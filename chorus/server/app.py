"""FastAPI aggregation server for federated LoRA training."""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File, Depends, Header, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from safetensors.torch import load as safetensors_load, save_file
from starlette.background import BackgroundTask

from chorus.exceptions import DuplicateClientError
from chorus.server.aggregation import AggregationStrategy, FedExLoRA, get_strategy
from chorus.privacy.mechanism import apply_dp
from chorus.server.storage import DeltaStorage, RoundState
from chorus.server.ws import ConnectionManager

logger = logging.getLogger("chorus.server")


# --- Rate Limiter ---


class RateLimiter:
    """Simple in-memory per-IP rate limiter."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        if self.requests_per_minute <= 0:
            return True
        now = time.time()
        window_start = now - 60
        reqs = self._requests[client_ip]
        self._requests[client_ip] = [t for t in reqs if t > window_start]
        if len(self._requests[client_ip]) >= self.requests_per_minute:
            return False
        self._requests[client_ip].append(now)
        return True


# --- Server State ---


class ServerState:
    """Mutable server state, initialized at startup."""

    storage: DeltaStorage
    strategy: AggregationStrategy
    model_id: str
    min_deltas: int
    dp_epsilon: float | None
    dp_delta: float
    dp_max_norm: float
    api_keys: set[str]  # empty set = no auth required
    ws_manager: ConnectionManager
    _aggregation_lock: asyncio.Lock
    max_upload_bytes: int
    norm_bound: float | None
    outlier_threshold: float | None
    rate_limiter: RateLimiter | None
    accountants: dict  # {model_id: {client_id: PrivacyAccountant}}
    accountant_target_epsilon: float | None
    accountant_noise_multiplier: float | None
    accountant_sample_rate: float


state = ServerState()
state.api_keys = set()
state.ws_manager = ConnectionManager()
state._aggregation_lock = asyncio.Lock()
state.max_upload_bytes = 500 * 1024 * 1024
state.norm_bound = None
state.outlier_threshold = None
state.rate_limiter = None
state.accountants = {}
state.accountant_target_epsilon = None
state.accountant_noise_multiplier = None
state.accountant_sample_rate = 1.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load persisted residuals into FedEx-LoRA strategy on startup
    if isinstance(state.strategy, FedExLoRA):
        residuals = state.storage.load_residuals(state.model_id)
        if residuals:
            state.strategy._residuals = residuals
            logger.info(f"Restored {len(residuals)} residual layers from disk")

    # Recover rounds stuck in AGGREGATING state (from a previous crash)
    stuck_rounds = state.storage.find_stuck_rounds(state.model_id)
    for rid in stuck_rounds:
        state.storage.set_round_state(state.model_id, rid, RoundState.OPEN)
        logger.warning(f"Recovered stuck round {rid}: AGGREGATING -> OPEN")

    # Restore privacy accountants for the configured model
    if state.accountant_target_epsilon is not None:
        restored = state.storage.load_all_accountants(state.model_id)
        if restored:
            state.accountants[state.model_id] = restored
            logger.info(f"Restored {len(restored)} privacy accountants from disk")

    logger.info(
        f"Chorus server started — model={state.model_id}, "
        f"strategy={state.strategy.name}, min_deltas={state.min_deltas}"
    )
    yield
    # Persist residuals on shutdown
    if isinstance(state.strategy, FedExLoRA):
        state.storage.save_residuals(state.model_id, state.strategy.get_residuals())
        logger.info("Residuals persisted to disk")
    logger.info("Chorus server shutting down")


app = FastAPI(
    title="Chorus Aggregation Server",
    description="Federated LoRA adapter aggregation via REST API",
    version="0.1.0",
    lifespan=lifespan,
)


def configure(
    model_id: str,
    data_dir: str = "./chorus_data",
    strategy: str = "fedex-lora",
    min_deltas: int = 2,
    dp_epsilon: float | None = None,
    dp_delta: float = 1e-5,
    dp_max_norm: float = 1.0,
    api_keys: list[str] | None = None,
    max_upload_bytes: int = 500 * 1024 * 1024,
    norm_bound: float | None = None,
    outlier_threshold: float | None = None,
    rate_limit: int = 0,
    accountant_target_epsilon: float | None = None,
    accountant_noise_multiplier: float | None = None,
    accountant_sample_rate: float = 1.0,
) -> FastAPI:
    """Configure the server before starting."""
    state.storage = DeltaStorage(data_dir)
    state.strategy = get_strategy(strategy)
    state.model_id = model_id
    state.min_deltas = min_deltas
    state.dp_epsilon = dp_epsilon
    state.dp_delta = dp_delta
    state.dp_max_norm = dp_max_norm
    state.api_keys = set(api_keys) if api_keys else set()
    state.ws_manager = ConnectionManager()
    state._aggregation_lock = asyncio.Lock()
    state.max_upload_bytes = max_upload_bytes
    state.norm_bound = norm_bound
    state.outlier_threshold = outlier_threshold
    state.rate_limiter = RateLimiter(rate_limit) if rate_limit > 0 else None
    state.accountant_target_epsilon = accountant_target_epsilon
    state.accountant_noise_multiplier = accountant_noise_multiplier
    state.accountant_sample_rate = accountant_sample_rate
    state.accountants = {}
    return app


# --- Authentication ---


async def require_auth(authorization: str | None = Header(default=None)):
    """Dependency: require Bearer token if api_keys are configured."""
    if not state.api_keys:
        return  # No auth configured, allow all
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    # Accept "Bearer <token>" or raw token
    token = authorization
    if token.startswith("Bearer "):
        token = token[7:]
    if not any(hmac.compare_digest(token, key) for key in state.api_keys):
        raise HTTPException(status_code=403, detail="Invalid API key")


# --- Rate Limiting Dependency ---


async def check_rate_limit(request: Request):
    """Dependency: enforce per-IP rate limiting if configured."""
    if state.rate_limiter is not None:
        client_ip = request.client.host if request.client else "unknown"
        if not state.rate_limiter.is_allowed(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")


# --- Privacy Accountant ---


def _ensure_accountant(model_id: str, client_id: str):
    """Return the accountant for (model_id, client_id), creating + persisting on first use.

    Returns None if accounting is not configured at the server level.
    """
    if state.accountant_target_epsilon is None or state.accountant_noise_multiplier is None:
        return None
    from chorus.privacy import PrivacyAccountant

    by_client = state.accountants.setdefault(model_id, {})
    if client_id in by_client:
        return by_client[client_id]

    # Try to restore from disk
    restored = state.storage.load_accountant(model_id, client_id)
    if restored is not None:
        by_client[client_id] = restored
        return restored

    # Create a fresh one
    a = PrivacyAccountant(
        target_epsilon=state.accountant_target_epsilon,
        target_delta=state.dp_delta,
        noise_multiplier=state.accountant_noise_multiplier,
        sample_rate=state.accountant_sample_rate,
    )
    by_client[client_id] = a
    state.storage.save_accountant(model_id, client_id, a)
    return a


# --- Upload Size Limit ---


async def _read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read uploaded file with size limit."""
    chunks = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)  # 1MB chunks
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds maximum size of {max_bytes} bytes",
            )
        chunks.append(chunk)
    return b"".join(chunks)


# --- API Endpoints ---


@app.get("/health")
async def health():
    """Health check — always public, no auth required."""
    return {
        "status": "ok",
        "model_id": state.model_id,
        "strategy": state.strategy.name,
        "ws_clients": state.ws_manager.connected_count,
    }


@app.get("/models/{model_id:path}/status", dependencies=[Depends(require_auth)])
async def model_status(model_id: str):
    current_round = state.storage.get_current_round(model_id)
    submitted = state.storage.list_deltas(model_id, current_round)
    latest_round = state.storage.get_latest_round(model_id)
    round_state = state.storage.get_round_state(model_id, current_round)
    return {
        "model_id": model_id,
        "current_round": current_round,
        "round_state": round_state.value,
        "deltas_submitted": len(submitted),
        "min_deltas": state.min_deltas,
        "latest_aggregated_round": latest_round,
        "strategy": state.strategy.name,
    }


@app.get("/models/{model_id:path}/clients/{client_id}/privacy", dependencies=[Depends(require_auth)])
async def get_client_privacy(model_id: str, client_id: str):
    """Return the privacy budget state for a specific client on this model."""
    if state.accountant_target_epsilon is None:
        raise HTTPException(status_code=404, detail="Privacy accounting is not enabled on this server")
    accountant = _ensure_accountant(model_id, client_id)
    eps_remaining, _ = accountant.remaining()
    return {
        "model_id": model_id,
        "client_id": client_id,
        "epsilon_consumed": accountant.get_epsilon(),
        "epsilon_target": accountant.target_epsilon,
        "epsilon_remaining": eps_remaining,
        "delta": accountant.target_delta,
        "exhausted": accountant.is_exhausted(),
    }


@app.post("/rounds/{round_id}/deltas", dependencies=[Depends(require_auth), Depends(check_rate_limit)])
async def submit_delta(
    round_id: int,
    file: UploadFile = File(...),
    client_id: str = Query(default=None),
    model_id: str = Query(default=None),
    dataset_size: int | None = Query(default=None),
):
    """Submit a LoRA delta for a given round.

    The delta should be a safetensors file containing LoRA A and B matrices.
    Rejects submissions to closed or aggregating rounds.
    """
    mid = model_id or state.model_id
    cid = client_id or str(uuid.uuid4())[:8]

    # Validate dataset_size
    if dataset_size is not None and dataset_size <= 0:
        raise HTTPException(status_code=400, detail="dataset_size must be a positive integer")

    # Validate round_id
    if round_id < 0:
        raise HTTPException(status_code=400, detail="round_id must be non-negative")
    current_round = state.storage.get_current_round(mid)
    if round_id > current_round:
        raise HTTPException(
            status_code=400,
            detail=f"round_id {round_id} is ahead of current round {current_round}",
        )

    # Check round state — reject late submissions
    if not state.storage.is_round_accepting(mid, round_id):
        round_state = state.storage.get_round_state(mid, round_id)
        logger.warning(f"Late submission rejected: client={cid}, round={round_id} ({round_state.value})")
        raise HTTPException(
            status_code=409,
            detail=f"Round {round_id} is {round_state.value}, not accepting submissions",
        )

    # Privacy-budget enforcement
    accountant = _ensure_accountant(mid, cid)
    if accountant is not None and accountant.is_exhausted():
        eps_consumed = accountant.get_epsilon()
        raise HTTPException(
            status_code=403,
            detail=(
                f"Privacy budget exhausted for client '{cid}' on model '{mid}'. "
                f"ε consumed = {eps_consumed:.4f}, target ε = {accountant.target_epsilon}. "
                f"Refusing further submissions."
            ),
        )

    # Read the uploaded safetensors file (with size limit)
    content = await _read_upload_limited(file, state.max_upload_bytes)
    try:
        tensors = safetensors_load(content)
    except Exception as e:
        logger.error(f"Invalid safetensors upload from client {cid}: {e}")
        raise HTTPException(status_code=400, detail="Invalid safetensors file")

    # Apply server-side DP if configured
    dp_applied = False
    if state.dp_epsilon is not None:
        tensors = apply_dp(
            tensors,
            epsilon=state.dp_epsilon,
            delta=state.dp_delta,
            max_norm=state.dp_max_norm,
        )
        dp_applied = True

    # Advance privacy accountant for this submission.
    # Step whenever an accountant exists, regardless of whether the *server*
    # applied noise — the client may be doing its own DP and the server is
    # purely tracking the budget.
    if accountant is not None:
        accountant.step()
        state.storage.save_accountant(mid, cid, accountant)

    # Build delta metadata (includes DP params if applied)
    delta_meta = {}
    if dataset_size is not None:
        delta_meta["dataset_size"] = dataset_size
    if dp_applied:
        delta_meta["dp_epsilon"] = state.dp_epsilon
        delta_meta["dp_delta"] = state.dp_delta
        delta_meta["dp_max_norm"] = state.dp_max_norm

    # Store the delta (atomic duplicate detection happens in storage layer)
    try:
        state.storage.save_delta(mid, round_id, cid, tensors, metadata=delta_meta)
    except DuplicateClientError:
        logger.warning(f"Duplicate submission rejected: client={cid}, round={round_id}")
        raise HTTPException(
            status_code=409,
            detail=f"Client '{cid}' has already submitted to round {round_id}",
        )

    submitted = state.storage.list_deltas(mid, round_id)
    num_submitted = len(submitted)

    logger.info(f"Delta received: model={mid}, round={round_id}, client={cid} ({num_submitted}/{state.min_deltas})")

    # Check if we have enough deltas to aggregate
    aggregated = False
    if num_submitted >= state.min_deltas:
        # Run aggregation in a thread to avoid blocking the event loop
        await _run_aggregation_async(mid, round_id)
        aggregated = True

    response: dict = {
        "status": "accepted",
        "client_id": cid,
        "round_id": round_id,
        "model_id": mid,
        "deltas_received": num_submitted,
        "min_deltas": state.min_deltas,
        "aggregated": aggregated,
        "next_round": round_id + 1,
    }
    if accountant is not None:
        eps_remaining, _ = accountant.remaining()
        response["privacy"] = {
            "epsilon_consumed": accountant.get_epsilon(),
            "epsilon_target": accountant.target_epsilon,
            "epsilon_remaining": eps_remaining,
            "delta": accountant.target_delta,
            "exhausted": accountant.is_exhausted(),
        }
    return response


@app.get("/models/{model_id:path}/latest", dependencies=[Depends(require_auth)])
async def get_latest(model_id: str):
    """Download the latest aggregated adapter."""
    latest_path = state.storage._model_dir(model_id) / "latest.safetensors"
    if not latest_path.exists():
        raise HTTPException(status_code=404, detail=f"No aggregated model found for '{model_id}'")
    return FileResponse(
        str(latest_path),
        media_type="application/octet-stream",
        filename="aggregated_adapter.safetensors",
    )


@app.get("/models/{model_id:path}/rounds/{round_id}", dependencies=[Depends(require_auth)])
async def get_round(model_id: str, round_id: int):
    """Download the aggregated adapter for a specific round."""
    agg_path = state.storage._round_dir(model_id, round_id) / "aggregated.safetensors"
    if not agg_path.exists():
        raise HTTPException(status_code=404, detail=f"No aggregation found for round {round_id}")
    return FileResponse(
        str(agg_path),
        media_type="application/octet-stream",
        filename=f"aggregated_round_{round_id}.safetensors",
    )


@app.post("/models/{model_id:path}/base-weights", dependencies=[Depends(require_auth), Depends(check_rate_limit)])
async def upload_base_weights(model_id: str, file: UploadFile = File(...)):
    """Upload base model weights (safetensors file)."""
    content = await _read_upload_limited(file, state.max_upload_bytes)
    try:
        tensors = safetensors_load(content)
    except Exception as e:
        logger.error(f"Invalid base weights upload for model {model_id}: {e}")
        raise HTTPException(status_code=400, detail="Invalid safetensors file")
    state.storage.save_base_weights(model_id, tensors)
    return {"status": "ok", "model_id": model_id, "num_tensors": len(tensors)}


@app.get("/models/{model_id:path}/base-weights", dependencies=[Depends(require_auth)])
async def download_base_weights(model_id: str):
    """Download the current base model weights."""
    path = state.storage._base_weights_path(model_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No base weights found for '{model_id}'")
    return FileResponse(
        str(path),
        media_type="application/octet-stream",
        filename="base_weights.safetensors",
    )


@app.get("/models/{model_id:path}/checkpoint", dependencies=[Depends(require_auth)])
async def get_checkpoint(model_id: str):
    """Download base weights merged with the latest adapter (full checkpoint).

    If no base weights are set, returns 404. If no adapter exists yet,
    returns the base weights as-is.
    """
    base = state.storage.load_base_weights(model_id)
    if base is None:
        raise HTTPException(status_code=404, detail=f"No base weights found for '{model_id}'")

    adapter = state.storage.load_aggregated(model_id)
    if adapter:
        from chorus.server.weight_manager import merge_adapter_into_base
        merged = merge_adapter_into_base(base, adapter)
    else:
        merged = base

    # Save to temp file and serve, with cleanup after response is sent
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    save_file(merged, tmp.name)
    tmp_path = tmp.name
    tmp.close()
    return FileResponse(
        tmp_path,
        media_type="application/octet-stream",
        filename="checkpoint.safetensors",
        background=BackgroundTask(os.unlink, tmp_path),
    )


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str, token: str | None = None):
    """WebSocket endpoint for real-time aggregation notifications."""
    # Authenticate WebSocket connections if API keys are configured
    if state.api_keys:
        if not token or not any(hmac.compare_digest(token, key) for key in state.api_keys):
            await websocket.close(code=4003, reason="Authentication required")
            return

    await state.ws_manager.connect(client_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await state.ws_manager.disconnect(client_id)


async def _run_aggregation_async(model_id: str, round_id: int) -> None:
    """Run aggregation in a background thread to avoid blocking the event loop."""
    async with state._aggregation_lock:
        # Double-check: another coroutine may have already aggregated this round
        if state.storage.get_round_state(model_id, round_id) != RoundState.OPEN:
            logger.info(f"Round {round_id} already aggregated/aggregating, skipping")
            return

        # Mark round as aggregating so no more submissions arrive
        state.storage.set_round_state(model_id, round_id, RoundState.AGGREGATING)
        try:
            residuals_folded = await asyncio.to_thread(_run_aggregation, model_id, round_id)
            # Mark round as closed after successful aggregation
            state.storage.set_round_state(model_id, round_id, RoundState.CLOSED)
            # Broadcast round completion to WebSocket clients
            await state.ws_manager.broadcast({
                "event": "round_complete",
                "model_id": model_id,
                "round_id": round_id,
                "next_round": round_id + 1,
                "adapter_ready": True,
                "base_weights_updated": bool(residuals_folded),
            })
        except Exception as exc:
            logger.error(f"Aggregation failed for round {round_id}: {exc}", exc_info=True)
            # Re-open round on failure so clients can retry
            state.storage.set_round_state(model_id, round_id, RoundState.OPEN)
            raise


def _run_aggregation(model_id: str, round_id: int) -> bool:
    """Run the aggregation algorithm on all deltas for a round (CPU-bound).

    Returns True if residuals were folded into base weights.
    """
    from chorus.server.aggregation import norm_bound_deltas, filter_outlier_deltas

    deltas = state.storage.load_all_deltas(model_id, round_id)
    if not deltas:
        return False

    logger.info(f"Aggregating {len(deltas)} deltas for model={model_id}, round={round_id}")

    # Apply Byzantine defenses if configured
    if state.norm_bound is not None:
        deltas = norm_bound_deltas(deltas, state.norm_bound)
        logger.info(f"Applied norm bounding (max_norm={state.norm_bound})")

    if state.outlier_threshold is not None:
        original_count = len(deltas)
        deltas = filter_outlier_deltas(deltas, state.outlier_threshold)
        filtered = original_count - len(deltas)
        if filtered > 0:
            logger.warning(f"Filtered {filtered} outlier deltas (threshold={state.outlier_threshold})")
        if not deltas:
            logger.error("All deltas filtered as outliers, skipping aggregation")
            return False

    # Compute dataset-size-proportional weights if available
    # Note: after filtering, we re-load metadata for surviving clients only
    metadata_list = state.storage.load_all_delta_metadata(model_id, round_id)
    # If outlier filtering reduced deltas, metadata_list still has all entries.
    # Use len(deltas) to determine count; if mismatch, fall back to uniform weights.
    if len(metadata_list) != len(deltas):
        weights = None
    else:
        sizes = [m.get("dataset_size") for m in metadata_list]
        if sizes and all(s is not None for s in sizes):
            total = sum(sizes)
            weights = [s / total for s in sizes] if total > 0 else None
            logger.info(f"Using dataset-size weights: {weights}")
        else:
            weights = None

    result = state.strategy.aggregate(deltas, weights)

    # Persist residuals after each aggregation (FedEx-LoRA)
    residuals_folded = False
    if isinstance(state.strategy, FedExLoRA):
        residuals = state.strategy.get_residuals()
        state.storage.save_residuals(model_id, residuals)

        # Auto-fold residuals into base weights if available
        base_weights = state.storage.load_base_weights(model_id)
        if base_weights and residuals:
            from chorus.server.weight_manager import fold_residuals_into_base
            updated_base = fold_residuals_into_base(base_weights, residuals)
            state.storage.save_base_weights(model_id, updated_base, meta={
                "last_fold_round": round_id,
            })
            state.strategy.reset_residuals()
            state.storage.save_residuals(model_id, {})
            residuals_folded = True
            logger.info(f"Residuals folded into base weights for model={model_id}")

    state.storage.save_aggregated(model_id, round_id, result)

    logger.info(f"Aggregation complete for model={model_id}, round={round_id}")
    return residuals_folded
