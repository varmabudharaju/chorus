"""FastAPI aggregation server for federated LoRA training."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Depends, Header, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from safetensors.torch import load as safetensors_load, save_file

from chorus.server.aggregation import AggregationStrategy, FedExLoRA, get_strategy
from chorus.server.privacy import apply_dp
from chorus.server.storage import DeltaStorage, RoundState
from chorus.server.ws import ConnectionManager

logger = logging.getLogger("chorus.server")


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


state = ServerState()
state.api_keys = set()
state.ws_manager = ConnectionManager()
state._aggregation_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load persisted residuals into FedEx-LoRA strategy on startup
    if isinstance(state.strategy, FedExLoRA):
        residuals = state.storage.load_residuals(state.model_id)
        if residuals:
            state.strategy._residuals = residuals
            logger.info(f"Restored {len(residuals)} residual layers from disk")

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
    if token not in state.api_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")


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


@app.post("/rounds/{round_id}/deltas", dependencies=[Depends(require_auth)])
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

    # Check round state — reject late submissions
    if not state.storage.is_round_accepting(mid, round_id):
        round_state = state.storage.get_round_state(mid, round_id)
        raise HTTPException(
            status_code=409,
            detail=f"Round {round_id} is {round_state.value}, not accepting submissions",
        )

    # Reject duplicate client submissions within a round
    existing = state.storage.list_deltas(mid, round_id)
    if cid in existing:
        raise HTTPException(
            status_code=409,
            detail=f"Client '{cid}' has already submitted to round {round_id}",
        )

    # Read the uploaded safetensors file
    content = await file.read()
    try:
        tensors = safetensors_load(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid safetensors file: {e}")

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

    # Build delta metadata (includes DP params if applied)
    delta_meta = {}
    if dataset_size is not None:
        delta_meta["dataset_size"] = dataset_size
    if dp_applied:
        delta_meta["dp_epsilon"] = state.dp_epsilon
        delta_meta["dp_delta"] = state.dp_delta
        delta_meta["dp_max_norm"] = state.dp_max_norm

    # Store the delta
    state.storage.save_delta(mid, round_id, cid, tensors, metadata=delta_meta)
    submitted = state.storage.list_deltas(mid, round_id)
    num_submitted = len(submitted)

    logger.info(f"Delta received: model={mid}, round={round_id}, client={cid} ({num_submitted}/{state.min_deltas})")

    # Check if we have enough deltas to aggregate
    aggregated = False
    if num_submitted >= state.min_deltas:
        # Run aggregation in a thread to avoid blocking the event loop
        await _run_aggregation_async(mid, round_id)
        aggregated = True

    return {
        "status": "accepted",
        "client_id": cid,
        "round_id": round_id,
        "model_id": mid,
        "deltas_received": num_submitted,
        "min_deltas": state.min_deltas,
        "aggregated": aggregated,
        "next_round": round_id + 1,
    }


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


@app.post("/models/{model_id:path}/base-weights", dependencies=[Depends(require_auth)])
async def upload_base_weights(model_id: str, file: UploadFile = File(...)):
    """Upload base model weights (safetensors file)."""
    content = await file.read()
    try:
        tensors = safetensors_load(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid safetensors file: {e}")
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

    # Save to temp file and serve
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    save_file(merged, tmp.name)
    return FileResponse(
        tmp.name,
        media_type="application/octet-stream",
        filename="checkpoint.safetensors",
    )


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time aggregation notifications."""
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
        except Exception:
            # Re-open round on failure so clients can retry
            state.storage.set_round_state(model_id, round_id, RoundState.OPEN)
            raise


def _run_aggregation(model_id: str, round_id: int) -> bool:
    """Run the aggregation algorithm on all deltas for a round (CPU-bound).

    Returns True if residuals were folded into base weights.
    """
    deltas = state.storage.load_all_deltas(model_id, round_id)
    if not deltas:
        return False

    logger.info(f"Aggregating {len(deltas)} deltas for model={model_id}, round={round_id}")

    # Compute dataset-size-proportional weights if available
    metadata_list = state.storage.load_all_delta_metadata(model_id, round_id)
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
