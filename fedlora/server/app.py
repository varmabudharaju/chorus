"""FastAPI aggregation server for federated LoRA training."""

from __future__ import annotations

import io
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from safetensors.torch import load as safetensors_load, save_file

from fedlora.server.aggregation import AggregationStrategy, get_strategy
from fedlora.server.privacy import apply_dp
from fedlora.server.storage import DeltaStorage

logger = logging.getLogger("fedlora.server")


class ServerState:
    """Mutable server state, initialized at startup."""

    storage: DeltaStorage
    strategy: AggregationStrategy
    model_id: str
    min_deltas: int
    dp_epsilon: float | None
    dp_delta: float
    dp_max_norm: float


state = ServerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"FedLoRA server started — model={state.model_id}, "
        f"strategy={state.strategy.name}, min_deltas={state.min_deltas}"
    )
    yield
    logger.info("FedLoRA server shutting down")


app = FastAPI(
    title="FedLoRA Aggregation Server",
    description="Federated LoRA adapter aggregation via REST API",
    version="0.1.0",
    lifespan=lifespan,
)


def configure(
    model_id: str,
    data_dir: str = "./fedlora_data",
    strategy: str = "fedex-lora",
    min_deltas: int = 2,
    dp_epsilon: float | None = None,
    dp_delta: float = 1e-5,
    dp_max_norm: float = 1.0,
) -> FastAPI:
    """Configure the server before starting."""
    state.storage = DeltaStorage(data_dir)
    state.strategy = get_strategy(strategy)
    state.model_id = model_id
    state.min_deltas = min_deltas
    state.dp_epsilon = dp_epsilon
    state.dp_delta = dp_delta
    state.dp_max_norm = dp_max_norm
    return app


# --- API Endpoints ---


@app.get("/health")
async def health():
    return {"status": "ok", "model_id": state.model_id, "strategy": state.strategy.name}


@app.get("/models/{model_id}/status")
async def model_status(model_id: str):
    current_round = state.storage.get_current_round(model_id)
    submitted = state.storage.list_deltas(model_id, current_round)
    latest_round = state.storage.get_latest_round(model_id)
    return {
        "model_id": model_id,
        "current_round": current_round,
        "deltas_submitted": len(submitted),
        "min_deltas": state.min_deltas,
        "latest_aggregated_round": latest_round,
        "strategy": state.strategy.name,
    }


@app.post("/rounds/{round_id}/deltas")
async def submit_delta(
    round_id: int,
    file: UploadFile = File(...),
    client_id: str = Query(default=None),
    model_id: str = Query(default=None),
):
    """Submit a LoRA delta for a given round.

    The delta should be a safetensors file containing LoRA A and B matrices.
    """
    mid = model_id or state.model_id
    cid = client_id or str(uuid.uuid4())[:8]

    # Read the uploaded safetensors file
    content = await file.read()
    try:
        tensors = safetensors_load(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid safetensors file: {e}")

    # Apply server-side DP if configured
    if state.dp_epsilon is not None:
        tensors = apply_dp(
            tensors,
            epsilon=state.dp_epsilon,
            delta=state.dp_delta,
            max_norm=state.dp_max_norm,
        )

    # Store the delta
    state.storage.save_delta(mid, round_id, cid, tensors)
    submitted = state.storage.list_deltas(mid, round_id)
    num_submitted = len(submitted)

    logger.info(f"Delta received: model={mid}, round={round_id}, client={cid} ({num_submitted}/{state.min_deltas})")

    # Check if we have enough deltas to aggregate
    aggregated = False
    if num_submitted >= state.min_deltas:
        _run_aggregation(mid, round_id)
        aggregated = True

    return {
        "status": "accepted",
        "client_id": cid,
        "round_id": round_id,
        "model_id": mid,
        "deltas_received": num_submitted,
        "min_deltas": state.min_deltas,
        "aggregated": aggregated,
    }


@app.get("/models/{model_id}/latest")
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


@app.get("/models/{model_id}/rounds/{round_id}")
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


def _run_aggregation(model_id: str, round_id: int) -> None:
    """Run the aggregation algorithm on all deltas for a round."""
    deltas = state.storage.load_all_deltas(model_id, round_id)
    if not deltas:
        return

    logger.info(f"Aggregating {len(deltas)} deltas for model={model_id}, round={round_id}")

    result = state.strategy.aggregate(deltas)
    state.storage.save_aggregated(model_id, round_id, result)

    logger.info(f"Aggregation complete for model={model_id}, round={round_id}")
