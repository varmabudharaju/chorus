# Chorus — Project Guide

## Overview

Chorus (formerly FedLoRA) is a **full continuous federated LoRA training framework**. Clients train locally, submit deltas, the server aggregates and folds improvements back into base weights, then pushes notifications so clients retrain on the improved model. FastAPI server + Python client SDK + CLI. Package name: `chorus`. Entry point: `chorus.cli.main:cli`.

## Commands

```bash
pip install -e ".[dev]"       # Install with dev deps
python3 -m pytest tests/ -v   # Run all tests (124 tests)
chorus server --model <id>    # Start aggregation server
chorus server --model <id> --base-weights model.safetensors  # With base weight folding
chorus train --server http://localhost:8080 --model <hf-id> --dataset <ds>  # Full loop
chorus simulate --compare     # Run FedAvg vs FedEx-LoRA comparison
chorus submit --server <url> --adapter <path> --dataset-size 5000
```

## Architecture

```
chorus/
├── patterns.py              # Shared LoRA key patterns (LORA_A_PATTERN, LORA_B_PATTERN, get_layer_pairs)
├── server/
│   ├── app.py               # FastAPI endpoints + auth + async aggregation + WebSocket
│   ├── aggregation.py       # FedAvg + FedEx-LoRA (SVD-optimal) + Byzantine defenses
│   ├── storage.py           # Filesystem delta/base-weight storage + round state + residuals
│   ├── weight_manager.py    # Residual folding into base weights + adapter merging
│   ├── ws.py                # WebSocket connection manager for live notifications
│   └── privacy.py           # Gaussian DP mechanism + global L2 clipping
├── client/
│   ├── sdk.py               # ChorusClient class (submit, pull, listen, train_loop)
│   ├── trainer.py           # LoRATrainer wrapper for HF PEFT + transformers
│   └── delta.py             # LoRA matrix extraction from PEFT adapters
├── cli/main.py              # Click CLI (server, submit, pull, simulate, train)
└── simulate/runner.py       # Synthetic multi-client federation runner
```

## Key Patterns

- Aggregation strategies implement `AggregationStrategy` ABC in `aggregation.py`
- `safetensors` format everywhere — no pickle
- Server state is a module-level `ServerState` singleton configured via `configure()`
- LoRA keys follow pattern: `{layer}.lora_A.weight` / `{layer}.lora_B.weight`
- PEFT prefix `base_model.model.` is stripped in `delta.py`
- Shared regex patterns in `chorus/patterns.py` (not duplicated)
- Auth via Bearer token, configured with `--api-key` flags (multiple allowed)
- Round lifecycle: OPEN → AGGREGATING → CLOSED. Late submissions rejected. Next round auto-opens.
- Aggregation runs in `asyncio.to_thread()` to avoid blocking the event loop
- FedEx-LoRA residuals persisted to disk via `storage.save_residuals()` / `load_residuals()`
- Residuals auto-folded into base weights after aggregation (if base weights are set)
- Dataset-size-proportional client weighting via `dataset_size` query param
- WebSocket broadcasts `round_complete` events after aggregation
- DP clipping uses global L2 norm across all tensors (user-level DP)

## Continuous Improvement Loop

```
Client                           Server
  |                                |
  |--- train locally (LoRA) ----> |
  |--- submit delta + size -----> |  (stores delta + metadata)
  |                                |  (enough deltas? aggregate!)
  |                                |  (fold residuals into base weights)
  |  <-- WS: round_complete ---   |  (broadcast to all clients)
  |--- pull updated adapter ----> |
  |--- repeat from train -------> |
```

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check (public, includes ws_clients count) |
| GET | `/models/{id}/status` | Round state, delta count, latest round |
| POST | `/rounds/{round_id}/deltas` | Submit LoRA delta (dataset_size param for weighting) |
| GET | `/models/{id}/latest` | Download latest aggregated adapter |
| GET | `/models/{id}/rounds/{round_id}` | Download round-specific adapter |
| POST | `/models/{id}/base-weights` | Upload base model weights |
| GET | `/models/{id}/base-weights` | Download current base weights |
| GET | `/models/{id}/checkpoint` | Download base + adapter merged |
| WS | `/ws/{client_id}` | WebSocket for live round notifications |

---

## Remaining Issues (Known Debt)

### Security
1. **No TLS/HTTPS.** Server binds to `0.0.0.0:8080` over plain HTTP. Deploy behind a reverse proxy (nginx/caddy) for production.

### Correctness
2. **FedEx-LoRA gives optimal-but-not-exact aggregation.** The SVD gives the optimal rank-r approximation (Eckart-Young theorem). Residuals capture what's lost and are now auto-folded into base weights, making the combined result exact.

### Functional Gaps
3. **Never tested with real models.** All tests use synthetic tensors. Never validated with actual PEFT adapters from real LoRA training.
4. **No adapter_config.json handling on server side.** Server only deals with raw tensors.
