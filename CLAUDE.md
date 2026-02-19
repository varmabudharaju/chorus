# Chorus — Project Guide

## Overview

Chorus (formerly FedLoRA) is a federated LoRA adapter aggregation framework. FastAPI server + Python client SDK + CLI. Package name: `chorus`. Entry point: `chorus.cli.main:cli`.

## Commands

```bash
pip install -e ".[dev]"       # Install with dev deps
python3 -m pytest tests/ -v   # Run all tests (80 tests)
chorus server --model <id>    # Start aggregation server
chorus simulate --compare     # Run FedAvg vs FedEx-LoRA comparison
```

## Architecture

```
chorus/
├── patterns.py          # Shared LoRA key patterns (LORA_A_PATTERN, LORA_B_PATTERN, get_layer_pairs)
├── server/
│   ├── app.py           # FastAPI endpoints + auth + async aggregation
│   ├── aggregation.py   # FedAvg + FedEx-LoRA (SVD-optimal) + Byzantine defenses
│   ├── storage.py       # Filesystem delta storage + round state + residual persistence
│   └── privacy.py       # Gaussian DP mechanism + global L2 clipping
├── client/
│   ├── sdk.py           # ChorusClient class
│   └── delta.py         # LoRA matrix extraction from PEFT adapters
├── cli/main.py          # Click CLI (server, submit, pull, simulate)
└── simulate/runner.py   # Synthetic multi-client federation runner
```

## Key Patterns

- Aggregation strategies implement `AggregationStrategy` ABC in `aggregation.py`
- `safetensors` format everywhere — no pickle
- Server state is a module-level `ServerState` singleton configured via `configure()`
- LoRA keys follow pattern: `{layer}.lora_A.weight` / `{layer}.lora_B.weight`
- PEFT prefix `base_model.model.` is stripped in `delta.py`
- Shared regex patterns in `chorus/patterns.py` (not duplicated)
- Auth via Bearer token, configured with `--api-key` flags (multiple allowed)
- Round lifecycle: OPEN → AGGREGATING → CLOSED. Late submissions rejected.
- Aggregation runs in `asyncio.to_thread()` to avoid blocking the event loop
- FedEx-LoRA residuals persisted to disk via `storage.save_residuals()` / `load_residuals()`
- DP clipping uses global L2 norm across all tensors (user-level DP)

---

## Remaining Issues (Known Debt)

### Security
1. **No TLS/HTTPS.** Server binds to `0.0.0.0:8080` over plain HTTP. Adapter weights transit in cleartext. Deploy behind a reverse proxy (nginx/caddy) for production.

### Architecture
2. **No model weight management.** The `--model` flag is just a string ID. Server doesn't store/serve base weights. No mechanism to fold residuals into base weights.

### Correctness
3. **FedEx-LoRA gives optimal-but-not-exact aggregation.** The average of N rank-r matrices has rank up to N*r, which cannot be exactly represented in rank r. The SVD gives the mathematically optimal rank-r approximation (Eckart-Young theorem), with 1.1-2.0x improvement over FedAvg. Residuals track what couldn't be captured and can be folded into base weights.
4. **No client weighting by data size.** Weights are always uniform `1/n`. Real FL systems weight clients by dataset size. The API supports custom weights but the CLI/client don't expose this.

### Functional Gaps
5. **No actual training integration.** Framework only handles aggregation. No code to fine-tune a model and produce adapters. Clients must bring their own PEFT pipeline.
6. **Never tested with real models.** All tests and simulations use `torch.randn()` synthetic tensors. Never validated with actual PEFT adapters from real LoRA training.
7. **No adapter_config.json handling on server side.** Server only deals with raw tensors. `apply_delta_to_adapter()` in `delta.py` copies config but is never called by the server or aggregation pipeline.
8. **No heterogeneous federation support.** Cannot handle clients with different LoRA ranks, different target modules, or different base model quantizations.

### Minor Issues
9. **`_run_comparison`** in `runner.py` returns `ex_result` from only the last round, not accumulated across rounds.
