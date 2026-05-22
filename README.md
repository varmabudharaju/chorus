# Chorus

[![CI](https://github.com/varmabudharaju/chorus/actions/workflows/ci.yml/badge.svg)](https://github.com/varmabudharaju/chorus/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/chorus-fl.svg)](https://pypi.org/project/chorus-fl/)

**Federated LoRA fine-tuning with mathematically exact aggregation — when residuals are folded into base weights every round ([details](docs/honest-tradeoffs.md#exactness)).**

Chorus is a framework for federated fine-tuning of large language models using LoRA adapters. Multiple clients train on their private data, submit adapter deltas to a central server, and receive back aggregated improvements — without sharing any raw data.

The key insight: standard FedAvg is **broken for LoRA** because `avg(B @ A) != avg(B) @ avg(A)`. Chorus implements [FedEx-LoRA](https://arxiv.org/abs/2501.03075) (ACL/ICLR 2025), which provides **exact** federated aggregation by tracking and folding SVD residuals.

## How It Works

```
Client 1 (private data)          Aggregation Server           Client 2 (private data)
┌─────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
│  1. Train LoRA      │       │                     │       │  1. Train LoRA      │
│  2. Submit delta  ──┼──POST─┼→ Collect deltas     │←─POST─┼── 2. Submit delta   │
│                     │       │  FedEx-LoRA agg     │       │                     │
│                     │       │  Fold residuals     │       │                     │
│  3. Pull updated  ←─┼──GET──┼─ Serve result       │──GET──┼→ 3. Pull updated    │
│  4. Repeat          │       │  WS: round_complete │       │  4. Repeat          │
└─────────────────────┘       └─────────────────────┘       └─────────────────────┘
```

## Honest tradeoffs

Chorus's claims hold under conditions that aren't always obvious from a quickstart.
Before relying on any of them, read [docs/honest-tradeoffs.md](docs/honest-tradeoffs.md).
Highlights:

- **"Mathematically exact" aggregation** holds when residuals are folded into base weights every round — the default server path, but not yet the eval harness ([#19](https://github.com/varmabudharaju/chorus/issues/19)). [More](docs/honest-tradeoffs.md#exactness)
- **Differential privacy** is per-submission Gaussian noise with a stateful accountant — but the accountant is opt-in: set `--accountant-target-epsilon` and `--accountant-noise-multiplier` on the server to bound the privacy budget. Without them, privacy loss accumulates unbounded. [More](docs/honest-tradeoffs.md#differential-privacy)
- **"Byzantine defenses"** are sanity checks against naive attackers (norm bound + z-score outlier). They will not stop an adaptive adversary. Real Byzantine-robust aggregation is on the Phase 2 roadmap. [More](docs/honest-tradeoffs.md#byzantine-robustness)
- **API keys are global.** No per-model scoping in v0.2.0. Run one server per trust boundary. [More](docs/honest-tradeoffs.md#multi-tenant-scope)
- **Heterogeneous-rank clients** work on `fedex-lora` and crash on `fedavg`. [More](docs/honest-tradeoffs.md#heterogeneous-clients)
- **Alpha software:** single-process server, in-memory rate limiter, filesystem storage, HTTP. Not hardened for multi-tenant production. [More](docs/honest-tradeoffs.md#production-readiness)

## Installation

```bash
pip install chorus-fl
```

With optional dependencies:

```bash
# For local LoRA training (PEFT + Transformers)
pip install "chorus-fl[peft]"

# For differential privacy
pip install "chorus-fl[privacy]"

# Everything
pip install "chorus-fl[all]"
```

From source:

```bash
git clone https://github.com/varmabudharaju/chorus.git
cd chorus
pip install -e ".[dev]"
```

## Quick Start

### 1. Start the server

```bash
chorus server --model meta-llama/Llama-3.2-3B --min-deltas 3
```

### 2. Submit adapters from clients

```python
from chorus import ChorusClient

client = ChorusClient(
    server="http://localhost:8080",
    model_id="meta-llama/Llama-3.2-3B",
)

# After your local LoRA training...
client.submit_delta(adapter_path="./my-adapter")

# Pull the aggregated global adapter
client.pull_latest(output_path="./updated-adapter")

client.close()
```

### 3. Run a simulation (no server needed)

```bash
# Compare FedAvg vs FedEx-LoRA
chorus simulate --clients 10 --rounds 5 --compare
```

## Why FedEx-LoRA?

LoRA decomposes weight updates as `W = B @ A` (two low-rank matrices). When you naively average across clients:

```
avg(B_i @ A_i)  !=  avg(B_i) @ avg(A_i)
```

**FedAvg produces mathematically inexact aggregation for LoRA.** FedEx-LoRA fixes this:

1. Computes the exact weighted average of full-rank products `B_i @ A_i`
2. Uses SVD to get the optimal rank-r approximation (Eckart-Young theorem)
3. Tracks the residual between exact and approximate results
4. Folds residuals into base weights, making the combined result **exact**

## CLI Reference

### `chorus server`

Start the aggregation server.

```bash
chorus server --model <model-id> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | *required* | Model ID (e.g. `meta-llama/Llama-3.2-3B`) |
| `--port` | `8080` | Port to listen on |
| `--host` | `0.0.0.0` | Host to bind to |
| `--data-dir` | `./chorus_data` | Data directory for storage |
| `--strategy` | `fedex-lora` | Aggregation strategy (`fedavg` or `fedex-lora`) |
| `--min-deltas` | `2` | Minimum deltas before aggregation triggers |
| `--dp-epsilon` | *disabled* | Server-side differential privacy epsilon |
| `--api-key` | *disabled* | API key for auth (can specify multiple times) |
| `--base-weights` | *none* | Path to base model weights (`.safetensors`) |
| `--norm-bound` | *disabled* | Max L2 norm for Byzantine defense |
| `--outlier-threshold` | *disabled* | Z-score threshold for outlier detection |
| `--rate-limit` | `0` | Max requests per minute per IP (0 = disabled) |
| `-v, --verbose` | | Verbose logging |

### `chorus submit`

Submit a LoRA adapter delta to the server.

```bash
chorus submit --server <url> --adapter <path> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--server` | *required* | Server URL |
| `--adapter` | *required* | Path to adapter directory or `.safetensors` file |
| `--model-id` | *auto* | Model ID (auto-detected from server) |
| `--client-id` | *auto* | Client identifier |
| `--round-id` | *current* | Target round |
| `--dp-epsilon` | *disabled* | Local DP epsilon |
| `--dataset-size` | *none* | Dataset size for weighted aggregation |
| `--api-key` | *none* | API key for authentication |

### `chorus pull`

Pull the latest aggregated adapter from the server.

```bash
chorus pull --server <url> --output <path> [options]
```

### `chorus train`

Run the full federated training loop (train -> submit -> wait -> pull -> repeat).

```bash
chorus train --server <url> --model <hf-model-id> --dataset <dataset> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--server` | *required* | Server URL |
| `--model` | *required* | HuggingFace model ID |
| `--dataset` | *required* | HuggingFace dataset or local path |
| `--rounds` | *infinite* | Number of training rounds |
| `--lora-rank` | `16` | LoRA rank |
| `--max-steps` | `-1` | Max training steps per round (-1 = full epoch) |
| `--dp-epsilon` | *disabled* | Local DP epsilon |

### `chorus simulate`

Run a simulated federation with synthetic data.

```bash
chorus simulate --clients 10 --rounds 5 --compare
```

### `chorus status`

Show the current status of a Chorus server.

```bash
chorus status --server <url>
```

### `chorus export`

Export a merged model (base + aggregated adapter) ready for deployment.

```bash
chorus export --server <url> --model <hf-model-id> --output ./merged/
```

## Python SDK

### `ChorusClient`

```python
from chorus import ChorusClient

client = ChorusClient(
    server="http://localhost:8080",
    model_id="my-model",
    client_id="client-1",          # optional, auto-generated if omitted
    api_key="secret",              # optional, for authenticated servers
    dp_epsilon=1.0,                # optional, local differential privacy
    dp_delta=1e-5,                 # optional, DP delta parameter
    dp_max_norm=1.0,               # optional, DP clipping norm
    timeout=120.0,                 # optional, HTTP timeout in seconds
)

# Check server status
status = client.status()

# Submit a trained LoRA adapter
result = client.submit_delta(
    adapter_path="./my-adapter",   # PEFT adapter dir or .safetensors
    round_id=None,                 # None = current round
    dataset_size=5000,             # for weighted aggregation
)

# Submit raw tensors directly
result = client.submit_tensors(tensors={"layer.lora_A.weight": tensor_a, ...})

# Pull the latest aggregated adapter
client.pull_latest(output_path="./updated-adapter")

# Pull a specific round
client.pull_round(round_id=3, output_path="./round-3-adapter")

# Export merged model (requires chorus[peft])
client.export_model(
    base_model="meta-llama/Llama-3.2-3B",
    output_dir="./merged-model",
)

# Full training loop (requires chorus[peft])
client.train_loop(
    trainer=my_trainer,            # LoRATrainer instance
    rounds=5,
)

# Listen for round completion via WebSocket
for event in client.listen():
    print(f"Round {event['round_id']} complete!")

client.close()
# Or use as context manager:
# with ChorusClient(...) as client:
#     ...
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (public, includes `ws_clients` count) |
| `GET` | `/models/{id}/status` | Round state, delta count, latest round |
| `POST` | `/rounds/{round_id}/deltas` | Submit LoRA delta (`dataset_size` param for weighting) |
| `GET` | `/models/{id}/latest` | Download latest aggregated adapter |
| `GET` | `/models/{id}/rounds/{round_id}` | Download round-specific adapter |
| `POST` | `/models/{id}/base-weights` | Upload base model weights |
| `GET` | `/models/{id}/base-weights` | Download current base weights |
| `GET` | `/models/{id}/checkpoint` | Download base + adapter merged checkpoint |
| `WS` | `/ws/{client_id}` | WebSocket for live round notifications |

## Architecture

```
chorus/
├── patterns.py              # Shared LoRA key patterns
├── exceptions.py            # Exception hierarchy (ChorusError, etc.)
├── server/
│   ├── app.py               # FastAPI endpoints + auth + async aggregation
│   ├── aggregation.py       # FedAvg + FedEx-LoRA (SVD) + Byzantine defenses
│   ├── storage.py           # Filesystem storage for deltas, base weights, round state
│   ├── weight_manager.py    # Residual folding into base weights
│   ├── ws.py                # WebSocket connection manager
│   └── privacy.py           # Gaussian DP mechanism + L2 clipping
├── client/
│   ├── sdk.py               # ChorusClient (submit, pull, listen, train_loop, export)
│   ├── trainer.py           # LoRATrainer wrapper for HF PEFT
│   └── delta.py             # LoRA matrix extraction from PEFT adapters
├── cli/
│   └── main.py              # Click CLI with error handling
└── simulate/
    └── runner.py            # Synthetic multi-client federation runner
```

## Security Features

Chorus includes several security mechanisms for production deployments:

- **Authentication** — Bearer token auth via `--api-key` (supports multiple keys)
- **Differential privacy** — Per-submission Gaussian DP with global L2 clipping (`--dp-epsilon`), plus a stateful `PrivacyAccountant` (RDP composition via Google's `dp-accounting`, `opacus` fallback). Set `--accountant-target-epsilon` and `--accountant-noise-multiplier` on the server to halt before you exceed your budget; without those, privacy loss accumulates unbounded across rounds. The eval harness does not yet apply DP ([#20](https://github.com/varmabudharaju/chorus/issues/20)); the *server* path does. [Details](docs/honest-tradeoffs.md#differential-privacy).
- **Sanity-check defenses against naive attackers** — L2 norm bounding (`--norm-bound`) and z-score outlier detection (`--outlier-threshold`) reject deltas with absurd magnitude or those several standard deviations from the round's median. These catch random-noise injection and trivial corruption; they **do not** stop coordinated attackers staying under the bound, label-flipping at the task level, or gradient-inversion attacks ([details](docs/honest-tradeoffs.md#byzantine-robustness)). Real Byzantine-robust aggregation (Krum, coordinate-wise median) is on the Phase 2 roadmap.
- **Rate limiting** — Per-IP request throttling via `--rate-limit`
- **safetensors only** — All weight serialization uses safetensors format (no pickle deserialization)

> **Note:** Chorus serves over HTTP. For production, deploy behind a TLS-terminating reverse proxy (nginx, Caddy, etc.).

## Aggregation Strategies

| Strategy | Exact? | Description |
|----------|--------|-------------|
| `fedex-lora` (default) | Yes | SVD-based exact aggregation with residual folding |
| `fedavg` | No | Naive independent averaging of A and B matrices |

## Configuration Examples

### Secure production server

```bash
chorus server \
  --model meta-llama/Llama-3.2-3B \
  --min-deltas 5 \
  --api-key $SECRET_KEY_1 \
  --api-key $SECRET_KEY_2 \
  --dp-epsilon 2.0 \
  --norm-bound 10.0 \
  --outlier-threshold 3.0 \
  --rate-limit 60 \
  --base-weights ./base-model.safetensors
```

### Client with local DP

```python
client = ChorusClient(
    server="http://chorus.internal:8080",
    model_id="meta-llama/Llama-3.2-3B",
    api_key="my-secret-key",
    dp_epsilon=1.0,        # strong local DP
    dp_max_norm=1.0,       # clip before noising
)
```

### Full training loop

```bash
chorus train \
  --server http://localhost:8080 \
  --model meta-llama/Llama-3.2-3B \
  --dataset wikitext \
  --rounds 10 \
  --lora-rank 16
```

### Production readiness

Chorus is **alpha software** in v0.2.0. Suitable for research, small internal federations, and benchmarks; not hardened for multi-tenant production. Single-process FastAPI server, in-memory rate limiter, filesystem storage, HTTP (terminate TLS at a reverse proxy). API keys are global — run one server per trust boundary. See [docs/honest-tradeoffs.md#production-readiness](docs/honest-tradeoffs.md#production-readiness) for the full list of caveats and what's on the roadmap to fix each.

## Examples

See the [`examples/`](examples/) directory:

- **[`quickstart.py`](examples/quickstart.py)** — Basic 2-client workflow with synthetic adapters
- **[`health_metrics/federated_health.py`](examples/health_metrics/federated_health.py)** — Multi-hospital federated training simulation with DP

## Development

```bash
git clone https://github.com/varmabudharaju/chorus.git
cd chorus
pip install -e ".[dev]"

# Run tests (165 tests)
pytest tests/ -v

# Run benchmarks
python benchmarks/benchmark.py
```

## Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and add tests
4. Run the test suite (`pytest tests/ -v`)
5. Submit a pull request

Please open an issue first to discuss significant changes.

## License

Apache 2.0 — see [LICENSE](LICENSE) for the full text.
