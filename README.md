# Chorus — Federated LoRA Adapter Aggregation

Federated fine-tuning of LLMs using LoRA, with **mathematically correct aggregation**.

Standard FedAvg is broken for LoRA because `avg(B*A) ≠ avg(B)*avg(A)`. Chorus implements **FedEx-LoRA** (ACL/ICLR 2025), which provides exact federated aggregation of LoRA adapters.

## Install

```bash
pip install chorus
```

## Quick Start

### 1. Start the aggregation server

```bash
chorus server --model meta-llama/Llama-3.2-3B --min-deltas 3
```

### 2. Submit LoRA adapters from clients

```python
from chorus import ChorusClient

client = ChorusClient(server="http://localhost:8080", model_id="meta-llama/Llama-3.2-3B")

# After your normal PEFT/LoRA training...
client.submit_delta(adapter_path="./my-adapter")

# Pull the aggregated global adapter
client.pull_latest(output_path="./updated-adapter")
```

Or via CLI:

```bash
chorus submit --server http://localhost:8080 --adapter ./my-adapter
chorus pull --server http://localhost:8080 --output ./updated-adapter
```

### 3. Run a simulation (no server needed)

```bash
# Basic simulation
chorus simulate --clients 10 --rounds 5

# Compare FedAvg vs FedEx-LoRA
chorus simulate --clients 10 --rounds 5 --compare
```

## Why FedEx-LoRA?

LoRA decomposes weight updates as `ΔW = B @ A` (two low-rank matrices). When you naively average across clients:

```
avg(B_i @ A_i) ≠ avg(B_i) @ avg(A_i)
```

This means **FedAvg produces mathematically inexact aggregation** for LoRA. FedEx-LoRA fixes this by:

1. Computing the exact weighted average of full-rank products `B_i @ A_i`
2. Tracking the residual between the exact and approximate (LoRA-structured) result
3. Folding the residual back into the B matrices via pseudoinverse correction

Result: **exact aggregation** with no approximation error.

## Differential Privacy

Chorus supports optional differential privacy at both client and server side:

```python
# Client-side DP (noise added before sending to server)
client = ChorusClient(
    server="http://localhost:8080",
    model_id="my-model",
    dp_epsilon=1.0,
)
```

```bash
# Server-side DP
chorus server --model my-model --dp-epsilon 1.0
```

## Architecture

```
Client 1                Aggregation Server              Client 2
┌──────────────┐       ┌──────────────────┐       ┌──────────────┐
│ Local data    │ POST  │ FastAPI           │ POST  │ Local data    │
│ LoRA training ├──────→│ /rounds/{id}/deltas│←──────┤ LoRA training │
│ submit_delta()│       │ → FedEx-LoRA agg  │       │ submit_delta()│
│ pull_latest() ├──────→│ → serve result    │←──────┤ pull_latest() │
└──────────────┘  GET  └──────────────────┘  GET  └──────────────┘
```

## Aggregation Strategies

| Strategy | Exact? | Description |
|----------|--------|-------------|
| `fedavg` | No | Naive averaging of A and B independently. Baseline only. |
| `fedex-lora` | Yes | Exact aggregation with residual correction (default). |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/models/{id}/status` | GET | Model round status |
| `/rounds/{id}/deltas` | POST | Submit a LoRA delta |
| `/models/{id}/latest` | GET | Download latest aggregated adapter |
| `/models/{id}/rounds/{id}` | GET | Download specific round's adapter |

## CLI Commands

```bash
chorus server    # Start aggregation server
chorus submit    # Submit a delta
chorus pull      # Pull aggregated adapter
chorus simulate  # Run local simulation
```

## Development

```bash
git clone https://github.com/chorus/chorus
cd chorus
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0
