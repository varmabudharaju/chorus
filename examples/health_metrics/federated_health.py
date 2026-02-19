"""Federated Health Metrics Example

Demonstrates federated LoRA fine-tuning for a health domain use case.
Multiple hospitals train on their private patient data, then aggregate
their adapters without sharing raw data.

This is a simulation — no real health data or model weights are used.

Usage:
    # Option 1: Run the full simulation locally (no server needed)
    fedlora simulate --clients 5 --rounds 3 --compare

    # Option 2: Run with a server
    # Terminal 1: fedlora server --model health-v1 --min-deltas 3
    # Terminal 2: python examples/health_metrics/federated_health.py
"""

import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from fedlora import FedLoRAClient
from fedlora.simulate.runner import run_simulation


def run_local_simulation():
    """Run a local simulation comparing FedAvg vs FedEx-LoRA.

    This demonstrates the mathematical advantage of FedEx-LoRA
    without needing a running server.
    """
    print("=" * 60)
    print("Federated Health Model — Local Simulation")
    print("=" * 60)
    print()
    print("Scenario: 5 hospitals, each with private patient data,")
    print("collaboratively fine-tune a health LLM using LoRA.")
    print("No raw data is shared — only adapter weight updates.")
    print()

    # Run comparison
    result = run_simulation(
        num_clients=5,
        num_rounds=3,
        rank=16,
        hidden_dim=512,
        compare_strategies=True,
    )

    print("\nResults:")
    print("-" * 50)
    for r in result.rounds:
        print(
            f"Round {r['round_id']}: "
            f"FedAvg error={r['fedavg_error']:.6f}, "
            f"FedEx-LoRA error={r['fedex_error']:.6f} "
            f"({r['improvement']} better)"
        )

    print()
    print("FedEx-LoRA produces mathematically exact aggregation,")
    print("while FedAvg introduces approximation error due to the")
    print("non-linearity of LoRA's low-rank decomposition (B @ A).")

    # Also run with differential privacy
    print()
    print("=" * 60)
    print("With Differential Privacy (epsilon=1.0)")
    print("=" * 60)

    dp_result = run_simulation(
        num_clients=5,
        num_rounds=3,
        rank=16,
        hidden_dim=512,
        strategy="fedex-lora",
        dp_epsilon=1.0,
    )
    print(dp_result.summary())
    print()
    print("With DP enabled, each client's contribution is clipped")
    print("and noised before submission, providing formal privacy")
    print("guarantees even if the server is compromised.")


def run_with_server():
    """Run with a live FedLoRA server (must be started separately).

    Start the server first:
        fedlora server --model health-v1 --min-deltas 3 --strategy fedex-lora
    """
    SERVER = "http://localhost:8080"
    MODEL = "health-v1"

    hospitals = ["hospital_a", "hospital_b", "hospital_c"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for hospital in hospitals:
            print(f"\n[{hospital}] Training on private data...")
            # Simulate training
            torch.manual_seed(hash(hospital) % 2**32)
            adapter_dir = tmpdir / hospital
            adapter_dir.mkdir()
            tensors = {
                f"base_model.model.model.layers.{l}.self_attn.{p}.lora_{ab}.default.weight":
                    torch.randn(16 if ab == "A" else 512, 512 if ab == "A" else 16) * 0.01
                for l in range(4) for p in ["q_proj", "v_proj"] for ab in ["A", "B"]
            }
            save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))

            # Submit
            client = FedLoRAClient(
                server=SERVER,
                model_id=MODEL,
                client_id=hospital,
                dp_epsilon=2.0,  # Local DP for extra privacy
            )
            result = client.submit_delta(adapter_path=adapter_dir)
            print(f"[{hospital}] Submitted ({result['deltas_received']}/{result['min_deltas']})")

            if result["aggregated"]:
                print(f"\n[Aggregation triggered!]")
                for h in hospitals:
                    out = tmpdir / f"{h}_updated"
                    client.pull_latest(output_path=out)
                    print(f"  [{h}] Pulled aggregated adapter → {out}")

            client.close()


if __name__ == "__main__":
    import sys

    if "--server" in sys.argv:
        run_with_server()
    else:
        run_local_simulation()
