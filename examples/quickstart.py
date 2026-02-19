"""Chorus Quickstart Example

This demonstrates the basic Chorus workflow:
1. Start the aggregation server (in a separate terminal)
2. Multiple clients train LoRA adapters locally
3. Clients submit their deltas to the server
4. Server aggregates using FedEx-LoRA
5. Clients pull the improved global adapter

Run the server first:
    chorus server --model my-model --min-deltas 2

Then run this script (simulates 2 clients):
    python examples/quickstart.py
"""

import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from chorus import ChorusClient


SERVER_URL = "http://localhost:8080"
MODEL_ID = "my-model"


def simulate_local_training(client_id: str, output_dir: Path) -> Path:
    """Simulate local LoRA training by creating synthetic adapter weights.

    In a real scenario, you'd use PEFT/HuggingFace to train a LoRA adapter
    on your local private data.
    """
    torch.manual_seed(hash(client_id) % 2**32)

    # Create a fake PEFT adapter directory
    adapter_dir = output_dir / f"adapter_{client_id}"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Simulate LoRA A and B matrices (as PEFT would produce)
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(8, 256) * 0.01,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(256, 8) * 0.01,
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight": torch.randn(8, 256) * 0.01,
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight": torch.randn(256, 8) * 0.01,
    }
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))

    print(f"[{client_id}] Local training complete → {adapter_dir}")
    return adapter_dir


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # --- Client 1: Train and submit ---
        print("\n=== Client 1 ===")
        adapter_1 = simulate_local_training("client_1", tmpdir)

        client_1 = ChorusClient(
            server=SERVER_URL,
            model_id=MODEL_ID,
            client_id="client_1",
        )
        result = client_1.submit_delta(adapter_path=adapter_1)
        print(f"[client_1] Submitted: {result['deltas_received']}/{result['min_deltas']} deltas")

        # --- Client 2: Train and submit ---
        print("\n=== Client 2 ===")
        adapter_2 = simulate_local_training("client_2", tmpdir)

        client_2 = ChorusClient(
            server=SERVER_URL,
            model_id=MODEL_ID,
            client_id="client_2",
        )
        result = client_2.submit_delta(adapter_path=adapter_2)
        print(f"[client_2] Submitted: {result['deltas_received']}/{result['min_deltas']} deltas")

        if result["aggregated"]:
            print("\n=== Aggregation triggered! ===")

            # Both clients pull the aggregated adapter
            output_1 = tmpdir / "updated_1"
            client_1.pull_latest(output_path=output_1)
            print(f"[client_1] Pulled aggregated adapter → {output_1}")

            output_2 = tmpdir / "updated_2"
            client_2.pull_latest(output_path=output_2)
            print(f"[client_2] Pulled aggregated adapter → {output_2}")

        client_1.close()
        client_2.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
