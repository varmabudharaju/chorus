"""Simulated multi-client federation for testing and demos."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from chorus.server.aggregation import AggregationStrategy, get_strategy
from chorus.server.privacy import apply_dp
from chorus.server.storage import DeltaStorage

logger = logging.getLogger("chorus.simulate")


def generate_synthetic_lora_delta(
    layer_names: list[str],
    rank: int = 8,
    hidden_dim: int = 256,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Generate a synthetic LoRA delta for testing.

    Creates random A and B matrices for each layer name, simulating
    what a real LoRA training run would produce.
    """
    if seed is not None:
        torch.manual_seed(seed)

    tensors = {}
    for name in layer_names:
        # A: (rank, hidden_dim), B: (hidden_dim, rank) — standard LoRA shapes
        tensors[f"{name}.lora_A.weight"] = torch.randn(rank, hidden_dim) * 0.01
        tensors[f"{name}.lora_B.weight"] = torch.randn(hidden_dim, rank) * 0.01
    return tensors


DEFAULT_LAYERS = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.1.self_attn.q_proj",
    "model.layers.1.self_attn.v_proj",
]


class SimulationResult:
    """Results from a simulated federation run."""

    def __init__(self):
        self.rounds: list[dict] = []
        self.final_adapter: dict[str, torch.Tensor] | None = None

    def summary(self) -> str:
        lines = [f"Simulation completed: {len(self.rounds)} rounds"]
        for r in self.rounds:
            lines.append(
                f"  Round {r['round_id']}: {r['num_clients']} clients, "
                f"strategy={r['strategy']}"
            )
        if self.final_adapter:
            num_params = sum(t.numel() for t in self.final_adapter.values())
            lines.append(f"  Final adapter: {len(self.final_adapter)} tensors, {num_params:,} parameters")
        return "\n".join(lines)


def run_simulation(
    num_clients: int = 5,
    num_rounds: int = 3,
    strategy: str = "fedex-lora",
    rank: int = 8,
    hidden_dim: int = 256,
    layer_names: list[str] | None = None,
    dp_epsilon: float | None = None,
    dp_delta: float = 1e-5,
    dp_max_norm: float = 1.0,
    data_dir: str | None = None,
    compare_strategies: bool = False,
) -> SimulationResult:
    """Run a simulated multi-client federation.

    Each client generates synthetic LoRA deltas (random A and B matrices),
    submits them, and the server aggregates after each round.

    Args:
        num_clients: Number of simulated clients per round.
        num_rounds: Number of federation rounds.
        strategy: Aggregation strategy ("fedavg" or "fedex-lora").
        rank: LoRA rank for synthetic adapters.
        hidden_dim: Hidden dimension for synthetic adapters.
        layer_names: Layer names for synthetic adapters. Defaults to 4 layers.
        dp_epsilon: If set, apply DP noise to each client's delta.
        dp_delta: DP delta parameter.
        dp_max_norm: DP clipping norm.
        data_dir: Directory for storage. Uses a temp dir if None.
        compare_strategies: If True, run both FedAvg and FedEx-LoRA and compare.

    Returns:
        SimulationResult with per-round details.
    """
    layers = layer_names or DEFAULT_LAYERS

    if compare_strategies:
        return _run_comparison(
            num_clients, num_rounds, rank, hidden_dim, layers,
            dp_epsilon, dp_delta, dp_max_norm,
        )

    use_temp = data_dir is None
    if use_temp:
        tmp = tempfile.mkdtemp(prefix="chorus_sim_")
        data_dir = tmp

    storage = DeltaStorage(data_dir)
    agg_strategy = get_strategy(strategy)
    model_id = "simulation"
    result = SimulationResult()

    for round_id in range(num_rounds):
        logger.info(f"=== Round {round_id + 1}/{num_rounds} ===")

        # Each client generates and submits a delta
        for client_idx in range(num_clients):
            delta = generate_synthetic_lora_delta(
                layers, rank=rank, hidden_dim=hidden_dim,
                seed=round_id * num_clients + client_idx,
            )

            if dp_epsilon is not None:
                delta = apply_dp(delta, epsilon=dp_epsilon, delta=dp_delta, max_norm=dp_max_norm)

            storage.save_delta(model_id, round_id, f"client_{client_idx}", delta)

        # Aggregate
        deltas = storage.load_all_deltas(model_id, round_id)
        aggregated = agg_strategy.aggregate(deltas)
        storage.save_aggregated(model_id, round_id, aggregated)

        result.rounds.append({
            "round_id": round_id,
            "num_clients": num_clients,
            "strategy": agg_strategy.name,
            "num_tensors": len(aggregated),
        })

        logger.info(f"Round {round_id} aggregated: {len(aggregated)} tensors")

    result.final_adapter = storage.load_aggregated(model_id)
    return result


def _run_comparison(
    num_clients: int,
    num_rounds: int,
    rank: int,
    hidden_dim: int,
    layers: list[str],
    dp_epsilon: float | None,
    dp_delta: float,
    dp_max_norm: float,
) -> SimulationResult:
    """Run both FedAvg and FedEx-LoRA on identical data and compare."""
    result = SimulationResult()

    for round_id in range(num_rounds):
        # Generate identical deltas for both strategies
        client_deltas = []
        for client_idx in range(num_clients):
            delta = generate_synthetic_lora_delta(
                layers, rank=rank, hidden_dim=hidden_dim,
                seed=round_id * num_clients + client_idx,
            )
            if dp_epsilon is not None:
                delta = apply_dp(delta, epsilon=dp_epsilon, delta=dp_delta, max_norm=dp_max_norm)
            client_deltas.append(delta)

        # Compute exact average: avg(B_i @ A_i)
        from chorus.patterns import get_layer_pairs as _get_layer_pairs
        layer_pairs = _get_layer_pairs(client_deltas[0])

        # Run both strategies
        fedavg = get_strategy("fedavg")
        fedex = get_strategy("fedex-lora")

        avg_result = fedavg.aggregate(client_deltas)
        ex_result = fedex.aggregate(client_deltas)

        # Compare: measure how close each is to the exact full-rank average
        errors_fedavg = []
        errors_fedex = []

        for layer_name, (a_key, b_key) in layer_pairs.items():
            # Exact average of B_i @ A_i
            exact = torch.zeros_like(client_deltas[0][b_key].float() @ client_deltas[0][a_key].float())
            for d in client_deltas:
                exact += (d[b_key].float() @ d[a_key].float()) / num_clients

            # FedAvg reconstruction
            fedavg_recon = avg_result[b_key].float() @ avg_result[a_key].float()
            fedavg_err = torch.norm(exact - fedavg_recon).item()
            errors_fedavg.append(fedavg_err)

            # FedEx-LoRA reconstruction
            fedex_recon = ex_result[b_key].float() @ ex_result[a_key].float()
            fedex_err = torch.norm(exact - fedex_recon).item()
            errors_fedex.append(fedex_err)

        avg_fedavg_err = sum(errors_fedavg) / len(errors_fedavg)
        avg_fedex_err = sum(errors_fedex) / len(errors_fedex)

        result.rounds.append({
            "round_id": round_id,
            "num_clients": num_clients,
            "strategy": "comparison",
            "fedavg_error": avg_fedavg_err,
            "fedex_error": avg_fedex_err,
            "improvement": f"{avg_fedavg_err / max(avg_fedex_err, 1e-12):.2f}x",
        })

        logger.info(
            f"Round {round_id}: FedAvg error={avg_fedavg_err:.6f}, "
            f"FedEx-LoRA error={avg_fedex_err:.6f}, "
            f"improvement={avg_fedavg_err / max(avg_fedex_err, 1e-12):.2f}x"
        )

    result.final_adapter = ex_result
    return result
