#!/usr/bin/env python3
"""Benchmark FedEx-LoRA vs FedAvg across configurations.

Usage:
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --with-dp
    python benchmarks/benchmark.py --realistic
"""

from __future__ import annotations

import argparse
import time

import torch
from rich.console import Console
from rich.table import Table

from chorus.server.aggregation import FedAvg, FedExLoRA, norm_bound_deltas
from chorus.patterns import get_layer_pairs

console = Console()


def make_delta(layers, rank, dim, seed, scale=0.01):
    """Generate a synthetic LoRA delta (scale mimics real training magnitudes)."""
    torch.manual_seed(seed)
    tensors = {}
    for name in layers:
        tensors[f"{name}.lora_A.weight"] = torch.randn(rank, dim) * scale
        tensors[f"{name}.lora_B.weight"] = torch.randn(dim, rank) * scale
    return tensors


def compute_exact_avg(deltas, weights=None):
    n = len(deltas)
    if weights is None:
        weights = [1.0 / n] * n
    pairs = get_layer_pairs(deltas[0])
    exact = {}
    for layer_name, (a_key, b_key) in pairs.items():
        product = torch.zeros_like(deltas[0][b_key].float() @ deltas[0][a_key].float())
        for i, d in enumerate(deltas):
            product += weights[i] * (d[b_key].float() @ d[a_key].float())
        exact[layer_name] = product
    return exact


def measure_error(result, deltas, weights=None):
    """Returns (max_abs_error, max_relative_error, mean_relative_error) across layers."""
    exact = compute_exact_avg(deltas, weights)
    pairs = get_layer_pairs(result)
    abs_errors = []
    rel_errors = []
    for layer_name, (a_key, b_key) in pairs.items():
        recon = result[b_key].float() @ result[a_key].float()
        abs_err = torch.norm(exact[layer_name] - recon).item()
        target_norm = torch.norm(exact[layer_name]).item()
        rel_err = abs_err / max(target_norm, 1e-12)
        abs_errors.append(abs_err)
        rel_errors.append(rel_err)
    return max(abs_errors), max(rel_errors), sum(rel_errors) / len(rel_errors)


REALISTIC_LAYERS = [
    f"model.layers.{i}.self_attn.{proj}_proj"
    for i in range(4)
    for proj in ["q", "k", "v", "o"]
]

BASIC_LAYERS = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.1.self_attn.q_proj",
    "model.layers.1.self_attn.v_proj",
]


def run_benchmark(configs, layers, dp_epsilon=None):
    table = Table(title="FedEx-LoRA vs FedAvg Benchmark")
    table.add_column("Clients", style="cyan", justify="right")
    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("Dim", style="cyan", justify="right")
    table.add_column("Layers", style="cyan", justify="right")
    table.add_column("FedAvg RelErr", style="red", justify="right")
    table.add_column("FedEx RelErr", style="green", justify="right")
    table.add_column("Improvement", style="yellow", justify="right")
    table.add_column("FedEx Time", style="dim", justify="right")

    for n_clients, rank, dim in configs:
        deltas = [make_delta(layers, rank, dim, seed=i) for i in range(n_clients)]

        if dp_epsilon is not None:
            from chorus.server.privacy import apply_dp
            deltas = [apply_dp(d, epsilon=dp_epsilon) for d in deltas]

        # FedAvg
        fedavg = FedAvg()
        avg_result = fedavg.aggregate(deltas)
        _, fedavg_rel, _ = measure_error(avg_result, deltas)

        # FedEx-LoRA
        fedex = FedExLoRA()
        t0 = time.perf_counter()
        ex_result = fedex.aggregate(deltas)
        dt = time.perf_counter() - t0
        _, fedex_rel, _ = measure_error(ex_result, deltas)

        improvement = fedavg_rel / max(fedex_rel, 1e-12)

        table.add_row(
            str(n_clients),
            str(rank),
            str(dim),
            str(len(layers)),
            f"{fedavg_rel:.4f}",
            f"{fedex_rel:.4f}",
            f"{improvement:.2f}x",
            f"{dt * 1000:.1f}ms",
        )

    console.print(table)


def run_multi_round_benchmark(n_clients, n_rounds, rank, dim, layers):
    """Show how residuals accumulate across rounds."""
    table = Table(title=f"Multi-Round Benchmark ({n_clients} clients, rank={rank}, dim={dim})")
    table.add_column("Round", style="cyan", justify="right")
    table.add_column("FedAvg RelErr", style="red", justify="right")
    table.add_column("FedEx RelErr", style="green", justify="right")
    table.add_column("Residual Norm", style="yellow", justify="right")
    table.add_column("Improvement", style="yellow", justify="right")

    fedex = FedExLoRA()

    for round_id in range(n_rounds):
        deltas = [make_delta(layers, rank, dim, seed=round_id * n_clients + i) for i in range(n_clients)]

        fedavg_result = FedAvg().aggregate(deltas)
        _, fedavg_rel, _ = measure_error(fedavg_result, deltas)

        ex_result = fedex.aggregate(deltas)
        _, fedex_rel, _ = measure_error(ex_result, deltas)

        residuals = fedex.get_residuals()
        max_residual = max(torch.norm(r).item() for r in residuals.values()) if residuals else 0

        table.add_row(
            str(round_id),
            f"{fedavg_rel:.4f}",
            f"{fedex_rel:.4f}",
            f"{max_residual:.6f}",
            f"{fedavg_rel / max(fedex_rel, 1e-12):.2f}x",
        )

    console.print(table)


def run_byzantine_benchmark(n_clients, rank, dim, layers):
    """Show effect of poisoned clients with and without norm bounding."""
    table = Table(title=f"Byzantine Robustness ({n_clients} clients, 1 poisoned)")
    table.add_column("Defense", style="cyan")
    table.add_column("FedEx RelErr", style="green", justify="right")
    table.add_column("Result Finite", style="yellow", justify="right")

    deltas = [make_delta(layers, rank, dim, seed=i) for i in range(n_clients)]

    # Compute typical norm
    norms = []
    for d in deltas:
        flat = torch.cat([t.float().flatten() for t in d.values()])
        norms.append(torch.norm(flat).item())
    typical_norm = sum(norms) / len(norms)

    # Poison client 0
    deltas[0] = {k: v * 10000 for k, v in deltas[0].items()}

    # No defense
    result = FedExLoRA().aggregate(deltas)
    _, rel_err, _ = measure_error(result, deltas)
    finite = all(torch.isfinite(t).all().item() for t in result.values())
    table.add_row("None", f"{rel_err:.4f}", str(finite))

    # Norm bounding
    bounded = norm_bound_deltas(deltas, max_norm=typical_norm * 2)
    result = FedExLoRA().aggregate(bounded)
    _, rel_err, _ = measure_error(result, bounded)
    finite = all(torch.isfinite(t).all().item() for t in result.values())
    table.add_row(f"Norm bound ({typical_norm * 2:.1f})", f"{rel_err:.4f}", str(finite))

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Benchmark FedEx-LoRA vs FedAvg")
    parser.add_argument("--realistic", action="store_true", help="Use realistic layer counts (16 layers)")
    parser.add_argument("--with-dp", action="store_true", help="Apply DP noise (epsilon=1.0)")
    parser.add_argument("--multi-round", action="store_true", help="Run multi-round benchmark")
    parser.add_argument("--byzantine", action="store_true", help="Run Byzantine robustness benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    layers = REALISTIC_LAYERS if args.realistic else BASIC_LAYERS

    if args.all or not any([args.multi_round, args.byzantine]):
        console.print("\n[bold]== Configuration Sweep ==[/bold]\n")
        configs = [
            # (clients, rank, dim)
            (2, 4, 64),
            (5, 4, 64),
            (10, 4, 64),
            (5, 8, 128),
            (5, 16, 256),
            (10, 16, 256),
            (20, 8, 128),
            (50, 8, 128),
        ]
        dp = 1.0 if args.with_dp else None
        run_benchmark(configs, layers, dp_epsilon=dp)

    if args.all or args.multi_round:
        console.print("\n[bold]== Multi-Round Residual Tracking ==[/bold]\n")
        run_multi_round_benchmark(5, 10, rank=8, dim=128, layers=layers)

    if args.all or args.byzantine:
        console.print("\n[bold]== Byzantine Robustness ==[/bold]\n")
        run_byzantine_benchmark(10, rank=8, dim=128, layers=layers)


if __name__ == "__main__":
    main()
