"""Tests for eval metrics (Frobenius reconstruction error + task metrics)."""

import torch
import pytest

from chorus.eval.metrics import frobenius_reconstruction_error


def _make_lora_delta(rank: int = 4, dim: int = 16, seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "layer.0.lora_A.weight": torch.randn(rank, dim),
        "layer.0.lora_B.weight": torch.randn(dim, rank),
    }


def test_frobenius_error_zero_for_single_client():
    """A single client's aggregation should reconstruct exactly (FedEx-LoRA invariant)."""
    from chorus.server.aggregation import FedExLoRA

    delta = _make_lora_delta(rank=4, dim=16, seed=0)
    result = FedExLoRA().aggregate([delta])
    err = frobenius_reconstruction_error(result, [delta])
    assert err < 1e-4, f"Single-client FedEx-LoRA must reconstruct exactly; got {err}"


def test_frobenius_error_positive_for_multiple_clients_fedavg():
    """FedAvg has nonzero reconstruction error with >1 clients."""
    from chorus.server.aggregation import FedAvg

    deltas = [_make_lora_delta(seed=i) for i in range(3)]
    result = FedAvg().aggregate(deltas)
    err = frobenius_reconstruction_error(result, deltas)
    assert err > 0.01, f"Multi-client FedAvg should have nonzero error; got {err}"


def test_frobenius_error_lower_for_fedex_than_fedavg():
    """The whole point of FedEx-LoRA: lower reconstruction error than FedAvg."""
    from chorus.server.aggregation import FedAvg, FedExLoRA

    deltas = [_make_lora_delta(seed=i) for i in range(5)]
    fedavg_err = frobenius_reconstruction_error(FedAvg().aggregate(deltas), deltas)
    fedex_err = frobenius_reconstruction_error(FedExLoRA().aggregate(deltas), deltas)
    assert fedex_err < fedavg_err
