"""Tests for aggregation algorithms."""

import torch
import pytest

from chorus.server.aggregation import FedAvg, FedExLoRA, get_strategy
from chorus.patterns import get_layer_pairs as _get_layer_pairs


def _make_delta(layers, rank=4, dim=16, seed=None):
    """Create a synthetic LoRA delta."""
    if seed is not None:
        torch.manual_seed(seed)
    tensors = {}
    for name in layers:
        tensors[f"{name}.lora_A.weight"] = torch.randn(rank, dim)
        tensors[f"{name}.lora_B.weight"] = torch.randn(dim, rank)
    return tensors


LAYERS = ["layer.0.attn.q_proj", "layer.0.attn.v_proj"]


class TestFedAvg:
    def test_single_client(self):
        delta = _make_delta(LAYERS, seed=42)
        strategy = FedAvg()
        result = strategy.aggregate([delta])
        for key in delta:
            assert torch.allclose(result[key], delta[key])

    def test_two_clients_equal_weight(self):
        d1 = _make_delta(LAYERS, seed=1)
        d2 = _make_delta(LAYERS, seed=2)
        strategy = FedAvg()
        result = strategy.aggregate([d1, d2])
        for key in d1:
            expected = (d1[key] + d2[key]) / 2
            assert torch.allclose(result[key], expected, atol=1e-6)

    def test_weighted_average(self):
        d1 = _make_delta(LAYERS, seed=1)
        d2 = _make_delta(LAYERS, seed=2)
        strategy = FedAvg()
        result = strategy.aggregate([d1, d2], weights=[3.0, 1.0])
        for key in d1:
            expected = 0.75 * d1[key] + 0.25 * d2[key]
            assert torch.allclose(result[key], expected, atol=1e-6)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No deltas"):
            FedAvg().aggregate([])

    def test_is_inexact(self):
        """Demonstrate that FedAvg is mathematically inexact for LoRA."""
        deltas = [_make_delta(LAYERS, seed=i) for i in range(5)]
        strategy = FedAvg()
        result = strategy.aggregate(deltas)

        pairs = _get_layer_pairs(deltas[0])
        for layer_name, (a_key, b_key) in pairs.items():
            # Exact average of B @ A
            exact = torch.zeros_like(deltas[0][b_key] @ deltas[0][a_key])
            for d in deltas:
                exact += (d[b_key] @ d[a_key]) / len(deltas)

            # FedAvg reconstruction
            naive = result[b_key] @ result[a_key]

            # These should NOT be equal (FedAvg is broken for LoRA)
            error = torch.norm(exact - naive).item()
            assert error > 1e-6, "FedAvg should have non-zero aggregation error"


class TestFedExLoRA:
    def test_single_client_exact(self):
        delta = _make_delta(LAYERS, seed=42)
        strategy = FedExLoRA()
        result = strategy.aggregate([delta])

        pairs = _get_layer_pairs(delta)
        for layer_name, (a_key, b_key) in pairs.items():
            exact = delta[b_key] @ delta[a_key]
            recon = result[b_key] @ result[a_key]
            assert torch.allclose(exact, recon, atol=1e-5)

    def test_multi_client_more_exact_than_fedavg(self):
        """FedEx-LoRA should have lower aggregation error than FedAvg."""
        deltas = [_make_delta(LAYERS, seed=i) for i in range(5)]

        fedavg = FedAvg()
        fedex = FedExLoRA()

        avg_result = fedavg.aggregate(deltas)
        ex_result = fedex.aggregate(deltas)

        pairs = _get_layer_pairs(deltas[0])
        for layer_name, (a_key, b_key) in pairs.items():
            exact = torch.zeros_like(deltas[0][b_key].float() @ deltas[0][a_key].float())
            for d in deltas:
                exact += (d[b_key].float() @ d[a_key].float()) / len(deltas)

            fedavg_err = torch.norm(exact - avg_result[b_key].float() @ avg_result[a_key].float()).item()
            fedex_err = torch.norm(exact - ex_result[b_key].float() @ ex_result[a_key].float()).item()

            assert fedex_err < fedavg_err, (
                f"FedEx-LoRA error ({fedex_err}) should be less than FedAvg ({fedavg_err})"
            )

    def test_preserves_dtype(self):
        delta = _make_delta(LAYERS, seed=42)
        # Convert to float16
        delta = {k: v.half() for k, v in delta.items()}
        strategy = FedExLoRA()
        result = strategy.aggregate([delta])
        for key in result:
            assert result[key].dtype == torch.float16

    def test_residual_tracking(self):
        strategy = FedExLoRA()
        deltas = [_make_delta(LAYERS, seed=i) for i in range(3)]
        strategy.aggregate(deltas)
        residuals = strategy.get_residuals()
        # After one round, residuals should exist
        assert len(residuals) > 0

    def test_reset_residuals(self):
        strategy = FedExLoRA()
        deltas = [_make_delta(LAYERS, seed=i) for i in range(3)]
        strategy.aggregate(deltas)
        strategy.reset_residuals()
        assert len(strategy.get_residuals()) == 0


class TestGetStrategy:
    def test_fedavg(self):
        s = get_strategy("fedavg")
        assert isinstance(s, FedAvg)

    def test_fedex_lora(self):
        s = get_strategy("fedex-lora")
        assert isinstance(s, FedExLoRA)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("nonexistent")
