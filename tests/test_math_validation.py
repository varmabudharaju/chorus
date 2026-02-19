"""Rigorous mathematical validation tests for FedEx-LoRA.

Tests verify:
1. SVD-based FedEx-LoRA always produces lower error than FedAvg
2. Single client is identity (exact reconstruction)
3. Multi-round residual accumulation works correctly
4. Edge cases: rank-deficient A, single client, many clients
5. Numerical stability at float16
6. Byzantine robustness utilities
7. Residual persistence round-trip

Note on error expectations:
    The average of N rank-r matrices has rank up to N*r, which cannot be
    exactly represented in rank r. The SVD gives the optimal rank-r approx
    (Eckart-Young theorem), so FedEx-LoRA will always have SOME error with
    N > 1 clients. What we verify is that it's always better than FedAvg.
"""

import torch
import pytest

from chorus.server.aggregation import (
    FedAvg,
    FedExLoRA,
    norm_bound_deltas,
    detect_outlier_deltas,
    filter_outlier_deltas,
)
from chorus.patterns import get_layer_pairs


LAYERS = ["layer.0.attn.q_proj", "layer.0.attn.v_proj"]


def _make_delta(layers, rank=4, dim=16, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    tensors = {}
    for name in layers:
        tensors[f"{name}.lora_A.weight"] = torch.randn(rank, dim)
        tensors[f"{name}.lora_B.weight"] = torch.randn(dim, rank)
    return tensors


def _compute_exact_avg(client_deltas, weights=None):
    """Compute the ground-truth exact average of B_i @ A_i."""
    n = len(client_deltas)
    if weights is None:
        weights = [1.0 / n] * n

    pairs = get_layer_pairs(client_deltas[0])
    exact = {}
    for layer_name, (a_key, b_key) in pairs.items():
        product = torch.zeros_like(client_deltas[0][b_key].float() @ client_deltas[0][a_key].float())
        for i, d in enumerate(client_deltas):
            product += weights[i] * (d[b_key].float() @ d[a_key].float())
        exact[layer_name] = product
    return exact


def _max_reconstruction_error(result, client_deltas, weights=None):
    """Compute max reconstruction error across all layers."""
    exact = _compute_exact_avg(client_deltas, weights)
    pairs = get_layer_pairs(result)
    errors = []
    for layer_name, (a_key, b_key) in pairs.items():
        recon = result[b_key].float() @ result[a_key].float()
        err = torch.norm(exact[layer_name] - recon).item()
        errors.append(err)
    return max(errors)


def _max_relative_error(result, client_deltas, weights=None):
    """Compute max relative reconstruction error across all layers."""
    exact = _compute_exact_avg(client_deltas, weights)
    pairs = get_layer_pairs(result)
    errors = []
    for layer_name, (a_key, b_key) in pairs.items():
        recon = result[b_key].float() @ result[a_key].float()
        abs_err = torch.norm(exact[layer_name] - recon).item()
        target_norm = torch.norm(exact[layer_name]).item()
        rel_err = abs_err / max(target_norm, 1e-12)
        errors.append(rel_err)
    return max(errors)


class TestFedExLoRABeatsBaseline:
    """Core property: FedEx-LoRA should always produce lower error than FedAvg."""

    def test_two_clients(self):
        deltas = [_make_delta(LAYERS, rank=4, dim=16, seed=i) for i in range(2)]
        fedavg_err = _max_reconstruction_error(FedAvg().aggregate(deltas), deltas)
        fedex_err = _max_reconstruction_error(FedExLoRA().aggregate(deltas), deltas)
        assert fedex_err < fedavg_err

    def test_five_clients(self):
        deltas = [_make_delta(LAYERS, rank=8, dim=64, seed=i) for i in range(5)]
        fedavg_err = _max_reconstruction_error(FedAvg().aggregate(deltas), deltas)
        fedex_err = _max_reconstruction_error(FedExLoRA().aggregate(deltas), deltas)
        assert fedex_err < fedavg_err

    def test_ten_clients(self):
        deltas = [_make_delta(LAYERS, rank=8, dim=128, seed=i) for i in range(10)]
        fedavg_err = _max_reconstruction_error(FedAvg().aggregate(deltas), deltas)
        fedex_err = _max_reconstruction_error(FedExLoRA().aggregate(deltas), deltas)
        assert fedex_err < fedavg_err

    def test_weighted_clients(self):
        deltas = [_make_delta(LAYERS, rank=4, dim=32, seed=i) for i in range(4)]
        weights = [0.5, 0.2, 0.2, 0.1]
        fedavg_err = _max_reconstruction_error(FedAvg().aggregate(deltas, weights), deltas, weights)
        fedex_err = _max_reconstruction_error(FedExLoRA().aggregate(deltas, weights), deltas, weights)
        assert fedex_err < fedavg_err

    def test_always_beats_fedavg_across_seeds(self):
        """Across 10 random trials, FedEx-LoRA should always beat FedAvg."""
        for trial in range(10):
            deltas = [_make_delta(LAYERS, rank=4, dim=32, seed=trial * 5 + i) for i in range(5)]
            fedavg_err = _max_reconstruction_error(FedAvg().aggregate(deltas), deltas)
            fedex_err = _max_reconstruction_error(FedExLoRA().aggregate(deltas), deltas)
            assert fedex_err <= fedavg_err + 1e-6, (
                f"Trial {trial}: FedEx ({fedex_err:.6f}) > FedAvg ({fedavg_err:.6f})"
            )


class TestFedExLoRASingleClient:
    """With one client, FedEx-LoRA must reproduce the input exactly."""

    def test_single_client_exact(self):
        delta = _make_delta(LAYERS, rank=4, dim=16, seed=42)
        result = FedExLoRA().aggregate([delta])
        pairs = get_layer_pairs(delta)
        for layer_name, (a_key, b_key) in pairs.items():
            original = delta[b_key] @ delta[a_key]
            recon = result[b_key] @ result[a_key]
            assert torch.allclose(original, recon, atol=1e-5), (
                f"Single client: layer {layer_name} not reproduced exactly"
            )

    def test_single_client_near_zero_residual(self):
        delta = _make_delta(LAYERS, rank=4, dim=16, seed=42)
        strategy = FedExLoRA()
        strategy.aggregate([delta])
        for name, r in strategy.get_residuals().items():
            # SVD of a rank-r matrix truncated to rank-r has small numerical error
            assert torch.norm(r).item() < 1e-4, f"Residual for {name} should be near 0 with 1 client"


class TestFedExLoRAResiduals:
    """Test multi-round residual accumulation behavior."""

    def test_residuals_exist_after_aggregation(self):
        deltas = [_make_delta(LAYERS, rank=4, dim=16, seed=i) for i in range(3)]
        strategy = FedExLoRA()
        strategy.aggregate(deltas)
        residuals = strategy.get_residuals()
        assert len(residuals) > 0

    def test_residuals_accumulate_across_rounds(self):
        strategy = FedExLoRA()

        # Round 1
        deltas_r1 = [_make_delta(LAYERS, rank=4, dim=16, seed=i) for i in range(3)]
        strategy.aggregate(deltas_r1)
        r1 = {k: v.clone() for k, v in strategy.get_residuals().items()}

        # Round 2 — residuals should change (they incorporate the previous residual)
        deltas_r2 = [_make_delta(LAYERS, rank=4, dim=16, seed=10 + i) for i in range(3)]
        strategy.aggregate(deltas_r2)
        r2 = strategy.get_residuals()

        assert len(r1) > 0 and len(r2) > 0
        # Residuals will differ between rounds
        for key in r1:
            assert not torch.allclose(r1[key], r2[key]), "Residuals should change across rounds"

    def test_init_with_preloaded_residuals(self):
        """FedExLoRA should accept pre-loaded residuals (for persistence)."""
        deltas = [_make_delta(LAYERS, rank=4, dim=16, seed=i) for i in range(3)]
        s1 = FedExLoRA()
        s1.aggregate(deltas)
        saved_residuals = s1.get_residuals()

        s2 = FedExLoRA(residuals=saved_residuals)
        loaded = s2.get_residuals()
        for key in saved_residuals:
            assert torch.allclose(saved_residuals[key], loaded[key])

    def test_reset_clears_residuals(self):
        deltas = [_make_delta(LAYERS, rank=4, dim=16, seed=i) for i in range(3)]
        strategy = FedExLoRA()
        strategy.aggregate(deltas)
        strategy.reset_residuals()
        assert len(strategy.get_residuals()) == 0


class TestFedExLoRAEdgeCases:
    """Test edge cases and numerical stability."""

    def test_rank_equals_dim(self):
        """When rank == dim, matrices are square. Should still work."""
        deltas = [_make_delta(LAYERS, rank=8, dim=8, seed=i) for i in range(3)]
        strategy = FedExLoRA()
        result = strategy.aggregate(deltas)
        assert len(result) == 4
        # With square matrices, exact avg IS rank-r representable, so error should be tiny
        err = _max_reconstruction_error(result, deltas)
        assert err < 1e-3

    def test_very_small_rank(self):
        """rank=1 should still work and beat FedAvg."""
        deltas = [_make_delta(LAYERS, rank=1, dim=32, seed=i) for i in range(4)]
        fedavg_err = _max_reconstruction_error(FedAvg().aggregate(deltas), deltas)
        fedex_err = _max_reconstruction_error(FedExLoRA().aggregate(deltas), deltas)
        assert fedex_err <= fedavg_err + 1e-6

    def test_large_rank_large_dim(self):
        """Realistic sizes: rank=16, dim=256. Should beat FedAvg."""
        deltas = [_make_delta(LAYERS, rank=16, dim=256, seed=i) for i in range(5)]
        fedavg_err = _max_reconstruction_error(FedAvg().aggregate(deltas), deltas)
        fedex_err = _max_reconstruction_error(FedExLoRA().aggregate(deltas), deltas)
        assert fedex_err < fedavg_err

    def test_float16_preserves_dtype(self):
        deltas = [_make_delta(LAYERS, rank=4, dim=16, seed=i) for i in range(3)]
        deltas = [{k: v.half() for k, v in d.items()} for d in deltas]
        result = FedExLoRA().aggregate(deltas)
        for key in result:
            assert result[key].dtype == torch.float16

    def test_float16_still_beats_fedavg(self):
        """Float16 inputs should still produce better results than FedAvg."""
        deltas = [_make_delta(LAYERS, rank=4, dim=32, seed=i) for i in range(3)]
        deltas_f16 = [{k: v.half() for k, v in d.items()} for d in deltas]

        fedavg_err = _max_reconstruction_error(FedAvg().aggregate(deltas_f16), deltas_f16)
        fedex_err = _max_reconstruction_error(FedExLoRA().aggregate(deltas_f16), deltas_f16)
        assert fedex_err <= fedavg_err + 1e-3  # small tolerance for f16 quantization

    def test_many_clients(self):
        """50 clients should work and beat FedAvg."""
        deltas = [_make_delta(LAYERS, rank=4, dim=32, seed=i) for i in range(50)]
        fedavg_err = _max_reconstruction_error(FedAvg().aggregate(deltas), deltas)
        fedex_err = _max_reconstruction_error(FedExLoRA().aggregate(deltas), deltas)
        assert fedex_err < fedavg_err

    def test_result_tensors_are_finite(self):
        """No NaN or Inf in outputs."""
        deltas = [_make_delta(LAYERS, rank=8, dim=64, seed=i) for i in range(5)]
        result = FedExLoRA().aggregate(deltas)
        for key, tensor in result.items():
            assert torch.isfinite(tensor).all(), f"Non-finite values in {key}"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No deltas"):
            FedExLoRA().aggregate([])


class TestNormBoundDeltas:
    def test_clips_large_deltas(self):
        deltas = [_make_delta(LAYERS, rank=4, dim=16, seed=i) for i in range(3)]
        deltas[1] = {k: v * 1000 for k, v in deltas[1].items()}
        clipped = norm_bound_deltas(deltas, max_norm=1.0)
        for d in clipped:
            flat = torch.cat([t.float().flatten() for t in d.values()])
            assert torch.norm(flat).item() <= 1.0 + 1e-5

    def test_preserves_small_deltas(self):
        deltas = [_make_delta(LAYERS, rank=4, dim=16, seed=i) for i in range(3)]
        clipped = norm_bound_deltas(deltas, max_norm=1e6)
        for orig, clip in zip(deltas, clipped):
            for key in orig:
                assert torch.allclose(orig[key], clip[key])

    def test_uniform_scaling(self):
        """All tensors in a clipped delta should be scaled by the same factor."""
        delta = _make_delta(LAYERS, rank=4, dim=16, seed=42)
        delta = {k: v * 1000 for k, v in delta.items()}
        clipped = norm_bound_deltas([delta], max_norm=1.0)[0]
        ratios = []
        for key in delta:
            orig_norm = torch.norm(delta[key].float()).item()
            clip_norm = torch.norm(clipped[key].float()).item()
            ratios.append(clip_norm / orig_norm)
        # All ratios should be the same
        assert all(abs(r - ratios[0]) < 1e-6 for r in ratios)


class TestOutlierDetection:
    def test_detects_outlier(self):
        deltas = [_make_delta(LAYERS, rank=4, dim=16, seed=i) for i in range(10)]
        deltas[5] = {k: v * 10000 for k, v in deltas[5].items()}
        outliers = detect_outlier_deltas(deltas, threshold=2.0)
        assert 5 in outliers

    def test_no_outliers_in_uniform_data(self):
        deltas = [_make_delta(LAYERS, rank=4, dim=16, seed=i) for i in range(20)]
        outliers = detect_outlier_deltas(deltas, threshold=3.0)
        assert len(outliers) <= 1

    def test_filter_removes_outliers(self):
        deltas = [_make_delta(LAYERS, rank=4, dim=16, seed=i) for i in range(10)]
        deltas[3] = {k: v * 10000 for k, v in deltas[3].items()}
        filtered = filter_outlier_deltas(deltas, threshold=2.0)
        assert len(filtered) < len(deltas)
        assert len(filtered) >= 9


class TestByzantineWithAggregation:
    def test_poisoned_client_bounded(self):
        deltas = [_make_delta(LAYERS, rank=4, dim=32, seed=i) for i in range(5)]
        norms = []
        for d in deltas:
            flat = torch.cat([t.float().flatten() for t in d.values()])
            norms.append(torch.norm(flat).item())
        typical_norm = sum(norms) / len(norms)

        deltas[2] = {k: v * 1000 for k, v in deltas[2].items()}
        bounded = norm_bound_deltas(deltas, max_norm=typical_norm * 2)
        result = FedExLoRA().aggregate(bounded)
        for key, tensor in result.items():
            assert torch.isfinite(tensor).all(), f"Non-finite values in {key}"


class TestResidualPersistence:
    def test_storage_roundtrip(self, tmp_path):
        from chorus.server.storage import DeltaStorage
        storage = DeltaStorage(str(tmp_path))

        deltas = [_make_delta(LAYERS, rank=4, dim=16, seed=i) for i in range(3)]
        strategy = FedExLoRA()
        strategy.aggregate(deltas)
        original_residuals = strategy.get_residuals()

        storage.save_residuals("test-model", original_residuals)
        loaded = storage.load_residuals("test-model")
        assert len(loaded) == len(original_residuals)
        for key in original_residuals:
            assert key in loaded
            assert torch.allclose(original_residuals[key], loaded[key])

    def test_empty_residuals(self, tmp_path):
        from chorus.server.storage import DeltaStorage
        storage = DeltaStorage(str(tmp_path))
        loaded = storage.load_residuals("nonexistent-model")
        assert loaded == {}


class TestRoundStateManagement:
    def test_default_state_is_open(self, tmp_path):
        from chorus.server.storage import DeltaStorage, RoundState
        storage = DeltaStorage(str(tmp_path))
        assert storage.get_round_state("m", 0) == RoundState.OPEN

    def test_state_transitions(self, tmp_path):
        from chorus.server.storage import DeltaStorage, RoundState
        storage = DeltaStorage(str(tmp_path))

        storage.set_round_state("m", 0, RoundState.AGGREGATING)
        assert storage.get_round_state("m", 0) == RoundState.AGGREGATING
        assert not storage.is_round_accepting("m", 0)

        storage.set_round_state("m", 0, RoundState.CLOSED)
        assert storage.get_round_state("m", 0) == RoundState.CLOSED
        assert not storage.is_round_accepting("m", 0)
