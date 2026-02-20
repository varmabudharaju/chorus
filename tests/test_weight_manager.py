"""Tests for base weight management and residual folding."""

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save, load

from chorus.server.app import app, configure, state
from chorus.server.weight_manager import fold_residuals_into_base, merge_adapter_into_base


LAYERS = ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.v_proj"]


def _make_base_weights(layers, dim=16, seed=0):
    """Create synthetic base model weights."""
    torch.manual_seed(seed)
    weights = {}
    for name in layers:
        weights[f"{name}.weight"] = torch.randn(dim, dim)
    return weights


def _make_residuals(layers, dim=16, seed=10):
    """Create synthetic residuals (keyed by base layer name, not .weight suffix)."""
    torch.manual_seed(seed)
    return {name: torch.randn(dim, dim) * 0.01 for name in layers}


def _make_adapter(layers, rank=4, dim=16, seed=20):
    """Create a synthetic LoRA adapter."""
    torch.manual_seed(seed)
    tensors = {}
    for name in layers:
        tensors[f"{name}.lora_A.weight"] = torch.randn(rank, dim)
        tensors[f"{name}.lora_B.weight"] = torch.randn(dim, rank)
    return tensors


class TestFoldResidualsIntoBase:
    def test_basic_fold(self):
        base = _make_base_weights(LAYERS)
        residuals = _make_residuals(LAYERS)
        updated = fold_residuals_into_base(base, residuals)

        for layer in LAYERS:
            key = f"{layer}.weight"
            expected = base[key].float() + residuals[layer].float()
            assert torch.allclose(updated[key].float(), expected, atol=1e-6)

    def test_preserves_dtype(self):
        base = {f"{l}.weight": torch.randn(8, 8).half() for l in LAYERS}
        residuals = {l: torch.randn(8, 8) * 0.01 for l in LAYERS}
        updated = fold_residuals_into_base(base, residuals)
        for key in updated:
            assert updated[key].dtype == torch.float16

    def test_missing_base_key_skipped(self):
        """Residuals for layers not in base weights are silently skipped."""
        base = {f"{LAYERS[0]}.weight": torch.randn(8, 8)}
        residuals = {l: torch.randn(8, 8) * 0.01 for l in LAYERS}
        updated = fold_residuals_into_base(base, residuals)
        # Only the first layer should be updated
        assert f"{LAYERS[0]}.weight" in updated
        assert f"{LAYERS[1]}.weight" not in updated

    def test_empty_residuals(self):
        base = _make_base_weights(LAYERS)
        updated = fold_residuals_into_base(base, {})
        for key in base:
            assert torch.allclose(updated[key], base[key])

    def test_accumulates_across_rounds(self):
        """Folding twice should accumulate correctly."""
        base = _make_base_weights(LAYERS)
        r1 = _make_residuals(LAYERS, seed=10)
        r2 = _make_residuals(LAYERS, seed=20)

        step1 = fold_residuals_into_base(base, r1)
        step2 = fold_residuals_into_base(step1, r2)

        for layer in LAYERS:
            key = f"{layer}.weight"
            expected = base[key].float() + r1[layer].float() + r2[layer].float()
            assert torch.allclose(step2[key].float(), expected, atol=1e-5)

    def test_exact_recovery_math(self):
        """Verify the key math: base + residual + new_B@new_A == base + target."""
        from chorus.server.aggregation import FedExLoRA
        from chorus.patterns import get_layer_pairs

        # Create some clients
        deltas = [_make_adapter(LAYERS, rank=4, dim=16, seed=i) for i in range(3)]
        strategy = FedExLoRA()
        result = strategy.aggregate(deltas)
        residuals = strategy.get_residuals()

        base = _make_base_weights(LAYERS, dim=16)

        # Compute exact target: avg(B_i @ A_i)
        pairs = get_layer_pairs(deltas[0])
        targets = {}
        for layer_name, (a_key, b_key) in pairs.items():
            t = torch.zeros(16, 16)
            for d in deltas:
                t += (d[b_key].float() @ d[a_key].float()) / len(deltas)
            targets[layer_name] = t

        # Fold residuals into base
        updated_base = fold_residuals_into_base(base, residuals)

        # Verify: updated_base + new_B@new_A == original_base + target
        for layer_name, (a_key, b_key) in pairs.items():
            key = f"{layer_name}.weight"
            lhs = updated_base[key].float() + result[b_key].float() @ result[a_key].float()
            rhs = base[key].float() + targets[layer_name]
            assert torch.allclose(lhs, rhs, atol=1e-4), (
                f"Exact recovery failed for {layer_name}"
            )


class TestMergeAdapterIntoBase:
    def test_basic_merge(self):
        base = _make_base_weights(LAYERS)
        adapter = _make_adapter(LAYERS)
        merged = merge_adapter_into_base(base, adapter)

        from chorus.patterns import get_layer_pairs
        pairs = get_layer_pairs(adapter)
        for layer_name, (a_key, b_key) in pairs.items():
            key = f"{layer_name}.weight"
            expected = base[key].float() + adapter[b_key].float() @ adapter[a_key].float()
            assert torch.allclose(merged[key].float(), expected, atol=1e-5)


class TestBaseWeightEndpoints:
    @pytest.fixture
    def client(self, tmp_path):
        configure(
            model_id="bw-test",
            data_dir=str(tmp_path / "data"),
            strategy="fedex-lora",
            min_deltas=2,
        )
        return TestClient(app)

    def test_upload_and_download_base_weights(self, client):
        base = _make_base_weights(LAYERS)
        base_bytes = save(base)

        resp = client.post(
            "/models/bw-test/base-weights",
            files={"file": ("base.safetensors", base_bytes, "application/octet-stream")},
        )
        assert resp.status_code == 200
        assert resp.json()["num_tensors"] == len(base)

        resp = client.get("/models/bw-test/base-weights")
        assert resp.status_code == 200
        downloaded = load(resp.content)
        for key in base:
            assert torch.allclose(base[key], downloaded[key])

    def test_download_missing_base_weights(self, client):
        resp = client.get("/models/bw-test/base-weights")
        assert resp.status_code == 404

    def test_checkpoint_no_base_weights(self, client):
        resp = client.get("/models/bw-test/checkpoint")
        assert resp.status_code == 404

    def test_checkpoint_with_base_no_adapter(self, client):
        """Checkpoint returns base weights when no adapter exists."""
        base = _make_base_weights(LAYERS)
        client.post(
            "/models/bw-test/base-weights",
            files={"file": ("base.safetensors", save(base), "application/octet-stream")},
        )
        resp = client.get("/models/bw-test/checkpoint")
        assert resp.status_code == 200
        checkpoint = load(resp.content)
        for key in base:
            assert torch.allclose(base[key], checkpoint[key])

    def test_checkpoint_merges_adapter(self, client):
        """Checkpoint returns base + adapter merged."""
        base = _make_base_weights(LAYERS, dim=16)
        client.post(
            "/models/bw-test/base-weights",
            files={"file": ("base.safetensors", save(base), "application/octet-stream")},
        )

        # Submit enough deltas to trigger aggregation
        for i, cid in enumerate(["c1", "c2"]):
            torch.manual_seed(i)
            delta = _make_adapter(LAYERS, rank=4, dim=16, seed=i)
            delta_bytes = save(delta)
            client.post(
                "/rounds/0/deltas",
                files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                params={"model_id": "bw-test", "client_id": cid},
            )

        resp = client.get("/models/bw-test/checkpoint")
        assert resp.status_code == 200
        checkpoint = load(resp.content)
        # Checkpoint should have same keys as base
        for key in base:
            assert key in checkpoint


class TestResidualFoldingIntegration:
    def test_residuals_cleared_after_folding(self, tmp_path):
        """When base weights exist, residuals should be folded and cleared."""
        configure(
            model_id="fold-test",
            data_dir=str(tmp_path / "data"),
            strategy="fedex-lora",
            min_deltas=2,
        )
        tc = TestClient(app)

        # Upload base weights
        base = _make_base_weights(
            ["layer.0.attn.q_proj", "layer.0.attn.v_proj"], dim=16
        )
        tc.post(
            "/models/fold-test/base-weights",
            files={"file": ("base.safetensors", save(base), "application/octet-stream")},
        )

        # Submit deltas to trigger aggregation
        def make_bytes(seed):
            torch.manual_seed(seed)
            tensors = {
                "layer.0.attn.q_proj.lora_A.weight": torch.randn(4, 16),
                "layer.0.attn.q_proj.lora_B.weight": torch.randn(16, 4),
                "layer.0.attn.v_proj.lora_A.weight": torch.randn(4, 16),
                "layer.0.attn.v_proj.lora_B.weight": torch.randn(16, 4),
            }
            return save(tensors)

        for i, cid in enumerate(["c1", "c2"]):
            tc.post(
                "/rounds/0/deltas",
                files={"file": ("delta.safetensors", make_bytes(i), "application/octet-stream")},
                params={"model_id": "fold-test", "client_id": cid},
            )

        # Residuals should be cleared (folded into base)
        residuals = state.storage.load_residuals("fold-test")
        assert len(residuals) == 0

        # Base weights should be updated (different from original)
        updated = state.storage.load_base_weights("fold-test")
        assert updated is not None
        for key in base:
            # Should NOT be identical (residuals were added)
            if not torch.allclose(base[key], updated[key]):
                break  # at least one changed
        else:
            pytest.fail("Base weights should have changed after folding")
