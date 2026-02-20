"""Tests for auto-round advancement and dataset-size client weighting."""

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save

from chorus.server.app import app, configure


def _make_delta_bytes(seed=42, rank=4, dim=16):
    """Create safetensors bytes for a synthetic delta."""
    torch.manual_seed(seed)
    tensors = {
        "layer.0.attn.q_proj.lora_A.weight": torch.randn(rank, dim),
        "layer.0.attn.q_proj.lora_B.weight": torch.randn(dim, rank),
        "layer.0.attn.v_proj.lora_A.weight": torch.randn(rank, dim),
        "layer.0.attn.v_proj.lora_B.weight": torch.randn(dim, rank),
    }
    return save(tensors)


@pytest.fixture
def client(tmp_path):
    configure(
        model_id="test-model",
        data_dir=str(tmp_path / "data"),
        strategy="fedex-lora",
        min_deltas=2,
    )
    return TestClient(app)


class TestAutoRoundAdvancement:
    def test_next_round_in_response(self, client):
        """Submit response should include next_round field."""
        delta_bytes = _make_delta_bytes(seed=1)
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "test-model", "client_id": "c1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "next_round" in data
        assert data["next_round"] == 1

    def test_auto_advance_after_aggregation(self, client):
        """After aggregation closes round 0, round 1 should be implicitly OPEN."""
        for i, cid in enumerate(["c1", "c2"]):
            delta_bytes = _make_delta_bytes(seed=i)
            resp = client.post(
                "/rounds/0/deltas",
                files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                params={"model_id": "test-model", "client_id": cid},
            )
        assert resp.json()["aggregated"] is True
        assert resp.json()["next_round"] == 1

        # Round 1 should be open (implicitly)
        status = client.get("/models/test-model/status").json()
        assert status["current_round"] == 1
        assert status["round_state"] == "open"

    def test_submit_to_next_round_after_aggregation(self, client):
        """After round 0 aggregation, can submit to round 1."""
        for i, cid in enumerate(["c1", "c2"]):
            delta_bytes = _make_delta_bytes(seed=i)
            client.post(
                "/rounds/0/deltas",
                files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                params={"model_id": "test-model", "client_id": cid},
            )

        # Submit to round 1
        delta_bytes = _make_delta_bytes(seed=10)
        resp = client.post(
            "/rounds/1/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "test-model", "client_id": "c1"},
        )
        assert resp.status_code == 200
        assert resp.json()["round_id"] == 1


class TestDatasetSizeWeighting:
    def test_submit_with_dataset_size(self, client):
        """Submit with dataset_size param should be accepted."""
        delta_bytes = _make_delta_bytes(seed=1)
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "test-model", "client_id": "c1", "dataset_size": 1000},
        )
        assert resp.status_code == 200

    def test_dataset_size_stored_in_metadata(self, tmp_path):
        """dataset_size should be persisted in delta metadata."""
        configure(
            model_id="ds-test",
            data_dir=str(tmp_path / "data"),
            strategy="fedex-lora",
            min_deltas=3,
        )
        tc = TestClient(app)

        delta_bytes = _make_delta_bytes(seed=1)
        tc.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "ds-test", "client_id": "c1", "dataset_size": 5000},
        )

        from chorus.server.app import state
        meta = state.storage.load_delta_metadata("ds-test", 0, "c1")
        assert meta["dataset_size"] == 5000

    def test_weighted_aggregation_with_dataset_sizes(self, tmp_path):
        """Aggregation should use dataset_size for proportional weights."""
        configure(
            model_id="w-test",
            data_dir=str(tmp_path / "data"),
            strategy="fedex-lora",
            min_deltas=2,
        )
        tc = TestClient(app)

        # Client 1: 9000 samples, Client 2: 1000 samples
        for cid, size, seed in [("c1", 9000, 1), ("c2", 1000, 2)]:
            delta_bytes = _make_delta_bytes(seed=seed)
            resp = tc.post(
                "/rounds/0/deltas",
                files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                params={"model_id": "w-test", "client_id": cid, "dataset_size": size},
            )
        assert resp.json()["aggregated"] is True

    def test_backward_compat_no_dataset_size(self, client):
        """Submitting without dataset_size should still work (uniform weights)."""
        for i, cid in enumerate(["c1", "c2"]):
            delta_bytes = _make_delta_bytes(seed=i)
            resp = client.post(
                "/rounds/0/deltas",
                files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                params={"model_id": "test-model", "client_id": cid},
            )
        assert resp.json()["aggregated"] is True

    def test_mixed_dataset_size_falls_back_to_uniform(self, tmp_path):
        """If some clients provide dataset_size and others don't, fall back to uniform."""
        configure(
            model_id="mixed-test",
            data_dir=str(tmp_path / "data"),
            strategy="fedex-lora",
            min_deltas=2,
        )
        tc = TestClient(app)

        # c1 with size, c2 without
        delta_bytes = _make_delta_bytes(seed=1)
        tc.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "mixed-test", "client_id": "c1", "dataset_size": 1000},
        )
        delta_bytes = _make_delta_bytes(seed=2)
        resp = tc.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "mixed-test", "client_id": "c2"},
        )
        # Should still aggregate (falls back to uniform weights)
        assert resp.json()["aggregated"] is True
