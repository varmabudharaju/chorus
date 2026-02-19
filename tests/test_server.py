"""Tests for the FastAPI aggregation server."""

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save

from chorus.server.app import app, configure


@pytest.fixture
def client(tmp_path):
    """Create a test client with a temporary data directory."""
    configure(
        model_id="test-model",
        data_dir=str(tmp_path / "data"),
        strategy="fedex-lora",
        min_deltas=2,
    )
    return TestClient(app)


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


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_id"] == "test-model"
        assert data["strategy"] == "fedex-lora"


class TestModelStatus:
    def test_status_empty(self, client):
        resp = client.get("/models/test-model/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_round"] == 0
        assert data["deltas_submitted"] == 0


class TestSubmitDelta:
    def test_submit_single(self, client):
        delta_bytes = _make_delta_bytes(seed=1)
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "test-model", "client_id": "c1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["client_id"] == "c1"
        assert data["deltas_received"] == 1
        assert data["aggregated"] is False

    def test_submit_triggers_aggregation(self, client):
        # Submit 2 deltas (min_deltas=2)
        for i, cid in enumerate(["c1", "c2"]):
            delta_bytes = _make_delta_bytes(seed=i)
            resp = client.post(
                "/rounds/0/deltas",
                files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                params={"model_id": "test-model", "client_id": cid},
            )
            assert resp.status_code == 200

        data = resp.json()
        assert data["aggregated"] is True
        assert data["deltas_received"] == 2

    def test_invalid_file(self, client):
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("bad.safetensors", b"not a valid file", "application/octet-stream")},
            params={"model_id": "test-model", "client_id": "c1"},
        )
        assert resp.status_code == 400


class TestGetLatest:
    def test_no_aggregation_yet(self, client):
        resp = client.get("/models/test-model/latest")
        assert resp.status_code == 404

    def test_get_after_aggregation(self, client):
        # Submit enough deltas to trigger aggregation
        for i, cid in enumerate(["c1", "c2"]):
            delta_bytes = _make_delta_bytes(seed=i)
            client.post(
                "/rounds/0/deltas",
                files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                params={"model_id": "test-model", "client_id": cid},
            )

        resp = client.get("/models/test-model/latest")
        assert resp.status_code == 200
        assert len(resp.content) > 0


class TestGetRound:
    def test_no_round(self, client):
        resp = client.get("/models/test-model/rounds/0")
        assert resp.status_code == 404

    def test_get_specific_round(self, client):
        for i, cid in enumerate(["c1", "c2"]):
            delta_bytes = _make_delta_bytes(seed=i)
            client.post(
                "/rounds/0/deltas",
                files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                params={"model_id": "test-model", "client_id": cid},
            )

        resp = client.get("/models/test-model/rounds/0")
        assert resp.status_code == 200


class TestDPServer:
    def test_dp_enabled(self, tmp_path):
        """Test that server-side DP noise is applied."""
        configure(
            model_id="dp-test",
            data_dir=str(tmp_path / "data"),
            strategy="fedavg",
            min_deltas=1,
            dp_epsilon=1.0,
        )
        dp_client = TestClient(app)

        delta_bytes = _make_delta_bytes(seed=42)
        resp = dp_client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "dp-test", "client_id": "c1"},
        )
        assert resp.status_code == 200
        assert resp.json()["aggregated"] is True


class TestAuthentication:
    @pytest.fixture
    def auth_client(self, tmp_path):
        configure(
            model_id="auth-test",
            data_dir=str(tmp_path / "data"),
            strategy="fedavg",
            min_deltas=2,
            api_keys=["secret-key-123", "another-key"],
        )
        return TestClient(app)

    def test_health_is_public(self, auth_client):
        resp = auth_client.get("/health")
        assert resp.status_code == 200

    def test_status_requires_auth(self, auth_client):
        resp = auth_client.get("/models/auth-test/status")
        assert resp.status_code == 401

    def test_status_with_valid_bearer(self, auth_client):
        resp = auth_client.get(
            "/models/auth-test/status",
            headers={"Authorization": "Bearer secret-key-123"},
        )
        assert resp.status_code == 200

    def test_status_with_invalid_key(self, auth_client):
        resp = auth_client.get(
            "/models/auth-test/status",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 403

    def test_submit_requires_auth(self, auth_client):
        delta_bytes = _make_delta_bytes(seed=1)
        resp = auth_client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "auth-test", "client_id": "c1"},
        )
        assert resp.status_code == 401

    def test_submit_with_auth(self, auth_client):
        delta_bytes = _make_delta_bytes(seed=1)
        resp = auth_client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "auth-test", "client_id": "c1"},
            headers={"Authorization": "Bearer secret-key-123"},
        )
        assert resp.status_code == 200

    def test_no_auth_when_no_keys_configured(self, client):
        """When no api_keys are set, everything works without auth."""
        resp = client.get("/models/test-model/status")
        assert resp.status_code == 200


class TestRoundManagement:
    def test_rejects_submission_to_closed_round(self, client):
        """After aggregation, round should be closed and reject new submissions."""
        # Submit 2 deltas to trigger aggregation (min_deltas=2)
        for i, cid in enumerate(["c1", "c2"]):
            delta_bytes = _make_delta_bytes(seed=i)
            resp = client.post(
                "/rounds/0/deltas",
                files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                params={"model_id": "test-model", "client_id": cid},
            )
        assert resp.json()["aggregated"] is True

        # Try to submit to the now-closed round
        delta_bytes = _make_delta_bytes(seed=99)
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "test-model", "client_id": "c3"},
        )
        assert resp.status_code == 409

    def test_rejects_duplicate_client(self, client):
        """Same client_id should not be able to submit twice to same round."""
        delta_bytes = _make_delta_bytes(seed=1)
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "test-model", "client_id": "c1"},
        )
        assert resp.status_code == 200

        # Same client again
        delta_bytes = _make_delta_bytes(seed=2)
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "test-model", "client_id": "c1"},
        )
        assert resp.status_code == 409

    def test_status_includes_round_state(self, client):
        resp = client.get("/models/test-model/status")
        assert "round_state" in resp.json()
        assert resp.json()["round_state"] == "open"
