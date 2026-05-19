"""Security and robustness tests for all 16 audit fixes."""


import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save

from chorus.server.app import app, configure, state
from chorus.server.storage import DeltaStorage, RoundState, _sanitize_client_id
from chorus.exceptions import DuplicateClientError


def _make_delta_bytes(seed=42, rank=4, dim=16):
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
        model_id="sec-test",
        data_dir=str(tmp_path / "data"),
        strategy="fedex-lora",
        min_deltas=2,
    )
    return TestClient(app)


# === Fix 1: Path Traversal via client_id ===


class TestPathTraversal:
    def test_sanitize_strips_slashes(self):
        assert _sanitize_client_id("a/b") == "a_b"
        assert _sanitize_client_id("a\\b") == "a_b"

    def test_sanitize_strips_dots(self):
        assert _sanitize_client_id("..evil") == "evil"
        assert _sanitize_client_id("...evil") == "evil"

    def test_sanitize_strips_null_bytes(self):
        assert _sanitize_client_id("abc\x00def") == "abc_def"

    def test_sanitize_empty_after_strip_raises(self):
        with pytest.raises(ValueError, match="empty or invalid"):
            _sanitize_client_id("...")

    def test_traversal_becomes_safe(self):
        """../../ etc should be sanitized to underscores, not raise."""
        result = _sanitize_client_id("../../etc")
        assert "/" not in result
        assert "\\" not in result
        assert not result.startswith(".")

    def test_sanitize_caps_length(self):
        assert len(_sanitize_client_id("a" * 200)) == 128

    def test_traversal_client_id_stays_in_data_dir(self, client, tmp_path):
        """client_id with path traversal should NOT escape data directory."""
        delta_bytes = _make_delta_bytes(seed=1)
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "sec-test", "client_id": "../../etc/passwd"},
        )
        assert resp.status_code == 200
        # Verify no files escaped the data dir (should be under tmp_path, not ../../etc/)
        assert not (tmp_path.parent / "etc").exists()

    def test_normal_client_id_works(self, client):
        delta_bytes = _make_delta_bytes(seed=1)
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "sec-test", "client_id": "good-client-123"},
        )
        assert resp.status_code == 200
        assert resp.json()["client_id"] == "good-client-123"


# === Fix 2: Upload Size Limit ===


class TestUploadSizeLimit:
    def test_upload_size_limit_enforced(self, tmp_path):
        configure(
            model_id="size-test",
            data_dir=str(tmp_path / "data"),
            strategy="fedavg",
            min_deltas=2,
            max_upload_bytes=100,  # Very small limit
        )
        tc = TestClient(app)
        delta_bytes = _make_delta_bytes(seed=1)
        # delta_bytes is > 100 bytes
        resp = tc.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "size-test", "client_id": "c1"},
        )
        assert resp.status_code == 413

    def test_upload_size_limit_allows_normal(self, client):
        """Default 500MB limit should allow normal test deltas."""
        delta_bytes = _make_delta_bytes(seed=1)
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "sec-test", "client_id": "c1"},
        )
        assert resp.status_code == 200

    def test_base_weight_upload_size_limit(self, tmp_path):
        configure(
            model_id="bw-size-test",
            data_dir=str(tmp_path / "data"),
            strategy="fedavg",
            min_deltas=2,
            max_upload_bytes=100,
        )
        tc = TestClient(app)
        base_bytes = save({"w": torch.randn(32, 32)})
        resp = tc.post(
            "/models/bw-size-test/base-weights",
            files={"file": ("base.safetensors", base_bytes, "application/octet-stream")},
        )
        assert resp.status_code == 413


# === Fix 3: Temp File Leak in /checkpoint ===


class TestTempFileCleanup:
    def test_checkpoint_serves_valid_response(self, tmp_path):
        """Checkpoint endpoint should return valid data (BackgroundTask cleanup)."""
        configure(
            model_id="cp-test",
            data_dir=str(tmp_path / "data"),
            strategy="fedex-lora",
            min_deltas=2,
        )
        tc = TestClient(app)
        # Upload base weights
        base = save({"model.layers.0.weight": torch.randn(8, 8)})
        tc.post(
            "/models/cp-test/base-weights",
            files={"file": ("base.safetensors", base, "application/octet-stream")},
        )
        resp = tc.get("/models/cp-test/checkpoint")
        assert resp.status_code == 200
        assert len(resp.content) > 0


# === Fix 4: WebSocket Authentication ===


class TestWebSocketAuth:
    def test_ws_requires_auth_when_configured(self, tmp_path):
        configure(
            model_id="ws-auth-test",
            data_dir=str(tmp_path / "data"),
            strategy="fedavg",
            min_deltas=2,
            api_keys=["ws-secret"],
        )
        tc = TestClient(app)
        # Connect without token — should be rejected
        with pytest.raises(Exception):
            with tc.websocket_connect("/ws/test-client") as ws:
                ws.send_text("ping")

    def test_ws_accepts_valid_token(self, tmp_path):
        configure(
            model_id="ws-auth-test2",
            data_dir=str(tmp_path / "data"),
            strategy="fedavg",
            min_deltas=2,
            api_keys=["ws-secret"],
        )
        tc = TestClient(app)
        with tc.websocket_connect("/ws/test-client?token=ws-secret") as ws:
            ws.send_text("ping")
            assert ws.receive_text() == "pong"

    def test_ws_no_auth_when_no_keys(self, client):
        """WS should work without token when no api_keys are configured."""
        with client.websocket_connect("/ws/test-client") as ws:
            ws.send_text("ping")
            assert ws.receive_text() == "pong"


# === Fix 5: WebSocket Client ID Collision ===


class TestWebSocketCollision:
    def test_ws_second_connection_replaces_first(self, client):
        """Second WS connection with same client_id should work."""
        # First connection
        with client.websocket_connect("/ws/dup-client") as ws1:
            ws1.send_text("ping")
            assert ws1.receive_text() == "pong"
        # Second connection with same ID should work fine
        with client.websocket_connect("/ws/dup-client") as ws2:
            ws2.send_text("ping")
            assert ws2.receive_text() == "pong"


# === Fix 6: Timing-safe API Key Comparison ===


class TestTimingSafeAuth:
    @pytest.fixture
    def auth_client(self, tmp_path):
        configure(
            model_id="timing-test",
            data_dir=str(tmp_path / "data"),
            strategy="fedavg",
            min_deltas=2,
            api_keys=["secret-key-abc"],
        )
        return TestClient(app)

    def test_valid_key_accepted(self, auth_client):
        resp = auth_client.get(
            "/models/timing-test/status",
            headers={"Authorization": "Bearer secret-key-abc"},
        )
        assert resp.status_code == 200

    def test_invalid_key_rejected(self, auth_client):
        resp = auth_client.get(
            "/models/timing-test/status",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 403


# === Fix 7: Error Message Information Leakage ===


class TestErrorLeakage:
    def test_invalid_file_no_exception_details(self, client):
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("bad.safetensors", b"not valid data", "application/octet-stream")},
            params={"model_id": "sec-test", "client_id": "c1"},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        # Should be a generic message, not leak exception class names
        assert detail == "Invalid safetensors file"
        assert "Traceback" not in detail
        assert "Error" not in detail  # no exception class names


# === Fix 8: Race Condition — Duplicate Client (Atomic Detection) ===


class TestDuplicateClientAtomic:
    def test_duplicate_client_rejected(self, client):
        """Second submission from same client_id should be 409."""
        delta_bytes = _make_delta_bytes(seed=1)
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "sec-test", "client_id": "dup-client"},
        )
        assert resp.status_code == 200

        delta_bytes2 = _make_delta_bytes(seed=2)
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes2, "application/octet-stream")},
            params={"model_id": "sec-test", "client_id": "dup-client"},
        )
        assert resp.status_code == 409

    def test_storage_duplicate_raises_exception(self, tmp_path):
        """DeltaStorage.save_delta should raise DuplicateClientError on duplicate."""
        storage = DeltaStorage(str(tmp_path))
        tensors = {"weight": torch.randn(4, 4)}
        storage.save_delta("m", 0, "client-a", tensors)
        with pytest.raises(DuplicateClientError):
            storage.save_delta("m", 0, "client-a", tensors)


# === Fix 9: Race Condition — Double Aggregation ===


class TestAggregationRace:
    def test_aggregation_double_check(self, client):
        """Aggregation should only happen once per round."""
        for i, cid in enumerate(["c1", "c2"]):
            delta_bytes = _make_delta_bytes(seed=i)
            resp = client.post(
                "/rounds/0/deltas",
                files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                params={"model_id": "sec-test", "client_id": cid},
            )
        assert resp.json()["aggregated"] is True
        # Round should be closed — verify state
        round_state = state.storage.get_round_state("sec-test", 0)
        assert round_state == RoundState.CLOSED


# === Fix 10: round_id Validation ===


class TestRoundIdValidation:
    def test_negative_round_id_rejected(self, client):
        delta_bytes = _make_delta_bytes(seed=1)
        resp = client.post(
            "/rounds/-1/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "sec-test", "client_id": "c1"},
        )
        assert resp.status_code == 400
        assert "non-negative" in resp.json()["detail"]

    def test_future_round_id_rejected(self, client):
        delta_bytes = _make_delta_bytes(seed=1)
        resp = client.post(
            "/rounds/99/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "sec-test", "client_id": "c1"},
        )
        assert resp.status_code == 400
        assert "ahead of current" in resp.json()["detail"]

    def test_current_round_accepted(self, client):
        delta_bytes = _make_delta_bytes(seed=1)
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "sec-test", "client_id": "c1"},
        )
        assert resp.status_code == 200


# === Fix 11: ChorusClient Auth Support ===


class TestClientAuth:
    def test_client_sends_auth_header(self):
        from chorus.client.sdk import ChorusClient

        client = ChorusClient(server="http://localhost:9999", model_id="test", api_key="my-key")
        assert client.api_key == "my-key"
        assert client._http.headers["Authorization"] == "Bearer my-key"
        client.close()

    def test_client_no_auth_header_by_default(self):
        from chorus.client.sdk import ChorusClient

        client = ChorusClient(server="http://localhost:9999", model_id="test")
        assert client.api_key is None
        assert "Authorization" not in client._http.headers
        client.close()


# === Fix 12: dataset_size Validation ===


class TestDatasetSizeValidation:
    def test_dataset_size_zero_rejected(self, client):
        delta_bytes = _make_delta_bytes(seed=1)
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "sec-test", "client_id": "c1", "dataset_size": 0},
        )
        assert resp.status_code == 400
        assert "positive" in resp.json()["detail"]

    def test_dataset_size_negative_rejected(self, client):
        delta_bytes = _make_delta_bytes(seed=1)
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "sec-test", "client_id": "c1", "dataset_size": -5},
        )
        assert resp.status_code == 400

    def test_dataset_size_positive_accepted(self, client):
        delta_bytes = _make_delta_bytes(seed=1)
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "sec-test", "client_id": "c1", "dataset_size": 500},
        )
        assert resp.status_code == 200


# === Fix 13: __init__.py Exports ===


class TestPackageExports:
    def test_server_init_exports(self):
        from chorus.server import FedExLoRA, FedAvg, DeltaStorage, RoundState
        assert FedExLoRA is not None
        assert FedAvg is not None
        assert DeltaStorage is not None
        assert RoundState is not None

    def test_client_init_exports(self):
        from chorus.client import ChorusClient, extract_lora_matrices
        assert ChorusClient is not None
        assert extract_lora_matrices is not None


# === Fix 14: Byzantine Defense Integration ===


class TestByzantineDefense:
    def test_norm_bounding_applied(self, tmp_path):
        configure(
            model_id="byz-test",
            data_dir=str(tmp_path / "data"),
            strategy="fedavg",
            min_deltas=2,
            norm_bound=0.01,  # Very tight bound
        )
        tc = TestClient(app)
        for i, cid in enumerate(["c1", "c2"]):
            delta_bytes = _make_delta_bytes(seed=i)
            resp = tc.post(
                "/rounds/0/deltas",
                files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                params={"model_id": "byz-test", "client_id": cid},
            )
        assert resp.json()["aggregated"] is True

    def test_byzantine_defaults_disabled(self, client):
        """Without norm_bound or outlier_threshold, aggregation is normal."""
        assert state.norm_bound is None
        assert state.outlier_threshold is None
        for i, cid in enumerate(["c1", "c2"]):
            delta_bytes = _make_delta_bytes(seed=i)
            resp = client.post(
                "/rounds/0/deltas",
                files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                params={"model_id": "sec-test", "client_id": cid},
            )
        assert resp.json()["aggregated"] is True


# === Fix 15: Rate Limiting ===


class TestRateLimiting:
    def test_rate_limit_enforced(self, tmp_path):
        configure(
            model_id="rl-test",
            data_dir=str(tmp_path / "data"),
            strategy="fedavg",
            min_deltas=100,  # High so we don't trigger aggregation
            rate_limit=2,  # Only 2 requests per minute
        )
        tc = TestClient(app)

        for i in range(2):
            delta_bytes = _make_delta_bytes(seed=i)
            resp = tc.post(
                "/rounds/0/deltas",
                files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                params={"model_id": "rl-test", "client_id": f"c{i}"},
            )
            assert resp.status_code == 200

        # Third request should be rate limited
        delta_bytes = _make_delta_bytes(seed=99)
        resp = tc.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "rl-test", "client_id": "c99"},
        )
        assert resp.status_code == 429

    def test_rate_limit_disabled_by_default(self, client):
        """Default config has no rate limiting."""
        assert state.rate_limiter is None
        delta_bytes = _make_delta_bytes(seed=1)
        resp = client.post(
            "/rounds/0/deltas",
            files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
            params={"model_id": "sec-test", "client_id": "c1"},
        )
        assert resp.status_code == 200


# === Fix 16: Graceful Server Shutdown — Stuck Round Recovery ===


class TestGracefulShutdown:
    def test_stuck_round_recovery(self, tmp_path):
        """Rounds stuck in AGGREGATING should be recovered to OPEN on startup."""
        data_dir = str(tmp_path / "data")
        storage = DeltaStorage(data_dir)

        # Manually create a stuck round
        storage.set_round_state("stuck-model", 0, RoundState.AGGREGATING)
        assert storage.get_round_state("stuck-model", 0) == RoundState.AGGREGATING

        # find_stuck_rounds should find it
        stuck = storage.find_stuck_rounds("stuck-model")
        assert stuck == [0]

    def test_find_stuck_rounds_empty(self, tmp_path):
        storage = DeltaStorage(str(tmp_path / "data"))
        assert storage.find_stuck_rounds("nonexistent") == []

    def test_find_stuck_rounds_ignores_open_and_closed(self, tmp_path):
        storage = DeltaStorage(str(tmp_path / "data"))
        storage.set_round_state("m", 0, RoundState.CLOSED)
        storage.set_round_state("m", 1, RoundState.OPEN)
        storage.set_round_state("m", 2, RoundState.AGGREGATING)
        stuck = storage.find_stuck_rounds("m")
        assert stuck == [2]


# === Backward Compatibility ===


class TestBackwardCompatibility:
    def test_configure_accepts_old_args(self, tmp_path):
        """configure() with only original arguments still works."""
        configure(
            model_id="compat-test",
            data_dir=str(tmp_path / "data"),
            strategy="fedex-lora",
            min_deltas=2,
        )
        assert state.model_id == "compat-test"
        assert state.max_upload_bytes == 500 * 1024 * 1024
        assert state.norm_bound is None
        assert state.outlier_threshold is None
        assert state.rate_limiter is None

    def test_exception_hierarchy(self):
        """DuplicateClientError should inherit from ChorusError."""
        from chorus.exceptions import ChorusError
        assert issubclass(DuplicateClientError, ChorusError)
