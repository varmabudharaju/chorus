"""Tests for WebSocket connection manager and live notifications."""

import asyncio
import json

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save

from chorus.server.app import app, configure, state
from chorus.server.ws import ConnectionManager


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
        model_id="ws-test",
        data_dir=str(tmp_path / "data"),
        strategy="fedex-lora",
        min_deltas=2,
    )
    return TestClient(app)


class TestConnectionManager:
    @pytest.mark.asyncio
    async def test_connected_count(self):
        mgr = ConnectionManager()
        assert mgr.connected_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_empty(self):
        """Broadcasting to no connections should not fail."""
        mgr = ConnectionManager()
        await mgr.broadcast({"event": "test"})

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent(self):
        """Disconnecting a non-connected client should not fail."""
        mgr = ConnectionManager()
        await mgr.disconnect("nonexistent")


class TestWebSocketEndpoint:
    def test_ws_connect_and_ping(self, client):
        """WebSocket ping/pong keepalive works."""
        with client.websocket_connect("/ws/test-client") as ws:
            ws.send_text("ping")
            resp = ws.receive_text()
            assert resp == "pong"

    def test_ws_health_shows_client_count(self, client):
        """Health endpoint should show ws_clients count."""
        resp = client.get("/health")
        assert "ws_clients" in resp.json()
        assert resp.json()["ws_clients"] == 0

    def test_ws_broadcast_on_aggregation(self, client):
        """WebSocket should receive round_complete event after aggregation."""
        with client.websocket_connect("/ws/listener") as ws:
            # Submit enough deltas to trigger aggregation
            for i, cid in enumerate(["c1", "c2"]):
                delta_bytes = _make_delta_bytes(seed=i)
                resp = client.post(
                    "/rounds/0/deltas",
                    files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                    params={"model_id": "ws-test", "client_id": cid},
                )
            assert resp.json()["aggregated"] is True

            # Should receive the broadcast
            msg = json.loads(ws.receive_text())
            assert msg["event"] == "round_complete"
            assert msg["model_id"] == "ws-test"
            assert msg["round_id"] == 0
            assert msg["next_round"] == 1
            assert msg["adapter_ready"] is True

    def test_ws_multiple_clients(self, client):
        """Multiple WebSocket clients should all receive broadcast."""
        with client.websocket_connect("/ws/client-a") as ws_a:
            with client.websocket_connect("/ws/client-b") as ws_b:
                # Trigger aggregation
                for i, cid in enumerate(["c1", "c2"]):
                    delta_bytes = _make_delta_bytes(seed=i)
                    client.post(
                        "/rounds/0/deltas",
                        files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                        params={"model_id": "ws-test", "client_id": cid},
                    )

                msg_a = json.loads(ws_a.receive_text())
                msg_b = json.loads(ws_b.receive_text())
                assert msg_a["event"] == "round_complete"
                assert msg_b["event"] == "round_complete"
