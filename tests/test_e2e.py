"""End-to-end tests: simulated federation and full round-trip."""

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save, load

from chorus.server.app import app, configure
from chorus.simulate.runner import run_simulation, generate_synthetic_lora_delta


class TestSimulation:
    def test_basic_simulation(self):
        result = run_simulation(num_clients=3, num_rounds=2, strategy="fedex-lora")
        assert len(result.rounds) == 2
        assert result.final_adapter is not None
        assert len(result.final_adapter) > 0

    def test_fedavg_simulation(self):
        result = run_simulation(num_clients=3, num_rounds=2, strategy="fedavg")
        assert len(result.rounds) == 2

    def test_comparison_simulation(self):
        result = run_simulation(num_clients=5, num_rounds=2, compare_strategies=True)
        assert len(result.rounds) == 2
        for r in result.rounds:
            assert "fedavg_error" in r
            assert "fedex_error" in r
            # FedEx-LoRA should have lower error
            assert r["fedex_error"] < r["fedavg_error"]

    def test_simulation_with_dp(self):
        result = run_simulation(num_clients=3, num_rounds=2, dp_epsilon=1.0)
        assert len(result.rounds) == 2
        assert result.final_adapter is not None

    def test_summary_output(self):
        result = run_simulation(num_clients=2, num_rounds=1)
        summary = result.summary()
        assert "Simulation completed" in summary
        assert "1 rounds" in summary

    def test_comparison_accumulates_across_rounds(self):
        """Verify that _run_comparison reuses strategies across rounds.

        The FedExLoRA strategy should accumulate residuals across rounds,
        meaning the strategy object persists (not recreated each round).
        """
        from unittest.mock import patch
        from chorus.server.aggregation import FedExLoRA

        created_instances: list[FedExLoRA] = []
        original_get_strategy = __import__(
            "chorus.server.aggregation", fromlist=["get_strategy"]
        ).get_strategy

        def tracking_get_strategy(name):
            strategy = original_get_strategy(name)
            if isinstance(strategy, FedExLoRA):
                created_instances.append(strategy)
            return strategy

        with patch("chorus.simulate.runner.get_strategy", side_effect=tracking_get_strategy):
            result = run_simulation(
                num_clients=3, num_rounds=3, compare_strategies=True,
            )

        # Should create exactly one FedExLoRA instance (not one per round)
        assert len(created_instances) == 1
        # The single instance should have residuals (accumulated across 3 rounds)
        assert len(created_instances[0].get_residuals()) > 0
        # All 3 rounds should be recorded
        assert len(result.rounds) == 3


class TestFullRoundTrip:
    """Test the full flow: submit deltas via HTTP, trigger aggregation, pull result."""

    @pytest.fixture
    def test_client(self, tmp_path):
        configure(
            model_id="e2e-test",
            data_dir=str(tmp_path / "data"),
            strategy="fedex-lora",
            min_deltas=3,
        )
        return TestClient(app)

    def _make_delta_bytes(self, seed):
        torch.manual_seed(seed)
        tensors = {
            "layer.0.q.lora_A.weight": torch.randn(4, 32),
            "layer.0.q.lora_B.weight": torch.randn(32, 4),
            "layer.0.v.lora_A.weight": torch.randn(4, 32),
            "layer.0.v.lora_B.weight": torch.randn(32, 4),
        }
        return save(tensors)

    def test_full_round(self, test_client):
        # 1. Check status — no aggregation yet
        resp = test_client.get("/models/e2e-test/status")
        assert resp.json()["current_round"] == 0
        assert resp.json()["deltas_submitted"] == 0

        # 2. Submit 3 deltas
        for i in range(3):
            delta_bytes = self._make_delta_bytes(seed=i)
            resp = test_client.post(
                "/rounds/0/deltas",
                files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                params={"model_id": "e2e-test", "client_id": f"client_{i}"},
            )
            assert resp.status_code == 200

        # 3. Last submission should trigger aggregation
        assert resp.json()["aggregated"] is True

        # 4. Pull the aggregated result
        resp = test_client.get("/models/e2e-test/latest")
        assert resp.status_code == 200

        # 5. Verify the result is valid safetensors
        result_tensors = load(resp.content)
        assert len(result_tensors) == 4
        assert "layer.0.q.lora_A.weight" in result_tensors
        assert "layer.0.q.lora_B.weight" in result_tensors

    def test_multiple_rounds(self, test_client):
        # Round 0
        for i in range(3):
            delta_bytes = self._make_delta_bytes(seed=i)
            test_client.post(
                "/rounds/0/deltas",
                files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                params={"model_id": "e2e-test", "client_id": f"client_{i}"},
            )

        # Round 1
        for i in range(3):
            delta_bytes = self._make_delta_bytes(seed=100 + i)
            resp = test_client.post(
                "/rounds/1/deltas",
                files={"file": ("delta.safetensors", delta_bytes, "application/octet-stream")},
                params={"model_id": "e2e-test", "client_id": f"client_{i}"},
            )

        assert resp.json()["aggregated"] is True

        # Both rounds should be available
        assert test_client.get("/models/e2e-test/rounds/0").status_code == 200
        assert test_client.get("/models/e2e-test/rounds/1").status_code == 200

        # Latest should be round 1
        resp = test_client.get("/models/e2e-test/latest")
        assert resp.status_code == 200


class TestPrivacy:
    def test_dp_noise_magnitude(self):
        """Verify DP noise is calibrated — higher epsilon = less noise."""
        from chorus.server.privacy import GaussianMechanism

        low_eps = GaussianMechanism(epsilon=0.1, sensitivity=1.0)
        high_eps = GaussianMechanism(epsilon=10.0, sensitivity=1.0)

        assert low_eps.sigma > high_eps.sigma

    def test_dp_changes_tensors(self):
        from chorus.server.privacy import apply_dp

        tensors = {"weight": torch.ones(100, 100)}
        noised = apply_dp(tensors, epsilon=1.0, max_norm=1.0)

        # Should not be identical (with overwhelming probability)
        assert not torch.allclose(tensors["weight"], noised["weight"])

    def test_clipping_global_norm(self):
        """Verify clipping uses global L2 norm across all tensors."""
        from chorus.server.privacy import clip_delta

        tensors = {
            "a": torch.ones(10) * 100,
            "b": torch.ones(10) * 100,
        }
        clipped = clip_delta(tensors, max_norm=1.0)
        # Global norm should be <= max_norm
        flat = torch.cat([clipped["a"].float().flatten(), clipped["b"].float().flatten()])
        global_norm = torch.norm(flat)
        assert global_norm <= 1.0 + 1e-6

        # Both tensors should be scaled by the same factor
        ratio_a = torch.norm(clipped["a"].float()) / torch.norm(tensors["a"].float())
        ratio_b = torch.norm(clipped["b"].float()) / torch.norm(tensors["b"].float())
        assert torch.allclose(ratio_a, ratio_b, atol=1e-6)

    def test_clipping_no_clip_when_under_norm(self):
        from chorus.server.privacy import clip_delta

        tensors = {"weight": torch.ones(3) * 0.01}  # small norm
        clipped = clip_delta(tensors, max_norm=100.0)
        assert torch.allclose(clipped["weight"], tensors["weight"])
