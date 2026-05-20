"""Tests for the /models/{id}/clients/{cid}/privacy endpoint."""

import pytest
from fastapi.testclient import TestClient

from chorus.server import app as app_module


@pytest.fixture
def configured_app(tmp_path):
    app_module.configure(
        model_id="test-model",
        data_dir=str(tmp_path),
        strategy="fedex-lora",
        min_deltas=1,
        dp_epsilon=1.0,
        dp_delta=1e-5,
        dp_max_norm=1.0,
        accountant_target_epsilon=10.0,
        accountant_noise_multiplier=1.0,
        accountant_sample_rate=1.0,
    )
    return app_module.app


def test_privacy_endpoint_returns_zero_for_unknown_client(configured_app):
    client = TestClient(configured_app)
    resp = client.get("/models/test-model/clients/unknown-client/privacy")
    assert resp.status_code == 200
    data = resp.json()
    assert data["epsilon_consumed"] == pytest.approx(0.0, abs=1e-9)
    assert data["exhausted"] is False


def test_privacy_endpoint_404_when_accounting_disabled(tmp_path):
    app_module.configure(
        model_id="test-model",
        data_dir=str(tmp_path),
        strategy="fedex-lora",
        min_deltas=1,
        # NO accountant_target_epsilon
    )
    client = TestClient(app_module.app)
    resp = client.get("/models/test-model/clients/any/privacy")
    assert resp.status_code == 404
