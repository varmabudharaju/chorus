"""Tests for accountant persistence in DeltaStorage."""

import json
from pathlib import Path

from chorus.privacy import PrivacyAccountant
from chorus.server.storage import DeltaStorage


def test_save_and_load_accountant_roundtrip(tmp_path: Path):
    storage = DeltaStorage(tmp_path)
    a = PrivacyAccountant(
        target_epsilon=10.0, target_delta=1e-5,
        noise_multiplier=1.0, sample_rate=1.0,
    )
    a.step()
    a.step()
    storage.save_accountant("model-x", "client-1", a)
    restored = storage.load_accountant("model-x", "client-1")
    assert restored is not None
    assert restored.target_epsilon == a.target_epsilon
    assert abs(restored.get_epsilon() - a.get_epsilon()) < 1e-9


def test_load_accountant_missing_returns_none(tmp_path: Path):
    storage = DeltaStorage(tmp_path)
    assert storage.load_accountant("model-x", "nobody") is None


def test_load_all_accountants_for_model(tmp_path: Path):
    storage = DeltaStorage(tmp_path)
    for cid in ("alice", "bob"):
        a = PrivacyAccountant(
            target_epsilon=10.0, target_delta=1e-5,
            noise_multiplier=1.0, sample_rate=1.0,
        )
        a.step()
        storage.save_accountant("model-x", cid, a)
    all_a = storage.load_all_accountants("model-x")
    assert set(all_a.keys()) == {"alice", "bob"}
    for accountant in all_a.values():
        assert accountant.get_epsilon() > 0


def test_save_accountant_atomic_overwrite(tmp_path: Path):
    storage = DeltaStorage(tmp_path)
    a = PrivacyAccountant(
        target_epsilon=10.0, target_delta=1e-5,
        noise_multiplier=1.0, sample_rate=1.0,
    )
    storage.save_accountant("model-x", "client-1", a)
    a.step()
    storage.save_accountant("model-x", "client-1", a)
    restored = storage.load_accountant("model-x", "client-1")
    assert restored is not None
    assert restored.get_epsilon() > 0


def test_accountants_restore_on_server_startup(tmp_path: Path):
    from fastapi.testclient import TestClient
    from chorus.privacy import PrivacyAccountant
    from chorus.server import app as app_module
    from chorus.server.storage import DeltaStorage

    # Pre-seed the storage with an accountant
    storage = DeltaStorage(tmp_path)
    a = PrivacyAccountant(
        target_epsilon=10.0, target_delta=1e-5,
        noise_multiplier=1.0, sample_rate=1.0,
    )
    a.step()
    a.step()
    storage.save_accountant("test-model", "preexisting", a)
    eps_before = a.get_epsilon()

    # Configure server pointing at the same data_dir
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

    # Use TestClient as a context manager so lifespan runs
    with TestClient(app_module.app) as client:
        resp = client.get("/models/test-model/clients/preexisting/privacy")
        assert resp.status_code == 200
        data = resp.json()
        assert abs(data["epsilon_consumed"] - eps_before) < 1e-9


def test_accountant_path_uses_sanitized_client_id(tmp_path: Path):
    storage = DeltaStorage(tmp_path)
    a = PrivacyAccountant(
        target_epsilon=10.0, target_delta=1e-5,
        noise_multiplier=1.0, sample_rate=1.0,
    )
    # Path-traversal attempt — sanitization should neutralize this
    storage.save_accountant("model-x", "../etc/passwd", a)
    privacy_dir = tmp_path / "model-x" / "privacy"
    # File must land inside the model's privacy dir, not outside it
    files = list(privacy_dir.glob("*.json"))
    assert len(files) == 1
    # Confirm no path traversal: no file escaped to parent directories
    assert files[0].parent == privacy_dir
