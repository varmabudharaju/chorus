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
