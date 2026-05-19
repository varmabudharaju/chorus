"""Tests for the PrivacyAccountant — RDP composition + budget tracking."""

import math

import pytest

# Module-under-test; this import will fail until Task 7 lands the class.
from chorus.privacy.accountant import PrivacyAccountant


class TestBasicBookkeeping:
    def test_zero_steps_means_zero_epsilon(self):
        a = PrivacyAccountant(
            target_epsilon=10.0, target_delta=1e-5,
            noise_multiplier=1.0, sample_rate=1.0,
        )
        assert a.get_epsilon() == pytest.approx(0.0, abs=1e-9)
        assert not a.is_exhausted()

    def test_epsilon_grows_monotonically_with_steps(self):
        a = PrivacyAccountant(
            target_epsilon=100.0, target_delta=1e-5,
            noise_multiplier=1.0, sample_rate=1.0,
        )
        eps = []
        for _ in range(5):
            a.step()
            eps.append(a.get_epsilon())
        # Strictly increasing
        for i in range(1, len(eps)):
            assert eps[i] > eps[i - 1]

    def test_higher_noise_multiplier_gives_smaller_epsilon(self):
        a_low = PrivacyAccountant(
            target_epsilon=100.0, target_delta=1e-5,
            noise_multiplier=1.0, sample_rate=1.0,
        )
        a_high = PrivacyAccountant(
            target_epsilon=100.0, target_delta=1e-5,
            noise_multiplier=5.0, sample_rate=1.0,
        )
        for _ in range(5):
            a_low.step()
            a_high.step()
        assert a_high.get_epsilon() < a_low.get_epsilon()


class TestExhaustion:
    def test_is_exhausted_triggers_at_threshold(self):
        # With noise_multiplier=0.5 and sample_rate=1, a few steps blow past ε=0.5
        a = PrivacyAccountant(
            target_epsilon=0.5, target_delta=1e-5,
            noise_multiplier=0.5, sample_rate=1.0,
        )
        # Step until exhausted, but bound the loop to avoid infinite spin
        for _ in range(50):
            if a.is_exhausted():
                break
            a.step()
        assert a.is_exhausted()
        assert a.get_epsilon() >= 0.5

    def test_remaining_decreases(self):
        a = PrivacyAccountant(
            target_epsilon=10.0, target_delta=1e-5,
            noise_multiplier=1.0, sample_rate=1.0,
        )
        eps_remaining_before, _ = a.remaining()
        a.step()
        eps_remaining_after, _ = a.remaining()
        assert eps_remaining_after < eps_remaining_before


class TestSerialization:
    def test_serialize_roundtrip_preserves_epsilon(self):
        a = PrivacyAccountant(
            target_epsilon=10.0, target_delta=1e-5,
            noise_multiplier=1.0, sample_rate=1.0,
        )
        for _ in range(3):
            a.step()
        eps_before = a.get_epsilon()
        serialized = a.serialize()
        restored = PrivacyAccountant.deserialize(serialized)
        assert math.isclose(restored.get_epsilon(), eps_before, rel_tol=1e-9)
        assert restored.target_epsilon == a.target_epsilon
        assert restored.target_delta == a.target_delta

    def test_serialize_is_json_safe(self):
        import json
        a = PrivacyAccountant(
            target_epsilon=10.0, target_delta=1e-5,
            noise_multiplier=1.0, sample_rate=1.0,
        )
        a.step()
        # Must be JSON-serializable as-is
        encoded = json.dumps(a.serialize())
        decoded = json.loads(encoded)
        restored = PrivacyAccountant.deserialize(decoded)
        assert math.isclose(restored.get_epsilon(), a.get_epsilon(), rel_tol=1e-9)


class TestValidation:
    def test_invalid_epsilon_rejected(self):
        with pytest.raises(ValueError):
            PrivacyAccountant(
                target_epsilon=0.0, target_delta=1e-5,
                noise_multiplier=1.0,
            )

    def test_invalid_delta_rejected(self):
        with pytest.raises(ValueError):
            PrivacyAccountant(
                target_epsilon=1.0, target_delta=0.0,
                noise_multiplier=1.0,
            )
        with pytest.raises(ValueError):
            PrivacyAccountant(
                target_epsilon=1.0, target_delta=1.5,
                noise_multiplier=1.0,
            )

    def test_invalid_noise_multiplier_rejected(self):
        with pytest.raises(ValueError):
            PrivacyAccountant(
                target_epsilon=1.0, target_delta=1e-5,
                noise_multiplier=0.0,
            )

    def test_invalid_sample_rate_rejected(self):
        with pytest.raises(ValueError):
            PrivacyAccountant(
                target_epsilon=1.0, target_delta=1e-5,
                noise_multiplier=1.0, sample_rate=0.0,
            )
        with pytest.raises(ValueError):
            PrivacyAccountant(
                target_epsilon=1.0, target_delta=1e-5,
                noise_multiplier=1.0, sample_rate=1.5,
            )
