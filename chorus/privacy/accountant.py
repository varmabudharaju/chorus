"""Stateful per-client privacy accountant for federated LoRA.

Tracks (ε, δ) consumed across DP-noised submissions using RDP composition.
Default backend is Google's `dp-accounting` library; falls back to
`opacus.accountants.RDPAccountant` if `dp-accounting` is unavailable.

This module does NOT add noise. It only accounts for noise added elsewhere
by `chorus.privacy.mechanism.apply_dp`.
"""

from __future__ import annotations

from typing import Any

try:
    from dp_accounting import dp_event
    from dp_accounting.rdp import RdpAccountant as _DpAccountingRDP

    _BACKEND = "dp-accounting"
except ImportError:  # pragma: no cover - exercised only when dp-accounting absent
    try:
        from opacus.accountants import RDPAccountant as _OpacusRDP

        _BACKEND = "opacus"
    except ImportError:  # pragma: no cover
        raise ImportError(
            "PrivacyAccountant requires either 'dp-accounting' or 'opacus'. "
            "Install with: pip install 'chorus-fl[privacy]'"
        )


class PrivacyAccountant:
    """RDP-based privacy accountant for the Gaussian mechanism.

    Each call to `step()` records one application of the Gaussian mechanism
    at the configured `noise_multiplier` and `sample_rate`. The accountant
    tracks RDP at multiple orders and converts to (ε, δ) on demand.

    Args:
        target_epsilon: Maximum ε allowed before `is_exhausted()` becomes True.
        target_delta: δ for the (ε, δ)-DP guarantee.
        noise_multiplier: σ / sensitivity for the Gaussian mechanism.
        sample_rate: Fraction of the dataset sampled per round (1.0 = full).
    """

    def __init__(
        self,
        target_epsilon: float,
        target_delta: float,
        noise_multiplier: float,
        sample_rate: float = 1.0,
    ) -> None:
        if target_epsilon <= 0:
            raise ValueError("target_epsilon must be positive")
        if not (0 < target_delta < 1):
            raise ValueError("target_delta must be in (0, 1)")
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        if not (0 < sample_rate <= 1):
            raise ValueError("sample_rate must be in (0, 1]")

        self.target_epsilon = float(target_epsilon)
        self.target_delta = float(target_delta)
        self.noise_multiplier = float(noise_multiplier)
        self.sample_rate = float(sample_rate)
        self._steps = 0
        self._accountant = self._make_backend()

    def _make_backend(self) -> Any:
        if _BACKEND == "dp-accounting":
            return _DpAccountingRDP()
        return _OpacusRDP()  # pragma: no cover

    def step(self) -> None:
        """Record one round of Gaussian-mechanism noise application."""
        if _BACKEND == "dp-accounting":
            event = dp_event.PoissonSampledDpEvent(
                sampling_probability=self.sample_rate,
                event=dp_event.GaussianDpEvent(noise_multiplier=self.noise_multiplier),
            )
            self._accountant.compose(event, count=1)
        else:  # pragma: no cover
            self._accountant.step(
                noise_multiplier=self.noise_multiplier,
                sample_rate=self.sample_rate,
            )
        self._steps += 1

    def get_epsilon(self, delta: float | None = None) -> float:
        """Return the ε consumed so far at the given δ (default: target_delta)."""
        d = delta if delta is not None else self.target_delta
        if self._steps == 0:
            return 0.0
        if _BACKEND == "dp-accounting":
            return float(self._accountant.get_epsilon(target_delta=d))
        return float(self._accountant.get_epsilon(delta=d))  # pragma: no cover

    def is_exhausted(self) -> bool:
        """True if get_epsilon() at target_delta has reached or exceeded target_epsilon."""
        return self.get_epsilon() >= self.target_epsilon

    def remaining(self) -> tuple[float, float]:
        """Return (epsilon_remaining, target_delta).

        epsilon_remaining is `max(target_epsilon - get_epsilon(), 0.0)`.
        """
        return (max(self.target_epsilon - self.get_epsilon(), 0.0), self.target_delta)

    def serialize(self) -> dict[str, Any]:
        """Return a JSON-safe dict representing this accountant's state.

        Backend-agnostic: stores config + step count. Backend state is
        rebuilt by replaying steps on deserialize (cheap for small N).
        """
        return {
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "noise_multiplier": self.noise_multiplier,
            "sample_rate": self.sample_rate,
            "steps": self._steps,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "PrivacyAccountant":
        """Reconstruct a PrivacyAccountant from serialize() output.

        Older serialized state may include a 'backend' field; it is ignored
        because backend selection is now determined at import time by which
        library is installed.
        """
        a = cls(
            target_epsilon=float(data["target_epsilon"]),
            target_delta=float(data["target_delta"]),
            noise_multiplier=float(data["noise_multiplier"]),
            sample_rate=float(data.get("sample_rate", 1.0)),
            # 'backend' field intentionally ignored if present
        )
        steps = int(data.get("steps", 0))
        for _ in range(steps):
            a.step()
        return a

    def __repr__(self) -> str:
        return (
            f"PrivacyAccountant(steps={self._steps}, "
            f"epsilon={self.get_epsilon():.4f}/{self.target_epsilon}, "
            f"delta={self.target_delta}, backend={_BACKEND})"
        )
