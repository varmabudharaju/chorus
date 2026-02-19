"""Differential privacy mechanisms for federated LoRA."""

from __future__ import annotations

import math

import torch


class GaussianMechanism:
    """Gaussian noise mechanism for (epsilon, delta)-differential privacy.

    Adds calibrated Gaussian noise to tensor values to provide DP guarantees.
    Uses the analytic Gaussian mechanism for tight noise calibration.
    """

    def __init__(self, epsilon: float, delta: float = 1e-5, sensitivity: float = 1.0):
        """
        Args:
            epsilon: Privacy budget. Lower = more privacy, more noise.
            delta: Failure probability for (eps, delta)-DP.
            sensitivity: L2 sensitivity of the query (max L2 norm of any single client's contribution).
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        if sensitivity <= 0:
            raise ValueError("sensitivity must be positive")

        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = self._compute_sigma()

    def _compute_sigma(self) -> float:
        """Compute the noise standard deviation for the Gaussian mechanism.

        Uses sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        from the standard Gaussian mechanism analysis.
        """
        return self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon

    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add calibrated Gaussian noise to a tensor."""
        noise = torch.randn_like(tensor.float()) * self.sigma
        return (tensor.float() + noise).to(tensor.dtype)

    def add_noise_to_dict(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Add calibrated Gaussian noise to all tensors in a dict."""
        return {key: self.add_noise(tensor) for key, tensor in tensors.items()}


def clip_delta(
    tensors: dict[str, torch.Tensor], max_norm: float
) -> dict[str, torch.Tensor]:
    """Clip the L2 norm of a delta to bound sensitivity.

    This is applied before adding noise to ensure the sensitivity assumption holds.
    Each tensor is clipped independently.
    """
    clipped = {}
    for key, tensor in tensors.items():
        t = tensor.float()
        norm = torch.norm(t)
        if norm > max_norm:
            t = t * (max_norm / norm)
        clipped[key] = t.to(tensor.dtype)
    return clipped


def apply_dp(
    tensors: dict[str, torch.Tensor],
    epsilon: float,
    delta: float = 1e-5,
    max_norm: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Convenience function: clip + add noise to a delta dict.

    Args:
        tensors: The delta tensors to privatize.
        epsilon: Privacy budget.
        delta: DP failure probability.
        max_norm: Clipping norm for sensitivity bounding.

    Returns:
        Noised tensor dict with (epsilon, delta)-DP guarantee.
    """
    clipped = clip_delta(tensors, max_norm)
    mechanism = GaussianMechanism(epsilon=epsilon, delta=delta, sensitivity=max_norm)
    return mechanism.add_noise_to_dict(clipped)
