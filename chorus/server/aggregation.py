"""Aggregation algorithms for federated LoRA adapters.

Implements:
- FedAvg (naive, mathematically inexact for LoRA — baseline only)
- FedEx-LoRA (mathematically exact aggregation from ACL/ICLR 2025)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import torch

LORA_A_PATTERN = re.compile(r"(.+)\.lora_A\.(?:default\.)?weight$")
LORA_B_PATTERN = re.compile(r"(.+)\.lora_B\.(?:default\.)?weight$")


def _get_layer_pairs(tensors: dict[str, torch.Tensor]) -> dict[str, tuple[str, str]]:
    """Map base layer name -> (lora_A_key, lora_B_key) from a tensor dict."""
    a_keys: dict[str, str] = {}
    b_keys: dict[str, str] = {}

    for key in tensors:
        m = LORA_A_PATTERN.match(key)
        if m:
            a_keys[m.group(1)] = key
            continue
        m = LORA_B_PATTERN.match(key)
        if m:
            b_keys[m.group(1)] = key

    pairs = {}
    for layer_name in a_keys:
        if layer_name in b_keys:
            pairs[layer_name] = (a_keys[layer_name], b_keys[layer_name])
    return pairs


class AggregationStrategy(ABC):
    """Base class for LoRA aggregation strategies."""

    @abstractmethod
    def aggregate(
        self,
        client_deltas: list[dict[str, torch.Tensor]],
        weights: list[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Aggregate multiple client deltas into a single global update."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class FedAvg(AggregationStrategy):
    """Naive Federated Averaging — averages A and B matrices independently.

    WARNING: This is mathematically incorrect for LoRA because
    avg(B_i @ A_i) != avg(B_i) @ avg(A_i). Included as a baseline only.
    """

    @property
    def name(self) -> str:
        return "fedavg"

    def aggregate(
        self,
        client_deltas: list[dict[str, torch.Tensor]],
        weights: list[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        if not client_deltas:
            raise ValueError("No deltas to aggregate")

        n = len(client_deltas)
        if weights is None:
            weights = [1.0 / n] * n
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        result: dict[str, torch.Tensor] = {}
        all_keys = client_deltas[0].keys()

        for key in all_keys:
            stacked = torch.stack([d[key].float() for d in client_deltas])
            w = torch.tensor(weights, dtype=torch.float32).reshape(-1, *([1] * (stacked.dim() - 1)))
            result[key] = (stacked * w).sum(dim=0).to(client_deltas[0][key].dtype)

        return result


class FedExLoRA(AggregationStrategy):
    """FedEx-LoRA: Mathematically exact federated LoRA aggregation.

    From "Exact Federated Learning of LoRA Adapters" (ACL/ICLR 2025).

    The key insight: naive averaging of B_i and A_i independently loses
    information because avg(B_i @ A_i) != avg(B_i) @ avg(A_i).

    FedEx-LoRA fixes this by:
    1. Computing the exact average of the full-rank updates: avg(B_i @ A_i)
    2. Computing the naive LoRA average: avg(B_i) @ avg(A_i)
    3. Tracking the residual: R = avg(B_i @ A_i) - avg(B_i) @ avg(A_i)
    4. The residual is accumulated and can be folded into base weights.

    This gives mathematically exact aggregation while keeping the LoRA structure.
    """

    def __init__(self) -> None:
        # Accumulated residuals per layer, carried across rounds
        self._residuals: dict[str, torch.Tensor] = {}

    @property
    def name(self) -> str:
        return "fedex-lora"

    def aggregate(
        self,
        client_deltas: list[dict[str, torch.Tensor]],
        weights: list[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        if not client_deltas:
            raise ValueError("No deltas to aggregate")

        n = len(client_deltas)
        if weights is None:
            weights = [1.0 / n] * n
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        # Identify LoRA layer pairs from the first client
        layer_pairs = _get_layer_pairs(client_deltas[0])

        result: dict[str, torch.Tensor] = {}

        for layer_name, (a_key, b_key) in layer_pairs.items():
            # Collect A and B matrices from all clients
            a_matrices = [d[a_key].float() for d in client_deltas]
            b_matrices = [d[b_key].float() for d in client_deltas]

            # Step 1: Compute the exact average of full-rank updates
            # exact_avg = sum(w_i * B_i @ A_i)
            exact_avg = torch.zeros_like(b_matrices[0] @ a_matrices[0])
            for i in range(n):
                exact_avg += weights[i] * (b_matrices[i] @ a_matrices[i])

            # Step 2: Compute naive LoRA average
            # avg_A = sum(w_i * A_i), avg_B = sum(w_i * B_i)
            w_t = torch.tensor(weights, dtype=torch.float32)

            avg_a = torch.zeros_like(a_matrices[0])
            avg_b = torch.zeros_like(b_matrices[0])
            for i in range(n):
                avg_a += weights[i] * a_matrices[i]
                avg_b += weights[i] * b_matrices[i]

            naive_product = avg_b @ avg_a

            # Step 3: Compute the gap we need to correct.
            # This includes the current round's approximation error plus any
            # accumulated residual from previous rounds that couldn't be
            # represented in the LoRA structure.
            round_gap = exact_avg - naive_product
            prev_residual = self._residuals.get(layer_name, torch.zeros_like(exact_avg))
            total_correction_needed = round_gap + prev_residual

            # Step 4: Fold the correction into B using the pseudoinverse of A.
            # corrected_B @ avg_A = avg_B @ avg_A + correction @ pinv(A) @ A
            # ≈ naive_product + total_correction_needed (when A has full row rank)
            a_pinv = torch.linalg.pinv(avg_a)
            b_correction = total_correction_needed @ a_pinv
            corrected_b = avg_b + b_correction

            # Step 5: Track what couldn't be represented (null space component).
            # For typical LoRA (rank << dim), A has full row rank and the
            # residual should be near zero.
            actual_product = corrected_b @ avg_a
            desired_product = naive_product + total_correction_needed
            self._residuals[layer_name] = desired_product - actual_product

            orig_dtype = client_deltas[0][a_key].dtype
            result[a_key] = avg_a.to(orig_dtype)
            result[b_key] = corrected_b.to(orig_dtype)

        # Handle any non-LoRA keys (e.g., biases) with simple averaging
        lora_keys = set()
        for _, (a_key, b_key) in layer_pairs.items():
            lora_keys.add(a_key)
            lora_keys.add(b_key)

        for key in client_deltas[0]:
            if key not in lora_keys:
                stacked = torch.stack([d[key].float() for d in client_deltas])
                w = torch.tensor(weights, dtype=torch.float32).reshape(
                    -1, *([1] * (stacked.dim() - 1))
                )
                result[key] = (stacked * w).sum(dim=0).to(client_deltas[0][key].dtype)

        return result

    def get_residuals(self) -> dict[str, torch.Tensor]:
        """Get the current accumulated residuals (for inspection/debugging)."""
        return dict(self._residuals)

    def reset_residuals(self) -> None:
        """Reset accumulated residuals (e.g., after folding into base weights)."""
        self._residuals.clear()


STRATEGIES: dict[str, type[AggregationStrategy]] = {
    "fedavg": FedAvg,
    "fedex-lora": FedExLoRA,
}


def get_strategy(name: str) -> AggregationStrategy:
    """Get an aggregation strategy by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]()
