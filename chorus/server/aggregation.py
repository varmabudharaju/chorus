"""Aggregation algorithms for federated LoRA adapters.

Implements:
- FedAvg (naive, mathematically inexact for LoRA — baseline only)
- FedEx-LoRA (mathematically exact aggregation from ACL/ICLR 2025)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from chorus.patterns import get_layer_pairs as _get_layer_pairs


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

        # FedAvg requires all clients to have identical tensor shapes
        ref_shapes = {k: v.shape for k, v in client_deltas[0].items()}
        for i, delta in enumerate(client_deltas[1:], 1):
            for k, v in delta.items():
                if k in ref_shapes and v.shape != ref_shapes[k]:
                    raise ValueError(
                        f"FedAvg requires uniform tensor shapes. Client {i} has "
                        f"{k} shape {v.shape}, expected {ref_shapes[k]}. "
                        f"Use fedex-lora for heterogeneous LoRA ranks."
                    )

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
    """FedEx-LoRA: SVD-optimal federated LoRA aggregation.

    Inspired by "Exact Federated Learning of LoRA Adapters" (2025).

    The key insight: naive averaging of B_i and A_i independently loses
    information because avg(B_i @ A_i) != avg(B_i) @ avg(A_i).
    The average of N rank-r updates has rank up to N*r, so it cannot be
    exactly represented as a single rank-r product.

    This implementation:
    1. Computes the exact weighted average of the full-rank updates: sum(w_i * B_i @ A_i)
    2. Uses truncated SVD to find the optimal rank-r approximation
    3. Tracks the residual (what couldn't be captured in rank-r)
       per round, for optional folding into base model weights

    Residuals are stored per-round but NOT automatically injected into
    subsequent rounds. Each round's aggregation is independent. To recover
    the lost information, fold residuals into the base model weights
    between rounds using get_residuals().
    """

    def __init__(
        self,
        residuals: dict[str, torch.Tensor] | None = None,
        output_rank: int | None = None,
    ) -> None:
        # Residuals from the most recent round, available for base-weight folding
        self._residuals: dict[str, torch.Tensor] = dict(residuals) if residuals else {}
        self._output_rank = output_rank

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

        # Discover LoRA layer pairs from ALL clients (union), so clients
        # with different target modules are handled correctly.
        layer_pairs: dict[str, tuple[str, str]] = {}
        for delta in client_deltas:
            for layer_name, pair in _get_layer_pairs(delta).items():
                if layer_name not in layer_pairs:
                    layer_pairs[layer_name] = pair

        result: dict[str, torch.Tensor] = {}

        for layer_name, (a_key, b_key) in layer_pairs.items():
            # Collect A and B matrices only from clients that have this layer
            client_indices = [i for i, d in enumerate(client_deltas) if a_key in d and b_key in d]
            a_matrices = [client_deltas[i][a_key].float() for i in client_indices]
            b_matrices = [client_deltas[i][b_key].float() for i in client_indices]
            layer_weights = [weights[i] for i in client_indices]

            # Re-normalise weights for this layer's subset of clients
            w_total = sum(layer_weights)
            layer_weights = [w / w_total for w in layer_weights]

            # Output rank: explicit override, or max across clients (preserves most info)
            rank = self._output_rank or max(a.shape[0] for a in a_matrices)

            # Step 1: Compute the exact weighted average of full-rank updates.
            # This has rank up to N*r, which exceeds r.
            target = torch.zeros_like(b_matrices[0] @ a_matrices[0])
            for i in range(len(client_indices)):
                target += layer_weights[i] * (b_matrices[i] @ a_matrices[i])

            # Step 2: Truncated SVD to find the best rank-r approximation.
            # This is the Eckart-Young-Mirsky theorem: the truncated SVD
            # gives the optimal rank-r approximation in Frobenius norm.
            U, S, Vt = torch.linalg.svd(target, full_matrices=False)

            # B_new = U[:, :r] * S[:r]  (shape: dim x r)
            # A_new = Vt[:r, :]          (shape: r x dim)
            # .contiguous() is required for safetensors serialization
            new_b = (U[:, :rank] * S[:rank].unsqueeze(0)).contiguous()  # (dim, r)
            new_a = Vt[:rank, :].contiguous()                            # (r, dim)

            # Step 3: Track residual — what the rank-r approximation couldn't capture.
            # Available via get_residuals() for folding into base model weights.
            approximation = new_b @ new_a
            self._residuals[layer_name] = target - approximation

            # Use dtype from the first client that has this layer
            orig_dtype = client_deltas[client_indices[0]][a_key].dtype
            result[a_key] = new_a.to(orig_dtype)
            result[b_key] = new_b.to(orig_dtype)

        # Handle any non-LoRA keys (e.g., biases) with simple averaging
        lora_keys = set()
        for _, (a_key, b_key) in layer_pairs.items():
            lora_keys.add(a_key)
            lora_keys.add(b_key)

        # Discover all non-LoRA keys across all clients (union)
        all_non_lora_keys: set[str] = set()
        for delta in client_deltas:
            all_non_lora_keys.update(k for k in delta if k not in lora_keys)

        for key in all_non_lora_keys:
            # Only average across clients that have this key
            indices = [i for i, d in enumerate(client_deltas) if key in d]
            key_weights = [weights[i] for i in indices]
            w_total = sum(key_weights)
            key_weights = [w / w_total for w in key_weights]

            stacked = torch.stack([client_deltas[i][key].float() for i in indices])
            w = torch.tensor(key_weights, dtype=torch.float32).reshape(
                -1, *([1] * (stacked.dim() - 1))
            )
            result[key] = (stacked * w).sum(dim=0).to(client_deltas[indices[0]][key].dtype)

        return result

    def get_residuals(self) -> dict[str, torch.Tensor]:
        """Get the current accumulated residuals (for inspection/debugging)."""
        return dict(self._residuals)

    def reset_residuals(self) -> None:
        """Reset accumulated residuals (e.g., after folding into base weights)."""
        self._residuals.clear()


def norm_bound_deltas(
    client_deltas: list[dict[str, torch.Tensor]],
    max_norm: float,
) -> list[dict[str, torch.Tensor]]:
    """Clip each client's delta to a maximum global L2 norm.

    This is a pre-aggregation defense against Byzantine/poisoning attacks.
    A malicious client sending a very large delta will be clipped to max_norm,
    limiting the damage they can do to the global model.

    Args:
        client_deltas: List of client delta dicts.
        max_norm: Maximum allowed global L2 norm per client.

    Returns:
        List of clipped deltas.
    """
    clipped = []
    for delta in client_deltas:
        flat = torch.cat([t.float().flatten() for t in delta.values()])
        global_norm = torch.norm(flat)
        if global_norm > max_norm:
            scale = max_norm / global_norm
            clipped.append({k: (v.float() * scale).to(v.dtype) for k, v in delta.items()})
        else:
            clipped.append(delta)
    return clipped


def detect_outlier_deltas(
    client_deltas: list[dict[str, torch.Tensor]],
    threshold: float = 3.0,
) -> list[int]:
    """Detect outlier deltas using norm-based z-score detection.

    Returns indices of deltas whose global L2 norm is more than
    `threshold` standard deviations from the mean.

    Args:
        client_deltas: List of client delta dicts.
        threshold: Number of standard deviations for outlier detection.

    Returns:
        List of indices of outlier deltas.
    """
    norms = []
    for delta in client_deltas:
        flat = torch.cat([t.float().flatten() for t in delta.values()])
        norms.append(torch.norm(flat).item())

    norms_t = torch.tensor(norms)
    mean = norms_t.mean()
    std = norms_t.std()

    if std < 1e-8:
        return []  # All norms are essentially the same

    z_scores = (norms_t - mean).abs() / std
    return [i for i, z in enumerate(z_scores.tolist()) if z > threshold]


def filter_outlier_deltas(
    client_deltas: list[dict[str, torch.Tensor]],
    threshold: float = 3.0,
) -> list[dict[str, torch.Tensor]]:
    """Remove outlier deltas based on norm z-score detection.

    Args:
        client_deltas: List of client delta dicts.
        threshold: Z-score threshold for outlier detection.

    Returns:
        Filtered list with outliers removed.
    """
    outliers = set(detect_outlier_deltas(client_deltas, threshold))
    return [d for i, d in enumerate(client_deltas) if i not in outliers]


STRATEGIES: dict[str, type[AggregationStrategy]] = {
    "fedavg": FedAvg,
    "fedex-lora": FedExLoRA,
}


def get_strategy(name: str) -> AggregationStrategy:
    """Get an aggregation strategy by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]()
