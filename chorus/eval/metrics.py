"""Metrics for evaluating aggregation strategies in the eval harness.

Two families:
- Algorithmic: Frobenius reconstruction error of aggregated B@A vs the exact
  weighted average of per-client B_i @ A_i.
- Task: perplexity (causal LM), accuracy (classification), F1 (classification).
  Backed by HuggingFace `evaluate`.
"""

from __future__ import annotations

import math

import torch

from chorus.patterns import get_layer_pairs


def frobenius_reconstruction_error(
    aggregated: dict[str, torch.Tensor],
    client_deltas: list[dict[str, torch.Tensor]],
    weights: list[float] | None = None,
    extra_base: dict[str, torch.Tensor] | None = None,
) -> float:
    """Frobenius norm of (exact_avg(B_i @ A_i) - reconstruction), maxed over layers.

    Args:
        aggregated: The aggregated adapter from the final round (B, A tensors).
        client_deltas: Per-client deltas from the final round.
        weights: Per-client weights; uniform if None.
        extra_base: Optional accumulated residuals from prior rounds that have
            been folded into the base model (keyed by layer name, dense matrices).
            When provided, the reconstruction is (extra_base[layer] + B_new @ A_new),
            reflecting the total effective update across all rounds.  When
            fold_residuals=True and strategy=fedex-lora, passing the accumulated
            residuals here gives a lower (more accurate) Frobenius error vs the
            fold_residuals=False case where extra_base is empty.
    """
    n = len(client_deltas)
    if weights is None:
        weights = [1.0 / n] * n

    pairs = get_layer_pairs(client_deltas[0])
    max_err = 0.0
    for layer_name, (a_key, b_key) in pairs.items():
        # Exact: sum w_i * B_i @ A_i
        exact = torch.zeros_like(
            client_deltas[0][b_key].float() @ client_deltas[0][a_key].float()
        )
        for i, d in enumerate(client_deltas):
            if a_key in d and b_key in d:
                exact += weights[i] * (d[b_key].float() @ d[a_key].float())

        # Reconstructed from aggregated adapter plus any folded residuals from prior rounds.
        if a_key not in aggregated or b_key not in aggregated:
            continue
        recon = aggregated[b_key].float() @ aggregated[a_key].float()
        if extra_base and layer_name in extra_base:
            recon = recon + extra_base[layer_name].float()
        err = torch.norm(exact - recon).item()
        max_err = max(max_err, err)

    return max_err


def compute_task_metric(
    metric_name: str,
    predictions: list,
    references: list,
) -> dict[str, float]:
    """Compute a task metric using HF `evaluate`.

    Args:
        metric_name: One of "accuracy", "f1", "perplexity".
        predictions: Predicted labels (for classification) or logits (for LM).
        references: Ground truth labels (for classification) or input ids (for LM).

    Returns:
        Metric dict from `evaluate.load(metric_name).compute(...)`.
    """
    import evaluate

    if metric_name == "perplexity":
        # Perplexity needs predictions as text strings under HF's `evaluate` API;
        # for our use, we compute it directly from cross-entropy in the runner.
        # This path is kept simple as a no-op stub since callers do PPL inline.
        raise NotImplementedError(
            "Perplexity is computed inline in EvalRunner, not via evaluate.load. "
            "Use compute_perplexity_from_loss() instead."
        )

    metric = evaluate.load(metric_name)
    return metric.compute(predictions=predictions, references=references)


def compute_perplexity_from_loss(loss: float) -> float:
    """Perplexity = exp(cross-entropy loss). Loss assumed to be natural-log base."""
    return math.exp(loss)
