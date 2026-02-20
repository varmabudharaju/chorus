"""Base model weight management and residual folding."""

from __future__ import annotations

import torch


def fold_residuals_into_base(
    base_weights: dict[str, torch.Tensor],
    residuals: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Add FedExLoRA residuals to the corresponding base model weight layers.

    For each layer with a residual R (shape: hidden_dim x hidden_dim),
    find the corresponding base weight and add R to it.
    The residual represents: exact_avg(B_i @ A_i) - best_rank_r_approx(B_i @ A_i)
    This is a dense correction that belongs in the base weights.

    After folding, the effective weight becomes:
        W_base_new + new_B @ new_A = (W_base + residual) + (new_B @ new_A)
                                   = W_base + target  (exact, no information lost)

    Args:
        base_weights: Full model state dict (or subset with target layers).
            Keys are like "model.layers.0.self_attn.q_proj.weight".
        residuals: Dict of layer_name -> residual matrix from FedExLoRA.
            Keys are like "model.layers.0.self_attn.q_proj".

    Returns:
        Updated base_weights dict with residuals folded in.
    """
    updated = dict(base_weights)
    for layer_name, residual in residuals.items():
        # The residual layer_name is like "model.layers.0.self_attn.q_proj"
        # The base weight key is typically "{layer_name}.weight"
        base_key = f"{layer_name}.weight"
        if base_key in updated:
            updated[base_key] = (updated[base_key].float() + residual.float()).to(
                base_weights[base_key].dtype
            )
    return updated


def merge_adapter_into_base(
    base_weights: dict[str, torch.Tensor],
    adapter: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Merge a LoRA adapter (B @ A) into base weights for checkpoint export.

    Args:
        base_weights: Base model weights.
        adapter: LoRA adapter tensors with lora_A.weight / lora_B.weight keys.

    Returns:
        Base weights with adapter merged in (W + B @ A).
    """
    from chorus.patterns import get_layer_pairs

    updated = dict(base_weights)
    pairs = get_layer_pairs(adapter)

    for layer_name, (a_key, b_key) in pairs.items():
        base_key = f"{layer_name}.weight"
        if base_key in updated:
            a = adapter[a_key].float()
            b = adapter[b_key].float()
            delta = b @ a
            updated[base_key] = (updated[base_key].float() + delta).to(
                base_weights[base_key].dtype
            )

    return updated
