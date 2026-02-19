"""Extract and apply LoRA deltas from PEFT adapters."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from chorus.patterns import LORA_A_PATTERN, LORA_B_PATTERN


def extract_lora_matrices(adapter_path: str | Path) -> dict[str, torch.Tensor]:
    """Extract LoRA A and B matrices from a PEFT adapter directory or safetensors file.

    Returns a flat dict of tensors with keys like:
        "layers.0.self_attn.q_proj.lora_A.weight"
        "layers.0.self_attn.q_proj.lora_B.weight"
    """
    adapter_path = Path(adapter_path)

    if adapter_path.is_dir():
        # Look for adapter_model.safetensors (PEFT standard)
        sf_path = adapter_path / "adapter_model.safetensors"
        if sf_path.exists():
            state_dict = load_file(str(sf_path))
        else:
            # Try .bin format
            bin_path = adapter_path / "adapter_model.bin"
            if bin_path.exists():
                state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=True)
            else:
                raise FileNotFoundError(
                    f"No adapter_model.safetensors or .bin found in {adapter_path}"
                )
    elif not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    elif adapter_path.suffix == ".safetensors":
        state_dict = load_file(str(adapter_path))
    else:
        raise ValueError(f"Unsupported adapter format: {adapter_path}")

    # Filter to only LoRA A and B matrices
    lora_tensors = {}
    for key, tensor in state_dict.items():
        # Normalize key: strip "base_model.model." prefix if present
        clean_key = key
        if clean_key.startswith("base_model.model."):
            clean_key = clean_key[len("base_model.model."):]
        elif clean_key.startswith("base_model."):
            clean_key = clean_key[len("base_model."):]

        if LORA_A_PATTERN.match(clean_key) or LORA_B_PATTERN.match(clean_key):
            lora_tensors[clean_key] = tensor.clone()

    if not lora_tensors:
        raise ValueError(f"No LoRA matrices found in {adapter_path}")

    return lora_tensors


def get_lora_layer_names(tensors: dict[str, torch.Tensor]) -> list[str]:
    """Get the base layer names (without .lora_A/.lora_B suffix) from a tensor dict."""
    names = set()
    for key in tensors:
        m = LORA_A_PATTERN.match(key)
        if m:
            names.add(m.group(1))
        m = LORA_B_PATTERN.match(key)
        if m:
            names.add(m.group(1))
    return sorted(names)


def save_lora_delta(tensors: dict[str, torch.Tensor], output_path: str | Path) -> Path:
    """Save LoRA tensors to a safetensors file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path))
    return output_path


def load_lora_delta(path: str | Path) -> dict[str, torch.Tensor]:
    """Load LoRA tensors from a safetensors file."""
    return load_file(str(path))


def apply_delta_to_adapter(
    base_adapter_path: str | Path,
    delta_tensors: dict[str, torch.Tensor],
    output_path: str | Path,
) -> Path:
    """Apply aggregated delta tensors to a base adapter, saving the result.

    This overwrites the LoRA A and B matrices in the adapter with the
    aggregated versions while preserving other adapter config.
    """
    base_adapter_path = Path(base_adapter_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the original adapter
    original = extract_lora_matrices(base_adapter_path)

    # Merge: use delta tensors where available, keep original otherwise
    merged = {**original, **delta_tensors}

    # Save
    sf_out = output_path / "adapter_model.safetensors"
    save_file(merged, str(sf_out))

    # Copy adapter_config.json if it exists
    config_src = base_adapter_path / "adapter_config.json" if base_adapter_path.is_dir() else None
    if config_src and config_src.exists():
        import shutil
        shutil.copy2(str(config_src), str(output_path / "adapter_config.json"))

    return output_path
