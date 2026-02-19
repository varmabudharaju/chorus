"""Shared LoRA key patterns used across chorus modules."""

from __future__ import annotations

import re

import torch

# Pattern to identify LoRA A and B matrices in PEFT state dicts
LORA_A_PATTERN = re.compile(r"(.+)\.lora_A\.(?:default\.)?weight$")
LORA_B_PATTERN = re.compile(r"(.+)\.lora_B\.(?:default\.)?weight$")


def get_layer_pairs(tensors: dict[str, torch.Tensor]) -> dict[str, tuple[str, str]]:
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
