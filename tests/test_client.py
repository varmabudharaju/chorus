"""Tests for the client SDK and delta extraction."""

import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from fedlora.client.delta import (
    extract_lora_matrices,
    get_lora_layer_names,
    save_lora_delta,
    load_lora_delta,
)


@pytest.fixture
def adapter_dir(tmp_path):
    """Create a fake PEFT adapter directory."""
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(8, 256),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(256, 8),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight": torch.randn(8, 256),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight": torch.randn(256, 8),
    }
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))

    config = '{"peft_type": "LORA", "r": 8, "target_modules": ["q_proj", "v_proj"]}'
    (tmp_path / "adapter_config.json").write_text(config)
    return tmp_path


@pytest.fixture
def adapter_file(tmp_path):
    """Create a standalone safetensors file with LoRA weights."""
    tensors = {
        "layer.0.q_proj.lora_A.weight": torch.randn(4, 128),
        "layer.0.q_proj.lora_B.weight": torch.randn(128, 4),
    }
    path = tmp_path / "adapter.safetensors"
    save_file(tensors, str(path))
    return path


class TestExtractLoraMatrices:
    def test_from_directory(self, adapter_dir):
        tensors = extract_lora_matrices(adapter_dir)
        assert len(tensors) == 4
        # Should strip base_model.model. prefix
        for key in tensors:
            assert not key.startswith("base_model.")

    def test_from_file(self, adapter_file):
        tensors = extract_lora_matrices(adapter_file)
        assert len(tensors) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_lora_matrices(tmp_path / "nonexistent")

    def test_empty_adapter_raises(self, tmp_path):
        # Create a file with no LoRA matrices
        save_file({"some.other.weight": torch.randn(4, 4)}, str(tmp_path / "bad.safetensors"))
        with pytest.raises(ValueError, match="No LoRA matrices"):
            extract_lora_matrices(tmp_path / "bad.safetensors")


class TestGetLayerNames:
    def test_names(self, adapter_file):
        tensors = extract_lora_matrices(adapter_file)
        names = get_lora_layer_names(tensors)
        assert names == ["layer.0.q_proj"]


class TestSaveLoadDelta:
    def test_roundtrip(self, tmp_path):
        original = {
            "layer.0.lora_A.weight": torch.randn(4, 16),
            "layer.0.lora_B.weight": torch.randn(16, 4),
        }
        path = save_lora_delta(original, tmp_path / "delta.safetensors")
        loaded = load_lora_delta(path)
        for key in original:
            assert torch.allclose(original[key], loaded[key])
