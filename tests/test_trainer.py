"""Tests for the LoRATrainer interface (mock-based, no GPU required)."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from chorus.client.trainer import LoRATrainer


class TestLoRATrainerInit:
    def test_default_params(self):
        trainer = LoRATrainer(
            base_model="test-model",
            dataset="test-dataset",
        )
        assert trainer.base_model == "test-model"
        assert trainer.dataset == "test-dataset"
        assert trainer.output_dir == Path("./chorus_adapter")
        assert trainer.lora_rank == 16
        assert trainer.lora_alpha == 32
        assert trainer.lora_target_modules == ["q_proj", "v_proj"]
        assert trainer.learning_rate == 2e-4
        assert trainer.num_epochs == 1
        assert trainer.max_steps == -1
        assert trainer.per_device_batch_size == 4
        assert trainer.bf16 is True
        assert trainer.adapter_path is None

    def test_custom_params(self, tmp_path):
        trainer = LoRATrainer(
            base_model="custom-model",
            dataset=tmp_path / "data.json",
            output_dir=tmp_path / "output",
            lora_rank=8,
            lora_alpha=16,
            lora_target_modules=["q_proj", "k_proj", "v_proj"],
            learning_rate=1e-4,
            num_epochs=3,
            max_steps=100,
            per_device_batch_size=8,
            bf16=False,
            adapter_path=tmp_path / "prev_adapter",
        )
        assert trainer.base_model == "custom-model"
        assert trainer.lora_rank == 8
        assert trainer.lora_alpha == 16
        assert trainer.num_epochs == 3
        assert trainer.max_steps == 100
        assert trainer.bf16 is False
        assert trainer.adapter_path == tmp_path / "prev_adapter"

    def test_extra_kwargs_stored(self):
        trainer = LoRATrainer(
            base_model="test",
            dataset="test",
            warmup_steps=100,
            weight_decay=0.01,
        )
        assert trainer.training_kwargs["warmup_steps"] == 100
        assert trainer.training_kwargs["weight_decay"] == 0.01

    def test_adapter_path_converted_to_path(self, tmp_path):
        trainer = LoRATrainer(
            base_model="test",
            dataset="test",
            adapter_path=str(tmp_path),
        )
        assert isinstance(trainer.adapter_path, Path)


class TestLoRATrainerGetDatasetSize:
    def test_dataset_size_from_object(self):
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=1234)

        trainer = LoRATrainer(base_model="test", dataset=mock_ds)
        assert trainer.get_dataset_size() == 1234

    def test_dataset_size_from_object_list(self):
        trainer = LoRATrainer(base_model="test", dataset=[1, 2, 3, 4, 5])
        assert trainer.get_dataset_size() == 5
