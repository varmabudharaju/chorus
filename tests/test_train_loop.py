"""Tests for the continuous training loop (mock-based)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
from safetensors.torch import save_file

from chorus.client.sdk import ChorusClient


class TestTrainLoop:
    @pytest.fixture
    def mock_trainer(self, tmp_path):
        """Create a mock LoRATrainer that produces a fake adapter."""
        trainer = MagicMock()
        trainer.adapter_path = None
        trainer.output_dir = tmp_path / "adapter"
        trainer.output_dir.mkdir()

        def fake_train():
            # Create a fake adapter file
            out = tmp_path / "adapter" / "adapter_model.safetensors"
            out.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {"layer.0.lora_A.weight": torch.randn(4, 16),
                 "layer.0.lora_B.weight": torch.randn(16, 4)},
                str(out),
            )
            return out.parent

        trainer.train.side_effect = fake_train
        trainer.get_dataset_size.return_value = 1000
        return trainer

    @patch.object(ChorusClient, "pull_latest")
    @patch.object(ChorusClient, "listen")
    @patch.object(ChorusClient, "submit_delta")
    @patch.object(ChorusClient, "status")
    def test_single_round_loop(
        self, mock_status, mock_submit, mock_listen, mock_pull, mock_trainer, tmp_path
    ):
        """Test one iteration of the training loop."""
        mock_status.return_value = {"current_round": 0}
        mock_submit.return_value = {
            "aggregated": True,
            "round_id": 0,
            "next_round": 1,
            "deltas_received": 2,
            "min_deltas": 2,
        }
        pulled_path = tmp_path / "aggregated" / "adapter_model.safetensors"
        pulled_path.parent.mkdir(parents=True)
        save_file(
            {"layer.0.lora_A.weight": torch.randn(4, 16),
             "layer.0.lora_B.weight": torch.randn(16, 4)},
            str(pulled_path),
        )
        mock_pull.return_value = pulled_path

        client = ChorusClient(server="http://localhost:8080", model_id="test")
        rounds_completed = []

        def on_complete(round_num, result):
            rounds_completed.append(round_num)

        client.train_loop(
            trainer=mock_trainer,
            num_rounds=1,
            on_round_complete=on_complete,
        )

        assert mock_trainer.train.call_count == 1
        assert mock_submit.call_count == 1
        # Verify dataset_size was passed
        call_kwargs = mock_submit.call_args[1]
        assert call_kwargs["dataset_size"] == 1000
        assert "adapter_path" in call_kwargs
        assert mock_pull.call_count == 1
        assert len(rounds_completed) == 1
        # listen() should NOT be called since aggregated=True
        mock_listen.assert_not_called()

    @patch.object(ChorusClient, "pull_latest")
    @patch.object(ChorusClient, "listen")
    @patch.object(ChorusClient, "submit_delta")
    @patch.object(ChorusClient, "status")
    def test_waits_for_aggregation_when_not_triggered(
        self, mock_status, mock_submit, mock_listen, mock_pull, mock_trainer, tmp_path
    ):
        """When aggregation isn't triggered, loop should wait via WebSocket."""
        mock_status.return_value = {"current_round": 0}
        mock_submit.return_value = {
            "aggregated": False,
            "round_id": 0,
            "next_round": 1,
            "deltas_received": 1,
            "min_deltas": 2,
        }

        # listen yields a round_complete event
        mock_listen.return_value = iter([
            {"event": "round_complete", "model_id": "test", "round_id": 0, "next_round": 1}
        ])

        pulled_path = tmp_path / "aggregated" / "adapter_model.safetensors"
        pulled_path.parent.mkdir(parents=True)
        save_file(
            {"layer.0.lora_A.weight": torch.randn(4, 16),
             "layer.0.lora_B.weight": torch.randn(16, 4)},
            str(pulled_path),
        )
        mock_pull.return_value = pulled_path

        client = ChorusClient(server="http://localhost:8080", model_id="test")
        client.train_loop(trainer=mock_trainer, num_rounds=1)

        # listen() should be called since aggregated=False
        mock_listen.assert_called_once()

    @patch.object(ChorusClient, "pull_latest")
    @patch.object(ChorusClient, "listen")
    @patch.object(ChorusClient, "submit_delta")
    @patch.object(ChorusClient, "status")
    def test_adapter_path_updated_between_rounds(
        self, mock_status, mock_submit, mock_listen, mock_pull, mock_trainer, tmp_path
    ):
        """The trainer's adapter_path should be updated after each round."""
        mock_status.return_value = {"current_round": 0}
        mock_submit.return_value = {
            "aggregated": True,
            "round_id": 0,
            "next_round": 1,
            "deltas_received": 2,
            "min_deltas": 2,
        }
        pulled_path = tmp_path / "aggregated" / "adapter_model.safetensors"
        pulled_path.parent.mkdir(parents=True)
        save_file(
            {"layer.0.lora_A.weight": torch.randn(4, 16),
             "layer.0.lora_B.weight": torch.randn(16, 4)},
            str(pulled_path),
        )
        mock_pull.return_value = pulled_path

        client = ChorusClient(server="http://localhost:8080", model_id="test")
        client.train_loop(trainer=mock_trainer, num_rounds=2)

        # adapter_path should have been set before the second train() call
        assert mock_trainer.train.call_count == 2
