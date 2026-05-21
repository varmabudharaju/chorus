"""Tests for EvalConfig dataclass + YAML loader."""

import logging
import textwrap
from pathlib import Path

import pytest

from chorus.eval.config import EvalConfig
from chorus.exceptions import EvalConfigError


def _write(tmp_path: Path, contents: str) -> Path:
    p = tmp_path / "eval.yaml"
    p.write_text(textwrap.dedent(contents).strip() + "\n")
    return p


class TestRequiredFields:
    def test_minimal_valid_config_loads(self, tmp_path):
        p = _write(tmp_path, """
            model_id: tiny-gpt2
            dataset:
              name: tiny
              split: train
            num_clients: 5
            num_rounds: 3
            strategies: [fedex-lora]
            rank: 8
            seeds: [42]
        """)
        cfg = EvalConfig.from_yaml(p)
        assert cfg.model_id == "tiny-gpt2"
        assert cfg.num_clients == 5
        assert cfg.num_rounds == 3
        assert cfg.strategies == ["fedex-lora"]
        assert cfg.rank == 8
        assert cfg.seeds == [42]

    def test_missing_model_id_raises(self, tmp_path):
        p = _write(tmp_path, """
            dataset: {name: tiny, split: train}
            num_clients: 2
            num_rounds: 1
            strategies: [fedavg]
            rank: 4
            seeds: [0]
        """)
        with pytest.raises(EvalConfigError, match="model_id"):
            EvalConfig.from_yaml(p)

    def test_missing_strategies_raises(self, tmp_path):
        p = _write(tmp_path, """
            model_id: tiny
            dataset: {name: tiny, split: train}
            num_clients: 2
            num_rounds: 1
            rank: 4
            seeds: [0]
        """)
        with pytest.raises(EvalConfigError, match="strategies"):
            EvalConfig.from_yaml(p)

    def test_invalid_strategy_rejected(self, tmp_path):
        p = _write(tmp_path, """
            model_id: tiny
            dataset: {name: tiny, split: train}
            num_clients: 2
            num_rounds: 1
            strategies: [nonsense]
            rank: 4
            seeds: [0]
        """)
        with pytest.raises(EvalConfigError, match="Unknown strategy"):
            EvalConfig.from_yaml(p)


class TestOptionalFields:
    def test_dp_fields_default_none(self, tmp_path):
        p = _write(tmp_path, """
            model_id: t
            dataset: {name: t, split: train}
            num_clients: 2
            num_rounds: 1
            strategies: [fedex-lora]
            rank: 4
            seeds: [0]
        """)
        cfg = EvalConfig.from_yaml(p)
        assert cfg.dp_epsilon is None
        assert cfg.dp_delta == 1e-5  # documented default
        assert cfg.fold_residuals is True  # documented default

    def test_heterogeneous_rank_supported(self, tmp_path):
        p = _write(tmp_path, """
            model_id: t
            dataset: {name: t, split: train}
            num_clients: 4
            num_rounds: 1
            strategies: [fedex-lora]
            rank: 4
            heterogeneous_rank: [4, 8, 16, 32]
            seeds: [0]
        """)
        cfg = EvalConfig.from_yaml(p)
        assert cfg.heterogeneous_rank == [4, 8, 16, 32]

    def test_heterogeneous_rank_length_mismatch_raises(self, tmp_path):
        p = _write(tmp_path, """
            model_id: t
            dataset: {name: t, split: train}
            num_clients: 4
            num_rounds: 1
            strategies: [fedex-lora]
            rank: 4
            heterogeneous_rank: [4, 8]
            seeds: [0]
        """)
        with pytest.raises(EvalConfigError, match="length"):
            EvalConfig.from_yaml(p)


class TestUnknownKeys:
    def test_from_dict_warns_on_unknown_keys(self, caplog):
        """Unknown YAML keys should be ignored with a warning, not silently dropped."""
        data = {
            "model_id": "x",
            "dataset": {"name": "t", "split": "train"},
            "num_clients": 2,
            "num_rounds": 1,
            "strategies": ["fedex-lora"],
            "rank": 4,
            "seeds": [0],
            "stratigies": ["fedavg"],  # typo of "strategies" — should be flagged
            "totally_made_up_key": 42,
        }
        with caplog.at_level(logging.WARNING, logger="chorus.eval.config"):
            EvalConfig.from_dict(data)
        msgs = " ".join(r.message for r in caplog.records)
        assert "stratigies" in msgs
        assert "totally_made_up_key" in msgs


class TestValidation:
    def test_num_clients_must_be_positive(self, tmp_path):
        p = _write(tmp_path, """
            model_id: t
            dataset: {name: t, split: train}
            num_clients: 0
            num_rounds: 1
            strategies: [fedex-lora]
            rank: 4
            seeds: [0]
        """)
        with pytest.raises(EvalConfigError, match="num_clients"):
            EvalConfig.from_yaml(p)

    def test_seeds_must_be_nonempty(self, tmp_path):
        p = _write(tmp_path, """
            model_id: t
            dataset: {name: t, split: train}
            num_clients: 2
            num_rounds: 1
            strategies: [fedex-lora]
            rank: 4
            seeds: []
        """)
        with pytest.raises(EvalConfigError, match="seeds"):
            EvalConfig.from_yaml(p)
