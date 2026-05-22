"""Tests for the sweep-expansion helper used by benchmarks/run_all.py."""

from pathlib import Path
import textwrap

import pytest


def _import_expand():
    """Import lazily so the test collects even before run_all.py exists."""
    from benchmarks.run_all import expand_sweep
    return expand_sweep


def _write(tmp_path: Path, yaml_text: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(yaml_text).strip() + "\n")
    return p


class TestSingleton:
    def test_yaml_with_no_sweep_yields_one_config(self, tmp_path):
        expand = _import_expand()
        p = _write(tmp_path, """
            model_id: tiny
            dataset: {name: synthetic-tiny, split: train}
            num_clients: 2
            num_rounds: 1
            strategies: [fedex-lora]
            rank: 4
            seeds: [0]
        """)
        runs = list(expand(p))
        assert len(runs) == 1
        run_key, cfg = runs[0]
        assert run_key == "base"
        assert cfg.model_id == "tiny"
        assert cfg.num_clients == 2


class TestCartesianExpansion:
    def test_single_axis_sweep(self, tmp_path):
        expand = _import_expand()
        p = _write(tmp_path, """
            model_id: tiny
            dataset: {name: synthetic-tiny, split: train}
            num_rounds: 1
            strategies: [fedex-lora]
            rank: 4
            seeds: [0]
            sweep:
              num_clients: [2, 5, 10]
        """)
        runs = list(expand(p))
        assert len(runs) == 3
        keys = [k for k, _ in runs]
        assert keys == ["num_clients=2", "num_clients=5", "num_clients=10"]
        client_values = [cfg.num_clients for _, cfg in runs]
        assert client_values == [2, 5, 10]

    def test_two_axis_sweep_is_cartesian(self, tmp_path):
        expand = _import_expand()
        p = _write(tmp_path, """
            model_id: tiny
            dataset: {name: synthetic-tiny, split: train}
            num_clients: 2
            num_rounds: 1
            strategies: [fedex-lora]
            seeds: [0]
            sweep:
              rank: [4, 8]
              fold_residuals: [true, false]
        """)
        runs = list(expand(p))
        assert len(runs) == 4
        keys = sorted(k for k, _ in runs)
        # Order within a key is sweep-key-declared order
        assert keys == sorted([
            "rank=4,fold_residuals=True",
            "rank=4,fold_residuals=False",
            "rank=8,fold_residuals=True",
            "rank=8,fold_residuals=False",
        ])

    def test_sweep_with_null_value_preserved(self, tmp_path):
        """dp_epsilon: [null, 1.0] must keep null as Python None."""
        expand = _import_expand()
        p = _write(tmp_path, """
            model_id: tiny
            dataset: {name: synthetic-tiny, split: train}
            num_clients: 2
            num_rounds: 1
            strategies: [fedex-lora]
            rank: 4
            seeds: [0]
            sweep:
              dp_epsilon: [null, 1.0]
        """)
        runs = list(expand(p))
        assert len(runs) == 2
        eps_values = [cfg.dp_epsilon for _, cfg in runs]
        assert None in eps_values
        assert 1.0 in eps_values


class TestErrors:
    def test_sweep_axis_must_be_list(self, tmp_path):
        expand = _import_expand()
        p = _write(tmp_path, """
            model_id: tiny
            dataset: {name: synthetic-tiny, split: train}
            num_clients: 2
            num_rounds: 1
            strategies: [fedex-lora]
            rank: 4
            seeds: [0]
            sweep:
              num_clients: 5
        """)
        with pytest.raises(ValueError, match="sweep.num_clients"):
            list(expand(p))

    def test_sweep_axis_must_target_known_field(self, tmp_path):
        expand = _import_expand()
        p = _write(tmp_path, """
            model_id: tiny
            dataset: {name: synthetic-tiny, split: train}
            num_clients: 2
            num_rounds: 1
            strategies: [fedex-lora]
            rank: 4
            seeds: [0]
            sweep:
              nonsense_axis: [1, 2]
        """)
        with pytest.raises(ValueError, match="nonsense_axis"):
            list(expand(p))

    def test_sweep_collision_with_base_key_is_rejected(self, tmp_path):
        """If `num_clients: 2` is set at base AND in sweep, that's ambiguous — reject."""
        expand = _import_expand()
        p = _write(tmp_path, """
            model_id: tiny
            dataset: {name: synthetic-tiny, split: train}
            num_clients: 2
            num_rounds: 1
            strategies: [fedex-lora]
            rank: 4
            seeds: [0]
            sweep:
              num_clients: [5, 10]
        """)
        with pytest.raises(ValueError, match="num_clients"):
            list(expand(p))
