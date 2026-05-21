"""Tests for EvalRunner end-to-end behavior on a tiny synthetic setup."""

import logging
import os
from pathlib import Path

import pytest
import torch


# Mark this whole module as requiring the peft extra
peft = pytest.importorskip("peft", reason="EvalRunner requires the [peft] extra")
transformers = pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")


def _write_smoke_config(tmp_path: Path) -> Path:
    """Write a YAML config that runs on a tiny model + tiny synthetic dataset."""
    cfg_text = """
model_id: hf-internal-testing/tiny-random-LlamaForCausalLM
dataset:
  name: synthetic-tiny
  split: train
  max_examples: 16
num_clients: 2
num_rounds: 1
strategies: [fedavg, fedex-lora]
rank: 4
seeds: [0]
max_steps_per_round: 2
eval_batch_size: 2
target_modules: [q_proj, v_proj]
"""
    p = tmp_path / "smoke.yaml"
    p.write_text(cfg_text)
    return p


def test_runner_check_only_does_not_train(tmp_path: Path):
    """--check-only must succeed without loading the model or training."""
    from chorus.eval import EvalConfig, EvalRunner

    cfg_path = _write_smoke_config(tmp_path)
    cfg = EvalConfig.from_yaml(cfg_path)
    runner = EvalRunner(cfg)
    # check_only should pass validation without loading model or running training
    runner.check_only()


def test_multi_round_logs_warning(caplog, tmp_path):
    """num_rounds > 1 should log a warning explaining the collapsed semantics."""
    if os.environ.get("CHORUS_SKIP_HF_NETWORK"):
        pytest.skip("Skipping test that requires HF model download")
    from chorus.eval import EvalConfig, EvalRunner

    cfg = EvalConfig.from_dict({
        "model_id": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "dataset": {"name": "synthetic-tiny", "split": "train", "max_examples": 4},
        "num_clients": 1,
        "num_rounds": 3,
        "strategies": ["fedex-lora"],
        "rank": 4,
        "seeds": [0],
        "max_steps_per_round": 1,
    })
    runner = EvalRunner(cfg)
    with caplog.at_level(logging.WARNING, logger="chorus.eval"):
        runner.run()
    assert any("collapses multi-round" in r.message for r in caplog.records)


def test_evaluate_aggregated_warns_on_no_key_match(caplog, tmp_path):
    """If aggregated keys don't match any PEFT state_dict key, log a warning."""
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    from chorus.eval import EvalConfig, EvalRunner

    cfg = EvalConfig.from_dict({
        "model_id": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "dataset": {"name": "synthetic-tiny", "split": "train", "max_examples": 4},
        "num_clients": 1,
        "num_rounds": 1,
        "strategies": ["fedex-lora"],
        "rank": 4,
        "seeds": [0],
        "max_steps_per_round": 1,
        "target_modules": ["q_proj", "v_proj"],
    })
    runner = EvalRunner(cfg)
    # Aggregated with bogus keys that won't match any PEFT layout
    bogus = {
        "nonexistent.lora_A.weight": torch.zeros(4, 8),
        "nonexistent.lora_B.weight": torch.zeros(8, 4),
    }
    eval_data = [{"text": "hi"}]
    with caplog.at_level(logging.WARNING, logger="chorus.eval"):
        try:
            runner._evaluate_aggregated(bogus, eval_data)
        except Exception:
            pass  # downstream perplexity calc may fail; we only care about the warning
    assert any("No aggregated adapter weights matched" in r.message for r in caplog.records)


@pytest.mark.network
def test_runner_run_produces_report_with_both_strategies(tmp_path: Path):
    """Full run on a tiny model should produce a report with results for each strategy."""
    if os.environ.get("CHORUS_SKIP_HF_NETWORK"):
        pytest.skip("Skipping test that requires HF model download")

    from chorus.eval import EvalConfig, EvalRunner

    cfg_path = _write_smoke_config(tmp_path)
    cfg = EvalConfig.from_yaml(cfg_path)
    cfg.output_dir = str(tmp_path / "out")

    report = EvalRunner(cfg).run()

    assert report.model_id == "hf-internal-testing/tiny-random-LlamaForCausalLM"
    assert report.num_clients == 2
    # One StrategyResult per (strategy, seed)
    assert len(report.results) == 2
    strategies = {r.strategy for r in report.results}
    assert strategies == {"fedavg", "fedex-lora"}
    for r in report.results:
        assert isinstance(r.frobenius_error, float)
        assert r.frobenius_error >= 0
