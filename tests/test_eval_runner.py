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


@pytest.mark.network
def test_multi_round_run_completes(tmp_path):
    """num_rounds > 1 with fedex-lora should complete without error and return a report."""
    if os.environ.get("CHORUS_SKIP_HF_NETWORK"):
        pytest.skip("Skipping test that requires HF model download")
    from chorus.eval import EvalConfig, EvalRunner

    cfg = EvalConfig.from_dict({
        "model_id": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "dataset": {"name": "synthetic-tiny", "split": "train", "max_examples": 4},
        "num_clients": 2,
        "num_rounds": 2,
        "strategies": ["fedex-lora"],
        "rank": 4,
        "seeds": [0],
        "max_steps_per_round": 1,
        "fold_residuals": True,
    })
    cfg.output_dir = str(tmp_path / "out")
    report = EvalRunner(cfg).run()
    assert len(report.results) == 1
    assert report.results[0].strategy == "fedex-lora"
    assert isinstance(report.results[0].frobenius_error, float)


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


# ---------------------------------------------------------------------------
# Issue #19: fold_residuals threading
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_fold_residuals_produces_different_frobenius_than_no_fold(tmp_path: Path):
    """fold_residuals=True vs False must produce different Frobenius errors over num_rounds>=2.

    With fold_residuals=True, each round's FedExLoRA residuals are folded into the
    base model weights before the next round's clients train (chorus/server/weight_manager.py
    fold helper, mirroring the server path). The final round's client deltas
    therefore differ between fold=True and fold=False because they were trained
    against different effective base models. That difference shows up as a
    different per-round Frobenius reconstruction error.
    """
    if os.environ.get("CHORUS_SKIP_HF_NETWORK"):
        pytest.skip("Skipping test that requires HF model download")

    from chorus.eval import EvalConfig, EvalRunner

    base_cfg = {
        "model_id": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "dataset": {"name": "synthetic-tiny", "split": "train", "max_examples": 8},
        "num_clients": 2,
        "num_rounds": 2,
        "strategies": ["fedex-lora"],
        "rank": 4,
        "seeds": [42],
        "max_steps_per_round": 2,
        "target_modules": ["q_proj", "v_proj"],
    }

    cfg_fold = EvalConfig.from_dict({**base_cfg, "fold_residuals": True})
    cfg_fold.output_dir = str(tmp_path / "fold_on")
    report_fold = EvalRunner(cfg_fold).run()

    cfg_nofold = EvalConfig.from_dict({**base_cfg, "fold_residuals": False})
    cfg_nofold.output_dir = str(tmp_path / "fold_off")
    report_nofold = EvalRunner(cfg_nofold).run()

    frob_fold = report_fold.results[0].frobenius_error
    frob_nofold = report_nofold.results[0].frobenius_error

    assert frob_fold != frob_nofold, (
        f"Expected fold_residuals=True ({frob_fold:.6f}) and "
        f"fold_residuals=False ({frob_nofold:.6f}) to produce different Frobenius errors, "
        "but they are identical. The fold_residuals path is not being exercised."
    )
    # The directional claim (fold reduces or increases error) depends on data
    # and noise floor; on a tiny synthetic dataset both signs are plausible. The
    # only contract we lock in here is that the two paths produce *different*
    # numbers, which proves the cross-round base re-injection is wired up.


# ---------------------------------------------------------------------------
# Issue #20: dp_epsilon threading
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_dp_epsilon_produces_different_frobenius_than_no_dp(tmp_path: Path):
    """dp_epsilon=1.0 vs None must produce different Frobenius errors (stochastic noise).

    When dp_epsilon is set, Gaussian noise is added to each client's delta before
    aggregation; the noised deltas produce a different (higher) Frobenius error than
    the clean deltas with dp_epsilon=None.
    """
    if os.environ.get("CHORUS_SKIP_HF_NETWORK"):
        pytest.skip("Skipping test that requires HF model download")

    from chorus.eval import EvalConfig, EvalRunner

    base_cfg = {
        "model_id": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "dataset": {"name": "synthetic-tiny", "split": "train", "max_examples": 8},
        "num_clients": 2,
        "num_rounds": 1,
        "strategies": ["fedex-lora"],
        "rank": 4,
        "seeds": [7],
        "max_steps_per_round": 2,
        "target_modules": ["q_proj", "v_proj"],
        "dp_delta": 1e-5,
        "dp_max_norm": 1.0,
    }

    cfg_dp = EvalConfig.from_dict({**base_cfg, "dp_epsilon": 1.0})
    cfg_dp.output_dir = str(tmp_path / "dp_on")
    report_dp = EvalRunner(cfg_dp).run()

    cfg_nodp = EvalConfig.from_dict({**base_cfg, "dp_epsilon": None})
    cfg_nodp.output_dir = str(tmp_path / "dp_off")
    report_nodp = EvalRunner(cfg_nodp).run()

    frob_dp = report_dp.results[0].frobenius_error
    frob_nodp = report_nodp.results[0].frobenius_error

    assert frob_dp != frob_nodp, (
        f"Expected dp_epsilon=1.0 ({frob_dp:.6f}) and dp_epsilon=None ({frob_nodp:.6f}) "
        "to produce different Frobenius errors, but they are identical. "
        "The dp_epsilon noise path is not being exercised."
    )
