"""Tests for the `chorus eval` CLI."""

from pathlib import Path
import textwrap

from click.testing import CliRunner

from chorus.cli.main import cli


def _write_min_config(tmp_path: Path) -> Path:
    p = tmp_path / "min.yaml"
    p.write_text(
        textwrap.dedent("""
        model_id: hf-internal-testing/tiny-random-LlamaForCausalLM
        dataset: {name: synthetic-tiny, split: train, max_examples: 4}
        num_clients: 2
        num_rounds: 1
        strategies: [fedex-lora]
        rank: 4
        seeds: [0]
        max_steps_per_round: 1
    """).strip()
        + "\n"
    )
    return p


def test_eval_help_works():
    runner = CliRunner()
    res = runner.invoke(cli, ["eval", "--help"])
    assert res.exit_code == 0
    assert "--config" in res.output
    assert "--check-only" in res.output


def test_eval_check_only_succeeds(tmp_path):
    runner = CliRunner()
    cfg = _write_min_config(tmp_path)
    res = runner.invoke(cli, ["eval", "--config", str(cfg), "--check-only"])
    assert res.exit_code == 0, res.output
    assert "OK" in res.output or "check" in res.output.lower()


def test_eval_check_only_rejects_bad_config(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("model_id: x\n")  # missing required fields
    runner = CliRunner()
    res = runner.invoke(cli, ["eval", "--config", str(p), "--check-only"])
    assert res.exit_code != 0
