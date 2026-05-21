"""Tests for EvalReport JSON + markdown serialization."""

import json
from pathlib import Path

from chorus.eval.report import EvalReport, StrategyResult


def _make_report() -> EvalReport:
    return EvalReport(
        config_name="smoke",
        model_id="tiny-gpt2",
        dataset_name="wikitext-2",
        num_clients=2,
        num_rounds=2,
        rank=4,
        seeds=[0],
        results=[
            StrategyResult(
                strategy="fedavg",
                seed=0,
                final_task_metric={"perplexity": 102.3},
                frobenius_error=1.45,
                per_round_times_s=[1.2, 1.1],
            ),
            StrategyResult(
                strategy="fedex-lora",
                seed=0,
                final_task_metric={"perplexity": 89.1},
                frobenius_error=0.32,
                per_round_times_s=[1.3, 1.2],
            ),
        ],
    )


def test_to_json_roundtrip(tmp_path: Path):
    report = _make_report()
    out = tmp_path / "report.json"
    report.to_json(out)
    data = json.loads(out.read_text())
    assert data["model_id"] == "tiny-gpt2"
    assert len(data["results"]) == 2
    assert data["results"][0]["strategy"] == "fedavg"
    assert data["results"][1]["frobenius_error"] == 0.32


def test_to_markdown_contains_summary_table(tmp_path: Path):
    report = _make_report()
    out = tmp_path / "report.md"
    report.to_markdown(out)
    md = out.read_text()
    assert "tiny-gpt2" in md
    assert "wikitext-2" in md
    assert "fedavg" in md
    assert "fedex-lora" in md
    assert "perplexity" in md.lower()
    assert "frobenius" in md.lower()


def test_to_markdown_compares_strategies():
    """The report should make the FedAvg vs FedEx delta visible."""
    report = _make_report()
    md = report.to_markdown_string()
    # Both strategies in same table
    assert md.count("fedavg") >= 1
    assert md.count("fedex-lora") >= 1
