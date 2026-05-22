"""Tests for benchmarks/verify_smoke_results.py."""

import json
from pathlib import Path

import pytest


def _import_verify():
    from benchmarks.verify_smoke_results import verify
    return verify


def _write_report(path: Path, fedavg_frob: float, fedex_frob: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "config_name": "smoke",
        "model_id": "tiny",
        "dataset_name": "synthetic-tiny",
        "num_clients": 2,
        "num_rounds": 1,
        "rank": 4,
        "seeds": [0],
        "results": [
            {
                "strategy": "fedavg",
                "seed": 0,
                "final_task_metric": {"perplexity": 10.0},
                "frobenius_error": fedavg_frob,
                "per_round_times_s": [1.0],
                "notes": "",
            },
            {
                "strategy": "fedex-lora",
                "seed": 0,
                "final_task_metric": {"perplexity": 9.5},
                "frobenius_error": fedex_frob,
                "per_round_times_s": [1.0],
                "notes": "",
            },
        ],
    }))


def _write_multiseed_report(
    path: Path,
    fedavg_frobs: list[float],
    fedex_frobs: list[float],
):
    """Write a report with multiple StrategyResults per strategy (one per seed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for seed, frob in enumerate(fedavg_frobs):
        results.append({
            "strategy": "fedavg", "seed": seed, "final_task_metric": {},
            "frobenius_error": frob, "per_round_times_s": [], "notes": "",
        })
    for seed, frob in enumerate(fedex_frobs):
        results.append({
            "strategy": "fedex-lora", "seed": seed, "final_task_metric": {},
            "frobenius_error": frob, "per_round_times_s": [], "notes": "",
        })
    path.write_text(json.dumps({"results": results}))


class TestPasses:
    def test_fedex_strictly_better(self, tmp_path):
        verify = _import_verify()
        report = tmp_path / "report.json"
        _write_report(report, fedavg_frob=1.5, fedex_frob=0.1)
        # Should not raise
        verify(report)

    def test_fedex_equal_within_tolerance(self, tmp_path):
        """FedEx within tolerance of FedAvg is acceptable (float32 noise floor)."""
        from benchmarks.verify_smoke_results import TOLERANCE

        verify = _import_verify()
        report = tmp_path / "report.json"
        # Sit at half of TOLERANCE so this test stays correct even if TOLERANCE
        # is later tightened slightly. The earlier 0.5001 vs 0.5000 value had
        # exactly TOLERANCE of headroom and would flip false-positive on any
        # downward tweak.
        _write_report(
            report,
            fedavg_frob=0.5000,
            fedex_frob=0.5000 + TOLERANCE * 0.5,
        )
        verify(report)


class TestFails:
    def test_missing_report_file(self, tmp_path):
        verify = _import_verify()
        with pytest.raises(FileNotFoundError):
            verify(tmp_path / "does-not-exist.json")

    def test_fedex_worse_than_fedavg(self, tmp_path):
        verify = _import_verify()
        report = tmp_path / "report.json"
        _write_report(report, fedavg_frob=0.1, fedex_frob=1.5)
        with pytest.raises(AssertionError, match="frobenius"):
            verify(report)

    def test_missing_strategy(self, tmp_path):
        verify = _import_verify()
        report = tmp_path / "report.json"
        report.write_text(json.dumps({
            "results": [
                {"strategy": "fedavg", "seed": 0, "frobenius_error": 1.0,
                 "final_task_metric": {}, "per_round_times_s": [], "notes": ""},
            ],
        }))
        with pytest.raises(AssertionError, match="fedex"):
            verify(report)


class TestMultiSeedReducer:
    def test_uses_mean_not_min_across_seeds(self, tmp_path):
        """FedEx beats FedAvg on min but loses on mean — verify must fail.

        With `min`, fedex.min=0.05 < fedavg.min=0.10 would pass. With `mean`,
        fedex.mean=(0.05+2.0)/2=1.025 > fedavg.mean=(0.10+0.20)/2=0.15, so
        verify must raise. This guards against silently regressing back to a
        cherry-picking reducer.
        """
        verify = _import_verify()
        report = tmp_path / "report.json"
        _write_multiseed_report(
            report,
            fedavg_frobs=[0.10, 0.20],
            fedex_frobs=[0.05, 2.00],
        )
        with pytest.raises(AssertionError, match="mean frobenius"):
            verify(report)

    def test_mean_passes_when_both_seeds_within_tolerance(self, tmp_path):
        """Two-seed report where FedEx is consistently below FedAvg should pass."""
        verify = _import_verify()
        report = tmp_path / "report.json"
        _write_multiseed_report(
            report,
            fedavg_frobs=[0.50, 0.60],
            fedex_frobs=[0.10, 0.20],
        )
        verify(report)


def test_cli_exit_code_pass(tmp_path):
    """The `if __name__ == '__main__'` path exits 0 on a passing report."""
    from benchmarks import verify_smoke_results

    report = tmp_path / "report.json"
    _write_report(report, fedavg_frob=1.0, fedex_frob=0.5)

    rc = verify_smoke_results.main([str(report)])
    assert rc == 0


def test_cli_exit_code_fail(tmp_path):
    from benchmarks import verify_smoke_results

    report = tmp_path / "report.json"
    _write_report(report, fedavg_frob=0.1, fedex_frob=2.0)

    rc = verify_smoke_results.main([str(report)])
    assert rc != 0
