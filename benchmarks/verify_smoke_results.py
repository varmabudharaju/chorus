"""Post-run tripwire: assert the smoke benchmark didn't regress.

Runs after `python benchmarks/run_all.py --config benchmarks/configs/smoke.yaml`
on rented compute. Fails loudly if FedEx-LoRA is worse than FedAvg on Frobenius
reconstruction error — that would invalidate the core claim of the project.

Usage:
    python benchmarks/verify_smoke_results.py path/to/report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Frobenius float32 reconstruction tolerance — same value the eval harness uses.
TOLERANCE = 1e-4


def verify(report_path: str | Path) -> None:
    """Load a report.json and assert FedEx-LoRA's Frobenius <= FedAvg's (within tolerance).

    Across multiple seeds, both strategies' errors are reduced to their mean.
    The earlier reducer was `min`, which let each strategy cherry-pick its best
    seed independently — a fair regression check needs a stable summary.

    Raises:
        FileNotFoundError: if the report does not exist.
        AssertionError: if a required strategy is missing or FedEx is worse than FedAvg.
    """
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    data = json.loads(report_path.read_text())
    results = data.get("results", [])

    by_strategy: dict[str, list[dict]] = {}
    for r in results:
        by_strategy.setdefault(r["strategy"], []).append(r)

    if "fedavg" not in by_strategy:
        raise AssertionError(f"{report_path}: missing fedavg results")
    if "fedex-lora" not in by_strategy:
        raise AssertionError(f"{report_path}: missing fedex-lora results")

    fedavg_frob = statistics.mean(r["frobenius_error"] for r in by_strategy["fedavg"])
    fedex_frob = statistics.mean(r["frobenius_error"] for r in by_strategy["fedex-lora"])

    if fedex_frob > fedavg_frob + TOLERANCE:
        raise AssertionError(
            f"{report_path}: fedex-lora mean frobenius_error ({fedex_frob:.6f}) > "
            f"fedavg ({fedavg_frob:.6f}) + tolerance ({TOLERANCE}) — "
            f"the exact-aggregation claim is contradicted by this smoke run."
        )

    print(
        f"OK: fedex-lora mean frobenius ({fedex_frob:.6f}) <= "
        f"fedavg ({fedavg_frob:.6f}) + tol ({TOLERANCE})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        help="Path to a report.json produced by EvalRunner",
    )
    args = parser.parse_args(argv)

    try:
        verify(args.report)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
