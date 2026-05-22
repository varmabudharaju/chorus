"""Drive a sweep of EvalRunner runs from one or more YAML configs.

A sweep YAML looks like:

    model_id: meta-llama/Llama-3.2-1B
    dataset: {name: tatsu-lab/alpaca, split: train, max_examples: 1000}
    num_rounds: 5
    strategies: [fedavg, fedex-lora]
    seeds: [0, 1, 2]
    rank: 8
    sweep:
      num_clients: [2, 5, 10, 20]

The Cartesian product over `sweep:` axes yields N concrete EvalConfig instances;
each gets executed via EvalRunner. Outputs land in
benchmarks/results/v0.2.0/<config-stem>/<run-key>/{report.json,report.md}.

Usage:
    python benchmarks/run_all.py --config benchmarks/configs/foo.yaml
    python benchmarks/run_all.py --all
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

import yaml

from chorus.eval import EvalConfig, EvalRunner

logger = logging.getLogger("chorus.benchmarks.run_all")

# Fields on EvalConfig that may be swept. Anything else is rejected at parse time
# so typos in a sweep YAML fail loudly.
_SWEEPABLE_FIELDS: set[str] = {
    "num_clients",
    "num_rounds",
    "rank",
    "dp_epsilon",
    "dp_delta",
    "dp_max_norm",
    "fold_residuals",
    "partition",
    "dirichlet_alpha",
    "learning_rate",
    "max_steps_per_round",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return data


def expand_sweep(path: str | Path) -> Iterator[tuple[str, EvalConfig]]:
    """Yield (run_key, EvalConfig) pairs from a sweep YAML.

    If no `sweep:` key is present, yields one ("base", EvalConfig) pair.
    Otherwise yields the Cartesian product over the sweep axes; the run_key
    is a comma-joined "axis=value" string in YAML-declared order.
    """
    path = Path(path)
    data = _load_yaml(path)
    sweep = data.pop("sweep", None)
    if sweep is None:
        yield "base", EvalConfig.from_dict(data)
        return

    if not isinstance(sweep, dict):
        raise ValueError(f"{path}: `sweep:` must be a mapping of axis -> list")

    # Validate each axis up front so all errors surface before we start iterating.
    for axis, values in sweep.items():
        if axis not in _SWEEPABLE_FIELDS:
            raise ValueError(
                f"{path}: unknown sweep axis `sweep.{axis}`. "
                f"Allowed: {sorted(_SWEEPABLE_FIELDS)}"
            )
        if not isinstance(values, list):
            raise ValueError(
                f"{path}: `sweep.{axis}` must be a list, got {type(values).__name__}"
            )
        if axis in data:
            raise ValueError(
                f"{path}: `{axis}` set in both base config and `sweep:` — remove one"
            )

    axes = list(sweep.keys())
    value_lists = [sweep[a] for a in axes]
    for combo in itertools.product(*value_lists):
        run_data = dict(data)
        run_key_parts: list[str] = []
        for axis, value in zip(axes, combo):
            run_data[axis] = value
            run_key_parts.append(f"{axis}={value}")
        yield ",".join(run_key_parts), EvalConfig.from_dict(run_data)


def _run_one(
    config_stem: str,
    run_key: str,
    cfg: EvalConfig,
    output_root: Path,
) -> Path:
    out_dir = output_root / config_stem / run_key
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)
    logger.info("Running %s :: %s -> %s", config_stem, run_key, out_dir)
    EvalRunner(cfg).run()
    return out_dir


def run_config(path: Path, output_root: Path) -> list[Path]:
    """Expand a sweep YAML and execute every resulting EvalConfig. Returns output dirs."""
    config_stem = path.stem
    out_dirs: list[Path] = []
    for run_key, cfg in expand_sweep(path):
        out_dirs.append(_run_one(config_stem, run_key, cfg, output_root))
    return out_dirs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="Path to a single YAML config")
    group.add_argument(
        "--all",
        action="store_true",
        help="Iterate every YAML in benchmarks/configs/ except smoke.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmarks/results/v0.2.0"),
        help="Root directory for per-run outputs (default: benchmarks/results/v0.2.0)",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("benchmarks/configs"),
        help="Directory to iterate when --all is given (default: benchmarks/configs)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.config is not None:
        run_config(args.config, args.output_root)
        return 0

    # --all
    yamls = sorted(
        p for p in args.configs_dir.glob("*.yaml") if p.name != "smoke.yaml"
    )
    if not yamls:
        logger.warning("No YAMLs found under %s", args.configs_dir)
        return 0
    for p in yamls:
        run_config(p, args.output_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
