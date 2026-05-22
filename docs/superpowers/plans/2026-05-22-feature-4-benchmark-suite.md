# Feature 4: Benchmark Suite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Before starting, ensure you are in the worktree at `~/chorus-worktrees/feat-benchmark-suite` on branch `feat/benchmark-suite` (created via `git worktree add` from the parent checkout at `~/chorus`).

**Goal:** Land the *code* of the v0.2.0 paper-appendix benchmark suite — YAML config files for each ablation axis, a `run_all.py` driver that expands sweep axes and runs `EvalRunner` over the cartesian product, and a `verify_smoke_results.py` CI tripwire that asserts the smoke output exists and that FedEx-LoRA's Frobenius reconstruction error is not worse than FedAvg's. The *published numbers* are out-of-scope for this PR — they will be generated in a single paid GPU burst after F4 merges (~80 GPU-hours on a single A100), and land in a follow-up `chore/v0.2.0-benchmark-results` PR.

**Architecture:** Three additions plus one move:
1. `benchmarks/legacy/benchmark.py` — the existing synthetic-data + timing script gets archived intact under `benchmarks/legacy/`. It is no longer the canonical benchmark; that role passes to `chorus.eval` + the new YAMLs.
2. `benchmarks/configs/*.yaml` — one YAML per experiment family. Each YAML either describes a single concrete `EvalConfig` (singletons like `smoke.yaml`, which already exists) or a *sweep* YAML that lists multiple values for one or more axes. Sweep YAMLs have a `sweep:` top-level key whose subkeys map sweep-axis names (e.g., `num_clients`, `rank`, `fold_residuals`, `dp_epsilon`) to lists; everything else under the YAML root is the shared base config. Sweep expansion is a Cartesian product over all listed axes.
3. `benchmarks/run_all.py` — CLI driver that takes `--config <yaml>` (or `--all` to iterate every YAML in `benchmarks/configs/` except `smoke.yaml`), expands sweeps into individual `EvalConfig` instances, and invokes `EvalRunner.run()` for each. Per-run outputs land in `benchmarks/results/v0.2.0/<config-stem>/<run-key>/{report.json,report.md}`; an aggregate `summary.md` per config gets written alongside.
4. `benchmarks/verify_smoke_results.py` — script that loads a smoke run's `report.json` and asserts (a) the file exists, (b) it contains at least one StrategyResult for `fedex-lora` and one for `fedavg`, (c) `fedex-lora`'s Frobenius reconstruction error is `<=` `fedavg`'s within tolerance. Intended for the post-`run_all.py` step on rented compute; not added to CI in F4 (CI already runs `chorus eval --check-only` from F3).

**No changes to `chorus/eval/`.** `EvalConfig`, `EvalRunner`, and friends are reused as-is. Sweep expansion is a `run_all.py`-local concept; `EvalConfig` continues to represent a single concrete run.

**Tech Stack:** No new runtime dependencies. Uses `chorus.eval`, `pyyaml` (already a dep), `rich` for the CLI (already a dep, used elsewhere). Tests use the existing `pytest` harness; nothing here needs `@pytest.mark.network` because the run_all and verify scripts can be exercised against `synthetic-tiny` or a stub `EvalRunner`.

**Spec section:** `docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md` §4.4
**Master plan:** `docs/superpowers/plans/2026-05-19-phase-1-execution-plan.md` F4 row

---

## Hard-won context from F1–F3 to respect

Read `docs/superpowers/handoff/2026-05-22-session-handoff.md` first if not yet done. Specifically:

- **No `Co-Authored-By: Claude` trailers. No "Generated with Claude Code" footers.** Sole attribution. See master plan §3.
- **Conventional commits**: `feat(scope):`, `fix(scope):`, `docs(scope):`, `test(scope):`, `chore(scope):`, `ci:`.
- **Network-gated tests** carry `@pytest.mark.network`. F4 should not need any.
- **`chorus.eval` quirks already documented**: `partition_non_iid_dirichlet` already pins `cuts[-1] = len(idxs)`; `to_markdown_string` already guards non-numeric metric values; PEFT key prefix handling already lives in `_evaluate_aggregated`. F4 does not re-touch these.
- **Frobenius float32 tolerance is ~1e-4**. The `verify_smoke_results.py` comparison must use a tolerance, not strict `<`. Default: `fedex_err <= fedavg_err + 1e-4`.
- **`chorus eval --check-only` is the existing CI smoke step.** F4 must not regress it.

---

## File Structure

- **Move:** `benchmarks/benchmark.py` → `benchmarks/legacy/benchmark.py` (verbatim; preserve `git log --follow` history).
- **Create:** `benchmarks/legacy/__init__.py` — empty marker (so Python tooling treats it as a package; not strictly required since nothing imports it, but ruff/pytest discovery is happier).
- **Create:** `benchmarks/configs/tinyllama_glue_sst2.yaml` — small-model classification sanity check.
- **Create:** `benchmarks/configs/phi3_alpaca_clients_sweep.yaml` — sweep `num_clients ∈ {2, 5, 10, 20}` on Phi-3-mini + Alpaca-1k.
- **Create:** `benchmarks/configs/llama3_1b_alpaca_rank_ablation.yaml` — sweep `rank ∈ {4, 8, 16}` on Llama-3.2-1B + Alpaca-1k.
- **Create:** `benchmarks/configs/llama3_1b_alpaca_dp_ablation.yaml` — sweep `dp_epsilon ∈ {null, 1.0, 4.0}` (null = DP off).
- **Create:** `benchmarks/configs/llama3_1b_alpaca_fold_ablation.yaml` — sweep `fold_residuals ∈ {true, false}`.
- **Create:** `benchmarks/configs/llama3_1b_alpaca_hetero_rank.yaml` — single-point config with `heterogeneous_rank: [4, 8, 16, 32]` over 4 clients.
- **Create:** `benchmarks/run_all.py` — sweep expander + driver.
- **Create:** `benchmarks/verify_smoke_results.py` — post-run tripwire.
- **Create:** `tests/test_benchmark_sweep.py` — unit tests for the sweep-expansion helper (the only piece of `run_all.py` with logic worth testing in isolation).
- **Create:** `tests/test_benchmark_verify.py` — unit tests for `verify_smoke_results.py`.
- **Create:** `tests/test_benchmark_run_all.py` — integration test for `run_all.py` with a stubbed `EvalRunner`.

---

## Pre-flight (Task 0)

Run from `~/chorus` (the main checkout). The planning PR (this doc) has already merged into master before you reach this step; pull first.

```bash
cd ~/chorus
git checkout master && git pull origin master
git worktree add ../chorus-worktrees/feat-benchmark-suite -b feat/benchmark-suite
cd ../chorus-worktrees/feat-benchmark-suite
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,peft,privacy]"
```

All subsequent paths in this plan are relative to `~/chorus-worktrees/feat-benchmark-suite`.

---

## Task 1: Open GitHub issue

- [ ] **Step 1: Create the issue**

```bash
gh issue create \
  --title "[Phase 1.4] Add benchmark suite (configs + run_all + verify_smoke)" \
  --label "phase-1" \
  --body "$(cat <<'EOF'
**Phase:** 1 (Credibility & Honesty)
**Spec:** [docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md](../blob/master/docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md) §4.4
**Plan:** [docs/superpowers/plans/2026-05-22-feature-4-benchmark-suite.md](../blob/master/docs/superpowers/plans/2026-05-22-feature-4-benchmark-suite.md)

## Scope
Land the *code* of the v0.2.0 paper-appendix benchmark suite. Adds YAML configs for each ablation axis, a `run_all.py` sweep driver, a `verify_smoke_results.py` tripwire, and archives the legacy synthetic benchmark under `benchmarks/legacy/`. The published numbers (~80 GPU-hours on A100) are out of scope for this PR; they land later in a `chore/v0.2.0-benchmark-results` PR.

## Acceptance criteria
- [ ] `benchmarks/benchmark.py` moved to `benchmarks/legacy/benchmark.py` with history preserved.
- [ ] One YAML config per ablation axis (clients sweep, rank ablation, DP ablation, fold ablation, heterogeneous rank, plus the TinyLlama GLUE-SST2 sanity check).
- [ ] `python benchmarks/run_all.py --config <yaml>` expands sweep axes and runs `EvalRunner` for each combination; writes `report.json` + `report.md` per run under `benchmarks/results/v0.2.0/<config-stem>/<run-key>/`.
- [ ] `python benchmarks/run_all.py --all` iterates every YAML in `benchmarks/configs/` except `smoke.yaml`.
- [ ] `python benchmarks/verify_smoke_results.py <results-dir>` exits 0 when FedEx-LoRA's Frobenius error is `<=` FedAvg's (within tolerance) and at least one StrategyResult exists per strategy. Nonzero on missing files or regression.
- [ ] Tests cover: sweep expansion (Cartesian product, singletons, errors), `verify_smoke_results.py` (pass + fail + missing-file paths), `run_all.py` integration via stubbed `EvalRunner`.
- [ ] All existing 225 tests still pass. Ruff clean.

## Out of scope
- Running the full benchmark on rented GPU. That happens after F4 merges, in a separate paid burst.
- Authoring `docs/honest-tradeoffs.md` or rewriting README — that's F5.
- Adding new aggregation strategies or new metrics. Reuses `chorus.eval` as-is.
- Distributed / multi-GPU execution inside `run_all.py`. Sequential per-run; users on multi-GPU rigs can shard configs across processes themselves.
EOF
)"
```

Note the issue number; you'll reference it in the PR body.

---

## Task 2: Archive the legacy synthetic benchmark

**Files:**
- Move: `benchmarks/benchmark.py` → `benchmarks/legacy/benchmark.py`
- Create: `benchmarks/legacy/__init__.py`

- [ ] **Step 1: Use `git mv` so history follows**

```bash
mkdir -p benchmarks/legacy
git mv benchmarks/benchmark.py benchmarks/legacy/benchmark.py
touch benchmarks/legacy/__init__.py
```

- [ ] **Step 2: Add a one-line module docstring at the top of `benchmarks/legacy/__init__.py`**

```python
"""Archived: legacy synthetic-data benchmark from v0.1.0. Kept for reference; not run by CI or `run_all.py`."""
```

- [ ] **Step 3: Verify the legacy script still imports cleanly (no path-dependent imports)**

```bash
python -c "from benchmarks.legacy import benchmark"
```

If it errors due to `python -c` not having `benchmarks/` on `sys.path`, prefix with `PYTHONPATH=.`:

```bash
PYTHONPATH=. python -c "from benchmarks.legacy import benchmark"
```

It should print no output. If imports break, *do not* try to fix the legacy script — that's archived; instead update this step's description to record the breakage and leave the file as-is. The legacy script is not on the supported path.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/legacy/
git commit -m "chore(benchmarks): archive legacy synthetic benchmark under benchmarks/legacy/"
```

---

## Task 3: Sweep-expansion helper (TDD — failing tests first)

The only logic in `run_all.py` worth unit-testing is the function that takes a sweep YAML dict and returns a list of `EvalConfig` instances (one per Cartesian-product point). Implement it as a standalone function so it's testable without invoking `EvalRunner`.

**Files:**
- Create: `tests/test_benchmark_sweep.py`

- [ ] **Step 1: Failing tests**

```python
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
```

```bash
pytest tests/test_benchmark_sweep.py -v 2>&1 | tail -15
git add tests/test_benchmark_sweep.py
git commit -m "test(benchmarks): add sweep-expansion tests (RED)"
```

Expected: import-error (module not yet created). That's the RED state.

---

## Task 4: Implement `run_all.py` (just enough to make Task 3 tests pass)

**Files:**
- Create: `benchmarks/run_all.py`

The function `expand_sweep(path)` is implemented first; the CLI shell around it comes in Task 5.

- [ ] **Step 1: Create `benchmarks/run_all.py`**

```python
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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

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
```

- [ ] **Step 2: Run sweep tests (GREEN)**

```bash
pytest tests/test_benchmark_sweep.py -v
```

All 7 tests should pass.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/run_all.py
git commit -m "feat(benchmarks): add run_all.py with sweep-expansion helper"
```

---

## Task 5: Test `run_all.py` end-to-end with a stubbed `EvalRunner`

**Files:**
- Create: `tests/test_benchmark_run_all.py`

The goal is to verify `run_config` correctly hands every expanded `EvalConfig` to an `EvalRunner` and that each run produces its own output directory — without paying the cost of real training.

- [ ] **Step 1: Tests with monkeypatched `EvalRunner`**

```python
"""Integration tests for benchmarks/run_all.py with EvalRunner stubbed out."""

from pathlib import Path
import textwrap

import pytest


def _write_sweep(tmp_path: Path) -> Path:
    p = tmp_path / "tiny_sweep.yaml"
    p.write_text(textwrap.dedent("""
        model_id: tiny
        dataset: {name: synthetic-tiny, split: train}
        num_rounds: 1
        strategies: [fedex-lora]
        seeds: [0]
        rank: 4
        sweep:
          num_clients: [2, 5]
    """).strip() + "\n")
    return p


def test_run_config_dispatches_one_runner_per_combo(tmp_path, monkeypatch):
    """expand_sweep yields 2 combos -> EvalRunner.run is called twice."""
    from benchmarks import run_all

    call_log: list[dict] = []

    class StubRunner:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            # Capture cfg + write a minimal report.json so downstream tooling can find it.
            Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.cfg.output_dir) / "report.json").write_text("{}")
            call_log.append({
                "num_clients": self.cfg.num_clients,
                "output_dir": self.cfg.output_dir,
            })

    monkeypatch.setattr(run_all, "EvalRunner", StubRunner)

    yaml_path = _write_sweep(tmp_path)
    out_root = tmp_path / "out"
    out_dirs = run_all.run_config(yaml_path, out_root)

    assert len(out_dirs) == 2
    assert len(call_log) == 2
    seen_clients = sorted(c["num_clients"] for c in call_log)
    assert seen_clients == [2, 5]

    # Output directories are per run_key and exist on disk
    for out_dir in out_dirs:
        assert out_dir.exists()
        assert (out_dir / "report.json").exists()


def test_run_config_writes_per_run_subdirs(tmp_path, monkeypatch):
    """Output layout: <output_root>/<config_stem>/<run_key>/"""
    from benchmarks import run_all

    class StubRunner:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.cfg.output_dir) / "report.json").write_text("{}")

    monkeypatch.setattr(run_all, "EvalRunner", StubRunner)

    yaml_path = _write_sweep(tmp_path)
    out_root = tmp_path / "out"
    run_all.run_config(yaml_path, out_root)

    # Stem of the YAML file = tiny_sweep
    config_dir = out_root / "tiny_sweep"
    assert config_dir.exists()
    subdirs = sorted(p.name for p in config_dir.iterdir() if p.is_dir())
    assert subdirs == ["num_clients=2", "num_clients=5"]


def test_main_all_skips_smoke_yaml(tmp_path, monkeypatch):
    """--all iterates every YAML in configs/ except smoke.yaml."""
    from benchmarks import run_all

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

    # Two real configs + smoke (which must be skipped)
    for name in ["a.yaml", "b.yaml", "smoke.yaml"]:
        (configs_dir / name).write_text(textwrap.dedent(f"""
            model_id: {name}
            dataset: {{name: synthetic-tiny, split: train}}
            num_clients: 2
            num_rounds: 1
            strategies: [fedex-lora]
            rank: 4
            seeds: [0]
        """).strip() + "\n")

    seen_model_ids: list[str] = []

    class StubRunner:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.cfg.output_dir) / "report.json").write_text("{}")
            seen_model_ids.append(self.cfg.model_id)

    monkeypatch.setattr(run_all, "EvalRunner", StubRunner)

    run_all.main([
        "--all",
        "--configs-dir", str(configs_dir),
        "--output-root", str(tmp_path / "out"),
    ])

    assert sorted(seen_model_ids) == ["a.yaml", "b.yaml"]
    assert "smoke.yaml" not in seen_model_ids
```

```bash
pytest tests/test_benchmark_run_all.py -v
git add tests/test_benchmark_run_all.py
git commit -m "test(benchmarks): integration tests for run_all.py with stubbed EvalRunner"
```

If any test fails, fix `run_all.py` (not the tests). The `StubRunner` pattern is the contract: `run_all.py` must construct `EvalRunner(cfg)`, call `.run()`, and respect `cfg.output_dir`.

---

## Task 6: `verify_smoke_results.py` (TDD)

**Files:**
- Create: `tests/test_benchmark_verify.py`

- [ ] **Step 1: Failing tests**

```python
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


class TestPasses:
    def test_fedex_strictly_better(self, tmp_path):
        verify = _import_verify()
        report = tmp_path / "report.json"
        _write_report(report, fedavg_frob=1.5, fedex_frob=0.1)
        # Should not raise
        verify(report)

    def test_fedex_equal_within_tolerance(self, tmp_path):
        """FedEx within 1e-4 of FedAvg is acceptable (float32 noise floor)."""
        verify = _import_verify()
        report = tmp_path / "report.json"
        _write_report(report, fedavg_frob=0.5000, fedex_frob=0.5001)
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
```

```bash
pytest tests/test_benchmark_verify.py -v 2>&1 | tail -15
git add tests/test_benchmark_verify.py
git commit -m "test(benchmarks): add verify_smoke_results tests (RED)"
```

- [ ] **Step 2: Implement `benchmarks/verify_smoke_results.py`**

```python
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
import sys
from pathlib import Path

# Frobenius float32 reconstruction tolerance — same value the eval harness uses.
TOLERANCE = 1e-4


def verify(report_path: str | Path) -> None:
    """Load a report.json and assert FedEx-LoRA's Frobenius <= FedAvg's (within tolerance).

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

    # Compare best (i.e., minimum) Frobenius error per strategy across seeds.
    fedavg_frob = min(r["frobenius_error"] for r in by_strategy["fedavg"])
    fedex_frob = min(r["frobenius_error"] for r in by_strategy["fedex-lora"])

    if fedex_frob > fedavg_frob + TOLERANCE:
        raise AssertionError(
            f"{report_path}: fedex-lora frobenius_error ({fedex_frob:.6f}) > "
            f"fedavg ({fedavg_frob:.6f}) + tolerance ({TOLERANCE}) — "
            f"the exact-aggregation claim is contradicted by this smoke run."
        )

    print(
        f"OK: fedex-lora frobenius ({fedex_frob:.6f}) <= "
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
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_benchmark_verify.py -v
git add benchmarks/verify_smoke_results.py
git commit -m "feat(benchmarks): add verify_smoke_results.py tripwire"
```

---

## Task 7: Author the YAML configs

These YAMLs are the *recipe* for the v0.2.0 published numbers. They're hand-authored, not generated. Each one declares the experiment shape; `run_all.py` does the rest.

Treat the exact model IDs and dataset slices as *defaults*, not commitments — the user can substitute equivalents if licensing or availability changes. Field choices below come from spec §4.4 and §2 "Success criteria."

- [ ] **Step 1: `benchmarks/configs/tinyllama_glue_sst2.yaml`**

```yaml
# TinyLlama on GLUE/SST-2 (sentiment classification).
# Small-model sanity check: confirms the pipeline works on a real classification
# task, not just synthetic LM data. ~5 min/run on CPU; ~30s on A100.

model_id: TinyLlama/TinyLlama-1.1B-Chat-v1.0
dataset:
  name: glue
  config_name: sst2
  split: train
  max_examples: 2000
num_clients: 5
num_rounds: 3
strategies: [fedavg, fedex-lora]
rank: 8
seeds: [0, 1, 2]
max_steps_per_round: 100
eval_batch_size: 8
target_modules: [q_proj, v_proj]
partition: iid
output_dir: benchmarks/results/v0.2.0/tinyllama_glue_sst2
```

- [ ] **Step 2: `benchmarks/configs/phi3_alpaca_clients_sweep.yaml`**

```yaml
# Phi-3-mini on Alpaca-1k: sweep num_clients ∈ {2, 5, 10, 20}.
# Measures how aggregation quality degrades as the federation grows.
# ~15 min/run on A100; 4 client points × 3 seeds × 2 strategies = 24 runs ≈ 6 GPU-hr.

model_id: microsoft/Phi-3-mini-4k-instruct
dataset:
  name: tatsu-lab/alpaca
  split: train
  max_examples: 1000
num_rounds: 5
strategies: [fedavg, fedex-lora]
rank: 8
seeds: [0, 1, 2]
max_steps_per_round: 200
eval_batch_size: 4
target_modules: [q_proj, v_proj]
partition: iid
sweep:
  num_clients: [2, 5, 10, 20]
```

(No `num_clients` at the base level — the sweep supplies it.)

- [ ] **Step 3: `benchmarks/configs/llama3_1b_alpaca_rank_ablation.yaml`**

```yaml
# Llama-3.2-1B on Alpaca-1k: sweep rank ∈ {4, 8, 16}.
# Tests whether the FedEx-LoRA advantage scales with rank.
# ~10 min/run on A100; 3 ranks × 3 seeds × 2 strategies = 18 runs ≈ 3 GPU-hr.

model_id: meta-llama/Llama-3.2-1B
dataset:
  name: tatsu-lab/alpaca
  split: train
  max_examples: 1000
num_clients: 10
num_rounds: 5
strategies: [fedavg, fedex-lora]
seeds: [0, 1, 2]
max_steps_per_round: 200
eval_batch_size: 4
target_modules: [q_proj, v_proj]
partition: iid
sweep:
  rank: [4, 8, 16]
```

- [ ] **Step 4: `benchmarks/configs/llama3_1b_alpaca_dp_ablation.yaml`**

```yaml
# Llama-3.2-1B on Alpaca-1k: sweep dp_epsilon ∈ {null, 1.0, 4.0}.
# null = DP off; 1.0 = strong DP; 4.0 = relaxed DP.
# Measures the privacy-utility tradeoff at fixed clients/rank.
# 3 eps × 3 seeds × 2 strategies = 18 runs ≈ 3 GPU-hr.

model_id: meta-llama/Llama-3.2-1B
dataset:
  name: tatsu-lab/alpaca
  split: train
  max_examples: 1000
num_clients: 10
num_rounds: 5
strategies: [fedavg, fedex-lora]
rank: 8
seeds: [0, 1, 2]
max_steps_per_round: 200
eval_batch_size: 4
target_modules: [q_proj, v_proj]
partition: iid
dp_delta: 1.0e-5
dp_max_norm: 1.0
sweep:
  dp_epsilon: [null, 1.0, 4.0]
```

- [ ] **Step 5: `benchmarks/configs/llama3_1b_alpaca_fold_ablation.yaml`**

```yaml
# Llama-3.2-1B on Alpaca-1k: sweep fold_residuals ∈ {true, false}.
# The headline "mathematically exact" claim holds only when fold_residuals=true.
# This ablation makes the cost of fold-off explicit in the published numbers.
# 2 fold × 3 seeds × 2 strategies = 12 runs ≈ 2 GPU-hr.

model_id: meta-llama/Llama-3.2-1B
dataset:
  name: tatsu-lab/alpaca
  split: train
  max_examples: 1000
num_clients: 10
num_rounds: 5
strategies: [fedavg, fedex-lora]
rank: 8
seeds: [0, 1, 2]
max_steps_per_round: 200
eval_batch_size: 4
target_modules: [q_proj, v_proj]
partition: iid
sweep:
  fold_residuals: [true, false]
```

- [ ] **Step 6: `benchmarks/configs/llama3_1b_alpaca_hetero_rank.yaml`**

```yaml
# Llama-3.2-1B on Alpaca-1k with heterogeneous-rank clients.
# Single-point experiment (no sweep): 4 clients with ranks [4, 8, 16, 32].
# Tests that FedEx-LoRA correctly aggregates across mismatched ranks.

model_id: meta-llama/Llama-3.2-1B
dataset:
  name: tatsu-lab/alpaca
  split: train
  max_examples: 1000
num_clients: 4
num_rounds: 5
strategies: [fedavg, fedex-lora]
rank: 8                  # default; per-client overrides below
heterogeneous_rank: [4, 8, 16, 32]
seeds: [0, 1, 2]
max_steps_per_round: 200
eval_batch_size: 4
target_modules: [q_proj, v_proj]
partition: iid
```

- [ ] **Step 7: Smoke-validate every config via `--check-only`**

```bash
for cfg in benchmarks/configs/*.yaml; do
  echo "=== $cfg ==="
  # Sweep YAMLs can't be loaded by `chorus eval` directly (no `num_clients` at root).
  # Expand them locally and check the first run.
  python -c "
from benchmarks.run_all import expand_sweep
runs = list(expand_sweep('$cfg'))
print(f'{len(runs)} run(s) expanded; first key: {runs[0][0]}')
"
done
```

Expected: each prints `N run(s) expanded`. If any config raises, fix that YAML.

- [ ] **Step 8: Commit**

```bash
git add benchmarks/configs/
git commit -m "feat(benchmarks): add per-experiment YAML configs (clients, rank, DP, fold, hetero-rank, GLUE-SST2)"
```

---

## Task 8: Update `benchmarks/results/` README (lightweight provenance note)

**Files:**
- Create: `benchmarks/results/README.md`

This is a placeholder so reviewers know where the numbers will land and how to interpret them once they exist. Reads like project documentation, not implementation.

- [ ] **Step 1: Write the README**

```bash
mkdir -p benchmarks/results
cat > benchmarks/results/README.md <<'EOF'
# Benchmark results

This directory holds the *committed* output of `python benchmarks/run_all.py`.

- `v0.2.0/` — populated by the paid GPU run between F4 and F6 of the v0.2.0
  Phase 1 release. Contains one subdirectory per YAML config under
  `benchmarks/configs/`, with per-run `report.json` + `report.md` underneath.
  Tracked in git so the published numbers are reproducible and reviewable.

- `smoke/` — local-only outputs from the CI smoke config (`benchmarks/configs/smoke.yaml`).
  Not committed; gitignored.

To regenerate v0.2.0 results from scratch (requires ~$80–$160 of A100 time):

    python benchmarks/run_all.py --all --output-root benchmarks/results/v0.2.0
    python benchmarks/verify_smoke_results.py benchmarks/results/v0.2.0/smoke/base/report.json
EOF
```

- [ ] **Step 2: Update root `.gitignore` so smoke runs don't get committed**

If `benchmarks/results/smoke/` is not already gitignored, append to `.gitignore`:

```
benchmarks/results/smoke/
```

Check first:

```bash
grep -q "benchmarks/results/smoke" .gitignore || echo "benchmarks/results/smoke/" >> .gitignore
```

- [ ] **Step 3: Commit**

```bash
git add benchmarks/results/README.md .gitignore
git commit -m "docs(benchmarks): add results README + gitignore smoke outputs"
```

---

## Task 9: Final verification and PR

- [ ] **Step 1: Full suite + ruff**

```bash
ruff check chorus tests benchmarks
pytest tests/ -m "not network" -q --tb=short
```

Expected:
- Ruff clean.
- Test count 225 → ~245 (7 sweep tests + 6 verify tests + 3 run_all tests = 16 new, but a few may merge). At minimum: no regressions, all new tests pass.

- [ ] **Step 2: Manually smoke-run `run_all.py` against the existing smoke YAML**

```bash
python benchmarks/run_all.py --config benchmarks/configs/smoke.yaml --output-root /tmp/chorus-f4-smoke
```

This actually trains the tiny model (network access needed). Expected: completes in ~1–3 min on CPU and produces `/tmp/chorus-f4-smoke/smoke/base/report.json`.

```bash
python benchmarks/verify_smoke_results.py /tmp/chorus-f4-smoke/smoke/base/report.json
```

Expected: exit 0, prints "OK: fedex-lora frobenius ... <= fedavg ...".

If network is unavailable in this environment, skip this step and record the deviation in the PR body; the test suite already covers the wiring.

- [ ] **Step 3: Push**

```bash
git push -u origin feat/benchmark-suite
```

- [ ] **Step 4: Open PR**

```bash
gh pr create \
  --base master \
  --head feat/benchmark-suite \
  --title "[Phase 1.4] Add benchmark suite (configs + run_all + verify_smoke)" \
  --body "$(cat <<'EOF'
## Summary
- Archives the legacy synthetic benchmark under `benchmarks/legacy/` (no functional change; preserved for reference).
- Adds per-experiment YAML configs covering each ablation axis from spec §4.4: TinyLlama-GLUE-SST2 sanity, Phi-3-mini clients sweep, Llama-3.2-1B rank/DP/fold ablations, and a heterogeneous-rank single-point.
- Adds `benchmarks/run_all.py`: expands `sweep:` axes into the Cartesian product of `EvalConfig` instances, runs `EvalRunner` for each, writes results under `benchmarks/results/v0.2.0/<config-stem>/<run-key>/`.
- Adds `benchmarks/verify_smoke_results.py`: post-run tripwire that asserts FedEx-LoRA Frobenius error is no worse than FedAvg (within float32 tolerance).
- 16 new tests (sweep expansion, verify, run_all integration).

## Closes
- #<ISSUE-NUMBER>

## Test plan
- [x] `pytest tests/ -m "not network"` green
- [x] Ruff clean
- [x] `python benchmarks/run_all.py --config benchmarks/configs/smoke.yaml` produces a usable `report.json` end-to-end
- [x] `python benchmarks/verify_smoke_results.py <report.json>` returns 0 on the smoke result
- [x] All six new ablation YAMLs expand via `expand_sweep()` without error

## Notes for reviewer
- This PR lands the **code** of the v0.2.0 benchmark suite. The **published numbers** are a separate paid GPU run (~80 GPU-hours on A100, ~$80–160) that lands in a follow-up `chore/v0.2.0-benchmark-results` PR.
- No changes to `chorus/eval/`. Sweep semantics live entirely in `run_all.py`; `EvalConfig` continues to represent a single concrete run.
- The fold/DP/heterogeneous-rank YAMLs assume `EvalRunner` already honors the `fold_residuals`, `dp_epsilon`, and `heterogeneous_rank` fields end-to-end. F3 wired the fields onto `EvalConfig`; if any of them aren't fully threaded through `EvalRunner.run()`, the missing wiring should be filed as a follow-up issue, not bundled here.
EOF
)"
```

Replace `<ISSUE-NUMBER>` with the real issue number from Task 1.

- [ ] **Step 5: Watch CI**

```bash
gh pr view --json statusCheckRollup --jq '[.statusCheckRollup[] | {name, status, conclusion}]'
```

Wait for all green. Fix any failures.

---

## Self-review checklist

- [ ] `benchmarks/benchmark.py` no longer exists at the original location; `benchmarks/legacy/benchmark.py` does, with `git log --follow` showing the original commits.
- [ ] Every YAML under `benchmarks/configs/` (except `smoke.yaml`) declares either a single concrete EvalConfig or uses `sweep:` for sweep axes — no halfway configs.
- [ ] `expand_sweep()` rejects: unknown sweep axes, non-list values, axes that collide with base-level keys, non-dict top-level YAML.
- [ ] `run_all.py` writes per-run subdirs at `<output_root>/<config_stem>/<run_key>/`.
- [ ] `run_all.py --all` correctly skips `smoke.yaml`.
- [ ] `verify_smoke_results.py` returns 0 on a passing report, 1 on a regression, 2 on a missing file.
- [ ] `verify_smoke_results.TOLERANCE` is `1e-4` (matches the documented Frobenius float32 noise floor).
- [ ] All existing 225 tests still pass; new test count is at least +13.
- [ ] Ruff clean across `chorus`, `tests`, `benchmarks`.
- [ ] No `Co-Authored-By` trailers on any commit; no AI-attribution footer in PR/issue body.
- [ ] PR references `Closes #<issue>` and links to this plan in the body.

---

## Out-of-scope for this PR (do not creep)

- Actually running the full benchmark on a GPU and committing the numbers. That's the separate `chore/v0.2.0-benchmark-results` PR after this one merges.
- Authoring or editing `docs/honest-tradeoffs.md` — F5.
- Editing README. F5.
- Adding new metrics to `chorus.eval.metrics`. If a metric is missing, file an issue.
- Adding new aggregation strategies. Out of scope for v0.2.0 entirely.
- Threading any missing `EvalConfig` field (e.g., `fold_residuals`) through `EvalRunner.run()`. If a YAML option doesn't reach the runner, file an issue; don't fix it here.
- Distributed / multi-GPU dispatch inside `run_all.py`. Sequential is fine for v0.2.0.
- A second comparison library (Flower, PySyft). Spec §10 defers this to Phase 2.
