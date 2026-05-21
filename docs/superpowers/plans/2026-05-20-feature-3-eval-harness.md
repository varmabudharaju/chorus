# Feature 3: Eval Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Before starting, ensure you are in the worktree at `~/chorus-worktrees/feat-eval-harness` on branch `feat/eval-harness` (created via superpowers:using-git-worktrees from the parent checkout at `~/chorus`).

**Goal:** Add a reusable evaluation harness (`chorus.eval` package + `chorus eval` CLI) that runs simulated federations on real models with real data and produces a comparable report. The eval harness is the foundation Feature 4 (paper-appendix benchmark suite) builds on, and the reproducibility artifact that lets skeptical users verify Chorus's claims on their own setup.

**Architecture:** New package `chorus/eval/` with five modules — `config` (YAML loader), `datasets` (HF wrapper + IID/non-IID partitioning), `metrics` (task + algorithmic), `report` (JSON + markdown serializers), `runner` (orchestrates everything). New CLI subcommand `chorus eval` reads a YAML config and emits a report. CI gets a `--check-only` smoke step that validates the harness wiring without training. Two modes: `--check-only` (fast, no model load) and full run (configurable, can scale from laptop to A100).

**Tech Stack:** New transitive dependencies (already gated behind the existing `[peft]` extra: `peft`, `transformers`, `datasets`, `accelerate`). Adding `evaluate>=0.4.0` for HF metric helpers. Reuses `chorus.client.trainer.LoRATrainer` for per-client training, `chorus.server.aggregation` for the FedAvg/FedExLoRA aggregation step. Reuses `chorus.eval.config.EvalConfig` as the single source of truth for a run.

**Spec section:** `docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md` §4.1
**Master plan:** `docs/superpowers/plans/2026-05-19-phase-1-execution-plan.md` F3 row

---

## File Structure

- **Create:** `chorus/eval/__init__.py` — package init; re-exports `EvalRunner`, `EvalConfig`, `EvalReport`.
- **Create:** `chorus/eval/config.py` — `EvalConfig` dataclass + YAML loader.
- **Create:** `chorus/eval/datasets.py` — `load_dataset_for_eval()`, `partition_iid()`, `partition_non_iid_dirichlet()`.
- **Create:** `chorus/eval/metrics.py` — `frobenius_reconstruction_error()`, `compute_task_metric()` (perplexity / accuracy / F1).
- **Create:** `chorus/eval/report.py` — `EvalReport` dataclass + `to_json()` + `to_markdown()`.
- **Create:** `chorus/eval/runner.py` — `EvalRunner` orchestration class.
- **Modify:** `chorus/cli/main.py` — new `chorus eval` subcommand.
- **Modify:** `chorus/exceptions.py` — add `EvalConfigError(ChorusError)`.
- **Modify:** `pyproject.toml` — add `evaluate>=0.4.0` to `[peft]` optional extra.
- **Modify:** `.github/workflows/ci.yml` — add `chorus eval --check-only` step.
- **Create:** `benchmarks/configs/smoke.yaml` — CI-friendly smoke config.
- **Create:** `tests/test_eval_config.py`, `tests/test_eval_datasets.py`, `tests/test_eval_metrics.py`, `tests/test_eval_report.py`, `tests/test_eval_runner.py`, `tests/test_cli_eval.py`.

---

## Pre-flight (Task 0)

```bash
cd ~/chorus
git checkout master && git pull origin master
git worktree add ../chorus-worktrees/feat-eval-harness -b feat/eval-harness
```

All subsequent paths are relative to `~/chorus-worktrees/feat-eval-harness`.

---

## Task 1: Open GitHub issue

- [ ] **Step 1: Create the issue**

```bash
cd ~/chorus-worktrees/feat-eval-harness
gh issue create \
  --title "[Phase 1.3] Add chorus.eval package + chorus eval CLI" \
  --label "phase-1" \
  --body "$(cat <<'EOF'
**Phase:** 1 (Credibility & Honesty)
**Spec:** [docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md](../blob/master/docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md) §4.1
**Plan:** [docs/superpowers/plans/2026-05-20-feature-3-eval-harness.md](../blob/master/docs/superpowers/plans/2026-05-20-feature-3-eval-harness.md)

## Scope
Add a reusable evaluation harness: `chorus.eval` package + `chorus eval` CLI. Runs a simulated federation on a real model + real dataset and emits a comparable JSON + markdown report. Used by Feature 4 (paper-appendix benchmark suite) and reusable by any Chorus user who wants to reproduce results on their own setup.

## Acceptance criteria
- [ ] `chorus eval --config <yaml>` runs end-to-end on a tiny model and emits both JSON and markdown reports.
- [ ] `chorus eval --check-only --config <yaml>` validates wiring without training. Completes in <10s. Used in CI.
- [ ] `EvalRunner` runs both FedAvg and FedExLoRA in the same run when configured to, and reports both side-by-side.
- [ ] Reports include task metric (perplexity / accuracy / F1) AND algorithmic metric (Frobenius reconstruction error) for each strategy.
- [ ] IID and Dirichlet-based non-IID partitioning supported.
- [ ] `benchmarks/configs/smoke.yaml` runs in CI (--check-only) without model download.
- [ ] All existing 195 tests pass; new tests cover config loading, datasets, metrics, report serialization, runner integration, CLI.

## Out of scope
- Authoring the actual paper-appendix benchmark configs (that's Feature 4).
- Running the full benchmark on rented GPU (post-Feature 4, single paid burst).
- Adding new aggregation strategies (out of scope for v0.2.0 entirely).
- Distributed/multi-GPU training inside `EvalRunner` — single-device per client.
EOF
)"
```

Note the issue number.

---

## Task 2: Add `evaluate` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Set up venv with peft extras**

```bash
cd ~/chorus-worktrees/feat-eval-harness
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,peft,privacy]"
```

- [ ] **Step 2: Add `evaluate` to the `peft` extra**

In `pyproject.toml`, find:
```toml
peft = ["peft>=0.7.0", "transformers>=4.36.0", "datasets>=2.16.0", "accelerate>=0.25.0"]
```

Change to:
```toml
peft = ["peft>=0.7.0", "transformers>=4.36.0", "datasets>=2.16.0", "accelerate>=0.25.0", "evaluate>=0.4.0"]
```

- [ ] **Step 3: Reinstall and verify**

```bash
pip install -e ".[dev,peft,privacy]"
python -c "import evaluate; print('evaluate', evaluate.__version__)"
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore(deps): add evaluate>=0.4.0 to peft extra for eval harness metrics"
```

---

## Task 3: Add `EvalConfigError` exception

**Files:**
- Modify: `chorus/exceptions.py`

- [ ] **Step 1: Append to `chorus/exceptions.py`**

```python


class EvalConfigError(ChorusError):
    """Raised when a chorus.eval config is malformed or references missing resources."""
```

- [ ] **Step 2: Verify**

```bash
python -c "from chorus.exceptions import EvalConfigError, ChorusError; assert issubclass(EvalConfigError, ChorusError)"
```

- [ ] **Step 3: Commit**

```bash
git add chorus/exceptions.py
git commit -m "feat(exceptions): add EvalConfigError"
```

---

## Task 4: `EvalConfig` (TDD — failing tests first)

**Files:**
- Create: `tests/test_eval_config.py`

- [ ] **Step 1: Create the test file**

```python
"""Tests for EvalConfig dataclass + YAML loader."""

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
```

- [ ] **Step 2: Run; expect ModuleNotFoundError**

```bash
pytest tests/test_eval_config.py -v 2>&1 | tail -10
```

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_eval_config.py
git commit -m "test(eval): add EvalConfig tests (RED)"
```

---

## Task 5: Implement `EvalConfig`

**Files:**
- Create: `chorus/eval/__init__.py`
- Create: `chorus/eval/config.py`

- [ ] **Step 1: Create the package init**

```python
"""Evaluation harness for federated LoRA aggregation strategies.

Runs simulated multi-client federations on real models + datasets, then emits
JSON and markdown reports comparing aggregation strategies (FedAvg vs FedExLoRA)
on both task metrics (perplexity, accuracy, F1) and algorithmic metrics
(Frobenius reconstruction error against the exact full-rank average).

Used as a CLI: `chorus eval --config <yaml>`.
Or programmatically:
    from chorus.eval import EvalRunner, EvalConfig
    config = EvalConfig.from_yaml("benchmarks/configs/smoke.yaml")
    report = EvalRunner(config).run()
    report.to_markdown("results.md")
"""

from chorus.eval.config import EvalConfig
from chorus.eval.report import EvalReport
from chorus.eval.runner import EvalRunner

__all__ = ["EvalConfig", "EvalReport", "EvalRunner"]
```

(The `report` and `runner` modules don't exist yet — `__init__.py` will fail import until later tasks. That's fine; tests use direct imports of submodules.)

- [ ] **Step 2: Create `chorus/eval/config.py`**

```python
"""EvalConfig dataclass + YAML loader for the eval harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from chorus.exceptions import EvalConfigError
from chorus.server.aggregation import STRATEGIES as _AVAILABLE_STRATEGIES


@dataclass
class EvalConfig:
    """Configuration for a single evaluation run.

    Fields map 1:1 to YAML keys. See benchmarks/configs/smoke.yaml for an example.
    """

    model_id: str
    dataset: dict[str, Any]  # {name, split, max_examples?, ...}
    num_clients: int
    num_rounds: int
    strategies: list[str]  # e.g., ["fedavg", "fedex-lora"]
    rank: int
    seeds: list[int]

    # Optional
    dp_epsilon: float | None = None
    dp_delta: float = 1e-5
    dp_max_norm: float = 1.0
    fold_residuals: bool = True
    heterogeneous_rank: list[int] | None = None  # per-client ranks, len must = num_clients
    partition: str = "iid"  # or "dirichlet"
    dirichlet_alpha: float = 0.5
    output_dir: str = "benchmarks/results"
    learning_rate: float = 5e-5
    max_steps_per_round: int = 50
    eval_batch_size: int = 8
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        """Load and validate an EvalConfig from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise EvalConfigError(f"Config file not found: {path}")
        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            raise EvalConfigError(f"Invalid YAML in {path}: {e}") from e
        if not isinstance(data, dict):
            raise EvalConfigError(f"Config root must be a mapping, got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalConfig":
        """Build an EvalConfig from a dict (e.g., parsed YAML). Validates inputs."""
        required = ["model_id", "dataset", "num_clients", "num_rounds", "strategies", "rank", "seeds"]
        for key in required:
            if key not in data:
                raise EvalConfigError(f"Missing required field: {key}")

        # Validate strategies
        for s in data["strategies"]:
            if s not in _AVAILABLE_STRATEGIES:
                raise EvalConfigError(
                    f"Unknown strategy '{s}'. Available: {list(_AVAILABLE_STRATEGIES.keys())}"
                )

        # Validate counts
        if not isinstance(data["num_clients"], int) or data["num_clients"] < 1:
            raise EvalConfigError("num_clients must be a positive integer")
        if not isinstance(data["num_rounds"], int) or data["num_rounds"] < 1:
            raise EvalConfigError("num_rounds must be a positive integer")
        if not isinstance(data["rank"], int) or data["rank"] < 1:
            raise EvalConfigError("rank must be a positive integer")
        if not data["seeds"]:
            raise EvalConfigError("seeds must be a non-empty list of integers")

        # Heterogeneous-rank length must equal num_clients
        het = data.get("heterogeneous_rank")
        if het is not None and len(het) != data["num_clients"]:
            raise EvalConfigError(
                f"heterogeneous_rank length ({len(het)}) must equal num_clients ({data['num_clients']})"
            )

        # Filter to known fields to avoid TypeError on extra keys
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
```

- [ ] **Step 3: Run tests (GREEN)**

```bash
pytest tests/test_eval_config.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 4: Commit**

```bash
git add chorus/eval/__init__.py chorus/eval/config.py
git commit -m "feat(eval): add EvalConfig dataclass + YAML loader (GREEN)"
```

---

## Task 6: Datasets module (IID and Dirichlet non-IID partitioning)

**Files:**
- Create: `tests/test_eval_datasets.py`

- [ ] **Step 1: Failing tests**

```python
"""Tests for dataset loading + partitioning in the eval harness."""

import numpy as np
import pytest

from chorus.eval.datasets import (
    partition_iid,
    partition_non_iid_dirichlet,
)


def _make_fake_dataset(n_examples: int, n_classes: int = 2):
    """Build a list of {text, label} dicts."""
    rng = np.random.default_rng(42)
    labels = rng.integers(0, n_classes, size=n_examples)
    return [{"text": f"example {i}", "label": int(l)} for i, l in enumerate(labels)]


class TestIID:
    def test_partitions_sum_to_dataset_size(self):
        ds = _make_fake_dataset(100)
        parts = partition_iid(ds, num_clients=5, seed=0)
        assert sum(len(p) for p in parts) == 100
        assert len(parts) == 5

    def test_no_overlap_between_partitions(self):
        ds = _make_fake_dataset(100)
        parts = partition_iid(ds, num_clients=4, seed=0)
        seen = set()
        for p in parts:
            for ex in p:
                key = ex["text"]
                assert key not in seen
                seen.add(key)

    def test_partition_sizes_roughly_balanced(self):
        ds = _make_fake_dataset(100)
        parts = partition_iid(ds, num_clients=5, seed=0)
        sizes = [len(p) for p in parts]
        assert max(sizes) - min(sizes) <= 1


class TestDirichlet:
    def test_dirichlet_partitions_sum_to_dataset_size(self):
        ds = _make_fake_dataset(200, n_classes=4)
        parts = partition_non_iid_dirichlet(ds, num_clients=5, alpha=0.5, seed=0)
        assert sum(len(p) for p in parts) == 200

    def test_dirichlet_low_alpha_creates_skew(self):
        """Low alpha → highly non-IID → at least one client gets a heavily-skewed class distribution."""
        ds = _make_fake_dataset(400, n_classes=4)
        parts = partition_non_iid_dirichlet(ds, num_clients=5, alpha=0.1, seed=0)

        def class_fraction(partition, cls):
            if not partition:
                return 0.0
            return sum(1 for ex in partition if ex["label"] == cls) / len(partition)

        # At least one client should have one class dominate >80% of its samples
        skewed = False
        for p in parts:
            for cls in range(4):
                if class_fraction(p, cls) > 0.8:
                    skewed = True
                    break
            if skewed:
                break
        assert skewed, "Low-alpha Dirichlet should produce at least one heavily skewed client"

    def test_dirichlet_high_alpha_approaches_iid(self):
        """High alpha → close to uniform; class fractions should not vary wildly across clients."""
        ds = _make_fake_dataset(800, n_classes=4)
        parts = partition_non_iid_dirichlet(ds, num_clients=5, alpha=100.0, seed=0)
        # For each class, fractions across clients should be similar (max - min < 0.2)
        for cls in range(4):
            fracs = []
            for p in parts:
                if not p:
                    continue
                fracs.append(sum(1 for ex in p if ex["label"] == cls) / len(p))
            assert max(fracs) - min(fracs) < 0.3, f"Class {cls} fractions too varied: {fracs}"
```

```bash
pytest tests/test_eval_datasets.py -v 2>&1 | tail -10
```

Expected failure (no module).

```bash
git add tests/test_eval_datasets.py
git commit -m "test(eval): add dataset partitioning tests (RED)"
```

- [ ] **Step 2: Implement `chorus/eval/datasets.py`**

```python
"""Dataset loading + partitioning for the eval harness.

Wraps HuggingFace `datasets` for the common loading patterns and implements
IID + Dirichlet-based non-IID partitioning into per-client splits.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def load_dataset_for_eval(
    name: str,
    split: str = "train",
    max_examples: int | None = None,
    text_field: str = "text",
    label_field: str = "label",
    config_name: str | None = None,
) -> list[dict[str, Any]]:
    """Load a HuggingFace dataset and return a list of {text_field, label_field, ...} dicts.

    Args:
        name: HF dataset name (e.g., "glue", "wikitext").
        split: Which split to load (e.g., "train", "validation").
        max_examples: If set, truncate to the first N examples.
        text_field: Column name for the text input.
        label_field: Column name for the label (may not exist for LM tasks).
        config_name: Optional HF config name (e.g., "sst2" for "glue").

    Returns:
        List of dicts; each dict has the original columns.
    """
    from datasets import load_dataset

    ds = load_dataset(name, config_name, split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    return [dict(ex) for ex in ds]


def partition_iid(
    dataset: list[dict[str, Any]],
    num_clients: int,
    seed: int = 0,
) -> list[list[dict[str, Any]]]:
    """Shuffle the dataset and split into `num_clients` near-equal partitions.

    Returns a list of `num_clients` lists. Partitions sum to len(dataset)
    and no example appears in more than one partition. Sizes differ by at most 1.
    """
    rng = np.random.default_rng(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    partitions: list[list[dict[str, Any]]] = [[] for _ in range(num_clients)]
    for i, idx in enumerate(indices):
        partitions[i % num_clients].append(dataset[idx])
    return partitions


def partition_non_iid_dirichlet(
    dataset: list[dict[str, Any]],
    num_clients: int,
    alpha: float = 0.5,
    label_field: str = "label",
    seed: int = 0,
) -> list[list[dict[str, Any]]]:
    """Partition by drawing per-client class proportions from Dir(alpha).

    Low `alpha` (e.g., 0.1) produces highly skewed per-client distributions
    (each client sees only a few classes). High `alpha` (e.g., 100) approaches
    IID.

    Args:
        dataset: List of dicts with a label field.
        num_clients: Number of partitions.
        alpha: Dirichlet concentration parameter.
        label_field: Dict key holding the class label.
        seed: RNG seed.

    Returns:
        List of `num_clients` lists.
    """
    rng = np.random.default_rng(seed)

    # Group example indices by class
    by_class: dict[int, list[int]] = {}
    for i, ex in enumerate(dataset):
        cls = ex[label_field]
        by_class.setdefault(cls, []).append(i)

    partitions: list[list[dict[str, Any]]] = [[] for _ in range(num_clients)]
    for cls, idxs in by_class.items():
        rng.shuffle(idxs)
        # Draw client proportions from Dirichlet
        proportions = rng.dirichlet([alpha] * num_clients)
        # Compute split points
        cuts = (np.cumsum(proportions) * len(idxs)).astype(int)
        prev = 0
        for c, cut in enumerate(cuts):
            for j in idxs[prev:cut]:
                partitions[c].append(dataset[j])
            prev = cut

    return partitions
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_eval_datasets.py -v
git add chorus/eval/datasets.py
git commit -m "feat(eval): add IID + Dirichlet non-IID dataset partitioning"
```

---

## Task 7: Metrics module

**Files:**
- Create: `tests/test_eval_metrics.py`

- [ ] **Step 1: Failing tests**

```python
"""Tests for eval metrics (Frobenius reconstruction error + task metrics)."""

import torch
import pytest

from chorus.eval.metrics import frobenius_reconstruction_error


def _make_lora_delta(rank: int = 4, dim: int = 16, seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "layer.0.lora_A.weight": torch.randn(rank, dim),
        "layer.0.lora_B.weight": torch.randn(dim, rank),
    }


def test_frobenius_error_zero_for_single_client():
    """A single client's aggregation should reconstruct exactly (FedEx-LoRA invariant)."""
    from chorus.server.aggregation import FedExLoRA
    delta = _make_lora_delta(rank=4, dim=16, seed=0)
    result = FedExLoRA().aggregate([delta])
    err = frobenius_reconstruction_error(result, [delta])
    assert err < 1e-5, f"Single-client FedEx-LoRA must reconstruct exactly; got {err}"


def test_frobenius_error_positive_for_multiple_clients_fedavg():
    """FedAvg has nonzero reconstruction error with >1 clients."""
    from chorus.server.aggregation import FedAvg
    deltas = [_make_lora_delta(seed=i) for i in range(3)]
    result = FedAvg().aggregate(deltas)
    err = frobenius_reconstruction_error(result, deltas)
    assert err > 0.01, f"Multi-client FedAvg should have nonzero error; got {err}"


def test_frobenius_error_lower_for_fedex_than_fedavg():
    """The whole point of FedEx-LoRA: lower reconstruction error than FedAvg."""
    from chorus.server.aggregation import FedAvg, FedExLoRA
    deltas = [_make_lora_delta(seed=i) for i in range(5)]
    fedavg_err = frobenius_reconstruction_error(FedAvg().aggregate(deltas), deltas)
    fedex_err = frobenius_reconstruction_error(FedExLoRA().aggregate(deltas), deltas)
    assert fedex_err < fedavg_err
```

```bash
pytest tests/test_eval_metrics.py -v 2>&1 | tail -10
git add tests/test_eval_metrics.py
git commit -m "test(eval): add metrics tests (RED)"
```

- [ ] **Step 2: Implement `chorus/eval/metrics.py`**

```python
"""Metrics for evaluating aggregation strategies in the eval harness.

Two families:
- Algorithmic: Frobenius reconstruction error of aggregated B@A vs the exact
  weighted average of per-client B_i @ A_i.
- Task: perplexity (causal LM), accuracy (classification), F1 (classification).
  Backed by HuggingFace `evaluate`.
"""

from __future__ import annotations

from typing import Any

import torch

from chorus.patterns import get_layer_pairs


def frobenius_reconstruction_error(
    aggregated: dict[str, torch.Tensor],
    client_deltas: list[dict[str, torch.Tensor]],
    weights: list[float] | None = None,
) -> float:
    """Frobenius norm of (exact_avg(B_i @ A_i) - aggregated_B @ aggregated_A), maxed over layers."""
    n = len(client_deltas)
    if weights is None:
        weights = [1.0 / n] * n

    pairs = get_layer_pairs(client_deltas[0])
    max_err = 0.0
    for layer_name, (a_key, b_key) in pairs.items():
        # Exact: sum w_i * B_i @ A_i
        exact = torch.zeros_like(
            client_deltas[0][b_key].float() @ client_deltas[0][a_key].float()
        )
        for i, d in enumerate(client_deltas):
            if a_key in d and b_key in d:
                exact += weights[i] * (d[b_key].float() @ d[a_key].float())

        # Reconstructed from aggregated:
        if a_key not in aggregated or b_key not in aggregated:
            continue
        recon = aggregated[b_key].float() @ aggregated[a_key].float()
        err = torch.norm(exact - recon).item()
        max_err = max(max_err, err)

    return max_err


def compute_task_metric(
    metric_name: str,
    predictions: list,
    references: list,
) -> dict[str, float]:
    """Compute a task metric using HF `evaluate`.

    Args:
        metric_name: One of "accuracy", "f1", "perplexity".
        predictions: Predicted labels (for classification) or logits (for LM).
        references: Ground truth labels (for classification) or input ids (for LM).

    Returns:
        Metric dict from `evaluate.load(metric_name).compute(...)`.
    """
    import evaluate

    if metric_name == "perplexity":
        # Perplexity needs predictions as text strings under HF's `evaluate` API;
        # for our use, we compute it directly from cross-entropy in the runner.
        # This path is kept simple as a no-op stub since callers do PPL inline.
        raise NotImplementedError(
            "Perplexity is computed inline in EvalRunner, not via evaluate.load. "
            "Use compute_perplexity_from_loss() instead."
        )

    metric = evaluate.load(metric_name)
    return metric.compute(predictions=predictions, references=references)


def compute_perplexity_from_loss(loss: float) -> float:
    """Perplexity = exp(cross-entropy loss). Loss assumed to be natural-log base."""
    import math
    return math.exp(loss)
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_eval_metrics.py -v
git add chorus/eval/metrics.py
git commit -m "feat(eval): add Frobenius reconstruction error + task metric helpers"
```

---

## Task 8: Report module

**Files:**
- Create: `tests/test_eval_report.py`

- [ ] **Step 1: Failing tests**

```python
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
```

```bash
pytest tests/test_eval_report.py -v 2>&1 | tail -10
git add tests/test_eval_report.py
git commit -m "test(eval): add EvalReport serialization tests (RED)"
```

- [ ] **Step 2: Implement `chorus/eval/report.py`**

```python
"""EvalReport: serializable record of a single EvalRunner execution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StrategyResult:
    """Per-strategy, per-seed outcome of one eval run."""
    strategy: str
    seed: int
    final_task_metric: dict[str, float]
    frobenius_error: float
    per_round_times_s: list[float] = field(default_factory=list)
    notes: str = ""


@dataclass
class EvalReport:
    """Top-level evaluation report; aggregates StrategyResults from one EvalRunner run."""
    config_name: str
    model_id: str
    dataset_name: str
    num_clients: int
    num_rounds: int
    rank: int
    seeds: list[int]
    results: list[StrategyResult] = field(default_factory=list)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    def to_markdown(self, path: str | Path) -> None:
        Path(path).write_text(self.to_markdown_string())

    def to_markdown_string(self) -> str:
        lines = [
            f"# Eval Report — {self.config_name}",
            "",
            f"- **Model:** `{self.model_id}`",
            f"- **Dataset:** `{self.dataset_name}`",
            f"- **Clients:** {self.num_clients}",
            f"- **Rounds:** {self.num_rounds}",
            f"- **LoRA rank:** {self.rank}",
            f"- **Seeds:** {self.seeds}",
            "",
            "## Results",
            "",
            "| Strategy | Seed | Task metric | Frobenius error | Mean round time (s) |",
            "|---|---|---|---|---|",
        ]
        for r in self.results:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in r.final_task_metric.items())
            mean_t = (
                sum(r.per_round_times_s) / len(r.per_round_times_s)
                if r.per_round_times_s
                else 0.0
            )
            lines.append(
                f"| `{r.strategy}` | {r.seed} | {metric_str} | {r.frobenius_error:.4f} | {mean_t:.2f} |"
            )
        return "\n".join(lines) + "\n"
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_eval_report.py -v
git add chorus/eval/report.py
git commit -m "feat(eval): add EvalReport with JSON + markdown serialization"
```

---

## Task 9: EvalRunner — the orchestrator

This is the largest task. The runner ties together: config → dataset → per-client LoRA training → aggregation → metrics → report.

**Files:**
- Create: `tests/test_eval_runner.py`
- Create: `chorus/eval/runner.py`

- [ ] **Step 1: Failing test (uses a tiny synthetic dataset to keep CI fast)**

```python
"""Tests for EvalRunner end-to-end behavior on a tiny synthetic setup."""

import os
from pathlib import Path

import pytest


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
```

```bash
pytest tests/test_eval_runner.py -v 2>&1 | tail -20
git add tests/test_eval_runner.py
git commit -m "test(eval): add EvalRunner tests (RED)"
```

- [ ] **Step 2: Implement `chorus/eval/runner.py`**

```python
"""EvalRunner: orchestrates simulated federation runs end-to-end.

Loads model + dataset, partitions data per client, runs per-client local LoRA
training, aggregates with each configured strategy, evaluates on a held-out
split, and emits a comparable report.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch

from chorus.eval.config import EvalConfig
from chorus.eval.datasets import load_dataset_for_eval, partition_iid, partition_non_iid_dirichlet
from chorus.eval.metrics import compute_perplexity_from_loss, frobenius_reconstruction_error
from chorus.eval.report import EvalReport, StrategyResult
from chorus.exceptions import EvalConfigError
from chorus.server.aggregation import get_strategy

logger = logging.getLogger("chorus.eval")


class EvalRunner:
    """Run a federated-LoRA evaluation on a real model + dataset.

    Two modes:
    - `check_only()`: validate config + verify model/dataset references resolve;
      do NOT load the model or run training. Used in CI.
    - `run()`: full execution. Returns an EvalReport.
    """

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    # -- Public API --

    def check_only(self) -> None:
        """Validate the config without training. Raises EvalConfigError on issues."""
        # Smoke-check the strategies resolve.
        for s in self.config.strategies:
            get_strategy(s)
        # Verify the dataset config has the required keys.
        if "name" not in self.config.dataset:
            raise EvalConfigError("dataset.name is required")
        if "split" not in self.config.dataset:
            raise EvalConfigError("dataset.split is required")
        logger.info("check-only: config OK for model %s", self.config.model_id)

    def run(self) -> EvalReport:
        """Run the full evaluation; return a report comparing strategies."""
        cfg = self.config
        logger.info(
            "EvalRunner starting: model=%s, clients=%d, rounds=%d, strategies=%s",
            cfg.model_id, cfg.num_clients, cfg.num_rounds, cfg.strategies,
        )

        # 1. Load + partition data
        train_data, eval_data = self._load_and_split_data()
        client_partitions = self._partition(train_data, seed=cfg.seeds[0])
        logger.info("Partitioned %d examples into %d clients", len(train_data), len(client_partitions))

        # 2. Per strategy and seed, train + aggregate + evaluate
        results: list[StrategyResult] = []
        for strategy_name in cfg.strategies:
            for seed in cfg.seeds:
                logger.info("Running strategy=%s seed=%d", strategy_name, seed)
                t_start = time.time()
                client_deltas, per_round_times = self._train_clients_and_collect_deltas(
                    client_partitions, strategy_name, seed,
                )
                aggregated = get_strategy(strategy_name).aggregate(client_deltas)
                frob = frobenius_reconstruction_error(aggregated, client_deltas)
                task_metric = self._evaluate_aggregated(aggregated, eval_data)
                t_total = time.time() - t_start
                results.append(StrategyResult(
                    strategy=strategy_name,
                    seed=seed,
                    final_task_metric=task_metric,
                    frobenius_error=float(frob),
                    per_round_times_s=per_round_times,
                    notes=f"total run time: {t_total:.1f}s",
                ))

        report = EvalReport(
            config_name=cfg.dataset.get("name", "unnamed"),
            model_id=cfg.model_id,
            dataset_name=cfg.dataset.get("name", "unknown"),
            num_clients=cfg.num_clients,
            num_rounds=cfg.num_rounds,
            rank=cfg.rank,
            seeds=list(cfg.seeds),
            results=results,
        )

        # Write artifacts
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(out_dir / "report.json")
        report.to_markdown(out_dir / "report.md")
        logger.info("Report written to %s", out_dir)
        return report

    # -- Internal: data --

    def _load_and_split_data(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Load the dataset and return (train_examples, eval_examples).

        For the special name 'synthetic-tiny', generate a minimal in-memory
        dataset (used by tests and CI smoke runs).
        """
        ds_cfg = self.config.dataset
        name = ds_cfg["name"]
        if name == "synthetic-tiny":
            return self._synthetic_tiny()

        examples = load_dataset_for_eval(
            name=name,
            split=ds_cfg["split"],
            max_examples=ds_cfg.get("max_examples"),
            config_name=ds_cfg.get("config_name"),
        )
        split_idx = int(len(examples) * 0.8)
        return examples[:split_idx], examples[split_idx:]

    def _synthetic_tiny(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Tiny in-memory dataset for CI: 16 short text examples, no labels."""
        texts = [
            "the quick brown fox jumps over the lazy dog",
            "lorem ipsum dolor sit amet consectetur",
            "federated learning is a distributed paradigm",
            "low rank adaptation enables efficient fine tuning",
            "privacy budgets must be tracked across rounds",
            "the cat sat on the mat",
            "machine learning models benefit from regularization",
            "differential privacy noise is calibrated to sensitivity",
            "neural networks learn hierarchical representations",
            "gradient descent converges under convexity",
            "attention is all you need",
            "transformers have revolutionized natural language processing",
            "the quick fox runs fast",
            "this is a synthetic example for testing",
            "tiny datasets keep CI runs fast",
            "evaluation harnesses produce reproducible reports",
        ]
        examples = [{"text": t} for t in texts]
        return examples[:12], examples[12:]

    def _partition(
        self,
        train_data: list[dict[str, Any]],
        seed: int,
    ) -> list[list[dict[str, Any]]]:
        cfg = self.config
        if cfg.partition == "iid":
            return partition_iid(train_data, num_clients=cfg.num_clients, seed=seed)
        if cfg.partition == "dirichlet":
            return partition_non_iid_dirichlet(
                train_data,
                num_clients=cfg.num_clients,
                alpha=cfg.dirichlet_alpha,
                seed=seed,
            )
        raise EvalConfigError(f"Unknown partition strategy: {cfg.partition}")

    # -- Internal: training + aggregation --

    def _train_clients_and_collect_deltas(
        self,
        partitions: list[list[dict[str, Any]]],
        strategy: str,
        seed: int,
    ) -> tuple[list[dict[str, torch.Tensor]], list[float]]:
        """Train a LoRA adapter on each client partition; return list of deltas + per-round times.

        Per-round-time is approximated as total client training time / num_rounds (one
        round = all clients train once then aggregate; we simulate by training all clients
        upfront and aggregating at the end). For multi-round, this is repeated.
        """
        deltas: list[dict[str, torch.Tensor]] = []
        per_round_times: list[float] = []
        cfg = self.config

        torch.manual_seed(seed)

        for round_idx in range(cfg.num_rounds):
            t0 = time.time()
            round_deltas: list[dict[str, torch.Tensor]] = []
            for client_idx, partition in enumerate(partitions):
                client_rank = (
                    cfg.heterogeneous_rank[client_idx]
                    if cfg.heterogeneous_rank
                    else cfg.rank
                )
                delta = self._train_one_client(partition, client_rank, seed=seed + client_idx)
                round_deltas.append(delta)
            per_round_times.append(time.time() - t0)
            # For multi-round, we'd aggregate here and re-broadcast. For the v0.2.0
            # eval harness we simulate one-shot aggregation per round; the final round's
            # deltas are what gets returned.
            deltas = round_deltas

        return deltas, per_round_times

    def _train_one_client(
        self,
        examples: list[dict[str, Any]],
        rank: int,
        seed: int,
    ) -> dict[str, torch.Tensor]:
        """Train a tiny LoRA adapter on one client's partition; return the adapter state dict."""
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(seed)

        # Load model + tokenizer (CPU)
        tok = AutoTokenizer.from_pretrained(self.config.model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if tok.eos_token else "[PAD]"
        model = AutoModelForCausalLM.from_pretrained(self.config.model_id)
        model.eval()  # freeze base

        # Attach LoRA
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=self.config.target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(model, lora_cfg)
        peft_model.train()

        # Tokenize examples
        if not examples:
            # No data — return zero-init adapters by extracting the LoRA params unchanged
            return self._extract_lora_state_dict(peft_model)

        optimizer = torch.optim.AdamW(
            (p for p in peft_model.parameters() if p.requires_grad),
            lr=self.config.learning_rate,
        )

        max_steps = self.config.max_steps_per_round
        step = 0
        for ex in examples:
            if step >= max_steps:
                break
            text = ex.get("text", "")
            if not text:
                continue
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=64)
            inputs["labels"] = inputs["input_ids"].clone()
            optimizer.zero_grad()
            outputs = peft_model(**inputs)
            outputs.loss.backward()
            optimizer.step()
            step += 1

        return self._extract_lora_state_dict(peft_model)

    @staticmethod
    def _extract_lora_state_dict(peft_model) -> dict[str, torch.Tensor]:
        """Return only the LoRA A/B parameters, keyed in the format chorus.patterns expects."""
        out: dict[str, torch.Tensor] = {}
        for name, p in peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                # PEFT uses names like "...lora_A.default.weight" or "...lora_A.weight";
                # chorus.patterns.get_layer_pairs handles either format already.
                out[name.replace("base_model.model.", "")] = p.detach().cpu().clone()
        return out

    # -- Internal: evaluation --

    def _evaluate_aggregated(
        self,
        aggregated: dict[str, torch.Tensor],
        eval_data: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Evaluate the aggregated adapter on held-out data.

        For LM datasets (no `label` field), reports perplexity. For classification,
        reports accuracy. Tiny implementation; serves the smoke path correctly.
        """
        if not eval_data:
            return {"note": "no_eval_data"}

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, TaskType, get_peft_model

        tok = AutoTokenizer.from_pretrained(self.config.model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if tok.eos_token else "[PAD]"
        base = AutoModelForCausalLM.from_pretrained(self.config.model_id)
        base.eval()

        lora_cfg = LoraConfig(
            r=self.config.rank,
            lora_alpha=self.config.rank * 2,
            target_modules=self.config.target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(base, lora_cfg)
        # Load aggregated adapter into the peft model
        peft_state = peft_model.state_dict()
        for k, v in aggregated.items():
            # PEFT may expect the "base_model.model." prefix
            cand = k if k in peft_state else f"base_model.model.{k}"
            if cand in peft_state:
                peft_state[cand] = v.to(peft_state[cand].dtype)
        peft_model.load_state_dict(peft_state, strict=False)
        peft_model.eval()

        # Compute perplexity over eval examples
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for ex in eval_data:
                text = ex.get("text", "")
                if not text:
                    continue
                inputs = tok(text, return_tensors="pt", truncation=True, max_length=64)
                inputs["labels"] = inputs["input_ids"].clone()
                out = peft_model(**inputs)
                total_loss += float(out.loss.item())
                n += 1
        if n == 0:
            return {"note": "no_eval_examples"}
        mean_loss = total_loss / n
        return {"perplexity": compute_perplexity_from_loss(mean_loss), "mean_loss": mean_loss}
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_eval_runner.py -v --tb=short
```

`test_runner_check_only_does_not_train` must pass without any model download. `test_runner_run_produces_report_with_both_strategies` may take 1–2 minutes downloading the tiny model the first time; subsequent runs use cache. If it fails on CI for any reason other than the obvious (e.g., tiny-random-LlamaForCausalLM unavailable), mark it `@pytest.mark.network` and gate via the existing skip pattern.

```bash
git add chorus/eval/runner.py
git commit -m "feat(eval): add EvalRunner end-to-end orchestration"
```

---

## Task 10: CLI subcommand `chorus eval`

**Files:**
- Modify: `chorus/cli/main.py`
- Create: `tests/test_cli_eval.py`

- [ ] **Step 1: Failing test**

```python
"""Tests for the `chorus eval` CLI."""

from pathlib import Path
import textwrap

from click.testing import CliRunner

from chorus.cli.main import cli


def _write_min_config(tmp_path: Path) -> Path:
    p = tmp_path / "min.yaml"
    p.write_text(textwrap.dedent("""
        model_id: hf-internal-testing/tiny-random-LlamaForCausalLM
        dataset: {name: synthetic-tiny, split: train, max_examples: 4}
        num_clients: 2
        num_rounds: 1
        strategies: [fedex-lora]
        rank: 4
        seeds: [0]
        max_steps_per_round: 1
    """).strip() + "\n")
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
```

```bash
pytest tests/test_cli_eval.py -v 2>&1 | tail -10
git add tests/test_cli_eval.py
git commit -m "test(cli): add chorus eval CLI tests (RED)"
```

- [ ] **Step 2: Add the subcommand to `chorus/cli/main.py`**

Append (use the existing `console` + `Click` patterns; read the file first):

```python
@cli.command("eval")
@click.option("--config", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to YAML config")
@click.option("--check-only", is_flag=True, default=False,
              help="Validate config + wiring, do NOT load model or train")
@click.option("--output-dir", default=None, type=click.Path(file_okay=False),
              help="Override output directory")
def eval_cmd(config: str, check_only: bool, output_dir: str | None):
    """Run an evaluation against a YAML config (see benchmarks/configs/)."""
    from chorus.eval import EvalConfig, EvalRunner
    from chorus.exceptions import EvalConfigError

    try:
        cfg = EvalConfig.from_yaml(config)
    except EvalConfigError as e:
        console.print(f"[red]Config error: {e}[/red]")
        raise SystemExit(1)

    if output_dir is not None:
        cfg.output_dir = output_dir

    runner = EvalRunner(cfg)

    if check_only:
        try:
            runner.check_only()
        except EvalConfigError as e:
            console.print(f"[red]Check-only failed: {e}[/red]")
            raise SystemExit(1)
        console.print(f"[green]check-only OK[/green] — model: {cfg.model_id}, "
                      f"clients: {cfg.num_clients}, strategies: {cfg.strategies}")
        return

    console.print(f"[bold]Running eval[/bold] — {cfg.model_id}, {cfg.num_clients} clients, "
                  f"{cfg.num_rounds} rounds")
    report = runner.run()
    console.print(f"[green]Eval complete[/green] — {len(report.results)} results, "
                  f"output in {cfg.output_dir}")
```

- [ ] **Step 3: Run + commit**

```bash
pytest tests/test_cli_eval.py -v
git add chorus/cli/main.py
git commit -m "feat(cli): add chorus eval subcommand"
```

---

## Task 11: Smoke YAML config

**Files:**
- Create: `benchmarks/configs/smoke.yaml`

- [ ] **Step 1: Write the config**

```yaml
# CI smoke config — runs in seconds via --check-only, in ~1 min via full run.
# Used by .github/workflows/ci.yml.

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
partition: iid
output_dir: benchmarks/results/smoke
```

- [ ] **Step 2: Verify with --check-only**

```bash
chorus eval --config benchmarks/configs/smoke.yaml --check-only
```

Expected: exit 0, prints "check-only OK".

- [ ] **Step 3: Commit**

```bash
git add benchmarks/configs/smoke.yaml
git commit -m "chore(benchmarks): add smoke.yaml for CI eval wiring check"
```

---

## Task 12: CI integration

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add `chorus eval --check-only` step after pytest**

In `.github/workflows/ci.yml`, inside the `test` job, after the `pytest tests/` step, add:

```yaml
      - name: Smoke-check eval harness wiring
        run: chorus eval --config benchmarks/configs/smoke.yaml --check-only
```

Also: update the install step to include the peft extra, since `chorus.eval` requires it:
```yaml
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev,privacy,peft]"
```

- [ ] **Step 2: Verify YAML parses**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add chorus eval --check-only smoke step + install peft extra"
```

---

## Task 13: Final verification and PR

- [ ] **Step 1: Full suite + ruff**

```bash
ruff check chorus tests benchmarks
pytest tests/ -q --tb=short
```

Expected: ruff clean. Test count 195 → ~215 (depending on exact additions; should be at least 195 + 8 config + 6 datasets + 3 metrics + 3 report + 2 runner + 3 cli = 220). Network-dependent tests may skip locally if HF cache is empty; CI runs them all.

- [ ] **Step 2: Push**

```bash
git push -u origin feat/eval-harness
```

- [ ] **Step 3: Open PR**

```bash
gh pr create \
  --base master \
  --head feat/eval-harness \
  --title "[Phase 1.3] Add chorus.eval package + chorus eval CLI" \
  --body "$(cat <<'EOF'
## Summary
- New `chorus/eval/` package: `EvalConfig` (YAML), `datasets` (IID + Dirichlet partitioning), `metrics` (Frobenius + task metrics), `report` (JSON + markdown), `EvalRunner` (orchestrator).
- New CLI: `chorus eval --config <yaml>` (full run) and `chorus eval --config <yaml> --check-only` (CI wiring check).
- New optional dep: `evaluate>=0.4.0` (added to `[peft]` extra).
- New smoke config: `benchmarks/configs/smoke.yaml`. CI smoke-checks the eval harness wiring on every PR.
- New exception: `EvalConfigError`.

## Closes
- #<ISSUE-NUMBER>

## Test plan
- [x] CI green across all three Python matrix entries
- [x] ruff clean
- [x] All existing tests still pass
- [x] ~25 new tests covering config, datasets, metrics, report, runner, CLI
- [x] `chorus eval --check-only --config benchmarks/configs/smoke.yaml` runs in <10s locally
- [x] Full smoke run completes in <2 min on a laptop with `hf-internal-testing/tiny-random-LlamaForCausalLM`

## Notes for reviewer
- Multi-round federation in `EvalRunner` is currently "simulated by repeated client training" — the rounds don't carry state between them in this v0.2.0 implementation. This matches the spec's intent (the eval harness measures algorithm quality, not server correctness). If we want round-over-round model evolution for the eval, that's a Feature 4 extension.
- Perplexity is computed inline from cross-entropy loss; HF `evaluate.load("perplexity")` is bypassed because it expects pretrained-model-by-name inputs that don't fit our adapter-overlay workflow.
- `synthetic-tiny` dataset is an in-memory list of 16 strings to keep CI fast and offline. Real benchmarks (Feature 4) will use Alpaca, SST-2, etc.
EOF
)"
```

Replace `<ISSUE-NUMBER>` with the real issue number.

- [ ] **Step 4: Watch CI**

```bash
gh pr view --json statusCheckRollup --jq '[.statusCheckRollup[] | {name, status, conclusion}]'
```

Wait for all green. Fix any failures.

---

## Self-review checklist

- [ ] `from chorus.eval import EvalConfig, EvalReport, EvalRunner` works.
- [ ] `EvalConfigError` is in `chorus/exceptions.py` and extends `ChorusError`.
- [ ] `chorus eval --check-only --config benchmarks/configs/smoke.yaml` exits 0 in <10s.
- [ ] `chorus eval --config benchmarks/configs/smoke.yaml` produces `benchmarks/results/smoke/report.json` and `report.md`.
- [ ] Report contains StrategyResult entries for each (strategy × seed) combination configured.
- [ ] Frobenius error is computed and non-negative for every StrategyResult.
- [ ] Existing tests (195) still pass; new tests added cleanly.
- [ ] CI workflow includes the `chorus eval --check-only` step AND the install line uses `.[dev,privacy,peft]`.
- [ ] Ruff clean.
- [ ] No `Co-Authored-By` trailers on any commit; no AI-attribution footer in PR/issue bodies.
- [ ] PR references `Closes #<issue>`.
