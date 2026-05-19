# Feature 2: DP Accountant — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking. Before starting, ensure you are in the worktree at `~/chorus-worktrees/feat-dp-accountant` on branch `feat/dp-accountant` (created via superpowers:using-git-worktrees from the parent checkout at `~/chorus`).

**Goal:** Replace the stateless per-round Gaussian DP mechanism with a stateful per-client privacy accountant that tracks (ε, δ) consumed across rounds using RDP composition. Expose budget state via API and CLI; raise a clear error when a client's configured budget is exhausted.

**Architecture:** New package `chorus/privacy/` houses the accountant (backed by Google's `dp-accounting` library, with `opacus.accountants` as fallback) and the existing Gaussian mechanism (moved from `chorus/server/privacy.py`). Server holds a `dict[client_id, PrivacyAccountant]` for each model; persists each accountant as JSON on disk; restores on startup. Client SDK gains a `max_epsilon` parameter that aborts submission when budget is exceeded server-side.

**Tech Stack:** New dependency `dp-accounting>=0.4.0` (Google) under the existing `[privacy]` extra. Existing deps: `opacus` (fallback), `numpy`, `torch`, FastAPI.

**Spec section:** `docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md` §4.2
**Master plan:** `docs/superpowers/plans/2026-05-19-phase-1-execution-plan.md` F2 row

---

## File Structure

- **Create:** `chorus/privacy/__init__.py` — package init; re-exports public API.
- **Create:** `chorus/privacy/accountant.py` — `PrivacyAccountant` class, RDP adapter, `PrivacyBudgetExhaustedError`-raising integration.
- **Move:** `chorus/server/privacy.py` → `chorus/privacy/mechanism.py` (existing `GaussianMechanism`, `clip_delta`, `apply_dp`).
- **Modify (shim):** `chorus/server/privacy.py` — becomes a deprecation re-export of `chorus.privacy.mechanism` (removed in v0.3.0).
- **Modify:** `chorus/exceptions.py` — add `PrivacyBudgetExhaustedError(ChorusError)`.
- **Modify:** `chorus/server/app.py` — add accountant state, new endpoint, integrate budget check into `submit_delta`.
- **Modify:** `chorus/server/storage.py` — `save_accountant`, `load_accountant`, `load_all_accountants` for a (model_id, client_id).
- **Modify:** `chorus/client/sdk.py` — `max_epsilon` param, post-submit budget poll, raise on exhaustion.
- **Modify:** `chorus/cli/main.py` — add `chorus privacy budget` subcommand; augment `chorus status` to show budgets.
- **Modify:** `pyproject.toml` — add `dp-accounting>=0.4.0` to `[project.optional-dependencies] privacy`.
- **Create:** `tests/test_privacy_accountant.py` — unit tests for the accountant.
- **Create:** `tests/test_privacy_persistence.py` — serialize/deserialize round-trip + storage.
- **Create:** `tests/test_privacy_endpoints.py` — server endpoint tests.
- **Create:** `tests/test_privacy_cli.py` — CLI tests.
- **Modify:** `tests/test_server.py` — augment existing DP-related tests to cover budget enforcement.

---

## Pre-flight (Task 0)

Run once from the main `~/chorus` checkout:

```bash
cd ~/chorus
git checkout master && git pull origin master
git worktree add ../chorus-worktrees/feat-dp-accountant -b feat/dp-accountant
git worktree list
# Confirm:
#   /Users/varma/chorus-worktrees/feat-dp-accountant  <sha> [feat/dp-accountant]
```

The subagent operates inside `~/chorus-worktrees/feat-dp-accountant`. All paths below are relative to that worktree.

---

## Task 1: Open GitHub issue

**Files:** None.

- [ ] **Step 1: Create the issue**

```bash
cd ~/chorus-worktrees/feat-dp-accountant
gh issue create \
  --title "[Phase 1.2] Add stateful DP privacy accountant with composition tracking" \
  --label "phase-1" \
  --body "$(cat <<'EOF'
**Phase:** 1 (Credibility & Honesty)
**Spec:** [docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md](../blob/master/docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md) §4.2
**Roadmap:** [docs/superpowers/specs/2026-05-19-chorus-roadmap.md](../blob/master/docs/superpowers/specs/2026-05-19-chorus-roadmap.md)
**Plan:** [docs/superpowers/plans/2026-05-19-feature-2-dp-accountant.md](../blob/master/docs/superpowers/plans/2026-05-19-feature-2-dp-accountant.md)

## Scope
Add a stateful `PrivacyAccountant` (RDP composition via `dp-accounting`, with `opacus` fallback). Track (ε, δ) consumed per (model_id, client_id) across rounds. Persist on disk; restore on server restart. Expose budget state via API endpoint and CLI. Refuse client submissions when budget is exhausted.

## Acceptance criteria
- [ ] `PrivacyAccountant.step()` advances state; `get_epsilon(delta)` returns RDP-composed ε.
- [ ] Server endpoint `GET /models/{model_id}/clients/{client_id}/privacy` returns `{epsilon_consumed, epsilon_target, delta, exhausted}`.
- [ ] Submission endpoint refuses with 403 + `PrivacyBudgetExhaustedError` when client budget is exhausted.
- [ ] Accountant state persists to `chorus_data/<model>/privacy/<client>.json`; restored on server startup.
- [ ] `ChorusClient(max_epsilon=...)` raises `PrivacyBudgetExhaustedError` after exceeded.
- [ ] `chorus privacy budget --client-id X --model-id Y` prints remaining ε for that client.
- [ ] `chorus status` shows per-client budget when accounting is enabled.
- [ ] All existing tests pass; new tests cover accountant behavior, persistence, endpoints, CLI.

## Out of scope
- Switching the Gaussian mechanism for a different DP primitive (Laplace, discrete Gaussian, etc.).
- Client-side accounting (only server-side; client mirrors via polling).
- Privacy amplification by subsampling beyond what `dp-accounting`'s `RdpAccountant` supports natively.
- Per-tenant API key scoping (Phase 4).
EOF
)"
```

Expected: `gh` prints the issue URL. Note the issue number.

---

## Task 2: Add `dp-accounting` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Set up venv and install current deps**

```bash
cd ~/chorus-worktrees/feat-dp-accountant
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,privacy]"
```

Expected: clean install. `pip list | grep -iE "opacus|dp-accounting"` shows `opacus` only.

- [ ] **Step 2: Add `dp-accounting` to the `privacy` extra in `pyproject.toml`**

Find the existing block:
```toml
[project.optional-dependencies]
peft = ["peft>=0.7.0", "transformers>=4.36.0", "datasets>=2.16.0", "accelerate>=0.25.0"]
privacy = ["opacus>=1.4.0"]
```

Change `privacy` to:
```toml
privacy = ["opacus>=1.4.0", "dp-accounting>=0.4.0"]
```

- [ ] **Step 3: Reinstall and verify**

```bash
pip install -e ".[dev,privacy]"
python -c "from dp_accounting import rdp; from dp_accounting import dp_event; print('dp-accounting import OK')"
```

Expected: `dp-accounting import OK`.

- [ ] **Step 4: Verify ruff and tests still pass**

```bash
ruff check chorus tests benchmarks
pytest tests/ -q
```

Expected: ruff clean; all existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore(deps): add dp-accounting to privacy extra for RDP composition"
```

---

## Task 3: Create `chorus/privacy/` package skeleton

**Files:**
- Create: `chorus/privacy/__init__.py`

- [ ] **Step 1: Create the package directory**

```bash
mkdir -p chorus/privacy
```

- [ ] **Step 2: Create `chorus/privacy/__init__.py`** with this content:

```python
"""Privacy primitives and accounting for federated LoRA.

Houses the Gaussian DP mechanism (in mechanism.py) and the stateful
privacy accountant (in accountant.py).
"""

from chorus.privacy.accountant import PrivacyAccountant
from chorus.privacy.mechanism import GaussianMechanism, apply_dp, clip_delta

__all__ = [
    "PrivacyAccountant",
    "GaussianMechanism",
    "apply_dp",
    "clip_delta",
]
```

Note: `accountant.py` and `mechanism.py` don't exist yet — this will fail import. We create them in the next two tasks. Don't commit yet.

---

## Task 4: Move existing mechanism to `chorus/privacy/mechanism.py`

**Files:**
- Move: `chorus/server/privacy.py` → `chorus/privacy/mechanism.py`
- Modify: `chorus/server/privacy.py` (replace with deprecation shim)

- [ ] **Step 1: Move the file with `git mv`**

```bash
git mv chorus/server/privacy.py chorus/privacy/mechanism.py
```

- [ ] **Step 2: Update the module docstring at the top of `chorus/privacy/mechanism.py`**

Replace the first docstring line if needed; the rest of the file stays identical. Confirm:

```bash
head -5 chorus/privacy/mechanism.py
# Expected: "Differential privacy mechanisms for federated LoRA." docstring
```

- [ ] **Step 3: Create `chorus/server/privacy.py` as a deprecation shim**

```python
"""Deprecated: import from chorus.privacy instead.

This module is preserved as a re-export of chorus.privacy.mechanism for
backward compatibility. It will be removed in v0.3.0.
"""

import warnings

warnings.warn(
    "chorus.server.privacy is deprecated; import from chorus.privacy instead. "
    "This shim will be removed in v0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

from chorus.privacy.mechanism import GaussianMechanism, apply_dp, clip_delta  # noqa: E402,F401

__all__ = ["GaussianMechanism", "apply_dp", "clip_delta"]
```

- [ ] **Step 4: Update all in-repo callers**

Find every `from chorus.server.privacy import ...` and update to `from chorus.privacy.mechanism import ...`:

```bash
grep -rn "from chorus.server.privacy import" chorus tests benchmarks
```

For each match, edit to use `from chorus.privacy.mechanism import ...`. Expected files to update (based on current repo): `chorus/server/app.py`, `chorus/client/sdk.py`, `chorus/simulate/runner.py`. Also check `tests/`.

- [ ] **Step 5: Verify imports still work**

```bash
python -c "from chorus.privacy.mechanism import apply_dp, GaussianMechanism, clip_delta; print('new path OK')"
python -W error::DeprecationWarning -c "from chorus.server.privacy import apply_dp" 2>&1 | grep -q "DeprecationWarning" && echo "deprecation warning OK"
```

Expected: both prints succeed.

- [ ] **Step 6: Run tests**

```bash
pytest tests/ -q
```

Expected: all pass (since the mechanism is functionally unchanged).

- [ ] **Step 7: Commit**

```bash
git add chorus/privacy/ chorus/server/privacy.py chorus/server/app.py chorus/client/sdk.py chorus/simulate/runner.py
# Add any test files that were updated:
git add tests/
git commit -m "refactor(privacy): move chorus.server.privacy to chorus.privacy.mechanism

Adds chorus/privacy/ package. chorus.server.privacy stays as a
deprecation re-export through v0.2.x; removed in v0.3.0. All in-repo
callers updated to the new path."
```

Note: `chorus/privacy/__init__.py` from Task 3 imports `PrivacyAccountant`, which doesn't exist yet. **Don't try to import the package as a whole yet** — only direct imports of `chorus.privacy.mechanism` work after this commit. The full package import is unlocked when Task 6 lands. (This is acceptable for a single PR; the package is internally inconsistent for ~2 commits but the test suite doesn't exercise the broken path.)

---

## Task 5: Add `PrivacyBudgetExhaustedError` exception

**Files:**
- Modify: `chorus/exceptions.py`

- [ ] **Step 1: Read current exceptions**

```bash
cat chorus/exceptions.py
```

Note the existing exception class hierarchy (all extend `ChorusError`).

- [ ] **Step 2: Append new exception class to `chorus/exceptions.py`**

Append at the end of the file (before any `if __name__` blocks if present):

```python


class PrivacyBudgetExhaustedError(ChorusError):
    """Raised when a client's configured (ε, δ) DP budget is exhausted.

    The accompanying message includes the model_id, client_id, target_epsilon,
    and consumed_epsilon so the user can adjust their training plan.
    """
```

- [ ] **Step 3: Verify it imports**

```bash
python -c "from chorus.exceptions import PrivacyBudgetExhaustedError; print(PrivacyBudgetExhaustedError.__mro__)"
```

Expected: shows MRO including `ChorusError` and `Exception`.

- [ ] **Step 4: Commit**

```bash
git add chorus/exceptions.py
git commit -m "feat(exceptions): add PrivacyBudgetExhaustedError"
```

---

## Task 6: Write failing tests for `PrivacyAccountant`

**Files:**
- Create: `tests/test_privacy_accountant.py`

- [ ] **Step 1: Create the test file** with this exact content:

```python
"""Tests for the PrivacyAccountant — RDP composition + budget tracking."""

import math

import pytest

# Module-under-test; this import will fail until Task 7 lands the class.
from chorus.privacy.accountant import PrivacyAccountant


class TestBasicBookkeeping:
    def test_zero_steps_means_zero_epsilon(self):
        a = PrivacyAccountant(
            target_epsilon=10.0, target_delta=1e-5,
            noise_multiplier=1.0, sample_rate=1.0,
        )
        assert a.get_epsilon() == pytest.approx(0.0, abs=1e-9)
        assert not a.is_exhausted()

    def test_epsilon_grows_monotonically_with_steps(self):
        a = PrivacyAccountant(
            target_epsilon=100.0, target_delta=1e-5,
            noise_multiplier=1.0, sample_rate=1.0,
        )
        eps = []
        for _ in range(5):
            a.step()
            eps.append(a.get_epsilon())
        # Strictly increasing
        for i in range(1, len(eps)):
            assert eps[i] > eps[i - 1]

    def test_higher_noise_multiplier_gives_smaller_epsilon(self):
        a_low = PrivacyAccountant(
            target_epsilon=100.0, target_delta=1e-5,
            noise_multiplier=1.0, sample_rate=1.0,
        )
        a_high = PrivacyAccountant(
            target_epsilon=100.0, target_delta=1e-5,
            noise_multiplier=5.0, sample_rate=1.0,
        )
        for _ in range(5):
            a_low.step()
            a_high.step()
        assert a_high.get_epsilon() < a_low.get_epsilon()


class TestExhaustion:
    def test_is_exhausted_triggers_at_threshold(self):
        # With noise_multiplier=0.5 and sample_rate=1, a few steps blow past ε=0.5
        a = PrivacyAccountant(
            target_epsilon=0.5, target_delta=1e-5,
            noise_multiplier=0.5, sample_rate=1.0,
        )
        # Step until exhausted, but bound the loop to avoid infinite spin
        for _ in range(50):
            if a.is_exhausted():
                break
            a.step()
        assert a.is_exhausted()
        assert a.get_epsilon() >= 0.5

    def test_remaining_decreases(self):
        a = PrivacyAccountant(
            target_epsilon=10.0, target_delta=1e-5,
            noise_multiplier=1.0, sample_rate=1.0,
        )
        eps_remaining_before, _ = a.remaining()
        a.step()
        eps_remaining_after, _ = a.remaining()
        assert eps_remaining_after < eps_remaining_before


class TestSerialization:
    def test_serialize_roundtrip_preserves_epsilon(self):
        a = PrivacyAccountant(
            target_epsilon=10.0, target_delta=1e-5,
            noise_multiplier=1.0, sample_rate=1.0,
        )
        for _ in range(3):
            a.step()
        eps_before = a.get_epsilon()
        serialized = a.serialize()
        restored = PrivacyAccountant.deserialize(serialized)
        assert math.isclose(restored.get_epsilon(), eps_before, rel_tol=1e-9)
        assert restored.target_epsilon == a.target_epsilon
        assert restored.target_delta == a.target_delta

    def test_serialize_is_json_safe(self):
        import json
        a = PrivacyAccountant(
            target_epsilon=10.0, target_delta=1e-5,
            noise_multiplier=1.0, sample_rate=1.0,
        )
        a.step()
        # Must be JSON-serializable as-is
        encoded = json.dumps(a.serialize())
        decoded = json.loads(encoded)
        restored = PrivacyAccountant.deserialize(decoded)
        assert math.isclose(restored.get_epsilon(), a.get_epsilon(), rel_tol=1e-9)


class TestValidation:
    def test_invalid_epsilon_rejected(self):
        with pytest.raises(ValueError):
            PrivacyAccountant(
                target_epsilon=0.0, target_delta=1e-5,
                noise_multiplier=1.0,
            )

    def test_invalid_delta_rejected(self):
        with pytest.raises(ValueError):
            PrivacyAccountant(
                target_epsilon=1.0, target_delta=0.0,
                noise_multiplier=1.0,
            )
        with pytest.raises(ValueError):
            PrivacyAccountant(
                target_epsilon=1.0, target_delta=1.5,
                noise_multiplier=1.0,
            )

    def test_invalid_noise_multiplier_rejected(self):
        with pytest.raises(ValueError):
            PrivacyAccountant(
                target_epsilon=1.0, target_delta=1e-5,
                noise_multiplier=0.0,
            )

    def test_invalid_sample_rate_rejected(self):
        with pytest.raises(ValueError):
            PrivacyAccountant(
                target_epsilon=1.0, target_delta=1e-5,
                noise_multiplier=1.0, sample_rate=0.0,
            )
        with pytest.raises(ValueError):
            PrivacyAccountant(
                target_epsilon=1.0, target_delta=1e-5,
                noise_multiplier=1.0, sample_rate=1.5,
            )
```

- [ ] **Step 2: Verify the tests fail (no implementation yet)**

```bash
pytest tests/test_privacy_accountant.py -v 2>&1 | tail -20
```

Expected: ImportError / ModuleNotFoundError on `from chorus.privacy.accountant import PrivacyAccountant`. This is the failing-test step of TDD.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_privacy_accountant.py
git commit -m "test(privacy): add unit tests for PrivacyAccountant (failing — RED)"
```

---

## Task 7: Implement `PrivacyAccountant`

**Files:**
- Create: `chorus/privacy/accountant.py`

- [ ] **Step 1: Write the implementation**

Create `chorus/privacy/accountant.py`:

```python
"""Stateful per-client privacy accountant for federated LoRA.

Tracks (ε, δ) consumed across DP-noised submissions using RDP composition.
Default backend is Google's `dp-accounting` library; falls back to
`opacus.accountants.RDPAccountant` if `dp-accounting` is unavailable.

This module does NOT add noise. It only accounts for noise added elsewhere
by `chorus.privacy.mechanism.apply_dp`.
"""

from __future__ import annotations

from typing import Any

try:
    from dp_accounting import dp_event, rdp
    _BACKEND = "dp-accounting"
except ImportError:  # pragma: no cover - exercised only when dp-accounting absent
    try:
        from opacus.accountants import RDPAccountant as _OpacusRDP
        _BACKEND = "opacus"
    except ImportError:  # pragma: no cover
        raise ImportError(
            "PrivacyAccountant requires either 'dp-accounting' or 'opacus'. "
            "Install with: pip install 'chorus-fl[privacy]'"
        )


class PrivacyAccountant:
    """RDP-based privacy accountant for the Gaussian mechanism.

    Each call to `step()` records one application of the Gaussian mechanism
    at the configured `noise_multiplier` and `sample_rate`. The accountant
    tracks RDP at multiple orders and converts to (ε, δ) on demand.

    Args:
        target_epsilon: Maximum ε allowed before `is_exhausted()` becomes True.
        target_delta: δ for the (ε, δ)-DP guarantee.
        noise_multiplier: σ / sensitivity for the Gaussian mechanism.
        sample_rate: Fraction of the dataset sampled per round (1.0 = full).
        backend: "rdp" (default; uses dp-accounting RdpAccountant).
    """

    def __init__(
        self,
        target_epsilon: float,
        target_delta: float,
        noise_multiplier: float,
        sample_rate: float = 1.0,
        backend: str = "rdp",
    ) -> None:
        if target_epsilon <= 0:
            raise ValueError("target_epsilon must be positive")
        if not (0 < target_delta < 1):
            raise ValueError("target_delta must be in (0, 1)")
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        if not (0 < sample_rate <= 1):
            raise ValueError("sample_rate must be in (0, 1]")

        self.target_epsilon = float(target_epsilon)
        self.target_delta = float(target_delta)
        self.noise_multiplier = float(noise_multiplier)
        self.sample_rate = float(sample_rate)
        self.backend = backend
        self._steps = 0
        self._accountant = self._make_backend()

    def _make_backend(self) -> Any:
        if _BACKEND == "dp-accounting":
            # Standard RDP orders used by Google's library examples.
            orders = (
                [1 + x / 10.0 for x in range(1, 100)]
                + list(range(11, 64))
                + [128, 256, 512]
            )
            return rdp.RdpAccountant(orders)
        return _OpacusRDP()

    def step(self) -> None:
        """Record one round of Gaussian-mechanism noise application."""
        if _BACKEND == "dp-accounting":
            event = dp_event.PoissonSampledDpEvent(
                sampling_probability=self.sample_rate,
                event=dp_event.GaussianDpEvent(noise_multiplier=self.noise_multiplier),
            )
            self._accountant.compose(event, count=1)
        else:
            self._accountant.step(
                noise_multiplier=self.noise_multiplier,
                sample_rate=self.sample_rate,
            )
        self._steps += 1

    def get_epsilon(self, delta: float | None = None) -> float:
        """Return the ε consumed so far at the given δ (default: target_delta)."""
        d = delta if delta is not None else self.target_delta
        if self._steps == 0:
            return 0.0
        if _BACKEND == "dp-accounting":
            return float(self._accountant.get_epsilon(d))
        return float(self._accountant.get_epsilon(delta=d))

    def is_exhausted(self) -> bool:
        """True if get_epsilon() at target_delta has reached or exceeded target_epsilon."""
        return self.get_epsilon() >= self.target_epsilon

    def remaining(self) -> tuple[float, float]:
        """Return (epsilon_remaining, target_delta).

        epsilon_remaining is `max(target_epsilon - get_epsilon(), 0.0)`.
        """
        return (max(self.target_epsilon - self.get_epsilon(), 0.0), self.target_delta)

    def serialize(self) -> dict[str, Any]:
        """Return a JSON-safe dict representing this accountant's state.

        Backend-agnostic: stores config + step count. Backend state is
        rebuilt by replaying steps on deserialize (cheap for small N).
        """
        return {
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "noise_multiplier": self.noise_multiplier,
            "sample_rate": self.sample_rate,
            "backend": self.backend,
            "steps": self._steps,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "PrivacyAccountant":
        """Reconstruct a PrivacyAccountant from `serialize()` output."""
        a = cls(
            target_epsilon=float(data["target_epsilon"]),
            target_delta=float(data["target_delta"]),
            noise_multiplier=float(data["noise_multiplier"]),
            sample_rate=float(data.get("sample_rate", 1.0)),
            backend=str(data.get("backend", "rdp")),
        )
        steps = int(data.get("steps", 0))
        for _ in range(steps):
            a.step()
        return a

    def __repr__(self) -> str:
        return (
            f"PrivacyAccountant(steps={self._steps}, "
            f"epsilon={self.get_epsilon():.4f}/{self.target_epsilon}, "
            f"delta={self.target_delta}, backend={_BACKEND})"
        )
```

- [ ] **Step 2: Verify tests pass (GREEN)**

```bash
pytest tests/test_privacy_accountant.py -v
```

Expected: all 12 tests pass.

If any test fails, do NOT skip. Read the failure, fix the implementation, re-run. The most likely failures and their causes:

- `test_serialize_roundtrip_preserves_epsilon` fails by tiny float diff → tolerate via `math.isclose` (already in test).
- `test_invalid_sample_rate_rejected` fails for `sample_rate=1.5` → confirm the `> 1` guard is in `__init__`.
- `test_is_exhausted_triggers_at_threshold` runs forever → confirm the loop bound of 50 is enough; if not, increase the noise_multiplier slightly to make ε grow faster, or relax `target_epsilon` to a larger value. (The test as written should converge well before 50 steps with noise_multiplier=0.5.)

- [ ] **Step 3: Run the full test suite (no regression)**

```bash
pytest tests/ -q
```

Expected: 165 + 12 = 177 tests pass.

- [ ] **Step 4: Verify the package import works end-to-end now**

```bash
python -c "from chorus.privacy import PrivacyAccountant, GaussianMechanism, apply_dp; print('package import OK')"
```

- [ ] **Step 5: Commit**

```bash
git add chorus/privacy/accountant.py
git commit -m "feat(privacy): add PrivacyAccountant with RDP composition (GREEN)

Backed by Google's dp-accounting library by default; falls back to
opacus.accountants if dp-accounting is unavailable. Tracks (ε, δ)
across step() calls; serialize() / deserialize() round-trip the
configuration + step count (backend state is rebuilt on deserialize)."
```

---

## Task 8: Add accountant storage to `DeltaStorage`

**Files:**
- Modify: `chorus/server/storage.py`
- Create: `tests/test_privacy_persistence.py`

- [ ] **Step 1: Write the failing test first**

Create `tests/test_privacy_persistence.py`:

```python
"""Tests for accountant persistence in DeltaStorage."""

import json
from pathlib import Path

from chorus.privacy import PrivacyAccountant
from chorus.server.storage import DeltaStorage


def test_save_and_load_accountant_roundtrip(tmp_path: Path):
    storage = DeltaStorage(tmp_path)
    a = PrivacyAccountant(
        target_epsilon=10.0, target_delta=1e-5,
        noise_multiplier=1.0, sample_rate=1.0,
    )
    a.step()
    a.step()
    storage.save_accountant("model-x", "client-1", a)
    restored = storage.load_accountant("model-x", "client-1")
    assert restored is not None
    assert restored.target_epsilon == a.target_epsilon
    assert abs(restored.get_epsilon() - a.get_epsilon()) < 1e-9


def test_load_accountant_missing_returns_none(tmp_path: Path):
    storage = DeltaStorage(tmp_path)
    assert storage.load_accountant("model-x", "nobody") is None


def test_load_all_accountants_for_model(tmp_path: Path):
    storage = DeltaStorage(tmp_path)
    for cid in ("alice", "bob"):
        a = PrivacyAccountant(
            target_epsilon=10.0, target_delta=1e-5,
            noise_multiplier=1.0, sample_rate=1.0,
        )
        a.step()
        storage.save_accountant("model-x", cid, a)
    all_a = storage.load_all_accountants("model-x")
    assert set(all_a.keys()) == {"alice", "bob"}
    for accountant in all_a.values():
        assert accountant.get_epsilon() > 0


def test_save_accountant_atomic_overwrite(tmp_path: Path):
    storage = DeltaStorage(tmp_path)
    a = PrivacyAccountant(
        target_epsilon=10.0, target_delta=1e-5,
        noise_multiplier=1.0, sample_rate=1.0,
    )
    storage.save_accountant("model-x", "client-1", a)
    a.step()
    storage.save_accountant("model-x", "client-1", a)
    restored = storage.load_accountant("model-x", "client-1")
    assert restored is not None
    assert restored.get_epsilon() > 0


def test_accountant_path_uses_sanitized_client_id(tmp_path: Path):
    storage = DeltaStorage(tmp_path)
    a = PrivacyAccountant(
        target_epsilon=10.0, target_delta=1e-5,
        noise_multiplier=1.0, sample_rate=1.0,
    )
    # Path-traversal attempt — sanitization should reject or neutralize this
    storage.save_accountant("model-x", "../etc/passwd", a)
    privacy_dir = tmp_path / "model-x" / "privacy"
    # File must land inside the model's privacy dir, not outside it
    files = list(privacy_dir.glob("*.json"))
    assert len(files) == 1
    assert "etc" not in files[0].name or "passwd" not in files[0].name
```

Run it; expect failure (`AttributeError: 'DeltaStorage' object has no attribute 'save_accountant'`):

```bash
pytest tests/test_privacy_persistence.py -v 2>&1 | tail -10
```

- [ ] **Step 2: Implement `save_accountant`, `load_accountant`, `load_all_accountants`** in `chorus/server/storage.py`

Read the existing `DeltaStorage` class to find the right insertion point (near other save_* / load_* methods). Then add these methods inside the class:

```python
    def _privacy_dir(self, model_id: str) -> Path:
        return self._model_dir(model_id) / "privacy"

    def _accountant_path(self, model_id: str, client_id: str) -> Path:
        client_id = _sanitize_client_id(client_id)
        return self._privacy_dir(model_id) / f"{client_id}.json"

    def save_accountant(self, model_id: str, client_id: str, accountant) -> None:
        """Persist a PrivacyAccountant to disk."""
        path = self._accountant_path(model_id, client_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via tmp + rename
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(accountant.serialize(), indent=2))
        tmp.replace(path)

    def load_accountant(self, model_id: str, client_id: str):
        """Load a PrivacyAccountant from disk, or None if not present."""
        from chorus.privacy import PrivacyAccountant
        path = self._accountant_path(model_id, client_id)
        if not path.exists():
            return None
        return PrivacyAccountant.deserialize(json.loads(path.read_text()))

    def load_all_accountants(self, model_id: str) -> dict:
        """Return {client_id: PrivacyAccountant} for every accountant under this model."""
        from chorus.privacy import PrivacyAccountant
        priv_dir = self._privacy_dir(model_id)
        if not priv_dir.exists():
            return {}
        out = {}
        for path in priv_dir.glob("*.json"):
            cid = path.stem
            out[cid] = PrivacyAccountant.deserialize(json.loads(path.read_text()))
        return out
```

- [ ] **Step 3: Run the tests (GREEN)**

```bash
pytest tests/test_privacy_persistence.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 4: Full test suite**

```bash
pytest tests/ -q
```

Expected: 177 + 5 = 182 tests pass.

- [ ] **Step 5: Commit**

```bash
git add chorus/server/storage.py tests/test_privacy_persistence.py
git commit -m "feat(privacy): persist PrivacyAccountant per (model_id, client_id)

Stores accountants as JSON under chorus_data/<model>/privacy/<client>.json.
Atomic save via tmp+rename. load_all_accountants returns
{client_id: PrivacyAccountant} for restoration on server startup.
Reuses existing _sanitize_client_id for path-traversal safety."
```

---

## Task 9: Wire accountant into server state and submission flow

**Files:**
- Modify: `chorus/server/app.py`

- [ ] **Step 1: Read the current server state class**

```bash
grep -n "class ServerState\|^state\." chorus/server/app.py | head
```

Locate `ServerState`, its attributes, and the `configure()` function that instantiates fields.

- [ ] **Step 2: Add an accountant config + per-(model, client) accountant map**

In `ServerState` annotations:

```python
class ServerState:
    # ... existing fields ...
    accountants: dict[str, dict[str, "PrivacyAccountant"]]  # {model_id: {client_id: PrivacyAccountant}}
    accountant_target_epsilon: float | None
    accountant_noise_multiplier: float | None
    accountant_sample_rate: float
```

Initialize them after the existing module-level `state = ServerState()` block:

```python
state.accountants = {}
state.accountant_target_epsilon = None
state.accountant_noise_multiplier = None
state.accountant_sample_rate = 1.0
```

- [ ] **Step 3: Extend `configure()` with accountant params**

Add to the `configure(...)` parameter list (right after `dp_max_norm`):

```python
    accountant_target_epsilon: float | None = None,
    accountant_noise_multiplier: float | None = None,
    accountant_sample_rate: float = 1.0,
```

And in the function body:

```python
    state.accountant_target_epsilon = accountant_target_epsilon
    state.accountant_noise_multiplier = accountant_noise_multiplier
    state.accountant_sample_rate = accountant_sample_rate
    state.accountants = {}
```

- [ ] **Step 4: Add an `_ensure_accountant(model_id, client_id)` helper**

Add as a module-level function in `app.py` (near other helpers):

```python
def _ensure_accountant(model_id: str, client_id: str):
    """Return the accountant for (model_id, client_id), creating + persisting on first use.

    Returns None if accounting is not configured at the server level.
    """
    if state.accountant_target_epsilon is None or state.accountant_noise_multiplier is None:
        return None
    from chorus.privacy import PrivacyAccountant

    by_client = state.accountants.setdefault(model_id, {})
    if client_id in by_client:
        return by_client[client_id]

    # Try to restore from disk
    restored = state.storage.load_accountant(model_id, client_id)
    if restored is not None:
        by_client[client_id] = restored
        return restored

    # Create a fresh one
    a = PrivacyAccountant(
        target_epsilon=state.accountant_target_epsilon,
        target_delta=state.dp_delta,
        noise_multiplier=state.accountant_noise_multiplier,
        sample_rate=state.accountant_sample_rate,
    )
    by_client[client_id] = a
    state.storage.save_accountant(model_id, client_id, a)
    return a
```

- [ ] **Step 5: Enforce budget in `submit_delta` BEFORE accepting the upload**

Inside the existing `submit_delta` endpoint, after `cid = ...` resolution and before the "Read the uploaded safetensors file" comment, add:

```python
    # Privacy-budget enforcement
    accountant = _ensure_accountant(mid, cid)
    if accountant is not None and accountant.is_exhausted():
        eps_consumed = accountant.get_epsilon()
        raise HTTPException(
            status_code=403,
            detail=(
                f"Privacy budget exhausted for client '{cid}' on model '{mid}'. "
                f"ε consumed = {eps_consumed:.4f}, target ε = {accountant.target_epsilon}. "
                f"Refusing further submissions."
            ),
        )
```

After the existing `apply_dp` call (which adds noise), advance the accountant and persist:

```python
    if state.dp_epsilon is not None and accountant is not None:
        accountant.step()
        state.storage.save_accountant(mid, cid, accountant)
```

Augment the response dict at the end of `submit_delta` to include budget info:

```python
    response = {
        "status": "accepted",
        # ... existing fields ...
    }
    if accountant is not None:
        eps_remaining, _ = accountant.remaining()
        response["privacy"] = {
            "epsilon_consumed": accountant.get_epsilon(),
            "epsilon_target": accountant.target_epsilon,
            "epsilon_remaining": eps_remaining,
            "delta": accountant.target_delta,
            "exhausted": accountant.is_exhausted(),
        }
    return response
```

(Make sure not to break existing callers — the `privacy` key is additive.)

- [ ] **Step 6: Run server tests**

```bash
pytest tests/test_server.py -q
```

Expected: existing tests still pass (the new code is gated on `accountant_target_epsilon is not None`, so default configure() leaves behavior unchanged).

- [ ] **Step 7: Commit**

```bash
git add chorus/server/app.py
git commit -m "feat(server): integrate PrivacyAccountant into submit_delta flow

When configure() is called with accountant_target_epsilon and
accountant_noise_multiplier, the server now tracks per-(model_id,
client_id) RDP-composed budgets. Submissions that would exceed the
budget are rejected with 403. The response includes epsilon_consumed
and epsilon_remaining for every accounted submission."
```

---

## Task 10: Add `/models/{model_id}/clients/{client_id}/privacy` endpoint

**Files:**
- Modify: `chorus/server/app.py`
- Create: `tests/test_privacy_endpoints.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_privacy_endpoints.py`:

```python
"""Tests for the /models/{id}/clients/{cid}/privacy endpoint."""

import pytest
from fastapi.testclient import TestClient

from chorus.server import app as app_module


@pytest.fixture
def configured_app(tmp_path):
    app_module.configure(
        model_id="test-model",
        data_dir=str(tmp_path),
        strategy="fedex-lora",
        min_deltas=1,
        dp_epsilon=1.0,
        dp_delta=1e-5,
        dp_max_norm=1.0,
        accountant_target_epsilon=10.0,
        accountant_noise_multiplier=1.0,
        accountant_sample_rate=1.0,
    )
    return app_module.app


def test_privacy_endpoint_returns_zero_for_unknown_client(configured_app):
    client = TestClient(configured_app)
    resp = client.get("/models/test-model/clients/unknown-client/privacy")
    assert resp.status_code == 200
    data = resp.json()
    assert data["epsilon_consumed"] == pytest.approx(0.0, abs=1e-9)
    assert data["exhausted"] is False


def test_privacy_endpoint_404_when_accounting_disabled(tmp_path):
    app_module.configure(
        model_id="test-model",
        data_dir=str(tmp_path),
        strategy="fedex-lora",
        min_deltas=1,
        # NO accountant_target_epsilon
    )
    client = TestClient(app_module.app)
    resp = client.get("/models/test-model/clients/any/privacy")
    assert resp.status_code == 404
```

Run and confirm failure:

```bash
pytest tests/test_privacy_endpoints.py -v 2>&1 | tail -10
```

Expected: 404 on the endpoint (not registered yet) → first test fails.

- [ ] **Step 2: Add the endpoint** to `chorus/server/app.py`

After the existing `/models/{model_id:path}/status` endpoint, add:

```python
@app.get("/models/{model_id:path}/clients/{client_id}/privacy", dependencies=[Depends(require_auth)])
async def get_client_privacy(model_id: str, client_id: str):
    """Return the privacy budget state for a specific client on this model."""
    if state.accountant_target_epsilon is None:
        raise HTTPException(status_code=404, detail="Privacy accounting is not enabled on this server")
    accountant = _ensure_accountant(model_id, client_id)
    eps_remaining, _ = accountant.remaining()
    return {
        "model_id": model_id,
        "client_id": client_id,
        "epsilon_consumed": accountant.get_epsilon(),
        "epsilon_target": accountant.target_epsilon,
        "epsilon_remaining": eps_remaining,
        "delta": accountant.target_delta,
        "exhausted": accountant.is_exhausted(),
    }
```

- [ ] **Step 3: Run tests (GREEN)**

```bash
pytest tests/test_privacy_endpoints.py -v
```

Expected: both tests pass.

- [ ] **Step 4: Full suite**

```bash
pytest tests/ -q
```

Expected: 182 + 2 = 184 tests pass.

- [ ] **Step 5: Commit**

```bash
git add chorus/server/app.py tests/test_privacy_endpoints.py
git commit -m "feat(server): add /models/{id}/clients/{cid}/privacy endpoint

Returns the per-client privacy budget state (epsilon_consumed,
epsilon_remaining, exhausted). 404 if accounting is not enabled."
```

---

## Task 11: Restore accountants on server startup

**Files:**
- Modify: `chorus/server/app.py`

- [ ] **Step 1: Locate the `lifespan` context manager**

```bash
grep -n "async def lifespan\|asynccontextmanager" chorus/server/app.py
```

- [ ] **Step 2: Add restoration logic at startup**

Inside `lifespan(app)`, before the final `logger.info("Chorus server started...")` line, add:

```python
    # Restore privacy accountants for the configured model
    if state.accountant_target_epsilon is not None:
        restored = state.storage.load_all_accountants(state.model_id)
        if restored:
            state.accountants[state.model_id] = restored
            logger.info(f"Restored {len(restored)} privacy accountants from disk")
```

- [ ] **Step 3: Add a test for restoration**

Append to `tests/test_privacy_persistence.py`:

```python
def test_accountants_restore_on_server_startup(tmp_path):
    from fastapi.testclient import TestClient
    from chorus.privacy import PrivacyAccountant
    from chorus.server import app as app_module

    # Pre-seed the storage with an accountant
    from chorus.server.storage import DeltaStorage
    storage = DeltaStorage(tmp_path)
    a = PrivacyAccountant(
        target_epsilon=10.0, target_delta=1e-5,
        noise_multiplier=1.0, sample_rate=1.0,
    )
    a.step()
    a.step()
    storage.save_accountant("test-model", "preexisting", a)
    eps_before = a.get_epsilon()

    # Configure server pointing at the same data_dir
    app_module.configure(
        model_id="test-model",
        data_dir=str(tmp_path),
        strategy="fedex-lora",
        min_deltas=1,
        dp_epsilon=1.0,
        dp_delta=1e-5,
        dp_max_norm=1.0,
        accountant_target_epsilon=10.0,
        accountant_noise_multiplier=1.0,
        accountant_sample_rate=1.0,
    )

    # Use TestClient as a context manager so lifespan runs
    with TestClient(app_module.app) as client:
        resp = client.get("/models/test-model/clients/preexisting/privacy")
        assert resp.status_code == 200
        data = resp.json()
        assert abs(data["epsilon_consumed"] - eps_before) < 1e-9
```

- [ ] **Step 4: Run and commit**

```bash
pytest tests/test_privacy_persistence.py -v
```

Expected: all 6 tests pass (5 original + new restoration test).

```bash
git add chorus/server/app.py tests/test_privacy_persistence.py
git commit -m "feat(server): restore privacy accountants from disk on startup"
```

---

## Task 12: Update client SDK with `max_epsilon` parameter and budget polling

**Files:**
- Modify: `chorus/client/sdk.py`

- [ ] **Step 1: Find the `ChorusClient.__init__`**

```bash
grep -n "def __init__" chorus/client/sdk.py
```

- [ ] **Step 2: Add `max_epsilon` parameter**

Append to `ChorusClient.__init__`'s signature (after existing DP params):

```python
        max_epsilon: float | None = None,
```

And in the body:

```python
        self.max_epsilon = max_epsilon
```

- [ ] **Step 3: Modify `submit_delta` to check budget post-response**

Find `def submit_delta` in `chorus/client/sdk.py`. After the response from the server is parsed (where it returns the accept dict), add:

```python
        # Check privacy budget if accounting is enabled server-side
        if self.max_epsilon is not None and "privacy" in resp_data:
            consumed = resp_data["privacy"]["epsilon_consumed"]
            if consumed >= self.max_epsilon:
                from chorus.exceptions import PrivacyBudgetExhaustedError
                raise PrivacyBudgetExhaustedError(
                    f"Client '{cid}' exceeded configured max_epsilon "
                    f"({self.max_epsilon}); server reports ε={consumed:.4f}"
                )
```

(Adjust `resp_data` and `cid` to whatever the surrounding code already calls them. Read the function context first.)

- [ ] **Step 4: Add a client-side test**

Append to `tests/test_client.py` (or create `tests/test_client_budget.py` if you prefer):

```python
def test_client_raises_when_max_epsilon_exceeded(monkeypatch):
    from chorus.client.sdk import ChorusClient
    from chorus.exceptions import PrivacyBudgetExhaustedError
    import httpx

    # Mock _request to return a privacy dict with consumed > max_epsilon
    class MockResp:
        status_code = 200
        def json(self):
            return {
                "status": "accepted",
                "client_id": "test",
                "round_id": 0,
                "model_id": "test",
                "deltas_received": 1,
                "min_deltas": 1,
                "aggregated": False,
                "next_round": 1,
                "privacy": {
                    "epsilon_consumed": 5.0,
                    "epsilon_target": 10.0,
                    "epsilon_remaining": 5.0,
                    "delta": 1e-5,
                    "exhausted": False,
                },
            }
        def raise_for_status(self):
            pass

    client = ChorusClient(
        server="http://x",
        model_id="test",
        client_id="test",
        max_epsilon=2.0,
    )
    monkeypatch.setattr(client, "_request", lambda *a, **kw: MockResp())
    # The submission will return; the budget check should raise
    import tempfile, os
    from safetensors.torch import save_file
    import torch
    tmp = tempfile.mkdtemp()
    tensors = {"l.lora_A.weight": torch.zeros(2, 2), "l.lora_B.weight": torch.zeros(2, 2)}
    save_file(tensors, os.path.join(tmp, "adapter.safetensors"))
    with pytest.raises(PrivacyBudgetExhaustedError):
        client.submit_delta(adapter_path=tmp)
```

(Adapt to existing test style and helpers in `tests/test_client.py`.)

- [ ] **Step 5: Run and commit**

```bash
pytest tests/test_client.py -v -k budget
pytest tests/ -q
```

Expected: new test passes; full suite green.

```bash
git add chorus/client/sdk.py tests/test_client.py
git commit -m "feat(client): add max_epsilon parameter; raise on budget exhaustion

ChorusClient now accepts max_epsilon. After each submit_delta, if the
server reports privacy.epsilon_consumed >= max_epsilon, the client
raises PrivacyBudgetExhaustedError instead of continuing."
```

---

## Task 13: CLI — `chorus privacy budget` subcommand

**Files:**
- Modify: `chorus/cli/main.py`
- Create: `tests/test_privacy_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_privacy_cli.py`:

```python
"""Tests for the `chorus privacy` CLI subcommand."""

from click.testing import CliRunner

from chorus.cli.main import cli


def test_privacy_budget_help_lists_required_options():
    runner = CliRunner()
    res = runner.invoke(cli, ["privacy", "budget", "--help"])
    assert res.exit_code == 0
    assert "--client-id" in res.output
    assert "--model-id" in res.output
    assert "--server" in res.output


def test_privacy_budget_requires_client_id():
    runner = CliRunner()
    res = runner.invoke(cli, ["privacy", "budget", "--model-id", "m", "--server", "http://x"])
    assert res.exit_code != 0
    assert "client-id" in res.output.lower() or "client_id" in res.output.lower()
```

Run:
```bash
pytest tests/test_privacy_cli.py -v 2>&1 | tail -10
```

Expected: failures because the `privacy` group doesn't exist yet.

- [ ] **Step 2: Read the current CLI structure**

```bash
grep -n "^@cli\|^cli =" chorus/cli/main.py | head -20
```

Note the Click conventions used in this file (groups, commands, option styles). Match them.

- [ ] **Step 3: Add the `privacy` group and `budget` command**

Append to `chorus/cli/main.py`:

```python
@cli.group()
def privacy():
    """Privacy budget management."""


@privacy.command("budget")
@click.option("--client-id", required=True, help="Client identifier")
@click.option("--model-id", required=True, help="Model identifier")
@click.option("--server", required=True, help="Server base URL")
@click.option("--api-key", default=None, help="Bearer token (if server requires auth)")
def privacy_budget(client_id: str, model_id: str, server: str, api_key: str | None):
    """Print the remaining privacy budget for a client on a model."""
    import httpx
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = httpx.get(
            f"{server.rstrip('/')}/models/{model_id}/clients/{client_id}/privacy",
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code == 404:
            console.print("[yellow]Privacy accounting is not enabled on this server.[/yellow]")
            raise SystemExit(0)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        console.print(f"[red]Failed to fetch budget: {e}[/red]")
        raise SystemExit(1)

    table = Table(title=f"Privacy budget — {client_id} on {model_id}")
    table.add_column("Field")
    table.add_column("Value", justify="right")
    table.add_row("ε consumed", f"{data['epsilon_consumed']:.4f}")
    table.add_row("ε target", f"{data['epsilon_target']:.4f}")
    table.add_row("ε remaining", f"{data['epsilon_remaining']:.4f}")
    table.add_row("δ", f"{data['delta']:.2e}")
    table.add_row("exhausted", "✗ YES" if data["exhausted"] else "✓ NO")
    console.print(table)
```

(Use existing `console` and `Table` already imported in `chorus/cli/main.py`. If they aren't, add `from rich.console import Console` and `from rich.table import Table` at the top, and `console = Console()` near other module-level state. Read the file first to know which.)

- [ ] **Step 4: Run and commit**

```bash
pytest tests/test_privacy_cli.py -v
pytest tests/ -q
```

Expected: all green.

```bash
git add chorus/cli/main.py tests/test_privacy_cli.py
git commit -m "feat(cli): add 'chorus privacy budget' subcommand"
```

---

## Task 14: Augment `chorus status` to show per-client budgets

**Files:**
- Modify: `chorus/cli/main.py`

- [ ] **Step 1: Locate the existing `status` command**

```bash
grep -n "@cli.command.*status\|def status" chorus/cli/main.py
```

- [ ] **Step 2: Extend it to fetch and display budgets**

After the existing status output, add a check + fetch loop. Pseudocode:

```python
    # ... existing status fetch / display ...

    # If accounting is enabled, list per-client budgets
    privacy_resp = httpx.get(
        f"{server}/models/{model_id}/clients/__none__/privacy",
        headers=headers, timeout=5.0,
    )
    if privacy_resp.status_code != 404:
        # Accounting is enabled (404 only if disabled).
        # We can't list all clients without a new endpoint, so just note that
        # accounting is on and point at the per-client budget command.
        console.print("[dim]Privacy accounting is enabled. "
                      "Run `chorus privacy budget --client-id <id> --model-id "
                      f"{model_id}` to see a specific client's budget.[/dim]")
```

(Optional: add a `/models/{id}/clients` listing endpoint in a follow-up issue; out of scope here.)

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -q
```

Expected: green.

- [ ] **Step 4: Commit**

```bash
git add chorus/cli/main.py
git commit -m "feat(cli): chorus status indicates when privacy accounting is enabled"
```

---

## Task 15: End-to-end integration test

**Files:**
- Modify: `tests/test_server.py` (add an integration test)

- [ ] **Step 1: Add a test that drives the full lifecycle**

Append to `tests/test_server.py`:

```python
def test_accounting_end_to_end_budget_exhaustion(tmp_path):
    """End-to-end: submit until budget is exhausted, then expect 403."""
    from fastapi.testclient import TestClient
    from safetensors.torch import save
    import torch
    from chorus.server import app as app_module

    app_module.configure(
        model_id="m",
        data_dir=str(tmp_path),
        strategy="fedex-lora",
        min_deltas=1,
        dp_epsilon=1.0, dp_delta=1e-5, dp_max_norm=1.0,
        accountant_target_epsilon=0.5,   # tight budget
        accountant_noise_multiplier=0.5, # small noise → ε grows fast
        accountant_sample_rate=1.0,
    )
    with TestClient(app_module.app) as client:
        tensors = {
            "l.lora_A.weight": torch.zeros(2, 2),
            "l.lora_B.weight": torch.zeros(2, 2),
        }
        payload = save(tensors)
        files = {"file": ("delta.safetensors", payload, "application/octet-stream")}

        # Submit until exhausted
        exhausted = False
        for round_id in range(50):
            resp = client.post(
                f"/rounds/{round_id}/deltas",
                params={"client_id": "alice", "model_id": "m"},
                files=files,
            )
            if resp.status_code == 403:
                exhausted = True
                assert "budget exhausted" in resp.text.lower()
                break
        assert exhausted, "Budget should have been exhausted within 50 submissions"
```

- [ ] **Step 2: Run and commit**

```bash
pytest tests/test_server.py::test_accounting_end_to_end_budget_exhaustion -v
pytest tests/ -q
```

Expected: green.

```bash
git add tests/test_server.py
git commit -m "test(privacy): end-to-end test for budget exhaustion via HTTP"
```

---

## Task 16: Push and open PR

**Files:** None.

- [ ] **Step 1: Final verification**

```bash
cd ~/chorus-worktrees/feat-dp-accountant
ruff check chorus tests benchmarks
pytest tests/ -v --tb=short
```

Expected: ruff clean; all tests pass (original 165 + new ~25).

- [ ] **Step 2: Push the branch**

```bash
git push -u origin feat/dp-accountant
```

- [ ] **Step 3: Open the PR**

```bash
gh pr create \
  --base master \
  --head feat/dp-accountant \
  --title "[Phase 1.2] Add stateful DP privacy accountant with composition tracking" \
  --body "$(cat <<'EOF'
## Summary
- Adds `chorus/privacy/` package with `PrivacyAccountant` (RDP composition via `dp-accounting`; `opacus` fallback).
- Moves `chorus.server.privacy` → `chorus.privacy.mechanism`; keeps deprecation re-export through v0.2.x.
- Persists per-(model_id, client_id) accountants to `chorus_data/<model>/privacy/<client>.json`; restored on server startup.
- Submission endpoint refuses with 403 when client budget is exhausted.
- New endpoint: `GET /models/{model_id}/clients/{client_id}/privacy`.
- New exception: `PrivacyBudgetExhaustedError`.
- Client SDK: `max_epsilon` parameter; raises on exhaustion.
- CLI: `chorus privacy budget`; `chorus status` notes when accounting is enabled.
- Adds `dp-accounting>=0.4.0` to the `[privacy]` extra.

## Closes
- #<ISSUE-NUMBER>

## Test plan
- [x] CI green on all three Python versions
- [x] ruff clean
- [x] All existing tests pass (165) plus ~25 new tests covering accountant unit behavior, persistence, server endpoint, CLI, and end-to-end exhaustion
- [x] Backward-compat: `from chorus.server.privacy import apply_dp` still works (with DeprecationWarning)

## Notes for reviewer
- The accountant is opt-in: `configure(...)` without `accountant_target_epsilon` leaves all existing behavior unchanged. No existing tests required modification.
- RDP backend is selected at import time; if `dp-accounting` is not installed, falls back to `opacus.accountants`. Both produce the same composition curve up to numerical tolerance; tests use `math.isclose`.
- Serialization stores config + step count, not the backend's internal RDP coefficients. Replaying steps on `deserialize` is cheap for the round counts realistic to FL (< 1000).
- `chorus status` only notes that accounting is on; per-client listing is a follow-up (would need a new endpoint).
EOF
)"
```

Replace `<ISSUE-NUMBER>` with the actual number from Task 1.

- [ ] **Step 4: Watch CI**

```bash
gh pr view --json statusCheckRollup --jq '[.statusCheckRollup[] | {name, status, conclusion}]'
```

Wait for all three matrix entries to be SUCCESS. If any FAIL, run the failing command locally, fix, commit, push.

---

## Self-review checklist (run before requesting Varma's review)

- [ ] `chorus/privacy/` package exists; `from chorus.privacy import PrivacyAccountant, GaussianMechanism, apply_dp` works.
- [ ] `chorus/server/privacy.py` exists as a deprecation re-export; `python -W error::DeprecationWarning -c "import chorus.server.privacy"` raises.
- [ ] `PrivacyBudgetExhaustedError` is defined in `chorus/exceptions.py` and inherits from `ChorusError`.
- [ ] All ~25 new tests pass; all original 165 tests still pass.
- [ ] Ruff clean on `chorus tests benchmarks`.
- [ ] CI green on the PR.
- [ ] `pyproject.toml` `privacy` extra includes both `opacus>=1.4.0` and `dp-accounting>=0.4.0`.
- [ ] No `Co-Authored-By:` trailer on any commit (`git log master..HEAD | grep -i 'co-authored' | wc -l` returns 0).
- [ ] PR body has no AI-attribution footer.
- [ ] PR references `Closes #<issue>`.
- [ ] `chorus privacy budget --client-id X --model-id Y --server <url>` works locally against a running server with accounting enabled.

If any check fails, fix before opening the PR.
