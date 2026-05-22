# Feature 6: v0.2.0 Release — Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax. Before starting, ensure you are in the worktree at `~/chorus-worktrees/release-v0.2.0` on branch `release/v0.2.0` (created via `git worktree add` from the parent checkout at `~/chorus`).

**Goal:** Prepare the v0.2.0 release on master. This PR lands every code change required for the release — version bumps in all source-of-truth locations, CHANGELOG entry, release-notes draft, PyPI publish workflow — but does **not** cut the git tag or publish to PyPI. Those externally-visible actions are reserved for the user (`varmabudharaju`) to trigger explicitly once they've reviewed the prepared release artifacts.

The paid GPU benchmark run that produces `benchmarks/results/v0.2.0/*.json` is **not blocking F6**. The release ships the code; the published numbers will land in a follow-up `chore/v0.2.0-benchmark-results` PR (either before the tag is cut, or as an addendum to the v0.2.0 GH release page). CHANGELOG and release notes describe this honestly.

**Architecture:** Five touchpoints:
1. **Version triplet** — `pyproject.toml`, `chorus/__init__.py:__version__`, `chorus/server/app.py:128`'s FastAPI version kwarg. All three must agree.
2. **CHANGELOG.md** — new `## [0.2.0]` section in Keep-a-Changelog format, summarizing the F1–F5 deliverables and the follow-up PRs.
3. **`docs/releases/v0.2.0.md`** — narrative GH-release-notes draft. Mirrors what gets pasted into the GitHub release UI when the tag goes live.
4. **`.github/workflows/release.yml`** — workflow that triggers on `v*` tag pushes, builds the wheel + sdist, and publishes to PyPI via trusted publishing (OIDC, no stored API token). Requires a one-time PyPI trusted-publisher configuration in the project settings — documented in the release notes file.
5. **`tests/test_version.py`** — a one-test guard that asserts the three version sources agree, so future drift fails CI loudly.

**Tone for CHANGELOG and release notes:** Same voice as `docs/honest-tradeoffs.md`. Concrete, direct, no marketing. Lead with what changed; surface the credibility-mandate framing once, near the top.

**Spec sections:**
- `docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md` §11 (Definition of Done)
- Master plan: `docs/superpowers/plans/2026-05-19-phase-1-execution-plan.md` F6 row

---

## Hard-won context from F1–F5 to respect

- **No `Co-Authored-By: Claude` trailers. No "Generated with Claude Code" footers.** Sole attribution. See master plan §3.
- **Conventional commits:** `chore(release):`, `docs(changelog):`, `test(version):`, `ci(release):`. One logical change per commit.
- **Cutting the git tag and publishing to PyPI are NOT part of this PR.** The implementer prepares the artifacts; the user triggers the externally-visible actions.
- **`benchmarks/results/v0.2.0/` is empty as of this PR.** Don't fabricate numbers. The CHANGELOG and release notes must be honest about this.
- **Three lingering follow-ups** the user might want to address in a follow-up chore PR, not blocking F6:
  - README "Aggregation Strategies" table row for `fedex-lora` could carry the same "([when?](docs/honest-tradeoffs.md#exactness))" qualifier the hero line got.
  - Configuration Examples "Secure production server" code block could be updated to show `--accountant-target-epsilon` + `--accountant-noise-multiplier` alongside `--dp-epsilon` now that they exist.
  - `chorus.eval._evaluate_aggregated` does not inject accumulated residuals into the base when computing perplexity — only Frobenius reflects the fold path. For the GPU benchmark run, the perplexity numbers in fold-on vs fold-off will be identical (only Frobenius differs). Worth a follow-up issue.

---

## File Structure

- **Modify:** `pyproject.toml` — bump `version = "0.1.0"` → `"0.2.0"`.
- **Modify:** `chorus/__init__.py` — bump `__version__ = "0.1.0"` → `"0.2.0"`.
- **Modify:** `chorus/server/app.py` — bump the FastAPI `version="0.1.0"` kwarg on line 128 → `"0.2.0"`.
- **Modify:** `CHANGELOG.md` — prepend the `## [0.2.0]` section described below.
- **Create:** `docs/releases/v0.2.0.md` — GH release notes draft.
- **Create:** `.github/workflows/release.yml` — PyPI publish workflow on `v*` tags.
- **Create:** `tests/test_version.py` — version-triplet consistency guard.

---

## Pre-flight (Task 0)

```bash
cd ~/chorus
git checkout master && git pull origin master
git worktree add ../chorus-worktrees/release-v0.2.0 -b release/v0.2.0
cd ../chorus-worktrees/release-v0.2.0
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,peft,privacy]"
```

All subsequent paths are relative to `~/chorus-worktrees/release-v0.2.0`.

---

## Task 1: Open GitHub issue

- [ ] **Step 1: Create the issue**

```bash
gh issue create \
  --title "[Phase 1.6] v0.2.0 release preparation" \
  --label "phase-1" \
  --body "$(cat <<'EOF'
**Phase:** 1 (Credibility & Honesty)
**Spec:** [docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md](../blob/master/docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md) §11
**Plan:** [docs/superpowers/plans/2026-05-22-feature-6-v020-release.md](../blob/master/docs/superpowers/plans/2026-05-22-feature-6-v020-release.md)

## Scope
Land the v0.2.0 release-prep code changes on master: version bumps everywhere they appear, CHANGELOG entry, release-notes draft, PyPI publish workflow on tag, and a version-consistency test guard. Does NOT cut the v0.2.0 git tag or publish to PyPI — those are reserved for explicit user action after review.

## Acceptance criteria
- [ ] pyproject.toml, chorus/__init__.py, and chorus/server/app.py all report version 0.2.0.
- [ ] tests/test_version.py asserts the three sources agree (fails loudly on future drift).
- [ ] CHANGELOG.md has a complete [0.2.0] section in Keep-a-Changelog format describing every merged PR in Phase 1 plus the three follow-up PRs.
- [ ] docs/releases/v0.2.0.md exists with narrative GH-release-notes copy.
- [ ] .github/workflows/release.yml triggers on v* tags and publishes to PyPI via trusted publishing.
- [ ] All existing tests still pass; new version-consistency test passes.

## Out of scope
- Cutting the v0.2.0 git tag.
- Publishing to PyPI.
- Generating the v0.2.0 benchmark numbers (separate paid GPU run + chore PR).
- Any code changes outside release plumbing.
EOF
)"
```

Record the issue number.

---

## Task 2: Version triplet bump

**Files:**
- Modify: `pyproject.toml`
- Modify: `chorus/__init__.py`
- Modify: `chorus/server/app.py`

- [ ] **Step 1: pyproject.toml**

```bash
sed -i.bak 's/^version = "0\.1\.0"$/version = "0.2.0"/' pyproject.toml && rm pyproject.toml.bak
```

Verify: `grep '^version' pyproject.toml` → `version = "0.2.0"`.

- [ ] **Step 2: chorus/__init__.py**

```bash
sed -i.bak 's/^__version__ = "0\.1\.0"$/__version__ = "0.2.0"/' chorus/__init__.py && rm chorus/__init__.py.bak
```

Verify: `grep __version__ chorus/__init__.py` → `__version__ = "0.2.0"`.

- [ ] **Step 3: chorus/server/app.py line 128**

The FastAPI app constructor has `version="0.1.0"` as a kwarg. Find it (use a precise grep so you don't accidentally edit version strings elsewhere) and bump it.

```bash
grep -n 'version="0\.1\.0"' chorus/server/app.py
```

Should report exactly one line (around 128). Replace just that occurrence using `Edit` or a careful `sed` with surrounding context:

```bash
sed -i.bak 's/version="0\.1\.0"/version="0.2.0"/' chorus/server/app.py && rm chorus/server/app.py.bak
grep -n 'version=' chorus/server/app.py
```

- [ ] **Step 4: Reinstall in editable mode so `chorus --version` picks up the bump**

```bash
pip install -e ".[dev,peft,privacy]"
chorus --version  # if Click exposes one; otherwise:
python -c "import chorus; print(chorus.__version__)"
```

Expected output: `0.2.0`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml chorus/__init__.py chorus/server/app.py
git commit -m "chore(release): bump version 0.1.0 -> 0.2.0 in all three sources of truth"
```

---

## Task 3: Version-consistency test guard

**File:**
- Create: `tests/test_version.py`

```python
"""Sanity check: the version string is defined in three places (pyproject, package
init, FastAPI app) and they must agree. This test fails loudly if any drifts."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(path: Path) -> str:
    return (REPO_ROOT / path).read_text()


def _pyproject_version() -> str:
    text = _read(Path("pyproject.toml"))
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert m, "pyproject.toml has no version line"
    return m.group(1)


def _package_version() -> str:
    import chorus
    return chorus.__version__


def _fastapi_version() -> str:
    text = _read(Path("chorus/server/app.py"))
    m = re.search(r'version\s*=\s*"([^"]+)"', text)
    assert m, "chorus/server/app.py has no FastAPI version kwarg"
    return m.group(1)


def test_version_triplet_agrees():
    pp = _pyproject_version()
    pkg = _package_version()
    fa = _fastapi_version()
    assert pp == pkg == fa, (
        f"Version drift detected: pyproject={pp!r}, package={pkg!r}, fastapi={fa!r}. "
        "Update all three together for any version bump."
    )
```

- [ ] **Step 1: Run the test**

```bash
pytest tests/test_version.py -v
```

Expected: pass. If it fails, one of the three version locations didn't get bumped in Task 2.

- [ ] **Step 2: Commit**

```bash
git add tests/test_version.py
git commit -m "test(version): assert pyproject/package/fastapi version strings agree"
```

---

## Task 4: CHANGELOG entry

**File:**
- Modify: `CHANGELOG.md`

Prepend a `## [0.2.0]` section between the existing top header and the `## [0.1.0]` entry. The section uses Keep-a-Changelog subheadings (`Added`, `Changed`, `Fixed`, `Deprecated`, `Removed`, `Security`). Skip any subheading that has no entries.

Use this content (adjust dates if they need to change before merge):

```markdown
## [0.2.0] - 2026-05-22

This is the Phase 1 "Credibility & Honesty" release. Every claim in the README and `docs/honest-tradeoffs.md` is grounded in code citations. The aggregation server's exactness story is qualified with the residual-folding precondition; the DP story is qualified with the accountant requirement; the Byzantine story is qualified as sanity checks against naive attackers; the multi-tenant and production-readiness limits are stated plainly.

Benchmark numbers from the v0.2.0 paid GPU run will land in a follow-up PR (`chore/v0.2.0-benchmark-results`) committing `benchmarks/results/v0.2.0/` and updating the README's results section. They are not in this release tarball.

### Added

- **`chorus.privacy.accountant.PrivacyAccountant`** — Stateful RDP composition tracker (Google `dp-accounting` primary, `opacus.accountants.RDPAccountant` fallback). Persisted per `(model_id, client_id)` under `chorus_data/<model>/privacy/<client>.json`; restored on server restart. Server refuses submissions with HTTP 403 once the configured `(epsilon, delta)` budget is exhausted; the client SDK maps that to `PrivacyBudgetExhaustedError`.
- **`--accountant-target-epsilon` and `--accountant-noise-multiplier` flags** on `chorus server` — Both must be set together for bounded DP loss. Without them, privacy loss accumulates unbounded across rounds.
- **`chorus/eval/` package** — Reusable evaluation harness: `EvalConfig` (YAML), `EvalRunner` (orchestrates per-client LoRA training → strategy aggregation → metric collection), `frobenius_reconstruction_error` and `compute_perplexity_from_loss` (`chorus.eval.metrics`), IID + Dirichlet partitioning (`chorus.eval.datasets`), JSON + markdown reports (`chorus.eval.report`).
- **`chorus eval` CLI** — `--config <yaml>`, `--check-only` (fast CI wiring check), `--output-dir`. Cross-round residual folding is wired through `EvalRunner._train_one_client` via the existing `chorus.server.weight_manager.fold_residuals_into_base` helper; DP noise is applied per client delta when `dp_epsilon` is configured. Both surface the ablations the benchmark YAMLs encode.
- **`benchmarks/configs/*.yaml`** — Six per-experiment configs covering TinyLlama-GLUE-SST2, Phi-3-mini clients sweep (`num_clients ∈ {2,5,10,20}`), Llama-3.2-1B rank ablation (`rank ∈ {4,8,16}`), Llama-3.2-1B DP ablation (`dp_epsilon ∈ {null,1.0,4.0}`), Llama-3.2-1B fold ablation (`fold_residuals ∈ {true,false}`), and a heterogeneous-rank single-point (`[4,8,16,32]`). The legacy synthetic `benchmarks/benchmark.py` is preserved under `benchmarks/legacy/`.
- **`benchmarks/run_all.py`** — Sweep expander + driver. Reads a YAML with optional `sweep:` axes, expands the Cartesian product into individual `EvalConfig` instances, runs `EvalRunner` for each, writes per-run reports under `benchmarks/results/v0.2.0/<config-stem>/<run-key>/`. `--all` iterates every YAML in `benchmarks/configs/` except `smoke.yaml`.
- **`benchmarks/verify_smoke_results.py`** — Post-run tripwire. Asserts FedEx-LoRA's mean Frobenius reconstruction error is no worse than FedAvg's within `TOLERANCE = 1e-4`. Mean across seeds rather than `min` so the regression check can't be cherry-picked.
- **GitHub Actions CI** — pytest on Python 3.10/3.11/3.12, ruff, `chorus eval --check-only` wiring smoke. Status badge in README.
- **`docs/honest-tradeoffs.md`** — Seven sections covering exactness, DP, Byzantine, multi-tenant, heterogeneous clients, production readiness, and the roadmap. Every behavioral claim is grounded in a `file:line` or function-name citation against master.
- **Light docs-link integrity test** (`tests/test_docs_links.py`) — Asserts every internal Markdown link in `README.md` and `docs/honest-tradeoffs.md` resolves to a file and anchor that exist.

### Changed

- **README** is now qualified throughout. The "mathematically exact" hero line includes the residual-folding precondition; a new "Honest tradeoffs" section near the top one-lines each major caveat with links into the new doc; the Security Features section's Byzantine bullet is retitled "Sanity-check defenses against naive attackers" with an explicit catch/miss list; the DP bullet describes the accountant and the unbounded-loss risk; a production-readiness paragraph labels the project alpha software.
- **Aggregation server** keeps the same surface area but now exposes the accountant via CLI flags (previously only via `chorus.server.app.configure()` programmatically).

### Security

- **DP accountant integration** prevents unbounded privacy loss across federation rounds when the new server flags are set. Without them, the per-round Gaussian mechanism still runs but composition is not tracked or bounded — this is now stated plainly in the README, the new tradeoffs doc, and the server startup banner.

### Test coverage

- v0.1.0: 165 tests passing.
- v0.2.0 baseline: **249 tests passing** on `pytest -m "not network"`. Two additional tests are gated behind `@pytest.mark.network` (fold-residuals and DP-epsilon end-to-end on `hf-internal-testing/tiny-random-LlamaForCausalLM`); both pass locally with a warm HF cache.

### Known gaps shipping with this release

- `benchmarks/results/v0.2.0/` is empty pending the paid GPU run. Numbers will land in a follow-up PR.
- `chorus.eval.runner._evaluate_aggregated` does not inject accumulated residuals into the base when computing perplexity — only Frobenius reflects the fold path. For the planned GPU benchmark run, fold-on vs fold-off perplexity numbers will be identical; only the Frobenius column will differ. Worth a follow-up before v0.2.1.
- API keys remain global; no per-model scoping. Documented in `docs/honest-tradeoffs.md#multi-tenant-scope`. Phase 4 work.

[0.2.0]: https://github.com/varmabudharaju/chorus/releases/tag/v0.2.0
```

Also update the link-reference block at the bottom of `CHANGELOG.md` so the existing `[0.1.0]` reference link stays intact and the new `[0.2.0]` link is added.

- [ ] **Step 1: Apply the edit**

Use `Edit` to insert the new section after the introductory paragraph and before `## [0.1.0]`. Add the `[0.2.0]: ...` link reference adjacent to the existing `[0.1.0]: ...`.

- [ ] **Step 2: Verify the docs-link integrity test still passes**

```bash
pytest tests/test_docs_links.py -v
```

(CHANGELOG isn't in `TARGET_FILES` for that test, but if you added any new cross-links into `docs/honest-tradeoffs.md` they need to resolve.)

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): add [0.2.0] entry covering F1-F5 + the three follow-up PRs"
```

---

## Task 5: GH release notes draft

**File:**
- Create: `docs/releases/v0.2.0.md`

Tracking the release notes as a committed file (rather than only on the GitHub release UI) lets the doc be reviewed, linked from CHANGELOG, and reproduced if the GH release ever needs to be recreated. Content is a narrative summary that complements (not duplicates) the CHANGELOG.

```markdown
# v0.2.0 — Phase 1: Credibility & Honesty

**Release date:** 2026-05-22 (TBD — replace with the actual tag date when the user cuts `v0.2.0`).

## What changed

Chorus v0.1.0 made three claims its code didn't fully back: "mathematically exact" aggregation (which was only true with residual folding), "differential privacy" (which was per-round Gaussian noise with no composition tracking), and "Byzantine defenses" (which were sanity checks against naive attackers). v0.2.0 closes those credibility gaps:

- The mathematically-exact claim now ships with a stateful **privacy accountant** (RDP composition via Google `dp-accounting`, `opacus` fallback) and CLI flags (`--accountant-target-epsilon`, `--accountant-noise-multiplier`) that bound DP loss across rounds. Without them, privacy loss accumulates unbounded — stated plainly in the README, the new tradeoffs doc, and the server startup banner.
- A new `chorus.eval` package and `chorus eval` CLI run simulated federations on real models with real data and emit comparable JSON + markdown reports. Cross-round residual folding and DP application are both wired through `EvalRunner` so the benchmark YAMLs are no longer Potemkin configs.
- The `benchmarks/` directory now hosts six per-experiment YAML configs covering each ablation axis (clients sweep, rank ablation, DP ablation, fold ablation, heterogeneous-rank, TinyLlama+GLUE-SST2 sanity), a `run_all.py` sweep driver, and a `verify_smoke_results.py` post-run tripwire that uses **mean** Frobenius across seeds (not `min`, which would cherry-pick).
- The README is qualified throughout: the hero line carries the residual-folding precondition, a new "Honest tradeoffs" section surfaces every caveat, and the production-readiness paragraph labels the project alpha software.
- `docs/honest-tradeoffs.md` lands as the explicit honest-disclosure document — seven sections, every behavioral claim grounded in a `file:line` citation.
- CI runs pytest across Python 3.10/3.11/3.12, ruff, and a `chorus eval --check-only` wiring smoke on every PR. Test count grew from 165 → 249 (network-gated tests not counted).

For the full per-PR breakdown, see [CHANGELOG.md](../../CHANGELOG.md).

## What's not in this release

- **Benchmark numbers.** `benchmarks/results/v0.2.0/` is empty pending the paid GPU run. They will land in a follow-up PR (`chore/v0.2.0-benchmark-results`) and be appended to this release's GitHub release page once committed. The benchmark *code* is in this release; the published *numbers* are not.
- **Cross-round residual folding in the perplexity path.** `_evaluate_aggregated` only injects accumulated residuals into the base for the Frobenius metric, not the perplexity computation. Fold-on and fold-off will show identical perplexity in the planned benchmark run; only Frobenius will differ. Slated for v0.2.1.

## Upgrading from v0.1.0

`pip install --upgrade chorus-fl` once the release is published. No config migration is required. If you were relying on per-round DP without budget enforcement, configure `--accountant-target-epsilon` and `--accountant-noise-multiplier` on the server to opt into bounded loss.

## One-time PyPI trusted-publisher setup

The new `.github/workflows/release.yml` publishes to PyPI via OIDC (no stored API token). Before the first tag publish, configure PyPI's trusted publisher:

1. Sign in to https://pypi.org as the `chorus-fl` package owner.
2. Project → Publishing → "Add a new pending publisher" with:
   - **Owner:** `varmabudharaju`
   - **Repository:** `chorus`
   - **Workflow filename:** `release.yml`
   - **Environment:** (leave blank, or set to `pypi` if you adopt environments later)
3. Save. The first tag push (`v0.2.0`) will then complete the publish.

Re-run this configuration once per major release line if you scope publishers per environment.

## Cutting the tag

When you're ready:

```bash
cd ~/chorus
git checkout master && git pull origin master
git tag -s v0.2.0 -m "v0.2.0 — Phase 1: Credibility & Honesty"
git push origin v0.2.0
```

The release workflow takes it from there: builds the wheel + sdist, runs the smoke pytest matrix, and publishes to PyPI. After the workflow completes, edit the auto-created GH release on the Releases page and paste the content of this file into the description.
```

- [ ] **Step 1: Write the file**
- [ ] **Step 2: Commit**

```bash
mkdir -p docs/releases
git add docs/releases/v0.2.0.md
git commit -m "docs(release): add v0.2.0 GitHub release notes draft"
```

---

## Task 6: PyPI publish workflow

**File:**
- Create: `.github/workflows/release.yml`

The release workflow triggers on `v*` tag pushes, builds the package (wheel + sdist), runs a final smoke pytest matrix, and publishes to PyPI via trusted publishing (OIDC, no API token in repo secrets). Uses pinned action versions because CI safety here is non-negotiable — supply-chain attacks on release tooling are catastrophic.

```yaml
name: release

# Runs only on tag pushes matching v*.*.* (semver). Bypassed for branch pushes.
on:
  push:
    tags:
      - "v[0-9]+.[0-9]+.[0-9]+"

# Concurrency: only one release at a time, no cancellation (don't kill mid-publish).
concurrency:
  group: release
  cancel-in-progress: false

permissions:
  contents: read

jobs:
  test:
    name: Smoke test before publish
    runs-on: ubuntu-latest
    strategy:
      fail-fast: true
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev,peft,privacy]"
      - name: Lint
        run: ruff check chorus tests benchmarks
      - name: Test
        run: pytest -m "not network" -q

  build:
    name: Build distribution
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install build tools
        run: |
          python -m pip install --upgrade pip
          pip install build
      - name: Build wheel + sdist
        run: python -m build
      - name: Upload distribution as artifact
        uses: actions/upload-artifact@v4
        with:
          name: chorus-fl-dist
          path: dist/*
          if-no-files-found: error
          retention-days: 7

  publish:
    name: Publish to PyPI
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write  # required for OIDC / trusted publishing
    steps:
      - name: Download distribution
        uses: actions/download-artifact@v4
        with:
          name: chorus-fl-dist
          path: dist
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        # No `with:` block — credentials come from OIDC via the trusted publisher
        # configured at https://pypi.org/manage/project/chorus-fl/settings/publishing/
```

- [ ] **Step 1: Validate the YAML parses cleanly**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml'))"
```

- [ ] **Step 2: Confirm `actions/checkout@v4`, `actions/setup-python@v5`, and `actions/upload-artifact@v4` are the versions the existing `ci.yml` uses too (avoid drift). If the existing CI workflow uses different versions, match those so the security review is consistent.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci(release): add PyPI publish workflow triggered on v*.*.* tags via trusted publishing"
```

---

## Task 7: Final verification + PR

- [ ] **Step 1: Full local check**

```bash
ruff check chorus tests benchmarks
pytest -m "not network" -q
```

Expected: 250 passed (249 baseline + 1 new version-consistency test). Ruff clean.

- [ ] **Step 2: Dry-build the package locally**

```bash
pip install build
python -m build
```

Inspect `dist/`. There should be a `chorus_fl-0.2.0-py3-none-any.whl` and `chorus_fl-0.2.0.tar.gz`. Confirm both filenames carry `0.2.0`. If they say anything else, the version triplet didn't propagate.

Clean up after inspection:
```bash
rm -rf dist build *.egg-info
```

- [ ] **Step 3: Push + open PR**

```bash
git push -u origin release/v0.2.0
gh pr create \
  --base master \
  --head release/v0.2.0 \
  --title "[Phase 1.6] v0.2.0 release preparation" \
  --body "$(cat <<'EOF'
## Summary

Lands all v0.2.0 release-preparation code on master. Does NOT cut the tag or publish to PyPI — those are user-triggered after this merges.

- Bumps version 0.1.0 -> 0.2.0 in pyproject.toml, chorus/__init__.py, and chorus/server/app.py (the three sources of truth).
- Adds tests/test_version.py to assert the triplet agrees (one-line guard against future drift).
- CHANGELOG [0.2.0] section in Keep-a-Changelog format covering F1-F5 + the three follow-up PRs.
- docs/releases/v0.2.0.md narrative release-notes draft (mirrors what gets pasted into the GH release UI when the tag is cut).
- .github/workflows/release.yml publishes to PyPI via OIDC trusted publishing on v*.*.* tag pushes. Requires the user to configure the trusted publisher on PyPI once — instructions in the release notes file.

## Closes
- #<ISSUE-NUMBER>

## Test plan
- [x] `pytest -m "not network" -q` green (249 -> 250)
- [x] Ruff clean
- [x] `python -m build` produces chorus_fl-0.2.0-py3-none-any.whl + chorus_fl-0.2.0.tar.gz
- [x] `python -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml'))"` exits 0

## Notes for reviewer / for the user
- The benchmark numbers for v0.2.0 are NOT in this PR. They land in a follow-up `chore/v0.2.0-benchmark-results` PR after the paid GPU run. CHANGELOG and release notes are honest about this.
- To cut the actual release after this merges:
  1. `gh pr merge` this PR
  2. `git checkout master && git pull`
  3. `git tag -s v0.2.0 -m "v0.2.0 — Phase 1: Credibility & Honesty"`
  4. `git push origin v0.2.0`
  5. The release workflow runs; verify on the Actions tab.
  6. Edit the auto-created GH release and paste `docs/releases/v0.2.0.md` into the description.
- Before the first tag push, configure PyPI trusted publishing once — instructions in `docs/releases/v0.2.0.md`.
EOF
)"
```

Replace `<ISSUE-NUMBER>` with the real issue number from Task 1.

- [ ] **Step 4: Watch CI**

```bash
gh pr checks --watch
```

All three Python matrix jobs in the existing `ci.yml` should pass. The new `release.yml` workflow does NOT run on PRs — it only triggers on tags.

---

## Self-review checklist

- [ ] `pyproject.toml`, `chorus/__init__.py`, and `chorus/server/app.py` all say `0.2.0`.
- [ ] `tests/test_version.py` passes.
- [ ] `CHANGELOG.md` has a `## [0.2.0]` section with all subheadings populated honestly, including the "Known gaps shipping with this release" callout.
- [ ] `docs/releases/v0.2.0.md` reads as direct prose that complements (does not duplicate) the CHANGELOG.
- [ ] `.github/workflows/release.yml` uses pinned-major action versions matching `ci.yml`.
- [ ] `python -m build` produces wheels with `0.2.0` in the filename.
- [ ] No `Co-Authored-By` trailer on any commit; no AI-attribution footer in PR body.
- [ ] PR body explicitly states the tag + PyPI publish are reserved for the user.

---

## Out-of-scope for this PR

- Cutting the `v0.2.0` git tag.
- Publishing to PyPI.
- Generating v0.2.0 benchmark numbers (separate paid GPU run + chore PR).
- Updating the README "Aggregation Strategies" table or Configuration Examples blocks for consistency with F5's qualifier work (separate small chore PR).
- Wiring residual folding into `_evaluate_aggregated`'s perplexity path (file an issue, target v0.2.1).
- Any other code change in `chorus/` source.
