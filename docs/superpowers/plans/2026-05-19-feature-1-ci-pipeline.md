# Feature 1: CI Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Before starting, ensure you are in the worktree at `~/chorus-worktrees/feat-ci-pipeline` on branch `feat/ci-pipeline` (created via superpowers:using-git-worktrees from the parent checkout at `~/chorus`).

**Goal:** Add a GitHub Actions CI pipeline that runs ruff + pytest on every push and PR, across Python 3.10/3.11/3.12, and adds a status badge to the README.

**Architecture:** A single `.github/workflows/ci.yml` workflow with one `test` job that uses a matrix over Python versions. No external services, no model downloads, no GPU. The eval-harness `--check-only` smoke step described in the spec §4.5 is **deferred to Feature 3** (it requires the `chorus eval` subcommand that doesn't yet exist).

**Tech Stack:** GitHub Actions (Ubuntu runners), `actions/checkout@v4`, `actions/setup-python@v5`, `actions/cache@v4`, `ruff` (already in `[dev]` extra), `pytest` + `pytest-asyncio` + `pytest-httpx` (already in `[dev]` extra).

**Spec section:** `docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md` §4.5
**Master plan:** `docs/superpowers/plans/2026-05-19-phase-1-execution-plan.md` F1 row

---

## File Structure

- **Create:** `.github/workflows/ci.yml` — single workflow file with one `test` job and a Python version matrix.
- **Modify:** `README.md` — add the CI status badge line near the top, beside the existing badges (License, Python 3.10+, PyPI version).

That's it. CI is intentionally minimal in this feature; F3 will extend it.

---

## Pre-flight (Task 0)

Before writing any code, the dispatching user (Varma) does this once, in the **main** `~/chorus` checkout:

```bash
# Confirm we're on master and up to date
cd ~/chorus
git checkout master
git pull origin master

# Create the worktree on a new feature branch
git worktree add ../chorus-worktrees/feat-ci-pipeline -b feat/ci-pipeline

# Confirm
git worktree list
# Expected output includes:
#   /Users/varma/chorus-worktrees/feat-ci-pipeline  <sha> [feat/ci-pipeline]
```

The subagent then operates entirely inside `~/chorus-worktrees/feat-ci-pipeline`. All `cd` commands and file paths in the tasks below are relative to that worktree unless explicitly stated.

---

## Task 1: Open the GitHub issue

**Files:** None (issue is on GitHub, not in repo)

- [ ] **Step 1: Create the issue using `gh`**

Run from the worktree:

```bash
cd ~/chorus-worktrees/feat-ci-pipeline

gh issue create \
  --title "[Phase 1.1] Add GitHub Actions CI pipeline" \
  --label "phase-1" \
  --body "$(cat <<'EOF'
**Phase:** 1 (Credibility & Honesty)
**Spec:** [docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md](../blob/master/docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md) §4.5
**Roadmap:** [docs/superpowers/specs/2026-05-19-chorus-roadmap.md](../blob/master/docs/superpowers/specs/2026-05-19-chorus-roadmap.md)
**Plan:** [docs/superpowers/plans/2026-05-19-feature-1-ci-pipeline.md](../blob/master/docs/superpowers/plans/2026-05-19-feature-1-ci-pipeline.md)

## Scope
Add a GitHub Actions workflow that runs ruff and pytest on every push and PR, matrixed across Python 3.10, 3.11, and 3.12. Add a status badge to the README so adoption signals are visible from the repo landing page.

## Acceptance criteria
- [ ] `.github/workflows/ci.yml` exists and is syntactically valid (`actionlint` clean)
- [ ] On every push and PR, the workflow runs ruff + pytest across the Python version matrix
- [ ] All existing tests pass under all three Python versions
- [ ] README displays a CI status badge linking to the workflow page
- [ ] CI runs in under 5 minutes per matrix entry

## Out of scope
- `chorus eval --check-only` step (deferred to Feature 3 — requires the eval CLI which doesn't exist yet)
- pre-commit hooks (defer to a follow-up DX-bundle PR if requested)
- Coverage upload to Codecov / similar (defer)
- Auto-deploy to PyPI on tag (defer to F6)
EOF
)"
```

Expected: `gh` prints the new issue URL. Note the issue number (e.g. `#1`) — you'll reference it in the PR.

- [ ] **Step 2: Record the issue number**

In this plan file, you don't need to edit anything yet. The issue number lands in the master plan §1 table and in the PR body later.

---

## Task 2: Confirm the local environment is healthy

**Files:** None (read-only verification)

This is the equivalent of "write the failing test, see it fail before fixing." If pytest is broken on master, CI will fail on day 1 and we'll think it's a CI bug when it's actually an existing-tests bug. So we verify first.

- [ ] **Step 1: Install the project in editable mode with dev extras**

```bash
cd ~/chorus-worktrees/feat-ci-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Expected: pip prints `Successfully installed chorus-fl-0.1.0` along with all dependencies. No errors.

- [ ] **Step 2: Run ruff and confirm it passes (or note pre-existing issues)**

```bash
ruff check chorus tests benchmarks
```

Expected: `All checks passed!`

If ruff reports issues, **do not fix them in this PR** (out of scope). Note them and create a follow-up issue:
```bash
gh issue create --title "[chore] Fix pre-existing ruff issues" --label "chore" \
  --body "Ruff reports the following on master:\n\n\`\`\`\n<paste output>\n\`\`\`\n\nFix in a standalone PR so the CI-pipeline PR (#<ci-issue>) stays focused."
```

- [ ] **Step 3: Run pytest and confirm all existing tests pass**

```bash
pytest tests/ -v
```

Expected: All tests pass (the changelog claims 165 tests). Note the exact pass count for use in the PR body.

If any tests fail on master, **stop**. The CI feature cannot be merged on top of broken tests because CI will be red on merge. File a separate issue and report back to Varma before proceeding.

- [ ] **Step 4: Note Python version locally**

```bash
python --version
```

Record this — useful for diagnosing matrix issues later.

---

## Task 3: Create `.github/workflows/ci.yml`

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the workflows directory**

```bash
mkdir -p .github/workflows
```

Expected: directory is created. Confirm with:
```bash
ls -la .github/workflows
```

- [ ] **Step 2: Write the workflow file**

Create `.github/workflows/ci.yml` with **exactly** this content:

```yaml
name: ci

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    name: test (py${{ matrix.python }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python: ["3.10", "3.11", "3.12"]
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
          cache: pip
          cache-dependency-path: pyproject.toml

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Lint with ruff
        run: ruff check chorus tests benchmarks

      - name: Run tests with pytest
        run: pytest tests/ -v --tb=short
```

**Design notes (kept in this plan, not in the YAML file):**
- `concurrency` cancels superseded runs when a PR is force-pushed, saving CI minutes.
- `fail-fast: false` so a 3.12-only failure doesn't hide 3.10 results.
- `cache: pip` with `cache-dependency-path: pyproject.toml` halves cold-install time.
- No `--cov` flag yet — coverage upload is deferred (see issue scope). Adding `--cov` without an upload step just adds runtime for no benefit.
- No `setup-python` `check-latest: true` — we want pinned minor versions for reproducibility, not the absolute latest patch.
- Job name template `test (py3.10)` etc. makes the GitHub UI readable.

- [ ] **Step 3: Verify the YAML is valid**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
```

Expected: no output (clean parse). If you get a YAMLError, fix the indentation and rerun.

Optional, if `actionlint` is installed locally:
```bash
actionlint .github/workflows/ci.yml
```
Expected: no output. If not installed, skip — GitHub Actions will validate on push.

- [ ] **Step 4: Commit the workflow file**

```bash
git add .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
ci: add GitHub Actions workflow for ruff + pytest

Runs on every push to master and every PR targeting master.
Matrixes Python 3.10, 3.11, 3.12. Caches pip via pyproject.toml.
Cancels superseded runs via concurrency group.

Part of Phase 1 credibility release (closes #<ISSUE-NUMBER>).
Spec: docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md §4.5
EOF
)"
```

Replace `<ISSUE-NUMBER>` with the actual number from Task 1. Expected: `git log -1 --oneline` shows the new commit.

---

## Task 4: Push the branch and observe the first CI run

**Files:** None (interacts with GitHub)

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/ci-pipeline
```

Expected: GitHub prints the URL for opening a PR. Don't open the PR yet — we want to see CI pass first.

- [ ] **Step 2: Watch the workflow run**

```bash
gh run watch
```

This blocks until the run completes. Alternatively:
```bash
gh run list --branch feat/ci-pipeline --limit 1
gh run view <run-id>
```

Expected outcomes:
- **All three matrix entries pass.** Proceed to Task 5.
- **One or more matrix entries fail.** Proceed to Task 4a (troubleshooting).

- [ ] **Step 3: If all green, fetch the workflow URL for the badge**

```bash
gh api repos/varmabudharaju/chorus/actions/workflows --jq '.workflows[] | select(.name=="ci") | .html_url'
```

Expected: prints something like `https://github.com/varmabudharaju/chorus/actions/workflows/ci.yml`. Note this URL.

---

## Task 4a: Troubleshooting CI failures (only if Task 4 fails)

**Skip this task if all three matrix entries passed in Task 4.**

Common failure modes and their fixes:

### Failure: `pip install -e ".[dev]"` errors out

Likely cause: a transitive dep doesn't have a wheel for the matrix Python version.

- [ ] **Step 1: Reproduce locally**

```bash
# Locally, install the same Python version that failed and retry
pyenv install 3.12  # or via your version manager
pyenv shell 3.12
python -m venv .venv-3.12 && source .venv-3.12/bin/activate
pip install -e ".[dev]"
```

If it reproduces, look at the error: usually it's a torch wheel missing for a specific platform/Python combo. Either pin the torch version in `pyproject.toml` to a known-good range or drop the failing Python version from the matrix (with a comment explaining why).

### Failure: pytest fails on a specific Python version but passes elsewhere

Likely cause: code uses a Python 3.11+ feature (e.g., `Self` from `typing`, `tomllib`) but `requires-python = ">=3.10"` is set.

- [ ] **Step 1: Inspect the traceback** in the GitHub Actions log.
- [ ] **Step 2: Fix the offending code** to be 3.10-compatible, OR raise `requires-python` to `>=3.11` in `pyproject.toml` and drop 3.10 from the matrix. Choose the cheaper option — generally fixing the code is preferred unless multiple call sites are affected.
- [ ] **Step 3: Commit the fix** in a separate commit:

```bash
git add <changed-files>
git commit -m "fix: <description> for Python 3.x compatibility"
```

### Failure: ruff reports issues that didn't reproduce locally

Likely cause: local ruff is a different version than CI installed.

- [ ] **Step 1: Pin ruff** in `pyproject.toml` under `[project.optional-dependencies] dev`:

```python
# Change this line:
dev = [..., "ruff>=0.1.0", ...]
# To this:
dev = [..., "ruff>=0.4.0,<0.5.0", ...]
```

- [ ] **Step 2: Commit:**

```bash
git add pyproject.toml
git commit -m "chore: pin ruff to 0.4.x for reproducible CI"
```

### Failure: a network-dependent test times out

Likely cause: a test reaches out to HuggingFace or PyPI.

- [ ] **Step 1: Identify the test** from the pytest output.
- [ ] **Step 2: If it's a legitimate offline test, fix it** (mock the network call). If it requires network, mark it with `@pytest.mark.network` and add a pytest config to skip it by default:

```toml
# pyproject.toml [tool.pytest.ini_options]
markers = ["network: requires network access"]
```

And update the CI command:
```yaml
- name: Run tests with pytest
  run: pytest tests/ -v --tb=short -m "not network"
```

- [ ] **Step 3: Commit and rerun.**

### After any fix in Task 4a

- [ ] **Step N: Re-push and re-watch**

```bash
git push
gh run watch
```

When all three matrix entries pass, return to Task 5.

---

## Task 5: Add the CI badge to the README

**Files:**
- Modify: `README.md` (badge row near the top)

- [ ] **Step 1: Read the current badge block**

The first lines of README.md (after `# Chorus`) currently contain three badges (License, Python, PyPI). Open the file and confirm the structure.

```bash
head -10 README.md
```

Expected output includes lines like:
```
# Chorus

[![License: Apache 2.0](...)](...)
[![Python 3.10+](...)](...)
[![PyPI version](...)](...)
```

- [ ] **Step 2: Insert the CI badge as the first badge**

Use the Edit tool (or `sed`) to insert the badge line. The exact insertion replaces the `License:` badge line by prepending the CI badge:

**Old:**
```markdown
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
```

**New (two lines):**
```markdown
[![CI](https://github.com/varmabudharaju/chorus/actions/workflows/ci.yml/badge.svg)](https://github.com/varmabudharaju/chorus/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
```

- [ ] **Step 3: Verify the rendered badge URL works**

```bash
curl -s -o /dev/null -w "%{http_code}\n" "https://github.com/varmabudharaju/chorus/actions/workflows/ci.yml/badge.svg"
```

Expected: `200`. If it's `404`, the badge URL is wrong — double-check the path.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: add CI status badge to README

Surfaces build health on the repo landing page.
EOF
)"
```

- [ ] **Step 5: Push**

```bash
git push
```

Expected: a second CI run starts on the new commit. It should pass for the same reasons the first one did. Quick sanity-check with `gh run list --branch feat/ci-pipeline --limit 1` until it's green.

---

## Task 6: Open the pull request

**Files:** None (PR is on GitHub)

- [ ] **Step 1: Open the PR with `gh`**

```bash
gh pr create \
  --base master \
  --head feat/ci-pipeline \
  --title "[Phase 1.1] Add GitHub Actions CI pipeline" \
  --body "$(cat <<'EOF'
## Summary
- Adds `.github/workflows/ci.yml` with a Python 3.10/3.11/3.12 matrix that runs `ruff check` and `pytest` on every push and PR.
- Caches pip via pyproject.toml for ~50% faster cold installs.
- Cancels superseded workflow runs via concurrency group.
- Adds a CI status badge to README.

## Closes
- #<ISSUE-NUMBER>

## Test plan
- [x] CI is green on this branch across all three Python versions
- [x] Existing pytest suite passes (X tests)
- [x] Ruff is clean
- [x] Badge URL returns 200

## Notes for reviewer
- The eval-harness smoke step described in spec §4.5 is intentionally deferred to Feature 3 (it depends on `chorus eval`, which doesn't exist yet). When F3 lands, that step gets added in the F3 PR.
- No coverage upload, no pre-commit hooks, no auto-deploy. Each is a separate follow-up.
EOF
)"
```

Replace `<ISSUE-NUMBER>` with the real issue number and `X tests` with the actual count from Task 2 Step 3.

Expected: `gh` prints the PR URL.

- [ ] **Step 2: Verify the PR shows the green CI check**

```bash
gh pr view --json statusCheckRollup --jq '.statusCheckRollup'
```

Expected: array of three SUCCESS entries (one per matrix Python version).

If any are still pending, wait a minute and re-run. If any are FAILED, return to Task 4a.

---

## Task 7: Handle review (if requested)

**Files:** As requested by reviewer

If the reviewer (Varma) requests changes:

- [ ] **Step 1: Read the review comments**

```bash
gh pr view --comments
```

- [ ] **Step 2: Make the requested changes** in the worktree.

- [ ] **Step 3: Commit + push.** Each round of changes gets its own commit (no force-push) so the review trail is preserved:

```bash
git add <files>
git commit -m "<scope>: address review feedback — <specific change>"
git push
```

- [ ] **Step 4: Reply to the review threads** marking them resolved where appropriate.

When the PR is approved, Varma merges (the subagent does not merge its own PR).

---

## Post-merge cleanup (Varma)

After the PR merges:

```bash
cd ~/chorus
git checkout master
git pull origin master

# Remove the worktree
git worktree remove ~/chorus-worktrees/feat-ci-pipeline

# Delete the local branch
git branch -d feat/ci-pipeline

# Optionally delete the remote branch (gh does this on merge if "auto-delete head branches" is on)
git push origin --delete feat/ci-pipeline 2>/dev/null || true
```

Update the master plan (`docs/superpowers/plans/2026-05-19-phase-1-execution-plan.md`) to add F1 to the "Completed features" section with the PR number and merge date.

---

## Self-review checklist (run before requesting Varma's review)

Before opening the PR (Task 6), the implementing subagent runs this checklist:

- [ ] `.github/workflows/ci.yml` exists and is syntactically valid YAML.
- [ ] The workflow runs on `push` to master AND on `pull_request` against master.
- [ ] All three matrix entries (3.10, 3.11, 3.12) pass on this branch.
- [ ] `ruff check chorus tests benchmarks` passes locally.
- [ ] `pytest tests/ -v` passes locally with the same count as on master before this PR.
- [ ] README displays a CI badge as the first item in the badge row.
- [ ] The badge image URL returns HTTP 200.
- [ ] No unrelated files are modified (run `git diff master --stat` and confirm only `.github/workflows/ci.yml` and `README.md` appear).
- [ ] Commit messages follow the convention in the master plan §3 (conventional-commits style; no Co-Authored-By or AI-attribution trailer).
- [ ] The PR body references the issue with `Closes #<n>`.

If any check fails, fix before opening the PR.
