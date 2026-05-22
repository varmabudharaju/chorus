# Chorus Phase 1 — Session Handoff (2026-05-22)

> **If you're a new session picking this up:** read this whole doc first, then `docs/superpowers/plans/2026-05-19-phase-1-execution-plan.md` (the master plan). Everything else is reference.

## TL;DR — Quick Start

**Where we are:** Chorus v0.2.0 in progress. Phase 1 (Credibility & Honesty) is half done.
- Master is at the merge commit of PR #14.
- **F1 ✅ F2 ✅ F3 ✅** merged on master, each with its own feature PR + followup cleanup PR.
- **F4, F5, F6 remain** for v0.2.0 release.

**Next step:** Start F4 (benchmark suite). Same flow as F2/F3:
1. Branch `plan/f4-benchmark-suite` off master.
2. Update master plan §1 status table (mark F3 ✅, F4 planning) + add F3 entry to "Completed features".
3. Write `docs/superpowers/plans/<date>-feature-4-benchmark-suite.md` per the spec at `docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md` §4.4.
4. Open + merge planning PR.
5. Create worktree `~/chorus-worktrees/feat-benchmark-suite` on branch `feat/benchmark-suite`.
6. Dispatch implementer subagent (general-purpose, sonnet model).
7. Two-stage review (spec compliance → code quality).
8. Merge.

Don't reinvent — copy the prompt patterns from how F2 and F3 were dispatched.

## Phase 1 status

| # | Feature | Status | Feature PR | Cleanup PR | Issue |
|---|---|---|---|---|---|
| F1 | CI pipeline | ✅ merged | #3 | (in #10) | #2 |
| F2 | DP accountant | ✅ merged | #9 | #10 | #8 |
| F3 | Eval harness | ✅ merged | #13 | #14 | #12 |
| F4 | Benchmark suite | pending | — | — | — |
| F5 | README + honest-tradeoffs | pending | — | — | — |
| F6 | v0.2.0 release | pending | — | — | — |

Also: paid GPU benchmark run between F4 and F6 — out-of-session work, ~$80–160, ~half a day of the user's time on Modal/Lambda/RunPod, then a `chore/v0.2.0-benchmark-results` PR commits the JSON+markdown reports into `benchmarks/results/v0.2.0/`.

## Working conventions

### Worktree layout
- Main checkout: `~/chorus` (always on master).
- Per-feature worktrees: `~/chorus-worktrees/<branch-name>/` (created by `git worktree add` from main checkout).
- Worktrees are deleted via `git worktree remove --force <path>` after their PR merges. `--force` is needed because implementer subagents create a `.venv/` inside.

### Branch names
- `feat/<kebab>` — features
- `plan/<kebab>` — planning-only PRs (master plan update + per-feature plan doc)
- `chore/<kebab>` — cleanup, deps, tooling
- `release/v<X.Y.Z>` — release branches
- `docs/<kebab>` — docs-only

### Commit attribution (strict, no exceptions)
- Author: `varmabudharaju <sairam.vzf33@gmail.com>` (inherited automatically; do not change git config).
- **NO `Co-Authored-By: Claude` trailer.**
- **NO** "Generated with Claude Code" footer or robot emoji in PR/issue bodies.
- Convention is enshrined in `docs/superpowers/plans/2026-05-19-phase-1-execution-plan.md` §3. Subagent prompts must repeat this rule explicitly.

### Conventional commits
`feat(scope): ...`, `fix(scope): ...`, `docs(scope): ...`, `test(scope): ...`, `refactor(scope): ...`, `chore(scope): ...`, `ci: ...`, `perf(scope): ...`.

### PR + issue templates
See master plan §3. Don't add AI-attribution footers.

### Subagent-driven flow (per feature)
1. **Planning PR first** — adds detailed plan to `docs/superpowers/plans/`. User merges.
2. **Implementer subagent** in the feature worktree, given full plan + scene-setting context. Use `general-purpose` agent type, `sonnet` model (cheap enough, capable enough for the work in Phase 1).
3. **Spec compliance reviewer** subagent — independent verification that requirements met, no scope creep. Read the actual code; don't trust the implementer's report.
4. **Code quality reviewer** subagent — independent quality review. **This is where the real bugs surface.** Both F2 and F3 had blockers caught here that spec compliance missed.
5. Fix iterations (same subagent type, smaller scope) until both reviews ✅.
6. Merge the PR. Cleanup worktree + branch.

### The two-review pattern actually works
- F2: code quality reviewer caught 4 blockers (eager-import regression, `dp_applied` gating, missing 403 mapping, budget-burn on duplicate). Spec compliance missed all of them.
- F3: code quality reviewer caught 2 blockers (Dirichlet data loss, markdown crash on string metrics) plus 3 non-blockers.
- Lesson: don't skip the code quality review even if spec compliance is ✅.

### Cleanup PR pattern
After each feature merges, non-blocker observations from the code reviewer become a single batched chore PR (e.g., #10 for F1+F2, #14 for F3). One commit per fix. Don't proliferate small PRs.

### Network-gated tests
HF Hub downloads use `@pytest.mark.network`. The marker is declared in `pyproject.toml` `[tool.pytest.ini_options].markers`. CI runs `pytest -m "not network"`. Every test that calls `runner.run()` or downloads a model must have the decorator.

## Concrete lessons from F1–F3 (gotchas for F4+)

### From F1 (CI pipeline)
- Pre-existing ruff issues blocked clean CI on day 1 — the implementer fixed them in a separate preceding commit rather than filing as follow-up. The plan said "file a followup", but the practical reality was: 35 ruff violations make CI red forever. Accept this deviation when it surfaces.
- `concurrency.group` must be `${{ github.workflow }}-${{ github.head_ref || github.ref }}` (PR-scoped cancellation only, not master pushes). Already in master.

### From F2 (DP accountant)
- **`chorus/privacy/__init__.py` cannot eager-import `PrivacyAccountant`** — the accountant requires `dp-accounting` or `opacus`, gated behind the `[privacy]` extra. If `__init__.py` imports it unconditionally, `import chorus` breaks for users without the extra. Always wrap in `try: ... except ImportError:`.
- **Privacy budgets advance per-submission, NOT per-DP-application.** Don't gate `accountant.step()` on `dp_applied`; gate it on `accountant is not None`.
- **`accountant.step()` must run AFTER `save_delta` succeeds**, not before. If `save_delta` raises `DuplicateClientError` (409), we don't want to burn budget on a rejected delta.
- **Client SDK must map server 403 budget-exhausted → `PrivacyBudgetExhaustedError`**, not generic `ChorusError`.
- `_ensure_accountant` uses double-checked locking via `_accountant_ensure_lock` — cache hits lock-free, create-path serialized. Don't break this when extending the function.

### From F3 (eval harness)
- **`partition_non_iid_dirichlet` must explicitly set `cuts[-1] = len(idxs)`** after `np.cumsum() * len(idxs)` truncation — otherwise ~26–72% of seeds drop trailing data silently.
- **`to_markdown_string` must handle non-numeric metric values** (e.g., `{"note": "no_eval_data"}`). Use `isinstance(v, (int, float))` check before applying `:.4f`.
- **Float32 SVD reconstruction error tolerance is ~1e-4, not 1e-5.** Don't tighten the threshold.
- **PEFT state_dict keys may need `base_model.model.` prefix** when loading aggregated weights. Check both `k in peft_state` and `f"base_model.model.{k}" in peft_state`. Warn if zero keys match.
- **`_train_one_client` allocates a full model** — for any non-tiny model, do `del peft_model, model, optimizer; gc.collect()` before returning, or RSS balloons with `num_clients × model_size`.
- **`EvalRunner._train_clients_and_collect_deltas` currently collapses multi-round** — each round is independent (same seed, same data, same training); only the last round's deltas are kept. A real cross-round federation with state-carrying is deferred. `run()` warns when `num_rounds > 1`.
- **`from_dict` warns on unknown config keys** — catches typos like `stratigies: [...]` early. Don't silently filter.
- **`hf-internal-testing/tiny-random-LlamaForCausalLM`** is the canonical tiny test model. Has `q_proj` / `v_proj` layers PEFT can hook.

### What F4 should expect to encounter
- F4 builds on `chorus.eval` from F3. Most of F4 is YAML configs + a `run_all.py` driver + `verify_smoke_results.py` for CI.
- Per spec §4.4 the existing `benchmarks/benchmark.py` moves to `benchmarks/legacy/`. It's synthetic-data + timing only; not a real-data benchmark.
- F4 doesn't generate the actual published numbers — that's the paid GPU run between F4 merge and F6.
- F4's CI smoke step is already in place (from F3: `chorus eval --check-only --config benchmarks/configs/smoke.yaml`). F4 doesn't need to add another.
- Watch for: heterogeneous-rank configs, DP-on/off ablations, FedAvg-vs-FedExLoRA comparison configs. Each is its own YAML in `benchmarks/configs/`.

## Open issues

- **#6 SHA-pin GitHub Actions** — Phase 4 supply-chain hardening, deferred per master plan. Leave open.

(All other Phase 1 issues were closed when their PRs merged.)

## User preferences learned

- **Honest reviews always; no gaslighting.** When asked to opine on something they built, be candid — including pointing out marketing-vs-substance gaps.
- **Sole attribution on commits.** No AI co-author trailer, no AI-attribution footer in PR/issue bodies. This is a hard rule; documented in [[feedback-commit-attribution]] memory.
- **"Yeah you do it" / "lets do it"** = full delegation; merge PRs, handle the workflow end-to-end, don't pause for trivial decisions. Pause only for non-reversible or scope-changing decisions.
- **Continuous execution** within a session; don't ask "should I continue?" between tasks the user has already authorized.
- **Cleanup batches** — when code review surfaces multiple non-blockers, prefer one chore PR over multiple issue tickets.
- **GPU budget**: ~$200 hard cap for the v0.2.0 benchmark run; if extrapolation exceeds, scope down (drop a model or an ablation).
- **No `rm -rf` shortcuts** — got pushback on this early. Investigate, use unique paths, use `git worktree remove --force` only when the worktree's purpose is fulfilled.

## What's saved in auto-memory

- `user_profile.md` — Varma: enterprise-focused ML product builder.
- `project_roadmap.md` — 8 sub-project plan to productize Chorus, enterprise-first.
- `feedback_commit_attribution.md` — no AI co-author trailers; sole attribution rule.
- (This handoff doc is *not* in auto-memory by design — it's in the repo so it stays version-controlled and survives memory pruning.)

## Picking up tomorrow

If the user says "continue with F4" or equivalent:

```
1. cd ~/chorus && git pull origin master
2. git checkout -b plan/f4-benchmark-suite master
3. Edit docs/superpowers/plans/2026-05-19-phase-1-execution-plan.md
   - Mark F3 status: ✅ merged (PR #13)
   - Add F3 entry to Completed features (mirror the F1/F2 entries)
   - Mark F4 status: planning
4. Write docs/superpowers/plans/<today>-feature-4-benchmark-suite.md
   - Spec source: docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md §4.4
   - Use the F3 plan as a structural template
   - Tasks: open issue → write YAML configs → run_all.py → verify_smoke_results.py → tests → PR
5. Commit + push + open planning PR + merge
6. Create worktree at ~/chorus-worktrees/feat-benchmark-suite
7. Dispatch implementer (general-purpose, sonnet model)
8. Two-stage review (spec compliance + code quality)
9. Merge feature PR
10. File any non-blockers as a chore PR
```

Estimated session time for F4: ~45–60 min based on F1/F3 pacing. F5 (~45 min) and F6 (~30 min) can follow in the same session if you have the patience for it.

## Test count and CI baseline

As of master `6438c54`:
- 225 tests pass with `-m "not network"` (CI baseline)
- 1 test gated as `@pytest.mark.network` (`test_runner_run_produces_report_with_both_strategies`) — runs locally if HF cache is warm.
- Ruff: clean.
- Test count growth: 165 (v0.1.0) → 188 (F2) → 195 (F1/F2 cleanup) → 221 (F3) → 225 (F3 cleanup). F4 should add ~5–10 more (smoke verifier + config loader tests).
