# Feature 5: README rewrite + `docs/honest-tradeoffs.md` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Before starting, ensure you are in the worktree at `~/chorus-worktrees/feat-honest-tradeoffs` on branch `feat/honest-tradeoffs` (created via `git worktree add` from the parent checkout at `~/chorus`).

**Goal:** Close the marketing-vs-substance gap in the public README and ship `docs/honest-tradeoffs.md` — the explicit honest-disclosure document that the entire Phase 1 release is named for. Every claim in the README must either be backed by code-verifiable behavior or be qualified with the precise condition under which it holds. Every sharp edge in the system must appear in `docs/honest-tradeoffs.md` with enough specificity that a skeptical reader can verify it against the source.

**Architecture:** Two files change.
1. **README.md** — surgical edits at six anchor points (hero line, new Honest-tradeoffs section, Byzantine subsection rewrite, DP subsection augmentation, production-readiness paragraph, link block). The flow and major structure stay intact. No content is removed without replacement.
2. **docs/honest-tradeoffs.md** — new file, seven sections per spec §4.6. Each section: one-paragraph statement of the sharp edge, code citations (`file:line` or `function-name`) proving the claim, what's safe to do with the system today, what would fix it (with a roadmap pointer when applicable).

**Tone:** Same voice as the spec and the handoff doc. Direct, concrete, no hedging that softens the truth. No marketing prose. The user (see `[[user-profile]]` memory) is explicit: "Honest reviews always; no gaslighting." This document is the public face of that principle.

**Spec section:** `docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md` §4.6
**Master plan:** `docs/superpowers/plans/2026-05-19-phase-1-execution-plan.md` F5 row

---

## Hard-won context from F1–F4 to respect

- **No `Co-Authored-By: Claude` trailers. No "Generated with Claude Code" footers. No robot emoji.** Sole attribution. See master plan §3.
- **Conventional commits:** `docs(scope):`, `feat(scope):`, `fix(scope):`, etc. F5 is mostly `docs(readme):` and `docs(tradeoffs):` commits.
- **Verify every claim against the source.** A README that says "single-process server" must be true today; a tradeoff doc that says "rate limiter is in-memory" must point at the file. Don't write what *should* be true; write what *is* true and (separately) what's planned.
- **Roadmap references:** Phase 2/3/4 items live in `docs/superpowers/specs/2026-05-19-chorus-roadmap.md` and the master execution plan §1 "Non-goals". When the tradeoff doc says "fix in Phase N", cite the doc.
- **Open issues that affect this PR's claims:**
  - **#19** — `fold_residuals` is not threaded through `EvalRunner`. The README hero-line qualification ("when residuals are folded into base weights every round") is *true for the server path* but *not yet exercised by the eval harness*. The exactness section must be honest about this — exactness holds in the *server's* aggregation loop, but the *benchmark numbers* will only confirm it once #19 closes.
  - **#20** — `dp_epsilon` is not applied inside `EvalRunner`. The DP section must say: the *server* applies DP and the accountant tracks composition; the *eval harness* does not yet apply DP, so DP-on numbers from the benchmark are pending issue #20.
- **No re-running the F4 GPU benchmark in this PR.** The numbers will eventually replace placeholder text in the README. Today, the README must use language that's honest about the absence: "Benchmark results pending the v0.2.0 GPU run" (or equivalent) rather than fabricated numbers.

---

## File Structure

- **Modify:** `README.md` — six surgical edits, no broad rewrites.
- **Create:** `docs/honest-tradeoffs.md` — seven sections per spec §4.6.
- **Optional create:** `tests/test_docs_links.py` — light integrity check that every internal Markdown link in README + the new doc resolves to an anchor or file that exists. (Worth the ~30 lines if you have time; skip if the implementer is rushed.)

---

## Pre-flight (Task 0)

```bash
cd ~/chorus
git checkout master && git pull origin master
git worktree add ../chorus-worktrees/feat-honest-tradeoffs -b feat/honest-tradeoffs
cd ../chorus-worktrees/feat-honest-tradeoffs
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,peft,privacy]"
```

All subsequent paths are relative to `~/chorus-worktrees/feat-honest-tradeoffs`.

---

## Task 1: Open GitHub issue

- [ ] **Step 1: Create the issue**

```bash
gh issue create \
  --title "[Phase 1.5] README rewrite + docs/honest-tradeoffs.md" \
  --label "phase-1" \
  --body "$(cat <<'EOF'
**Phase:** 1 (Credibility & Honesty)
**Spec:** [docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md](../blob/master/docs/superpowers/specs/2026-05-19-chorus-phase-1-credibility-design.md) §4.6
**Plan:** [docs/superpowers/plans/2026-05-22-feature-5-honest-tradeoffs.md](../blob/master/docs/superpowers/plans/2026-05-22-feature-5-honest-tradeoffs.md)

## Scope
Close the marketing-vs-substance gap in the public-facing README and ship `docs/honest-tradeoffs.md` (the explicit honest-disclosure document Phase 1 is named for). README gets six surgical edits at named anchors; new doc covers all seven sections in spec §4.6.

## Acceptance criteria
- [ ] README hero line qualifies "mathematically exact" with the residual-folding precondition and links to the new doc.
- [ ] README has an "Honest tradeoffs" section near the top with a one-line bullet per major caveat, each linking to the relevant section of the new doc.
- [ ] README Security Features → Byzantine entry is rewritten as "Sanity-check defenses against naive attackers" with what they catch / what they miss.
- [ ] README Security Features → DP entry mentions the accountant and the unbounded-loss risk if `--max-epsilon` is not set.
- [ ] README has a production-readiness paragraph below Configuration Examples that explicitly says \"alpha software, not multi-tenant production\".
- [ ] \`docs/honest-tradeoffs.md\` exists, covers all seven spec §4.6 sections (exactness, DP, Byzantine, multi-tenant, heterogeneous clients, production, roadmap), each section grounded in code citations.
- [ ] No unqualified claim remains in the README that should be qualified.
- [ ] All existing tests still pass.

## Out of scope
- Benchmark numbers (those land in the post-F4 paid GPU run + a follow-up PR).
- New features or behavior changes anywhere in \`chorus/\`.
- Logo, design, screenshots, or any visual asset work.
- Translations.
EOF
)"
```

Note the issue number.

---

## Task 2: Pre-write reconnaissance (no edits yet)

Before writing a single word, build an accurate mental model of the system as it exists today. The implementer should run these and keep the results handy for citations.

- [ ] **Step 1: Confirm "exactness" preconditions**

```bash
# Find where residuals are folded into base weights:
grep -rn "fold_residuals\|residuals" chorus/server/ chorus/server/weight_manager.py 2>&1 | head -20
# Find the FedExLoRA implementation and confirm it stores residuals:
grep -n "_residuals\|residual" chorus/server/aggregation.py | head -20
```

Record: which file:line implements the fold, and the precondition for it to run (which CLI flag / config field).

- [ ] **Step 2: Confirm DP plumbing**

```bash
grep -rn "apply_dp\|GaussianMechanism" chorus/ | head -20
grep -n "PrivacyAccountant\|max_epsilon" chorus/server/app.py chorus/client/ 2>&1 | head -30
```

Record: where DP is applied on the server path, where the accountant is consulted, and how the client SDK opts in (`max_epsilon` parameter). Note that the eval harness does NOT apply DP (issue #20).

- [ ] **Step 3: Confirm Byzantine defenses**

```bash
grep -rn "norm_bound\|outlier_threshold\|z_score\|byzantine" chorus/server/aggregation.py chorus/server/app.py | head -20
```

Record: which CLI flags exist (`--norm-bound`, `--outlier-threshold`), what the checks compute (L2 norm of the delta, z-score across the round's clients), and explicitly what they do NOT catch (e.g., adaptive attacks, gradient inversion, label-flipping at the *task* level, anything that stays under the norm bound).

- [ ] **Step 4: Confirm multi-tenant scope**

```bash
grep -rn "api_key\|api-key\|model_id" chorus/server/auth.py chorus/server/app.py 2>&1 | head -30
```

Record: how API keys are scoped. The spec calls out "global API keys, no per-model scoping" — verify this is still true on master.

- [ ] **Step 5: Confirm heterogeneous-client behavior**

```bash
grep -n "rank\|target_modules" chorus/server/aggregation.py chorus/simulate/runner.py | head -30
```

Record: what happens when client A submits a rank-4 LoRA and client B submits a rank-8 LoRA against the same model. Per F4 review: `FedAvg` raises `ValueError` for shape mismatches; `FedExLoRA` handles it. The README/tradeoff should call this out specifically: heterogeneous-rank works on FedExLoRA but fails loudly on FedAvg.

- [ ] **Step 6: Confirm production-readiness claims**

```bash
grep -n "rate_limit\|Limiter" chorus/server/app.py chorus/server/rate_limit.py 2>&1 | head -10
grep -rn "filesystem\|storage\|FileSystemStorage\|atomic" chorus/server/ | head -10
```

Record: rate limiter is in-memory (per-process); storage is local filesystem; single-process FastAPI server (no built-in horizontal scale).

Do not commit anything in this task. The artifact is a notes file (write it to a scratchpad in the worktree; do not check it in). The citations gathered here populate the tradeoffs doc and back the README qualifications.

---

## Task 3: Draft `docs/honest-tradeoffs.md`

**File:** `docs/honest-tradeoffs.md` (new)

The doc has seven sections in this order, matching spec §4.6. Each section follows the same template:

1. **One-paragraph statement** of the sharp edge, plain English, no hedging.
2. **What the code actually does today** — short, with `file:line` or `function-name` citations from Task 2.
3. **What's safe / unsafe to do given this** — concrete guidance.
4. **What would fix it / when** — link to the relevant Phase 2/3/4 roadmap item in the spec or master plan if applicable. If something is on a tracked issue, link it.

- [ ] **Step 1: Write the doc**

The skeleton below is a *starting point*, not the final text. The implementer must replace the bracketed placeholders with content that survives a skeptical read against the source.

```markdown
# Honest Tradeoffs

> Every sharp edge in Chorus today, with citations. If you find a claim in this
> document that doesn't match the source, file an issue — that's a credibility
> bug and we fix it on sight.

## Table of contents

1. [When "exact" applies](#exactness)
2. [Differential privacy](#differential-privacy)
3. [Byzantine robustness](#byzantine-robustness)
4. [Multi-tenant scope](#multi-tenant-scope)
5. [Heterogeneous clients](#heterogeneous-clients)
6. [Production readiness](#production-readiness)
7. [What's on the roadmap to fix each of these](#roadmap)

---

## Exactness <a id="exactness"></a>

Chorus's headline claim is "mathematically exact federated aggregation for LoRA."
The claim is **true under one precondition**: the aggregated residual must be folded
into the base model's weights between rounds. If you skip that step, you get the
same rank-r approximation FedAvg gives you — not exact.

**Today's behavior:**
- The server's aggregation loop computes the exact result `sum(w_i * B_i @ A_i)` and
  decomposes it into a rank-r `(B, A)` pair plus a residual matrix [cite
  `chorus/server/aggregation.py:<line>`].
- When `fold_residuals=True` (default on the server path), the residual is added
  into the base weights [cite `chorus/server/weight_manager.py:<function>`]. Pulling
  the model on the next round gives you the exact combined result.
- When `fold_residuals=False`, you get the rank-r approximation only.

**Safe to do:**
- Run with default settings. You get exactness.

**Unsafe / surprising:**
- Disabling `fold_residuals` and assuming you still have exact aggregation. You don't.
- The `chorus.eval` harness does **not** yet apply the fold (issue
  [#19](https://github.com/varmabudharaju/chorus/issues/19)). Eval-harness reports
  that compare `fedex-lora` vs `fedavg` measure the *per-round aggregation step*
  exactness, not the *across-round* exactness that real federated training would
  produce. The published v0.2.0 benchmark numbers will close this gap once #19
  lands and the GPU run completes.

**Cost:**
- Folding requires shipping the residual matrix with each pull. Bandwidth grows as
  `model_dim * rank * sizeof(fp32)` per LoRA layer per round — small for typical
  ranks (4–32) but worth knowing.

## Differential privacy <a id="differential-privacy"></a>

Chorus implements Gaussian-mechanism DP at the per-round level and tracks
composition with a stateful privacy accountant (RDP via Google's `dp-accounting`,
`opacus` fallback). The library will refuse submissions once a configured
`(epsilon, delta)` budget is exhausted — but only if you opt in by setting
`max_epsilon` on the client.

**Today's behavior:**
- Server applies Gaussian noise to incoming deltas after L2 clipping; the
  `PrivacyAccountant` increments its budget per submission [cite
  `chorus/server/app.py:<line>` and `chorus/privacy/accountant.py:<class>`].
- Accountants are persisted per (model_id, client_id) under
  `chorus_data/<model>/privacy/<client>.json` and restored on server restart.
- If `max_epsilon` is reached, the submission endpoint returns HTTP 403; the
  client SDK maps this to `PrivacyBudgetExhaustedError`.

**What the accountant does not model:**
- Privacy loss from the client's *local training step* (gradient updates against
  private data before noise is applied). The mechanism noises the **delta**
  submitted to the server, not the gradient.
- Attacks against the *base model* (e.g., model-inversion that doesn't depend on
  delta sensitivity).
- Auxiliary information leakage outside the protocol (logs, side channels).

**Safe to do:**
- Set `max_epsilon` on the client. The library will halt before you blow past it.

**Unsafe / surprising:**
- Running with `dp_epsilon` set but no `max_epsilon`. The server noises every
  round, but no upper bound is enforced — you accumulate privacy loss
  indefinitely.
- The `chorus.eval` harness does **not** yet apply DP (issue
  [#20](https://github.com/varmabudharaju/chorus/issues/20)). The eval-harness
  DP-ablation YAML currently produces identical numbers across `dp_epsilon`
  values; published numbers wait on #20.

## Byzantine robustness <a id="byzantine-robustness"></a>

Chorus's "Byzantine defenses" today are **sanity checks against naive attackers**.
They reject obvious garbage — a delta with absurd magnitude, or one whose norm is
several standard deviations from the round's median. They will not stop an
adaptive adversary.

**Today's behavior:**
- `--norm-bound <X>` rejects any delta whose L2 norm exceeds `X` [cite
  `chorus/server/aggregation.py:<function>`].
- `--outlier-threshold <Z>` computes the per-round z-score of delta norms and
  drops any client more than `Z` standard deviations from the round's median.

**What this catches:**
- Random-noise injection (will produce a huge-norm delta).
- A single client submitting a wildly larger update than peers.
- Trivial corruption / serialization errors.

**What this misses:**
- Coordinated attacks: N malicious clients, each staying just under the norm
  bound, conspiring on direction.
- Label-flipping at the *task* level. The delta's norm can be perfectly normal
  while the model is being poisoned semantically.
- Gradient-inversion attacks recovering training data from honest updates.
- Anything below the threshold.

**Safe to do:**
- Use these as defense-in-depth, not as a primary trust boundary.
- Combine with `dp_epsilon` (noise raises the bar for any attack that depends on
  precise direction).

**Unsafe / surprising:**
- Treating "Byzantine defenses on" as equivalent to "robust to malicious clients."
  It is not. The phrase "Byzantine" in the academic literature implies stronger
  guarantees (e.g., Krum, coordinate-wise median) than Chorus implements today.
  Real Byzantine-robust aggregation is on the Phase 2 roadmap.

## Multi-tenant scope <a id="multi-tenant-scope"></a>

API keys are **global** in v0.2.0. Any key can read or write any model on the
server.

**Today's behavior:**
- `--api-key` flag(s) configure a flat list of acceptable bearer tokens [cite
  `chorus/server/auth.py:<line>`]. There is no per-model scoping, no role
  separation, no per-client API key.
- A compromised key gives the attacker write access to *every* model the server
  hosts.

**Safe to do:**
- Run one Chorus server per trust boundary (one model, or one team).
- Rotate keys with `--api-key` (multi-key support lets you grace-period rotation).

**Unsafe / surprising:**
- Hosting multiple unrelated federations on the same server with different API
  keys assuming the keys segregate them. They don't.

**Planned fix:** Per-(model, client) API keys and scoped read/write permissions —
on the Phase 4 roadmap [cite spec / roadmap doc].

## Heterogeneous clients <a id="heterogeneous-clients"></a>

Clients can submit LoRA adapters at different ranks targeting different modules.
`FedExLoRA` handles rank mismatch correctly. `FedAvg` **does not** — and will
fail loudly if you try.

**Today's behavior:**
- `FedExLoRA.aggregate()` accepts heterogeneous-shape deltas; it computes the
  exact weighted sum at full rank before decomposing back to the server's target
  rank [cite `chorus/server/aggregation.py:<line>`].
- `FedAvg.aggregate()` checks for uniform tensor shapes and raises
  `ValueError("FedAvg requires uniform tensor shapes. Use fedex-lora for
  heterogeneous LoRA ranks.")` [cite `chorus/server/aggregation.py:<line>`].
- Target-module mismatch (client A trains `q_proj`, client B trains `q_proj,k_proj`)
  is **silently tolerated**: keys present in some deltas but not others get
  averaged over the clients that have them, with no warning.

**Safe to do:**
- Mix LoRA ranks across clients when using `fedex-lora`. The library handles it.

**Unsafe / surprising:**
- Configuring `fedavg` with heterogeneous ranks. The server will return a 500
  the moment two mismatched-shape deltas land.
- Assuming target-module mismatch will warn. It won't; you have to inspect
  pulled state-dict keys yourself.

## Production readiness <a id="production-readiness"></a>

Chorus is **alpha software** in v0.2.0. It is suitable for research, small
internal federations, and reproducing benchmark results. It is **not** hardened
for multi-tenant production deployments.

**What this means concretely:**

- **Single-process FastAPI server.** No built-in horizontal scaling. Aggregation
  runs in the server process; restart loses any in-flight round state that
  hasn't been persisted to disk.
- **In-memory rate limiter.** Per-IP throttles reset on restart and don't span
  replicas. Front-load with nginx/Caddy/Cloudflare rate limiting for any real
  exposure.
- **Filesystem storage.** Aggregated weights and accountant state live in
  `chorus_data/` on local disk. No S3, Postgres, or other shared-storage
  backends.
- **HTTP, not HTTPS.** Terminate TLS at a reverse proxy.
- **No audit log.** Submissions are logged via the standard server log, but
  there's no append-only audit trail with cryptographic integrity.
- **No multi-tenant scoping.** See [§4](#multi-tenant-scope).

**Safe deployments:**
- Single-team internal federation behind a TLS-terminating reverse proxy on a
  trusted network.
- Reproducible research / benchmarking on a single machine.
- Local development and CI.

**Unsafe deployments:**
- Public-facing service accepting submissions from untrusted clients.
- Multi-customer SaaS without per-tenant isolation.
- Anything with HIPAA / SOC 2 / PCI scope as of v0.2.0.

## Roadmap to fix each of these <a id="roadmap"></a>

| Sharp edge | Fix lands in | Tracked by |
|---|---|---|
| Eval-harness `fold_residuals` threading | Pre-v0.2.0 GPU run | [#19](https://github.com/varmabudharaju/chorus/issues/19) |
| Eval-harness `dp_epsilon` threading | Pre-v0.2.0 GPU run | [#20](https://github.com/varmabudharaju/chorus/issues/20) |
| Real Byzantine-robust aggregation (Krum / median) | Phase 2 | spec §3 non-goals → Phase 2 roadmap |
| Per-(model, client) API keys & scoped permissions | Phase 4 | spec §3 non-goals → Phase 4 roadmap |
| Target-module-mismatch warnings | Phase 2 polish | not yet filed |
| Distributed server / shared storage | Phase 3 | spec §3 non-goals → Phase 3 roadmap |
| Audit log with cryptographic integrity | Phase 4 (compliance) | spec §3 non-goals → Phase 4 roadmap |
| Multi-process / horizontal scale | Phase 3 | spec §3 non-goals → Phase 3 roadmap |
```

(Replace every `<line>`, `<function>`, and `<class>` placeholder with the exact citation gathered in Task 2.)

- [ ] **Step 2: Commit**

```bash
git add docs/honest-tradeoffs.md
git commit -m "docs(tradeoffs): add docs/honest-tradeoffs.md covering all seven spec sections"
```

---

## Task 4: README hero-line qualification

**File:** `README.md`

- [ ] **Step 1: Find the current hero line**

```bash
grep -n "Mathematically exact" README.md
```

There are at least two occurrences (the marketing line near the top and a section header further down). The spec §4.6 only requires qualifying the *first* one (the headline). The later occurrence is a section explaining how the algorithm works — leave that one alone.

- [ ] **Step 2: Apply the surgical edit**

Replace:
```
**Federated LoRA fine-tuning with mathematically exact aggregation.**
```

With:
```
**Federated LoRA fine-tuning with mathematically exact aggregation — when residuals are folded into base weights every round ([details](docs/honest-tradeoffs.md#exactness)).**
```

The em-dash + parenthetical link keeps the line scannable but makes the precondition unmissable.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): qualify the mathematically-exact hero line with residual-folding precondition"
```

---

## Task 5: README "Honest tradeoffs" section

**File:** `README.md`

- [ ] **Step 1: Choose the insertion point**

Insert immediately after the "How It Works" diagram block but before the "Installation" section. This places it high enough that a casual reader sees it, but below the one-liner architectural overview that grounds the reader in what the system *is*.

- [ ] **Step 2: Add the section**

```markdown
## Honest tradeoffs

Chorus's claims hold under conditions that aren't always obvious from a quickstart.
Before relying on any of them, read [docs/honest-tradeoffs.md](docs/honest-tradeoffs.md).
Highlights:

- **"Mathematically exact" aggregation** holds when residuals are folded into base weights every round — the default server path, but not yet the eval harness ([#19](https://github.com/varmabudharaju/chorus/issues/19)). [More](docs/honest-tradeoffs.md#exactness)
- **Differential privacy** is per-submission Gaussian noise with a stateful accountant — but only if you set `max_epsilon` on the client. Without it, privacy loss accumulates unbounded. [More](docs/honest-tradeoffs.md#differential-privacy)
- **"Byzantine defenses"** are sanity checks against naive attackers (norm bound + z-score outlier). They will not stop an adaptive adversary. Real Byzantine-robust aggregation is on the Phase 2 roadmap. [More](docs/honest-tradeoffs.md#byzantine-robustness)
- **API keys are global.** No per-model scoping in v0.2.0. Run one server per trust boundary. [More](docs/honest-tradeoffs.md#multi-tenant-scope)
- **Heterogeneous-rank clients** work on `fedex-lora` and crash on `fedavg`. [More](docs/honest-tradeoffs.md#heterogeneous-clients)
- **Alpha software:** single-process server, in-memory rate limiter, filesystem storage, HTTP. Not hardened for multi-tenant production. [More](docs/honest-tradeoffs.md#production-readiness)
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): add Honest tradeoffs section linking to docs/honest-tradeoffs.md"
```

---

## Task 6: Security Features section — Byzantine + DP rewrites

**File:** `README.md`

The current Security Features bullets read as marketing. The spec wants them rewritten to be honest about what they catch and what they don't.

- [ ] **Step 1: Locate the section**

```bash
grep -n "Security Features" README.md
```

- [ ] **Step 2: Replace the Byzantine bullet**

Find:
```
- **Byzantine defenses** — L2 norm bounding (`--norm-bound`) and z-score outlier detection (`--outlier-threshold`) reject malicious or corrupted deltas
```

Replace with:
```
- **Sanity-check defenses against naive attackers** — L2 norm bounding (`--norm-bound`) and z-score outlier detection (`--outlier-threshold`) reject deltas with absurd magnitude or those several standard deviations from the round's median. These catch random-noise injection and trivial corruption; they **do not** stop coordinated attackers staying under the bound, label-flipping at the task level, or gradient-inversion attacks ([details](docs/honest-tradeoffs.md#byzantine-robustness)). Real Byzantine-robust aggregation (Krum, coordinate-wise median) is on the Phase 2 roadmap.
```

- [ ] **Step 3: Replace the DP bullet**

Find:
```
- **Differential privacy** — Gaussian DP with global L2 clipping at both client and server level
```

Replace with:
```
- **Differential privacy** — Per-submission Gaussian DP with global L2 clipping, plus a stateful `PrivacyAccountant` (RDP composition via Google's `dp-accounting`, `opacus` fallback). Set `max_epsilon` on the client to halt before you exceed your budget; without it, privacy loss accumulates unbounded across rounds. The eval harness does not yet apply DP ([#20](https://github.com/varmabudharaju/chorus/issues/20)); the *server* path does. [Details](docs/honest-tradeoffs.md#differential-privacy).
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): rewrite Byzantine + DP bullets to be honest about what they catch"
```

---

## Task 7: Production-readiness paragraph

**File:** `README.md`

- [ ] **Step 1: Locate the Configuration Examples section**

```bash
grep -n "Configuration Examples" README.md
```

- [ ] **Step 2: Insert a production-readiness paragraph below it**

Find the end of the Configuration Examples section (just before the next `##` heading) and add:

```markdown
### Production readiness

Chorus is **alpha software** in v0.2.0. Suitable for research, small internal federations, and benchmarks; not hardened for multi-tenant production. Single-process FastAPI server, in-memory rate limiter, filesystem storage, HTTP (terminate TLS at a reverse proxy). API keys are global — run one server per trust boundary. See [docs/honest-tradeoffs.md#production-readiness](docs/honest-tradeoffs.md#production-readiness) for the full list of caveats and what's on the roadmap to fix each.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): add production-readiness paragraph below Configuration Examples"
```

---

## Task 8: (Optional) Markdown link integrity test

If time permits, lock in the cross-references so a future docs PR can't silently break them.

**Files:**
- Create: `tests/test_docs_links.py`

- [ ] **Step 1: Write the test**

```python
"""Light integrity check: every internal Markdown link in our top-level docs
resolves to a file or anchor that actually exists.

Catches the most common docs regression: an anchor or filename gets renamed and
all the references quietly stop working. This is not a full Markdown link
checker (no external URLs, no relative `../` traversal beyond what we use). It
covers exactly the link patterns the F5 PR introduces.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files we check. Anything else is out of scope for this test.
TARGET_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "honest-tradeoffs.md",
]

# Match [text](path[#anchor]) where path is a relative file path (no scheme,
# no protocol-relative, no fragment-only links — those are tested separately
# by ensuring the anchor exists in the same file).
LINK_RE = re.compile(r"\[(?:[^\]]+)\]\(([^)]+)\)")


def _collect_anchors(md_text: str) -> set[str]:
    """Return the set of anchor ids defined in a Markdown file.

    Recognizes both:
    - explicit `<a id="foo"></a>` blocks
    - GitHub-style anchors from `## Heading` (lowercased, spaces → dashes,
      punctuation stripped). This is approximate but good enough for our
      docs.
    """
    anchors: set[str] = set()
    anchors.update(re.findall(r'<a\s+id="([^"]+)"', md_text))
    for line in md_text.splitlines():
        m = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if not m:
            continue
        # GitHub anchor: lowercase, strip punctuation, spaces → dashes
        slug = m.group(1).lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"\s+", "-", slug).strip("-")
        anchors.add(slug)
    return anchors


def test_internal_links_resolve():
    for md_path in TARGET_FILES:
        text = md_path.read_text()
        for link in LINK_RE.findall(text):
            # Skip external links and mailtos
            if link.startswith(("http://", "https://", "mailto:", "#")):
                # Same-file anchor — verify it exists in this doc
                if link.startswith("#"):
                    anchors = _collect_anchors(text)
                    assert link.lstrip("#") in anchors, (
                        f"{md_path.name}: same-file anchor {link!r} not found. "
                        f"Available: {sorted(anchors)}"
                    )
                continue
            # Split path and anchor
            if "#" in link:
                path_part, anchor = link.split("#", 1)
            else:
                path_part, anchor = link, ""
            # Resolve relative to the markdown file's directory
            target = (md_path.parent / path_part).resolve()
            assert target.exists(), (
                f"{md_path.name}: link {link!r} → {target} does not exist"
            )
            if anchor:
                target_text = target.read_text()
                target_anchors = _collect_anchors(target_text)
                assert anchor in target_anchors, (
                    f"{md_path.name}: link {link!r} → anchor #{anchor} not "
                    f"found in {target.name}. Available: {sorted(target_anchors)}"
                )
```

- [ ] **Step 2: Run + commit**

```bash
pytest tests/test_docs_links.py -v
```

If any link fails, fix it (probably in README.md or honest-tradeoffs.md, not in the test). Common cause: anchor casing or punctuation mismatch.

```bash
git add tests/test_docs_links.py
git commit -m "test(docs): add Markdown internal-link integrity check"
```

---

## Task 9: Final verification + PR

- [ ] **Step 1: Full suite + ruff + docs link test**

```bash
ruff check chorus tests benchmarks
pytest -m "not network" -q
```

Expected: ruff clean. Test count delta: 243 → 244 if Task 8 was done (just the docs-link test). No regression in the existing 243.

- [ ] **Step 2: Manually render the README locally**

```bash
# Easiest: use a local Markdown previewer or just visual-inspect the diff.
git diff master -- README.md docs/honest-tradeoffs.md | head -200
```

Read it as a *first-time visitor*. Does it surface the right caveats without burying the headline value? If it reads as a self-deprecating ramble, dial back. If it reads as marketing, you're not done.

- [ ] **Step 3: Push**

```bash
git push -u origin feat/honest-tradeoffs
```

- [ ] **Step 4: Open PR**

```bash
gh pr create \
  --base master \
  --head feat/honest-tradeoffs \
  --title "[Phase 1.5] Honest README + docs/honest-tradeoffs.md" \
  --body "$(cat <<'EOF'
## Summary
- Qualifies the README's \"mathematically exact\" hero line with the residual-folding precondition.
- Adds a new \"Honest tradeoffs\" section near the top of README that one-lines each major caveat and links to the new doc.
- Rewrites the Security Features Byzantine + DP bullets to honestly state what they catch and what they miss.
- Adds a production-readiness paragraph below Configuration Examples explicitly labeling Chorus alpha software.
- Adds \`docs/honest-tradeoffs.md\` covering all seven spec §4.6 sections (exactness, DP, Byzantine, multi-tenant, heterogeneous clients, production, roadmap) with code citations.
- (If Task 8 was done:) adds a light Markdown internal-link integrity test.

## Closes
- #<ISSUE-NUMBER>

## Test plan
- [x] \`pytest -m \"not network\" -q\` green
- [x] Ruff clean
- [x] Every claim in the new doc verified against the source on master (Task 2 reconnaissance)
- [x] README renders correctly on GitHub (visual inspection)

## Notes for reviewer
- The doc honestly flags issues #19 (\`fold_residuals\` threading) and #20 (\`dp_epsilon\` threading) as gaps in the *eval harness*, not the server path. Once those issues close, the relevant lines in this doc should be updated to remove the gap callouts.
- No changes to \`chorus/\` source. This PR is documentation-only.
EOF
)"
```

- [ ] **Step 5: Watch CI**

```bash
gh pr checks --watch
```

---

## Self-review checklist

- [ ] Every claim in `docs/honest-tradeoffs.md` has a `file:line` or `function-name` citation or links to an issue / roadmap doc.
- [ ] No claim in the README is unqualified that should be qualified (you've re-read the README front-to-back as a first-time visitor).
- [ ] The hero line's precondition is unmissable.
- [ ] The "Honest tradeoffs" section is high enough on the page to be seen.
- [ ] Issues #19 and #20 are referenced where the eval harness is the gap.
- [ ] No `Co-Authored-By` trailer on any commit; no AI-attribution footer in PR/issue bodies.
- [ ] No new claim in the README about *benchmark numbers* — those wait on the post-F4 GPU run.
- [ ] Tone matches the spec/handoff: direct, concrete, no hedging that softens the truth, no marketing prose.
- [ ] Existing 243 tests still pass (244 with the optional docs-link test).

---

## Out-of-scope (do not creep)

- Benchmark numbers (waiting on the v0.2.0 GPU run).
- New features or behavior changes in `chorus/`.
- Threading `fold_residuals` or `dp_epsilon` through `EvalRunner` (issues #19/#20 — separate PRs).
- v0.2.0 release notes, CHANGELOG, PyPI publish — that's F6.
- Logo, design, screenshots, translations.
