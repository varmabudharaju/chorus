# Chorus Roadmap: From Demo to Defensible OSS

**Status:** Draft
**Date:** 2026-05-19
**Author:** Varma (varmabudharaju)
**Scope:** Top-level roadmap. Not an implementation spec. Each phase has (or will have) its own design doc.

---

## Context

Chorus today (v0.1.0, Feb 2026) is a clean, technically-correct federated LoRA aggregation framework with a real implementation of FedEx-LoRA, solid Python hygiene, and a 165-test suite. It has 3 stars, no users, one commit on master, and a README that overclaims in three specific places ("mathematically exact" without the residual-folding caveat, "differential privacy" without composition tracking, "Byzantine defenses" for what's really an outlier filter).

The goal of this roadmap is to close the marketing-vs-substance gap, then progressively harden Chorus into something a stranger would actually adopt for their own federated training.

This is an open-source project. The roadmap does not optimize for any single deployment scenario (SaaS, consortium, single-org, research) — it optimizes for *general trust and usability*, so that any of those users can fork and adapt without hitting a sharp edge.

## Why this order

A common failure mode for OSS projects is to harden production deployment before establishing that the algorithm itself works on real data and the claims match reality. This roadmap inverts that:

1. **Earn trust before adding features.** Credibility comes first because nothing else matters if a skeptical reader bounces in 60 seconds.
2. **Reduce friction before hardening.** Once claims are honest, the next adoption barrier is "can I get this running in 10 minutes?" — not "does it scale to 100 tenants?"
3. **Harden the algorithm before the deployment.** Secure aggregation, real Byzantine defenses, and rank-pinning are research-adjacent work that's hard to retrofit once users depend on current behavior. Better to land them while the user base is small.
4. **Production hardening last.** Most OSS users will run their own fork on their own infra. Multi-tenant isolation, Helm charts, and audit logs help only the small subset who host Chorus as a service. Wait until that demand is real.

## Phases

### Phase 1 — Credibility & Honesty (≈6–7 weeks)

**Goal:** Close the gap between what the README claims and what the code actually delivers. Make Chorus citable.

**Deliverables (high level):**
- `chorus eval` subcommand — reusable benchmark CLI any user can run on their own model/dataset.
- Paper-appendix-grade benchmark suite (multi-model, multi-dataset, multi-seed, ablations) running on top of `chorus eval`. Code lands in this phase; the full published numbers are generated in one paid GPU burst when the suite is ready.
- Real DP accountant (RDP-based, via `dp-accounting` or `opacus.accountants`), with per-client (ε, δ) budget tracking, an API endpoint, and CLI surfacing.
- GitHub Actions CI: matrix pytest + ruff + smoke-config `chorus eval`. Status badge in README.
- README rewrite: qualify "exact" claim, surface the residual-folding bandwidth tradeoff, rename "Byzantine defenses" to "sanity-check defenses" with explicit limits, link to the accountant from the DP section.
- `docs/honest-tradeoffs.md` — single page that enumerates every sharp edge.

**Unblocks:** Phase 2 (you can't onboard users until what they're onboarding to is credible) and Phase 3 (the accountant and benchmark are prerequisites for the secure-aggregation comparison).

**Detailed spec:** `2026-05-19-chorus-phase-1-credibility-design.md`

### Phase 2 — Developer Experience (≈3 weeks)

**Goal:** Make Chorus delightful to try. Convert curious readers into users.

**Sketch (to be re-brainstormed when Phase 1 ships):**
- `docker compose up` for a local server + N simulated clients with one command.
- Notebook tutorial that fine-tunes a small real model (e.g., Phi-3-mini) across 5 simulated hospitals on a real dataset, end-to-end, in <10 min on a free Colab GPU.
- Friendlier CLI errors (current ones are FastAPI HTTPException strings; replace with actionable messages).
- 3–5 minute video walkthrough of the notebook (hosted on YouTube, linked from README).
- "Migrate from Flower / PySyft / FedML" comparison page — short, honest, captures users from adjacent projects.

**Unblocks:** Phase 3 testing (more users = more bug reports on correctness). Validates whether the algorithm-level improvements in Phase 3 are actually wanted.

### Phase 3 — Correctness & Robustness (≈5 weeks)

**Goal:** Make the algorithm defensible under adversarial and messy conditions.

**Sketch (to be re-brainstormed when Phase 2 ships):**
- Cross-client schema pinning — server publishes a "model schema" (rank, target modules, dtype) at round start, rejects mismatched deltas with a clear error.
- Optional secure aggregation (Bonawitz-style mask exchange) so the server cannot see individual client deltas. Pluggable: not on by default. (This is a real research-adjacent body of work — may end up scoped down to "additive masking via pairwise PRG seeds" rather than full DDH-based.)
- Real Byzantine defenses — coordinate-wise median and Krum, in addition to current norm-bound + z-score. Documented honestly about what each catches and what it misses.
- "Failure-loud" mode for residuals — when base weights aren't configured, the server refuses to start in `--strict-exact` mode rather than silently degrading.

**Unblocks:** Phase 4 enterprise hardening (compliance/audit teams want to see that adversarial robustness is addressed).

### Phase 4 — Production Hardening (≈6 weeks)

**Goal:** Make Chorus deployable at scale by an organization that didn't build it.

**Sketch (to be re-brainstormed when Phase 3 ships):**
- Pluggable storage backend — abstract `StorageBackend` interface, ship Postgres + S3 implementation alongside the existing filesystem one. Removes the "single-process, single-machine" constraint.
- Multi-tenant API key scoping — keys scoped to (model_id, role) pairs; current global key behavior becomes a deprecated default.
- Audit log — every delta submission, aggregation, fold, and key event written to a tamper-evident append-only log (or external SIEM hook).
- Helm chart + Terraform module for AWS/GCP.
- OpenAPI spec + auto-generated client SDKs for JS/Go.

**Unblocks:** Phase 5+ (whatever the user base actually demands next — likely either a managed-service offering or vertical-specific compliance work).

## Sequencing notes

- Phases are mostly sequential, but a few items can run in parallel as fillers when the user has spare cycles: the README rewrite (Phase 1) can start day 1; the docker-compose work (Phase 2) can be done by anyone with a free weekend even during Phase 1.
- Each phase's design doc will be written *just before* that phase starts, not now. Detail decays with distance from execution.
- The benchmark suite from Phase 1 should be re-run at the end of every subsequent phase to confirm no regressions in algorithmic quality. This is one of the reasons it ships as a reusable CLI rather than a one-off script.

## Out of scope (for the foreseeable horizon)

- Cross-silo federation over WAN with real network conditions (latency simulation, partial-failure handling beyond crash recovery).
- Model architectures beyond decoder LLMs (vision, multimodal). LoRA exists in those domains but the aggregation math is the same; rather than building out, document that and accept community PRs.
- Cryptographic privacy beyond DP + optional secure aggregation. Homomorphic encryption, MPC, TEE-based approaches are out of scope.
- A managed-service / hosted Chorus offering. This roadmap is for the OSS project only.

## Success criteria (whole roadmap)

- A skeptical engineer reading the repo for the first time finishes with: "the claims match the code, there's evidence it works on real data, it's clear what tradeoffs I'm accepting, and there's a path from `git clone` to a working federation in under an hour."
- ≥10 unaffiliated users (not Varma, not friends-of-Varma) have run Chorus on real data and either contributed back, filed substantive issues, or starred with a non-trivial reason.
- Chorus is referenced in at least one third-party blog post, paper, or conference talk as a usable open-source FL framework.
