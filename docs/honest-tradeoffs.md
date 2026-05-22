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

<a id="exactness"></a>
## When "exact" applies

Chorus's headline claim is "mathematically exact federated aggregation for LoRA."
The claim is **true under one precondition**: the aggregated residual must be folded
into the base model's weights between rounds. If you skip that step, you get the
same rank-r approximation FedAvg gives you — not exact.

**What the code does today:**

- `FedExLoRA.aggregate()` computes `sum(w_i * B_i @ A_i)` and decomposes it via
  truncated SVD into a rank-r `(B_new, A_new)` pair, tracking the residual as
  `target - (B_new @ A_new)` (`chorus/server/aggregation.py:176`). This residual
  is the information that a single rank-r product cannot represent.
- After each aggregation, `_run_aggregation()` in `chorus/server/app.py:634` calls
  `fold_residuals_into_base()` when base weights are present. The fold adds the
  residual matrix directly into the base weights (`chorus/server/weight_manager.py:8`,
  `fold_residuals_into_base`). After the fold, clients who pull the base weights
  receive an exact representation of the combined update — no approximation loss
  accumulated across rounds.
- The fold runs automatically on the server path when base weights have been
  uploaded. It is not optional; you either uploaded base weights and get exactness,
  or you didn't and get the per-round rank-r approximation only.

**Safe to do:**

- Upload base weights to the server before the first round (`POST
  /models/{id}/base-weights`). Run with default settings. You get exactness.

**Unsafe / surprising:**

- Running without uploading base weights. The aggregation still runs and produces
  a valid rank-r adapter, but residuals accumulate in memory and are never folded.
  Clients pulling the adapter get the per-round approximation, not the exact result.
- The `chorus.eval` harness (`chorus/eval/runner.py`) does **not** yet thread
  `fold_residuals` through its training loop (issue
  [#19](https://github.com/varmabudharaju/chorus/issues/19)). Eval-harness
  comparisons of `fedex-lora` vs `fedavg` measure the per-round aggregation
  step only; the across-round exactness guarantee that real federated training
  produces is not exercised until #19 closes and the v0.2.0 GPU run completes.

**Bandwidth cost:**

Folding requires shipping the residual matrix alongside each round's pull. The
residual is a dense `(hidden_dim × hidden_dim)` matrix per LoRA layer — small
for typical ranks (4–32) but grows with model width. Worth sizing before
deploying over narrow links.

---

<a id="differential-privacy"></a>
## Differential privacy

Chorus implements Gaussian-mechanism DP at the per-submission level and tracks
composition with a stateful privacy accountant (RDP via Google's `dp-accounting`,
`opacus` fallback). The server will **refuse further submissions** once a configured
`(epsilon, delta)` budget is exhausted — but only if you opt in to the accountant
by configuring `--accountant-target-epsilon` on the server. Without it, privacy
loss accumulates unbounded.

**What the code does today:**

- `submit_delta()` in `chorus/server/app.py:379` calls `apply_dp()` from
  `chorus/privacy/mechanism.py:74` if `dp_epsilon` is set on the server. This
  clips the incoming delta to `dp_max_norm` L2 norm, then adds calibrated
  Gaussian noise (`sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon`).
- `PrivacyAccountant` in `chorus/privacy/accountant.py:32` tracks RDP composition
  per `(model_id, client_id)` pair. Accountant state is persisted to
  `chorus_data/<model>/privacy/<client_id>.json` and restored on server restart.
- When the accountant's `is_exhausted()` returns `True`, `submit_delta()` returns
  HTTP 403 (`chorus/server/app.py:358`). The client SDK maps this to
  `PrivacyBudgetExhaustedError`.

**What the accountant does not model:**

- Privacy loss from the client's local training step before the delta is submitted.
  The Gaussian mechanism noises the **submitted delta**, not the per-example
  gradient. Clients training without local DP contribute a sensitivity that depends
  on their raw data.
- Attacks against the base model itself (model-inversion or membership inference
  that doesn't depend on submitted delta sensitivity).
- Side-channel leakage: logs, timing, network metadata.

**Safe to do:**

- Configure both `--dp-epsilon` (server-side noise) and
  `--accountant-target-epsilon` / `--accountant-noise-multiplier` (accountant
  enforcement). Together they noise every submission and halt before the budget
  limit.
- Verify the accountant is active by hitting `GET
  /models/{id}/clients/{client_id}/privacy` and checking `epsilon_consumed`.

**Unsafe / surprising:**

- Running with `--dp-epsilon` set but no `--accountant-target-epsilon`. The server
  noises every round but no upper bound is enforced. Privacy loss accumulates
  indefinitely across rounds with no halt.
- The `chorus.eval` harness does **not** yet apply DP to client deltas (issue
  [#20](https://github.com/varmabudharaju/chorus/issues/20)). DP-ablation YAMLs
  in the benchmark suite currently produce identical numbers across `dp_epsilon`
  values because `EvalRunner._train_clients_and_collect_deltas()` submits raw
  deltas without noising. Published DP-on vs DP-off numbers wait on #20.

---

<a id="byzantine-robustness"></a>
## Byzantine robustness

Chorus's "Byzantine defenses" today are **sanity checks against naive attackers**.
They reject obvious garbage — a delta with an absurd magnitude or one whose norm
is several standard deviations from the round's median. They will not stop an
adaptive adversary who knows the defense parameters.

**What the code does today:**

- `norm_bound_deltas()` in `chorus/server/aggregation.py:218` clips every delta
  whose global L2 norm exceeds `--norm-bound`. Any client submitting a wildly
  large update gets its delta scaled down to the bound before aggregation.
- `filter_outlier_deltas()` in `chorus/server/aggregation.py:279` computes the
  per-round z-score of delta norms and drops any client more than
  `--outlier-threshold` standard deviations from the round's mean norm. Both
  defenses run in `_run_aggregation()` at `chorus/server/app.py:594`.

**What this catches:**

- Random-noise injection (produces a huge-norm delta that exceeds the bound).
- A single rogue client submitting an update far larger than peers.
- Trivial corruption or serialization errors producing garbage tensors.

**What this misses:**

- Coordinated attacks: multiple malicious clients each staying just under the norm
  bound and conspiring on update direction. The z-score check computes outliers
  relative to the round's set of clients — if most clients are attackers, honest
  clients become the outliers.
- Label-flipping at the task level. A delta's L2 norm can be perfectly normal
  while the model is being steered toward a backdoor target. Norm checks cannot
  detect semantic poisoning.
- Gradient-inversion attacks that recover training data from honest client
  updates. These are passive and leave no norm footprint.
- Any attack that stays under the threshold. The defenses bound magnitude, not
  direction.

**Safe to do:**

- Use these as defense-in-depth, not as the primary trust boundary. Combine with
  `--dp-epsilon` (noise raises the bar for attacks that depend on precise update
  direction) and client authentication.

**Unsafe / surprising:**

- Treating "Byzantine defenses on" as equivalent to "robust to malicious clients."
  It is not. The academic literature uses "Byzantine robustness" to mean
  guarantees that survive up to `f` arbitrary malicious clients even with full
  knowledge of the algorithm (e.g., Krum, coordinate-wise median). Chorus's norm
  and z-score checks are weaker than that bar. Real Byzantine-robust aggregation
  is on the Phase 2 roadmap.

---

<a id="multi-tenant-scope"></a>
## Multi-tenant scope

API keys are **global** in v0.2.0. Any valid key can read or write any model on
the server.

**What the code does today:**

- `configure()` in `chorus/server/app.py:133` takes a flat `api_keys: list[str]`
  that populates `state.api_keys: set[str]`. The `require_auth` dependency
  (`chorus/server/app.py:175`) checks every request against this flat set — there
  is no per-model, per-client, or role-based scoping.
- A compromised key gives the bearer write access to every model the server hosts,
  including the ability to submit malicious deltas to models belonging to other
  teams.

**Safe to do:**

- Run one Chorus server per trust boundary: one model per server, or one team per
  server. This is the supported multi-tenancy model today.
- Use the multi-key support (`--api-key` can be repeated) for zero-downtime key
  rotation. Retire old keys once all clients have updated.

**Unsafe / surprising:**

- Hosting multiple unrelated federations on the same server instance with different
  API keys, assuming the keys segregate write access. They don't. Any key that can
  submit a delta to model A can also submit to model B.

**Planned fix:** Per-(model, client) API keys and scoped read/write permissions
are on the Phase 4 roadmap
(`docs/superpowers/specs/2026-05-19-chorus-roadmap.md`).

---

<a id="heterogeneous-clients"></a>
## Heterogeneous clients

Clients can submit LoRA adapters at different ranks targeting different modules.
`FedExLoRA` handles rank mismatch correctly. `FedAvg` does not — it fails loudly
if shapes don't match. Target-module mismatch is silently tolerated by both
strategies.

**What the code does today:**

- `FedExLoRA.aggregate()` (`chorus/server/aggregation.py:118`) collects A and B
  matrices across all clients that have each layer, then computes the exact
  weighted average at full rank before decomposing. The output rank is
  `max(client ranks)` by default (`chorus/server/aggregation.py:154`), so a
  mix of rank-4 and rank-8 clients produces a rank-8 aggregated adapter with no
  information loss from the rank mismatch.
- `FedAvg.aggregate()` (`chorus/server/aggregation.py:46`) checks that all clients
  have identical tensor shapes and raises `ValueError: "FedAvg requires uniform
  tensor shapes. Use fedex-lora for heterogeneous LoRA ranks."` if any mismatch
  is found (`chorus/server/aggregation.py:58`). This error surfaces as an HTTP 500
  to clients.
- Target-module mismatch (client A trains `q_proj`, client B trains
  `q_proj, k_proj`) is **silently tolerated** in `FedExLoRA`: layers present in
  some deltas but not others are averaged over the subset of clients that have
  them (`chorus/server/aggregation.py:144`). No warning is logged.

**Safe to do:**

- Mix LoRA ranks across clients when using `fedex-lora`. The library handles it
  and the output rank is deterministic (max of contributing ranks).

**Unsafe / surprising:**

- Configuring `fedavg` with heterogeneous ranks. The server returns HTTP 500 the
  moment two mismatched-shape deltas land in the same round. Use `fedex-lora`
  if your clients have different ranks.
- Assuming target-module mismatch will warn you. It won't. If you want to verify
  which modules were aggregated, inspect the keys in the pulled state dict
  yourself.

---

<a id="production-readiness"></a>
## Production readiness

Chorus is **alpha software** in v0.2.0. It is suitable for research, small
internal federations, and reproducing benchmark results. It is **not** hardened
for multi-tenant production deployments.

**What this means concretely:**

- **Single-process FastAPI server.** Aggregation runs in the server process via
  `asyncio.to_thread` (`chorus/server/app.py:561`). No built-in horizontal
  scaling. Restarting the server loses any in-flight round state not yet
  persisted to disk (rounds stuck in `AGGREGATING` are recovered on next
  startup, but round state mid-aggregation is not).
- **In-memory rate limiter.** `RateLimiter` in `chorus/server/app.py:31` is a
  plain dict of per-IP timestamps. It resets on every restart and does not span
  replicas. For any real external exposure, front-load with nginx/Caddy/Cloudflare
  rate limiting.
- **Filesystem storage.** `DeltaStorage` in `chorus/server/storage.py:37` writes
  aggregated weights, accountant state, and round state to `chorus_data/` on
  local disk. No S3, Postgres, or shared-storage backends.
- **HTTP, not HTTPS.** The server binds plain HTTP. Terminate TLS at a reverse
  proxy (nginx, Caddy, etc.) before exposing to any network you don't fully
  trust.
- **No audit log.** Submissions are logged via the standard Python logger. There
  is no append-only audit trail with cryptographic integrity.
- **Global API keys.** See [Multi-tenant scope](#multi-tenant-scope).

**Safe deployments:**

- Single-team internal federation behind a TLS-terminating reverse proxy on a
  trusted network.
- Reproducible research and benchmarking on a single machine.
- Local development and CI.

**Unsafe deployments:**

- Public-facing service accepting submissions from untrusted clients.
- Multi-customer SaaS without per-tenant isolation.
- Any deployment with HIPAA, SOC 2, or PCI scope as of v0.2.0.

---

<a id="roadmap"></a>
## What's on the roadmap to fix each of these

| Sharp edge | Fix lands in | Tracked by |
|---|---|---|
| Eval-harness `fold_residuals` threading | Pre-v0.2.0 GPU run | [#19](https://github.com/varmabudharaju/chorus/issues/19) |
| Eval-harness `dp_epsilon` threading | Pre-v0.2.0 GPU run | [#20](https://github.com/varmabudharaju/chorus/issues/20) |
| Target-module-mismatch warning | Phase 2 polish | not yet filed |
| Real Byzantine-robust aggregation (Krum, coordinate-wise median) | Phase 2 | `docs/superpowers/specs/2026-05-19-chorus-roadmap.md` |
| Per-(model, client) API keys and scoped permissions | Phase 4 | `docs/superpowers/specs/2026-05-19-chorus-roadmap.md` |
| Distributed server / shared storage backends | Phase 3 | `docs/superpowers/specs/2026-05-19-chorus-roadmap.md` |
| Multi-process / horizontal scale | Phase 3 | `docs/superpowers/specs/2026-05-19-chorus-roadmap.md` |
| Audit log with cryptographic integrity | Phase 4 (compliance) | `docs/superpowers/specs/2026-05-19-chorus-roadmap.md` |
