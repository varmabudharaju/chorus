# Chorus Phase 1: Credibility & Honesty — Design

**Status:** Draft
**Date:** 2026-05-19
**Author:** Varma (varmabudharaju)
**Roadmap context:** `2026-05-19-chorus-roadmap.md`
**Target horizon:** ≈6–7 weeks part-time
**Branch:** `roadmap/credibility-phase-1` (this doc) → per-deliverable feature branches
**Release target:** Chorus v0.2.0

---

## 1. Problem

Chorus v0.1.0 makes three claims its code does not fully back:

1. **"Mathematically exact" aggregation** — true only when residuals are folded into base weights every round. The default `pull_latest` path delivers the lossy rank-r approximation. README doesn't surface this dependency.
2. **"Differential privacy"** — the Gaussian mechanism is implemented per-round, but no privacy accountant tracks composition across rounds. A user running 10 rounds at ε=1 is not getting (1, δ)-DP; the library never tells them.
3. **"Byzantine defenses"** — z-score on global L2 norm only catches naive attacks. The phrasing oversells what's really a sanity check.

Additionally:
- No benchmark on real data shows FedEx-LoRA actually improves downstream task quality over FedAvg.
- No CI runs the 165-test suite on PRs. The number is marketing without enforcement.
- No external contributors; no issues, no PRs, no recurring releases — the public repo has zero contribution-velocity signal.

Phase 1 closes these credibility gaps.

## 2. Success criteria

By end of Phase 1:

- A user can run `chorus eval --strategy fedex-lora --model phi-3-mini --dataset alpaca-1k --clients 10 --rounds 5` and get a JSON+markdown report comparing FedEx-LoRA vs FedAvg on a real downstream metric.
- `benchmarks/results/v0.2.0/` contains paper-appendix-grade results: ≥2 models × 2 datasets × 3 seeds × {2, 5, 10, 20} clients × rank ∈ {4, 8, 16} × {fold-on, fold-off} × {DP-on, DP-off} × heterogeneous-rank case. Results are reproducible by re-running the published config files.
- The CLI raises a clear error when a configured (ε, δ) budget is exhausted, and `chorus status` shows the budget remaining per client.
- GitHub Actions: every PR runs pytest (Python 3.10/3.11/3.12), ruff, and a 3-minute smoke `chorus eval` run. Green badge in README.
- README and `docs/honest-tradeoffs.md` accurately describe every sharp edge. No claim in the README is unqualified that should be qualified.

## 3. Non-goals

- Not in Phase 1: secure aggregation, coordinate-wise median / Krum, schema pinning, multi-tenant scoping, audit log, pluggable storage, docker-compose, notebook tutorial. All deferred to Phases 2–4.
- Not in Phase 1: new aggregation strategies, new privacy mechanisms beyond accounting the existing Gaussian one, GPU/distributed-aggregation server.
- Not in Phase 1: replacing existing tests. Phase 1 adds tests but does not refactor the existing suite beyond what's needed for the new code.

## 4. Architecture

Five new or modified subsystems, each in its own module to keep blast radius small:

### 4.1 `chorus/eval/` — Evaluation harness (NEW)

A new package that runs simulated federations end-to-end on real models and real data, then computes both algorithmic and task-level metrics.

**Files:**
- `chorus/eval/__init__.py` — public API.
- `chorus/eval/runner.py` — `EvalRunner` class. Takes a `EvalConfig`, returns `EvalReport`. Orchestrates: data split → per-client local LoRA training → simulated aggregation → final model evaluation.
- `chorus/eval/config.py` — `EvalConfig` dataclass + YAML loader. Fields: model_id, dataset, strategy, num_clients, num_rounds, rank, seeds, dp_epsilon, fold_residuals, heterogeneous_rank, output_dir.
- `chorus/eval/datasets.py` — thin wrapper around HuggingFace `datasets` for the supported tasks. IID and non-IID partitioning helpers.
- `chorus/eval/metrics.py` — task-level metrics (perplexity, accuracy, F1 via HF `evaluate`) plus algorithmic metrics (Frobenius reconstruction error, residual norm, rank of full update).
- `chorus/eval/report.py` — `EvalReport` dataclass + JSON + markdown table serializers.

**Public interface:**
```python
from chorus.eval import EvalRunner, EvalConfig

config = EvalConfig.from_yaml("benchmarks/configs/phi3_alpaca_10clients.yaml")
report = EvalRunner(config).run()
report.to_markdown("benchmarks/results/v0.2.0/phi3_alpaca_10clients.md")
```

**CLI surface:** new `chorus eval` subcommand in `chorus/cli/main.py`. Reads a YAML config (`--config path.yaml`) or accepts flags. Supports:
- `--smoke` — preset alias for `--config benchmarks/configs/smoke.yaml` (TinyLlama-1.1B, 2 clients, 1 round, 1 seed; finishes in ~3 min on CPU; used by CI).
- `--check-only` — loads and validates the config, instantiates the runner, then exits without training. Used as a fast import/wiring tripwire in CI to fail before paying the ~3-min smoke cost.
- `--full` — uses whatever the YAML specifies, no presets applied.

**Dependencies it depends on:** existing `chorus.simulate.runner`, `chorus.server.aggregation`, `chorus.client.trainer`. PEFT, transformers, datasets, evaluate (all already in `chorus[peft]` except `evaluate` — adding it).

**What it doesn't do:** It does not touch the server. Eval runs locally and in-process; no HTTP roundtrip. This is intentional: the eval harness measures *algorithm quality*, not *server correctness* (server correctness is what `tests/` already covers).

### 4.2 `chorus/privacy/accountant.py` — DP accountant (NEW)

Replaces the stateless per-round Gaussian mechanism with a stateful budget tracker.

**File:** `chorus/privacy/accountant.py` (note: moves `chorus/server/privacy.py` into a new `chorus/privacy/` package — see Migration below).

**Class:**
```python
class PrivacyAccountant:
    def __init__(self, target_epsilon: float, target_delta: float,
                 noise_multiplier: float, sample_rate: float = 1.0,
                 backend: str = "rdp"): ...

    def step(self) -> None:
        """Record one round of noise application."""

    def get_epsilon(self, delta: float | None = None) -> float:
        """Current ε consumed at the given δ."""

    def is_exhausted(self) -> bool:
        """True if get_epsilon() >= target_epsilon."""

    def remaining(self) -> tuple[float, float]:
        """(epsilon_remaining, target_delta)."""

    def serialize(self) -> dict: ...
    @classmethod
    def deserialize(cls, data: dict) -> "PrivacyAccountant": ...
```

**Backend choice:** Use `dp-accounting` (Google) by default — it's the canonical RDP/PLD library, well-maintained, no PyTorch dep. Fallback to `opacus.accountants.RDPAccountant` if `dp-accounting` is unavailable. Tested behind a thin adapter so the choice can change without touching call sites.

**Server integration:** `state` gains a `dict[client_id, PrivacyAccountant]` mapping. `apply_dp()` becomes an instance method that increments the accountant. Submission endpoint returns budget-remaining alongside `deltas_received`. New endpoint `GET /models/{model_id}/clients/{client_id}/privacy` returns the accountant state.

**Client integration:** `ChorusClient` gains `--max-epsilon` parameter; client polls server-side budget after each submit and raises `PrivacyBudgetExhaustedError` (new in `exceptions.py`) when exceeded.

**Persistence:** Accountants are stored per-(model_id, client_id) in `chorus_data/<model>/privacy/<client>.json`. Restored on server restart via the existing lifespan hook.

**Migration note:** Existing `chorus/server/privacy.py` (which contains `GaussianMechanism`, `clip_delta`, `apply_dp`) moves to `chorus/privacy/mechanism.py`. The old import path stays as a deprecation re-export for one release.

### 4.3 `chorus/cli/main.py` — CLI additions (MODIFIED)

- New `chorus eval` subcommand (see §4.1).
- New `chorus privacy` subcommand: `chorus privacy budget --client-id X --model-id Y` prints remaining ε.
- `chorus status` augmented to display per-client budget if accounting is enabled.

### 4.4 `benchmarks/` — Benchmark suite (NEW + REPLACES)

The existing `benchmarks/benchmark.py` (231 lines, synthetic-data correctness + timing — uses `torch.randn` matrices, not real model deltas) gets archived into `benchmarks/legacy/` and the directory becomes a real-data benchmark suite:

```
benchmarks/
├── configs/                    # YAML configs, one per experiment
│   ├── smoke.yaml              # used in CI, ~3 min
│   ├── tinyllama_glue_sst2.yaml
│   ├── phi3_alpaca_clients_sweep.yaml
│   ├── llama3_1b_alpaca_rank_ablation.yaml
│   ├── llama3_1b_alpaca_dp_ablation.yaml
│   ├── llama3_1b_alpaca_fold_ablation.yaml
│   └── llama3_1b_alpaca_hetero_rank.yaml
├── run_all.py                  # iterates configs, writes results/v0.2.0/
├── verify_smoke_results.py     # asserts smoke output exists + FedEx not worse than FedAvg (CI tripwire)
├── results/
│   └── v0.2.0/                 # committed JSON + markdown results
└── legacy/
    └── benchmark.py            # old synthetic benchmark, preserved
```

Each config encodes: model, dataset, num_clients, num_rounds, rank, strategies-to-compare, seeds, dp params, fold flag, output filename.

`run_all.py` is what the user invokes once on rented GPU time. Outputs are deterministic-per-config (seeded). The full sweep estimate at v0.2.0 launch: ~80 GPU-hours on a single A100 (rough, will be refined when smoke-config timings come in).

### 4.5 `.github/workflows/ci.yml` — CI pipeline (NEW)

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: pip install -e ".[all]"
      - run: ruff check chorus tests benchmarks
      - run: pytest tests/ -v --cov=chorus
      - run: chorus eval --config benchmarks/configs/smoke.yaml --check-only
```

CI does **not** run real-data benchmarks — those need GPU and are inappropriate for free-tier runners. `--check-only` exercises the eval harness wiring (config load, runner construction, dataset/model resolution dry-run) without training; that's enough to catch regressions in the new code path. `verify_smoke_results.py` is invoked manually after `run_all.py` completes, not in CI.

Badge added to README pointing at the workflow.

### 4.6 README + `docs/honest-tradeoffs.md` (MODIFIED + NEW)

**README changes:**
- The "Mathematically exact federated aggregation" hero line becomes: "Mathematically exact federated aggregation **when residuals are folded into base weights every round** (see ['When exactness applies'](docs/honest-tradeoffs.md#exactness))."
- New "Honest tradeoffs" section near the top that links to the new doc and one-lines each major caveat.
- "Security Features" section: "Byzantine defenses" → "Sanity-check defenses against naive attackers", with explicit list of what they catch and what they don't.
- "Security Features" / DP subsection: link to accountant docs, note that without `--max-epsilon` the library does *not* prevent unbounded privacy loss across rounds.
- Production-readiness paragraph added below the "Configuration Examples" section: "Chorus is alpha software. Single-process server, in-memory rate limiter, filesystem storage. Suitable for research and small federations; not yet hardened for multi-tenant production. See [docs/honest-tradeoffs.md](docs/honest-tradeoffs.md) for the full list."

**`docs/honest-tradeoffs.md` outline:**
1. When "exact" applies (residual folding, base weight redistribution, bandwidth cost).
2. Differential privacy (composition tracking, what the accountant does and doesn't model).
3. Byzantine robustness (what z-score and norm-bound catch and miss).
4. Multi-tenant scope (global API keys, no per-model scoping yet).
5. Heterogeneous clients (rank/target-module mismatch is tolerated silently).
6. Production readiness (single-process, in-memory components, filesystem state).
7. What's on the roadmap to fix each of these.

## 5. Data flow

For the eval path (the new, primary data flow):

```
EvalConfig (YAML)
   │
   ▼
EvalRunner
   ├──► load model + tokenizer (HF)
   ├──► load + partition dataset (chorus.eval.datasets)
   ├──► seed loop:
   │      ├──► per-client local LoRA training (chorus.client.trainer)
   │      ├──► collect deltas
   │      ├──► run aggregation N rounds (chorus.server.aggregation)
   │      │     └──► track residuals + (optionally) fold into base
   │      ├──► evaluate final model on held-out split (chorus.eval.metrics)
   │      └──► record both algorithmic and task metrics
   └──► EvalReport
          ├──► .to_json()
          └──► .to_markdown()
```

For the accountant path:

```
client.submit_delta()
   │ HTTP POST
   ▼
submit_delta endpoint
   ├──► apply_dp(noise) ──► accountant.step()  [if dp enabled]
   ├──► accountant.is_exhausted()?
   │       yes ──► 403 PrivacyBudgetExhausted
   │       no  ──► proceed
   └──► aggregate + return budget remaining in response
```

## 6. Testing strategy

Three layers, all added or expanded in this phase:

1. **Unit tests** for new code:
   - `tests/test_eval_runner.py` — runs `EvalRunner` on synthetic-but-tiny config; verifies report structure and that FedEx error < FedAvg error.
   - `tests/test_privacy_accountant.py` — verifies (a) ε grows monotonically with steps, (b) `is_exhausted` triggers at the right threshold, (c) serialize/deserialize round-trips correctly.
   - `tests/test_eval_metrics.py` — sanity-checks Frobenius error and HF metric integration on tiny inputs.

2. **CLI tests** for the new subcommands: `tests/test_cli_eval.py`, `tests/test_cli_privacy.py`. Use Click's `CliRunner`.

3. **Eval wiring check in CI** — `chorus eval --config smoke.yaml --check-only` loads + validates the config and constructs the runner without training. Catches import errors, dataset-resolution bugs, and config-schema breakage between releases. Cheap (<10s) and reliable (no model download, no GPU, no flake).

The existing 165 tests stay green throughout. Any tests that break due to the `chorus.server.privacy` → `chorus.privacy.mechanism` move get updated; no functional changes to existing tests.

## 7. Migration / backward compatibility

- `chorus.server.privacy` becomes a deprecation re-export: `from chorus.privacy.mechanism import *` plus a `DeprecationWarning` on import. Removed in v0.3.0.
- `apply_dp()` keeps its old signature; a new `apply_dp_with_accountant(tensors, accountant, ...)` is added. Old callers (server `_run_aggregation`, simulate runner) updated to use the new path.
- Existing `benchmarks/benchmark.py` is preserved at `benchmarks/legacy/benchmark.py`. The new `chorus eval` is additive.

## 8. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `dp-accounting` library API churn breaks the accountant | Low | Medium | Wrap it behind our `PrivacyAccountant` interface; tests catch breakage on dep bump |
| Smoke `chorus eval` in CI is flaky (network for HF model download) | Medium | Medium | Cache HF artifacts in CI, pin to a tiny model (TinyLlama or smaller), set `--offline` once cached |
| Paper-appendix benchmark takes way more GPU-hours than estimated | Medium | Medium | Smoke config first nails timing per (model, rank, clients) combo; full sweep launched only when extrapolated cost is acceptable |
| User runs `chorus eval` and gets results that contradict the README claims | Low | High | This is the *point* of the benchmark. If FedEx-LoRA doesn't actually beat FedAvg on a real task, the README is wrong and the README gets corrected before v0.2.0 ships. |
| Existing `chorus.server.privacy` import path breaks downstream users | Very low (no users yet) | Low | Keep deprecation re-export through v0.2.x |
| Accountant adds latency to hot path | Low | Low | `step()` is O(1) microbenchmark; serialize-on-shutdown only |

## 9. Sequencing (within Phase 1)

A suggested order, optimized so each piece unblocks the next:

1. **Week 1**: CI pipeline + README first-pass rewrite (deepest unrelated to other work; gets the trust signal up fast).
2. **Week 1–2**: `chorus.privacy.accountant` + tests + integration into `_run_aggregation`. Self-contained.
3. **Week 2–4**: `chorus.eval` package + `chorus eval` CLI + smoke config + smoke-in-CI. Biggest single chunk.
4. **Week 4–5**: Benchmark configs (`benchmarks/configs/*.yaml`) + `run_all.py`. Reuses `chorus.eval`.
5. **Week 5–6**: Full benchmark run on rented GPU. Land results in `benchmarks/results/v0.2.0/`. Update README with real numbers.
6. **Week 6–7**: `docs/honest-tradeoffs.md` + final README pass + v0.2.0 release notes + PyPI publish.

Each chunk lands on its own PR off `master`, each PR closes a sub-issue, the whole phase culminates in a v0.2.0 git tag.

## 10. Open questions (to resolve during implementation, not now)

- Exact model list for the full benchmark: TinyLlama-1.1B and Phi-3-mini are confirmed; the third (Llama-3.2-1B vs -3B) depends on whether smoke-config timings make -3B affordable.
- Exact dataset list: leaning toward (Alpaca-1k-slice for instruction tuning + SST-2 for classification), but open to substitution if HF licensing or partitioning is awkward.
- Whether to ship a small comparison against Flower or PySyft as part of the benchmark. Adds credibility but doubles the eval-harness surface area. Default: not in Phase 1; revisit in Phase 2.
- Whether the accountant should also account for the *client-side* noise the SDK can apply (`ChorusClient(dp_epsilon=...)`). Default yes — it's the user's privacy budget; server vs client noise is an implementation detail.

## 11. Definition of done

Phase 1 ships when:

1. All five subsystems above are merged to master.
2. CI is green on master, badge in README.
3. `chorus eval` works end-to-end on the smoke config from a fresh `pip install chorus-fl==0.2.0`.
4. `benchmarks/results/v0.2.0/` contains the full result set, generated and committed.
5. README claims match what the benchmark results show. Any claim that doesn't survive the benchmark gets removed.
6. `docs/honest-tradeoffs.md` exists, is linked from README, and covers all seven sections listed in §4.6.
7. v0.2.0 tag is cut, PyPI release is up, release notes are written.
