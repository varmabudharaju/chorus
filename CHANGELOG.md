# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [0.1.0] - 2025-02-22

Initial open-source release.

### Added

- **Aggregation server** — FastAPI-based server with round lifecycle management (OPEN -> AGGREGATING -> CLOSED)
- **FedEx-LoRA aggregation** — Exact federated LoRA aggregation using SVD with residual tracking, based on the FedEx-LoRA paper (ACL/ICLR 2025)
- **FedAvg baseline** — Standard federated averaging for comparison
- **Residual folding** — Automatic folding of FedEx-LoRA residuals into base weights for lossless continuous training
- **Python SDK** (`ChorusClient`) — Submit deltas, pull adapters, listen for round events, run full training loops, export merged models
- **CLI** — `chorus server`, `chorus submit`, `chorus pull`, `chorus train`, `chorus simulate`, `chorus status`, `chorus export`
- **Differential privacy** — Gaussian DP mechanism with global L2 clipping (client-side and server-side)
- **Byzantine defenses** — L2 norm bounding and z-score outlier detection for submitted deltas
- **Authentication** — Bearer token auth with support for multiple API keys
- **Rate limiting** — Per-IP request rate limiting on the server
- **WebSocket notifications** — Real-time `round_complete` broadcasts to connected clients with auto-reconnect
- **Dataset-size weighting** — Proportional client weighting based on dataset size
- **Model export** — Merge aggregated adapter with HuggingFace base model into a deployable model directory
- **Simulation runner** — Synthetic multi-client federation for testing and demos
- **safetensors format** — All weight serialization uses safetensors (no pickle)
- **165 tests** — Comprehensive test suite covering server, client, aggregation, privacy, security, and CLI

[0.1.0]: https://github.com/varmabudharaju/chorus/releases/tag/v0.1.0
