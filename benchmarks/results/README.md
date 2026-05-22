# Benchmark results

This directory holds the *committed* output of `python benchmarks/run_all.py`.

- `v0.2.0/` — populated by the paid GPU run between F4 and F6 of the v0.2.0
  Phase 1 release. Contains one subdirectory per YAML config under
  `benchmarks/configs/`, with per-run `report.json` + `report.md` underneath.
  Tracked in git so the published numbers are reproducible and reviewable.

- `smoke/` — local-only outputs from the CI smoke config (`benchmarks/configs/smoke.yaml`).
  Not committed; gitignored.

To regenerate v0.2.0 results from scratch (requires ~$80–$160 of A100 time):

    python benchmarks/run_all.py --all --output-root benchmarks/results/v0.2.0
    python benchmarks/verify_smoke_results.py benchmarks/results/v0.2.0/smoke/base/report.json
