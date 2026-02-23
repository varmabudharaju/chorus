# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
