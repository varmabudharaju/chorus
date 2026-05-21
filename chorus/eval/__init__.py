"""Evaluation harness for federated LoRA aggregation strategies.

Runs simulated multi-client federations on real models + datasets, then emits
JSON and markdown reports comparing aggregation strategies (FedAvg vs FedExLoRA)
on both task metrics (perplexity, accuracy, F1) and algorithmic metrics
(Frobenius reconstruction error against the exact full-rank average).

Used as a CLI: `chorus eval --config <yaml>`.
Or programmatically:
    from chorus.eval import EvalRunner, EvalConfig
    config = EvalConfig.from_yaml("benchmarks/configs/smoke.yaml")
    report = EvalRunner(config).run()
    report.to_markdown("results.md")
"""

from chorus.eval.config import EvalConfig
from chorus.eval.report import EvalReport
from chorus.eval.runner import EvalRunner

__all__ = ["EvalConfig", "EvalReport", "EvalRunner"]
