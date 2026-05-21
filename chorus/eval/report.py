"""EvalReport: serializable record of a single EvalRunner execution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class StrategyResult:
    """Per-strategy, per-seed outcome of one eval run."""

    strategy: str
    seed: int
    final_task_metric: dict[str, float]
    frobenius_error: float
    per_round_times_s: list[float] = field(default_factory=list)
    notes: str = ""


@dataclass
class EvalReport:
    """Top-level evaluation report; aggregates StrategyResults from one EvalRunner run."""

    config_name: str
    model_id: str
    dataset_name: str
    num_clients: int
    num_rounds: int
    rank: int
    seeds: list[int]
    results: list[StrategyResult] = field(default_factory=list)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    def to_markdown(self, path: str | Path) -> None:
        Path(path).write_text(self.to_markdown_string())

    def to_markdown_string(self) -> str:
        lines = [
            f"# Eval Report — {self.config_name}",
            "",
            f"- **Model:** `{self.model_id}`",
            f"- **Dataset:** `{self.dataset_name}`",
            f"- **Clients:** {self.num_clients}",
            f"- **Rounds:** {self.num_rounds}",
            f"- **LoRA rank:** {self.rank}",
            f"- **Seeds:** {self.seeds}",
            "",
            "## Results",
            "",
            "| Strategy | Seed | Task metric | Frobenius error | Mean round time (s) |",
            "|---|---|---|---|---|",
        ]
        for r in self.results:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in r.final_task_metric.items())
            mean_t = (
                sum(r.per_round_times_s) / len(r.per_round_times_s)
                if r.per_round_times_s
                else 0.0
            )
            lines.append(
                f"| `{r.strategy}` | {r.seed} | {metric_str} | {r.frobenius_error:.4f} | {mean_t:.2f} |"
            )
        return "\n".join(lines) + "\n"
