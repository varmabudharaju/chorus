"""EvalConfig dataclass + YAML loader for the eval harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from chorus.exceptions import EvalConfigError
from chorus.server.aggregation import STRATEGIES as _AVAILABLE_STRATEGIES


@dataclass
class EvalConfig:
    """Configuration for a single evaluation run.

    Fields map 1:1 to YAML keys. See benchmarks/configs/smoke.yaml for an example.
    """

    model_id: str
    dataset: dict[str, Any]  # {name, split, max_examples?, ...}
    num_clients: int
    num_rounds: int
    strategies: list[str]  # e.g., ["fedavg", "fedex-lora"]
    rank: int
    seeds: list[int]

    # Optional
    dp_epsilon: float | None = None
    dp_delta: float = 1e-5
    dp_max_norm: float = 1.0
    fold_residuals: bool = True
    heterogeneous_rank: list[int] | None = None  # per-client ranks, len must = num_clients
    partition: str = "iid"  # or "dirichlet"
    dirichlet_alpha: float = 0.5
    output_dir: str = "benchmarks/results"
    learning_rate: float = 5e-5
    max_steps_per_round: int = 50
    eval_batch_size: int = 8
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        """Load and validate an EvalConfig from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise EvalConfigError(f"Config file not found: {path}")
        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            raise EvalConfigError(f"Invalid YAML in {path}: {e}") from e
        if not isinstance(data, dict):
            raise EvalConfigError(f"Config root must be a mapping, got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalConfig":
        """Build an EvalConfig from a dict (e.g., parsed YAML). Validates inputs."""
        required = ["model_id", "dataset", "num_clients", "num_rounds", "strategies", "rank", "seeds"]
        for key in required:
            if key not in data:
                raise EvalConfigError(f"Missing required field: {key}")

        # Validate strategies
        for s in data["strategies"]:
            if s not in _AVAILABLE_STRATEGIES:
                raise EvalConfigError(
                    f"Unknown strategy '{s}'. Available: {list(_AVAILABLE_STRATEGIES.keys())}"
                )

        # Validate counts
        if not isinstance(data["num_clients"], int) or data["num_clients"] < 1:
            raise EvalConfigError("num_clients must be a positive integer")
        if not isinstance(data["num_rounds"], int) or data["num_rounds"] < 1:
            raise EvalConfigError("num_rounds must be a positive integer")
        if not isinstance(data["rank"], int) or data["rank"] < 1:
            raise EvalConfigError("rank must be a positive integer")
        if not data["seeds"]:
            raise EvalConfigError("seeds must be a non-empty list of integers")

        # Heterogeneous-rank length must equal num_clients
        het = data.get("heterogeneous_rank")
        if het is not None and len(het) != data["num_clients"]:
            raise EvalConfigError(
                f"heterogeneous_rank length ({len(het)}) must equal num_clients ({data['num_clients']})"
            )

        # Filter to known fields to avoid TypeError on extra keys; warn on unknowns
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data.keys()) - known
        if unknown:
            import logging
            _cfg_logger = logging.getLogger("chorus.eval.config")
            _cfg_logger.warning(
                "Ignoring unknown config keys: %s. Known keys: %s",
                sorted(unknown), sorted(known),
            )
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
