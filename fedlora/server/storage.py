"""Filesystem-based delta storage for the aggregation server."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from safetensors.torch import load_file, save_file


class DeltaStorage:
    """Stores LoRA deltas on the filesystem, organized by model and round."""

    def __init__(self, base_dir: str | Path = "./fedlora_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, model_id: str) -> Path:
        safe_name = model_id.replace("/", "__")
        return self.base_dir / safe_name

    def _round_dir(self, model_id: str, round_id: int) -> Path:
        return self._model_dir(model_id) / f"round_{round_id}"

    def _deltas_dir(self, model_id: str, round_id: int) -> Path:
        return self._round_dir(model_id, round_id) / "deltas"

    def save_delta(
        self,
        model_id: str,
        round_id: int,
        client_id: str,
        tensors: dict,
        metadata: dict | None = None,
    ) -> Path:
        """Save a client's LoRA delta to disk."""
        deltas_dir = self._deltas_dir(model_id, round_id)
        deltas_dir.mkdir(parents=True, exist_ok=True)

        delta_path = deltas_dir / f"{client_id}.safetensors"
        save_file(tensors, str(delta_path))

        # Save metadata alongside
        meta_path = deltas_dir / f"{client_id}.json"
        meta = {
            "client_id": client_id,
            "model_id": model_id,
            "round_id": round_id,
            "timestamp": time.time(),
            **(metadata or {}),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        return delta_path

    def list_deltas(self, model_id: str, round_id: int) -> list[str]:
        """List client IDs that have submitted deltas for a given round."""
        deltas_dir = self._deltas_dir(model_id, round_id)
        if not deltas_dir.exists():
            return []
        return [p.stem for p in deltas_dir.glob("*.safetensors")]

    def load_delta(self, model_id: str, round_id: int, client_id: str) -> dict:
        """Load a single client's delta tensors."""
        delta_path = self._deltas_dir(model_id, round_id) / f"{client_id}.safetensors"
        return load_file(str(delta_path))

    def load_all_deltas(self, model_id: str, round_id: int) -> list[dict]:
        """Load all client deltas for a round."""
        client_ids = self.list_deltas(model_id, round_id)
        return [self.load_delta(model_id, round_id, cid) for cid in client_ids]

    def save_aggregated(self, model_id: str, round_id: int, tensors: dict) -> Path:
        """Save the aggregated result for a round."""
        round_dir = self._round_dir(model_id, round_id)
        round_dir.mkdir(parents=True, exist_ok=True)

        agg_path = round_dir / "aggregated.safetensors"
        save_file(tensors, str(agg_path))

        # Update the latest symlink / pointer
        latest_path = self._model_dir(model_id) / "latest.safetensors"
        if latest_path.exists():
            latest_path.unlink()
        shutil.copy2(str(agg_path), str(latest_path))

        # Write round metadata
        meta_path = round_dir / "aggregation_meta.json"
        meta = {
            "model_id": model_id,
            "round_id": round_id,
            "num_clients": len(self.list_deltas(model_id, round_id)),
            "timestamp": time.time(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        return agg_path

    def load_aggregated(self, model_id: str, round_id: int | None = None) -> dict | None:
        """Load aggregated tensors. If round_id is None, load latest."""
        if round_id is not None:
            agg_path = self._round_dir(model_id, round_id) / "aggregated.safetensors"
        else:
            agg_path = self._model_dir(model_id) / "latest.safetensors"

        if not agg_path.exists():
            return None
        return load_file(str(agg_path))

    def get_latest_round(self, model_id: str) -> int | None:
        """Get the latest completed round number for a model."""
        model_dir = self._model_dir(model_id)
        if not model_dir.exists():
            return None
        rounds = []
        for p in model_dir.iterdir():
            if p.is_dir() and p.name.startswith("round_"):
                agg = p / "aggregated.safetensors"
                if agg.exists():
                    rounds.append(int(p.name.split("_")[1]))
        return max(rounds) if rounds else None

    def get_current_round(self, model_id: str) -> int:
        """Get the current round (latest completed + 1, or 0 if none)."""
        latest = self.get_latest_round(model_id)
        return 0 if latest is None else latest + 1

    def cleanup_round(self, model_id: str, round_id: int) -> None:
        """Remove delta files for a completed round to save space."""
        deltas_dir = self._deltas_dir(model_id, round_id)
        if deltas_dir.exists():
            shutil.rmtree(deltas_dir)
