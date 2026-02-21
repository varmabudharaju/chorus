"""Chorus client SDK for submitting deltas and pulling aggregated adapters."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Generator

import httpx
from safetensors.torch import save_file, load_file

from chorus.client.delta import extract_lora_matrices, save_lora_delta
from chorus.server.privacy import apply_dp

logger = logging.getLogger("chorus.client")


class ChorusClient:
    """Client for interacting with a Chorus aggregation server.

    Usage:
        client = ChorusClient(server="http://localhost:8080", model_id="my-model")
        # ... do local LoRA training ...
        client.submit_delta(adapter_path="./my-adapter")
        client.pull_latest(output_path="./updated-adapter")
    """

    def __init__(
        self,
        server: str,
        model_id: str,
        client_id: str | None = None,
        dp_epsilon: float | None = None,
        dp_delta: float = 1e-5,
        dp_max_norm: float = 1.0,
        timeout: float = 120.0,
    ):
        self.server = server.rstrip("/")
        self.model_id = model_id
        self.client_id = client_id
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_max_norm = dp_max_norm
        self._http = httpx.Client(base_url=self.server, timeout=timeout)

    def status(self) -> dict:
        """Get the current status of the model on the server."""
        resp = self._http.get(f"/models/{self.model_id}/status")
        resp.raise_for_status()
        return resp.json()

    def submit_delta(
        self,
        adapter_path: str | Path,
        round_id: int | None = None,
        dataset_size: int | None = None,
    ) -> dict:
        """Extract LoRA delta from an adapter and submit to the server.

        Args:
            adapter_path: Path to a PEFT adapter directory or .safetensors file.
            round_id: Round to submit to. If None, uses the server's current round.

        Returns:
            Server response dict with submission status.
        """
        # Extract LoRA matrices
        tensors = extract_lora_matrices(adapter_path)

        # Apply local DP if configured
        if self.dp_epsilon is not None:
            logger.info(f"Applying local DP (epsilon={self.dp_epsilon})")
            tensors = apply_dp(
                tensors,
                epsilon=self.dp_epsilon,
                delta=self.dp_delta,
                max_norm=self.dp_max_norm,
            )

        # Determine round
        if round_id is None:
            status = self.status()
            round_id = status["current_round"]

        return self._submit_tensors(tensors, round_id, dataset_size=dataset_size)

    def submit_tensors(
        self,
        tensors: dict,
        round_id: int | None = None,
        dataset_size: int | None = None,
    ) -> dict:
        """Submit raw tensors (already extracted) to the server.

        Useful when you've already extracted/modified the tensors yourself.
        """
        if round_id is None:
            status = self.status()
            round_id = status["current_round"]
        return self._submit_tensors(tensors, round_id, dataset_size=dataset_size)

    def _submit_tensors(self, tensors: dict, round_id: int, dataset_size: int | None = None) -> dict:
        """Internal: serialize and upload tensors."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=True) as tmp:
            save_file(tensors, tmp.name)
            tmp.seek(0)

            params = {"model_id": self.model_id}
            if self.client_id:
                params["client_id"] = self.client_id
            if dataset_size is not None:
                params["dataset_size"] = dataset_size

            with open(tmp.name, "rb") as f:
                resp = self._http.post(
                    f"/rounds/{round_id}/deltas",
                    files={"file": ("delta.safetensors", f, "application/octet-stream")},
                    params=params,
                )

        resp.raise_for_status()
        result = resp.json()
        logger.info(
            f"Delta submitted: round={round_id}, "
            f"received={result['deltas_received']}/{result['min_deltas']}, "
            f"aggregated={result['aggregated']}"
        )
        return result

    def pull_latest(self, output_path: str | Path, adapter_config: dict | None = None) -> Path:
        """Pull the latest aggregated adapter from the server.

        Args:
            output_path: Where to save the aggregated adapter.
                Can be a directory (saves adapter_model.safetensors inside)
                or a .safetensors file path.
            adapter_config: Optional PEFT adapter_config.json contents to write
                alongside the safetensors file. Required for PeftModel.from_pretrained()
                to load the adapter in subsequent rounds.

        Returns:
            Path to the saved file.
        """
        resp = self._http.get(f"/models/{self.model_id}/latest")
        if resp.status_code == 404:
            raise FileNotFoundError(
                f"No aggregated adapter available yet for model '{self.model_id}'"
            )
        resp.raise_for_status()

        output_path = Path(output_path)
        if output_path.suffix != ".safetensors":
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / "adapter_model.safetensors"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_bytes(resp.content)

        if adapter_config is not None:
            config_path = output_path.parent / "adapter_config.json"
            config_path.write_text(json.dumps(adapter_config, indent=2))
            logger.info(f"Wrote adapter_config.json to {config_path}")

        logger.info(f"Pulled latest aggregated adapter to {output_path}")
        return output_path

    def pull_round(self, round_id: int, output_path: str | Path, adapter_config: dict | None = None) -> Path:
        """Pull the aggregated adapter for a specific round."""
        resp = self._http.get(f"/models/{self.model_id}/rounds/{round_id}")
        if resp.status_code == 404:
            raise FileNotFoundError(
                f"No aggregation found for model '{self.model_id}' round {round_id}"
            )
        resp.raise_for_status()

        output_path = Path(output_path)
        if output_path.suffix != ".safetensors":
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / "adapter_model.safetensors"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_bytes(resp.content)

        if adapter_config is not None:
            config_path = output_path.parent / "adapter_config.json"
            config_path.write_text(json.dumps(adapter_config, indent=2))
            logger.info(f"Wrote adapter_config.json to {config_path}")

        logger.info(f"Pulled round {round_id} aggregated adapter to {output_path}")
        return output_path

    def train_loop(
        self,
        trainer: "LoRATrainer",
        num_rounds: int | None = None,
        on_round_complete: Callable | None = None,
    ):
        """Run the complete federated training loop.

        1. Train locally
        2. Submit delta to server
        3. Wait for aggregation via WebSocket
        4. Pull updated adapter
        5. Repeat

        Args:
            trainer: A LoRATrainer instance configured for local training.
            num_rounds: Number of rounds (None = infinite).
            on_round_complete: Optional callback(rounds_completed, result).
        """
        rounds_completed = 0
        adapter_path = trainer.adapter_path

        while num_rounds is None or rounds_completed < num_rounds:
            trainer.adapter_path = adapter_path

            logger.info(f"[Round {rounds_completed}] Training locally...")
            adapter_path = trainer.train()

            logger.info(f"[Round {rounds_completed}] Submitting delta...")
            result = self.submit_delta(
                adapter_path=adapter_path,
                dataset_size=trainer.get_dataset_size(),
            )

            if not result["aggregated"]:
                logger.info("Waiting for aggregation...")
                for event in self.listen():
                    if event.get("event") == "round_complete":
                        break

            logger.info("Pulling updated adapter...")
            pull_dir = trainer.output_dir / "aggregated"
            adapter_path = self.pull_latest(output_path=pull_dir)

            rounds_completed += 1
            if on_round_complete:
                on_round_complete(rounds_completed, result)

    def listen(self, on_round_complete: Callable | None = None) -> Generator[dict, None, None]:
        """Connect via WebSocket and listen for server events.

        Args:
            on_round_complete: Optional callback(event_dict) called when a round completes.

        Yields:
            Event dicts as they arrive.
        """
        import websockets.sync.client as ws_sync

        ws_url = self.server.replace("http://", "ws://").replace("https://", "wss://")
        cid = self.client_id or "listener"
        with ws_sync.connect(f"{ws_url}/ws/{cid}") as ws:
            while True:
                msg = json.loads(ws.recv())
                if on_round_complete and msg.get("event") == "round_complete":
                    on_round_complete(msg)
                yield msg

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
