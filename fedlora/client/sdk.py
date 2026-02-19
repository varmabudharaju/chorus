"""FedLoRA client SDK for submitting deltas and pulling aggregated adapters."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
from safetensors.torch import save_file, load_file

from fedlora.client.delta import extract_lora_matrices, save_lora_delta
from fedlora.server.privacy import apply_dp

logger = logging.getLogger("fedlora.client")


class FedLoRAClient:
    """Client for interacting with a FedLoRA aggregation server.

    Usage:
        client = FedLoRAClient(server="http://localhost:8080", model_id="my-model")
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

        return self._submit_tensors(tensors, round_id)

    def submit_tensors(
        self,
        tensors: dict,
        round_id: int | None = None,
    ) -> dict:
        """Submit raw tensors (already extracted) to the server.

        Useful when you've already extracted/modified the tensors yourself.
        """
        if round_id is None:
            status = self.status()
            round_id = status["current_round"]
        return self._submit_tensors(tensors, round_id)

    def _submit_tensors(self, tensors: dict, round_id: int) -> dict:
        """Internal: serialize and upload tensors."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=True) as tmp:
            save_file(tensors, tmp.name)
            tmp.seek(0)

            params = {"model_id": self.model_id}
            if self.client_id:
                params["client_id"] = self.client_id

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

    def pull_latest(self, output_path: str | Path) -> Path:
        """Pull the latest aggregated adapter from the server.

        Args:
            output_path: Where to save the aggregated adapter.
                Can be a directory (saves adapter_model.safetensors inside)
                or a .safetensors file path.

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
        logger.info(f"Pulled latest aggregated adapter to {output_path}")
        return output_path

    def pull_round(self, round_id: int, output_path: str | Path) -> Path:
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
        logger.info(f"Pulled round {round_id} aggregated adapter to {output_path}")
        return output_path

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
