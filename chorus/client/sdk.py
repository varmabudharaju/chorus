"""Chorus client SDK for submitting deltas and pulling aggregated adapters."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generator

if TYPE_CHECKING:
    from chorus.client.trainer import LoRATrainer

import httpx
from safetensors.torch import save_file, load_file

from chorus.client.delta import extract_lora_matrices
from chorus.exceptions import (
    AggregationPendingError,
    ChorusError,
    ExportError,
    RoundClosedError,
    ServerUnreachableError,
    SubmissionError,
)
from chorus.privacy.mechanism import apply_dp

logger = logging.getLogger("chorus.client")

_MAX_RETRIES = 3


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
        api_key: str | None = None,
        dp_epsilon: float | None = None,
        dp_delta: float = 1e-5,
        dp_max_norm: float = 1.0,
        timeout: float = 120.0,
        max_epsilon: float | None = None,
    ):
        self.server = server.rstrip("/")
        self.model_id = model_id
        self.client_id = client_id
        self.api_key = api_key
        self.max_epsilon = max_epsilon
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_max_norm = dp_max_norm
        self._warned_about_no_server_accounting = False
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(base_url=self.server, timeout=timeout, headers=headers)

    # --- Internal HTTP helper with retries ---

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Send an HTTP request with retries on transient failures.

        Retries on connection errors, timeouts, and 5xx responses (up to 3 attempts
        with exponential backoff: 1s, 2s, 4s). Does NOT retry 4xx errors.
        """
        resp: httpx.Response | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._http.request(method, path, **kwargs)
                if resp.status_code >= 500 and attempt < _MAX_RETRIES - 1:
                    logger.warning(
                        "Server error %d on %s %s, retrying (%d/%d)...",
                        resp.status_code, method, path, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(2 ** attempt)
                    continue
                return resp
            except httpx.ConnectError:
                if attempt < _MAX_RETRIES - 1:
                    logger.warning(
                        "Connection failed to %s, retrying (%d/%d)...",
                        self.server, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(2 ** attempt)
                    continue
                raise ServerUnreachableError(
                    f"Cannot connect to Chorus server at {self.server}"
                ) from None
            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    logger.warning(
                        "Request timed out, retrying (%d/%d)...",
                        attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(2 ** attempt)
                    continue
                raise ServerUnreachableError(
                    f"Request to {self.server} timed out"
                ) from None

        # Should be unreachable, but satisfies the type checker
        assert resp is not None
        return resp

    def _translate_response(self, resp: httpx.Response, context: str = "") -> None:
        """Translate HTTP error responses to Chorus exceptions."""
        if resp.status_code < 400:
            return

        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            detail = resp.text

        if resp.status_code == 409:
            if "not accepting" in detail.lower():
                raise RoundClosedError(detail)
            raise SubmissionError(detail)
        if resp.status_code == 404:
            if context == "pull":
                raise AggregationPendingError(detail)
            raise ChorusError(detail)
        if resp.status_code == 400:
            raise SubmissionError(detail)
        if resp.status_code == 403 and "budget exhausted" in detail.lower():
            from chorus.exceptions import PrivacyBudgetExhaustedError
            raise PrivacyBudgetExhaustedError(detail)
        raise ChorusError(f"Server error ({resp.status_code}): {detail}")

    # --- Public API ---

    def status(self) -> dict:
        """Get the current status of the model on the server."""
        resp = self._request("GET", f"/models/{self.model_id}/status")
        self._translate_response(resp)
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
                resp = self._request(
                    "POST",
                    f"/rounds/{round_id}/deltas",
                    files={"file": ("delta.safetensors", f, "application/octet-stream")},
                    params=params,
                )

        self._translate_response(resp)
        result = resp.json()
        logger.info(
            f"Delta submitted: round={round_id}, "
            f"received={result['deltas_received']}/{result['min_deltas']}, "
            f"aggregated={result['aggregated']}"
        )

        # Check privacy budget if accounting is enabled server-side
        if self.max_epsilon is not None:
            if "privacy" in result:
                consumed = result["privacy"]["epsilon_consumed"]
                if consumed >= self.max_epsilon:
                    from chorus.exceptions import PrivacyBudgetExhaustedError

                    raise PrivacyBudgetExhaustedError(
                        f"Client '{self.client_id}' exceeded configured max_epsilon "
                        f"({self.max_epsilon}); server reports ε={consumed:.4f}"
                    )
            elif not self._warned_about_no_server_accounting:
                logger.warning(
                    "ChorusClient was configured with max_epsilon=%s but the server does not "
                    "have privacy accounting enabled. The max_epsilon constraint will not be "
                    "enforced. To enable, configure the server with accountant_target_epsilon.",
                    self.max_epsilon,
                )
                self._warned_about_no_server_accounting = True

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
        resp = self._request("GET", f"/models/{self.model_id}/latest")
        self._translate_response(resp, context="pull")

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
        resp = self._request("GET", f"/models/{self.model_id}/rounds/{round_id}")
        self._translate_response(resp, context="pull")

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

    def export_model(self, base_model: str, output_dir: str | Path, round_id: int | None = None) -> Path:
        """Download the aggregated adapter and merge with base model into a deployable HF model directory.

        Creates: output_dir/config.json, model.safetensors, tokenizer.json, etc.
        Loadable via: AutoModelForCausalLM.from_pretrained(output_dir)

        Args:
            base_model: HuggingFace model ID (e.g. "Qwen/Qwen2.5-7B").
            output_dir: Directory to save the merged model.
            round_id: Specific round to export (None = latest).

        Returns:
            Path to the output directory.
        """
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ExportError(
                "Export requires 'transformers' and 'peft' packages. "
                "Install them with: pip install transformers peft"
            )

        output_dir = Path(output_dir)

        # 1. Pull the adapter to a temp directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            if round_id is not None:
                adapter_path = self.pull_round(round_id, tmpdir / "adapter")
            else:
                adapter_path = self.pull_latest(tmpdir / "adapter")

            adapter_dir = adapter_path.parent

            # 2. We need an adapter_config.json for PeftModel to load it.
            #    Create a minimal one if the server didn't provide one.
            config_path = adapter_dir / "adapter_config.json"
            if not config_path.exists():
                # Infer rank from the adapter tensors
                tensors = load_file(str(adapter_path))
                rank = None
                for key, tensor in tensors.items():
                    if "lora_A" in key:
                        rank = tensor.shape[0]
                        break
                if rank is None:
                    raise ExportError("Cannot infer LoRA rank from adapter tensors")

                target_modules = list({
                    k.rsplit(".lora_", 1)[0].rsplit(".", 1)[-1]
                    for k in tensors if "lora_A" in k or "lora_B" in k
                })

                config = {
                    "peft_type": "LORA",
                    "auto_mapping": None,
                    "base_model_name_or_path": base_model,
                    "r": rank,
                    "lora_alpha": rank,
                    "lora_dropout": 0.0,
                    "target_modules": target_modules,
                    "task_type": "CAUSAL_LM",
                    "fan_in_fan_out": False,
                    "bias": "none",
                }
                config_path.write_text(json.dumps(config, indent=2))

            # 3. Load base model
            logger.info(f"Loading base model: {base_model}")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model, torch_dtype="auto", device_map="cpu",
                )
            except Exception as exc:
                raise ExportError(f"Failed to load base model '{base_model}': {exc}") from exc

            # 4. Apply adapter and merge
            logger.info("Applying adapter and merging weights...")
            try:
                model = PeftModel.from_pretrained(model, str(adapter_dir))
                model = model.merge_and_unload()
            except Exception as exc:
                raise ExportError(f"Failed to merge adapter: {exc}") from exc

            # 5. Save merged model
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving merged model to {output_dir}")
            model.save_pretrained(output_dir)

            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model)
                tokenizer.save_pretrained(output_dir)
            except Exception:
                logger.warning("Could not save tokenizer (model may not have one)")

        logger.info(f"Export complete: {output_dir}")
        return output_dir

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

            # --- Train locally ---
            logger.info(f"[Round {rounds_completed}] Training locally...")
            try:
                adapter_path = trainer.train()
            except Exception as exc:
                logger.error(f"[Round {rounds_completed}] Training failed: {exc}")
                raise

            # --- Submit delta ---
            logger.info(f"[Round {rounds_completed}] Submitting delta...")
            try:
                result = self.submit_delta(
                    adapter_path=adapter_path,
                    dataset_size=trainer.get_dataset_size(),
                )
            except SubmissionError as exc:
                logger.error(f"[Round {rounds_completed}] Submission failed: {exc}")
                raise

            # --- Wait for aggregation ---
            if not result["aggregated"]:
                logger.info("Waiting for aggregation...")
                try:
                    for event in self.listen():
                        if event.get("event") == "round_complete":
                            break
                except Exception as exc:
                    logger.warning(f"WebSocket listen failed: {exc}, polling status instead...")
                    self._poll_for_aggregation(rounds_completed)

            # --- Pull updated adapter ---
            logger.info("Pulling updated adapter...")
            pull_dir = trainer.output_dir / "aggregated"
            for attempt in range(_MAX_RETRIES):
                try:
                    adapter_path = self.pull_latest(output_path=pull_dir)
                    break
                except AggregationPendingError:
                    if attempt < _MAX_RETRIES - 1:
                        logger.warning("Aggregated model not ready, retrying (%d/%d)...", attempt + 1, _MAX_RETRIES)
                        time.sleep(2 ** attempt)
                        continue
                    raise

            rounds_completed += 1
            if on_round_complete:
                on_round_complete(rounds_completed, result)

    def _poll_for_aggregation(self, current_round: int) -> None:
        """Fallback: poll the server status until the round advances."""
        for _ in range(60):  # up to ~5 minutes with 5s intervals
            try:
                st = self.status()
                if st.get("latest_aggregated_round") is not None and st["latest_aggregated_round"] >= current_round:
                    return
            except ChorusError:
                pass
            time.sleep(5)
        logger.warning("Timed out waiting for aggregation via polling")

    def listen(self, on_round_complete: Callable | None = None) -> Generator[dict, None, None]:
        """Connect via WebSocket and listen for server events.

        Automatically reconnects on connection drops (up to 3 attempts).

        Args:
            on_round_complete: Optional callback(event_dict) called when a round completes.

        Yields:
            Event dicts as they arrive.
        """
        import websockets.sync.client as ws_sync
        from websockets.exceptions import ConnectionClosed

        ws_url = self.server.replace("http://", "ws://").replace("https://", "wss://")
        cid = self.client_id or "listener"
        ws_endpoint = f"{ws_url}/ws/{cid}"
        if self.api_key:
            ws_endpoint += f"?token={self.api_key}"
        reconnect_attempts = 0

        while reconnect_attempts < _MAX_RETRIES:
            try:
                with ws_sync.connect(ws_endpoint) as ws:
                    reconnect_attempts = 0  # reset on successful connect
                    while True:
                        msg = json.loads(ws.recv(timeout=300))
                        if on_round_complete and msg.get("event") == "round_complete":
                            on_round_complete(msg)
                        yield msg
            except (ConnectionClosed, OSError):
                # ConnectionClosed = dropped during recv
                # OSError (including ConnectionRefusedError) = failed to reconnect
                reconnect_attempts += 1
                if reconnect_attempts < _MAX_RETRIES:
                    logger.warning(
                        "WebSocket disconnected, reconnecting (%d/%d)...",
                        reconnect_attempts, _MAX_RETRIES,
                    )
                    time.sleep(2)
                    continue
                logger.error("WebSocket reconnection failed after %d attempts", _MAX_RETRIES)
                raise
            except TimeoutError:
                logger.warning("WebSocket recv timed out (300s), reconnecting...")
                reconnect_attempts += 1
                if reconnect_attempts < _MAX_RETRIES:
                    continue
                raise

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
