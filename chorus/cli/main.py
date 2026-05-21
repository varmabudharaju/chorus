"""Chorus CLI — server, submit, pull, simulate, train, status, export commands."""

from __future__ import annotations

import functools
import logging
import sys

import click
from rich.console import Console
from rich.table import Table

from chorus.exceptions import (
    AggregationPendingError,
    ChorusError,
    ServerUnreachableError,
)

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def handle_errors(f):
    """Decorator that catches Chorus exceptions and prints clean error messages."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ServerUnreachableError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            console.print("[dim]Is the Chorus server running?[/dim]")
            sys.exit(1)
        except AggregationPendingError as exc:
            console.print(f"[yellow]Warning:[/yellow] {exc}")
            sys.exit(1)
        except ChorusError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            sys.exit(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            sys.exit(130)

    return wrapper


@click.group()
@click.version_option(package_name="chorus")
def cli():
    """Chorus — Federated LoRA Adapter Aggregation Framework."""
    pass


@cli.command()
@click.option("--model", required=True, help="Model ID (e.g. meta-llama/Llama-3.2-3B)")
@click.option("--port", default=8080, help="Port to listen on")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--data-dir", default="./chorus_data", help="Data directory for storage")
@click.option(
    "--strategy",
    type=click.Choice(["fedavg", "fedex-lora"]),
    default="fedex-lora",
    help="Aggregation strategy",
)
@click.option("--min-deltas", default=2, help="Minimum deltas before aggregation triggers")
@click.option("--dp-epsilon", type=float, default=None, help="Server-side DP epsilon (disabled if not set)")
@click.option("--api-key", multiple=True, help="API key(s) for authentication (can specify multiple)")
@click.option("--base-weights", type=click.Path(exists=True), default=None, help="Path to base model weights (.safetensors)")
@click.option("--norm-bound", type=float, default=None, help="Max L2 norm for Byzantine defense (disabled if not set)")
@click.option("--outlier-threshold", type=float, default=None, help="Z-score threshold for outlier detection (disabled if not set)")
@click.option("--rate-limit", type=int, default=0, help="Max requests per minute per IP (0 = disabled)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@handle_errors
def server(model, port, host, data_dir, strategy, min_deltas, dp_epsilon, api_key, base_weights, norm_bound, outlier_threshold, rate_limit, verbose):
    """Start the Chorus aggregation server."""
    _setup_logging(verbose)

    from chorus.server.app import configure

    configure(
        model_id=model,
        data_dir=data_dir,
        strategy=strategy,
        min_deltas=min_deltas,
        dp_epsilon=dp_epsilon,
        api_keys=list(api_key) if api_key else None,
        norm_bound=norm_bound,
        outlier_threshold=outlier_threshold,
        rate_limit=rate_limit,
    )

    # Load initial base weights if provided
    if base_weights:
        from safetensors.torch import load_file
        from chorus.server.app import state
        tensors = load_file(base_weights)
        state.storage.save_base_weights(model, tensors, meta={"source": base_weights})
        console.print(f"  Base weights: {base_weights} ({len(tensors)} tensors)")

    console.print("[bold green]Chorus Server[/bold green]")
    console.print(f"  Model:    {model}")
    console.print(f"  Strategy: {strategy}")
    console.print(f"  Min deltas: {min_deltas}")
    console.print(f"  DP epsilon: {dp_epsilon or 'disabled'}")
    console.print(f"  Listening: {host}:{port}")
    console.print()

    import uvicorn
    from chorus.server.app import app

    uvicorn.run(app, host=host, port=port, log_level="info" if not verbose else "debug")


@cli.command()
@click.option("--server", "server_url", required=True, help="Server URL")
@click.option("--adapter", required=True, type=click.Path(exists=True), help="Path to adapter")
@click.option("--model-id", default=None, help="Model ID (uses server default if not set)")
@click.option("--client-id", default=None, help="Client ID (auto-generated if not set)")
@click.option("--round-id", type=int, default=None, help="Round ID (uses current if not set)")
@click.option("--dp-epsilon", type=float, default=None, help="Local DP epsilon")
@click.option("--dataset-size", type=int, default=None, help="Dataset size for weighted aggregation")
@click.option("--api-key", default=None, help="API key for server authentication")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@handle_errors
def submit(server_url, adapter, model_id, client_id, round_id, dp_epsilon, dataset_size, api_key, verbose):
    """Submit a LoRA adapter delta to the server."""
    _setup_logging(verbose)

    from chorus.client.sdk import ChorusClient

    # Get model_id from server if not provided
    if model_id is None:
        import httpx
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = httpx.get(f"{server_url.rstrip('/')}/health", headers=headers)
        resp.raise_for_status()
        model_id = resp.json()["model_id"]

    client = ChorusClient(
        server=server_url,
        model_id=model_id,
        client_id=client_id,
        api_key=api_key,
        dp_epsilon=dp_epsilon,
    )

    with client:
        result = client.submit_delta(adapter_path=adapter, round_id=round_id, dataset_size=dataset_size)

    console.print("[bold green]Delta submitted[/bold green]")
    console.print(f"  Round:    {result['round_id']}")
    console.print(f"  Client:   {result['client_id']}")
    console.print(f"  Received: {result['deltas_received']}/{result['min_deltas']}")
    if result["aggregated"]:
        console.print("  [bold yellow]Aggregation triggered![/bold yellow]")


@cli.command()
@click.option("--server", "server_url", required=True, help="Server URL")
@click.option("--output", required=True, type=click.Path(), help="Output path")
@click.option("--model-id", default=None, help="Model ID (uses server default if not set)")
@click.option("--round-id", type=int, default=None, help="Specific round (latest if not set)")
@click.option("--api-key", default=None, help="API key for server authentication")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@handle_errors
def pull(server_url, output, model_id, round_id, api_key, verbose):
    """Pull the latest aggregated adapter from the server."""
    _setup_logging(verbose)

    from chorus.client.sdk import ChorusClient

    if model_id is None:
        import httpx
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = httpx.get(f"{server_url.rstrip('/')}/health", headers=headers)
        resp.raise_for_status()
        model_id = resp.json()["model_id"]

    client = ChorusClient(server=server_url, model_id=model_id, api_key=api_key)

    with client:
        if round_id is not None:
            path = client.pull_round(round_id, output)
        else:
            path = client.pull_latest(output)

    console.print(f"[bold green]Adapter pulled[/bold green] → {path}")


@cli.command()
@click.option("--clients", default=5, help="Number of simulated clients")
@click.option("--rounds", default=3, help="Number of federation rounds")
@click.option("--model", default=None, help="Model name (cosmetic, uses synthetic data)")
@click.option(
    "--strategy",
    type=click.Choice(["fedavg", "fedex-lora"]),
    default="fedex-lora",
    help="Aggregation strategy",
)
@click.option("--rank", default=8, help="LoRA rank")
@click.option("--hidden-dim", default=256, help="Hidden dimension")
@click.option("--dp-epsilon", type=float, default=None, help="DP epsilon per client")
@click.option("--compare", is_flag=True, help="Compare FedAvg vs FedEx-LoRA")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@handle_errors
def simulate(clients, rounds, model, strategy, rank, hidden_dim, dp_epsilon, compare, verbose):
    """Run a simulated federation (for testing/demos)."""
    _setup_logging(verbose)

    from chorus.simulate.runner import run_simulation

    console.print("[bold green]Chorus Simulation[/bold green]")
    console.print(f"  Clients:  {clients}")
    console.print(f"  Rounds:   {rounds}")
    console.print(f"  Strategy: {'comparison' if compare else strategy}")
    console.print(f"  Rank:     {rank}")
    console.print(f"  DP:       {dp_epsilon or 'disabled'}")
    console.print()

    result = run_simulation(
        num_clients=clients,
        num_rounds=rounds,
        strategy=strategy,
        rank=rank,
        hidden_dim=hidden_dim,
        dp_epsilon=dp_epsilon,
        compare_strategies=compare,
    )

    if compare:
        table = Table(title="FedAvg vs FedEx-LoRA Comparison")
        table.add_column("Round", style="cyan")
        table.add_column("FedAvg Error", style="red")
        table.add_column("FedEx-LoRA Error", style="green")
        table.add_column("Improvement", style="yellow")

        for r in result.rounds:
            table.add_row(
                str(r["round_id"]),
                f"{r['fedavg_error']:.6f}",
                f"{r['fedex_error']:.6f}",
                r["improvement"],
            )
        console.print(table)
    else:
        console.print(result.summary())

    console.print("\n[bold green]Simulation complete.[/bold green]")


@cli.command()
@click.option("--server", "server_url", required=True, help="Server URL")
@click.option("--model", "base_model", required=True, help="HF model ID for training")
@click.option("--dataset", required=True, help="HF dataset name or local path")
@click.option("--rounds", default=None, type=int, help="Number of rounds (infinite if not set)")
@click.option("--output-dir", default="./chorus_adapter", help="Output directory for adapter")
@click.option("--lora-rank", default=16, help="LoRA rank")
@click.option("--max-steps", default=-1, help="Max training steps per round (-1 for full epoch)")
@click.option("--client-id", default=None, help="Client ID")
@click.option("--dp-epsilon", type=float, default=None, help="Local DP epsilon")
@click.option("--api-key", default=None, help="API key for server authentication")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@handle_errors
def train(server_url, base_model, dataset, rounds, output_dir, lora_rank, max_steps, client_id, dp_epsilon, api_key, verbose):
    """Run the full federated training loop: train -> submit -> wait -> pull -> repeat."""
    _setup_logging(verbose)

    from chorus.client.sdk import ChorusClient
    from chorus.client.trainer import LoRATrainer

    # Get model_id from server
    import httpx
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = httpx.get(f"{server_url.rstrip('/')}/health", headers=headers)
    resp.raise_for_status()
    model_id = resp.json()["model_id"]

    trainer = LoRATrainer(
        base_model=base_model,
        dataset=dataset,
        output_dir=output_dir,
        lora_rank=lora_rank,
        max_steps=max_steps,
    )

    client = ChorusClient(
        server=server_url,
        model_id=model_id,
        client_id=client_id,
        api_key=api_key,
        dp_epsilon=dp_epsilon,
    )

    console.print("[bold green]Chorus Federated Training[/bold green]")
    console.print(f"  Server:     {server_url}")
    console.print(f"  Base model: {base_model}")
    console.print(f"  Dataset:    {dataset}")
    console.print(f"  Rounds:     {rounds or 'infinite'}")
    console.print(f"  LoRA rank:  {lora_rank}")
    console.print()

    def on_round_complete(round_num, result):
        console.print(f"  [bold yellow]Round {round_num} complete[/bold yellow]")

    with client:
        client.train_loop(
            trainer=trainer,
            num_rounds=rounds,
            on_round_complete=on_round_complete,
        )


@cli.command()
@click.option("--server", "server_url", required=True, help="Server URL")
@click.option("--api-key", default=None, help="API key for server authentication")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@handle_errors
def status(server_url, api_key, verbose):
    """Show the current status of a Chorus server."""
    _setup_logging(verbose)

    import httpx

    base_url = server_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # Fetch health
    try:
        health_resp = httpx.get(f"{base_url}/health", timeout=10)
        health_resp.raise_for_status()
    except httpx.ConnectError:
        raise ServerUnreachableError(f"Cannot connect to Chorus server at {base_url}")
    except httpx.TimeoutException:
        raise ServerUnreachableError(f"Request to {base_url} timed out")

    health = health_resp.json()
    model_id = health["model_id"]

    # Fetch model status
    try:
        status_resp = httpx.get(f"{base_url}/models/{model_id}/status", timeout=10, headers=headers)
        status_resp.raise_for_status()
        model_status = status_resp.json()
    except Exception:
        model_status = None

    console.print(f"[bold green]Chorus Server:[/bold green] {base_url}")
    console.print(f"  Model:     {health['model_id']}")
    console.print(f"  Strategy:  {health['strategy']}")
    console.print(f"  Clients:   {health['ws_clients']} connected")

    if model_status:
        round_state = model_status.get("round_state", "unknown").upper()
        console.print(f"  Round:     {model_status['current_round']} ({round_state})")
        console.print(f"  Deltas:    {model_status['deltas_submitted']} / {model_status['min_deltas']} received")
        latest = model_status.get("latest_aggregated_round")
        console.print(f"  Last agg:  {'Round ' + str(latest) if latest is not None else 'none'}")

    # Check if privacy accounting is enabled (probe any client endpoint)
    try:
        privacy_resp = httpx.get(
            f"{base_url}/models/{model_id}/clients/__probe__/privacy",
            headers=headers,
            timeout=5.0,
        )
        if privacy_resp.status_code != 404:
            console.print(
                "[dim]Privacy accounting is enabled. "
                "Run `chorus privacy budget --client-id <id> --model-id "
                f"{model_id} --server {base_url}` to see a client's budget.[/dim]"
            )
    except Exception:
        pass


@cli.command(name="export")
@click.option("--server", "server_url", required=True, help="Server URL")
@click.option("--model", "base_model", required=True, help="HuggingFace model ID (for base weights)")
@click.option("--output", required=True, type=click.Path(), help="Output directory for merged model")
@click.option("--round-id", type=int, default=None, help="Specific round (latest if not set)")
@click.option("--api-key", default=None, help="API key for server authentication")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@handle_errors
def export_cmd(server_url, base_model, output, round_id, api_key, verbose):
    """Export a merged model (base + adapter) ready for deployment."""
    _setup_logging(verbose)

    from chorus.client.sdk import ChorusClient

    import httpx
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = httpx.get(f"{server_url.rstrip('/')}/health", timeout=10, headers=headers)
        resp.raise_for_status()
    except httpx.ConnectError:
        raise ServerUnreachableError(f"Cannot connect to Chorus server at {server_url}")
    except httpx.TimeoutException:
        raise ServerUnreachableError(f"Request to {server_url} timed out")

    model_id = resp.json()["model_id"]

    client = ChorusClient(server=server_url, model_id=model_id, api_key=api_key)

    console.print("[bold green]Exporting merged model...[/bold green]")
    console.print(f"  Server:     {server_url}")
    console.print(f"  Base model: {base_model}")
    console.print(f"  Output:     {output}")
    if round_id is not None:
        console.print(f"  Round:      {round_id}")
    console.print()

    with client:
        output_dir = client.export_model(
            base_model=base_model,
            output_dir=output,
            round_id=round_id,
        )

    console.print(f"[bold green]Model exported to:[/bold green] {output_dir}")
    console.print(f"[dim]Load with: AutoModelForCausalLM.from_pretrained('{output_dir}')[/dim]")


@cli.group()
def privacy():
    """Privacy budget management."""


@privacy.command("budget")
@click.option("--client-id", required=True, help="Client identifier")
@click.option("--model-id", required=True, help="Model identifier")
@click.option("--server", required=True, help="Server base URL")
@click.option("--api-key", default=None, help="Bearer token (if server requires auth)")
def privacy_budget(client_id: str, model_id: str, server: str, api_key: str | None):
    """Print the remaining privacy budget for a client on a model."""
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = httpx.get(
            f"{server.rstrip('/')}/models/{model_id}/clients/{client_id}/privacy",
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code == 404:
            console.print("[yellow]Privacy accounting is not enabled on this server.[/yellow]")
            raise SystemExit(0)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        console.print(f"[red]Failed to fetch budget: {e}[/red]")
        raise SystemExit(1)

    table = Table(title=f"Privacy budget — {client_id} on {model_id}")
    table.add_column("Field")
    table.add_column("Value", justify="right")
    table.add_row("epsilon consumed", f"{data['epsilon_consumed']:.4f}")
    table.add_row("epsilon target", f"{data['epsilon_target']:.4f}")
    table.add_row("epsilon remaining", f"{data['epsilon_remaining']:.4f}")
    table.add_row("delta", f"{data['delta']:.2e}")
    table.add_row("exhausted", "YES" if data["exhausted"] else "NO")
    console.print(table)


@cli.command("eval")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to YAML config",
)
@click.option(
    "--check-only",
    is_flag=True,
    default=False,
    help="Validate config + wiring, do NOT load model or train",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Override output directory",
)
def eval_cmd(config: str, check_only: bool, output_dir: str | None):
    """Run an evaluation against a YAML config (see benchmarks/configs/)."""
    from chorus.eval import EvalConfig, EvalRunner
    from chorus.exceptions import EvalConfigError

    try:
        cfg = EvalConfig.from_yaml(config)
    except EvalConfigError as e:
        console.print(f"[red]Config error: {e}[/red]")
        raise SystemExit(1)

    if output_dir is not None:
        cfg.output_dir = output_dir

    runner = EvalRunner(cfg)

    if check_only:
        try:
            runner.check_only()
        except EvalConfigError as e:
            console.print(f"[red]Check-only failed: {e}[/red]")
            raise SystemExit(1)
        console.print(
            f"[green]check-only OK[/green] — model: {cfg.model_id}, "
            f"clients: {cfg.num_clients}, strategies: {cfg.strategies}"
        )
        return

    console.print(
        f"[bold]Running eval[/bold] — {cfg.model_id}, {cfg.num_clients} clients, "
        f"{cfg.num_rounds} rounds"
    )
    report = runner.run()
    console.print(
        f"[green]Eval complete[/green] — {len(report.results)} results, "
        f"output in {cfg.output_dir}"
    )


if __name__ == "__main__":
    cli()
