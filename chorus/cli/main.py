"""Chorus CLI — server, submit, pull, simulate commands."""

from __future__ import annotations

import logging
import sys

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


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
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def server(model, port, host, data_dir, strategy, min_deltas, dp_epsilon, api_key, verbose):
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
    )

    console.print(f"[bold green]Chorus Server[/bold green]")
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
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def submit(server_url, adapter, model_id, client_id, round_id, dp_epsilon, verbose):
    """Submit a LoRA adapter delta to the server."""
    _setup_logging(verbose)

    from chorus.client.sdk import ChorusClient

    # Get model_id from server if not provided
    if model_id is None:
        import httpx
        resp = httpx.get(f"{server_url.rstrip('/')}/health")
        resp.raise_for_status()
        model_id = resp.json()["model_id"]

    client = ChorusClient(
        server=server_url,
        model_id=model_id,
        client_id=client_id,
        dp_epsilon=dp_epsilon,
    )

    with client:
        result = client.submit_delta(adapter_path=adapter, round_id=round_id)

    console.print(f"[bold green]Delta submitted[/bold green]")
    console.print(f"  Round:    {result['round_id']}")
    console.print(f"  Client:   {result['client_id']}")
    console.print(f"  Received: {result['deltas_received']}/{result['min_deltas']}")
    if result["aggregated"]:
        console.print(f"  [bold yellow]Aggregation triggered![/bold yellow]")


@cli.command()
@click.option("--server", "server_url", required=True, help="Server URL")
@click.option("--output", required=True, type=click.Path(), help="Output path")
@click.option("--model-id", default=None, help="Model ID (uses server default if not set)")
@click.option("--round-id", type=int, default=None, help="Specific round (latest if not set)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def pull(server_url, output, model_id, round_id, verbose):
    """Pull the latest aggregated adapter from the server."""
    _setup_logging(verbose)

    from chorus.client.sdk import ChorusClient

    if model_id is None:
        import httpx
        resp = httpx.get(f"{server_url.rstrip('/')}/health")
        resp.raise_for_status()
        model_id = resp.json()["model_id"]

    client = ChorusClient(server=server_url, model_id=model_id)

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
def simulate(clients, rounds, model, strategy, rank, hidden_dim, dp_epsilon, compare, verbose):
    """Run a simulated federation (for testing/demos)."""
    _setup_logging(verbose)

    from chorus.simulate.runner import run_simulation

    console.print(f"[bold green]Chorus Simulation[/bold green]")
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


if __name__ == "__main__":
    cli()
