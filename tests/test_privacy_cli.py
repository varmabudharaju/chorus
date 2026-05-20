"""Tests for the `chorus privacy` CLI subcommand."""

from click.testing import CliRunner

from chorus.cli.main import cli


def test_privacy_budget_help_lists_required_options():
    runner = CliRunner()
    res = runner.invoke(cli, ["privacy", "budget", "--help"])
    assert res.exit_code == 0
    assert "--client-id" in res.output
    assert "--model-id" in res.output
    assert "--server" in res.output


def test_privacy_budget_requires_client_id():
    runner = CliRunner()
    res = runner.invoke(cli, ["privacy", "budget", "--model-id", "m", "--server", "http://x"])
    assert res.exit_code != 0
    assert "client-id" in res.output.lower() or "client_id" in res.output.lower()
