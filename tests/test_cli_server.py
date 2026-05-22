"""Tests for the `chorus server` CLI — accountant flag wiring."""

from unittest.mock import patch

from click.testing import CliRunner

from chorus.cli.main import cli


def test_server_help_lists_accountant_flags():
    runner = CliRunner()
    res = runner.invoke(cli, ["server", "--help"])
    assert res.exit_code == 0
    assert "--accountant-target-epsilon" in res.output
    assert "--accountant-noise-multiplier" in res.output


def test_server_flags_reach_configure(tmp_path):
    """--accountant-target-epsilon + --accountant-noise-multiplier land in server state."""
    runner = CliRunner()
    with patch("uvicorn.run") as mock_uvicorn, patch("chorus.cli.main.console"):
        res = runner.invoke(cli, [
            "server",
            "--model", "test-model",
            "--data-dir", str(tmp_path),
            "--accountant-target-epsilon", "1.5",
            "--accountant-noise-multiplier", "2.0",
        ])
    assert res.exit_code == 0, res.output
    assert mock_uvicorn.called

    from chorus.server.app import state
    assert state.accountant_target_epsilon == 1.5
    assert state.accountant_noise_multiplier == 2.0


def test_server_rejects_target_epsilon_without_noise_multiplier(tmp_path):
    """Providing one accountant flag without the other should fail with a clear message."""
    runner = CliRunner()
    res = runner.invoke(cli, [
        "server",
        "--model", "test-model",
        "--data-dir", str(tmp_path),
        "--accountant-target-epsilon", "1.5",
    ])
    assert res.exit_code != 0
    assert "must be set together" in res.output.lower() or "must be set together" in (res.stderr or "").lower()


def test_server_rejects_noise_multiplier_without_target_epsilon(tmp_path):
    runner = CliRunner()
    res = runner.invoke(cli, [
        "server",
        "--model", "test-model",
        "--data-dir", str(tmp_path),
        "--accountant-noise-multiplier", "2.0",
    ])
    assert res.exit_code != 0


def test_server_accountant_disabled_by_default(tmp_path):
    """Without the flags, accountant_target_epsilon should remain None in state."""
    runner = CliRunner()
    with patch("uvicorn.run"), patch("chorus.cli.main.console"):
        res = runner.invoke(cli, [
            "server",
            "--model", "test-model",
            "--data-dir", str(tmp_path),
        ])
    assert res.exit_code == 0, res.output

    from chorus.server.app import state
    assert state.accountant_target_epsilon is None
    assert state.accountant_noise_multiplier is None
