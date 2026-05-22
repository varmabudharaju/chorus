"""Integration tests for benchmarks/run_all.py with EvalRunner stubbed out."""

from pathlib import Path
import textwrap

import pytest


def _write_sweep(tmp_path: Path) -> Path:
    p = tmp_path / "tiny_sweep.yaml"
    p.write_text(textwrap.dedent("""
        model_id: tiny
        dataset: {name: synthetic-tiny, split: train}
        num_rounds: 1
        strategies: [fedex-lora]
        seeds: [0]
        rank: 4
        sweep:
          num_clients: [2, 5]
    """).strip() + "\n")
    return p


def test_run_config_dispatches_one_runner_per_combo(tmp_path, monkeypatch):
    """expand_sweep yields 2 combos -> EvalRunner.run is called twice."""
    from benchmarks import run_all

    call_log: list[dict] = []

    class StubRunner:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            # Capture cfg + write a minimal report.json so downstream tooling can find it.
            Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.cfg.output_dir) / "report.json").write_text("{}")
            call_log.append({
                "num_clients": self.cfg.num_clients,
                "output_dir": self.cfg.output_dir,
            })

    monkeypatch.setattr(run_all, "EvalRunner", StubRunner)

    yaml_path = _write_sweep(tmp_path)
    out_root = tmp_path / "out"
    out_dirs = run_all.run_config(yaml_path, out_root)

    assert len(out_dirs) == 2
    assert len(call_log) == 2
    seen_clients = sorted(c["num_clients"] for c in call_log)
    assert seen_clients == [2, 5]

    # Output directories are per run_key and exist on disk
    for out_dir in out_dirs:
        assert out_dir.exists()
        assert (out_dir / "report.json").exists()


def test_run_config_writes_per_run_subdirs(tmp_path, monkeypatch):
    """Output layout: <output_root>/<config_stem>/<run_key>/"""
    from benchmarks import run_all

    class StubRunner:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.cfg.output_dir) / "report.json").write_text("{}")

    monkeypatch.setattr(run_all, "EvalRunner", StubRunner)

    yaml_path = _write_sweep(tmp_path)
    out_root = tmp_path / "out"
    run_all.run_config(yaml_path, out_root)

    # Stem of the YAML file = tiny_sweep
    config_dir = out_root / "tiny_sweep"
    assert config_dir.exists()
    subdirs = sorted(p.name for p in config_dir.iterdir() if p.is_dir())
    assert subdirs == ["num_clients=2", "num_clients=5"]


def test_main_all_skips_smoke_yaml(tmp_path, monkeypatch):
    """--all iterates every YAML in configs/ except smoke.yaml."""
    from benchmarks import run_all

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

    # Two real configs + smoke (which must be skipped)
    for name in ["a.yaml", "b.yaml", "smoke.yaml"]:
        (configs_dir / name).write_text(textwrap.dedent(f"""
            model_id: {name}
            dataset: {{name: synthetic-tiny, split: train}}
            num_clients: 2
            num_rounds: 1
            strategies: [fedex-lora]
            rank: 4
            seeds: [0]
        """).strip() + "\n")

    seen_model_ids: list[str] = []

    class StubRunner:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self):
            Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.cfg.output_dir) / "report.json").write_text("{}")
            seen_model_ids.append(self.cfg.model_id)

    monkeypatch.setattr(run_all, "EvalRunner", StubRunner)

    run_all.main([
        "--all",
        "--configs-dir", str(configs_dir),
        "--output-root", str(tmp_path / "out"),
    ])

    assert sorted(seen_model_ids) == ["a.yaml", "b.yaml"]
    assert "smoke.yaml" not in seen_model_ids
