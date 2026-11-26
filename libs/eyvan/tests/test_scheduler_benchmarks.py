# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Benchmark receipts must be reproducible inputs to a fail-closed CI budget."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_records_measured_samples_and_checks_placement_results(tmp_path) -> None:
    receipt = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_scheduler.py",
            "--cells",
            "12",
            "--iterations",
            "2",
            "--output",
            str(receipt),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    document = json.loads(receipt.read_text())
    assert document["cells"] == 12
    assert document["iterations"] == 2
    assert "diagnostics" not in document
    assert set(document["phases"]) == {"snapshot", "schedule", "elastic_pack"}
    for phase in document["phases"].values():
        assert len(phase["samples_ms"]) == 2
        assert phase["p99_ms"] >= phase["p50_ms"] >= 0
    assert document["violations"] == []


def test_a_budget_regression_fails_and_keeps_its_receipt(tmp_path) -> None:
    budget = tmp_path / "budgets.json"
    budget.write_text(json.dumps({"12": {"snapshot": 0, "schedule": 0, "elastic_pack": 0}}))
    receipt = tmp_path / "failed.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_scheduler.py",
            "--cells",
            "12",
            "--iterations",
            "2",
            "--budget-file",
            str(budget),
            "--output",
            str(receipt),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads(receipt.read_text())["violations"]


def test_opt_in_diagnostics_preserve_the_failing_budget_and_record_cpu_gc(tmp_path) -> None:
    budget = tmp_path / "budgets.json"
    budget.write_text(json.dumps({"12": {"snapshot": 0, "schedule": 0, "elastic_pack": 0}}))
    receipt = tmp_path / "diagnostics.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_scheduler.py",
            "--cells",
            "12",
            "--iterations",
            "2",
            "--diagnostics",
            "--budget-file",
            str(budget),
            "--output",
            str(receipt),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    document = json.loads(receipt.read_text())
    assert len(document["violations"]) == 3
    diagnostics = document["diagnostics"]
    assert diagnostics["cpu_clock"] == "process_time"
    assert set(diagnostics["phases"]) == set(document["phases"])
    for phase in diagnostics["phases"].values():
        assert len(phase["cpu_samples_ms"]) == len(phase["gc_collections"]) == 2
        assert all(value >= 0 for value in phase["cpu_samples_ms"])
        assert all(len(counts) == 3 for counts in phase["gc_collections"])
        assert all(isinstance(count, int) and count >= 0 for counts in phase["gc_collections"] for count in counts)


def test_ci_diagnostics_do_not_replace_or_waive_the_original_budget() -> None:
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yaml").read_text())
    steps = [step for job in workflow["jobs"].values() for step in job.get("steps", [])]
    diagnostics = [step for step in steps if step.get("name") == "Diagnose scheduler budget failure"]
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    budget = next(step for step in steps if step.get("name") == "Scheduler regression budget")
    assert budget["id"] == "scheduler_budget"
    assert not budget.get("continue-on-error", False)
    assert "--diagnostics" not in budget["run"]
    assert "failure()" in diagnostic["if"] and "steps.scheduler_budget.conclusion == 'failure'" in diagnostic["if"]
    assert "--diagnostics" in diagnostic["run"]
    assert "--budget-file scripts/scheduler-budgets.json" in diagnostic["run"]
    assert "--cells 1000 --iterations 20" in diagnostic["run"]
    assert "benchmark-results/scheduler-diagnostics-1000.json" in diagnostic["run"]
