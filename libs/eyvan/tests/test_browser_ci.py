# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The browser qualification is a blocking, reproducible local-provider CI lane."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_browser_qualification_runs_after_build_with_installed_python() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yaml").read_text())
    steps = workflow["jobs"]["python"]["steps"]
    browser = next(step for step in steps if step.get("name") == "Browser BFF qualification")
    assert browser["if"] == "matrix.python-version == '3.11'"
    assert not browser.get("continue-on-error", False)
    assert "npm --prefix dashboard run test:browser" in browser["run"]
    assert '--python "$PWD/.venv/bin/python"' in browser["run"]
    assert '--output "$PWD/browser-results/bff.json"' in browser["run"]
    before = steps[: steps.index(browser)]
    assert any(step.get("run") == "npm --prefix dashboard run build" for step in before)
    assert any("--require-hashes -r requirements-dev.txt" in step.get("run", "") for step in before)
    assert any("browser:install -- --with-deps" in step.get("run", "") for step in before)
    receipts = next(step for step in steps if step.get("name") == "Retain browser qualification receipt")
    assert receipts["if"] == "!cancelled() && matrix.python-version == '3.11'"
    assert receipts["with"]["path"] == "browser-results/"


def test_browser_dependency_is_exact_and_the_harness_is_formatted() -> None:
    package = json.loads((ROOT / "dashboard/package.json").read_text())
    assert package["devDependencies"]["playwright"] == "1.63.0"
    assert package["scripts"]["test:browser"] == "node e2e/bff.mjs"
    assert "e2e/**/*.mjs" in package["scripts"]["format:check"]
    assert "e2e/**/*.mjs" in package["scripts"]["format"]
