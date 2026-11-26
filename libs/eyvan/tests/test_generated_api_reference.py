# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generated API docs stay tied to the schema the application actually serves."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_api_reference_generation_is_deterministic_and_detects_drift(tmp_path) -> None:
    command = [sys.executable, "scripts/generate_api_reference.py", "--output-dir", str(tmp_path)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    schema = json.loads((tmp_path / "openapi.json").read_text())
    assert "/api/v1/workloads" in schema["paths"]
    assert schema["components"]["schemas"]
    index = (tmp_path / "api-generated.md").read_text()
    assert "/api/v1/workloads" in index and "openapi.json" in index
    checked = subprocess.run([*command, "--check"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    (tmp_path / "openapi.json").write_text("{}\n")
    stale = subprocess.run([*command, "--check"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert stale.returncode == 1
    assert json.loads((tmp_path / "openapi.json").read_text()) == {}, "check mode must not rewrite files"
