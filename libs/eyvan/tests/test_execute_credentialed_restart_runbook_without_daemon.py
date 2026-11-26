# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute the credentialed-restart runbook without a daemon or real credentials."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _restart_snippet() -> str:
    runbook = (Path(__file__).resolve().parents[1] / "docs" / "operate.md").read_text(encoding="utf-8")
    contract = runbook.split("**Restart with credentials, or don't restart.**", 1)[1]
    match = re.search(r"```bash\n(.*?)\n\s*```", contract, re.DOTALL)
    assert match is not None, "credentialed-restart contract must contain an executable Bash snippet"
    return textwrap.dedent(match.group(1))


@pytest.mark.parametrize("profile", ["valid", "missing", "malformed", "empty", "null", "wrong-type"])
def test_documented_restart_requires_successful_credential_extraction(tmp_path: Path, profile: str) -> None:
    home = tmp_path / "home"
    home.mkdir()
    profile_path = home / ".eyvan" / "gcp-native-runtime.json"
    profile_path.parent.mkdir()
    secret = "mock secret with spaces; $never_evaluate"
    if profile == "malformed":
        profile_path.write_text("not-json", encoding="utf-8")
    elif profile != "missing":
        value = {"valid": secret, "empty": "", "null": None, "wrong-type": 123}[profile]
        profile_path.write_text(json.dumps({"agent_shared_secret": value}), encoding="utf-8")

    # Restrict PATH to the chosen interpreter; eyvan-server is only a local
    # reachability marker, so the copied runbook cannot contact an agent or cloud.
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    (bin_path / "python3").symlink_to(sys.executable)
    marker = tmp_path / "startup-secret"
    server = bin_path / "eyvan-server"
    server.write_text('#!/bin/bash\nprintf "%s" "$EYVAN_AGENT_SHARED_SECRET" > "$STARTUP_MARKER"\n', encoding="utf-8")
    server.chmod(0o700)
    result = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", _restart_snippet()],
        cwd=tmp_path,
        env={"HOME": str(home), "PATH": str(bin_path), "STARTUP_MARKER": str(marker)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if profile == "valid":
        assert result.returncode == 0, result.stderr
        assert marker.read_text(encoding="utf-8") == secret
        assert result.stderr == ""
    else:
        assert not marker.exists(), f"startup was reached after {profile} credential extraction: {result.stderr}"
        assert result.returncode != 0, "failed credential extraction must not report success"
    assert secret not in result.stdout + result.stderr
