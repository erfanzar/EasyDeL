# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute the credential-restoration runbooks without starting any daemon."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = ("docs/tutorial/14-operations.md", "docs/operate.md")


def _restart_snippet(document: str) -> str:
    text = (ROOT / document).read_text()
    blocks = re.findall(r"^[ \t]*```bash\n(.*?)^[ \t]*```", text, re.MULTILINE | re.DOTALL)
    matches = [block for block in blocks if "EYVAN_AGENT_SHARED_SECRET" in block and "eyvan-server" in block]
    assert len(matches) == 1, "The credential-restoration command must remain an executable runbook block"
    return textwrap.dedent(matches[0])


@pytest.mark.parametrize("document", DOCUMENTS)
@pytest.mark.parametrize("errexit", [False, True], ids=["normal-shell", "errexit-shell"])
@pytest.mark.parametrize(
    "profile_kind", ["valid", "missing", "malformed", "missing-secret", "empty-secret", "null-secret"]
)
def test_restart_runbook_restores_secret_or_refuses_startup(
    tmp_path: Path, document: str, errexit: bool, profile_kind: str
) -> None:
    home = tmp_path / "disposable-home"
    profile = home / ".eyvan/gcp-native-runtime.json"
    profile.parent.mkdir(parents=True)
    expected_secret = "disposable test secret with spaces"
    if profile_kind == "malformed":
        profile.write_text("not JSON")
    elif profile_kind != "missing":
        values: dict[str, str | None] = {}
        if profile_kind == "valid":
            values["agent_shared_secret"] = expected_secret
        elif profile_kind == "empty-secret":
            values["agent_shared_secret"] = ""
        elif profile_kind == "null-secret":
            values["agent_shared_secret"] = None
        profile.write_text(json.dumps(values))

    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "python3").symlink_to(sys.executable)
    server = binaries / "eyvan-server"
    server.write_text('#!/bin/sh\nprintf "%s" "$EYVAN_AGENT_SHARED_SECRET" > "$RESTART_RECEIPT"\n')
    server.chmod(0o700)
    receipt = tmp_path / "server-started-secret"
    command = ["/bin/bash", *(["-e"] if errexit else []), "-c", _restart_snippet(document)]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env={"HOME": str(home), "PATH": str(binaries), "RESTART_RECEIPT": str(receipt)},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    if profile_kind == "valid":
        assert result.returncode == 0, result.stderr
        assert receipt.is_file(), result.stderr
        assert receipt.read_text() == expected_secret
    else:
        assert not receipt.exists(), "Invalid credentials must not launch even a stub server"
        assert result.returncode != 0, "Credential extraction must not report successful recovery"
        assert result.stderr, "Refusal must leave an operator-visible diagnostic"
        assert expected_secret not in result.stderr
