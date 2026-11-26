# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute the CI secret gate against tracked, nonfunctional secret fixtures."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]


def _run(arguments: list[str], root: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, cwd=root, env=env, capture_output=True, text=True, check=False, timeout=30)


def _checkout(root: Path) -> dict[str, str]:
    """Isolate Git and replace only the uv launcher, not the real scanner."""
    bin_path = root / "bin"
    bin_path.mkdir()
    launcher = bin_path / "uv"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "assert sys.argv[1:4] == ['run', '--no-sync', 'python'], sys.argv\n"
        "os.execv(sys.executable, [sys.executable, *sys.argv[4:]])\n",
        encoding="utf-8",
    )
    launcher.chmod(0o700)
    guard = root / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import sys\n"
        "def deny_network(event, args):\n"
        "    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:\n"
        "        raise RuntimeError('Network is forbidden in the secret gate regression')\n"
        "sys.addaudithook(deny_network)\n",
        encoding="utf-8",
    )
    env = {"PATH": f"{bin_path}{os.pathsep}{os.defpath}", "HOME": str(root), "PYTHONPATH": str(guard)}
    result = _run(["git", "init", "--quiet"], root, env)
    assert result.returncode == 0, result.stderr
    (root / ".secrets.baseline").write_bytes((_ROOT / ".secrets.baseline").read_bytes())
    (root / ".gitignore").write_text("bin/\nguard/\nignored.pem\n", encoding="utf-8")
    return env


def _gate(root: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yaml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["secrets"]["steps"]
    command = next(step["run"] for step in steps if step.get("name") == "No new secrets against the baseline")
    # GitHub's bash step uses these same error-handling flags. No uv install or
    # network resolution is performed: the launcher runs the installed scanner.
    return _run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", command], root, env)


@pytest.mark.parametrize("filename", ["new-key.pem", "new key\nfixture.pem", "--new-key.pem"])
def test_actual_ci_gate_rejects_new_tracked_private_key_marker(tmp_path: Path, filename: str) -> None:
    env = _checkout(tmp_path)
    marker = "-----BEGIN " + "RSA PRIVATE KEY-----\nnonfunctional regression marker\n"
    (tmp_path / filename).write_text(marker, encoding="utf-8")
    tracked = _run(["git", "add", "--", ".secrets.baseline", ".gitignore", filename], tmp_path, env)
    assert tracked.returncode == 0, tracked.stderr
    control = _run(
        [
            sys.executable,
            "-m",
            "detect_secrets.pre_commit_hook",
            "--baseline",
            ".secrets.baseline",
            "--no-verify",
            "--",
            filename,
        ],
        tmp_path,
        env,
    )
    assert control.returncode == 1, control.stdout + control.stderr
    assert "Private Key" in control.stdout + control.stderr

    result = _gate(tmp_path, env)

    assert result.returncode != 0, "CI accepted a tracked private-key marker despite the positive scanner control"
    assert "Private Key" in result.stdout + result.stderr, result.stdout + result.stderr


def test_actual_ci_gate_accepts_clean_checkout_without_scanning_ignored_files(tmp_path: Path) -> None:
    env = _checkout(tmp_path)
    (tmp_path / "clean file.txt").write_text("No credentials here.\n", encoding="utf-8")
    (tmp_path / "ignored.pem").write_text("-----BEGIN " + "RSA PRIVATE KEY-----\n", encoding="utf-8")
    tracked = _run(["git", "add", "--", ".secrets.baseline", ".gitignore", "clean file.txt"], tmp_path, env)
    assert tracked.returncode == 0, tracked.stderr

    result = _gate(tmp_path, env)

    assert result.returncode == 0, result.stdout + result.stderr
