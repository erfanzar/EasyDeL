# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Per-job dependencies and workspaces work inside an OCI image that has no Eyvan.

The controller wraps a command as ``python -I -m eyvan.worker.runtime.dependency_entry``
(``--with``) and ``... workspace_entry`` (``--workspace``). Inside an arbitrary
image ``eyvan`` is not importable, so both died before the job started. The OCI
executor now rewrites them to standalone copies staged in its read-only bridge
mount, exactly as it already did for the callable worker.
"""

from __future__ import annotations

import hashlib
import shlex
import subprocess
import sys
from pathlib import Path

from eyvan._internal.workspace import build_workspace
from eyvan.server.controller.workload_commands import _ControllerWorkloadCommands
from eyvan.worker.runtime.oci_callable_stager import OCI_CALLABLE_RUNNER_CONTAINER_DIRECTORY
from tests.test_container_executor import FakeDockerSystemd, _context, _executor, _spec

STAGED = OCI_CALLABLE_RUNNER_CONTAINER_DIRECTORY


def _rendered(tmp_path: Path, entrypoint: str) -> tuple[str, Path]:
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    executor.submit(_spec(entrypoint=entrypoint), _context())
    launch = next(command for command in daemon.commands if command[0] == "systemd-run")
    return launch[-1], tmp_path / "oci-callable-runner"


def test_dependency_and_workspace_wrappers_are_rewritten_to_staged_standalone_copies(tmp_path: Path) -> None:
    workspace = (
        "python -I -m eyvan.worker.runtime.workspace_entry --workspace-sha256 "
        + "b" * 64
        + " --workspace-uri file:///tmp/ws.tar.gz -- /bin/sh -c 'set -e; python train.py'"
    )
    entrypoint = _ControllerWorkloadCommands._dependency_entrypoint(("numpy", "jax>=0.4"), workspace)

    rendered, staged_dir = _rendered(tmp_path, entrypoint)

    parts = shlex.split(rendered)
    assert not any(part.startswith("eyvan.worker.runtime.") for part in parts), rendered
    assert "-m" not in parts
    assert parts[:3] == ["python", "-I", f"{STAGED}/dependency_entry.py"]
    inner = parts.index(f"{STAGED}/workspace_entry.py")
    assert parts[inner - 2 : inner] == ["python", "-I"]
    # The user's shell command still travels as one argument, untouched.
    assert parts[-3:] == ["/bin/sh", "-c", "set -e; python train.py"]
    assert parts[parts.index("--require") + 1] == "numpy"
    assert (staged_dir / "dependency_entry.py").is_file()
    assert (staged_dir / "workspace_entry.py").is_file()


def test_a_callable_with_dependencies_has_both_layers_rewritten(tmp_path: Path) -> None:
    callable_command = shlex.join(
        ("python", "-m", "eyvan.worker.runtime.callable_worker", "--uri", "gs://b/c.pkl", "--sha256", "a" * 64)
    )
    entrypoint = _ControllerWorkloadCommands._dependency_entrypoint(("flax",), callable_command)

    rendered, _ = _rendered(tmp_path, entrypoint)

    parts = shlex.split(rendered)
    assert "eyvan.worker.runtime.dependency_entry" not in parts
    assert "eyvan.worker.runtime.callable_worker" not in parts
    assert f"{STAGED}/dependency_entry.py" in parts and f"{STAGED}/eyvan_oci_callable_worker.py" in parts


def test_a_setup_step_survives_the_container_rewrite_as_one_argument(tmp_path: Path) -> None:
    """An inline setup script may itself contain shell syntax; it is data, not composition."""

    import json

    from eyvan.records import WorkloadSetup

    setup = WorkloadSetup(script="cd /tmp && echo 'a; b' | cat\n", args=("--ref", "a b"))
    entrypoint = _ControllerWorkloadCommands._dependency_entrypoint((), "python train.py", setup)

    rendered, staged_dir = _rendered(tmp_path, entrypoint)

    parts = shlex.split(rendered)
    assert parts[:3] == ["python", "-I", f"{STAGED}/dependency_entry.py"]
    assert json.loads(parts[parts.index("--setup") + 1]) == setup.model_dump(mode="json")
    assert parts[-2:] == ["python", "train.py"]
    assert (staged_dir / "dependency_entry.py").is_file()


def test_unrelated_and_shell_composed_commands_are_left_alone(tmp_path: Path) -> None:
    for entrypoint in ("python train.py --steps 20", "python -m mypkg.train", "python -m json.tool x && echo ok"):
        rendered, _ = _rendered(tmp_path / str(abs(hash(entrypoint))), entrypoint)
        assert rendered == entrypoint


def _bare_python(tmp_path: Path) -> Path:
    """A venv on this interpreter with no site-packages: ``eyvan`` is not importable, as in a foreign image."""

    venv = tmp_path / "bare-venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, timeout=120)
    python = venv / "bin" / "python"
    probe = subprocess.run(
        ["/usr/bin/env", "-i", str(python), "-I", "-c", "import eyvan"], capture_output=True, text=True, check=False
    )
    assert probe.returncode != 0 and "No module named 'eyvan'" in probe.stderr
    return python


def test_the_staged_wrappers_run_on_an_interpreter_without_eyvan(tmp_path: Path) -> None:
    _, staged_dir = _rendered(tmp_path / "executor", "python train.py")
    python = _bare_python(tmp_path)

    source = tmp_path / "src"
    source.mkdir()
    (source / "marker.txt").write_text("shipped workspace\n", encoding="utf-8")
    archive = build_workspace(source, respect_gitignore=False)
    bundle = tmp_path / "ws.tar.gz"
    bundle.write_bytes(archive.payload)
    digest = hashlib.sha256(archive.payload).hexdigest()
    workdir = tmp_path / "workload"
    workdir.mkdir()

    completed = subprocess.run(
        [
            "/usr/bin/env",
            "-i",
            "PATH=/usr/bin:/bin",
            str(python),
            "-I",
            str(staged_dir / "workspace_entry.py"),
            "--workspace-uri",
            f"file://{bundle}",
            "--workspace-sha256",
            digest,
            "--",
            "/bin/sh",
            "-c",
            "cat marker.txt",
        ],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "shipped workspace\n"

    # A tampered digest is still refused before anything runs.
    refused = subprocess.run(
        [
            "/usr/bin/env",
            "-i",
            "PATH=/usr/bin:/bin",
            str(python),
            "-I",
            str(staged_dir / "workspace_entry.py"),
            "--workspace-uri",
            f"file://{bundle}",
            "--workspace-sha256",
            "0" * 64,
            "--",
            "/bin/true",
        ],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert refused.returncode != 0 and "does not match its content-addressed digest" in refused.stderr

    dependency = subprocess.run(
        [
            "/usr/bin/env",
            "-i",
            "PATH=/usr/bin:/bin",
            str(python),
            "-I",
            str(staged_dir / "dependency_entry.py"),
            "--",
            "/bin/true",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    # It loaded standalone and reached its own argument check (no import error).
    assert dependency.returncode == 2 and "at least one --require" in dependency.stderr
