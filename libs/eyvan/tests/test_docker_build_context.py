# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Docker build contexts must work from the standalone repository root.

The repository root is the package root: every COPY source is relative to it,
and the documented builds pass it as the context.
"""

from __future__ import annotations

import shlex
from pathlib import Path

EYVAN_ROOT = Path(__file__).resolve().parents[1]


def _local_copy_sources(dockerfile: Path) -> tuple[str, ...]:
    sources: list[str] = []
    logical_lines: list[str] = []
    pending = ""
    for raw_line in dockerfile.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        pending = f"{pending} {line}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical_lines.append(pending)
        pending = ""
    assert not pending
    for line in logical_lines:
        if not line.startswith("COPY ") or "--from=" in line:
            continue
        words = shlex.split(line)
        arguments = [word for word in words[1:] if not word.startswith("--")]
        assert len(arguments) >= 2, f"invalid COPY in {dockerfile.name}: {line}"
        sources.extend(arguments[:-1])
    return tuple(sources)


def test_all_release_dockerfiles_use_the_eyvan_package_directory_as_context() -> None:
    """Every local COPY source exists when the context is the repository root."""

    for name in ("Dockerfile", "Dockerfile.agent", "Dockerfile.worker"):
        dockerfile = EYVAN_ROOT / name
        sources = _local_copy_sources(dockerfile)
        assert sources
        for source in sources:
            assert not source.startswith("libs/eyvan"), (
                f"{name} requires a workspace-root context instead of the documented package context"
            )
            matches = tuple(EYVAN_ROOT.glob(source))
            assert matches, f"{name} COPY source does not exist: {source}"
            assert all(path.is_relative_to(EYVAN_ROOT) for path in matches)


def test_dockerfile_comments_document_both_context_shapes() -> None:
    for name in ("Dockerfile", "Dockerfile.agent", "Dockerfile.worker"):
        content = (EYVAN_ROOT / name).read_text(encoding="utf-8")
        assert f"-f {name}" in content
        assert f"-f libs/eyvan/{name}" in content
        assert "libs/eyvan" in content


def _documented_docker_builds(document: Path) -> tuple[tuple[str, str], ...]:
    """Return ``(dockerfile, context)`` from continued build commands."""

    commands: list[tuple[str, str]] = []
    pending = ""
    for raw_line in document.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        pending = f"{pending} {line}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        if pending.startswith("docker build "):
            words = shlex.split(pending)
            commands.append((words[words.index("-f") + 1], words[-1]))
        pending = ""
    assert not pending
    return tuple(commands)


def test_documented_build_commands_use_the_context_their_dockerfile_expects() -> None:
    commands = tuple(
        command
        for document in (EYVAN_ROOT / "README.md", EYVAN_ROOT / "deploy/helm/eyvan/README.md")
        for command in _documented_docker_builds(document)
    )

    for dockerfile, context in commands:
        assert context == ".", (dockerfile, context)
    for name in ("Dockerfile", "Dockerfile.agent", "Dockerfile.worker"):
        assert (name, ".") in commands
