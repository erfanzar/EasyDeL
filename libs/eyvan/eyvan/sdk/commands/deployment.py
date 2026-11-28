# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Which locally deployed Eyvan components are older than the source tree.

A source checkout deploys three separate surfaces -- the control plane, the cell
agent, and the staged worker wheel -- and nothing about them is linked to the
source they were built from. Updating one and forgetting another produces a
fleet that works and is wrong: the workload succeeds, on code nobody is looking
at.

This module is the single definition of "stale". ``eyvan runtime deploy`` uses
it to verify it left nothing behind, and ``eyvan doctor`` uses it to report
drift. Two implementations of the same question would eventually disagree, and
the one that disagreed quietly would be the one that mattered.
"""

from __future__ import annotations

import json
import subprocess
import typing as tp
from datetime import datetime
from pathlib import Path

import eyvan

WORKER_WHEEL_COMPONENT = "staged worker wheel"


def newest_source_mtime() -> float | None:
    """Return the newest Eyvan source mtime, or ``None`` outside a checkout.

    An installed wheel has no source to drift from, so there is nothing to
    compare against and nothing to report.
    """

    package_root = Path(eyvan.__file__).resolve().parent
    if not (package_root.parent / "pyproject.toml").is_file():
        return None
    return max(
        (path.stat().st_mtime for path in package_root.rglob("*.py") if "__pycache__" not in path.parts),
        default=None,
    )


def process_started_at(pid: int) -> float | None:
    """Return when a PID started, or ``None`` when that cannot be read."""

    completed = subprocess.run(
        ("ps", "-o", "lstart=", "-p", str(pid)),
        capture_output=True,
        text=True,
        check=False,
    )
    stamp = completed.stdout.strip()
    if completed.returncode != 0 or not stamp:
        return None
    try:
        return datetime.strptime(stamp, "%a %b %d %H:%M:%S %Y").timestamp()
    except ValueError:
        return None


def running_component_pids() -> dict[str, int]:
    """Return each locally supervised component's recorded PID.

    Discovered by reading the daemon directory rather than by reconstructing
    endpoints: the control plane and the cell agent are recorded under their
    *own* ports, so deriving one from the other finds neither.
    """

    from eyvan.server.local.daemon import LocalDaemonPaths, LocalDaemonRecord

    directory = LocalDaemonPaths().resolved_directory
    pids: dict[str, int] = {}
    for label, pattern in (("control plane", "control-plane-*.pid"), ("cell agent", "cell-agent-*.pid")):
        for path in sorted(directory.glob(pattern)):
            try:
                record = LocalDaemonRecord.from_json(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
            if record is not None:
                pids[f"{label} ({record.endpoint.url})"] = record.pid
    return pids


def stale_components(
    *,
    runtime_profile_path: str | Path | None,
    newest_source: float | None = None,
    component_pids: tp.Mapping[str, int] | None = None,
) -> tuple[str, ...]:
    """Return every local component running code older than the source tree."""

    newest = newest_source_mtime() if newest_source is None else newest_source
    if newest is None:
        return ()

    stale: list[str] = []
    if runtime_profile_path is not None:
        profile = Path(runtime_profile_path).expanduser()
        try:
            if profile.is_file() and profile.stat().st_mtime < newest:
                stale.append(WORKER_WHEEL_COMPONENT)
        except OSError:
            pass
    pids = running_component_pids() if component_pids is None else component_pids
    for name, pid in pids.items():
        started = process_started_at(pid)
        if started is not None and started < newest:
            stale.append(name)
    return tuple(sorted(stale))


__all__ = (
    "WORKER_WHEEL_COMPONENT",
    "newest_source_mtime",
    "process_started_at",
    "running_component_pids",
    "stale_components",
)
