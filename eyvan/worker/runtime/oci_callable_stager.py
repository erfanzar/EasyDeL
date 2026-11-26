# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Stage the dependency-light callable runtime into OCI workload mounts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from eyvan._internal import plane_tls, workspace

OCI_CALLABLE_RUNNER_CONTAINER_DIRECTORY = "/opt/eyvan-native/oci-callable-runner"
"""Read-only container path holding the staged callable runtime."""

OCI_CALLABLE_RUNNER_FILENAME = "eyvan_oci_callable_worker.py"
"""Stable runner filename used by the native OCI bridge."""

OCI_CALLABLE_PROTOCOL_FILENAME = "callable_protocol.py"
OCI_WORKSPACE_FILENAME = "workspace.py"
OCI_HTTP_RETRY_FILENAME = "http_retry.py"
OCI_PLANE_TLS_FILENAME = "plane_tls.py"
"""Shared protocol sources staged beside the OCI callable runner."""

OCI_DEPENDENCY_ENTRY_FILENAME = "dependency_entry.py"
OCI_WORKSPACE_ENTRY_FILENAME = "workspace_entry.py"
"""Per-job wrappers (``--with`` packages, ``--workspace`` trees) staged for images without Eyvan."""


def _stage_file(source: Path, destination: Path) -> None:
    """Atomically copy one immutable runtime source file into the state directory."""

    try:
        contents = source.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"could not read OCI callable runtime source {source}: {exc}") from exc
    try:
        if destination.is_file() and destination.read_bytes() == contents:
            destination.chmod(0o644)
            return
    except OSError as exc:
        raise RuntimeError(f"could not inspect OCI callable runtime file {destination}: {exc}") from exc

    descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        descriptor, raw_temporary_path = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
        temporary_path = Path(raw_temporary_path)
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except OSError as exc:
        raise RuntimeError(f"could not stage OCI callable runtime file at {destination}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def stage_oci_callable_runner(directory: str | os.PathLike[str]) -> Path:
    """Stage the OCI runner and shared protocol without exposing host packages.

    The two source files contain no Eyvan import.  The Docker executor mounts
    this directory read-only, so the workload receives only the bridge and not
    the host virtual environment or host credentials.
    """

    destination_directory = Path(directory).expanduser()
    if not destination_directory.is_absolute():
        raise ValueError("OCI callable runner directory must be absolute")
    try:
        destination_directory.mkdir(parents=True, exist_ok=True, mode=0o755)
        destination_directory.chmod(0o755)
    except OSError as exc:
        raise RuntimeError(f"could not create OCI callable runner directory {destination_directory}: {exc}") from exc

    source_directory = Path(__file__).parent
    protocol_destination = destination_directory / OCI_CALLABLE_PROTOCOL_FILENAME
    runner_destination = destination_directory / OCI_CALLABLE_RUNNER_FILENAME
    _stage_file(source_directory / OCI_CALLABLE_PROTOCOL_FILENAME, protocol_destination)
    _stage_file(source_directory / "oci_callable_runner.py", runner_destination)
    # callable_protocol imports the hardened workspace extractor as a sibling
    # once staged, so the single copy of the tar-slip defence travels with it.
    # Resolve it through the module rather than by a path hop: this stager runs
    # on the host and may import Eyvan, and a hop silently breaks whenever the
    # package layout changes.
    _stage_file(
        Path(workspace.__file__).resolve(),
        destination_directory / OCI_WORKSPACE_FILENAME,
    )
    # Same reasoning for the retry policy: callable_protocol imports it as a
    # sibling once staged, so the container gets the one policy the native
    # worker uses rather than a second copy that can drift from it.
    _stage_file(
        source_directory / OCI_HTTP_RETRY_FILENAME,
        destination_directory / OCI_HTTP_RETRY_FILENAME,
    )
    # And the control-plane CA trust, so a containerized callable verifies a
    # head's private CA exactly as a native one does. Stdlib-only by design.
    _stage_file(Path(plane_tls.__file__).resolve(), destination_directory / OCI_PLANE_TLS_FILENAME)
    # The command-job wrappers. The controller launches them as
    # ``python -I -m eyvan.worker.runtime.<wrapper>``, which an image without
    # Eyvan cannot import; the Docker executor rewrites them to these copies.
    # Both are stdlib-only (workspace_entry imports the siblings staged above).
    for filename in (OCI_DEPENDENCY_ENTRY_FILENAME, OCI_WORKSPACE_ENTRY_FILENAME):
        _stage_file(source_directory / filename, destination_directory / filename)
    return runner_destination


__all__ = (
    "OCI_CALLABLE_PROTOCOL_FILENAME",
    "OCI_CALLABLE_RUNNER_CONTAINER_DIRECTORY",
    "OCI_CALLABLE_RUNNER_FILENAME",
    "OCI_DEPENDENCY_ENTRY_FILENAME",
    "OCI_HTTP_RETRY_FILENAME",
    "OCI_PLANE_TLS_FILENAME",
    "OCI_WORKSPACE_ENTRY_FILENAME",
    "stage_oci_callable_runner",
)
