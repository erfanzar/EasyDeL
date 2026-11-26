# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Data files Eyvan locates on disk, rather than imports.

Regrouping the package into layers broke four of these at once, and none of it
showed up in the suite: every one was resolved by counting parent directories
from ``__file__``, which silently encodes how deep a module happens to sit.

    uv build failed: .../libs/eyvan/eyvan does not appear to be a Python project
    alembic: Path doesn't exist: .../eyvan/persistence/migrations
    could not read OCI callable runtime source .../eyvan/worker/workspace.py
    `eyvan dashboard-dev` cannot find .../eyvan/dashboard in a source checkout

These assert the resolved paths exist, so the next layout change fails here
instead of at a live ``eyvan runtime init``.
"""

from __future__ import annotations

import pathlib

import eyvan
from eyvan.sdk.commands.dashboard import _dashboard_root
from eyvan.sdk.commands.runtime import UvNativeWorkerWheelBuilder, _default_package_root
from eyvan.server.persistence.migrate import MIGRATIONS_DIR
from eyvan.worker.runtime.oci_callable_stager import OCI_WORKSPACE_FILENAME


def _repository_root() -> pathlib.Path:
    return pathlib.Path(eyvan.__file__).resolve().parent.parent


def test_the_wheel_builder_points_at_the_directory_holding_pyproject() -> None:
    """``uv build`` needs the project directory, not whatever is two levels up.

    This is the failure that broke ``eyvan runtime init`` outright: the builder
    was handed ``libs/eyvan/eyvan``, which has no ``pyproject.toml``.
    """

    root = _default_package_root()

    assert (root / "pyproject.toml").is_file(), f"{root} has no pyproject.toml"
    assert UvNativeWorkerWheelBuilder().package_root == root


def test_the_alembic_migrations_directory_resolves() -> None:
    """A wrong script_location makes every migration command fail at runtime."""

    assert MIGRATIONS_DIR.is_dir(), f"{MIGRATIONS_DIR} is not a directory"
    assert (MIGRATIONS_DIR / "env.py").is_file()


def test_the_staged_workspace_extractor_resolves() -> None:
    """The OCI bridge stages this file by path; a miss fails every callable run."""

    from eyvan._internal import workspace

    source = pathlib.Path(workspace.__file__).resolve()

    assert source.is_file()
    assert source.name == OCI_WORKSPACE_FILENAME


def test_the_dashboard_dev_sources_resolve_from_a_source_checkout() -> None:
    """``eyvan dashboard-dev`` runs Vite from these sources."""

    root = _dashboard_root()

    assert root == _repository_root() / "dashboard"
    assert (root / "package.json").is_file()
