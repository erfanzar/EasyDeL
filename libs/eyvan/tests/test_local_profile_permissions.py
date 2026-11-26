# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The profile directory must be private no matter which component creates it.

``Path.mkdir(mode=..., parents=True)`` applies the mode to the final directory
only; parents it creates get ``0o777 & ~umask``. Creating ``~/.eyvan/autoscale``
therefore left ``~/.eyvan`` itself group-readable under the ordinary 022 umask,
and the credential store -- correctly -- refused to use it. A first run could
lock itself out by running an autoscale command before a credentialed one. It
never reproduced on a developer machine, where ``~/.eyvan`` already existed at
0700 from an earlier run; a clean CI runner found it immediately.
"""

from __future__ import annotations

import os
import pathlib
import stat

import pytest

from eyvan._internal.workspace import create_private_directory
from eyvan.capacity.autoscale.cache import AutoscaleConfigCache
from eyvan.server.local.auth import LocalDevelopmentCredentialError, LocalDevelopmentCredentialStore


@pytest.fixture
def permissive_umask():
    """The default on CI runners and most Linux installs, but not on a dev box."""

    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path: pathlib.Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_creating_a_nested_directory_leaves_no_readable_parent(tmp_path, permissive_umask) -> None:
    create_private_directory(tmp_path / ".eyvan" / "autoscale" / "configs")

    for directory in (tmp_path / ".eyvan", tmp_path / ".eyvan" / "autoscale"):
        assert not _mode(directory) & 0o077, f"{directory} is group- or world-accessible"


def test_the_autoscale_cache_does_not_lock_the_credential_store_out(tmp_path, permissive_umask) -> None:
    """The two components share ``~/.eyvan``; the looser one must not win."""

    profile = tmp_path / ".eyvan"
    cache = AutoscaleConfigCache(profile / "autoscale")
    cache.save_defaults(cache.load_defaults())

    store = LocalDevelopmentCredentialStore(profile / "local-development-credential")
    try:
        store.load()
    except LocalDevelopmentCredentialError as error:  # pragma: no cover - the bug this pins
        pytest.fail(f"an autoscale command locked the credential store out: {error}")
    except FileNotFoundError:
        pass  # No credential issued yet, which is the ordinary first-run answer.

    assert not _mode(profile) & 0o077


def test_an_existing_directory_is_left_exactly_as_it_is(tmp_path) -> None:
    """Widening or narrowing a directory this process did not create is not ours.

    ``$HOME`` is one of the parents the walk passes through.
    """

    existing = tmp_path / "already-here"
    existing.mkdir(mode=0o755)

    create_private_directory(existing)

    assert _mode(existing) == 0o755


def test_the_artifact_store_does_not_widen_the_profile_directory(tmp_path, permissive_umask) -> None:
    """The default artifact store lives at ``~/.eyvan/managed-artifacts``.

    It created its tree with no mode at all, so it -- not the autoscale cache --
    was what actually locked a clean CI runner out of its own profile directory.
    """

    from eyvan.artifacts.managed import LocalManagedArtifactStore

    profile = tmp_path / ".eyvan"
    LocalManagedArtifactStore(profile / "managed-artifacts")

    assert not _mode(profile) & 0o077
    assert not _mode(profile / "managed-artifacts") & 0o077


def test_the_staged_callable_runner_imports_only_files_that_travel_with_it() -> None:
    """``callable_protocol`` is copied flat into the OCI runner directory.

    It may import a module only if the stager stages that module too, and only
    through the sibling-first form that survives being flattened. Importing the
    shared directory helper would have been the tidy change; it broke every
    staged callable, which is why that one walk is inlined.
    """

    import ast
    import pathlib as _pathlib

    from eyvan.worker.runtime import callable_protocol, oci_callable_stager

    staged = {
        "http_retry",
        "plane_tls",
        "workspace",
        _pathlib.Path(oci_callable_stager.OCI_CALLABLE_PROTOCOL_FILENAME).stem,
    }
    tree = ast.parse(_pathlib.Path(callable_protocol.__file__).read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("eyvan"):
            assert node.module is not None
            leaf = node.module.rsplit(".", 1)[-1]
            assert leaf in staged, f"callable_protocol imports {node.module}, which the stager does not stage"
