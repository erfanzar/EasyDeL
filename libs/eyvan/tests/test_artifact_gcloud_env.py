# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coverage for interpreter-steering scrubbing at the managed-artifact gcloud boundaries.

The GCS managed-artifact store spawns ``gcloud`` from two places: the store's
default transfer runner (every artifact cp/cat/rm the server-side store runs,
via ``managed_artifact_store_from_uri``) and the inventory token helper's
ambient ``gcloud auth print-access-token`` fallback. Both may run inside a
process that inherited a client's ``PYTHONPATH``, so both must start gcloud
with a scrubbed environment via ``eyvan._internal.subprocess_env``. Sibling coverage lives in
``tests/test_subprocess_env.py`` for the other module-level runners.
"""

from __future__ import annotations

import subprocess
import typing as tp

from eyvan.artifacts.managed import GCSManagedArtifactStore, _gcloud_inventory_access_token


def _captured_env_run(captured: dict[str, tp.Any]) -> tp.Callable[..., subprocess.CompletedProcess[bytes]]:
    """Return a subprocess.run stand-in that records the environment it was given."""

    def run(*args: tp.Any, **kwargs: tp.Any) -> subprocess.CompletedProcess[bytes]:
        captured.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, b"", b"")

    return run


def test_gcs_managed_store_default_runner_scrubs_interpreter_steering(monkeypatch: tp.Any) -> None:
    """The store's default transfer runner must not leak PYTHONPATH into gcloud."""

    monkeypatch.setenv("PYTHONPATH", "libs/easydel:libs/eyvan")
    store = GCSManagedArtifactStore("gs://eyvan-managed/artifacts")
    captured: dict[str, tp.Any] = {}
    monkeypatch.setattr(subprocess, "run", _captured_env_run(captured))

    store._run_gcloud(["gcloud", "storage", "cat", "gs://eyvan-managed/artifacts/results/x.result"])

    env = captured.get("env")
    assert env is not None, "the runner must pass an explicit scrubbed environment"
    assert "PYTHONPATH" not in env
    assert captured["timeout"] == 30.0


def test_gcloud_inventory_access_token_scrubs_interpreter_steering(monkeypatch: tp.Any) -> None:
    """The ambient-token fallback must not leak PYTHONPATH into gcloud either."""

    monkeypatch.setenv("PYTHONPATH", "libs/easydel:libs/eyvan")
    captured: dict[str, tp.Any] = {}
    monkeypatch.setattr(subprocess, "run", _captured_env_run(captured))

    token = _gcloud_inventory_access_token()

    env = captured.get("env")
    assert env is not None, "the token helper must pass an explicit scrubbed environment"
    assert "PYTHONPATH" not in env
    assert token == ""
