# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Token acquisition failures must retain fleet-state subprocess fallback."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx
import pytest

from eyvan._internal.json_types import JsonObject
from eyvan.capacity.fleet_state import ConflictError, GcsBackend
from eyvan.capacity.gcs_rest import GcsRestError, GcsRestObject
from eyvan.capacity.providers.gcp.accelerator_catalog import AcceleratorCatalogError

_URI = "gs://test-bucket/fleet.json"
_DOCUMENT: JsonObject = {
    "schema": "eyvan.fleet-state",
    "version": 1,
    "cells": {"owned-cell": {"generation": 7}},
    "cell_tombstones": {},
    "workload_fences": {},
    "lease": None,
}


class _Storage:
    """Local storage transport that enforces the supplied CAS generation."""

    def __init__(self) -> None:
        self.document = json.loads(json.dumps(_DOCUMENT))
        self.generation = 42
        self.commands: list[tuple[str, ...]] = []

    def run(self, args):
        self.commands.append(tuple(args))
        assert _URI in args
        if tuple(args[:3]) == ("storage", "objects", "describe"):
            return subprocess.CompletedProcess(args, 0, json.dumps({"generation": self.generation}), "")
        if tuple(args[:2]) == ("storage", "cat"):
            return subprocess.CompletedProcess(args, 0, json.dumps(self.document), "")
        assert tuple(args[:2]) == ("storage", "cp")
        if args[-1] != f"--if-generation-match={self.generation}":
            return subprocess.CompletedProcess(args, 1, "", "412 precondition failed")
        self.document = json.loads(Path(args[2]).read_text())
        self.generation += 1
        return subprocess.CompletedProcess(args, 0, "", "")


class _HttpClient:
    """An entirely local HTTP response source; never opens a socket."""

    def __init__(self, status: int) -> None:
        self.status = status
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def get(self, *args, **kwargs):
        self.calls += 1
        return httpx.Response(self.status, headers={"x-goog-generation": "77"}, json=_DOCUMENT)

    def post(self, *args, **kwargs):
        self.calls += 1
        return httpx.Response(self.status)


def _mint_failure(kind: str):
    if kind == "timeout":
        raise subprocess.TimeoutExpired(["gcloud", "auth", "print-access-token"], 30)
    if kind == "missing-executable":
        raise OSError("synthetic gcloud startup failure")
    if kind == "nonzero":
        return subprocess.CompletedProcess([], 1, "", "synthetic credential-sensitive diagnostic")
    assert kind == "empty"
    return subprocess.CompletedProcess([], 0, "  ", "synthetic credential-sensitive diagnostic")


@pytest.mark.parametrize("operation", ["read", "write"])
@pytest.mark.parametrize("auth_status", [None, 401, 403], ids=["initial", "refresh-401", "refresh-403"])
@pytest.mark.parametrize("failure", ["timeout", "missing-executable", "nonzero", "empty"])
def test_default_token_failure_uses_working_storage_fallback(monkeypatch, operation, auth_status, failure) -> None:
    """Returned fleet state and persisted conditional writes survive mint failures."""

    mint_calls = 0

    def mint(*args, **kwargs):
        nonlocal mint_calls
        mint_calls += 1
        assert args[0] == ["gcloud", "auth", "print-access-token"]
        if auth_status is not None and mint_calls == 1:
            return subprocess.CompletedProcess(args[0], 0, "expired-synthetic-token", "")
        return _mint_failure(failure)

    monkeypatch.setattr(subprocess, "run", mint)
    client = _HttpClient(auth_status or 200)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    storage = _Storage()
    rest = GcsRestObject(_URI)
    backend = GcsBackend(_URI, runner=storage.run, rest=rest)

    if operation == "read":
        document, generation = backend._read_fast()
        assert document == _DOCUMENT
        assert generation == 42
    else:
        updated: JsonObject = {**_DOCUMENT, "cells": {"replacement-cell": {"generation": 8}}}
        assert backend._write_fast(updated, 42) is None
        assert storage.document == updated
        assert storage.generation == 43
        # A stale conditional write still conflicts, without replacing stored state.
        with pytest.raises(ConflictError):
            backend._write_fast(_DOCUMENT, 42)
        assert storage.document == updated
        assert storage.generation == 43
    assert backend._rest is rest
    assert mint_calls >= (2 if auth_status is not None else 1)
    assert client.calls == (2 if operation == "write" and auth_status is not None else int(auth_status is not None))


@pytest.mark.parametrize("refresh", [False, True])
def test_expected_token_error_is_credential_free(monkeypatch, refresh) -> None:
    diagnostic = "synthetic credential-sensitive diagnostic"

    def fail():
        raise AcceleratorCatalogError(diagnostic)

    obj = GcsRestObject(_URI, token_source=fail)
    with pytest.raises(GcsRestError) as caught:
        obj._headers(refresh=refresh)
    assert diagnostic not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("operation", ["read", "write"])
def test_programming_error_in_token_source_is_not_treated_as_transport_failure(monkeypatch, operation) -> None:
    def fail():
        raise TypeError("unexpected token-source bug")

    client = _HttpClient(200)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    storage = _Storage()
    backend = GcsBackend(_URI, runner=storage.run, rest=GcsRestObject(_URI, token_source=fail))
    with pytest.raises(TypeError, match="unexpected token-source bug"):
        if operation == "read":
            backend._read_fast()
        else:
            backend._write_fast(_DOCUMENT, 42)
    assert storage.commands == []
    assert storage.document == _DOCUMENT
    assert storage.generation == 42


def test_failed_mint_does_not_disable_later_rest_reads(monkeypatch) -> None:
    attempts = 0

    def mint(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return _mint_failure("nonzero")
        return subprocess.CompletedProcess(args[0], 0, "recovered-synthetic-token", "")

    monkeypatch.setattr(subprocess, "run", mint)
    client = _HttpClient(200)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    storage = _Storage()
    backend = GcsBackend(_URI, runner=storage.run, rest=GcsRestObject(_URI))
    assert backend._read_fast() == (_DOCUMENT, 42)
    assert backend._read_fast() == (_DOCUMENT, 77)
    assert len(storage.commands) == 2
    assert client.calls == 1


def test_rest_precondition_conflict_does_not_fall_back(monkeypatch) -> None:
    client = _HttpClient(412)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    storage = _Storage()
    backend = GcsBackend(_URI, runner=storage.run, rest=GcsRestObject(_URI, token_source=lambda: "synthetic-token"))
    with pytest.raises(ConflictError):
        backend._write_fast(_DOCUMENT, 42)
    assert storage.commands == []
    assert storage.document == _DOCUMENT
    assert storage.generation == 42
