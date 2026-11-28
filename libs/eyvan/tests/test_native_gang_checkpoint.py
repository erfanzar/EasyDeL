# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Checkpointing a workload that spans several independently leased cells.

This was the package's only real unimplemented-feature marker: multi-cell gangs
raised ``NotImplementedError`` from ``checkpoint``, so a generic multi-cell
group could never be checkpointed and therefore never preempted -- Eyvan could
not reclaim its capacity for higher-priority work at all.

Two things differ from a single-cell slice, and both decide correctness:

* a gang holds one lease *per cell*, so each worker must be addressed with its
  own fence -- one shared context would be rejected by every worker but one;
* each cell's context carries a different idempotency key, so passing them
  through unchanged would mint one checkpoint ID per cell and leave nothing for
  the caller to poll, because ``status`` asks every rank about one logical
  checkpoint.

So each worker is addressed with its own fence carrying the *shared* checkpoint
ID -- the same asymmetry ``submit`` already has between per-cell fences and one
logical submission ID.
"""

from __future__ import annotations

import json
import typing as tp

import httpx
import pytest
from test_native_gang_executor import _client, _context, _members

from eyvan.worker.runtime.native_gang import NativeGangExecutor

ENDPOINTS = (
    "http://10.12.0.10:8100",
    "http://10.12.0.11:8100",
    "http://10.12.0.20:8100",
    "http://10.12.0.21:8100",
)
CHECKPOINT_CONTEXT = _context(key="ckpt-shared", generation=7, token="ignored-by-gang")


def _checkpoint_calls(handler: tp.Callable[[httpx.Request], httpx.Response]):
    clients = [_client(endpoint, handler) for endpoint in ENDPOINTS]
    return NativeGangExecutor(_members(), clients=clients), clients


def _payload(request: httpx.Request) -> dict:
    return json.loads(request.content.decode("utf-8"))


def test_every_cell_is_checkpointed_under_one_shared_id() -> None:
    """Without a shared ID nothing can poll "the checkpoint" afterwards."""

    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, _payload(request)))
        return httpx.Response(200, json={"remote_id": "ckpt-shared"})

    executor, clients = _checkpoint_calls(handler)
    try:
        checkpoint_id = executor.checkpoint(
            "gang-job",
            hook="python checkpoint.py",
            output_uri="gs://bucket/checkpoint",
            context=CHECKPOINT_CONTEXT,
        )
    finally:
        for client in clients:
            client.close()

    assert checkpoint_id == "ckpt-shared"
    assert len(seen) == len(ENDPOINTS), "every rank in every cell must run the hook"
    assert {payload["mutation"]["idempotency_key"] for _host, payload in seen} == {"ckpt-shared"}


def test_each_cell_is_addressed_with_its_own_fence() -> None:
    """A gang holds one lease per cell; a shared fence is rejected by all but one."""

    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, _payload(request)))
        return httpx.Response(200, json={"remote_id": "ckpt-shared"})

    executor, clients = _checkpoint_calls(handler)
    try:
        executor.checkpoint(
            "gang-job",
            hook="python checkpoint.py",
            output_uri="gs://bucket/checkpoint",
            context=CHECKPOINT_CONTEXT,
        )
    finally:
        for client in clients:
            client.close()

    tokens = {host: payload["mutation"]["fencing_token"] for host, payload in seen}
    assert tokens["10.12.0.10"] == "fence-a"
    assert tokens["10.12.0.11"] == "fence-a"
    assert tokens["10.12.0.20"] == "fence-b"
    assert tokens["10.12.0.21"] == "fence-b"
    assert tokens["10.12.0.10"] != tokens["10.12.0.20"], "cells must not share a fence"


def test_a_partial_launch_retires_what_it_started() -> None:
    """A surviving hook holds accelerators and can still write the checkpoint."""

    retired: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/retire"):
            retired.append((request.url.host, request.url.path))
            return httpx.Response(200, json={"remote_id": "ckpt-shared", "status": "STOPPED", "message": "retired"})
        if request.url.host == "10.12.0.20":
            return httpx.Response(500, json={"detail": "worker exploded"})
        return httpx.Response(200, json={"remote_id": "ckpt-shared"})

    executor, clients = _checkpoint_calls(handler)
    try:
        with pytest.raises(Exception):  # noqa: B017 - the transport error type is not the property under test
            executor.checkpoint(
                "gang-job",
                hook="python checkpoint.py",
                output_uri="gs://bucket/checkpoint",
                context=CHECKPOINT_CONTEXT,
            )
    finally:
        for client in clients:
            client.close()

    assert retired, "a partial checkpoint launch left hooks running"
    assert {host for host, _path in retired} >= {"10.12.0.10", "10.12.0.11"}


def test_a_worker_returning_a_different_id_is_refused() -> None:
    """Divergent IDs mean the ranks are not checkpointing one logical thing."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/retire"):
            return httpx.Response(200, json={"remote_id": "x", "status": "STOPPED", "message": "retired"})
        remote_id = "something-else" if request.url.host == "10.12.0.21" else "ckpt-shared"
        return httpx.Response(200, json={"remote_id": remote_id})

    executor, clients = _checkpoint_calls(handler)
    try:
        with pytest.raises(RuntimeError, match="unexpected checkpoint ID"):
            executor.checkpoint(
                "gang-job",
                hook="python checkpoint.py",
                output_uri="gs://bucket/checkpoint",
                context=CHECKPOINT_CONTEXT,
            )
    finally:
        for client in clients:
            client.close()


@pytest.mark.parametrize("hook", ["", "   "])
def test_an_empty_hook_is_refused_before_any_cell_is_touched(hook: str) -> None:
    """Launching half a checkpoint to discover the hook is empty is pure cost."""

    touched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        touched.append(request.url.host)
        return httpx.Response(200, json={"remote_id": "ckpt-shared"})

    executor, clients = _checkpoint_calls(handler)
    try:
        with pytest.raises(ValueError, match="hook must be non-empty"):
            executor.checkpoint(
                "gang-job",
                hook=hook,
                output_uri="gs://bucket/checkpoint",
                context=CHECKPOINT_CONTEXT,
            )
    finally:
        for client in clients:
            client.close()

    assert touched == []


def test_status_reports_each_ranks_derived_checkpoint_destination() -> None:
    """A status row may name a checkpoint only at the address its hook writes.

    The resume path derives the same per-rank address (see
    ``test_checkpoint_resume_addresses_agree``), so a destination that disagreed
    with the write would silently read as "no checkpoint" on restart.
    """

    from eyvan.worker.runtime.systemd_transient import _rank_checkpoint_uri

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/checkpoint"):
            return httpx.Response(200, json={"remote_id": "ckpt-shared"})
        return httpx.Response(200, json={"remote_id": "gang-job", "status": "RUNNING"})

    executor, clients = _checkpoint_calls(handler)
    try:
        executor.checkpoint(
            "gang-job",
            hook="python checkpoint.py",
            output_uri="gs://bucket/checkpoint",
            context=CHECKPOINT_CONTEXT,
        )
        report = executor.status("gang-job")
    finally:
        for client in clients:
            client.close()

    assert [worker["checkpoint_uri"] for worker in report["workers"]] == [
        _rank_checkpoint_uri("gs://bucket/checkpoint", rank) for rank in range(len(ENDPOINTS))
    ]


def test_status_leaves_the_checkpoint_destination_unknown_without_a_launch() -> None:
    """A status must not invent a destination no hook ever received."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"remote_id": "gang-job", "status": "RUNNING"})

    executor, clients = _checkpoint_calls(handler)
    try:
        report = executor.status("gang-job")
    finally:
        for client in clients:
            client.close()

    assert all(worker["checkpoint_uri"] is None for worker in report["workers"])
