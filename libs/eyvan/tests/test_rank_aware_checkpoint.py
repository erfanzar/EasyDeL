# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rank-aware checkpointing across a multi-host TPU slice.

``NativeSliceExecutor.checkpoint`` used to raise ``NotImplementedError``, so any
preemption policy relying on checkpoint hooks silently excluded every multi-host
pool -- that is, every real TPU training job. The blocker named in that
docstring was concrete: the single-host checkpoint discarded the source job's
environment, so a hook fanned across N hosts would have no JAX coordinator or
process id and every rank would race on one output URI.
"""

from __future__ import annotations

import httpx
import pytest
from test_systemd_transient import FakeSystemd

from eyvan.capacity.drivers.base import MutationContext
from eyvan.worker.runtime.native_remote import NativeRemoteExecutor
from eyvan.worker.runtime.native_slice import NativeSliceExecutor
from eyvan.worker.runtime.systemd_transient import SystemdTransientServiceExecutor


def _context(key: str = "ckpt-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=1, fencing_token="token-1")


def _spec(submission_id: str, environment: dict[str, str]) -> dict:
    return {
        "entrypoint": "python train.py",
        "submission_id": submission_id,
        "environment": environment,
        "gang": {"replicas": 1, "per_replica": {"cpu": 1}, "required_topology": {}},
        "metadata": {"eyvan_workload_id": submission_id},
    }


def _rank_environment(rank: int) -> dict[str, str]:
    return {
        "JAX_COORDINATOR_ADDRESS": "10.0.0.1:8476",
        "JAX_PROCESS_ID": str(rank),
        "JAX_PROCESS_COUNT": "2",
        "WANDB_MODE": "offline",
    }


def _checkpoint_environment(tmp_path, rank: int, *, output_uri: str | None = "gs://bucket/ckpt") -> dict[str, str]:
    daemon = FakeSystemd()
    executor = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=daemon,
        boot_id_reader=lambda: "boot-1",
        cgroup_pids_reader=lambda _group: (1,),
    )
    remote_id = executor.submit(_spec("wl-1", _rank_environment(rank)), _context("submit"))
    checkpoint_id = executor.checkpoint(remote_id, hook="python save.py", output_uri=output_uri, context=_context())
    # Read the env file the executor actually wrote, not a test double's record.
    environment = dict(
        line.split("=", 1)
        for line in executor._environment_file_path(checkpoint_id).read_text().splitlines()
        if "=" in line
    )
    environment["__source__"] = remote_id
    return environment


def test_the_hook_keeps_its_rank_identity_but_not_the_live_cluster_topology(tmp_path) -> None:
    """The source workload is still RUNNING while its hook executes.

    Inheriting JAX_COORDINATOR_* would make rank 0's hook bind the port the live
    training coordinator already holds. The hook still needs to know *which*
    rank it is, so identity is kept and cluster-forming topology is dropped.
    """

    environment = _checkpoint_environment(tmp_path, rank=1)

    assert environment["EYVAN_CHECKPOINT_RANK"] == "1"
    assert "JAX_COORDINATOR_ADDRESS" not in environment
    assert "JAX_COORDINATOR_PORT" not in environment
    assert "EYVAN_JAX_MULTIHOST" not in environment
    # Non-topology environment is still inherited.
    assert environment["WANDB_MODE"] == "offline"
    assert environment["EYVAN_SOURCE_JOB_ID"] == environment["__source__"]


def test_each_rank_writes_to_a_distinct_checkpoint_destination(tmp_path) -> None:
    """JAX membership does not serialize object-store writes.

    A single shared URI would let every rank race on one destination, which is
    what the multi-host checkpoint path previously produced.
    """

    rank0 = _checkpoint_environment(tmp_path / "r0", rank=0)
    rank1 = _checkpoint_environment(tmp_path / "r1", rank=1)

    assert rank0["EYVAN_CHECKPOINT_URI"] == "gs://bucket/ckpt/rank-0"
    assert rank1["EYVAN_CHECKPOINT_URI"] == "gs://bucket/ckpt/rank-1"
    assert rank0["EYVAN_CHECKPOINT_URI"] != rank1["EYVAN_CHECKPOINT_URI"]
    # The shared prefix stays addressable so a hook can still see the whole set.
    assert rank0["EYVAN_CHECKPOINT_BASE_URI"] == "gs://bucket/ckpt"
    assert rank1["EYVAN_CHECKPOINT_BASE_URI"] == "gs://bucket/ckpt"


def test_a_single_host_workload_keeps_an_unsuffixed_destination(tmp_path) -> None:
    """No rank means "not part of a slice", not "rank 0"."""

    from eyvan.worker.runtime.systemd_transient import _checkpoint_rank, _rank_checkpoint_uri

    assert _checkpoint_rank({"WANDB_MODE": "offline"}) is None
    assert _rank_checkpoint_uri("gs://bucket/ckpt", None) == "gs://bucket/ckpt"
    assert _rank_checkpoint_uri(None, 3) == ""


def _slice(handlers: list) -> NativeSliceExecutor:
    workers = [
        NativeRemoteExecutor(
            f"http://rank-{rank}",
            secret="s",
            client=httpx.Client(transport=httpx.MockTransport(handler), base_url=f"http://rank-{rank}"),
        )
        for rank, handler in enumerate(handlers)
    ]
    executor = NativeSliceExecutor.__new__(NativeSliceExecutor)
    executor.endpoints = tuple(f"http://rank-{rank}" for rank in range(len(handlers)))
    executor._workers = workers
    return executor


def test_slice_checkpoint_fans_out_to_every_rank_under_one_shared_id() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"remote_id": "ckpt-1"})

    executor = _slice([handler, handler])

    checkpoint_id = executor.checkpoint("wl-1", hook="python save.py", output_uri="gs://bucket/ckpt", context=_context())

    assert checkpoint_id == "ckpt-1"
    assert len(calls) == 2
    assert all("/checkpoint" in call for call in calls)
    for worker in executor._workers:
        worker.close()


def test_a_partial_checkpoint_launch_is_retired_not_left_running() -> None:
    """A surviving rank would hold accelerators and could still write output."""

    retired: list[str] = []

    def healthy(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/retire"):
            retired.append(str(request.url))
            return httpx.Response(200, json={"remote_id": "ckpt-1", "status": "STOPPED"})
        return httpx.Response(200, json={"remote_id": "ckpt-1"})

    def failing(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/retire"):
            retired.append(str(request.url))
            return httpx.Response(200, json={"remote_id": "ckpt-1", "status": "STOPPED"})
        return httpx.Response(500, json={"detail": "rank 1 is unreachable"})

    executor = _slice([healthy, failing])

    with pytest.raises(httpx.HTTPStatusError):
        executor.checkpoint("wl-1", hook="python save.py", output_uri="gs://b/c", context=_context())

    assert retired, "the rank that did launch must be retired"
    for worker in executor._workers:
        worker.close()


def test_an_empty_hook_is_rejected_before_any_rank_is_contacted() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"remote_id": "ckpt-1"})

    executor = _slice([handler, handler])

    with pytest.raises(ValueError, match="must be non-empty"):
        executor.checkpoint("wl-1", hook="   ", output_uri=None, context=_context())

    assert calls == []
    for worker in executor._workers:
        worker.close()
