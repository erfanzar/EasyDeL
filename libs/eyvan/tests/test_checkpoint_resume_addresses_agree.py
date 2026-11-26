# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derive matching rank-specific checkpoint addresses for writes and resumes.

The two paths must agree so a resumed attempt does not silently start from zero
when its checkpoint exists under a rank suffix.
"""

from __future__ import annotations

import pytest

from eyvan.worker.runtime._multihost_site import DEFAULT_GANG_INITIALIZATION
from eyvan.worker.runtime.native_slice import _RESUME_CHECKPOINT_ENV
from eyvan.worker.runtime.systemd_transient import _rank_checkpoint_uri

BASE = "gs://bucket/eyvan/checkpoints/job"


def _resume_env(executor, spec: dict, rank: int) -> dict[str, str]:
    return dict(executor._ranked_spec(spec, rank)["environment"])


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_a_rank_resumes_from_the_address_its_hook_wrote(rank: int) -> None:
    """The property that broke: write address and read address must match.

    Asserted as agreement between the two derivations rather than against a
    literal, so changing the layout cannot leave one side behind again.
    """

    from eyvan.worker.runtime.native_slice import NativeSliceExecutor

    executor = NativeSliceExecutor.__new__(NativeSliceExecutor)
    executor._workers = (None,) * 4  # type: ignore[attr-defined]
    executor.jax_cluster_detection = _detection()  # type: ignore[attr-defined]
    executor.initialization = DEFAULT_GANG_INITIALIZATION  # type: ignore[attr-defined]
    spec = {"entrypoint": "python train.py", "environment": {_RESUME_CHECKPOINT_ENV: BASE}}

    resume = _resume_env(executor, spec, rank)[_RESUME_CHECKPOINT_ENV]

    assert resume == _rank_checkpoint_uri(BASE, rank)
    assert resume.endswith(f"/rank-{rank}")


def _detection():
    from eyvan.worker.runtime.native_slice import JaxClusterDetectionMode

    return JaxClusterDetectionMode.GCE_TPU


def test_two_ranks_never_resume_from_the_same_object() -> None:
    """Each rank checkpointed its own state; sharing one would corrupt a resume."""

    from eyvan.worker.runtime.native_slice import NativeSliceExecutor

    executor = NativeSliceExecutor.__new__(NativeSliceExecutor)
    executor._workers = (None,) * 2  # type: ignore[attr-defined]
    executor.jax_cluster_detection = _detection()  # type: ignore[attr-defined]
    executor.initialization = DEFAULT_GANG_INITIALIZATION  # type: ignore[attr-defined]
    spec = {"entrypoint": "python train.py", "environment": {_RESUME_CHECKPOINT_ENV: BASE}}

    first = _resume_env(executor, spec, 0)[_RESUME_CHECKPOINT_ENV]
    second = _resume_env(executor, spec, 1)[_RESUME_CHECKPOINT_ENV]

    assert first != second


def test_a_first_attempt_is_given_no_resume_address() -> None:
    """An empty resume URI must stay empty, not become ``/rank-0``.

    A workload starting fresh would otherwise be pointed at an object that does
    not exist and cannot: there is nothing to resume from.
    """

    from eyvan.worker.runtime.native_slice import NativeSliceExecutor

    executor = NativeSliceExecutor.__new__(NativeSliceExecutor)
    executor._workers = (None,) * 2  # type: ignore[attr-defined]
    executor.jax_cluster_detection = _detection()  # type: ignore[attr-defined]
    executor.initialization = DEFAULT_GANG_INITIALIZATION  # type: ignore[attr-defined]
    spec = {"entrypoint": "python train.py", "environment": {_RESUME_CHECKPOINT_ENV: ""}}

    assert _resume_env(executor, spec, 0)[_RESUME_CHECKPOINT_ENV] == ""


def test_a_workload_with_no_resume_variable_is_left_alone() -> None:
    """Most workloads never checkpoint; the fix must not invent the variable."""

    from eyvan.worker.runtime.native_slice import NativeSliceExecutor

    executor = NativeSliceExecutor.__new__(NativeSliceExecutor)
    executor._workers = (None,) * 2  # type: ignore[attr-defined]
    executor.jax_cluster_detection = _detection()  # type: ignore[attr-defined]
    executor.initialization = DEFAULT_GANG_INITIALIZATION  # type: ignore[attr-defined]
    spec = {"entrypoint": "python train.py", "environment": {}}

    assert _RESUME_CHECKPOINT_ENV not in _resume_env(executor, spec, 0)


def test_a_trailing_slash_does_not_produce_a_double_separator() -> None:
    """Operators write both forms; both must resolve to the same object."""

    assert _rank_checkpoint_uri(f"{BASE}/", 0) == _rank_checkpoint_uri(BASE, 0)


def _gang_executor(worker_count: int):
    from eyvan.worker.runtime.native_gang import NativeGangExecutor

    executor = NativeGangExecutor.__new__(NativeGangExecutor)
    executor._workers = (None,) * worker_count  # type: ignore[attr-defined]
    executor.initialization = DEFAULT_GANG_INITIALIZATION  # type: ignore[attr-defined]
    executor.gpu_plan = None  # type: ignore[attr-defined]
    executor._coordinator_address = "10.12.0.10:12345"  # type: ignore[attr-defined]
    return executor


def _gang_worker():
    from eyvan.worker.runtime.native_gang import _GangWorker

    return _GangWorker(
        cell_id="cell-a",
        cell_index=0,
        endpoint="http://10.12.0.10:8100",
        context=None,  # type: ignore[arg-type]
        executor=None,  # type: ignore[arg-type]
    )


def _gang_resume_env(executor, spec: dict, rank: int) -> dict[str, str]:
    return dict(executor._ranked_spec(spec, rank, _gang_worker())["environment"])


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_a_gang_rank_resumes_from_the_address_its_hook_wrote(rank: int) -> None:
    """The gang dispatch serves the same hook the slice dispatch serves.

    Both write ``{base}/rank-{n}``, so both resume paths have to suffix the
    base the same way; this is the gang half of the agreement the slice tests
    above pin.
    """

    executor = _gang_executor(4)
    spec = {"entrypoint": "python train.py", "environment": {_RESUME_CHECKPOINT_ENV: BASE}}

    resume = _gang_resume_env(executor, spec, rank)[_RESUME_CHECKPOINT_ENV]

    assert resume == _rank_checkpoint_uri(BASE, rank)
    assert resume.endswith(f"/rank-{rank}")


def test_two_gang_ranks_never_resume_from_the_same_object() -> None:
    """Each gang rank checkpointed its own state; sharing one corrupts a resume."""

    executor = _gang_executor(2)
    spec = {"entrypoint": "python train.py", "environment": {_RESUME_CHECKPOINT_ENV: BASE}}

    first = _gang_resume_env(executor, spec, 0)[_RESUME_CHECKPOINT_ENV]
    second = _gang_resume_env(executor, spec, 1)[_RESUME_CHECKPOINT_ENV]

    assert first != second


def test_a_gang_workload_without_a_resume_address_is_left_alone() -> None:
    """A cold gang, or one whose URI is empty, must not be handed ``/rank-0``."""

    executor = _gang_executor(2)
    bare = {"entrypoint": "python train.py", "environment": {}}
    empty = {"entrypoint": "python train.py", "environment": {_RESUME_CHECKPOINT_ENV: ""}}

    assert _RESUME_CHECKPOINT_ENV not in _gang_resume_env(executor, bare, 0)
    assert _gang_resume_env(executor, empty, 0)[_RESUME_CHECKPOINT_ENV] == ""
