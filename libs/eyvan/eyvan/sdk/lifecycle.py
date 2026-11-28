# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What one durable workload record means, said once for every waiter.

Two clients wait on a job: the SDK's blocking ``result()`` and the CLI's
``eyvan run``. Both have to turn the same durable states into the same words,
and a second copy of that mapping is a second vocabulary -- the state whose
wording is fixed in one of them and not the other is exactly the state a user
is trying to understand when they hit it.
"""

from __future__ import annotations

from eyvan.records import AutoscaleBlockReason, WorkloadRecord, WorkloadState
from eyvan.sdk.job_receipt import JobLifecycleState, JobLifecycleUpdate

#: States a workload never leaves.
TERMINAL_STATES = frozenset({WorkloadState.SUCCEEDED, WorkloadState.FAILED, WorkloadState.CANCELLED})


def worker_count(count: int) -> str:
    """Format one worker count without exposing provider resource accounting."""

    return f"{count} worker" if count == 1 else f"{count} workers"


def worker_request(accelerator_type: str | None, workers: int) -> str:
    """Format a provider-neutral worker request for terminal output."""

    counted = worker_count(workers)
    if accelerator_type is None:
        return counted
    return f"{accelerator_type} x {counted}"


def terminal_workload_update(workload: WorkloadRecord) -> JobLifecycleUpdate:
    """Describe a workload that has stopped, from the record alone.

    A denied queue admission is reported as *blocked* rather than as an
    ordinary failure: nothing ran, the provider refused the request, and the
    user's next action is a different zone or a quota, not a code change.
    """

    if workload.autoscale_block_reason is AutoscaleBlockReason.QUEUE_ADMISSION_DENIED:
        target = workload.queue_admission_denial
        required_workers = workload.spec.cell_group_count * workload.spec.worker_count
        scope = worker_request(None if target is None else target.accelerator_type, required_workers)
        if target is not None:
            scope = f"{scope} | {target.zone}"
        return JobLifecycleUpdate(
            JobLifecycleState.BLOCKED,
            scope,
            "GCP denied queue admission; no queued resource was created",
            workload.failure_reason,
        )
    if workload.state is WorkloadState.SUCCEEDED:
        return JobLifecycleUpdate(JobLifecycleState.SUCCEEDED, None, "job completed")
    if workload.state is WorkloadState.FAILED:
        return JobLifecycleUpdate(JobLifecycleState.FAILED, None, "job failed", workload.failure_reason)
    if workload.state is WorkloadState.CANCELLED:
        return JobLifecycleUpdate(JobLifecycleState.CANCELLED, None, "job cancelled")
    raise AssertionError(f"workload {workload.id!r} is not terminal: {workload.state.value!r}")


def running_workload_update(workload: WorkloadRecord) -> JobLifecycleUpdate:
    """Describe a workload that has not stopped, from the record alone.

    Deliberately record-only. The progress endpoint says far more -- how many
    workers are ready, which zone was selected -- and a waiter that can reach it
    should. This is what remains true when it cannot, and it is never a guess:
    every durable non-terminal state has exactly one meaning here.
    """

    if workload.state is WorkloadState.PENDING:
        return JobLifecycleUpdate(JobLifecycleState.QUEUED, None, "waiting for matching capacity")
    if workload.state is WorkloadState.ADMITTED:
        return JobLifecycleUpdate(JobLifecycleState.QUEUED, None, "admitted | waiting for a capacity lease")
    if workload.state is WorkloadState.LEASED:
        return JobLifecycleUpdate(JobLifecycleState.DISPATCHING, None, "capacity leased | waiting for dispatch")
    if workload.state is WorkloadState.DISPATCHING:
        return JobLifecycleUpdate(JobLifecycleState.DISPATCHING, None, "starting native workers")
    if workload.state is WorkloadState.RUNNING:
        return JobLifecycleUpdate(JobLifecycleState.RUNNING, None, "workload is running")
    if workload.state is WorkloadState.CHECKPOINTING:
        return JobLifecycleUpdate(
            JobLifecycleState.CHECKPOINTING,
            None,
            "writing checkpoint before releasing capacity",
        )
    if workload.state is WorkloadState.CANCELLING:
        return JobLifecycleUpdate(JobLifecycleState.CANCELLING, None, "releasing workload capacity")
    raise AssertionError(f"workload {workload.id!r} has no fallback lifecycle update: {workload.state.value!r}")


def workload_lifecycle_update(workload: WorkloadRecord) -> JobLifecycleUpdate:
    """Describe any durable workload record, terminal or not."""

    if workload.state in TERMINAL_STATES:
        return terminal_workload_update(workload)
    return running_workload_update(workload)


__all__ = (
    "TERMINAL_STATES",
    "running_workload_update",
    "terminal_workload_update",
    "worker_count",
    "worker_request",
    "workload_lifecycle_update",
)
