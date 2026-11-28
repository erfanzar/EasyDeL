# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Batch fan-out built on ordinary durable workloads.

A map or array job is not a new execution kind. It is one parent workload plus N
children, each an ordinary durable workload that admits, schedules, leases,
settles, and audits exactly like any other. That is deliberate: a separate batch
runtime would need its own retry, preemption, and budget semantics, and those
are precisely the parts that are hard to get right.

Two bounds are enforced rather than left to the caller. ``max_fan_out`` caps how
many children one request may create, because a typo in a range expression
should fail at submission instead of admitting ten thousand TPU jobs. And the
whole expansion commits inside a single transaction, so a partially-created
array cannot exist.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from pydantic import ValidationError

from eyvan.records import WorkloadDefinition

#: Ceiling on children created by one batch request. Deliberately conservative:
#: this is accelerator capacity, and the cost of a mistake is measured in
#: dollars per hour per child.
DEFAULT_MAX_FAN_OUT = 1000

#: Environment variables each child receives so the task can find its slot.
BATCH_INDEX_ENV = "EYVAN_BATCH_INDEX"
BATCH_COUNT_ENV = "EYVAN_BATCH_COUNT"


class BatchError(ValueError):
    """Raised when a batch request cannot be expanded into valid workloads."""


@dataclass(frozen=True, slots=True)
class BatchRequest:
    """One fan-out request: a template plus the number of tasks to create."""

    template: WorkloadDefinition
    count: int
    name_prefix: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count < 1:
            raise BatchError("batch count must be a positive integer")


def expand(request: BatchRequest, *, max_fan_out: int = DEFAULT_MAX_FAN_OUT) -> list[WorkloadDefinition]:
    """Expand one batch request into per-task workload definitions.

    Each child gets ``EYVAN_BATCH_INDEX`` and ``EYVAN_BATCH_COUNT`` so the task
    can select its shard without the caller templating a command string. The
    template's own environment wins over neither: these names are reserved and
    a template that sets them is rejected rather than silently overridden.
    """

    if request.count > max_fan_out:
        raise BatchError(
            f"batch of {request.count} exceeds the {max_fan_out}-task fan-out limit; "
            "split the work or raise the limit deliberately"
        )
    reserved = {BATCH_INDEX_ENV, BATCH_COUNT_ENV} & set(request.template.environment)
    if reserved:
        raise BatchError(f"batch template must not set reserved variables: {', '.join(sorted(reserved))}")

    tasks: list[WorkloadDefinition] = []
    for index in range(request.count):
        prefix = request.name_prefix or request.template.name
        candidate = request.template.model_copy(
            update={
                "name": f"{prefix}-{index}",
                "environment": {
                    **request.template.environment,
                    BATCH_INDEX_ENV: str(index),
                    BATCH_COUNT_ENV: str(request.count),
                },
            }
        )
        # model_copy(update=...) SKIPS validation, so a derived name can exceed
        # the field's own limit. That record persists happily and then fails to
        # re-validate on read -- and the Postgres backend re-validates every row
        # in list_workloads, which the controller calls unfiltered. One bad batch
        # would stop placement for the entire control plane, not just its tenant.
        try:
            tasks.append(WorkloadDefinition.model_validate(candidate.model_dump()))
        except ValidationError as exc:
            raise BatchError(f"batch task {index} is not a valid workload: {exc}") from exc
    return tasks


def task_index(workload_environment: tp.Mapping[str, str]) -> int:
    """Return a task's batch index, or -1 when it is not a batch task.

    Used to order children by task rather than by their random identifier:
    callers index into the returned list, so ``children[3]`` must be task 3 on
    a replay just as it is on the original submission.
    """

    raw = workload_environment.get(BATCH_INDEX_ENV)
    if raw is None:
        return -1
    try:
        return int(raw)
    except ValueError:
        return -1


def child_idempotency_key(parent_key: str, index: int) -> str:
    """Derive a stable per-child key from the parent request's key.

    Derived rather than random so a retried batch submission replays the same
    children instead of creating a second array.
    """

    return f"{parent_key}:task:{index}"


__all__ = (
    "BATCH_COUNT_ENV",
    "BATCH_INDEX_ENV",
    "DEFAULT_MAX_FAN_OUT",
    "BatchError",
    "BatchRequest",
    "child_idempotency_key",
    "expand",
    "task_index",
)
