# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Heterogeneous groups: differently-shaped workloads that admit together.

A training run is often not one shape. It is one coordinator on a CPU cell, four
TPU slices, and two GPU cells serving a reward model -- and the useful property
is that *none* of them should start until *all* of them can. Submitted as
independent workloads, the four TPU slices lease immediately and burn accelerator
hours for however long the GPU cells stay unplaceable.

That single property -- atomic admission -- is the whole reason this exists over
the sibling model, and it is worth being precise about what is and is not
delivered.

**What this is.** A coordinator holding N differently-shaped members, all created
unschedulable, released together in one transaction once every member is
independently placeable in the same evaluation pass.

**What this is not: a reservation.** The check re-runs the pure scheduler over
current state; between that answer and the members' actual leases, another
workload can take the capacity. What the gate prevents is the *common* failure --
half a group holding accelerators for hours while the rest is structurally
infeasible -- not a race between two schedulers. Claiming otherwise would require
a reservation primitive the core scheduler does not have.

**Why not a parameter on the existing gang.** ``_schedule_cell_group`` places N
*identical* cells and ``_static_cell_failure`` requires ``worker_count``
equality, so one ``ResourceGang`` cannot describe two shapes. Widening it would
change the meaning of every existing gang; a coordinator over ordinary members
changes nothing that already ships.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from eyvan.records import WorkloadDefinition
from eyvan.server.service.batch import DEFAULT_MAX_FAN_OUT
from eyvan.server.service.errors import InputError

#: Most members one group may declare, counting every replica of every role. The
#: gate evaluates all of them together on the leader's hot path, so the bound is
#: what keeps one oversized group from stalling every other tenant's scheduling.
MAX_GROUP_MEMBERS = DEFAULT_MAX_FAN_OUT


class GroupError(InputError, ValueError):
    """Raised when a heterogeneous group request is not well formed.

    An ``InputError`` so the API maps it to 400 like every other domain
    precondition, and a ``ValueError`` so this module stays usable as a plain
    validator.
    """


@dataclass(frozen=True, slots=True)
class GroupRole:
    """One named shape within a group, and how many of it the group needs."""

    name: str
    template: WorkloadDefinition
    count: int = 1


def validate(roles: tp.Sequence[GroupRole], *, max_members: int = MAX_GROUP_MEMBERS) -> int:
    """Validate a group request and return its total member count.

    Raises:
        GroupError: If the request is empty, oversized, has duplicate role names,
            or declares a non-positive count.
    """

    if not roles:
        raise GroupError("a group must declare at least one role")
    names = [role.name for role in roles]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise GroupError(f"group role names must be unique; repeated: {', '.join(duplicates)}")
    for role in roles:
        if not role.name.strip():
            raise GroupError("group role names must not be blank")
        if role.count < 1:
            raise GroupError(f"group role {role.name!r} must declare a positive count")
    members = sum(role.count for role in roles)
    if members > max_members:
        raise GroupError(
            f"group of {members} members exceeds the {max_members}-member limit; "
            "split the run or raise the limit deliberately"
        )
    if len(roles) == 1:
        # One shape is a batch, and batch already does it with no admission gate
        # to evaluate every cycle. Refusing here keeps operators from paying the
        # gate's cost for a property they are not using.
        raise GroupError(
            f"group declares only the role {roles[0].name!r}; a single-shape run is a batch, "
            "which admits its tasks independently and costs nothing to gate"
        )
    return members


def member_names(roles: tp.Sequence[GroupRole], *, prefix: str = "") -> tuple[str, ...]:
    """Return each member's name, in declaration order.

    Deterministic so a replayed submission derives the same idempotency keys and
    therefore reuses the same member records rather than creating a second group.
    """

    stem = prefix or "group"
    return tuple(f"{stem}-{role.name}-{index}" for role in roles for index in range(role.count))


__all__ = ("MAX_GROUP_MEMBERS", "GroupError", "GroupRole", "member_names", "validate")
