# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Job-tree traversal for parent/child workloads.

**Why a tree is not a scheduling level.** Eyvan's fair-share hierarchy is
tenant -> project -> principal and is structural: ``IdentityScope`` caps its own
depth, ``WorkloadIdentity.scopes`` returns a literal three-tuple, and the DRF
queue writes that parent chain verbatim. Adding a fourth level would ripple into
quota scopes, budget scopes, and the per-scope lease accounting indexes. So a
child inherits its parent's ``(org, project, created_by)`` and competes as an
ordinary sibling; job structure is a *relation* plus a *cancellation cascade*,
nothing more. Revisit only if fairness *between* trees becomes a real complaint.

Traversal here is deliberately defensive. ``parent_workload_id`` is an ordinary
persisted field, so a corrupted or hand-edited record could describe a cycle or
an unbounded chain. Every walk is therefore visited-set guarded and depth-capped
rather than trusting the data to be a tree.
"""

from __future__ import annotations

import typing as tp

from eyvan.records import WorkloadRecord
from eyvan.server.service.errors import ConflictError, InputError

#: Maximum tree depth walked in one traversal. Deep enough for launcher ->
#: stage -> shard patterns, shallow enough that a malformed chain cannot make a
#: cancellation walk unbounded.
MAX_JOB_TREE_DEPTH = 32


class ChildLoader(tp.Protocol):
    """Load the direct children of one workload."""

    async def __call__(self, parent_workload_id: str) -> list[WorkloadRecord]: ...


class JobTreeError(InputError):
    """A named parent/child relation is not a valid tree.

    An ``EyvanServiceError`` rather than a bare ``ValueError`` because it reaches
    the HTTP layer: subclassing ``ValueError`` meant no handler matched and a
    caller naming a parent in another project got a 500 instead of being told
    what was wrong with their request.
    """


class JobTreeTooDeepError(ConflictError):
    """The durable tree is deeper than the cascade protocol will walk.

    Distinct from :class:`JobTreeError` and a 409, not a 400, because the request
    was well formed -- the *stored* tree is the problem. Cancelling such a parent
    with ``cascade=False`` is the way out, which is why that flag is exposed on
    the route.
    """


#: Live children one parent may hold at once. A ceiling on *concurrency*, not on
#: total fan-out: ten thousand children submitted one at a time is an ordinary
#: sweep, while ten thousand at once is a denial of service against the
#: scheduler and against every other tenant's queue position.
MAX_LIVE_CHILDREN = 256


class JobTreeTooWideError(ConflictError):
    """Raised when a parent already holds as many live children as it may.

    A 409 rather than a 429: the caller is not being rate limited and retrying
    in a moment will not help. Something has to finish first, and saying so is
    what lets a launcher block rather than spin.
    """


async def parent_depth(
    workload_id: str,
    load_parent: tp.Callable[[str], tp.Awaitable[WorkloadRecord | None]],
    *,
    max_depth: int = MAX_JOB_TREE_DEPTH,
) -> int:
    """Return how many ancestors a workload has, refusing an unbounded chain.

    ``collect_descendants`` walks *down* and cannot see this: a self-submitting
    callable extends the tree upward one link at a time, and every individual
    step looks like an ordinary parent with one child. Without an upward check
    the only thing that eventually stops it is the budget.
    """

    depth = 0
    seen: set[str] = {workload_id}
    current = workload_id
    while True:
        parent = await load_parent(current)
        if parent is None:
            return depth
        if parent.id in seen:
            # A cycle cannot be created through the API, but a corrupted or
            # hand-edited record should terminate rather than hang the request.
            raise JobTreeTooDeepError(f"job tree above {workload_id!r} contains a cycle at {parent.id!r}")
        depth += 1
        if depth >= max_depth:
            raise JobTreeTooDeepError(
                f"job tree above {workload_id!r} is deeper than {max_depth} levels; "
                "a self-submitting callable will not stop on its own"
            )
        seen.add(parent.id)
        current = parent.id


async def collect_descendants(
    root_id: str,
    load_children: ChildLoader,
    *,
    max_depth: int = MAX_JOB_TREE_DEPTH,
) -> list[WorkloadRecord]:
    """Return every descendant of ``root_id``, parents before their children.

    The order matters for cancellation: cancelling a parent before its children
    means the parent cannot admit new children while the walk is still running.

    A workload already visited is skipped rather than re-walked, so a cycle
    terminates instead of looping forever.
    """

    visited: set[str] = {root_id}
    ordered: list[WorkloadRecord] = []
    frontier = [(root_id, 0)]
    while frontier:
        parent_id, depth = frontier.pop(0)
        if depth >= max_depth:
            raise JobTreeTooDeepError(
                f"job tree below {root_id!r} is deeper than {max_depth} levels; refusing to walk further. "
                "Cancel individual subtrees, or cancel this workload alone with cascade=false."
            )
        for child in await load_children(parent_id):
            if child.id in visited:
                # Either a cycle or a diamond; both mean this node is already
                # accounted for, and re-walking it would not terminate.
                continue
            visited.add(child.id)
            ordered.append(child)
            frontier.append((child.id, depth + 1))
    return ordered


def validate_parent(child_scope: tuple[str, str], parent: WorkloadRecord) -> None:
    """Reject a parent that would create a cross-tenant or cross-project tree.

    A child must stay inside its parent's scope: it inherits the parent's
    fair-share leaf and budget, so a cross-project child would spend one
    project's budget against another's quota.
    """

    org_id, project_id = child_scope
    if parent.org_id != org_id or parent.project_id != project_id:
        raise JobTreeError(
            f"parent workload {parent.id!r} belongs to {parent.org_id}/{parent.project_id}, "
            f"not the child's {org_id}/{project_id}"
        )


__all__ = (
    "MAX_JOB_TREE_DEPTH",
    "MAX_LIVE_CHILDREN",
    "ChildLoader",
    "JobTreeError",
    "JobTreeTooDeepError",
    "JobTreeTooWideError",
    "collect_descendants",
    "parent_depth",
    "validate_parent",
)
