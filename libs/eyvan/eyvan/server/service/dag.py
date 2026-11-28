# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Small DAGs of whole workloads.

A DAG is the batch shape plus one relation: a node may declare predecessors that
must succeed before it is placed. Everything else is unchanged -- each node is an
ordinary durable workload that admits, schedules, leases, settles and audits like
any other, under a coordinator that owns them.

Three limits are deliberate, and this module is where they are enforced rather
than discovered:

**No data passing.** A dependency orders execution and nothing else. Results
already travel as ``eyvan://`` result URIs that a successor can read, so adding a
value channel here would turn the control plane into a data plane and put user
payloads in the scheduling database.

**Dependencies are by name, within one request.** A node cannot depend on an
arbitrary existing workload id. That single restriction is what makes cycles
detectable at submission, keeps every edge inside one org/project scope, and stops
a dependency from outliving the request that created it.

**Small.** Bounded fan-out, no fan-in aggregation, no conditional edges, no
retries-with-different-inputs. This is "run B after A succeeds" for a handful of
stages -- not a workflow engine, and the line is here so it stays that way.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from eyvan.records import WorkloadDefinition
from eyvan.server.service.batch import DEFAULT_MAX_FAN_OUT
from eyvan.server.service.errors import InputError

#: Longest chain a DAG may declare. Deeper than any staged training pipeline needs
#: and shallow enough that a release walk stays trivially bounded.
MAX_DAG_DEPTH = 32


class DagError(InputError, ValueError):
    """Raised when a DAG request is not a valid directed acyclic graph.

    An ``InputError`` so the API maps it to 400 like every other domain
    precondition, and a ``ValueError`` so this module remains usable as a plain
    validator by anything that has not adopted the service error hierarchy.
    """


@dataclass(frozen=True, slots=True)
class DagNode:
    """One named stage and the sibling stages it waits for."""

    name: str
    template: WorkloadDefinition
    depends_on: tuple[str, ...] = ()


def validate(nodes: tp.Sequence[DagNode], *, max_fan_out: int = DEFAULT_MAX_FAN_OUT) -> tuple[str, ...]:
    """Validate a DAG request and return its node names in topological order.

    Returning the order is the point: the caller creates records in this sequence,
    so a node's predecessors always already exist and their ids can be resolved
    without a second pass or a placeholder.

    Raises:
        DagError: If the request is empty, oversized, has duplicate or unknown
            names, declares a self-edge, or contains a cycle.
    """

    if not nodes:
        raise DagError("a DAG must declare at least one node")
    if len(nodes) > max_fan_out:
        raise DagError(
            f"DAG of {len(nodes)} nodes exceeds the {max_fan_out}-node limit; "
            "split the pipeline or raise the limit deliberately"
        )

    names = [node.name for node in nodes]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise DagError(f"DAG node names must be unique; repeated: {', '.join(duplicates)}")

    known = set(names)
    for node in nodes:
        if node.name in node.depends_on:
            raise DagError(f"DAG node {node.name!r} cannot depend on itself")
        unknown = sorted(set(node.depends_on) - known)
        if unknown:
            raise DagError(
                f"DAG node {node.name!r} depends on {unknown[0]!r}, which is not a node in this request; "
                "dependencies must name a sibling node"
            )
        if len(set(node.depends_on)) != len(node.depends_on):
            raise DagError(f"DAG node {node.name!r} repeats a dependency")

    return _topological_order(nodes)


def _topological_order(nodes: tp.Sequence[DagNode]) -> tuple[str, ...]:
    """Return node names ordered so every predecessor precedes its dependents.

    Kahn's algorithm over a deterministic queue: ties break by declaration order, so
    the same request always produces the same order and therefore the same derived
    idempotency keys on a replay.
    """

    remaining = {node.name: set(node.depends_on) for node in nodes}
    order: list[str] = []
    while remaining:
        ready = [name for name in (node.name for node in nodes) if name in remaining and not remaining[name]]
        if not ready:
            # Everything left has an unmet dependency inside the remaining set,
            # which is exactly a cycle.
            raise DagError(f"DAG contains a cycle among: {', '.join(sorted(remaining))}")
        for name in ready:
            del remaining[name]
            order.append(name)
        for pending in remaining.values():
            pending.difference_update(ready)

    depth = _longest_chain(nodes, order)
    if depth > MAX_DAG_DEPTH:
        raise DagError(f"DAG chain is {depth} nodes deep, above the {MAX_DAG_DEPTH}-node limit")
    return tuple(order)


def _longest_chain(nodes: tp.Sequence[DagNode], order: tp.Sequence[str]) -> int:
    """Return the longest path length using the already-validated topological order."""

    by_name = {node.name: node for node in nodes}
    depths: dict[str, int] = {}
    for name in order:
        predecessors = by_name[name].depends_on
        depths[name] = 1 + max((depths[parent] for parent in predecessors), default=0)
    return max(depths.values(), default=0)


__all__ = ("MAX_DAG_DEPTH", "DagError", "DagNode", "validate")
