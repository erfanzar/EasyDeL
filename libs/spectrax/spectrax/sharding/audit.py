# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Placement audit: check that every variable lives where its metadata says.

SpectraX resolves variable placement through several layers (logical axis
rules, ``assign_stage`` hints, spec sanitization) and degrades silently when
intent cannot be honored. :func:`audit_placement` makes those decisions
visible::

    report = spx.sharding.audit_placement(model, mesh)
    print(report.format())
    report.raise_for_errors()

Issue codes:

* ``misplaced`` (error): the value's sharding is not equivalent to the
  sharding its metadata resolves to (wrong stage sub-mesh, wrong spec).
* ``unresolvable`` (error): the metadata cannot be turned into a sharding.
* ``dropped_axis`` (warning): a declared axis is silently removed (pipeline
  axis, axis missing from the mesh, non-divisible dimension).
* ``replicated_across_stages`` (warning): an untagged variable on a
  multi-stage MPMD mesh, so every stage holds a full copy.
* ``unmaterialized`` (warning): a deferred variable that has no value yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
from jax.sharding import NamedSharding, PartitionSpec

from ..core.errors import ShardingError
from ..core.graph import live_variables
from ..core.module import Module
from ..core.sharding import Sharding, normalize_sharding
from ..core.stage_assignment import metadata_stage_assignment, resolve_stage_rank
from ..core.variable import DeferredBuffer, DeferredParameter
from .partition import (
    _axis_rules_for_mesh,
    _resolve_named_sharding_mesh,
    explain_partition_spec_sanitization,
)

if TYPE_CHECKING:
    from jax.sharding import Mesh

    from ..runtime.types.mesh import MpMdMesh
    from .mesh import SpxMesh

__all__ = ["PlacementIssue", "PlacementReport", "audit_placement"]

_ERROR_CODES = frozenset({"misplaced", "unresolvable"})


@dataclass(frozen=True)
class PlacementIssue:
    """One finding from :func:`audit_placement`."""

    code: str
    severity: str
    path: str
    collection: str
    message: str


@dataclass(frozen=True)
class PlacementReport:
    """Result of :func:`audit_placement`.

    Attributes:
        issues: Every finding, in variable-path order.
        checked: Number of variables inspected.
        stage_counts: ``{stage_index_or_None: variable_count}``.
    """

    issues: tuple[PlacementIssue, ...]
    checked: int
    stage_counts: dict[int | None, int] = field(default_factory=dict)

    @property
    def errors(self) -> tuple[PlacementIssue, ...]:
        """Findings with ``severity == "error"``."""
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[PlacementIssue, ...]:
        """Findings with ``severity == "warning"``."""
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def ok(self) -> bool:
        """``True`` when there are no errors (warnings are allowed)."""
        return not self.errors

    def codes(self) -> dict[str, int]:
        """Count findings per issue code."""
        out: dict[str, int] = {}
        for issue in self.issues:
            out[issue.code] = out.get(issue.code, 0) + 1
        return out

    def format(self) -> str:
        """Human-readable rendering."""
        stages = ", ".join(
            f"{'untagged' if k is None else f'stage{k}'}={v}"
            for k, v in sorted(self.stage_counts.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))
        )
        lines = [
            f"placement audit: {self.checked} variable(s), {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s); {stages or 'no variables'}"
        ]
        for issue in self.issues:
            lines.append(f"  [{issue.severity}] {issue.code} {issue.collection}:{issue.path}: {issue.message}")
        return "\n".join(lines)

    def raise_for_errors(self) -> None:
        """Raise :class:`~spectrax.ShardingError` if the report has errors."""
        if self.errors:
            raise ShardingError(self.format())


def _nbytes(value: object) -> int:
    size = getattr(value, "size", None)
    dtype = getattr(value, "dtype", None)
    try:
        return int(size) * int(jax.numpy.dtype(dtype).itemsize)
    except Exception:
        return 0


def audit_placement(
    module: Module,
    mesh: "Mesh | SpxMesh | MpMdMesh",
    *,
    replicated_warn_bytes: int = 1 << 20,
    check_values: bool = True,
) -> PlacementReport:
    """Audit where ``module``'s variables live against where their metadata says.

    Args:
        module: Root module.
        mesh: The mesh the model is (or will be) placed on.
        replicated_warn_bytes: Minimum size for a ``replicated_across_stages``
            warning; ``0`` reports every untagged variable.
        check_values: Compare each value's actual sharding with the expected
            one. Pass ``False`` to audit only the metadata (before placement).

    Returns:
        A :class:`PlacementReport`.
    """
    base_mesh, mpmd_mesh = _resolve_named_sharding_mesh(mesh)
    multi_stage = mpmd_mesh is not None and mpmd_mesh.mpmd_dim > 1
    rules = _axis_rules_for_mesh(base_mesh, mpmd_mesh)
    issues: list[PlacementIssue] = []
    stage_counts: dict[int | None, int] = {}
    checked = 0

    for path, var in live_variables(module):
        checked += 1
        collection = var.kind

        def add(code: str, message: str, *, _path: str = path, _collection: str = collection) -> None:
            severity = "error" if code in _ERROR_CODES else "warning"
            issues.append(PlacementIssue(code, severity, _path, _collection, message))

        if isinstance(var, DeferredParameter | DeferredBuffer) and not var.is_materialized:
            stage_counts[None] = stage_counts.get(None, 0) + 1
            add("unmaterialized", "deferred variable has no value; call module.init(rngs, *example) first")
            continue

        value = var._raw_get()
        shape = tuple(getattr(value, "shape", ()))
        try:
            owner = (
                resolve_stage_rank(metadata_stage_assignment(var.metadata), mpmd_mesh.mpmd_dim)
                if mpmd_mesh is not None
                else None
            )
            declared = normalize_sharding(var.metadata.get("sharding"))
            if declared is None and var.metadata.get("axis_names") is not None:
                declared = Sharding(axis_names=tuple(var.metadata["axis_names"]))
            spec = declared.to_partition_spec(rules) if declared is not None else PartitionSpec()
            spec, reasons = explain_partition_spec_sanitization(spec, mesh=mpmd_mesh or base_mesh, shape=shape)
            target_mesh = mpmd_mesh.submesh(owner) if (mpmd_mesh is not None and owner is not None) else base_mesh
            expected = NamedSharding(target_mesh, spec)
        except Exception as exc:
            stage_counts[None] = stage_counts.get(None, 0) + 1
            add("unresolvable", f"{type(exc).__name__}: {exc}")
            continue

        stage_counts[owner] = stage_counts.get(owner, 0) + 1
        for reason in reasons:
            add("dropped_axis", f"declared {declared} shape={shape}: {reason}")

        if multi_stage and owner is None:
            nbytes = _nbytes(value)
            if nbytes >= replicated_warn_bytes:
                add(
                    "replicated_across_stages",
                    f"no pipeline_stage hint; {nbytes} bytes replicated on all {mpmd_mesh.mpmd_dim} stages "
                    "(wrap construction in spx.assign_stage(...) to place it on one stage)",
                )

        if check_values and isinstance(value, jax.Array):
            actual = value.sharding
            try:
                equivalent = actual.is_equivalent_to(expected, len(shape))
            except Exception:
                equivalent = False
            if not equivalent:
                actual_desc = (
                    f"{getattr(actual, 'spec', actual)} on {len(actual.device_set)} device(s) "
                    f"{sorted(d.id for d in actual.device_set)[:8]}"
                )
                expected_desc = (
                    f"{expected.spec} on {'stage ' + str(owner) if owner is not None else 'full mesh'} "
                    f"{sorted(d.id for d in expected.device_set)[:8]}"
                )
                add("misplaced", f"actual {actual_desc}; expected {expected_desc}")

    return PlacementReport(issues=tuple(issues), checked=checked, stage_counts=stage_counts)
