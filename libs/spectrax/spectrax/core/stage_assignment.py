# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Construction-time pipeline stage ownership hints.

``assign_stage(total=..., current=...)`` lets model construction stamp a
coarse-grained "where in the logical layer order was this variable
created?" hint onto every :class:`~spectrax.Variable` allocated inside
the scope. MPMD runtimes can then resolve that hint against the active
pipeline width and place the variable on the owning stage sub-mesh.

Example::

    for i in range(32):
        with assign_stage(total=32, current=i):
            blocks.append(MyBlock(...))

On a 4-stage mesh, layers 0-7 map to stage 0, 8-15 to stage 1, and so
on. The hint is stored in ``Variable.metadata["pipeline_stage"]`` as the
hashable tuple ``(current, total)``.
"""

from __future__ import annotations

import operator
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from .context import get as _scope_get
from .context import scope

__all__ = [
    "PIPELINE_STAGE_METADATA_KEY",
    "assign_stage",
    "current_stage_assignment",
    "metadata_stage_assignment",
    "resolve_stage_rank",
]


_PIPELINE_STAGE_SCOPE_KEY = "_spx_pipeline_stage_assignment"
"""Internal scope-stack key used by :func:`assign_stage`.

Kept private because the wire format (a normalized ``(current, total)``
tuple) and the lookup mechanism may change without notice.
"""

PIPELINE_STAGE_METADATA_KEY = "pipeline_stage"
"""Variable-metadata key under which the stage hint is stored.

This is part of the public metadata schema: third-party code that
inspects :attr:`spectrax.Variable.metadata` may look up this key to
recover the ``(current, total)`` pair stamped at construction time.
"""


def _normalize_assignment(current: object, total: object) -> tuple[int, int]:
    """Validate and normalize a ``(current, total)`` stage hint.

    Coerces both arguments through :func:`operator.index` so that any
    integer-like value (numpy scalars, custom int subclasses) is accepted,
    then enforces the invariant ``0 <= current < total`` with
    ``total >= 1``. The same routine guards every public entry point so
    a malformed assignment is rejected once and consistently.

    Args:
        current: Logical zero-based slot index.
        total: Total number of logical slots; must be at least one.

    Returns:
        A 2-tuple ``(current, total)`` of normalized Python ``int``.

    Raises:
        TypeError: If either argument is not integer-coercible.
        ValueError: If ``total < 1`` or ``current`` falls outside
            ``[0, total)``.
    """
    cur = operator.index(current)
    tot = operator.index(total)
    if tot <= 0:
        raise ValueError(f"assign_stage(...): total must be >= 1; got {tot}.")
    if not 0 <= cur < tot:
        raise ValueError(
            f"assign_stage(...): current must satisfy 0 <= current < total; got current={cur}, total={tot}."
        )
    return cur, tot


@contextmanager
def assign_stage(*, total: int, current: int) -> Iterator[None]:
    """Stamp subsequently-created variables with a pipeline stage hint.

    Pushes a frame onto :mod:`spectrax.core.context` carrying a normalized
    ``(current, total)`` tuple. object :class:`~spectrax.Variable` constructed
    inside the ``with`` block (for which the subclass opts in via
    :attr:`~spectrax.Variable.inherit_stage_assignment`) records the tuple
    under :data:`PIPELINE_STAGE_METADATA_KEY` in
    :attr:`~spectrax.Variable.metadata`. The hint is purely *logical* — it
    describes a slot index inside an abstract pipeline, not a concrete
    device — so the same construction code targets meshes of varying
    pipeline width without modification.

    Args:
        total: Number of logical pipeline slots assumed by the model
            construction code; must be ``>= 1``.
        current: Zero-based slot index for the variables created inside
            the block; must satisfy ``0 <= current < total``.

    Yields:
        ``None``. The hint is active only for the dynamic extent of the
        ``with`` body.

    Returns:
        A context manager that yields ``None``.

    Raises:
        TypeError: If ``current`` or ``total`` are not integer-coercible.
        ValueError: If the normalized values violate the slot invariants.

    Example::

        for i in range(num_layers):
            with spx.assign_stage(total=num_layers, current=i):
                layers.append(TransformerBlock(...))
    """
    assignment = _normalize_assignment(current=current, total=total)
    with scope(**{_PIPELINE_STAGE_SCOPE_KEY: assignment}):
        yield


def current_stage_assignment() -> tuple[int, int] | None:
    """Return the innermost active :func:`assign_stage` hint, if any.

    Looks up the scope key written by :func:`assign_stage` on the
    per-task contextvars stack and re-normalizes the result. Variable
    constructors call this during ``__init__`` to decide whether to
    stamp the freshly-created cell with a pipeline-stage hint.

    Returns:
        The normalized ``(current, total)`` tuple of the innermost
        active scope, or ``None`` when no scope is active.
    """
    assignment = _scope_get(_PIPELINE_STAGE_SCOPE_KEY, None)
    if assignment is None:
        return None
    return _normalize_assignment(*assignment)


def metadata_stage_assignment(metadata: Mapping[str, object] | None) -> tuple[int, int] | None:
    """Extract a normalized stage hint from variable metadata.

    Reads :data:`PIPELINE_STAGE_METADATA_KEY` from a variable's
    metadata dict and validates the stored payload. Used by
    :class:`~spectrax.Variable` properties such as
    :attr:`~spectrax.Variable.stage_assignment` and by sharding helpers
    that need to know which pipeline stage owns a leaf.

    Args:
        metadata: A variable's metadata mapping (``None`` or empty maps
            are treated as "no hint present").

    Returns:
        The normalized ``(current, total)`` tuple, or ``None`` when the
        key is absent.

    Raises:
        ValueError: If the stored value is not a 2-element tuple, or
            if its components fail :func:`_normalize_assignment`'s
            invariants.
    """
    if not metadata or PIPELINE_STAGE_METADATA_KEY not in metadata:
        return None
    raw = metadata[PIPELINE_STAGE_METADATA_KEY]
    if not isinstance(raw, tuple) or len(raw) != 2:
        raise ValueError(
            f"Variable metadata[{PIPELINE_STAGE_METADATA_KEY!r}] must be a 2-tuple ``(current, total)``; got {raw!r}."
        )
    return _normalize_assignment(*raw)


def resolve_stage_rank(assignment: tuple[int, int] | None, mpmd_dim: int) -> int | None:
    """Resolve a logical ``(current, total)`` hint to a physical MPMD rank.

    Maps the abstract slot index recorded at construction time onto the
    pipeline width of the currently-active MPMD mesh. The mapping is
    proportional: ``rank = min(mpmd_dim - 1, current * mpmd_dim // total)``,
    so 32 logical layers on a 4-stage mesh place layers 0-7 on stage 0,
    8-15 on stage 1, and so on. The clamp guards against pathological
    cases where the proportional formula would round up to ``mpmd_dim``.

    Args:
        assignment: A previously-normalized ``(current, total)`` tuple,
            typically obtained from :func:`current_stage_assignment` or
            :func:`metadata_stage_assignment`. ``None`` short-circuits.
        mpmd_dim: Pipeline width of the target MPMD mesh; must be
            ``>= 1``.

    Returns:
        The zero-based physical stage rank in ``[0, mpmd_dim)``, or
        ``None`` when ``assignment`` is ``None``.

    Raises:
        ValueError: If ``mpmd_dim < 1`` or ``assignment`` violates
            :func:`_normalize_assignment`'s invariants.
    """
    if assignment is None:
        return None
    if mpmd_dim < 1:
        raise ValueError(f"mpmd_dim must be >= 1; got {mpmd_dim}.")
    current, total = _normalize_assignment(*assignment)
    return min(mpmd_dim - 1, (current * mpmd_dim) // total)
