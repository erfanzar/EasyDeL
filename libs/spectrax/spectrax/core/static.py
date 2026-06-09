# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Static module attributes.

A *static* module attribute is one whose identity and value contribute to
:class:`~spectrax.GraphDef`, and whose change therefore forces a retrace
under transforms. Most hyperparameters (``num_heads``, ``use_bias``, …)
are naturally static: they are plain immutable Python scalars assigned
during ``__init__``. :class:`Static` is the explicit generic form that
makes the static-ness visible in type signatures and also supports
wrapping compound-but-hashable values.
"""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Static(Generic[T]):
    """Explicit marker that an attribute belongs to :class:`~spectrax.GraphDef`.

    Wrapping a value in ``Static`` declares that it is a hyperparameter:
    its identity participates in graph-def equality and hashing, and any
    change triggers a retrace under spectrax transforms.

    Example:
        ``self.activation = Static("gelu")``

    The stored value is available as ``.value`` and is compared
    structurally.
    """

    __slots__ = ("value",)

    def __init__(self, value: T) -> None:
        """Wrap ``value`` as a static marker.

        Args:
            value: Value consumed by the helper.
        """
        self.value = value

    def __repr__(self) -> str:
        """Return ``Static(<repr of value>)``.

        Returns:
            Return ``Static(<repr of value>)``.
        """
        return f"Static({self.value!r})"

    def __eq__(self, other: object) -> bool:
        """Structural equality: two markers are equal iff their values are.

        Args:
            other: Other value consumed by this operation.

        Returns:
            Result described by this helper.
        """
        if isinstance(other, Static):
            return self.value == other.value
        return NotImplemented

    def __hash__(self) -> int:
        """Hash derived from the wrapped value.

        Returns:
            Result described by this helper.
        """
        return hash(("spectrax.Static", self.value))


def is_static_scalar(x: object) -> bool:
    """Return ``True`` iff ``x`` is safe to embed directly into a graph-def.

    A value is a static scalar if it is hashable, immutable, and semantically
    stable across processes: ``None``, Python numerics, strings, bytes,
    :class:`Static` markers, or tuples/frozensets recursively composed of
    the same. Anything else (lists, dicts, arbitrary objects) fails the
    check and must either be wrapped in :class:`Static` or stored as a
    :class:`~spectrax.Variable`/``Module`` child.

    Args:
        x: The value to test.

    Returns:
        ``True`` when ``x`` qualifies as a static scalar.
    """
    if x is None or isinstance(x, bool | int | float | complex | str | bytes):
        return True
    if isinstance(x, Static):
        return True
    if isinstance(x, tuple):
        return all(is_static_scalar(e) for e in x)
    if isinstance(x, frozenset):
        return all(is_static_scalar(e) for e in x)
    return False
