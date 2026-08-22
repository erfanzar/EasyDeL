# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":class:`QArray` — a quantized array with subchannel support.

A :class:`QArray` bundles the three things needed to reconstruct an
approximation of an original floating-point array::

    original ~= (qvalue - zero_point) * generic_broadcast(scale, original.shape)

The interesting part is *generic broadcast*. NumPy broadcasting only
stretches size-one axes, which restricts a scale to being either
per-tensor or per-channel. Quantization wants a third option: one scale
per contiguous *tile* of a long axis ("subchannel" or "group-wise"
quantization), where a ``(4096, 4096)`` weight carries a
``(32, 4096)`` scale meaning "one scale per 128 rows".

So this module defines its own broadcast rule: ``scale`` has the same
rank as ``qvalue``, and each of its axes either matches, is one, or
*divides* the corresponding ``qvalue`` axis. :func:`generic_broadcast_op`
implements that by reshaping both operands to a common split shape,
which costs nothing at runtime — XLA sees only reshapes.

The rest of the module is the mechanical support that rule implies:
:func:`split_axis` (turn a tiled axis into ``(tile_count, tile_size)``),
:func:`transpose_array` (a transpose that tolerates missing and new
axes, needed to line a scale up with a dot_general output), and
:func:`tiled_axes` (recover the tiling from the shapes alone, so a
``QArray`` never has to carry it as metadata).

Ported from Google's Qwix (``qwix._src.core.qarray``, Apache-2.0); see
:mod:`spectrax.quantization._numerics` for why it is reimplemented here
rather than depended on.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias

import jax
import jax.numpy as jnp

from ..core._typing import Array
from ._numerics import QType, qtype_name

__all__ = [
    "MaybeQArray",
    "QArray",
    "generic_broadcast_op",
    "resolve_tile_size",
    "split_axis",
    "tiled_axes",
    "transpose_array",
    "validate_qarray",
]


@dataclasses.dataclass(frozen=True)
class QArray:
    """A quantized array: integer or narrow-float values plus their scale.

    Attributes:
        qvalue: The quantized values. Same shape as the original array.
        scale: The per-tile scale. Same rank as ``qvalue``; each axis
            either equals, is one, or divides the matching ``qvalue`` axis.
        zero_point: The quantized value representing exact floating-point
            zero, or ``None`` for symmetric quantization. Same shape and
            dtype rules as ``scale``/``qvalue`` respectively.
        qtype: The *logical* type of ``qvalue``, which may differ from
            ``qvalue.dtype`` — a 3-bit value is stored in an ``int4``
            element, and an ``"nf4"`` value in a ``uint4`` element.
    """

    qvalue: Array
    scale: Array
    zero_point: Array | None = None
    qtype: QType | None = None

    def __post_init__(self) -> None:
        """Default :attr:`qtype` to the storage dtype when unspecified."""
        if self.qtype is None:
            object.__setattr__(self, "qtype", self.qvalue.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array this quantizes (that is, ``qvalue.shape``)."""
        return self.qvalue.shape

    @property
    def ndim(self) -> int:
        """Rank of the array this quantizes."""
        return self.qvalue.ndim

    @property
    def dtype(self) -> jnp.dtype:
        """Dtype the array dequantizes *to*, taken from :attr:`scale`."""
        return self.scale.dtype

    @property
    def scale_tile_shape(self) -> tuple[int, ...]:
        """Per-axis number of ``qvalue`` elements sharing one scale."""
        return tuple(v // s for v, s in zip(self.shape, self.scale.shape, strict=True))

    def astype(self, dtype: jnp.dtype) -> QArray:
        """Change the dtype this array dequantizes to.

        Only the scale is cast: the quantized values keep their storage
        type, and it is the scale's dtype that decides the output of
        :func:`~spectrax.quantization.dequantize`.

        Args:
            dtype: The floating-point dtype to dequantize into.

        Returns:
            A new :class:`QArray` with the recast scale.
        """
        return dataclasses.replace(self, scale=self.scale.astype(dtype))

    def transpose(self, *axes: int) -> QArray:
        """Transpose every component in lock step.

        Args:
            *axes: Permutation, as accepted by :meth:`jax.Array.transpose`.

        Returns:
            A transposed :class:`QArray`.
        """
        return jax.tree.map(lambda x: x.transpose(*axes), self)

    def swapaxes(self, axis1: int, axis2: int) -> QArray:
        """Swap two axes of every component in lock step.

        Args:
            axis1: First axis.
            axis2: Second axis.

        Returns:
            A :class:`QArray` with the two axes exchanged.
        """
        return jax.tree.map(lambda x: x.swapaxes(axis1, axis2), self)

    def __repr__(self) -> str:
        """Return a compact summary naming the logical type and tiling."""
        tiles = self.scale_tile_shape
        return (
            f"QArray(shape={self.shape}, qtype={qtype_name(self.qtype)}, "
            f"scale_tile_shape={tiles}, asymmetric={self.zero_point is not None})"
        )


MaybeQArray: TypeAlias = "Array | QArray"
"""Either a plain array or a quantized one; the operand type of quantized ops."""


def _qarray_flatten_with_keys(
    array: QArray,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, Array | None], ...], QType | None]:
    """Flatten a :class:`QArray` for :func:`jax.tree_util.register_pytree_with_keys`.

    Args:
        array: The array to flatten.

    Returns:
        A pair of the keyed children and the static :attr:`QArray.qtype`.
    """
    children = (
        (jax.tree_util.GetAttrKey("qvalue"), array.qvalue),
        (jax.tree_util.GetAttrKey("scale"), array.scale),
        (jax.tree_util.GetAttrKey("zero_point"), array.zero_point),
    )
    return children, array.qtype


def _qarray_unflatten(qtype: QType | None, children: Sequence[Array | None]) -> QArray:
    """Rebuild a :class:`QArray` from its flattened form.

    Args:
        qtype: The static logical type recorded at flatten time.
        children: The ``(qvalue, scale, zero_point)`` leaves.

    Returns:
        The reconstructed :class:`QArray`.
    """
    qvalue, scale, zero_point = children
    return QArray(qvalue=qvalue, scale=scale, zero_point=zero_point, qtype=qtype)


jax.tree_util.register_pytree_with_keys(
    QArray,
    _qarray_flatten_with_keys,
    _qarray_unflatten,
    lambda array: ((array.qvalue, array.scale, array.zero_point), array.qtype),
)


def validate_qarray(array: QArray) -> None:
    """Check a :class:`QArray`'s internal consistency.

    Called at the entry of quantized ops so that a malformed array is
    reported where it was built rather than as an opaque shape error
    inside a matmul.

    Args:
        array: The array to validate.

    Raises:
        ValueError: If the scale's rank differs from the value's, if any
            scale or zero-point axis does not divide the matching value
            axis, if the quantized values are wider than one byte, if the
            scale is not a floating-point type, or if the zero point's
            dtype does not match the quantized values'.
    """
    if not isinstance(array.qvalue, jax.Array | jnp.ndarray):
        return
    if array.qvalue.ndim != array.scale.ndim:
        raise ValueError(
            f"scale {array.scale.shape} must have the same rank as qvalue {array.qvalue.shape}; "
            "insert size-one axes rather than squeezing them away."
        )
    if not all(v % s == 0 for v, s in zip(array.qvalue.shape, array.scale.shape, strict=True)):
        raise ValueError(
            f"scale {array.scale.shape} is not generic-broadcastable to qvalue {array.qvalue.shape}: "
            "every scale axis must divide the matching qvalue axis."
        )
    if array.qvalue.dtype.itemsize > 1:
        raise ValueError(f"{array.qvalue.dtype} is too wide to hold quantized values.")
    if array.scale.dtype not in (jnp.bfloat16, jnp.float32, jnp.float64):
        raise ValueError(f"{array.scale.dtype} is not a valid scale dtype; use bfloat16, float32 or float64.")
    if array.zero_point is not None:
        if array.zero_point.ndim != array.qvalue.ndim:
            raise ValueError(
                f"zero_point {array.zero_point.shape} must have the same rank as qvalue {array.qvalue.shape}."
            )
        if not all(v % z == 0 for v, z in zip(array.qvalue.shape, array.zero_point.shape, strict=True)):
            raise ValueError(
                f"zero_point {array.zero_point.shape} is not generic-broadcastable to qvalue {array.qvalue.shape}."
            )
        if array.zero_point.dtype != array.qvalue.dtype:
            raise ValueError(
                f"zero_point dtype {array.zero_point.dtype} must match qvalue dtype {array.qvalue.dtype}."
            )


def resolve_tile_size(dim: int, tile_size: int | float) -> int:
    """Turn a tile-size spec into a concrete element count.

    An integer is a literal tile size. A float is interpreted as
    ``1 / tile_count``, which is what lets a rule say "split this axis
    into as many tiles as there are shards" without knowing the axis
    length. The two are deliberately not interchangeable: ``1`` means one
    element per tile (per-channel scales) while ``1.0`` means one tile
    for the whole axis (a shared scale).

    Args:
        dim: Length of the axis being tiled.
        tile_size: Literal size (``int``) or reciprocal tile count (``float``).

    Returns:
        The tile size in elements.

    Raises:
        ValueError: If the resolved tile size is not positive or does not
            divide ``dim`` evenly.
    """
    resolved = round(dim * tile_size) if isinstance(tile_size, float) else int(tile_size)
    if resolved <= 0 or dim % resolved != 0:
        raise ValueError(
            f"tile_size={tile_size!r} resolves to {resolved} elements, which does not evenly divide axis length {dim}."
        )
    return resolved


def split_axis(array: Array, tiled: Mapping[int, int | float]) -> Array:
    """Reshape tiled axes into ``(tile_count, tile_size)`` pairs.

    This is the reshape that makes subchannel quantization expressible
    with ordinary reductions: after splitting, "reduce within each tile"
    is just a reduction over the inserted trailing axis.

    Args:
        array: The array to reshape.
        tiled: Mapping from axis index to tile size, in the *original*
            axis numbering.

    Returns:
        The reshaped array, with one extra axis per entry in ``tiled``.

    Raises:
        ValueError: If a tile size does not divide its axis.
    """
    if not tiled:
        return array
    new_shape: list[int] = []
    for axis, dim in enumerate(array.shape):
        if axis in tiled:
            size = resolve_tile_size(dim, tiled[axis])
            new_shape.extend((dim // size, size))
        else:
            new_shape.append(dim)
    return array.reshape(new_shape)


def tiled_axes(array: QArray) -> dict[int, int]:
    """Recover which axes are subchannel-tiled, and by how much.

    Derived from the shapes rather than stored, so a :class:`QArray` that
    survives a round trip through a pytree transform cannot disagree with
    itself about its own tiling. An axis whose scale extent is 1
    (per-tensor) or equal to the value extent (per-channel) is not
    "tiled" in this sense and is omitted.

    Args:
        array: The quantized array to inspect.

    Returns:
        Mapping from axis index to tile size, for genuinely tiled axes.
    """
    return {
        axis: value_dim // scale_dim
        for axis, (value_dim, scale_dim) in enumerate(zip(array.qvalue.shape, array.scale.shape, strict=True))
        if value_dim != scale_dim and scale_dim != 1
    }


def transpose_array(array: Array, order: Sequence[int | None]) -> Array:
    """Transpose ``array``, allowing axes to be dropped and inserted.

    A generalisation of :func:`jax.numpy.transpose` needed to line a
    scale up with a ``dot_general`` result, where some of the scale's
    axes vanish (they were contracted) and some must be created (they
    belong to the other operand):

    * an entry of ``None`` inserts a new size-one axis;
    * an axis of ``array`` absent from ``order`` is squeezed away, which
      is only legal if its extent is one.

    The implementation prefers reshape and squeeze over a real transpose
    wherever the retained axes are already in order, because Pallas
    lowers reshapes but not general transposes.

    Args:
        array: The array to rearrange.
        order: Target layout: source axis index, or ``None`` for a new axis.

    Returns:
        The rearranged array, of rank ``len(order)``.

    Raises:
        ValueError: If an axis dropped by ``order`` has extent greater
            than one, which would lose data.
    """
    dropped = [axis for axis, dim in enumerate(array.shape) if axis not in order and dim > 1]
    if dropped:
        raise ValueError(f"Cannot transpose {array.shape} as {list(order)}: axes {dropped} would be dropped.")
    target_shape = [1 if axis is None else array.shape[axis] for axis in order]
    kept = [axis for axis in order if axis is not None and array.shape[axis] != 1]
    if sorted(kept) == kept:
        return array.reshape(target_shape)
    squeezed = array.squeeze([axis for axis in range(array.ndim) if axis not in kept])
    return squeezed.transpose([sum(other < axis for other in kept) for axis in kept]).reshape(target_shape)


def generic_broadcast_op(op: Callable[[Array, Array], Array], x: Array, y: Array) -> Array:
    """Apply an elementwise binary ``op`` under the generic broadcast rule.

    Standard broadcasting stretches only size-one axes. Here an axis of
    one operand may also *divide* the matching axis of the other, meaning
    "one value per tile". Such a pair is handled by splitting the longer
    axis into ``(tile_count, tile_size)`` and giving the shorter one a
    trailing size-one axis, after which ordinary broadcasting applies.
    The result is reshaped back, so callers never see the split shape.

    Args:
        op: An elementwise binary operation, e.g. :func:`jax.numpy.multiply`.
        x: First operand.
        y: Second operand, of the same rank as ``x``.

    Returns:
        ``op(x, y)`` at the elementwise-maximum shape.

    Raises:
        AssertionError: If the operands differ in rank.
        ValueError: If some axis pair neither matches, contains a one, nor
            divides evenly in one direction.
    """
    assert x.ndim == y.ndim, f"generic_broadcast_op needs equal ranks, got {x.shape} and {y.shape}."
    x_shape: list[int] = []
    y_shape: list[int] = []
    out_shape: list[int] = []
    for a, b in zip(x.shape, y.shape, strict=True):
        out_shape.append(max(a, b))
        if a == b or a == 1 or b == 1:
            x_shape.append(a)
            y_shape.append(b)
        elif a % b == 0:
            x_shape.extend((b, a // b))
            y_shape.extend((b, 1))
        elif b % a == 0:
            x_shape.extend((a, 1))
            y_shape.extend((a, b // a))
        else:
            raise ValueError(f"Cannot generic-broadcast between {x.shape} and {y.shape}: axis pair ({a}, {b}).")
    return op(x.reshape(x_shape), y.reshape(y_shape)).reshape(out_shape)
