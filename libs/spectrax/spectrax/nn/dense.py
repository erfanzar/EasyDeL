# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generalised dense layers — :class:`DenseGeneral` and :class:`Einsum`.

These two layers extend :class:`~spectrax.nn.Linear` along orthogonal
axes:

* :class:`DenseGeneral` keeps the implicit "input axes contract,
  output axes broadcast" structure of a standard dense layer, but lets
  callers pick *which* input axes to contract and *what shape* of new
  trailing axes to produce. It is implemented with
  :func:`jax.numpy.tensordot`.
* :class:`Einsum` exposes an arbitrary :func:`jax.numpy.einsum`
  equation between the input and a learnable weight tensor. It is the
  most flexible primitive and the one used internally by attention
  blocks that fold the per-head reshape into the matmul.

Both layers attach the standard "axis name" sharding hints; both
default to a Kaiming-uniform weight init and zero bias.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType, Initializer
from ..core.module import Module
from ..core.sharding import AxisNames, Sharding
from ..core.variable import Parameter
from ..init import kaiming_uniform, zeros
from ..rng.rngs import Rngs, resolve_rngs

__all__ = ["DenseGeneral", "Einsum"]


def _to_tuple(x: int | Sequence[int]) -> tuple[int, ...]:
    """Coerce ``x`` to a tuple, wrapping a bare ``int`` in a length-1 tuple.

    Args:
        x: Either a single ``int`` or any sequence of ints.

    Returns:
        ``(x,)`` when ``x`` is an ``int``; ``tuple(x)`` otherwise.
    """
    return (x,) if isinstance(x, int) else tuple(x)


def _normalize_axes(axes: Sequence[int], ndim: int) -> tuple[int, ...]:
    """Resolve possibly-negative axis indices against ``ndim``.

    Used to translate user-facing negative axis arguments (which are
    ergonomic but not accepted by every JAX primitive) into canonical
    non-negative positions.

    Args:
        axes: Axis indices, each in ``[-ndim, ndim)``.
        ndim: Rank of the array against which the axes were specified.

    Returns:
        A tuple of non-negative axis indices, each in ``[0, ndim)``.
    """
    return tuple(a % ndim for a in axes)


class DenseGeneral(Module):
    """Dense layer that contracts over arbitrary input axes.

    ``axis`` selects which axes of the input ``x`` participate in the
    contraction; ``features`` specifies the shape of the new trailing
    axes appended to the result. The weight has shape
    ``(*in_shape, *features)`` and the bias (when used) has shape
    ``features``. Internally the contraction is computed via
    :func:`jax.numpy.tensordot`.

    Because the weight is allocated eagerly the contracted-axis sizes
    must be supplied via ``in_shape=`` at construction — there is no
    deferred-shape mode.

    Example::

        >>> layer = DenseGeneral(features=(4, 8), axis=(-2, -1),
        ...                      in_shape=(2, 6), rngs=Rngs(0))
        >>> y = layer(jnp.zeros((3, 5, 2, 6)))
        >>> y.shape
        (3, 5, 4, 8)
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        features: int | Sequence[int],
        *,
        axis: int | Sequence[int] = -1,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        in_shape: Sequence[int] | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize the layer.

        Args:
            features: Shape of the new trailing axes appended by the
                contraction. Either a single ``int`` (one new axis)
                or a sequence of ints.
            axis: Input axis (or axes) to contract against the weight.
                Negative indices are accepted and resolved against the
                input's ndim at call time. The order of ``axis``
                determines the order in which the corresponding sizes
                appear in ``in_shape`` and on the leading axes of the
                weight.
            use_bias: When ``True``, allocate a bias of shape
                ``features``.
            rngs: PRNG source for parameter initialization. Accepts
                an :class:`Rngs`, an ``int`` seed, or ``None``.
            w_init: Weight initializer; defaults to
                :func:`~spectrax.init.kaiming_uniform` with the
                ``"linear"`` gain.
            b_init: Bias initializer; defaults to
                :func:`~spectrax.init.zeros`.
            dtype: Storage dtype for the parameters; defaults to
                ``float32`` when both ``dtype`` and ``param_dtype``
                are ``None``.
            param_dtype: Alias for ``dtype``; takes precedence when
                both are supplied.
            in_shape: **Required.** Sizes of the contracted axes in
                the same order as ``axis``. Used to allocate the
                weight eagerly.
            sharding: Optional sharding for the weight.
            bias_sharding: Optional sharding for the bias.

        Raises:
            ValueError: If ``in_shape`` is ``None`` or its length
                does not match ``axis``.
        """
        super().__init__()
        self.features = _to_tuple(features)
        self.axis = _to_tuple(axis)
        self.use_bias = use_bias
        if in_shape is None:
            raise ValueError("DenseGeneral requires in_shape=(..) for the contracted axes")
        self.in_shape = tuple(in_shape)
        if len(self.in_shape) != len(self.axis):
            raise ValueError("in_shape length must equal axis length")
        resolved = resolve_rngs(rngs)
        dt = param_dtype or dtype or jnp.float32
        w_init = w_init or kaiming_uniform("linear")
        wshape = (*self.in_shape, *self.features)
        self.weight = Parameter(w_init(resolved.parameters, wshape, dt), sharding=sharding)
        if use_bias:
            b_init = b_init or zeros
            self.bias = Parameter(
                b_init(resolved.parameters, tuple(self.features), dt),
                sharding=bias_sharding,
            )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Contract ``x`` with the weight along ``axis`` and add the bias.

        Computes ``jnp.tensordot(x, weight, axes=(axis, leading_weight_axes))``
        and broadcasts the bias along the result's non-feature axes.

        Args:
            x: Input tensor whose sizes along ``axis`` must match
                ``in_shape`` (in the same order).
            **_: Ignored; accepted for container interoperability.

        Returns:
            An array whose shape is ``x.shape`` with the contracted
            axes removed and the ``features`` shape appended.
        """
        xa = jnp.asarray(x)
        axes = _normalize_axes(self.axis, xa.ndim)
        contracting = list(axes)
        n_contract = len(contracting)
        n_features = len(self.features)
        y = jnp.tensordot(xa, self.weight.value, axes=(contracting, list(range(n_contract))))
        if self.use_bias:
            expand = (None,) * (y.ndim - n_features) + (slice(None),) * n_features
            y = y + self.bias.value[expand]
        return y


class Einsum(Module):
    """Learnable einsum: input combines with a learned weight via an equation.

    The equation describes how the input ``x`` and the learnable
    parameter ``weight`` combine; ``shape`` declares the weight's shape
    so it can be allocated at construction time. The equation must
    contain ``"->"`` (an explicit output specification); the operands
    are passed in the order ``(x, weight)``.

    Example::

        >>> e = Einsum("...ij,jk->...ik", shape=(4, 8), rngs=Rngs(0))
        >>> e(jnp.zeros((3, 2, 4))).shape
        (3, 2, 8)
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        equation: str,
        shape: Sequence[int],
        *,
        use_bias: bool = False,
        bias_shape: Sequence[int] | None = None,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize the layer.

        Args:
            equation: An :func:`jax.numpy.einsum` equation in
                explicit form (must contain ``"->"``). The first
                operand is the input ``x`` and the second is the
                learnable :attr:`weight`.
            shape: Shape of :attr:`weight` — used to allocate it
                eagerly via the chosen initializer.
            use_bias: When ``True``, allocate a bias added (with
                broadcasting) to the einsum output. Requires
                ``bias_shape`` to be specified.
            bias_shape: Shape of the bias. Required when
                ``use_bias=True``; not auto-inferred from the
                equation in this implementation.
            rngs: PRNG source for initialization. Accepts an
                :class:`Rngs`, an ``int`` seed, or ``None``.
            w_init: Weight initializer; defaults to
                :func:`~spectrax.init.kaiming_uniform` with the
                ``"linear"`` gain.
            b_init: Bias initializer; defaults to
                :func:`~spectrax.init.zeros`.
            dtype: Storage dtype for the parameters; defaults to
                ``float32`` when both ``dtype`` and ``param_dtype``
                are ``None``.
            param_dtype: Alias for ``dtype``; takes precedence when
                both are supplied.
            sharding: Optional sharding for the weight.
            bias_sharding: Optional sharding for the bias.

        Raises:
            ValueError: If ``equation`` does not contain ``"->"``,
                or if ``use_bias=True`` is given without a
                ``bias_shape``.
        """
        super().__init__()
        if "->" not in equation:
            raise ValueError("Einsum equation must contain '->'")
        self.equation = equation
        self.shape = tuple(shape)
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        dt = param_dtype or dtype or jnp.float32
        w_init = w_init or kaiming_uniform("linear")
        self.weight = Parameter(w_init(resolved.parameters, self.shape, dt), sharding=sharding)
        if use_bias:
            if bias_shape is None:
                raise ValueError("Einsum(use_bias=True) requires bias_shape=(..)")
            b_init = b_init or zeros
            self.bias = Parameter(
                b_init(resolved.parameters, tuple(bias_shape), dt),
                sharding=bias_sharding,
            )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Run ``jnp.einsum(equation, x, weight)`` and add the bias.

        Args:
            x: Input tensor whose shape must satisfy the input side
                of :attr:`equation`.
            **_: Ignored; accepted for container interoperability.

        Returns:
            The einsum result plus :attr:`bias` (broadcast) when
            ``use_bias=True``, otherwise just the einsum result.
        """
        y = jnp.einsum(self.equation, jnp.asarray(x), self.weight.value)
        if self.use_bias:
            y = y + self.bias.value
        return y
