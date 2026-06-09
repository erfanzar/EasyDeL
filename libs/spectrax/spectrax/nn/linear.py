# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dense (fully-connected) layers: :class:`Linear` and :class:`Bilinear`.

:class:`Linear` is the canonical ``y = x @ W + b`` layer; :class:`Bilinear`
implements a learned three-way interaction ``y[..., o] = sum(x1[..., i]
* W[i, j, o] * x2[..., j]) + b[o]`` between two distinct input streams.
Both layers obey the framework-wide conventions:

* Logical axis names ``("in", "out")`` (and ``("in1", "in2", "out")``
  for the bilinear weight) are attached to every parameter, so a
  surrounding mesh can resolve sharding by axis name.
* Mixed precision is opt-in through
  :func:`~spectrax.core.policy.current_policy`: if a policy with a
  non-``None`` ``compute_dtype`` is active, both the input and the
  parameters are downcast to that dtype before the matmul.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType, Initializer
from ..core.module import Module
from ..core.policy import current_policy
from ..core.sharding import AxisNames, Sharding
from ..core.variable import DeferredParameter, Parameter
from ..functional import linear as F_linear
from ..init import kaiming_uniform, zeros
from ..rng.rngs import Rngs, resolve_rngs


class Linear(Module):
    """Standard dense (fully-connected) layer: ``y = x @ W + b``.

    Stores the weight under the canonical name ``weight`` with shape
    ``(in_features, out_features)`` and (optionally) a bias under
    ``bias`` with shape ``(out_features,)``. Logical axis names
    ``("in", "out")`` are attached to the weight and ``("out",)`` to
    the bias so a mesh can resolve sharding by name.

    Mixed precision: when a :class:`~spectrax.core.policy.Policy` is
    active and exposes a non-``None`` ``compute_dtype``, the input and
    both parameters are cast to that dtype before the matmul. The
    stored parameters themselves are unchanged.

    Deferred shape inference: passing ``in_features=None`` allocates a
    :class:`~spectrax.core.variable.DeferredParameter` whose true
    shape is filled in from ``x.shape[-1]`` on the first forward
    call. The deferred path is not safe under JAX transforms and
    triggers the standard ``_resolve_deferred`` guard.
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        in_features: int | None,
        out_features: int,
        *,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize the dense layer.

        Args:
            in_features: Trailing input feature count. ``None`` defers
                shape inference until the first :meth:`forward` call,
                where the actual size is read off ``x.shape[-1]``.
            out_features: Output feature count (size of the trailing
                axis of ``y``).
            use_bias: When ``True`` (default), allocate and add a
                ``(out_features,)`` bias initialized via ``b_init``.
            rngs: Source of PRNG keys for parameter initialization.
                Accepts an :class:`Rngs`, an ``int`` seed, or
                ``None``; resolved via :func:`resolve_rngs`.
            w_init: Weight initializer. Defaults to
                :func:`~spectrax.init.kaiming_uniform` with the
                ``"linear"`` gain (i.e. unit gain — a plain
                Glorot-style Kaiming uniform sized for a linear layer
                with no following non-linearity).
            b_init: Bias initializer. Defaults to
                :func:`~spectrax.init.zeros`.
            dtype: Storage dtype for the parameters. Defaults to
                ``float32`` when both ``dtype`` and ``param_dtype``
                are ``None``.
            param_dtype: Alias for ``dtype``; takes precedence when
                both are supplied. Provided for parity with frameworks
                that split parameter storage and computation dtype.
            sharding: Optional sharding (or axis-name tuple) for the
                weight; combined with the auto-attached axis names
                ``("in", "out")``.
            bias_sharding: Optional sharding for the bias (axis names
                ``("out",)``).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        w_init = w_init or kaiming_uniform("linear")
        weight_dtype = param_dtype or dtype or jnp.float32
        if in_features is None:
            self.weight = DeferredParameter(
                (None, out_features),
                w_init,
                resolved.parameters,
                weight_dtype,
                sharding=sharding,
                axis_names=("in", "out"),
            )
        else:
            self.weight = Parameter(
                w_init(resolved.parameters, (in_features, out_features), weight_dtype),
                sharding=sharding,
                axis_names=("in", "out"),
            )
        if use_bias:
            b_init = b_init or zeros
            self.bias = Parameter(
                b_init(resolved.parameters, (out_features,), weight_dtype),
                sharding=bias_sharding,
                axis_names=("out",),
            )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Compute ``y = x @ W + b`` (bias added only when configured).

        On the first call, if ``in_features`` was deferred, the weight
        is materialised from ``x.shape[-1]``. When a mixed-precision
        policy is active, ``x`` and the parameters are cast to
        ``policy.compute_dtype`` before the dot.

        Args:
            x: Input tensor; trailing axis must equal
                :attr:`in_features` (or determines it on the first
                call when deferred).
            **_: Ignored; accepted for container interoperability.

        Returns:
            ``x @ W + b`` (or ``x @ W`` when ``use_bias=False``) with
            shape ``x.shape[:-1] + (out_features,)``.
        """
        xa = jnp.asarray(x)
        if self.in_features is None:
            in_features = int(xa.shape[-1])
            self._resolve_deferred(self.weight, (in_features, self.out_features))
            self.in_features = in_features
        pol = current_policy()
        W = self.weight.value
        b = self.bias.value if self.use_bias else None
        if pol is not None and pol.compute_dtype is not None:
            xa = xa.astype(pol.compute_dtype)
            W = W.astype(pol.compute_dtype)
            if b is not None:
                b = b.astype(pol.compute_dtype)
        if self.use_bias:
            return F_linear(xa, W, b)
        return F_linear(xa, W)


class Bilinear(Module):
    """Bilinear interaction layer.

    Implements ``y[..., o] = sum_ij x1[..., i] * W[i, j, o] * x2[..., j]
    + b[o]``, computed via :func:`jax.numpy.einsum` with the equation
    ``"...i,ijo,...j->...o"``. Use it for learned feature-feature
    products — e.g. encoder/decoder mixing, score functions, or
    second-order feature crosses.

    The leading shapes of ``x1`` and ``x2`` broadcast under standard
    NumPy rules, so a typical use is ``Bilinear(d, d, d_out)(x, x)``
    with shared inputs to obtain a learned quadratic form. Logical
    axis names ``("in1", "in2", "out")`` are attached to the weight.
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize the bilinear layer.

        Args:
            in1_features: Trailing feature count of the first input
                (size of the ``i`` axis on the weight).
            in2_features: Trailing feature count of the second input
                (size of the ``j`` axis on the weight).
            out_features: Output feature count (size of the ``o``
                axis on the weight, and of the bias).
            use_bias: When ``True`` (default), allocate and add a
                ``(out_features,)`` zero-initialized bias.
            rngs: Source of PRNG keys for parameter initialization.
                Accepts an :class:`Rngs`, an ``int`` seed, or
                ``None``; resolved via :func:`resolve_rngs`.
            dtype: Storage dtype for the parameters. Defaults to
                ``float32`` when both ``dtype`` and ``param_dtype``
                are ``None``.
            param_dtype: Alias for ``dtype``; takes precedence when
                both are supplied.
            sharding: Optional sharding for the three-axis weight
                (axis names ``("in1", "in2", "out")``).
            bias_sharding: Optional sharding for the bias (axis names
                ``("out",)``).
        """
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        init = kaiming_uniform("linear")
        weight_dtype = param_dtype or dtype or jnp.float32
        self.weight = Parameter(
            init(
                resolved.parameters,
                (in1_features, in2_features, out_features),
                weight_dtype,
            ),
            sharding=sharding,
            axis_names=("in1", "in2", "out"),
        )
        if use_bias:
            self.bias = Parameter(
                jnp.zeros((out_features,), dtype=weight_dtype),
                sharding=bias_sharding,
                axis_names=("out",),
            )

    def forward(self, x1: ArrayLike, x2: ArrayLike, **_: object) -> Array:
        """Compute the bilinear form and add the bias when configured.

        Args:
            x1: First input with trailing axis equal to
                :attr:`in1_features`.
            x2: Second input with trailing axis equal to
                :attr:`in2_features`. The leading shapes of ``x1`` and
                ``x2`` broadcast against each other.
            **_: Ignored; accepted for container interoperability.

        Returns:
            ``y`` with shape ``broadcast(x1.shape[:-1], x2.shape[:-1])
            + (out_features,)``.
        """
        y = jnp.einsum("...i,ijo,...j->...o", x1, self.weight.value, x2)
        if self.use_bias:
            y = y + self.bias.value
        return y
