# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Convolution layers — direct and transposed, ranks 1/2/3.

The dispatch matrix exposes one concrete class per spatial rank
(:class:`Conv1d`, :class:`Conv2d`, :class:`Conv3d`,
:class:`ConvTranspose1d`, :class:`ConvTranspose2d`,
:class:`ConvTranspose3d`) plus a rank-polymorphic :class:`Conv` that
infers the rank from its ``kernel_size`` argument (mirroring the
``flax.nnx.Conv`` API).

All layers share a common implementation in two private bases —
:class:`_ConvND` for direct convolutions and :class:`_ConvTransposeND`
for the transpose. Both store the kernel and bias as
:class:`~spectrax.Parameter` s and delegate the actual computation to
:func:`spectrax.functional.conv` / :func:`spectrax.functional.conv_transpose`.

Conventions
    * **Channels-last** layout: inputs are ``(N, *spatial, C_in)`` and
      outputs ``(N, *spatial_out, C_out)``.
    * **Kernel layout**: ``(*kernel_size, C_in / groups, C_out)`` for
      direct convolutions; ``(*kernel_size, C_in, C_out)`` for the
transpose.
    * **Logical axis names** ``(*"k" * rank, "in", "out")`` are
      attached to every kernel for sharding resolution; biases use
``("out",)``.
    * **Deferred shape inference**: passing ``in_channels=None``
      defers kernel allocation to the first :meth:`forward` call,
where ``C_in`` is read off ``x.shape[-1]``. Not safe inside JAX
transforms.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType
from ..core.module import Module
from ..core.sharding import AxisNames, Sharding
from ..core.variable import DeferredParameter, Parameter
from ..functional.conv import PaddingSpec
from ..functional.conv import conv as F_conv
from ..functional.conv import conv_transpose as F_conv_transpose
from ..init import kaiming_uniform, zeros
from ..rng.rngs import Rngs, resolve_rngs


def _tup(x: int | Sequence[int], n: int) -> tuple[int, ...]:
    """Broadcast an ``int`` to a length-``n`` tuple, or validate a sequence.

    Helper used to normalise the per-axis arguments (``kernel_size``,
    ``stride``, ``dilation``) accepted by every conv layer.

    Args:
        x: Either a single ``int`` (broadcast) or any sequence of
            ``n`` ints.
        n: Required length of the resulting tuple — equal to the
            spatial rank of the layer.

    Returns:
        A length-``n`` tuple of ints.

    Raises:
        ValueError: If ``x`` is a sequence whose length is not ``n``.
    """
    if isinstance(x, int):
        return (x,) * n
    t = tuple(x)
    if len(t) != n:
        raise ValueError(f"Expected length-{n} tuple, got {t}")
    return t


class _ConvND(Module):
    """Shared N-D direct convolution implementation.

    Concrete subclasses (:class:`Conv1d`, :class:`Conv2d`,
    :class:`Conv3d`) only need to set the class-level :attr:`_N` to
    the spatial rank — everything else (parameter allocation,
    deferred shape resolution, forward dispatch) lives here.

    The kernel has shape ``(*kernel_size, in_channels // groups,
    out_channels)`` with logical axis names
    ``(*"k" * _N, "in", "out")``; the bias (when used) has shape
    ``(out_channels,)`` with axis names ``("out",)``.
    """

    _N: ClassVar[int] = 0

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        in_channels: int | None,
        out_channels: int,
        kernel_size: int | Sequence[int],
        *,
        stride: int | Sequence[int] = 1,
        padding: PaddingSpec = "VALID",
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize the N-D convolution.

        Args:
            in_channels: Input channel count. Pass ``None`` to defer
                allocation until the first :meth:`forward` call (the
                value will be read off ``x.shape[-1]``); deferred mode
                is not safe inside JAX transforms.
            out_channels: Output channel count.
            kernel_size: Per-axis kernel size. A bare ``int`` is
                broadcast to all spatial axes; a length-``_N`` sequence
                is taken as-is.
            stride: Per-axis stride; broadcast / validated as
                ``kernel_size`` is.
            padding: Either a string accepted by the underlying
                primitive (``"SAME"``, ``"VALID"``, ``"CIRCULAR"``,
                ``"REFLECT"`` …) or a sequence of per-axis
                ``(low, high)`` integer pairs.
            dilation: Per-axis kernel dilation; broadcast / validated
                as ``kernel_size``.
            groups: Group count for grouped / depthwise convolutions.
                Must divide both ``in_channels`` and ``out_channels``.
            use_bias: When ``True`` (default), allocate and add an
                ``(out_channels,)`` zero-initialized bias.
            rngs: Source of PRNG keys for parameter initialization.
                Accepts an :class:`Rngs`, an ``int`` seed, or
                ``None``.
            dtype: Storage dtype for the parameters; defaults to
                ``float32`` when both ``dtype`` and ``param_dtype``
                are ``None``.
            param_dtype: Alias for ``dtype``; takes precedence when
                both are supplied.
            sharding: Optional sharding for the kernel; the auto-attached
                axis names are ``(*"k" * _N, "in", "out")``.
            bias_sharding: Optional sharding for the bias (axis names
                ``("out",)``).
        """
        super().__init__()
        n = self._N
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _tup(kernel_size, n)
        self.stride = _tup(stride, n)
        self.padding = padding if isinstance(padding, str) else tuple(tuple(p) for p in padding)
        self.dilation = _tup(dilation, n)
        self.groups = groups
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        init = kaiming_uniform("linear")
        weight_dtype = param_dtype or dtype or jnp.float32
        if in_channels is None:
            self.weight = DeferredParameter(
                (*self.kernel_size, None, out_channels),
                init,
                resolved.parameters,
                weight_dtype,
                sharding=sharding,
                axis_names=(*["k"] * n, "in", "out"),
            )
        else:
            kshape = (*self.kernel_size, in_channels // groups, out_channels)
            self.weight = Parameter(
                init(resolved.parameters, kshape, weight_dtype),
                sharding=sharding,
                axis_names=(*["k"] * n, "in", "out"),
            )
        if use_bias:
            self.bias = Parameter(
                zeros(resolved.parameters, (out_channels,), weight_dtype),
                sharding=bias_sharding,
                axis_names=("out",),
            )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply :func:`spectrax.functional.conv` with the stored parameters.

        Resolves any deferred kernel allocation on the first call,
        then dispatches to the functional convolution with the
        configured ``stride``, ``padding``, ``dilation``, and
        ``groups``.

        Args:
            x: Channels-last input of shape
                ``(N, *spatial, C_in)`` whose trailing axis equals
                :attr:`in_channels` (or determines it on the first
                call when deferred).
            **_: Ignored; accepted for container interoperability.

        Returns:
            ``(N, *spatial_out, C_out)`` channels-last output.
        """
        xa = jnp.asarray(x)
        if self.in_channels is None:
            in_channels = int(xa.shape[-1])
            self._resolve_deferred(self.weight, (*self.kernel_size, in_channels // self.groups, self.out_channels))
            self.in_channels = in_channels
        return F_conv(
            xa,
            self.weight.value,
            self.bias.value if self.use_bias else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv1d(_ConvND):
    """1-D convolution over ``(N, L, C)`` channels-last inputs.

    See :class:`_ConvND` for the constructor and forward contract.
    """

    _N: ClassVar[int] = 1


class Conv2d(_ConvND):
    """2-D convolution over ``(N, H, W, C)`` channels-last inputs.

    See :class:`_ConvND` for the constructor and forward contract.
    """

    _N: ClassVar[int] = 2


class Conv3d(_ConvND):
    """3-D convolution over ``(N, D, H, W, C)`` channels-last inputs.

    See :class:`_ConvND` for the constructor and forward contract.
    """

    _N: ClassVar[int] = 3


class Conv(_ConvND):
    """N-D convolution whose spatial rank is inferred from ``kernel_size``.

    Mirrors the ``flax.nnx.Conv`` API: a single ``int`` kernel size
    selects a 1-D convolution; a length-``n`` tuple selects an ``n``-D
    convolution. The rank is stamped onto the instance as ``_N`` (via
    :func:`object.__setattr__` to bypass module field validation),
    after which the regular :class:`_ConvND` machinery is used.
    """

    def __init__(
        self,
        in_channels: int | None,
        out_channels: int,
        kernel_size: int | Sequence[int],
        *,
        stride: int | Sequence[int] = 1,
        padding: PaddingSpec = "VALID",
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Construct an N-D conv layer; ``N`` is inferred from ``kernel_size``.

        An ``int`` ``kernel_size`` selects a 1-D convolution and stamps
        ``self._N = 1``; a length-``n`` tuple selects an ``n``-D
        convolution. All other arguments are forwarded unchanged to
        :class:`_ConvND`: see its constructor for ``in_channels``,
        ``out_channels``, ``stride``, ``padding``, ``dilation``,
        ``groups``, ``use_bias``, ``rngs``, ``dtype`` /
        ``param_dtype``, and the sharding arguments.

        Args:
            in_channels: In channels value consumed by this operation.
            out_channels: Out channels value consumed by this operation.
            kernel_size: Kernel size value consumed by this operation.
            stride: Stride value consumed by this operation.
            padding: Padding value consumed by this operation.
            dilation: Dilation value consumed by this operation.
            groups: Groups value consumed by this operation.
            use_bias: Use bias value consumed by this operation.
            rngs: Random-number generator collection used to initialize or run the module.
            dtype: Array dtype requested for the produced value.
            param_dtype: Param dtype value consumed by this operation.
            sharding: JAX sharding object describing how an array is placed.
            bias_sharding: Bias sharding value consumed by this operation.
        """
        if isinstance(kernel_size, int):
            object.__setattr__(self, "_N", 1)
        else:
            object.__setattr__(self, "_N", len(tuple(kernel_size)))
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding=sharding,
            bias_sharding=bias_sharding,
        )


class _ConvTransposeND(Module):
    """Shared N-D transposed (fractional-stride) convolution.

    Concrete subclasses set :attr:`_N` to the spatial rank. Unlike the
    direct convolution there is no ``groups`` parameter, and the
    kernel layout is ``(*kernel_size, in_channels, out_channels)`` —
    no groups divisor on the input axis.

    Delegates to :func:`spectrax.functional.conv_transpose` for the
    actual computation; supports the same deferred-shape behaviour as
    :class:`_ConvND`.
    """

    _N: ClassVar[int] = 0

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        in_channels: int | None,
        out_channels: int,
        kernel_size: int | Sequence[int],
        *,
        stride: int | Sequence[int] = 1,
        padding: PaddingSpec = "VALID",
        dilation: int | Sequence[int] = 1,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize the N-D transposed convolution.

        Args:
            in_channels: Input channel count, or ``None`` to defer to
                the first :meth:`forward` call.
            out_channels: Output channel count.
            kernel_size: Per-axis kernel size; ``int`` broadcasts.
            stride: Per-axis stride. Note that for the transpose,
                ``stride > 1`` upsamples the spatial dimensions.
            padding: ``"SAME"`` / ``"VALID"`` / explicit per-axis
                ``(low, high)`` pairs.
            dilation: Per-axis kernel dilation.
            use_bias: When ``True`` (default), allocate and add an
                ``(out_channels,)`` zero-initialized bias.
            rngs: PRNG source for parameter initialization.
            dtype: Storage dtype; defaults to ``float32``.
            param_dtype: Alias for ``dtype``.
            sharding: Optional sharding for the kernel (axis names
                ``(*"k" * _N, "in", "out")``).
            bias_sharding: Optional sharding for the bias (axis names
                ``("out",)``).
        """
        super().__init__()
        n = self._N
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _tup(kernel_size, n)
        self.stride = _tup(stride, n)
        self.padding = padding if isinstance(padding, str) else tuple(tuple(p) for p in padding)
        self.dilation = _tup(dilation, n)
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        init = kaiming_uniform("linear")
        weight_dtype = param_dtype or dtype or jnp.float32
        if in_channels is None:
            self.weight = DeferredParameter(
                (*self.kernel_size, None, out_channels),
                init,
                resolved.parameters,
                weight_dtype,
                sharding=sharding,
                axis_names=(*["k"] * n, "in", "out"),
            )
        else:
            kshape = (*self.kernel_size, in_channels, out_channels)
            self.weight = Parameter(
                init(resolved.parameters, kshape, weight_dtype),
                sharding=sharding,
                axis_names=(*["k"] * n, "in", "out"),
            )
        if use_bias:
            self.bias = Parameter(
                zeros(resolved.parameters, (out_channels,), weight_dtype),
                sharding=bias_sharding,
                axis_names=("out",),
            )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply :func:`spectrax.functional.conv_transpose` with stored params.

        Resolves any deferred kernel allocation on the first call,
        then dispatches to the functional transposed convolution with
        the configured ``stride``, ``padding``, and ``dilation``.

        Args:
            x: Channels-last input
                ``(N, *spatial, C_in)``; trailing axis equals
                :attr:`in_channels`.
            **_: Ignored; accepted for container interoperability.

        Returns:
            Channels-last output ``(N, *spatial_out, C_out)`` whose
            spatial size is determined by the stride / padding /
            dilation / kernel size combination.
        """
        xa = jnp.asarray(x)
        if self.in_channels is None:
            in_channels = int(xa.shape[-1])
            self._resolve_deferred(self.weight, (*self.kernel_size, in_channels, self.out_channels))
            self.in_channels = in_channels
        return F_conv_transpose(
            xa,
            self.weight.value,
            self.bias.value if self.use_bias else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


class ConvTranspose1d(_ConvTransposeND):
    """Transposed 1-D convolution over ``(N, L, C)`` channels-last inputs.

    See :class:`_ConvTransposeND` for the constructor and forward
    contract.
    """

    _N: ClassVar[int] = 1


class ConvTranspose2d(_ConvTransposeND):
    """Transposed 2-D convolution over ``(N, H, W, C)`` inputs.

    See :class:`_ConvTransposeND` for the constructor and forward
    contract.
    """

    _N: ClassVar[int] = 2


class ConvTranspose3d(_ConvTransposeND):
    """Transposed 3-D convolution over ``(N, D, H, W, C)`` inputs.

    See :class:`_ConvTransposeND` for the constructor and forward
    contract.
    """

    _N: ClassVar[int] = 3
