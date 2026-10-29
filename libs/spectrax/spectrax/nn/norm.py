# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Normalization layers.

Exports the standard suite — :class:`LayerNorm`, :class:`RMSNorm`,
:class:`BatchNorm1d`, :class:`BatchNorm2d`, :class:`GroupNorm`,
:class:`InstanceNorm`. All layers attach the logical axis name
``("features",)`` (or ``("channels",)`` for the convolutional /
batch-style variants) to every learned scale or bias so a surrounding
mesh can resolve sharding by name.

Train / eval semantics differ for :class:`BatchNorm1d` /
:class:`BatchNorm2d` only: those two maintain running statistics in
the ``"batch_stats"`` collection and update them during training. All
other normalizers are stateless across calls.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType
from ..core.module import Module
from ..core.sharding import AxisNames, Sharding
from ..core.variable import Buffer, Parameter
from ..functional import layer_norm as F_layer_norm
from ..functional import rms_norm as F_rms_norm

_FEATURE_AXIS: AxisNames = ("features",)
_CHANNEL_AXIS: AxisNames = ("channels",)


class LayerNorm(Module):
    """Per-sample layer normalization (Ba, Kiros & Hinton, 2016).

    Normalizes along the trailing ``features`` axis using the
    per-sample mean and variance, then optionally applies a learned
    per-feature affine transform. Mean/variance are computed in the
    input's dtype; the variance floor :attr:`eps` is added before
    the inverse square root for numerical stability.
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        features: int,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        use_bias: bool = True,
        dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize.

        Args:
            features: Size of the trailing (normalization) axis.
            eps: Variance floor added before the inverse square root,
                for numerical stability with near-zero variance
                inputs.
            elementwise_affine: When ``True`` (default), allocate a
                learned per-feature scale (and, if ``use_bias`` is
                also set, a bias).
            use_bias: When ``True`` and ``elementwise_affine`` is
                set, also allocate a learned bias.
            dtype: Storage dtype for the learnable parameters;
                defaults to ``float32``.
            sharding: Optional sharding for the scale (axis names
                ``("features",)``).
            bias_sharding: Optional sharding for the bias (axis names
                ``("features",)``).
        """
        super().__init__()
        self.features = features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = use_bias
        if elementwise_affine:
            self.weight = Parameter(
                jnp.ones((features,), dtype=dtype or jnp.float32),
                sharding=sharding,
                axis_names=_FEATURE_AXIS,
            )
            if use_bias:
                self.bias = Parameter(
                    jnp.zeros((features,), dtype=dtype or jnp.float32),
                    sharding=bias_sharding,
                    axis_names=_FEATURE_AXIS,
                )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Normalize ``x`` along the last axis and apply any affine transform.

        Args:
            x: Input tensor whose trailing axis equals
                :attr:`features`.
            **_: Ignored; accepted for container interoperability.

        Returns:
            Normalized tensor with the same shape and dtype as ``x``.
        """
        scale = self.weight.value if self.elementwise_affine else None
        bias = self.bias.value if (self.elementwise_affine and self.use_bias) else None
        return F_layer_norm(x, scale=scale, bias=bias, axis=-1, eps=self.eps)


class RMSNorm(Module):
    """Root-mean-square normalization (Zhang & Sennrich, 2019).

    Variant of :class:`LayerNorm` that drops the mean-subtraction
    step: ``y = x / sqrt(mean(x^2) + eps)`` followed by an optional
    per-feature scale. Cheaper than :class:`LayerNorm` and the
    canonical normalization for modern transformer stacks (LLaMA,
    GPT-J, …).
    """

    weight: Parameter

    def __init__(
        self,
        features: int,
        *,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize.

        Args:
            features: Size of the trailing (normalization) axis.
            eps: Floor added to the mean-of-squares before the
                inverse square root. Defaults to ``1e-6`` (smaller
                than :class:`LayerNorm`'s default because the absence
                of mean-subtraction means the variance term is
                bounded below by zero, not by the magnitude of the
                mean).
            elementwise_affine: When ``True`` (default), allocate a
                learned per-feature scale. There is no bias by design
                — RMSNorm is centring-free.
            dtype: Storage dtype for the scale; defaults to
                ``float32``.
            sharding: Optional sharding for the scale (axis names
                ``("features",)``).
        """
        super().__init__()
        self.features = features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(
                jnp.ones((features,), dtype=dtype or jnp.float32),
                sharding=sharding,
                axis_names=_FEATURE_AXIS,
            )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply RMSNorm along the last axis.

        Args:
            x: Input tensor whose trailing axis equals
                :attr:`features`.
            **_: Ignored; accepted for container interoperability.

        Returns:
            Normalized tensor with the same shape and dtype as ``x``.
        """
        scale = self.weight.value if self.elementwise_affine else None
        return F_rms_norm(x, scale=scale, axis=-1, eps=self.eps)


class _BatchNormND(Module):
    """Shared N-D batch-normalization implementation (channels-last).

    Reduction axes are everything except the trailing channel axis,
    so the same code services 1-D / 2-D / 3-D inputs (the
    ``_SPATIAL`` class attribute is documentation only — actual
    rank inference is dynamic).

    Running statistics live in :class:`~spectrax.Buffer` cells with
    kind ``"batch_stats"``: ``running_mean`` (zero-init) and
    ``running_var`` (one-init), each of shape ``(num_features,)``.
    Training-mode calls mutate them in place using an exponential
    moving average; the surrounding transform must therefore declare
    ``mutable="batch_stats"``.
    """

    _SPATIAL: ClassVar[int] = 0

    weight: Parameter
    bias: Parameter
    running_mean: Buffer
    running_var: Buffer

    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
        stats_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize.

        Args:
            num_features: Trailing channel count.
            eps: Variance floor added before the inverse square root
                for numerical stability.
            momentum: EMA factor used to fold each batch's
                statistics into the running estimates:
                ``running = (1 - momentum) * running + momentum * batch``.
                Default ``0.1`` matches PyTorch.
            affine: When ``True`` (default), allocate learned
                per-channel ``weight`` (one-init) and ``bias``
                (zero-init).
            dtype: Storage dtype for both the parameters and the
                running statistics; defaults to ``float32``.
            sharding: Optional sharding for the scale (axis names
                ``("channels",)``).
            bias_sharding: Optional sharding for the bias (axis names
                ``("channels",)``).
            stats_sharding: Optional sharding for the running mean
                and variance (axis names ``("channels",)``).
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = Parameter(
                jnp.ones((num_features,), dtype=dtype or jnp.float32),
                sharding=sharding,
                axis_names=_CHANNEL_AXIS,
            )
            self.bias = Parameter(
                jnp.zeros((num_features,), dtype=dtype or jnp.float32),
                sharding=bias_sharding,
                axis_names=_CHANNEL_AXIS,
            )
        self.running_mean = Buffer(
            jnp.zeros((num_features,), dtype=dtype or jnp.float32),
            kind="batch_stats",
            sharding=stats_sharding,
            axis_names=_CHANNEL_AXIS,
        )
        self.running_var = Buffer(
            jnp.ones((num_features,), dtype=dtype or jnp.float32),
            kind="batch_stats",
            sharding=stats_sharding,
            axis_names=_CHANNEL_AXIS,
        )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Normalize ``x`` and (in training mode) update running stats.

        Training mode reduces over every axis except the trailing
        channel axis to produce the batch mean and variance, then
        blends them into :attr:`running_mean` / :attr:`running_var`
        via the exponential moving average and uses the *batch*
        statistics for the normalization itself. Evaluation mode
        skips the update and normalizes with the stored running
        statistics verbatim.

        Args:
            x: Channels-last input. Trailing axis must equal
                :attr:`num_features`. Any preceding axes are reduced
                over for statistics.
            **_: Ignored; accepted for container interoperability.

        Returns:
            Normalized tensor with the same shape and dtype as ``x``.
        """
        xa = jnp.asarray(x)
        reduce_axes = tuple(range(xa.ndim - 1))
        if self.training:
            mean = jnp.mean(xa, axis=reduce_axes)
            var = jnp.var(xa, axis=reduce_axes)
            self.running_mean.value = (1 - self.momentum) * self.running_mean.value + self.momentum * mean
            self.running_var.value = (1 - self.momentum) * self.running_var.value + self.momentum * var
        else:
            mean = self.running_mean.value
            var = self.running_var.value
        inv = 1.0 / jnp.sqrt(var + self.eps)
        y = (xa - mean) * inv
        if self.affine:
            y = y * self.weight.value + self.bias.value
        return y


class BatchNorm1d(_BatchNormND):
    """BatchNorm for ``(N, L, C)`` channels-last inputs (1-D sequences).

    See :class:`_BatchNormND` for the constructor and forward
    contract.
    """

    _SPATIAL: ClassVar[int] = 1


class BatchNorm2d(_BatchNormND):
    """BatchNorm for ``(N, H, W, C)`` channels-last inputs (2-D images).

    See :class:`_BatchNormND` for the constructor and forward
    contract.
    """

    _SPATIAL: ClassVar[int] = 2


class GroupNorm(Module):
    """Group normalization (Wu & He, 2018).

    Splits the trailing channel axis into ``num_groups`` equally
    sized groups and computes mean/variance per group over both the
    spatial axes and the within-group channel axis. Reduces to
    :class:`LayerNorm`-on-channels when ``num_groups=num_channels``
    and to :class:`InstanceNorm` when ``num_groups=num_channels``;
    typical choice in vision architectures is ``num_groups=32``.

    Inputs are channels-last ``(..., spatial..., C)``.
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        *,
        eps: float = 1e-5,
        affine: bool = True,
        dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize.

        Args:
            num_groups: Number of groups; must exactly divide
                ``num_channels``.
            num_channels: Size of the trailing channel axis. Used
                both to allocate the learnable parameters and to
                validate the input shape on every forward call.
            eps: Variance floor for numerical stability.
            affine: When ``True`` (default), allocate learned
                per-channel ``weight`` (one-init) and ``bias``
                (zero-init).
            dtype: Storage dtype for the parameters; defaults to
                ``float32``.
            sharding: Optional sharding for the scale (axis names
                ``("channels",)``).
            bias_sharding: Optional sharding for the bias (axis names
                ``("channels",)``).

        Raises:
            ValueError: If ``num_groups`` does not divide
                ``num_channels``.
        """
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = Parameter(
                jnp.ones((num_channels,), dtype=dtype or jnp.float32),
                sharding=sharding,
                axis_names=_CHANNEL_AXIS,
            )
            self.bias = Parameter(
                jnp.zeros((num_channels,), dtype=dtype or jnp.float32),
                sharding=bias_sharding,
                axis_names=_CHANNEL_AXIS,
            )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Normalize within each channel group and apply the affine transform.

        The trailing channel axis is reshaped to
        ``(num_groups, num_channels // num_groups)`` so reductions
        over the within-group channel axis and the spatial axes
        produce per-group statistics. The result is reshaped back to
        the input layout before the (optional) per-channel scale and
        bias are applied.

        Args:
            x: Channels-last input. Trailing axis must equal
                :attr:`num_channels`.
            **_: Ignored; accepted for container interoperability.

        Returns:
            Normalized tensor with the same shape and dtype as ``x``.

        Raises:
            ValueError: If ``x.shape[-1]`` does not equal
                :attr:`num_channels`.
        """
        xa = jnp.asarray(x)
        c = xa.shape[-1]
        if c != self.num_channels:
            raise ValueError(f"GroupNorm expected {self.num_channels} channels, got {c}")
        g = self.num_groups
        shape = (*xa.shape[:-1], g, c // g)
        xg = xa.reshape(shape)
        reduce_axes = (*range(1, xg.ndim - 2), xg.ndim - 1)
        mean = jnp.mean(xg, axis=reduce_axes, keepdims=True)
        var = jnp.var(xg, axis=reduce_axes, keepdims=True)
        xn = (xg - mean) / jnp.sqrt(var + self.eps)
        y = xn.reshape(xa.shape)
        if self.affine:
            y = y * self.weight.value + self.bias.value
        return y


class InstanceNorm(Module):
    """Instance normalization (Ulyanov, Vedaldi & Lempitsky, 2016).

    Computes per-sample, per-channel mean and variance over the
    spatial axes only — i.e. each ``(sample, channel)`` slice is
    normalized independently. Inputs are channels-last
    ``(N, *spatial, C)``; reductions are over ``range(1, ndim - 1)``,
    so the leading batch axis is *not* reduced. Optional learned
    per-channel affine transform.
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        num_channels: int,
        *,
        eps: float = 1e-5,
        affine: bool = True,
        dtype: DType | None = None,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize.

        Args:
            num_channels: Trailing channel count.
            eps: Variance floor for numerical stability.
            affine: When ``True`` (default), allocate learned
                per-channel ``weight`` (one-init) and ``bias``
                (zero-init).
            dtype: Storage dtype for the parameters; defaults to
                ``float32``.
            sharding: Optional sharding for the scale (axis names
                ``("channels",)``).
            bias_sharding: Optional sharding for the bias (axis names
                ``("channels",)``).
        """
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = Parameter(
                jnp.ones((num_channels,), dtype=dtype or jnp.float32),
                sharding=sharding,
                axis_names=_CHANNEL_AXIS,
            )
            self.bias = Parameter(
                jnp.zeros((num_channels,), dtype=dtype or jnp.float32),
                sharding=bias_sharding,
                axis_names=_CHANNEL_AXIS,
            )

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Normalize per sample, per channel, over the spatial axes.

        Args:
            x: Channels-last input ``(N, *spatial, C)``; trailing
                axis must equal :attr:`num_channels`.
            **_: Ignored; accepted for container interoperability.

        Returns:
            Normalized tensor with the same shape and dtype as ``x``.

        Raises:
            ValueError: If ``x.shape[-1]`` does not equal
                :attr:`num_channels`.
        """
        xa = jnp.asarray(x)
        if xa.shape[-1] != self.num_channels:
            raise ValueError(f"InstanceNorm expected {self.num_channels} channels, got {xa.shape[-1]}")
        reduce_axes = tuple(range(1, xa.ndim - 1))
        mean = jnp.mean(xa, axis=reduce_axes, keepdims=True)
        var = jnp.var(xa, axis=reduce_axes, keepdims=True)
        y = (xa - mean) / jnp.sqrt(var + self.eps)
        if self.affine:
            y = y * self.weight.value + self.bias.value
        return y
