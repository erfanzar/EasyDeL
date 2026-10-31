# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pooling modules — max / average / adaptive-average over 1/2/3-D inputs.

All layers operate on channels-last inputs (``(N, *spatial, C)``) and
delegate to the corresponding functional kernel in
:mod:`spectrax.functional.pool`. The :class:`AdaptiveAvgPoolNd`
variants compute window/stride from the input and target output
shapes (PyTorch-style ragged windows) before delegating.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import ClassVar

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike
from ..core.module import Module
from ..functional.conv import PaddingSpec
from ..functional.pool import avg_pool as F_avg_pool
from ..functional.pool import max_pool as F_max_pool

__all__ = [
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
]


def _tup(x: int | Sequence[int], n: int) -> tuple[int, ...]:
    """Broadcast an ``int`` to a length-``n`` tuple, or validate a sequence.

    Args:
        x: Either a single ``int`` (broadcast across all spatial
            axes) or any sequence of ``n`` ints.
        n: Required length, equal to the spatial rank of the layer.

    Returns:
        A length-``n`` tuple of ints.

    Raises:
        ValueError: If ``x`` is a sequence of length ``!= n``.
    """
    if isinstance(x, int):
        return (x,) * n
    t = tuple(x)
    if len(t) != n:
        raise ValueError(f"Expected length-{n} sequence, got {t}")
    return t


class _PoolND(Module):
    """Shared N-D max/average pool implementation.

    Concrete subclasses set :attr:`_N` to the spatial rank and
    :attr:`_MODE` to either ``"max"`` or ``"avg"`` to pick the
    underlying functional kernel.
    """

    _N: ClassVar[int] = 0
    _MODE: ClassVar[str] = "max"

    def __init__(
        self,
        kernel_size: int | Sequence[int],
        *,
        stride: int | Sequence[int] | None = None,
        padding: PaddingSpec = "VALID",
        count_include_pad: bool = True,
    ) -> None:
        """Initialize the pool.

        Args:
            kernel_size: Per-axis window size; ``int`` broadcasts to
                length ``_N``.
            stride: Per-axis stride. ``None`` (default) means
                "same as kernel size" — non-overlapping windows.
            padding: ``"SAME"``, ``"VALID"``, ``"CIRCULAR"``,
                ``"REFLECT"``, or a sequence of per-axis
                ``(low, high)`` integer pairs.
            count_include_pad: Only meaningful for
                ``_MODE == "avg"``. When ``True`` (default), padded
                positions count toward the denominator; when
                ``False``, only the in-bounds positions do.

        Raises:
            ValueError: If ``count_include_pad`` is set to a
                non-default value on a max-pool layer.
        """
        super().__init__()
        if self._MODE == "max" and count_include_pad is not True:
            raise ValueError("count_include_pad is only meaningful for AvgPool layers.")
        self.kernel_size = _tup(kernel_size, self._N)
        self.stride = None if stride is None else _tup(stride, self._N)
        self.padding = padding if isinstance(padding, str) else tuple(tuple(p) for p in padding)
        self.count_include_pad = count_include_pad

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply max- or average-pool depending on :attr:`_MODE`.

        Args:
            x: Channels-last input ``(N, *spatial, C)``.
            **_: Ignored; accepted for container interoperability.

        Returns:
            Pooled tensor with the same batch and channel axes; the
            spatial axes are reduced according to ``kernel_size``,
            ``stride``, and ``padding``.
        """
        if self._MODE == "max":
            return F_max_pool(x, self.kernel_size, strides=self.stride, padding=self.padding)
        return F_avg_pool(
            x,
            self.kernel_size,
            strides=self.stride,
            padding=self.padding,
            count_include_pad=self.count_include_pad,
        )


class MaxPool1d(_PoolND):
    """Max-pool over ``(N, L, C)`` channels-last inputs.

    See :class:`_PoolND` for the constructor and forward contract.
    """

    _N: ClassVar[int] = 1
    _MODE: ClassVar[str] = "max"


class MaxPool2d(_PoolND):
    """Max-pool over ``(N, H, W, C)`` channels-last inputs.

    See :class:`_PoolND` for the constructor and forward contract.
    """

    _N: ClassVar[int] = 2
    _MODE: ClassVar[str] = "max"


class MaxPool3d(_PoolND):
    """Max-pool over ``(N, D, H, W, C)`` channels-last inputs.

    See :class:`_PoolND` for the constructor and forward contract.
    """

    _N: ClassVar[int] = 3
    _MODE: ClassVar[str] = "max"


class AvgPool1d(_PoolND):
    """Average-pool over ``(N, L, C)`` channels-last inputs.

    See :class:`_PoolND` for the constructor and forward contract;
    ``count_include_pad`` is honored.
    """

    _N: ClassVar[int] = 1
    _MODE: ClassVar[str] = "avg"


class AvgPool2d(_PoolND):
    """Average-pool over ``(N, H, W, C)`` channels-last inputs.

    See :class:`_PoolND` for the constructor and forward contract;
    ``count_include_pad`` is honored.
    """

    _N: ClassVar[int] = 2
    _MODE: ClassVar[str] = "avg"


class AvgPool3d(_PoolND):
    """Average-pool over ``(N, D, H, W, C)`` channels-last inputs.

    See :class:`_PoolND` for the constructor and forward contract;
    ``count_include_pad`` is honored.
    """

    _N: ClassVar[int] = 3
    _MODE: ClassVar[str] = "avg"


class _AdaptiveAvgPoolND(Module):
    """Shared N-D adaptive average-pool implementation.

    Implements the PyTorch ``AdaptiveAvgPoolNd`` semantics: the
    output spatial shape is fixed and the per-output-position
    windows are derived from the input/output shape ratio. Window
    boundaries are ``(i*S/O, (i+1)*S/O)`` rounded so that each
    output position covers a contiguous, possibly ragged, slice of
    the input — adjacent windows may overlap when ``S/O`` is not
    integer.

    Concrete subclasses set :attr:`_N` to the spatial rank.
    """

    _N: ClassVar[int] = 0

    def __init__(self, output_size: int | Sequence[int]) -> None:
        """Initialize with the desired output spatial size.

        Args:
            output_size: Per-axis target output size; ``int``
                broadcasts to all ``_N`` axes.
        """
        super().__init__()
        self.output_size = _tup(output_size, self._N)

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Adaptive average-pool ``x`` to :attr:`output_size`.

        Per-output-position window boundaries are computed in pure
        Python (``starts = (i*S)//O``, ``ends = ((i+1)*S + O - 1)//O``)
        and the corresponding slices of ``x`` are mean-reduced. The
        result is reshaped to ``(N, *output_size, C)``.

        Args:
            x: Channels-last input ``(N, *spatial, C)`` whose number
                of spatial axes equals :attr:`_N`.
            **_: Ignored; accepted for container interoperability.

        Returns:
            ``(N, *output_size, C)`` channels-last tensor.

        Raises:
            ValueError: If the number of spatial axes on ``x`` is
                not :attr:`_N`, or if any element of
                :attr:`output_size` exceeds the corresponding input
                spatial dimension.
        """
        xa = jnp.asarray(x)
        spatial = xa.shape[1:-1]
        if len(spatial) != self._N:
            raise ValueError(f"AdaptiveAvgPool{self._N}d expected {self._N} spatial axes, got {len(spatial)}")
        if any(o > s for s, o in zip(spatial, self.output_size, strict=True)):
            raise ValueError(f"output_size {self.output_size} exceeds spatial dims {spatial}")

        starts = [tuple((i * s) // o for i in range(o)) for s, o in zip(spatial, self.output_size, strict=True)]
        ends = [
            tuple(((i + 1) * s + o - 1) // o for i in range(o)) for s, o in zip(spatial, self.output_size, strict=True)
        ]
        pooled = []
        for out_idx in itertools.product(*(range(o) for o in self.output_size)):
            slices = [slice(None)]
            for dim, pos in enumerate(out_idx):
                slices.append(slice(starts[dim][pos], ends[dim][pos]))
            slices.append(slice(None))
            pooled.append(xa[tuple(slices)].mean(axis=tuple(range(1, self._N + 1))))
        flat = jnp.stack(pooled, axis=1)
        return flat.reshape(xa.shape[0], *self.output_size, xa.shape[-1])


class AdaptiveAvgPool1d(_AdaptiveAvgPoolND):
    """Adaptive average pool over ``(N, L, C)`` inputs.

    See :class:`_AdaptiveAvgPoolND` for the constructor and forward
    contract.
    """

    _N: ClassVar[int] = 1


class AdaptiveAvgPool2d(_AdaptiveAvgPoolND):
    """Adaptive average pool over ``(N, H, W, C)`` inputs.

    See :class:`_AdaptiveAvgPoolND` for the constructor and forward
    contract.
    """

    _N: ClassVar[int] = 2


class AdaptiveAvgPool3d(_AdaptiveAvgPoolND):
    """Adaptive average pool over ``(N, D, H, W, C)`` inputs.

    See :class:`_AdaptiveAvgPoolND` for the constructor and forward
    contract.
    """

    _N: ClassVar[int] = 3
