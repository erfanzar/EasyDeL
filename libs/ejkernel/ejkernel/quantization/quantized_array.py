# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Container type for quantized tensors and metadata.

This module defines :class:`QuantizedArray`, a frozen dataclass that bundles
packed quantized weight data with all metadata (scales, zeros, mode, group_size,
bits, axis) needed for dequantization and quantized matrix multiplication.

It also provides two convenience factory functions:

- :func:`quantize_array` — quantize a raw weight tensor into a ``QuantizedArray``.
- :func:`prepack_quantized_array` — quantize and prepack weights (with axis
  convention compatible with fused kernel wrappers) into a ``QuantizedArray``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal

import jax

from ._quants.quantizations import dequantize, prepack_quantized_weights, quantize
from ._quants.quantizations import quantized_matmul as dense_quantized_matmul
from ._utils.qparams import (
    QuantizationAxis,
    QuantizationMode,
    normalize_axis,
    resolve_prepack_axis,
    resolve_qparams,
)
from .runtime import QuantRuntimeConfig


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class QuantizedArray:
    """Packed quantized tensor with all metadata required for runtime use.

    A frozen, JAX-pytree-registered container that stores bit-packed quantized
    weights together with the per-group scale/zero-point metadata and all static
    parameters needed to dequantize or run a fused quantized matmul.

    Attributes:
        data: Bit-packed uint32 array of quantized codes produced by
            :func:`quantize` or :func:`prepack_quantized_weights`.  Shape
            depends on ``axis`` and ``bits``:

            - ``axis='row'``: shape ``(K, ceil(N * bits / 32))``.
            - ``axis='col'``: shape ``(N, ceil(K * bits / 32))``.

        scales: Per-group scale tensor.  Shape is ``(K, N // group_size)`` for
            ``axis='row'`` or ``(N, K // group_size)`` for ``axis='col'``.
            Dtype is float for ``affine``/``nf4`` modes, ``uint8`` for
            ``mxfp*``/``nvfp*`` modes (shared exponents or E4M3 codes).

        zeros: Per-group zero-point tensor with the same shape as ``scales``.
            Required for ``mode='affine'`` (dequantization formula:
            ``(q - zero) * scale``).  Must be ``None`` for all other modes.

        mode: Quantization mode string.  One of ``"affine"``, ``"nf4"``,
            ``"mxfp4"``, ``"mxfp8"``, ``"nvfp4"``, ``"nvfp8"``.

        group_size: Number of elements per quantization group.  Defaults and
            valid values depend on ``mode`` (see :func:`resolve_qparams`).

        bits: Effective bit-width of stored codes (1 through 8 for affine;
            fixed for other modes).

        axis: Quantization axis.  ``"row"`` groups over output channels;
            ``"col"`` groups over input channels.

        runtime_config: Optional :class:`QuantRuntimeConfig` that controls
            implementation fast paths (compute dtype, unpack policy, etc.).
            ``None`` means defaults are resolved at runtime.
    """

    data: jax.Array
    scales: jax.Array
    zeros: jax.Array | None
    mode: QuantizationMode
    group_size: int
    bits: int
    axis: QuantizationAxis
    runtime_config: QuantRuntimeConfig | None = None

    def tree_flatten(self):
        """Flatten for JAX pytree traversal.

        Returns:
            Tuple of (children, aux) where:
            - children: ``(data, scales, zeros)`` — the traced array leaves.
            - aux: ``(mode, group_size, bits, axis, runtime_config)`` — static
              metadata preserved across JAX transformations.
        """
        children = (self.data, self.scales, self.zeros)
        aux = (self.mode, int(self.group_size), int(self.bits), self.axis, self.runtime_config)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Reconstruct from flattened pytree representation.

        Args:
            aux: Static metadata tuple ``(mode, group_size, bits, axis,
                runtime_config)`` as returned by :meth:`tree_flatten`.
            children: Array leaves ``(data, scales, zeros)`` as returned by
                :meth:`tree_flatten`.

        Returns:
            Reconstructed :class:`QuantizedArray` instance.
        """
        mode, group_size, bits, axis, runtime_config = aux
        data, scales, zeros = children
        return cls(
            data=data,
            scales=scales,
            zeros=zeros,
            mode=mode,
            group_size=int(group_size),
            bits=int(bits),
            axis=axis,
            runtime_config=runtime_config,
        )

    @staticmethod
    def _shape_or_none(x: jax.Array | None) -> tuple[int, ...] | None:
        """Return the shape tuple of *x*, or ``None`` if *x* is ``None``."""
        if x is None:
            return None
        return tuple(x.shape)

    @staticmethod
    def _numel(x: jax.Array | None) -> int:
        """Return the total element count of *x*, or 0 if *x* is ``None``."""
        if x is None:
            return 0
        if x.ndim == 0:
            return 1
        return int(prod(int(dim) for dim in x.shape))

    @staticmethod
    def _storage_bits(x: jax.Array | None) -> int:
        """Return storage size of *x* in bits, or 0 if *x* is ``None``."""
        if x is None:
            return 0
        return int(QuantizedArray._numel(x) * int(x.dtype.itemsize) * 8)

    @property
    def logical_num_values(self) -> int:
        """Number of dequantized scalar values represented by this payload."""
        if self.scales.ndim == 0:
            return int(self.group_size)
        prefix = 1 if self.scales.ndim == 1 else int(prod(int(dim) for dim in self.scales.shape[:-1]))
        n_groups = int(self.scales.shape[-1])
        return int(prefix * n_groups * int(self.group_size))

    @property
    def data_storage_bits(self) -> int:
        """Storage size of the packed weight data array in bits."""
        return self._storage_bits(self.data)

    @property
    def scales_storage_bits(self) -> int:
        """Storage size of the scales metadata array in bits."""
        return self._storage_bits(self.scales)

    @property
    def zeros_storage_bits(self) -> int:
        """Storage size of the zeros metadata array in bits (0 for non-affine modes)."""
        return self._storage_bits(self.zeros)

    @property
    def storage_bits(self) -> int:
        """Total storage in bits: packed data + scales + zeros (if present)."""
        return self.data_storage_bits + self.scales_storage_bits + self.zeros_storage_bits

    @property
    def storage_bytes(self) -> int:
        """Total storage in bytes (``storage_bits // 8``)."""
        return self.storage_bits // 8

    @property
    def storage_kib(self) -> float:
        """Total storage in kibibytes (KiB = 1024 bytes)."""
        return float(self.storage_bytes / 1024.0)

    @property
    def storage_mib(self) -> float:
        """Total storage in mebibytes (MiB = 1024 KiB)."""
        return float(self.storage_bytes / (1024.0 * 1024.0))

    @property
    def payload_bits_per_value(self) -> float:
        """Bits per dequantized value contributed by the packed payload only.

        Does not include metadata (scales/zeros) overhead.
        Returns NaN when ``logical_num_values`` is zero.
        """
        count = self.logical_num_values
        if count == 0:
            return float("nan")
        return float(self.data_storage_bits / count)

    @property
    def metadata_bits_per_value(self) -> float:
        """Bits per dequantized value contributed by scales and zeros overhead.

        Returns NaN when ``logical_num_values`` is zero.
        """
        count = self.logical_num_values
        if count == 0:
            return float("nan")
        return float((self.scales_storage_bits + self.zeros_storage_bits) / count)

    @property
    def actual_bits_per_value(self) -> float:
        """Effective bits/value including packed payload and metadata overhead."""
        count = self.logical_num_values
        if count == 0:
            return float("nan")
        return float(self.storage_bits / count)

    @property
    def effective_bits_per_value(self) -> float:
        """Alias for ``actual_bits_per_value``."""
        return self.actual_bits_per_value

    def __repr__(self) -> str:
        return (
            "QuantizedArray("
            f"data_shape={self._shape_or_none(self.data)}, "
            f"scales_shape={self._shape_or_none(self.scales)}, "
            f"zeros_shape={self._shape_or_none(self.zeros)}, "
            f"actual_bits_per_value={self.actual_bits_per_value:.3f}, "
            f"storage_bytes={self.storage_bytes}, "
            f"storage_kib={self.storage_kib:.3f}, "
            f"mode={self.mode!r}, "
            f"group_size={self.group_size}, "
            f"bits={self.bits}, "
            f"axis={self.axis!r}, "
            f"runtime_config={self.runtime_config!r}"
            ")"
        )

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def from_quantized(
        cls,
        data: jax.Array,
        scales: jax.Array,
        zeros: jax.Array | None = None,
        *,
        group_size: int | None = None,
        bits: int | None = None,
        mode: QuantizationMode = "affine",
        axis: QuantizationAxis = "row",
        runtime_config: QuantRuntimeConfig | None = None,
    ) -> QuantizedArray:
        """Construct a container from already-quantized buffers.

        Use this constructor when you already have the packed ``data``,
        ``scales``, and (for affine mode) ``zeros`` tensors from a prior call
        to :func:`quantize` or :func:`prepack_quantized_weights`.

        Args:
            data: Bit-packed uint32 weight array.
            scales: Per-group scale metadata.
            zeros: Per-group zero-point metadata.  Required for
                ``mode='affine'``; must be ``None`` for all other modes.
            group_size: Number of elements per group, or ``None`` for the
                mode-specific default (see :func:`resolve_qparams`).
            bits: Bit-width of stored codes, or ``None`` for the
                mode-specific default.
            mode: Quantization mode string.
            axis: Quantization axis (``"row"`` or ``"col"``).
            runtime_config: Optional runtime fast-path policy.

        Returns:
            A :class:`QuantizedArray` with normalized parameters.

        Raises:
            ValueError: If ``zeros`` is ``None`` for affine mode, or if
                ``zeros`` is not ``None`` for a non-affine mode.
        """
        axis_n = normalize_axis(axis)
        mode_n, group_size_n, bits_n, _ = resolve_qparams(mode, group_size, bits)

        if mode_n == "affine" and zeros is None:
            raise ValueError("affine QuantizedArray requires `zeros`.")
        if mode_n != "affine" and zeros is not None:
            raise ValueError("zeros must be None for non-affine QuantizedArray modes.")

        return cls(
            data=data,
            scales=scales,
            zeros=zeros,
            mode=mode_n,
            group_size=group_size_n,
            bits=bits_n,
            axis=axis_n,
            runtime_config=runtime_config,
        )

    def as_tuple(
        self,
    ) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
        """Return the payload as a legacy tuple compatible with low-level APIs.

        Returns:
            For affine mode: ``(data, scales, zeros)``.
            For all other modes: ``(data, scales)``.
        """
        if self.mode == "affine":
            assert self.zeros is not None
            return self.data, self.scales, self.zeros
        return self.data, self.scales

    def dequantize(self, *, runtime_config: QuantRuntimeConfig | None = None) -> jax.Array:
        """Dequantize this container into floating-point weights.

        Calls :func:`dequantize` with the stored metadata and returns the
        reconstructed weight tensor.  The output dtype is controlled by
        ``runtime_config.dequant_output_dtype`` (or the stored
        ``self.runtime_config`` when *runtime_config* is ``None``).

        Args:
            runtime_config: Optional override for the stored
                ``self.runtime_config``.  When ``None``, the instance's own
                ``runtime_config`` is used, falling back to backend defaults.

        Returns:
            Dequantized weight tensor.  Shape follows the last dimension of
            the ``scales`` array times ``group_size``, with leading batch
            dimensions preserved.
        """
        runtime_cfg = self.runtime_config if runtime_config is None else runtime_config
        return dequantize(
            self.data,
            self.scales,
            self.zeros,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            axis=self.axis,
            runtime_config=runtime_cfg,
        )

    def matmul(
        self,
        x: jax.Array,
        *,
        fuse: bool = True,
        strict_fuse: bool | None = None,
        tpu_path: Literal["packed"] | None = None,
        allow_dense_fallback: bool | None = None,
        transpose: bool | None = None,
        axis: QuantizationAxis | None = None,
        platform: Literal["triton", "pallas", "cuda", "cute", "xla", "auto"] | None = None,
    ) -> jax.Array:
        """Run quantized matmul against activation tensor *x*.

        When ``fuse=True`` (the default), dispatches to
        ``ejkernel.modules.operations.quantized_matmul``, which selects the
        fastest fused kernel available for the current backend and shape.

        When ``fuse=False``, falls back to the dense reference path:
        dequantize the weights first, then compute ``x @ w`` (or ``x @ w.T``
        depending on *transpose*/*axis*).

        Args:
            x: Activation tensor of shape ``(..., K)``.
            fuse: Whether to use a fused quantized-matmul kernel.  Default
                ``True``.
            strict_fuse: If ``True``, raise when no fused kernel is available
                instead of falling back.  Forwarded to the fused kernel; ignored
                when ``fuse=False``.
            tpu_path: Optional TPU-specific kernel path override.  Only
                meaningful on TPU backends; forwarded to the fused kernel.
            allow_dense_fallback: Whether the fused kernel may silently fall
                back to a dequantize-then-dense path.  Forwarded to the fused
                kernel.
            transpose: Whether to transpose the weight matrix before matmul.
                When ``None`` (default), inferred from ``axis``:
                ``axis='col'`` implies ``transpose=True``.
            axis: Quantization axis override.  When ``None``, uses
                ``self.axis``.
            platform: Backend platform hint for the fused kernel.  One of
                ``"triton"``, ``"pallas"``, ``"cuda"``, ``"cute"``,
                ``"xla"``, ``"auto"``.

        Returns:
            Result of the matrix multiplication, shape ``(..., N)``.
        """
        axis_n = self.axis if axis is None else normalize_axis(axis)
        transpose_n = (axis_n == "col") if transpose is None else bool(transpose)
        if fuse:
            from ejkernel.modules.operations import quantized_matmul as fused_quantized_matmul

            return fused_quantized_matmul(
                x,
                self.data,
                self.scales,
                self.zeros,
                transpose=transpose_n,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
                axis=axis_n,
                platform=platform,
                fuse=True,
                strict_fuse=strict_fuse,
                tpu_path=tpu_path,
                allow_dense_fallback=allow_dense_fallback,
            )

        return dense_quantized_matmul(
            x,
            self.data,
            self.scales,
            self.zeros,
            transpose=transpose_n,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            axis=axis_n,
        )


def quantize_array(
    w: jax.Array,
    /,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    axis: QuantizationAxis | None = None,
    runtime_config: QuantRuntimeConfig | None = None,
) -> QuantizedArray:
    """Quantize *w* and return a :class:`QuantizedArray` container.

    This is the primary high-level entry point for quantizing a weight tensor.
    It calls :func:`quantize` and wraps the result in a :class:`QuantizedArray`
    that can be used directly with :meth:`QuantizedArray.matmul` or
    :meth:`QuantizedArray.dequantize`.

    If *axis* is omitted, a compatible axis is inferred from the input shape:

    - Prefer ``axis='row'`` when ``w.shape[-2]`` is divisible by ``group_size``
      (groups over output channels, common for column-major weight packing).
    - Fall back to ``axis='col'`` when ``w.shape[-1]`` is divisible by
      ``group_size`` (groups over input channels).
    - Raise :exc:`ValueError` when neither dimension is compatible.

    Args:
        w: Weight tensor with at least 2 dimensions.
        group_size: Number of elements per quantization group.  ``None`` uses
            the mode-specific default (e.g., 64 for affine/nf4).
        bits: Bit-width of quantized codes.  ``None`` uses the mode-specific
            default (e.g., 4 for affine/nf4).
        mode: Quantization mode.  One of ``"affine"``, ``"nf4"``,
            ``"mxfp4"``, ``"mxfp8"``, ``"nvfp4"``, ``"nvfp8"``.
        axis: Quantization axis.  When ``None``, inferred from ``w.shape``
            as described above.
        runtime_config: Optional runtime fast-path policy.

    Returns:
        A :class:`QuantizedArray` containing the packed weights and metadata.

    Raises:
        ValueError: If ``w`` has fewer than 2 dimensions, or if *axis* is
            ``None`` and neither ``w.shape[-2]`` nor ``w.shape[-1]`` is
            divisible by ``group_size``.
    """
    mode_n, group_size_n, bits_n, _ = resolve_qparams(mode, group_size, bits)
    if axis is None:
        if w.ndim < 2:
            raise ValueError("quantize_array expects inputs with two or more dimensions.")
        if int(w.shape[-2]) % group_size_n == 0:
            axis_n: QuantizationAxis = "row"
        elif int(w.shape[-1]) % group_size_n == 0:
            axis_n = "col"
        else:
            raise ValueError(
                "group_size is incompatible with both possible grouping axes. "
                f"input_shape={tuple(w.shape)}, group_size={group_size_n}, "
                f"dim[-2]={int(w.shape[-2])}, dim[-1]={int(w.shape[-1])}. "
                "Pass axis='row' or axis='col' explicitly."
            )
    else:
        axis_n = normalize_axis(axis)
    out = quantize(
        w,
        group_size=group_size_n,
        bits=bits_n,
        mode=mode_n,
        axis=axis_n,
        runtime_config=runtime_config,
    )
    if mode_n == "affine":
        data, scales, zeros = out
    else:
        data, scales = out
        zeros = None
    return QuantizedArray(
        data=data,
        scales=scales,
        zeros=zeros,
        mode=mode_n,
        group_size=group_size_n,
        bits=bits_n,
        axis=axis_n,
        runtime_config=runtime_config,
    )


def prepack_quantized_array(
    w: jax.Array,
    /,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    transpose: bool = True,
    axis: QuantizationAxis | None = None,
    runtime_config: QuantRuntimeConfig | None = None,
) -> QuantizedArray:
    """Quantize and prepack *w* into a :class:`QuantizedArray` for fused kernels.

    Thin wrapper around :func:`prepack_quantized_weights` that bundles the
    result into a :class:`QuantizedArray`.  The axis convention follows
    :func:`resolve_prepack_axis`:

    - If *axis* is provided, it is used directly.
    - Otherwise, ``transpose=True`` maps to ``axis='row'`` (the legacy default)
      and ``transpose=False`` maps to ``axis='col'``.

    This function is intended for offline weight preparation.  The returned
    :class:`QuantizedArray` can be passed directly to fused kernel wrappers or
    used via :meth:`QuantizedArray.matmul`.

    Args:
        w: Weight tensor of shape ``(out_features, in_features)`` or with
            additional leading batch dimensions.
        group_size: Number of elements per quantization group, or ``None``
            for the mode-specific default.
        bits: Bit-width of quantized codes, or ``None`` for the mode-specific
            default.
        mode: Quantization mode.  One of ``"affine"``, ``"nf4"``,
            ``"mxfp4"``, ``"mxfp8"``, ``"nvfp4"``, ``"nvfp8"``.
        transpose: Legacy flag used to infer ``axis`` when *axis* is ``None``.
            ``True`` -> ``axis='row'``.
        axis: Explicit quantization axis.  Overrides *transpose* when provided.
        runtime_config: Optional runtime fast-path policy.

    Returns:
        A :class:`QuantizedArray` ready for use with fused matmul kernels.
    """
    axis_n = resolve_prepack_axis(axis=axis, transpose=transpose)
    mode_n, group_size_n, bits_n, _ = resolve_qparams(mode, group_size, bits)
    out = prepack_quantized_weights(
        w,
        group_size=group_size_n,
        bits=bits_n,
        mode=mode_n,
        axis=axis_n,
        runtime_config=runtime_config,
    )
    if mode_n == "affine":
        data, scales, zeros = out
    else:
        data, scales = out
        zeros = None
    return QuantizedArray(
        data=data,
        scales=scales,
        zeros=zeros,
        mode=mode_n,
        group_size=group_size_n,
        bits=bits_n,
        axis=axis_n,
        runtime_config=runtime_config,
    )


__all__ = ("QuantizedArray", "prepack_quantized_array", "quantize_array")
