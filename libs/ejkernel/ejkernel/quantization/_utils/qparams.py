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

"""Shared quantization parameter normalization utilities.

This module centralizes normalization and validation for quantization mode,
bit-width, group-size, and quantization axis.
"""

from __future__ import annotations

from typing import Literal

from .grouping import _require_bits

QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]
QuantizationAxis = Literal["row", "col"]
BackendQuantizationMode = QuantizationMode
GemvMode = Literal["auto", "on", "off"]
RevSplitKMode = Literal["auto", "on", "off"]
KernelFamily = Literal["gemm", "gemm_splitk", "gemv_splitk", "gemv_revsplitk"]
AFFINE_NF4_GROUP_SIZES = {16, 32, 64, 128, 256, 512, 1024}


def normalize_axis(axis: str | None, *, default: QuantizationAxis = "row") -> QuantizationAxis:
    """Normalize and validate the quantization axis name.

    Converts the axis string to lowercase and validates it against the
    allowed set of axis names.

    Args:
        axis: Axis name string, or None to use the default.
        default: Default axis to return when axis is None.

    Returns:
        Normalized axis name, either "row" or "col".

    Raises:
        ValueError: If axis is not one of {"row", "col"}.
    """
    if axis is None:
        return default
    axis = axis.lower()
    if axis not in {"row", "col"}:
        raise ValueError("axis must be one of {'row','col'}.")
    return axis  # type: ignore[return-value]


def normalize_mode_and_bits(
    mode: str,
    bits: int | None,
) -> tuple[QuantizationMode, int | None, bool]:
    """Normalize and validate explicit quantization mode and optional bit-width.

    Converts mode to lowercase and checks it against the set of supported
    modes. Ambiguous shorthand names ("mxfp", "nvfp") are rejected with
    a helpful error message.

    Args:
        mode: Quantization mode string (case-insensitive). Must be one of
            {"affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"}.
        bits: Optional bit-width override. Converted to int if provided.

    Returns:
        Tuple of (canonical_mode, bits, used_legacy_alias) where
        used_legacy_alias is always False for current modes.

    Raises:
        ValueError: If mode is ambiguous ("mxfp", "nvfp") or unsupported.
    """
    mode = mode.lower()
    if mode in {"mxfp", "nvfp"}:
        raise ValueError("Use explicit mode names: {'affine','nf4','mxfp4','mxfp8','nvfp4','nvfp8'}.")
    if mode not in {"affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"}:
        raise ValueError("Unsupported quantization mode. Use one of {'affine','nf4','mxfp4','mxfp8','nvfp4','nvfp8'}.")

    if bits is not None:
        bits = int(bits)

    return mode, bits, False  # type: ignore[return-value]


def resolve_qparams(
    mode: str,
    group_size: int | None,
    bits: int | None,
) -> tuple[QuantizationMode, int, int, bool]:
    """Resolve and validate the full set of quantization parameters.

    Applies mode-specific defaults and constraints for bit-width and
    group-size, raising clear errors when user-supplied values violate
    the rules for the chosen quantization mode.

    Mode-specific rules:
        - affine: bits in {1, 2, 3, 4, 5, 6, 7, 8} (default 4),
          group_size in {16, 32, 64, 128, 256, 512, 1024} (default 64).
        - nf4: bits fixed to 4, group_size in
          {16, 32, 64, 128, 256, 512, 1024} (default 64).
        - mxfp4 / mxfp8: bits fixed to 4 or 8 respectively, group_size must be 32.
        - nvfp4 / nvfp8: bits fixed to 4 or 8 respectively, group_size must be 16.

    Args:
        mode: Quantization mode string (passed through normalize_mode_and_bits).
        group_size: Number of elements per quantization group, or None for
            mode-specific default.
        bits: Bit-width for quantized values, or None for mode-specific default.

    Returns:
        Tuple of (mode, group_size, bits, used_legacy_alias).

    Raises:
        ValueError: If group_size or bits are invalid for the chosen mode.
    """
    mode, bits, used_legacy = normalize_mode_and_bits(mode, bits)

    if mode == "affine":
        bits = 4 if bits is None else _require_bits(bits, set(range(1, 9)))
        group_size = 64 if group_size is None else int(group_size)
        if group_size not in AFFINE_NF4_GROUP_SIZES:
            raise ValueError("affine mode supports group_size in {16,32,64,128,256,512,1024}.")
        return mode, group_size, bits, used_legacy

    if mode == "nf4":
        bits = 4
        group_size = 64 if group_size is None else int(group_size)
        if group_size not in AFFINE_NF4_GROUP_SIZES:
            raise ValueError("nf4 mode supports group_size in {16,32,64,128,256,512,1024}.")
        return mode, group_size, bits, used_legacy

    if mode in {"mxfp4", "mxfp8"}:
        bits = 4 if mode == "mxfp4" else 8
        group_size = 32 if group_size is None else int(group_size)
        if group_size != 32:
            raise ValueError(f"{mode} requires group_size=32.")
        return mode, group_size, bits, used_legacy

    bits = 4 if mode == "nvfp4" else 8
    group_size = 16 if group_size is None else int(group_size)
    if group_size != 16:
        raise ValueError(f"{mode} requires group_size=16.")
    return mode, group_size, bits, used_legacy


def to_backend_mode(mode: QuantizationMode, bits: int) -> BackendQuantizationMode:
    """Map a user-facing quantization mode to the backend mode key.

    In the current implementation, the backend mode is identical to the
    user-facing mode. The ``bits`` parameter is accepted for backward
    compatibility but is ignored.

    Args:
        mode: User-facing quantization mode.
        bits: Bit-width (ignored; kept for API compatibility).

    Returns:
        Backend quantization mode string, identical to the input mode.
    """
    del bits
    return mode


def normalize_gemv_mode(mode: str | None) -> GemvMode:
    """Normalize and validate the GEMV dispatch override mode.

    Args:
        mode: GEMV mode string ("auto", "on", "off"), or None for "auto".

    Returns:
        Normalized GEMV mode string.

    Raises:
        ValueError: If mode is not one of {"auto", "on", "off"}.
    """
    if mode is None:
        return "auto"
    norm = str(mode).lower()
    if norm not in {"auto", "on", "off"}:
        raise ValueError("gemv_mode must be one of {'auto','on','off'}.")
    return norm  # type: ignore[return-value]


def normalize_revsplitk_mode(mode: str | None) -> RevSplitKMode:
    """Normalize and validate the reverse split-K dispatch override mode.

    Args:
        mode: Reverse split-K mode string ("auto", "on", "off"), or None for "auto".

    Returns:
        Normalized reverse split-K mode string.

    Raises:
        ValueError: If mode is not one of {"auto", "on", "off"}.
    """
    if mode is None:
        return "auto"
    norm = str(mode).lower()
    if norm not in {"auto", "on", "off"}:
        raise ValueError("revsplit_k must be one of {'auto','on','off'}.")
    return norm  # type: ignore[return-value]


def normalize_revsplitk_parts(parts: int | None) -> int | None:
    """Normalize and validate the optional reverse split-K partition count.

    Only small powers of two are supported to ensure efficient GPU utilization.

    Args:
        parts: Number of split-K partitions, or None to leave unset.
            Must be one of {1, 2, 4, 8, 16} if provided.

    Returns:
        Validated partition count as int, or None if not provided.

    Raises:
        ValueError: If parts is not one of {1, 2, 4, 8, 16}.
    """
    if parts is None:
        return None
    parts = int(parts)
    if parts not in {1, 2, 4, 8, 16}:
        raise ValueError("revsplit_k_parts must be one of {1,2,4,8,16}.")
    return parts


def is_effective_4bit_mode(mode: QuantizationMode, bits: int) -> bool:
    """Check whether the effective runtime quantization is 4-bit.

    For affine mode, the bit-width is user-configurable from 1 through 8.
    For other modes, the effective bit-width is determined by the mode name
    (e.g., nf4, mxfp4, nvfp4 are all 4-bit).

    Args:
        mode: Quantization mode.
        bits: Bit-width parameter (only used for affine mode).

    Returns:
        True if the runtime quantization uses 4-bit values, False otherwise.
    """
    if mode == "affine":
        return int(bits) == 4
    return mode in {"nf4", "mxfp4", "nvfp4"}


def select_qmm_kernel_family(
    *,
    m: int,
    mode: QuantizationMode,
    bits: int,
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
) -> tuple[KernelFamily, int | None]:
    """Select the quantized matmul kernel family using a GemLite-style policy.

    The selection policy considers the activation batch dimension (M), the
    effective bit-width, and user overrides (gemv_mode, revsplit_k) to choose
    the best kernel family for the workload.

    Selection policy:
        - M > 64: "gemm" (standard batched GEMM).
        - 1 < M <= 64: "gemm_splitk" (split-K GEMM for moderate batch).
        - M == 1 with MX modes: "gemm_splitk" (MX paths use GEMM-SplitK for M=1).
        - M == 1, 4-bit effective: "gemv_revsplitk" (reverse split-K GEMV).
        - M == 1, 8-bit effective: "gemv_splitk" (split-K GEMV).

    Args:
        m: Activation leading dimension (batch size for matmul). Must be >= 1.
        mode: Quantization mode.
        bits: Bit-width parameter.
        gemv_mode: GEMV dispatch override ("auto", "on", "off").
        revsplit_k: Reverse split-K override ("auto", "on", "off").
        revsplit_k_parts: Number of reverse split-K partitions, or None.

    Returns:
        Tuple of (kernel_family, revsplitk_parts) where kernel_family is one
        of {"gemm", "gemm_splitk", "gemv_splitk", "gemv_revsplitk"} and
        revsplitk_parts is the number of partitions (or None if not applicable).

    Raises:
        ValueError: If m < 1, gemv_mode="on" with M != 1, or revsplit_k="on"
            with a non-4-bit mode.
    """
    m = int(m)
    if m <= 0:
        raise ValueError("Input activation leading dimension M must be >= 1.")

    gemv_mode_n = normalize_gemv_mode(gemv_mode)
    revsplit_k_n = normalize_revsplitk_mode(revsplit_k)
    revsplit_k_parts_n = normalize_revsplitk_parts(revsplit_k_parts)
    is_4bit = is_effective_4bit_mode(mode, bits)

    if gemv_mode_n == "on" and m != 1:
        raise ValueError("gemv_mode='on' requires M == 1.")

    if m == 1 and mode in {"mxfp4", "mxfp8"}:
        return "gemm_splitk", None

    use_gemv = (m == 1) if gemv_mode_n == "auto" else (gemv_mode_n == "on")
    if not use_gemv:
        return ("gemm" if m > 64 else "gemm_splitk"), None

    if revsplit_k_n == "on":
        if not is_4bit:
            raise ValueError("revsplit_k='on' requires an effective 4-bit mode.")
        parts = 2 if revsplit_k_parts_n is None else revsplit_k_parts_n
        if parts < 2:
            raise ValueError("revsplit_k='on' requires revsplit_k_parts >= 2.")
        return "gemv_revsplitk", parts

    if revsplit_k_n == "off":
        return "gemv_splitk", None

    if is_4bit:
        if revsplit_k_parts_n is None:
            return "gemv_revsplitk", 2
        return "gemv_revsplitk", max(2, revsplit_k_parts_n)
    return "gemv_splitk", None


def resolve_runtime_axis_and_transpose(
    *,
    axis: str | None,
    transpose: bool,
) -> tuple[QuantizationAxis, bool]:
    """Resolve runtime quantization axis and ensure transpose consistency.

    Enforces the bidirectional mapping between axis and transpose:
        - axis="row" requires transpose=False (no transpose needed).
        - axis="col" requires transpose=True (weight is transposed).

    When axis is omitted (None), the transpose flag determines the axis.

    Args:
        axis: Quantization axis string ("row" or "col"), or None to
            infer from the transpose flag.
        transpose: Whether the weight matrix is transposed at runtime.

    Returns:
        Tuple of (resolved_axis, resolved_transpose).

    Notes:
        When ``axis`` is explicitly provided, it is treated as the source of
        truth and ``transpose`` is ignored. This keeps the public API ergonomic:
        callers may pass ``axis='col'`` without also setting ``transpose=True``.
    """
    if axis is None:
        return ("col" if transpose else "row"), transpose

    axis_n = normalize_axis(axis)
    expected_transpose = axis_n == "col"
    return axis_n, expected_transpose


def resolve_prepack_axis(*, axis: str | None, transpose: bool) -> QuantizationAxis:
    """Resolve the quantization axis for the prepack API.

    When axis is explicitly provided, it is normalized and returned directly.
    When axis is omitted (None), the transpose flag provides backward-compatible
    axis inference:
        - transpose=True maps to axis="row" (legacy default).
        - transpose=False maps to axis="col".

    Args:
        axis: Explicit axis string ("row" or "col"), or None to infer
            from the transpose flag.
        transpose: Legacy transpose flag used for axis inference when
            axis is None.

    Returns:
        Resolved quantization axis, either "row" or "col".
    """
    if axis is not None:
        return normalize_axis(axis)
    return "row" if bool(transpose) else "col"


def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError(f"ceil_div expects b > 0, got b={b}.")
    return (int(a) + int(b) - 1) // int(b)


def validate_packed_quantized_matmul_layout(
    x,
    w_q,
    scales,
    zeros,
    *,
    mode: QuantizationMode,
    group_size: int,
    bits: int,
    axis: QuantizationAxis,
    transpose: bool,
) -> tuple[int, int]:
    """Validate packed QMM weight/metadata layout against the canonical contract.

    This is a fast, shape-only preflight intended to catch mismatched packed
    layouts (e.g., swapping axis='row'/'col' inputs) before dispatching into
    fused kernels where layout mismatches can silently trigger slow fallbacks.

    Canonical runtime contract:
      - ``axis='row'`` (``transpose=False``): ``w_q`` is ``(K, ceil(N * bits / 32))``,
        ``scales``/``zeros`` are ``(K, N // group_size)``.
      - ``axis='col'`` (``transpose=True``): ``w_q`` is ``(N, ceil(K * bits / 32))``,
        ``scales``/``zeros`` are ``(N, K // group_size)``.

    Args:
        x: Activation array with ``x.shape[-1] == K``.
        w_q: Packed uint32 weight codes, rank-2.
        scales: Scale metadata array, rank-2.  Must be float for affine/nf4
            modes and uint8 for mx/nv modes.
        zeros: Zero-point array for affine mode (same shape as ``scales``),
            or ``None`` for all other modes.
        mode: Quantization mode; one of ``{"affine", "nf4", "mxfp4", "mxfp8",
            "nvfp4", "nvfp8"}``.
        group_size: Number of elements per quantization group.
        bits: Bit-width of the packed codes.
        axis: Quantization axis, either ``"row"`` or ``"col"``.
        transpose: Whether the weight is transposed at runtime.  Must be
            consistent with ``axis``: ``"row"`` ↔ ``False``, ``"col"`` ↔ ``True``.

    Returns:
        Tuple ``(K, N)`` where ``K`` is the input channel dimension and ``N``
        is the output channel dimension, both inferred from the input shapes.

    Raises:
        ValueError: When shapes or dtypes are inconsistent with the contract,
            including mismatched ``axis``/``transpose``, incorrect ``w_q`` dtype
            (must be uint32), wrong scale dtype, missing or unexpected ``zeros``,
            or shape mismatches between ``x``, ``w_q``, and ``scales``.
    """
    axis_n = normalize_axis(axis)
    transpose_n = bool(transpose)
    expected_transpose = axis_n == "col"
    if expected_transpose != transpose_n:
        raise ValueError(
            "Inconsistent axis/transpose combination: "
            f"axis={axis_n!r} requires transpose={expected_transpose}, got transpose={transpose_n}."
        )

    group_size = int(group_size)
    bits = int(bits)
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}.")
    if bits <= 0:
        raise ValueError(f"bits must be > 0, got {bits}.")
    if bits > 32:
        raise ValueError(f"Invalid bits={bits}: bit-width must be <= 32.")

    x_shape = getattr(x, "shape", None)
    w_shape = getattr(w_q, "shape", None)
    s_shape = getattr(scales, "shape", None)
    if x_shape is None or w_shape is None or s_shape is None:
        raise ValueError("validate_packed_quantized_matmul_layout expects array-like inputs with .shape.")

    if len(x_shape) < 1:
        raise ValueError(f"x must have at least 1 dimension, got x.shape={tuple(x_shape)}.")
    if len(w_shape) != 2:
        raise ValueError(f"w_q must be rank-2, got w_q.shape={tuple(w_shape)}.")
    if len(s_shape) != 2:
        raise ValueError(f"scales must be rank-2, got scales.shape={tuple(s_shape)}.")

    w_dtype = getattr(w_q, "dtype", None)
    if w_dtype is not None and str(w_dtype) != "uint32":
        raise ValueError(f"w_q must be uint32 packed codes, got w_q.dtype={w_dtype!s}.")

    s_dtype = getattr(scales, "dtype", None)
    s_name = str(s_dtype) if s_dtype is not None else "<unknown>"
    if mode in {"mxfp4", "mxfp8", "nvfp4", "nvfp8"}:
        if s_dtype is not None and str(s_dtype) != "uint8":
            raise ValueError(f"{mode} scales must be uint8, got scales.dtype={s_name}.")
    else:
        if s_dtype is not None and "float" not in s_name:
            raise ValueError(f"{mode} scales must be a floating dtype, got scales.dtype={s_name}.")

    if mode == "affine":
        if zeros is None:
            raise ValueError("affine quantized_matmul requires `zeros`.")
        z_shape = getattr(zeros, "shape", None)
        if z_shape is None:
            raise ValueError("zeros must be array-like when provided.")
        if tuple(z_shape) != tuple(s_shape):
            raise ValueError(
                "affine zeros must have the same shape as scales. "
                f"zeros.shape={tuple(z_shape)} scales.shape={tuple(s_shape)}."
            )
        z_dtype = getattr(zeros, "dtype", None)
        z_name = str(z_dtype) if z_dtype is not None else "<unknown>"
        if z_dtype is not None and "float" not in z_name:
            raise ValueError(f"affine zeros must be a floating dtype, got zeros.dtype={z_name}.")
    else:
        if zeros is not None:
            raise ValueError(f"zeros must be None for non-affine mode={mode!r}.")

    K = int(x_shape[-1])
    if axis_n == "row":
        if int(w_shape[0]) != K:
            raise ValueError(
                "Packed layout mismatch for axis='row': expected w_q.shape[0] == K. "
                f"x.shape={tuple(x_shape)} w_q.shape={tuple(w_shape)} scales.shape={tuple(s_shape)}."
            )
        if int(s_shape[0]) != K:
            raise ValueError(
                "Packed layout mismatch for axis='row': expected scales.shape[0] == K. "
                f"x.shape={tuple(x_shape)} w_q.shape={tuple(w_shape)} scales.shape={tuple(s_shape)}."
            )
        N = int(s_shape[1]) * group_size
        exp_words = _ceil_div(N * bits, 32)
        if int(w_shape[1]) != exp_words:
            raise ValueError(
                "Packed layout mismatch for axis='row': expected w_q.shape[1] == ceil(N * bits / 32). "
                f"got w_q.shape[1]={int(w_shape[1])}, expected={exp_words}, "
                f"N={N}, bits={bits}."
            )
        return K, N

    exp_words = _ceil_div(K * bits, 32)
    if int(s_shape[1]) * group_size == K and int(w_shape[1]) == exp_words and int(w_shape[0]) == int(s_shape[0]):
        return K, int(w_shape[0])
    if int(s_shape[0]) * group_size == K and int(w_shape[0]) == exp_words and int(w_shape[1]) == int(s_shape[1]):
        return K, int(w_shape[1])
    raise ValueError(
        "Packed layout mismatch for axis='col': expected either "
        "w_q.shape=(N, ceil(K * bits / 32)), scales.shape=(N, K/group_size) "
        "or k-major w_q.shape=(ceil(K * bits / 32), N), scales.shape=(K/group_size, N). "
        f"x.shape={tuple(x_shape)} w_q.shape={tuple(w_shape)} scales.shape={tuple(s_shape)} "
        f"group_size={group_size} bits={bits} expected_words={exp_words}."
    )
