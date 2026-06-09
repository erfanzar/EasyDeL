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

"""Quantized matrix multiplication operation with automatic optimization.

This module implements fused quantized matrix multiplication, performing
dequantization and matmul in a single kernel pass for maximum throughput.
It supports multiple quantization formats and automatically selects the
optimal backend (Triton, Pallas, CUDA, CuTe, or XLA) based on the target
hardware and input characteristics.

Supported Quantization Modes:
    - affine: Standard asymmetric quantization with scales and zero-points
    - nf4: NormalFloat4 (QLoRA-style 4-bit quantization)
    - mxfp4: Microscaling FP4
    - mxfp8: Microscaling FP8
    - nvfp4: NVIDIA FP4
    - nvfp8: NVIDIA FP8

Key Features:
    - Fused dequantize + matmul in a single kernel pass
    - Automatic platform selection (Triton/Pallas/CUDA/CuTe/XLA)
    - Configurable block sizes with autotuning support
    - Split-K for improved parallelism on tall-skinny matrices
    - GEMV specialization for M=1 workloads
    - Custom VJP for backward pass compatibility
    - TPU default strategy selection between predecode-once and packed fused paths

Mathematical Formulation:
    output = x @ dequantize(w, scales, zeros)
    output = x @ dequantize(w, scales, zeros).T  (when transpose=True)

Performance Characteristics:
    - Eliminates memory bandwidth overhead of separate dequantization
    - Automatic split-K selection for small batch sizes
    - Hardware-specific heuristics for block size selection
    - Persistent configuration caching across runs

References:
    - QLoRA: https://arxiv.org/abs/2305.14314
    - GPTQ: https://arxiv.org/abs/2210.17323
    - Microscaling: https://arxiv.org/abs/2310.10537
"""

from __future__ import annotations

import functools
import math
import os
import warnings
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ejkernel.kernels._registry import Platform, kernel_registry
from ejkernel.ops import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    Invocation,
    Kernel,
    Tuner,
    policy_override,
)
from ejkernel.ops.config.persistent import PersistentCache
from ejkernel.quantization._utils.qparams import (
    GemvMode,
    QuantizationAxis,
    QuantizationMode,
    RevSplitKMode,
    normalize_gemv_mode,
    normalize_revsplitk_mode,
    normalize_revsplitk_parts,
    resolve_qparams,
    resolve_runtime_axis_and_transpose,
    select_qmm_kernel_family,
    validate_packed_quantized_matmul_layout,
)

from ..base import detect_platform
from .configs import QuantizedMatmulConfig


def _resolve_qparams(mode: str, group_size: int | None, bits: int | None) -> tuple[int, int]:
    """Resolve quantization parameters from mode, group_size, and bits.

    Args:
        mode: Quantization mode string (e.g., "affine", "nf4").
        group_size: Optional group size override.
        bits: Optional bit-width override.

    Returns:
        Tuple of (group_size, bits) with defaults applied based on mode.
    """
    _, group_size, bits, _ = resolve_qparams(mode, group_size, bits)
    return group_size, bits


def _static_bool(value, name: str) -> bool:
    """Extract a concrete boolean value, raising if it is a JAX tracer.

    Args:
        value: The value to concretize.
        name: Parameter name for the error message.

    Returns:
        The concrete boolean value.
    """
    return jax.core.concrete_or_error(bool, value, f"{name} must be static.")


def _static_int(value, name: str) -> int:
    """Extract a concrete integer value, raising if it is a JAX tracer.

    Args:
        value: The value to concretize.
        name: Parameter name for the error message.

    Returns:
        The concrete integer value.
    """
    return jax.core.concrete_or_error(int, value, f"{name} must be static.")


def _lcm(a: int, b: int) -> int:
    """Compute the least common multiple of two integers.

    Args:
        a: First integer. If <= 0, returns b.
        b: Second integer. If <= 0, returns a.

    Returns:
        The least common multiple of a and b.
    """
    if a <= 0:
        return int(b)
    if b <= 0:
        return int(a)
    return abs(a * b) // math.gcd(a, b)


def _bit_aligned_values(bits: int) -> int:
    """Return the value count that starts and ends on a 32-bit word boundary."""
    return 32 // math.gcd(32, int(bits))


def _ceil_div(a: int, b: int) -> int:
    """Compute ceiling division of a by b.

    Args:
        a: Numerator.
        b: Denominator.

    Returns:
        The smallest integer >= a/b.
    """
    return (a + b - 1) // b


def _nearest_choices(value: int, choices: tuple[int, ...], count: int = 2) -> list[int]:
    """Select the nearest choices to a target value from a set of options.

    Args:
        value: Target value to match.
        choices: Available choices to select from.
        count: Number of nearest choices to return.

    Returns:
        Sorted list of the `count` choices closest to `value`.
    """
    ranked = sorted(set(choices), key=lambda x: abs(x - value))
    return sorted(ranked[:count])


def _expand_choices(value: int, choices: tuple[int, ...]) -> list[int]:
    """Expand a value into a neighborhood of choices.

    Returns the value itself plus its immediate neighbors in the sorted
    choices list, providing a small search window for autotuning.

    Args:
        value: The base value to expand around.
        choices: Sorted tuple of available choices.

    Returns:
        Sorted list of up to 3 choices: the value and its neighbors.
    """
    choices = tuple(sorted(set(choices)))
    try:
        idx = choices.index(value)
    except ValueError:
        idx = 0
    out = {choices[idx]}
    if idx > 0:
        out.add(choices[idx - 1])
    if idx + 1 < len(choices):
        out.add(choices[idx + 1])
    return sorted(out)


def _ensure_aligned(choices: list[int], align: int, max_choice: int) -> list[int]:
    """Filter or round choices to ensure alignment.

    Returns only choices that are multiples of `align`. If none of the
    original choices are aligned, rounds up and filters by max_choice.

    Args:
        choices: List of candidate block sizes.
        align: Required alignment.
        max_choice: Maximum allowed value after rounding.

    Returns:
        List of aligned choices, or [align] as fallback.
    """
    if align <= 1:
        return choices
    aligned = [c for c in choices if c % align == 0]
    if aligned:
        return aligned
    rounded = [((c + align - 1) // align) * align for c in choices]
    aligned = [c for c in rounded if c <= max_choice]
    if aligned:
        return sorted(set(aligned))
    return [align]


def _inv_arg(inv: Invocation[QuantizedMatmulConfig, Array], name: str, index: int):
    """Resolve a positional-or-keyword argument from an Invocation.

    Args:
        inv: The kernel invocation containing args and kwargs.
        name: The keyword argument name to look up.
        index: The positional argument index to try first.

    Returns:
        The resolved argument value.
    """
    if len(inv.args) > index:
        return inv.args[index]
    return inv.kwargs[name]


def _infer_mkn(inv: Invocation[QuantizedMatmulConfig, Array], group_size: int) -> tuple[int, int, int, bool]:
    """Infer the M, K, N dimensions and transpose flag from an invocation.

    Extracts shape information from the input tensors (x, w, scales)
    to determine the effective matmul dimensions.

    Args:
        inv: The kernel invocation containing the input tensors.
        group_size: Quantization group size, used to compute N when
            transpose is False.

    Returns:
        Tuple of (M, K, N, transpose) where M is the batch dimension,
        K is the reduction dimension, N is the output dimension, and
        transpose indicates whether the weight is in transposed layout.
    """
    x = _inv_arg(inv, "x", 0)
    w = _inv_arg(inv, "w", 1)
    scales = _inv_arg(inv, "scales", 2)
    transpose = _static_bool(inv.kwargs.get("transpose", False), "transpose")
    M, K = x.shape
    if transpose:
        _, bits = _resolve_qparams(
            str(inv.kwargs.get("mode", "affine")),
            inv.kwargs.get("group_size"),
            inv.kwargs.get("bits"),
        )
        exp_words = _ceil_div(int(K) * int(bits), 32)
        if int(w.shape[0]) == exp_words and int(scales.shape[0]) * int(group_size) == int(K):
            N = w.shape[1]
        else:
            N = w.shape[0]
    else:
        N = scales.shape[1] * group_size
    return int(M), int(K), int(N), transpose


def _prefer_bf16(x: Array) -> bool:
    """Determine whether to prefer bfloat16 accumulation for the given input.

    Returns True only when the activation dtype is already bfloat16.

    On GPU, float16 is the typical fast-path compute type for quantized matmul
    and matches the default quant/dequant runtime config. Using bfloat16
    for float32 activations can introduce extra rounding error vs the reference
    dequantize+matmul path.

    Args:
        x: Input array to check dtype of.

    Returns:
        True if bfloat16 is preferred, False otherwise.
    """
    dt = getattr(x, "dtype", None)
    if dt is None:
        return True
    return dt == jnp.bfloat16


@functools.lru_cache(maxsize=1)
def _tilelang_runtime_available() -> bool:
    """Return whether TileLang FFI can actually run in this environment."""
    try:
        from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support

        return bool(has_tilelang_ffi_support())
    except Exception:
        return False


def _pick_split_k(m: int, k: int, block_k: int) -> int:
    """Select the split-K factor for improved parallelism on small M.

    When the M dimension is small and K is large, splitting the K
    reduction across multiple thread blocks improves GPU utilization.

    Args:
        m: M dimension (number of rows in the output).
        k: K dimension (reduction dimension).
        block_k: Block size along the K dimension.

    Returns:
        Split-K factor (1, 2, 4, or 8). Returns 1 for no split.
    """
    if block_k <= 0:
        return 1
    tiles = math.ceil(k / block_k)
    if tiles <= 1:
        return 1
    if m <= 128:
        if tiles >= 256:
            return 8
        if tiles >= 128:
            return 4
        if tiles >= 64:
            return 2
    return 1


def _xla_choices(hardware: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return block size choice tuples for XLA backend on the given hardware.

    Args:
        hardware: Hardware target string ("cpu", "tpu", or "gpu").

    Returns:
        Tuple of (block_m_choices, block_n_choices, block_k_choices) where
        each element is a tuple of valid block sizes for that dimension.
    """
    if hardware == "cpu":
        return (128, 256, 512), (128, 256, 512), (64, 128, 256)
    if hardware == "tpu":
        return (256, 512, 1024, 2048), (256, 512, 1024, 2048), (128, 256, 512)
    return (256, 512, 1024, 2048), (256, 512, 1024, 2048), (128, 256, 512)


def _parse_positive_int_env(name: str, default: int) -> int:
    """Parse a positive integer environment variable with a safe fallback.

    Args:
        name: The environment variable name to read.
        default: Fallback value used when the variable is unset or non-integer.

    Returns:
        The parsed integer value, clamped to at least 1.
    """
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return max(1, value)


def _parse_nonnegative_int_env(name: str, default: int) -> int:
    """Parse a non-negative integer environment variable with a safe fallback.

    Args:
        name: The environment variable name to read.
        default: Fallback value used when the variable is unset or non-integer.

    Returns:
        The parsed integer value, clamped to at least 0.
    """
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return max(0, value)


def _tpu_tile_working_set_bytes(*, block_m: int, block_n: int, block_k: int) -> int:
    """Estimate the per-tile bytes touched by the fused TPU QMM hot loop.

    Computes ``(block_m * block_k + block_k * block_n) * 2`` as a proxy for
    the bfloat16 working set size of one tile iteration (activations + weights).

    Args:
        block_m: Tile size along the M (batch) dimension.
        block_n: Tile size along the N (output) dimension.
        block_k: Tile size along the K (reduction) dimension.

    Returns:
        Estimated working-set size in bytes.
    """
    return int((block_m * block_k + block_k * block_n) * 2)


def _is_tracer_value(x: object) -> bool:
    """Return ``True`` when ``x`` is an abstract JAX tracer.

    Used to detect whether quantized weights are concrete (already materialised)
    or symbolic (still traced through JIT), which determines whether the
    predecode-once path can be used.

    Args:
        x: Any Python object to inspect.

    Returns:
        ``True`` if ``x`` is a ``jax.core.Tracer`` instance, else ``False``.
    """
    return isinstance(x, jax.core.Tracer)


def _env_tpu_default_strategy() -> str:
    """Read the TPU default QMM strategy from the environment.

    Controlled by the environment variable ``EJKERNEL_QMM_TPU_DEFAULT_STRATEGY``.

    Supported values:
        - ``"predecode_once"`` (default): Attempt predecode-once dense matmul
          first, then fall back to the fused Pallas packed path if inapplicable.
        - ``"packed"``: Skip the predecode-once attempt and dispatch directly to
          the fused Pallas packed path.

    Returns:
        One of ``"predecode_once"`` or ``"packed"``.
    """
    raw = os.getenv("EJKERNEL_QMM_TPU_DEFAULT_STRATEGY", "predecode_once").strip().lower()
    if raw in {"predecode_once", "packed"}:
        return raw
    return "predecode_once"


_QMM_BESTCFG_NON_AFFINE_MODES = frozenset({"nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"})
_QMM_TPU_BESTCFG_POLICY: dict[str, dict[str, dict[str, object]]] = {
    "affine": {
        "small": {"fuse": False, "platform": "xla"},
        "large": {
            "fuse": True,
            "platform": "pallas",
            "strict_fuse": True,
            "allow_dense_fallback": False,
            "tpu_path": "predecode",
        },
        "default": {"fuse": False, "platform": "xla"},
    },
    "non_affine": {
        "small": {
            "fuse": True,
            "platform": "pallas",
            "strict_fuse": False,
            "allow_dense_fallback": True,
            "tpu_path": "packed",
        },
        "large": {
            "fuse": True,
            "platform": "pallas",
            "strict_fuse": False,
            "allow_dense_fallback": True,
            "tpu_path": "packed",
        },
        "default": {
            "fuse": True,
            "platform": "pallas",
            "strict_fuse": False,
            "allow_dense_fallback": True,
            "tpu_path": "packed",
        },
    },
    "default": {
        "default": {"fuse": False, "platform": "xla"},
    },
}


def _qmm_bestcfg_mode_key(mode: str) -> str:
    """Map a quantization mode string to a best-config policy table key.

    Args:
        mode: Raw quantization mode string (e.g., ``"affine"``, ``"nf4"``).

    Returns:
        One of ``"affine"``, ``"non_affine"``, or ``"default"``.
    """
    mode_n = str(mode).strip().lower()
    if mode_n == "affine":
        return "affine"
    if mode_n in _QMM_BESTCFG_NON_AFFINE_MODES:
        return "non_affine"
    return "default"


def _lookup_best_qmm_policy(
    *,
    backend_name: str,
    mode: str,
    m_tokens: int,
    runtime_axis: QuantizationAxis,
    runtime_transpose: bool,
    weights_concrete: bool,
) -> dict[str, object]:
    """Return best-known policy controls for QMM runtime selection.

    For non-affine modes on TPU, the fused Pallas packed kernel is used
    regardless of whether weights are concrete or traced.  The Pallas
    kernel's in-kernel dequant uses pure arithmetic (no table lookups),
    making it both trace-safe and significantly faster than the XLA
    gather-based codebook decode path.
    """
    if backend_name != "tpu":
        return {}
    if runtime_axis != "row" or runtime_transpose:
        return {}

    threshold = _parse_positive_int_env("EJKERNEL_QMM_TPU_BESTCFG_SMALL_M", 1024)
    size_key = "small" if int(m_tokens) <= int(threshold) else "large"
    mode_key = _qmm_bestcfg_mode_key(mode)
    mode_table = _QMM_TPU_BESTCFG_POLICY.get(mode_key, _QMM_TPU_BESTCFG_POLICY["default"])
    chosen = mode_table.get(size_key, mode_table.get("default", {}))
    return dict(chosen) if isinstance(chosen, dict) else {}


def _should_try_tpu_predecode_once_default(
    *,
    fuse: bool,
    backend_name: str,
    runtime_axis: QuantizationAxis,
    runtime_transpose: bool,
    platform: Literal["triton", "pallas", "cuda", "cute", "tilelang", "xla", "auto"] | None,
    strategy_override: Literal["predecode_once", "packed"] | None = None,
) -> bool:
    """Determine whether to attempt the TPU predecode-once dense matmul path.

    The predecode-once path dequantizes weights to bfloat16 once outside JIT
    and stores the result, then performs dense bfloat16 matmul on each call.
    It is faster than the fused Pallas packed path when weights are static
    (concrete) and the model is repeatedly called with the same weights.

    Returns ``False`` whenever any guard condition is not met:
        - ``fuse`` must be ``True``.
        - ``backend_name`` must be ``"tpu"``.
        - ``runtime_axis`` must be ``"row"`` (``transpose=False``).
        - ``platform`` must not be an explicit non-Pallas selection.
        - The resolved strategy (from ``strategy_override`` or env var) must be
          ``"predecode_once"``.

    Args:
        fuse: Whether fused kernel dispatch is enabled.
        backend_name: Name of the JAX default backend (e.g. ``"tpu"``).
        runtime_axis: Resolved quantization axis (``"row"`` or ``"col"``).
        runtime_transpose: Whether the weight layout requires transposition.
        platform: Explicit platform override, if any.
        strategy_override: Per-call strategy override (``"predecode_once"`` or
            ``"packed"``), or ``None`` to read from the environment variable.

    Returns:
        ``True`` if the predecode-once path should be attempted, else ``False``.
    """
    if not fuse:
        return False
    if backend_name != "tpu":
        return False
    if runtime_axis != "row" or runtime_transpose:
        return False
    if platform not in (None, "auto", "pallas"):
        return False
    strategy = strategy_override if strategy_override is not None else _env_tpu_default_strategy()
    return strategy == "predecode_once"


def _maybe_tpu_predecode_once_matmul(
    x: Array,
    w: Array,
    scales: Array,
    zeros: Array | None,
    *,
    mode: QuantizationMode,
    group_size: int,
    bits: int,
) -> Array | None:
    """Try the TPU predecode-once + dense matmul path.

    Dequantizes ``w`` to a bfloat16 dense weight matrix using
    ``get_predecoded_dense_weight`` and performs a bfloat16
    ``dot_general`` matmul.  The dense weight is cached across calls so
    subsequent invocations only pay the cost of the matmul.

    Returns ``None`` when the path is inapplicable — e.g. when any of
    ``w``/``scales``/``zeros`` are JAX tracers (indicating JIT-traced weights
    that cannot be pre-decoded), or when the import of the predecode helper
    fails. In all these cases the caller should fall back to the fused kernel
    dispatch.

    Args:
        x: Input activation matrix ``(M, K)``.
        w: Packed quantized weights.
        scales: Per-group scale factors.
        zeros: Per-group zero-points (``None`` for non-affine modes).
        mode: Quantization mode (``"affine"`` or non-affine).
        group_size: Elements per quantization group.
        bits: Quantization bit-width, from 1 through 8 for affine mode.

    Returns:
        Dense matmul output ``(M, N)`` in bfloat16, or ``None`` if the path
        cannot be used.
    """
    if _is_tracer_value(w) or _is_tracer_value(scales) or (zeros is not None and _is_tracer_value(zeros)):
        return None
    try:
        from ejkernel.kernels._pallas.tpu.quantized_matmul._pallas_impl_core import get_predecoded_dense_weight
    except Exception:
        return None

    if mode == "affine":
        if zeros is None:
            return None
        safe_scale = jnp.where(scales == 0, jnp.ones_like(scales), scales)
        biases = -zeros * safe_scale
    else:
        biases = None

    try:
        dense_w = get_predecoded_dense_weight(
            w,
            scales,
            biases,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
    except Exception:
        return None

    return jax.lax.dot_general(
        x.astype(jnp.bfloat16),
        dense_w.astype(jnp.bfloat16),
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.bfloat16,
    )


def _packed_legal_block_n(
    n: int,
    *,
    group_size: int,
    bits: int,
    align_n: int,
    vmem_cap: int = 4096,
) -> int:
    """Return the smallest packed-legal block_n for the given N dimension.

    Mosaic TPU tiling requires that for a 2-D BlockSpec over packed words the
    trailing-dimension tile either equals the padded dimension or is a multiple
    of 128. The cheapest legal choice is ``block_n == n_pad`` which makes both
    packed-word and scale-group tile sizes exactly equal to the padded dimension.

    For very large *N* where ``n_pad`` would blow the VMEM budget, the fallback
    is the packed-lane alignment ``lcm(128 * bit_alignment, 128 * group_size)``
    which guarantees both tile sizes are multiples of 128.

    Args:
        n: The output dimension (unpacked, pre-padding).
        group_size: Elements per quantization group.
        bits: Quantization bit width, from 1 through 8.
        align_n: Base alignment for n (``lcm(128, lcm(group_size, bit_alignment))``).
        vmem_cap: Maximum acceptable block_n (default 4096).

    Returns:
        A packed-legal ``block_n`` value.
    """
    n_pad = max(align_n, _ceil_div(n, align_n) * align_n)
    if n_pad <= vmem_cap:
        return n_pad
    value_alignment = _bit_aligned_values(bits)
    packed_lane_align = _lcm(128 * value_alignment, 128 * group_size)
    bn = max(packed_lane_align, _ceil_div(n, packed_lane_align) * packed_lane_align)
    return min(bn, n_pad)


def _pallas_tpu_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
    """Generate a heuristic configuration for TPU Pallas quantized matmul.

    Selects block sizes based on matrix dimensions and quantization parameters.
    For packed-fused execution the block_n must satisfy Mosaic TPU tiling
    constraints; this heuristic prefers ``n_pad`` (the full padded N) which is
    always packed-legal and avoids the costly normalization fallback.

    Args:
        inv: The kernel invocation containing input tensors and kwargs.

    Returns:
        A QuantizedMatmulConfig tuned for TPU Pallas execution.
    """
    mode = str(inv.kwargs.get("mode", "affine"))
    group_size, bits = _resolve_qparams(mode, inv.kwargs.get("group_size"), inv.kwargs.get("bits"))
    m, k, n, _ = _infer_mkn(inv, group_size)
    value_alignment = _bit_aligned_values(bits)
    align_n = _lcm(128, _lcm(group_size, value_alignment))

    block_m = 256 if m >= 2048 else 128
    block_k = 256 if k >= 4096 else 128
    block_k = max(128, _ceil_div(block_k, 128) * 128)

    block_n = _packed_legal_block_n(n, group_size=group_size, bits=bits, align_n=align_n)

    return QuantizedMatmulConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=4,
        num_stages=2,
        use_bf16=True,
        split_k=None,
        tpu_path="packed",
        platform="pallas",
        backend="tpu",
    )


def _pallas_tpu_candidate_cfgs(inv: Invocation[QuantizedMatmulConfig, Array]) -> list[QuantizedMatmulConfig]:
    """Generate candidate configurations for autotuning TPU Pallas quantized matmul.

    Creates a grid of block size combinations, filtering by alignment
    constraints and TPU memory model limits. Configurations that are legal
    for the packed TPU path are preferred.

    Args:
        inv: The kernel invocation containing input tensors and kwargs.

    Returns:
        List of QuantizedMatmulConfig candidates for TPU autotuning.
    """
    mode = str(inv.kwargs.get("mode", "affine"))
    group_size, bits = _resolve_qparams(mode, inv.kwargs.get("group_size"), inv.kwargs.get("bits"))
    m, k, n, _ = _infer_mkn(inv, group_size)

    value_alignment = _bit_aligned_values(bits)
    align_n = _lcm(128, _lcm(group_size, value_alignment))
    bm_opts = (128, 256)
    n_pad = _ceil_div(n, align_n) * align_n
    packed_lane_align = _lcm(align_n, _lcm(group_size * 128, value_alignment * 128))
    bn_seed = {
        128,
        256,
        512,
        1024,
        n_pad,
        packed_lane_align,
        _ceil_div(n, packed_lane_align) * packed_lane_align,
    }
    if n_pad > align_n:
        bn_seed.add(_ceil_div(max(align_n, n_pad // 2), align_n) * align_n)
    bk_opts = (128, 256, 512)

    bn_opts = []
    for v in sorted(bn_seed):
        aligned = max(align_n, _ceil_div(v, align_n) * align_n)
        bn_opts.append(aligned)
    bn_opts = sorted(set(bn_opts))

    large_problem = m >= 128 and n >= 4096 and k >= 4096
    if large_problem:
        small_bn = max(align_n, 128)
        pressure_bn = {
            small_bn,
            n_pad,
            _ceil_div(n, packed_lane_align) * packed_lane_align,
        }
        bn_opts = sorted(bn for bn in bn_opts if bn in pressure_bn)
        if not bn_opts:
            bn_opts = [small_bn]
        bk_opts = (128,)

    max_candidates_default = _parse_positive_int_env("EJKERNEL_QMM_TPU_AUTOTUNE_MAX_CANDIDATES", 12)
    max_candidates_large_default = _parse_positive_int_env("EJKERNEL_QMM_TPU_AUTOTUNE_MAX_CANDIDATES_LARGE", 8)
    min_tile_bytes_large_default = _parse_nonnegative_int_env(
        "EJKERNEL_QMM_TPU_AUTOTUNE_MIN_TILE_BYTES_LARGE",
        96 * 1024,
    )
    min_tile_bytes = min_tile_bytes_large_default if large_problem else 0

    x = _inv_arg(inv, "x", 0)
    w = _inv_arg(inv, "w", 1)
    scales = _inv_arg(inv, "scales", 2)
    try:
        from ejkernel.kernels._pallas.tpu.quantized_matmul._pallas_impl_core import (
            is_packed_tpu_legal_forward as _packed_legal_forward,
        )
    except Exception:
        _packed_legal_forward = None

    configs: list[QuantizedMatmulConfig] = []
    target_bm = 256 if m >= 2048 else 128
    target_bn = _nearest_choices(n, tuple(bn_opts), count=1)[0]
    target_bk = 256 if k >= 4096 else 128

    for bm in bm_opts:
        for bn in bn_opts:
            for bk in bk_opts:
                if (
                    min_tile_bytes > 0
                    and _tpu_tile_working_set_bytes(block_m=bm, block_n=bn, block_k=bk) < min_tile_bytes
                ):
                    continue
                packed_legal = False
                if _packed_legal_forward is not None:
                    packed_legal = bool(
                        _packed_legal_forward(
                            x,
                            w,
                            scales,
                            group_size=group_size,
                            bits=bits,
                            block_m=bm,
                            block_n=bn,
                            block_k=bk,
                        )
                    )
                if packed_legal:
                    configs.append(
                        QuantizedMatmulConfig(
                            block_m=bm,
                            block_n=bn,
                            block_k=bk,
                            num_warps=4,
                            num_stages=2,
                            use_bf16=True,
                            split_k=None,
                            tpu_path="packed",
                            platform="pallas",
                            backend="tpu",
                        )
                    )

    if configs:

        def _score(cfg: QuantizedMatmulConfig) -> tuple[int, int]:
            distance = abs(cfg.block_m - target_bm) + abs(cfg.block_n - target_bn) + abs(cfg.block_k - target_bk)
            tile_work = cfg.block_m * cfg.block_n * cfg.block_k
            return (distance, -tile_work)

        dedup: dict[
            tuple[int, int, int, str, str, str],
            QuantizedMatmulConfig,
        ] = {}
        for cfg in configs:
            key = (
                int(cfg.block_m),
                int(cfg.block_n),
                int(cfg.block_k),
                str(cfg.tpu_path),
                str(cfg.platform),
                str(cfg.backend),
            )
            dedup.setdefault(key, cfg)
        ranked = sorted(dedup.values(), key=_score)
        max_candidates = max_candidates_large_default if large_problem else max_candidates_default
        return ranked[:max_candidates]

    if not configs:
        configs.append(_pallas_tpu_heuristic_cfg(inv))
    return configs


def _normalize_pallas_tpu_packed_cfg_forward(
    *,
    x: Array,
    w: Array,
    scales: Array,
    group_size: int,
    bits: int,
    cfg: QuantizedMatmulConfig,
) -> QuantizedMatmulConfig:
    """Coerce TPU Pallas forward tiles to a legal packed BlockSpec.

    This guards against stale cache entries or manual cfg overrides that use
    a block_n legal for dense/XLA but illegal for packed TPU BlockSpecs.
    """
    try:
        from ejkernel.kernels._pallas.tpu.quantized_matmul._pallas_impl_core import (
            is_packed_tpu_legal_forward as _packed_legal_forward,
        )
    except Exception:
        return cfg

    m_dim = int(x.shape[0]) if x.ndim >= 1 else int(cfg.block_m)
    k_dim = int(x.shape[1]) if x.ndim >= 2 else int(cfg.block_k)
    bm = max(8, _ceil_div(min(int(cfg.block_m), max(8, m_dim)), 8) * 8)
    bk = max(128, _ceil_div(min(int(cfg.block_k), max(128, k_dim)), 128) * 128)
    value_alignment = _bit_aligned_values(int(bits))
    align_n = _lcm(128, _lcm(int(group_size), value_alignment))
    n = int(scales.shape[-1]) * int(group_size)
    n_pad = max(align_n, _ceil_div(n, align_n) * align_n)
    packed_lane_align = _lcm(align_n, _lcm(int(group_size) * 128, value_alignment * 128))

    def _legal(block_n: int) -> bool:
        return bool(
            _packed_legal_forward(
                x,
                w,
                scales,
                group_size=int(group_size),
                bits=int(bits),
                block_m=bm,
                block_n=block_n,
                block_k=bk,
            )
        )

    bn_seed = {
        int(cfg.block_n),
        n_pad,
        packed_lane_align,
        _ceil_div(n, packed_lane_align) * packed_lane_align,
    }
    bn_opts: list[int] = []
    for v in sorted(bn_seed):
        aligned = max(align_n, _ceil_div(v, align_n) * align_n)
        bn_opts.append(int(aligned))
    bn_opts = sorted(set(bn_opts), key=lambda bn: (abs(bn - int(cfg.block_n)), bn))

    for bn in bn_opts:
        if _legal(bn):
            if bm == int(cfg.block_m) and bn == int(cfg.block_n) and bk == int(cfg.block_k):
                return cfg
            return QuantizedMatmulConfig(
                block_m=bm,
                block_n=bn,
                block_k=bk,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                use_bf16=cfg.use_bf16,
                split_k=cfg.split_k,
                tpu_path=cfg.tpu_path,
                platform=cfg.platform,
                backend=cfg.backend,
            )
    return cfg


def _xla_candidate_cfgs(inv: Invocation[QuantizedMatmulConfig, Array], hardware: str) -> list[QuantizedMatmulConfig]:
    """Generate candidate configurations for autotuning XLA quantized matmul.

    Creates block size combinations based on matrix dimensions and hardware
    target, selecting nearby power-of-2 choices and ensuring alignment with
    quantization group size. Returns up to 6 candidates sorted by proximity
    to the actual matrix dimensions.

    Args:
        inv: The kernel invocation containing input tensors and kwargs.
        hardware: Hardware target string ("cpu", "tpu", or "gpu").

    Returns:
        List of up to 6 QuantizedMatmulConfig candidates for XLA autotuning.
    """
    mode = str(inv.kwargs.get("mode", "affine"))
    group_size, bits = _resolve_qparams(mode, inv.kwargs.get("group_size"), inv.kwargs.get("bits"))
    M, K, N, transpose = _infer_mkn(inv, group_size)
    value_alignment = _bit_aligned_values(bits)
    align = _lcm(group_size, value_alignment)

    bm_choices, bn_choices, bk_choices = _xla_choices(hardware)
    base_m = _nearest_choices(M, bm_choices, count=1)[0]
    base_n = _nearest_choices(N, bn_choices, count=1)[0]
    base_k = _nearest_choices(K, bk_choices, count=1)[0]

    bm_opts = _expand_choices(base_m, bm_choices)
    bn_opts = _expand_choices(base_n, bn_choices)
    bk_opts = _expand_choices(base_k, bk_choices)

    if transpose:
        bk_opts = _ensure_aligned(bk_opts, align, max(bk_choices))
    else:
        bn_opts = _ensure_aligned(bn_opts, align, max(bn_choices))

    use_bf16 = False if hardware == "cpu" else _prefer_bf16(_inv_arg(inv, "x", 0))
    configs = []
    for bm in bm_opts:
        for bn in bn_opts:
            for bk in bk_opts:
                configs.append(
                    QuantizedMatmulConfig(
                        block_m=bm,
                        block_n=bn,
                        block_k=bk,
                        num_warps=4,
                        num_stages=3,
                        use_bf16=use_bf16,
                        split_k=None,
                        platform="xla",
                        backend="any",
                    )
                )

    def _score(cfg: QuantizedMatmulConfig) -> int:
        """Score a config by Manhattan distance from actual matrix dimensions."""
        return abs(cfg.block_m - M) + abs(cfg.block_n - N) + abs(cfg.block_k - K)

    configs.sort(key=_score)
    return configs[:6]


def _xla_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array], hardware: str) -> QuantizedMatmulConfig:
    """Generate a heuristic configuration for XLA quantized matmul.

    Returns the top-ranked candidate from _xla_candidate_cfgs, or a
    minimal fallback configuration if no candidates are generated.

    Args:
        inv: The kernel invocation containing input tensors and kwargs.
        hardware: Hardware target string ("cpu", "tpu", or "gpu").

    Returns:
        A QuantizedMatmulConfig tuned for XLA execution.
    """
    candidates = _xla_candidate_cfgs(inv, hardware)
    return candidates[0] if candidates else QuantizedMatmulConfig(platform="xla", backend="any")


def _triton_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
    """Generate a heuristic configuration for Triton GPU quantized matmul.

    Selects block sizes, warp counts, pipeline stages, and split-K factor
    based on matrix dimensions and quantization parameters. Includes
    shared memory usage estimation to avoid exceeding GPU limits.

    Args:
        inv: The kernel invocation containing input tensors and kwargs.

    Returns:
        A QuantizedMatmulConfig tuned for Triton GPU execution.
    """
    mode = str(inv.kwargs.get("mode", "affine"))
    group_size, bits = _resolve_qparams(mode, inv.kwargs.get("group_size"), inv.kwargs.get("bits"))
    M, K, N, _ = _infer_mkn(inv, group_size)

    block_m = 128 if M >= 128 else 64
    block_n = 128 if N >= 128 else 64
    if bits == 8 or group_size >= 128:
        block_k = 32
        num_stages = 2
    else:
        block_k = 64 if K >= 1024 else 32
        num_stages = 3 if block_k >= 64 else 2
    num_warps = 4 if (block_m >= 128 and block_n >= 128) else 2

    elem_size = 2
    smem_limit = 96 * 1024
    smem = (block_m * block_k + block_k * block_n) * elem_size * num_stages
    if smem > smem_limit:
        num_stages = 2
        smem = (block_m * block_k + block_k * block_n) * elem_size * num_stages
    if smem > smem_limit:
        block_k = 32
        num_stages = 2
        smem = (block_m * block_k + block_k * block_n) * elem_size * num_stages
    if smem > smem_limit:
        block_m = 64
        block_n = 64
        num_warps = 2
    split_k = _pick_split_k(M, K, block_k)

    return QuantizedMatmulConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        use_bf16=_prefer_bf16(_inv_arg(inv, "x", 0)),
        split_k=split_k,
        platform="triton",
        backend="gpu",
    )


def _cuda_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
    """Generate a heuristic configuration for CUDA custom-call quantized matmul.

    Uses fixed 128x128x64 block sizes with 4 warps and 2 pipeline stages,
    which are well-suited for CUDA's custom-call codepath.

    Args:
        inv: The kernel invocation containing input tensors and kwargs.

    Returns:
        A QuantizedMatmulConfig tuned for CUDA custom-call execution.
    """
    mode = str(inv.kwargs.get("mode", "affine"))
    group_size, _bits = _resolve_qparams(mode, inv.kwargs.get("group_size"), inv.kwargs.get("bits"))
    M, _K, _N, _transpose = _infer_mkn(inv, group_size)
    block_n = 256 if M < 16 else 128
    block_k = 128 if M < 16 else 64
    return QuantizedMatmulConfig(
        block_m=128,
        block_n=block_n,
        block_k=block_k,
        num_warps=4,
        num_stages=2,
        use_bf16=_prefer_bf16(_inv_arg(inv, "x", 0)),
        split_k=None,
        platform="cuda",
        backend="gpu",
    )


def _tilelang_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
    """Generate a launch-safe heuristic configuration for TileLang QMM."""
    mode = str(inv.kwargs.get("mode", "affine"))
    group_size, bits = _resolve_qparams(mode, inv.kwargs.get("group_size"), inv.kwargs.get("bits"))
    M, _K, N, _ = _infer_mkn(inv, group_size)
    block_m = 128 if M >= 128 else 64 if M >= 64 else 32 if M >= 16 else 16
    block_m_tall = 256 if M >= 256 else block_m
    num_stages = 2
    if mode == "affine":
        block_n = 64 if M < 16 and N >= 64 else 128 if N >= 128 else 64
        block_k = 64 if M < 16 else 128
        num_warps = 4
        num_stages = 3 if bits == 8 and M >= 512 else 2
        if bits == 8 and M >= 512:
            block_n = 64 if N >= 64 else 32
            num_warps = 8
            num_stages = 4
    elif mode == "nf4":
        block_m = block_m_tall
        block_n = 32
        block_k = 64
        num_warps = 4
    elif mode == "mxfp4":
        block_m = block_m_tall
        block_n = 64 if N >= 64 else 32
        block_k = 64
        num_warps = 4
    elif mode in {"nvfp4", "nvfp8"}:
        block_m = block_m_tall
        block_n = 64 if N >= 64 else 32
        block_k = 64
        num_warps = 4
    elif mode == "mxfp8":
        block_n = 64 if N >= 64 else 32
        block_k = 128
        num_warps = 4
    else:
        block_n = 64 if N >= 64 else 32
        block_k = 64
        num_warps = 4
    return QuantizedMatmulConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        use_bf16=_prefer_bf16(_inv_arg(inv, "x", 0)),
        split_k=None,
        platform="tilelang",
        backend="gpu",
    )


def _cute_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
    """Generate a heuristic configuration for CuTe DSL quantized matmul.

    Uses fixed 128x128x64 block sizes with 4 warps and 2 pipeline stages,
    matching the CuTe DSL kernel's default tile shape.

    Args:
        inv: The kernel invocation containing input tensors and kwargs.

    Returns:
        A QuantizedMatmulConfig tuned for CuTe DSL execution.
    """
    return QuantizedMatmulConfig(
        block_m=128,
        block_n=128,
        block_k=64,
        num_warps=4,
        num_stages=2,
        use_bf16=_prefer_bf16(_inv_arg(inv, "x", 0)),
        split_k=None,
        platform="cute",
        backend="gpu",
    )


class QuantizedMatmul(Kernel[QuantizedMatmulConfig, Array]):
    """Quantized matrix multiplication kernel with configurable tiling and backend selection.

    This kernel performs matrix multiplication between a dense input matrix and a
    quantized weight matrix, supporting explicit quantization modes:
    affine, nf4, mxfp4, mxfp8, nvfp4, and nvfp8.

    The kernel automatically selects the optimal backend (Triton or XLA) based on
    the target platform and input characteristics. It supports autotuning for
    optimal block sizes and pipeline configurations.

    Attributes:
        op_id: Operation identifier ("quantized_matmul").

    Example:
        >>> kernel = QuantizedMatmul()
        >>> cfg = QuantizedMatmulConfig(block_m=128, block_n=128, block_k=64)
        >>> output = kernel.run(x, w_q, scales, zeros, cfg=cfg)
    """

    version = "16"

    def __init__(self) -> None:
        """Initialize the quantized matmul kernel."""
        super().__init__(op_id="quantized_matmul")

    def _resolve_inv_platform(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> Platform:
        """Resolve the target platform from invocation parameters.

        Args:
            inv: The kernel invocation containing configuration and arguments.

        Returns:
            The resolved Platform enum value.
        """
        platform = inv.kwargs.get("platform", None)
        if platform is None and inv.override_cfg is not None:
            platform = inv.override_cfg.platform
        return detect_platform(self.op_id, platform if platform is not None else "auto")

    def get_impl(self, cfg: QuantizedMatmulConfig):
        """Get the kernel implementation for the given configuration.

        Args:
            cfg: Kernel configuration specifying platform and backend.

        Returns:
            The registered kernel implementation function.
        """
        try:
            backend_name = jax.default_backend()
        except Exception:
            backend_name = "cpu"
        platform = detect_platform(
            self.op_id,
            cfg.platform,
            prefer_pallas=backend_name == "tpu",
            prefer_cuda=backend_name in ("gpu", "cuda"),
        )
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(
        self,
        x: Float[Array, "m k"],
        w: Array,
        scales: Array,
        zeros: Array | None = None,
        transpose: bool = False,
        group_size: int | None = None,
        bits: int | None = None,
        mode: QuantizationMode = "affine",
        axis: QuantizationAxis | None = None,
        gemv_mode: GemvMode = "auto",
        revsplit_k: RevSplitKMode = "auto",
        revsplit_k_parts: int | None = None,
        allow_dense_fallback: bool = True,
        _resolved_platform: str | None = None,
        platform: Literal["triton", "pallas", "cuda", "cute", "tilelang", "xla", "auto"] | None = None,
        *,
        cfg: QuantizedMatmulConfig,
    ) -> Float[Array, "m n"]:
        """Execute quantized matmul with the selected backend.

        Performs the computation:
            - ``output = x @ dequantize(w, scales, zeros)``
            - ``output = x @ dequantize(w, scales, zeros).T`` when ``transpose=True``

        Args:
            x: Input matrix of shape ``(M, K)`` in float dtype.
            w: Packed uint32 weights produced by ``quantize()``. Shape depends on
                ``transpose`` and ``bits`` settings.
            scales: Per-group scale factors. Shape is ``(N, K//group_size)`` for
                ``transpose=True`` or ``(K, N//group_size)`` for ``transpose=False``.
            zeros: Per-group affine zero-points. Required for affine mode; must be
                ``None`` for all non-affine modes.
            transpose: If ``True``, compute ``x @ dequantize(w).T``
                (weights stored in KxN layout). If ``False``, compute
                ``x @ dequantize(w)`` (weights in transposed layout).
            group_size: Quantization group size. ``None`` uses the mode default.
            bits: Bit-width used in quantization. Honoured for affine
                (``1`` through ``8``); ignored for
                nf4/mxfp4/mxfp8/nvfp4/nvfp8.
            mode: Quantization mode string. One of
                ``{"affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"}``.
            axis: Optional quantization axis alias. ``"row"`` maps to
                ``transpose=False``; ``"col"`` maps to ``transpose=True``.
            gemv_mode: GEMV specialisation mode (``"auto"``, ``"on"``, ``"off"``).
                ``"auto"`` enables the GEMV kernel when M == 1 and mode supports it.
            revsplit_k: Reverse split-K reduction mode
                (``"auto"``, ``"on"``, ``"off"``). Splits the K reduction for
                improved parallelism on tall-skinny outputs.
            revsplit_k_parts: Number of parts for reverse split-K. ``None`` means
                the mode default is used.
            allow_dense_fallback: When dispatching to XLA or Pallas, whether the
                implementation may fall back to a dequantize+matmul decomposition
                when blocked-fusion preconditions are not met.
            platform: Platform override (``triton``/``pallas``/``cuda``/``cute``/
                ``xla``/``auto``).
            cfg: Kernel configuration with block sizes and settings.

        Returns:
            Matrix multiplication result of shape ``(M, N)``. The CUDA backend
            preserves the dtype of ``x``; all other backends return float32.

        Note:
            For best Triton performance, prepack weights in KxN layout using
            ``prepack_quantized_weights()`` and call with ``transpose=False``.
            For affine mode, backend wrappers convert ``zeros`` to internal
            additive offsets immediately before kernel launch.
        """
        _ = _resolved_platform
        if platform is not None:
            cfg = QuantizedMatmulConfig(
                block_m=cfg.block_m,
                block_n=cfg.block_n,
                block_k=cfg.block_k,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                use_bf16=cfg.use_bf16,
                split_k=cfg.split_k,
                tpu_path=cfg.tpu_path,
                platform=platform,
                backend=cfg.backend,
            )

        resolved_platform = _resolved_platform
        if resolved_platform is None:
            try:
                backend_name = jax.default_backend()
            except Exception:
                backend_name = "cpu"
            resolved_platform = detect_platform(
                self.op_id,
                cfg.platform,
                prefer_pallas=backend_name == "tpu",
                prefer_cuda=backend_name in ("gpu", "cuda"),
            ).value

        if (
            resolved_platform == Platform.PALLAS.value
            and bool(transpose) is False
            and group_size is not None
            and bits is not None
        ):
            cfg = _normalize_pallas_tpu_packed_cfg_forward(
                x=x,
                w=w,
                scales=scales,
                group_size=int(group_size),
                bits=int(bits),
                cfg=cfg,
            )

        impl = self.get_impl(cfg)
        impl_kwargs = dict(
            transpose=transpose,
            group_size=group_size,
            bits=bits,
            mode=mode,
            axis=axis,
            gemv_mode=gemv_mode,
            revsplit_k=revsplit_k,
            revsplit_k_parts=revsplit_k_parts,
            allow_dense_fallback=allow_dense_fallback,
            block_m=cfg.block_m,
            block_n=cfg.block_n,
            block_k=cfg.block_k,
            use_bf16=cfg.use_bf16,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
            split_k=cfg.split_k,
        )
        if resolved_platform == Platform.PALLAS.value and cfg.tpu_path is not None:
            impl_kwargs["tpu_path"] = cfg.tpu_path

        return impl(
            x,
            w,
            scales,
            zeros,
            **impl_kwargs,
        )

    def heuristic_cfg(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
        """Return default heuristic configuration for any platform.

        Args:
            inv: The kernel invocation (unused for default heuristics).

        Returns:
            A default QuantizedMatmulConfig with balanced block sizes.
        """
        return _xla_heuristic_cfg(inv, "cpu")

    def candidate_cfgs(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> list[QuantizedMatmulConfig]:
        """Return candidate configurations for autotuning.

        Args:
            inv: The kernel invocation (unused for default candidates).

        Returns:
            List of QuantizedMatmulConfig candidates to try during autotuning.
        """
        return _xla_candidate_cfgs(inv, "cpu")

    def heuristic_cfg_gpu(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
        """Return heuristic configuration optimized for GPU.

        Args:
            inv: The kernel invocation for platform resolution.

        Returns:
            A QuantizedMatmulConfig optimized for GPU execution.
        """
        resolved = self._resolve_inv_platform(inv)
        if resolved == Platform.TRITON:
            return _triton_heuristic_cfg(inv)
        if resolved == Platform.CUDA:
            return _cuda_heuristic_cfg(inv)
        if resolved == Platform.TILELANG:
            return _tilelang_heuristic_cfg(inv)
        if resolved == Platform.CUTE:
            return _cute_heuristic_cfg(inv)
        return _xla_heuristic_cfg(inv, "gpu")

    def heuristic_cfg_cpu(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
        """Return heuristic configuration optimized for CPU.

        Args:
            inv: The kernel invocation (unused for CPU heuristics).

        Returns:
            A QuantizedMatmulConfig optimized for CPU execution.
        """
        return _xla_heuristic_cfg(inv, "cpu")

    def heuristic_cfg_tpu(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
        """Return heuristic configuration optimized for TPU.

        Args:
            inv: The kernel invocation (unused for TPU heuristics).

        Returns:
            A QuantizedMatmulConfig optimized for TPU execution.
        """
        resolved = self._resolve_inv_platform(inv)
        if resolved == Platform.PALLAS:
            return _pallas_tpu_heuristic_cfg(inv)
        return _xla_heuristic_cfg(inv, "tpu")

    def candidate_cfgs_gpu(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> list[QuantizedMatmulConfig]:
        """Return GPU-specific candidate configurations for autotuning.

        Args:
            inv: The kernel invocation for platform resolution.

        Returns:
            List of QuantizedMatmulConfig candidates optimized for GPU.
        """
        requested = inv.kwargs.get("platform", None)
        if requested is None and inv.override_cfg is not None:
            requested = inv.override_cfg.platform
        if requested in (None, "auto"):
            candidates: list[QuantizedMatmulConfig] = []
            for platform in (Platform.CUDA, Platform.CUTE, Platform.TRITON, Platform.TILELANG):
                candidates.extend(self._candidate_cfgs_gpu_for_platform(inv, platform))
            candidates.extend(_xla_candidate_cfgs(inv, "gpu")[:4])
            return candidates
        return self._candidate_cfgs_gpu_for_platform(inv, self._resolve_inv_platform(inv))

    def _candidate_cfgs_gpu_for_platform(
        self,
        inv: Invocation[QuantizedMatmulConfig, Array],
        resolved: Platform,
    ) -> list[QuantizedMatmulConfig]:
        """Return candidate configurations for a concrete GPU platform."""
        if resolved == Platform.TRITON:
            mode = str(inv.kwargs.get("mode", "affine"))
            group_size, bits = _resolve_qparams(mode, inv.kwargs.get("group_size"), inv.kwargs.get("bits"))
            M, K, N, _ = _infer_mkn(inv, group_size)
            use_bf16 = _prefer_bf16(_inv_arg(inv, "x", 0))
            if bits == 8:
                bk_choices = (64, 128)
            elif group_size >= 128:
                bk_choices = (32, 64)
            else:
                bk_choices = (32, 64, 128)
            bm_choices = _nearest_choices(M, (32, 64, 128, 256), 3)
            bn_choices = _nearest_choices(N, (64, 128, 256, 512), 3)
            candidates: list[QuantizedMatmulConfig] = []
            seen: set[tuple[int, int, int, int, int, int]] = set()
            for bm in bm_choices:
                for bn in bn_choices:
                    for bk in bk_choices:
                        for stages in (1, 2) if bk >= 128 else (2, 3):
                            smem = (bm * bk + bk * bn) * 2 * stages
                            if smem > 96 * 1024:
                                continue
                            warps = 8 if (bm >= 128 and bn >= 256) else 4 if (bm >= 128 or bn >= 128) else 2
                            split_k = _pick_split_k(M, K, bk)
                            key = (bm, bn, bk, warps, stages, split_k)
                            if key in seen:
                                continue
                            seen.add(key)
                            candidates.append(
                                QuantizedMatmulConfig(
                                    block_m=bm,
                                    block_n=bn,
                                    block_k=bk,
                                    num_warps=warps,
                                    num_stages=stages,
                                    use_bf16=use_bf16,
                                    split_k=split_k,
                                    platform="triton",
                                    backend="gpu",
                                )
                            )
            return candidates[:16] or [_triton_heuristic_cfg(inv)]
        if resolved == Platform.CUDA:
            base = _cuda_heuristic_cfg(inv)
            return [
                QuantizedMatmulConfig(
                    block_m=bm,
                    block_n=bn,
                    block_k=bk,
                    num_warps=warps,
                    num_stages=stages,
                    use_bf16=base.use_bf16,
                    split_k=None,
                    platform="cuda",
                    backend="gpu",
                )
                for bm, bn, bk, warps, stages in (
                    (128, 256, 128, 4, 2),
                    (128, 128, 128, 4, 2),
                    (128, 128, 64, 4, 2),
                    (128, 256, 64, 4, 2),
                    (64, 128, 64, 4, 2),
                )
            ]
        if resolved == Platform.CUTE:
            base = _cute_heuristic_cfg(inv)
            return [
                QuantizedMatmulConfig(
                    block_m=bm,
                    block_n=bn,
                    block_k=bk,
                    num_warps=base.num_warps,
                    num_stages=base.num_stages,
                    use_bf16=base.use_bf16,
                    split_k=None,
                    platform="cute",
                    backend="gpu",
                )
                for bm, bn, bk in (
                    (128, 128, 64),
                    (128, 64, 64),
                    (64, 128, 64),
                    (64, 64, 32),
                    (128, 128, 32),
                )
            ]
        if resolved == Platform.TILELANG:
            mode = str(inv.kwargs.get("mode", "affine"))
            group_size, bits = _resolve_qparams(mode, inv.kwargs.get("group_size"), inv.kwargs.get("bits"))
            M, _K, N, _ = _infer_mkn(inv, group_size)
            block_m = 128 if M >= 128 else 64 if M >= 64 else 32 if M >= 16 else 16
            block_m_tall = 256 if M >= 256 else block_m
            block_n_hi = 128 if N >= 128 else 64
            block_n_lo = 64 if N >= 64 else 32
            block_n_micro = 32
            use_bf16 = _prefer_bf16(_inv_arg(inv, "x", 0))
            if mode == "affine":
                if M < 16:
                    shapes = (
                        (block_m, block_n_lo, 64, 4),
                        (block_m, block_n_lo, 128, 4),
                        (block_m, block_n_hi, 64, 4),
                        (block_m, block_n_hi, 128, 4),
                    )
                else:
                    if bits == 8 and M >= 512:
                        shapes = (
                            (block_m, block_n_lo, 128, 8),
                            (block_m_tall, block_n_lo, 64, 4),
                            (block_m, block_n_hi, 128, 4),
                            (block_m, block_n_hi, 64, 4),
                            (block_m, block_n_lo, 64, 4),
                        )
                    else:
                        shapes = (
                            (block_m, block_n_hi, 128, 4),
                            (block_m, block_n_hi, 64, 4),
                            (block_m, block_n_lo, 64, 4),
                            (block_m, block_n_hi, 32, 4),
                        )
            elif mode == "nf4":
                shapes = (
                    (block_m_tall, block_n_micro, 64, 4),
                    (block_m_tall, block_n_lo, 64, 4),
                    (64 if block_m >= 64 else block_m, block_n_lo, 64, 4),
                    (block_m, block_n_lo, 32, 4),
                )
            elif mode == "mxfp4":
                shapes = (
                    (block_m_tall, block_n_lo, 64, 4),
                    (block_m_tall, block_n_lo, 128, 4),
                    (block_m, block_n_hi, 64, 4),
                    (block_m, block_n_lo, 32, 4),
                    (block_m, block_n_lo, 64, 4),
                )
            elif mode == "mxfp8":
                shapes = (
                    (block_m_tall, block_n_lo, 128, 4),
                    (block_m, block_n_lo, 64, 4),
                    (block_m, block_n_hi, 64, 4),
                    (block_m_tall, block_n_lo, 64, 4),
                    (block_m, block_n_lo, 32, 4),
                )
            elif mode == "nvfp4":
                shapes = (
                    (block_m_tall, block_n_lo, 64, 4),
                    (block_m, block_n_lo, 64, 4),
                    (block_m, block_n_lo, 32, 4),
                    (block_m, block_n_hi, 64, 4),
                )
            else:
                shapes = (
                    (block_m_tall, block_n_lo, 64, 4),
                    (block_m, block_n_lo, 32, 4),
                    (block_m, block_n_hi, 64, 4),
                    (block_m, block_n_lo, 64, 4),
                )

            def _tilelang_num_stages(bk: int, warps: int) -> int:
                if mode == "affine" and bits == 8 and M >= 512:
                    return 4 if warps >= 8 else 3 if bk >= 128 else 2
                return 2

            return [
                QuantizedMatmulConfig(
                    block_m=bm,
                    block_n=bn,
                    block_k=bk,
                    num_warps=warps,
                    num_stages=_tilelang_num_stages(bk, warps),
                    use_bf16=use_bf16,
                    split_k=None,
                    platform="tilelang",
                    backend="gpu",
                )
                for bm, bn, bk, warps in shapes
            ]
        return _xla_candidate_cfgs(inv, "gpu")

    def candidate_cfgs_cpu(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> list[QuantizedMatmulConfig]:
        """Return CPU-specific candidate configurations for autotuning.

        Args:
            inv: The kernel invocation (unused for CPU candidates).

        Returns:
            List of QuantizedMatmulConfig candidates optimized for CPU.
        """
        return _xla_candidate_cfgs(inv, "cpu")

    def candidate_cfgs_tpu(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> list[QuantizedMatmulConfig]:
        """Return TPU-specific candidate configurations for autotuning.

        Args:
            inv: The kernel invocation (unused for TPU candidates).

        Returns:
            List of QuantizedMatmulConfig candidates optimized for TPU.
        """
        requested = inv.kwargs.get("platform", None)
        if requested is None and inv.override_cfg is not None:
            requested = inv.override_cfg.platform
        if requested in (None, "auto"):
            return [*_pallas_tpu_candidate_cfgs(inv), *_xla_candidate_cfgs(inv, "tpu")]
        resolved = self._resolve_inv_platform(inv)
        if resolved == Platform.PALLAS:
            return _pallas_tpu_candidate_cfgs(inv)
        return _xla_candidate_cfgs(inv, "tpu")

    candidate_cfgs_shard_map_gpu = candidate_cfgs_gpu


_quantized_matmul_executor: Executor[QuantizedMatmulConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=False,
        ),
        tuner=Tuner(
            warmup=_parse_nonnegative_int_env("EJKERNEL_QMM_AUTOTUNE_WARMUP", 5),
            iters=_parse_positive_int_env("EJKERNEL_QMM_AUTOTUNE_ITERS", 100),
        ),
        persistent=PersistentCache("quantized-matmul", cfg_type=QuantizedMatmulConfig),
    )
)


def _quantized_matmul_impl(
    x: Float[Array, "m k"],
    w: Array,
    scales: Array,
    zeros: Array | None = None,
    /,
    *,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    axis: QuantizationAxis | None = None,
    gemv_mode: GemvMode = "auto",
    revsplit_k: RevSplitKMode = "auto",
    revsplit_k_parts: int | None = None,
    allow_dense_fallback: bool = True,
    platform: Literal["triton", "pallas", "cuda", "cute", "tilelang", "xla", "auto"] | None = None,
    cfg: QuantizedMatmulConfig | None = None,
) -> Float[Array, "m n"]:
    """Execute quantized matrix multiplication with normalized qparams.

    Internal implementation that normalizes quantization parameters (mode,
    group_size, bits, axis, transpose), resolves the kernel family (GEMM vs
    GEMV vs revsplit-K), validates zeros requirements, and dispatches to the
    appropriate platform executor.

    Args:
        x: Input matrix of shape ``(M, K)`` in float dtype.
        w: Packed quantized weights.
        scales: Per-group scale factors.
        zeros: Per-group zero-points (required for affine mode, ``None`` otherwise).
        transpose: If ``True``, compute ``x @ dequantize(w).T``.
        group_size: Quantization group size.
        bits: Quantization bit-width.
        mode: Quantization mode string.
        axis: Resolved quantization axis (must be ``"row"`` or ``"col"``; see
            ``resolve_runtime_axis_and_transpose``). Callers must pass this
            explicitly — passing ``None`` raises ``ValueError``.
        gemv_mode: GEMV kernel selection mode (``"auto"``, ``"on"``, ``"off"``).
        revsplit_k: Reverse split-K mode (``"auto"``, ``"on"``, ``"off"``).
        revsplit_k_parts: Number of parts for reverse split-K.
        allow_dense_fallback: When dispatching to XLA or Pallas, controls
            whether the implementation may fall back to a separate
            dequantize+matmul when blocked-fusion preconditions are not met.
        platform: Platform override.
        cfg: Optional configuration override.

    Returns:
        Matrix multiplication result of shape ``(M, N)``.

    Raises:
        ValueError: If ``axis`` is ``None`` or not in ``{"row", "col"}``.
        ValueError: If ``axis``/``transpose`` are inconsistent with each other.
        ValueError: If affine mode is used without ``zeros``.
        ValueError: If a non-affine mode is used with a non-``None`` ``zeros``.
    """
    transpose = _static_bool(transpose, "transpose")
    if group_size is not None:
        group_size = _static_int(group_size, "group_size")
    if bits is not None:
        bits = _static_int(bits, "bits")
    mode, group_size, bits, _ = resolve_qparams(mode, group_size, bits)
    if axis is None:
        raise ValueError("_quantized_matmul_impl expects axis to be resolved (pass axis='row' or axis='col').")
    if axis not in {"row", "col"}:
        raise ValueError(f"_quantized_matmul_impl expected axis in {{'row','col'}}, got {axis!r}.")
    expected_transpose = axis == "col"
    if expected_transpose != bool(transpose):
        raise ValueError(
            "_quantized_matmul_impl received inconsistent axis/transpose: "
            f"axis={axis!r} requires transpose={expected_transpose}, got transpose={bool(transpose)}."
        )
    gemv_mode = normalize_gemv_mode(gemv_mode)
    revsplit_k = normalize_revsplitk_mode(revsplit_k)
    revsplit_k_parts = normalize_revsplitk_parts(revsplit_k_parts)
    allow_dense_fallback = _static_bool(allow_dense_fallback, "allow_dense_fallback")

    if int(x.shape[0]) == 1 and mode in {"mxfp4", "mxfp8"} and gemv_mode == "on":
        warnings.warn(
            "gemv_mode='on' with MX modes at M==1 is mapped to GEMM-SplitK for GemLite parity.",
            RuntimeWarning,
            stacklevel=2,
        )

    family, family_revsplit_parts = select_qmm_kernel_family(
        m=int(x.shape[0]),
        mode=mode,
        bits=bits,
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
    )
    if family == "gemv_revsplitk":
        revsplit_k_parts = family_revsplit_parts

    if mode == "affine" and zeros is None:
        raise ValueError("affine quantized_matmul requires `zeros`.")
    if mode != "affine" and zeros is not None:
        raise ValueError("zeros must be None for non-affine modes.")

    try:
        backend_name = jax.default_backend()
    except Exception:
        backend_name = "cpu"

    prefer_cuda_col_lowbit = (
        backend_name in ("gpu", "cuda")
        and axis == "col"
        and mode == "affine"
        and bits in {1, 2, 4, 8}
        and int(x.shape[0]) >= 512
    )
    prefer_tilelang = (
        backend_name in ("gpu", "cuda")
        and axis == "col"
        and mode == "affine"
        and bits in {1, 2, 4, 8}
        and int(x.shape[0]) >= 512
        and not prefer_cuda_col_lowbit
        and _tilelang_runtime_available()
    )
    prefer_cuda = backend_name in ("gpu", "cuda") and (axis != "col" or prefer_cuda_col_lowbit)
    prefer_triton = backend_name in ("gpu", "cuda") and axis == "col" and not prefer_cuda and not prefer_tilelang
    resolved = detect_platform(
        "quantized_matmul",
        platform if platform is not None else (cfg.platform if cfg is not None else "auto"),
        prefer_pallas=backend_name == "tpu",
        prefer_cuda=prefer_cuda,
        prefer_triton=prefer_triton,
        prefer_tilelang=prefer_tilelang,
    )
    dispatch_platform = resolved.value
    extra_kwargs = {}
    if resolved in (Platform.XLA, Platform.PALLAS):
        extra_kwargs["allow_dense_fallback"] = allow_dense_fallback

    if resolved in (Platform.TRITON, Platform.CUDA, Platform.CUTE, Platform.TILELANG):
        with policy_override(
            _quantized_matmul_executor.chooser,
            allow_autotune=False,
            cache_miss_fallback="heuristics",
        ):
            return _quantized_matmul_executor(
                QuantizedMatmul(),
                x=x,
                w=w,
                scales=scales,
                zeros=zeros,
                transpose=transpose,
                group_size=group_size,
                bits=bits,
                mode=mode,
                axis=axis,
                gemv_mode=gemv_mode,
                revsplit_k=revsplit_k,
                revsplit_k_parts=revsplit_k_parts,
                _resolved_platform=resolved.value,
                platform=dispatch_platform,
                _cfg=cfg,
                **extra_kwargs,
            )

    return _quantized_matmul_executor(
        QuantizedMatmul(),
        x=x,
        w=w,
        scales=scales,
        zeros=zeros,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        axis=axis,
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
        _resolved_platform=resolved.value,
        platform=dispatch_platform,
        _cfg=cfg,
        **extra_kwargs,
    )


def quantized_matmul(
    x: Float[Array, "m k"],
    w: Array,
    scales: Array,
    zeros: Array | None = None,
    /,
    *,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    axis: QuantizationAxis | None = None,
    gemv_mode: GemvMode = "auto",
    revsplit_k: RevSplitKMode = "auto",
    revsplit_k_parts: int | None = None,
    fuse: bool = True,
    strict_fuse: bool | None = None,
    tpu_path: Literal["packed", "hybrid", "predecode"] | None = None,
    allow_dense_fallback: bool | None = None,
    use_best_config: bool = False,
    platform: Literal["triton", "pallas", "cuda", "cute", "tilelang", "xla", "auto"] | None = None,
    cfg: QuantizedMatmulConfig | None = None,
) -> Float[Array, "m n"]:
    """Quantized matrix multiplication with fused dequantization and custom VJP.

    Performs output = x @ dequantize(w, scales, zeros) with automatic backend
    selection and a custom backward pass that dequantizes weights for the
    gradient computation. Supports affine, nf4, mxfp4, mxfp8, nvfp4, and
    nvfp8 quantization modes.

    TPU default strategy:
        When running on TPU with ``fuse=True`` and row-wise layout
        (``axis='row'`` / ``transpose=False``), this API first attempts a
        predecode-once dense matmul path by default. If that path is not
        applicable (e.g. traced quantized metadata), execution falls back to
        the fused backend dispatcher (typically TPU Pallas packed path).
        This behavior is controlled by environment variable
        ``EJKERNEL_QMM_TPU_DEFAULT_STRATEGY`` with supported values:
        ``"predecode_once"`` (default) and ``"packed"``.

    Args:
        x: Input matrix of shape (M, K) in float dtype.
        w: Packed uint32 weights produced by quantize().
        scales: Per-group scale factors.
        zeros: Per-group zero-points. Required for affine mode, must be
            None for non-affine modes.
        transpose: If True, compute x @ dequantize(w).T.
        group_size: Quantization group size. If None, uses mode default.
        bits: Quantization bit-width. Honored for affine ({1,2,3,4,5,6,7,8});
            ignored for nf4/mxfp4/mxfp8/nvfp4/nvfp8.
        mode: Quantization mode. One of
            {"affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"}.
        axis: Optional quantization axis convenience alias. "row" maps to
            transpose=False; "col" maps to transpose=True.
        gemv_mode: GEMV kernel selection mode ("auto", "on", "off").
        revsplit_k: Reverse split-K mode ("auto", "on", "off").
        revsplit_k_parts: Number of parts for reverse split-K.
        fuse: If True, run platform fused quantized kernels. If False, force
            reference path (dequantize then matmul) using XLA/JAX ops.
            On TPU row-wise fused calls, default behavior may route through
            predecode-once first (see TPU default strategy above).
        strict_fuse: If True, disallow dense dequantize+matmul fallbacks inside
            fused implementations (notably the XLA backend). When None, reads
            environment variable ``EJKERNEL_QMM_STRICT_FUSED``.
        tpu_path: Optional TPU strategy override for fused TPU execution.
            Accepted values:
            - ``"packed"``: force packed fused path dispatch.
            - ``"predecode"`` or ``"hybrid"``: force predecode-once attempt
              first, then fall back to packed fused dispatch when needed.
            This override applies per call and takes precedence over
            ``EJKERNEL_QMM_TPU_DEFAULT_STRATEGY``.
        allow_dense_fallback: Explicitly control whether fused implementations
            may fall back to dense dequantize+matmul. If None, defaults to
            ``not strict_fuse``.
        use_best_config: If True, apply ejkernel's built-in backend/mode/size
            policy table to fill runtime controls (fuse/platform/TPU strategy).
            For non-affine modes on TPU, this policy is tracer-aware:
            traced packed metadata prefers unfused XLA, while concrete packed
            metadata prefers predecode-once first.
            Explicit non-default values for ``strict_fuse``,
            ``allow_dense_fallback``, ``platform``, and ``tpu_path`` are
            preserved. ``fuse=True`` may still be downshifted to ``False`` by
            policy.
        platform: Platform override (triton/pallas/cuda/cute/xla/auto).
        cfg: Optional configuration override.

    Returns:
        Matrix multiplication result of shape (M, N).
    """
    mode_n, group_size_n, bits_n, _ = resolve_qparams(mode, group_size, bits)
    runtime_axis, runtime_transpose = resolve_runtime_axis_and_transpose(axis=axis, transpose=transpose)
    use_best_config_n = _static_bool(use_best_config, "use_best_config")
    weights_concrete = (
        not _is_tracer_value(w) and not _is_tracer_value(scales) and (zeros is None or not _is_tracer_value(zeros))
    )

    try:
        backend_name = jax.default_backend()
    except Exception:
        backend_name = "cpu"

    if use_best_config_n:
        policy = _lookup_best_qmm_policy(
            backend_name=backend_name,
            mode=mode_n,
            m_tokens=int(x.shape[0]),
            runtime_axis=runtime_axis,
            runtime_transpose=runtime_transpose,
            weights_concrete=weights_concrete,
        )
        if policy:
            if fuse is True and "fuse" in policy:
                fuse = bool(policy["fuse"])
            if strict_fuse is None and "strict_fuse" in policy:
                strict_fuse = bool(policy["strict_fuse"])
            if allow_dense_fallback is None and "allow_dense_fallback" in policy:
                allow_dense_fallback = bool(policy["allow_dense_fallback"])
            if platform in (None, "auto") and "platform" in policy:
                policy_platform = str(policy["platform"]).strip().lower()
                if policy_platform in {"triton", "pallas", "cuda", "cute", "tilelang", "xla", "auto"}:
                    platform = policy_platform
            if tpu_path is None and "tpu_path" in policy:
                policy_tpu_path = str(policy["tpu_path"]).strip().lower()
                if policy_tpu_path in {"packed", "hybrid", "predecode"}:
                    tpu_path = policy_tpu_path

    if strict_fuse is None:
        env = os.getenv("EJKERNEL_QMM_STRICT_FUSED", "").strip().lower()
        strict_fuse = env in {"1", "true", "on", "yes"}
    strict_fuse_n = _static_bool(strict_fuse, "strict_fuse")
    if allow_dense_fallback is None:
        allow_dense_fallback_n = not strict_fuse_n
    else:
        allow_dense_fallback_n = _static_bool(allow_dense_fallback, "allow_dense_fallback")
        if strict_fuse_n and allow_dense_fallback_n:
            raise ValueError("allow_dense_fallback=True is incompatible with strict_fuse=True.")

    fuse = _static_bool(fuse, "fuse")
    if strict_fuse_n and not fuse:
        raise ValueError("strict_fuse=True requires fuse=True.")

    tpu_strategy_override: Literal["predecode_once", "packed"] | None = None
    if tpu_path is not None:
        tpu_path_n = str(tpu_path).strip().lower()
        if tpu_path_n not in {"hybrid", "packed", "predecode"}:
            raise ValueError(f"tpu_path must be one of {{'hybrid','packed','predecode'}}, got {tpu_path!r}.")
        tpu_strategy_override = "predecode_once" if tpu_path_n in {"hybrid", "predecode"} else "packed"
        if tpu_path_n == "packed":
            if cfg is None:
                cfg = QuantizedMatmulConfig(tpu_path="packed")
            else:
                cfg = QuantizedMatmulConfig(
                    block_m=cfg.block_m,
                    block_n=cfg.block_n,
                    block_k=cfg.block_k,
                    num_warps=cfg.num_warps,
                    num_stages=cfg.num_stages,
                    use_bf16=cfg.use_bf16,
                    split_k=cfg.split_k,
                    tpu_path="packed",
                    platform=cfg.platform,
                    backend=cfg.backend,
                )

    if fuse and backend_name == "mps":
        msg = "fuse=True on MPS currently falls back to reference dequantize+matmul for stability."
        if strict_fuse_n:
            raise ValueError(msg)
        warnings.warn(
            msg,
            RuntimeWarning,
            stacklevel=2,
        )
        fuse = False
    use_tpu_predecode_once_default = _should_try_tpu_predecode_once_default(
        fuse=fuse,
        backend_name=backend_name,
        runtime_axis=runtime_axis,
        runtime_transpose=runtime_transpose,
        platform=platform,
        strategy_override=tpu_strategy_override,
    )
    validate_packed_quantized_matmul_layout(
        x,
        w,
        scales,
        zeros,
        mode=mode_n,
        group_size=group_size_n,
        bits=bits_n,
        axis=runtime_axis,
        transpose=runtime_transpose,
    )

    def _inner(xi, wi, si, zi):
        """Dispatch to _quantized_matmul_impl with captured quantization parameters."""
        if not fuse:
            from ejkernel.quantization._quants.quantizations import quantized_matmul as dense_quantized_matmul

            return dense_quantized_matmul(
                xi,
                wi,
                si,
                zi,
                transpose=runtime_transpose,
                group_size=group_size,
                bits=bits,
                mode=mode,
                axis=runtime_axis,
            )

        if use_tpu_predecode_once_default:
            out = _maybe_tpu_predecode_once_matmul(
                xi,
                wi,
                si,
                zi,
                mode=mode_n,
                group_size=group_size_n,
                bits=bits_n,
            )
            if out is not None:
                return out

        return _quantized_matmul_impl(
            xi,
            wi,
            si,
            zi,
            transpose=runtime_transpose,
            group_size=group_size,
            bits=bits,
            mode=mode,
            axis=runtime_axis,
            gemv_mode=gemv_mode,
            revsplit_k=revsplit_k,
            revsplit_k_parts=revsplit_k_parts,
            allow_dense_fallback=allow_dense_fallback_n,
            platform=platform,
            cfg=cfg,
        )

    @jax.custom_vjp
    def _inner_vjp(xi, wi, si, zi):
        """Forward pass wrapper decorated with custom_vjp for backward compatibility."""
        return _inner(xi, wi, si, zi)

    def _inner_fwd(xi, wi, si, zi):
        """Custom VJP forward: compute output and save residuals (w, scales, zeros)."""
        y = _inner(xi, wi, si, zi)
        return y, (wi, si, zi)

    def _inner_bwd(res, g):
        """Custom VJP backward: dequantize weights and compute grad_x (grad_w/scales/zeros are zero)."""
        wi, si, zi = res
        from ejkernel.quantization._quants.quantizations import dequantize

        dequant_axis: QuantizationAxis = "col" if runtime_transpose else "row"
        w_f = dequantize(
            wi,
            si,
            zi,
            group_size=group_size,
            bits=bits,
            mode=mode,
            axis=dequant_axis,
        )
        if runtime_transpose:
            grad_x = g @ w_f
        else:
            grad_x = g @ w_f.T

        grad_w = jnp.zeros_like(wi)
        grad_scales = jnp.zeros_like(si)
        grad_zeros = jnp.zeros_like(zi) if zi is not None else None
        return grad_x, grad_w, grad_scales, grad_zeros

    _inner_vjp.defvjp(_inner_fwd, _inner_bwd)

    return _inner_vjp(x, w, scales, zeros)
