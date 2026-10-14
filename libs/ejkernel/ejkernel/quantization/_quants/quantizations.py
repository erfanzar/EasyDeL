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

"""Quantization functions for weight compression.

Supported quantization modes:
- ``affine`` with bits in ``{1, 2, 3, 4, 5, 6, 7, 8}``
- ``nf4`` (4-bit normal-float codebook)
- ``mxfp4`` / ``mxfp8``
- ``nvfp4`` / ``nvfp8``

Quantization axis is explicit via ``axis``:
- ``axis='row'``: group over output channels (logical weight rows)
- ``axis='col'``: group over input channels (logical weight cols)
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from threading import Lock

import jax
from jax import numpy as jnp

from ejkernel.callib._ejit import ejit

from .._utils.bitpack import _pack_bits, _unpack_bits
from .._utils.fp_tables import (
    _decode_e2m1_codes,
    _decode_e4m3_codes,
    _get_e2m1_max,
    _get_e2m1_table,
    _get_e2m1_threshold_map,
    _get_e4m3_max,
    _get_e4m3_q_threshold_map,
    _get_e4m3_table,
    _get_e4m3_table_q,
    _get_nf4_table,
    _get_nf4_threshold_map,
)
from .._utils.grouping import _quantize_to_codebook, _reshape_groups
from .._utils.qparams import (
    QuantizationAxis,
    QuantizationMode,
    normalize_axis,
    resolve_prepack_axis,
    resolve_qparams,
    resolve_runtime_axis_and_transpose,
    validate_packed_quantized_matmul_layout,
)
from ..runtime import QuantRuntimeConfig, resolve_runtime_config

_AUTOTUNE_CACHE_LOCK = Lock()
_AUTOTUNE_QUANT_CFG_CACHE: dict[tuple, QuantRuntimeConfig] = {}
_AUTOTUNE_DEQUANT_CFG_CACHE: dict[tuple, QuantRuntimeConfig] = {}


def clear_runtime_autotune_cache() -> None:
    """Clear cached runtime autotune decisions for quantize/dequantize."""
    with _AUTOTUNE_CACHE_LOCK:
        _AUTOTUNE_QUANT_CFG_CACHE.clear()
        _AUTOTUNE_DEQUANT_CFG_CACHE.clear()


def runtime_autotune_cache_sizes() -> tuple[int, int]:
    """Return ``(quantize_cells, dequantize_cells)`` cached by runtime autotune."""
    with _AUTOTUNE_CACHE_LOCK:
        return len(_AUTOTUNE_QUANT_CFG_CACHE), len(_AUTOTUNE_DEQUANT_CFG_CACHE)


def _autotune_enabled() -> bool:
    """Check whether runtime autotuning is enabled."""
    flag = os.getenv("EJKERNEL_QRUNTIME_AUTOTUNE", "1").strip().lower()
    return flag not in {"0", "false", "off", "no"}


def _is_tracing_tree(tree) -> bool:
    """Return True if any leaf in *tree* is a JAX tracer."""
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, jax.core.Tracer):
            return True
    return False


def _block_tree(tree) -> None:
    """Block on all array leaves in *tree*."""
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _bench_ms(
    fn,
    args: tuple,
    *,
    warmup: int | None = None,
    iters: int | None = None,
) -> float:
    """Measure median runtime in milliseconds for a jitted callable."""
    if warmup is None:
        warmup = max(1, int(os.getenv("EJKERNEL_QRUNTIME_AUTOTUNE_WARMUP", "2")))
    if iters is None:
        iters = max(1, int(os.getenv("EJKERNEL_QRUNTIME_AUTOTUNE_ITERS", "4")))
    out = fn(*args)
    _block_tree(out)
    for _ in range(warmup):
        out = fn(*args)
        _block_tree(out)

    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        _block_tree(out)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return float(samples[len(samples) // 2])


def _autotune_min_gain() -> float:
    """Return minimum relative win required to replace the baseline config."""
    return max(0.0, float(os.getenv("EJKERNEL_QRUNTIME_AUTOTUNE_MIN_GAIN", "0.03")))


def _dedupe_cfgs(candidates: list[QuantRuntimeConfig]) -> list[QuantRuntimeConfig]:
    """Deduplicate candidate runtime configs while preserving order."""
    seen: set[tuple] = set()
    out: list[QuantRuntimeConfig] = []
    for cfg in candidates:
        sig = (
            cfg.enable_u4_u8_fastpath,
            cfg.enable_threshold_codebook,
            cfg.enable_parity_fallback,
            cfg.strict_shape_alignment,
            cfg.prefer_compute_dtype,
            cfg.affine_metadata_dtype,
            cfg.dequant_output_dtype,
            cfg.dequant_unpack_policy,
        )
        if sig in seen:
            continue
        seen.add(sig)
        out.append(cfg)
    return out


def _quant_autotune_key(
    w: jax.Array,
    *,
    mode: QuantizationMode,
    group_size: int,
    bits: int,
    axis: QuantizationAxis,
) -> tuple:
    """Build cache key for quantize autotuning."""
    return (
        jax.default_backend(),
        "quantize",
        mode,
        int(group_size),
        int(bits),
        axis,
        tuple(int(d) for d in w.shape),
        str(w.dtype),
    )


def _dequant_autotune_key(
    w_q: jax.Array,
    scales: jax.Array,
    zeros: jax.Array | None,
    *,
    mode: QuantizationMode,
    group_size: int,
    bits: int,
    axis: QuantizationAxis,
) -> tuple:
    """Build cache key for dequantize autotuning."""
    return (
        jax.default_backend(),
        "dequantize",
        mode,
        int(group_size),
        int(bits),
        axis,
        tuple(int(d) for d in w_q.shape),
        str(w_q.dtype),
        tuple(int(d) for d in scales.shape),
        str(scales.dtype),
        None if zeros is None else str(zeros.dtype),
    )


def _quantize_candidate_cfgs(
    base: QuantRuntimeConfig,
    *,
    mode: QuantizationMode,
) -> list[QuantRuntimeConfig]:
    """Build quantize autotune candidates around *base*."""
    del mode
    return [base]


def _dequantize_candidate_cfgs(base: QuantRuntimeConfig) -> list[QuantRuntimeConfig]:
    """Build dequantize autotune candidates around *base*."""
    backend = jax.default_backend()
    if backend == "tpu":
        compute_opts = ("bf16", "fp32")
    elif backend in {"gpu", "cuda", "mps"}:
        compute_opts = ("fp16", "fp32")
    else:
        compute_opts = ("fp32",)

    if backend in {"gpu", "cuda", "mps", "tpu"}:
        output_opts = ("compute", "fp32")
    else:
        output_opts = ("fp32",)

    unpack_opts = ("auto", "fast", "generic")
    candidates = [base]
    for compute_dtype in compute_opts:
        for output_dtype in output_opts:
            for unpack_policy in unpack_opts:
                candidates.append(
                    replace(
                        base,
                        prefer_compute_dtype=compute_dtype,
                        dequant_output_dtype=output_dtype,
                        dequant_unpack_policy=unpack_policy,
                    )
                )
    max_candidates = max(1, int(os.getenv("EJKERNEL_QRUNTIME_AUTOTUNE_MAX_CANDIDATES", "12")))
    return _dedupe_cfgs(candidates)[:max_candidates]


def _maybe_autotune_quantize_runtime_cfg(
    w: jax.Array,
    *,
    mode: QuantizationMode,
    group_size: int,
    bits: int,
    axis: QuantizationAxis,
    base_cfg: QuantRuntimeConfig,
) -> QuantRuntimeConfig:
    """Pick and cache a fast quantize runtime config for this shape/mode cell."""
    if not _autotune_enabled() or _is_tracing_tree(w):
        return base_cfg

    key = _quant_autotune_key(w, mode=mode, group_size=group_size, bits=bits, axis=axis)
    with _AUTOTUNE_CACHE_LOCK:
        cached = _AUTOTUNE_QUANT_CFG_CACHE.get(key)
    if cached is not None:
        return cached

    best_cfg = base_cfg
    best_ms = float("inf")
    min_gain = _autotune_min_gain()
    for cand in _quantize_candidate_cfgs(base_cfg, mode=mode):
        try:
            fn = jax.jit(
                lambda x, _cand=cand: quantize(
                    x,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                    axis=axis,
                    runtime_config=_cand,
                )
            )
            ms = _bench_ms(fn, (w,))
        except Exception:
            continue
        if ms < best_ms * (1.0 - min_gain):
            best_ms = ms
            best_cfg = cand

    with _AUTOTUNE_CACHE_LOCK:
        _AUTOTUNE_QUANT_CFG_CACHE.setdefault(key, best_cfg)
        return _AUTOTUNE_QUANT_CFG_CACHE[key]


def _maybe_autotune_dequantize_runtime_cfg(
    w_q: jax.Array,
    scales: jax.Array,
    zeros: jax.Array | None,
    *,
    mode: QuantizationMode,
    group_size: int,
    bits: int,
    axis: QuantizationAxis,
    base_cfg: QuantRuntimeConfig,
) -> QuantRuntimeConfig:
    """Pick and cache a fast dequantize runtime config for this shape/mode cell."""
    if not _autotune_enabled() or _is_tracing_tree((w_q, scales, zeros)):
        return base_cfg

    key = _dequant_autotune_key(
        w_q,
        scales,
        zeros,
        mode=mode,
        group_size=group_size,
        bits=bits,
        axis=axis,
    )
    with _AUTOTUNE_CACHE_LOCK:
        cached = _AUTOTUNE_DEQUANT_CFG_CACHE.get(key)
    if cached is not None:
        return cached

    best_cfg = base_cfg
    best_ms = float("inf")
    min_gain = _autotune_min_gain()
    for cand in _dequantize_candidate_cfgs(base_cfg):
        try:
            fn = jax.jit(
                lambda q, s, z, _cand=cand: dequantize(
                    q,
                    s,
                    z,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                    axis=axis,
                    runtime_config=_cand,
                )
            )
            ms = _bench_ms(fn, (w_q, scales, zeros))
        except Exception:
            continue
        if ms < best_ms * (1.0 - min_gain):
            best_ms = ms
            best_cfg = cand

    if best_cfg != base_cfg:
        try:
            confirm_warmup = max(2, int(os.getenv("EJKERNEL_QRUNTIME_AUTOTUNE_CONFIRM_WARMUP", "3")))
            confirm_iters = max(4, int(os.getenv("EJKERNEL_QRUNTIME_AUTOTUNE_CONFIRM_ITERS", "8")))
            fn_base = jax.jit(
                lambda q, s, z: dequantize(
                    q,
                    s,
                    z,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                    axis=axis,
                    runtime_config=base_cfg,
                )
            )
            fn_best = jax.jit(
                lambda q, s, z: dequantize(
                    q,
                    s,
                    z,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                    axis=axis,
                    runtime_config=best_cfg,
                )
            )
            base_ms = _bench_ms(fn_base, (w_q, scales, zeros), warmup=confirm_warmup, iters=confirm_iters)
            tuned_ms = _bench_ms(fn_best, (w_q, scales, zeros), warmup=confirm_warmup, iters=confirm_iters)
            if tuned_ms >= base_ms * (1.0 - min_gain):
                best_cfg = base_cfg
        except Exception:
            best_cfg = base_cfg

    with _AUTOTUNE_CACHE_LOCK:
        _AUTOTUNE_DEQUANT_CFG_CACHE.setdefault(key, best_cfg)
        return _AUTOTUNE_DEQUANT_CFG_CACHE[key]


def _to_quant_layout(w: jax.Array, axis: QuantizationAxis) -> jax.Array:
    """Map logical weight layout to quantization/runtime layout.

    For ``axis='row'``, swaps the last two dimensions so that the grouping
    dimension (last axis) runs over output channels. For ``axis='col'``,
    the layout is returned unchanged since grouping already runs over
    input channels.

    Args:
        w: Weight tensor with at least 2 dimensions, typically shaped
            ``(..., out_features, in_features)``.
        axis: Quantization axis. "row" transposes the last two dims;
            "col" leaves the layout unchanged.

    Returns:
        Weight tensor in quantization layout, possibly transposed.

    Raises:
        ValueError: If w has fewer than 2 dimensions.
    """
    if w.ndim < 2:
        raise ValueError("quantize expects inputs with two or more dimensions.")
    return jnp.swapaxes(w, -2, -1) if axis == "row" else w


def _resolve_affine_metadata_dtype(w_layout: jax.Array, runtime_cfg: QuantRuntimeConfig) -> jnp.dtype:
    """Resolve storage dtype for affine scales/zeros metadata.

    Args:
        w_layout: Weight tensor in quantization layout.  Its dtype is used
            when ``runtime_cfg.affine_metadata_dtype == "input"``.
        runtime_cfg: Runtime config whose ``affine_metadata_dtype`` field
            selects the storage dtype.

    Returns:
        JAX dtype for scale/zero-point storage.

    Raises:
        ValueError: If ``affine_metadata_dtype`` is not one of ``{"input",
            "bf16", "fp16", "fp32"}``.
    """
    mode = runtime_cfg.affine_metadata_dtype
    if mode == "input":
        return w_layout.dtype
    if mode == "bf16":
        return jnp.bfloat16
    if mode == "fp16":
        return jnp.float16
    if mode == "fp32":
        return jnp.float32
    raise ValueError(f"Unsupported affine metadata dtype mode: {mode!r}.")


def _pack_group_codes(
    w_layout: jax.Array,
    q: jax.Array,
    bits: int,
    runtime_cfg: QuantRuntimeConfig,
) -> jax.Array:
    """Pack grouped quantization codes into uint32 storage layout.

    Reshapes the code tensor to match the original weight layout (before
    grouping) then delegates to :func:`_pack_bits`.

    Args:
        w_layout: Weight tensor in quantization layout (used for shape
            reference only; its data is not read).
        q: uint32 array of quantization codes with shape
            ``(*w_layout.shape[:-1], n_groups, group_size)``.
        bits: Number of bits per code.
        runtime_cfg: Runtime config forwarded to ``_pack_bits``.

    Returns:
        Packed uint32 array of shape
        ``(*w_layout.shape[:-1], ceil(n_values * bits / 32))``.
    """
    return _pack_bits(
        q.reshape(*w_layout.shape[:-1], -1),
        bits,
        prefer_fast_u4_u8=runtime_cfg.enable_u4_u8_fastpath,
        strict_shape_alignment=runtime_cfg.strict_shape_alignment,
    )


def _use_argmin_codebook(runtime_cfg: QuantRuntimeConfig) -> bool:
    """Return ``True`` when the slow argmin codebook path should be used.

    The argmin path is selected when ``enable_parity_fallback=True`` (explicit
    parity mode) or when ``enable_threshold_codebook=False`` (threshold codebook
    disabled).
    """
    return runtime_cfg.enable_parity_fallback or not runtime_cfg.enable_threshold_codebook


def _quantize_affine(
    w_layout: jax.Array,
    w_groups: jax.Array,
    *,
    bits: int,
    runtime_cfg: QuantRuntimeConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Quantize groups with affine (scale + zero-point) metadata.

    For each group the per-group min/max determine:
    - ``scale = (max - min) / (2^bits - 1)``  (clipped to 1 when zero)
    - ``zero  = -min / scale``

    Quantization formula: ``q = clip(round(w / scale + zero), 0, 2^bits - 1)``

    Dequantization (inverse): ``w ≈ (q - zero) * scale``

    Args:
        w_layout: Weight tensor in quantization layout (shape reference).
        w_groups: Weight tensor reshaped as ``(..., n_groups, group_size)``.
        bits: Bit-width.  Controls the dynamic range ``[0, 2^bits - 1]``.
        runtime_cfg: Runtime config.  Affects metadata dtype and pack strategy.

    Returns:
        Tuple of ``(packed_data, scales, zeros)``:
        - ``packed_data``: uint32, shape ``(*w_layout.shape[:-1], n_words)``.
        - ``scales``: float, shape ``(*w_layout.shape[:-1], n_groups)``.
        - ``zeros``: float, same shape as ``scales``.
    """
    qmax = 2**bits - 1
    meta_dtype = _resolve_affine_metadata_dtype(w_layout, runtime_cfg)
    alpha = jnp.max(w_groups, axis=-1)
    beta = jnp.min(w_groups, axis=-1)

    scale = (alpha - beta) / qmax
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    zero = -beta / scale

    q = jnp.round(w_groups / scale[..., None] + zero[..., None])
    q = jnp.clip(q, 0, qmax).astype(jnp.uint32)
    packed = _pack_group_codes(w_layout, q, bits, runtime_cfg)
    return packed, scale.astype(meta_dtype), zero.astype(meta_dtype)


def _quantize_nf4(
    w_layout: jax.Array,
    w_groups: jax.Array,
    *,
    bits: int,
    runtime_cfg: QuantRuntimeConfig,
) -> tuple[jax.Array, jax.Array]:
    """Quantize groups into NF4 codebook indices and per-group absmax scales.

    Normalizes each group by its absolute-maximum value (so each element lies
    in ``[-1, 1]``), then maps to the nearest NF4 codebook entry using the
    configured search strategy (threshold or argmin).

    Dequantization: ``w ≈ nf4_table[q] * scale``

    Args:
        w_layout: Weight tensor in quantization layout (shape reference).
        w_groups: Weight tensor reshaped as ``(..., n_groups, group_size)``.
        bits: Must be 4 (NF4 always uses 4-bit codes); passed through for
            consistency with other quantize helpers.
        runtime_cfg: Runtime config.  Controls codebook search strategy and
            pack fast path.

    Returns:
        Tuple of ``(packed_data, scales)``:
        - ``packed_data``: uint32 packed codes.
        - ``scales``: float, per-group absmax values with same dtype as
          ``w_layout``.
    """
    codebook = _get_nf4_table()
    nf4_sorted_idx, nf4_boundaries = _get_nf4_threshold_map()
    max_abs = jnp.max(jnp.abs(w_groups), axis=-1)
    scale = jnp.where(max_abs == 0, jnp.ones_like(max_abs), max_abs)
    normalized = w_groups / scale[..., None]
    q = _quantize_to_codebook(
        normalized,
        codebook,
        use_argmin_fallback=_use_argmin_codebook(runtime_cfg),
        sorted_idx=nf4_sorted_idx,
        boundaries=nf4_boundaries,
    )
    packed = _pack_group_codes(w_layout, q, bits, runtime_cfg)
    return packed, scale.astype(w_layout.dtype)


def _quantize_mxfp(
    w_layout: jax.Array,
    w_groups: jax.Array,
    *,
    bits: int,
    runtime_cfg: QuantRuntimeConfig,
) -> tuple[jax.Array, jax.Array]:
    """Quantize groups using MXFP shared-exponent scaling.

    Computes a per-group shared exponent ``exp = ceil(log2(max_abs / vmax))``
    (clipped to the int8 range ``[-128, 127]``), then normalizes each group by
    ``2^exp`` and maps to the nearest E2M1 (4-bit) or E4M3 (8-bit) codebook
    entry.

    Dequantization: ``w ≈ e2m1_or_e4m3_table[q] * 2^exp``

    Args:
        w_layout: Weight tensor in quantization layout (shape reference).
        w_groups: Weight tensor reshaped as ``(..., n_groups, group_size)``.
        bits: 4 for MXFP4 (E2M1 codes) or 8 for MXFP8 (E4M3 codes).
        runtime_cfg: Runtime config.  Controls codebook search strategy and
            pack fast path.

    Returns:
        Tuple of ``(packed_data, scales)``:
        - ``packed_data``: uint32 packed E2M1 or E4M3 codes.
        - ``scales``: uint8 per-group shared exponents (int8 interpreted
          as uint8 for storage).
    """
    max_abs = jnp.max(jnp.abs(w_groups), axis=-1)
    if bits == 4:
        vmax = _get_e2m1_max()
        codebook, _ = _get_e2m1_table()
        codebook_sorted_idx, codebook_boundaries = _get_e2m1_threshold_map()
    else:
        vmax = _get_e4m3_max()
        codebook = _get_e4m3_table_q()
        codebook_sorted_idx, codebook_boundaries = _get_e4m3_q_threshold_map()

    exp = jnp.where(max_abs > 0, jnp.ceil(jnp.log2(max_abs / vmax)), 0.0)
    exp = jnp.clip(exp, -128, 127).astype(jnp.int8)

    scale = jnp.exp2(exp.astype(jnp.float32))
    scale = jnp.where(scale == 0, 1.0, scale)
    normalized = w_groups / scale[..., None]
    q = _quantize_to_codebook(
        normalized,
        codebook,
        use_argmin_fallback=_use_argmin_codebook(runtime_cfg),
        sorted_idx=codebook_sorted_idx,
        boundaries=codebook_boundaries,
    )
    packed = _pack_group_codes(w_layout, q, bits, runtime_cfg)
    return packed, exp.astype(jnp.uint8)


def _quantize_nvfp(
    w_layout: jax.Array,
    w_groups: jax.Array,
    *,
    bits: int,
    runtime_cfg: QuantRuntimeConfig,
) -> tuple[jax.Array, jax.Array]:
    """Quantize groups using NVFP value codes and E4M3 quantized scales.

    Unlike MXFP, NVFP stores a *quantized* (E4M3-coded) scale rather than a
    raw shared exponent:

    1. Compute ``scale_raw = max_abs / vmax``.
    2. Quantize ``scale_raw`` into an E4M3 index and decode back to float.
    3. Normalize each group element by the decoded scale.
    4. Quantize each element into the target code (E2M1 for 4-bit, E4M3 for
       8-bit).

    Dequantization:
    ``w ≈ q_table[q] * e4m3_table[scale_q]``

    Args:
        w_layout: Weight tensor in quantization layout (shape reference).
        w_groups: Weight tensor reshaped as ``(..., n_groups, group_size)``.
        bits: 4 for NVFP4 (E2M1 value codes) or 8 for NVFP8 (E4M3 value codes).
        runtime_cfg: Runtime config.  Controls codebook search strategy and
            pack fast path.

    Returns:
        Tuple of ``(packed_data, scales)``:
        - ``packed_data``: uint32 packed E2M1 or E4M3 value codes.
        - ``scales``: uint8 per-group E4M3 scale codes.
    """
    scale_sorted_idx, scale_boundaries = _get_e4m3_q_threshold_map()
    if bits == 4:
        vmax = _get_e2m1_max()
        q_codebook, _ = _get_e2m1_table()
        q_sorted_idx, q_boundaries = _get_e2m1_threshold_map()
    else:
        vmax = _get_e4m3_max()
        q_codebook = _get_e4m3_table_q()
        q_sorted_idx, q_boundaries = _get_e4m3_q_threshold_map()

    max_abs = jnp.max(jnp.abs(w_groups), axis=-1)
    scale_raw = jnp.where(max_abs > 0, max_abs / vmax, 0.0)
    scale_codebook = _get_e4m3_table_q()
    scale_q = _quantize_to_codebook(
        scale_raw,
        scale_codebook,
        use_argmin_fallback=_use_argmin_codebook(runtime_cfg),
        sorted_idx=scale_sorted_idx,
        boundaries=scale_boundaries,
    ).astype(jnp.uint32)

    e4m3_table, _ = _get_e4m3_table()
    scale = e4m3_table[scale_q.astype(jnp.int32)]
    scale = jnp.where(scale == 0, 1.0, scale)
    normalized = w_groups / scale[..., None]

    q = _quantize_to_codebook(
        normalized,
        q_codebook,
        use_argmin_fallback=_use_argmin_codebook(runtime_cfg),
        sorted_idx=q_sorted_idx,
        boundaries=q_boundaries,
    )
    packed = _pack_group_codes(w_layout, q, bits, runtime_cfg)
    return packed, scale_q.astype(jnp.uint8)


def _resolve_compute_dtype(runtime_cfg: QuantRuntimeConfig) -> jnp.dtype:
    """Resolve arithmetic dtype for dequantization math.

    Args:
        runtime_cfg: Config whose ``prefer_compute_dtype`` field is read.

    Returns:
        JAX dtype for intermediate arithmetic.

    Raises:
        ValueError: If ``prefer_compute_dtype`` is not ``"bf16"``, ``"fp16"``,
            or ``"fp32"``.
    """
    mode = runtime_cfg.prefer_compute_dtype
    if mode == "bf16":
        return jnp.bfloat16
    if mode == "fp16":
        return jnp.float16
    if mode == "fp32":
        return jnp.float32
    raise ValueError(f"Unsupported compute dtype mode: {mode!r}.")


def _resolve_dequant_output_dtype(runtime_cfg: QuantRuntimeConfig, compute_dtype: jnp.dtype) -> jnp.dtype:
    """Resolve output dtype for dequantized tensors.

    Args:
        runtime_cfg: Config whose ``dequant_output_dtype`` field is read.
        compute_dtype: The resolved compute dtype (used when
            ``dequant_output_dtype == "compute"``).

    Returns:
        JAX dtype for the dequantized output tensor.

    Raises:
        ValueError: If ``dequant_output_dtype`` is not one of
            ``{"compute", "bf16", "fp16", "fp32"}``.
    """
    mode = runtime_cfg.dequant_output_dtype
    if mode == "compute":
        return compute_dtype
    if mode == "bf16":
        return jnp.bfloat16
    if mode == "fp16":
        return jnp.float16
    if mode == "fp32":
        return jnp.float32
    raise ValueError(f"Unsupported dequant output dtype mode: {mode!r}.")


def _prefer_fast_unpack(
    *,
    runtime_cfg: QuantRuntimeConfig,
    mode: QuantizationMode,
    bits: int,
    batch_size_hint: int,
) -> bool:
    """Return ``True`` to select the fast grouped unpack path.

    The decision is driven by ``runtime_cfg.dequant_unpack_policy``:

    - ``"fast"``: always use the fast path (when
      ``enable_u4_u8_fastpath=True``).
    - ``"generic"``: always use the generic scatter/gather path.
    - ``"auto"``: equivalent to the fast path when
      ``enable_u4_u8_fastpath=True``.

    Args:
        runtime_cfg: Runtime config.
        mode: Quantization mode (currently unused; reserved for future
            per-mode dispatch).
        bits: Bit-width (currently unused; reserved for future dispatch).
        batch_size_hint: Product of leading dimensions (currently unused;
            reserved for future batch-size-aware dispatch).

    Returns:
        ``True`` to use the fast grouped unpacker.

    Raises:
        ValueError: If ``dequant_unpack_policy`` is not one of
            ``{"auto", "fast", "generic"}``.
    """
    del mode, bits, batch_size_hint
    policy = runtime_cfg.dequant_unpack_policy
    if policy == "fast":
        return runtime_cfg.enable_u4_u8_fastpath
    if policy == "generic":
        return False
    if policy != "auto":
        raise ValueError(f"Unsupported dequant unpack policy: {policy!r}.")
    return runtime_cfg.enable_u4_u8_fastpath


def _prefer_arith_minifloat_decode(runtime_cfg: QuantRuntimeConfig) -> bool:
    """Return ``True`` to use arithmetic decode instead of table lookup.

    Table lookup (the default) maps integer codes through a pre-computed
    float32 table via gather.  Arithmetic decode reconstructs float values
    from sign/exponent/mantissa fields using bitwise ops.

    The choice is driven by ``runtime_cfg.minifloat_decode_policy``:

    - ``"arith"``: always use arithmetic decode.
    - ``"table"``: always use table lookup.
    - ``"auto"``: currently always returns ``False`` (table lookup), as table
      lookup maps well to constant-memory gathers on GPU.

    Raises:
        ValueError: If ``minifloat_decode_policy`` is not one of
            ``{"auto", "table", "arith"}``.
    """
    policy = runtime_cfg.minifloat_decode_policy
    if policy == "arith":
        return True
    if policy == "table":
        return False
    if policy != "auto":
        raise ValueError(f"Unsupported minifloat decode policy: {policy!r}.")
    return False


def _unpack_groups(
    w_q: jax.Array,
    scales: jax.Array,
    *,
    mode: QuantizationMode,
    group_size: int,
    bits: int,
    runtime_cfg: QuantRuntimeConfig,
) -> tuple[jax.Array, int]:
    """Unpack bitpacked codes and reshape to ``(..., n_groups, group_size)``.

    Args:
        w_q: Packed uint32 array from :func:`quantize`.
        scales: Per-group scale array.  Its shape is used to infer the number
            of groups (``n_groups = scales.shape[-1]``).
        mode: Quantization mode (forwarded to ``_prefer_fast_unpack``).
        group_size: Number of elements per group.
        bits: Bits per code.
        runtime_cfg: Runtime config controlling the unpack strategy.

    Returns:
        Tuple of ``(codes, n)`` where:
        - ``codes``: uint32 array of shape
          ``(*scales.shape[:-1], n_groups, group_size)``.
        - ``n``: total number of dequantized values
          (``n_groups * group_size``).
    """
    n_groups = scales.shape[-1]
    n = n_groups * group_size
    batch_size_hint = 1
    if scales.ndim > 1:
        for dim in scales.shape[:-1]:
            batch_size_hint *= int(dim)
    q = _unpack_bits(
        w_q,
        n,
        bits,
        prefer_fast_u4_u8=_prefer_fast_unpack(
            runtime_cfg=runtime_cfg,
            mode=mode,
            bits=bits,
            batch_size_hint=batch_size_hint,
        ),
    )
    return q.reshape(*scales.shape[:-1], n_groups, group_size), n


def _dequantize_affine_bits(
    w_q: jax.Array,
    scales: jax.Array,
    zeros: jax.Array,
    *,
    group_size: int,
    bits: int,
    runtime_cfg: QuantRuntimeConfig,
) -> jax.Array:
    """Affine dequantization specialized by static bit-width.

    Formula: ``w ≈ (q - zero) * scale``

    Args:
        w_q: Packed uint32 codes.
        scales: Float per-group scales.
        zeros: Float per-group zero-points (same shape as *scales*).
        group_size: Elements per group.
        bits: Bit-width of the stored codes.
        runtime_cfg: Controls compute/output dtypes and unpack strategy.

    Returns:
        Dequantized float array of shape
        ``(*scales.shape[:-1], n_groups * group_size)``.
    """
    compute_dtype = _resolve_compute_dtype(runtime_cfg)
    output_dtype = _resolve_dequant_output_dtype(runtime_cfg, compute_dtype)
    q, n = _unpack_groups(w_q, scales, mode="affine", group_size=group_size, bits=bits, runtime_cfg=runtime_cfg)
    q = q.astype(compute_dtype)
    scales_f = scales.astype(compute_dtype)
    zeros_f = zeros.astype(compute_dtype)
    out = (q - zeros_f[..., None]) * scales_f[..., None]
    if out.dtype != output_dtype:
        out = out.astype(output_dtype)
    return out.reshape(*scales.shape[:-1], n)


def _dequantize_nf4(
    w_q: jax.Array,
    scales: jax.Array,
    *,
    group_size: int,
    runtime_cfg: QuantRuntimeConfig,
) -> jax.Array:
    """NF4 dequantization: table lookup then scale by per-group absmax.

    Formula: ``w ≈ nf4_table[q] * scale``

    Args:
        w_q: Packed uint32 NF4 codes.
        scales: Float per-group absmax scales.
        group_size: Elements per group.
        runtime_cfg: Controls compute/output dtypes and unpack strategy.

    Returns:
        Dequantized float array of shape
        ``(*scales.shape[:-1], n_groups * group_size)``.
    """
    compute_dtype = _resolve_compute_dtype(runtime_cfg)
    output_dtype = _resolve_dequant_output_dtype(runtime_cfg, compute_dtype)
    q, n = _unpack_groups(w_q, scales, mode="nf4", group_size=group_size, bits=4, runtime_cfg=runtime_cfg)
    q = q.astype(jnp.int32)
    table = _get_nf4_table().astype(compute_dtype)
    vals = table[q]
    out = vals * scales.astype(compute_dtype)[..., None]
    if out.dtype != output_dtype:
        out = out.astype(output_dtype)
    return out.reshape(*scales.shape[:-1], n)


def _dequantize_mxfp_bits(
    w_q: jax.Array,
    scales: jax.Array,
    *,
    group_size: int,
    bits: int,
    runtime_cfg: QuantRuntimeConfig,
) -> jax.Array:
    """MXFP dequantization specialized for 4-bit (E2M1) or 8-bit (E4M3) payloads.

    Decodes E2M1 or E4M3 codes (via lookup or arithmetic) then scales by
    ``2^exp`` using :func:`jnp.ldexp` for integer exponents.

    Formula: ``w ≈ e2m1_or_e4m3_table[q] * 2^exp``

    Args:
        w_q: Packed uint32 E2M1 or E4M3 codes.
        scales: uint8 per-group shared exponents (stored as uint8; cast to
            int8 before use).
        group_size: Elements per group.
        bits: 4 for MXFP4 or 8 for MXFP8.
        runtime_cfg: Controls compute/output dtypes, unpack, and decode
            strategy.

    Returns:
        Dequantized float array of shape
        ``(*scales.shape[:-1], n_groups * group_size)``.
    """
    compute_dtype = _resolve_compute_dtype(runtime_cfg)
    output_dtype = _resolve_dequant_output_dtype(runtime_cfg, compute_dtype)
    q, n = _unpack_groups(
        w_q,
        scales,
        mode="mxfp4" if bits == 4 else "mxfp8",
        group_size=group_size,
        bits=bits,
        runtime_cfg=runtime_cfg,
    )
    if _prefer_arith_minifloat_decode(runtime_cfg):
        vals = _decode_e2m1_codes(q, dtype=compute_dtype) if bits == 4 else _decode_e4m3_codes(q, dtype=compute_dtype)
    else:
        q_i = q.astype(jnp.int32)
        if bits == 4:
            table, _ = _get_e2m1_table()
        else:
            table, _ = _get_e4m3_table()
        table = table.astype(compute_dtype)
        vals = table[q_i]
    exp_i = scales.astype(jnp.int8).astype(jnp.int32)
    out = jnp.ldexp(vals, exp_i[..., None])
    if out.dtype != output_dtype:
        out = out.astype(output_dtype)
    return out.reshape(*scales.shape[:-1], n)


def _dequantize_nvfp_bits(
    w_q: jax.Array,
    scales: jax.Array,
    *,
    group_size: int,
    bits: int,
    runtime_cfg: QuantRuntimeConfig,
) -> jax.Array:
    """NVFP dequantization specialized for 4-bit (E2M1) or 8-bit (E4M3) payloads.

    Decodes E4M3 scale codes and E2M1/E4M3 value codes then multiplies:

    Formula: ``w ≈ q_table[q] * e4m3_table[scale_q]``

    Args:
        w_q: Packed uint32 E2M1 or E4M3 value codes.
        scales: uint8 per-group E4M3 scale codes.
        group_size: Elements per group.
        bits: 4 for NVFP4 or 8 for NVFP8.
        runtime_cfg: Controls compute/output dtypes, unpack, and decode
            strategy.

    Returns:
        Dequantized float array of shape
        ``(*scales.shape[:-1], n_groups * group_size)``.
    """
    compute_dtype = _resolve_compute_dtype(runtime_cfg)
    output_dtype = _resolve_dequant_output_dtype(runtime_cfg, compute_dtype)
    q, n = _unpack_groups(
        w_q,
        scales,
        mode="nvfp4" if bits == 4 else "nvfp8",
        group_size=group_size,
        bits=bits,
        runtime_cfg=runtime_cfg,
    )
    if _prefer_arith_minifloat_decode(runtime_cfg):
        q_vals = _decode_e2m1_codes(q, dtype=compute_dtype) if bits == 4 else _decode_e4m3_codes(q, dtype=compute_dtype)
        scale_vals = _decode_e4m3_codes(scales, dtype=compute_dtype)
        out = q_vals * scale_vals[..., None]
    else:
        q_i = q.astype(jnp.int32)
        e4m3_table, _ = _get_e4m3_table()
        if bits == 4:
            q_table, _ = _get_e2m1_table()
        else:
            q_table = e4m3_table
        q_table = q_table.astype(compute_dtype)
        scale_table = e4m3_table.astype(compute_dtype)
        vals = q_table[q_i]
        out = vals * scale_table[scales.astype(jnp.int32)][..., None]
    if out.dtype != output_dtype:
        out = out.astype(output_dtype)
    return out.reshape(*scales.shape[:-1], n)


def quantize(
    w: jax.Array,
    /,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    axis: QuantizationAxis = "row",
    runtime_config: QuantRuntimeConfig | None = None,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Quantize weights into packed uint32 codes with per-group scaling.

    Supports multiple quantization modes, each producing different output
    tuples. The weight tensor is first transposed to the quantization layout
    (based on axis), then grouped and quantized per-group.

    Args:
        w: Weight tensor with at least 2 dimensions. The last dimension
            must be divisible by the resolved group_size.
        group_size: Number of elements per quantization group, or None for
            mode-specific default (e.g., 64 for affine, 32 for mxfp4).
        bits: Bit-width for quantized values, or None for mode-specific
            default (e.g., 4 for affine and nf4).
        mode: Quantization mode. One of "affine", "nf4", "mxfp4", "mxfp8",
            "nvfp4", "nvfp8".
        axis: Quantization axis determining how groups are formed:
            "row" groups over output channels, "col" groups over input channels.
        runtime_config: Optional runtime fast-path policy. When omitted,
            defaults are used.

    Returns:
        For affine mode: tuple of (w_q, scales, zeros) where the dequantization
            formula is ``(q - zero) * scale``.
        For all other modes: tuple of (w_q, scales) where scales encode either
            per-group float scales (nf4) or shared exponents (mxfp/nvfp).
    """
    axis = normalize_axis(axis)
    mode, group_size, bits, _ = resolve_qparams(mode, group_size, bits)
    runtime_cfg = resolve_runtime_config(runtime_config)
    if runtime_config is None:
        runtime_cfg = _maybe_autotune_quantize_runtime_cfg(
            w,
            mode=mode,
            group_size=group_size,
            bits=bits,
            axis=axis,
            base_cfg=runtime_cfg,
        )

    w_layout = _to_quant_layout(w, axis)
    if w_layout.shape[-1] % group_size != 0:
        raise ValueError(
            "group_size is incompatible with the effective grouping axis. "
            f"input_shape={tuple(w.shape)}, axis={axis!r}, runtime_layout_shape={tuple(w_layout.shape)}, "
            f"group_dim={w_layout.shape[-1]}, group_size={group_size}. "
            "For grouping over the original last dimension, use axis='col'."
        )
    w_groups, _ = _reshape_groups(w_layout, group_size)

    if mode == "affine":
        return _quantize_affine(w_layout, w_groups, bits=bits, runtime_cfg=runtime_cfg)

    if mode == "nf4":
        return _quantize_nf4(w_layout, w_groups, bits=bits, runtime_cfg=runtime_cfg)

    if mode in {"mxfp4", "mxfp8"}:
        return _quantize_mxfp(w_layout, w_groups, bits=bits, runtime_cfg=runtime_cfg)

    return _quantize_nvfp(w_layout, w_groups, bits=bits, runtime_cfg=runtime_cfg)


def dequantize(
    w_q: jax.Array,
    scales: jax.Array,
    zeros: jax.Array | None = None,
    *,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    axis: QuantizationAxis = "row",
    runtime_config: QuantRuntimeConfig | None = None,
) -> jax.Array:
    """Dequantize packed uint32 weights back to floating-point values.

    Reverses the quantization performed by ``quantize()``, unpacking the
    bit-packed codes and applying the appropriate inverse transformation
    for the specified mode.

    Args:
        w_q: Packed uint32 array of quantized codes from ``quantize()``.
        scales: Per-group scale factors (float for affine/nf4, uint8
            exponents for mxfp/nvfp modes).
        zeros: Per-group zero-point offsets. Required for affine mode
            (dequantization formula: ``(q - zero) * scale``). Must be
            None for all other modes.
        group_size: Number of elements per quantization group, or None
            for mode-specific default.
        bits: Bit-width for quantized values, or None for mode-specific default.
        mode: Quantization mode matching the one used during quantization.
        axis: Quantization axis (kept for API symmetry; not used in
            current implementation).
        runtime_config: Optional runtime fast-path policy. When omitted,
            defaults are used.

    Returns:
        Dequantized tensor with the same leading dimensions as scales and last
        dimension equal to ``n_groups * group_size``. Output dtype follows
        ``runtime_config.dequant_output_dtype``.

    Raises:
        ValueError: If zeros is None for affine mode.
    """
    axis = normalize_axis(axis)
    axis_n = axis
    del axis
    mode, group_size, bits, _ = resolve_qparams(mode, group_size, bits)
    runtime_cfg = resolve_runtime_config(runtime_config)
    if runtime_config is None:
        runtime_cfg = _maybe_autotune_dequantize_runtime_cfg(
            w_q,
            scales,
            zeros,
            mode=mode,
            group_size=group_size,
            bits=bits,
            axis=axis_n,
            base_cfg=runtime_cfg,
        )

    if mode == "affine":
        if zeros is None:
            raise ValueError("affine dequantize requires `zeros`.")
        return _dequantize_affine_bits(
            w_q,
            scales,
            zeros,
            group_size=group_size,
            bits=bits,
            runtime_cfg=runtime_cfg,
        )

    if mode == "nf4":
        return _dequantize_nf4(
            w_q,
            scales,
            group_size=group_size,
            runtime_cfg=runtime_cfg,
        )

    if mode in {"mxfp4", "mxfp8"}:
        if bits == 4:
            return _dequantize_mxfp_bits(
                w_q,
                scales,
                group_size=group_size,
                bits=4,
                runtime_cfg=runtime_cfg,
            )
        return _dequantize_mxfp_bits(
            w_q,
            scales,
            group_size=group_size,
            bits=8,
            runtime_cfg=runtime_cfg,
        )

    if bits == 4:
        return _dequantize_nvfp_bits(
            w_q,
            scales,
            group_size=group_size,
            bits=4,
            runtime_cfg=runtime_cfg,
        )
    return _dequantize_nvfp_bits(
        w_q,
        scales,
        group_size=group_size,
        bits=8,
        runtime_cfg=runtime_cfg,
    )


def _resolve_matmul_align_multiple(align_multiple: int | None) -> int:
    """Resolve K/N padding multiple for the dense dequantize-then-matmul path.

    Reads ``EJKERNEL_QMM_ALIGN_MULTIPLE`` from the environment when
    *align_multiple* is ``None`` and no env var is set.  Backend-specific
    defaults: 128 on GPU/TPU/MPS, 0 (no alignment) on CPU.

    Args:
        align_multiple: Explicit padding multiple, or ``None`` to use the
            environment variable / backend default.

    Returns:
        Non-negative int alignment multiple.  0 or 1 means no padding.
    """
    if align_multiple is not None:
        return max(0, int(align_multiple))
    env = os.getenv("EJKERNEL_QMM_ALIGN_MULTIPLE")
    if env is not None and env.strip() != "":
        return max(0, int(env))
    backend = jax.default_backend()
    if backend in {"gpu", "cuda", "mps", "tpu"}:
        return 128
    return 0


def _pad_axis_to_multiple(x: jax.Array, axis: int, multiple: int) -> tuple[jax.Array, int]:
    """Pad *x* along *axis* to the nearest multiple, returning ``(padded, original_size)``.

    Does nothing when ``multiple <= 1`` or when the dimension is already
    aligned.

    Args:
        x: Input array.
        axis: Axis to pad (negative indices supported).
        multiple: Target alignment multiple.

    Returns:
        Tuple of ``(padded_array, original_size)`` where ``original_size`` is
        the un-padded size along ``axis``.
    """
    if multiple <= 1:
        return x, int(x.shape[axis])
    axis_n = axis if axis >= 0 else x.ndim + axis
    n = int(x.shape[axis_n])
    rem = n % multiple
    if rem == 0:
        return x, n
    pad = multiple - rem
    pad_spec = [(0, 0)] * x.ndim
    pad_spec[axis_n] = (0, pad)
    return jnp.pad(x, pad_spec), n


@ejit(static_argnames=["transpose", "group_size", "bits", "mode", "axis", "align_multiple"])
def quantized_matmul(
    x: jax.Array,
    w: jax.Array,
    /,
    scales: jax.Array,
    zeros: jax.Array | None = None,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    axis: QuantizationAxis | None = None,
    align_multiple: int | None = None,
) -> jax.Array:
    """Dense reference quantized matrix multiplication via dequantize-then-matmul.

    First dequantizes the packed weight tensor back to full precision, then
    performs a standard matrix multiplication with the input activations.
    This is a reference implementation; for fused high-performance variants,
    see ``ejkernel.modules.operations.quantized_matmul``.

    Args:
        x: Input activation tensor of shape ``(..., K)`` where K is the
            contraction dimension.
        w: Packed uint32 weight tensor from ``quantize()`` or
            ``prepack_quantized_weights()``.
        scales: Per-group scale factors for dequantization.
        zeros: Per-group zero-point offsets (required for affine mode,
            must be None for other modes).
        transpose: If True, transposes the dequantized weight before matmul
            (``x @ w.T``). If False, uses ``x @ w``.
        group_size: Number of elements per quantization group, or None
            for mode-specific default.
        bits: Bit-width for quantized values, or None for mode-specific default.
        mode: Quantization mode matching the one used during quantization.
        axis: Explicit quantization axis. If provided, overrides the transpose
            flag for consistency.
        align_multiple: Optional K/N padding multiple for the dense reference
            path. ``None`` uses backend defaults (e.g., 128 on TPU/GPU/MPS).

    Returns:
        Matrix multiplication result with shape ``(..., N)`` where N is the
        output dimension of the weight matrix.
    """
    mode_n, group_size_n, bits_n, _ = resolve_qparams(mode, group_size, bits)
    runtime_axis, transpose = resolve_runtime_axis_and_transpose(axis=axis, transpose=transpose)
    validate_packed_quantized_matmul_layout(
        x,
        w,
        scales,
        zeros,
        mode=mode_n,
        group_size=group_size_n,
        bits=bits_n,
        axis=runtime_axis,
        transpose=transpose,
    )

    dequant_axis: QuantizationAxis = runtime_axis
    w_f = dequantize(
        w,
        scales,
        zeros,
        group_size=group_size_n,
        bits=bits_n,
        mode=mode_n,
        axis=dequant_axis,
    )
    rhs = w_f.T if transpose else w_f
    align_n = _resolve_matmul_align_multiple(align_multiple)
    if align_n <= 1:
        return x @ rhs

    x_pad, _ = _pad_axis_to_multiple(x, axis=-1, multiple=align_n)
    rhs_pad, _ = _pad_axis_to_multiple(rhs, axis=-2, multiple=align_n)
    rhs_pad, n_orig = _pad_axis_to_multiple(rhs_pad, axis=-1, multiple=align_n)
    out = x_pad @ rhs_pad
    return out[..., :n_orig]


def prepack_quantized_weights(
    w: jax.Array,
    /,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    transpose: bool = True,
    axis: QuantizationAxis | None = None,
    runtime_config: QuantRuntimeConfig | None = None,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Prepack logical ``(out_features, in_features)`` weights for quantized matmul.

    Convenience wrapper around ``quantize()`` that resolves the quantization
    axis from either the explicit ``axis`` parameter or the legacy
    ``transpose`` flag. The output is ready for use with
    ``quantized_matmul()`` or the fused kernel variants.

    Backward compatibility when ``axis`` is omitted:
        - ``transpose=True`` maps to ``axis='row'`` (group over out features).
        - ``transpose=False`` maps to ``axis='col'`` (group over in features).

    Args:
        w: Weight tensor of shape ``(out_features, in_features)`` or with
            additional leading batch dimensions.
        group_size: Number of elements per quantization group, or None for
            mode-specific default.
        bits: Bit-width for quantized values, or None for mode-specific default.
        mode: Quantization mode. One of "affine", "nf4", "mxfp4", "mxfp8",
            "nvfp4", "nvfp8".
        transpose: Legacy flag for axis inference when ``axis`` is None.
        axis: Explicit quantization axis ("row" or "col"). Overrides
            ``transpose`` when provided.
        runtime_config: Optional runtime fast-path policy passed to ``quantize``.

    Returns:
        For affine mode: tuple of (w_q, scales, zeros).
        For other modes: tuple of (w_q, scales).
    """
    axis = resolve_prepack_axis(axis=axis, transpose=transpose)
    return quantize(
        w,
        group_size=group_size,
        bits=bits,
        mode=mode,
        axis=axis,
        runtime_config=runtime_config,
    )
