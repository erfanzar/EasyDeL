"""Opt-in TPU streaming CHANNELWISE grouped matmul for A16/A4/A8.

Only the primal streams integer weights. AD uses the XLA represented-weight
reference, which may materialize floating weights; no backward memory saving
is promised. Codes and group membership are frozen, scales differentiate exactly.
"""

import jax
import jax.numpy as jnp

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.grouped_matmul_quant._channelwise import grouped_matmul_channelwise as _reference
from ..grouped_matmulv3._pallas_impl import TileSizes, grouped_matmulv3_pallas_impl


def _integer_streaming_tiles(k, n, dtype):
    """Bound each of three RHS buffers to 2 MiB, including very wide N."""
    tk = ((k + 127) // 128) * 128
    tn = max(128, ((n + 255) // 256) * 128)
    bits = jax.dtypes.itemsize_bits(dtype)
    while tk * tn * bits // 8 > 2 * 1024 * 1024:
        if tk > 128:
            tk = max(128, ((tk // 2 + 127) // 128) * 128)
        else:
            tn = max(128, ((tn // 2 + 127) // 128) * 128)
    return (32, tk, tn)


_A8_SLAB_MAX_ROWS = 1024
"""Rows (decode buckets) up to which A8 int8 experts stream one whole (K, N) slab per grid step with 32-row tiles."""
_SLAB_MAX_BYTES = 2 * 1024 * 1024
"""RHS bytes per pipeline buffer, the budget ``_integer_streaming_tiles`` keeps (three buffers: 6 MiB of VMEM)."""
_A8_PREFILL_SLAB_FAMILIES = frozenset({(72, 4096, 512), (72, 512, 4096)})
"""Expert-stacked ``(E, K, N)`` int8 families whose whole-slab A8 tiles were measured at prefill rows on TPU v5p: the
per-chip routed experts (gate/up, down) of GLM-5.3-Flash served at ep4 x tp4."""
_A8_PREFILL_SLAB_MAX_ROWS = 81920
"""Largest measured prefill buffer: the deferred MoE dispatch's window of one 32,768-token step at ep4 (top-8)."""
_A8_PREFILL_TILE_M = ((55, 64), (200, 128), (400, 256))
"""``(max rows per expert, tile_m)`` of the prefill slab tiles; 512 above the last bound (see ``_a8_slab_tiles``)."""


def _a8_slab_tiles(m, activation_bits, e, k, n, dtype):
    """``(tile_m, K, N)`` tiles holding one whole int8 expert slab, or ``None`` (keep ``_integer_streaming_tiles``).

    Applies to A8 activations with int8 codes whose ``(K, N)`` slab is 128-aligned and fits one 2 MiB RHS buffer:
    at up to ``_A8_SLAB_MAX_ROWS`` rows with 32-row tiles, and on the measured ``_A8_PREFILL_SLAB_FAMILIES`` up to
    ``_A8_PREFILL_SLAB_MAX_ROWS`` rows with the ``_A8_PREFILL_TILE_M`` row tile of the buffer's rows per expert
    ``m / E``. Experts without rows are never read, and the entry applies both scales inside the kernel. Measured on
    TPU v5p, one chip, on the per-chip expert shapes of GLM-5.3-Flash served at ep4 x tp4 (int8 [72, 4096, 512]
    gate/up, [72, 512, 4096] down), every output bitwise equal to the XLA entry's:

    * decode, 8 layers with distinct weights, 256 dispatch rows of which 52-64 routed: the raw int8 dot costs
      0.865 / 0.816 us per active expert (2.42 / 2.57 TB/s) against 1.47 / 1.19 us for XLA's ragged dot, and at 37
      active experts 36.0 us per call against 42.9 (gate) and 44.4 (down) with ``(32, K, ~N/2)`` tiles;
    * prefill, the chained gate + up + down FFN of 2 layers at the ep4 windows of the 512 ... 32,768-token buckets
      (about 4/5 of each window routed), us per layer for XLA / tile 32 / 64 / 128 / 256 / 512, with the window's
      rows per expert: 2,048 rows (28) 477 / 343 / 317 / 423 / 673 / 1187; 3,072 (43) 571 / 505 / 382 / 462 / 703 /
      1216; 5,120 (71) 754 / 691 / 583 / 532 / 779 / 1284; 10,240 (142) 1166 / 1104 / 917 / 861 / 928 / 1427;
      20,480 (284) 1905 / 1959 / 1625 / 1398 / 1302 / 1683; 40,960 (569) 3361 / 3548 / 2931 / 2601 / 2365 / 2294;
      81,920 (1138) 6355 / 6781 / 5559 / 5019 / 4712 / 4564. The tile bounds sit at the geometric midpoints of the
      measured rows per expert where the fastest tile changes.
    """
    if activation_bits != 8 or jnp.dtype(dtype) != jnp.dtype(jnp.int8) or m <= 0:
        return None
    if k % 128 or n % 128 or k * n > _SLAB_MAX_BYTES:
        return None
    if m <= _A8_SLAB_MAX_ROWS:
        return (32, k, n)
    if m > _A8_PREFILL_SLAB_MAX_ROWS or (e, k, n) not in _A8_PREFILL_SLAB_FAMILIES:
        return None
    rows_per_expert = m / e
    tile_m = next((tile for bound, tile in _A8_PREFILL_TILE_M if rows_per_expert <= bound), 512)
    return (tile_m, k, n)


# Expert-stacked ``(E, K, N)`` families whose full-K decode tiles were measured on TPU; others keep generic tiles.
# DeepSeek-V4.1 tp=4 per-device families (TPU v5, 48 rows, int4, bit-exact vs XLA): gate/up [96,5120,2304]
# 127 -> 62 us, fused gate_up [96,5120,4608] 204 -> 110 us, down [96,2304,5120] 107 -> 62 us.
_MEASURED_DECODE_FAMILIES = frozenset(
    {
        (128, 2560, 1280),
        (128, 640, 2560),
        (96, 5120, 2304),
        (96, 5120, 4608),
        (96, 2304, 5120),
    }
)


# Families whose full-K prefill tiles were measured on TPU v5 (int4, bit-exact vs XLA) for 129..12288 rows, with
# tile_m=64: gate/up [96,5120,2304] 2.05 -> 1.18 ms at 1950 rows and 5.43 -> 1.36 ms at 12288 rows; down
# [96,2304,5120] 1.78 -> 1.21 ms and 4.85 -> 1.63 ms (prior generic (16,512,1024) tiles; XLA 2.55-3.17 ms).
_MEASURED_PREFILL_FAMILIES = frozenset({(96, 5120, 2304), (96, 2304, 5120)})
_MEASURED_PREFILL_MAX_ROWS = 12288


def _weight_only_streaming_tiles(m, e, k, n, dtype):
    """A16 tiles for measured families: full K, N split into the widest 128-multiple that divides N and keeps each
    RHS buffer within 2 MiB; ``tile_m`` 16 for decode rows (<= 128) and 64 for measured prefill rows. Unmeasured
    shapes and row ranges (including other expert counts) keep the prior generic tiles."""
    family = (e, k, n)
    tile_m = None
    if 0 < m <= 128 and family in _MEASURED_DECODE_FAMILIES:
        tile_m = 16
    elif 128 < m <= _MEASURED_PREFILL_MAX_ROWS and family in _MEASURED_PREFILL_FAMILIES:
        tile_m = 64
    if tile_m is not None:
        budget = (2 * 1024 * 1024 * 8) // (k * jax.dtypes.itemsize_bits(dtype))
        divisors = [t for t in range(128, min(n, budget) + 1, 128) if n % t == 0]
        if divisors:
            return (tile_m, k, divisors[-1])
    return (16, 512 if k >= 512 else 128, 1024 if n >= 1024 else 128)


def _expand_channel_scales(scales, sizes, count):
    """Avoid repeat's scatter-based construction on bounded decode shapes."""
    from ...._xla.grouped_matmul_quant._scale_rows import expand_group_scales

    return expand_group_scales(scales.astype(jnp.float32), sizes, count)


@kernel_registry.register("grouped_matmul_channelwise", Platform.PALLAS, Backend.TPU)
def grouped_matmul_channelwise(
    lhs,
    codes,
    scales,
    group_sizes,
    *,
    activation_bits=16,
    preferred_element_type=jnp.bfloat16,
    tiling=None,
):
    """Stream signed int4/int8 weights with explicit activation precision.

    A16 uses BF16 inputs and FP32 accumulation. A4/A8 quantize each input row,
    then use INT8 arithmetic with INT32 accumulation; the A4 grid is unchanged
    by the arithmetic widening and INT4 RHS storage stays packed in HBM.
    Scales [E,1,N] are applied AFTER the raw dot, never rounded to BF16.
    ``tiling`` is (M,K,N). A16 uses full K for the measured decode matrix
    families and retains prior defaults elsewhere. Integer activations use
    M=32, full K within the RHS buffer budget, and roughly N/2, except A8 with
    int8 codes whose whole 128-aligned (K, N) slab fits the 2 MiB buffer, at
    up to 1024 rows and on the measured prefill families up to 81,920 rows
    (``_a8_slab_tiles``): one slab per grid step, both scales applied in the
    kernel into f32, the tail mask and the cast in XLA.
    Group sizes must be nonnegative and sum to at most M. Trailing padding has
    zero output/cotangents, including when every group is empty.
    """
    if activation_bits not in (4, 8, 16):
        raise ValueError("activation_bits must be 4, 8, or 16")
    if activation_bits == 4 and codes.dtype != jnp.dtype(jnp.int4):
        raise ValueError("activation_bits=4 requires int4 weights")
    if lhs.dtype != jnp.dtype(jnp.bfloat16):
        raise TypeError("Pallas CHANNELWISE requires bfloat16 lhs; use platform='xla' for other dtypes")
    if codes.dtype not in (jnp.dtype(jnp.int4), jnp.dtype(jnp.int8)):
        raise TypeError("codes must have signed int4 or int8 dtype")
    if lhs.ndim != 2 or codes.ndim != 3 or lhs.shape[1] != codes.shape[1]:
        raise ValueError("expected lhs[M,K] and codes[E,K,N]")
    if scales.shape != (codes.shape[0], 1, codes.shape[2]):
        raise ValueError("expected scales[E,1,N]")
    if not jnp.issubdtype(scales.dtype, jnp.floating):
        raise TypeError("scales must be floating point")
    if group_sizes.shape != (codes.shape[0],) or not jnp.issubdtype(group_sizes.dtype, jnp.integer):
        raise ValueError("expected integer group_sizes[E]")
    if not jnp.issubdtype(jnp.dtype(preferred_element_type), jnp.floating):
        raise TypeError("preferred_element_type must be floating point")
    slab_epilogue = False
    if tiling is None:
        if activation_bits == 16:
            tiling = _weight_only_streaming_tiles(lhs.shape[0], *codes.shape, codes.dtype)
        else:
            tiling = _a8_slab_tiles(lhs.shape[0], activation_bits, *codes.shape, codes.dtype)
            slab_epilogue = tiling is not None
            if tiling is None:
                tiling = _integer_streaming_tiles(lhs.shape[1], codes.shape[2], codes.dtype)
    if len(tiling) != 3 or any(not isinstance(t, int) or t <= 0 for t in tiling):
        raise ValueError("tiling must be a positive (M,K,N) integer tuple")
    if tiling[1] % 128 or tiling[2] % 128:
        raise ValueError("Pallas tiling K and N must be multiples of 128")
    if jax.default_backend() != "tpu":
        raise ValueError("Pallas CHANNELWISE requires TPU hardware; use platform='xla' elsewhere")
    tiles = TileSizes(tile_m=tiling[0], tile_k=tiling[1], tile_n=tiling[2])

    if activation_bits != 16 and tiling[0] % 32:
        raise ValueError("integer activation tiling M must be a multiple of 32")

    @jax.custom_jvp
    def apply(x, weights, s, sizes):
        if activation_bits == 16:
            # Pad rows to the sublane-safe multiple the integer path already uses: the v3 kernel's zero-fill
            # reshapes its output into (-1, sublane, lanes), which fails when M is not a multiple of the
            # sublane tiling (e.g. DeepSeek-V4 325 tokens x top-6 = 1950 rows). Padded rows sit past
            # sum(sizes), so they are zero and sliced off.
            padded_m = ((x.shape[0] + 31) // 32) * 32
            raw = grouped_matmulv3_pallas_impl(
                jnp.pad(x, ((0, padded_m - x.shape[0]), (0, 0))),
                weights,
                sizes,
                maybe_quantize_lhs=False,
                tile_info=tiles,
                preferred_element_type=jnp.float32,
                acc_dtype=jnp.float32,
            )[: x.shape[0]]
        else:
            from ...._xla.quantized_matmul._integer_quantization import quantize_rows

            valid = jnp.arange(x.shape[0]) < jnp.sum(sizes)
            q, activation_scale = quantize_rows(jnp.where(valid[:, None], x, 0), activation_bits)
            # INT4 grid is preserved exactly; only arithmetic widens to INT8.
            # RHS stays packed in HBM and is widened inside VMEM as needed.
            q = q.astype(jnp.int8)
            padded_m = ((x.shape[0] + 31) // 32) * 32
            q = jnp.pad(q, ((0, padded_m - x.shape[0]), (0, 0)))
            if slab_epilogue:
                # Whole expert slabs: both scales are applied in the kernel after the complete int32 contraction,
                # in the XLA entry's order ((acc * row scale) * channel scale), into an f32 output. The tail mask
                # and the final cast stay in XLA, written as the XLA entry ends, so the graph after this op is the
                # XLA entry's. A launch needs at least one grid step: with no routed row, group 0 owns the zero
                # row 0, whose output the mask discards.
                kernel_sizes = jnp.where(
                    (jnp.arange(sizes.shape[0]) == 0) & (jnp.sum(sizes) == 0), jnp.ones_like(sizes), sizes
                )
                scaled = grouped_matmulv3_pallas_impl(
                    q,
                    weights,
                    kernel_sizes,
                    maybe_quantize_lhs=False,
                    tile_info=tiles,
                    preferred_element_type=jnp.float32,
                    acc_dtype=jnp.int32,
                    output_row_scale=jnp.pad(activation_scale, ((0, padded_m - x.shape[0]), (0, 0)), constant_values=1),
                    output_channel_scale=s.astype(jnp.float32),
                )[: x.shape[0]]
                return jnp.where(valid[:, None], scaled, 0).astype(preferred_element_type)
            fuse_output = (
                1280 <= x.shape[0] <= 81920
                and weights.shape in ((128, 2560, 1280), (128, 640, 2560))
                and jnp.dtype(preferred_element_type) in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32))
            )
            if fuse_output:
                from jax.experimental.pallas import tpu as pltpu

                fuse_output = pltpu.get_tpu_info().chip_version.value == "v5p"
            has_work = jnp.sum(sizes) > 0

            def _fused_epilogue(_):
                padded_scale = jnp.pad(activation_scale, ((0, padded_m - x.shape[0]), (0, 0)), constant_values=1)
                return grouped_matmulv3_pallas_impl(
                    q,
                    weights,
                    sizes,
                    maybe_quantize_lhs=False,
                    tile_info=tiles,
                    preferred_element_type=preferred_element_type,
                    acc_dtype=jnp.int32,
                    output_row_scale=padded_scale,
                    output_channel_scale=s.astype(jnp.float32),
                )[: x.shape[0]]

            def _unfused(_):
                return grouped_matmulv3_pallas_impl(
                    q,
                    weights,
                    sizes,
                    maybe_quantize_lhs=False,
                    tile_info=tiles,
                    preferred_element_type=jnp.int32,
                    acc_dtype=jnp.int32,
                )[: x.shape[0]]

            # All-zero group sizes would launch an empty pipeline grid; the
            # contract requires an all-zero output instead.
            if fuse_output:
                return jax.lax.cond(
                    has_work,
                    _fused_epilogue,
                    lambda _: jnp.zeros((x.shape[0], weights.shape[2]), preferred_element_type),
                    None,
                )
            raw = (
                jax.lax.cond(
                    has_work,
                    _unfused,
                    lambda _: jnp.zeros((x.shape[0], weights.shape[2]), jnp.int32),
                    None,
                ).astype(jnp.float32)
                * activation_scale
            )
        row_scales = _expand_channel_scales(s, sizes, x.shape[0])
        scaled = raw * row_scales
        valid = jnp.arange(x.shape[0]) < jnp.sum(sizes)
        return jnp.where(valid[:, None], scaled, 0).astype(preferred_element_type)

    # All arrays, including frozen codes/groups, are operands. Never close over
    # traced integer arrays: AD of an already-jitted call must remain valid.
    @apply.defjvp
    def apply_jvp(primals, tangents):
        def reference(x, weights, s, sizes):
            return _reference(
                x, weights, s, sizes, activation_bits=activation_bits, preferred_element_type=preferred_element_type
            )

        _, tangent = jax.jvp(reference, primals, tangents)
        return apply(*primals), tangent

    return apply(lhs, codes, scales, group_sizes)


__all__ = ("grouped_matmul_channelwise",)
