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
    M=32, full K within the RHS buffer budget, and roughly N/2.
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
    if tiling is None:
        if activation_bits == 16:
            tiling = _weight_only_streaming_tiles(lhs.shape[0], *codes.shape, codes.dtype)
        else:
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
