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

_SLAB_MAX_BYTES = 2 * 1024 * 1024
"""HBM bytes of one RHS pipeline buffer (three are in flight: 6 MiB of VMEM), the budget every tile rule below keeps.
Bytes count the codes at their own width (``jax.dtypes.itemsize_bits``): an int4 tile holds twice the int8 elements."""


def _fits_rhs_buffer(k, n, dtype):
    """Whether a ``(k, n)`` tile of ``dtype`` codes fits one RHS buffer (``_SLAB_MAX_BYTES``), counted in bits."""
    return k * n * jax.dtypes.itemsize_bits(dtype) <= 8 * _SLAB_MAX_BYTES


def _integer_streaming_tiles(k, n, dtype):
    """Bound each of three RHS buffers to 2 MiB, including very wide N."""
    tk = ((k + 127) // 128) * 128
    tn = max(128, ((n + 255) // 256) * 128)
    while not _fits_rhs_buffer(tk, tn, dtype):
        if tk > 128:
            tk = max(128, ((tk // 2 + 127) // 128) * 128)
        else:
            tn = max(128, ((tn // 2 + 127) // 128) * 128)
    return (32, tk, tn)


_A8_SLAB_MAX_ROWS = 1024
"""Rows (decode buckets) up to which A8 experts stream one whole (K, N) slab per grid step with 32-row tiles."""
_A8_PREFILL_SLAB_MAX_ROWS = 81920
"""Largest measured row buffer: the deferred MoE dispatch's window of one 32,768-token step at ep4 (top-8)."""
_A8_PREFILL_TILE_M = ((55, 64), (200, 128), (400, 256))
"""``(max rows per expert, tile_m)`` of the prefill slab tiles; 512 above the last bound (see ``_a8_slab_tiles``)."""
_A8_SLAB_MAX_TILE_ELEMENTS = 512 * 4096
"""Largest ``tile_m * K`` and ``tile_m * N`` of a slab tile: the measured 512-row tiles of the ``K = 4096`` gate/up and
the ``N = 4096`` down slabs. A slab with a wider ``K`` or ``N`` halves its row tile until both products fit, keeping
the row block and the output tile within the sizes that were compiled and measured."""


def _a8_slab_tiles(m, activation_bits, e, k, n, dtype):
    """``(tile_m, K, N)`` tiles holding one whole expert slab, or ``None`` (keep ``_integer_streaming_tiles``).

    Applies to A8 activations with int8 or int4 codes whose ``(K, N)`` slab is 128-aligned and fits one RHS buffer of
    ``_SLAB_MAX_BYTES`` counted at the codes' own bit width (an int4 slab holds up to twice the elements of an int8
    one; the kernel widens it to int8 in VMEM), for any expert count: at up to ``_A8_SLAB_MAX_ROWS`` rows with 32-row
    tiles, and up to ``_A8_PREFILL_SLAB_MAX_ROWS`` rows with the ``_A8_PREFILL_TILE_M`` row tile of the buffer's rows
    per expert ``m / E``, halved while ``tile_m * max(K, N)`` exceeds ``_A8_SLAB_MAX_TILE_ELEMENTS``. Pure shape
    policy: it queries no device. Experts without rows are never read, and the entry applies both scales inside the
    kernel. The row tiles were measured on TPU v5p, one chip, on the per-chip expert shapes of GLM-5.3-Flash served at
    ep4 x tp4 (int8 [72, 4096, 512] gate/up, [72, 512, 4096] down), every output bitwise equal to the XLA entry's:

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
    if activation_bits != 8 or jnp.dtype(dtype) not in (jnp.dtype(jnp.int8), jnp.dtype(jnp.int4)) or m <= 0:
        return None
    if k % 128 or n % 128 or not _fits_rhs_buffer(k, n, dtype):
        return None
    if m <= _A8_SLAB_MAX_ROWS:
        return (32, k, n)
    if m > _A8_PREFILL_SLAB_MAX_ROWS:
        return None
    rows_per_expert = m / e
    tile_m = next((tile for bound, tile in _A8_PREFILL_TILE_M if rows_per_expert <= bound), 512)
    while tile_m > 32 and tile_m * max(k, n) > _A8_SLAB_MAX_TILE_ELEMENTS:
        tile_m //= 2
    return (tile_m, k, n)


def _slab_core_count(n):
    """TensorCores the A8 whole-slab kernel splits its ``N`` output columns over.

    Every TensorCore of a megacore device (``TpuInfo.num_cores``; 2 on TPU v5p) when ``N`` splits into 128-aligned
    column tiles, else 1. Each core streams its own column tile of every active expert's slab and writes its own output
    columns; every output element keeps the same int32 contraction and scale epilogue, so the output is unchanged.
    """
    from jax.experimental.pallas import tpu as pltpu

    cores = pltpu.get_tpu_info().num_cores
    return cores if cores > 1 and n % (cores * 128) == 0 else 1


_A16_DECODE_MAX_ROWS = 128
"""Rows up to which A16 full-K tiles use 16-row tiles (decode); 64-row tiles above, up to ``_A16_FULL_K_MAX_ROWS``."""
_A16_FULL_K_MAX_ROWS = 12288
"""Largest row count the A16 full-K tiles were measured at; above it A16 keeps the generic partial-K tiles."""
_INTEGER_PREFILL_ROWS = (1280, 81920)
"""Row range of the measured W4A4 prefill (and of the non-slab integer calls that write their scaled output from the
kernel instead of materializing the raw ``M x N`` int32 dot)."""
_A4_DECODE_MAX_ROWS = 128
"""Rows up to which A4 decode was measured faster on the Pallas entry than on XLA."""


def _weight_only_full_k_tiles(m, e, k, n, dtype):
    """A16 ``(tile_m, K, tile_n)`` tiles that read the whole contraction per grid step, or ``None``.

    ``tile_n`` is the widest 128-multiple dividing ``N`` whose ``(K, tile_n)`` tile of codes fits one RHS buffer
    (``_SLAB_MAX_BYTES`` counted at the codes' bit width, so an int4 tile may be twice as wide as an int8 one);
    ``tile_m`` is 16 up to ``_A16_DECODE_MAX_ROWS`` rows and 64 up to ``_A16_FULL_K_MAX_ROWS``. ``None`` when ``K`` or
    ``N`` is not 128-aligned, when no such ``tile_n`` exists, or outside ``1 .. _A16_FULL_K_MAX_ROWS`` rows. Pure shape
    policy for any expert count. The tiles were measured on TPU v5 (int4, bit-exact vs XLA) on the DeepSeek-V4.1 tp=4
    per-device experts: decode at 48 rows, gate/up [96,5120,2304] 127 -> 62 us, fused gate_up [96,5120,4608] 204 ->
    110 us, down [96,2304,5120] 107 -> 62 us; prefill with 64-row tiles, gate/up 2.05 -> 1.18 ms at 1950 rows and 5.43
    -> 1.36 ms at 12288, down 1.78 -> 1.21 and 4.85 -> 1.63 ms (prior generic (16,512,1024) tiles; XLA 2.55-3.17 ms).

    Args:
        m: Rows of ``lhs``.
        e: Expert count (unused: the tiles hold for any count).
        k: Contraction width ``K``.
        n: Output width ``N``.
        dtype: Dtype of the codes.

    Returns:
        ``(tile_m, K, tile_n)`` or ``None``.
    """
    del e
    if m <= 0 or m > _A16_FULL_K_MAX_ROWS or k % 128 or n % 128:
        return None
    widths = [t for t in range(128, n + 1, 128) if n % t == 0 and _fits_rhs_buffer(k, t, dtype)]
    if not widths:
        return None
    return (16 if m <= _A16_DECODE_MAX_ROWS else 64, k, widths[-1])


def _weight_only_streaming_tiles(m, e, k, n, dtype):
    """A16 tiles: ``_weight_only_full_k_tiles`` where they apply, else the generic partial-K tiles."""
    tiles = _weight_only_full_k_tiles(m, e, k, n, dtype)
    if tiles is not None:
        return tiles
    return (16, 512 if k >= 512 else 128, 1024 if n >= 1024 else 128)


def _a4_streaming_tiles(m, e, k, n, dtype):
    """A4 ``_integer_streaming_tiles`` in the row ranges where A4 was measured faster than XLA, else ``None``.

    Applies to int4 codes with a 128-aligned ``(K, N)``, for any expert count, at up to ``_A4_DECODE_MAX_ROWS`` rows
    and inside ``_INTEGER_PREFILL_ROWS``.
    """
    del e
    if jnp.dtype(dtype) != jnp.dtype(jnp.int4) or k % 128 or n % 128:
        return None
    if 0 < m <= _A4_DECODE_MAX_ROWS or _INTEGER_PREFILL_ROWS[0] <= m <= _INTEGER_PREFILL_ROWS[1]:
        return _integer_streaming_tiles(k, n, dtype)
    return None


def _pallas_path_tiles(m, activation_bits, e, k, n, dtype):
    """Tiles of the Pallas entry's measured-fast path for a call, or ``None`` (the call should stay on XLA).

    One rule per activation width, from the weight shape, the codes' bit width, 128-alignment, the RHS buffer budget
    and row bounds only: A8 ``_a8_slab_tiles`` (whole slab, int8 or int4 codes), A16 ``_weight_only_full_k_tiles``
    (full K), A4 ``_a4_streaming_tiles``. Pure shape policy: it queries no device.
    """
    if activation_bits == 8:
        return _a8_slab_tiles(m, activation_bits, e, k, n, dtype)
    if activation_bits == 16:
        if jnp.dtype(dtype) not in (jnp.dtype(jnp.int8), jnp.dtype(jnp.int4)):
            return None
        return _weight_only_full_k_tiles(m, e, k, n, dtype)
    if activation_bits == 4:
        return _a4_streaming_tiles(m, e, k, n, dtype)
    return None


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
    mask_padding=True,
):
    """Stream signed int4/int8 weights with explicit activation precision.

    A16 uses BF16 inputs and FP32 accumulation. A4/A8 quantize each input row,
    then use INT8 arithmetic with INT32 accumulation; the A4 grid is unchanged
    by the arithmetic widening and INT4 RHS storage stays packed in HBM.
    Scales [E,1,N] are applied AFTER the raw dot, never rounded to BF16.
    ``tiling`` is (M,K,N). A16 uses full K with the widest N tile that fits
    the RHS buffer at up to 12,288 rows (``_weight_only_full_k_tiles``) and
    the prior partial-K defaults elsewhere. Integer activations use M=32,
    full K within the RHS buffer budget, and roughly N/2, except A8 with int8
    or int4 codes whose whole 128-aligned (K, N) slab fits the 2 MiB buffer
    at the codes' bit width, at up to 81,920 rows with a row tile chosen from
    the rows per expert (``_a8_slab_tiles``): one slab per grid step, both
    scales applied in the kernel into f32, the tail mask and the cast in XLA.
    Group sizes must be nonnegative and sum to at most M. Trailing padding has
    zero output/cotangents, including when every group is empty.
    ``mask_padding=False`` allows the rows past ``sum(group_sizes)`` to be left
    unspecified (possibly non-finite): the A8 whole-slab path then writes the
    ``preferred_element_type`` output from the kernel, skipping the XLA tail
    mask and cast (the same values on every routed row). Other paths mask.
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
    slab_cores = _slab_core_count(tiling[2]) if slab_epilogue else 1
    tiles = TileSizes(tile_m=tiling[0], tile_k=tiling[1], tile_n=tiling[2] // slab_cores)

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
            if slab_epilogue and mask_padding:
                # Whole expert slabs: both scales are applied in the kernel after the complete int32 contraction,
                # in the XLA entry's order ((acc * row scale) * channel scale), into an f32 output. The tail mask
                # and the final cast stay in XLA, written as the XLA entry ends, so the graph after this op is the
                # XLA entry's. A launch needs at least one grid step: with no routed row, group 0 owns the zero
                # row 0, whose output the mask discards. That mask also zeroes every dispatch-tail row, so the kernel
                # skips its own zero-fill of those rows (zero_initialize=False), and on a megacore chip both
                # TensorCores take a column half of every slab (``_slab_core_count``).
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
                    zero_initialize=False,
                    output_row_scale=jnp.pad(activation_scale, ((0, padded_m - x.shape[0]), (0, 0)), constant_values=1),
                    output_channel_scale=s.astype(jnp.float32),
                    num_cores=slab_cores,
                )[: x.shape[0]]
                return jnp.where(valid[:, None], scaled, 0).astype(preferred_element_type)
            if slab_epilogue and not mask_padding:
                kernel_sizes = jnp.where(
                    (jnp.arange(sizes.shape[0]) == 0) & (jnp.sum(sizes) == 0), jnp.ones_like(sizes), sizes
                )
                # Routed rows get the same ((acc * row scale) * channel scale) value and the same final cast, done in
                # the kernel; rows past sum(sizes) stay unwritten (the caller allowed unspecified padding).
                return grouped_matmulv3_pallas_impl(
                    q,
                    weights,
                    kernel_sizes,
                    maybe_quantize_lhs=False,
                    tile_info=tiles,
                    preferred_element_type=preferred_element_type,
                    acc_dtype=jnp.int32,
                    zero_initialize=False,
                    output_row_scale=jnp.pad(activation_scale, ((0, padded_m - x.shape[0]), (0, 0)), constant_values=1),
                    output_channel_scale=s.astype(jnp.float32),
                    num_cores=slab_cores,
                )[: x.shape[0]]
            # Prefill-sized calls write the scaled output from the kernel instead of materializing the raw M x N
            # int32 dot (same ((acc * row scale) * channel scale) order and final cast as the unfused path).
            fuse_output = _INTEGER_PREFILL_ROWS[0] <= x.shape[0] <= _INTEGER_PREFILL_ROWS[1] and jnp.dtype(
                preferred_element_type
            ) in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32))
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
