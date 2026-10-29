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

"""XLA reference for the chunked KDA forward with per-channel decay, plus the shared per-chunk math.

The recurrence (per batch row and head, per token ``t``)::

    S_t = diag(exp(g_t)) S_{t-1};   S_t += k_t (beta_t (v_t - k_t^T S_t))^T;   o_t = S_t^T q_t

is evaluated chunk by chunk. Inside a chunk of ``cs`` tokens, with ``gc`` the within-chunk prefix sums of ``g``:

* ``pair[i, j] = sum_k k_beta[i, k] k[j, k] exp(gc[i, k] - gc[j, k])`` on ``j < i`` and
  ``T = (I + pair)^-1`` (unit lower triangular),
* ``value_local = T v_beta``, ``k_cumdecay = T (k_beta exp(gc))``,
* ``v_new = value_local - k_cumdecay S``, ``out = (q exp(gc)) S + attn_qk v_new`` with
  ``attn_qk[i, j] = sum_k q[i, k] k[j, k] exp(gc[i, k] - gc[j, k])`` on ``j <= i``,
* ``S <- S * exp(gc[-1])[:, None] + (k exp(gc[-1] - gc))^T v_new``.

Packed rows (``segment_ids``): a token whose id differs from the previous token's starts a new run with a zero
state; negative ids are padding. Inputs of padding tokens are zeroed, which makes them exactly inert (zero output,
identity decay, no state write). Every chunk carries a per-token *local run index* ``run - run_prev`` (``run`` the
global run id, ``run_prev`` the run that was open when the chunk started): local run 0 continues the carried state,
only the chunk's last run writes the outgoing state, and the carried state survives only when the whole chunk is
local run 0. With ``continue_initial_state`` padding does not break a run and the first live run inherits
``initial_state`` (cached continuation); otherwise the first token always starts from zero.

This module is shared by the Pallas TPU backend (which runs :func:`chunk_core` on one head's VMEM tiles), so every
function used in-kernel sticks to Mosaic-lowerable primitives: static iota masks, lane reductions, 2-D dots, and
the iota-masked block-recursive inverse :func:`unit_lower_inverse_masked`. The XLA reference itself uses an exact
batched ``triangular_solve`` for the chunk inverse instead, so the two backends do not share that step.
"""

from __future__ import annotations

import functools
import math
import typing as tp

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

LANES = 128
SUBLANES = 8
INVERSE_LEAF = 8
META_ROW_RUN = 0

PrecisionLike = str | lax.Precision | None


def resolve_precision(precision: PrecisionLike) -> lax.Precision:
    """Map ``"highest"`` / ``"default"`` / a ``lax.Precision`` / ``None`` (HIGHEST) to ``lax.Precision``.

    Args:
        precision: Requested precision.

    Returns:
        The ``lax.Precision`` value.

    Raises:
        ValueError: For an unknown precision name.
    """
    if precision is None:
        return lax.Precision.HIGHEST
    if isinstance(precision, lax.Precision):
        return precision
    name = str(precision).strip().lower()
    if name == "highest":
        return lax.Precision.HIGHEST
    if name == "default":
        return lax.Precision.DEFAULT
    raise ValueError(f"precision must be 'highest' or 'default', got {precision!r}")


def validate_options(
    chunk_size: int,
    precision: PrecisionLike,
    backward: str,
    bwd_reduce: str,
    pairwise: str = "auto",
    decay_lower_bound: float | None = None,
) -> None:
    """Validate the static options shared by every ``kda_chunk`` backend.

    Args:
        chunk_size: Chunk length.
        precision: Precision of the non-inverse dots.
        backward: ``"pallas"`` or ``"xla"``.
        bwd_reduce: ``"vpu"`` or ``"mxu"``.
        pairwise: ``"auto"`` or ``"dense"``.
        decay_lower_bound: ``None`` or a finite log-decay bound ``<= 0``.

    Raises:
        ValueError: On any invalid option.
    """
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    resolve_precision(precision)
    if backward not in ("pallas", "xla"):
        raise ValueError(f"backward must be 'pallas' or 'xla', got {backward!r}")
    if bwd_reduce not in ("vpu", "mxu"):
        raise ValueError(f"bwd_reduce must be 'vpu' or 'mxu', got {bwd_reduce!r}")
    if pairwise not in ("auto", "dense"):
        raise ValueError(f"pairwise must be 'auto' or 'dense', got {pairwise!r}")
    validate_decay_lower_bound(decay_lower_bound)


def _nan_to_num(x: Array) -> Array:
    """``nan_to_num(x, 0, 0, 0)`` written with Mosaic-lowerable primitives."""
    return jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x))


DECAY_FACTOR_MAX_LOG = 80.0
"""Largest exponent a factored-decay term may reach (fp32 overflows past ~88.7; the margin covers the K-sum)."""


def companion_precision(precision: lax.Precision) -> lax.Precision:
    """Precision of the chunk inverse and the factored-decay weighted reductions for a given dot ``precision``.

    ``HIGHEST`` stays ``HIGHEST``. Below it (the bf16 ``"default"`` MXU passes) these terms run at ``HIGH`` (bf16x3):
    a single bf16 pass on the backward's decay reductions loses accuracy in the log-decay gradient, while ``HIGH``
    matches ``HIGHEST`` to within the bf16 floor that the other dots already set.

    Args:
        precision: Precision of the non-inverse dots.

    Returns:
        ``HIGHEST`` for ``HIGHEST``, otherwise ``HIGH``.
    """
    return lax.Precision.HIGHEST if precision == lax.Precision.HIGHEST else lax.Precision.HIGH


def validate_decay_lower_bound(decay_lower_bound: float | None) -> float | None:
    """Return the bound as a float, or raise unless it is ``None`` or a finite value ``<= 0``.

    Args:
        decay_lower_bound: Promised lower bound of every per-token log-decay ``g`` (``g >= bound``), or ``None``.

    Returns:
        The bound as a Python float, or ``None``.

    Raises:
        ValueError: For a positive or non-finite bound.
    """
    if decay_lower_bound is None:
        return None
    bound = float(decay_lower_bound)
    if not math.isfinite(bound) or bound > 0.0:
        raise ValueError(f"decay_lower_bound must be a finite log-decay bound <= 0, got {decay_lower_bound!r}")
    return bound


def pairwise_sub_block(decay_lower_bound: float | None, chunk_size: int) -> int:
    """Sub-block size of the factored (MXU) pairwise decay, or ``0`` for the dense ``[cs, cs, K]`` form.

    With every per-token log-decay ``g >= decay_lower_bound``, the pairwise decay ``exp(gc[i] - gc[j])`` of a row
    ``i`` in sub-block ``p`` factors as ``exp(gc[i] - r_p) * exp(r_p - gc[j])`` around the prefix-sum row ``r_p`` at
    the sub-block's middle, for every ``j`` up to the end of ``p``. Every factor and every product the masked matmuls
    form (including the masked-out ones above the diagonal) stays below ``exp((sub - 1) * |bound|)``, so the largest
    power-of-two divisor of ``chunk_size`` (at least 8) with ``(sub - 1) * |bound| <= DECAY_FACTOR_MAX_LOG`` is used.
    An unknown or looser bound keeps the dense form. Far-back columns may underflow to zero; their true decay is then
    below ``exp(-DECAY_FACTOR_MAX_LOG / 2)``.

    Args:
        decay_lower_bound: Lower bound of the per-token log-decay, or ``None``.
        chunk_size: Chunk length ``cs``.

    Returns:
        The sub-block size (a power of two in ``[8, cs]`` dividing ``cs``), or ``0``.

    Raises:
        ValueError: For an invalid bound (see :func:`validate_decay_lower_bound`).
    """
    bound = validate_decay_lower_bound(decay_lower_bound)
    cs = int(chunk_size)
    if bound is None or cs < 8:
        return 0
    sub = 1 << (cs.bit_length() - 1)
    while sub >= 8 and (cs % sub or (sub - 1) * -bound > DECAY_FACTOR_MAX_LOG):
        sub //= 2
    return sub if sub >= 8 else 0


def dot_nt(a: Array, b: Array, precision) -> Array:
    """``a @ b.T`` as a ``dot_general`` contracting the last dims of both."""
    return lax.dot_general(a, b, (((1,), (1,)), ((), ())), precision=precision)


def factored_decay(g_cumsum: Array, sub: int) -> tuple[Array, list[Array], list[Array]]:
    """Row factor, per-sub-block column factors and row masks of the factored pairwise decay.

    For ``i`` in sub-block ``p`` and ``j`` up to the end of ``p``: ``exp(gc[i] - gc[j]) = a[i] * b[p][j]``. Past the
    end of ``p`` the column factor is 1 (those pairs lie above the diagonal and every caller masks them).

    Args:
        g_cumsum: Within-chunk prefix sums of the log-decay ``[cs, K]``.
        sub: Sub-block size from :func:`pairwise_sub_block`.

    Returns:
        ``(a [cs, K], b [cs, K] per sub-block, rows [cs, K] bool per sub-block)``.
    """
    cs, kdim = g_cumsum.shape
    nb = cs // sub
    row = lax.broadcasted_iota(jnp.int32, (cs, kdim), 0)
    blk = lax.shift_right_logical(row, _log2_int(sub))
    refs = [jnp.sum(jnp.where(row == p * sub + sub // 2 - 1, g_cumsum, 0.0), axis=0, keepdims=True) for p in range(nb)]
    ref_rows = jnp.broadcast_to(refs[0], (cs, kdim))
    for p in range(1, nb):
        ref_rows = jnp.where(blk == p, refs[p], ref_rows)
    a = jnp.exp(g_cumsum - ref_rows)
    b = [jnp.exp(jnp.where(row < (p + 1) * sub, refs[p] - g_cumsum, 0.0)) for p in range(nb)]
    rows = [blk == p for p in range(nb)]
    return a, b, rows


def factored_pair_dot(x: Array, a: Array, k_cols: list[Array], rows: list[Array], precision) -> Array:
    """``out[i, j] = sum_k x[i, k] k[j, k] exp(gc[i, k] - gc[j, k])`` (valid for ``j`` up to ``i``'s sub-block end).

    Args:
        x: Row operand ``[cs, K]``.
        a: Row factor from :func:`factored_decay`.
        k_cols: ``k * b[p]`` per sub-block.
        rows: Sub-block row masks.
        precision: Dot precision.

    Returns:
        ``[cs, cs]``; entries above the diagonal are finite garbage the caller masks.
    """
    xa = x * a
    out = None
    for m, kp in zip(rows, k_cols, strict=True):
        t = dot_nt(jnp.where(m, xa, 0.0), kp, precision)
        out = t if out is None else out + t
    return out


def iota_rc(n: int) -> tuple[Array, Array]:
    """Row / column index grids of an ``[n, n]`` tile (int32)."""
    return lax.broadcasted_iota(jnp.int32, (n, n), 0), lax.broadcasted_iota(jnp.int32, (n, n), 1)


def _log2_int(n: int) -> int:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"expected a power of two, got {n}")
    return n.bit_length() - 1


def unit_lower_inverse_masked(
    a: Array, chunk_size: int, leaf: int = INVERSE_LEAF, precision: lax.Precision = lax.Precision.HIGHEST
) -> Array:
    """``(I - A)^-1`` for strictly lower-triangular ``A`` ``[cs, cs]``, block-recursive on full tiles.

    Leaves: the nilpotent squaring series ``(I + A)(I + A^2)(I + A^4)`` on the ``leaf x leaf`` diagonal blocks. The
    series is only used at that block size: on a full 64-wide chunk with slow decay and correlated keys its partial
    products grow far beyond the answer and cancel catastrophically in fp32, while at 8x8 the error stays ~1e-6.
    Merges: ``log2(cs / leaf)`` levels of the lower-triangular block identity
    ``[[T, 0], [-A_L, B]]^-1 = [[T^-1, 0], [B^-1 A_L T^-1, B^-1]]``. ``inv`` is kept as a full block-diagonal
    ``[cs, cs]`` matrix and each level adds ``inv @ mask_pair(A) @ inv``: the extra product terms are exact zeros,
    so no sub-block extraction, strided slice or concatenate is needed (Mosaic-lowerable).

    Args:
        a: Strictly lower-triangular ``[cs, cs]`` (entries on/above the diagonal are ignored).
        chunk_size: ``cs``, a power of two ``>= leaf``.
        leaf: Leaf block size.
        precision: Dot precision (``HIGHEST`` by default; the factored-decay path passes
            :func:`companion_precision`).

    Returns:
        ``(I - A)^-1`` (lower triangular, unit diagonal), non-finite entries zeroed.
    """
    if chunk_size < leaf or chunk_size & (chunk_size - 1):
        raise ValueError(f"chunk_size={chunk_size} must be a power of two >= {leaf}")
    hi = precision
    row, col = iota_rc(chunk_size)
    eye = (row == col).astype(jnp.float32)
    a32 = _nan_to_num(jnp.where(col < row, a.astype(jnp.float32), 0.0))

    leaf_shift = _log2_int(leaf)
    same_leaf = lax.shift_right_logical(row, leaf_shift) == lax.shift_right_logical(col, leaf_shift)
    blocks = jnp.where(same_leaf, a32, 0.0)
    result = eye + blocks
    power = blocks
    for _ in range(max(math.ceil(math.log2(leaf)) - 1, 0)):
        power = lax.dot(power, power, precision=hi)
        result = lax.dot(result, eye + power, precision=hi)
    inv = result

    size = leaf
    while size < chunk_size:
        s_shift = _log2_int(size)
        same_pair = lax.shift_right_logical(row, s_shift + 1) == lax.shift_right_logical(col, s_shift + 1)
        row_odd = (lax.shift_right_logical(row, s_shift) & 1) == 1
        col_even = (lax.shift_right_logical(col, s_shift) & 1) == 0
        lower_left = jnp.where(same_pair & row_odd & col_even, a32, 0.0)
        inv = inv + lax.dot(lax.dot(inv, lower_left, precision=hi), inv, precision=hi)
        size *= 2
    return _nan_to_num(jnp.where(col <= row, inv, 0.0))


def unit_lower_inverse_solve(a: Array) -> Array:
    """``(I - A)^-1`` for strictly lower-triangular ``A`` ``[..., cs, cs]`` by exact forward substitution.

    Args:
        a: Strictly lower-triangular matrices (entries on/above the diagonal are ignored).

    Returns:
        ``(I - A)^-1`` float32, same shape.
    """
    cs = a.shape[-1]
    eye = jnp.eye(cs, dtype=jnp.float32)
    a32 = jnp.where(jnp.tril(jnp.ones((cs, cs), bool), k=-1), a.astype(jnp.float32), 0.0)
    return lax.linalg.triangular_solve(
        eye - a32, jnp.broadcast_to(eye, a32.shape), left_side=True, lower=True, unit_diagonal=True
    )


class SegmentMasks(tp.NamedTuple):
    """Packed-row masks of one chunk (a pytree, so it can be vmapped).

    Attributes:
        incl: ``[cs, cs]`` bool, ``j <= i`` and same run.
        strict: ``[cs, cs]`` bool, ``j < i`` and same run.
        uses_state: ``[cs, 1]`` f32, token in local run 0 (reads the carried state).
        in_last: ``[cs, 1]`` f32, token in the chunk's last run (writes the outgoing state).
        keep_state: ``[1, 1]`` f32, 1.0 iff the whole chunk is local run 0 (the carried state survives).
    """

    incl: Array
    strict: Array
    uses_state: Array
    in_last: Array
    keep_state: Array


def segment_masks(local_run: Array, chunk_size: int) -> SegmentMasks:
    """Masks from a chunk's local run indices given as a ROW ``[1, cs]`` (small exact integers, any float dtype).

    The column orientation is built as ``sum(eye * row, axis=1)`` (exact: one nonzero per row), so no transpose or
    relayout is needed in-kernel. Local run indices are nondecreasing, so the row max is the last run.

    Args:
        local_run: ``[1, cs]`` local run index of every token of the chunk.
        chunk_size: ``cs``.

    Returns:
        The chunk's :class:`SegmentMasks`.
    """
    row, col = iota_rc(chunk_size)
    eye = (row == col).astype(jnp.float32)
    ri = local_run.astype(jnp.float32)
    ri_col = jnp.sum(eye * ri, axis=1, keepdims=True)
    last = jnp.max(ri, axis=1, keepdims=True)
    same = ri_col == ri
    return SegmentMasks(
        incl=(col <= row) & same,
        strict=(col < row) & same,
        uses_state=(ri_col == 0.0).astype(jnp.float32),
        in_last=(ri_col == last).astype(jnp.float32),
        keep_state=(last == 0.0).astype(jnp.float32),
    )


def chunk_core(
    state: Array,
    q: Array,
    k: Array,
    k_beta: Array,
    v_beta: Array,
    g_cumsum: Array,
    g_last: Array,
    seg: SegmentMasks | None,
    precision: lax.Precision,
    inverse: str = "masked",
    sub: int = 0,
) -> tuple[Array, Array]:
    """One head, one chunk, on 2-D tiles: the per-chunk KDA math (see the module docstring).

    Dense form (``sub == 0``): the pairwise decay ``exp(gc[i] - gc[j])`` is masked to the static lower triangle in
    log space before the ``exp`` (3-D iotas; no 2-D -> 3-D mask relayout); same-run masking is applied to the two
    ``[cs, cs]`` results after the K-reduction, which is exact because masked entries are zeroed either way. Both
    pairwise contractions are fp32 VPU multiply + lane reduce over a ``[cs, cs, K]`` tensor.

    Factored form (``sub > 0``, see :func:`pairwise_sub_block`): both pairwise contractions are ``cs / sub`` masked
    MXU matmuls on per-sub-block decay factors (:func:`factored_pair_dot`); no ``[cs, cs, K]`` tensor exists.
    ``pair`` (it feeds the inverse) runs at HIGHEST, ``attn_qk`` at ``precision``, and the masked inverse at
    :func:`companion_precision`.

    Args:
        state: Carried state ``[K, V]`` fp32.
        q: Scaled (and L2-normalised) queries ``[cs, K]``.
        k: (L2-normalised) keys ``[cs, K]``.
        k_beta: ``k * beta`` ``[cs, K]``.
        v_beta: ``v * beta`` ``[cs, V]``.
        g_cumsum: Within-chunk prefix sums of the per-channel log-decay ``[cs, K]``.
        g_last: Its last row ``[1, K]``.
        seg: ``None`` (unsegmented) or the chunk's :class:`SegmentMasks`.
        precision: Precision of the non-inverse dots (dense form: the inverse is always HIGHEST / exact).
        inverse: ``"masked"`` (:func:`unit_lower_inverse_masked`, power-of-two ``cs``) or ``"solve"``
            (:func:`unit_lower_inverse_solve`, XLA only).
        sub: Factored-decay sub-block size, or ``0`` for the dense form.

    Returns:
        ``(new_state [K, V], core_out [cs, V])`` fp32.
    """
    cs, kdim = q.shape
    row, col = iota_rc(cs)
    if seg is None:
        incl, strict = col <= row, col < row
    else:
        incl, strict = seg.incl, seg.strict

    if sub:
        fa, fb, frows = factored_decay(g_cumsum, sub)
        k_cols = [k * bp for bp in fb]
        inv_precision = companion_precision(precision)

        def pair_dot(x: Array, prec) -> Array:
            return factored_pair_dot(x, fa, k_cols, frows, prec)

    else:
        i3 = lax.broadcasted_iota(jnp.int32, (cs, cs, kdim), 0)
        j3 = lax.broadcasted_iota(jnp.int32, (cs, cs, kdim), 1)
        g_diff = jnp.where(j3 <= i3, g_cumsum[:, None, :] - g_cumsum[None, :, :], -jnp.inf)
        decay_mask = jnp.exp(g_diff)
        inv_precision = lax.Precision.HIGHEST

        def pair_dot(x: Array, prec) -> Array:
            del prec  # exact fp32 VPU reduction
            return jnp.sum(x[:, None, :] * k[None, :, :] * decay_mask, axis=-1)

    pair = pair_dot(k_beta, lax.Precision.HIGHEST)
    a = jnp.where(strict, -pair, 0.0)
    if inverse == "masked":
        attn = unit_lower_inverse_masked(a, cs, precision=inv_precision)
    else:
        attn = unit_lower_inverse_solve(a)

    exp_gc = jnp.exp(g_cumsum)
    value_local = lax.dot(attn, v_beta, precision=precision)
    k_cumdecay = lax.dot(attn, k_beta * exp_gc, precision=precision)
    v_prime = lax.dot(k_cumdecay, state, precision=precision)
    attn_inter = lax.dot(q * exp_gc, state, precision=precision)
    if seg is not None:
        v_prime = v_prime * seg.uses_state
        attn_inter = attn_inter * seg.uses_state
    v_new = value_local - v_prime

    attn_qk = jnp.where(incl, pair_dot(q, precision), 0.0)
    core_out = attn_inter + lax.dot(attn_qk, v_new, precision=precision)

    k_scaled = k * jnp.exp(g_last - g_cumsum)
    krow, kcol = iota_rc(kdim)
    exp_last_col = jnp.sum((krow == kcol).astype(jnp.float32) * jnp.exp(g_last), axis=1, keepdims=True)  # [K, 1]
    new_state = state * exp_last_col
    if seg is not None:
        k_scaled = k_scaled * seg.in_last
        new_state = jnp.where(seg.keep_state > 0.5, new_state, 0.0)
    new_state = new_state + lax.dot(k_scaled.T, v_new, precision=precision)
    return new_state, core_out


def l2norm(x: Array, eps: float = 1e-6) -> Array:
    """``x * rsqrt(sum(x * x) + eps)`` over the trailing axis."""
    return x * lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def local_run_index(
    segment_ids: Array, seq_len: int, pad_size: int, chunk_size: int, continue_initial_state: bool
) -> tuple[Array, Array]:
    """Per-chunk local run indices and token validity from packed segment ids.

    Args:
        segment_ids: ``[B, L']`` ids (negative = padding); truncated / ``-1``-padded to ``seq_len``.
        seq_len: ``L``.
        pad_size: Chunk padding appended after ``L`` (continues the last run, inputs zero).
        chunk_size: ``cs``.
        continue_initial_state: Padding does not break runs and the first live run inherits the initial state.

    Returns:
        ``(local_run [B, C, cs] int32, valid [B, L] bool)``.
    """
    seg = jnp.asarray(segment_ids)[:, :seq_len].astype(jnp.int32)
    batch = seg.shape[0]
    if seg.shape[1] < seq_len:
        seg = jnp.pad(seg, ((0, 0), (0, seq_len - seg.shape[1])), constant_values=-1)
    valid = seg >= 0
    if continue_initial_state:
        idx = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len))
        last_valid = lax.associative_scan(jnp.maximum, jnp.where(valid, idx, -1), axis=1)
        previous = jnp.concatenate([jnp.full((batch, 1), -1, jnp.int32), last_valid[:, :-1]], axis=1)
        previous_segment = jnp.take_along_axis(seg, jnp.maximum(previous, 0), axis=1)
        change = valid & (previous >= 0) & (seg != previous_segment)
        run = 1 + jnp.cumsum(change.astype(jnp.int32), axis=1)
    else:
        change = jnp.concatenate([jnp.ones((batch, 1), bool), seg[:, 1:] != seg[:, :-1]], axis=1)
        run = jnp.cumsum(change.astype(jnp.int32), axis=1)
    if pad_size:
        run = jnp.pad(run, ((0, 0), (0, pad_size)), mode="edge")
    num_chunks = (seq_len + pad_size) // chunk_size
    run_c = run.reshape(batch, num_chunks, chunk_size)
    first = run_c[:, 0, 0] if continue_initial_state else run_c[:, 0, 0] - 1
    run_prev = jnp.concatenate([first[:, None], run_c[:, :-1, -1]], axis=1)
    return run_c - run_prev[:, :, None], valid


class PreparedInputs(tp.NamedTuple):
    """Fp32, chunk-padded kernel inputs (``Lp = L + pad`` a multiple of ``chunk_size``).

    Attributes:
        q: Scaled (``K**-0.5``), optionally L2-normalised queries ``[B, H, Lp, K]``.
        k: Optionally L2-normalised keys ``[B, H, Lp, K]``.
        v: Values ``[B, H, Lp, V]``.
        beta: Gates ``[B, H, Lp]``.
        g: Per-channel log-decay ``[B, H, Lp, K]``.
        state: Initial state ``[B, H, K, V]``.
        local_run: ``[B, C, cs]`` int32 local run indices, or ``None`` without segments.
    """

    q: Array
    k: Array
    v: Array
    beta: Array
    g: Array
    state: Array
    local_run: Array | None


def prepare_inputs(
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    initial_state: Array | None,
    segment_ids: Array | None,
    chunk_size: int,
    use_qk_l2norm: bool,
    continue_initial_state: bool,
) -> PreparedInputs:
    """Elementwise prologue shared by every backend (plain JAX; differentiates on its own).

    Args:
        query: ``[B, H, L, K]``.
        key: ``[B, H, L, K]``.
        value: ``[B, H, L, V]``.
        beta: ``[B, H, L]``.
        decay: ``[B, H, L, K]`` per-channel log-decay (``<= 0``).
        initial_state: Optional ``[B, H, K, V]``.
        segment_ids: Optional packed ids ``[B, L]``.
        chunk_size: Chunk length.
        use_qk_l2norm: L2-normalise queries and keys.
        continue_initial_state: See :func:`local_run_index`.

    Returns:
        :class:`PreparedInputs`.
    """
    B, H, L, K = query.shape
    V = value.shape[-1]
    if use_qk_l2norm:
        query = l2norm(query.astype(jnp.float32))
        key = l2norm(key.astype(jnp.float32))
    q = query.astype(jnp.float32) * (1.0 / (K**0.5))
    k = key.astype(jnp.float32)
    v = value.astype(jnp.float32)
    b = beta.astype(jnp.float32)
    g = decay.astype(jnp.float32)
    pad_size = (chunk_size - L % chunk_size) % chunk_size
    local_run = None
    if segment_ids is not None:
        local_run, valid = local_run_index(segment_ids, L, pad_size, chunk_size, continue_initial_state)
        v4 = valid[:, None, :, None]
        q, k, v, g = (jnp.where(v4, x, 0.0) for x in (q, k, v, g))
        b = jnp.where(valid[:, None, :], b, 0.0)
    if pad_size:
        p4 = ((0, 0), (0, 0), (0, pad_size), (0, 0))
        q, k, v, g = (jnp.pad(x, p4) for x in (q, k, v, g))
        b = jnp.pad(b, ((0, 0), (0, 0), (0, pad_size)))
    if initial_state is None:
        state = jnp.zeros((B, H, K, V), jnp.float32)
    else:
        state = initial_state.astype(jnp.float32)
    return PreparedInputs(q, k, v, b, g, state, local_run)


def _to_chunks(x: Array, chunk_size: int) -> Array:
    """``[B, H, Lp, D] -> [C, B, H, cs, D]``."""
    B, H, Lp, D = x.shape
    return x.reshape(B, H, Lp // chunk_size, chunk_size, D).transpose(2, 0, 1, 3, 4)


def batched_chunk_step(
    state: Array,
    q: Array,
    k: Array,
    v: Array,
    beta: Array,
    g: Array,
    local_run: Array | None,
    precision: lax.Precision,
    inverse: str,
    sub: int = 0,
) -> tuple[Array, Array]:
    """One chunk for every ``(B, H)``: prefix sums + :func:`chunk_core` vmapped over heads and rows.

    Args:
        state: ``[B, H, K, V]``.
        q: ``[B, H, cs, K]``.
        k: ``[B, H, cs, K]``.
        v: ``[B, H, cs, V]``.
        beta: ``[B, H, cs]``.
        g: Raw (not prefix-summed) log-decay ``[B, H, cs, K]``.
        local_run: ``[B, cs]`` local run indices or ``None``.
        precision: Non-inverse dot precision.
        inverse: Chunk-inverse method (see :func:`chunk_core`).
        sub: Factored-decay sub-block size, or ``0`` for the dense form (see :func:`pairwise_sub_block`).

    Returns:
        ``(new_state [B, H, K, V], core_out [B, H, cs, V])``.
    """
    cs = q.shape[-2]
    gcs = jnp.cumsum(g, axis=-2)
    seg = None
    if local_run is not None:
        seg = jax.vmap(lambda r: segment_masks(r[None, :], cs))(local_run)
    core = functools.partial(chunk_core, precision=precision, inverse=inverse, sub=sub)
    over_h = jax.vmap(core, in_axes=(0, 0, 0, 0, 0, 0, 0, None))
    over_b = jax.vmap(over_h, in_axes=(0, 0, 0, 0, 0, 0, 0, 0 if seg is not None else None))
    kb = k * beta[..., None]
    vb = v * beta[..., None]
    return over_b(state, q, k, kb, vb, gcs, gcs[..., -1:, :], seg)


def kda_chunk_xla(
    query: Array,
    key: Array,
    value: Array,
    beta: Array,
    decay: Array,
    initial_state: Array | None = None,
    segment_ids: Array | None = None,
    *,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    continue_initial_state: bool = False,
    return_chunk_states: bool = False,
    precision: PrecisionLike = "highest",
    decay_lower_bound: float | None = None,
    pairwise: str = "auto",
) -> tuple[Array, ...]:
    """XLA chunked KDA: ``lax.scan`` over chunks, each step rematerialised under autodiff.

    The scan carry is the ``[B, H, K, V]`` fp32 state; the per-chunk ``[cs, cs, K]`` decay tensors (or, with a
    ``decay_lower_bound``, their factored form) are rebuilt in the backward (``jax.checkpoint`` around the step), so
    residuals stay at the chunk inputs plus one state per chunk. The chunk inverse is an exact batched
    ``triangular_solve``.

    Args:
        query: ``[B, H, L, K]``.
        key: ``[B, H, L, K]``.
        value: ``[B, H, L, V]``.
        beta: ``[B, H, L]``.
        decay: ``[B, H, L, K]`` per-channel log-decay.
        initial_state: Optional ``[B, H, K, V]``.
        segment_ids: Optional packed ids ``[B, L]`` (negative = padding).
        chunk_size: Chunk length.
        use_qk_l2norm: L2-normalise queries and keys.
        continue_initial_state: Cached-continuation segment semantics (see :func:`local_run_index`).
        return_chunk_states: Also return the state after every chunk ``[B, C, H, K, V]``.
        precision: Precision of the non-inverse dots.
        decay_lower_bound: Promise that every log-decay ``g >= decay_lower_bound`` (e.g. a safe-gate bound). It
            enables the factored pairwise decay (see :func:`pairwise_sub_block`); ``None`` keeps the dense form.
        pairwise: ``"auto"`` (factored whenever the bound allows it) or ``"dense"``.

    Returns:
        ``(out [B, H, L, V] fp32, final_state [B, H, K, V] fp32)`` (+ ``chunk_states``).
    """
    L = query.shape[2]
    prec = resolve_precision(precision)
    sub = pairwise_sub_block(decay_lower_bound, chunk_size) if pairwise == "auto" else 0
    p = prepare_inputs(
        query, key, value, beta, decay, initial_state, segment_ids, chunk_size, use_qk_l2norm, continue_initial_state
    )
    xs = {
        "q": _to_chunks(p.q, chunk_size),
        "k": _to_chunks(p.k, chunk_size),
        "v": _to_chunks(p.v, chunk_size),
        "beta": _to_chunks(p.beta[..., None], chunk_size)[..., 0],
        "g": _to_chunks(p.g, chunk_size),
    }
    if p.local_run is not None:
        xs["run"] = p.local_run.transpose(1, 0, 2)  # [C, B, cs]

    def step(state, x):
        new_state, out = batched_chunk_step(
            state, x["q"], x["k"], x["v"], x["beta"], x["g"], x.get("run"), prec, inverse="solve", sub=sub
        )
        return new_state, ((out, new_state) if return_chunk_states else out)

    final_state, ys = lax.scan(jax.checkpoint(step, prevent_cse=False), p.state, xs)
    out = ys[0] if return_chunk_states else ys
    C, B, H, cs, V = out.shape
    out = out.transpose(1, 2, 0, 3, 4).reshape(B, H, C * cs, V)[:, :, :L, :]
    if return_chunk_states:
        return out, final_state, ys[1].transpose(1, 0, 2, 3, 4)  # [B, C, H, K, V]
    return out, final_state
