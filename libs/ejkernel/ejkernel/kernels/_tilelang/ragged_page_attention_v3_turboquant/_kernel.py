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

"""TileLang prim_func builder for the TurboQuant page-update kernel (RPA v3).

The update kernel compresses new K/V tokens into TurboQuant page format and
writes the results back to the page arrays.  It does **not** perform attention;
the attention step delegates to
:func:`~.ragged_page_attention_v2_turboquant.ragged_page_attention_v2_turboquant`.

Compression algorithm per token
---------------------------------
For each new token at position ``(seq, logical_pos)`` mapped to physical page
``pp``, page offset ``po``, and head ``kh``:

1. **Normalise**: compute L2-norm of the raw key/value vector.
2. **Rotate**: apply the orthogonal rotation matrix ``R`` to the normalised vector.
3. **Quantise keys**:
   a. Codebook lookup (per dimension, nearest centroid in ``KeyCodebook``).
   b. Pack two 4-bit indices per byte into ``KeyIndicesOut``.
   c. Compute residual = rotated_key - centroid; record its L2-norm.
   d. Project residual with ``QJLProjection`` and store sign bits.
4. **Quantise values**: same codebook lookup, pack indices into ``ValueIndicesOut``.
5. Write norms to ``KeyNormsOut[:, 0]`` (original), ``[:, 1]`` (residual) and
   ``ValueNormsOut``.

Grid
----
``T.Kernel(num_kv_heads, page_size, num_pages)`` — one CTA per
``(head, page_offset_within_page, physical_page)`` triple.

Each CTA scans all sequences and all logical-page assignments to discover
whether its ``(physical_page, page_offset)`` slot is being updated in the
current batch.  If no matching token is found the existing page data is
copied through unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _dtype_str(dtype) -> str:
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
        jnp.dtype(jnp.uint8): "uint8",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported dtype for ragged_page_attention_v3_turboquant: {dtype}")
    return mapping[canonical]


def make_rpa_v3_turboquant_update_prim_func(
    *,
    total_tokens: int,
    max_num_seqs: int,
    num_kv_heads: int,
    num_pages: int,
    page_size: int,
    pages_per_seq: int,
    head_dim: int,
    packed_idx_dim: int,
    packed_sign_dim: int,
    qjl_dim: int,
    key_levels: int,
    value_levels: int,
    kv_dtype,
    norm_dtype,
    codebook_dtype,
    threads: int = 128,
):
    """Build the TurboQuant page-update kernel ``@T.prim_func``.

    Grid: ``T.Kernel(num_kv_heads, page_size, num_pages)``.

    Each CTA processes a single ``(head, page_offset, physical_page)`` slot.
    It scans all active sequences to determine whether the slot is being
    updated.  If yes, it compresses the new token; otherwise it copies the
    existing compressed data through unchanged.

    Args:
        total_tokens: Total new token count ``TQ``.
        max_num_seqs: Upper bound on sequence count ``NS``.
        num_kv_heads: ``HKV``.
        num_pages: Physical page pool size ``P``.
        page_size: Tokens per page ``PS``.
        pages_per_seq: Maximum pages per sequence ``PPS``.
        head_dim: Head dimension ``D``.
        packed_idx_dim: ``ceil(D/2)`` — bytes per token/head for index arrays.
        packed_sign_dim: ``ceil(qjl_dim/8)`` — bytes per token/head for sign arrays.
        qjl_dim: QJL projection dimension ``QJL``.
        key_levels: Codebook size for keys ``KL``.
        value_levels: Codebook size for values ``VL``.
        kv_dtype: Floating-point dtype of ``KNew``/``VNew``.
        norm_dtype: Dtype of norm output arrays.
        codebook_dtype: Dtype of ``Rotation``, ``QJLProjection``, and codebooks.
        threads: Threads per CTA (default 128).

    Returns:
        A TileLang ``@T.prim_func`` (``rpa_v3_tq_update``) with signature::

            (KNew, VNew,
             KeyIndicesIn, KeySignsIn, KeyNormsIn,
             ValueIndicesIn, ValueNormsIn,
             KVLens, BlockTables, QueryStartLoc, Distribution,
             Rotation, QJLProjection, KeyCodebook, ValueCodebook,
             KeyIndicesOut, KeySignsOut, KeyNormsOut,
             ValueIndicesOut, ValueNormsOut)
    """
    kv_ts = _dtype_str(kv_dtype)
    norm_ts = _dtype_str(norm_dtype)
    cb_ts = _dtype_str(codebook_dtype)
    accum = "float32"
    TQ, NS, HKV, P, PS, PPS, D, PID, PSD, QJL, KL, VL = (
        total_tokens,
        max_num_seqs,
        num_kv_heads,
        num_pages,
        page_size,
        pages_per_seq,
        head_dim,
        packed_idx_dim,
        packed_sign_dim,
        qjl_dim,
        key_levels,
        value_levels,
    )

    @T.prim_func
    def rpa_v3_tq_update(
        KNew: T.Tensor((TQ, HKV, D), kv_ts),
        VNew: T.Tensor((TQ, HKV, D), kv_ts),
        KeyIndicesIn: T.Tensor((P, PS, HKV, PID), "uint8"),
        KeySignsIn: T.Tensor((P, PS, HKV, PSD), "uint8"),
        KeyNormsIn: T.Tensor((P, PS, HKV, 2), norm_ts),
        ValueIndicesIn: T.Tensor((P, PS, HKV, PID), "uint8"),
        ValueNormsIn: T.Tensor((P, PS, HKV), norm_ts),
        KVLens: T.Tensor((NS,), "int32"),
        BlockTables: T.Tensor((NS * PPS,), "int32"),
        QueryStartLoc: T.Tensor((NS + 1,), "int32"),
        Distribution: T.Tensor((3,), "int32"),
        Rotation: T.Tensor((D, D), cb_ts),
        QJLProjection: T.Tensor((QJL, D), cb_ts),
        KeyCodebook: T.Tensor((KL,), cb_ts),
        ValueCodebook: T.Tensor((VL,), cb_ts),
        KeyIndicesOut: T.Tensor((P, PS, HKV, PID), "uint8"),
        KeySignsOut: T.Tensor((P, PS, HKV, PSD), "uint8"),
        KeyNormsOut: T.Tensor((P, PS, HKV, 2), norm_ts),
        ValueIndicesOut: T.Tensor((P, PS, HKV, PID), "uint8"),
        ValueNormsOut: T.Tensor((P, PS, HKV), norm_ts),
    ):
        with T.Kernel(HKV, PS, P, threads=threads) as (kh, po, pp):
            update_idx = T.alloc_fragment((1,), "int32")
            update_valid = T.alloc_fragment((1,), "int32")
            k_norm = T.alloc_fragment((D,), accum)
            v_norm = T.alloc_fragment((D,), accum)
            k_rot = T.alloc_fragment((D,), accum)
            v_rot = T.alloc_fragment((D,), accum)
            k_res = T.alloc_fragment((D,), accum)
            k_code = T.alloc_fragment((D,), "int32")
            v_code = T.alloc_fragment((D,), "int32")
            sum_key = T.alloc_fragment((1,), accum)
            sum_val = T.alloc_fragment((1,), accum)
            sum_res = T.alloc_fragment((1,), accum)
            inv_key = T.alloc_fragment((1,), accum)
            inv_val = T.alloc_fragment((1,), accum)
            dim_ref = T.alloc_fragment((1,), accum)
            _kv_ref = T.alloc_fragment((1,), kv_ts)
            _norm_ref = T.alloc_fragment((1,), norm_ts)
            _cb_ref = T.alloc_fragment((1,), cb_ts)
            _pid_ref = T.alloc_fragment((PID,), accum)
            _psd_ref = T.alloc_fragment((PSD,), accum)

            dim_ref[0] = T.Cast(accum, TQ + NS + HKV + P + PS + PPS + D + PID + PSD + QJL + KL + VL)
            update_idx[0] = 0
            update_valid[0] = 0
            num_seqs = T.Cast("int32", Distribution[2])
            for s in T.serial(NS):
                q_start = T.Cast("int32", QueryStartLoc[s])
                q_end = T.Cast("int32", QueryStartLoc[s + 1])
                q_len = q_end - q_start
                kv_len = T.Cast("int32", KVLens[s])
                write_start = kv_len - q_len
                for lp in T.serial(PPS):
                    phys = T.Cast("int32", BlockTables[s * PPS + lp])
                    kv_pos = lp * PS + po
                    rel = kv_pos - write_start
                    live = (s < num_seqs) & (phys == pp) & (rel >= 0) & (rel < q_len) & (kv_pos < kv_len)
                    update_idx[0] = T.if_then_else(live, q_start + rel, update_idx[0])
                    update_valid[0] = T.if_then_else(live, 1, update_valid[0])

            for b in T.Parallel(PID):
                KeyIndicesOut[pp, po, kh, b] = KeyIndicesIn[pp, po, kh, b]
                ValueIndicesOut[pp, po, kh, b] = ValueIndicesIn[pp, po, kh, b]
            for b in T.Parallel(PSD):
                KeySignsOut[pp, po, kh, b] = KeySignsIn[pp, po, kh, b]
            for n2 in T.Parallel(2):
                KeyNormsOut[pp, po, kh, n2] = KeyNormsIn[pp, po, kh, n2]
            ValueNormsOut[pp, po, kh] = ValueNormsIn[pp, po, kh]

            if update_valid[0] == 1:
                sum_key[0] = 0.0
                sum_val[0] = 0.0
                for d in T.serial(D):
                    kval = T.Cast(accum, KNew[update_idx[0], kh, d])
                    vval = T.Cast(accum, VNew[update_idx[0], kh, d])
                    k_norm[d] = kval
                    v_norm[d] = vval
                    sum_key[0] = sum_key[0] + kval * kval
                    sum_val[0] = sum_val[0] + vval * vval

                key_norm_value = T.sqrt(T.max(sum_key[0], 0.0))
                value_norm_value = T.sqrt(T.max(sum_val[0], 0.0))
                inv_key[0] = 1.0 / T.max(key_norm_value, 1e-8)
                inv_val[0] = 1.0 / T.max(value_norm_value, 1e-8)

                for rd in T.Parallel(D):
                    k_rot[rd] = 0.0
                    v_rot[rd] = 0.0
                    for d in T.serial(D):
                        r = T.Cast(accum, Rotation[rd, d])
                        k_rot[rd] = k_rot[rd] + k_norm[d] * inv_key[0] * r
                        v_rot[rd] = v_rot[rd] + v_norm[d] * inv_val[0] * r

                for d in T.serial(D):
                    best_dist = T.alloc_fragment((1,), accum)
                    best_idx = T.alloc_fragment((1,), "int32")
                    best_dist[0] = 1e30
                    best_idx[0] = 0
                    for c in T.serial(KL):
                        diff = k_rot[d] - T.Cast(accum, KeyCodebook[c])
                        dist = T.if_then_else(diff >= 0.0, diff, -diff)
                        take = dist < best_dist[0]
                        best_dist[0] = T.if_then_else(take, dist, best_dist[0])
                        best_idx[0] = T.if_then_else(take, c, best_idx[0])
                    k_code[d] = best_idx[0]
                    k_res[d] = k_rot[d] - T.Cast(accum, KeyCodebook[best_idx[0]])

                    best_dist[0] = 1e30
                    best_idx[0] = 0
                    for c in T.serial(VL):
                        diff = v_rot[d] - T.Cast(accum, ValueCodebook[c])
                        dist = T.if_then_else(diff >= 0.0, diff, -diff)
                        take = dist < best_dist[0]
                        best_dist[0] = T.if_then_else(take, dist, best_dist[0])
                        best_idx[0] = T.if_then_else(take, c, best_idx[0])
                    v_code[d] = best_idx[0]

                sum_res[0] = 0.0
                for d in T.serial(D):
                    sum_res[0] = sum_res[0] + k_res[d] * k_res[d]
                res_norm_value = T.sqrt(T.max(sum_res[0], 0.0))

                for b in T.serial(PID):
                    d0 = b * 2
                    d1 = d0 + 1
                    low = T.if_then_else(d0 < D, k_code[d0], 0)
                    high = T.if_then_else(d1 < D, k_code[d1], 0)
                    KeyIndicesOut[pp, po, kh, b] = T.Cast("uint8", low | (high << 4))
                    low_v = T.if_then_else(d0 < D, v_code[d0], 0)
                    high_v = T.if_then_else(d1 < D, v_code[d1], 0)
                    ValueIndicesOut[pp, po, kh, b] = T.Cast("uint8", low_v | (high_v << 4))

                for b in T.serial(PSD):
                    byte_val = T.alloc_fragment((1,), "int32")
                    byte_val[0] = 0
                    for bit in T.serial(8):
                        m = b * 8 + bit
                        proj = T.alloc_fragment((1,), accum)
                        proj[0] = 0.0
                        if m < QJL:
                            for d in T.serial(D):
                                proj[0] = proj[0] + k_res[d] * T.Cast(accum, QJLProjection[m, d])
                        byte_val[0] = byte_val[0] + T.if_then_else(
                            (m < QJL) & (proj[0] >= 0.0),
                            T.Cast("int32", 1) << bit,
                            0,
                        )
                    KeySignsOut[pp, po, kh, b] = T.Cast("uint8", byte_val[0])

                KeyNormsOut[pp, po, kh, 0] = T.Cast(norm_ts, key_norm_value)
                KeyNormsOut[pp, po, kh, 1] = T.Cast(norm_ts, res_norm_value)
                ValueNormsOut[pp, po, kh] = T.Cast(norm_ts, value_norm_value)

    return rpa_v3_tq_update
