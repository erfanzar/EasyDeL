# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

import functools
import typing as tp

import jax
import jax.lax as lax
import jax.numpy as jnp
from einops import rearrange
from jax.experimental import pallas as pl  # type: ignore[import]
from jax.experimental.pallas import tpu as pltpu  # type: ignore[import]

from ._utils import (
    DEFAULT_MASK_VALUE,
    MIN_BLOCK_SIZE,
    NUM_LANES,
    NUM_SUBLANES,
    TRANS_B_DIM_NUMBERS,
    BlockSizes,
    PatchBlockSpec,
    SegmentIds,
    _verify_block,
    below_or_on_diag,
)


def _ring_flash_attention_fwd_tpu(
    q,
    k,
    v,
    attn_bias,
    segment_ids,
    cache_idx,
    axis_name,
    float32_logits,
    query_chunk_size,
    key_chunk_size,
    causal_block_size,
):
    if float32_logits:
        q, k = q.astype(jnp.float32), k.astype(jnp.float32)
    q, k, v = map(lambda x: rearrange(x, "b q h d -> b h q d"), [q, k, v])
    batch, num_heads, q_len, dim_per_head = q.shape
    batch, num_heads, kv_len, dim_per_head = k.shape

    o = jnp.zeros((batch, num_heads, q_len, dim_per_head)).astype(q.dtype)
    lse_ = jnp.zeros((batch, num_heads, q_len)).astype(q.dtype)
    m = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(q.dtype)
    if attn_bias is not None:
        attn_bias = attn_bias[:, 0, :, :]
    axis_size = lax.psum(1, axis_name)
    q_block_size, kv_block_size = (q_len, kv_len)

    if segment_ids is not None:
        if cache_idx is None:
            q_offset = lax.axis_index(axis_name) * q_len
        else:
            q_offset = cache_idx
        q_segment_ids = lax.dynamic_slice_in_dim(segment_ids, q_offset, q_len, axis=-1)

    block_sizes = BlockSizes(
        block_q=query_chunk_size,
        block_k_major=key_chunk_size,
        block_k=key_chunk_size,
        block_b=1,
        block_q_major_dkv=query_chunk_size,
        block_k_major_dkv=key_chunk_size,
        block_k_dkv=key_chunk_size,
        block_q_dkv=query_chunk_size,
        block_k_major_dq=key_chunk_size,
        block_k_dq=key_chunk_size,
        block_q_dq=query_chunk_size,
    )

    scale = q.shape[-1] ** -0.5

    def scan_kv_block(carry, idx):
        o, lse_, m, k, v = carry
        pad = (lax.axis_index(axis_name) - idx) % axis_size
        if attn_bias is not None:
            attn_bias_slice = lax.dynamic_slice_in_dim(
                attn_bias,
                pad * kv_len,
                kv_len,
                axis=-1,
            )
        else:
            attn_bias_slice = None
        if segment_ids is not None:
            kv_segment_ids = lax.dynamic_slice_in_dim(
                segment_ids,
                pad * kv_len,
                kv_len,
                axis=-1,
            )
            segment_ids_slice = SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
        else:
            segment_ids_slice = None
        if cache_idx is None:
            q_block_idx = lax.axis_index(axis_name)
            q_chunk_idx_start = q_block_idx * (q_block_size // query_chunk_size)
        else:
            q_chunk_idx_start = cache_idx // query_chunk_size
        k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size
        k_chunk_idx_start = k_block_idx * (kv_block_size // key_chunk_size)
        o, lse_, m = _flash_attention_fwd(
            q,
            k,
            v,
            carry=(o, lse_, m),
            q_chunk_idx_start=q_chunk_idx_start,
            k_chunk_idx_start=k_chunk_idx_start,
            ab=attn_bias_slice,
            segment_ids=segment_ids_slice,
            save_residuals=False,
            causal_block_size=causal_block_size,
            sm_scale=scale,
            block_sizes=block_sizes,
            debug=False,
        )
        k, v = map(
            lambda x: lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]),
            (k, v),
        )
        return (o, lse_, m, k, v), None

    (o, lse_, m, _, _), _ = lax.scan(scan_kv_block, init=(o, lse_, m, k, v), xs=jnp.arange(0, axis_size))
    output = rearrange(o.astype(v.dtype), "b h q d -> b q h d")
    return output, (o, q, k, v, attn_bias, segment_ids, cache_idx, lse_, m)


def _flash_attention(
    q,
    k,
    v,
    carry,
    q_chunk_idx_start,
    k_chunk_idx_start,
    ab,
    segment_ids,
    save_residuals,
    causal_block_size,
    sm_scale,
    block_sizes,
    debug,
):
    return _flash_attention_impl(
        q,
        k,
        v,
        carry,
        q_chunk_idx_start,
        k_chunk_idx_start,
        ab,
        segment_ids,
        save_residuals,
        causal_block_size,
        sm_scale,
        block_sizes.block_b,
        block_sizes.block_q,
        block_sizes.block_k_major,
        block_sizes.block_k,
        debug,
    )


def _flash_attention_fwd(
    q,
    k,
    v,
    carry,
    q_chunk_idx_start,
    k_chunk_idx_start,
    ab,
    segment_ids,
    save_residuals,
    causal_block_size,
    sm_scale,
    block_sizes,
    debug,
):
    if save_residuals:
        raise NotImplementedError("Higher-order AD not supported")
    o, lse_, m = _flash_attention(
        q,
        k,
        v,
        carry,
        q_chunk_idx_start,
        k_chunk_idx_start,
        ab,
        segment_ids,
        True,
        causal_block_size,
        sm_scale,
        block_sizes,
        debug,
    )
    return o, lse_, m


def _flash_attention_kernel(
    q_idx_chunk_start,
    k_idx_chunk_start,
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    acc_tile_ref,
    l_tile_ref,
    m_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    o_tile_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    l_ref,
    m_ref,
    causal_block_size,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
    block_q,
):
    block_b = q_tile_ref.shape[0]

    if block_k == kv_seq_len:
        raise NotImplementedError()

    for batch_idx in range(block_b):
        _flash_attention_kernel_single_batch(
            (batch_idx, 0),
            q_idx_chunk_start,
            k_idx_chunk_start,
            q_tile_ref,
            k_tile_ref,
            v_tile_ref,
            acc_tile_ref,
            l_tile_ref,
            m_tile_ref,
            ab_tile_ref,
            q_segment_ids_tile_ref,
            kv_segment_ids_tile_ref,
            o_tile_ref,
            m_scratch_ref,
            l_scratch_ref,
            acc_scratch_ref,
            l_ref,
            m_ref,
            causal_block_size=causal_block_size,
            sm_scale=sm_scale,
            block_k=block_k,
            kv_seq_len=kv_seq_len,
            mask_value=mask_value,
            block_q=block_q,
        )


def _flash_attention_kernel_single_batch(
    batch_idx: tuple[int, ...],
    q_chunk_idx_start_ref,
    k_chunk_idx_start_ref,
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    acc_tile_ref,
    l_tile_ref,
    m_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    o_tile_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    l_ref: tp.Any | None = None,
    m_ref: tp.Any | None = None,
    *,
    causal_block_size,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
    block_q,
):
    block_k_major = k_tile_ref.shape[2]
    block_q = q_tile_ref.shape[2]
    head_dim = q_tile_ref.shape[-1]

    kv_seq_idx = pl.program_id(3)

    @pl.when(kv_seq_idx == 0)
    def start_new_sequence():
        m_scratch_ref[batch_idx] = m_tile_ref[batch_idx]
        l_scratch_ref[batch_idx] = l_tile_ref[batch_idx]
        acc_scratch_ref[batch_idx] = acc_tile_ref[batch_idx]

    q_chunk_idx_start = q_chunk_idx_start_ref[0]
    k_chunk_idx_start = k_chunk_idx_start_ref[0]

    q_seq_idx = pl.program_id(2)
    if causal_block_size is not None:
        should_run = below_or_on_diag(
            q_seq_idx + q_chunk_idx_start,
            block_q,
            kv_seq_idx + k_chunk_idx_start,
            block_k_major,
            causal_block_size,
        )
    else:
        should_run = True

    @pl.when(should_run)
    def run():
        @functools.partial(
            lax.fori_loop,
            0,
            block_k_major // block_k,
            init_val=None,
            unroll=True,
        )
        def body(i, _):
            m_prev = m_scratch_ref[batch_idx]
            l_prev = l_scratch_ref[batch_idx]
            q = q_tile_ref[batch_idx]  # [block_q, head_dim]
            start_k = i * block_k
            k = pl.load(
                k_tile_ref,
                (*batch_idx, pl.dslice(start_k, block_k), slice(None)),
            )

            s = jax.lax.dot_general(
                q,
                k,
                TRANS_B_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            )

            # Add attention bias if needed.
            if ab_tile_ref is not None:
                ab = pl.load(
                    ab_tile_ref,
                    (batch_idx[0], pl.dslice(0, block_q), pl.dslice(start_k, block_k)),
                ).astype(s.dtype)
                s += ab

            if sm_scale != 1.0:
                s *= sm_scale

            mask = None
            if q_segment_ids_tile_ref is not None:
                repeats, rem = divmod(block_k, NUM_LANES)
                if rem:
                    raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
                q_segment_ids = pltpu.repeat(
                    q_segment_ids_tile_ref[batch_idx[0]],
                    repeats,
                    axis=1,
                )
                kv_segment_ids = pl.load(
                    kv_segment_ids_tile_ref,
                    (batch_idx[0], pl.dslice(1), pl.dslice(start_k, block_k)),
                )
                mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

            if causal_block_size is not None:
                mask_shape = (block_q, block_k)
                row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
                row_ids += (q_seq_idx + q_chunk_idx_start) * block_q
                row_ids = jax.lax.div(row_ids, causal_block_size)
                col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
                col_ids += (kv_seq_idx + k_chunk_idx_start) * block_k_major + start_k
                col_ids = jax.lax.div(col_ids, causal_block_size)
                causal_mask = col_ids <= row_ids
                mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

            s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

            m_curr = jnp.max(s, axis=1)[:, None]  # Row max, shape [block_q, 1].
            m_next = jnp.maximum(m_prev, m_curr)  # Shape [block_q, 128].

            block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
            if rem:
                raise NotImplementedError(f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}")
            p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))

            alpha = jnp.exp(m_prev - m_next)  # Shape [block_q, 128].

            l_corr = alpha * l_prev

            l_next = jnp.sum(p, axis=1)[:, None] + l_corr  # Shape [block_q, 128]

            head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
            l_broadcast = lambda lse_: pltpu.repeat(lse_, head_dim_repeats, 1)  # noqa:E731
            if rem:
                if head_dim_repeats == 0:
                    l_broadcast = lambda lse_: lse_[:, :head_dim]  # noqa:E731
                else:
                    raise NotImplementedError(f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger")
            l_scratch_ref[batch_idx] = l_next
            m_scratch_ref[batch_idx] = m_next

            l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
            acc_scratch_ref[batch_idx] *= l_broadcast(l_corr * l_next_inv_safe)
            v = pl.load(v_tile_ref, (*batch_idx, pl.dslice(start_k, block_k), slice(None)))
            o_curr = jax.lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32)
            acc_scratch_ref[batch_idx] += o_curr * l_broadcast(l_next_inv_safe)

    @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
    def store_output():
        o_tile_ref[batch_idx] = acc_scratch_ref[batch_idx].astype(o_tile_ref.dtype)
        if l_ref is not None:
            l_ref[batch_idx] = l_scratch_ref[batch_idx].astype(l_ref.dtype)
        if m_ref is not None:
            m_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_ref.dtype)


def _flash_attention_impl(
    q,
    k,
    v,
    carry,
    q_chunk_idx_start,
    k_chunk_idx_start,
    ab,
    segment_ids,
    save_residuals,
    causal_block_size,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    debug,
):
    if causal_block_size is not None:
        assert causal_block_size % block_q == 0 or block_q % causal_block_size == 0
        assert causal_block_size % block_k == 0 or block_k % causal_block_size == 0
    assert block_k_major == block_k, (block_k_major, block_k)
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    acc, l_prev, m_prev = carry
    l_prev, m_prev = map(
        lambda x: jnp.broadcast_to(x[..., None], (*x.shape, MIN_BLOCK_SIZE)),
        (l_prev, m_prev),
    )
    q_chunk_idx_start, k_chunk_idx_start = (
        q_chunk_idx_start[None],
        k_chunk_idx_start[None],
    )
    _verify_block("block_q", "q_seq_len", block_q, q_seq_len, should_divide=False)
    _verify_block("block_k_major", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k", "kv_seq_len", block_k, kv_seq_len)
    _verify_block("block_b", "batch", block_b, batch_size, should_divide=False)

    grid = (
        pl.cdiv(batch_size, block_b),
        num_heads,
        pl.cdiv(q_seq_len, block_q),
        kv_seq_len // block_k_major,
    )

    def q_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, q_seq_index, 0)

    def kv_index_map(
        batch_index,
        head_index,
        q_seq_index,
        kv_seq_index,
        q_idx_ref,
        k_idx_ref,
    ):
        if causal_block_size is not None:
            next_kv_index = lax.select(
                below_or_on_diag(
                    q_seq_index + q_idx_ref[0],
                    block_q,
                    kv_seq_index + k_idx_ref[0],
                    block_k_major,
                    causal_block_size,
                ),
                kv_seq_index,
                0,
            )
        else:
            next_kv_index = kv_seq_index
        return (batch_index, head_index, next_kv_index, 0)

    def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref):
        if causal_block_size is not None:
            should_run = below_or_on_diag(
                q_seq_index + q_idx_ref[0],
                block_q,
                kv_seq_index + k_idx_ref[0],
                block_k_major,
                causal_block_size,
            )
            next_kv_index = lax.select(should_run, kv_seq_index, 0)
        else:
            next_kv_index = kv_seq_index

        return (batch_index, 0, next_kv_index)

    def o_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, q_seq_index, 0)

    def lm_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, q_seq_index, 0)

    kernel = functools.partial(
        _flash_attention_kernel,
        causal_block_size=causal_block_size,
        mask_value=DEFAULT_MASK_VALUE,
        sm_scale=sm_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
        block_q=block_q,
    )
    out_shape = [jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)]
    out_specs = [PatchBlockSpec(o_index_map, (block_b, 1, block_q, head_dim))]

    if block_k != kv_seq_len:
        scratch_shape = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)
        m_scratch = scratch_shape((block_b, 1, block_q, MIN_BLOCK_SIZE))
        l_scratch = scratch_shape((block_b, 1, block_q, MIN_BLOCK_SIZE))
        acc_scratch = scratch_shape((block_b, 1, block_q, head_dim))
        out_shape += [m_scratch, l_scratch, acc_scratch]
        out_specs += [
            PatchBlockSpec(lambda *_: (0, 0, 0, 0), m_scratch.shape),
            PatchBlockSpec(lambda *_: (0, 0, 0, 0), l_scratch.shape),
            PatchBlockSpec(lambda *_: (0, 0, 0, 0), acc_scratch.shape),
        ]
    else:
        raise NotImplementedError()

    if save_residuals:
        out_specs = [
            *out_specs,
            PatchBlockSpec(lm_index_map, (block_b, 1, block_q, MIN_BLOCK_SIZE)),
            PatchBlockSpec(lm_index_map, (block_b, 1, block_q, MIN_BLOCK_SIZE)),
        ]
        lse_ = jax.ShapeDtypeStruct((batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32)
        m = jax.ShapeDtypeStruct((batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32)
        out_shape = (*out_shape, lse_, m)

    ab_block_spec = PatchBlockSpec(ab_index_map, (block_b, block_q, block_k_major)) if ab is not None else None

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(
            batch_index,
            head_index,
            q_seq_index,
            _,
            q_idx_ref,
            k_idx_ref,
        ):
            del head_index
            return (batch_index, q_seq_index, 0)

        def kv_segment_ids_index_map(
            batch_index,
            head_index,
            q_seq_index,
            kv_seq_index,
            q_idx_ref,
            k_idx_ref,
        ):
            del head_index
            if causal_block_size is not None:
                next_kv_index = lax.select(
                    below_or_on_diag(
                        q_seq_index + q_idx_ref[0],
                        block_q,
                        kv_seq_index + k_idx_ref[0],
                        block_k_major,
                        causal_block_size,
                    ),
                    kv_seq_index,
                    0,
                )
            else:
                next_kv_index = kv_seq_index
            return (batch_index, 0, next_kv_index)

        q_segment_ids_spec = PatchBlockSpec(
            q_segment_ids_index_map,
            (block_b, block_q, NUM_LANES),
        )
        kv_segment_ids_spec = PatchBlockSpec(
            kv_segment_ids_index_map,
            (block_b, NUM_SUBLANES, block_k_major),
        )

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        PatchBlockSpec(q_index_map, (block_b, 1, block_q, head_dim)),
        PatchBlockSpec(kv_index_map, (block_b, 1, block_k_major, head_dim)),
        PatchBlockSpec(kv_index_map, (block_b, 1, block_k_major, head_dim)),
        PatchBlockSpec(q_index_map, (block_b, 1, block_q, head_dim)),
        PatchBlockSpec(lm_index_map, (block_b, 1, block_q, MIN_BLOCK_SIZE)),
        PatchBlockSpec(lm_index_map, (block_b, 1, block_q, MIN_BLOCK_SIZE)),
        ab_block_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
    ]

    o, *aux = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
        ),
        debug=debug,
        compiler_params=dict(
            mosaic=dict(
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
            )
        ),
    )(
        q_chunk_idx_start,
        k_chunk_idx_start,
        q,
        k,
        v,
        acc,
        l_prev,
        m_prev,
        ab,
        q_segment_ids,
        kv_segment_ids,
    )
    if save_residuals:
        lse_, m = (v[..., 0] for v in aux[-2:])
        return (o, lse_, m)
    else:
        return o
