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


def _ring_flash_attention_bwd_tpu(
    axis_name,
    float32_logits,
    query_chunk_size,
    key_chunk_size,
    causal_block_size,
    res,
    g,
):
    del float32_logits
    o, q, k, v, attn_bias, segment_ids, cache_idx, lse_, m = res
    batch, num_heads, kv_len, dim_per_head = k.shape
    axis_size = lax.psum(1, axis_name)
    dq = jnp.zeros_like(q, dtype=jnp.float32)
    dk = jnp.zeros_like(k, dtype=jnp.float32)
    dv = jnp.zeros_like(v, dtype=jnp.float32)
    q_block_size, kv_block_size = (
        q.shape[2],
        k.shape[2],
    )  # assumes this function is pre-sharded inside shard_map
    scale = q.shape[-1] ** -0.5

    if segment_ids is not None:
        if cache_idx is None:
            q_offset = lax.axis_index(axis_name) * q_block_size
        else:
            q_offset = cache_idx
        q_segment_ids = lax.dynamic_slice_in_dim(segment_ids, q_offset, q_block_size, axis=-1)
    g = rearrange(g, "b q h d -> b h q d")

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

    def scan_kv_block(carry, idx):
        dq, dk, dv, k, v = carry
        if attn_bias is not None:
            attn_bias_slice = lax.dynamic_slice_in_dim(
                attn_bias,
                (lax.axis_index(axis_name) - idx) % axis_size * kv_len,
                kv_len,
                axis=-1,
            )
        else:
            attn_bias_slice = None
        if segment_ids is not None:
            kv_segment_ids = lax.dynamic_slice_in_dim(
                segment_ids,
                (lax.axis_index(axis_name) - idx) % axis_size * kv_len,
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
        (
            dq_i,
            dk_i,
            dv_i,
        ) = _flash_attention_bwd(
            save_residuals=False,
            causal_block_size=causal_block_size,
            sm_scale=scale,
            block_sizes=block_sizes,
            debug=False,
            q_chunk_idx_start=q_chunk_idx_start,
            k_chunk_idx_start=k_chunk_idx_start,
            residuals=(q, k, v, attn_bias_slice, segment_ids_slice, o, lse_, m),
            do=g,
        )
        dq += dq_i
        dk += dk_i
        dv += dv_i
        k, v, dk, dv = map(
            lambda x: lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)]),
            (k, v, dk, dv),
        )
        return (dq, dk, dv, k, v), None

    (dq, dk, dv, k, v), _ = lax.scan(scan_kv_block, init=(dq, dk, dv, k, v), xs=jnp.arange(0, axis_size))
    dq, dk, dv = dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)
    dq, dk, dv = map(lambda x: rearrange(x, "b h q d -> b q h d"), (dq, dk, dv))
    return dq, dk, dv, None, None, None


def _flash_attention_bwd(
    save_residuals: bool,
    causal_block_size: int | None,
    sm_scale: float,
    block_sizes: BlockSizes,
    debug: bool,
    q_chunk_idx_start,
    k_chunk_idx_start,
    residuals,
    do,
):
    """VJP rule for FlashAttention."""
    if save_residuals:
        raise NotImplementedError("Higher-order AD not supported")
    (q, k, v, ab, segment_ids, o, lse_, m) = residuals
    if not block_sizes.has_backward_blocks:
        raise ValueError("Program is being differentiated, but not all backward blocks are specified")

    di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)  # [batch_size, num_heads, q_seq_len]

    dk, dv = _flash_attention_bwd_dkv(
        q_chunk_idx_start,
        k_chunk_idx_start,
        q,
        k,
        v,
        ab,
        segment_ids,
        lse_,
        m,
        do,
        di,
        block_q_major=block_sizes.block_q_major_dkv,
        block_k_major=block_sizes.block_k_major_dkv,
        block_k=block_sizes.block_k_dkv,
        block_q=block_sizes.block_q_dkv,
        sm_scale=sm_scale,
        causal_block_size=causal_block_size,
        mask_value=DEFAULT_MASK_VALUE,
        debug=debug,
    )

    dq, ds = _flash_attention_bwd_dq(
        q_chunk_idx_start,
        k_chunk_idx_start,
        q,
        k,
        v,
        ab,
        segment_ids,
        lse_,
        m,
        do,
        di,
        block_q_major=block_sizes.block_q_dq,
        block_k_major=block_sizes.block_k_major_dq,
        block_k=block_sizes.block_k_dq,
        sm_scale=sm_scale,
        causal_block_size=causal_block_size,
        mask_value=DEFAULT_MASK_VALUE,
        debug=debug,
    )
    return dq, dk, dv


def _flash_attention_dkv_kernel(
    q_chunk_idx_start_ref,
    k_chunk_idx_start_ref,
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dk_tile_ref,
    dv_tile_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    sm_scale: float,
    causal_block_size: int | None,
    mask_value: float,
    q_seq_len: int,
    block_q: int,
    block_k: int,
):
    _, _, block_q_major, _ = q_tile_ref.shape
    _, _, block_k_major, _ = k_tile_ref.shape

    q_seq_index = pl.program_id(axis=3)
    kv_seq_index = pl.program_id(axis=2)

    q_chunk_idx_start = q_chunk_idx_start_ref[0]
    k_chunk_idx_start = k_chunk_idx_start_ref[0]

    @pl.when(q_seq_index == 0)
    def start_new_sequence():
        dk_scratch_ref[:, :] = jnp.zeros(dk_scratch_ref.shape, dk_scratch_ref.dtype)
        dv_scratch_ref[:, :] = jnp.zeros(dv_scratch_ref.shape, dv_scratch_ref.dtype)

    def q_body(j, _):
        start_q = j * block_q

        def k_body(i, _):
            start_k = i * block_k
            k = pl.load(k_tile_ref, (0, 0, pl.ds(start_k, block_k), slice(None)))
            v = pl.load(v_tile_ref, (0, 0, pl.ds(start_k, block_k), slice(None)))
            q = pl.load(q_tile_ref, (0, 0, pl.ds(start_q, block_q), slice(None)))  # [block_q, head_dim]
            lse_ = pl.load(l_tile_ref, (0, 0, pl.ds(start_q, block_q), slice(None)))  # [block_q, 128]
            m = pl.load(m_tile_ref, (0, 0, pl.ds(start_q, block_q), slice(None)))  # [block_q, 128]
            do = pl.load(do_tile_ref, (0, 0, pl.ds(start_q, block_q), slice(None)))  # [block_q, 128]
            di = pl.load(di_tile_ref, (0, 0, pl.ds(start_q, block_q), slice(None))).astype(jnp.float32)  # [block_q, 128]

            capped_logits = lax.dot_general(
                q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
            )  # [block_q_major, block_k]

            if ab_tile_ref is not None:
                ab = pl.load(
                    ab_tile_ref,
                    (
                        0,
                        pl.dslice(0, block_q),
                        pl.dslice(i * block_k, block_k),
                    ),
                ).astype(jnp.float32)
                capped_logits += ab

            if sm_scale != 1.0:
                capped_logits *= sm_scale

            mask = None
            if q_segment_ids_tile_ref is not None:
                repeats, rem = divmod(block_k, NUM_LANES)
                if rem:
                    raise NotImplementedError()
                q_segment_ids = pl.load(
                    q_segment_ids_tile_ref, (0, pl.ds(start_q, block_q), slice(None))
                )  # [block_q, NUM_LANES].
                q_segment_ids = pltpu.repeat(q_segment_ids, repeats, axis=1)  # [block_q, block_k].
                kv_segment_ids = pl.load(
                    kv_segment_ids_tile_ref, (slice(None), 0, pl.ds(start_k, block_k))
                )  # [1, block_k].
                mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

            if causal_block_size is not None:
                mask_shape = (block_q, block_k)
                row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
                row_ids += (q_seq_index + q_chunk_idx_start) * block_q_major + start_q
                row_ids = jax.lax.div(row_ids, causal_block_size)
                col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
                col_ids += (kv_seq_index + k_chunk_idx_start) * block_k_major + start_k
                col_ids = jax.lax.div(col_ids, causal_block_size)
                causal_mask = col_ids <= row_ids
                mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

            capped_logits = capped_logits if mask is None else capped_logits + jnp.where(mask, 0.0, mask_value)

            p = jnp.exp(capped_logits - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1))
            p = p * pltpu.repeat(1 / lse_, block_k // MIN_BLOCK_SIZE, axis=1)  # [block_q_major, block_k_major]
            dv = lax.dot(p.T.astype(do.dtype), do, preferred_element_type=jnp.float32)
            pl.store(
                dv_scratch_ref,
                (pl.ds(start_k, block_k), slice(None)),
                pl.load(dv_scratch_ref, (pl.ds(start_k, block_k), slice(None))) + dv.astype(dv_scratch_ref.dtype),
            )

            # di: [block_q, 128]
            # do: [block_q, head_dim]
            # v: [block_k_major, head_dim]
            dp = lax.dot_general(do, v, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)
            ds = (dp - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1)) * p

            if sm_scale != 1.0:
                ds = ds * sm_scale

            # ds: [block_q_major, block_k_major]
            # q: [block_q_major, head_dim]
            dk = lax.dot(ds.T.astype(do.dtype), q, preferred_element_type=jnp.float32)
            pl.store(
                dk_scratch_ref,
                (pl.ds(start_k, block_k), slice(None)),
                pl.load(dk_scratch_ref, (pl.ds(start_k, block_k), slice(None))) + dk.astype(dk_scratch_ref.dtype),
            )

        lax.fori_loop(0, block_k_major // block_k, k_body, None, unroll=True)

    if causal_block_size is not None:
        should_run = below_or_on_diag(
            q_seq_index + q_chunk_idx_start,
            block_q_major,
            kv_seq_index + k_chunk_idx_start,
            block_k_major,
            causal_block_size,
        )
    else:
        should_run = True

    @pl.when(should_run)
    def run():
        lax.fori_loop(0, block_q_major // block_q, q_body, None, unroll=True)

    @pl.when(q_seq_index == q_seq_len // block_q_major - 1)
    def end_of_q_sequence():
        dv_tile_ref[0, 0, :, :] = dv_scratch_ref[...].astype(dv_tile_ref)
        dk_tile_ref[0, 0, :, :] = dk_scratch_ref[...].astype(dk_tile_ref)


def _flash_attention_bwd_dkv(
    q_chunk_idx_start,
    k_chunk_idx_start,
    q,
    k,
    v,
    ab,
    segment_ids,
    lse_,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_q: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal_block_size: int | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    debug: bool = False,
):
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    q_chunk_idx_start, k_chunk_idx_start = (
        q_chunk_idx_start[None],
        k_chunk_idx_start[None],
    )
    _verify_block("block_q_major_dkv", "q_seq_len", block_q_major, q_seq_len)
    _verify_block("block_q_dkv", "q_seq_len", block_q, q_seq_len)
    _verify_block("block_k_major_dkv", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k_dkv", "kv_seq_len", block_k, kv_seq_len)

    # Broadcast out scalar values
    m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
    lse_ = jnp.broadcast_to(lse_[..., None], (*lse_.shape, MIN_BLOCK_SIZE))
    # Preprocess contraction for bwd pass
    di = jnp.broadcast_to(di[..., None], (*di.shape, MIN_BLOCK_SIZE))

    # kv index needs to be before q index since q index is the contractng
    # dimension.
    grid = (
        batch_size,
        num_heads,
        kv_seq_len // block_k_major,
        q_seq_len // block_q_major,
    )

    def qo_index_map(batch_index, head_index, kv_seq_index, q_seq_index, q_idx_ref, k_idx_ref):
        if causal_block_size is not None:
            # If the q block is skipped, stay at the 0th q block.
            next_q_index = lax.select(
                below_or_on_diag(
                    q_seq_index + q_idx_ref[0],
                    block_q_major,
                    kv_seq_index + k_idx_ref[0],
                    block_k_major,
                    causal_block_size,
                ),
                q_seq_index,
                0,
            )
        else:
            next_q_index = q_seq_index

        return (batch_index, head_index, next_q_index, 0)

    qo_spec = PatchBlockSpec(qo_index_map, (1, 1, block_q_major, head_dim))
    assert qo_spec.block_shape is not None
    assert q.ndim == len(qo_spec.block_shape)
    do_spec = qo_spec
    assert do.ndim == len(qo_spec.block_shape)

    def kv_index_map(batch_index, head_index, kv_seq_index, _, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, kv_seq_index, 0)

    kv_spec = PatchBlockSpec(kv_index_map, (1, 1, block_k_major, head_dim))
    assert kv_spec.block_shape is not None
    assert k.ndim == len(kv_spec.block_shape)
    assert v.ndim == len(kv_spec.block_shape)

    def lm_index_map(batch_index, head_index, _, q_seq_index, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, q_seq_index, 0)

    lm_spec = PatchBlockSpec(lm_index_map, (1, 1, block_q_major, MIN_BLOCK_SIZE))
    assert lm_spec.block_shape is not None
    assert lse_.ndim == len(lm_spec.block_shape)
    assert m.ndim == len(lm_spec.block_shape)

    di_spec = PatchBlockSpec(qo_index_map, (1, 1, block_q_major, MIN_BLOCK_SIZE))
    assert di_spec.block_shape is not None
    assert di.ndim == len(di_spec.block_shape)

    def ab_index_map(batch_index, head_index, kv_seq_index, q_seq_index, q_idx_ref, k_idx_ref):
        return (batch_index, 0, kv_seq_index)

    if ab is not None:
        ab = ab[:, None].repeat(block_q_major, axis=1)

    dab_spec = PatchBlockSpec(ab_index_map, (1, block_q_major, block_k_major)) if ab is not None else None

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(batch_index, head_index, kv_seq_index, q_seq_index, q_idx_ref, k_idx_ref):
            del head_index
            if causal_block_size is not None:
                next_q_index = lax.select(
                    below_or_on_diag(
                        q_seq_index + q_idx_ref[0],
                        block_q_major,
                        kv_seq_index + k_idx_ref[0],
                        block_k_major,
                        causal_block_size,
                    ),
                    q_seq_index,
                    0,
                )
            else:
                next_q_index = q_seq_index
            return (batch_index, next_q_index, 0)

        def kv_segment_ids_index_map(batch_index, head_index, kv_seq_index, _, q_idx_ref, k_idx_ref):
            del head_index
            return (batch_index, 0, kv_seq_index)

        q_segment_ids_spec = PatchBlockSpec(q_segment_ids_index_map, (1, block_q_major, NUM_LANES))
        kv_segment_ids_spec = PatchBlockSpec(kv_segment_ids_index_map, (1, NUM_SUBLANES, block_k_major))

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
        qo_spec,
        kv_spec,
        kv_spec,
        dab_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
        lm_spec,
        lm_spec,
        do_spec,
        di_spec,
    ]

    out_shapes = [
        jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim), k.dtype),
        jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim), v.dtype),
        jax.ShapeDtypeStruct((block_k_major, head_dim), jnp.float32),
        jax.ShapeDtypeStruct((block_k_major, head_dim), jnp.float32),
    ]

    def dkv_index_map(batch_index, head_index, kv_seq_index, _, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, kv_seq_index, 0)

    dkv_spec = PatchBlockSpec(dkv_index_map, (1, 1, block_k_major, head_dim))
    out_specs = [
        dkv_spec,
        dkv_spec,
        PatchBlockSpec(lambda *_: (0, 0), (block_k_major, head_dim)),
        PatchBlockSpec(lambda *_: (0, 0), (block_k_major, head_dim)),
    ]

    kernel = functools.partial(
        _flash_attention_dkv_kernel,
        block_q=block_q,
        block_k=block_k,
        sm_scale=sm_scale,
        causal_block_size=causal_block_size,
        mask_value=mask_value,
        q_seq_len=q_seq_len,
    )
    name_scope = f"flash_mha_bwd_dkv_{block_q_major=}_{block_q=}_{block_k_major=}_{block_k=}"
    with jax.named_scope(name_scope):
        dk, dv, _, _ = pl.pallas_call(
            kernel,
            out_shape=out_shapes,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2, in_specs=in_specs, out_specs=out_specs, grid=grid
            ),
            debug=debug,
            compiler_params=dict(mosaic=dict(dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"))),
        )(
            q_chunk_idx_start,
            k_chunk_idx_start,
            q,
            k,
            v,
            ab,
            q_segment_ids,
            kv_segment_ids,
            lse_,
            m,
            do,
            di,
        )
        assert dk.shape == k.shape
        assert dv.shape == v.shape
    return dk, dv


def _flash_attention_dq_kernel(
    q_chunk_idx_start_ref,
    k_chunk_idx_start_ref,
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dq_tile_ref,
    dq_scratch_ref,
    ds_tile_ref,
    *,
    sm_scale: float,
    causal_block_size: int | None,
    mask_value: float,
    kv_seq_len: int,
    block_k: int,
):
    _, _, block_k_major, _ = k_tile_ref.shape
    _, _, block_q_major, _ = q_tile_ref.shape

    kv_seq_index = pl.program_id(axis=3)
    q_seq_index = pl.program_id(axis=2)

    q_chunk_idx_start = q_chunk_idx_start_ref[0]
    k_chunk_idx_start = k_chunk_idx_start_ref[0]

    @pl.when(kv_seq_index == 0)
    def start_new_sequence():
        dq_scratch_ref[:, :] = jnp.zeros(dq_scratch_ref.shape, dq_scratch_ref.dtype)

    def body(i, _):
        k_slice = pl.ds(i * block_k, block_k)
        q = q_tile_ref[0, 0, :, :]
        k = pl.load(
            k_tile_ref,
            (0, 0, k_slice, slice(None)),
        )  # [block_k, head_dim]
        v = pl.load(
            v_tile_ref,
            (0, 0, k_slice, slice(None)),
        )  # [block_k, head_dim]
        lse_ = l_tile_ref[0, 0, :, :]  # [block_q_major, 128]
        m = m_tile_ref[0, 0, :, :]  # [block_q_major, 128]
        do = do_tile_ref[0, 0, :, :]  # [block_q_major, head_dim]
        di = di_tile_ref[0, 0, :].astype(jnp.float32)  # [block_q_major, 128]

        capped_logits = jax.lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)

        if ab_tile_ref is not None:
            ab = pl.load(
                ab_tile_ref,
                (0, pl.dslice(0, block_q_major), pl.dslice(i * block_k, block_k)),
            ).astype(jnp.float32)
            capped_logits += ab

        if sm_scale != 1.0:
            capped_logits *= sm_scale

        mask = None
        if q_segment_ids_tile_ref is not None:
            repeats, rem = divmod(block_k, NUM_LANES)
            if rem:
                raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
            q_segment_ids = pltpu.repeat(q_segment_ids_tile_ref[0], repeats, axis=1)
            kv_segment_ids = pl.load(kv_segment_ids_tile_ref, (slice(None), 0, k_slice))
            mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

        if causal_block_size is not None:
            mask_shape = (block_q_major, block_k)
            row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            row_ids += (q_seq_index + q_chunk_idx_start) * block_q_major
            row_ids = jax.lax.div(row_ids, causal_block_size)
            col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
            col_ids += (kv_seq_index + k_chunk_idx_start) * block_k_major + i * block_k
            col_ids = jax.lax.div(col_ids, causal_block_size)
            causal_mask = col_ids <= row_ids
            mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        capped_logits = capped_logits if mask is None else capped_logits + jnp.where(mask, 0.0, mask_value)

        p = jnp.exp(capped_logits - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1))
        p = p * pltpu.repeat(1 / lse_, block_k // MIN_BLOCK_SIZE, axis=1)  # [block_q_major, block_k]

        # di: [block_q_major, 128]
        # do: [block_q_major, head_dim]
        # v: [block_k_major, head_dim]
        dp = jax.lax.dot_general(
            do,
            v,
            TRANS_B_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
        ds = (dp - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1)) * p

        if sm_scale != 1.0:
            ds = ds * sm_scale

        if ds_tile_ref is not None:
            pl.store(
                ds_tile_ref,
                (0, pl.dslice(None), pl.dslice(i * block_k, block_k)),
                ds.astype(ds_tile_ref.dtype),
            )

        # dp: [block_q_major, block_k]
        # k: [block_k, head_dim]
        dq_scratch_ref[:, :] += lax.dot(
            ds.astype(k.dtype),
            k,
            preferred_element_type=jnp.float32,
        ).astype(dq_scratch_ref.dtype)

    if causal_block_size is not None:
        should_run = below_or_on_diag(
            q_seq_index + q_chunk_idx_start,
            block_q_major,
            kv_seq_index + k_chunk_idx_start,
            block_k_major,
            causal_block_size,
        )
        should_not_run = lax.select(should_run, False, True)
    else:
        should_run = True
        should_not_run = False  # type: ignore

    @pl.when(should_run)
    def run():
        lax.fori_loop(0, block_k_major // block_k, body, None, unroll=True)

    @pl.when(should_not_run)
    def zero_out_ds():
        if ds_tile_ref is not None:
            ds_tile_ref[...] = jnp.zeros_like(ds_tile_ref)

    @pl.when(kv_seq_index == kv_seq_len // block_k_major - 1)
    def end_of_kv_sequence():
        dq_tile_ref[0, 0, :, :] = dq_scratch_ref[...].astype(dq_tile_ref)
        dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)


def _flash_attention_bwd_dq(
    q_chunk_idx_start,
    k_chunk_idx_start,
    q,
    k,
    v,
    ab,
    segment_ids,
    lse_,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal_block_size: int | None,
    mask_value: float,
    debug: bool,
):
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    q_chunk_idx_start, k_chunk_idx_start = (
        q_chunk_idx_start[None],
        k_chunk_idx_start[None],
    )
    _verify_block("block_q_dq", "q_seq_len", block_q_major, q_seq_len)
    _verify_block("block_k_major_dq", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k_dq", "block_k", block_k, kv_seq_len)

    # Broadcast out scalar values
    m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
    lse_ = jnp.broadcast_to(lse_[..., None], (*lse_.shape, MIN_BLOCK_SIZE))
    # Preprocess contraction for bwd pass
    di = jnp.broadcast_to(di[..., None], (*di.shape, block_k_major))

    grid = (
        batch_size,
        num_heads,
        q_seq_len // block_q_major,
        kv_seq_len // block_k_major,
    )

    def qo_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, q_seq_index, 0)

    qo_spec = PatchBlockSpec(qo_index_map, (1, 1, block_q_major, head_dim))
    do_spec = qo_spec

    def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref):
        if causal_block_size is not None:
            # If the kv block is skipped, prefetch the next valid kv block, i.e. the
            # 0th one to be used for the next block_q rows.
            next_kv_index = lax.select(
                below_or_on_diag(
                    q_seq_index + q_idx_ref[0],
                    block_q_major,
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

    kv_spec = PatchBlockSpec(kv_index_map, (1, 1, block_k_major, head_dim))
    assert kv_spec.block_shape is not None
    assert k.ndim == len(kv_spec.block_shape)
    assert v.ndim == len(kv_spec.block_shape)

    def lm_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
        return (batch_index, head_index, q_seq_index, 0)

    lm_spec = PatchBlockSpec(lm_index_map, (1, 1, block_q_major, MIN_BLOCK_SIZE))
    assert lm_spec.block_shape is not None
    assert lse_.ndim == len(lm_spec.block_shape)
    assert m.ndim == len(lm_spec.block_shape)

    di_spec = PatchBlockSpec(qo_index_map, (1, 1, block_q_major, MIN_BLOCK_SIZE))
    assert di_spec.block_shape is not None
    assert di.ndim == len(di_spec.block_shape)

    def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref):
        return (batch_index, 0, kv_seq_index)

    if ab is not None:
        ab = ab[:, None].repeat(block_q_major, axis=1)

    dab_spec = PatchBlockSpec(ab_index_map, (1, block_q_major, block_k_major)) if ab is not None else None

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _, q_idx_ref, k_idx_ref):
            del head_index
            return (batch_index, q_seq_index, 0)

        def kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index, q_idx_ref, k_idx_ref):
            del head_index
            if causal_block_size is not None:
                # If the kv block is skipped, prefetch the next valid kv block, i.e. the
                # 0th one to be used for the next block_q rows.
                next_kv_index = lax.select(
                    below_or_on_diag(
                        q_seq_index + q_idx_ref[0],
                        block_q_major,
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

        q_segment_ids_spec = PatchBlockSpec(q_segment_ids_index_map, (1, block_q_major, NUM_LANES))
        kv_segment_ids_spec = PatchBlockSpec(kv_segment_ids_index_map, (1, NUM_SUBLANES, block_k_major))

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
        qo_spec,
        kv_spec,
        kv_spec,
        dab_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
        lm_spec,
        lm_spec,
        do_spec,
        di_spec,
    ]

    out_shapes = [
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct((block_q_major, head_dim), jnp.float32),
        jax.ShapeDtypeStruct(ab.shape, ab.dtype) if ab is not None else None,
    ]
    dq_spec = PatchBlockSpec(qo_index_map, (1, 1, block_q_major, head_dim))
    out_specs = [
        dq_spec,
        PatchBlockSpec(lambda *_: (0, 0), (block_q_major, head_dim)),
        dab_spec,
    ]

    kernel = functools.partial(
        _flash_attention_dq_kernel,
        sm_scale=sm_scale,
        causal_block_size=causal_block_size,
        mask_value=mask_value,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
    )
    name_scope = f"flash_mha_bwd_dq_{block_q_major=}_{block_k_major=}_{block_k=}"
    with jax.named_scope(name_scope):
        dq, _, ds = pl.pallas_call(
            kernel,
            out_shape=out_shapes,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2, in_specs=in_specs, out_specs=out_specs, grid=grid
            ),
            debug=debug,
            compiler_params=dict(mosaic=dict(dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"))),
        )(
            q_chunk_idx_start,
            k_chunk_idx_start,
            q,
            k,
            v,
            ab,
            q_segment_ids,
            kv_segment_ids,
            lse_,
            m,
            do,
            di,
        )

    return dq, ds
