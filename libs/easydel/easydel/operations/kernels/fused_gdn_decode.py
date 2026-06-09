# Copyright 2026 Google LLC
# Copyright 2026 EasyDeL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.
"""serving-style fused recurrent GDN decode kernel for TPU.

This is adapted from the TPU-inference branch. It differs from the
older EasyDeL decode kernel in one critical way: the recurrent state cache is
an input/output alias and is updated directly by the Pallas program. That
avoids building a separate ``new_state`` prefix and scattering it back after
the kernel, which is costly in single-request decode.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _validate_decode_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    state: jax.Array,
    state_indices: jax.Array,
    b: jax.Array,
) -> tuple[int, int, int, int, jnp.dtype, int, int]:
    """Validate fused decode inputs and return shape metadata.

    Checks shape, dtype and TPU lane/packing constraints on every argument
    of :func:`fused_gdn_decode` so that mismatches fail before reaching the
    Pallas program.

    Args:
        q: Query tensor of shape ``(T, H, K)``.
        k: Key tensor of shape ``(T, H, K)``.
        v: Value tensor of shape ``(T, H, V)``.
        g: Log-decay tensor of shape ``(T, H, K)`` in ``float32``.
        state: Recurrent state pool of shape ``(num_states, H, K, V)``.
        state_indices: ``int32`` request-to-state index map.
        b: Raw beta logits broadcast to TPU lanes, shape ``(T, H, lanes)``.

    Returns:
        tuple: ``(T, H, K, V, dtype, num_states, num_lanes)`` extracted from
        the validated inputs.

    Raises:
        ValueError: If any shape, dtype or divisibility constraint required
            by the TPU kernel is violated.
    """

    t, h, k_dim = q.shape
    v_dim = v.shape[2]
    dtype = q.dtype
    num_states = state.shape[0]
    num_lanes = pltpu.get_tpu_info().num_lanes
    packing = 32 // dtypes.itemsize_bits(dtype)

    if k.shape != (t, h, k_dim):
        raise ValueError(f"k shape {k.shape} != q shape {q.shape}")
    if v.shape != (t, h, v_dim):
        raise ValueError(f"v shape {v.shape} must be ({t}, {h}, {v_dim})")
    if g.shape != (t, h, k_dim):
        raise ValueError(f"g shape {g.shape} must be ({t}, {h}, {k_dim})")
    if b.shape != (t, h, num_lanes):
        raise ValueError(f"b shape {b.shape} must be ({t}, {h}, {num_lanes})")
    if state.shape[1:] != (h, k_dim, v_dim):
        raise ValueError(f"state trailing shape {state.shape[1:]} must be ({h}, {k_dim}, {v_dim})")
    if state_indices.dtype != jnp.int32:
        raise ValueError(f"state_indices must be int32, got {state_indices.dtype}")
    if k_dim % num_lanes != 0 or v_dim % num_lanes != 0:
        raise ValueError(f"K={k_dim}, V={v_dim} must be multiples of TPU lanes={num_lanes}")
    if h % packing != 0:
        raise ValueError(f"H={h} must be a multiple of packing={packing}")
    if k.dtype != dtype or v.dtype != dtype:
        raise ValueError(f"q/k/v must share dtype; got q={dtype}, k={k.dtype}, v={v.dtype}")
    if g.dtype != jnp.float32:
        raise ValueError(f"g must be float32, got {g.dtype}")
    if state.dtype not in (jnp.float32, jnp.bfloat16, jnp.float16):
        raise ValueError(f"state must be float32, bfloat16, or float16; got {state.dtype}")

    return t, h, k_dim, v_dim, dtype, num_states, num_lanes


def _default_decode_bt(
    *,
    t: int,
    h: int,
    k_dim: int,
    v_dim: int,
    dtype,
    state_dtype,
) -> int:
    """Choose a token block size that fits comfortably in TPU VMEM.

    Computes the largest power-of-two number of decode tokens (``bt``) whose
    per-block live working set (double-buffered state, q/k/v/g shards and
    beta lanes) stays under 90% of the device's VMEM capacity.

    Args:
        t: Number of decode tokens in the batch.
        h: Number of attention heads.
        k_dim: Per-head key dimension.
        v_dim: Per-head value dimension.
        dtype: dtype of the activation tensors ``q``/``k``/``v``.
        state_dtype: dtype of the recurrent state pool.

    Returns:
        int: Token block size, clipped to ``[1, t]`` and rounded down to
        the nearest power of two.
    """

    ibits = dtypes.itemsize_bits(dtype)
    sbits = dtypes.itemsize_bits(state_dtype)
    num_lanes = pltpu.get_tpu_info().num_lanes
    vmem_bytes_limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

    per_bt_bits = 2 * h * k_dim * v_dim * sbits + 2 * (
        2 * h * k_dim * ibits + h * k_dim * 32 + 2 * h * v_dim * ibits + h * num_lanes * ibits
    )
    bt = max(1, (vmem_bytes_limit * 8) // per_bt_bits)
    bt = 1 << (bt.bit_length() - 1)
    return max(1, min(int(t), int(bt)))


def _decode_kernel_main(
    q_hbm,
    k_hbm,
    v_hbm,
    g_hbm,
    b_hbm,
    state_indices_ref,
    distribution_ref,
    _state_init_ref,
    o_hbm,
    state_hbm,
    h_bufs,
    h_load_sems,
    h_store_sems,
    *,
    h: int,
    k_dim: int,
    v_dim: int,
    bt: int,
):
    """Pallas program that runs the fused GDN recurrent decode update.

    Iterates over the decode tokens in blocks of ``bt`` using a
    double-buffered ``pltpu.emit_pipeline`` that prefetches the next state
    from HBM, performs the per-token recurrent update in VMEM, and stores
    the updated state back into the in-place state pool referenced by
    ``state_indices_ref``.

    Args:
        q_hbm: HBM input ref for queries, shape ``(T, H, K)``.
        k_hbm: HBM input ref for keys, shape ``(T, H, K)``.
        v_hbm: HBM input ref for values, shape ``(T, H, V)``.
        g_hbm: HBM input ref for log-decay ``g``, shape ``(T, H, K)``.
        b_hbm: HBM input ref for beta lanes, shape ``(T, H, num_lanes)``.
        state_indices_ref: SMEM ref of per-token state pool indices.
        distribution_ref: SMEM ref carrying ``[decode_end, total]``.
        _state_init_ref: Unused input alias of the state pool (kept to
            wire up ``input_output_aliases``).
        o_hbm: HBM output ref for the per-token decode result, shape
            ``(T, H, V)``.
        state_hbm: HBM input/output ref for the recurrent state pool,
            shape ``(num_states, H, K, V)``; updated in place.
        h_bufs: VMEM scratch double buffer holding the live recurrent
            state slabs.
        h_load_sems: DMA semaphores for asynchronous state loads.
        h_store_sems: DMA semaphores for asynchronous state stores.
        h: Number of attention heads.
        k_dim: Per-head key dimension.
        v_dim: Per-head value dimension.
        bt: Number of decode tokens processed per pipeline block.
    """
    decode_end = distribution_ref[0]
    nb_t = (decode_end + bt - 1) // bt
    bounded_bt = pl.BoundedSlice(bt)

    def token_map(i):
        t_start = i * bt
        t_size = jnp.minimum(bt, decode_end - t_start)
        return (pl.ds(t_start, t_size), 0, 0)

    qkv_spec = pl.BlockSpec((bounded_bt, h, k_dim), token_map)
    v_spec = pl.BlockSpec((bounded_bt, h, v_dim), token_map)
    b_spec = pl.BlockSpec((bounded_bt, h, b_hbm.shape[2]), token_map)

    for i_t in range(bt):

        @pl.when(i_t < decode_end)
        def _first_load(i_t=i_t):
            si = state_indices_ref[i_t]
            pltpu.make_async_copy(
                state_hbm.at[pl.ds(si, 1), :, :, :],
                h_bufs.at[0, pl.ds(i_t, 1), :, :, :],
                h_load_sems.at[0],
            ).start()

    def _inner_kernel(
        q_ref,
        k_ref,
        v_ref,
        g_ref,
        b_ref,
        o_ref,
        h_bufs_s,
        state_indices_s,
        h_load_sems_s,
        h_store_sems_s,
    ):
        block_id = pl.program_id(0)
        t_start = block_id * bt
        block_len = jnp.minimum(bt, decode_end - t_start)
        buf_idx = block_id % 2
        next_buf_idx = (block_id + 1) % 2

        next_t_start = t_start + bt
        next_block_len = jnp.maximum(jnp.minimum(bt, decode_end - next_t_start), 0)
        for i_t in range(bt):

            @pl.when(i_t < next_block_len)
            def _prefetch(i_t=i_t):
                next_si = state_indices_s[next_t_start + i_t]
                pltpu.make_async_copy(
                    state_hbm.at[pl.ds(next_si, 1), :, :, :],
                    h_bufs_s.at[next_buf_idx, pl.ds(i_t, 1), :, :, :],
                    h_load_sems_s.at[next_buf_idx],
                ).start()

        pltpu.make_async_copy(
            state_hbm.at[pl.ds(0, block_len), :, :, :],
            h_bufs_s.at[buf_idx, pl.ds(0, block_len), :, :, :],
            h_load_sems_s.at[buf_idx],
        ).wait()

        for i_t in range(bt):

            @pl.when(i_t < block_len)
            def _process_token(i_t=i_t):
                h0 = h_bufs_s[buf_idx, i_t].astype(jnp.float32)
                q_t = q_ref[i_t].astype(jnp.float32)
                k_t = k_ref[i_t].astype(jnp.float32)
                v_t = v_ref[i_t].astype(jnp.float32)
                g_t = g_ref[i_t].astype(jnp.float32)
                beta_t = jax.nn.sigmoid(b_ref[i_t].astype(jnp.float32)[:, 0])

                h_pre = h0 * jnp.exp(g_t[:, :, None])
                kh = jax.lax.dot_general(
                    k_t.reshape(h, 1, k_dim),
                    h_pre,
                    (((2,), (1,)), ((0,), (0,))),
                    preferred_element_type=jnp.float32,
                ).reshape(h, v_dim)
                v_diff = v_t - kh
                b_v = beta_t[:, None] * v_diff

                o_step1 = jax.lax.dot_general(
                    q_t.reshape(h, 1, k_dim),
                    h_pre,
                    (((2,), (1,)), ((0,), (0,))),
                    preferred_element_type=jnp.float32,
                ).reshape(h, v_dim)
                qk_dot = jnp.sum(q_t * k_t, axis=-1, keepdims=True)
                o_t = o_step1 + qk_dot * b_v
                h_new = h_pre + k_t[:, :, None] * b_v[:, None, :]

                o_ref[i_t] = o_t.astype(o_ref.dtype)
                h_bufs_s[buf_idx, i_t] = h_new.astype(h_bufs_s.dtype)

        prev_t_start = jnp.maximum((block_id - 2) * bt, 0)
        prev_block_len = jnp.where(block_id >= 2, jnp.minimum(bt, decode_end - prev_t_start), 0)

        @pl.when(prev_block_len > 0)
        def _wait_prev_store():
            pltpu.make_async_copy(
                h_bufs_s.at[buf_idx, pl.ds(0, prev_block_len), :, :, :],
                state_hbm.at[pl.ds(0, prev_block_len), :, :, :],
                h_store_sems_s.at[buf_idx],
            ).wait()

        for i_t in range(bt):

            @pl.when(i_t < block_len)
            def _start_store(i_t=i_t):
                si = state_indices_s[t_start + i_t]
                pltpu.make_async_copy(
                    h_bufs_s.at[buf_idx, pl.ds(i_t, 1), :, :, :],
                    state_hbm.at[pl.ds(si, 1), :, :, :],
                    h_store_sems_s.at[buf_idx],
                ).start()

    pltpu.emit_pipeline(
        _inner_kernel,
        grid=(nb_t,),
        in_specs=[qkv_spec, qkv_spec, v_spec, qkv_spec, b_spec],
        out_specs=v_spec,
    )(
        q_hbm,
        k_hbm,
        v_hbm,
        g_hbm,
        b_hbm,
        o_hbm,
        scratches=[h_bufs, state_indices_ref, h_load_sems, h_store_sems],
    )

    last_buf_idx = (nb_t - 1) % 2
    other_buf_idx = nb_t % 2
    last_block_len = jnp.minimum(bt, decode_end - (nb_t - 1) * bt)
    pltpu.make_async_copy(
        h_bufs.at[last_buf_idx, pl.ds(0, last_block_len), :, :, :],
        state_hbm.at[pl.ds(0, last_block_len), :, :, :],
        h_store_sems.at[last_buf_idx],
    ).wait()

    other_block_len = jnp.where(nb_t >= 2, jnp.minimum(bt, decode_end - (nb_t - 2) * bt), 0)

    @pl.when(other_block_len > 0)
    def _drain_other():
        pltpu.make_async_copy(
            h_bufs.at[other_buf_idx, pl.ds(0, other_block_len), :, :, :],
            state_hbm.at[pl.ds(0, other_block_len), :, :, :],
            h_store_sems.at[other_buf_idx],
        ).wait()


@functools.partial(jax.jit, donate_argnames=("v", "state"))
def fused_gdn_decode(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    state: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Run the fused TPU decode kernel for Gated Delta Rule recurrent attention.

    Wraps the in-place Pallas pipeline :func:`_decode_kernel_main` with
    proper validation, VMEM-aware block sizing, and donation of the value
    and state buffers so that the recurrent state pool is updated without
    an extra scatter.

    Args:
        q: Pre-normalized and pre-scaled queries, shape ``(T, H, K)``.
        k: Pre-normalized keys, shape ``(T, H, K)``.
        v: Values, shape ``(T, H, V)``. Donated.
        g: Log decay, broadcast to shape ``(T, H, K)`` in ``float32``.
        b: Raw beta logits broadcast to TPU lanes, shape ``(T, H, lanes)``.
        state: Recurrent state pool, shape ``(num_states, H, K, V)``.
            Donated and updated in place.
        state_indices: Request-to-state mapping, ``int32`` shape
            ``(max_reqs,)``.
        distribution: 2-element ``int32`` vector ``[decode_end, total]``;
            only ``decode_end`` is consumed.

    Returns:
        tuple: ``(output, updated_state)`` where ``output`` has shape
        ``(T, H, V)`` (same dtype as ``q``) and ``updated_state`` carries the
        post-decode recurrent state with shape and dtype matching ``state``.
    """

    t, h, k_dim, v_dim, dtype, num_states, _num_lanes = _validate_decode_inputs(q, k, v, g, state, state_indices, b)
    bt = _default_decode_bt(t=t, h=h, k_dim=k_dim, v_dim=v_dim, dtype=dtype, state_dtype=state.dtype)

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    decode_end = distribution[0]
    grid_dim = jnp.where(decode_end > 0, 1, 0)

    out, updated_state = pl.pallas_call(
        functools.partial(
            _decode_kernel_main,
            h=h,
            k_dim=k_dim,
            v_dim=v_dim,
            bt=bt,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                any_spec,  # q
                any_spec,  # k
                any_spec,  # v
                any_spec,  # g
                any_spec,  # b
                smem_spec,  # state_indices
                smem_spec,  # distribution
                any_spec,  # state init
            ],
            out_specs=[any_spec, any_spec],
            grid=(grid_dim,),
            scratch_shapes=[
                pltpu.VMEM((2, bt, h, k_dim, v_dim), state.dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        input_output_aliases={
            2: 0,
            7: 1,
        },
        out_shape=[
            jax.ShapeDtypeStruct((t, h, v_dim), dtype),
            jax.ShapeDtypeStruct((num_states, h, k_dim, v_dim), state.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        ),
        name=f"easydel_fused_gdn_decode_bt{bt}",
    )(
        q,
        k,
        v,
        g,
        b,
        state_indices,
        distribution.astype(jnp.int32),
        state,
    )
    return out, updated_state
