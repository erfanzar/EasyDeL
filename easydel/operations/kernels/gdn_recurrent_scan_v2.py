# Copyright 2026 Google LLC
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

"""Fused Pallas Gated Delta Rule recurrent scan kernel (v2) for TPU.

Implements the production GDN recurrent scan used by EasyDeL's continuous
batching runtime on TPU v7+. The kernel processes prefill chunks and decode
tokens in a single pipelined pallas grid, using the schedule table from
:mod:`gdn_compute_schedule_v2` to dispatch each grid iteration to the
correct code path:

* **Regular prefill** — chunkwise WY-form reduction of the GDR recurrence
  for a single request, producing the chunk output and updated state.
* **Transition prefill** — token-by-token math for sublane rows that
  straddle a sequence (or decode/prefill) boundary.
* **Decode** — per-token recurrent update plus output emission for a
  batch of single-token requests.

Public entry point: :func:`recurrent_scan` (jit-compiled wrapper).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import gdn_compute_schedule_v2 as compute_schedule_table_v2


def invert_triangular_matrix(A, block_size=16):
    """Inverts a unit lower triangular matrix A block-wise.

    Args:
      A: Unit lower triangular matrix of shape (B, N, N).
      block_size: Size of the blocks for Gaussian elimination.

    Returns:
      Inverse of A, of shape (B, N, N).
    """
    B, N, _ = A.shape
    num_blocks = N // block_size

    def local_forward_sub(A_mat, b_mat):
        x_list = []
        for i in range(block_size):
            b_i = b_mat[:, i, :]
            if i == 0:
                x_i = b_i
            else:
                stacked_x = jnp.stack(x_list, axis=1)
                all_prev_A = A_mat[:, i, :i]
                prev_sum = jnp.sum(all_prev_A[..., None] * stacked_x, axis=1)
                x_i = b_i - prev_sum
            x_list.append(x_i)
        return jnp.stack(x_list, axis=1)

    x_blocks = []
    for i in range(num_blocks):
        start, end = i * block_size, (i + 1) * block_size
        e_block = jnp.eye(N, dtype=A.dtype)[start:end, :]
        e_block = jnp.broadcast_to(e_block, (B, block_size, N))

        if i == 0:
            target_b = e_block
        else:
            interaction_A = A[:, start:end, :start]
            solved_x = jnp.concatenate(x_blocks, axis=1)
            prev_sum = jnp.matmul(interaction_A, solved_x, precision=jax.lax.Precision.HIGHEST)
            target_b = e_block - prev_sum

        local_A = A[:, start:end, start:end]
        x_block = local_forward_sub(local_A, target_b)
        x_blocks.append(x_block)

    return jnp.concatenate(x_blocks, axis=1)


def inner_kernel(
    prefill_qkv_ref,
    decode_qkv_ref,
    prefill_a_raw_ref,
    decode_a_raw_ref,
    prefill_b_raw_ref,
    decode_b_raw_ref,
    a_log_ref,
    dt_bias_ref,
    prefill_output_ref,
    decode_output_ref,
    schedule_table,
    state_indices,
    has_initial_state,
    *,
    recurrent_state_in,
    recurrent_state_out,
    C: int,
    BT: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    use_qk_norm_in_gdn: bool,
    sublanesize: int,
    prefill_only: bool,
    prefill_scratch,
    decode_state_scratch,
    state_commit_scratch,
    decode_output_scratch,
    decode_read_semaphores,
    decode_write_semaphore,
    prefill_semaphore,
    decode_tokens,
):
    """Per-grid-step body that runs one decode batch or one prefill block.

    Called once per ``pl.program_id(0)`` step. Reads the schedule table row
    for the current step and dispatches to ``process_decode`` for decode
    batches, ``process_regular_prefill`` for chunk-aligned prefill blocks,
    and ``process_transition_prefill`` for sublane rows that straddle a
    request boundary. Recurrent state is loaded from / stored back into
    ``recurrent_state_in`` / ``recurrent_state_out`` via async DMAs, with
    double-buffered VMEM scratch to overlap compute with HBM traffic.

    Args:
        prefill_qkv_ref: VMEM ref for the prefill QKV chunk, shape
            ``(C, 2 * n_kq * d_k + n_v * d_v)``.
        decode_qkv_ref: VMEM ref for the decode QKV batch, shape
            ``(BT, 2 * n_kq * d_k + n_v * d_v)``.
        prefill_a_raw_ref: VMEM ref for prefill ``a`` gate inputs, shape
            ``(C, 128)``.
        decode_a_raw_ref: VMEM ref for decode ``a`` gate inputs, shape
            ``(BT, 128)``.
        prefill_b_raw_ref: VMEM ref for prefill ``b`` gate inputs, shape
            ``(C, 128)``.
        decode_b_raw_ref: VMEM ref for decode ``b`` gate inputs, shape
            ``(BT, 128)``.
        a_log_ref: VMEM ref for the per-head ``A_log`` parameter, shape
            ``(n_v,)``.
        dt_bias_ref: VMEM ref for the per-head ``dt_bias`` parameter, shape
            ``(n_v,)``.
        prefill_output_ref: VMEM output ref for the prefill chunk, shape
            ``(C, n_v * d_v)``.
        decode_output_ref: VMEM output ref for the decode batch, shape
            ``(BT, n_v * d_v)``.
        schedule_table: SMEM ref carrying the per-step work descriptors.
        state_indices: SMEM ref mapping request ids to state-pool slots.
        has_initial_state: SMEM flag per request indicating whether the
            pre-existing recurrent state should be loaded from HBM.
        recurrent_state_in: HBM input state pool, shape
            ``(B, n_v, d_k, d_v)``.
        recurrent_state_out: HBM output state pool, same shape.
        C: Static chunk size for prefill.
        BT: Static block size for decode batches.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Key dimension.
        d_v: Value dimension.
        use_qk_norm_in_gdn: Whether to L2-normalize Q/K before the recurrence.
        sublanesize: TPU sublane size used for alignment / row math.
        prefill_only: When ``True``, skip the decode branch entirely.
        prefill_scratch: Double-buffered VMEM scratch holding the live
            prefill recurrent state, shape ``(2, n_v, d_k, d_v)``.
        decode_state_scratch: VMEM scratch for the active decode state,
            shape ``(1, n_v, d_k, d_v)``.
        state_commit_scratch: VMEM scratch in ``recurrent_state`` dtype
            used to cast results before DMA-storing them back to HBM.
        decode_output_scratch: VMEM scratch for decode outputs awaiting
            DMA-out, shape ``(BT, n_v * d_v)``.
        decode_read_semaphores: DMA semaphores for state loads.
        decode_write_semaphore: DMA semaphore for state stores.
        prefill_semaphore: DMA semaphore for prefill state I/O.
        decode_tokens: Scalar number of decode tokens in the batch.
    """
    step = pl.program_id(0)


    prefill_valid = schedule_table[step, 0][...]
    prefill_req_id = schedule_table[step, 2][...]

    decode_valid = schedule_table[step, 4][...]
    decode_offset = schedule_table[step, 5][...]
    decode_req_id = schedule_table[step, 6][...]
    decode_count = schedule_table[step, 7][...]

    prefill_offset = schedule_table[step, 1][...]
    is_transition = schedule_table[step, 10][...]

    is_last_chunk = schedule_table[step, 8][...]
    is_first_chunk = schedule_table[step, 9][...]

    def l2_normalize(x, eps=1e-6):
        norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)
        return x / norm

    if not prefill_only:
        @pl.when(decode_valid > 0)
        def decode_wrapper():

            def get_target_idx(b):
                safe_req_id = jnp.minimum(decode_req_id + b, state_indices.shape[0] - 1)
                return state_indices[safe_req_id][...]

            def process_decode(b, _):
                is_valid = b < decode_count

                @pl.when(is_valid)
                def do_work():
                    target_idx = get_target_idx(b)

                    copy_op = pltpu.make_async_copy(
                        src_ref=recurrent_state_in.at[pl.ds(target_idx, 1)],
                        dst_ref=state_commit_scratch,
                        sem=decode_read_semaphores.at[0],
                    )
                    copy_op.start()
                    copy_op.wait()
                    decode_state_scratch[pl.ds(0, 1)] = state_commit_scratch[...].astype(jnp.float32)

                    key_dim = n_kq * d_k
                    b_aligned = (b // sublanesize) * sublanesize
                    qkv_block_data = decode_qkv_ref[pl.ds(b_aligned, sublanesize), :].astype(jnp.float32)
                    lane = b % sublanesize
                    lane_mask = (jnp.arange(sublanesize) == lane).astype(jnp.float32)[:, None]
                    qkv_row = jnp.sum(qkv_block_data * lane_mask, axis=0, keepdims=True)
                    qkv_row = jax.nn.silu(qkv_row)
                    q = qkv_row[:, :key_dim].reshape(n_kq, d_k)
                    k = qkv_row[:, key_dim : 2 * key_dim].reshape(n_kq, d_k)
                    v = qkv_row[:, 2 * key_dim :].reshape(n_v, d_v)

                    if use_qk_norm_in_gdn:
                        q = l2_normalize(q)
                        k = l2_normalize(k)

                    repeat_factor = n_v // n_kq
                    if repeat_factor > 1:
                        q = jnp.repeat(q, repeat_factor, axis=0)
                        k = jnp.repeat(k, repeat_factor, axis=0)

                    scale = d_k**-0.5
                    q = q * scale

                    b_aligned = (b // sublanesize) * sublanesize

                    g_block_new = decode_a_raw_ref[pl.ds(b_aligned, sublanesize), :]
                    beta_block_new = decode_b_raw_ref[pl.ds(b_aligned, sublanesize), :]

                    curr_g_slice_new = jnp.sum(
                        g_block_new.astype(jnp.float32) * lane_mask,
                        axis=0,
                        keepdims=True,
                    )
                    curr_beta_slice_new = jnp.sum(
                        beta_block_new.astype(jnp.float32) * lane_mask,
                        axis=0,
                        keepdims=True,
                    )

                    a_raw_new = curr_g_slice_new[:, :n_v].reshape(n_v).astype(jnp.float32)
                    b_raw_new = curr_beta_slice_new[:, :n_v].reshape(n_v).astype(jnp.float32)

                    curr_beta = jax.nn.sigmoid(b_raw_new)
                    curr_g = -jnp.exp(a_log_ref[...].astype(jnp.float32)) * jax.nn.softplus(
                        a_raw_new + dt_bias_ref[...].astype(jnp.float32)
                    )
                    curr_g = jnp.maximum(curr_g, -100.0)
                    decay = jnp.exp(curr_g)

                    current_state = decode_state_scratch[0]

                    out_list = []
                    new_state_list = []
                    for h in range(n_v):
                        q_h = q[h : h + 1, :]  # (1, d_k)
                        k_h = k[h : h + 1, :]  # (1, d_k)
                        v_h = v[h : h + 1, :]  # (1, d_v)

                        state_h = current_state[h]  # (d_k, d_v)

                        k_state_h = pl.dot(k_h, state_h, precision=jax.lax.Precision.HIGHEST)  # (1, d_v)

                        decay_k_state = jnp.where(
                            jnp.isinf(k_state_h),
                            0.0,
                            decay[h].astype(jnp.float32) * k_state_h,
                        )
                        v_diff_h = v_h - decay_k_state
                        v_new_h = curr_beta[h].astype(jnp.float32) * v_diff_h

                        q_state_h = pl.dot(q_h, state_h, precision=jax.lax.Precision.HIGHEST)  # (1, d_v)

                        q_k_h = jnp.sum(q_h * k_h, axis=-1, keepdims=True)  # (1, 1)

                        decay_q_state = jnp.where(jnp.isinf(q_state_h), 0.0, decay[h] * q_state_h)
                        out_h = decay_q_state + q_k_h * v_new_h
                        out_list.append(out_h)

                        k_v_new_h = pl.dot(
                            k_h, v_new_h, trans_a=True, precision=jax.lax.Precision.HIGHEST
                        )  # (d_k, 1) @ (1, d_v) -> (d_k, d_v)
                        decay_state = jnp.where(jnp.isinf(state_h), 0.0, state_h * decay[h])
                        new_state_h = decay_state + k_v_new_h
                        new_state_list.append(new_state_h)

                    out = jnp.concatenate(out_list, axis=0)  # (n_v, d_v)
                    new_state = jnp.stack(new_state_list, axis=0)  # (n_v, d_k, d_v)





                    decode_state_scratch[pl.ds(0, 1)] = new_state[None, ...].astype(current_state.dtype)

                    decode_output_scratch[pl.ds(b, 1), :] = out.reshape(1, n_v * d_v).astype(decode_output_scratch.dtype)

                    state_commit_scratch[0] = decode_state_scratch[0].astype(state_commit_scratch.dtype)
                    copy_op = pltpu.make_async_copy(
                        src_ref=state_commit_scratch,
                        dst_ref=recurrent_state_out.at[pl.ds(target_idx, 1)],
                        sem=decode_write_semaphore.at[0],
                    )
                    copy_op.start()
                    copy_op.wait()

                    return None

                return None

            decode_output_scratch[...] = jnp.zeros_like(decode_output_scratch[...])

            jax.lax.fori_loop(0, BT, process_decode, None)

            decode_output_ref[...] = decode_output_scratch[...]

            return None

    @pl.when(prefill_valid > 0)
    def process_prefill():

        prefill_slot = prefill_req_id % 2

        def process_regular_prefill():
            @pl.when(is_first_chunk > 0)
            def init_state():
                has_init = has_initial_state[prefill_req_id][...]

                def load_from_hbm():
                    state_idx = state_indices[prefill_req_id][...]
                    copy_op = pltpu.make_async_copy(
                        src_ref=recurrent_state_in.at[pl.ds(state_idx, 1)],
                        dst_ref=state_commit_scratch,
                        sem=prefill_semaphore.at[prefill_slot],
                    )
                    copy_op.start()
                    copy_op.wait()
                    prefill_scratch[prefill_slot] = state_commit_scratch[0].astype(prefill_scratch.dtype)

                def zero_state():
                    prefill_scratch[prefill_slot] = jnp.zeros((n_v, d_k, d_v), dtype=prefill_scratch.dtype)

                jax.lax.cond(has_init > 0, load_from_hbm, zero_state)
                return None

            key_dim = n_kq * d_k

            qkv_chunk = prefill_qkv_ref[...].astype(jnp.float32)  # (C, d)
            qkv_chunk = jax.nn.silu(qkv_chunk)
            q = qkv_chunk[:, :key_dim]
            k = qkv_chunk[:, key_dim : 2 * key_dim]
            v = qkv_chunk[:, 2 * key_dim :]

            a_raw_chunk = prefill_a_raw_ref[...]  # (C, 128)
            b_raw_chunk = prefill_b_raw_ref[...]  # (C, 128)

            a_raw_processed = a_raw_chunk[:, :n_v].T
            b_raw_processed = b_raw_chunk[:, :n_v].T

            a_raw_processed = a_raw_processed.astype(jnp.float32)
            b_raw_processed = b_raw_processed.astype(jnp.float32)
            beta = jax.nn.sigmoid(b_raw_processed)
            a_log_f32 = a_log_ref[...].astype(jnp.float32)
            dt_bias_f32 = dt_bias_ref[...].astype(jnp.float32)
            g = -jnp.exp(a_log_f32[:, None]) * jax.nn.softplus(a_raw_processed + dt_bias_f32[:, None])
            g = jnp.maximum(g, -100.0)
            prefill_count = schedule_table[step, 3][...]
            mask_float = (jnp.arange(C) < prefill_count).astype(q.dtype)
            q = jnp.where(mask_float[:, None] > 0, q, 0.0)
            k = jnp.where(mask_float[:, None] > 0, k, 0.0)
            g = jnp.where(mask_float[None, :] > 0, g, 0.0)
            v = jnp.where(mask_float[:, None] > 0, v, 0.0)
            beta = jnp.where(mask_float[None, :] > 0, beta, 0.0)

            q = q.reshape(C, n_kq, d_k)
            k = k.reshape(C, n_kq, d_k)
            v = v.reshape(C, n_v, d_v)

            if use_qk_norm_in_gdn:
                q = l2_normalize(q)
                k = l2_normalize(k)

            repeat_factor = n_v // n_kq
            if repeat_factor > 1:
                q = jnp.repeat(q, repeat_factor, axis=1)
                k = jnp.repeat(k, repeat_factor, axis=1)

            q = q.transpose(1, 0, 2)
            k = k.transpose(1, 0, 2)
            v = v.transpose(1, 0, 2)

            scale = d_k**-0.5
            q = q * scale

            g_cumsum_list = []
            current_sum = jnp.zeros((n_v,), dtype=jnp.float32)
            for i in range(C):
                current_sum = current_sum + g[:, i].astype(jnp.float32)
                g_cumsum_list.append(current_sum)
            g_cumsum = jnp.stack(g_cumsum_list, axis=-1)
            k_beta = k * beta[..., None]

            S = jnp.matmul(
                k_beta.astype(jnp.float32),
                k.transpose(0, 2, 1).astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )

            g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
            i = jnp.arange(C)[:, None]
            j = jnp.arange(C)[None, :]
            mask_float = (i > j).astype(jnp.float32)

            g_diff_safe = jnp.minimum(g_diff, 0.0)
            S = jnp.where(mask_float[None, :, :] > 0, S * jnp.exp(g_diff_safe), 0.0)

            S_q = jnp.matmul(
                q.astype(jnp.float32),
                k.transpose(0, 2, 1).astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )
            mask_float_q = (i >= j).astype(jnp.float32)
            g_diff_Sq = g_diff_safe * mask_float_q[None, ...] + (1.0 - mask_float_q[None, ...]) * (-1e30)
            S_q = S_q * jnp.exp(g_diff_Sq)
            S_q = S_q * mask_float_q[None, ...]

            I_plus_S = jnp.eye(C, dtype=jnp.float32)[None, ...] + S
            A_inv = invert_triangular_matrix(I_plus_S, block_size=16)

            v_beta = v * beta[..., None]
            u = jnp.matmul(A_inv, v_beta.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)

            k_beta_g = k_beta * jnp.exp(g_cumsum)[..., None]
            w = jnp.matmul(
                A_inv,
                k_beta_g.astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )

            q_g = q * jnp.exp(g_cumsum)[..., None]
            current_state = prefill_scratch[prefill_slot]
            attn_inter = jnp.matmul(
                q_g.astype(jnp.float32),
                current_state.astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )
            v_prime = jnp.matmul(
                w,
                current_state.astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )
            v_new = u - v_prime
            term2 = jnp.matmul(S_q, v_new, precision=jax.lax.Precision.HIGHEST)
            o_c = attn_inter + term2

            g_i_last_exp = jnp.exp(g_cumsum[..., -1, None, None])
            g_diff_exp_state = jnp.exp(g_cumsum[..., -1, None] - g_cumsum)[..., None]
            k_i_g_diff = k * g_diff_exp_state

            update_term = jnp.matmul(
                k_i_g_diff.transpose(0, 2, 1).astype(jnp.float32),
                v_new,
                precision=jax.lax.Precision.HIGHEST,
            )
            h_new = current_state * g_i_last_exp + update_term

            prefill_scratch[prefill_slot] = h_new.astype(prefill_scratch.dtype)

            @pl.when(is_last_chunk > 0)
            def store_state():
                state_commit_scratch[0] = prefill_scratch[prefill_slot].astype(state_commit_scratch.dtype)
                state_idx = state_indices[prefill_req_id][...]
                copy_op = pltpu.make_async_copy(
                    src_ref=state_commit_scratch,
                    dst_ref=recurrent_state_out.at[pl.ds(state_idx, 1)],
                    sem=prefill_semaphore.at[prefill_slot],
                )
                copy_op.start()
                copy_op.wait()
                return None

            o_c_tr = o_c.transpose(1, 0, 2)
            o_c_flat = o_c_tr.reshape(C, n_v * d_v)

            prefill_count = schedule_table[step, 3][...]
            mask_float = (jnp.arange(C) < prefill_count).astype(o_c_flat.dtype)
            o_c_flat_masked = o_c_flat * mask_float[:, None]
            prefill_output_ref[...] = o_c_flat_masked.astype(prefill_output_ref.dtype)
            return None

        def process_transition_prefill():
            C_trans = sublanesize
            key_dim = n_kq * d_k

            qkv_chunk = prefill_qkv_ref[:C_trans, :].astype(jnp.float32)
            qkv_chunk = jax.nn.silu(qkv_chunk)
            q = qkv_chunk[:, :key_dim]
            k = qkv_chunk[:, key_dim : 2 * key_dim]
            v = qkv_chunk[:, 2 * key_dim :]

            a_raw_chunk = prefill_a_raw_ref[...]  # (C, 128)
            b_raw_chunk = prefill_b_raw_ref[...]  # (C, 128)

            a_raw_processed = a_raw_chunk[:C_trans, :n_v].T
            b_raw_processed = b_raw_chunk[:C_trans, :n_v].T

            a_raw_processed = a_raw_processed.astype(jnp.float32)
            b_raw_processed = b_raw_processed.astype(jnp.float32)
            beta_chunk = jax.nn.sigmoid(b_raw_processed)
            a_log_f32 = a_log_ref[...].astype(jnp.float32)
            dt_bias_f32 = dt_bias_ref[...].astype(jnp.float32)
            g_chunk = -jnp.exp(a_log_f32[:, None]) * jax.nn.softplus(a_raw_processed + dt_bias_f32[:, None])
            g_chunk = jnp.maximum(g_chunk, -100.0)
            q = q.reshape(C_trans, n_kq, d_k)
            k = k.reshape(C_trans, n_kq, d_k)
            v = v.reshape(C_trans, n_v, d_v)

            if use_qk_norm_in_gdn:
                q = l2_normalize(q)
                k = l2_normalize(k)

            repeat_factor = n_v // n_kq
            if repeat_factor > 1:
                q = jnp.repeat(q, repeat_factor, axis=1)
                k = jnp.repeat(k, repeat_factor, axis=1)

            q = q.transpose(1, 0, 2)
            k = k.transpose(1, 0, 2)
            v = v.transpose(1, 0, 2)

            scale = d_k**-0.5
            q = q * scale

            first_req_id = schedule_table[step, 11][...]
            first_is_first = schedule_table[step, 11 + C_trans][...]
            first_slot = first_req_id % 2
            first_has_init = has_initial_state[first_req_id][...]

            @pl.when((first_is_first > 0) & (first_has_init > 0))
            def load_first_state():
                state_idx = state_indices[first_req_id][...]
                copy_op = pltpu.make_async_copy(
                    src_ref=recurrent_state_in.at[pl.ds(state_idx, 1)],
                    dst_ref=state_commit_scratch,
                    sem=prefill_semaphore.at[first_slot],
                )
                copy_op.start()
                copy_op.wait()
                prefill_scratch[first_slot] = state_commit_scratch[0].astype(prefill_scratch.dtype)

            h = prefill_scratch[first_slot]
            h = jnp.where((first_is_first > 0) & (first_has_init == 0), jnp.zeros_like(h), h)

            current_r = first_req_id
            sequence_valid = True

            for i in range(sublanesize):
                t_req = schedule_table[step, 11 + i][...]
                t_is_first = schedule_table[step, 11 + C_trans + i][...]
                t_is_last = schedule_table[step, 11 + 2 * C_trans + i][...]

                is_new_seq = t_req != current_r
                sequence_valid = jnp.where(is_new_seq, True, sequence_valid)

                is_decode_token = t_req < decode_tokens
                sequence_valid = jnp.where(is_decode_token, False, sequence_valid)

                c_slot = current_r % 2

                h0 = prefill_scratch[0]
                h1 = prefill_scratch[1]
                prefill_scratch[0] = jnp.where(c_slot == 0, h, h0)
                prefill_scratch[1] = jnp.where(c_slot == 1, h, h1)

                state_commit_scratch[0] = prefill_scratch[c_slot].astype(state_commit_scratch.dtype)

                def do_write(current_r=current_r, c_slot=c_slot):
                    state_idx = state_indices[current_r][...]
                    copy_op = pltpu.make_async_copy(
                        src_ref=state_commit_scratch,
                        dst_ref=recurrent_state_out.at[pl.ds(state_idx, 1)],
                        sem=prefill_semaphore.at[c_slot],
                    )
                    copy_op.start()
                    copy_op.wait()
                    return None

                is_current_r_prefill = current_r >= decode_tokens
                should_write = is_current_r_prefill & is_new_seq
                jax.lax.cond(should_write, do_write, lambda: None)

                t_slot = t_req % 2
                t_has_init = has_initial_state[t_req][...]

                def load_t_state(t_req=t_req, t_slot=t_slot):
                    state_idx = state_indices[t_req][...]
                    copy_op = pltpu.make_async_copy(
                        src_ref=recurrent_state_in.at[pl.ds(state_idx, 1)],
                        dst_ref=state_commit_scratch,
                        sem=prefill_semaphore.at[t_slot],
                    )
                    copy_op.start()
                    copy_op.wait()
                    prefill_scratch[t_slot] = state_commit_scratch[0].astype(prefill_scratch.dtype)

                should_load_t = (t_is_first > 0) & (t_has_init > 0)
                jax.lax.cond(should_load_t, load_t_state, lambda: None)

                h0_new = prefill_scratch[0]
                h1_new = prefill_scratch[1]
                new_h = jnp.where(t_slot == 0, h0_new, h1_new)

                new_h = jnp.where((t_is_first > 0) & (t_has_init == 0), jnp.zeros_like(new_h), new_h)
                h = new_h

                current_r = t_req

                k_i = k[:, i, :]
                v_i = v[:, i, :]
                g_i = g_chunk[:, i]
                beta_i = beta_chunk[:, i]
                q_i = q[:, i, :]

                decay = jnp.exp(g_i)[..., None]

                k_state = jnp.sum(k_i[..., None] * h, axis=1)
                v_diff = v_i - decay * k_state
                v_new = beta_i[:, None] * v_diff

                q_state = jnp.sum(q_i[..., None] * h, axis=1)
                q_k = jnp.sum(q_i * k_i, axis=-1, keepdims=True)

                out_i = decay * q_state + q_k * v_new

                k_v_new = k_i[..., None] * v_new[:, None, :]
                h_new = h * decay[..., None] + k_v_new

                h = jnp.where(sequence_valid, h_new, h)

                out_i = jnp.where(sequence_valid, out_i, 0.0)

                sequence_valid = jnp.where(t_is_last > 0, False, sequence_valid)

                prefill_output_ref[i, :] = out_i.reshape(n_v * d_v).astype(prefill_output_ref.dtype)

            final_slot = current_r % 2
            prefill_scratch[final_slot] = h
            state_commit_scratch[0] = h.astype(state_commit_scratch.dtype)

            is_current_r_prefill = current_r >= decode_tokens

            @pl.when(is_current_r_prefill)
            def do_final_write():
                state_idx = state_indices[current_r][...]
                copy_op = pltpu.make_async_copy(
                    src_ref=state_commit_scratch,
                    dst_ref=recurrent_state_out.at[pl.ds(state_idx, 1)],
                    sem=prefill_semaphore.at[final_slot],
                )
                copy_op.start()
                copy_op.wait()
                return None

            return None

        is_transition = schedule_table[step, 10][...]

        def process_prefill_dispatch():
            return jax.lax.cond(
                is_transition > 0,
                lambda _: process_transition_prefill(),
                lambda _: process_regular_prefill(),
                operand=None,
            )

        process_prefill_dispatch()
        return None

    def do_stitch():
        local_start = prefill_offset - decode_offset
        local_split = decode_tokens - prefill_offset

        safe_local_start = pl.multiple_of(local_start, sublanesize)

        decode_overlap = decode_output_ref[pl.ds(safe_local_start, sublanesize), :]
        prefill_arr = prefill_output_ref[pl.ds(0, sublanesize), :]

        iota = jax.lax.broadcasted_iota(jnp.int32, (sublanesize,), 0)
        is_decode_mask = (iota < local_split).astype(jnp.int32)[:, None]

        merged_overlap = jnp.where(is_decode_mask, decode_overlap, prefill_arr)

        decode_output_ref[pl.ds(safe_local_start, sublanesize), :] = merged_overlap
        prefill_output_ref[pl.ds(0, sublanesize), :] = merged_overlap

        return None

    if not prefill_only:
        is_first_block = pl.program_id(0) == 0
        needs_stitching = (is_transition > 0) & is_first_block & (decode_valid > 0)
        jax.lax.cond(needs_stitching, do_stitch, lambda: None)


def get_qkv_index_map_v2(
    step,
    schedule_table,
    valid_col,
    offset_col,
    count_col,
    alignment=16,
    block_size=64,
    sink_offset=0,
):
    """Compute the ``(offset, 0)`` BlockSpec index for a QKV block.

    Reads validity and offset columns from ``schedule_table`` for the
    current grid step and returns a sublane-aligned slice descriptor
    suitable for ``pl.BlockSpec``. When the step is not valid the index
    is routed to ``sink_offset`` so the kernel pipeline can DMA into a
    safe sink region.

    Args:
        step: Grid program id (``pl.program_id(0)``).
        schedule_table: SMEM ref carrying the work schedule.
        valid_col: Column index that signals whether this step is valid.
        offset_col: Column index carrying the block token offset.
        count_col: Column index for the block token count (unused here
            but kept in the signature for symmetry with related helpers).
        alignment: Required offset alignment (typically the sublane size).
        block_size: Size of the block along the token axis.
        sink_offset: Token offset used as the safe sink when the step is
            inactive.

    Returns:
        tuple: ``(pl.ds(safe_offset, block_size), 0)`` ready to be returned
        from a ``pl.BlockSpec.index_map``.
    """
    valid = schedule_table[step, valid_col][...]
    offset = schedule_table[step, offset_col][...]
    offset = pl.multiple_of(offset, alignment)

    safe_offset = jnp.where(valid > 0, offset, sink_offset)
    safe_offset = pl.multiple_of(safe_offset, alignment)

    return (pl.ds(safe_offset, block_size), 0)


def create_block_specs(
    schedule_table,
    chunk_size,
    BT,
    d,
    n_v,
    d_v,
    alignment=16,
    sink_offset=0,
):
    """Build the input/output ``pl.BlockSpec`` lists for the GDN scan kernel.

    Produces the per-input and per-output block specs consumed by
    :func:`fused_kernel`. The prefill specs slice token windows of size
    ``chunk_size``; the decode specs slice windows of size ``BT``. All
    QKV / a / b specs share the same index-map closure so the pipeline
    streams the same coordinates through every tensor.

    Args:
        schedule_table: SMEM ref carrying the per-step work schedule.
        chunk_size: Token block size for prefill chunks.
        BT: Token block size for decode batches.
        d: Inner feature width of the packed QKV tensor.
        n_v: Number of value heads.
        d_v: Per-head value dimension.
        alignment: Sublane alignment for block offsets.
        sink_offset: Safe sink token offset used when a grid step is
            inactive.

    Returns:
        tuple: ``(in_specs, out_specs)`` lists ready to be passed to
        ``pltpu.emit_pipeline``.
    """

    prefill_qkv_index_map = functools.partial(
        get_qkv_index_map_v2,
        schedule_table=schedule_table,
        valid_col=0,
        offset_col=1,
        count_col=3,
        alignment=alignment,
        block_size=chunk_size,
        sink_offset=sink_offset,
    )

    decode_qkv_index_map = functools.partial(
        get_qkv_index_map_v2,
        schedule_table=schedule_table,
        valid_col=4,
        offset_col=5,
        count_col=7,
        alignment=alignment,
        block_size=BT,
        sink_offset=sink_offset,
    )

    prefill_qkv_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), d),
        index_map=prefill_qkv_index_map,
    )
    decode_qkv_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), d),
        index_map=decode_qkv_index_map,
    )

    prefill_output_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), n_v * d_v),
        index_map=prefill_qkv_index_map,
    )
    decode_output_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), n_v * d_v),
        index_map=decode_qkv_index_map,
    )

    a_log_spec = pl.BlockSpec(block_shape=(n_v,), index_map=lambda _: (0,))
    dt_bias_spec = pl.BlockSpec(block_shape=(n_v,), index_map=lambda _: (0,))
    prefill_a_raw_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), 128),
        index_map=prefill_qkv_index_map,
    )
    decode_a_raw_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), 128),
        index_map=decode_qkv_index_map,
    )
    prefill_b_raw_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), 128),
        index_map=prefill_qkv_index_map,
    )
    decode_b_raw_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), 128),
        index_map=decode_qkv_index_map,
    )

    return [
        prefill_qkv_spec,
        decode_qkv_spec,
        prefill_a_raw_spec,
        decode_a_raw_spec,
        prefill_b_raw_spec,
        decode_b_raw_spec,
        a_log_spec,
        dt_bias_spec,
    ], [prefill_output_spec, decode_output_spec]


def fused_kernel(
    mixed_qkv_ref,
    aliased_recurrent_state_ref,
    state_indices_ref,
    has_initial_state_ref,
    a_raw_ref,
    b_raw_ref,
    a_log_ref,
    dt_bias_ref,
    schedule_table_ref,
    decode_tokens_ref,
    total_blocks_ref,
    recurrent_state_ref,
    output_ref,
    *,
    C: int,
    BT: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    use_qk_norm_in_gdn: bool,
    sublanesize: int,
    prefill_only: bool,
):
    """Outer Pallas program that drives the fused GDN recurrent scan.

    Pulls ``decode_tokens`` and ``total_blocks`` out of SMEM, builds the
    block specs for QKV / a / b streams, and emits the pipelined inner
    kernel via ``pltpu.emit_pipeline``. All VMEM scratch buffers and DMA
    semaphores required by :func:`inner_kernel` are allocated through
    ``pl.run_scoped`` so they have the correct lifetime.

    Args:
        mixed_qkv_ref: HBM ref for the padded packed QKV tensor, shape
            ``(num_tokens + pad, 2 * n_kq * d_k + n_v * d_v)``.
        aliased_recurrent_state_ref: Aliased HBM input ref of the state
            pool used to satisfy ``input_output_aliases``.
        state_indices_ref: SMEM ref of the per-request state slot map.
        has_initial_state_ref: SMEM ref of per-request initial-state flags.
        a_raw_ref: HBM ref for padded ``a`` gate inputs, shape
            ``(num_tokens + pad, 128)``.
        b_raw_ref: HBM ref for padded ``b`` gate inputs, shape
            ``(num_tokens + pad, 128)``.
        a_log_ref: HBM ref for the per-head ``A_log`` parameter.
        dt_bias_ref: HBM ref for the per-head ``dt_bias`` parameter.
        schedule_table_ref: SMEM ref carrying the work schedule table.
        decode_tokens_ref: 1-element SMEM ref with the decode-token count.
        total_blocks_ref: 1-element SMEM ref with the total grid size.
        recurrent_state_ref: HBM output ref for the updated state pool.
        output_ref: HBM output ref for the packed token outputs, shape
            ``(num_tokens + pad, n_v * d_v)``.
        C: Static prefill chunk size.
        BT: Static decode batch size.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Key dimension.
        d_v: Value dimension.
        use_qk_norm_in_gdn: Whether to apply QK L2 normalization.
        sublanesize: TPU sublane size used for alignment.
        prefill_only: When ``True``, skip decode dispatch entirely.
    """
    decode_tokens = decode_tokens_ref[0]
    total_blocks = total_blocks_ref[0]

    d = mixed_qkv_ref.shape[-1]
    pad_size = max(C, BT)
    sink_offset = mixed_qkv_ref.shape[0] - pad_size

    in_specs, out_specs = create_block_specs(
        schedule_table_ref,
        C,
        BT,
        d,
        n_v,
        d_v,
        alignment=sublanesize,
        sink_offset=sink_offset,
    )

    def _run_with_scratch(
        scratch_ref,
        decode_state_scratch_ref,
        state_commit_scratch_ref,
        decode_output_scratch_ref,
        decode_read_sems,
        decode_write_sem,
        prefill_sem,
    ):

        pipeline_func = pltpu.emit_pipeline(
            body=functools.partial(
                inner_kernel,
                C=C,
                BT=BT,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                use_qk_norm_in_gdn=use_qk_norm_in_gdn,
                sublanesize=sublanesize,
                prefill_only=prefill_only,
                prefill_scratch=scratch_ref,
                decode_state_scratch=decode_state_scratch_ref,
                decode_output_scratch=decode_output_scratch_ref,
                state_commit_scratch=state_commit_scratch_ref,
                decode_read_semaphores=decode_read_sems,
                decode_write_semaphore=decode_write_sem,
                prefill_semaphore=prefill_sem,
                decode_tokens=decode_tokens,
                recurrent_state_in=aliased_recurrent_state_ref,
                recurrent_state_out=recurrent_state_ref,
            ),
            grid=(total_blocks,),
            in_specs=in_specs,
            out_specs=out_specs,
        )

        pipeline_func(
            mixed_qkv_ref,
            mixed_qkv_ref,
            a_raw_ref,
            a_raw_ref,
            b_raw_ref,
            b_raw_ref,
            a_log_ref,
            dt_bias_ref,
            output_ref,
            output_ref,
            scratches=[schedule_table_ref, state_indices_ref, has_initial_state_ref],
        )

    pl.run_scoped(
        _run_with_scratch,
        pltpu.VMEM((2, n_v, d_k, d_v), jnp.float32),  # prefill_scratch (double buffered)
        pltpu.VMEM((1, n_v, d_k, d_v), jnp.float32),  # decode_state_scratch
        pltpu.VMEM((1, n_v, d_k, d_v), recurrent_state_ref.dtype),  # state_commit_scratch
        pltpu.VMEM((BT, n_v * d_v), mixed_qkv_ref.dtype),  # decode_output_scratch
        pltpu.SemaphoreType.DMA((1,)),  # decode_read_semaphores
        pltpu.SemaphoreType.DMA((1,)),  # decode_write_semaphore
        pltpu.SemaphoreType.DMA((2,)),  # prefill_semaphore
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "chunk_size",
        "BT",
        "use_qk_norm_in_gdn",
        "prefill_only",
    ],
)
def recurrent_scan(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    recurrent_state: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    chunk_size: int = 128,
    BT: int = 128,
    use_qk_norm_in_gdn: bool = True,
    has_initial_state: jax.Array | None = None,
    prefill_only: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Fused recurrent scan kernel for GDN on TPU v7+.

    Pads the token-axis tensors to sublane alignment, builds the schedule
    table via :func:`compute_schedule_table_v2.compute_schedule_table_v2`,
    and launches :func:`fused_kernel` to produce the per-token GDR output
    and the updated recurrent-state pool. State is updated in place via
    ``input_output_aliases``.

    Args:
        mixed_qkv: Packed Q/K/V tokens, shape
            ``(num_tokens, 2 * n_kq * d_k + n_v * d_v)``.
        b: Per-token beta-gate inputs, shape ``(num_tokens, n_v)``.
        a: Per-token g-gate inputs, shape ``(num_tokens, n_v)``.
        recurrent_state: Recurrent state pool, shape
            ``(max_reqs, n_v, d_k, d_v)``.
        A_log: Log of the per-head ``A`` parameter, shape ``(n_v,)``.
        dt_bias: Per-head dt bias, shape ``(n_v,)``.
        query_start_loc: CSR-style cumulative token offsets per request,
            shape ``(num_requests + 1,)``.
        state_indices: Mapping from request id to state-pool slot.
        distribution: ``int32`` vector ``[decode_tokens, total_tokens,
            num_valid_seqs]``.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Per-head key dimension.
        d_v: Per-head value dimension.
        chunk_size: Token block size used for prefill chunks.
        BT: Token block size used for decode batches.
        use_qk_norm_in_gdn: Whether to apply QK L2 normalization inside
            the recurrence.
        has_initial_state: Optional ``int32`` per-request flag indicating
            that the existing recurrent state should be loaded from HBM.
            Defaults to zeros (cold start) when ``None``.
        prefill_only: When ``True``, skip the decode branch and dispatch
            only prefill / transition work.

    Returns:
        tuple: ``(updated_recurrent_state, output)`` where
        ``updated_recurrent_state`` has the same shape and dtype as
        ``recurrent_state`` and ``output`` has shape
        ``(num_tokens, n_v * d_v)`` (trimmed back to the unpadded length).
    """
    if has_initial_state is None:
        has_initial_state = jnp.zeros(state_indices.shape[0], dtype=jnp.int32)
    else:
        has_initial_state = has_initial_state.astype(jnp.int32)

    num_tokens = mixed_qkv.shape[0]
    tpu_info = pltpu.get_tpu_info()
    sublanesize = 4 // mixed_qkv.itemsize * tpu_info.num_sublanes

    block_size = max(chunk_size, BT)
    sink_offset = ((num_tokens + sublanesize - 1) // sublanesize) * sublanesize
    pad_rows = sink_offset + block_size - num_tokens
    mixed_qkv = jnp.pad(mixed_qkv, ((0, pad_rows), (0, 0)))

    a_padded = jnp.pad(a, ((0, pad_rows), (0, 128 - n_v)))
    b_padded = jnp.pad(b, ((0, pad_rows), (0, 128 - n_v)))

    decode_tokens = distribution[0]
    schedule_table, total_blocks = compute_schedule_table_v2.compute_schedule_table_v2(
        query_start_loc,
        decode_tokens,
        distribution[2],
        num_tokens,
        chunk_size,
        BT,
        alignment=sublanesize,
    )

    decode_tokens_arr = jnp.expand_dims(decode_tokens, 0)
    total_blocks_arr = jnp.expand_dims(total_blocks, 0)

    grid_spec = pl.GridSpec(
        grid=(1,),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(block_shape=(1,), index_map=lambda _: (0,)),
            pl.BlockSpec(block_shape=(1,), index_map=lambda _: (0,)),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
    )

    updated_recurrent_state, output_padded = pl.pallas_call(
        functools.partial(
            fused_kernel,
            C=chunk_size,
            BT=BT,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            use_qk_norm_in_gdn=use_qk_norm_in_gdn,
            sublanesize=sublanesize,
            prefill_only=prefill_only,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(recurrent_state.shape, recurrent_state.dtype),
            jax.ShapeDtypeStruct((sink_offset + block_size, n_v * d_v), mixed_qkv.dtype),
        ),
        grid_spec=grid_spec,
        input_output_aliases={1: 0},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    )(
        mixed_qkv,
        recurrent_state,
        state_indices,
        has_initial_state,
        a_padded,
        b_padded,
        A_log,
        dt_bias,
        schedule_table,
        decode_tokens_arr,
        total_blocks_arr,
    )
    return updated_recurrent_state, output_padded[:num_tokens]
