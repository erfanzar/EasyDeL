"""Regression tests for PR #306 ejKernel fixes.

Covers:
- Finding #8: TPU Pallas GDR custom-VJP backward (`_chunk_gdr_bwd`,
  `_chunk_gdr_grouped_bwd`, `_single_step_gdr_bwd`) preserves the primal dtype of
  ``initial_state`` (and ``beta``/``decay``) under ``jax.checkpoint`` (remat) and
  mixed-precision cotangents (bf16 Q/K/V with fp32 recurrent state).
- Finding #29: ``flash_attention`` position-bias ``shard_map`` TP shape mismatch
  with broadcastable caller bias, SP ``DuplicateSpecError`` / sequence-sharding
  guard, and ``platform="cuda"`` / GPU candidate routing for 4-D position bias.
- Finding #44: ``blocksparse_attention`` does not mutate ``mask_info._attention_mask``
  (or positions/segments) in place when ``method is None``, preserving
  ``mask_info.uses_position_causality()`` across repeated calls and keeping custom
  stored dense masks intact.
"""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.gated_delta_rule import _pallas_impl_bwd as gdr_bwd_mod
from ejkernel.kernels._pallas.tpu.gated_delta_rule import _pallas_impl_fwd as gdr_fwd_mod
from ejkernel.kernels._registry import Backend, Platform
from ejkernel.modules.operations import blocksparse_attention, flash_attention
from ejkernel.modules.operations.configs import FlashAttentionConfig
from ejkernel.modules.operations.flash_attention import FlashAttention
from ejkernel.ops import Invocation
from ejkernel.types import MaskInfo
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

needs_4_devices = pytest.mark.skipif(jax.device_count() < 4, reason="needs a 4-device CPU mesh")


# -------------------------------------------------------------------------------------- Finding #8


def _install_cpu_gdr_step_mocks(monkeypatch):
    """Replace TPU-only Pallas step launchers with pure-JAX equivalents executing the same math.

    This exercises the REAL ``@jax.custom_vjp`` forward/backward rules, residual packing,
    and dtype casting of ``_chunk_gdr_fwd_pallas_chunk``, ``_chunk_gdr_grouped_fwd_pallas_chunk``,
    and ``_single_step_gdr_fwd`` on CPU.
    """

    def _dense_step(state, q_i, k_i, v_i, beta_i, g_cumsum_i, *, output_dtype, state_dtype):
        def _one(st, q, k, v, beta, g_cs):
            C = q.shape[0]
            v_loc, k_cum, attn_qk, q_sc, k_sc, g_end_exp = gdr_fwd_mod._phase1_chunk_values(
                q.astype(jnp.float32),
                k.astype(jnp.float32),
                v.astype(jnp.float32),
                beta.astype(jnp.float32),
                g_cs.astype(jnp.float32),
                C,
            )
            st_f = st.astype(jnp.float32)
            v_new = v_loc - jnp.matmul(k_cum, st_f)
            out = jnp.matmul(q_sc, st_f) + jnp.matmul(attn_qk, v_new)
            new_st = st_f * g_end_exp + jnp.matmul(k_sc.T, v_new)
            return new_st.astype(state_dtype), out.astype(output_dtype)

        return jax.vmap(jax.vmap(_one))(state, q_i, k_i, v_i, beta_i, g_cumsum_i)

    def _grouped_dense_step(state, q_i, k_i, v_i, beta_i, g_cumsum_i, *, expand_ratio, output_dtype, state_dtype):
        q_rep = jnp.repeat(q_i, expand_ratio, axis=1)
        k_rep = jnp.repeat(k_i, expand_ratio, axis=1)
        return _dense_step(
            state, q_rep, k_rep, v_i, beta_i, g_cumsum_i, output_dtype=output_dtype, state_dtype=state_dtype
        )

    def _seg_step(
        state, state_seg, q_i, k_i, v_i, beta_i, g_cumsum_i, seg_i, has_valid_i, seg_last_i, g_end_i, *, output_dtype
    ):
        def _one(st, st_seg, q, k, v, beta, g_cs, seg, has_v, s_last, g_end):
            C = q.shape[0]
            v_loc, k_cum, attn_qk, q_sc, k_sc, g_end_exp = gdr_fwd_mod._phase1_chunk_values(
                q.astype(jnp.float32),
                k.astype(jnp.float32),
                v.astype(jnp.float32),
                beta.astype(jnp.float32),
                g_cs.astype(jnp.float32),
                C,
                seg=seg,
                has_valid=has_v.astype(jnp.float32),
                seg_last=s_last,
                g_end_value=g_end.astype(jnp.float32),
            )
            st_f = st.astype(jnp.float32)
            valid = (seg >= 0).astype(jnp.float32)[:, None]
            read = (seg == st_seg).astype(jnp.float32)[:, None] * valid
            v_new = v_loc - jnp.matmul(k_cum, st_f) * read
            out = (jnp.matmul(q_sc, st_f) * read + jnp.matmul(attn_qk, v_new)) * valid
            survive = (s_last == st_seg).astype(jnp.float32) * has_v.astype(jnp.float32)
            cand = st_f * g_end_exp * survive + jnp.matmul(k_sc.T, v_new)
            keep_old = has_v.astype(jnp.float32) < 0.5
            new_st = jnp.where(keep_old, st_f, cand)
            new_seg = jnp.where(keep_old, st_seg, s_last).astype(jnp.int32)
            return new_st.astype(jnp.float32), new_seg, out.astype(output_dtype)

        return jax.vmap(jax.vmap(_one))(
            state, state_seg, q_i, k_i, v_i, beta_i, g_cumsum_i, seg_i, has_valid_i, seg_last_i, g_end_i
        )

    def _grouped_seg_step(
        state,
        state_seg,
        q_i,
        k_i,
        v_i,
        beta_i,
        g_cumsum_i,
        seg_i,
        has_valid_i,
        seg_last_i,
        g_end_i,
        *,
        expand_ratio,
        output_dtype,
    ):
        q_rep = jnp.repeat(q_i, expand_ratio, axis=1)
        k_rep = jnp.repeat(k_i, expand_ratio, axis=1)
        return _seg_step(
            state,
            state_seg,
            q_rep,
            k_rep,
            v_i,
            beta_i,
            g_cumsum_i,
            seg_i,
            has_valid_i,
            seg_last_i,
            g_end_i,
            output_dtype=output_dtype,
        )

    def _bwd_dense_step(state_pre, q_i, k_i, v_i, beta_i, decay_i, d_out_i, d_state_next, *, grad_dtype=jnp.float32):
        # q_i: [B, H, num_fused, C, K]; beta_i: [B, H, 1, num_fused, C]
        num_fused = q_i.shape[2]
        C = q_i.shape[3]

        def _one(st_pre, qg, kg, vg, bg, dg, dog, dst_next):
            cur_dst = dst_next.astype(jnp.float32)
            dqs, dks, dvs, dbs, dds = [], [], [], [], []
            for idx in range(num_fused - 1, -1, -1):
                sp = st_pre.astype(jnp.float32)
                for j in range(idx):
                    _, sp = gdr_fwd_mod._process_one_chunk(
                        qg[j].astype(jnp.float32),
                        kg[j].astype(jnp.float32),
                        vg[j].astype(jnp.float32),
                        bg[0, j].astype(jnp.float32),
                        dg[0, j].astype(jnp.float32),
                        sp,
                        C,
                    )
                cur_dst, dq, dk, dv, db, dd = gdr_bwd_mod._bwd_one_chunk(
                    qg[idx].astype(jnp.float32),
                    kg[idx].astype(jnp.float32),
                    vg[idx].astype(jnp.float32),
                    bg[0, idx].astype(jnp.float32),
                    dg[0, idx].astype(jnp.float32),
                    dog[idx].astype(jnp.float32),
                    sp,
                    cur_dst,
                    C,
                )
                dqs.append(dq.astype(grad_dtype))
                dks.append(dk.astype(grad_dtype))
                dvs.append(dv.astype(grad_dtype))
                dbs.append(db.astype(grad_dtype))
                dds.append(dd.astype(grad_dtype))
            dqs.reverse()
            dks.reverse()
            dvs.reverse()
            dbs.reverse()
            dds.reverse()
            return (
                cur_dst.astype(jnp.float32),
                jnp.stack(dqs, axis=0),
                jnp.stack(dks, axis=0),
                jnp.stack(dvs, axis=0),
                jnp.stack(dbs, axis=0),
                jnp.stack(dds, axis=0),
            )

        return jax.vmap(jax.vmap(_one))(state_pre, q_i, k_i, v_i, beta_i, decay_i, d_out_i, d_state_next)

    def _bwd_seg_step(
        state_pre,
        state_seg_pre,
        q_i,
        k_i,
        v_i,
        beta_i,
        decay_i,
        g_cumsum_i,
        d_out_i,
        seg_i,
        has_valid_i,
        seg_last_i,
        g_end_i,
        d_state_next,
        *,
        grad_dtype=jnp.float32,
    ):
        C = q_i.shape[3]

        def _one(st_pre, st_seg, qg, kg, vg, bg, dg, gcg, dog, segg, hasg, lastg, geg, dst_next):
            dst, dq, dk, dv, db, dd = gdr_bwd_mod._bwd_one_chunk_segmented(
                qg[0].astype(jnp.float32),
                kg[0].astype(jnp.float32),
                vg[0].astype(jnp.float32),
                bg[0, 0].astype(jnp.float32),
                dg[0, 0].astype(jnp.float32),
                gcg[0].astype(jnp.float32),
                dog[0].astype(jnp.float32),
                st_pre.astype(jnp.float32),
                dst_next.astype(jnp.float32),
                segg[0],
                st_seg,
                hasg[0].astype(jnp.float32),
                lastg[0],
                geg[0].astype(jnp.float32),
                C,
            )
            return (
                dst.astype(jnp.float32),
                dq[None].astype(grad_dtype),
                dk[None].astype(grad_dtype),
                dv[None].astype(grad_dtype),
                db[None].astype(grad_dtype),
                dd[None].astype(grad_dtype),
            )

        return jax.vmap(jax.vmap(_one))(
            state_pre,
            state_seg_pre,
            q_i,
            k_i,
            v_i,
            beta_i,
            decay_i,
            g_cumsum_i,
            d_out_i,
            seg_i,
            has_valid_i,
            seg_last_i,
            g_end_i,
            d_state_next,
        )

    def _single_step_fwd(query, key, value, beta, decay, recurrent_state):
        def _one(q, k, v, b, d, st):
            q_t = q[0].astype(jnp.float32)
            k_t = k[0].astype(jnp.float32)
            v_t = v[0].astype(jnp.float32)
            b_t = b[0].astype(jnp.float32)
            g_exp = jnp.exp(jnp.clip(d[0].astype(jnp.float32), -20.0, 20.0))
            st_dec = st.astype(jnp.float32) * g_exp
            kv_mem = jnp.sum(st_dec * k_t[:, None], axis=0)
            delta = (v_t - kv_mem) * b_t
            new_st = st_dec + k_t[:, None] * delta[None, :]
            out = jnp.sum(new_st * q_t[:, None], axis=0)
            return out[None].astype(query.dtype), new_st.astype(recurrent_state.dtype)

        return jax.vmap(jax.vmap(_one))(query, key, value, beta, decay, recurrent_state)

    def _single_step_bwd(query, key, value, beta, decay, recurrent_state, d_out, d_state_next):
        def _one(q, k, v, b, d, st, do, ds_next):
            q_t = q[0].astype(jnp.float32)
            k_t = k[0].astype(jnp.float32)
            v_t = v[0].astype(jnp.float32)
            b_t = b[0].astype(jnp.float32)
            d_t = d[0].astype(jnp.float32)
            st_prev = st.astype(jnp.float32)
            do_t = do[0].astype(jnp.float32)
            ds_n = ds_next.astype(jnp.float32)
            g_exp = jnp.exp(jnp.clip(d_t, -20.0, 20.0))
            st_dec = st_prev * g_exp
            kv_mem = jnp.sum(st_dec * k_t[:, None], axis=0)
            delta_raw = v_t - kv_mem
            delta = delta_raw * b_t
            state = st_dec + k_t[:, None] * delta[None, :]
            d_s = ds_n + q_t[:, None] * do_t[None, :]
            d_q = jnp.sum(state * do_t[None, :], axis=-1)
            d_k = jnp.sum(d_s * delta[None, :], axis=-1)
            d_delta = jnp.sum(d_s * k_t[:, None], axis=0)
            d_beta = jnp.sum(d_delta * delta_raw)
            d_v = d_delta * b_t
            d_kv_mem = -d_delta * b_t
            d_st_dec = d_s + k_t[:, None] * d_kv_mem[None, :]
            d_k = d_k + jnp.sum(st_dec * d_kv_mem[None, :], axis=-1)
            d_state = d_st_dec * g_exp
            d_decay = jnp.sum(d_st_dec * st_prev) * g_exp
            return d_q[None], d_k[None], d_v[None], d_beta.reshape(1, 1), d_decay.reshape(1, 1), d_state

        return jax.vmap(jax.vmap(_one))(query, key, value, beta, decay, recurrent_state, d_out, d_state_next)

    monkeypatch.setattr(gdr_fwd_mod, "_run_dense_fwd_step", _dense_step)
    monkeypatch.setattr(gdr_fwd_mod, "_run_grouped_dense_fwd_step", _grouped_dense_step)
    monkeypatch.setattr(gdr_fwd_mod, "_run_segmented_fwd_step", _seg_step)
    monkeypatch.setattr(gdr_fwd_mod, "_run_grouped_segmented_fwd_step", _grouped_seg_step)
    monkeypatch.setattr(gdr_fwd_mod, "_run_single_step_forward", _single_step_fwd)
    monkeypatch.setattr(gdr_bwd_mod, "_run_bwd_grad_step", _bwd_dense_step)
    monkeypatch.setattr(gdr_bwd_mod, "_run_bwd_grad_step_segmented", _bwd_seg_step)
    monkeypatch.setattr(gdr_bwd_mod, "_run_single_step_backward", _single_step_bwd)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("state_dtype", [jnp.float32, jnp.bfloat16])
def test_pallas_gdr_chunk_bwd_preserves_initial_state_primal_dtype_under_remat(
    monkeypatch, grouped, segmented, state_dtype
):
    """Pallas GDR custom-VJP backward must cast d_initial_state to primal initial_state.dtype.

    Reproduces the Qwen3Next bf16 + fp32-state remat / jnp.take prefix-fork crash where
    casting d_initial_state to query.dtype (bf16) caused mixed-precision remat / AD failures.
    """
    _install_cpu_gdr_step_mocks(monkeypatch)
    rng = np.random.default_rng(101)
    prompt_batch, num_gen, L, K, V, C = 2, 2, 16, 8, 8, 8
    fork_batch = prompt_batch * num_gen
    Hq = 2 if grouped else 4
    Hv = 4

    q = jnp.asarray(rng.standard_normal((fork_batch, Hq, L, K)), jnp.bfloat16)
    k = jnp.asarray(rng.standard_normal((fork_batch, Hq, L, K)), jnp.bfloat16)
    v = jnp.asarray(rng.standard_normal((fork_batch, Hv, L, V)), jnp.bfloat16)
    beta = jnp.asarray(jax.nn.sigmoid(rng.standard_normal((fork_batch, Hv, L))), jnp.bfloat16)
    decay = jnp.asarray(-np.abs(rng.standard_normal((fork_batch, Hv, L))) * 0.2, jnp.bfloat16)
    # Captured prompt state gathered across G generations with jnp.take (prefix-shared fork).
    captured_state = jnp.asarray(rng.standard_normal((prompt_batch, Hv, K, V)) * 0.1, state_dtype)
    seg_ids = (
        jnp.asarray([[0] * 12 + [-1] * 4] * fork_batch, dtype=jnp.int32)
        if segmented
        else None
    )
    fn = gdr_fwd_mod._chunk_gdr_grouped_fwd_pallas_chunk if grouped else gdr_fwd_mod._chunk_gdr_fwd_pallas_chunk
    rows = jnp.arange(fork_batch, dtype=jnp.int32) // num_gen

    @jax.jit
    def loss_fn(q_in, k_in, v_in, beta_in, decay_in, cap_state):
        # Remat wrapper around the GDR layer + coexisting cotangent on cap_state.
        init_state = jnp.take(cap_state, rows, axis=0)

        def layer(q_, k_, v_, b_, d_, s_):
            out, final_state = fn(q_, k_, v_, b_, d_, seg_ids, C, s_, True, False, False)
            return out, final_state

        out, final_state = jax.checkpoint(layer)(q_in, k_in, v_in, beta_in, decay_in, init_state)
        return (
            jnp.sum(out.astype(jnp.float32))
            + 0.1 * jnp.sum(final_state.astype(jnp.float32))
            + 0.05 * jnp.sum((cap_state * jnp.asarray(2.0, state_dtype)).astype(jnp.float32))
        )

    val, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, beta, decay, captured_state)
    dq, dk, dv, dbeta, ddecay, dstate = grads
    assert jnp.isfinite(val)
    assert dq.dtype == q.dtype and dq.shape == q.shape
    assert dk.dtype == k.dtype and dk.shape == k.shape
    assert dv.dtype == v.dtype and dv.shape == v.shape
    assert dbeta.dtype == beta.dtype and dbeta.shape == beta.shape
    assert ddecay.dtype == decay.dtype and ddecay.shape == decay.shape
    assert dstate.dtype == state_dtype and dstate.shape == captured_state.shape
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)


def test_pallas_gdr_single_step_bwd_preserves_state_primal_dtype_under_remat(monkeypatch):
    """Single-step Pallas GDR backward must also preserve fp32 recurrent_state primal dtype."""
    _install_cpu_gdr_step_mocks(monkeypatch)
    rng = np.random.default_rng(102)
    B, H, K, V = 2, 4, 8, 8
    q = jnp.asarray(rng.standard_normal((B, H, 1, K)), jnp.bfloat16)
    k = jnp.asarray(rng.standard_normal((B, H, 1, K)), jnp.bfloat16)
    v = jnp.asarray(rng.standard_normal((B, H, 1, V)), jnp.bfloat16)
    beta = jnp.asarray(jax.nn.sigmoid(rng.standard_normal((B, H, 1))), jnp.bfloat16)
    decay = jnp.asarray(-np.abs(rng.standard_normal((B, H, 1))) * 0.2, jnp.float32)
    state = jnp.asarray(rng.standard_normal((B, H, K, V)) * 0.1, jnp.float32)

    @jax.jit
    def loss_fn(q_in, k_in, v_in, beta_in, decay_in, state_in):
        out, final_state = jax.checkpoint(
            lambda q_, k_, v_, b_, d_, s_: gdr_fwd_mod._single_step_gdr_fwd(q_, k_, v_, b_, d_, s_, True)
        )(q_in, k_in, v_in, beta_in, decay_in, state_in)
        return (
            jnp.sum(out.astype(jnp.float32))
            + 0.1 * jnp.sum(final_state.astype(jnp.float32))
            + 0.05 * jnp.sum(state_in * jnp.float32(2.0))
        )

    val, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, beta, decay, state)
    dq, dk, dv, dbeta, ddecay, dstate = grads
    assert jnp.isfinite(val)
    assert dq.dtype == jnp.bfloat16
    assert dk.dtype == jnp.bfloat16
    assert dv.dtype == jnp.bfloat16
    assert dbeta.dtype == jnp.bfloat16
    assert ddecay.dtype == jnp.float32
    assert dstate.dtype == jnp.float32
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)


@pytest.mark.parametrize("grouped", [False, True])
def test_pallas_gdr_none_initial_state_and_fp32_decay_grad_dtypes(monkeypatch, grouped):
    """initial_state=None keeps None grad; fp32 decay with bf16 Q/K/V keeps fp32 d_decay."""
    _install_cpu_gdr_step_mocks(monkeypatch)
    rng = np.random.default_rng(103)
    B, L, K, V, C = 2, 16, 8, 8, 8
    Hq = 2 if grouped else 4
    Hv = 4
    q = jnp.asarray(rng.standard_normal((B, Hq, L, K)), jnp.bfloat16)
    k = jnp.asarray(rng.standard_normal((B, Hq, L, K)), jnp.bfloat16)
    v = jnp.asarray(rng.standard_normal((B, Hv, L, V)), jnp.bfloat16)
    beta = jnp.asarray(jax.nn.sigmoid(rng.standard_normal((B, Hv, L))), jnp.bfloat16)
    decay = jnp.asarray(-np.abs(rng.standard_normal((B, Hv, L))) * 0.2, jnp.float32)
    fn = gdr_fwd_mod._chunk_gdr_grouped_fwd_pallas_chunk if grouped else gdr_fwd_mod._chunk_gdr_fwd_pallas_chunk

    @jax.jit
    def loss_fn(q_in, k_in, v_in, beta_in, decay_in):
        out, final_state = jax.checkpoint(lambda *args: fn(*args, None, C, None, True, False, False))(
            q_in, k_in, v_in, beta_in, decay_in
        )
        return (
            jnp.sum(out.astype(jnp.float32))
            + 0.1 * jnp.sum(final_state.astype(jnp.float32))
            + 0.05 * jnp.sum(decay_in * jnp.float32(2.0))
        )

    val, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v, beta, decay)
    assert jnp.isfinite(val)
    assert grads[4].dtype == jnp.float32
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)


# -------------------------------------------------------------------------------------- Finding #29


def _prefix_flash_inputs(batch: int = 4, prefix: int = 12, completion: int = 8, hq: int = 4, hkv: int = 2, d: int = 16):
    rng = np.random.default_rng(201)
    kv_len = prefix + completion
    q = jnp.asarray(rng.standard_normal((batch, completion, hq, d)), jnp.float32)
    k = jnp.asarray(rng.standard_normal((batch, kv_len, hkv, d)), jnp.float32)
    v = jnp.asarray(rng.standard_normal((batch, kv_len, hkv, d)), jnp.float32)
    prompt_valid = np.ones((batch, prefix), dtype=bool)
    prompt_valid[0, :3] = False
    comp_valid = np.ones((batch, completion), dtype=bool)
    comp_valid[1, -2:] = False
    q_seg = jnp.asarray(np.where(comp_valid, 0, -1), jnp.int32)
    kv_seg = jnp.asarray(np.concatenate([np.where(prompt_valid, 0, -1), np.asarray(q_seg)], axis=1), jnp.int32)
    q_pos = jnp.broadcast_to(prefix + jnp.arange(completion, dtype=jnp.int32), (batch, completion))
    kv_pos = jnp.broadcast_to(jnp.arange(kv_len, dtype=jnp.int32), (batch, kv_len))
    mask = MaskInfo.from_segments(q_segment_ids=q_seg, kv_segment_ids=kv_seg, q_positions=q_pos, kv_positions=kv_pos)
    return q, k, v, mask


def _dense_ref_bthd(q, k, v, mask_info, bias=None):
    hq, hkv, d = q.shape[2], k.shape[2], q.shape[3]
    k_rep = jnp.repeat(k, hq // hkv, axis=2)
    v_rep = jnp.repeat(v, hq // hkv, axis=2)
    visible = mask_info.get_position_aware_attention_mask(causal=True, sliding_window=None)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k_rep, precision=lax.Precision.HIGHEST) * (d**-0.5)
    if bias is not None:
        logits = logits + bias.astype(jnp.float32)
    logits = jnp.where(visible, logits, -1e30)
    weights = jnp.where(visible, jax.nn.softmax(logits, axis=-1), 0.0)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v_rep, precision=lax.Precision.HIGHEST)


@needs_4_devices
@pytest.mark.parametrize(
    "bias_shape,bias_spec",
    [
        (None, None),
        ((4, 1, 8, 20), PartitionSpec("dp", None, None, None)),
        ((1, 1, 8, 20), PartitionSpec(None, None, None, None)),
        ((4, 4, 8, 20), PartitionSpec("dp", "tp", None, None)),
    ],
)
def test_flash_attention_position_bias_tp_mesh_with_broadcast_caller_bias(bias_shape, bias_spec):
    """TP+DP shard_map must recompute bias_spec when position_bias expands caller bias to num_heads."""
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("dp", "tp"))
    q, k, v, mask = _prefix_flash_inputs(batch=4, prefix=12, completion=8, hq=4, hkv=2, d=16)
    rng = np.random.default_rng(202)
    bias = None if bias_shape is None else jnp.asarray(rng.standard_normal(bias_shape), jnp.float32)

    q_spec = PartitionSpec("dp", None, "tp", None)
    k_spec = PartitionSpec("dp", None, "tp", None)
    v_spec = PartitionSpec("dp", None, "tp", None)
    in_specs = (q_spec, k_spec, v_spec, bias_spec, None, None, None)

    q_s = jax.device_put(q, NamedSharding(mesh, q_spec))
    k_s = jax.device_put(k, NamedSharding(mesh, k_spec))
    v_s = jax.device_put(v, NamedSharding(mesh, v_spec))
    bias_s = None if bias is None else jax.device_put(bias, NamedSharding(mesh, bias_spec))
    cot = jnp.asarray(rng.standard_normal(q.shape), jnp.float32)
    q_valid = (mask._q_segment_ids >= 0)[:, :, None, None].astype(jnp.float32)

    def sharded_loss(q_in, k_in, v_in):
        out = flash_attention(
            q_in,
            k_in,
            v_in,
            bias_s,
            mask_info=mask,
            causal=True,
            precision=lax.Precision.HIGHEST,
            platform="xla",
            mesh=mesh,
            in_specs=in_specs,
            out_specs=q_spec,
        )
        return jnp.sum(out * cot * q_valid), out

    def ref_loss(q_in, k_in, v_in):
        out = _dense_ref_bthd(q_in, k_in, v_in, mask, bias=bias)
        return jnp.sum(out * cot * q_valid), out

    with jax.default_matmul_precision("highest"), mesh:
        (_, out_got), grads_got = jax.value_and_grad(sharded_loss, argnums=(0, 1, 2), has_aux=True)(q_s, k_s, v_s)
        (_, out_ref), grads_ref = jax.value_and_grad(ref_loss, argnums=(0, 1, 2), has_aux=True)(q, k, v)

    np.testing.assert_allclose(np.asarray(out_got), np.asarray(out_ref), rtol=1e-4, atol=1e-4)
    for g_got, g_ref in zip(grads_got, grads_ref, strict=True):
        np.testing.assert_allclose(np.asarray(g_got), np.asarray(g_ref), rtol=1e-4, atol=1e-4)


@needs_4_devices
@pytest.mark.parametrize(
    "q_seq,k_seq",
    [
        ("sp", "sp"),
        ("sp", None),
        (None, "sp"),
    ],
)
def test_flash_attention_position_bias_rejects_active_sp_sequence_sharding(q_seq, k_seq):
    """Active SP sequence sharding on position-bias path must raise NotImplementedError (not DuplicateSpecError)."""
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("sp", "tp"))
    q, k, v, mask = _prefix_flash_inputs(batch=2, prefix=12, completion=8, hq=4, hkv=2, d=16)
    q_spec = PartitionSpec(None, q_seq, "tp", None)
    k_spec = PartitionSpec(None, k_seq, "tp", None)
    in_specs = (q_spec, k_spec, k_spec, None, None, None, None)

    with pytest.raises(NotImplementedError, match="sequence axis"):
        flash_attention(
            q,
            k,
            v,
            mask_info=mask,
            causal=True,
            platform="xla",
            mesh=mesh,
            in_specs=in_specs,
            out_specs=q_spec,
        )


@needs_4_devices
def test_flash_attention_position_bias_allows_size1_sp_label_without_duplicate_spec_error():
    """When sp mesh axis has size 1, naming 'sp' on both q_spec and k_spec must not trigger DuplicateSpecError."""
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 1, 2), ("dp", "sp", "tp"))
    q, k, v, mask = _prefix_flash_inputs(batch=4, prefix=12, completion=8, hq=4, hkv=2, d=16)
    q_spec = PartitionSpec("dp", "sp", "tp", None)
    k_spec = PartitionSpec("dp", "sp", "tp", None)
    in_specs = (q_spec, k_spec, k_spec, None, None, None, None)

    with jax.default_matmul_precision("highest"), mesh:
        out = flash_attention(
            jax.device_put(q, NamedSharding(mesh, q_spec)),
            jax.device_put(k, NamedSharding(mesh, k_spec)),
            jax.device_put(v, NamedSharding(mesh, k_spec)),
            mask_info=mask,
            causal=True,
            precision=lax.Precision.HIGHEST,
            platform="xla",
            mesh=mesh,
            in_specs=in_specs,
            out_specs=q_spec,
        )
        ref = _dense_ref_bthd(q, k, v, mask)

    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-4, atol=1e-4)


def test_flash_attention_position_bias_cuda_platform_routing_and_gpu_candidates(monkeypatch):
    """Position-bias path must route away from CUDA and exclude CUDA from GPU candidates for 4-D bias."""
    q, k, v, mask = _prefix_flash_inputs(batch=2, prefix=12, completion=8, hq=4, hkv=2, d=16)
    with jax.default_matmul_precision("highest"):
        ref = _dense_ref_bthd(q, k, v, mask)
        out_plat = flash_attention(
            q, k, v, mask_info=mask, causal=True, precision=lax.Precision.HIGHEST, platform="cuda"
        )
        out_cfg = flash_attention(
            q,
            k,
            v,
            mask_info=mask,
            causal=True,
            precision=lax.Precision.HIGHEST,
            cfg=FlashAttentionConfig(platform="cuda", backend="gpu"),
        )
    np.testing.assert_allclose(np.asarray(out_plat), np.asarray(ref), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(out_cfg), np.asarray(ref), rtol=1e-4, atol=1e-4)

    # GPU autotune candidates omit "cuda" when a 4-D additive bias is present, but retain it for None / 1-D ALiBi.
    kernel = FlashAttention()
    q_np = np.empty((1, 64, 4, 64), dtype=np.float16)
    k_np = np.empty((1, 64, 2, 64), dtype=np.float16)
    bias_4d = np.empty((1, 4, 64, 64), dtype=np.float32)
    bias_alibi = np.empty((4,), dtype=np.float32)

    cfgs_4d = kernel.candidate_cfgs_gpu(
        Invocation(op_id="flash_attention", args=(), kwargs={"query": q_np, "key": k_np, "bias": bias_4d})
    )
    assert "cuda" not in {c.platform for c in cfgs_4d}
    assert "triton" in {c.platform for c in cfgs_4d} and "xla" in {c.platform for c in cfgs_4d}

    cfgs_alibi = kernel.candidate_cfgs_gpu(
        Invocation(op_id="flash_attention", args=(), kwargs={"query": q_np, "key": k_np, "bias": bias_alibi})
    )
    assert "cuda" in {c.platform for c in cfgs_alibi}

    # Auto-detection fallback when detect_platform still returns CUDA on a 4-D bias call.
    flash_mod = importlib.import_module("ejkernel.modules.operations.flash_attention")
    captured = {}

    def fake_detect(*args, **kwargs):
        captured["kwargs"] = kwargs
        return Platform.CUDA

    def fake_get(*, algorithm, platform, backend):
        captured["resolved"] = (algorithm, platform, backend)
        return lambda **kw: kw["query"]

    monkeypatch.setattr(flash_mod, "detect_platform", fake_detect)
    monkeypatch.setattr(flash_mod.kernel_registry, "get", fake_get)
    kernel.get_impl(FlashAttentionConfig(platform="auto", backend="any"), prefer_cuda=False)
    assert captured["kwargs"]["prefer_cuda"] is False
    assert captured["resolved"] == ("flash_attention", Platform.XLA, Backend.ANY)


@pytest.mark.parametrize("platform_arg", [None, "auto"])
@pytest.mark.parametrize("use_shard_map", [False, True])
def test_flash_attention_position_bias_nvidia_prefer_cuda_cfg_never_dispatches_cuda(
    monkeypatch, platform_arg, use_shard_map
):
    """Simulated NVIDIA GPU with Executor._prefer_cuda_cfg must never dispatch 4-D position_bias to Platform.CUDA."""
    if use_shard_map and jax.device_count() < 4:
        pytest.skip("needs a 4-device CPU mesh")

    exec_mod = importlib.import_module("ejkernel.ops.execution.executor")
    flash_mod = importlib.import_module("ejkernel.modules.operations.flash_attention")
    from ejkernel.ops import ConfigCache, policy_override

    executor = flash_mod._flash_executor
    monkeypatch.setattr(executor.chooser, "cache", ConfigCache())
    monkeypatch.setattr(executor.chooser, "persistent", None)
    monkeypatch.setattr(executor.chooser, "_autotuned_keys", set())

    # Simulate an NVIDIA GPU with CUDA available and no CuTe implementation, so
    # Executor._prefer_cuda_cfg actively upgrades heuristic/cached cfg to platform="cuda", backend="gpu".
    monkeypatch.setattr(exec_mod, "get_device_platform", lambda: "gpu")
    monkeypatch.setattr(executor, "_is_nvidia_gpu", lambda: True)
    monkeypatch.setattr(executor, "_has_cute_impl", lambda op_id: False)
    monkeypatch.setattr(executor, "_has_cuda_impl", lambda op_id: True)

    prefer_calls = []
    orig_prefer = executor._prefer_cuda_cfg

    def spy_prefer(cfg, kernel, inv):
        out_cfg = orig_prefer(cfg, kernel, inv)
        prefer_calls.append(out_cfg)
        return out_cfg

    monkeypatch.setattr(executor, "_prefer_cuda_cfg", spy_prefer)

    dispatched = []
    orig_registry_get = flash_mod.kernel_registry.get

    def spy_registry_get(*, algorithm, platform, backend):
        dispatched.append((algorithm, platform, backend))
        assert platform != Platform.CUDA, f"Platform.CUDA must not be dispatched for 4-D position_bias: {dispatched}"
        return orig_registry_get(algorithm=algorithm, platform=platform, backend=backend)

    # Even if detect_platform returns CUDA for "auto", get_impl must fall back to XLA + Backend.ANY.
    monkeypatch.setattr(
        flash_mod,
        "detect_platform",
        lambda alg, plat, **kw: Platform.CUDA if plat in ("auto", None, "cuda") else Platform(plat),
    )
    monkeypatch.setattr(flash_mod.kernel_registry, "get", spy_registry_get)

    batch = 4 if use_shard_map else 2
    q, k, v, mask = _prefix_flash_inputs(batch=batch, prefix=12, completion=8, hq=4, hkv=2, d=16)

    with (
        jax.default_matmul_precision("highest"),
        policy_override(executor.chooser, allow_autotune=False, cache_miss_fallback="heuristics"),
    ):
        ref = _dense_ref_bthd(q, k, v, mask)
        if use_shard_map:
            mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("dp", "tp"))
            q_spec = PartitionSpec("dp", None, "tp", None)
            in_specs = (q_spec, q_spec, q_spec, None, None, None, None)
            with mesh:
                out = flash_attention(
                    jax.device_put(q, NamedSharding(mesh, q_spec)),
                    jax.device_put(k, NamedSharding(mesh, q_spec)),
                    jax.device_put(v, NamedSharding(mesh, q_spec)),
                    mask_info=mask,
                    causal=True,
                    precision=lax.Precision.HIGHEST,
                    platform=platform_arg,
                    mesh=mesh,
                    in_specs=in_specs,
                    out_specs=q_spec,
                )
        else:
            out = flash_attention(
                q,
                k,
                v,
                mask_info=mask,
                causal=True,
                precision=lax.Precision.HIGHEST,
                platform=platform_arg,
            )

    assert prefer_calls and any(str(c.platform) == "cuda" and str(c.backend) == "gpu" for c in prefer_calls), (
        f"Expected _prefer_cuda_cfg to upgrade cfg to cuda/gpu, saw: {prefer_calls}"
    )
    assert dispatched == [("flash_attention", Platform.XLA, Backend.ANY)], dispatched
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-4, atol=1e-4)

    # Confirm explicit platform="cuda" on a non-position_bias call still resolves to Platform.CUDA.
    explicit_dispatched = []
    monkeypatch.setattr(
        flash_mod.kernel_registry,
        "get",
        lambda *, algorithm, platform, backend: explicit_dispatched.append((platform, backend))
        or (lambda **kw: kw["query"]),
    )
    FlashAttention().run(
        query=q,
        key=k,
        value=v,
        bias=jnp.zeros((batch, 4, 8, 20), dtype=jnp.float32),
        platform="cuda",
        cfg=FlashAttentionConfig(platform="auto", backend="gpu"),
    )
    assert explicit_dispatched == [(Platform.CUDA, "gpu")]


# -------------------------------------------------------------------------------------- Finding #44


def _blocksparse_inputs(batch: int = 2, prefix: int = 12, completion: int = 8, hq: int = 4, hkv: int = 2, d: int = 16):
    rng = np.random.default_rng(301)
    kv_len = prefix + completion
    q = jnp.asarray(rng.standard_normal((batch, hq, completion, d)), jnp.float32)
    k = jnp.asarray(rng.standard_normal((batch, hkv, kv_len, d)), jnp.float32)
    v = jnp.asarray(rng.standard_normal((batch, hkv, kv_len, d)), jnp.float32)
    prompt_valid = np.ones((batch, prefix), dtype=bool)
    prompt_valid[0, :3] = False
    comp_valid = np.ones((batch, completion), dtype=bool)
    comp_valid[1, -2:] = False
    q_seg = jnp.asarray(np.where(comp_valid, 0, -1), jnp.int32)
    kv_seg = jnp.asarray(np.concatenate([np.where(prompt_valid, 0, -1), np.asarray(q_seg)], axis=1), jnp.int32)
    q_pos = jnp.broadcast_to(prefix + jnp.arange(completion, dtype=jnp.int32), (batch, completion))
    kv_pos = jnp.broadcast_to(jnp.arange(kv_len, dtype=jnp.int32), (batch, kv_len))
    return q, k, v, q_seg, kv_seg, q_pos, kv_pos


def test_blocksparse_attention_does_not_mutate_mask_info_on_reuse():
    """Calling blocksparse_attention with method=None must not mutate mask_info._attention_mask."""
    q, k, v, q_seg, kv_seg, q_pos, kv_pos = _blocksparse_inputs()
    mask_info = MaskInfo.from_segments(
        q_segment_ids=q_seg,
        kv_segment_ids=kv_seg,
        q_positions=q_pos,
        kv_positions=kv_pos,
    )
    assert mask_info._attention_mask is None
    assert mask_info.uses_position_causality() is True

    with jax.default_matmul_precision("highest"):
        out1 = blocksparse_attention(q, k, v, mask_info=mask_info, causal=True, platform="xla")
        assert mask_info._attention_mask is None
        assert mask_info.uses_position_causality() is True
        out2 = blocksparse_attention(q, k, v, mask_info=mask_info, causal=True, platform="xla")
        assert mask_info._attention_mask is None
        assert mask_info.uses_position_causality() is True

    np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))
    ref = _dense_ref_bthd(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3), mask_info)
    np.testing.assert_allclose(np.asarray(out1.transpose(0, 2, 1, 3)), np.asarray(ref), rtol=1e-4, atol=1e-4)


def test_blocksparse_attention_does_not_mutate_segment_only_mask_positions():
    """Segment-only MaskInfo without explicit positions must keep q_positions/kv_positions as None on reuse."""
    q, k, v, q_seg, kv_seg, _, _ = _blocksparse_inputs()
    mask_info = MaskInfo.from_segments(q_segment_ids=q_seg, kv_segment_ids=kv_seg)
    assert mask_info.q_positions is None
    assert mask_info.kv_positions is None
    assert mask_info._attention_mask is None
    assert mask_info.uses_position_causality() is False

    out1 = blocksparse_attention(q, k, v, mask_info=mask_info, causal=True, platform="xla")
    assert mask_info.q_positions is None
    assert mask_info.kv_positions is None
    assert mask_info._attention_mask is None
    assert mask_info.uses_position_causality() is False

    out2 = blocksparse_attention(q, k, v, mask_info=mask_info, causal=True, platform="xla")
    np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))


def test_blocksparse_attention_preserves_stored_dense_custom_mask_without_mutating_caller():
    """A MaskInfo carrying a custom stored dense mask must still apply it without mutating the instance."""
    q, k, v, _, _, _, _ = _blocksparse_inputs(batch=2, prefix=4, completion=6, hq=4, hkv=2, d=16)
    kv_len = k.shape[2]
    q_len = q.shape[2]
    # Non-segment-reducible pattern: query i attends only to keys with (i + j) % 2 == 0.
    pattern = ((np.arange(q_len)[:, None] + np.arange(kv_len)[None, :]) % 2) == 0
    dense = jnp.broadcast_to(jnp.asarray(pattern)[None, None, :, :], (2, 1, q_len, kv_len))
    mask_info = MaskInfo(_attention_mask=dense)
    assert mask_info._q_segment_ids is None

    with jax.default_matmul_precision("highest"):
        out = blocksparse_attention(q, k, v, mask_info=mask_info, causal=False, platform="xla")
        k_rep = jnp.repeat(k, 2, axis=1)
        v_rep = jnp.repeat(v, 2, axis=1)
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k_rep, precision=lax.Precision.HIGHEST) * (16**-0.5)
        logits = jnp.where(dense, logits, -1e30)
        weights = jnp.where(dense, jax.nn.softmax(logits, axis=-1), 0.0)
        ref = jnp.einsum("bhqk,bhkd->bhqd", weights, v_rep, precision=lax.Precision.HIGHEST)

    assert mask_info._q_segment_ids is None
    assert mask_info._kv_segment_ids is None
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-4, atol=1e-4)


def test_blocksparse_pallas_position_gate_still_triggers_on_reused_mask_info(monkeypatch):
    """Reusing MaskInfo across two Pallas-gated calls must keep the position-offset mask_builder active."""
    bs_mod = importlib.import_module("ejkernel.modules.operations.blocksparse_attention")
    q, k, v, q_seg, kv_seg, q_pos, kv_pos = _blocksparse_inputs()
    mask_info = MaskInfo.from_segments(
        q_segment_ids=q_seg,
        kv_segment_ids=kv_seg,
        q_positions=q_pos,
        kv_positions=kv_pos,
    )
    seen_builders = []
    seen_masks = []

    def fake_executor(op, **kwargs):
        del op
        seen_builders.append(kwargs["mask_builder"])
        seen_masks.append(kwargs.get("attention_mask"))
        return jnp.zeros_like(kwargs["query"])

    monkeypatch.setattr(bs_mod, "_resolves_to_pallas_tpu", lambda platform, cfg: True)
    monkeypatch.setattr(bs_mod, "_executor", fake_executor)

    blocksparse_attention(q, k, v, mask_info=mask_info, causal=True)
    blocksparse_attention(q, k, v, mask_info=mask_info, causal=True)

    assert mask_info._attention_mask is None
    assert mask_info.uses_position_causality() is True
    assert len(seen_builders) == 2
    assert seen_builders[0] is not None and seen_builders[1] is not None
    assert seen_masks == [None, None]
