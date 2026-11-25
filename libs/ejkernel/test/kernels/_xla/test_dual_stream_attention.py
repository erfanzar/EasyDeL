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

"""XLA reference tests for Dual-Stream Hybrid Attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.dual_stream_attention import dual_stream_attention


def _naive_dense_dual_stream(q, kg, vg, kl, vl, g_mask, l_mask, scale):
    """Unblocked dense fp32 reference over concatenated ``[global | local]`` KV streams."""
    nq = q.shape[1]
    nkv = kg.shape[1]
    groups = nq // nkv
    kg_rep = jnp.repeat(kg, groups, axis=1).astype(jnp.float32)
    vg_rep = jnp.repeat(vg, groups, axis=1).astype(jnp.float32)
    kl_rep = jnp.repeat(kl, groups, axis=1).astype(jnp.float32)
    vl_rep = jnp.repeat(vl, groups, axis=1).astype(jnp.float32)
    k_all = jnp.concatenate([kg_rep, kl_rep], axis=2)
    v_all = jnp.concatenate([vg_rep, vl_rep], axis=2)
    m_all = jnp.concatenate([g_mask, l_mask], axis=-1)
    scores = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32), k_all) * scale
    has_any = jnp.any(m_all, axis=-1, keepdims=True)
    scores_masked = jnp.where(m_all, scores, -1e30)
    m = jnp.max(scores_masked, axis=-1, keepdims=True)
    probs = jnp.where(m_all, jnp.exp(scores_masked - m), 0.0)
    probs = probs / jnp.maximum(jnp.sum(probs, axis=-1, keepdims=True), 1e-30)
    out = jnp.einsum("bhqk,bhkd->bhqd", probs, v_all)
    return jnp.where(has_any, out, 0.0).astype(q.dtype)


@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("q_len,q_chunk_size,kv_chunk_size", [(32, 64, 64), (96, 32, 32)])
def test_xla_dual_stream_matches_dense_reference_and_grads(overlap, q_len, q_chunk_size, kv_chunk_size):
    """Both direct and K-blocked XLA paths match the concatenated dense reference in value and grad."""
    rng = np.random.default_rng(7 if overlap else 11)
    bsz, nq, nkv, dim, win = 2, 4, 2, 16, 12
    q = jnp.asarray(rng.standard_normal((bsz, nq, q_len, dim)) * 0.3, jnp.float32)
    kg = jnp.asarray(rng.standard_normal((bsz, nkv, q_len, dim)) * 0.3, jnp.float32)
    vg = jnp.asarray(rng.standard_normal((bsz, nkv, q_len, dim)) * 0.3, jnp.float32)
    kl = jnp.asarray(rng.standard_normal((bsz, nkv, q_len, dim)) * 0.3, jnp.float32)
    vl = jnp.asarray(rng.standard_normal((bsz, nkv, q_len, dim)) * 0.3, jnp.float32)

    # Packed segments including trailing -1 padding tokens.
    seg = np.zeros((bsz, q_len), dtype=np.int32)
    seg[0, q_len // 2 : q_len - 4] = 1
    seg[0, q_len - 4 :] = -1
    seg[1, q_len // 3 : 2 * q_len // 3] = 1
    seg[1, 2 * q_len // 3 :] = 2
    seg_j = jnp.asarray(seg)

    pos = np.arange(q_len, dtype=np.int32)
    rel = pos[:, None] - pos[None, :]
    same = (seg[:, :, None] >= 0) & (seg[:, :, None] == seg[:, None, :])
    g_mask = jnp.asarray(((rel >= 0) if overlap else (rel >= win))[None, None, :, :] & same[:, None, :, :])
    l_mask = jnp.asarray(((rel >= 0) & (rel < win))[None, None, :, :] & same[:, None, :, :])
    scale = dim**-0.5

    cot = jnp.asarray(rng.standard_normal((bsz, nq, q_len, dim)), jnp.float32)

    def _xla_loss(qa, kga, vga, kla, vla):
        out = dual_stream_attention(
            qa,
            kga,
            vga,
            kla,
            vla,
            g_q_seg=seg_j,
            g_kv_seg=seg_j,
            l_q_seg=seg_j,
            l_kv_seg=seg_j,
            softmax_scale=scale,
            sliding_window=win,
            overlap=overlap,
            q_chunk_size=q_chunk_size,
            kv_chunk_size=kv_chunk_size,
        )
        return jnp.sum(out * cot), out

    def _ref_loss(qa, kga, vga, kla, vla):
        out = _naive_dense_dual_stream(qa, kga, vga, kla, vla, g_mask, l_mask, scale)
        return jnp.sum(out * cot), out

    with jax.default_matmul_precision("highest"):
        (_, out_xla), grads_xla = jax.value_and_grad(_xla_loss, argnums=(0, 1, 2, 3, 4), has_aux=True)(
            q, kg, vg, kl, vl
        )
        (_, out_ref), grads_ref = jax.value_and_grad(_ref_loss, argnums=(0, 1, 2, 3, 4), has_aux=True)(
            q, kg, vg, kl, vl
        )

    assert np.isfinite(np.asarray(out_xla)).all()
    np.testing.assert_allclose(np.asarray(out_xla), np.asarray(out_ref), atol=1e-5, rtol=1e-5)
    for gx, gr in zip(grads_xla, grads_ref, strict=True):
        assert np.isfinite(np.asarray(gx)).all()
        np.testing.assert_allclose(np.asarray(gx), np.asarray(gr), atol=1e-4, rtol=1e-4)


def test_xla_dual_stream_cached_decode_asymmetric_kv():
    """Explicit masks handle single-token decode over asymmetric global/local KV cache lengths."""
    rng = np.random.default_rng(19)
    bsz, nq, nkv, dim = 2, 4, 2, 16
    kv_g_len, kv_l_len = 64, 16
    q = jnp.asarray(rng.standard_normal((bsz, nq, 1, dim)) * 0.3, jnp.float32)
    kg = jnp.asarray(rng.standard_normal((bsz, nkv, kv_g_len, dim)) * 0.3, jnp.float32)
    vg = jnp.asarray(rng.standard_normal((bsz, nkv, kv_g_len, dim)) * 0.3, jnp.float32)
    kl = jnp.asarray(rng.standard_normal((bsz, nkv, kv_l_len, dim)) * 0.3, jnp.float32)
    vl = jnp.asarray(rng.standard_normal((bsz, nkv, kv_l_len, dim)) * 0.3, jnp.float32)

    g_mask = jnp.asarray(rng.random((bsz, 1, 1, kv_g_len)) > 0.3)
    l_mask = jnp.asarray(rng.random((bsz, 1, 1, kv_l_len)) > 0.3)
    scale = dim**-0.5

    with jax.default_matmul_precision("highest"):
        out_xla = dual_stream_attention(
            q, kg, vg, kl, vl, global_mask=g_mask, local_mask=l_mask, softmax_scale=scale, sliding_window=16
        )
        out_ref = _naive_dense_dual_stream(q, kg, vg, kl, vl, g_mask, l_mask, scale)
    np.testing.assert_allclose(np.asarray(out_xla), np.asarray(out_ref), atol=1e-5, rtol=1e-5)
