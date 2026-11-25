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

"""Pallas TPU parity tests for Dual-Stream Hybrid Attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _has_tpu() -> bool:
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require a TPU backend")


@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("use_seg_ids", [True, False])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_pallas_dual_stream_matches_xla_fwd_and_bwd(overlap, use_seg_ids, dtype):
    """Pallas TPU dual-stream SplashAttention matches XLA reference across forward and all 5 gradients."""
    from ejkernel.kernels._pallas.tpu.dual_stream_attention import dual_stream_attention as pallas_impl
    from ejkernel.kernels._xla.dual_stream_attention import dual_stream_attention as xla_impl

    rng = np.random.default_rng(101 if overlap else 202)
    bsz, nq, nkv, seq_len, dim, win = 2, 4, 2, 256, 128, 64
    q = jnp.asarray(rng.standard_normal((bsz, nq, seq_len, dim)) * 0.25, dtype)
    kg = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.25, dtype)
    vg = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.25, dtype)
    kl = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.25, dtype)
    vl = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.25, dtype)
    cot = jnp.asarray(rng.standard_normal((bsz, nq, seq_len, dim)) * 0.25, dtype)

    seg_j = None
    if use_seg_ids:
        seg = np.zeros((bsz, seq_len), dtype=np.int32)
        seg[0, 128:240] = 1
        seg[0, 240:] = -1
        seg[1, 96:192] = 1
        seg[1, 192:] = 2
        seg_j = jnp.asarray(seg)

    scale = dim**-0.5

    def _loss(impl, qa, kga, vga, kla, vla):
        out = impl(
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
        )
        return jnp.sum(out.astype(jnp.float32) * cot.astype(jnp.float32)), out

    (_, out_p), grads_p = jax.value_and_grad(
        lambda *a: _loss(pallas_impl, *a), argnums=(0, 1, 2, 3, 4), has_aux=True
    )(q, kg, vg, kl, vl)
    (_, out_x), grads_x = jax.value_and_grad(
        lambda *a: _loss(xla_impl, *a), argnums=(0, 1, 2, 3, 4), has_aux=True
    )(q, kg, vg, kl, vl)

    out_tol = 2e-2 if dtype == jnp.bfloat16 else 2e-3
    grad_tol = 6e-2 if dtype == jnp.bfloat16 else 5e-3
    assert np.isfinite(np.asarray(out_p, dtype=np.float32)).all()
    assert float(np.max(np.abs(np.asarray(out_p, np.float32) - np.asarray(out_x, np.float32)))) < out_tol
    for gp, gx in zip(grads_p, grads_x, strict=True):
        gp_f = np.asarray(gp, np.float32)
        gx_f = np.asarray(gx, np.float32)
        assert np.isfinite(gp_f).all()
        assert float(np.max(np.abs(gp_f - gx_f))) < grad_tol


def test_pallas_unaligned_falls_back_to_xla_cleanly():
    """Shapes outside SplashAttention alignment (e.g. q_len=37 or decode q_len=1) fall back to XLA seamlessly."""
    from ejkernel.kernels._pallas.tpu.dual_stream_attention import dual_stream_attention as pallas_impl
    from ejkernel.kernels._xla.dual_stream_attention import dual_stream_attention as xla_impl

    rng = np.random.default_rng(77)
    bsz, nq, nkv, seq_len, dim = 1, 4, 2, 37, 64
    q = jnp.asarray(rng.standard_normal((bsz, nq, seq_len, dim)) * 0.2, jnp.float32)
    kg = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.2, jnp.float32)
    vg = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.2, jnp.float32)
    kl = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.2, jnp.float32)
    vl = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.2, jnp.float32)

    out_p = pallas_impl(q, kg, vg, kl, vl, sliding_window=16, overlap=False)
    out_x = xla_impl(q, kg, vg, kl, vl, sliding_window=16, overlap=False)
    np.testing.assert_allclose(np.asarray(out_p), np.asarray(out_x), atol=1e-5, rtol=1e-5)


def test_pallas_head_dim_256_fits_scoped_vmem_fwd_and_bwd():
    """Verify Qwen3.8-27B head_dim=256 compiles and differentiates within TPU's 16 MiB scoped VMEM."""
    from ejkernel.kernels._pallas.tpu.dual_stream_attention import dual_stream_attention as pallas_impl

    rng = np.random.default_rng(256)
    bsz, nq, nkv, seq_len, dim, win = 1, 6, 1, 1024, 256, 128
    q = jnp.asarray(rng.standard_normal((bsz, nq, seq_len, dim)) * 0.2, jnp.bfloat16)
    kg = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.2, jnp.bfloat16)
    vg = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.2, jnp.bfloat16)
    kl = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.2, jnp.bfloat16)
    vl = jnp.asarray(rng.standard_normal((bsz, nkv, seq_len, dim)) * 0.2, jnp.bfloat16)
    seg = jnp.zeros((bsz, seq_len), dtype=jnp.int32)
    cot = jnp.asarray(rng.standard_normal((bsz, nq, seq_len, dim)) * 0.1, jnp.bfloat16)

    @jax.jit
    def _vg(qa, kga, vga, kla, vla):
        def _loss(q_i, kg_i, vg_i, kl_i, vl_i):
            o = pallas_impl(
                q_i, kg_i, vg_i, kl_i, vl_i, g_q_seg=seg, g_kv_seg=seg, l_q_seg=seg, l_kv_seg=seg, sliding_window=win
            )
            return jnp.sum(o.astype(jnp.float32) * cot.astype(jnp.float32)), o

        return jax.value_and_grad(_loss, argnums=(0, 1, 2, 3, 4), has_aux=True)(qa, kga, vga, kla, vla)

    (_, out), grads = _vg(q, kg, vg, kl, vl)
    assert out.shape == q.shape
    assert np.isfinite(np.asarray(out, dtype=np.float32)).all()
    for g in grads:
        assert np.isfinite(np.asarray(g, dtype=np.float32)).all()
