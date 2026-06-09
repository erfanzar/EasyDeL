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

"""Tests for TPU Pallas block-sparse attention implementation."""

import random

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._pallas.tpu.blocksparse_attention import blocksparse_attention
from ejkernel.kernels._xla.blocksparse_attention import blocksparse_attention as blocksparse_attention_xla
from ejkernel.types.mask import MaskInfo


def _has_tpu() -> bool:
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")


def test_full_attention_shape_and_finite():
    key = jax.random.PRNGKey(0)
    key, kq, kk, kv = jax.random.split(key, 4)

    batch, num_heads, seq_len, head_dim = 1, 4, 128, 64
    q = jax.random.normal(kq, (batch, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (batch, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (batch, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

    out = blocksparse_attention(q, k, v, causal=False)

    assert out.shape == q.shape
    assert jnp.isfinite(out).all()


def test_segment_ids_parity_vs_xla():
    key = jax.random.PRNGKey(1)
    key, kq, kk, kv = jax.random.split(key, 4)

    batch, num_heads, seq_len, head_dim = 1, 4, 128, 64
    q = jax.random.normal(kq, (batch, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (batch, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (batch, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

    seg = jnp.array([[0] * (seq_len // 2) + [1] * (seq_len // 2)], dtype=jnp.int32)

    out_tpu = blocksparse_attention(q, k, v, q_segment_ids=seg, kv_segment_ids=seg, causal=False)
    out_xla = blocksparse_attention_xla(q, k, v, q_segment_ids=seg, kv_segment_ids=seg, causal=False)

    assert out_tpu.shape == out_xla.shape
    assert jnp.allclose(out_tpu, out_xla, rtol=0.0, atol=0.2)


def test_causal_sliding_window_parity_vs_xla():
    key = jax.random.PRNGKey(2)
    key, kq, kk, kv = jax.random.split(key, 4)

    batch, num_heads, seq_len, head_dim = 1, 4, 128, 64
    q = jax.random.normal(kq, (batch, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (batch, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (batch, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

    out_tpu = blocksparse_attention(q, k, v, causal=True, sliding_window=(32, 0))
    out_xla = blocksparse_attention_xla(q, k, v, causal=True, sliding_window=(32, 0))

    assert out_tpu.shape == out_xla.shape
    assert jnp.allclose(out_tpu, out_xla, rtol=0.0, atol=0.2)


def test_softmax_aux_parity_vs_xla():
    key = jax.random.PRNGKey(3)
    key, kq, kk, kv, ks = jax.random.split(key, 5)

    batch, num_heads, seq_len, head_dim = 4, 4, 1024, 128
    q = jax.random.normal(kq, (batch, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (batch, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (batch, num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

    num_to_attach = 4
    q_segment_ids = jnp.zeros((batch, seq_len), "i4") - 1

    for i in range(batch):
        mperv = seq_len // 4
        for atc in range(num_to_attach):
            min_go = random.randint((seq_len // 3) // num_to_attach, (seq_len // 2) // num_to_attach)
            q_segment_ids = q_segment_ids.at[i, mperv : mperv + min_go].set(atc)
            mperv += min_go

    info = MaskInfo.from_segments(q_segment_ids=q_segment_ids)

    softmax_aux = jax.random.normal(ks, (4,), dtype=jnp.float32)
    q_segment_ids = info.q_segment_ids
    if q_segment_ids.ndim == 3:
        q_segment_ids = q_segment_ids[:, 0, :]
    q_mask = q_segment_ids >= 0
    out_tpu = blocksparse_attention(
        q,
        k,
        v,
        causal=True,
        softmax_aux=softmax_aux,
        q_segment_ids=info.q_segment_ids,
        kv_segment_ids=info.kv_segment_ids,
    ) * q_mask[:, None, :, None].astype(jnp.bfloat16)
    out_xla = blocksparse_attention_xla(
        q,
        k,
        v,
        causal=True,
        softmax_aux=softmax_aux,
        q_segment_ids=info.q_segment_ids,
        kv_segment_ids=info.kv_segment_ids,
    ) * q_mask[:, None, :, None].astype(jnp.bfloat16)

    assert out_tpu.shape == out_xla.shape
    assert jnp.allclose(out_tpu, out_xla, rtol=0.0, atol=0.2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
