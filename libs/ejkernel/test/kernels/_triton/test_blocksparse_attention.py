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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._triton.blocksparse_attention import blocksparse_attention as triton_blocksparse_attention
from ejkernel.kernels._xla.blocksparse_attention import blocksparse_attention as xla_blocksparse_attention
from ejkernel.ops import BwdParams, FwdParams

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def test_blocksparse_attention_matches_xla():
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    batch, num_heads, q_len, head_dim = 1, 4, 64, 32
    num_kv_heads, kv_len = 2, 64

    q = jax.random.normal(kq, (batch, num_heads, q_len, head_dim), dtype=jnp.float16)
    k = jax.random.normal(kk, (batch, num_kv_heads, kv_len, head_dim), dtype=jnp.float16)
    v = jax.random.normal(kv, (batch, num_kv_heads, kv_len, head_dim), dtype=jnp.float16)

    fwd_params = FwdParams(q_blocksize=32, kv_blocksize=32, num_warps=4, num_stages=2)
    bwd_params = BwdParams(q_blocksize=32, kv_blocksize=32, num_warps=4, num_stages=2)

    out_triton = triton_blocksparse_attention(
        q,
        k,
        v,
        causal=True,
        sliding_window=(32, 32),
        softmax_scale=head_dim**-0.5,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
    )
    out_xla = xla_blocksparse_attention(
        q,
        k,
        v,
        causal=True,
        sliding_window=(32, 32),
        softmax_scale=head_dim**-0.5,
    )

    out_triton = jax.block_until_ready(out_triton)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


def test_blocksparse_sinks_counted_once_across_kv_tiles():
    """Sink mass must enter the softmax denominator once per row, not once per KV tile (r15 S05-1).

    kv_len=256 with kv_blocksize=32 spans 8 tiles; a single tile hides the bug.
    """
    kq, kk, kv, kw = jax.random.split(jax.random.PRNGKey(1), 4)
    batch, num_heads, num_kv_heads, seq, head_dim = 1, 4, 2, 256, 32
    q = jax.random.normal(kq, (batch, num_heads, seq, head_dim), dtype=jnp.float32)
    k = jax.random.normal(kk, (batch, num_kv_heads, seq, head_dim), dtype=jnp.float32)
    v = jax.random.normal(kv, (batch, num_kv_heads, seq, head_dim), dtype=jnp.float32)
    w = jax.random.normal(kw, (batch, num_heads, seq, head_dim), dtype=jnp.float32)
    # Length 3 != num_heads: a shared sink set under every backend's layout rule. Values near the
    # row logsumexp so the sinks take a material share of the mass.
    sinks = jnp.array([1.0, 2.0, 3.0], jnp.float32)
    params = {
        "fwd_params": FwdParams(q_blocksize=32, kv_blocksize=32, num_warps=4, num_stages=2),
        "bwd_params": BwdParams(q_blocksize=32, kv_blocksize=32, num_warps=4, num_stages=2),
    }

    def triton(q, aux):
        return triton_blocksparse_attention(
            q, k, v, softmax_aux=aux, causal=True, softmax_scale=head_dim**-0.5, **params
        )

    def xla(q, aux):
        return xla_blocksparse_attention(q, k, v, softmax_aux=aux, causal=True, softmax_scale=head_dim**-0.5)

    out_triton = np.asarray(triton(q, sinks), np.float32)
    out_xla = np.asarray(xla(q, sinks), np.float32)
    out_no_sink = np.asarray(xla(q, None), np.float32)
    assert np.abs(out_xla - out_no_sink).max() > 0.1  # negative control: these sinks matter
    np.testing.assert_allclose(out_triton, out_xla, rtol=2e-2, atol=2e-2)

    dq_triton = jax.grad(lambda x: jnp.sum(triton(x, sinks) * w))(q)
    dq_xla = jax.grad(lambda x: jnp.sum(xla(x, sinks) * w))(q)
    np.testing.assert_allclose(np.asarray(dq_triton), np.asarray(dq_xla), rtol=3e-2, atol=3e-2)
