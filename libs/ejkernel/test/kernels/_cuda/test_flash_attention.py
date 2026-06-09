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

"""CUDA flash attention correctness tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ejkernel.errors import EjkernelRuntimeError
from ejkernel.kernels._cuda.flash_attention import flash_attention as cuda_flash_attention
from ejkernel.kernels._xla.flash_attention import flash_attention as xla_flash_attention

if jax.devices()[0].platform != "gpu":
    pytest.skip("CUDA tests require GPU backend", allow_module_level=True)

try:
    from ejkernel.kernels._cuda.flash_attention._build import build_cuda_lib

    build_cuda_lib()
except RuntimeError as exc:
    pytest.skip(f"CUDA flash_attention build failed: {exc}", allow_module_level=True)


def _assert_close(out_cuda, out_ref, rtol=2e-2, atol=2e-2):
    np.testing.assert_allclose(
        np.asarray(out_cuda, dtype=np.float32),
        np.asarray(out_ref, dtype=np.float32),
        rtol=rtol,
        atol=atol,
    )


class TestFlashAttentionCuda:
    def test_forward_matches_xla(self):
        key = jax.random.PRNGKey(0)
        kq, kk, kv = jax.random.split(key, 3)
        batch, seq_len, num_heads, head_dim = 2, 64, 4, 64

        q = jax.random.normal(kq, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(kk, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(kv, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)

        out_cuda = cuda_flash_attention(q, k, v, causal=True)
        out_xla = xla_flash_attention(q, k, v, causal=True)

        out_cuda = jax.block_until_ready(out_cuda)
        out_xla = jax.block_until_ready(out_xla)

        _assert_close(out_cuda, out_xla)

    def test_sliding_window_tuple_matches_xla(self):
        key = jax.random.PRNGKey(5)
        kq, kk, kv = jax.random.split(key, 3)
        batch, seq_len, num_heads, head_dim = 1, 32, 4, 64

        q = jax.random.normal(kq, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(kk, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(kv, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)

        window = (8, 4)
        out_cuda = cuda_flash_attention(q, k, v, causal=False, sliding_window=window)
        out_xla = xla_flash_attention(q, k, v, causal=False, sliding_window=window)

        out_cuda = jax.block_until_ready(out_cuda)
        out_xla = jax.block_until_ready(out_xla)

        _assert_close(out_cuda, out_xla, rtol=3e-2, atol=3e-2)

    def test_rejects_attention_mask(self):
        key = jax.random.PRNGKey(4)
        kq, kk, kv, km = jax.random.split(key, 4)
        batch, seq_len, num_heads, head_dim = 1, 24, 2, 64

        q = jax.random.normal(kq, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(kk, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(kv, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        mask = jax.random.bernoulli(km, p=0.7, shape=(batch, 1, seq_len, seq_len))

        with pytest.raises(EjkernelRuntimeError):
            _ = cuda_flash_attention(q, k, v, attention_mask=mask, causal=False)

    def test_rejects_softmax_aux(self):
        key = jax.random.PRNGKey(6)
        kq, kk, kv, ks = jax.random.split(key, 4)
        batch, seq_len, num_heads, head_dim = 1, 16, 2, 32

        q = jax.random.normal(kq, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(kk, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(kv, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        sinks = jax.random.normal(ks, (num_heads,), dtype=jnp.float32)

        with pytest.raises(EjkernelRuntimeError):
            _ = cuda_flash_attention(q, k, v, softmax_aux=sinks, causal=True)

    def test_rejects_segment_ids(self):
        key = jax.random.PRNGKey(2)
        kq, kk, kv = jax.random.split(key, 3)
        batch, seq_len, num_heads, head_dim = 1, 24, 2, 32

        q = jax.random.normal(kq, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(kk, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(kv, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)

        q_segment_ids = jnp.concatenate(
            [
                jnp.zeros((batch, seq_len // 2), dtype=jnp.int32),
                jnp.ones((batch, seq_len - seq_len // 2), dtype=jnp.int32),
            ],
            axis=1,
        )
        kv_segment_ids = q_segment_ids.copy()

        with pytest.raises(EjkernelRuntimeError):
            _ = cuda_flash_attention(
                q,
                k,
                v,
                causal=False,
                q_segment_ids=q_segment_ids,
                kv_segment_ids=kv_segment_ids,
            )

    def test_rejects_non_alibi_bias(self):
        key = jax.random.PRNGKey(7)
        kq, kk, kv, kb = jax.random.split(key, 4)
        batch, seq_len, num_heads, head_dim = 1, 16, 2, 32

        q = jax.random.normal(kq, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(kk, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(kv, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        bias = jax.random.normal(kb, (batch, num_heads, seq_len, seq_len), dtype=jnp.float32)

        with pytest.raises(EjkernelRuntimeError):
            _ = cuda_flash_attention(q, k, v, bias=bias, causal=True)

    def test_gradients_finite(self):
        key = jax.random.PRNGKey(3)
        kq, kk, kv = jax.random.split(key, 3)
        batch, seq_len, num_heads, head_dim = 1, 16, 2, 32

        q = jax.random.normal(kq, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(kk, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(kv, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)

        def loss_fn(q, k, v):
            out = cuda_flash_attention(q, k, v, causal=True)
            return jnp.sum(out.astype(jnp.float32) ** 2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        for grad in grads:
            assert jnp.all(jnp.isfinite(grad))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
