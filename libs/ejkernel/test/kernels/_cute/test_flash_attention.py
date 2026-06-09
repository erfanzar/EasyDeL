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

import importlib
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ejkernel.errors import EjkernelRuntimeError

_has_cutlass = importlib.util.find_spec("cutlass") is not None
if not _has_cutlass:
    pytest.skip("CUTLASS CuTe DSL not installed", allow_module_level=True)
if jax.devices()[0].platform != "gpu":
    pytest.skip("CUTE tests require GPU backend", allow_module_level=True)

_cuda_flash_module = importlib.import_module("ejkernel.kernels._cuda.flash_attention")
cuda_flash_attention = _cuda_flash_module.flash_attention
_cute_flash_module = importlib.import_module("ejkernel.kernels._cute.flash_attention")
cute_flash_attention = _cute_flash_module.flash_attention
_xla_flash_module = importlib.import_module("ejkernel.kernels._xla.flash_attention")
xla_flash_attention = _xla_flash_module.flash_attention

_HAS_CUDA_FLASH = True
try:
    from ejkernel.kernels._cuda.flash_attention._build import build_cuda_lib

    build_cuda_lib()
except Exception:
    _HAS_CUDA_FLASH = False


@pytest.mark.parametrize(
    "causal,sliding_window",
    [
        (False, None),
        (True, None),
        (True, (8, 0)),
    ],
)
def test_flash_attention_cute_matches_xla(causal: bool, sliding_window: tuple[int, int] | None):
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    q = jax.random.normal(kq, (2, 32, 4, 32), dtype=jnp.float16)
    k = jax.random.normal(kk, (2, 32, 2, 32), dtype=jnp.float16)
    v = jax.random.normal(kv, (2, 32, 2, 32), dtype=jnp.float16)

    out_cute = cute_flash_attention(q, k, v, causal=causal, sliding_window=sliding_window)
    out_xla = xla_flash_attention(q, k, v, causal=causal, sliding_window=sliding_window)

    out_cute = jax.block_until_ready(out_cute)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_cute, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=5e-2,
        atol=5e-2,
    )


def test_flash_attention_cute_mask_bias_softcap_matches_xla():
    key = jax.random.PRNGKey(11)
    kq, kk, kv, km, kb = jax.random.split(key, 5)

    batch, q_len, k_len, q_heads, kv_heads, dim = 2, 24, 24, 4, 2, 32
    q = jax.random.normal(kq, (batch, q_len, q_heads, dim), dtype=jnp.float16)
    k = jax.random.normal(kk, (batch, k_len, kv_heads, dim), dtype=jnp.float16)
    v = jax.random.normal(kv, (batch, k_len, kv_heads, dim), dtype=jnp.float16)

    attention_mask = (jax.random.uniform(km, (batch, 1, q_len, k_len), dtype=jnp.float32) > 0.15).astype(jnp.int32)
    diag = jnp.arange(min(q_len, k_len))
    attention_mask = attention_mask.at[:, :, diag, diag].set(1)
    bias = jax.random.normal(kb, (batch, q_heads, q_len, k_len), dtype=jnp.float32).astype(jnp.float16) * 0.25

    out_cute = cute_flash_attention(
        q,
        k,
        v,
        attention_mask=attention_mask,
        bias=bias,
        causal=True,
        sliding_window=(10, 0),
        logits_soft_cap=8.0,
    )
    out_xla = xla_flash_attention(
        q,
        k,
        v,
        attention_mask=attention_mask,
        bias=bias,
        causal=True,
        sliding_window=(10, 0),
        logits_soft_cap=8.0,
    )

    out_cute = jax.block_until_ready(out_cute)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_cute, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=7e-2,
        atol=7e-2,
    )


def test_flash_attention_cute_gradients_are_finite():
    key = jax.random.PRNGKey(42)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (1, 16, 4, 32), dtype=jnp.float16)
    k = jax.random.normal(kk, (1, 16, 2, 32), dtype=jnp.float16)
    v = jax.random.normal(kv, (1, 16, 2, 32), dtype=jnp.float16)

    def loss_fn(q_in, k_in, v_in):
        out = cute_flash_attention(q_in, k_in, v_in, causal=True, sliding_window=(8, 0))
        return jnp.sum(out.astype(jnp.float32) ** 2)

    grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
    for grad in grads:
        assert jnp.all(jnp.isfinite(grad))


def test_flash_attention_cute_softmax_aux_1d_matches_xla():
    key = jax.random.PRNGKey(314)
    kq, kk, kv, ks = jax.random.split(key, 4)

    q = jax.random.normal(kq, (2, 24, 4, 32), dtype=jnp.float16)
    k = jax.random.normal(kk, (2, 24, 2, 32), dtype=jnp.float16)
    v = jax.random.normal(kv, (2, 24, 2, 32), dtype=jnp.float16)
    sinks = jax.random.normal(ks, (3,), dtype=jnp.float32) * 0.2

    out_cute = cute_flash_attention(q, k, v, causal=True, softmax_aux=sinks)
    out_xla = xla_flash_attention(q, k, v, causal=True, softmax_aux=sinks)

    out_cute = jax.block_until_ready(out_cute)
    out_xla = jax.block_until_ready(out_xla)
    np.testing.assert_allclose(
        np.asarray(out_cute, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=7e-2,
        atol=7e-2,
    )


def test_flash_attention_cute_varlen_matches_equivalent_mask():
    key = jax.random.PRNGKey(2718)
    kq, kk, kv = jax.random.split(key, 3)
    batch, q_len, k_len = 2, 24, 24

    q = jax.random.normal(kq, (batch, q_len, 4, 32), dtype=jnp.float16)
    k = jax.random.normal(kk, (batch, k_len, 2, 32), dtype=jnp.float16)
    v = jax.random.normal(kv, (batch, k_len, 2, 32), dtype=jnp.float16)

    q_lens = jnp.asarray([20, 9], dtype=jnp.int32)
    k_lens = jnp.asarray([24, 13], dtype=jnp.int32)
    cum_q = jnp.concatenate([jnp.asarray([0], dtype=jnp.int32), jnp.cumsum(q_lens)])
    cum_k = jnp.concatenate([jnp.asarray([0], dtype=jnp.int32), jnp.cumsum(k_lens)])

    q_idx = jnp.arange(q_len, dtype=jnp.int32)[None, :]
    k_idx = jnp.arange(k_len, dtype=jnp.int32)[None, :]
    shift = (k_lens - q_lens)[:, None, None]
    shifted_causal = k_idx[:, None, :] <= (q_idx[:, :, None] + shift)
    var_mask = ((q_idx < q_lens[:, None])[:, :, None] & (k_idx < k_lens[:, None])[:, None, :] & shifted_causal)[
        :, None, :, :
    ]

    out_cute = cute_flash_attention(
        q,
        k,
        v,
        attention_mask=None,
        causal=True,
        cum_seqlens_q=cum_q,
        cum_seqlens_k=cum_k,
    )
    out_cute_mask = cute_flash_attention(
        q,
        k,
        v,
        attention_mask=var_mask,
        causal=False,
    )

    out_cute = jax.block_until_ready(out_cute)
    out_cute_mask = jax.block_until_ready(out_cute_mask)
    np.testing.assert_allclose(
        np.asarray(out_cute, dtype=np.float32),
        np.asarray(out_cute_mask, dtype=np.float32),
        rtol=5e-2,
        atol=5e-2,
    )


def test_flash_attention_cute_softmax_aux_varlen_gradients_are_finite():
    key = jax.random.PRNGKey(1618)
    kq, kk, kv, ks = jax.random.split(key, 4)

    q = jax.random.normal(kq, (1, 12, 4, 32), dtype=jnp.float16)
    k = jax.random.normal(kk, (1, 12, 2, 32), dtype=jnp.float16)
    v = jax.random.normal(kv, (1, 12, 2, 32), dtype=jnp.float16)
    sinks = jax.random.normal(ks, (2,), dtype=jnp.float32) * 0.3
    cum_q = jnp.asarray([0, 9], dtype=jnp.int32)
    cum_k = jnp.asarray([0, 7], dtype=jnp.int32)

    def loss_fn(q_in, k_in, v_in):
        out = cute_flash_attention(
            q_in,
            k_in,
            v_in,
            causal=True,
            softmax_aux=sinks,
            cum_seqlens_q=cum_q,
            cum_seqlens_k=cum_k,
        )
        return jnp.sum(out.astype(jnp.float32) ** 2)

    grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
    for grad in grads:
        assert jnp.all(jnp.isfinite(grad))


def test_flash_attention_cute_dropout_is_explicitly_rejected():
    key = jax.random.PRNGKey(99)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (1, 16, 4, 32), dtype=jnp.float16)
    k = jax.random.normal(kk, (1, 16, 4, 32), dtype=jnp.float16)
    v = jax.random.normal(kv, (1, 16, 4, 32), dtype=jnp.float16)

    with pytest.raises(EjkernelRuntimeError, match=r"dropout_seed is not supported|dropout_prob must be 0\.0"):
        _ = cute_flash_attention(q, k, v, dropout_prob=0.1, dropout_seed=7, causal=True)


@pytest.mark.skipif(not _HAS_CUDA_FLASH, reason="CUDA flash attention build unavailable")
def test_flash_attention_cute_paged_kv_matches_cuda():
    key = jax.random.PRNGKey(123)
    kq, kk, kv = jax.random.split(key, 3)

    batch = 1
    num_q_heads, num_kv_heads, head_dim = 4, 2, 32
    block_size, max_blocks = 8, 3
    seq_k = block_size * max_blocks
    q_len = seq_k

    q = jax.random.normal(kq, (batch, q_len, num_q_heads, head_dim), dtype=jnp.float16)
    dense_k = jax.random.normal(kk, (batch, seq_k, num_kv_heads, head_dim), dtype=jnp.float16)
    dense_v = jax.random.normal(kv, (batch, seq_k, num_kv_heads, head_dim), dtype=jnp.float16)

    key_cache = dense_k.reshape(batch, max_blocks, block_size, num_kv_heads, head_dim).reshape(
        batch * max_blocks, block_size, num_kv_heads, head_dim
    )
    value_cache = dense_v.reshape(batch, max_blocks, block_size, num_kv_heads, head_dim).reshape(
        batch * max_blocks, block_size, num_kv_heads, head_dim
    )
    block_tables = jnp.arange(max_blocks, dtype=jnp.int32)[None, :]

    out_cute = cute_flash_attention(
        q,
        key_cache,
        value_cache,
        causal=True,
        block_tables=block_tables,
    )
    out_cuda = cuda_flash_attention(
        q,
        key_cache,
        value_cache,
        causal=True,
        block_tables=block_tables,
    )

    out_cute = jax.block_until_ready(out_cute)
    out_cuda = jax.block_until_ready(out_cuda)

    np.testing.assert_allclose(
        np.asarray(out_cute, dtype=np.float32),
        np.asarray(out_cuda, dtype=np.float32),
        rtol=9e-2,
        atol=9e-2,
    )


@pytest.mark.skipif(not _HAS_CUDA_FLASH, reason="CUDA flash attention build unavailable")
def test_flash_attention_cute_paged_kv_backward_is_explicitly_unsupported():
    key = jax.random.PRNGKey(77)
    kq, kk, kv = jax.random.split(key, 3)

    batch, q_len = 1, 8
    num_q_heads, num_kv_heads, head_dim = 4, 2, 32
    block_size, max_blocks = 8, 2
    seq_k = block_size * max_blocks

    q = jax.random.normal(kq, (batch, q_len, num_q_heads, head_dim), dtype=jnp.float16)
    dense_k = jax.random.normal(kk, (batch, seq_k, num_kv_heads, head_dim), dtype=jnp.float16)
    dense_v = jax.random.normal(kv, (batch, seq_k, num_kv_heads, head_dim), dtype=jnp.float16)

    key_cache = dense_k.reshape(batch, max_blocks, block_size, num_kv_heads, head_dim).reshape(
        batch * max_blocks, block_size, num_kv_heads, head_dim
    )
    value_cache = dense_v.reshape(batch, max_blocks, block_size, num_kv_heads, head_dim).reshape(
        batch * max_blocks, block_size, num_kv_heads, head_dim
    )
    block_tables = jnp.arange(max_blocks, dtype=jnp.int32)[None, :]

    def loss_fn(q_in):
        out = cute_flash_attention(
            q_in,
            key_cache,
            value_cache,
            causal=True,
            block_tables=block_tables,
        )
        return jnp.sum(out.astype(jnp.float32))

    with pytest.raises(EjkernelRuntimeError, match="paged_kv backward is not supported"):
        _ = jax.grad(loss_fn)(q)
