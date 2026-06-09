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
import pytest

from ejkernel.callib._cute_ffi import has_cute_ffi_support
from ejkernel.modules.operations import flash_attention

from ._utils import assert_allclose, device_platform, has_cutlass, rand_qkv

pytestmark = [
    pytest.mark.skipif(device_platform() != "gpu", reason="GPU-only CUTE validation"),
    pytest.mark.skipif(not has_cutlass(), reason="CUTLASS CuTe DSL not installed"),
    pytest.mark.skipif(not has_cute_ffi_support(), reason="CuTe primitive support unavailable"),
]


def test_flash_attention_module_cute_matches_xla():
    key = jax.random.PRNGKey(0)
    q, k, v = rand_qkv(
        key,
        batch=2,
        q_len=32,
        kv_len=32,
        q_heads=4,
        kv_heads=2,
        head_dim=32,
        dtype=jnp.float16,
    )

    out_cute = flash_attention(q, k, v, causal=True, sliding_window=(8, 0), platform="cute")
    out_xla = flash_attention(q, k, v, causal=True, sliding_window=(8, 0), platform="xla")

    out_cute = jax.block_until_ready(out_cute)
    out_xla = jax.block_until_ready(out_xla)
    assert_allclose(out_cute, out_xla, atol=7e-2, rtol=7e-2)


def test_flash_attention_module_cute_mask_bias_softcap_matches_xla():
    key = jax.random.PRNGKey(1)
    kq, kk, kv, km, kb = jax.random.split(key, 5)

    batch, q_len, k_len, q_heads, kv_heads, dim = 2, 24, 24, 4, 2, 32
    q = jax.random.normal(kq, (batch, q_len, q_heads, dim), dtype=jnp.float16)
    k = jax.random.normal(kk, (batch, k_len, kv_heads, dim), dtype=jnp.float16)
    v = jax.random.normal(kv, (batch, k_len, kv_heads, dim), dtype=jnp.float16)
    attention_mask = (jax.random.uniform(km, (batch, 1, q_len, k_len), dtype=jnp.float32) > 0.1).astype(jnp.int32)
    diag = jnp.arange(min(q_len, k_len))
    attention_mask = attention_mask.at[:, :, diag, diag].set(1)
    bias = jax.random.normal(kb, (batch, q_heads, q_len, k_len), dtype=jnp.float32).astype(jnp.float16) * 0.2

    out_cute = flash_attention(
        q,
        k,
        v,
        bias,
        mask_info=None,
        softmax_scale=None,
        causal=True,
        sliding_window=(12, 0),
        logits_soft_cap=8.0,
        platform="cute",
        cfg=None,
    )
    out_xla = flash_attention(
        q,
        k,
        v,
        bias,
        mask_info=None,
        softmax_scale=None,
        causal=True,
        sliding_window=(12, 0),
        logits_soft_cap=8.0,
        platform="xla",
        cfg=None,
    )

    out_cute = jax.block_until_ready(out_cute)
    out_xla = jax.block_until_ready(out_xla)
    assert_allclose(out_cute, out_xla, atol=7e-2, rtol=7e-2)


def test_flash_attention_module_cute_jit_runs():
    key = jax.random.PRNGKey(2)
    q, k, v = rand_qkv(
        key,
        batch=1,
        q_len=16,
        kv_len=16,
        q_heads=4,
        kv_heads=2,
        head_dim=32,
        dtype=jnp.float16,
    )

    @jax.jit
    def run(q_in, k_in, v_in):
        return flash_attention(q_in, k_in, v_in, causal=True, platform="cute")

    out_jit = run(q, k, v)
    out_ref = flash_attention(q, k, v, causal=True, platform="xla")
    out_jit = jax.block_until_ready(out_jit)
    out_ref = jax.block_until_ready(out_ref)
    assert_allclose(out_jit, out_ref, atol=7e-2, rtol=7e-2)


def test_flash_attention_module_cute_paged_kv_runs():
    key = jax.random.PRNGKey(3)
    kq, kk, kv = jax.random.split(key, 3)

    batch, q_len = 2, 16
    num_q_heads, num_kv_heads, head_dim = 4, 2, 32
    block_size, max_blocks = 8, 3
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
    block_tables = (
        jnp.arange(batch, dtype=jnp.int32)[:, None] * jnp.int32(max_blocks)
        + jnp.arange(max_blocks, dtype=jnp.int32)[None, :]
    )

    out = flash_attention(
        q,
        key_cache,
        value_cache,
        None,
        None,
        None,
        None,
        block_tables,
        causal=True,
        platform="cute",
    )
    out = jax.block_until_ready(out)
    assert out.shape == q.shape
    assert jnp.all(jnp.isfinite(out))


def test_flash_attention_module_auto_paged_kv_matches_explicit_cute():
    key = jax.random.PRNGKey(4)
    kq, kk, kv = jax.random.split(key, 3)

    batch, q_len = 2, 12
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
    block_tables = (
        jnp.arange(batch, dtype=jnp.int32)[:, None] * jnp.int32(max_blocks)
        + jnp.arange(max_blocks, dtype=jnp.int32)[None, :]
    )

    out_cute = flash_attention(
        q,
        key_cache,
        value_cache,
        None,
        None,
        None,
        None,
        block_tables,
        causal=True,
        platform="cute",
    )
    out_auto = flash_attention(
        q,
        key_cache,
        value_cache,
        None,
        None,
        None,
        None,
        block_tables,
        causal=True,
        platform="auto",
    )

    out_cute = jax.block_until_ready(out_cute)
    out_auto = jax.block_until_ready(out_auto)
    assert_allclose(out_auto, out_cute, atol=7e-2, rtol=7e-2)


def test_flash_attention_module_cute_supports_varlen_and_softmax_aux():
    key = jax.random.PRNGKey(5)
    kq, kk, kv, ks = jax.random.split(key, 4)
    batch, q_len, k_len = 2, 20, 20

    q = jax.random.normal(kq, (batch, q_len, 4, 32), dtype=jnp.float16)
    k = jax.random.normal(kk, (batch, k_len, 2, 32), dtype=jnp.float16)
    v = jax.random.normal(kv, (batch, k_len, 2, 32), dtype=jnp.float16)
    softmax_aux = jax.random.normal(ks, (3,), dtype=jnp.float32) * 0.2

    q_lens = jnp.asarray([20, 11], dtype=jnp.int32)
    k_lens = jnp.asarray([17, 9], dtype=jnp.int32)
    cum_q = jnp.concatenate([jnp.asarray([0], dtype=jnp.int32), jnp.cumsum(q_lens)])
    cum_k = jnp.concatenate([jnp.asarray([0], dtype=jnp.int32), jnp.cumsum(k_lens)])

    out_cute = flash_attention(
        q,
        k,
        v,
        None,
        cum_q,
        cum_k,
        softmax_aux,
        None,
        causal=True,
        platform="cute",
    )

    # Module-level varlen path is validated by kernel-level parity tests.
    out_cute = jax.block_until_ready(out_cute)
    assert out_cute.shape == q.shape
    assert jnp.all(jnp.isfinite(out_cute))
