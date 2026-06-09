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

"""Tests for DeepSeek Sparse Attention (DSA) XLA kernel with MLA-style inputs."""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ejkernel.kernels._xla.deepseek_attn import deepseek_attn


@pytest.fixture
def rng():
    """Provide a deterministic PRNG key for reproducible tests."""
    return jax.random.PRNGKey(42)


def _make_mla_inputs(
    rng,
    batch=2,
    seq_len=128,
    q_heads=8,
    kv_heads=8,
    d_nope=64,
    kv_lora_rank=32,
    index_heads=16,
    index_head_dim=64,
    dtype=jnp.bfloat16,
):
    """Create MLA-style inputs for DSA testing.

    Args:
        rng: JAX PRNG key.
        batch: Batch size.
        seq_len: Sequence length.
        q_heads: Number of query heads.
        kv_heads: Number of KV heads.
        d_nope: Non-positional embedding dimension.
        kv_lora_rank: KV latent compression rank.
        index_heads: Number of Lightning Indexer heads.
        index_head_dim: Indexer head dimension.
        dtype: Data type for tensors.

    Returns:
        Dictionary of MLA input tensors.
    """
    keys = jax.random.split(rng, 8)
    v_head_dim = d_nope
    return dict(
        query=jax.random.normal(keys[0], (batch, seq_len, q_heads, d_nope), dtype=dtype),
        key_value=jax.random.normal(keys[1], (batch, seq_len, kv_lora_rank), dtype=dtype),
        w_kc=jax.random.normal(keys[2], (kv_lora_rank, kv_heads, d_nope), dtype=dtype),
        w_vc=jax.random.normal(keys[3], (kv_lora_rank, kv_heads, v_head_dim), dtype=dtype),
        query_index=jax.random.normal(keys[4], (batch, seq_len, index_heads, index_head_dim), dtype=dtype),
        key_index=jax.random.normal(keys[5], (batch, seq_len, index_head_dim), dtype=dtype),
        index_weights=jax.random.normal(keys[6], (batch, seq_len, index_heads), dtype=jnp.float32),
    )


class TestDeepSeekAttnShapes:
    """Validate output tensor shapes for various head configurations."""

    def test_basic_output_shape(self, rng):
        """Verify default-config output matches expected [B, T, H_Q, D_V] shape."""
        inputs = _make_mla_inputs(rng)
        output = deepseek_attn(**inputs, index_topk=32, causal=True)
        batch, seq_len, q_heads, _ = inputs["query"].shape
        v_head_dim = inputs["w_vc"].shape[2]
        assert output.shape == (batch, seq_len, q_heads, v_head_dim)

    def test_gqa_output_shape(self, rng):
        """GQA: 16 query heads, 1 KV head (MQA)."""
        inputs = _make_mla_inputs(rng, q_heads=16, kv_heads=1)
        output = deepseek_attn(**inputs, index_topk=16, causal=True)
        assert output.shape == (2, 128, 16, 64)


class TestDeepSeekAttnCausality:
    """Verify causal masking behavior."""

    def test_causal_context_matters(self, rng):
        """Verify changing past tokens affects output of later tokens."""
        inputs = _make_mla_inputs(
            rng,
            batch=1,
            seq_len=32,
            q_heads=4,
            kv_heads=4,
            d_nope=32,
            kv_lora_rank=16,
            index_heads=4,
            index_head_dim=32,
            dtype=jnp.float32,
        )
        out1 = deepseek_attn(**inputs, index_topk=32, causal=True)

        inputs2 = {**inputs}
        inputs2["key_value"] = inputs["key_value"].at[:, :16, :].set(inputs["key_value"][:, :16, :] * 100.0)
        out2 = deepseek_attn(**inputs2, index_topk=32, causal=True)

        assert not jnp.allclose(out1[:, -1], out2[:, -1], atol=1e-3)


class TestDeepSeekAttnSparsity:
    """Validate sparsity behavior and numerical stability."""

    def test_no_nan(self, rng):
        """Verify output contains no NaN or Inf values."""
        inputs = _make_mla_inputs(rng)
        output = deepseek_attn(**inputs, index_topk=32, causal=True)
        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))

    def test_topk_equals_seqlen_matches_dense_mla(self, rng):
        """When topk >= seq_len, DSA should match dense MLA."""
        batch, seq_len, q_heads, d_nope = 1, 16, 4, 32
        kv_heads, kv_lora_rank = 4, 16
        inputs = _make_mla_inputs(
            rng,
            batch=batch,
            seq_len=seq_len,
            q_heads=q_heads,
            kv_heads=kv_heads,
            d_nope=d_nope,
            kv_lora_rank=kv_lora_rank,
            index_heads=4,
            index_head_dim=32,
            dtype=jnp.float32,
        )

        out_dsa = deepseek_attn(**inputs, index_topk=seq_len, causal=True)

        q = inputs["query"]
        kv = inputs["key_value"]
        w_kc = inputs["w_kc"]
        w_vc = inputs["w_vc"]

        k_nope = jnp.einsum("btl,lhd->bthd", kv, w_kc)
        v = jnp.einsum("btl,lhd->bthd", kv, w_vc)

        G = q_heads // kv_heads
        k_exp = jnp.repeat(k_nope, G, axis=2)
        v_exp = jnp.repeat(v, G, axis=2)

        scale = d_nope**-0.5
        scores = jnp.einsum("bshd,bthd->bhst", q, k_exp) * scale
        causal_mask = jnp.triu(jnp.full((seq_len, seq_len), jnp.finfo(jnp.float32).min), k=1)
        scores = scores + causal_mask[None, None, :, :]
        weights = jax.nn.softmax(scores, axis=-1)
        out_dense = jnp.einsum("bhst,bthd->bshd", weights, v_exp)

        npt.assert_allclose(out_dsa, out_dense, atol=1e-4)


class TestDeepSeekAttnGrad:
    """Validate gradient computation through DSA."""

    def test_grad_no_nan(self, rng):
        """Verify gradients w.r.t. query and key_value contain no NaN."""
        inputs = _make_mla_inputs(
            rng,
            batch=1,
            seq_len=32,
            q_heads=4,
            kv_heads=4,
            d_nope=32,
            kv_lora_rank=16,
            index_heads=4,
            index_head_dim=32,
            dtype=jnp.float32,
        )
        q, kv = inputs["query"], inputs["key_value"]
        rest = {k: v for k, v in inputs.items() if k not in ("query", "key_value")}

        def loss_fn(q, kv):
            out = deepseek_attn(q, kv, **rest, index_topk=8, causal=True)
            return out.sum()

        grads = jax.grad(loss_fn, argnums=(0, 1))(q, kv)
        for g in grads:
            assert not jnp.any(jnp.isnan(g)), "Gradient contains NaN"

    def test_weight_grads_no_nan(self, rng):
        """Verify gradients w.r.t. projection weights contain no NaN."""
        inputs = _make_mla_inputs(
            rng,
            batch=1,
            seq_len=32,
            q_heads=4,
            kv_heads=4,
            d_nope=32,
            kv_lora_rank=16,
            index_heads=4,
            index_head_dim=32,
            dtype=jnp.float32,
        )
        q = inputs["query"]
        kv = inputs["key_value"]
        w_kc = inputs["w_kc"]
        w_vc = inputs["w_vc"]
        rest = {k: v for k, v in inputs.items() if k not in ("query", "key_value", "w_kc", "w_vc")}

        def loss_fn(q, kv, w_kc, w_vc):
            out = deepseek_attn(q, kv, w_kc, w_vc, **rest, index_topk=8, causal=True)
            return out.sum()

        grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3))(q, kv, w_kc, w_vc)
        for g in grads:
            assert not jnp.any(jnp.isnan(g)), "Gradient contains NaN"

    def test_indexer_grads_are_zero(self, rng):
        """Verify Lightning Indexer inputs receive zero gradients (stop_gradient)."""
        inputs = _make_mla_inputs(
            rng,
            batch=1,
            seq_len=32,
            q_heads=4,
            kv_heads=4,
            d_nope=32,
            kv_lora_rank=16,
            index_heads=4,
            index_head_dim=32,
            dtype=jnp.float32,
        )
        qi = inputs["query_index"]
        ki = inputs["key_index"]
        iw = inputs["index_weights"]
        rest = {k: v for k, v in inputs.items() if k not in ("query_index", "key_index", "index_weights")}

        def loss_fn(qi, ki, iw):
            out = deepseek_attn(
                **rest,
                query_index=qi,
                key_index=ki,
                index_weights=iw,
                index_topk=8,
                causal=True,
            )
            return out.sum()

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(qi, ki, iw)
        for g in grads:
            npt.assert_allclose(g, jnp.zeros_like(g), atol=0.0)


class TestModuleAPI:
    """Validate public module-level imports and availability."""

    def test_module_import(self):
        """Verify DeepSeekAttention is importable from operations module."""
        from ejkernel.modules.operations import DeepSeekAttention, DeepSeekAttentionConfig
        from ejkernel.modules.operations import deepseek_attn as dsa_fn

        assert callable(dsa_fn)
        assert DeepSeekAttention is not None
        assert DeepSeekAttentionConfig is not None

    def test_pallas_kernel_import(self):
        """Verify Pallas TPU kernel is importable."""
        from ejkernel.kernels._pallas.tpu.deepseek_attn import deepseek_attn as dsa_pallas

        assert callable(dsa_pallas)
