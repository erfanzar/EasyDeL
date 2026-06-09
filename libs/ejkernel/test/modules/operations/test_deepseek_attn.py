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

"""Operation-level tests for deepseek_attn."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel import modules
from ejkernel.modules import operations
from ejkernel.modules.operations import (
    DeepSeekAttention,
    DeepSeekAttentionConfig,
    deepseek_attn,
)


def _make_inputs(seed: int = 0):
    key = jax.random.PRNGKey(seed)
    k0, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)

    batch = 1
    seq_len = 16
    q_heads = 4
    kv_heads = 4
    d_nope = 32
    kv_lora_rank = 16
    index_heads = 4
    index_head_dim = 32

    return dict(
        query=jax.random.normal(k0, (batch, seq_len, q_heads, d_nope), dtype=jnp.float32),
        key_value=jax.random.normal(k1, (batch, seq_len, kv_lora_rank), dtype=jnp.float32),
        w_kc=jax.random.normal(k2, (kv_lora_rank, kv_heads, d_nope), dtype=jnp.float32),
        w_vc=jax.random.normal(k3, (kv_lora_rank, kv_heads, d_nope), dtype=jnp.float32),
        query_index=jax.random.normal(k4, (batch, seq_len, index_heads, index_head_dim), dtype=jnp.float32),
        key_index=jax.random.normal(k5, (batch, seq_len, index_head_dim), dtype=jnp.float32),
        index_weights=jax.random.normal(k6, (batch, seq_len, index_heads), dtype=jnp.float32),
    )


def test_operation_is_exported_from_modules_init_files():
    """Verify DeepSeekAttention exports are available from public module packages."""
    assert hasattr(operations, "deepseek_attn")
    assert hasattr(operations, "DeepSeekAttention")
    assert hasattr(operations, "DeepSeekAttentionConfig")

    assert hasattr(modules, "deepseek_attn")
    assert hasattr(modules, "DeepSeekAttention")
    assert hasattr(modules, "DeepSeekAttentionConfig")


def test_deepseek_attn_operation_functional_api_runs_xla():
    """Verify the public functional API runs on the XLA backend."""
    inputs = _make_inputs(seed=1)
    output = deepseek_attn(
        inputs["query"],
        inputs["key_value"],
        inputs["w_kc"],
        inputs["w_vc"],
        inputs["query_index"],
        inputs["key_index"],
        inputs["index_weights"],
        index_topk=8,
        causal=True,
        platform="xla",
    )

    assert output.shape == (1, 16, 4, 32)
    assert jnp.isfinite(output).all()


def test_deepseek_attn_operation_class_api_runs_xla():
    """Verify the Kernel.run API runs on the XLA backend."""
    inputs = _make_inputs(seed=2)
    op = DeepSeekAttention()
    cfg = DeepSeekAttentionConfig(index_topk=8, platform="xla")

    output = op.run(
        inputs["query"],
        inputs["key_value"],
        inputs["w_kc"],
        inputs["w_vc"],
        inputs["query_index"],
        inputs["key_index"],
        inputs["index_weights"],
        causal=True,
        cfg=cfg,
    )

    assert output.shape == (1, 16, 4, 32)
    assert jnp.isfinite(output).all()
