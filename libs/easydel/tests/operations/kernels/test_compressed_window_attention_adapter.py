# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Tests for the CompressedWindowAttn EasyDeL operation adapter.

Validates registry wiring, the public-ejkernel-only import boundary, that the
adapter reproduces an independent dense reference for the DeepSeek-V4
sink-augmented shared-KV full-sequence attention, and that it is differentiable
(the training-path contract).
"""

import os
import pathlib
import re

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_NEG = float(np.finfo(np.float32).min)


def _dense_reference(q, kv, bias, sinks, scale):
    """Independent dense float64 reference (sink-augmented shared-KV attention)."""
    batch, heads, q_len, head_dim = q.shape
    kv_len = kv.shape[1]
    out = np.zeros((batch, heads, q_len, head_dim), np.float64)
    for b in range(batch):
        for h in range(heads):
            for s in range(q_len):
                logit = scale * (q[b, h, s].astype(np.float64) @ kv[b].astype(np.float64).T) + bias[b, s].astype(
                    np.float64
                )
                col = np.concatenate([logit, [sinks[h]]]) if sinks is not None else logit
                col = col - col.max()
                p = np.exp(col)
                p = p / p.sum()
                out[b, h, s] = p[:kv_len] @ kv[b].astype(np.float64)
    return out


def test_registered_in_operation_registry():
    """The adapter is registered under its impl name."""
    from easydel.operations import OperationRegistry
    from easydel.operations.kernels import CompressedWindowAttn

    assert CompressedWindowAttn.get_impl_name() == "compressed_window_attention"
    assert OperationRegistry.get("compressed_window_attention") is CompressedWindowAttn


def test_no_private_ejkernel_imports():
    """The adapter imports ejkernel only through public module surfaces."""
    from easydel.operations.kernels import compressed_window_attention as adapter_mod

    src = pathlib.Path(adapter_mod.__file__).read_text()
    forbidden = r"ejkernel\.kernels\.|_pallas|_xla|_triton|pallas_call|jax\.experimental\.pallas|\bpl\."
    hits = [ln for ln in src.splitlines() if re.search(forbidden, ln)]
    assert not hits, f"forbidden private-kernel references: {hits}"
    assert "from ejkernel.modules import compressed_window_attention" in src


def _metadata():
    """Build an OperationMetadata from a tiny DeepSeek-V4 config."""
    import easydel as ed
    from easydel.operations._operation_meta import OperationMetadata

    cfg = ed.DeepseekV4Config(
        vocab_size=64,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        layer_types=["sliding_attention", "heavily_compressed_attention"],
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
        mlp_layer_types=["moe", "moe"],
        sliding_window=8,
        o_groups=2,
        o_lora_rank=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=128,
        partial_rotary_factor=0.25,
    )
    cfg.attn_dtype = jnp.float32
    return OperationMetadata.from_config(cfg)


@pytest.mark.parametrize("use_sinks", [True, False])
def test_forward_matches_dense_reference(use_sinks):
    """forward_native reproduces the dense reference in fp32 (full sequence)."""
    from easydel.operations.kernels import CompressedWindowAttn

    op = CompressedWindowAttn(_metadata())
    batch, heads, q_len, head_dim, kv_len = 2, 4, 16, 32, 40
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal((batch, heads, q_len, head_dim)), jnp.float32)
    kv = jnp.asarray(rng.standard_normal((batch, kv_len, head_dim)), jnp.float32)
    bias_np = np.asarray(rng.standard_normal((batch, q_len, kv_len)) * 0.5, np.float32)
    bias_np[rng.random((batch, q_len, kv_len)) < 0.3] = _NEG
    bias = jnp.asarray(bias_np)
    sinks = jnp.asarray(rng.standard_normal((heads,)), jnp.float32) if use_sinks else None
    scale = head_dim**-0.5

    out = np.asarray(op(q, kv, bias, sinks, softmax_scale=scale).attention_outputs)
    ref = _dense_reference(np.asarray(q), np.asarray(kv), bias_np, None if sinks is None else np.asarray(sinks), scale)
    assert out.shape == (batch, heads, q_len, head_dim)
    assert np.isfinite(out).all()
    assert np.abs(out - ref).max() < 1e-4


def test_adapter_is_differentiable():
    """The adapter forward is differentiable w.r.t. query / kv / sinks."""
    from easydel.operations.kernels import CompressedWindowAttn

    op = CompressedWindowAttn(_metadata())
    batch, heads, q_len, head_dim, kv_len = 1, 4, 12, 32, 24
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.standard_normal((batch, heads, q_len, head_dim)), jnp.float32)
    kv = jnp.asarray(rng.standard_normal((batch, kv_len, head_dim)), jnp.float32)
    bias = jnp.asarray(rng.standard_normal((batch, q_len, kv_len)) * 0.5, jnp.float32)
    sinks = jnp.asarray(rng.standard_normal((heads,)), jnp.float32)
    scale = head_dim**-0.5

    def _loss(qq, kk, ss):
        return jnp.sum(op(qq, kk, bias, ss, softmax_scale=scale).attention_outputs ** 2)

    dq, dkv, dsink = jax.grad(_loss, argnums=(0, 1, 2))(q, kv, sinks)
    assert jnp.isfinite(dq).all() and jnp.isfinite(dkv).all() and jnp.isfinite(dsink).all()
    assert float(jnp.max(jnp.abs(dq))) > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
