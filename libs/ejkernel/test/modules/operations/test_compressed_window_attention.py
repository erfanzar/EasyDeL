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

"""Public-module tests for compressed_window_attention.

Validates the ``ejkernel.modules.compressed_window_attention`` public wrapper
(executor + registry dispatch) against a dense NumPy reference, gradient flow
through the executor path, the config round-trip, the shard_map structure, and
signature parity across registered backends.
"""

import jax
import numpy as np
import pytest
from ejkernel.kernels._registry import kernel_registry
from ejkernel.modules import (
    CompressedWindowAttention,
    CompressedWindowAttentionConfig,
    compressed_window_attention,
)
from jax import numpy as jnp

_NEG = float(np.finfo(np.float32).min)


def _naive(q, kv, bias, sinks, scale):
    """Dense float64 reference."""
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


@pytest.mark.parametrize("use_sinks", [True, False])
def test_public_wrapper_matches_reference(use_sinks):
    """The public wrapper (executor dispatch -> XLA) matches the dense reference."""
    batch, heads, q_len, head_dim, kv_len = 2, 8, 24, 64, 80
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal((batch, heads, q_len, head_dim)), jnp.float32)
    kv = jnp.asarray(rng.standard_normal((batch, kv_len, head_dim)), jnp.float32)
    bias_np = np.asarray(rng.standard_normal((batch, q_len, kv_len)) * 0.5, np.float32)
    bias_np[rng.random((batch, q_len, kv_len)) < 0.3] = _NEG
    bias = jnp.asarray(bias_np)
    sinks = jnp.asarray(rng.standard_normal((heads,)), jnp.float32) if use_sinks else None
    scale = head_dim**-0.5

    out = np.asarray(compressed_window_attention(q, kv, bias, sinks, softmax_scale=scale, platform="xla"))
    ref = _naive(np.asarray(q), np.asarray(kv), bias_np, None if sinks is None else np.asarray(sinks), scale)
    assert out.shape == (batch, heads, q_len, head_dim)
    assert np.abs(out - ref).max() < 1e-4


def test_public_wrapper_is_differentiable():
    """``jax.grad`` flows through the executor -> XLA path to q/kv/sinks."""
    batch, heads, q_len, head_dim, kv_len = 1, 4, 16, 64, 48
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.standard_normal((batch, heads, q_len, head_dim)), jnp.float32)
    kv = jnp.asarray(rng.standard_normal((batch, kv_len, head_dim)), jnp.float32)
    bias = jnp.asarray(rng.standard_normal((batch, q_len, kv_len)) * 0.5, jnp.float32)
    sinks = jnp.asarray(rng.standard_normal((heads,)), jnp.float32)
    scale = head_dim**-0.5

    def _loss(qq, kk, ss):
        return jnp.sum(compressed_window_attention(qq, kk, bias, ss, softmax_scale=scale, platform="xla") ** 2)

    dq, dkv, dsink = jax.grad(_loss, argnums=(0, 1, 2))(q, kv, sinks)
    assert jnp.isfinite(dq).all() and jnp.isfinite(dkv).all() and jnp.isfinite(dsink).all()
    assert float(jnp.max(jnp.abs(dq))) > 0.0


def test_config_roundtrip_and_signatures():
    """Config JSON round-trips and both backends share a validated signature."""
    cfg = CompressedWindowAttentionConfig(platform="xla", backend="any")
    assert CompressedWindowAttentionConfig.from_json(cfg.to_json()).platform == "xla"
    assert CompressedWindowAttention().op_id == "compressed_window_attention"
    assert kernel_registry.validate_signatures("compressed_window_attention")


def test_shard_map_wrapper_defined_on_class():
    """The shard_map path is defined directly on the operation class."""
    assert "create_shard_map_wrapper" in CompressedWindowAttention.__dict__


def test_both_platforms_registered():
    """Both (Platform, Backend) pairs resolve via direct registry lookup."""
    from ejkernel.kernels._registry import Backend, Platform

    assert kernel_registry.get("compressed_window_attention", platform=Platform.XLA, backend=Backend.ANY) is not None
    assert kernel_registry.get("compressed_window_attention", platform=Platform.PALLAS, backend=Backend.TPU) is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
