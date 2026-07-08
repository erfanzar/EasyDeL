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

"""XLA-reference tests for compressed-window full-sequence attention.

Validates the registered XLA kernel (DeepSeek-V4 training / prefill forward)
against an independent dense float64 reference over a small shape grid, plus
gradient parity (``jax.grad`` w.r.t. query / kv / sinks) against the dense
reference — this is the training-path contract the Pallas kernel must match.
"""

import jax
import numpy as np
import pytest
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.kernels._xla.compressed_window_attention import compressed_window_attention
from jax import numpy as jnp

_NEG = float(np.finfo(np.float32).min)


def _naive(q, kv, bias, sinks, scale):
    """Dense float64 reference for sink-augmented shared-KV attention."""
    batch, heads, q_len, head_dim = q.shape
    kv_len = kv.shape[1]
    out = np.zeros((batch, heads, q_len, head_dim), np.float64)
    for b in range(batch):
        for h in range(heads):
            for s in range(q_len):
                logit = scale * (q[b, h, s].astype(np.float64) @ kv[b].astype(np.float64).T)
                logit = logit + bias[b, s].astype(np.float64)
                col = np.concatenate([logit, [sinks[h]]]) if sinks is not None else logit
                col = col - col.max()
                p = np.exp(col)
                p = p / p.sum()
                out[b, h, s] = p[:kv_len] @ kv[b].astype(np.float64)
    return out


def _inputs(batch, heads, q_len, head_dim, kv_len, use_sinks, seed=0):
    """Random forward inputs with ~30% of KV positions masked out."""
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((batch, heads, q_len, head_dim)), jnp.float32)
    kv = jnp.asarray(rng.standard_normal((batch, kv_len, head_dim)), jnp.float32)
    bias = np.asarray(rng.standard_normal((batch, q_len, kv_len)) * 0.5, np.float32)
    bias[rng.random((batch, q_len, kv_len)) < 0.3] = _NEG
    bias = jnp.asarray(bias)
    sinks = jnp.asarray(rng.standard_normal((heads,)), jnp.float32) if use_sinks else None
    return q, kv, bias, sinks


@pytest.mark.parametrize(
    ("batch", "heads", "q_len", "head_dim", "kv_len"),
    [
        (2, 4, 16, 32, 40),  # short prefill, ragged kv
        (1, 8, 32, 64, 80),  # medium prefill
        (2, 16, 8, 512, 132),  # HCA-like: real head_dim, window + few entries
        (1, 4, 17, 128, 300),  # multi-tile kv (spans several 128 tiles)
    ],
)
@pytest.mark.parametrize("use_sinks", [True, False])
def test_matches_dense_reference(batch, heads, q_len, head_dim, kv_len, use_sinks):
    """The XLA kernel matches the dense float64 reference to fp32 precision."""
    q, kv, bias, sinks = _inputs(batch, heads, q_len, head_dim, kv_len, use_sinks)
    scale = head_dim**-0.5
    got = np.asarray(compressed_window_attention(q, kv, bias, sinks, softmax_scale=scale))
    ref = _naive(np.asarray(q), np.asarray(kv), np.asarray(bias), None if sinks is None else np.asarray(sinks), scale)
    assert got.shape == (batch, heads, q_len, head_dim)
    assert np.isfinite(got).all()
    assert np.abs(got - ref).max() < 1e-4, np.abs(got - ref).max()


@pytest.mark.parametrize(
    ("batch", "heads", "q_len", "head_dim", "kv_len"),
    [(2, 4, 16, 32, 40), (2, 16, 8, 512, 132), (1, 4, 17, 128, 300)],
)
def test_gradient_matches_reference(batch, heads, q_len, head_dim, kv_len):
    """``jax.grad`` w.r.t. query/kv/sinks matches the dense reference (fp32 tight).

    The bias is a stop-gradient structural mask; gradients flow only to the
    query, the shared KV (both key and value roles), and the per-head sinks.
    """
    q, kv, bias, sinks = _inputs(batch, heads, q_len, head_dim, kv_len, use_sinks=True, seed=1)
    scale = head_dim**-0.5
    rng = np.random.default_rng(2)
    cot = jnp.asarray(rng.standard_normal((batch, heads, q_len, head_dim)), jnp.float32)

    def _ref(qq, kk, ss):
        logits = jnp.einsum("bhsd,bld->bhsl", qq, kk).astype(jnp.float32) * scale + bias[:, None, :, :]
        sk = jnp.broadcast_to(ss.reshape(1, heads, 1, 1), (*logits.shape[:3], 1))
        comb = jnp.concatenate([logits, sk], axis=-1)
        comb = comb - jnp.max(comb, axis=-1, keepdims=True)
        p = jax.nn.softmax(comb, axis=-1)[..., :kv_len]
        return jnp.sum(jnp.einsum("bhsl,bld->bhsd", p, kk) * cot)

    def _kern(qq, kk, ss):
        return jnp.sum(compressed_window_attention(qq, kk, bias, ss, softmax_scale=scale) * cot)

    gr = jax.grad(_ref, argnums=(0, 1, 2))(q, kv, sinks)
    gk = jax.grad(_kern, argnums=(0, 1, 2))(q, kv, sinks)
    for ref_g, kern_g in zip(gr, gk, strict=True):
        assert jnp.isfinite(kern_g).all()
        assert float(jnp.max(jnp.abs(ref_g - kern_g))) < 1e-4, float(jnp.max(jnp.abs(ref_g - kern_g)))


def test_registry_and_signatures():
    """Both platforms are registered under the op id and their signatures agree."""
    assert kernel_registry.get("compressed_window_attention", platform=Platform.XLA, backend=Backend.ANY) is not None
    assert kernel_registry.get("compressed_window_attention", platform=Platform.PALLAS, backend=Backend.TPU) is not None
    assert kernel_registry.validate_signatures("compressed_window_attention")


def test_fully_masked_query_row_defined():
    """A query whose only open column is the sink yields a finite (~zero) output."""
    rng = np.random.default_rng(3)
    q = jnp.asarray(rng.standard_normal((1, 2, 4, 32)), jnp.float32)
    kv = jnp.asarray(rng.standard_normal((1, 8, 32)), jnp.float32)
    bias = jnp.full((1, 4, 8), _NEG, jnp.float32)
    sinks = jnp.zeros((2,), jnp.float32)
    out = np.asarray(compressed_window_attention(q, kv, bias, sinks, softmax_scale=0.1))
    assert np.isfinite(out).all()
    assert np.abs(out).max() < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
