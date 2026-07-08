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

"""XLA-reference tests for compressed-window decode attention.

Validates the registered XLA kernel against an independent dense NumPy
reference (float64) over a small shape grid covering the DeepSeek-V4 decode
regimes: sink and no-sink paths, single- and multi-query, masked KV positions,
and the shared-KV (``K == V``) broadcast over heads.
"""

import numpy as np
import pytest
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.kernels._xla.compressed_window_decode import compressed_window_decode
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


@pytest.mark.parametrize(
    ("batch", "heads", "q_len", "head_dim", "kv_len"),
    [
        (3, 4, 1, 32, 17),  # tiny decode, ragged kv_len
        (2, 16, 1, 512, 132),  # HCA-like: real head_dim, window + few entries
        (1, 8, 1, 128, 256),  # CSA-like: window + many compressed entries
        (2, 4, 3, 64, 40),  # short prefill (q_len > 1)
    ],
)
@pytest.mark.parametrize("use_sinks", [True, False])
def test_matches_dense_reference(batch, heads, q_len, head_dim, kv_len, use_sinks):
    """The XLA kernel matches the dense float64 reference to fp32 precision."""
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal((batch, heads, q_len, head_dim)), jnp.float32)
    kv = jnp.asarray(rng.standard_normal((batch, kv_len, head_dim)), jnp.float32)
    bias = np.asarray(rng.standard_normal((batch, q_len, kv_len)) * 0.5, np.float32)
    bias[rng.random((batch, q_len, kv_len)) < 0.3] = _NEG  # mask ~30% of KV
    bias = jnp.asarray(bias)
    sinks = jnp.asarray(rng.standard_normal((heads,)), jnp.float32) if use_sinks else None
    scale = head_dim**-0.5

    got = np.asarray(compressed_window_decode(q, kv, bias, sinks, softmax_scale=scale))
    ref = _naive(np.asarray(q), np.asarray(kv), np.asarray(bias), None if sinks is None else np.asarray(sinks), scale)

    assert got.shape == (batch, heads, q_len, head_dim)
    assert np.isfinite(got).all()
    assert np.abs(got - ref).max() < 1e-4


def test_registry_and_signatures():
    """Both platforms are registered and their signatures agree."""
    assert kernel_registry.get("compressed_window_decode", platform=Platform.XLA, backend=Backend.ANY) is not None
    assert kernel_registry.get("compressed_window_decode", platform=Platform.PALLAS, backend=Backend.TPU) is not None
    assert kernel_registry.validate_signatures("compressed_window_decode")


def test_fully_masked_query_row_defined():
    """A query whose only open KV positions are sinks yields finite output."""
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.standard_normal((1, 2, 1, 32)), jnp.float32)
    kv = jnp.asarray(rng.standard_normal((1, 8, 32)), jnp.float32)
    bias = jnp.full((1, 1, 8), _NEG, jnp.float32)  # every KV masked
    sinks = jnp.zeros((2,), jnp.float32)  # sink present -> denominator is exp(0)
    out = np.asarray(compressed_window_decode(q, kv, bias, sinks, softmax_scale=0.1))
    assert np.isfinite(out).all()
    assert np.abs(out).max() < 1e-3  # all mass on the sink -> ~zero value output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
