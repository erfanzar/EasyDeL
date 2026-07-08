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

"""Pallas-TPU parity tests for compressed-window decode attention.

Compares the Pallas TPU kernel against the XLA reference over the DeepSeek-V4
decode shapes (real head_dim 512; sliding-window ring plus HCA/CSA compressed
entries; sinks on and off; masked KV positions). Skipped when no TPU is present.
"""

import numpy as np
import pytest
from jax import numpy as jnp

_NEG = float(np.finfo(np.float32).min)


def _has_tpu():
    """Return True when a TPU backend is available."""
    try:
        import jax

        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require a TPU backend")


def _inputs(batch, heads, q_len, head_dim, kv_len, dtype, seed=0):
    """Build random decode inputs with ~30% of KV positions masked out."""
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((batch, heads, q_len, head_dim)), dtype)
    kv = jnp.asarray(rng.standard_normal((batch, kv_len, head_dim)), dtype)
    bias = np.asarray(rng.standard_normal((batch, q_len, kv_len)) * 0.5, np.float32)
    bias[rng.random((batch, q_len, kv_len)) < 0.3] = _NEG
    bias = jnp.asarray(bias)
    sinks = jnp.asarray(rng.standard_normal((heads,)), jnp.float32)
    return q, kv, bias, sinks


@pytest.mark.parametrize(
    ("batch", "heads", "q_len", "head_dim", "kv_len"),
    [
        (16, 16, 1, 512, 128),  # sliding-only layer (window 128), full decode batch
        (16, 16, 1, 512, 132),  # HCA layer: window 128 + 4 compressed entries
        (16, 16, 1, 512, 256),  # CSA layer: window 128 + 128 compressed entries
        (4, 4, 1, 512, 132),  # per-device TP shard (heads/4)
    ],
)
@pytest.mark.parametrize("use_sinks", [True, False])
def test_pallas_matches_xla(batch, heads, q_len, head_dim, kv_len, use_sinks):
    """Pallas TPU output matches the XLA reference on fp32 decode shapes."""
    from ejkernel.kernels._pallas.tpu.compressed_window_decode import compressed_window_decode as pallas_impl
    from ejkernel.kernels._xla.compressed_window_decode import compressed_window_decode as xla_impl

    q, kv, bias, sinks = _inputs(batch, heads, q_len, head_dim, kv_len, jnp.float32)
    aux = sinks if use_sinks else None
    scale = head_dim**-0.5

    out_xla = np.asarray(xla_impl(q, kv, bias, aux, softmax_scale=scale))
    out_tpu = np.asarray(pallas_impl(q, kv, bias, aux, softmax_scale=scale))

    assert out_tpu.shape == out_xla.shape
    assert np.isfinite(out_tpu).all()
    assert np.abs(out_tpu - out_xla).max() < 2e-4, np.abs(out_tpu - out_xla).max()


def test_pallas_matches_xla_bf16():
    """bf16 Pallas output matches the bf16 XLA reference within the MXU floor."""
    from ejkernel.kernels._pallas.tpu.compressed_window_decode import compressed_window_decode as pallas_impl
    from ejkernel.kernels._xla.compressed_window_decode import compressed_window_decode as xla_impl

    q, kv, bias, sinks = _inputs(16, 16, 1, 512, 256, jnp.bfloat16)
    scale = 512**-0.5
    out_xla = np.asarray(xla_impl(q, kv, bias, sinks, softmax_scale=scale).astype(jnp.float32))
    out_tpu = np.asarray(pallas_impl(q, kv, bias, sinks, softmax_scale=scale).astype(jnp.float32))
    assert np.isfinite(out_tpu).all()
    assert np.abs(out_tpu - out_xla).max() < 5e-2, np.abs(out_tpu - out_xla).max()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
