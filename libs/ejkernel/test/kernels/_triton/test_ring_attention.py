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
import numpy as np
import pytest

from ejkernel.kernels._triton.ring_attention._interface import ring_attention as triton_ring_attention
from ejkernel.kernels._xla.ring_attention._interface import ring_attention as xla_ring_attention
from ejkernel.ops import BwdParams, FwdParams

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def test_ring_attention_matches_xla_single_device():
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    batch, seq_len, num_heads, head_dim = 1, 64, 4, 32
    # Keep K/V heads equal to Q heads for parity with the current XLA implementation.
    num_kv_heads = num_heads

    q = jax.random.normal(kq, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
    k = jax.random.normal(kk, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
    v = jax.random.normal(kv, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)

    fwd_params = FwdParams(q_blocksize=64, kv_blocksize=64, num_warps=4, num_stages=2)
    bwd_params = BwdParams(q_blocksize=64, kv_blocksize=64, num_warps=4, num_stages=2)

    out_triton = triton_ring_attention(
        q,
        k,
        v,
        causal=True,
        sliding_window=(32, 32),
        softmax_scale=head_dim**-0.5,
        axis_name=None,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
    )
    out_xla = xla_ring_attention(
        q,
        k,
        v,
        causal=True,
        sliding_window=(32, 32),
        softmax_scale=head_dim**-0.5,
        axis_name=None,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
    )

    out_triton = jax.block_until_ready(out_triton)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
