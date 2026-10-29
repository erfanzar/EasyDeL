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

"""Native TPU grouped-decode regression coverage for model-sized recurrent states."""

from __future__ import annotations

from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.gated_delta_rule_grouped_decode import (
    gated_delta_rule_grouped_decode as grouped_decode_pallas,
)
from ejkernel.kernels._xla.gated_delta_rule_grouped_decode import (
    gated_delta_rule_grouped_decode as grouped_decode_xla,
)

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Requires native TPU Pallas lowering")


@pytest.mark.parametrize(
    ("batch", "num_k_heads", "dtype"),
    [
        pytest.param(8, 1, jnp.bfloat16, id="tp-local-scoped-vmem-regression"),
        pytest.param(9, 1, jnp.bfloat16, id="odd-request-count"),
        pytest.param(9, 2, jnp.bfloat16, id="multiple-key-head-groups"),
        pytest.param(9, 2, jnp.float32, id="float32-state"),
    ],
)
def test_grouped_decode_native_tpu_matches_xla(batch, num_k_heads, dtype):
    """Native Pallas grouped decode matches XLA for every request/head across two steps.

    The first case uses TP=4-local Qwen3-Next decode dimensions. The odd batch
    exercises the final request, and multiple key heads detect incorrect
    expansion-group output ordering. Calls go directly to the native Pallas
    backend (no interpretation or automatic backend fallback).
    """
    head_dim = value_dim = 128
    expand_ratio = 4
    num_v_heads = num_k_heads * expand_ratio
    rng = jax.random.key(2026)

    def normal(index, shape):
        return jax.random.normal(jax.random.fold_in(rng, index), shape, dtype=jnp.float32)

    qk_shape = (batch, num_k_heads, head_dim)
    value_shape = (batch, num_k_heads, expand_ratio, value_dim)
    gate_shape = (batch, num_k_heads, expand_ratio)
    state_shape = (batch, num_v_heads, head_dim, value_dim)
    # Model decode uses normalized q/k. Keep independently sampled requests and
    # value heads so permutations or dropped tail writes cannot pass by symmetry.
    query = normal(0, qk_shape)
    key = normal(1, qk_shape)
    query = (query / jnp.linalg.norm(query, axis=-1, keepdims=True)).astype(dtype)
    key = (key / jnp.linalg.norm(key, axis=-1, keepdims=True)).astype(dtype)
    value = normal(2, value_shape).astype(dtype)
    beta = jax.nn.sigmoid(normal(3, gate_shape)).astype(dtype)
    decay = (-jax.nn.softplus(normal(4, gate_shape))).astype(dtype)
    state = normal(5, state_shape).astype(dtype)

    native_step = jax.jit(grouped_decode_pallas)
    reference_step = jax.jit(grouped_decode_xla)
    native_state = reference_state = state
    # Request full-fp32 products only for the float32 case. Mosaic's fp32
    # contract precision requires fp32 inputs; applying it to bf16 is invalid.
    precision_context = jax.default_matmul_precision("highest") if dtype == jnp.float32 else nullcontext()
    with precision_context:
        for _ in range(2):
            native_out, native_state = jax.block_until_ready(native_step(query, key, value, beta, decay, native_state))
            reference_out, reference_state = jax.block_until_ready(
                reference_step(query, key, value, beta, decay, reference_state)
            )
            assert native_out.shape == (batch, num_v_heads, value_dim)
            assert native_state.shape == state_shape
            assert native_out.dtype == native_state.dtype == dtype
            tolerance = {"rtol": 0.02, "atol": 0.05} if dtype == jnp.bfloat16 else {"rtol": 2e-5, "atol": 2e-5}
            for actual, expected in ((native_out, reference_out), (native_state, reference_state)):
                actual = np.asarray(actual.astype(jnp.float32))
                expected = np.asarray(expected.astype(jnp.float32))
                assert np.isfinite(actual).all()
                np.testing.assert_allclose(actual, expected, **tolerance)
