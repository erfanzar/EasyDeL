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

"""TPU Pallas registration tests for ragged GDN v2."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from ejkernel.kernels._pallas.tpu.ragged_gated_delta_rule_v2 import (
    ragged_gated_delta_rule_v2 as pallas_gdn_v2,
)
from ejkernel.kernels._registry import kernel_registry
from ejkernel.kernels._xla.ragged_gated_delta_rule_v2 import ragged_gated_delta_rule_v2 as xla_gdn_v2


def test_ragged_gated_delta_rule_v2_pallas_is_registered():
    """Assert the Pallas TPU GDN v2 kernel is registered with a valid signature.

    Looks up ``"ragged_gated_delta_rule_v2"`` in the kernel registry for
    ``platform="pallas", backend="tpu"`` and asserts the returned impl is the
    imported ``pallas_gdn_v2`` callable, then asserts
    ``kernel_registry.validate_signatures`` accepts all registered impls of that
    operation (i.e. the Pallas and reference signatures agree). Runs without a
    TPU since it only inspects registry metadata.
    """
    impl = kernel_registry.get("ragged_gated_delta_rule_v2", platform="pallas", backend="tpu")
    assert impl is pallas_gdn_v2
    assert kernel_registry.validate_signatures("ragged_gated_delta_rule_v2")


@pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="TPU-only Pallas execution")
def test_ragged_gated_delta_rule_v2_pallas_matches_xla_decode_smoke():
    """Smoke-test that the Pallas TPU GDN v2 decode matches the XLA reference.

    Builds a tiny decode-only batch of 2 single-token requests with one KQ and
    one V head (``d_k = d_v = 128``): a fused ``mixed_qkv`` projection of shape
    ``[tokens, 2 * key_dim + value_dim]``, zero gate inputs ``a``/``b``, zero
    ``A_log``/``dt_bias``, a per-token CSR offset array, slot ``state_indices``,
    and a ``distribution`` descriptor. It runs both ``xla_gdn_v2`` and
    ``pallas_gdn_v2`` from fresh zero recurrent states and asserts the decode
    outputs and updated states agree within ``atol=1e-2``. TPU-only via the
    function-level ``skipif``.
    """
    tokens, n_kq, n_v, d_k, d_v = 2, 1, 1, 128, 128
    key_dim = n_kq * d_k
    value_dim = n_v * d_v
    mixed_qkv = (
        jax.random.normal(
            jax.random.PRNGKey(0),
            (tokens, 2 * key_dim + value_dim),
            dtype=jnp.float32,
        )
        * 0.05
    )
    b = jnp.zeros((tokens, n_v), dtype=jnp.float32)
    a = jnp.zeros((tokens, n_v), dtype=jnp.float32)
    A_log = jnp.zeros((n_v,), dtype=jnp.float32)
    dt_bias = jnp.zeros((n_v,), dtype=jnp.float32)
    query_start_loc = jnp.array([0, 1, 2], dtype=jnp.int32)
    state_indices = jnp.array([0, 1], dtype=jnp.int32)
    distribution = jnp.array([2, 2, 2], dtype=jnp.int32)

    def fresh_state():
        """Allocate a zero-initialized recurrent GDN state for one kernel call.

        Returns:
            A zero ``float32`` array of shape ``[tokens, n_v, d_k, d_v]`` giving
            each call (XLA and Pallas) its own independent starting state so the
            two runs are compared from identical initial conditions.
        """
        return jnp.zeros((tokens, n_v, d_k, d_v), dtype=jnp.float32)

    state_xla, out_xla = xla_gdn_v2(
        mixed_qkv,
        b,
        a,
        fresh_state(),
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=2,
    )
    state_pallas, out_pallas = pallas_gdn_v2(
        mixed_qkv,
        b,
        a,
        fresh_state(),
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=2,
    )

    assert jnp.allclose(out_pallas, out_xla, atol=1e-2)
    assert jnp.allclose(state_pallas, state_xla, atol=1e-2)
