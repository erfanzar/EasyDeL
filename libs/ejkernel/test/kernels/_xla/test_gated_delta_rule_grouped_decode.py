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

"""XLA kernel tests for grouped GDR single-step decode."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.kernels._registry import kernel_registry
from ejkernel.kernels._xla.gated_delta_rule_grouped_decode import (
    gated_delta_rule_grouped_decode,
)


def test_gated_delta_rule_grouped_decode_kernel_is_registered_and_signature_compatible():
    """Assert the ``gated_delta_rule_grouped_decode`` XLA kernel is registered and signature-consistent.

    Resolves the operation from the global :data:`kernel_registry` for ``platform="xla"`` and
    ``backend="any"``, asserting a non-``None`` implementation, then asserts that
    :meth:`kernel_registry.validate_signatures` confirms the registered backends share a compatible
    signature.

    Raises:
        AssertionError: If no XLA implementation is registered, or if signature validation fails.
    """
    impl = kernel_registry.get("gated_delta_rule_grouped_decode", platform="xla", backend="any")
    assert impl is not None
    assert kernel_registry.validate_signatures("gated_delta_rule_grouped_decode")


def test_gated_delta_rule_grouped_decode_xla_shape_and_finite():
    """Check output shapes and finiteness of one XLA grouped GDR decode step.

    Constructs constant query/key/value/beta/decay tensors and a recurrent state for a grouped
    layout where ``num_v_heads == num_k_heads * expand_ratio`` (using a small negative decay), runs
    :func:`gated_delta_rule_grouped_decode`, and asserts the output is
    ``[batch, num_v_heads, value_dim]`` and the new state is
    ``[batch, num_v_heads, head_dim, value_dim]``, and that both contain only finite values.

    Raises:
        AssertionError: If either returned array has the wrong shape or contains non-finite
            (NaN/Inf) entries.
    """
    batch, num_k_heads, head_dim = 1, 2, 4
    expand_ratio, value_dim = 3, 5
    num_v_heads = num_k_heads * expand_ratio

    query = jnp.ones((batch, num_k_heads, head_dim), dtype=jnp.float32)
    key = jnp.ones((batch, num_k_heads, head_dim), dtype=jnp.float32) * 0.5
    value = jnp.ones((batch, num_k_heads, expand_ratio, value_dim), dtype=jnp.float32)
    beta = jnp.ones((batch, num_k_heads, expand_ratio), dtype=jnp.float32) * 0.1
    decay = jnp.ones((batch, num_k_heads, expand_ratio), dtype=jnp.float32) * -0.01
    recurrent_state = jnp.ones((batch, num_v_heads, head_dim, value_dim), dtype=jnp.float32)

    output, new_state = gated_delta_rule_grouped_decode(query, key, value, beta, decay, recurrent_state)

    assert output.shape == (batch, num_v_heads, value_dim)
    assert new_state.shape == (batch, num_v_heads, head_dim, value_dim)
    assert jnp.all(jnp.isfinite(output))
    assert jnp.all(jnp.isfinite(new_state))


def test_gated_delta_rule_grouped_decode_xla_no_decay_matches_zero_decay():
    """Check that ``decay=None`` is numerically equivalent to a zero decay tensor.

    Draws random query/key/value/beta and recurrent-state tensors, then runs
    :func:`gated_delta_rule_grouped_decode` twice: once with ``decay=None`` (state carried
    unchanged) and once with an explicit ``jnp.zeros_like(beta)`` decay (state multiplied by
    ``exp(0) == 1``). Asserts both the outputs and the updated states agree within ``atol=1e-5``,
    confirming the no-decay branch matches the identity-gate computation.

    Raises:
        AssertionError: If the ``None``-decay and zero-decay outputs or states differ beyond the
            tolerance.
    """
    batch, num_k_heads, head_dim = 1, 2, 4
    expand_ratio, value_dim = 2, 3
    num_v_heads = num_k_heads * expand_ratio

    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (batch, num_k_heads, head_dim), dtype=jnp.float32)
    k = jax.random.normal(key, (batch, num_k_heads, head_dim), dtype=jnp.float32)
    v = jax.random.normal(key, (batch, num_k_heads, expand_ratio, value_dim), dtype=jnp.float32)
    b = jax.random.normal(key, (batch, num_k_heads, expand_ratio), dtype=jnp.float32)
    s = jax.random.normal(key, (batch, num_v_heads, head_dim, value_dim), dtype=jnp.float32)

    out_none, state_none = gated_delta_rule_grouped_decode(q, k, v, b, None, s)
    out_zero, state_zero = gated_delta_rule_grouped_decode(q, k, v, b, jnp.zeros_like(b), s)

    assert jnp.allclose(out_none, out_zero, atol=1e-5)
    assert jnp.allclose(state_none, state_zero, atol=1e-5)
