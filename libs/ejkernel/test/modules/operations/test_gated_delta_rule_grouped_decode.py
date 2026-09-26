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

"""Module-level tests for grouped GDR decode operation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from ejkernel.modules.operations import (
    GatedDeltaRuleGroupedDecode,
    GatedDeltaRuleGroupedDecodeConfig,
    gated_delta_rule_grouped_decode,
    gated_delta_rule_grouped_decode_op,
)


def test_gated_delta_rule_grouped_decode_function_class_and_op_match():
    """Check the three grouped-GDR-decode entry points produce identical results.

    Draws random query/key/value/beta/decay and recurrent-state tensors for a grouped layout
    (``num_v_heads == num_k_heads * expand_ratio``) and runs them through the module-level
    :func:`gated_delta_rule_grouped_decode` function, the :class:`GatedDeltaRuleGroupedDecode`
    kernel's ``run`` method, and the :data:`gated_delta_rule_grouped_decode_op` alias, all with an
    XLA-pinned :class:`GatedDeltaRuleGroupedDecodeConfig`. Asserts the function's output shape is
    ``[batch, num_v_heads, value_dim]`` and state shape is ``[batch, num_v_heads, head_dim,
    value_dim]``, then asserts the class- and op-routed outputs and states match the function's
    within :func:`jnp.allclose` tolerance.

    Raises:
        AssertionError: If any of the three paths disagree in shape or value.
    """
    batch, num_k_heads, head_dim = 1, 2, 4
    expand_ratio, value_dim = 2, 3
    num_v_heads = num_k_heads * expand_ratio

    key = jax.random.PRNGKey(0)
    query = jax.random.normal(key, (batch, num_k_heads, head_dim), dtype=jnp.float32)
    k = jax.random.normal(key, (batch, num_k_heads, head_dim), dtype=jnp.float32)
    v = jax.random.normal(key, (batch, num_k_heads, expand_ratio, value_dim), dtype=jnp.float32)
    beta = jax.random.normal(key, (batch, num_k_heads, expand_ratio), dtype=jnp.float32)
    decay = jax.random.normal(key, (batch, num_k_heads, expand_ratio), dtype=jnp.float32)
    state = jax.random.normal(key, (batch, num_v_heads, head_dim, value_dim), dtype=jnp.float32)
    cfg = GatedDeltaRuleGroupedDecodeConfig(platform="xla")

    direct_out, direct_state = gated_delta_rule_grouped_decode(query, k, v, beta, decay, state, cfg=cfg)
    class_out, class_state = GatedDeltaRuleGroupedDecode().run(query, k, v, beta, decay, state, cfg=cfg)
    op_out, op_state = gated_delta_rule_grouped_decode_op(query, k, v, beta, decay, state, cfg=cfg)

    assert direct_out.shape == (batch, num_v_heads, value_dim)
    assert direct_state.shape == (batch, num_v_heads, head_dim, value_dim)
    assert jnp.allclose(class_out, direct_out)
    assert jnp.allclose(class_state, direct_state)
    assert jnp.allclose(op_out, direct_out)
    assert jnp.allclose(op_state, direct_state)


def test_gated_delta_rule_grouped_decode_config_roundtrips():
    """Check :class:`GatedDeltaRuleGroupedDecodeConfig` survives a JSON serialization round-trip.

    Builds a config with ``platform="xla"`` and ``backend="any"``, serializes it via ``to_json``
    and reconstructs it with ``from_json``, then asserts the ``platform`` and ``backend`` fields are
    preserved.

    Raises:
        AssertionError: If either field differs after the serialize/deserialize round-trip.
    """
    cfg = GatedDeltaRuleGroupedDecodeConfig(platform="xla", backend="any")
    rt = GatedDeltaRuleGroupedDecodeConfig.from_json(cfg.to_json())
    assert rt.platform == "xla"
    assert rt.backend == "any"
