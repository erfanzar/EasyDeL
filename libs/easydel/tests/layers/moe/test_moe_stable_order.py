# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""The deferred MoE dispatch's pair order: ranked in one pass, equal to the two stable argsorts it replaces."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.layers.moe._moe_module import _RANKED_ORDER_MAX_PAIRS, _stable_order


def _keys(kind: str, pairs: int, local_experts: int, seed: int) -> np.ndarray:
    """Per-pair keys: a local expert id, or ``local_experts`` for a pair another shard serves."""
    rng = np.random.default_rng(seed)
    ids = np.arange(pairs)
    if kind == "routed":  # a quarter of the pairs local (four expert shards), uniform over the local experts
        keys = np.where(rng.random(pairs) < 0.25, rng.integers(0, local_experts, size=pairs), local_experts)
    elif kind == "one_expert":
        keys = np.zeros(pairs)
    elif kind == "none_local":
        keys = np.full(pairs, local_experts)
    elif kind == "all_local":
        keys = rng.integers(0, local_experts, size=pairs)
    elif kind == "sorted":
        keys = np.sort(ids % (local_experts + 1))
    elif kind == "reversed":
        keys = np.sort(ids % (local_experts + 1))[::-1]
    else:  # alternating local and non-local pairs
        keys = np.where(ids % 2 == 0, local_experts, ids % local_experts)
    return keys.astype(np.int32)


KINDS = ("routed", "one_expert", "none_local", "all_local", "sorted", "reversed", "alternating")


@pytest.mark.parametrize("pairs", [64, 256, _RANKED_ORDER_MAX_PAIRS, 2 * _RANKED_ORDER_MAX_PAIRS])
@pytest.mark.parametrize("local_experts", [16, 72])
@pytest.mark.parametrize("kind", KINDS)
def test_stable_order_equals_the_two_stable_argsorts(pairs, local_experts, kind):
    """``(order, positions)`` equal ``argsort(keys)`` and ``argsort(order)`` on both sides of the one-pass limit."""
    keys = jnp.asarray(_keys(kind, pairs, local_experts, seed=pairs + local_experts))

    order, positions = jax.jit(_stable_order)(keys)
    want_order = jnp.argsort(keys).astype(jnp.int32)
    want_positions = jnp.argsort(want_order).astype(jnp.int32)

    assert order.dtype == jnp.int32 and positions.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(order), np.asarray(want_order))
    np.testing.assert_array_equal(np.asarray(positions), np.asarray(want_positions))
    host = np.argsort(np.asarray(keys), kind="stable")
    np.testing.assert_array_equal(np.asarray(order), host)
    np.testing.assert_array_equal(np.asarray(order)[np.asarray(positions)], np.arange(pairs))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
