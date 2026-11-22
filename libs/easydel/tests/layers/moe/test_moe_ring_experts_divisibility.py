# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Regression: ring-of-experts requires n_routed_experts divisible by ep_size.

Without this guard, ``permute``'s roll-then-truncate logic silently drops
tokens routed to experts ``g >= (n_routed_experts // ep_size) * ep_size``.
The math: ``flatten_selected_experts = (selected - roll) % num_experts``
followed by ``group_sizes[:experts_per_shard]`` truncates ``num_experts %
ep_size`` experts from every shard simultaneously.

The assertion lives in ``BaseMoeModule._sparse_moe_call`` right after
``ep_size`` is computed; it must NOT fire when ``use_ring_of_experts=False``
(default) and MUST fire on the bug condition.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import spectrax as spx

AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


def _build_qwen3_moe_block(*, num_experts: int, ep_axis: int, use_ring_of_experts: bool):
    """Return an instantiated Qwen3MoeSparseBlock under the given mesh shape."""
    from easydel.modules.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseBlock
    from easydel.modules.qwen3_moe.qwen3_moe_configuration import Qwen3MoeConfig

    config = Qwen3MoeConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        num_experts=num_experts,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        sharding_axis_dims=(1, 1, 1, ep_axis, 1, 1),
        sharding_axis_names=AXIS_NAMES,
        scan_layers=False,
    )
    config.add_basic_configurations(use_ring_of_experts=use_ring_of_experts)
    return Qwen3MoeSparseBlock(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )


@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires >= 4 devices to make ep=4 nontrivial")
def test_ring_of_experts_uneven_divisibility_raises():
    """n_routed_experts=10, ep=4 -> experts 8..9 would be silently dropped without the guard."""
    block = _build_qwen3_moe_block(num_experts=10, ep_axis=4, use_ring_of_experts=True)
    x = jnp.ones((1, 4, 32), dtype=jnp.float32)

    with pytest.raises(ValueError) as excinfo:
        with block.config.mesh:
            block(x)

    msg = str(excinfo.value)
    assert "use_ring_of_experts" in msg
    assert "n_routed_experts (10)" in msg
    assert "ep_size (4)" in msg

    assert "[8..9]" in msg


@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires >= 4 devices for ep=4")
def test_ring_of_experts_assertion_dormant_when_disabled():
    """Default ``use_ring_of_experts=False`` must NOT trigger the assertion even with uneven config."""
    block = _build_qwen3_moe_block(num_experts=10, ep_axis=4, use_ring_of_experts=False)
    x = jnp.ones((1, 4, 32), dtype=jnp.float32)

    try:
        with block.config.mesh:
            block(x)
    except ValueError as ve:
        assert "use_ring_of_experts" not in str(ve), f"divisibility assertion fired with use_ring_of_experts=False: {ve}"
    except Exception:
        pass


@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires >= 4 devices for ep=4")
def test_ring_of_experts_even_divisibility_passes_assertion():
    """n_routed_experts=8, ep=4 (clean divisibility) must pass the assertion."""
    block = _build_qwen3_moe_block(num_experts=8, ep_axis=4, use_ring_of_experts=True)
    x = jnp.ones((1, 4, 32), dtype=jnp.float32)

    try:
        with block.config.mesh:
            block(x)
    except ValueError as ve:
        assert "use_ring_of_experts" not in str(ve), f"divisibility assertion fired on clean (8, 4) config: {ve}"
    except Exception:
        pass
