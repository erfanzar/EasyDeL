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

"""moe_fsdp_shard_expert_weights must be honored under ep-bound fsdp.

At ep=1/tp=1 the default expert weight spec P('ep', ...) is fully replicated
per device; the knob is the only way to shard the expert dimension over fsdp
at rest, so it must not be masked by ``fsdp_is_ep_bound`` (which is True by
default). Regression for the 30B-MoE-cannot-train-on-4-chips failure: with
the mask, runs with the knob on and off compiled byte-identical programs.

Construction-level spec assertions only: the ep-bound sparse forward uses
``ragged_all_to_all``, which XLA:CPU does not implement.
"""

import spectrax as spx
from easydel.modules.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from jax import numpy as jnp
from jax.sharding import PartitionSpec


def _build(knob: bool) -> Qwen3MoeForCausalLM:
    cfg = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        sharding_axis_dims=(1, 1, 4, 1, 1, 1),
        moe_fsdp_shard_expert_weights=knob,
    )
    assert cfg.fsdp_is_ep_bound, "test targets the ep-bound default"
    return Qwen3MoeForCausalLM(config=cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))


def _expert_specs(model):
    experts = model.model.layers[0].mlp.experts
    gate_up = jnp.asarray(experts.gate_up_proj.weight)
    down = jnp.asarray(experts.down_proj.weight)
    return gate_up.sharding.spec, down.sharding.spec


def test_knob_shards_expert_dim_over_fsdp_under_ep_bound():
    gate_up_spec, down_spec = _expert_specs(_build(True))
    assert gate_up_spec[0] == ("ep", "fsdp"), gate_up_spec
    assert down_spec[0] == ("ep", "fsdp"), down_spec


def test_default_keeps_ep_only_expert_spec():
    gate_up_spec, down_spec = _expert_specs(_build(False))
    assert gate_up_spec == PartitionSpec("ep", None, "tp"), gate_up_spec
    assert down_spec[0] == "ep", down_spec
