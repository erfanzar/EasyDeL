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

"""Per-expert checkpoint keys vs transformers>=5 merged expert tensors.

The resolver reads transformers' own conversion mapping, so these tests pin
its answers for model types whose checkpoints name experts differently.
"""

import torch
from easydel.utils.parameters_transformation import (
    StateDictConverter,
    canonical_expert_leaf,
    resolve_expert_merge,
)


def test_resolver_follows_transformers_renames_and_part_order():
    merged, expert, spec, part = resolve_expert_merge("model.layers.3.block_sparse_moe.experts.5.w3.weight", "mixtral")
    assert merged == "model.layers.3.mlp.experts.gate_up_proj"
    assert (expert, part, spec.parts, spec.concat_dim) == (5, 1, ("w1", "w3"), 1)
    assert canonical_expert_leaf(spec, part) == "up_proj"

    merged, _, spec, part = resolve_expert_merge("model.layers.3.block_sparse_moe.experts.5.w2.weight", "mixtral")
    assert merged == "model.layers.3.mlp.experts.down_proj"
    assert canonical_expert_leaf(spec, part) == "down_proj"

    assert resolve_expert_merge("model.layers.1.mlp.gate.weight", "qwen3_moe") is None


def test_unknown_model_type_uses_the_transformers_convention():
    merged, expert, spec, part = resolve_expert_merge("model.layers.1.mlp.experts.7.up_proj.weight", "not_a_model")
    assert (merged, expert, part, spec.target) == ("model.layers.1.mlp.experts.gate_up_proj", 7, 1, "gate_up_proj")


def _per_expert(num_experts=3, inter=4, hidden=5):
    torch.manual_seed(0)
    state = {}
    for e in range(num_experts):
        state[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = torch.randn(inter, hidden)
        state[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = torch.randn(inter, hidden)
    return state


def test_merge_builds_the_transformers_layout_for_single_source_rules():
    state = _per_expert()
    expected = torch.cat(
        [
            torch.stack([state[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] for e in range(3)]),
            torch.stack([state[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] for e in range(3)]),
        ],
        dim=1,
    )
    rules = {
        "model.layers.0.mlp.experts.gate_up_proj.weight$": {
            "sources": ("model.layers.0.mlp.experts.gate_up_proj",),
            "fuser": lambda x: x,
        }
    }
    StateDictConverter.merge_hf_expert_tensors(state, rules, "qwen3_moe")
    assert list(state) == ["model.layers.0.mlp.experts.gate_up_proj"]
    torch.testing.assert_close(state["model.layers.0.mlp.experts.gate_up_proj"], expected)


def test_unfuse_restores_per_expert_keys_for_separate_source_rules():
    per_expert = _per_expert()
    merged = dict(per_expert)
    StateDictConverter.merge_hf_expert_tensors(
        merged,
        {"r$": {"sources": ("model.layers.0.mlp.experts.gate_up_proj",), "fuser": lambda x: x}},
        "deepseek_v3",
    )
    rules = {
        "model.layers.0.mlp.experts.gate_up_proj.weight$": {
            "sources": ("model.layers.0.mlp.experts.gate_proj.weight", "model.layers.0.mlp.experts.up_proj.weight"),
            "fuser": lambda gate, up: (gate, up),
        }
    }
    StateDictConverter.unfuse_hf_expert_tensors(merged, rules, "deepseek_v3")
    assert set(merged) == set(per_expert)
    for key, value in per_expert.items():
        torch.testing.assert_close(merged[key], value)
