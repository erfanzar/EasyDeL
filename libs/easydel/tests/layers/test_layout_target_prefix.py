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

"""``target_prefix`` of the fused layouts/builders is honoured, and the public reform builders stay consistent."""

import pytest
from easydel.layers.layouts import (
    dense_gate_up_layout,
    dense_qkv_layout,
    gate_up_fusion_reform_param,
    qkv_fusion_reform_param,
)

pytestmark = pytest.mark.cpu_ok


def test_layout_default_target_prefix_drives_reform_keys():
    layout = dense_qkv_layout(8, 4, target_prefix="attn.qkv")
    assert set(layout.reform_param()) == {"attn.qkv.weight$"}
    assert set(layout.reform_param(include_bias=True)) == {"attn.qkv.weight$", "attn.qkv.bias$"}


def test_explicit_prefix_overrides_layout_prefix():
    layout = dense_gate_up_layout(8, target_prefix="mlp.gate_up")
    assert set(layout.reform_param("other")) == {"other.weight$"}


def test_missing_prefix_raises():
    with pytest.raises(ValueError, match="target_prefix"):
        dense_qkv_layout(8, 4).reform_param()


def test_prefix_not_part_of_default_layout_behaviour():
    layout = dense_gate_up_layout(8)
    assert layout.target_prefix is None
    assert set(layout.reform_param("gate_up_proj")) == {"gate_up_proj.weight$"}


def test_gate_up_builder_and_layout_emit_expected_sources():
    expected = {
        "gate_up_proj.weight$": ("gate_proj.weight", "up_proj.weight"),
        "gate_up_proj.bias$": ("gate_proj.bias", "up_proj.bias"),
    }
    for rules in (
        dense_gate_up_layout(8).reform_param("gate_up_proj", include_bias=True),
        gate_up_fusion_reform_param(include_bias=True),
    ):
        assert {key: tuple(rule["sources"]) for key, rule in rules.items()} == expected


def test_hf_per_expert_swiglu_builder_emits_per_expert_splits():
    from easydel.layers.layouts import hf_per_expert_swiglu_reform_param

    rules = hf_per_expert_swiglu_reform_param(num_experts=2)
    assert set(rules) == {"router.weight$", "experts.gate_up_proj$", "experts.down_proj$"}
    gate_up_names = [split["name"] for split in rules["experts.gate_up_proj$"]["splits"]]
    assert gate_up_names == [
        "experts.0.w1.weight",
        "experts.0.w3.weight",
        "experts.1.w1.weight",
        "experts.1.w3.weight",
    ]
    down_names = [split["name"] for split in rules["experts.down_proj$"]["splits"]]
    assert down_names == ["experts.0.w2.weight", "experts.1.w2.weight"]


def test_qkv_builder_and_layout_emit_expected_sources():
    expected = {
        "qkv_proj.weight$": ("q_proj.weight", "k_proj.weight", "v_proj.weight"),
        "qkv_proj.bias$": ("q_proj.bias", "k_proj.bias", "v_proj.bias"),
    }
    for rules in (
        dense_qkv_layout(8, 4).reform_param("qkv_proj", include_bias=True),
        qkv_fusion_reform_param(include_bias=True, segment_sizes=(8, 4, 4)),
    ):
        assert {key: tuple(rule["sources"]) for key, rule in rules.items()} == expected
