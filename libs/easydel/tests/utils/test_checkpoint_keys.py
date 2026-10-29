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

"""Semantics of the declarative checkpoint-key rule engine (pure string transforms)."""

import pytest
from easydel.utils.checkpoint_keys import KEEP, KeyRuleSet, Map, Nested, Rewrite

pytestmark = pytest.mark.cpu_ok


def test_first_terminal_match_wins_and_none_is_unowned():
    rules = KeyRuleSet([Map(r"vision\..*", None), Map(r"(?P<rest>.*)\.w", "{rest}.weight"), Map(r".*", "late")])
    assert rules("vision.blocks.0.w") is None
    assert rules("layers.0.w") == "layers.0.weight"
    assert rules("other") == "late"


def test_rewrites_chain_before_terminal_rules():
    rules = KeyRuleSet([Rewrite(r"\.attn\.", ".self_attn."), Rewrite(r"\.ffn\.", ".mlp.")], otherwise=KEEP)
    assert rules("l.0.attn.ffn.attn.x") == "l.0.self_attn.mlp.self_attn.x"


def test_rename_tables_apply_to_named_groups_and_pass_unknown_values():
    rules = KeyRuleSet([Map(r"(?P<p>.*)\.(?P<leaf>[^.]+)", "{p}.{leaf}", rename={"leaf": {"w1": "gate_proj"}})])
    assert rules("mlp.w1") == "mlp.gate_proj"
    assert rules("mlp.w9") == "mlp.w9"


def test_otherwise_modes():
    assert KeyRuleSet([])("x") is None
    assert KeyRuleSet([], otherwise=KEEP)("x") == "x"
    fallback = KeyRuleSet([Map(r"x", "from-fallback")])
    assert KeyRuleSet([Rewrite(r"^y$", "x")], otherwise=fallback)("y") == "from-fallback"


def test_nested_reuses_inner_rules_and_rejects_foreign_results():
    inner = KeyRuleSet([Map(r"layers\.(?P<i>\d+)\.(?P<r>.*)", "model.layers.{i}.{r}"), Map(r"bad", "elsewhere")])
    nested = Nested(
        r"mtp\.(?P<s>\d+)\.(?P<r>.+)",
        inner,
        inner_key="layers.{s}.{r}",
        inner_result=r"model\.layers\.{s}\.(?P<tail>.*)",
        target="stages.{s}.{tail}",
    )
    rules = KeyRuleSet([nested])
    assert rules("mtp.2.attn.wq.weight") == "stages.2.attn.wq.weight"
    assert KeyRuleSet([Nested(r"(?P<s>bad)", inner, "{s}", r"model\..*", "never")])("bad") is None
    assert rules("unrelated") is None


def test_rejects_non_rule_entries():
    with pytest.raises(TypeError):
        KeyRuleSet([lambda key: key])
