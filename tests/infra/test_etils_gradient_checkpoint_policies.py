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

import jax

from easydel.infra.etils import GRADIENT_CHECKPOINT_TARGETS, EasyDeLGradientCheckPointers
from easydel.infra.utils import get_gradient_checkpoint_policy


def test_mlp_notsaveable_uses_non_mlp_targets(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_save_only_these_names(*names):
        captured["names"] = names
        return sentinel

    monkeypatch.setattr(jax.checkpoint_policies, "save_only_these_names", fake_save_only_these_names)

    policy = get_gradient_checkpoint_policy(EasyDeLGradientCheckPointers.MLP_NOTSAVEABLE)

    assert policy is sentinel
    assert set(captured["names"]) == {name for name in GRADIENT_CHECKPOINT_TARGETS if not name.startswith("mlp_")}


def test_attn_notsaveable_uses_non_attn_targets(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_save_only_these_names(*names):
        captured["names"] = names
        return sentinel

    monkeypatch.setattr(jax.checkpoint_policies, "save_only_these_names", fake_save_only_these_names)

    policy = get_gradient_checkpoint_policy(EasyDeLGradientCheckPointers.ATTN_NOTSAVEABLE)

    assert policy is sentinel
    assert set(captured["names"]) == {name for name in GRADIENT_CHECKPOINT_TARGETS if not name.startswith("attn_")}


def test_mlp_attn_notsaveable_uses_non_mlp_non_attn_targets(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_save_only_these_names(*names):
        captured["names"] = names
        return sentinel

    monkeypatch.setattr(jax.checkpoint_policies, "save_only_these_names", fake_save_only_these_names)

    policy = get_gradient_checkpoint_policy(EasyDeLGradientCheckPointers.MLP_ATTN_NOTSAVEABLE)

    assert policy is sentinel
    assert set(captured["names"]) == {
        name for name in GRADIENT_CHECKPOINT_TARGETS if not name.startswith("mlp_") and not name.startswith("attn_")
    }
