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
import pytest

from easydel.infra.etils import GRADIENT_CHECKPOINT_TARGETS, EasyDeLGradientCheckPointers
from easydel.infra.utils import get_gradient_checkpoint_policy


@pytest.mark.parametrize(
    ("policy_name", "excluded_prefixes"),
    [
        (EasyDeLGradientCheckPointers.MLP_NOTSAVEABLE, ("mlp_",)),
        (EasyDeLGradientCheckPointers.ATTN_NOTSAVEABLE, ("attn_",)),
        (EasyDeLGradientCheckPointers.MLP_ATTN_NOTSAVEABLE, ("mlp_", "attn_")),
        (EasyDeLGradientCheckPointers.LMHEAD_NOTSAVEABLE, ("lm_head_",)),
        (EasyDeLGradientCheckPointers.ATTN_LMHEAD_NOTSAVEABLE, ("attn_", "lm_head_")),
        (EasyDeLGradientCheckPointers.MLP_LMHEAD_NOTSAVEABLE, ("mlp_", "lm_head_")),
    ],
)
def test_notsaveable_policies_exclude_expected_target_families(monkeypatch, policy_name, excluded_prefixes):
    captured = {}
    sentinel = object()

    def fake_save_only_these_names(*names):
        captured["names"] = names
        return sentinel

    monkeypatch.setattr(jax.checkpoint_policies, "save_only_these_names", fake_save_only_these_names)

    policy = get_gradient_checkpoint_policy(policy_name)

    assert policy is sentinel
    assert set(captured["names"]) == {
        name for name in GRADIENT_CHECKPOINT_TARGETS if not name.startswith(excluded_prefixes)
    }
    assert any(name.startswith(prefix) for prefix in excluded_prefixes for name in GRADIENT_CHECKPOINT_TARGETS)
    assert not any(name.startswith(excluded_prefixes) for name in captured["names"])
