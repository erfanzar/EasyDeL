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

"""Pins the precedence of a nested ``text_config`` over flat kwargs for GLM-5-Next (no array compute)."""

import pytest

pytestmark = pytest.mark.cpu_ok


def test_nested_text_config_wins_over_flat_named_argument():
    """Named parameters take the nested value: the checkpoint is authoritative and schedules follow it."""
    import easydel as ed

    cfg = ed.Glm5NextTextConfig(text_config={"num_hidden_layers": 8}, num_hidden_layers=12)
    assert cfg.num_hidden_layers == 8
    assert len(cfg.layer_types) == 8
    assert len(cfg.mlp_layer_types) == 8


def test_flat_named_argument_used_without_nested_value():
    """Without a nested value, the flat argument drives every derived schedule."""
    import easydel as ed

    cfg = ed.Glm5NextTextConfig(text_config={"hidden_size": 64}, num_hidden_layers=12)
    assert cfg.hidden_size == 64
    assert cfg.num_hidden_layers == 12
    assert len(cfg.layer_types) == 12
