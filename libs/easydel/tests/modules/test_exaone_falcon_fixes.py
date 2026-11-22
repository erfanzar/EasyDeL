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

"""Regression tests for exaone4 and falcon_mamba."""

import easydel as ed
import pytest


@pytest.mark.cpu_ok
def test_exaone4_config_without_sliding_window_defaults_to_full_attention():
    config = ed.Exaone4Config(sliding_window=None, num_hidden_layers=4)
    assert config.layer_types == ["full_attention"] * 4


@pytest.mark.cpu_ok
def test_exaone4_config_default_pattern_still_alternates():
    config = ed.Exaone4Config(sliding_window=16, sliding_window_pattern=4, num_hidden_layers=8)
    assert config.layer_types == ["sliding_attention"] * 3 + ["full_attention"] + ["sliding_attention"] * 3 + [
        "full_attention"
    ]


@pytest.mark.cpu_ok
def test_exaone4_config_preserves_explicit_layer_types_without_window():
    layer_types = ["full_attention", "full_attention"]
    config = ed.Exaone4Config(sliding_window=None, num_hidden_layers=2, layer_types=layer_types)
    assert config.layer_types == layer_types
