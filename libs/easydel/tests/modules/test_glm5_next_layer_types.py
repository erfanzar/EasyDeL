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

"""GLM-5-Next layer schedule predicates and the cache schedule they drive."""

import pytest

pytestmark = pytest.mark.cpu_ok


def test_cache_schedule_follows_kda_dsa_predicates():
    from easydel.modules.glm5_next.glm5_next_configuration import Glm5NextTextConfig

    config = Glm5NextTextConfig(num_hidden_layers=8)
    schedule = config.get_layer_types()

    assert len(schedule) == config.num_hidden_layers
    for idx, cache_type in enumerate(schedule):
        assert config.is_kda_layer(idx) != config.is_dsa_layer(idx)
        assert (cache_type == "full_attention") == config.is_dsa_layer(idx)
        assert (cache_type == "kda_linear_attention") == config.is_kda_layer(idx)
    # Default 3:1 KDA:DSA schedule.
    assert [config.is_dsa_layer(i) for i in range(8)] == [False, False, False, True] * 2
