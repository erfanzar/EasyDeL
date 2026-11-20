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

"""``kda_replicate_gate_low_rank`` config plumbing (no array computation).

The layer behaviour (no gate-pair all-gather, bitwise-equal outputs) is covered on TPU by
``tests/modules/test_glm5_next_kda_serving_ep_split.py``.
"""

import easydel as ed
import pytest

pytestmark = pytest.mark.cpu_ok


def test_defaults_off():
    assert ed.Glm5NextTextConfig().kda_replicate_gate_low_rank is False


def test_constructor_and_add_basic_configurations_set_it():
    assert ed.Glm5NextTextConfig(kda_replicate_gate_low_rank=True).kda_replicate_gate_low_rank is True
    config = ed.Glm5NextTextConfig()
    config.add_basic_configurations(kda_replicate_gate_low_rank=True)
    assert config.kda_replicate_gate_low_rank is True


def test_survives_a_dict_round_trip():
    config = ed.Glm5NextTextConfig(kda_replicate_gate_low_rank=True)
    assert ed.Glm5NextTextConfig.from_dict(config.to_dict()).kda_replicate_gate_low_rank is True
