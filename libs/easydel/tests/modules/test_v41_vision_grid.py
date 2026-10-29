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

"""The aligner token grid (vision tower) agrees with the processor's token grid (pure Python)."""

import pytest
from easydel.modules.deepseek_v41.modeling_deepseek_v41_vision import llm_grid
from easydel.modules.deepseek_v41.processing_deepseek_v41 import llm_token_grid

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("height,width", [(14, 14), (28, 42), (70, 112), (126, 28), (210, 322)])
@pytest.mark.parametrize("ratio", [1, 2, 3])
def test_llm_grid_matches_processor_token_grid(height, width, ratio):
    patch = 14
    assert llm_grid(height // patch, width // patch, ratio) == llm_token_grid(height, width, patch, ratio)
