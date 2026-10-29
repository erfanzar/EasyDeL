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

"""HunYuan-VL generation metadata transitions without array computation."""

from types import SimpleNamespace

import pytest
from easydel.modules.hunyuan_vl.modeling_hunyuan_vl import HunYuanVLForConditionalGeneration


@pytest.mark.cpu_ok
def test_generation_updates_preserve_grid_and_drop_prefill_only_inputs():
    """The active generation API preserves the image grid across decode steps."""
    grid = ((1, 2, 4), (1, 4, 4))
    deltas = object()
    cache = object()
    kwargs = {
        "image_grid_thw": grid,
        "rope_deltas": deltas,
        "pixel_values": object(),
        "mm_token_type_ids": object(),
    }
    for output_deltas in (None, object()):
        outputs = SimpleNamespace(past_key_values=cache, rope_deltas=output_deltas)
        kwargs = HunYuanVLForConditionalGeneration.update_inputs_for_generation(None, outputs, kwargs)
        if output_deltas is not None:
            deltas = output_deltas
        assert kwargs == {"image_grid_thw": grid, "rope_deltas": deltas, "past_key_values": cache}
        assert kwargs["image_grid_thw"] is grid
