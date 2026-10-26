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

"""Non-JAX regressions for Qwen2-VL serving and generation metadata."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.multimodal.manager import MultiModalManager
from easydel.modules.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from easydel.modules.qwen2_vl.qwen2_vl_configuration import Qwen2VLConfig

pytestmark = pytest.mark.cpu_ok


class _Image:
    """RGB image protocol with no resize or Pillow dependency."""

    size = (4, 4)

    def convert(self, mode):
        assert mode == "RGB"
        return self

    def __array__(self, dtype=None, copy=None):
        return np.full((4, 4, 3), 128, dtype=np.uint8 if dtype is None else dtype)


def test_flat_patch_capability_drives_preprocessing_after_config_roundtrip():
    config = Qwen2VLConfig(vision_config={"patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2})
    config = Qwen2VLConfig.from_dict(config.to_dict())
    assert config.esurge_flat_patch_inputs is True
    # A new/subclassed family must not rely on the manager's model_type allow-list.
    config.model_type = "custom_qwen2_vl_variant"
    manager = MultiModalManager(
        model=SimpleNamespace(config=config), processor=None, resolution_buckets=[(4, 4)], enable_cache=False
    )

    pixels, grid = manager.process_images([_Image()])

    assert pixels.shape == (4, 3 * 2 * 2 * 2)
    assert pixels.dtype == np.float32
    assert np.all(np.isfinite(pixels))
    np.testing.assert_array_equal(grid, [[1, 2, 2]])


def test_flat_patch_capability_allows_serialized_explicit_opt_out():
    config = Qwen2VLConfig(esurge_flat_patch_inputs=False)
    config = Qwen2VLConfig.from_dict(config.to_dict())
    assert config.esurge_flat_patch_inputs is False
    manager = MultiModalManager(model=SimpleNamespace(config=config), resolution_buckets=[(4, 4)], enable_cache=False)

    with pytest.raises(ValueError, match="does not expose a compatible"):
        manager.process_images([_Image()])


def test_generation_update_preserves_grid_metadata_and_updates_cache_and_rope():
    image_grid, video_grid = object(), object()
    old_rope, new_rope, cache = object(), object(), object()
    kwargs = {
        "image_grid_thw": image_grid,
        "video_grid_thw": video_grid,
        "rope_deltas": old_rope,
        "position_ids": None,
        "pixel_values": object(),
        "pixel_values_videos": object(),
        "token_type_ids": object(),
    }
    outputs = SimpleNamespace(past_key_values=cache, rope_deltas=new_rope)

    updated = Qwen2VLForConditionalGeneration.update_inputs_for_generation(None, outputs, kwargs)

    assert updated["image_grid_thw"] is image_grid
    assert updated["video_grid_thw"] is video_grid
    assert updated["past_key_values"] is cache
    assert updated["rope_deltas"] is new_rope
    assert "pixel_values" not in updated
    assert "pixel_values_videos" not in updated
    assert "token_type_ids" not in updated
