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

"""CPU-only regressions for GLM's declared serving patch-input capability."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.multimodal.manager import MultiModalManager
from easydel.modules.glm4v import Glm4vConfig
from easydel.modules.glm4v_moe import Glm4vMoeConfig
from easydel.modules.glm46v import Glm46VConfig

pytestmark = pytest.mark.cpu_ok


class _Image:
    """Minimal RGB image protocol; matching bucket sizes need no Pillow resize."""

    size = (4, 4)

    def convert(self, mode):
        assert mode == "RGB"
        return self

    def __array__(self, dtype=None, copy=None):
        return np.full((4, 4, 3), 128, dtype=np.uint8 if dtype is None else dtype)


@pytest.mark.parametrize("config_class", [Glm4vConfig, Glm4vMoeConfig, Glm46VConfig])
def test_declared_flat_patch_capability_survives_config_roundtrip_and_drives_preprocessing(config_class):
    config = config_class(vision_config={"patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2})
    config = config_class.from_dict(config.to_dict())
    assert config.esurge_flat_patch_inputs is True
    # An inherited/new model type must not depend on the legacy manager allow-list.
    config.model_type = "custom_glm_vision_variant"
    manager = MultiModalManager(
        model=SimpleNamespace(config=config), processor=None, resolution_buckets=[(4, 4)], enable_cache=False
    )

    pixel_values, grid = manager.process_images([_Image()])

    assert pixel_values.shape == (4, 3 * 2 * 2 * 2)
    assert pixel_values.dtype == np.float32
    assert np.all(np.isfinite(pixel_values))
    np.testing.assert_array_equal(grid, [[1, 2, 2]])


@pytest.mark.parametrize("config_class", [Glm4vConfig, Glm4vMoeConfig, Glm46VConfig])
def test_explicit_config_opt_out_is_preserved(config_class):
    config = config_class(esurge_flat_patch_inputs=False)
    config = config_class.from_dict(config.to_dict())
    assert config.esurge_flat_patch_inputs is False
    manager = MultiModalManager(model=SimpleNamespace(config=config), resolution_buckets=[(4, 4)], enable_cache=False)
    with pytest.raises(ValueError, match="does not expose a compatible"):
        manager.process_images([_Image()])
