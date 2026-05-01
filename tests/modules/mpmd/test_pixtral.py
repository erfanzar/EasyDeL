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

"""Tests for Pixtral vision model."""

import pytest
import transformers

import easydel as ed
from tests.modules.mpmd._scheduler_utils import LOSS_SCHEDULE_KINDS
from tests.modules.test_utils import BaseModuleTester


class TestPixtral:
    """Test suite for Pixtral vision model."""

    @pytest.fixture
    def pixtral_config(self, small_model_config):
        """Create Pixtral vision config."""
        return ed.PixtralVisionConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=1024,
            patch_size=16,
        )

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_vision_model(self, pixtral_config, small_model_config, mpmd_schedule_kind):
        """Test PixtralVisionModel."""
        tester = BaseModuleTester()
        result = tester.run(
            module_name="pixtral",
            hf_class=transformers.PixtralVisionModel,
            task=ed.TaskType.BASE_VISION,
            config=pixtral_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Pixtral vision failed: {result.error_message or result.comparison.details}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
