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

"""Tests for MPT model."""

import pytest
import transformers

import easydel as ed
from tests.modules.mpmd._scheduler_utils import GENERATION_SCHEDULE_KIND, LOSS_SCHEDULE_KINDS
from tests.modules.test_utils import CausalLMTester


class TestMPT:
    """Test suite for MPT model."""

    @pytest.fixture
    def mpt_config(self, small_model_config):
        """Create MPT-specific config."""
        return ed.MptConfig(
            d_model=small_model_config["hidden_size"],
            n_heads=small_model_config["num_attention_heads"],
            n_layers=4,
            attn_config=ed.MptAttentionConfig(),
            sharding_axis_dims=(1, 1, 1, 1, -1),
        )

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_causal_lm(self, mpt_config, small_model_config, mpmd_schedule_kind):
        """Test MptForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="mpt",
            hf_class=transformers.MptForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=mpt_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"MPT CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    @pytest.mark.parametrize("mpmd_schedule_kind", [GENERATION_SCHEDULE_KIND], indirect=True)
    def test_generation(self, mpt_config, small_model_config, mpmd_schedule_kind):
        """Test MPT text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="mpt",
            hf_class=transformers.MptForCausalLM,
            config=mpt_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"MPT generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
