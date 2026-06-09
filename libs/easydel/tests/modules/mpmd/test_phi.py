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

"""Tests for Phi model."""

import pytest
import transformers

import easydel as ed
from tests.modules.mpmd._scheduler_utils import GENERATION_SCHEDULE_KIND, LOSS_SCHEDULE_KINDS
from tests.modules.test_utils import CausalLMTester


class TestPhi:
    """Test suite for Phi model."""

    @pytest.fixture
    def phi_config(self, small_model_config):
        """Create Phi-specific config."""
        return ed.PhiConfig(
            vocab_size=51200,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=None,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attention_dropout=0.0,
            hidden_act="gelu_new",
            max_position_embeddings=small_model_config["max_position_embeddings"],
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            partial_rotary_factor=0.5,
            qk_layernorm=False,
            bos_token_id=1,
            eos_token_id=2,
        )

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_causal_lm(self, phi_config, small_model_config, mpmd_schedule_kind):
        """Test PhiForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="phi",
            hf_class=transformers.PhiForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=phi_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Phi CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    @pytest.mark.parametrize("mpmd_schedule_kind", [GENERATION_SCHEDULE_KIND], indirect=True)
    def test_generation(self, phi_config, small_model_config, mpmd_schedule_kind):
        """Test Phi text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="phi",
            hf_class=transformers.PhiForCausalLM,
            config=phi_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Phi generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
