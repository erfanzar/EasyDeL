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

"""Tests for RoBERTa model."""

import pytest
import transformers

import easydel as ed
from tests.modules.mpmd._scheduler_utils import LOSS_SCHEDULE_KINDS
from tests.modules.test_utils import BaseModuleTester, SequenceClassificationTester


class TestRoBERTa:
    """Test suite for RoBERTa model."""

    @pytest.fixture
    def roberta_config(self, small_model_config):
        """Create RoBERTa-specific config."""
        return ed.RobertaConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            # RoBERTa uses a position embedding offset (padding_idx + 1),
            # so max_position_embeddings must be >= sequence_length + 2.
            max_position_embeddings=small_model_config["max_position_embeddings"] + 2,
        )

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_base_module(self, roberta_config, small_model_config, mpmd_schedule_kind):
        """Test RobertaModel base module."""
        tester = BaseModuleTester()
        result = tester.run(
            module_name="roberta",
            hf_class=transformers.RobertaModel,
            task=ed.TaskType.BASE_MODULE,
            config=roberta_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"RoBERTa BASE_MODULE failed: {result.error_message or result.comparison.details}"

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_sequence_classification(self, roberta_config, small_model_config, mpmd_schedule_kind):
        """Test RobertaForSequenceClassification."""
        roberta_config.num_labels = 2
        tester = SequenceClassificationTester()
        result = tester.run(
            module_name="roberta",
            hf_class=transformers.RobertaForSequenceClassification,
            task=ed.TaskType.SEQUENCE_CLASSIFICATION,
            config=roberta_config,
            small_model_config=small_model_config,
        )
        assert result.success, (
            f"RoBERTa SEQUENCE_CLASSIFICATION failed: {result.error_message or result.comparison.details}"
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
