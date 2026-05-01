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

"""Tests for Qwen3 embedding model."""

import pytest

import easydel as ed
from tests.modules.mpmd._scheduler_utils import LOSS_SCHEDULE_KINDS
from tests.modules.test_utils import EmbeddingTester


class TestQwen3Embedding:
    """Test suite for Qwen3ForEmbedding."""

    @pytest.fixture
    def qwen3_config(self, small_model_config):
        """Create Qwen3-specific config for embedding."""
        return ed.Qwen3Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    @pytest.mark.parametrize("mpmd_schedule_kind", LOSS_SCHEDULE_KINDS, indirect=True)
    def test_embedding(self, qwen3_config, small_model_config, mpmd_schedule_kind):
        """Test Qwen3ForEmbedding produces correct shapes and normalized output."""
        tester = EmbeddingTester()
        result = tester.run(
            module_name="qwen3",
            task=ed.TaskType.EMBEDDING,
            config=qwen3_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Qwen3 EMBEDDING failed: {result.error_message}"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
