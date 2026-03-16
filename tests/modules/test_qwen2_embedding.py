"""Tests for Qwen2 embedding model."""

import pytest

import easydel as ed

try:
    from .test_utils import EmbeddingTester
except ImportError:
    from test_utils import EmbeddingTester  # pyright: ignore[reportImplicitRelativeImport]


class TestQwen2Embedding:
    """Test suite for Qwen2ForEmbedding."""

    @pytest.fixture
    def qwen2_config(self, small_model_config):
        """Create Qwen2-specific config for embedding."""
        return ed.Qwen2Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
        )

    def test_embedding(self, qwen2_config, small_model_config):
        """Test Qwen2ForEmbedding produces correct shapes and normalized output."""
        tester = EmbeddingTester()
        result = tester.run(
            module_name="qwen2",
            task=ed.TaskType.EMBEDDING,
            config=qwen2_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Qwen2 EMBEDDING failed: {result.error_message}"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
