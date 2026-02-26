"""Tests for RoBERTa model."""

# pyright: reportPrivateLocalImportUsage=false

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import BaseModuleTester, SequenceClassificationTester
except ImportError:
    from test_utils import (  # pyright: ignore[reportImplicitRelativeImport]
        BaseModuleTester,
        SequenceClassificationTester,
    )


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

    def test_base_module(self, roberta_config, small_model_config):
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

    def test_sequence_classification(self, roberta_config, small_model_config):
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
