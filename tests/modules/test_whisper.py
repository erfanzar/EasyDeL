"""Tests for Whisper model."""

import pytest
import transformers

import easydel as ed

try:
    from .test_utils import Seq2SeqTester
except ImportError:
    from test_utils import Seq2SeqTester


class TestWhisper:
    """Test suite for Whisper model."""

    @pytest.fixture
    def whisper_config(self, small_model_config):
        """Create Whisper config."""
        return ed.WhisperConfig(
            vocab_size=small_model_config["vocab_size"],
            d_model=small_model_config["hidden_size"],
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=small_model_config["intermediate_size"],
            decoder_ffn_dim=small_model_config["intermediate_size"],
            max_source_positions=1500,
            max_target_positions=448,
            num_mel_bins=80,
        )

    def test_seq2seq(self, whisper_config, small_model_config):
        """Test WhisperForConditionalGeneration."""
        tester = Seq2SeqTester()
        result = tester.run(
            module_name="whisper",
            hf_class=transformers.WhisperForConditionalGeneration,
            task=ed.TaskType.SPEECH_SEQUENCE_TO_SEQUENCE,
            config=whisper_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Whisper failed: {result.error_message or result.comparison.details}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
