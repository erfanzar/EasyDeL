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

"""Tests for Whisper model."""

import easydel as ed
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import transformers

try:
    from tests.modules.test_utils import Seq2SeqTester, setup_config
    from tests.modules.test_utils.model_factory import cleanup_models, create_ed_model, create_hf_model
except ImportError:
    from tests.modules.test_utils import Seq2SeqTester, setup_config  # pyright: ignore[reportImplicitRelativeImport]
    from tests.modules.test_utils.model_factory import (  # pyright: ignore[reportImplicitRelativeImport]
        cleanup_models,
        create_ed_model,
        create_hf_model,
    )


def _random_input_features(config, batch_size: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    shape = (batch_size, config.num_mel_bins, config.max_source_positions * 2)
    return rng.standard_normal(shape).astype(np.float32)


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

    def test_generation(self, whisper_config, small_model_config):
        """Test Whisper generation."""
        tester = Seq2SeqTester()
        result = tester.test_generation(
            module_name="whisper",
            hf_class=transformers.WhisperForConditionalGeneration,
            task=ed.TaskType.SPEECH_SEQUENCE_TO_SEQUENCE,
            config=whisper_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Whisper generation failed: {result.error_message}"

    def test_encoder_conv_frontend_matches_hf(self, whisper_config, small_model_config):
        """Conv front-end (num_mel_bins -> d_model, symmetric padding=1) and hidden-state count match HF.

        Regression: conv1 was built with ``d_model`` input channels and both
        convs used ``"SAME"`` padding, which pads the stride-2 conv2 as (0, 1)
        instead of HF's (1, 1) (a half-frame shift of every encoder frame).
        The encoder also emitted ``num_layers + 2`` hidden states.
        """
        config = setup_config(whisper_config, small_model_config)
        assert config.num_mel_bins != config.d_model, "test needs num_mel_bins != d_model"
        hf_model = create_hf_model(transformers.WhisperModel, config)

        with config.mesh:
            ed_model = create_ed_model(
                module_name="whisper",
                task=ed.TaskType.BASE_MODULE,
                config=config,
                small_model_config=small_model_config,
                hf_model=hf_model,
            )
            assert ed_model.encoder.conv1.weight.value.shape == (3, config.num_mel_bins, config.d_model)

            features = _random_input_features(config, small_model_config["batch_size"])
            with torch.no_grad():
                hf_out = hf_model.get_encoder()(torch.from_numpy(features), output_hidden_states=True)
            ed_out = ed_model.encoder(input_features=jnp.asarray(features), output_hidden_states=True)

            assert len(ed_out.hidden_states) == len(hf_out.hidden_states) == config.encoder_layers + 1
            # hidden_states[0] = gelu(conv2(gelu(conv1(x)))) + positions: no attention involved.
            np.testing.assert_allclose(
                np.asarray(ed_out.hidden_states[0]),
                hf_out.hidden_states[0].numpy(),
                atol=2e-2,
                rtol=0,
            )
            for idx, (ed_h, hf_h) in enumerate(zip(ed_out.hidden_states, hf_out.hidden_states, strict=True)):
                np.testing.assert_allclose(np.asarray(ed_h), hf_h.numpy(), atol=5e-2, rtol=0, err_msg=f"state {idx}")
            np.testing.assert_allclose(
                np.asarray(ed_out.last_hidden_state),
                hf_out.last_hidden_state.numpy(),
                atol=5e-2,
                rtol=0,
            )

        cleanup_models(hf_model)

    def test_audio_classification_weighted_layer_sum(self, whisper_config, small_model_config):
        """``use_weighted_layer_sum`` loads HF ``layer_weights`` and stacks ``encoder_outputs.hidden_states``."""
        whisper_config.use_weighted_layer_sum = True
        whisper_config.classifier_proj_size = 64
        whisper_config.num_labels = 3
        config = setup_config(whisper_config, small_model_config)
        hf_model = create_hf_model(transformers.WhisperForAudioClassification, config)
        with torch.no_grad():
            # Non-uniform weights so a missed load (uniform init) cannot pass.
            hf_model.layer_weights.copy_(torch.tensor([0.5, -1.0, 2.0][: config.encoder_layers + 1]))

        with config.mesh:
            ed_model = create_ed_model(
                module_name="whisper",
                task=ed.TaskType.AUDIO_CLASSIFICATION,
                config=config,
                small_model_config=small_model_config,
                hf_model=hf_model,
            )
            np.testing.assert_allclose(
                np.asarray(ed_model.layer_weights.value),
                hf_model.layer_weights.detach().numpy(),
                atol=1e-6,
            )

            features = _random_input_features(config, small_model_config["batch_size"])
            with torch.no_grad():
                hf_logits = hf_model(input_features=torch.from_numpy(features)).logits.numpy()
            ed_out = ed_model(input_features=jnp.asarray(features), output_hidden_states=False)
            np.testing.assert_allclose(np.asarray(ed_out.logits), hf_logits, atol=5e-2, rtol=0)

        cleanup_models(hf_model)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
