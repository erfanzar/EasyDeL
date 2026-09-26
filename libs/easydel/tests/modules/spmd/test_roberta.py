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

import easydel as ed
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import transformers

try:
    from tests.modules.test_utils import BaseModuleTester, SequenceClassificationTester, setup_config
    from tests.modules.test_utils.model_factory import cleanup_models, create_ed_model, create_hf_model
except ImportError:
    from tests.modules.test_utils import (  # pyright: ignore[reportImplicitRelativeImport]
        BaseModuleTester,
        SequenceClassificationTester,
        setup_config,
    )
    from tests.modules.test_utils.model_factory import (  # pyright: ignore[reportImplicitRelativeImport]
        cleanup_models,
        create_ed_model,
        create_hf_model,
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

    def test_position_ids_follow_padding_tokens(self, roberta_config, small_model_config):
        """Default position ids follow HF ``create_position_ids_from_input_ids`` (``input_ids != padding_idx``).

        Regression: positions were derived from ``attention_mask``, so pad tokens
        present in ``input_ids`` without a matching mask shifted every later position.
        """
        config = setup_config(roberta_config, small_model_config)
        hf_model = create_hf_model(transformers.RobertaModel, config)
        pad = config.pad_token_id

        rng = np.random.default_rng(0)
        batch_size, seq_len = small_model_config["batch_size"], 32
        input_ids_np = rng.integers(pad + 1, config.vocab_size, size=(batch_size, seq_len), dtype=np.int64)
        input_ids_np[0, 5:8] = pad  # interior pad tokens, no attention_mask supplied
        input_ids_np[1, -4:] = pad

        with config.mesh:
            ed_model = create_ed_model(
                module_name="roberta",
                task=ed.TaskType.BASE_MODULE,
                config=config,
                small_model_config=small_model_config,
                hf_model=hf_model,
            )
            with torch.no_grad():
                hf_hidden = hf_model(input_ids=torch.from_numpy(input_ids_np)).last_hidden_state.numpy()
            ed_hidden = np.asarray(ed_model(input_ids=jnp.asarray(input_ids_np, dtype="i4")).last_hidden_state)
            np.testing.assert_allclose(ed_hidden, hf_hidden, atol=5e-2, rtol=0)

        cleanup_models(hf_model)

    def test_decoder_cross_attention_is_bidirectional(self, roberta_config, small_model_config):
        """Cross-attention is non-causal and its output feeds the feed-forward block (HF parity).

        Regression: cross-attention was built with ``causal=True`` and its output
        was discarded (the FFN consumed the self-attention output instead).
        """
        roberta_config.is_decoder = True
        roberta_config.add_cross_attention = True
        config = setup_config(roberta_config, small_model_config)
        hf_model = create_hf_model(transformers.RobertaModel, config)

        rng = np.random.default_rng(0)
        batch_size, seq_len, enc_len = small_model_config["batch_size"], 16, 24
        input_ids_np = rng.integers(config.pad_token_id + 1, config.vocab_size, size=(batch_size, seq_len))
        encoder_hidden_np = rng.standard_normal((batch_size, enc_len, config.hidden_size)).astype(np.float32)
        encoder_mask_np = np.ones((batch_size, enc_len), dtype=np.int64)
        encoder_mask_np[1, 18:] = 0

        with config.mesh:
            ed_model = create_ed_model(
                module_name="roberta",
                task=ed.TaskType.BASE_MODULE,
                config=config,
                small_model_config=small_model_config,
                hf_model=hf_model,
            )
            with torch.no_grad():
                hf_hidden = hf_model(
                    input_ids=torch.from_numpy(input_ids_np),
                    encoder_hidden_states=torch.from_numpy(encoder_hidden_np),
                    encoder_attention_mask=torch.from_numpy(encoder_mask_np),
                    use_cache=False,
                ).last_hidden_state.numpy()
            ed_hidden = np.asarray(
                ed_model(
                    input_ids=jnp.asarray(input_ids_np, dtype="i4"),
                    encoder_hidden_states=jnp.asarray(encoder_hidden_np),
                    encoder_attention_mask=jnp.asarray(encoder_mask_np, dtype=jnp.bool_),
                ).last_hidden_state
            )
            np.testing.assert_allclose(ed_hidden, hf_hidden, atol=5e-2, rtol=0)

        cleanup_models(hf_model)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
