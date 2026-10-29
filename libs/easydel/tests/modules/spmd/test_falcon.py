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

"""Tests for Falcon model."""

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import transformers

try:
    from tests.modules.test_utils import CausalLMTester
except ImportError:
    from tests.modules.test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestFalcon:
    """Test suite for Falcon model."""

    @pytest.fixture
    def falcon_config(self, small_model_config):
        """Create Falcon-specific config."""
        return ed.FalconConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=small_model_config["num_attention_heads"],
            num_kv_heads=small_model_config["num_key_value_heads"],
            ffn_hidden_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            new_decoder_architecture=True,
            num_ln_in_parallel_attn=2,
            parallel_attn=True,
        )

    def test_causal_lm(self, falcon_config, small_model_config):
        """Test FalconForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="falcon",
            hf_class=transformers.FalconForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=falcon_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Falcon CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, falcon_config, small_model_config):
        """Test Falcon text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="falcon",
            hf_class=transformers.FalconForCausalLM,
            config=falcon_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Falcon generation failed: {result.error_message}"

    def test_alibi_cached_decode_matches_full_forward(self):
        """Decode must use each cached key's ALiBi offset, not only the query mask."""
        config = ed.FalconConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_kv_heads=2,
            ffn_hidden_size=64,
            max_position_embeddings=64,
            new_decoder_architecture=True,
            num_ln_in_parallel_attn=2,
            parallel_attn=True,
            alibi=True,
            attn_dtype=jnp.float32,
            kvdtype=jnp.float32,
            attn_mechanism="vanilla",
            sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        )
        model = ed.AutoEasyDeLModelForCausalLM.from_config(
            config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST
        )
        ids = jnp.asarray([[3, 5, 7, 9, 11, 13], [4, 6, 8, 10, 12, 14]], dtype=jnp.int32)
        full = model(input_ids=ids, output_attentions=True)
        cache = model.init_cache(batch_size=2, max_length=12)
        prefill = model(input_ids=ids[:, :-1], past_key_values=cache)
        decode = model(
            input_ids=ids[:, -1:],
            position_ids=jnp.full((2, 1), ids.shape[1] - 1, dtype=jnp.int32),
            past_key_values=prefill.past_key_values,
            output_attentions=True,
        )
        np.testing.assert_allclose(
            np.asarray(decode.attentions[0][..., : ids.shape[1]]),
            np.asarray(full.attentions[0][:, :, -1:, :]),
            rtol=2e-4,
            atol=2e-4,
        )
        np.testing.assert_allclose(
            np.asarray(decode.logits[:, -1, :]), np.asarray(full.logits[:, -1, :]), rtol=2e-3, atol=2e-3
        )

    def test_alibi_decode_explicit_window_mask_matches_full_forward(self):
        """An explicit full-window decode mask (HF ``generate`` style) keeps ALiBi positions exact."""
        config = ed.FalconConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_kv_heads=2,
            ffn_hidden_size=64,
            max_position_embeddings=64,
            new_decoder_architecture=True,
            num_ln_in_parallel_attn=2,
            parallel_attn=True,
            alibi=True,
            attn_dtype=jnp.float32,
            kvdtype=jnp.float32,
            attn_mechanism="vanilla",
            sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        )
        model = ed.AutoEasyDeLModelForCausalLM.from_config(
            config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST
        )
        ids = jnp.asarray([[3, 5, 7, 9, 11, 13], [4, 6, 8, 10, 12, 14]], dtype=jnp.int32)
        full = model(input_ids=ids)
        cache = model.init_cache(batch_size=2, max_length=12)
        prefill = model(input_ids=ids[:, :-1], past_key_values=cache)
        # Prompt plus the token being decoded, padded to the cache width (as HF extends it).
        window = jnp.zeros((2, 12), dtype=jnp.int32).at[:, : ids.shape[1]].set(1)
        decode = model(
            input_ids=ids[:, -1:],
            attention_mask=window,
            position_ids=jnp.full((2, 1), ids.shape[1] - 1, dtype=jnp.int32),
            past_key_values=prefill.past_key_values,
        )
        np.testing.assert_allclose(
            np.asarray(decode.logits[:, -1, :]), np.asarray(full.logits[:, -1, :]), rtol=2e-3, atol=2e-3
        )

    def test_alibi_multi_token_cached_continuation_matches_full_forward(self):
        """A multi-token continuation over a populated dense cache keeps absolute ALiBi positions (r15 S10-1)."""
        config = ed.FalconConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_kv_heads=2,
            ffn_hidden_size=64,
            max_position_embeddings=64,
            new_decoder_architecture=True,
            num_ln_in_parallel_attn=2,
            parallel_attn=True,
            alibi=True,
            attn_dtype=jnp.float32,
            kvdtype=jnp.float32,
            attn_mechanism="vanilla",
            sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        )
        model = ed.AutoEasyDeLModelForCausalLM.from_config(
            config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST
        )
        ids = jnp.asarray([[3, 5, 7, 9, 11, 13, 15], [4, 6, 8, 10, 12, 14, 16]], dtype=jnp.int32)
        split = 4
        full = model(input_ids=ids)
        cache = model.init_cache(batch_size=2, max_length=12)
        prefill = model(input_ids=ids[:, :split], past_key_values=cache)
        cont_len = ids.shape[1] - split
        cont = model(
            input_ids=ids[:, split:],
            position_ids=jnp.broadcast_to(jnp.arange(split, ids.shape[1], dtype=jnp.int32), (2, cont_len)),
            past_key_values=prefill.past_key_values,
        )
        np.testing.assert_allclose(np.asarray(cont.logits), np.asarray(full.logits[:, split:, :]), rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
