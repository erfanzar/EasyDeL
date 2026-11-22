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

"""Tests for LLaMA model."""

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


class TestLlama:
    """Test suite for LLaMA model variants."""

    @pytest.fixture
    def llama_config(self, small_model_config):
        """Create LLaMA-specific config with rope scaling."""
        config_dict = small_model_config.copy()
        config_dict["rope_scaling"] = {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }
        return ed.LlamaConfig(
            vocab_size=config_dict["vocab_size"],
            hidden_size=config_dict["hidden_size"],
            num_hidden_layers=config_dict["num_hidden_layers"],
            num_attention_heads=config_dict["num_attention_heads"],
            num_key_value_heads=config_dict["num_key_value_heads"],
            intermediate_size=config_dict["intermediate_size"],
            max_position_embeddings=config_dict["max_position_embeddings"],
            rope_scaling=config_dict["rope_scaling"],
        )

    def test_causal_lm(self, llama_config, small_model_config):
        """Test LlamaForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="llama",
            hf_class=transformers.LlamaForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=llama_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Llama CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, llama_config, small_model_config):
        """Test LLaMA text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="llama",
            hf_class=transformers.LlamaForCausalLM,
            config=llama_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Llama generation failed: {result.error_message}"

    def test_multi_token_cached_continuation_matches_full_forward(self):
        """A multi-token step over a populated dense cache must use absolute causal rows (r15 S10-1).

        A query-local mask used to reach the kernel's top-left (Q, K) causal triangle, so row r of a
        continuation written at cache index i saw keys <= r instead of <= i + r.
        """
        config = ed.LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
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
        np.testing.assert_allclose(np.asarray(prefill.logits), np.asarray(full.logits[:, :split]), rtol=2e-3, atol=2e-3)
        cont_len = ids.shape[1] - split
        cont = model(
            input_ids=ids[:, split:],
            position_ids=jnp.broadcast_to(jnp.arange(split, ids.shape[1], dtype=jnp.int32), (2, cont_len)),
            past_key_values=prefill.past_key_values,
        )
        np.testing.assert_allclose(np.asarray(cont.logits), np.asarray(full.logits[:, split:]), rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
