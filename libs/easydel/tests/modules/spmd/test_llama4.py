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

"""Tests for LLaMA4 text model."""

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


class TestLlama4:
    """Test suite for LLaMA4 text model."""

    @pytest.fixture
    def llama4_config(self, small_model_config):
        """Create LLaMA4 text-specific config."""
        return ed.Llama4TextConfig(
            hidden_size=128,
            intermediate_size=512,
            intermediate_size_mlp=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=128,
            use_qk_norm=False,
        )

    def test_causal_lm(self, llama4_config, small_model_config):
        """Test Llama4ForCausalLM."""
        tester = CausalLMTester()
        llama4_config.moe_force_xla_gmm = True
        result = tester.run(
            module_name="llama4_text",
            hf_class=transformers.Llama4ForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=llama4_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Llama4 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, llama4_config, small_model_config):
        """Test LLaMA4 text generation."""
        tester = CausalLMTester()
        llama4_config.moe_force_xla_gmm = True
        result = tester.test_generation(
            module_name="llama4_text",
            hf_class=transformers.Llama4ForCausalLM,
            config=llama4_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Llama4 generation failed: {result.error_message}"

    def test_custom_rope_and_chunk_schedule_parity(self, llama4_config, small_model_config):
        """Non-default rotary flags and explicit mask types match HF attention."""
        tester = CausalLMTester()
        llama4_config.moe_force_xla_gmm = True
        llama4_config.no_rope_layers = [0, 1, 0, 1]
        llama4_config.layer_types = ["full_attention", "chunked_attention", "full_attention", "chunked_attention"]
        llama4_config.attention_chunk_size = 8
        llama4_config.floor_scale = 8
        result = tester.run(
            module_name="llama4_text",
            hf_class=transformers.Llama4ForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=llama4_config,
            small_model_config=small_model_config,
        )
        assert result.success, (
            f"Llama4 custom attention schedule parity failed: {result.error_message or result.comparison.details}"
        )

    def test_chunked_mask_and_nope_temperature_parity(self, llama4_config, small_model_config):
        """HF parity when chunked attention and NoPE temperature tuning are active.

        With the default ``attention_chunk_size=8192`` and ``floor_scale=8192``
        neither mechanism changes results at the suite's 128-token sequence
        length, so a regression in the per-layer chunked-mask dispatch or the
        NoPE-layer query rescaling is invisible. Shrink both knobs so the
        chunked mask bites on RoPE layers and the log-position temperature is
        non-trivial on NoPE layers, then require exact parity with HF.
        """
        tester = CausalLMTester()
        llama4_config.moe_force_xla_gmm = True
        llama4_config.attention_chunk_size = 8
        llama4_config.attn_temperature_tuning = True
        llama4_config.attn_scale = 0.5
        llama4_config.floor_scale = 8
        result = tester.run(
            module_name="llama4_text",
            hf_class=transformers.Llama4ForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=llama4_config,
            small_model_config=small_model_config,
        )
        assert result.success, (
            f"Llama4 chunked/NoPE-temperature parity failed: {result.error_message or result.comparison.details}"
        )

    def test_cached_decode_matches_full_forward(self):
        """Cached one-token decode equals the full forward on RoPE/chunked and NoPE/full layers.

        A tiny chunk size and ``floor_scale`` make the chunk mask, the causal mask on NoPE layers and
        the log-position temperature all matter, so a decode mask that only sees key 0 or uses the wrong
        absolute position changes the logits.
        """
        config = ed.Llama4TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            intermediate_size_mlp=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_local_experts=4,
            num_experts_per_tok=1,
            max_position_embeddings=64,
            use_qk_norm=False,
            no_rope_layers=[1, 0, 1, 0],
            attention_chunk_size=4,
            attn_temperature_tuning=True,
            attn_scale=0.5,
            floor_scale=4,
            attn_dtype=jnp.float32,
            kvdtype=jnp.float32,
            attn_mechanism="vanilla",
            moe_force_xla_gmm=True,
            sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        )
        model = ed.AutoEasyDeLModelForCausalLM.from_config(
            config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST
        )
        ids = jnp.asarray([[3, 5, 7, 9, 11, 13, 15, 17, 19, 21], [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]], jnp.int32)
        prefix = 6
        full = model(input_ids=ids)
        cache = model.init_cache(batch_size=2, max_length=16)
        out = model(input_ids=ids[:, :prefix], past_key_values=cache)
        for step in range(prefix, ids.shape[1]):
            out = model(
                input_ids=ids[:, step : step + 1],
                position_ids=jnp.full((2, 1), step, dtype=jnp.int32),
                past_key_values=out.past_key_values,
            )
            np.testing.assert_allclose(
                np.asarray(out.logits[:, -1, :]), np.asarray(full.logits[:, step, :]), rtol=2e-3, atol=2e-3
            )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
