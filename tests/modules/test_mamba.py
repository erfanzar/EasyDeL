"""Tests for Mamba model."""

import pytest
import transformers
from jax import numpy as jnp

import easydel as ed

try:
    from .test_utils import CausalLMTester
except ImportError:
    from test_utils import CausalLMTester


class TestMamba:
    """Test suite for Mamba model."""

    @pytest.fixture
    def mamba_config(self, small_model_config):
        """Create Mamba-specific config."""
        return ed.MambaConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            intermediate_size=small_model_config["intermediate_size"],
        )

    def test_causal_lm(self, mamba_config, small_model_config):
        """Test MambaForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="mamba",
            hf_class=transformers.MambaForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=mamba_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Mamba CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, mamba_config, small_model_config):
        """Test Mamba text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="mamba",
            hf_class=transformers.MambaForCausalLM,
            config=mamba_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Mamba generation failed: {result.error_message}"

    def test_left_padding_invariance(self, mamba_config, small_model_config):
        """Mamba generation should be invariant to left padding when masked."""
        from transformers import GenerationConfig

        from tests.modules.test_utils.model_factory import create_ed_model_only, setup_config

        config = setup_config(mamba_config, small_model_config)
        config.sharding_axis_dims = (1, 1, -1, 1, 1)

        with config.mesh:
            ed_model = create_ed_model_only(
                module_name="mamba",
                task=ed.TaskType.CAUSAL_LM,
                config=config,
                small_model_config=small_model_config,
            )
            if not hasattr(ed_model, "generation_config") or ed_model.generation_config is None:
                ed_model.generation_config = GenerationConfig(
                    max_length=small_model_config.get("max_position_embeddings", 128),
                    max_new_tokens=4,
                    pad_token_id=getattr(config, "pad_token_id", 0) or 0,
                    eos_token_id=getattr(config, "eos_token_id", 1) or 1,
                    bos_token_id=getattr(config, "bos_token_id", 0) or 0,
                )

            prompt = jnp.array([[5, 6, 7, 8]], dtype="i4")
            attention_mask = jnp.ones(prompt.shape, dtype=jnp.bool_)

            pad_amt = 4
            padded = jnp.concatenate(
                [jnp.full((1, pad_amt), int(ed_model.generation_config.pad_token_id), dtype="i4"), prompt],
                axis=1,
            )
            padded_mask = jnp.concatenate([jnp.zeros((1, pad_amt), dtype=jnp.bool_), attention_mask], axis=1)

            max_new = 4
            out_unpadded = ed_model.generate(
                input_ids=prompt,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
                do_sample=False,
            )
            out_padded = ed_model.generate(
                input_ids=padded,
                attention_mask=padded_mask,
                max_new_tokens=max_new,
                do_sample=False,
            )

            assert bool(jnp.all(out_unpadded.sequences[:, -max_new:] == out_padded.sequences[:, -max_new:]))


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
