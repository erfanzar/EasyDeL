"""KV cache correctness tests for EasyDeL models.

This module tests that key-value caching works correctly for autoregressive generation.
Tests include:
- Cache initialization
- Cache update correctness
- Multi-step generation with cache
- Cache consistency across steps
"""

import jax.numpy as jnp
import pytest
from flax import nnx as nn

import easydel as ed


class TestKVCacheCorrectness:
    """Test KV cache functionality and correctness."""

    @pytest.mark.parametrize(
        "model_name,hf_class",
        [
            ("llama", "transformers.LlamaForCausalLM"),
            ("mistral", "transformers.MistralForCausalLM"),
            ("qwen2", "transformers.Qwen2ForCausalLM"),
            ("gemma", "transformers.GemmaForCausalLM"),
        ],
    )
    def test_cache_initialization(self, model_name, hf_class, model_factory):
        """Test that cache is properly initialized."""
        hf_class = eval(hf_class)

        ed_model, _, config, _ = model_factory(
            model_name,
            hf_class,
            ed.TaskType.CAUSAL_LM,
        )

        batch_size = 2
        max_length = 128

        # Initialize cache
        with config.mesh:
            cache = ed_model.init_cache(
                batch_size=batch_size,
                max_length=max_length,
            )

        # Verify cache structure
        assert cache is not None, f"Cache not initialized for {model_name}"

        # Check cache dimensions
        if hasattr(cache, "views"):
            assert (
                len(cache.views) == config.num_hidden_layers
            ), f"Cache should have {config.num_hidden_layers} layers, got {len(cache.views)}"

    @pytest.mark.parametrize(
        "model_name,hf_class",
        [
            ("llama", "transformers.LlamaForCausalLM"),
            ("mistral", "transformers.MistralForCausalLM"),
        ],
    )
    def test_cache_update_correctness(self, model_name, hf_class, model_factory, random_input_ids):
        """Test that cache is updated correctly after forward pass."""
        hf_class = eval(hf_class)

        ed_model, _, config, _ = model_factory(
            model_name,
            hf_class,
            ed.TaskType.CAUSAL_LM,
        )

        _, jax_input_ids = random_input_ids
        batch_size, seq_len = jax_input_ids.shape

        # Initialize cache
        with config.mesh:
            cache = ed_model.init_cache(
                batch_size=batch_size,
                max_length=seq_len * 2,
            )

            # Forward pass with cache
            @ed.ejit(static_argnums=(1,))
            def forward_with_cache(ids, gd, gs, go, past_cache):
                model = nn.merge(gd, gs, go)
                return model(
                    input_ids=ids,
                    attention_mask=jnp.ones_like(ids, dtype=jnp.bool),
                    past_key_values=past_cache,
                )

            output = forward_with_cache(
                jax_input_ids,
                *ed_model.split_module(),
                cache,
            )

        # Verify output has cache
        assert hasattr(output, "past_key_values") or hasattr(
            output, "cache"
        ), f"Output should contain cache for {model_name}"

        # Get returned cache
        returned_cache = getattr(output, "past_key_values", None) or getattr(output, "cache", None)
        assert returned_cache is not None, f"Returned cache is None for {model_name}"

    def test_cached_vs_uncached_equivalence(self, model_factory, random_input_ids, small_model_config):
        """Test that cached and uncached forward passes produce same results."""
        import transformers

        ed_model, _, config, _ = model_factory(
            "llama",
            transformers.LlamaForCausalLM,
            ed.TaskType.CAUSAL_LM,
        )

        _, jax_input_ids = random_input_ids
        batch_size, seq_len = jax_input_ids.shape

        with config.mesh:
            # Uncached forward pass
            @ed.ejit(static_argnums=(1,))
            def forward_no_cache(ids, gd, gs, go):
                model = nn.merge(gd, gs, go)
                return model(
                    input_ids=ids,
                    attention_mask=jnp.ones_like(ids, dtype=jnp.bool),
                )

            output_no_cache = forward_no_cache(
                jax_input_ids,
                *ed_model.split_module(),
            )

            # Cached forward pass (first step)
            cache = ed_model.init_cache(
                batch_size=batch_size,
                max_length=seq_len * 2,
            )

            @ed.ejit(static_argnums=(1,))
            def forward_with_cache(ids, gd, gs, go, past_cache):
                model = nn.merge(gd, gs, go)
                return model(
                    input_ids=ids,
                    attention_mask=jnp.ones_like(ids, dtype=jnp.bool),
                    past_key_values=past_cache,
                )

            output_with_cache = forward_with_cache(
                jax_input_ids,
                *ed_model.split_module(),
                cache,
            )

        # Compare logits (first step should match)
        logits_no_cache = output_no_cache.logits
        logits_with_cache = output_with_cache.logits

        # They should be very close (allowing for numerical differences)
        assert jnp.allclose(logits_no_cache, logits_with_cache, atol=1e-4, rtol=1e-4), (
            f"Cached and uncached logits differ significantly. "
            f"Max diff: {jnp.abs(logits_no_cache - logits_with_cache).max()}"
        )

    def test_multi_step_generation_with_cache(self, model_factory, random_input_ids, small_model_config):
        """Test multi-step generation using cache."""
        import transformers

        ed_model, _, config, _ = model_factory(
            "llama",
            transformers.LlamaForCausalLM,
            ed.TaskType.CAUSAL_LM,
        )

        _, jax_input_ids = random_input_ids
        batch_size, seq_len = jax_input_ids.shape
        num_gen_steps = 5  # Generate 5 tokens

        with config.mesh:
            # Initialize cache
            cache = ed_model.init_cache(
                batch_size=batch_size,
                max_length=seq_len + num_gen_steps,
            )

            @ed.ejit(static_argnums=(1,))
            def forward_step(ids, gd, gs, go, past_cache):
                model = nn.merge(gd, gs, go)
                return model(
                    input_ids=ids,
                    attention_mask=jnp.ones_like(ids, dtype=jnp.bool),
                    past_key_values=past_cache,
                )

            # First step with full sequence
            output = forward_step(
                jax_input_ids,
                *ed_model.split_module(),
                cache,
            )

            assert output is not None, "First forward pass failed"
            current_cache = getattr(output, "past_key_values", None) or getattr(output, "cache", None)

            # Subsequent steps with single token
            for step in range(num_gen_steps):
                # Get next token (greedy decoding)
                next_token = jnp.argmax(output.logits[:, -1:, :], axis=-1)

                # Forward pass with single token and cache
                output = forward_step(
                    next_token,
                    *ed_model.split_module(),
                    current_cache,
                )

                assert output is not None, f"Generation step {step} failed"
                current_cache = getattr(output, "past_key_values", None) or getattr(output, "cache", None)

                # Verify cache was updated
                assert current_cache is not None, f"Cache is None at step {step}"

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_cache_with_different_batch_sizes(self, batch_size, model_factory, small_model_config):
        """Test cache works with different batch sizes."""
        import numpy as np
        import transformers

        ed_model, _, config, _ = model_factory(
            "llama",
            transformers.LlamaForCausalLM,
            ed.TaskType.CAUSAL_LM,
        )

        seq_len = small_model_config["sequence_length"]
        vocab_size = small_model_config["vocab_size"]

        # Create random input
        np.random.seed(42)  # noqa
        np_input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))  # noqa
        jax_input_ids = jnp.asarray(np_input_ids, dtype="i4")

        with config.mesh:
            # Initialize cache for this batch size
            cache = ed_model.init_cache(
                batch_size=batch_size,
                max_length=seq_len * 2,
            )

            @ed.ejit(static_argnums=(1,))
            def forward_with_cache(ids, gd, gs, go, past_cache):
                model = nn.merge(gd, gs, go)
                return model(
                    input_ids=ids,
                    attention_mask=jnp.ones_like(ids, dtype=jnp.bool),
                    past_key_values=past_cache,
                )

            output = forward_with_cache(
                jax_input_ids,
                *ed_model.split_module(),
                cache,
            )

        # Verify output shape matches batch size
        assert (
            output.logits.shape[0] == batch_size
        ), f"Output batch size {output.logits.shape[0]} != input batch size {batch_size}"


class TestCacheEdgeCases:
    """Test edge cases and error handling for KV cache."""

    def test_cache_with_empty_input(self, model_factory, small_model_config):
        """Test behavior with edge case inputs."""
        import transformers

        ed_model, _, config, _ = model_factory(
            "llama",
            transformers.LlamaForCausalLM,
            ed.TaskType.CAUSAL_LM,
        )

        batch_size = 1
        seq_len = 1  # Single token

        # Create minimal input
        jax_input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        with config.mesh:
            cache = ed_model.init_cache(
                batch_size=batch_size,
                max_length=128,
            )

            @ed.ejit(static_argnums=(1,))
            def forward_with_cache(ids, gd, gs, go, past_cache):
                model = nn.merge(gd, gs, go)
                return model(
                    input_ids=ids,
                    attention_mask=jnp.ones_like(ids, dtype=jnp.bool),
                    past_key_values=past_cache,
                )

            output = forward_with_cache(
                jax_input_ids,
                *ed_model.split_module(),
                cache,
            )

        assert output is not None, "Forward pass with single token failed"
        assert output.logits.shape[1] == seq_len, "Output sequence length mismatch"


class TestPagedAttentionCache:
    """Test paged attention / ragged pages cache if available."""

    @pytest.mark.skip(reason="Paged attention tests require specific setup")
    def test_paged_cache_initialization(self, model_factory):
        """Test paged cache initialization."""
        # TODO: Implement paged attention cache tests when available
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
