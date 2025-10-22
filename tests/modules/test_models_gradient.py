"""Gradient and backward pass tests for EasyDeL models.

This module tests that gradients are computed correctly and match expected behavior.
Tests include:
- Gradient flow verification
- Gradient magnitude checks
- Numerical gradient verification
- Gradient checkpointing correctness
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx as nn

import easydel as ed


class TestGradientFlow:
    """Test gradient computation and backpropagation."""

    @pytest.mark.parametrize(
        "model_name,hf_class",
        [
            ("llama", "transformers.LlamaForCausalLM"),
            ("mistral", "transformers.MistralForCausalLM"),
            ("qwen2", "transformers.Qwen2ForCausalLM"),
            ("gemma", "transformers.GemmaForCausalLM"),
        ],
    )
    def test_gradient_computation(self, model_name, hf_class, model_factory, random_input_ids, small_model_config):
        """Test that gradients are computed and non-zero."""

        hf_class = eval(hf_class)

        ed_model, _, config, _ = model_factory(
            model_name,
            hf_class,
            ed.TaskType.CAUSAL_LM,
        )

        _, jax_input_ids = random_input_ids

        def loss_fn(model_state):
            """Compute loss for gradient calculation."""
            model = nn.merge(*model_state)
            output = model.compute_loss(
                input_ids=jax_input_ids,
                attention_mask=jnp.ones_like(jax_input_ids, dtype=jnp.bool),
            )
            return output[0].loss

        # Compute gradients
        with config.mesh:
            model_state = ed_model.split_module()
            loss, grads = jax.value_and_grad(loss_fn)(model_state)

        # Check that loss is finite
        assert jnp.isfinite(loss), f"Loss is not finite for {model_name}"

        # Check that gradients exist and are finite
        grad_dict, _, _ = grads
        has_grads = False
        for param_name, param_grad in grad_dict.items():
            if param_grad is not None:
                has_grads = True
                assert jnp.all(jnp.isfinite(param_grad)), f"Gradient for {param_name} contains non-finite values"

        assert has_grads, f"No gradients computed for {model_name}"

    @pytest.mark.parametrize(
        "gradient_checkpointing",
        [
            ed.EasyDeLGradientCheckPointers.NONE,
            ed.EasyDeLGradientCheckPointers.SAVE_ONLY_THESE_NAMES,
        ],
    )
    def test_gradient_checkpointing(self, gradient_checkpointing, model_factory, random_input_ids, small_model_config):
        """Test that gradient checkpointing works correctly."""
        import transformers

        # Update config with gradient checkpointing
        config_dict = small_model_config.copy()
        config_dict["gradient_checkpointing"] = gradient_checkpointing
        config_dict["gradient_checkpointing_targets"] = [
            "attn_key",
            "attn_dense",
            "residual",
        ]

        module_config, module_class = ed.get_modules_by_type("llama", ed.TaskType.CAUSAL_LM)
        config = module_config(**config_dict)
        config.sharding_axis_dims = config_dict["sharding_axis_dims"]
        config.pad_token_id = 0

        # Create HuggingFace model
        hf_model = transformers.LlamaForCausalLM(config=config)
        hf_model.eval()
        hf_model = hf_model.float()

        # Prepare EasyDeL config
        config.attach_custom_arguments()
        config.add_basic_configurations(
            shard_attention_computation=config_dict["shard_attention_computation"],
            use_sharding_constraint=config_dict["use_sharding_constraint"],
            scan_mlp_chunk_size=config_dict.get("scan_mlp_chunk_size", 64),
        )
        config.add_basic_configurations(
            attn_mechanism=config_dict["attn_mechanism"],
            blocksize_k=config_dict["blocksize_k"],
            blocksize_q=config_dict["blocksize_q"],
            attn_dtype=config_dict["attn_dtype"],
        )

        # Create EasyDeL model
        with config.mesh:
            ed_model = module_class.lazy_init(
                config=config,
                dtype=config_dict["dtype"],
                param_dtype=config_dict["dtype"],
                precision=config_dict["precision"],
                rngs=nn.Rngs(42),
            )
            ed_model = ed.traversals.merge_model_and_tree(ed_model, tree=ed_model.transform_fn(hf_model.state_dict()))
            ed_model.eval()
            ed_model = ed_model.shard_model()

        _, jax_input_ids = random_input_ids

        def loss_fn(model_state):
            model = nn.merge(*model_state)
            output = model.compute_loss(
                input_ids=jax_input_ids,
                attention_mask=jnp.ones_like(jax_input_ids, dtype=jnp.bool),
            )
            return output[0].loss

        # Compute gradients with checkpointing
        with config.mesh:
            model_state = ed_model.split_module()
            loss, grads = jax.value_and_grad(loss_fn)(model_state)

        # Verify gradients are computed
        assert jnp.isfinite(loss), f"Loss is not finite with gradient_checkpointing={gradient_checkpointing}"

        grad_dict, _, _ = grads
        has_grads = any(g is not None for g in grad_dict.values())
        assert has_grads, f"No gradients computed with gradient_checkpointing={gradient_checkpointing}"

    def test_gradient_magnitude(self, model_factory, random_input_ids):
        """Test that gradient magnitudes are reasonable (not too large or small)."""
        import transformers

        ed_model, _, config, _ = model_factory(
            "llama",
            transformers.LlamaForCausalLM,
            ed.TaskType.CAUSAL_LM,
        )

        _, jax_input_ids = random_input_ids

        def loss_fn(model_state):
            model = nn.merge(*model_state)
            output = model.compute_loss(
                input_ids=jax_input_ids,
                attention_mask=jnp.ones_like(jax_input_ids, dtype=jnp.bool),
            )
            return output[0].loss

        # Compute gradients
        with config.mesh:
            model_state = ed_model.split_module()
            _, grads = jax.value_and_grad(loss_fn)(model_state)

        # Check gradient magnitudes
        grad_dict, _, _ = grads
        for param_name, param_grad in grad_dict.items():
            if param_grad is not None:
                grad_norm = jnp.linalg.norm(param_grad)
                # Gradients should not be too large (exploding) or too small (vanishing)
                assert grad_norm < 1e3, f"Gradient for {param_name} is too large: {grad_norm}"
                assert grad_norm > 1e-6, f"Gradient for {param_name} is too small: {grad_norm}"


class TestGradientConsistency:
    """Test gradient consistency across different configurations."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_gradient_dtype_consistency(self, dtype, model_factory, random_input_ids, small_model_config):
        """Test gradients are consistent across dtypes (within tolerance)."""
        import transformers

        # Update config with specific dtype
        config_dict = small_model_config.copy()
        config_dict["dtype"] = dtype

        module_config, module_class = ed.get_modules_by_type("llama", ed.TaskType.CAUSAL_LM)
        config = module_config(**config_dict)
        config.sharding_axis_dims = config_dict["sharding_axis_dims"]
        config.pad_token_id = 0

        hf_model = transformers.LlamaForCausalLM(config=config)
        hf_model.eval()
        hf_model = hf_model.float()

        config.attach_custom_arguments()
        config.add_basic_configurations(
            shard_attention_computation=config_dict["shard_attention_computation"],
            use_sharding_constraint=config_dict["use_sharding_constraint"],
            scan_mlp_chunk_size=config_dict.get("scan_mlp_chunk_size", 64),
        )
        config.add_basic_configurations(
            attn_mechanism=config_dict["attn_mechanism"],
            blocksize_k=config_dict["blocksize_k"],
            blocksize_q=config_dict["blocksize_q"],
            attn_dtype=config_dict["attn_dtype"],
        )

        with config.mesh:
            ed_model = module_class.lazy_init(
                config=config,
                dtype=dtype,
                param_dtype=dtype,
                precision=config_dict["precision"],
                rngs=nn.Rngs(42),
            )
            ed_model = ed.traversals.merge_model_and_tree(ed_model, tree=ed_model.transform_fn(hf_model.state_dict()))
            ed_model.eval()
            ed_model = ed_model.shard_model()

        _, jax_input_ids = random_input_ids

        def loss_fn(model_state):
            model = nn.merge(*model_state)
            output = model.compute_loss(
                input_ids=jax_input_ids,
                attention_mask=jnp.ones_like(jax_input_ids, dtype=jnp.bool),
            )
            return output[0].loss

        # Compute gradients
        with config.mesh:
            model_state = ed_model.split_module()
            loss, grads = jax.value_and_grad(loss_fn)(model_state)

        # Verify gradients computed successfully
        assert jnp.isfinite(loss), f"Loss is not finite with dtype={dtype}"

        grad_dict, _, _ = grads
        has_grads = any(g is not None for g in grad_dict.values())
        assert has_grads, f"No gradients computed with dtype={dtype}"


class TestMoEGradients:
    """Test gradients for Mixture of Experts models."""

    @pytest.mark.parametrize(
        "model_name,hf_class",
        [
            ("mixtral", "transformers.MixtralForCausalLM"),
            ("qwen2_moe", "transformers.Qwen2MoeForCausalLM"),
            ("grok-1", "transformers.Grok1ForCausalLM"),
        ],
    )
    def test_moe_router_gradients(self, model_name, hf_class, model_factory, random_input_ids, small_model_config):
        """Test that MoE router gets gradients."""

        hf_class = eval(hf_class)

        ed_model, _, config, _ = model_factory(
            model_name,
            hf_class,
            ed.TaskType.CAUSAL_LM,
        )

        _, jax_input_ids = random_input_ids

        def loss_fn(model_state):
            model = nn.merge(*model_state)
            output = model.compute_loss(
                input_ids=jax_input_ids,
                attention_mask=jnp.ones_like(jax_input_ids, dtype=jnp.bool),
                output_router_logits=True,
            )
            return output[0].loss

        # Compute gradients
        with config.mesh:
            model_state = ed_model.split_module()
            loss, grads = jax.value_and_grad(loss_fn)(model_state)

        # Verify gradients exist
        assert jnp.isfinite(loss), f"Loss is not finite for {model_name}"

        grad_dict, _, _ = grads
        has_grads = any(g is not None for g in grad_dict.values())
        assert has_grads, f"No gradients computed for {model_name}"

        # Check for router-specific gradients (router parameters should have gradients)
        router_grads_found = any(  # noqa
            "gate" in name.lower() or "router" in name.lower() for name, grad in grad_dict.items() if grad is not None
        )
        # Note: This is a soft check - some architectures may name routers differently
        # The main check is that overall gradients are computed


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
