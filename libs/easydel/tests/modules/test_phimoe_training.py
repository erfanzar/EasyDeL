"""Training/evaluation routing state transitions for Phi-MoE (accelerator only)."""

import jax
import numpy as np
import spectrax as spx
from easydel.modules.phimoe.modeling_phimoe import PhiMoeSparseMoeBlock
from easydel.modules.phimoe.phimoe_configuration import PhiMoeConfig
from jax import numpy as jnp


def make_block(jitter):
    config = PhiMoeConfig(
        hidden_size=16,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=4,
        input_jitter_noise=jitter,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape((1,) * 6), ("pp", "dp", "fsdp", "ep", "tp", "sp"))
    config.set_model_mesh(mesh)
    return PhiMoeSparseMoeBlock(config, dtype=jnp.float32, param_dtype=jnp.float32, layer_idx=0, rngs=spx.Rngs(9))


def test_input_jitter_follows_train_eval_and_deterministic_override():
    block = make_block(0.2)
    inputs = jnp.ones((1, 16, 16), dtype=jnp.float32)
    block.eval()
    expected_output, expected_logits = block(inputs)
    np.testing.assert_array_equal(block(inputs)[1], expected_logits)
    block.train()
    first_output, first_logits = block(inputs)
    _, second_logits = block(inputs)
    assert not np.array_equal(first_logits, expected_logits)
    assert not np.array_equal(first_logits, second_logits)
    forced_output, forced_logits = block(inputs, deterministic=True)
    np.testing.assert_allclose(forced_output, expected_output, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(forced_logits, expected_logits)
    assert first_output.shape == inputs.shape


def test_training_samples_sparsemixer_experts_even_without_input_jitter():
    block = make_block(0.0)
    # Tied router scores leave every expert within the sparsity threshold;
    # greedy eval is repeatable, while training samples distinct expert pairs.
    block.router.weight.value = jnp.zeros_like(block.router.weight.value)
    inputs = jnp.ones((1, 32, 16), dtype=jnp.float32)
    block.eval()
    expected, _ = block(inputs)
    np.testing.assert_array_equal(block(inputs)[0], expected)
    block.train()
    first, logits = block(inputs)
    second, _ = block(inputs)
    np.testing.assert_array_equal(logits, jnp.zeros((1, 32, 4)))
    assert not np.array_equal(first, expected)
    assert not np.array_equal(first, second)
    assert np.isfinite(np.asarray(first)).all()
    gradients = spx.grad(lambda module: jnp.sum(module(inputs)[0]))(block)
    router_gradient = np.asarray(gradients["parameters"]["router"]["weight"])
    assert np.isfinite(router_gradient).all()
    assert np.any(router_gradient != 0)
