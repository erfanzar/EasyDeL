"""Non-SiLU Qwen2-MoE numerics against NumPy references (accelerator only)."""

import jax
import numpy as np
import spectrax as spx
from easydel.modules.qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP, Qwen2MoeMLPStack, Qwen2MoeSparseBlock
from easydel.modules.qwen2_moe.qwen2_moe_configuration import Qwen2MoeConfig
from jax import numpy as jnp


def relu_config():
    config = Qwen2MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        hidden_act="relu",
        norm_topk_prob=True,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    config.set_model_mesh(
        jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape((1,) * 6), ("pp", "dp", "fsdp", "ep", "tp", "sp"))
    )
    return config


def set_mlp_weights(mlp, seed):
    rng = np.random.default_rng(seed)
    # Powers-of-two values are exactly representable even in TPU bf16 products.
    for projection in (mlp.gate_up_proj, mlp.down_proj):
        projection.weight.value = jnp.asarray(rng.integers(-2, 3, size=projection.weight.value.shape) / 8, jnp.float32)


def numpy_mlp(inputs, mlp, expert=None):
    gate_up = np.asarray(mlp.gate_up_proj.weight.value)
    down = np.asarray(mlp.down_proj.weight.value)
    if expert is not None:
        gate_up, down = gate_up[expert], down[expert]
    gate, up = np.split(inputs @ gate_up, 2, axis=-1)
    return (np.maximum(gate, 0) * up) @ down


def test_dense_mlp_honors_non_silu_activation():
    model = Qwen2MoeMLP(
        relu_config(),
        intermediate_size=16,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    set_mlp_weights(model, 4)
    inputs = np.random.default_rng(5).integers(-2, 3, size=(1, 4, 16)).astype(np.float32) / 4
    np.testing.assert_allclose(model(jnp.asarray(inputs)), numpy_mlp(inputs, model), rtol=1e-6, atol=1e-6)


def test_expert_stack_honors_non_silu_activation():
    model = Qwen2MoeMLPStack(relu_config(), dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(1))
    set_mlp_weights(model, 6)
    inputs = np.random.default_rng(7).integers(-2, 3, size=(4, 16)).astype(np.float32) / 4
    expected = np.concatenate((numpy_mlp(inputs[:2], model, 0), numpy_mlp(inputs[2:], model, 1)))
    outputs = model(jnp.asarray(inputs), group_sizes=jnp.array([2, 2], jnp.int32))
    np.testing.assert_allclose(outputs, expected, rtol=1e-6, atol=1e-6)


def test_routed_and_shared_experts_honor_non_silu_activation():
    model = Qwen2MoeSparseBlock(
        relu_config(), dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(2)
    )
    set_mlp_weights(model.experts, 8)
    set_mlp_weights(model.shared_expert, 9)
    model.gate.weight.value = jnp.zeros_like(model.gate.weight.value)
    model.shared_expert_gate.weight.value = jnp.zeros_like(model.shared_expert_gate.weight.value)
    inputs = np.random.default_rng(10).integers(-2, 3, size=(1, 4, 16)).astype(np.float32) / 4
    # Both experts selected with p=1/2; shared sigmoid gate(0) is also 1/2.
    expected = (
        numpy_mlp(inputs, model.experts, 0)
        + numpy_mlp(inputs, model.experts, 1)
        + numpy_mlp(inputs, model.shared_expert)
    ) / 2
    actual, logits = model(jnp.asarray(inputs))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(logits, 0)
