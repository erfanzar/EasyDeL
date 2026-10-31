"""Full-block routing must survive chunking, padding and EP dispatch.

Numerical tests run on an accelerator only.
"""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.moe._communication_utils import permute, unpermute
from easydel.modules.arctic import ArcticConfig
from easydel.modules.arctic.modeling_arctic import ArcticMoeBlock
from jax import numpy as jnp


@pytest.mark.parametrize("rotation", [None, 2])
def test_preselected_routing_preserves_weights_indices_and_input_hooks(rotation):
    """Dispatch consumes selected routes verbatim, including zero-weight padding."""
    inputs = jnp.array([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]])
    weights = jnp.array([[0.25, 0.5], [0.75, 0.0], [0.0, 0.0]])
    experts = jnp.array([[3, 1], [2, 0], [0, 0]], dtype=jnp.int32)

    def must_not_run(*args):
        raise AssertionError("Preselected routing must not reselect or renormalize")

    def refine_inputs(x, w, shape):
        return x + jnp.sum(w, axis=-1, keepdims=True)

    def scale_inputs(x, w):
        return x * w[:, None]

    def dispatch(w):
        rows, order, routed_weights, sizes, sorted_experts = permute(
            inputs,
            gate_logits=jnp.zeros((3, 4)),
            num_experts_per_tok=2,
            num_experts=4,
            dtype=jnp.float32,
            roll_to_expert_id=rotation,
            select_hook=must_not_run,
            refine_weights_hook=must_not_run,
            refine_inputs_hook=refine_inputs,
            scale_replicated_inputs=scale_inputs,
            preselected_routing=(w, experts),
        )
        # Restore global IDs after the ring rotation for this simple expert reference.
        global_experts = (sorted_experts + (rotation or 0)) % 4
        expert_outputs = rows * (global_experts[:, None] + 1)
        output = unpermute(
            expert_outputs,
            order,
            routed_weights,
            batch_size=1,
            sequence_length=3,
            num_experts_per_tok=2,
            dtype=jnp.float32,
        )
        return output, sizes, routed_weights

    actual, sizes, routed_weights = jax.jit(dispatch)(weights)
    w_np = np.asarray(weights)
    x_np = np.asarray(inputs)[0] + w_np.sum(axis=-1, keepdims=True)
    expected = x_np * np.sum(w_np**2 * (np.asarray(experts) + 1), axis=-1, keepdims=True)
    np.testing.assert_allclose(actual[0], expected, atol=1e-6)
    np.testing.assert_array_equal(routed_weights, weights)
    rotated = (np.asarray(experts).ravel() - (rotation or 0)) % 4
    np.testing.assert_array_equal(sizes, np.bincount(rotated, minlength=4))
    # Independent derivative of sum_h (x_h + sum_k w_k) * sum_k w_k^2 * (expert_k + 1).
    coeff = np.asarray(experts) + 1
    expected_grad = (
        inputs.shape[-1] * np.sum(w_np**2 * coeff, axis=-1, keepdims=True)
        + x_np.sum(axis=-1, keepdims=True) * 2 * w_np * coeff
    )
    grad = jax.grad(lambda w: jnp.sum(dispatch(w)[0]))(weights)
    np.testing.assert_allclose(grad, expected_grad, atol=1e-6)


@pytest.mark.parametrize("ring,ep_size", [(False, 1), (True, 1), (False, 2), (True, 2)])
@pytest.mark.parametrize("chunk_size", [3, 99])
@pytest.mark.parametrize("minimum_capacity", [0, 7])
def test_capacity_chunking_matches_full_dispatch_and_input_gradients(ring, ep_size, chunk_size, minimum_capacity):
    """Block-wide capacity, including minimums, is independent of chunk shape."""
    devices = jax.device_count()
    if devices % ep_size:
        pytest.skip(f"needs a device count divisible by ep={ep_size}")
    config = ArcticConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_layer_frequency=1,
        max_position_embeddings=16,
        moe_method="fused_moe",
        moe_train_capacity_factor=1.0,
        moe_min_capacity=minimum_capacity,
        sharding_axis_dims=(1, 1, devices // ep_size, ep_size, 1, 1),
        sharding_axis_names=("pp", "dp", "fsdp", "ep", "tp", "sp"),
    )
    config.add_basic_configurations(
        use_ring_of_experts=ring,
        fsdp_is_ep_bound=False,
        sp_is_ep_bound=False,
        moe_chunk_size=None,
    )
    config.moe_force_xla_gmm = True
    block = ArcticMoeBlock(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(3), layer_idx=0)
    block.train()
    # All tokens choose the same two experts: later assignments must be dropped.
    block.gate.weight.value = jnp.zeros_like(block.gate.weight.value)
    hidden = jax.random.normal(jax.random.PRNGKey(8), (2 * devices, 5, 8))

    def loss(x):
        return jnp.sum(block(x)[0] ** 2)

    with config.mesh:
        config.moe_token_dropping = False
        unrestricted, _ = block(hidden)
        config.moe_token_dropping = True
        full, _ = block(hidden)
        full_grad = jax.grad(loss)(hidden)
        config.moe_chunk_size = chunk_size
        chunked, _ = block(hidden)
        chunked_grad = jax.grad(loss)(hidden)

    # This fixture must actually exercise capacity, not just compare unrestricted MoEs.
    assert np.any(np.linalg.norm(np.asarray(unrestricted), axis=-1) > 0)
    assert np.any(np.linalg.norm(np.asarray(full), axis=-1) == 0)
    assert np.any(np.linalg.norm(np.asarray(full), axis=-1) > 0)
    np.testing.assert_allclose(chunked, full, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(chunked_grad, full_grad, rtol=1e-5, atol=1e-6)
