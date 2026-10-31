"""DeepSeek-V2 grouped routing numerical contracts (accelerator only)."""

import functools

import jax
import numpy as np
import pytest
from easydel.modules.deepseek_v2.deepseek_configuration import DeepseekV2Config
from easydel.modules.deepseek_v2.modeling_deepseek import DeepseekV2MoE
from jax import numpy as jnp


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("norm_topk_prob", [False, True])
@pytest.mark.parametrize("groups,active_groups,k", [(2, 1, 2), (2, 1, 3), (4, 2, 3), (4, 2, 5)])
def test_grouped_selection_matches_independent_assignment_and_weight_reference(
    groups, active_groups, k, norm_topk_prob, sharded, compiled
):
    """Cover narrow sharded group selection, ties, underflow, and zero padding."""
    logits = np.asarray(
        [
            [7, -4, 3, 2, -1, 0, 1, 4],
            [-1, 2, 3, 8, -3, -2, 0, 1],
            [2, 1, -3, 7, 0, 2, 9, -1],
            [1, 2, 8, -5, 0, 10, 1, -2],
            [-10000, -10000, -10000, -10000, 10000, 0, 0, 0],
            [1, 2, 8, -5, 1, 2, 8, -5],
            [0, 1, 6, 5, 0, 1, 6, 5],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    experts = logits.shape[-1]
    config = DeepseekV2Config(
        hidden_size=8,
        n_routed_experts=experts,
        num_experts_per_tok=k,
        topk_method="group_limited_greedy",
        n_group=groups,
        topk_group=active_groups,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=1.7,
    )
    device_logits = jnp.asarray(logits)
    if sharded:
        device_logits = jax.device_put(
            device_logits,
            jax.sharding.NamedSharding(config.mesh.jax_mesh, jax.sharding.PartitionSpec("fsdp", None)),
        )
    selector = functools.partial(DeepseekV2MoE._select_hook(config), pre_bias_logits=None, k=k)
    compute = jax.jit(selector) if compiled else selector
    actual_weights, actual_indices = compute(device_logits)
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    group_scores = probabilities.reshape(-1, groups, experts // groups).max(axis=-1)
    chosen_groups = np.argsort(-group_scores, axis=-1, kind="stable")[:, :active_groups]
    allowed = np.zeros_like(group_scores, dtype=bool)
    np.put_along_axis(allowed, chosen_groups, True, axis=-1)
    eligibility = np.repeat(allowed, experts // groups, axis=-1)
    selection_scores = np.where(eligibility, probabilities, -np.inf)
    expected_indices = np.argsort(-selection_scores, axis=-1, kind="stable")[:, :k]
    expected_weights = np.take_along_axis(np.where(eligibility, probabilities, 0), expected_indices, axis=-1)
    if norm_topk_prob and k > 1:
        expected_weights /= expected_weights.sum(axis=-1, keepdims=True) + 1e-20
    expected_weights *= 1.7
    np.testing.assert_array_equal(actual_indices, expected_indices)
    np.testing.assert_allclose(actual_weights, expected_weights, rtol=1e-5, atol=1e-7)
    assert np.isfinite(np.asarray(actual_weights)).all()
    actual_gradient = jax.jit(jax.grad(lambda x: jnp.sum(selector(x)[0] ** 2)))(device_logits)
    assert np.isfinite(np.asarray(actual_gradient)).all()
