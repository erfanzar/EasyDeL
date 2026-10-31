"""DeepSeek-V3 excess-capacity routing contracts (accelerator only)."""

import functools

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.deepseek_v3.deepseek_configuration import DeepseekV3Config
from easydel.modules.deepseek_v3.modeling_deepseek import DeepseekV3MoE
from jax import numpy as jnp


def _score_selector(config, bias, k):
    """DeepSeek-V3's select hook fed scores directly, so the references need no sigmoid inversion."""
    hook = DeepseekV3MoE._select_hook(config, None if bias is None else jnp.asarray(bias))
    return functools.partial(hook, pre_bias_logits=None, k=k, score_fn="none")


def _token_sharded(config, array):
    return jax.device_put(
        array,
        jax.sharding.NamedSharding(config.mesh.jax_mesh, jax.sharding.PartitionSpec("fsdp", None)),
    )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("norm_topk_prob", [False, True])
@pytest.mark.parametrize("bias_kind", ["absent", "zero", "nonzero"])
def test_excess_capacity_has_finite_eligibility_masked_weights(bias_kind, norm_topk_prob, sharded, compiled):
    """Selection bias and -inf exclusion must never give padded experts weight."""
    probabilities = np.asarray(
        [[0.9, 0.8, 0.1, 0.2], [0.1, 0.2, 0.9, 0.8], [0, 0, 0.9, 0.8], [0.9, 0.8, 0, 0]],
        dtype=np.float32,
    )
    bias = None if bias_kind == "absent" else np.zeros(4, dtype=np.float32)
    if bias_kind == "nonzero":
        # Reverse group membership for some rows and permit negative choice scores.
        bias = np.asarray([-0.8, -0.7, -0.1, -0.2], dtype=np.float32)
    choices = probabilities if bias is None else probabilities + bias
    config = DeepseekV3Config(
        hidden_size=4,
        n_routed_experts=4,
        num_experts_per_tok=3,
        n_group=2,
        topk_group=1,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=1.7,
    )
    device_scores = jnp.asarray(probabilities)
    if sharded:
        device_scores = _token_sharded(config, device_scores)
    selector = _score_selector(config, bias, k=3)
    compute = jax.jit(selector) if compiled else selector
    actual_weights, actual_indices = compute(device_scores)
    # Independent noaux_tc reference: two experts per group means group rank
    # is the sum of both choice scores; only unbiased eligible scores combine.
    winning_group = choices.reshape(4, 2, 2).sum(axis=-1).argmax(axis=-1)
    eligibility = np.arange(4)[None, :] // 2 == winning_group[:, None]
    expected_indices = np.argsort(-np.where(eligibility, choices, -np.inf), axis=-1, kind="stable")[:, :3]
    expected_weights = np.take_along_axis(np.where(eligibility, probabilities, 0), expected_indices, axis=-1)
    if norm_topk_prob:
        expected_weights /= expected_weights.sum(axis=-1, keepdims=True) + 1e-20
    expected_weights *= 1.7
    np.testing.assert_array_equal(actual_indices, expected_indices)
    np.testing.assert_allclose(actual_weights, expected_weights, rtol=1e-5, atol=1e-7)
    assert np.isfinite(np.asarray(actual_weights)).all()
    assert np.all(np.asarray(actual_weights)[:, -1] == 0)
    gradient = jax.jit(jax.grad(lambda x: jnp.sum(selector(x)[0] ** 2)))(device_scores)
    assert np.isfinite(np.asarray(gradient)).all()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("norm_topk_prob", [False, True])
@pytest.mark.parametrize("biased", [False, True])
def test_multiple_groups_rank_by_top_two_mass_with_stable_ties(biased, norm_topk_prob, sharded, compiled):
    """Retain multiple groups by their top-two sum, not their maximum or total."""
    # Exact binary fractions avoid reference/TPU rounding differences at ties.
    # Group zero has the highest maximum, but groups one and two have more
    # top-two mass. Group three has the highest total, but smaller top-two mass.
    row = np.asarray([15, 0, 0, 0, 9, 9, 1, 0, 8, 8, 0, 0, 7, 7, 7, 7], dtype=np.float32) / 16
    probabilities = np.stack(
        [
            *(np.roll(row, 4 * shift) for shift in range(4)),
            np.tile(np.asarray([8, 4, 0, 0], dtype=np.float32) / 16, 4),
            np.zeros(16, dtype=np.float32),
            np.roll(row, 2),
            np.roll(row, 6),
        ]
    )
    bias = np.repeat(np.asarray([0, -0.125, 0.125, -0.25], dtype=np.float32), 4) if biased else None
    choices = probabilities if bias is None else probabilities + bias
    config = DeepseekV3Config(
        hidden_size=4,
        n_routed_experts=16,
        num_experts_per_tok=9,
        n_group=4,
        topk_group=2,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=1.7,
    )
    device_scores = jnp.asarray(probabilities)
    if sharded:
        device_scores = _token_sharded(config, device_scores)
    selector = _score_selector(config, bias, k=9)
    actual_weights, actual_indices = (jax.jit(selector) if compiled else selector)(device_scores)
    group_mass = np.sort(choices.reshape(8, 4, 4), axis=-1)[..., -2:].sum(axis=-1)
    winning_groups = np.argsort(-group_mass, axis=-1, kind="stable")[:, :2]
    eligible_groups = np.zeros((8, 4), dtype=bool)
    np.put_along_axis(eligible_groups, winning_groups, True, axis=-1)
    eligibility = np.repeat(eligible_groups, 4, axis=-1)
    expected_indices = np.argsort(-np.where(eligibility, choices, -np.inf), axis=-1, kind="stable")[:, :9]
    expected_weights = np.take_along_axis(np.where(eligibility, probabilities, 0), expected_indices, axis=-1)
    if norm_topk_prob:
        expected_weights /= expected_weights.sum(axis=-1, keepdims=True) + 1e-20
    expected_weights *= 1.7
    np.testing.assert_array_equal(actual_indices, expected_indices)
    np.testing.assert_allclose(actual_weights, expected_weights, rtol=1e-5, atol=1e-7)
    assert np.isfinite(np.asarray(actual_weights)).all()
    assert np.all(np.asarray(actual_weights)[:, -1] == 0)
    gradient = jax.jit(jax.grad(lambda x: jnp.sum(selector(x)[0] ** 2)))(device_scores)
    assert np.isfinite(np.asarray(gradient)).all()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("logit", [-45.0, -46.0, -50.0])
def test_small_positive_sigmoid_mass_has_correct_finite_gradients(logit, sharded, compiled):
    """Tiny positive mass needs a stable quotient derivative, not just a zero guard."""
    scale = 1.7
    config = DeepseekV3Config(
        hidden_size=4,
        n_routed_experts=4,
        num_experts_per_tok=3,
        n_group=2,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=scale,
    )
    logits = jnp.full((4, 4), logit, dtype=jnp.float32)
    if sharded:
        logits = _token_sharded(config, logits)
    route = functools.partial(DeepseekV3MoE._select_hook(config, None), pre_bias_logits=None, k=3)

    def loss(x):
        return jnp.sum(route(x)[0] ** 2)

    compute = jax.jit(route) if compiled else route
    differentiate = jax.jit(jax.grad(loss)) if compiled else jax.grad(loss)
    actual_weights, actual_indices = compute(logits)
    actual_gradient = differentiate(logits)
    # Independent float64 analytic reference, including the additive epsilon.
    # Stable ties retain group zero, experts zero and one, plus zero padding.
    probability = 1.0 / (1.0 + np.exp(-np.float64(logit)))
    denominator = 2 * probability + 1e-20
    expected_weights = np.broadcast_to([scale * probability / denominator] * 2 + [0], (4, 3))
    expected_gradient = np.zeros((4, 4), dtype=np.float64)
    expected_gradient[:, :2] = 2 * scale**2 * probability**2 * (1 - probability) * 1e-20 / denominator**3
    np.testing.assert_array_equal(actual_indices, np.broadcast_to([0, 1, 2], (4, 3)))
    np.testing.assert_allclose(actual_weights, expected_weights, rtol=1e-5, atol=1e-7)
    assert np.isfinite(np.asarray(actual_gradient)).all()
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-7)

    # Check the JVP independently too: a unit tangent in every logit sums the
    # analytic gradient, and exercises the same rule without reverse-mode AD.
    def directional_derivative(x):
        return jax.jvp(loss, (x,), (jnp.ones_like(x),))[1]

    actual_directional = (jax.jit(directional_derivative) if compiled else directional_derivative)(logits)
    np.testing.assert_allclose(actual_directional, expected_gradient.sum(), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("norm_topk_prob", [False, True])
def test_public_dispatch_with_zero_live_bias_drops_excluded_expert_contribution(norm_topk_prob):
    """Exercise the live-bias forward path with nonuniform scores."""
    config = DeepseekV3Config(
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=4,
        n_shared_experts=None,
        n_routed_experts=4,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=3,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=1.0,
        use_expert_tensor_mode=False,
    )
    config.moe_force_xla_gmm = True
    moe = DeepseekV3MoE(
        config,
        layer_idx=0,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(1),
    )
    probabilities = np.asarray([0.9, 0.8, 0.1, 0.2], dtype=np.float32)
    logits = np.log(probabilities / (1 - probabilities))
    moe.gate.weight.value = jnp.asarray(np.broadcast_to(logits, (4, 4)).copy())
    moe.gate.e_score_correction_bias.value = jnp.zeros(4, dtype=jnp.float32)
    moe.experts.gate_up_proj.weight.value = jnp.ones_like(moe.experts.gate_up_proj.weight.value)
    coefficients = np.asarray([1, 2, 10, 20], dtype=np.float32)
    moe.experts.down_proj.weight.value = jnp.asarray(
        np.broadcast_to(coefficients[:, None, None], moe.experts.down_proj.weight.value.shape).copy()
    )
    hidden = jnp.full((4, 2, 4), 0.25, dtype=jnp.float32)
    actual, router_logits = moe(hidden)
    np.testing.assert_allclose(router_logits, np.broadcast_to(logits, router_logits.shape), rtol=1e-6, atol=1e-6)
    weights = probabilities[:2].copy()
    if norm_topk_prob:
        weights /= weights.sum()
    expected = 4 / (1 + np.exp(-1.0)) * np.dot(weights, coefficients[:2])
    np.testing.assert_allclose(actual, np.full(actual.shape, expected), rtol=2e-5, atol=2e-5)
