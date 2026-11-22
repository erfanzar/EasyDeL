"""DeepSeek routing and sequence objectives against independent numerical references (accelerator only)."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.deepseek_v2.deepseek_configuration import DeepseekV2Config
from easydel.modules.deepseek_v2.modeling_deepseek import DeepseekV2ForCausalLM
from easydel.modules.deepseek_v2.modeling_deepseek import MoEGate as V2Gate
from easydel.modules.deepseek_v3.deepseek_configuration import DeepseekV3Config
from easydel.modules.deepseek_v3.modeling_deepseek import DeepseekV3MoE
from jax import numpy as jnp


def test_v2_square_gate_uses_expert_major_checkpoint_layout():
    config = DeepseekV2Config(hidden_size=4, n_routed_experts=4, num_experts_per_tok=1)
    gate = V2Gate(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    weight = np.asarray([[1, 0, 0, 0], [2, 3, 0, 0], [0, 0, 4, 0], [0, 0, 0, 5]], dtype=np.float32)
    gate.weight.value = jnp.asarray(weight)
    hidden = np.eye(4, dtype=np.float32)
    actual = gate(jnp.asarray(hidden))
    np.testing.assert_allclose(actual, hidden @ weight.T, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(actual).argmax(axis=-1), [1, 1, 2, 3])


@pytest.mark.parametrize("top_k", [1, 2, 3])
@pytest.mark.parametrize("winning_group", [0, 1])
def test_v3_negative_correction_scores_dispatch_only_to_winning_group(top_k, winning_group):
    config = DeepseekV3Config(
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=4,
        n_shared_experts=None,
        n_routed_experts=4,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=top_k,
        norm_topk_prob=True,
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
    moe.gate.weight.value = jnp.zeros_like(moe.gate.weight.value)
    bias = np.asarray([-0.6, -0.7, -1.3, -1.4], dtype=np.float32)
    if winning_group:
        bias = np.roll(bias, 2)
    moe.gate.e_score_correction_bias.value = jnp.asarray(bias)
    moe.experts.gate_up_proj.weight.value = jnp.ones_like(moe.experts.gate_up_proj.weight.value)
    coefficients = np.arange(1, 5, dtype=np.float32)
    moe.experts.down_proj.weight.value = jnp.asarray(
        np.broadcast_to(coefficients[:, None, None], moe.experts.down_proj.weight.value.shape).copy()
    )
    hidden = jnp.full((4, 2, 4), 0.25, dtype=jnp.float32)
    actual, router_logits = moe(hidden)
    # The router returns raw logits; the correction bias enters selection only.
    np.testing.assert_array_equal(router_logits, np.zeros(router_logits.shape, np.float32))
    # Each expert emits 4*silu(1)*its_coefficient. Selection-only biases
    # must not change the combine weights: sigmoid(0)=0.5 for each expert.
    selected = np.arange(2 * winning_group, 2 * winning_group + min(top_k, 2))
    expected = 4 / (1 + np.exp(-1.0)) * 0.5 * coefficients[selected].sum()
    np.testing.assert_allclose(actual, np.full(actual.shape, expected), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("seq_aux", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_v2_opposite_sequence_collapse_does_not_cancel_with_seq_aux(seq_aux, masked):
    config = DeepseekV2Config(
        hidden_size=4,
        n_routed_experts=2,
        num_experts_per_tok=1,
        topk_method="greedy",
        aux_loss_alpha=1.0,
        seq_aux=seq_aux,
    )
    # Test the numerical objective with controlled router logits, independently
    # of transformer activations. The public-forward tests cover integration.
    model = object.__new__(DeepseekV2ForCausalLM)
    object.__setattr__(model, "config", config)
    logits = np.asarray([[[10, 0], [10, 0], [0, 20]], [[0, 10], [0, 10], [20, 0]]] * 2, dtype=np.float32)
    mask = np.asarray([[1, 1, 0]] * 4, dtype=np.float32) if masked else np.ones((4, 3), dtype=np.float32)
    outputs = SimpleNamespace(
        last_hidden_state=jnp.zeros((4, 3, 4)), router_logits=(jnp.asarray(logits.reshape(-1, 2)),)
    )
    actual = model._compute_aux_loss(outputs, jnp.asarray(mask) if masked else None)
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    counts = np.eye(2)[logits.argmax(axis=-1)]
    if not seq_aux:
        mask = mask.reshape(1, -1)
        probabilities = probabilities.reshape(1, -1, 2)
        counts = counts.reshape(1, -1, 2)
    n = mask.sum(axis=1, keepdims=True)
    frequency = (counts * mask[..., None]).sum(axis=1) / n
    mean_probability = (probabilities * mask[..., None]).sum(axis=1) / n
    expected = 2 * (frequency * mean_probability).sum(axis=-1).mean()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    if masked:
        np.testing.assert_allclose(actual, 2 / (1 + np.exp(-10)) if seq_aux else 1, rtol=1e-6, atol=1e-6)
