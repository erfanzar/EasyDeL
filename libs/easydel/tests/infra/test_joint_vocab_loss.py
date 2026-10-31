"""Joint-vocabulary chunked loss/gradient parity (accelerator-only; not run on CPU)."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.infra.loss_utils import CausalLMLossStrategy, LossConfig
from easydel.modules.llama.llama_configuration import VisionLlamaConfig
from jax import numpy as jnp


@pytest.mark.parametrize("token_chunk_size", [1, 3])
@pytest.mark.parametrize("smoothing,z_loss", [(0.0, 0.0), (0.1, 0.001)])
def test_joint_vocab_chunked_loss_and_gradient_match_independent_dense_reference(token_chunk_size, smoothing, z_loss):
    rng = np.random.default_rng(41)
    hidden = rng.normal(size=(2, 5, 3)).astype(np.float32)
    kernel = rng.normal(size=(3, 24)).astype(np.float32)
    # Both namespaces occur as targets; ignore_index also exercises masked chunks.
    labels = np.asarray([[0, 16, -100, 23, 2], [1, 22, 4, 19, 5]], dtype=np.int32)
    config = VisionLlamaConfig(vocab_size=16, vision_vocab_size=8)

    class JointProjection:
        def __init__(self):
            self.config = config
            self.vocab_size = config.vocab_size

        def forward(self, input_ids, apply_lm_head=True):
            raise AssertionError("Only the public loss strategy is exercised")

        def compute_lm_logits(self, hidden_states):
            return jnp.matmul(hidden_states, jnp.asarray(kernel), precision=jax.lax.Precision.HIGHEST)

    module = JointProjection()
    loss_config = LossConfig(
        chunk_token_size=token_chunk_size,
        label_smoothing=smoothing,
        z_loss=z_loss,
        compute_dtype="fp32",
        cross_entropy_platform="xla",
    )
    strategy = CausalLMLossStrategy()
    label_array = jnp.asarray(labels)
    plan = strategy.plan_forward(module=module, labels=label_array, loss_config=loss_config, batch={}, loss_kwargs={})
    assert plan.forward_kwargs == {"apply_lm_head": False}

    def evaluate(states):
        return strategy.compute(
            module=module,
            outputs=SimpleNamespace(last_hidden_state=states),
            labels=label_array,
            loss_config=loss_config,
            batch={},
            loss_kwargs={},
            paxis=None,
            forward_plan=plan,
        )

    metrics = evaluate(jnp.asarray(hidden))
    gradient = jax.grad(lambda states: evaluate(states).loss)(jnp.asarray(hidden))

    logits = hidden[:, :-1].astype(np.float64) @ kernel.astype(np.float64)
    targets = labels[:, 1:]
    valid = targets != -100
    shifted = logits - logits.max(-1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(-1)) + logits.max(-1)
    log_probs = logits - log_z[..., None]
    probabilities = np.exp(log_probs)
    target_distribution = np.full_like(logits, smoothing / 23)
    np.put_along_axis(target_distribution, np.maximum(targets, 0)[..., None], 1 - smoothing, axis=-1)
    log_targets = np.zeros_like(target_distribution)
    np.log(target_distribution, out=log_targets, where=target_distribution > 0)
    per_token = np.sum(target_distribution * (log_targets - log_probs), axis=-1)
    expected_z = np.sum(z_loss * log_z**2 * valid) / valid.sum()
    expected_loss = np.sum(per_token * valid) / valid.sum() + expected_z
    logits_gradient = (
        (probabilities - target_distribution + 2 * z_loss * log_z[..., None] * probabilities)
        * valid[..., None]
        / valid.sum()
    )
    expected_gradient = np.zeros_like(hidden)
    expected_gradient[:, :-1] = logits_gradient @ kernel.T
    expected_accuracy = np.sum((logits.argmax(-1) == targets) * valid) / valid.sum()

    np.testing.assert_allclose(metrics.loss, expected_loss, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(metrics.z_loss, expected_z, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(metrics.accuracy, expected_accuracy, atol=1e-6)
    np.testing.assert_allclose(metrics.weight_sum, valid.sum())
    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-5, atol=3e-6)
