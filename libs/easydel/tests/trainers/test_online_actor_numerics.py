"""Accelerator-only numerical coverage of the actual online actor objective."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.trainers._online_rl.actor_critic import actor_step
from easydel.trainers._shared import concatenated_inputs
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec


class _Policy:
    def __init__(self, logits, mesh):
        self.logits = logits
        self.mesh = mesh

    def __call__(self, input_ids, **kwargs):
        # The first prompt token identifies each trajectory, including after
        # the real minibatch loop slices the rows for accumulation.
        return SimpleNamespace(logits=self.logits[input_ids[:, 0]])


@jax.tree_util.register_pytree_node_class
class _State:
    def __init__(self, graphstate, step, mesh):
        self.graphstate = graphstate
        self.step = step
        self.mesh = mesh

    @property
    def model(self):
        return self.merge(self.graphstate)

    def merge(self, tree):
        return _Policy(tree, self.mesh)

    def apply_gradients(self, grads):
        return _State(self.graphstate - 0.05 * grads, self.step + 1, self.mesh)

    def tree_flatten(self):
        return (self.graphstate, self.step), self.mesh

    @classmethod
    def tree_unflatten(cls, mesh, children):
        return cls(*children, mesh)


def _reference(logits, behavior, advantages, mask, algorithm, normalizer):
    shifted = logits[:, :-1] - np.max(logits[:, :-1], axis=-1, keepdims=True)
    logps_all = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    logps = logps_all[..., 0]
    entropy = -np.sum(np.exp(logps_all) * logps_all, axis=-1)
    ratio = np.exp(logps - behavior)
    if algorithm == "ppo":
        raw = -advantages * ratio
        clipped = -advantages * np.clip(ratio, 0.8, 1.2)
        policy_loss = np.sum(np.maximum(raw, clipped) * mask) / normalizer
        clip_fraction = np.sum((clipped > raw) * mask) / mask.sum()
        keep = mask
        score_coeff = -advantages * ratio * (raw >= clipped) * mask / normalizer
    else:
        keep = ((logps - behavior > np.log1p(-0.2)) & (logps - behavior < np.log1p(0.2))) * mask
        policy_loss = -np.sum(ratio * keep * advantages * logps * mask) / normalizer
        clip_fraction = 1 - np.sum(keep * mask) / mask.sum()
        # SAO's ratio is deliberately detached: do not differentiate the
        # reference's numerical loss with ordinary finite differences.
        score_coeff = -ratio * keep * advantages * mask / normalizer
    entropy_mean = np.sum(entropy * mask) / normalizer
    probabilities = np.exp(logps_all)
    target_derivative = -probabilities
    target_derivative[..., 0] += 1
    gradient = score_coeff[..., None] * target_derivative
    entropy_derivative = -probabilities * (logps_all + entropy[..., None])
    gradient -= 0.1 * entropy_derivative * mask[..., None] / normalizer
    return policy_loss, entropy_mean, clip_fraction, gradient


@pytest.mark.parametrize("algorithm", ["ppo", "sao"])
@pytest.mark.parametrize("accumulation_steps", [1, 2])
def test_real_actor_loss_metrics_and_gradients(algorithm, accumulation_steps):
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("dp",))
    logits_np = np.asarray(
        [
            [[0.3, -0.2, 0.1], [0.1, 0.5, -0.1], [-0.2, 0.1, 0.4], [0.0, 0.0, 0.0]],
            [[0.7, 0.1, -0.1], [-0.3, 0.2, 0.4], [0.0, 0.1, 0.2], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    shifted = logits_np[:, :-1] - logits_np[:, :-1].max(axis=-1, keepdims=True)
    current_np = (shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True)))[..., 0]
    # Retain ratios both inside and outside the DIS/PPO bounds, on each side.
    ratios = np.asarray([[1.1, 1.4, 0.6], [0.9, 0.7, 1.3]], dtype=np.float32)
    behavior = current_np - np.log(ratios)
    advantages = np.asarray([[2.0, 3.0, 100.0], [-1.0, -2.0, 100.0]], dtype=np.float32)
    mask = np.asarray([[1, 1, 0], [1, 0, 0]], dtype=np.float32)
    batch = {
        "input_ids": jnp.asarray([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=jnp.int32),
        "attention_mask": jnp.ones((2, 4), dtype=jnp.int32),
        "behavior_logps": jnp.asarray(behavior),
        "advantages": jnp.asarray(advantages),
        "action_mask": jnp.asarray(mask),
        "num_items_in_batch": jnp.asarray(10.0),
    }

    def evaluate(logits, batch=batch, is_training=False):
        state = _State(logits, jnp.asarray(0), mesh)
        with mesh:
            return actor_step(
                state,
                batch,
                algorithm=algorithm,
                dis_epsilon_low=0.2,
                dis_epsilon_high=0.2,
                cliprange=0.2,
                entropy_coef=0.1,
                logprob_vocab_chunk_size=None,
                partition_spec=PartitionSpec(),
                gradient_accumulation_steps=accumulation_steps,
                use_global_normalizer=True,
                is_training=is_training,
            )

    logits = jnp.asarray(logits_np)
    metrics = evaluate(logits)
    policy_loss, entropy, clip_fraction, expected_grad = _reference(
        logits_np, behavior, advantages, mask, algorithm, 10.0 / accumulation_steps
    )
    np.testing.assert_allclose(metrics.loss, policy_loss - 0.1 * entropy, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(metrics.other_metrics["actor/policy_loss"], policy_loss, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(metrics.other_metrics["actor/entropy"], entropy, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(metrics.other_metrics["actor/clip_fraction"], clip_fraction, atol=2e-6)
    actual_grad = jax.grad(lambda values: evaluate(values).loss)(logits)
    np.testing.assert_allclose(actual_grad[:, :-1], expected_grad, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(actual_grad[:, -1], 0)
    for name in ("advantages", "behavior_logps"):
        detached_grad = jax.grad(lambda values, name=name: evaluate(logits, {**batch, name: values}).loss)(batch[name])
        np.testing.assert_array_equal(detached_grad, 0)

    # Exercise the real optimizer/update path, not just a standalone public
    # surrogate. A global denominator must yield the same SGD update with
    # one or two accumulation microbatches, including unequal token counts.
    updated, training_metrics = evaluate(logits, is_training=True)
    full_policy, full_entropy, _, full_gradient = _reference(logits_np, behavior, advantages, mask, algorithm, 10.0)
    expected_updated = logits_np.copy()
    expected_updated[:, :-1] -= 0.05 * full_gradient
    np.testing.assert_allclose(updated.graphstate, expected_updated, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(training_metrics.loss, full_policy - 0.1 * full_entropy, rtol=2e-5, atol=2e-6)
    assert int(updated.step) == 1


def test_shared_preference_inputs_pad_pairs_and_duplicate_multimodal_inputs():
    batch = {
        "prompt_input_ids": jnp.asarray([[7, 8]]),
        "prompt_attention_mask": jnp.asarray([[1, 1]]),
        "chosen_input_ids": jnp.asarray([[3, 4]]),
        "chosen_attention_mask": jnp.asarray([[1, 1]]),
        "rejected_input_ids": jnp.asarray([[5]]),
        "rejected_attention_mask": jnp.asarray([[1]]),
        "pixel_values": jnp.asarray([[[1.0, 2.0]]]),
        "pixel_attention_mask": jnp.asarray([[1]]),
        "image_sizes": jnp.asarray([[2, 3]]),
    }
    result = concatenated_inputs(batch, padding_value=99)
    np.testing.assert_array_equal(result["completion_input_ids"], [[3, 4], [5, 99]])
    np.testing.assert_array_equal(result["completion_attention_mask"], [[1, 1], [1, 0]])
    for key in ("prompt_input_ids", "prompt_attention_mask", "pixel_values", "pixel_attention_mask", "image_sizes"):
        np.testing.assert_array_equal(result[key], np.concatenate([np.asarray(batch[key])] * 2, axis=0))
