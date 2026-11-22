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
    # Eval runs once over the full batch without minibatch_call's 1/K averaging,
    # so the normalizer must stay 10.0 regardless of accumulation_steps.
    policy_loss, entropy, clip_fraction, expected_grad = _reference(
        logits_np, behavior, advantages, mask, algorithm, 10.0
    )
    np.testing.assert_allclose(metrics.loss, policy_loss - 0.1 * entropy, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(metrics.other_metrics["actor/policy_loss"], policy_loss, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(metrics.other_metrics["actor/entropy"], entropy, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(metrics.other_metrics["actor/clip_fraction"], clip_fraction, atol=2e-6)
    np.testing.assert_allclose(metrics.other_metrics["actor/action_tokens"], 3.0, atol=1e-6)
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
    np.testing.assert_allclose(training_metrics.other_metrics["actor/action_tokens"], 3.0, atol=1e-6)
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


class _CriticPolicy:
    def __init__(self, values, mesh):
        self.values = values
        self.mesh = mesh
        self.value_head = lambda hidden: hidden

    def __call__(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.values[input_ids[:, 0]][..., None])


@jax.tree_util.register_pytree_node_class
class _CriticState:
    def __init__(self, graphstate, step, mesh):
        self.graphstate = graphstate
        self.step = step
        self.mesh = mesh

    @property
    def model(self):
        return self.merge(self.graphstate)

    def merge(self, tree):
        return _CriticPolicy(tree, self.mesh)

    def replace(self, **updates):
        return _CriticState(
            updates.get("graphstate", self.graphstate),
            updates.get("step", self.step),
            updates.get("mesh", self.mesh),
        )

    def apply_gradients(self, grads):
        return _CriticState(self.graphstate - 0.05 * grads, self.step + 1, self.mesh)

    def tree_flatten(self):
        return (self.graphstate, self.step), self.mesh

    @classmethod
    def tree_unflatten(cls, mesh, children):
        return cls(*children, mesh)


class _PPOPolicy:
    def __init__(self, params, mesh):
        self.logits, self.values = params
        self.mesh = mesh
        self.value_head = lambda hidden: hidden

    def __call__(self, input_ids, **kwargs):
        row = input_ids[:, 0]
        return SimpleNamespace(logits=self.logits[row], last_hidden_state=self.values[row][..., None])


@jax.tree_util.register_pytree_node_class
class _PPOState:
    def __init__(self, graphstate, step, mesh):
        self.graphstate = graphstate
        self.step = step
        self.mesh = mesh

    @property
    def model(self):
        return self.merge(self.graphstate)

    def merge(self, tree):
        return _PPOPolicy(tree, self.mesh)

    def apply_gradients(self, grads):
        new_tree = jax.tree_util.tree_map(lambda p, g: p - 0.05 * g, self.graphstate, grads)
        return _PPOState(new_tree, self.step + 1, self.mesh)

    def tree_flatten(self):
        return (self.graphstate, self.step), self.mesh

    @classmethod
    def tree_unflatten(cls, mesh, children):
        return cls(*children, mesh)


def test_sao_policy_loss_extreme_log_ratio_produces_finite_loss_and_grad():
    from easydel.trainers._shared.policy_losses import sao_policy_loss

    behavior = jnp.asarray([[-120.0, 0.0, 120.0, -120.0]], dtype=jnp.float32)
    current = jnp.asarray([[0.0, 0.05, 0.0, 0.0]], dtype=jnp.float32)
    advantages = jnp.asarray([[3.0, 2.0, -4.0, 5.0]], dtype=jnp.float32)
    mask = jnp.asarray([[1.0, 1.0, 1.0, 0.0]], dtype=jnp.float32)

    def loss_fn(logps):
        return sao_policy_loss(logps, behavior, advantages, mask, epsilon_low=0.3, epsilon_high=5.0)

    loss, grad = jax.value_and_grad(loss_fn)(current)
    assert np.isfinite(float(loss))
    assert np.all(np.isfinite(np.asarray(grad)))
    # Only token 1 is inside DIS; tokens 0/2 (extreme log_ratio > 90 / < -90) and
    # token 3 (masked) must produce exact zero gradients without exp(large)*0=NaN.
    expected_grad = np.asarray([[0.0, -np.exp(0.05) * 2.0 / 3.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(grad), expected_grad, rtol=1e-5, atol=1e-6)


def test_clipped_ppo_policy_loss_extreme_log_ratio_produces_finite_loss_and_grad():
    from easydel.trainers._shared.policy_losses import clipped_ppo_policy_loss

    # Token 0: active adv > 0 with log_ratio = 100 (tests jnp.maximum VJP NaN fix).
    # Token 1: active interior token.
    # Token 2: active adv < 0 with log_ratio = 100 (unclipped extreme ratio).
    # Token 3: masked pad position with log_ratio = 100.
    # Token 4: active adv < 0 with log_ratio = -100 (clipped lower bound).
    behavior = jnp.asarray([[-100.0, 0.0, -100.0, -100.0, 100.0]], dtype=jnp.float32)
    current = jnp.asarray([[0.0, 0.05, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    advantages = jnp.asarray([[2.0, 1.5, -2.0, 10.0, -3.0]], dtype=jnp.float32)
    mask = jnp.asarray([[1.0, 1.0, 1.0, 0.0, 1.0]], dtype=jnp.float32)

    def loss_fn(logps):
        return clipped_ppo_policy_loss(logps, behavior, advantages, mask, cliprange=0.2)

    loss, grad = jax.value_and_grad(loss_fn)(current)
    assert np.isfinite(float(loss))
    assert np.all(np.isfinite(np.asarray(grad)))
    grad_np = np.asarray(grad)
    assert grad_np[0, 0] == 0.0
    assert grad_np[0, 3] == 0.0
    assert grad_np[0, 4] == 0.0
    np.testing.assert_allclose(grad_np[0, 1], -1.5 * np.exp(0.05) / 4.0, rtol=1e-5, atol=1e-6)

    # Also verify exact clipped value for adv > 0, log_ratio = 100 on its own.
    loss_pos, grad_pos = jax.value_and_grad(
        lambda logps: clipped_ppo_policy_loss(
            logps,
            jnp.asarray([[-100.0]], dtype=jnp.float32),
            jnp.asarray([[2.5]], dtype=jnp.float32),
            jnp.asarray([[1.0]], dtype=jnp.float32),
            cliprange=0.2,
        )
    )(jnp.asarray([[0.0]], dtype=jnp.float32))
    np.testing.assert_allclose(float(loss_pos), -2.5 * 1.2, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(grad_pos), [[0.0]])


@pytest.mark.parametrize("algorithm", ["sao", "ppo"])
def test_actor_step_extreme_log_ratio_finite_loss_grad_and_diagnostics(algorithm):
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("dp",))
    logits = jnp.asarray(
        [
            [[0.4, -0.2, 0.1], [0.1, 0.3, -0.2], [0.2, -0.1, 0.0], [0.0, 0.0, 0.0]],
            [[0.2, 0.1, -0.3], [-0.1, 0.4, 0.2], [0.3, 0.0, -0.2], [0.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    # Place log_ratio > 95 on active tokens (both positive and negative adv) and pad tokens.
    batch = {
        "input_ids": jnp.asarray([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=jnp.int32),
        "attention_mask": jnp.ones((2, 4), dtype=jnp.int32),
        "behavior_logps": jnp.asarray([[-100.0, -1.0, -120.0], [-100.0, -1.0, -120.0]], dtype=jnp.float32),
        "advantages": jnp.asarray([[2.0, 1.5, 50.0], [-2.0, -1.0, 50.0]], dtype=jnp.float32),
        "action_mask": jnp.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=jnp.float32),
        "num_items_in_batch": jnp.asarray(4.0, dtype=jnp.float32),
    }
    state = _State(logits, jnp.asarray(0), mesh)
    with mesh:
        updated, metrics = actor_step(
            state,
            batch,
            algorithm=algorithm,
            dis_epsilon_low=0.3,
            dis_epsilon_high=5.0,
            cliprange=0.2,
            entropy_coef=0.05,
            logprob_vocab_chunk_size=None,
            partition_spec=PartitionSpec(),
            gradient_accumulation_steps=1,
            use_global_normalizer=True,
            is_training=True,
        )
    assert np.isfinite(float(metrics.loss))
    assert np.all(np.isfinite(np.asarray(updated.graphstate)))
    for key, value in metrics.other_metrics.items():
        assert np.all(np.isfinite(np.asarray(value))), f"non-finite diagnostic {key}={value}"


def test_ppo_step_extreme_log_ratio_and_eval_nondivisible_batch():
    from easydel.trainers.proximal_policy_optimization_trainer._fn import ppo_step

    mesh = Mesh(np.asarray(jax.devices()[:1]), ("dp",))
    logits = jnp.asarray(
        [
            [[0.3, -0.1, 0.2], [0.2, 0.4, -0.1], [-0.2, 0.1, 0.3], [0.1, -0.2, 0.4]],
            [[0.1, 0.2, -0.3], [0.4, -0.1, 0.2], [0.0, 0.3, -0.2], [-0.1, 0.2, 0.1]],
        ],
        dtype=jnp.float32,
    )
    values = jnp.asarray(
        [
            [0.2, 0.5, -0.1, 0.3],
            [0.1, -0.2, 0.4, 0.0],
        ],
        dtype=jnp.float32,
    )
    # prompt_length=2 -> completion length=2 (positions 2, 3).
    # Position 0 of completion has log_ratio > 95 with adv > 0; position 1 is masked pad with log_ratio > 110.
    batch = {
        "input_ids": jnp.asarray([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=jnp.int32),
        "attention_mask": jnp.asarray([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=jnp.int32),
        "completion_mask": jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
        "old_logps": jnp.asarray([[-100.0, -120.0], [-1.0, -120.0]], dtype=jnp.float32),
        "old_values": jnp.asarray([[0.1, 50.0], [0.2, 50.0]], dtype=jnp.float32),
        "advantages": jnp.asarray([[2.0, 0.0], [1.5, 0.0]], dtype=jnp.float32),
        "returns": jnp.asarray([[1.0, 0.0], [0.5, 0.0]], dtype=jnp.float32),
    }
    state = _PPOState((logits, values), jnp.asarray(0), mesh)
    with mesh:
        updated, train_metrics = ppo_step(
            state,
            batch,
            prompt_length=2,
            cliprange=0.2,
            vf_coef=0.1,
            cliprange_value=0.2,
            entropy_coef=0.01,
            logprob_vocab_chunk_size=None,
            partition_spec=PartitionSpec(),
            gradient_accumulation_steps=1,
            is_training=True,
        )
        # Eval with gradient_accumulation_steps=4 on batch_size=2 (2 % 4 != 0) must not crash.
        eval_metrics = ppo_step(
            state,
            batch,
            prompt_length=2,
            cliprange=0.2,
            vf_coef=0.1,
            cliprange_value=0.2,
            entropy_coef=0.01,
            logprob_vocab_chunk_size=None,
            partition_spec=PartitionSpec(),
            gradient_accumulation_steps=4,
            is_training=False,
        )
    assert np.isfinite(float(train_metrics.loss))
    assert np.all(np.isfinite(np.asarray(updated.graphstate[0])))
    assert np.all(np.isfinite(np.asarray(updated.graphstate[1])))
    assert np.isfinite(float(eval_metrics.loss))
    for key, val in train_metrics.other_metrics.items():
        assert np.all(np.isfinite(np.asarray(val))), f"non-finite PPO diagnostic {key}={val}"


def test_ppo_preprocess_k3_kl_masks_extreme_pad_logps():
    from easydel.trainers.proximal_policy_optimization_trainer.ppo_trainer import PPOTrainer

    trainer = object.__new__(PPOTrainer)
    trainer.arguments = SimpleNamespace(
        mask_truncated_completions=False,
        missing_eos_penalty=None,
        kl_estimator="k3",
        kl_coef=0.1,
        whiten_rewards=True,
        whiten_advantages=True,
        gamma=1.0,
        lam=0.95,
        max_length=4,
    )
    trainer.ref_state = SimpleNamespace(graphstate=None, graphother=None)
    trainer.reward_funcs = [lambda **kwargs: [1.0, 2.0]]
    trainer.reward_processing_classes = [None]
    trainer.reward_weights = jnp.asarray([1.0], dtype=jnp.float32)
    trainer.reward_func_names = ["unit_reward"]
    trainer.train_is_conversational = False
    trainer.eval_is_conversational = False
    trainer._eos_token_id = [2]
    trainer._pad_token_id = 0
    trainer._apply_user_data_collator = lambda b: b
    trainer._extract_reward_batch_sidechannels = lambda b: {}
    trainer._purify_batch = lambda b: b
    trainer._make_attn_mask = lambda ids: (ids != 0).astype(jnp.int32)
    trainer._coerce_generation_texts = lambda primary, fallback=None: ["a", "b"]
    trainer._coerce_optional_generation_texts = lambda vals, target_len: [None] * target_len
    trainer._coerce_generation_metadata_list = lambda vals, target_len: [None] * target_len
    trainer._completion_aligned_reward_batch = lambda b, sc, **kw: b
    trainer._build_reward_call_kwargs = lambda fn, **kw: {}
    trainer._all_gather = lambda x: x
    trainer._log_training_generations_to_wandb = lambda **kw: None
    trainer.rollout = lambda **kw: SimpleNamespace(
        sequences=jnp.asarray([[1, 5, 6, 0], [1, 7, 0, 0]], dtype=jnp.int32),
        prompt_ids=jnp.asarray([[1], [1]], dtype=jnp.int32),
        prompt_mask=jnp.asarray([[1], [1]], dtype=jnp.int32),
        completion_ids=jnp.asarray([[5, 6, 0], [7, 0, 0]], dtype=jnp.int32),
        completion_prompts=["p0", "p1"],
        raw_text=["a", "b"],
        text=["a", "b"],
        reasoning=None,
        tool_calls=None,
    )
    # Pad positions have extreme log-ratio (+120); active positions have valid values (plus one +100 active outlier).
    trainer.compute_refmodel_logps = lambda *args: jnp.asarray(
        [[-0.2, 99.0, 120.0], [-0.5, 120.0, 120.0]], dtype=jnp.float32
    )
    trainer._score_rollout_policy_and_values = lambda *args: (
        jnp.asarray([[-0.3, -1.0, 0.0], [-0.6, 0.0, 0.0]], dtype=jnp.float32),
        jnp.asarray([[0.2, 0.3, 99.0], [0.1, 99.0, 99.0]], dtype=jnp.float32),
    )

    prepared, metrics = trainer._preprocess_batch_input(
        state=SimpleNamespace(),
        batch={"input_ids": jnp.asarray([[1], [1]]), "attention_mask": jnp.asarray([[1], [1]])},
        is_train=True,
    )
    assert np.isfinite(metrics["mean_kl"])
    assert np.isfinite(metrics["reward_mean"])
    for key in ("old_logps", "old_values", "advantages", "returns"):
        assert np.all(np.isfinite(np.asarray(prepared[key]))), f"non-finite {key}"
    # Pad positions must be zeroed in old_logps and old_values.
    assert float(prepared["old_logps"][0, 2]) == 0.0
    assert float(prepared["old_values"][1, 1]) == 0.0


@pytest.mark.parametrize("algorithm", ["sao", "ppo"])
def test_global_normalizer_clamps_before_dividing_by_k_and_eval_nondivisible(algorithm):
    from easydel.trainers._online_rl.actor_critic import critic_step

    mesh = Mesh(np.asarray(jax.devices()[:1]), ("dp",))
    # 4 rows, K=4 microbatches, ONLY 1 valid action token in the entire batch (global_count=1 < K=4).
    logits = jnp.asarray(
        [
            [[0.3, -0.2, 0.1], [0.2, 0.1, -0.3], [0.0, 0.0, 0.0]],
            [[0.1, 0.4, -0.2], [-0.1, 0.2, 0.3], [0.0, 0.0, 0.0]],
            [[0.5, -0.1, 0.0], [0.0, 0.2, -0.1], [0.0, 0.0, 0.0]],
            [[-0.2, 0.3, 0.1], [0.1, -0.3, 0.2], [0.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    shifted = logits[:, :-1] - jnp.max(logits[:, :-1], axis=-1, keepdims=True)
    current = (shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True)))[..., 0]
    behavior = current - jnp.log(jnp.asarray([[1.1, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=jnp.float32))
    mask = jnp.asarray([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=jnp.float32)
    batch = {
        "input_ids": jnp.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=jnp.int32),
        "attention_mask": jnp.ones((4, 3), dtype=jnp.int32),
        "behavior_logps": behavior,
        "advantages": jnp.asarray([[2.5, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=jnp.float32),
        "action_mask": mask,
        "num_items_in_batch": jnp.asarray(1.0, dtype=jnp.float32),
    }

    with mesh:
        state_k1, metrics_k1 = actor_step(
            _State(logits, jnp.asarray(0), mesh),
            batch,
            algorithm=algorithm,
            dis_epsilon_low=0.3,
            dis_epsilon_high=5.0,
            cliprange=0.2,
            entropy_coef=0.1,
            logprob_vocab_chunk_size=None,
            partition_spec=PartitionSpec(),
            gradient_accumulation_steps=1,
            use_global_normalizer=True,
            is_training=True,
        )
        state_k4, metrics_k4 = actor_step(
            _State(logits, jnp.asarray(0), mesh),
            batch,
            algorithm=algorithm,
            dis_epsilon_low=0.3,
            dis_epsilon_high=5.0,
            cliprange=0.2,
            entropy_coef=0.1,
            logprob_vocab_chunk_size=None,
            partition_spec=PartitionSpec(),
            gradient_accumulation_steps=4,
            use_global_normalizer=True,
            is_training=True,
        )
    np.testing.assert_allclose(metrics_k4.loss, metrics_k1.loss, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(state_k4.graphstate, state_k1.graphstate, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(metrics_k4.other_metrics["actor/action_tokens"], 1.0, atol=1e-6)

    # Non-divisible eval batch (3 rows with K=4) must not crash or inflate loss by 4x.
    eval_batch = {k: (v[:3] if v.ndim > 0 else v) for k, v in batch.items()}
    with mesh:
        eval_k1 = actor_step(
            _State(logits, jnp.asarray(0), mesh),
            eval_batch,
            algorithm=algorithm,
            dis_epsilon_low=0.3,
            dis_epsilon_high=5.0,
            cliprange=0.2,
            entropy_coef=0.1,
            logprob_vocab_chunk_size=None,
            partition_spec=PartitionSpec(),
            gradient_accumulation_steps=1,
            use_global_normalizer=True,
            is_training=False,
        )
        eval_k4 = actor_step(
            _State(logits, jnp.asarray(0), mesh),
            eval_batch,
            algorithm=algorithm,
            dis_epsilon_low=0.3,
            dis_epsilon_high=5.0,
            cliprange=0.2,
            entropy_coef=0.1,
            logprob_vocab_chunk_size=None,
            partition_spec=PartitionSpec(),
            gradient_accumulation_steps=4,
            use_global_normalizer=True,
            is_training=False,
        )
    np.testing.assert_allclose(eval_k4.loss, eval_k1.loss, rtol=1e-6, atol=1e-6)

    # Same global_count=1 < K=4 and non-divisible eval checks for critic_step.
    values = jnp.asarray([[0.2, 0.3, 0.0], [0.1, -0.2, 0.0], [0.4, 0.1, 0.0], [-0.3, 0.2, 0.0]], dtype=jnp.float32)
    critic_batch = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "action_mask": mask,
        "old_values": jnp.asarray([[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=jnp.float32),
        "critic_returns": jnp.asarray([[1.2, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=jnp.float32),
        "num_items_in_batch": jnp.asarray(1.0, dtype=jnp.float32),
    }
    with mesh:
        c_k1, cm_k1 = critic_step(
            _CriticState(values, jnp.asarray(0), mesh),
            critic_batch,
            cliprange_value=0.2,
            partition_spec=PartitionSpec(),
            gradient_accumulation_steps=1,
            use_global_normalizer=True,
            is_training=True,
        )
        c_k4, cm_k4 = critic_step(
            _CriticState(values, jnp.asarray(0), mesh),
            critic_batch,
            cliprange_value=0.2,
            partition_spec=PartitionSpec(),
            gradient_accumulation_steps=4,
            use_global_normalizer=True,
            is_training=True,
        )
        c_eval_k1 = critic_step(
            _CriticState(values, jnp.asarray(0), mesh),
            {k: (v[:3] if v.ndim > 0 else v) for k, v in critic_batch.items()},
            cliprange_value=0.2,
            partition_spec=PartitionSpec(),
            gradient_accumulation_steps=1,
            use_global_normalizer=True,
            is_training=False,
        )
        c_eval_k4 = critic_step(
            _CriticState(values, jnp.asarray(0), mesh),
            {k: (v[:3] if v.ndim > 0 else v) for k, v in critic_batch.items()},
            cliprange_value=0.2,
            partition_spec=PartitionSpec(),
            gradient_accumulation_steps=4,
            use_global_normalizer=True,
            is_training=False,
        )
    np.testing.assert_allclose(cm_k4.loss, cm_k1.loss, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(c_k4.graphstate, c_k1.graphstate, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cm_k4.other_metrics["critic/action_tokens"], 1.0, atol=1e-6)
    np.testing.assert_allclose(c_eval_k4.loss, c_eval_k1.loss, rtol=1e-6, atol=1e-6)


def test_empty_action_mask_sao_clip_fraction_is_zero():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("dp",))
    logits = jnp.zeros((2, 4, 3), dtype=jnp.float32)
    batch = {
        "input_ids": jnp.asarray([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=jnp.int32),
        "attention_mask": jnp.ones((2, 4), dtype=jnp.int32),
        "behavior_logps": jnp.zeros((2, 3), dtype=jnp.float32),
        "advantages": jnp.ones((2, 3), dtype=jnp.float32),
        "action_mask": jnp.zeros((2, 3), dtype=jnp.float32),
        "num_items_in_batch": jnp.asarray(0.0, dtype=jnp.float32),
    }
    with mesh:
        metrics = actor_step(
            _State(logits, jnp.asarray(0), mesh),
            batch,
            algorithm="sao",
            dis_epsilon_low=0.3,
            dis_epsilon_high=5.0,
            cliprange=0.2,
            entropy_coef=0.0,
            logprob_vocab_chunk_size=None,
            partition_spec=PartitionSpec(),
            gradient_accumulation_steps=2,
            use_global_normalizer=True,
            is_training=False,
        )
    assert float(metrics.loss) == 0.0
    assert float(metrics.other_metrics["actor/clip_fraction"]) == 0.0
    assert float(metrics.other_metrics["actor/action_tokens"]) == 0.0


@pytest.mark.parametrize("algorithm", ["sao", "ppo"])
@pytest.mark.parametrize("custom_num_items", [None, 10.0])
def test_scheduled_actor_and_critic_adapters_match_spmd_on_presliced_microbatches(algorithm, custom_num_items):
    from easydel.trainers._online_rl.actor_critic import critic_step
    from easydel.trainers.training_utils import ScheduledStepCall, get_scheduled_loss_adapter

    mesh = Mesh(np.asarray(jax.devices()[:1]), ("dp",))
    logits = jnp.asarray(
        [
            [[0.3, -0.2, 0.1], [0.2, 0.4, -0.1], [-0.1, 0.2, 0.3], [0.0, 0.0, 0.0]],
            [[0.1, 0.5, -0.2], [-0.2, 0.1, 0.4], [0.3, -0.2, 0.1], [0.0, 0.0, 0.0]],
            [[-0.3, 0.2, 0.4], [0.4, -0.1, 0.0], [0.1, 0.3, -0.2], [0.0, 0.0, 0.0]],
            [[0.2, -0.1, 0.3], [0.0, 0.2, -0.3], [-0.2, 0.4, 0.1], [0.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    shifted = logits[:, :-1] - jnp.max(logits[:, :-1], axis=-1, keepdims=True)
    current = (shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True)))[..., 0]
    behavior = current - jnp.log(
        jnp.asarray(
            [[1.1, 1.3, 0.9], [0.85, 1.05, 1.0], [1.0, 1.0, 1.0], [0.95, 1.15, 1.0]],
            dtype=jnp.float32,
        )
    )
    # Unequal action counts across 4 single-row microbatches: [1, 3, 0, 2] -> total = 6.
    mask = jnp.asarray(
        [[1, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0]],
        dtype=jnp.float32,
    )
    batch = {
        "input_ids": jnp.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]], dtype=jnp.int32),
        "attention_mask": jnp.ones((4, 4), dtype=jnp.int32),
        "behavior_logps": behavior,
        "advantages": jnp.asarray([[2.0, -1.0, 0.5], [1.5, -2.0, 0.8], [3.0, 1.0, -1.0], [-1.5, 2.5, 0.0]]),
        "action_mask": mask,
        "num_items_in_batch": jnp.asarray(6.0 if custom_num_items is None else custom_num_items, dtype=jnp.float32),
    }

    class _ReplaceableState(_State):
        def replace(self, **updates):
            return _ReplaceableState(
                updates.get("graphstate", self.graphstate),
                updates.get("step", self.step),
                updates.get("mesh", self.mesh),
            )

    state = _ReplaceableState(logits, jnp.asarray(0), mesh)
    # Reference full-batch SPMD loss and gradient.
    with mesh:
        ref_loss, ref_grad = jax.value_and_grad(
            lambda tree: actor_step(
                state.replace(graphstate=tree),
                batch,
                algorithm=algorithm,
                dis_epsilon_low=0.3,
                dis_epsilon_high=5.0,
                cliprange=0.2,
                entropy_coef=0.05,
                logprob_vocab_chunk_size=None,
                partition_spec=PartitionSpec(),
                gradient_accumulation_steps=1,
                use_global_normalizer=True,
                is_training=False,
            ).loss
        )(logits)

    # Simulate MPMD scheduled execution: K=2 outer GA * P=2 pipeline microbatches -> 1-row pre-sliced microbatches
    # where (1 % K != 0), reduced by make_microbatch_weight!
    adapter = get_scheduled_loss_adapter(actor_step)
    assert adapter is not None and adapter.make_microbatch_weight is not None
    call = ScheduledStepCall(
        step_fn=actor_step,
        state=state,
        batch=batch,
        args=(),
        kwargs={},
        bound_arguments={
            "algorithm": algorithm,
            "dis_epsilon_low": 0.3,
            "dis_epsilon_high": 5.0,
            "cliprange": 0.2,
            "entropy_coef": 0.05,
            "logprob_vocab_chunk_size": None,
            "loss_config": None,
            "learning_rate_fn": None,
            "partition_spec": PartitionSpec(),
            "gradient_accumulation_steps": 2,
            "use_global_normalizer": True,
            "straight_through_emulator": None,
        },
        schedule=None,
    )
    prepared = adapter.prepare_batch(call) if adapter.prepare_batch is not None else batch
    loss_fn = adapter.make_loss(call)
    weight_fn = adapter.make_microbatch_weight(call)
    assert weight_fn is not None

    with mesh:
        microbatches = [{k: (v[i : i + 1] if v.ndim > 0 else v) for k, v in prepared.items()} for i in range(4)]
        weights = [float(weight_fn(logits, mb)) for mb in microbatches]
        total_w = sum(weights)
        sched_loss = 0.0
        sched_grad = jnp.zeros_like(logits)
        for mb, w in zip(microbatches, weights, strict=True):
            l_i, g_i = jax.value_and_grad(loss_fn)(logits, mb)
            assert np.isfinite(float(l_i))
            assert np.all(np.isfinite(np.asarray(g_i)))
            sched_loss = sched_loss + (w / total_w) * float(l_i)
            sched_grad = sched_grad + (w / total_w) * g_i
    np.testing.assert_allclose(sched_loss, float(ref_loss), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(sched_grad), np.asarray(ref_grad), rtol=1e-5, atol=1e-6)

    # Same scheduled adapter check for critic_step on 1-row pre-sliced microbatches (1 % K != 0).
    values = jnp.asarray(
        [[0.2, 0.1, -0.1, 0.0], [0.3, -0.2, 0.4, 0.0], [0.1, 0.2, 0.3, 0.0], [-0.1, 0.5, 0.2, 0.0]],
        dtype=jnp.float32,
    )
    critic_batch = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "action_mask": mask,
        "old_values": jnp.zeros((4, 3), dtype=jnp.float32),
        "critic_returns": jnp.asarray([[1.0, 0.0, 0.0], [0.5, -0.5, 1.5], [0.0, 0.0, 0.0], [2.0, -1.0, 0.0]]),
        "num_items_in_batch": batch["num_items_in_batch"],
    }
    c_state = _CriticState(values, jnp.asarray(0), mesh)
    with mesh:
        c_ref_loss, c_ref_grad = jax.value_and_grad(
            lambda tree: critic_step(
                c_state.replace(graphstate=tree),
                critic_batch,
                cliprange_value=0.2,
                partition_spec=PartitionSpec(),
                gradient_accumulation_steps=1,
                use_global_normalizer=True,
                is_training=False,
            ).loss
        )(values)
    c_adapter = get_scheduled_loss_adapter(critic_step)
    assert c_adapter is not None and c_adapter.make_microbatch_weight is not None
    c_call = ScheduledStepCall(
        step_fn=critic_step,
        state=c_state,
        batch=critic_batch,
        args=(),
        kwargs={},
        bound_arguments={
            "cliprange_value": 0.2,
            "loss_config": None,
            "learning_rate_fn": None,
            "partition_spec": PartitionSpec(),
            "gradient_accumulation_steps": 2,
            "use_global_normalizer": True,
        },
        schedule=None,
    )
    c_prepared = c_adapter.prepare_batch(c_call) if c_adapter.prepare_batch is not None else critic_batch
    c_loss_fn = c_adapter.make_loss(c_call)
    c_weight_fn = c_adapter.make_microbatch_weight(c_call)
    assert c_weight_fn is not None
    with mesh:
        c_mbs = [{k: (v[i : i + 1] if v.ndim > 0 else v) for k, v in c_prepared.items()} for i in range(4)]
        c_weights = [float(c_weight_fn(values, mb)) for mb in c_mbs]
        c_total_w = sum(c_weights)
        c_sched_loss = 0.0
        c_sched_grad = jnp.zeros_like(values)
        for mb, w in zip(c_mbs, c_weights, strict=True):
            l_i, g_i = jax.value_and_grad(c_loss_fn)(values, mb)
            c_sched_loss = c_sched_loss + (w / c_total_w) * float(l_i)
            c_sched_grad = c_sched_grad + (w / c_total_w) * g_i
    np.testing.assert_allclose(c_sched_loss, float(c_ref_loss), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(c_sched_grad), np.asarray(c_ref_grad), rtol=1e-5, atol=1e-6)
