# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Discard protocol: a ``None`` reward from a discarding reward function drops its whole prompt group."""

import inspect
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.trainers.base_trainer import GenerationResults
from easydel.trainers.group_relative_policy_optimization._fn import grpo_step
from easydel.trainers.group_relative_policy_optimization.grpo_config import GRPOConfig
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import (
    GRPOTrainer,
    _apply_discarded_groups,
    _compute_rewards_and_advantages,
    _unscorable_group_rows,
)
from easydel.trainers.reward_protocol import RewardProtocol
from easydel.trainers.training_utils import ScheduledStepCall, get_scheduled_loss_adapter
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec

EPS = 1e-4
VOCAB = 5


def _advantages(rewards_per_func, discard, scale_rewards="group"):
    return _compute_rewards_and_advantages(
        rewards_per_func=jnp.asarray(rewards_per_func, dtype=jnp.float32),
        reward_weights=jnp.ones((np.asarray(rewards_per_func).shape[1],), dtype=jnp.float32),
        generation_factor=2,
        scale_rewards=scale_rewards,
        multi_objective_aggregation="sum_then_normalize",
        arguments=GRPOConfig(discard_unscorable_groups=discard),
    )


def test_unscorable_sample_in_any_function_discards_its_group():
    # Group 0 is fully scored; group 1 has one unscorable sample in the second function only.
    rewards = [[1.0, 0.0], [3.0, 0.0], [2.0, np.nan], [6.0, 1.0]]
    np.testing.assert_array_equal(np.asarray(_unscorable_group_rows(jnp.asarray(rewards), 2)), [0, 0, 1, 1])
    # Restricting to the first column ignores the second column's NaN.
    np.testing.assert_array_equal(
        np.asarray(_unscorable_group_rows(jnp.asarray(rewards), 2, np.array([True, False]))), [0, 0, 0, 0]
    )

    totals, advantages, std, _ = _advantages(rewards, discard=True)
    # Group 0: rewards [1, 3], mean 2, std 1. Group 1 contributes nothing.
    np.testing.assert_allclose(np.asarray(advantages), [-1 / (1 + EPS), 1 / (1 + EPS), 0.0, 0.0], rtol=1e-5)
    assert np.isnan(np.asarray(totals)[2:]).all()
    assert float(jnp.nanmean(totals)) == pytest.approx(2.0)
    assert float(jnp.nanmean(std)) == pytest.approx(1.0)


def test_batch_statistics_ignore_discarded_groups():
    rewards = [[1.0], [3.0], [np.nan], [100.0]]
    _, advantages, std, _ = _advantages(rewards, discard=True, scale_rewards="batch")
    # Batch std over the surviving rewards [1, 3] only (=1), not polluted by the 100.
    np.testing.assert_allclose(np.asarray(advantages), [-1 / (1 + EPS), 1 / (1 + EPS), 0.0, 0.0], rtol=1e-5)
    np.testing.assert_allclose(np.asarray(std), np.ones(4), rtol=1e-6)


NAN_AWARE_ADVANTAGES = [-1 / (1 + EPS), 1 / (1 + EPS), -2.5 / (2.5 + EPS), 2.5 / (2.5 + EPS)]
"""nansum turns the unscored reward into 0, so group 1 is [0, 5]: mean 2.5, std 2.5."""


@pytest.mark.parametrize("discard", [False, None])
def test_non_discarding_policies_keep_nan_aware_aggregation(discard):
    _, advantages, _, _ = _advantages([[1.0], [3.0], [np.nan], [5.0]], discard=discard)
    np.testing.assert_allclose(np.asarray(advantages), NAN_AWARE_ADVANTAGES, rtol=1e-5)


def test_apply_discarded_groups_masks_tokens_counts_and_weights():
    model_batch = {
        "completion_mask": jnp.ones((4, 3), dtype=jnp.int32),
        "advantages": jnp.asarray([0.5, -0.5, 1.0, -1.0]),
        "num_items_in_batch": jnp.asarray(12),
    }
    out, metrics = _apply_discarded_groups(model_batch, jnp.asarray([False, False, True, True]), 2)
    np.testing.assert_array_equal(np.asarray(out["completion_mask"]).sum(-1), [3, 3, 0, 0])
    np.testing.assert_array_equal(np.asarray(out["advantages"]), [0.5, -0.5, 0.0, 0.0])
    np.testing.assert_array_equal(np.asarray(out["difficulty_weights"]), [1.0, 1.0, 0.0, 0.0])
    assert int(out["num_items_in_batch"]) == 6
    assert metrics == {"rewards/discarded_groups": 1, "rewards/discarded_group_fraction": 0.5}

    # Existing weights are kept for surviving rows.
    weighted = dict(model_batch, difficulty_weights=jnp.asarray([2.0, 0.5, 3.0, 4.0]))
    out, _ = _apply_discarded_groups(weighted, jnp.asarray([True, True, False, False]), 2)
    np.testing.assert_array_equal(np.asarray(out["difficulty_weights"]), [0.0, 0.0, 3.0, 4.0])

    # Nothing discarded: the batch structure is unchanged (no weights invented).
    out, metrics = _apply_discarded_groups(model_batch, jnp.zeros((4,), dtype=bool), 2)
    assert "difficulty_weights" not in out
    assert metrics["rewards/discarded_groups"] == 0


def _stub_trainer(reward_values, discard, marked=False):
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = GRPOConfig(
        num_generations=2,
        max_prompt_length=1,
        max_completion_length=2,
        beta=0.0,
        esurge_importance_sampling_correction=False,
        scale_rewards="group",
        discard_unscorable_groups=discard,
    )
    trainer.steps_per_generation = 1
    trainer.num_iterations = 1
    trainer._buffered_grpo_batch = None
    trainer._buffered_grpo_remaining = 0
    trainer.ref_state = None
    trainer.train_is_conversational = False
    trainer.eval_is_conversational = False
    trainer.scale_rewards = "group"
    trainer.multi_objective_aggregation = "sum_then_normalize"
    trainer.reward_weights = jnp.ones((1,))

    def reward(**kwargs):
        return list(reward_values)

    if marked:
        reward.discard_unscorable = True
    trainer.reward_funcs = [reward]
    trainer.reward_processing_classes = [None]
    trainer.reward_func_names = ["reward"]
    trainer._pad_token_id = 0
    trainer._apply_user_data_collator = lambda batch: batch
    trainer._extract_reward_batch_sidechannels = lambda batch: {}
    trainer._purify_batch = lambda batch: batch
    trainer._all_gather = lambda value: value
    trainer._make_attn_mask = lambda ids: (ids != 0).astype(jnp.int32)
    trainer._tool_call_loop_enabled = lambda: False
    trainer._run_environment_feedback = lambda **kwargs: None
    trainer._build_reward_call_kwargs = lambda func, **kwargs: kwargs
    trainer._maybe_log_grpo_completions = lambda **kwargs: None
    trainer._log_training_generations_to_wandb = lambda **kwargs: None
    prompt_ids = jnp.asarray([[1], [2]])
    ids = jnp.asarray([[5, 6], [7, 0], [8, 9], [3, 4]])
    texts = [f"c{i}" for i in range(4)]
    trainer.rollout = lambda **kwargs: GenerationResults(
        generation_results=texts,
        prompt_ids=prompt_ids,
        prompt_mask=jnp.ones_like(prompt_ids),
        sequences=jnp.concatenate([prompt_ids.repeat(2, 0), ids], axis=1),
        completion_ids=ids,
        completion_mask=(ids != 0).astype(jnp.int32),
        decoded_prompts=["p0", "p1"],
        completion_prompts=["p0", "p0", "p1", "p1"],
        text=texts,
        raw_text=texts,
    )
    state = SimpleNamespace(model=lambda **kwargs: None, step=jnp.asarray(1))
    batch = {"input_ids": prompt_ids, "attention_mask": jnp.ones_like(prompt_ids)}
    return trainer, state, batch


@pytest.mark.parametrize(("discard", "marked"), [(None, True), (True, False), (True, True)])
def test_trainer_drops_group_with_unscorable_sample_from_the_loss(discard, marked):
    trainer, state, batch = _stub_trainer([1.0, 3.0, None, 5.0], discard=discard, marked=marked)
    model_batch, metrics = trainer._preprocess_batch_input(state, batch, True)

    np.testing.assert_array_equal(np.asarray(model_batch["completion_mask"]), [[1, 1], [1, 0], [0, 0], [0, 0]])
    np.testing.assert_array_equal(np.asarray(model_batch["difficulty_weights"]), [1.0, 1.0, 0.0, 0.0])
    assert int(model_batch["num_items_in_batch"]) == 3
    np.testing.assert_allclose(
        np.asarray(model_batch["advantages"]), [-1 / (1 + EPS), 1 / (1 + EPS), 0.0, 0.0], rtol=1e-5
    )
    assert metrics["rewards/discarded_groups"] == 1
    assert metrics["rewards/discarded_group_fraction"] == pytest.approx(0.5)
    assert metrics["reward_mean"] == pytest.approx(2.0)


@pytest.mark.parametrize(("discard", "marked"), [(None, False), (False, False), (False, True)])
def test_trainer_keeps_nan_aware_result_for_non_discarding_reward(discard, marked):
    # A plain reward function's None (default policy) or any None under discard=False keeps the
    # pre-existing NaN-aware aggregation: no rows masked, group 1 scored as [0, 5].
    trainer, state, batch = _stub_trainer([1.0, 3.0, None, 5.0], discard=discard, marked=marked)
    model_batch, metrics = trainer._preprocess_batch_input(state, batch, True)

    np.testing.assert_array_equal(np.asarray(model_batch["completion_mask"]), [[1, 1], [1, 0], [1, 1], [1, 1]])
    assert int(model_batch["num_items_in_batch"]) == 7
    assert "difficulty_weights" not in model_batch
    np.testing.assert_allclose(np.asarray(model_batch["advantages"]), NAN_AWARE_ADVANTAGES, rtol=1e-5)
    assert "rewards/discarded_groups" not in metrics
    assert metrics["reward_mean"] == pytest.approx(9.0 / 4.0)


def test_trainer_without_unscorable_samples_discards_nothing():
    trainer, state, batch = _stub_trainer([1.0, 3.0, 2.0, 5.0], discard=True)
    model_batch, metrics = trainer._preprocess_batch_input(state, batch, True)
    assert int(model_batch["num_items_in_batch"]) == 7
    assert metrics["rewards/discarded_groups"] == 0
    assert "difficulty_weights" not in model_batch


@pytest.mark.cpu_ok
def test_reward_protocol_declares_discard_unscorable():
    class Plain(RewardProtocol):
        def compute(self, **kwargs):
            return 1.0

    class Judge(RewardProtocol):
        discard_unscorable = True

        def compute(self, **kwargs):
            return None

    assert Plain().discard_unscorable is False
    assert Judge().discard_unscorable is True
    trainer = object.__new__(GRPOTrainer)
    trainer.reward_funcs = [Plain(), Judge()]
    trainer.arguments = GRPOConfig()
    np.testing.assert_array_equal(trainer._discard_columns(2), [False, True])
    # An environment-feedback column (beyond the reward functions) opts in under the default policy.
    np.testing.assert_array_equal(trainer._discard_columns(3), [False, True, True])
    trainer.reward_funcs = [Plain()]
    assert trainer._discard_columns(1) is None
    trainer.arguments = GRPOConfig(discard_unscorable_groups=True)
    np.testing.assert_array_equal(trainer._discard_columns(1), [True])
    trainer.reward_funcs = [Judge()]
    trainer.arguments = GRPOConfig(discard_unscorable_groups=False)
    assert trainer._discard_columns(1) is None


# --- loss normalisers exclude discarded groups ------------------------------------------------


def _stub_logits(input_ids, bias, xp):
    tilt = 0.1 * input_ids[:, :, None].astype(xp.float32) * xp.arange(VOCAB, dtype=xp.float32)[None, None, :]
    return bias[None, : input_ids.shape[1], :] + tilt


class _Model:
    def __init__(self, bias, mesh):
        self.bias = bias
        self.mesh = mesh

    def __call__(self, input_ids, **kwargs):
        return SimpleNamespace(logits=_stub_logits(input_ids, self.bias, jnp))


class _State:
    def __init__(self, graphstate, mesh):
        self.mesh = mesh
        self.model = _Model(graphstate, mesh)
        self.graphstate = graphstate
        self.graphdef = None
        self.graphother = None

    def replace(self, **kwargs):
        return _State(kwargs.get("graphstate", self.graphstate), self.mesh)

    def merge(self, tree):
        return _Model(tree, self.mesh)


def _loss_fn(state, batch, **kwargs):
    bound = inspect.signature(grpo_step).bind(
        state, batch, 2, 0.0, is_training=False, partition_spec=PartitionSpec(), **kwargs
    )
    bound.apply_defaults()
    call = ScheduledStepCall(
        step_fn=grpo_step, state=state, batch=batch, args=(), kwargs={}, bound_arguments=bound.arguments, schedule=None
    )
    return get_scheduled_loss_adapter(grpo_step).make_loss(call)


def _two_group_batch():
    rng = np.random.default_rng(3)
    prompt_ids = np.array([[1], [3]], dtype=np.int32)
    completion_ids = np.array([[0, 2, 4], [1, 1, 3], [4, 0, 2], [3, 2, 1]], dtype=np.int32)
    mask = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=np.float32)
    old = rng.uniform(-2.5, -0.5, size=mask.shape).astype(np.float32)
    batch = {
        "prompt_ids": jnp.asarray(prompt_ids),
        "prompt_mask": jnp.ones_like(jnp.asarray(prompt_ids)),
        "completion_ids": jnp.asarray(completion_ids),
        "completion_mask": jnp.asarray(mask),
        "advantages": jnp.asarray([1.5, -0.7, 0.4, -1.2], dtype=jnp.float32),
        "old_per_token_logps": jnp.asarray(old),
        "num_items_in_batch": jnp.asarray(mask.sum()),
    }
    bias = jnp.asarray(rng.normal(size=(4, VOCAB)).astype(np.float32))
    return batch, bias


@pytest.mark.parametrize("chunk_size", [None, 1])
@pytest.mark.parametrize("loss_type", ["grpo", "gmpo", "dr_grpo", "luspo"])
def test_discarded_group_loss_and_gradient_equal_kept_groups_alone(loss_type, chunk_size):
    batch, bias = _two_group_batch()
    mesh = Mesh(np.array(jax.devices()[:1]), ("dp",))
    state = _State(bias, mesh)
    discarded, _ = _apply_discarded_groups(batch, jnp.asarray([False, False, True, True]), 2)
    kept = {
        key: value[:1] if key.startswith("prompt") else value[:2]
        for key, value in batch.items()
        if key != "num_items_in_batch"
    }
    kept["num_items_in_batch"] = jnp.sum(kept["completion_mask"])

    kwargs = {"loss_type": loss_type, "epsilon": 0.3, "epsilon_high": 0.3, "completion_chunk_size": chunk_size}
    loss_d, grad_d = jax.value_and_grad(_loss_fn(state, discarded, **kwargs))(bias, discarded)
    loss_k, grad_k = jax.value_and_grad(_loss_fn(state, kept, **kwargs))(bias, kept)

    assert abs(float(loss_k)) > 1e-3
    np.testing.assert_allclose(float(loss_d), float(loss_k), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(grad_d), np.asarray(grad_k), rtol=1e-5, atol=1e-7)
