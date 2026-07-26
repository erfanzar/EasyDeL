# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent math and public-surface tests for the GLM-5.2 RL trainers."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import easydel as ed
import jax
import numpy as np
import pytest
from easydel.infra.loss_utils import LossMetrics
from easydel.trainers._online_rl.actor_critic import build_critic_optimizer
from easydel.trainers._online_rl.trainer import OnlineActorCriticTrainer
from easydel.trainers._online_rl.trajectory import (
    OnlineRLTrajectory,
    OnlineRLTrajectoryCollator,
    TrajectorySegment,
)
from easydel.trainers.compaction_reinforcement_learning_trainer import (
    CompactionRLConfig,
    CompactionRLTrainer,
)
from easydel.trainers.compaction_reinforcement_learning_trainer._fn import (
    apply_cross_segment_gae_correction,
    compaction_ppo_policy_loss,
    cross_segment_advantage_factors,
)
from easydel.trainers.proximal_policy_optimization_trainer import PPOTrainer
from easydel.trainers.single_rollout_asynchronous_optimization_trainer import (
    SAOConfig,
    SAOTrainer,
)
from easydel.trainers.single_rollout_asynchronous_optimization_trainer._fn import (
    direct_importance_sampling_mask,
    sao_policy_loss,
)
from easydel.trainers.training_configurations import TrainingArguments
from easydel.trainers.training_utils import get_scheduled_loss_adapter
from easydel.utils import Registry
from jax import numpy as jnp


def test_sao_dis_mask_uses_strict_double_sided_boundaries():
    epsilon_low = 0.3
    epsilon_high = 5.0
    ratios = jnp.asarray(
        [
            1.0 - epsilon_low,
            np.nextafter(1.0 - epsilon_low, np.inf),
            1.0,
            np.nextafter(1.0 + epsilon_high, -np.inf),
            1.0 + epsilon_high,
        ],
        dtype=jnp.float32,
    )
    current = jnp.log(ratios)
    mask = direct_importance_sampling_mask(
        current,
        jnp.zeros_like(current),
        jnp.ones_like(current),
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
    )

    # Float32 rounding can erase a one-ULP NumPy float64 perturbation, so the
    # exact endpoints and clearly interior points are the load-bearing checks.
    assert mask[0] == 0
    assert mask[2] == 1
    assert mask[-1] == 0


def test_sao_loss_gradient_is_detached_importance_weight_reference():
    current = jnp.log(jnp.asarray([[1.2, 0.8]], dtype=jnp.float32))
    behavior = jnp.zeros_like(current)
    advantages = jnp.asarray([[2.0, -3.0]], dtype=jnp.float32)
    action_mask = jnp.ones_like(current)

    def loss_fn(logps):
        return sao_policy_loss(
            logps,
            behavior,
            advantages,
            action_mask,
            epsilon_low=0.5,
            epsilon_high=1.0,
        )

    gradient = jax.grad(loss_fn)(current)
    expected = -(np.exp(np.asarray(current)) * np.asarray(advantages)) / 2.0
    np.testing.assert_allclose(np.asarray(gradient), expected, rtol=1e-6, atol=1e-6)


def test_sao_rejected_token_has_zero_gradient_independent_of_advantage_sign():
    current = jnp.log(jnp.asarray([[0.2, 3.0]], dtype=jnp.float32))
    behavior = jnp.zeros_like(current)
    advantages = jnp.asarray([[4.0, -7.0]], dtype=jnp.float32)
    action_mask = jnp.ones_like(current)
    gradient = jax.grad(
        lambda logps: sao_policy_loss(
            logps,
            behavior,
            advantages,
            action_mask,
            epsilon_low=0.3,
            epsilon_high=1.0,
        )
    )(current)
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros((1, 2), dtype=np.float32))


def test_skip_observation_gae_bridges_actions_and_ignores_observation_value():
    rewards = jnp.asarray([[0.0, 0.0, 3.0]], dtype=jnp.float32)
    action_mask = jnp.asarray([[1.0, 0.0, 1.0]], dtype=jnp.float32)
    done_mask = jnp.asarray([[0.0, 0.0, 1.0]], dtype=jnp.float32)

    def compute(observation_value):
        values = jnp.asarray([[1.0, observation_value, 2.0]], dtype=jnp.float32)
        return OnlineActorCriticTrainer._skip_observation_gae(
            rewards,
            values,
            action_mask,
            done_mask,
            gamma=1.0,
            gae_lambda=1.0,
        )[0]

    expected = np.asarray([[2.0, 0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(compute(100.0)), expected)
    np.testing.assert_allclose(np.asarray(compute(-999.0)), expected)


def test_online_actor_stays_registry_constructible_and_steps_have_mpmd_adapters():
    from easydel.trainers._online_rl.actor_critic import actor_step, critic_step

    assert PPOTrainer.requires_value_head is True
    assert OnlineActorCriticTrainer.requires_value_head is False
    assert get_scheduled_loss_adapter(actor_step) is not None
    assert get_scheduled_loss_adapter(critic_step) is not None


def test_provider_recompute_scores_the_explicit_rollout_state():
    trainer = OnlineActorCriticTrainer.__new__(OnlineActorCriticTrainer)
    trainer.arguments = SimpleNamespace(
        rollout_logprob_source="recompute",
        kl_coef=0.0,
    )
    seen_graphstates = []

    def score(graphstate, graphother, input_ids, attention_mask):
        del graphother, attention_mask
        seen_graphstates.append(int(graphstate))
        return jnp.full((input_ids.shape[0], input_ids.shape[1] - 1), graphstate, dtype=jnp.float32)

    trainer.compute_actor_logps_full = score
    trainer._prepare_target_batch = lambda batch: batch
    scoring_state = SimpleNamespace(graphstate=jnp.asarray(9), graphother=None)
    normalized = trainer._normalize_provider_batch(
        {
            "input_ids": np.asarray([[1, 2, 3]], dtype=np.int32),
            "attention_mask": np.ones((1, 3), dtype=np.int32),
            "target_action_mask": np.asarray([[1, 1]], dtype=np.int32),
            "target_rewards": np.asarray([[0.0, 1.0]], dtype=np.float32),
        },
        scoring_state=scoring_state,
    )

    assert seen_graphstates == [9]
    np.testing.assert_array_equal(np.asarray(normalized["behavior_logps"]), [[9.0, 9.0]])


def test_provider_rejects_multiple_causal_context_segments_in_one_row():
    trajectory = OnlineRLTrajectory(
        input_ids=np.arange(10, 18, dtype=np.int32),
        attention_mask=np.ones(8, dtype=np.bool_),
        action_mask=np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.bool_),
        rewards=np.asarray([0, 0, 0, 0.2, 0, -0.1, 0, 1.0], dtype=np.float32),
        behavior_logps=np.asarray([0, 0, -0.1, -0.2, 0, -0.3, 0, -0.4], dtype=np.float32),
        segments=(
            TrajectorySegment(0, 0, 5, "execution"),
            TrajectorySegment(1, 5, 7, "execution"),
            TrajectorySegment(2, 7, 8, "compaction", is_terminal=True),
        ),
        policy_version=3,
    )
    provider_batch = OnlineRLTrajectoryCollator(max_length=8, pad_token_id=0)([trajectory])
    trainer = OnlineActorCriticTrainer.__new__(OnlineActorCriticTrainer)
    trainer.arguments = SimpleNamespace(
        rollout_logprob_source="engine",
        kl_coef=0.0,
        max_policy_staleness=0,
    )
    trainer._prepare_target_batch = lambda batch: batch

    with pytest.raises(ValueError, match="different causal contexts"):
        trainer._normalize_provider_batch(
            provider_batch,
            scoring_state=SimpleNamespace(step=jnp.asarray(3)),
        )


def test_provider_accepts_separate_full_context_segment_rows_and_expands_scores():
    trajectories = (
        OnlineRLTrajectory(
            input_ids=np.arange(10, 15, dtype=np.int32),
            attention_mask=np.ones(5, dtype=np.bool_),
            action_mask=np.asarray([0, 0, 1, 1, 0], dtype=np.bool_),
            rewards=np.zeros(5, dtype=np.float32),
            behavior_logps=np.asarray([0, 0, -0.1, -0.2, 0], dtype=np.float32),
            segments=(TrajectorySegment(0, 0, 5, "execution"),),
            trajectory_id=9,
            policy_version=3,
        ),
        OnlineRLTrajectory(
            input_ids=np.arange(20, 25, dtype=np.int32),
            attention_mask=np.ones(5, dtype=np.bool_),
            action_mask=np.asarray([0, 0, 1, 0, 0], dtype=np.bool_),
            rewards=np.zeros(5, dtype=np.float32),
            behavior_logps=np.asarray([0, 0, -0.3, 0, 0], dtype=np.float32),
            segments=(TrajectorySegment(1, 0, 5, "execution"),),
            trajectory_id=9,
            policy_version=3,
        ),
        OnlineRLTrajectory(
            input_ids=np.arange(30, 35, dtype=np.int32),
            attention_mask=np.ones(5, dtype=np.bool_),
            action_mask=np.asarray([0, 1, 1, 0, 0], dtype=np.bool_),
            rewards=np.zeros(5, dtype=np.float32),
            behavior_logps=np.asarray([0, -0.4, -0.5, 0, 0], dtype=np.float32),
            segments=(TrajectorySegment(2, 0, 5, "compaction", is_terminal=True),),
            trajectory_id=9,
            policy_version=3,
        ),
    )
    provider_batch = OnlineRLTrajectoryCollator(max_length=5, pad_token_id=0)(trajectories)
    provider_batch["scores"] = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)
    trainer = OnlineActorCriticTrainer.__new__(OnlineActorCriticTrainer)
    trainer.arguments = SimpleNamespace(
        rollout_logprob_source="engine",
        kl_coef=0.0,
        max_policy_staleness=0,
    )
    trainer._prepare_target_batch = lambda batch: batch

    normalized = trainer._normalize_provider_batch(
        provider_batch,
        scoring_state=SimpleNamespace(step=jnp.asarray(3)),
    )

    np.testing.assert_allclose(
        np.asarray(normalized["behavior_logps"]),
        [
            [0.0, -0.1, -0.2, 0.0],
            [0.0, -0.3, 0.0, 0.0],
            [-0.4, -0.5, 0.0, 0.0],
        ],
    )
    np.testing.assert_array_equal(
        np.asarray(normalized["segment_ids"]),
        [0, 1, 2],
    )
    np.testing.assert_array_equal(
        np.asarray(normalized["trajectory_ids"]),
        [9, 9, 9],
    )
    np.testing.assert_array_equal(
        np.asarray(normalized["rewards"]),
        [
            [0, 0, 2, 0],
            [0, 2, 0, 0],
            [0, 2, 0, 0],
        ],
    )


def test_provider_rejects_rollouts_beyond_configured_policy_staleness():
    trainer = OnlineActorCriticTrainer.__new__(OnlineActorCriticTrainer)
    trainer.arguments = SimpleNamespace(
        rollout_logprob_source="engine",
        kl_coef=0.0,
        max_policy_staleness=1,
    )
    trainer._prepare_target_batch = lambda batch: batch

    with pytest.raises(ValueError, match="max_policy_staleness"):
        trainer._normalize_provider_batch(
            {
                "input_ids": np.asarray([[1, 2, 3]], dtype=np.int32),
                "attention_mask": np.ones((1, 3), dtype=np.int32),
                "target_action_mask": np.asarray([[1, 1]], dtype=np.int32),
                "target_rewards": np.asarray([[0.0, 1.0]], dtype=np.float32),
                "target_behavior_logps": np.asarray([[-0.2, -0.3]], dtype=np.float32),
                "policy_versions": np.asarray([2], dtype=np.int32),
            },
            scoring_state=SimpleNamespace(step=jnp.asarray(4)),
        )


def test_provider_rejects_action_target_predicted_from_padding():
    trainer = OnlineActorCriticTrainer.__new__(OnlineActorCriticTrainer)
    trainer.arguments = SimpleNamespace(
        rollout_logprob_source="engine",
        kl_coef=0.0,
        max_policy_staleness=None,
    )
    trainer._prepare_target_batch = lambda batch: batch

    with pytest.raises(ValueError, match="attended causal context"):
        trainer._normalize_provider_batch(
            {
                "input_ids": np.asarray([[0, 10, 11]], dtype=np.int32),
                "attention_mask": np.asarray([[0, 1, 1]], dtype=np.int32),
                "target_action_mask": np.asarray([[1, 0]], dtype=np.int32),
                "target_behavior_logps": np.asarray([[-0.2, 0.0]], dtype=np.float32),
            },
            scoring_state=SimpleNamespace(step=jnp.asarray(0)),
        )


def test_critic_checkpoint_restores_into_value_head_template(tmp_path, monkeypatch):
    import spectrax as spx
    from easydel.modules.llama.llama_configuration import LlamaConfig
    from easydel.modules.llama.modeling_llama import LlamaForCausalLM
    from easydel.trainers._online_rl.actor_critic import build_critic_trainable_selector
    from easydel.trainers.proximal_policy_optimization_trainer.modeling_value_head import (
        CausalLMWithValueHead,
    )

    def make_critic(seed):
        config = LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=32,
            max_position_embeddings=16,
            sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        )
        actor = LlamaForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=spx.Rngs(seed),
        )
        return CausalLMWithValueHead(actor, rngs=spx.Rngs(seed + 100))

    arguments = SAOConfig(max_training_steps=1, critic_pretrain_steps=0)
    selector = build_critic_trainable_selector(arguments.critic_trainable_mode)
    saved = make_critic(0).to_state(trainable_selector=selector)
    saved = saved.replace(
        step=jnp.asarray(7, dtype=jnp.int32),
        graphstate=jax.tree_util.tree_map(lambda value: value + 0.125, saved.graphstate),
    )

    checkpoint = tmp_path / "run-0"
    critic_directory = checkpoint / "critic"
    critic_directory.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text(json.dumps({"step": 0}))
    (critic_directory / "metadata.json").write_text(json.dumps({"step": 7, "has_optimizer_state": False}))
    (checkpoint / "online_rl_checkpoint.json").write_text(
        json.dumps(
            {
                "format": "easydel-online-actor-critic-v1",
                "actor_step": 0,
                "critic_step": 7,
                "actor_algorithm": "sao",
                "complete": True,
            }
        )
    )

    saved_full = saved.graphstate.merge(saved.graphother, copy=False).raw()

    class FakeCheckpointer:
        def __init__(self, **kwargs):
            del kwargs

        def load_pytree(self, **kwargs):
            assert kwargs["prefix"] == "model"
            assert kwargs["discover_latest"] is False
            return saved_full, {}

    monkeypatch.setattr("easydel.trainers._online_rl.trainer.Checkpointer", FakeCheckpointer)
    monkeypatch.setattr(
        "easydel.trainers._online_rl.trainer.EasyDeLState.load_state",
        lambda **_: (_ for _ in ()).throw(AssertionError("generic wrapper loading is invalid")),
    )
    trainer = OnlineActorCriticTrainer.__new__(OnlineActorCriticTrainer)
    trainer.actor_algorithm = "sao"
    trainer.arguments = arguments
    trainer.critic_state = make_critic(1).to_state(trainable_selector=selector)
    trainer._load_auxiliary_states(str(checkpoint))

    assert int(trainer.critic_state.step) == 7
    assert trainer.critic_state.tx is not None
    assert trainer.critic_state.opt_state is None
    saved_leaves = jax.tree_util.tree_leaves(saved.graphstate)
    loaded_leaves = jax.tree_util.tree_leaves(trainer.critic_state.graphstate)
    assert len(saved_leaves) == len(loaded_leaves)
    for expected, actual in zip(saved_leaves, loaded_leaves, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


def test_paired_checkpoint_marks_manifest_incomplete_before_critic_save(tmp_path, monkeypatch):
    trainer = OnlineActorCriticTrainer.__new__(OnlineActorCriticTrainer)
    trainer.actor_algorithm = "sao"
    trainer.arguments = SimpleNamespace(save_optimizer_state=False)
    trainer.critic_state = SimpleNamespace(
        model=SimpleNamespace(param_dtype=jnp.float32),
        save_state=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("critic save failed")),
    )
    actor_state = SimpleNamespace(step=jnp.asarray(4))
    monkeypatch.setattr(
        PPOTrainer,
        "_save_state",
        lambda self, state, save_directory, *args, **kwargs: str(save_directory),
    )

    with pytest.raises(RuntimeError, match="critic save failed"):
        trainer._save_state(actor_state, str(tmp_path))

    manifest = json.loads((tmp_path / "online_rl_checkpoint.json").read_text())
    assert manifest["actor_step"] == 4
    assert manifest["actor_algorithm"] == "sao"
    assert manifest["complete"] is False


@pytest.mark.parametrize(
    ("manifest_algorithm", "manifest_actor_step", "error"),
    [
        ("ppo", 5, "algorithm"),
        ("sao", 4, "Actor checkpoint step"),
    ],
)
def test_paired_checkpoint_validates_actor_half(
    tmp_path,
    manifest_algorithm,
    manifest_actor_step,
    error,
):
    (tmp_path / "critic").mkdir()
    (tmp_path / "metadata.json").write_text(json.dumps({"step": 5}))
    (tmp_path / "online_rl_checkpoint.json").write_text(
        json.dumps(
            {
                "format": "easydel-online-actor-critic-v1",
                "actor_step": manifest_actor_step,
                "critic_step": 0,
                "actor_algorithm": manifest_algorithm,
                "complete": True,
            }
        )
    )
    trainer = OnlineActorCriticTrainer.__new__(OnlineActorCriticTrainer)
    trainer.actor_algorithm = "sao"

    with pytest.raises(ValueError, match=error):
        trainer._load_auxiliary_states(str(tmp_path))


def test_resumed_critic_rebinds_the_fully_configured_optimizer(monkeypatch):
    class StateStub(SimpleNamespace):
        def replace(self, **updates):
            return StateStub(**(self.__dict__ | updates))

    trainer = OnlineActorCriticTrainer.__new__(OnlineActorCriticTrainer)
    trainer._resumed_from_checkpoint = True
    trainer.arguments = SimpleNamespace(init_tx=True)
    trainer.critic_tx = object()
    old_tx = object()
    optimizer_state = object()
    trainer.critic_state = StateStub(
        opt_state=optimizer_state,
        tx=old_tx,
        step=jnp.asarray(7),
        shardings="critic-shardings",
    )
    monkeypatch.setattr(PPOTrainer, "_configure_state", lambda self: None)

    trainer._configure_state()

    assert trainer.critic_state.tx is trainer.critic_tx
    assert trainer.critic_state.opt_state is optimizer_state
    assert int(trainer.critic_state.step) == 7
    assert trainer.critic_state_shardings == "critic-shardings"


def test_critic_pretraining_uses_distinct_batches_and_keeps_actor_frozen(monkeypatch):
    trainer = OnlineActorCriticTrainer.__new__(OnlineActorCriticTrainer)
    trainer.arguments = SimpleNamespace(critic_pretrain_steps=3)
    trainer._critic_pretraining_complete = False
    trainer.dataloader_train = [{"batch_id": 0}, {"batch_id": 1}, {"batch_id": 2}]
    trainer.data_collator = None
    trainer.model_state = SimpleNamespace(name="actor")
    trainer.critic_state = SimpleNamespace(step=0)
    trainer._critic_shared_fn_static_args = ()

    consumed = []

    def get_next_batch(iterator, dataset):
        del dataset
        return next(iterator), iterator

    def preprocess(*, state, batch, is_train):
        assert state is trainer.model_state
        assert is_train is True
        consumed.append(batch["batch_id"])
        return {"batch_id": jnp.asarray([batch["batch_id"]])}, {}

    def critic_step(state, batch):
        return SimpleNamespace(step=state.step + 1), LossMetrics(
            loss=batch["batch_id"][0].astype(jnp.float32),
            accuracy=jnp.asarray(1.0),
        )

    trainer._get_next_batch = get_next_batch
    trainer._preprocess_batch_input = preprocess
    trainer.sharded_critic_step_function = critic_step
    monkeypatch.setattr(
        "easydel.trainers.proximal_policy_optimization_trainer.PPOTrainer.start_training_hook",
        lambda self: None,
    )
    monkeypatch.setattr(
        "easydel.trainers._online_rl.trainer.jax.block_until_ready",
        lambda value: value,
    )

    actor_before = trainer.model_state
    trainer.start_training_hook()

    assert consumed == [0, 1, 2]
    assert trainer.model_state is actor_before
    assert trainer.critic_state.step == 3
    assert trainer._critic_pretraining_complete is True

    trainer.start_training_hook()
    assert consumed == [0, 1, 2]


def test_compaction_cross_segment_factors_count_only_later_action_tokens():
    trajectory_ids = jnp.asarray([0, 0, 0, 1], dtype=jnp.int32)
    segment_ids = jnp.asarray([0, 1, 2, 0], dtype=jnp.int32)
    action_mask = jnp.asarray(
        [
            [1, 1, 0],  # two action tokens; prompt/pad ignored
            [1, 0, 0],  # one action token
            [1, 1, 1],  # three later action tokens
            [1, 1, 0],  # different trajectory; never counted for trajectory 0
        ],
        dtype=jnp.float32,
    )
    gamma_lambda = 0.9 * 0.8
    factors = cross_segment_advantage_factors(
        trajectory_ids,
        segment_ids,
        action_mask,
        gamma=0.9,
        gae_lambdas=0.8,
    )
    expected = np.asarray(
        [
            gamma_lambda**4,
            gamma_lambda**3,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )
    # 1e-5 rather than 1e-6: the reference exponentiates in numpy while the kernel
    # accumulates on device, and TPU float32 lands ~2e-6 relative away from CPU.
    np.testing.assert_allclose(np.asarray(factors), expected, rtol=1e-5)

    advantages = jnp.ones_like(action_mask)
    corrected = apply_cross_segment_gae_correction(
        advantages,
        trajectory_ids,
        segment_ids,
        action_mask,
        gamma=0.9,
        gae_lambdas=0.8,
    )
    np.testing.assert_allclose(np.asarray(corrected[:, 0]), expected, rtol=1e-5)


def test_compaction_ppo_loss_uses_one_global_token_denominator():
    current = jnp.log(jnp.asarray([[1.1, 0.9, 2.0], [1.0, 1.0, 1.0]], dtype=jnp.float32))
    behavior = jnp.zeros_like(current)
    advantages = jnp.asarray([[2.0, 2.0, 100.0], [-1.0, 100.0, 100.0]], dtype=jnp.float32)
    mask = jnp.asarray([[1, 1, 0], [1, 0, 0]], dtype=jnp.float32)
    loss = compaction_ppo_policy_loss(
        current,
        behavior,
        advantages,
        mask,
        cliprange=0.2,
        normalizer=3,
    )
    ratios = np.exp(np.asarray(current))
    unclipped = -np.asarray(advantages) * ratios
    clipped = -np.asarray(advantages) * np.clip(ratios, 0.8, 1.2)
    expected = np.sum(np.maximum(unclipped, clipped) * np.asarray(mask)) / 3.0
    assert math.isclose(float(loss), float(expected), rel_tol=1e-6)


@pytest.mark.parametrize(
    ("config_cls", "trainer_cls", "canonical", "alias"),
    [
        (SAOConfig, SAOTrainer, "sao", "single_rollout_asynchronous_optimization"),
        (CompactionRLConfig, CompactionRLTrainer, "compaction_rl", "compaction-rl"),
    ],
)
def test_public_exports_registry_and_argument_roundtrip(tmp_path, config_cls, trainer_cls, canonical, alias):
    assert getattr(ed, config_cls.__name__) is config_cls
    assert getattr(ed, trainer_cls.__name__) is trainer_cls
    assert Registry.get("trainer", canonical) is trainer_cls
    assert Registry.get("trainer", alias) is trainer_cls
    assert Registry.get("trainer-arguments", canonical) is config_cls
    assert Registry.get("trainer-arguments", alias) is config_cls

    config = config_cls(critic_pretrain_steps=0)
    path = tmp_path / f"{canonical}.json"
    config.save_arguments(path)
    loaded = TrainingArguments.load_arguments(path)
    assert type(loaded) is config_cls
    assert loaded.to_dict() == config.to_dict()


def test_sao_rejects_more_than_one_rollout():
    assert SAOConfig().total_batch_size == 128
    with pytest.raises(ValueError, match="single-rollout"):
        SAOConfig(num_generations=2)


def test_sao_critic_warmup_builds_for_a_short_smoke_run():
    config = SAOConfig(max_training_steps=1)
    _, schedule = build_critic_optimizer(config, steps=2)

    assert float(schedule(0)) < config.critic_learning_rate
    assert math.isclose(float(schedule(10)), config.critic_learning_rate, rel_tol=1e-6)


def test_compaction_trainer_builds_default_controller(monkeypatch):
    captured = {}

    def fake_online_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(OnlineActorCriticTrainer, "__init__", fake_online_init)
    config = CompactionRLConfig(critic_pretrain_steps=0)
    trainer = CompactionRLTrainer(
        arguments=config,
        model=object(),
        reward_funcs=lambda **_: [0.0],
    )

    assert trainer.compaction_controller.context_budget == config.context_budget
    assert trainer.compaction_controller.compaction_threshold == config.compaction_threshold
    assert trainer.compaction_controller.recent_pairs == config.recent_steps
    assert trainer.compaction_controller.train_summary_tokens is config.train_summary_tokens
    assert trainer.compaction_controller.failure_policy.value == config.compaction_failure_policy
    assert captured["arguments"] is config


def test_compaction_config_requires_strict_budget_and_resume_placeholder():
    defaults = CompactionRLConfig(critic_pretrain_steps=0)
    assert defaults.total_batch_size == 128
    assert defaults.weight_decay == 0.0
    with pytest.raises(ValueError, match="smaller than"):
        CompactionRLConfig(compaction_threshold=65_536)
    with pytest.raises(ValueError, match=r"\{summary\}"):
        CompactionRLConfig(resume_template="missing placeholder")
    with pytest.raises(ValueError, match="global_token_normalization"):
        CompactionRLConfig(global_token_normalization=False)
