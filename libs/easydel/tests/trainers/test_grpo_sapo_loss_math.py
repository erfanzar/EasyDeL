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

from types import SimpleNamespace

import jax
import pandas as pd
import pytest
from jax import numpy as jnp

from easydel.trainers.group_relative_policy_optimization._fn import (
    _compute_grpo_policy_loss_terms,
    _compute_importance_weights,
    _compute_off_policy_sequence_mask,
    _vespo_gamma_weights,
)
from easydel.trainers.group_relative_policy_optimization.grpo_config import GRPOConfig
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import GRPOTrainer, _compute_rewards_and_advantages


def test_grpo_sapo_policy_terms_match_soft_advantage_formula():
    per_token_logps = jnp.log(jnp.asarray([[1.2, 0.7, 1.0], [0.9, 1.4, 0.8]], dtype=jnp.float32))
    old_per_token_logps = jnp.zeros_like(per_token_logps)
    advantages = jnp.asarray([[2.0], [-1.5]], dtype=jnp.float32)
    completion_mask = jnp.ones_like(per_token_logps)
    sapo_temperature_pos = 1.0
    sapo_temperature_neg = 1.05

    per_token_loss, ratios = _compute_grpo_policy_loss_terms(
        per_token_logps=per_token_logps,
        old_per_token_logps=old_per_token_logps,
        advantages=advantages,
        completion_mask=completion_mask,
        loss_type="sapo",
        epsilon=0.2,
        epsilon_high=0.2,
        delta=None,
        importance_sampling_level="token",
        sapo_temperature_pos=sapo_temperature_pos,
        sapo_temperature_neg=sapo_temperature_neg,
        vespo_k_pos=2.0,
        vespo_lambda_pos=3.0,
        vespo_k_neg=3.0,
        vespo_lambda_neg=2.0,
    )

    expected_ratios = jnp.exp(per_token_logps - old_per_token_logps)
    expected_temperature = jnp.where(advantages > 0, sapo_temperature_pos, sapo_temperature_neg)
    expected_multiplier = jax.nn.sigmoid(expected_temperature * (expected_ratios - 1.0)) * 4.0 / expected_temperature
    expected_loss = -expected_multiplier * advantages

    assert jnp.allclose(ratios, expected_ratios, atol=1e-6)
    assert jnp.allclose(per_token_loss, expected_loss, atol=1e-6)


def test_grpo_luspo_policy_terms_use_clipped_sequence_importance_weights():
    per_token_logps = jnp.log(jnp.asarray([[1.3, 0.7], [0.9, 1.2]], dtype=jnp.float32))
    old_per_token_logps = jnp.zeros_like(per_token_logps)
    advantages = jnp.asarray([[1.5], [-2.0]], dtype=jnp.float32)
    completion_mask = jnp.ones_like(per_token_logps)

    per_token_loss, ratios = _compute_grpo_policy_loss_terms(
        per_token_logps=per_token_logps,
        old_per_token_logps=old_per_token_logps,
        advantages=advantages,
        completion_mask=completion_mask,
        loss_type="luspo",
        epsilon=0.2,
        epsilon_high=0.3,
        delta=None,
        importance_sampling_level="sequence",
        sapo_temperature_pos=1.0,
        sapo_temperature_neg=1.05,
        vespo_k_pos=2.0,
        vespo_lambda_pos=3.0,
        vespo_k_neg=3.0,
        vespo_lambda_neg=2.0,
    )

    expected_log_weights = jnp.mean(per_token_logps - old_per_token_logps, axis=-1, keepdims=True)
    expected_ratios = jnp.exp(expected_log_weights)
    clipped_ratios = jnp.clip(expected_ratios, 0.8, 1.3)
    expected_loss = -jnp.minimum(expected_ratios * advantages, clipped_ratios * advantages)

    assert ratios.shape == (2, 1)
    assert jnp.allclose(ratios, expected_ratios, atol=1e-6)
    assert jnp.allclose(per_token_loss, expected_loss, atol=1e-6)


def test_grpo_sequence_token_importance_weights_match_gspo_token_formula():
    per_token_logps = jnp.asarray([[0.2, 0.4, -0.1]], dtype=jnp.float32)
    old_per_token_logps = jnp.asarray([[0.0, 0.1, 0.0]], dtype=jnp.float32)
    advantages = jnp.asarray([[1.0]], dtype=jnp.float32)
    completion_mask = jnp.asarray([[1.0, 1.0, 0.0]], dtype=jnp.float32)

    _, ratios = _compute_grpo_policy_loss_terms(
        per_token_logps=per_token_logps,
        old_per_token_logps=old_per_token_logps,
        advantages=advantages,
        completion_mask=completion_mask,
        loss_type="grpo",
        epsilon=0.2,
        epsilon_high=0.2,
        delta=None,
        importance_sampling_level="sequence_token",
        sapo_temperature_pos=1.0,
        sapo_temperature_neg=1.05,
        vespo_k_pos=2.0,
        vespo_lambda_pos=3.0,
        vespo_k_neg=3.0,
        vespo_lambda_neg=2.0,
    )

    sequence_log_weight = ((per_token_logps - old_per_token_logps) * completion_mask).sum(axis=-1, keepdims=True) / 2
    assert ratios.shape == per_token_logps.shape
    assert jnp.allclose(ratios, jnp.exp(jnp.broadcast_to(sequence_log_weight, per_token_logps.shape)), atol=1e-6)


def test_grpo_vespo_gamma_weights_match_trl_formula():
    log_ratio = jnp.asarray([[0.2, -0.1, 0.0], [0.4, 0.3, -0.2]], dtype=jnp.float32)
    mask = jnp.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=jnp.float32)
    advantages = jnp.asarray([[1.0], [-1.0]], dtype=jnp.float32)

    weights = _vespo_gamma_weights(
        advantages=advantages,
        log_ratio_per_token=log_ratio,
        mask=mask,
        importance_sampling_ratio=None,
        k_pos=2.0,
        lambda_pos=3.0,
        k_neg=3.0,
        lambda_neg=2.0,
    )

    seq_log_ratio = jnp.sum(jnp.clip(log_ratio, -20.0, 20.0) * mask, axis=-1, keepdims=True)
    w_seq = jnp.exp(jnp.clip(seq_log_ratio, jnp.log(1e-8), 20.0))
    k_seq = jnp.where(advantages >= 0, 2.0, 3.0)
    lambda_seq = jnp.where(advantages >= 0, 3.0, 2.0)
    expected = jnp.exp(lambda_seq + k_seq * jnp.log(w_seq) - lambda_seq * w_seq)

    assert weights.shape == (2, 1)
    assert jnp.allclose(weights, expected, atol=1e-6)


def test_rloo_advantages_use_leave_one_out_group_baseline():
    rewards_per_func = jnp.asarray([[1.0], [3.0], [7.0], [2.0]], dtype=jnp.float32)
    rewards, advantages, _, _ = _compute_rewards_and_advantages(
        rewards_per_func=rewards_per_func,
        reward_weights=jnp.asarray([1.0], dtype=jnp.float32),
        generation_factor=2,
        scale_rewards="none",
        multi_objective_aggregation="sum_then_normalize",
        arguments=SimpleNamespace(advantage_estimator="leave_one_out", reward_clip_range=None),
    )

    assert jnp.allclose(rewards, jnp.asarray([1.0, 3.0, 7.0, 2.0], dtype=jnp.float32))
    assert jnp.allclose(advantages, jnp.asarray([-2.0, 2.0, 5.0, -5.0], dtype=jnp.float32))


def test_grpo_config_accepts_new_loss_variants_and_validates_temperatures():
    cfg = GRPOConfig(loss_type="SAPO", sapo_temperature_pos=0.8, sapo_temperature_neg=1.2)

    assert cfg.loss_type == "sapo"
    assert cfg.sapo_temperature_pos == 0.8
    assert cfg.sapo_temperature_neg == 1.2
    assert GRPOConfig(loss_type="luspo").loss_type == "luspo"
    assert GRPOConfig(loss_type="vespo").loss_type == "vespo"

    with pytest.raises(ValueError, match="sapo_temperature_pos"):
        GRPOConfig(loss_type="sapo", sapo_temperature_pos=0.0)

    with pytest.raises(ValueError, match="vespo_lambda_neg"):
        GRPOConfig(loss_type="vespo", vespo_lambda_neg=0.0)

    with pytest.raises(ValueError, match="loss_type"):
        GRPOConfig(loss_type="unknown")


def test_grpo_off_policy_sequence_mask_keeps_positive_and_low_kl_negative_rows():
    per_token_logps = jnp.asarray(
        [
            [-2.0, -2.0],
            [-2.0, -2.0],
            [-2.0, -2.0],
        ],
        dtype=jnp.float32,
    )
    sampling_per_token_logps = jnp.asarray(
        [
            [-1.0, -1.0],
            [-1.8, -1.8],
            [-0.8, -0.8],
        ],
        dtype=jnp.float32,
    )
    advantages = jnp.asarray([[1.0], [-1.0], [-1.0]], dtype=jnp.float32)
    completion_mask = jnp.ones_like(per_token_logps)

    mask = _compute_off_policy_sequence_mask(
        per_token_logps=per_token_logps,
        sampling_per_token_logps=sampling_per_token_logps,
        advantages=advantages,
        completion_mask=completion_mask,
        threshold=0.5,
    )

    assert jnp.array_equal(mask, jnp.asarray([[1.0], [1.0], [0.0]], dtype=jnp.float32))


def test_grpo_importance_weights_for_bias_kl_are_unclipped_by_loss_delta():
    per_token_logps = jnp.log(jnp.asarray([[2.0, 1.5]], dtype=jnp.float32))
    old_per_token_logps = jnp.zeros_like(per_token_logps)
    completion_mask = jnp.ones_like(per_token_logps)
    advantages = jnp.asarray([[1.0]], dtype=jnp.float32)

    raw_weights = _compute_importance_weights(
        per_token_logps=per_token_logps,
        old_per_token_logps=old_per_token_logps,
        completion_mask=completion_mask,
        importance_sampling_level="token",
    )
    _, capped_weights = _compute_grpo_policy_loss_terms(
        per_token_logps=per_token_logps,
        old_per_token_logps=old_per_token_logps,
        advantages=advantages,
        completion_mask=completion_mask,
        loss_type="dapo",
        epsilon=0.2,
        epsilon_high=0.2,
        delta=1.1,
        importance_sampling_level="token",
        sapo_temperature_pos=1.0,
        sapo_temperature_neg=1.05,
        vespo_k_pos=2.0,
        vespo_lambda_pos=3.0,
        vespo_k_neg=3.0,
        vespo_lambda_neg=2.0,
    )

    assert jnp.allclose(raw_weights, jnp.asarray([[2.0, 1.5]], dtype=jnp.float32), atol=1e-6)
    assert jnp.allclose(capped_weights, jnp.asarray([[1.1, 1.1]], dtype=jnp.float32), atol=1e-6)


def test_grpo_config_accepts_off_policy_and_bias_correction_knobs():
    cfg = GRPOConfig(
        off_policy_mask_threshold=0.5,
        use_bias_correction_kl=True,
    )

    assert cfg.off_policy_mask_threshold == 0.5
    assert cfg.use_bias_correction_kl is True
    assert GRPOConfig(disable_dropout=False).disable_dropout is False
    with pytest.raises(ValueError, match="off_policy_mask_threshold"):
        GRPOConfig(off_policy_mask_threshold=0.0)
    with pytest.raises(ValueError, match="top_entropy_quantile"):
        GRPOConfig(top_entropy_quantile=1.1)
    with pytest.raises(ValueError, match="top_entropy_quantile"):
        GRPOConfig(top_entropy_quantile=-0.1)
    with pytest.raises(ValueError, match="ref_model_mixup_alpha"):
        GRPOConfig(ref_model_mixup_alpha=-0.1)
    with pytest.raises(ValueError, match="ref_model_sync_steps"):
        GRPOConfig(ref_model_sync_steps=0)


def test_grpo_esurge_generation_knobs_validate():
    cfg = GRPOConfig(
        cast_lm_head_to_fp32=True,
        use_transformers_paged=True,
        use_esurge_generation=True,
        esurge_hbm_utilization=0.7,
        esurge_importance_sampling_mode="sequence_mask",
    )
    assert cfg.use_esurge_generation is True
    assert cfg.esurge_hbm_utilization == 0.7
    with pytest.raises(ValueError, match="log_completions_hub_repo"):
        GRPOConfig(log_completions_hub_repo="repo")
    assert GRPOConfig(log_completions=True, log_completions_hub_repo="repo").log_completions_hub_repo == "repo"
    with pytest.raises(ValueError, match="esurge_importance_sampling_mode"):
        GRPOConfig(esurge_importance_sampling_mode="bad")


def test_grpo_disable_dropout_puts_policy_reference_and_reward_models_in_eval_mode():
    class Model:
        def __init__(self):
            self.eval_calls = 0

        def eval(self):
            self.eval_calls += 1

    policy = SimpleNamespace(model=Model())
    reference = SimpleNamespace(model=Model())
    reward = SimpleNamespace(model=Model())

    GRPOTrainer._disable_state_dropout(policy, reference, reward)

    assert policy.model.eval_calls == 1
    assert reference.model.eval_calls == 1
    assert reward.model.eval_calls == 1


def test_grpo_string_reward_func_is_rejected():
    with pytest.raises(ValueError, match="reward model"):
        GRPOTrainer._resolve_reward_func_model("repo/reward")


def test_grpo_shuffle_dataset_alias_updates_base_shuffle_flag():
    cfg = GRPOConfig(shuffle_dataset=False)

    assert cfg.shuffle_dataset is False
    assert cfg.shuffle_train_dataset is False


def test_grpo_pad_to_multiple_of_validates_positive_value():
    cfg = GRPOConfig(pad_to_multiple_of=8)

    assert cfg.pad_to_multiple_of == 8
    with pytest.raises(ValueError, match="pad_to_multiple_of"):
        GRPOConfig(pad_to_multiple_of=0)


def test_grpo_num_generations_eval_overrides_eval_rollout_count():
    cfg = GRPOConfig(num_generations=4, num_generations_eval=2)
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = cfg

    assert cfg.num_return_sequences == 4
    assert cfg.num_generations == 4
    assert cfg.num_generations_eval == 2
    assert trainer._generation_config_overrides_for_phase(is_train=True) == {"num_return_sequences": 4}
    assert trainer._generation_config_overrides_for_phase(is_train=False) == {"num_return_sequences": 2}

    with pytest.raises(ValueError, match="num_generations_eval"):
        GRPOConfig(num_generations_eval=0)


def test_grpo_generation_reuse_config_derives_steps_and_validates_exclusive_knobs():
    cfg = GRPOConfig(total_batch_size=4, generation_batch_size=12)

    assert cfg.steps_per_generation == 3
    assert cfg.generation_batch_size == 12
    assert GRPOConfig(total_batch_size=4, steps_per_generation=2).generation_batch_size == 8
    with pytest.raises(ValueError, match="mutually exclusive"):
        GRPOConfig(generation_batch_size=8, steps_per_generation=2)
    with pytest.raises(ValueError, match="divisible"):
        GRPOConfig(total_batch_size=4, generation_batch_size=10)
    with pytest.raises(ValueError, match="num_iterations"):
        GRPOConfig(num_iterations=0)


def test_grpo_generation_reuse_buffer_returns_cached_batch_for_configured_span():
    trainer = object.__new__(GRPOTrainer)
    trainer.steps_per_generation = 2
    trainer.num_iterations = 2
    trainer._buffered_grpo_batch = None
    trainer._buffered_grpo_remaining = 0
    batch = {"completion_ids": jnp.asarray([[1, 2]], dtype=jnp.int32)}
    metrics = {"generation_time": 1.0}

    returned_batch, returned_metrics = GRPOTrainer._store_buffered_grpo_batch(trainer, batch, metrics)

    assert returned_batch is batch
    assert returned_metrics["generation_reused"] == 0
    assert returned_metrics["generation_reuse_span"] == 4
    assert returned_metrics["generation_reuse_remaining"] == 3

    first_reuse = GRPOTrainer._take_buffered_grpo_batch(trainer)
    second_reuse = GRPOTrainer._take_buffered_grpo_batch(trainer)
    third_reuse = GRPOTrainer._take_buffered_grpo_batch(trainer)

    assert first_reuse is not None
    assert first_reuse[1]["generation_reused"] == 1
    assert first_reuse[1]["generation_reuse_remaining"] == 2
    assert second_reuse is not None
    assert third_reuse is not None
    assert GRPOTrainer._take_buffered_grpo_batch(trainer) is None


def test_grpo_completion_logging_selects_unique_limited_rows(monkeypatch):
    cfg = GRPOConfig(log_completions=True, num_completions_to_print=2, log_unique_prompts=True)
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = cfg
    logged = []

    monkeypatch.setattr(
        "easydel.trainers.group_relative_policy_optimization.grpo_trainer.logger.info",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )

    rows = trainer._maybe_log_grpo_completions(
        prompts=["p0", "p0", "p1", "p2"],
        completions=["c0", "c0b", "c1", "c2"],
        completion_lengths=jnp.asarray([1, 2, 3, 4], dtype=jnp.float32),
    )

    assert rows == [
        {"sample_idx": 0, "prompt": "p0", "completion": "c0", "completion_length": 1.0},
        {"sample_idx": 2, "prompt": "p1", "completion": "c1", "completion_length": 3.0},
    ]
    assert logged


def test_grpo_completion_logging_writes_parquet_and_triggers_hub_scheduler(monkeypatch, tmp_path):
    cfg = GRPOConfig(
        log_completions=True,
        log_completions_hub_repo="user/grpo-completions",
        save_directory=str(tmp_path),
    )
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = cfg
    trainer._completion_log_dir = None
    trainer._completion_commit_scheduler = None
    create_repo_calls = []
    scheduler_calls = []
    trigger_calls = []

    class FakeCommitScheduler:
        def __init__(self, **kwargs):
            scheduler_calls.append(kwargs)

        def trigger(self):
            trigger_calls.append(True)

    monkeypatch.setattr("easydel.trainers.group_relative_policy_optimization.grpo_trainer.jax.process_index", lambda: 0)
    monkeypatch.setattr("huggingface_hub.create_repo", lambda *args, **kwargs: create_repo_calls.append((args, kwargs)))
    monkeypatch.setattr("huggingface_hub.CommitScheduler", FakeCommitScheduler)

    rows = GRPOTrainer._maybe_log_grpo_completions(
        trainer,
        prompts=["p0", "p1"],
        completions=["c0", "c1"],
        completion_lengths=jnp.asarray([2, 3], dtype=jnp.float32),
        step=12,
    )

    assert len(rows) == 2
    assert create_repo_calls == [(("user/grpo-completions",), {"repo_type": "dataset", "exist_ok": True})]
    assert scheduler_calls
    assert scheduler_calls[0]["repo_id"] == "user/grpo-completions"
    assert scheduler_calls[0]["repo_type"] == "dataset"
    assert trigger_calls == [True]

    parquet_files = sorted(tmp_path.rglob("*.parquet"))
    assert len(parquet_files) == 1
    frame = pd.read_parquet(parquet_files[0])
    assert frame["step"].tolist() == [12, 12]
    assert frame["prompt"].tolist() == ["p0", "p1"]
    assert frame["completion"].tolist() == ["c0", "c1"]


def test_grpo_completion_logging_config_validates_limit():
    assert GRPOConfig(num_completions_to_print=1).num_completions_to_print == 1
    with pytest.raises(ValueError, match="num_completions_to_print"):
        GRPOConfig(num_completions_to_print=0)


def test_grpo_normalize_then_sum_aggregates_each_reward_before_batch_advantages():
    rewards_per_func = jnp.asarray(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [2.0, 30.0],
            [4.0, 60.0],
        ],
        dtype=jnp.float32,
    )
    reward_weights = jnp.asarray([1.0, 0.5], dtype=jnp.float32)
    cfg = GRPOConfig(multi_objective_aggregation="normalize_then_sum")

    rewards, advantages, std_rewards, is_std_zero = _compute_rewards_and_advantages(
        rewards_per_func=rewards_per_func,
        reward_weights=reward_weights,
        generation_factor=2,
        scale_rewards="group",
        multi_objective_aggregation=cfg.multi_objective_aggregation,
        arguments=cfg,
    )

    grouped = rewards_per_func.reshape(-1, 2, 2)
    normalized = (grouped - jnp.nanmean(grouped, axis=1, keepdims=True)) / (
        jnp.nanstd(grouped, axis=1, keepdims=True) + 1e-4
    )
    expected_rewards = jnp.nansum(normalized.reshape(-1, 2) * reward_weights[None, :], axis=1)
    expected_std = jnp.nanstd(expected_rewards)
    expected_advantages = (expected_rewards - jnp.nanmean(expected_rewards)) / (expected_std + 1e-4)

    assert jnp.allclose(rewards, expected_rewards, atol=1e-6)
    assert jnp.allclose(advantages, expected_advantages, atol=1e-6)
    assert jnp.allclose(std_rewards, jnp.broadcast_to(expected_std, rewards.shape), atol=1e-6)
    assert not bool(jnp.any(is_std_zero))
