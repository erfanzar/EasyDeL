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

"""Non-mocked regression suite for Online RL, GRPO/PPO, SDPO, Rollout & Agentic trainer fixes.

Covers verified review findings RL-C1..RL-C11, RL-H1..RL-H17, RL-M1..RL-M11,
RL-L1, and PKD-C9 (SDPOTrainer completion_mask intersection).
"""

from __future__ import annotations

import inspect
import signal
import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.infra.mixins.generation import EasyGenerationMixin
from easydel.trainers._online_rl.actor_critic import (
    _token_normalizer,
    actor_step,
    critic_step,
)
from easydel.trainers._online_rl.config import OnlineActorCriticConfig
from easydel.trainers._online_rl.trainer import OnlineActorCriticTrainer
from easydel.trainers._online_rl.trajectory import OnlineRLTrajectory, OnlineRLTrajectoryCollator
from easydel.trainers._shared import esurge_generation
from easydel.trainers._shared.environment import ToolEnvWrapper
from easydel.trainers._shared.policy_losses import clipped_ppo_policy_loss, sao_policy_loss
from easydel.trainers._shared.sandbox import Sandbox
from easydel.trainers._shared.tools import FileReadTool, Tool, WebFetchTool
from easydel.trainers.agentic_moshpit import compute_advantages_agentic
from easydel.trainers.agentic_moshpit.agentic_moshpit_config import AgenticMoshPitConfig
from easydel.trainers.agentic_moshpit.agentic_moshpit_trainer import AgenticMoshPitTrainer
from easydel.trainers.agentic_moshpit.env_manager import RolloutManager
from easydel.trainers.agentic_moshpit.self_play import SelfPlayEnvironment
from easydel.trainers.agentic_rollout.compaction import (
    CompactionController,
    CompactionFailurePolicy,
)
from easydel.trainers.agentic_rollout.guards import (
    RegexRuleFilter,
    RuleAction,
    RuleCheck,
)
from easydel.trainers.agentic_rollout.schema import (
    AtomicInteractionStep,
    PolicySummary,
    RolloutMessage,
    SegmentKind,
)
from easydel.trainers.async_grpo_trainer.async_grpo_config import AsyncGRPOConfig
from easydel.trainers.async_grpo_trainer.async_grpo_trainer import AsyncGRPOTrainer
from easydel.trainers.base_trainer import GenerationResults
from easydel.trainers.compaction_reinforcement_learning_trainer.compaction_rl_config import (
    CompactionRLConfig,
)
from easydel.trainers.esurge_rollout.generation import generate_rollout_completions
from easydel.trainers.group_relative_policy_optimization._fn import (
    _compute_dppo_divergence_mask,
    _global_token_normalizer,
    _make_grpo_microbatch_weight,
    _vespo_gamma_weights,
    grpo_step,
)
from easydel.trainers.group_relative_policy_optimization.grpo_config import GRPOConfig
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import (
    GRPOTrainer,
    _leave_one_out_group_baseline,
)
from easydel.trainers.group_sequence_policy_optimization.gspo_trainer import GSPOTrainer
from easydel.trainers.grpo_replay_buffer_trainer.grpo_replay_buffer_trainer import (
    GRPOWithReplayBufferTrainer,
)
from easydel.trainers.grpo_replay_buffer_trainer.replay_buffer import _ReplayBuffer
from easydel.trainers.nemo_gym_trainer._fn import _normalize_step_result
from easydel.trainers.nemo_gym_trainer.nemo_gym_trainer import NeMoGymTrainer
from easydel.trainers.papo_trainer.papo_trainer import PAPOTrainer
from easydel.trainers.proximal_policy_optimization_trainer._fn import ppo_step
from easydel.trainers.proximal_policy_optimization_trainer.modeling_value_head import (
    CausalLMWithValueHead,
)
from easydel.trainers.proximal_policy_optimization_trainer.ppo_config import PPOConfig
from easydel.trainers.proximal_policy_optimization_trainer.ppo_trainer import PPOTrainer
from easydel.trainers.rlvr_trainer.reward_verifiers import (
    CodeVerifier,
    JSONVerifier,
    MathVerifier,
    MultipleChoiceVerifier,
)
from easydel.trainers.rlvr_trainer.rlvr_config import RLVRConfig
from easydel.trainers.self_distillation_policy_optimization import SDPOConfig, SDPOTrainer
from easydel.trainers.self_distillation_policy_optimization._fn import (
    _full_vocab_sdpo_loss,
    _sdpo_step_with_teacher_state,
    sdpo_step,
)
from easydel.trainers.single_rollout_asynchronous_optimization_trainer.sao_config import SAOConfig

pytestmark = pytest.mark.cpu_ok


# ---------------------------------------------------------------------------
# Minimal differentiable JAX state/model helpers (real math, zero mocks)
# ---------------------------------------------------------------------------


class _LinearLM:
    """Tiny differentiable causal LM whose logits depend on prompt+completion history."""

    mesh = None
    config = SimpleNamespace(lmhead_chunksize=None, vocab_size=8)

    def __init__(self, weight: jax.Array, vocab_size: int = 8):
        self.weight = weight  # [vocab_size]
        self.vocab_size = vocab_size

    def __call__(self, input_ids: jax.Array, attention_mask: jax.Array, **kwargs):
        del kwargs
        # Running masked token sum gives prompt-dependent per-position context.
        ctx = jnp.cumsum((input_ids * attention_mask).astype(jnp.float32), axis=1) * 0.05
        vocab_idx = jnp.arange(self.vocab_size, dtype=jnp.float32)[None, None, :]
        logits = ctx[..., None] * self.weight[None, None, :] + 0.1 * vocab_idx
        return SimpleNamespace(logits=logits, last_hidden_state=logits)


class _LinearValueLM(_LinearLM):
    """Tiny differentiable CausalLM + value head for real ppo_step / critic_step checks."""

    def __init__(self, weight: jax.Array, value_weight: jax.Array, vocab_size: int = 8):
        super().__init__(weight=weight, vocab_size=vocab_size)
        self.value_weight = value_weight
        self.model = _LinearLM(weight=weight, vocab_size=vocab_size)
        self.value_head = lambda hidden: jnp.mean(hidden, axis=-1, keepdims=True) * self.value_weight

    def __call__(self, input_ids: jax.Array, attention_mask: jax.Array, **kwargs):
        out = super().__call__(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        values = self.value_head(out.last_hidden_state).squeeze(-1)
        return SimpleNamespace(logits=out.logits, values=values, last_hidden_state=out.logits)


class _ToyState:
    """Minimal EasyDeLState-compatible container supporting real jax.value_and_grad."""

    def __init__(self, weight: jax.Array, step: int = 0, value_weight: jax.Array | None = None, vocab_size: int = 8):
        self.step = jnp.asarray(step, dtype=jnp.int32)
        self.vocab_size = vocab_size
        self._has_value = value_weight is not None
        if value_weight is not None:
            self.graphstate = {"w": weight, "v": value_weight}
        else:
            self.graphstate = {"w": weight}
        self.model = self.merge(self.graphstate)
        self.opt_state = None

    def merge(self, tree):
        if self._has_value:
            return _LinearValueLM(tree["w"], tree["v"], vocab_size=self.vocab_size)
        return _LinearLM(tree["w"], vocab_size=self.vocab_size)

    def apply_gradients(self, grads):
        lr = 0.1
        new_tree = jax.tree_util.tree_map(lambda p, g: p - lr * g, self.graphstate, grads)
        return _ToyState(
            weight=new_tree["w"],
            step=int(self.step) + 1,
            value_weight=new_tree["v"] if self._has_value else None,
            vocab_size=self.vocab_size,
        )


# ---------------------------------------------------------------------------
# RL-C1: GRPO gradient_accumulation_steps > 1 prompt/completion pairing
# ---------------------------------------------------------------------------


def test_rl_c1_grpo_ga_gt_1_pairs_prompts_and_completions_identically():
    """GRPO K=1, K=2, and K=4 (K > G!) must produce identical parameter updates."""
    prompt_ids = jnp.asarray([[1, 2], [3, 4], [5, 6], [2, 1]], dtype=jnp.int32)  # B=4
    prompt_mask = jnp.ones((4, 2), dtype=jnp.int32)
    completion_ids = jnp.asarray(
        [
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 1],
            [5, 1, 2],
            [1, 3, 5],
            [2, 4, 1],
            [3, 5, 2],
        ],
        dtype=jnp.int32,
    )  # B*G=8 (G=2)
    completion_mask = jnp.ones((8, 3), dtype=jnp.int32)
    advantages = jnp.asarray([1.0, -1.0, 0.5, -0.5, 0.8, -0.8, 0.2, -0.2], dtype=jnp.float32)

    batch = {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "advantages": advantages,
        "num_items_in_batch": jnp.sum(completion_mask).astype(jnp.float32),
    }

    init_w = jnp.linspace(0.2, 0.9, 8, dtype=jnp.float32)
    outputs = {}
    for ga_steps in (1, 2, 4):
        updated_state, metrics = grpo_step(
            state=_ToyState(init_w),
            batch=batch,
            num_generations=2,
            beta=0.0,
            gradient_accumulation_steps=ga_steps,
            is_training=True,
        )
        outputs[ga_steps] = (updated_state.graphstate["w"], float(metrics.loss))

    # K=2 and K=4 (where K > G!) must match K=1 bit-closely.
    np.testing.assert_allclose(outputs[2][0], outputs[1][0], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(outputs[4][0], outputs[1][0], atol=1e-5, rtol=1e-5)
    assert outputs[2][1] == pytest.approx(outputs[1][1], abs=1e-5)
    assert outputs[4][1] == pytest.approx(outputs[1][1], abs=1e-5)


# ---------------------------------------------------------------------------
# RL-C2: log_ratio > 90 NaN resistance across SAO, PPO, Actor-Critic, SDPO
# ---------------------------------------------------------------------------


def test_rl_c2_log_ratio_gt_90_no_nan_in_sao_ppo_actor_sdpo():
    """Large log_ratio (>90) on rejected/active/padded tokens must never yield NaN loss or grads."""
    # 1. SAO policy loss (rejected token has log_ratio=120 > fp32 exp overflow 88.72)
    def sao_loss_fn(curr_lp):
        beh_lp = jnp.asarray([[-121.0, -1.0, -200.0]], dtype=jnp.float32)
        adv = jnp.asarray([[1.0, 0.5, 0.0]], dtype=jnp.float32)
        mask = jnp.asarray([[1.0, 1.0, 0.0]], dtype=jnp.float32)
        loss = sao_policy_loss(
            current_logps=curr_lp,
            behavior_logps=beh_lp,
            advantages=adv,
            action_mask=mask,
            epsilon_low=0.2,
            epsilon_high=0.2,
        )
        return loss

    curr = jnp.asarray([[-1.0, -1.05, -1.0]], dtype=jnp.float32)
    sao_val, sao_grad = jax.value_and_grad(sao_loss_fn)(curr)
    assert jnp.isfinite(sao_val)
    assert jnp.all(jnp.isfinite(sao_grad))

    # 2. Clipped PPO loss (active adv > 0 token + padded token both have log_ratio=120)
    def ppo_loss_fn(curr_lp):
        beh_lp = jnp.asarray([[-121.0, -1.0, -200.0]], dtype=jnp.float32)
        adv = jnp.asarray([[2.0, -1.0, 0.0]], dtype=jnp.float32)
        mask = jnp.asarray([[1.0, 1.0, 0.0]], dtype=jnp.float32)
        return clipped_ppo_policy_loss(
            current_logps=curr_lp,
            behavior_logps=beh_lp,
            advantages=adv,
            action_mask=mask,
            cliprange=0.2,
        )

    ppo_val, ppo_grad = jax.value_and_grad(ppo_loss_fn)(curr)
    assert jnp.isfinite(ppo_val)
    assert jnp.all(jnp.isfinite(ppo_grad))

    # 3. Full ppo_step with unzeroed pad old_logprobs = -200.0
    init_w = jnp.linspace(0.1, 0.8, 8, dtype=jnp.float32)
    init_v = jnp.asarray(0.5, dtype=jnp.float32)
    ppo_batch = {
        "input_ids": jnp.asarray([[1, 2, 3, 4, 0, 0]], dtype=jnp.int32),
        "attention_mask": jnp.asarray([[1, 1, 1, 1, 0, 0]], dtype=jnp.int32),
        "old_logps": jnp.asarray([[-1.0, -1.0, -200.0, -200.0]], dtype=jnp.float32),
        "old_values": jnp.asarray([[0.1, 0.2, 0.0, 0.0]], dtype=jnp.float32),
        "advantages": jnp.asarray([[1.5, -0.5, 0.0, 0.0]], dtype=jnp.float32),
        "returns": jnp.asarray([[1.6, -0.3, 0.0, 0.0]], dtype=jnp.float32),
        "completion_mask": jnp.asarray([[1.0, 1.0, 0.0, 0.0]], dtype=jnp.float32),
    }
    new_ppo_state, ppo_metrics = ppo_step(
        state=_ToyState(init_w, value_weight=init_v),
        batch=ppo_batch,
        prompt_length=2,
        cliprange=0.2,
        vf_coef=0.1,
        cliprange_value=0.2,
        entropy_coef=0.01,
        logprob_vocab_chunk_size=None,
        is_training=True,
    )
    assert jnp.isfinite(ppo_metrics.loss)
    assert jnp.all(jnp.isfinite(new_ppo_state.graphstate["w"]))
    assert jnp.all(jnp.isfinite(new_ppo_state.graphstate["v"]))

    # 4. Full sdpo_step k3 KL with ref - student > 120 on active and masked tokens (chunked & non-chunked)
    sdpo_batch = {
        "prompt_ids": jnp.asarray([[1, 2], [2, 3]], dtype=jnp.int32),
        "prompt_mask": jnp.ones((2, 2), dtype=jnp.int32),
        "completion_ids": jnp.asarray([[3, 4, 0], [4, 5, 0]], dtype=jnp.int32),
        "completion_mask": jnp.asarray([[1, 1, 0], [1, 1, 0]], dtype=jnp.int32),
        "teacher_ids": jnp.asarray([[1, 2, 5, 3, 4, 0], [2, 3, 6, 4, 5, 0]], dtype=jnp.int32),
        "teacher_mask": jnp.asarray([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]], dtype=jnp.int32),
        "ref_per_token_logps": jnp.asarray([[120.0, -1.0, 200.0], [120.0, -1.0, 200.0]], dtype=jnp.float32),
        "num_items_in_batch": jnp.asarray(4.0, dtype=jnp.float32),
    }
    for chunk_size in (None, 1):
        new_sdpo_state, sdpo_metrics = sdpo_step(
            state=_ToyState(init_w),
            batch=sdpo_batch,
            num_generations=1,
            teacher_prompt_length=3,
            beta=0.1,
            distillation_type="jsd",
            distillation_weight=1.0,
            logprob_vocab_chunk_size=None,
            max_loss_completion_tokens=None,
            completion_chunk_size=chunk_size,
            is_training=True,
        )
        assert jnp.isfinite(sdpo_metrics.loss)
        assert jnp.all(jnp.isfinite(new_sdpo_state.graphstate["w"]))


# ---------------------------------------------------------------------------
# RL-C3: GRPO dynamic prompt_length passed to logp helpers
# ---------------------------------------------------------------------------


def test_rl_c3_grpo_passes_dynamic_prompt_length_to_logp_helpers():
    """GRPOTrainer must pass actual prompt_ids.shape[-1], not stale max_prompt_length."""
    captured = {}

    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = GRPOConfig(
        max_prompt_length=4,
        max_completion_length=3,
        max_length=7,
        beta=0.1,
        num_iterations=2,
        steps_per_generation=1,
        mask_truncated_completions=False,
        scale_rewards="group",
        loss_type="dapo",
        esurge_importance_sampling_correction=False,
        log_completions=False,
    )
    trainer.num_generations = 1
    trainer.num_iterations = 2
    trainer.steps_per_generation = 1
    trainer.scale_rewards = "group"
    trainer.multi_objective_aggregation = "sum_then_normalize"
    trainer.ref_logps_chunk_size = None
    trainer._eos_token_id = [2]
    trainer._pad_token_id = 0
    trainer.padding_value = 0
    trainer.train_is_conversational = False
    trainer.eval_is_conversational = False
    trainer.processing_class = None
    trainer._environment_tools = None
    trainer._environment_tool_parser = None
    trainer.environment_factory = None
    trainer.reward_funcs = [lambda completions, **kw: [1.0] * len(completions)]
    trainer.reward_processing_classes = [None]
    trainer.reward_weights = jnp.asarray([1.0], dtype=jnp.float32)
    trainer.reward_func_names = ["r0"]
    trainer.ref_state = SimpleNamespace(
        graphstate="ref_gs",
        graphother="ref_go",
        model=SimpleNamespace(forward=lambda input_ids, attention_mask: None),
    )
    trainer._buffered_grpo_batch = None
    trainer._buffered_grpo_metrics = None
    trainer._buffered_grpo_remaining = 0
    trainer._buffered_grpo_queue = []
    trainer._apply_user_data_collator = lambda b: b
    trainer._extract_reward_batch_sidechannels = lambda b: {}
    trainer._purify_batch = lambda b: b
    trainer._all_gather = lambda x: x
    trainer._log_training_generations_to_wandb = lambda **kw: None
    trainer._make_attn_mask = lambda ids: jnp.ones_like(ids)

    # Actual prompt width is 6 (> configured max_prompt_length=4)!
    prompt_ids = jnp.asarray([[1, 2, 3, 4, 5, 6]], dtype=jnp.int32)
    prompt_mask = jnp.ones((1, 6), dtype=jnp.int32)
    comp_ids = jnp.asarray([[7, 8, 2]], dtype=jnp.int32)
    comp_mask = jnp.ones((1, 3), dtype=jnp.int32)

    trainer.rollout = lambda **kw: GenerationResults(
        generation_results=["ok"],
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        sequences=jnp.concatenate([prompt_ids, comp_ids], axis=1),
        completion_ids=comp_ids,
        completion_mask=comp_mask,
        decoded_prompts=["p"],
        completion_prompts=["p"],
        text=["ok"],
        raw_text=["ok"],
    )

    def _ref_logps(gs, go, ids, mask, model_kwargs=None, *, prompt_length=None, **kw):
        del gs, go, ids, mask, model_kwargs, kw
        captured["ref_prompt_length"] = prompt_length
        return jnp.zeros((1, 3), dtype=jnp.float32)

    def _state_logps(gs, go, ids, mask, model_kwargs=None, *, prompt_length=None, **kw):
        del gs, go, ids, mask, model_kwargs, kw
        captured["old_prompt_length"] = prompt_length
        return jnp.zeros((1, 3), dtype=jnp.float32)

    trainer.compute_refmodel_logps = _ref_logps
    trainer.compute_state_logps = _state_logps

    state = SimpleNamespace(
        step=jnp.asarray(0, dtype=jnp.int32),
        graphstate="s_gs",
        graphother="s_go",
        model=SimpleNamespace(forward=lambda input_ids, attention_mask: None),
    )
    batch = {"input_ids": prompt_ids, "attention_mask": prompt_mask}
    trainer._preprocess_batch_input(state=state, batch=batch, is_train=True)
    assert captured["ref_prompt_length"] == 6
    assert captured["old_prompt_length"] == 6


# ---------------------------------------------------------------------------
# RL-C4: weight_sync_steps=1 preserves rollouts (AsyncGRPO & OnlineActorCritic)
# ---------------------------------------------------------------------------


def test_rl_c4_online_actor_critic_weight_sync_steps_1_zero_discarded(monkeypatch):
    """OnlineActorCriticTrainer with weight_sync_steps=1 must not discard lookahead rollouts."""
    import easydel.trainers._online_rl.trainer as online_mod

    trainer = object.__new__(OnlineActorCriticTrainer)
    trainer.arguments = OnlineActorCriticConfig(
        max_prompt_length=8,
        max_completion_length=8,
        weight_sync_steps=1,
        max_policy_staleness=1,
        max_inflight_rollouts=4,
        asynchronous_rollouts=True,
    )
    trainer._online_rollout_idle = True
    trainer._online_rollout_snapshot = None
    trainer._online_generation_lock = None
    trainer._online_rollout_context = None
    trainer._online_multihost_async_warned = False
    trainer._process_count = lambda: 1

    rollout_calls = []

    def fake_provider(*, trainer, state, batch, is_train):
        del trainer, is_train
        rollout_calls.append((int(state.step), int(batch["id"])))
        return {"id": batch["id"], "policy_step": int(state.step)}, {}

    trainer.rollout_provider = fake_provider
    trainer._normalize_provider_batch = lambda batch, **kw: batch

    class _FakeSnap:
        def __init__(self, state):
            self.policy_step = int(state.step)
            self.state = state

        def as_policy_snapshot(self):
            return online_mod.PolicySnapshot(version=self.policy_step, state=self.state)

        def release(self):
            return None

    monkeypatch.setattr(
        online_mod.OwnedPolicySnapshot,
        "from_training_state",
        classmethod(lambda cls, state, **kw: _FakeSnap(state)),
    )

    step_state = [SimpleNamespace(step=jnp.asarray(0, dtype=jnp.int32), esurge_cache_scope_key="test-scope")]

    def prompt_source():
        for i in range(4):
            yield {"id": i}

    collected = []
    for item in trainer.collect_rollouts(
        state=step_state[0],
        batches=prompt_source(),
        is_train=True,
        learner_state=lambda: step_state[0],
    ):
        collected.append(item)
        step_state[0] = SimpleNamespace(
            step=jnp.asarray(int(step_state[0].step) + 1, dtype=jnp.int32),
            esurge_cache_scope_key="test-scope",
        )

    assert len(collected) == 4
    # With weight_sync_steps=1, exactly 4 rollouts generated and 0 discarded!
    assert len(rollout_calls) == 4


# ---------------------------------------------------------------------------
# RL-C5: SDPO EMA ref_state is actually used by sdpo_step
# ---------------------------------------------------------------------------


def test_rl_c5_sdpo_ema_teacher_state_drives_distillation_loss():
    """sdpo_step / _sdpo_step_with_teacher_state must score teacher_ids with teacher_state when supplied."""
    # Use reversed slopes (not a constant shift!) so log_softmax distributions genuinely differ.
    student_w = jnp.linspace(0.5, 4.0, 8, dtype=jnp.float32)
    ema_teacher_w = jnp.linspace(4.0, 0.5, 8, dtype=jnp.float32)
    student_state = _ToyState(student_w)
    ema_state = _ToyState(ema_teacher_w)

    batch = {
        "prompt_ids": jnp.asarray([[1, 2], [2, 3]], dtype=jnp.int32),
        "prompt_mask": jnp.ones((2, 2), dtype=jnp.int32),
        "completion_ids": jnp.asarray([[3, 4], [4, 5]], dtype=jnp.int32),
        "completion_mask": jnp.ones((2, 2), dtype=jnp.int32),
        # Same context as prompt+completion: if teacher_state were ignored, advantage would be 0!
        "teacher_ids": jnp.asarray([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=jnp.int32),
        "teacher_mask": jnp.ones((2, 4), dtype=jnp.int32),
        "num_items_in_batch": jnp.asarray(4.0, dtype=jnp.float32),
    }

    for full_vocab in (False, True):
        for chunk_size in (None, 1):
            _, self_metrics = sdpo_step(
                state=student_state,
                batch=batch,
                num_generations=1,
                teacher_prompt_length=2,
                beta=0.0,
                distillation_type="jsd",
                distillation_weight=1.0,
                logprob_vocab_chunk_size=None,
                max_loss_completion_tokens=None,
                completion_chunk_size=chunk_size,
                full_logit_distillation=full_vocab,
                is_training=True,
            )
            updated_ema_run, ema_metrics = _sdpo_step_with_teacher_state(
                state=student_state,
                batch=batch,
                teacher_state=ema_state,
                num_generations=1,
                teacher_prompt_length=2,
                beta=0.0,
                distillation_type="jsd",
                distillation_weight=1.0,
                logprob_vocab_chunk_size=None,
                max_loss_completion_tokens=None,
                completion_chunk_size=chunk_size,
                full_logit_distillation=full_vocab,
                is_training=True,
            )
            # Without EMA teacher, identical context -> 0 advantage & 0 loss.
            assert float(self_metrics.other_metrics["sdpo/advantage_mean"]) == pytest.approx(0.0, abs=1e-6)
            # With distinct EMA teacher_state, teacher_logps & loss must be nonzero and update student!
            assert abs(float(ema_metrics.other_metrics["sdpo/advantage_mean"])) > 1e-3
            assert not np.allclose(updated_ema_run.graphstate["w"], student_w)


# ---------------------------------------------------------------------------
# RL-C6 & PKD-C9: SDPO left-padded teacher prefix + results.completion_mask
# ---------------------------------------------------------------------------


def test_rl_c6_sdpo_teacher_prefix_left_padded_no_pad_at_boundary():
    """_build_teacher_prefix must eliminate PAD at teacher_prompt_length-1 and mid-sequence PAD holes."""
    trainer = object.__new__(SDPOTrainer)
    trainer._pad_token_id = 0
    trainer.padding_value = 0

    # Row 0: right-padded prompt + short right-padded feedback
    # Row 1: right-padded prompt + empty feedback (dont_reprompt_on_self_success=True)
    # Row 2: left-padded prompt + left-padded feedback
    prompt_ids = jnp.asarray([[10, 11, 0, 0], [20, 21, 22, 0], [0, 0, 30, 31]], dtype=jnp.int32)
    prompt_mask = jnp.asarray([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]], dtype=jnp.int32)
    feedback_ids = jnp.asarray([[40, 41, 0, 0], [0, 0, 0, 0], [0, 0, 50, 51]], dtype=jnp.int32)
    feedback_mask = jnp.asarray([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1]], dtype=jnp.int32)

    prefix_ids, prefix_mask = trainer._build_teacher_prefix(prompt_ids, prompt_mask, feedback_ids, feedback_mask)
    assert prefix_ids.shape == (3, 8)
    assert prefix_ids.tolist() == [
        [0, 0, 0, 0, 10, 11, 40, 41],
        [0, 0, 0, 0, 0, 20, 21, 22],
        [0, 0, 0, 0, 30, 31, 50, 51],
    ]
    assert prefix_mask.tolist() == [
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
    ]
    # Last prefix position is ALWAYS a real token (mask == 1), never PAD!
    assert all(int(prefix_mask[r, -1]) == 1 for r in range(3))


def test_pkd_c9_sdpo_preprocess_intersects_results_completion_mask():
    """SDPOTrainer._preprocess_batch_input must intersect completion_mask with results.completion_mask."""
    trainer = object.__new__(SDPOTrainer)
    trainer.arguments = SDPOConfig(
        max_prompt_length=4,
        max_completion_length=4,
        max_feedback_length=0,
        num_generations=1,
        beta=0.0,
        mask_truncated_completions=False,
        diagnostics_warning_interval=0,
    )
    trainer.arguments.num_loss_tokens_to_skip = 1
    trainer.num_generations = 1
    trainer._effective_feedback_length = 0
    trainer.teacher_prompt_length = 4
    trainer._eos_token_id = [99]  # EOS absent from completion_ids (e.g. stop string!)
    trainer._pad_token_id = 0
    trainer.padding_value = 0
    trainer.train_is_conversational = False
    trainer.eval_is_conversational = False
    trainer.feedback_func = None
    trainer.compute_refmodel_logps = None
    trainer.reward_funcs = [lambda completions, **kw: [1.0] * len(completions)]
    trainer.reward_processing_classes = [None]
    trainer.reward_weights = jnp.asarray([1.0], dtype=jnp.float32)
    trainer.reward_func_names = ["r0"]
    trainer._apply_user_data_collator = lambda b: b
    trainer._extract_reward_batch_sidechannels = lambda b: {}
    trainer._purify_batch = lambda b: b
    trainer._all_gather = lambda x: x
    trainer._log_training_generations_to_wandb = lambda **kw: None
    trainer._make_attn_mask = lambda ids: jnp.ones_like(ids)

    prompt_ids = jnp.asarray([[10, 11, 12, 13]], dtype=jnp.int32)
    prompt_mask = jnp.ones((1, 4), dtype=jnp.int32)
    # Completion has 3 active tokens and 1 right-padded PAD token, NO EOS!
    comp_ids = jnp.asarray([[20, 21, 22, 0]], dtype=jnp.int32)
    rollout_comp_mask = jnp.asarray([[1, 1, 1, 0]], dtype=jnp.int32)

    trainer.rollout = lambda **kw: GenerationResults(
        generation_results=["abc"],
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        sequences=jnp.concatenate([prompt_ids, comp_ids], axis=1),
        completion_ids=comp_ids,
        completion_mask=rollout_comp_mask,
        decoded_prompts=["p"],
        completion_prompts=["p"],
        text=["abc"],
        raw_text=["abc"],
    )

    out_batch, _ = trainer._preprocess_batch_input(
        state=SimpleNamespace(),
        batch={"input_ids": prompt_ids, "attention_mask": prompt_mask},
        is_train=True,
    )
    # PKD-C9: PAD tail at index 3 must be masked out!
    assert out_batch["completion_mask"].tolist() == [[1, 1, 1, 0]]
    # RL-H7 / PKD-H14: num_items_in_batch must account for num_loss_tokens_to_skip=1 -> 3 - 1 = 2!
    assert float(out_batch["num_items_in_batch"]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# RL-C7: AgenticMoshPit canonical prefix & reasoning template masking
# ---------------------------------------------------------------------------


def test_rl_c7_agentic_moshpit_reasoning_template_masks_only_assistant_tokens():
    """Reasoning templates that strip <think> in history or inject <think>\\n must not mis-mask observations."""
    import re

    class _ReasoningTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None, **kw):
            del tools, kw
            parts = []
            for idx, m in enumerate(messages):
                content = m["content"]
                # Mimic Qwen3 / DeepSeek-R1: strip <think> blocks from non-final assistant turns!
                if m["role"] == "assistant" and idx < len(messages) - 1:
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                parts.append(f"<{m['role']}>{content}</{m['role']}>")
            if add_generation_prompt:
                parts.append("<assistant><think>\n")
            rendered = "".join(parts)
            if tokenize:
                return {"input_ids": [[ord(c) for c in rendered]]}
            return rendered

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [ord(c) for c in text]

    class _TwoTurnEnv:
        def __init__(self):
            self.step_count = 0

        def reset(self, seed=None):
            del seed
            self.step_count = 0
            return SimpleNamespace(observation="Question?", info={})

        def step(self, action):
            del action
            self.step_count += 1
            if self.step_count == 1:
                return SimpleNamespace(observation="OBS_TOKEN", reward=0.0, terminated=False, truncated=False, info={})
            return SimpleNamespace(observation="", reward=1.0, terminated=True, truncated=False, info={})

        def close(self):
            return None

    tok = _ReasoningTokenizer()
    manager = RolloutManager(tokenizer=tok, max_steps=3, max_seq_length=2048)
    turn_texts = ["plan1</think>act1", "plan2</think>final"]
    # Use distinctive token IDs (> 1000) so we prove exact completion_ids/sampling_per_token_logps
    # aliases from last_generation_tokens are preserved rather than re-encoded via tok.encode().
    turn_token_ids = [
        [1001 + i for i in range(len(turn_texts[0]))],
        [2001 + i for i in range(len(turn_texts[1]))],
    ]
    turn_logps = [
        [-0.25] * len(turn_texts[0]),
        [-0.75] * len(turn_texts[1]),
    ]
    prompts_seen = []

    def gen_fn(prompts, **kw):
        del kw
        turn_idx = len(prompts_seen)
        prompts_seen.extend(prompts)
        txt = turn_texts[turn_idx]
        gen_fn.last_generation_tokens = [
            {
                "completion_ids": turn_token_ids[turn_idx],
                "sampling_per_token_logps": turn_logps[turn_idx],
            }
        ]
        gen_fn.last_generation_metadata = [{"text": txt, "raw_text": txt, "reasoning": None, "tool_calls": None}]
        return [txt]

    trajs = manager.run_grouped_episodes(
        env_factory=_TwoTurnEnv,
        generate_fn=gen_fn,
        group_size=1,
        num_groups=1,
    )
    traj = trajs[0]
    masked_ids = [
        int(t) for t, m in zip(traj.input_ids.tolist(), traj.response_mask.tolist(), strict=True) if m == 1
    ]
    # Exact sampled token IDs preserved for both assistant turns — zero observation/user tokens masked!
    assert masked_ids == turn_token_ids[0] + turn_token_ids[1]
    assert traj.sampling_logps is not None
    masked_logps = [
        float(lp) for lp, m in zip(traj.sampling_logps.tolist(), traj.response_mask.tolist(), strict=True) if m == 1
    ]
    assert masked_logps == pytest.approx(turn_logps[0] + turn_logps[1])


def test_rl_c7_sentinel_failure_clears_running_token_ids():
    """When sentinel transition extraction fails, running_token_ids must be cleared so obs isn't dropped."""

    class _NonSentinelTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None, **kw):
            del tools, kw
            # Strip sentinel tokens so _extract_transition_text returns sentinel_ok=False!
            rendered = "|".join(f"{m['role']}={m['content'].replace('__EASYDEL_AGENTIC_', '')}" for m in messages)
            if add_generation_prompt:
                rendered += "|assistant="
            if tokenize:
                return {"input_ids": [[ord(c) for c in rendered]]}
            return rendered

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [ord(c) for c in text]

    class _TwoStepObsEnv:
        def __init__(self):
            self.calls = 0

        def reset(self, seed=None):
            del seed
            self.calls = 0
            return SimpleNamespace(observation="Q0", info={})

        def step(self, action):
            del action
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(observation="CRITICAL_OBS", reward=0.0, terminated=False, truncated=False, info={})
            return SimpleNamespace(observation="", reward=1.0, terminated=True, truncated=False, info={})

        def close(self):
            return None

    seen_calls: list[tuple[list[str], object]] = []

    def gen_fn(prompts, prompt_token_ids=None, **kw):
        del kw
        seen_calls.append((list(prompts), prompt_token_ids))
        txt = "a1" if len(seen_calls) == 1 else "a2"
        gen_fn.last_generation_tokens = [
            {"completion_ids": [ord(c) for c in txt], "sampling_per_token_logps": [-0.5, -0.5]}
        ]
        gen_fn.last_generation_metadata = [{"text": txt, "raw_text": txt, "reasoning": None, "tool_calls": None}]
        return [txt]

    gen_fn._accepts_prompt_token_ids = True

    manager = RolloutManager(tokenizer=_NonSentinelTokenizer(), max_steps=3, max_seq_length=2048)
    manager.run_grouped_episodes(
        env_factory=_TwoStepObsEnv,
        generate_fn=gen_fn,
        group_size=1,
        num_groups=1,
    )
    assert len(seen_calls) == 2
    # Turn 0 passes running prompt_token_ids (proving pass_prompt_ids=True was active).
    assert seen_calls[0][1] is not None
    # Turn 1 must NOT pass stale running_token_ids (missing CRITICAL_OBS); prompt_token_ids must be None
    # and the text prompt must carry CRITICAL_OBS!
    assert "CRITICAL_OBS" in seen_calls[1][0][0]
    assert seen_calls[1][1] is None


# ---------------------------------------------------------------------------
# RL-C9 & RL-H15: NeMoGym Grain sidechannels + AgenticMoshPit is_train & knob guards
# ---------------------------------------------------------------------------


def test_rl_c9_h15_nemo_gym_grain_sidechannels_and_step_result_normalization():
    """NeMoGymTrainer must unpack Grain ndarray/PyTree sidechannels and validate rewards cleanly."""
    trainer = object.__new__(NeMoGymTrainer)
    trainer.arguments = SimpleNamespace(
        metadata_key="metadata",
        agent_ref_key="agent_ref",
    )

    # 1. Grain batch: 1D np.ndarray metadata + PyTree-transposed dict[str, np.ndarray] agent_ref
    grain_batch = {
        "input_ids": np.zeros((2, 4), dtype=np.int32),
        "metadata": np.array(['{"task":"a"}', '{"task":"b"}'], dtype=object),
        "agent_ref": {
            "name": np.array(["agent_a", "agent_b"], dtype=object),
            "version": np.array([1, 2], dtype=np.int32),
        },
    }
    metas, refs = trainer._extract_nemo_sidechannels(grain_batch)
    assert metas == [{"task": "a"}, {"task": "b"}]
    assert refs == [{"name": "agent_a", "version": 1}, {"name": "agent_b", "version": 2}]

    # 2. metadata=None must NOT drop agent_ref!
    only_ref_batch = {"agent_ref": [{"name": "a1"}, {"name": "a2"}]}
    metas2, refs2 = trainer._extract_nemo_sidechannels(only_ref_batch)
    assert metas2 == []
    assert refs2 == [{"name": "a1"}, {"name": "a2"}]

    # 3. Uncollated list[dict] batch works without crashing
    list_batch = [
        {"metadata": {"reward": 0.1}, "agent_ref": {"name": "x"}},
        {"metadata": {"reward": 0.2}, "agent_ref": {"name": "y"}},
    ]
    metas3, refs3 = trainer._extract_nemo_sidechannels(list_batch)
    assert metas3 == [{"reward": 0.1}, {"reward": 0.2}]
    assert refs3 == [{"name": "x"}, {"name": "y"}]

    # 4. _normalize_step_result handles reward=None cleanly and rejects bad tuple lengths
    norm = _normalize_step_result(("obs", None, True, {}))
    assert norm[1] == 0.0
    with pytest.raises(ValueError):
        _normalize_step_result((1, 2, 3, 4, 5, 6))


# ---------------------------------------------------------------------------
# RL-C10: CodeVerifier sys.exit(0) / os._exit(0) exploit + >128KiB + warning
# ---------------------------------------------------------------------------


def test_rl_c10_code_verifier_defeats_exit_exploits_and_handles_large_scripts():
    """CodeVerifier must reject sys.exit(0)/os._exit(0) exploits and execute >128KiB scripts."""
    verifier = CodeVerifier(timeout=5.0, use_jail="off")
    tests = {"tests": ["assert 1 == 2"]}

    assert verifier(completions=["```python\nimport sys\nsys.exit(0)\n```"], batch=tests) == [0.0]
    assert verifier(completions=["```python\nimport os\nos._exit(0)\n```"], batch=tests) == [0.0]

    # Script larger than Linux MAX_ARG_STRLEN (128 KiB) must succeed via staged file, not E2BIG!
    pad_comment = "# " + ("a" * 140_000) + "\n"
    large_code = f"```python\n{pad_comment}x = 42\n```"
    assert verifier(completions=[large_code], batch={"tests": ["assert x == 42"]}) == [1.0]

    # use_jail="auto" warns explicitly when no jail binary is available
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sb = Sandbox(timeout=2.0, use_jail="auto")
        if sb._jail is None:
            assert any(issubclass(w.category, RuntimeWarning) for w in caught)


# ---------------------------------------------------------------------------
# RL-C11: CausalLMWithValueHead propagates _esurge_cache_scope_key to self.model
# ---------------------------------------------------------------------------


def test_rl_c11_causal_lm_with_value_head_propagates_esurge_cache_scope_key():
    """Setting _esurge_cache_scope_key on CausalLMWithValueHead must propagate to self.model."""

    class _InnerModel(EasyGenerationMixin):
        def __init__(self):
            self.config = SimpleNamespace()
            self.calls = []

        def get_esurge(self, *args, **kwargs):
            self.calls.append(("get_esurge", self._esurge_cache_scope()))
            return self._esurge_cache_scope()

        def pause_esurge(self, *args, **kwargs):
            self.calls.append(("pause_esurge", self._esurge_cache_scope()))
            return self._esurge_cache_scope()

    wrapper = object.__new__(CausalLMWithValueHead)
    inner = _InnerModel()
    object.__setattr__(wrapper, "model", inner)
    # Mirror EasyDeLState.model setting _esurge_cache_scope_key only on the outer wrapper
    object.__setattr__(wrapper, "_esurge_cache_scope_key", "shared-ppo-state-scope-123")

    scope_get = wrapper.get_esurge()
    scope_pause = wrapper.pause_esurge()
    assert scope_get == "shared-ppo-state-scope-123"
    assert scope_pause == "shared-ppo-state-scope-123"
    assert getattr(inner, "_esurge_cache_scope_key", None) == "shared-ppo-state-scope-123"


# ---------------------------------------------------------------------------
# RL-H1: GRPO 2D advantages + max_loss_completion_tokens & compute_advantages_agentic
# ---------------------------------------------------------------------------


def test_rl_h1_grpo_2d_advantages_truncation_and_agentic_reinforce_per_step():
    """grpo_step must slice 2D advantages under max_loss_completion_tokens; agentic_reinforce normalizes per step."""
    init_w = jnp.linspace(0.2, 0.9, 8, dtype=jnp.float32)
    batch = {
        "prompt_ids": jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int32),
        "prompt_mask": jnp.ones((2, 2), dtype=jnp.int32),
        "completion_ids": jnp.asarray([[1, 2, 3, 4, 5], [2, 3, 4, 5, 1]], dtype=jnp.int32),
        "completion_mask": jnp.ones((2, 5), dtype=jnp.int32),
        "advantages": jnp.asarray([[1.0, 0.5, 0.2, -0.1, -0.5], [-1.0, -0.5, -0.2, 0.1, 0.5]], dtype=jnp.float32),
        "num_items_in_batch": jnp.asarray(10.0, dtype=jnp.float32),
    }
    for chunk_size in (None, 1):
        metrics = grpo_step(
            state=_ToyState(init_w),
            batch=batch,
            num_generations=1,
            beta=0.0,
            max_loss_completion_tokens=3,
            completion_chunk_size=chunk_size,
            is_training=False,
        )
        assert jnp.isfinite(metrics.loss)

    # Two identical 5-step sparse-reward rollouts with gamma=0.95 must have 0.0 advantage at every step!
    step_rewards = [np.asarray([0.0, 0.0, 0.0, 0.0, 1.0]), np.asarray([0.0, 0.0, 0.0, 0.0, 1.0])]
    response_masks = np.ones((2, 5), dtype=np.int32)
    turn_ids = np.asarray([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=np.int32)
    advs = compute_advantages_agentic(
        step_rewards_list=step_rewards,
        response_masks=response_masks,
        group_size=2,
        gamma=0.95,
        turn_ids=turn_ids,
    )
    np.testing.assert_allclose(np.asarray(advs), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# RL-H2: Eval 1/K inflation fix + 0 < N < K normalizer clamp + scheduled weight
# ---------------------------------------------------------------------------


def test_rl_h2_eval_no_k_inflation_and_small_count_normalizer():
    """Eval loss must not be scaled by K or fail non-divisible batch sizes; 0 < N < K normalizer must not clamp."""
    # _global_token_normalizer and _token_normalizer: clamp BEFORE dividing by K
    assert float(_global_token_normalizer(jnp.asarray(2.0), jnp.asarray(1.0), False, 4, is_training=True)) == (
        pytest.approx(0.5)
    )
    assert float(_global_token_normalizer(jnp.asarray(2.0), jnp.asarray(1.0), False, 4, is_training=False)) == (
        pytest.approx(2.0)
    )
    assert float(_token_normalizer({"num_items_in_batch": jnp.asarray(2.0)}, jnp.asarray(1.0), 4, True)) == (
        pytest.approx(0.5)
    )
    assert float(_token_normalizer({"num_items_in_batch": jnp.asarray(2.0)}, jnp.asarray(1.0), 1, True)) == (
        pytest.approx(2.0)
    )

    # Non-divisible eval batch (3 rows, K=4) must run and match K=1 identically across GRPO & SDPO!
    init_w = jnp.linspace(0.2, 0.9, 8, dtype=jnp.float32)
    grpo_eval_batch = {
        "prompt_ids": jnp.asarray([[1, 2], [2, 3], [3, 4]], dtype=jnp.int32),
        "prompt_mask": jnp.ones((3, 2), dtype=jnp.int32),
        "completion_ids": jnp.asarray([[1, 2], [2, 3], [3, 4]], dtype=jnp.int32),
        "completion_mask": jnp.ones((3, 2), dtype=jnp.int32),
        "advantages": jnp.asarray([1.0, -0.5, 0.25], dtype=jnp.float32),
        "num_items_in_batch": jnp.asarray(6.0, dtype=jnp.float32),
    }
    m1 = grpo_step(state=_ToyState(init_w), batch=grpo_eval_batch, num_generations=1, beta=0.0, gradient_accumulation_steps=1, is_training=False)
    m4 = grpo_step(state=_ToyState(init_w), batch=grpo_eval_batch, num_generations=1, beta=0.0, gradient_accumulation_steps=4, is_training=False)
    assert float(m4.loss) == pytest.approx(float(m1.loss), abs=1e-6)

    sdpo_eval_batch = {
        "prompt_ids": jnp.asarray([[1, 2], [2, 3], [3, 4]], dtype=jnp.int32),
        "prompt_mask": jnp.ones((3, 2), dtype=jnp.int32),
        "completion_ids": jnp.asarray([[1, 2], [2, 3], [3, 4]], dtype=jnp.int32),
        "completion_mask": jnp.ones((3, 2), dtype=jnp.int32),
        "teacher_ids": jnp.asarray([[1, 2, 5, 1, 2], [2, 3, 5, 2, 3], [3, 4, 5, 3, 4]], dtype=jnp.int32),
        "teacher_mask": jnp.ones((3, 5), dtype=jnp.int32),
        "num_items_in_batch": jnp.asarray(6.0, dtype=jnp.float32),
    }
    s1 = sdpo_step(
        state=_ToyState(init_w),
        batch=sdpo_eval_batch,
        num_generations=1,
        teacher_prompt_length=3,
        beta=0.0,
        distillation_type="kl",
        distillation_weight=1.0,
        logprob_vocab_chunk_size=None,
        max_loss_completion_tokens=None,
        completion_chunk_size=None,
        gradient_accumulation_steps=1,
        is_training=False,
    )
    s4 = sdpo_step(
        state=_ToyState(init_w),
        batch=sdpo_eval_batch,
        num_generations=1,
        teacher_prompt_length=3,
        beta=0.0,
        distillation_type="kl",
        distillation_weight=1.0,
        logprob_vocab_chunk_size=None,
        max_loss_completion_tokens=None,
        completion_chunk_size=None,
        gradient_accumulation_steps=4,
        is_training=False,
    )
    assert float(s4.loss) == pytest.approx(float(s1.loss), abs=1e-6)

    # Actor & Critic eval non-divisible batch (3 rows, K=4) matches K=1 identically
    # Target arrays align with next-token positions (width seq_len - 1 = 3).
    actor_eval_batch = {
        "input_ids": jnp.asarray([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]], dtype=jnp.int32),
        "attention_mask": jnp.ones((3, 4), dtype=jnp.int32),
        "action_mask": jnp.asarray([[0, 1, 1], [0, 1, 1], [0, 1, 1]], dtype=jnp.float32),
        "behavior_logps": jnp.full((3, 3), -2.0, dtype=jnp.float32),
        "advantages": jnp.asarray([[0, 1.0, -0.5], [0, 0.5, 0.2], [0, -1.0, 0.3]], dtype=jnp.float32),
        "num_items_in_batch": jnp.asarray(6.0, dtype=jnp.float32),
    }
    a1 = actor_step(
        state=_ToyState(init_w),
        batch=actor_eval_batch,
        algorithm="ppo",
        dis_epsilon_low=0.2,
        dis_epsilon_high=0.2,
        cliprange=0.2,
        entropy_coef=0.0,
        logprob_vocab_chunk_size=None,
        gradient_accumulation_steps=1,
        is_training=False,
    )
    a4 = actor_step(
        state=_ToyState(init_w),
        batch=actor_eval_batch,
        algorithm="ppo",
        dis_epsilon_low=0.2,
        dis_epsilon_high=0.2,
        cliprange=0.2,
        entropy_coef=0.0,
        logprob_vocab_chunk_size=None,
        gradient_accumulation_steps=4,
        is_training=False,
    )
    assert float(a4.loss) == pytest.approx(float(a1.loss), abs=1e-6)

    critic_eval_batch = {
        "input_ids": actor_eval_batch["input_ids"],
        "attention_mask": actor_eval_batch["attention_mask"],
        "action_mask": actor_eval_batch["action_mask"],
        "critic_returns": jnp.ones((3, 3), dtype=jnp.float32),
        "old_values": jnp.zeros((3, 3), dtype=jnp.float32),
        "num_items_in_batch": jnp.asarray(6.0, dtype=jnp.float32),
    }
    c_state = _ToyState(init_w, value_weight=jnp.asarray(0.5, dtype=jnp.float32))
    c1 = critic_step(state=c_state, batch=critic_eval_batch, cliprange_value=0.2, gradient_accumulation_steps=1, is_training=False)
    c4 = critic_step(state=c_state, batch=critic_eval_batch, cliprange_value=0.2, gradient_accumulation_steps=4, is_training=False)
    assert float(c4.loss) == pytest.approx(float(c1.loss), abs=1e-6)

    # GRPO scheduled microbatch weight helper returns exact token count for dapo
    weight_fn = _make_grpo_microbatch_weight({"loss_type": "dapo", "max_loss_completion_tokens": None})
    assert float(weight_fn(None, grpo_eval_batch)) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# RL-H3 & RL-H16 & RL-H17 & RL-M4: esurge_generation & esurge_rollout
# ---------------------------------------------------------------------------


def test_rl_h3_h16_h17_m4_esurge_generation_and_rollout_contracts():
    """Verify empty completion i>0 isolation, resolve_sampling extras, engine_kwargs sizing, and truncation sync."""
    # RL-H3: empty completion i=1 must NOT inherit output.accumulated_text from completion 0
    comp0 = SimpleNamespace(
        text="correct answer",
        raw_text="correct answer",
        token_ids=[10, 11],
        logprobs=None,
        reasoning_content=None,
        tool_calls=None,
        finish_reason="stop",
    )
    comp1 = SimpleNamespace(
        text="",
        raw_text="",
        token_ids=[],
        logprobs=None,
        reasoning_content=None,
        tool_calls=None,
        finish_reason="stop",
    )
    req_out = SimpleNamespace(
        prompt="hello",
        prompt_token_ids=[1, 2, 3],  # RL-M4: flat list[int] supported too!
        outputs=[comp0, comp1],
        accumulated_text="correct answer",
        raw_accumulated_text="correct answer",
        get_text=lambda: "correct answer",
    )
    res = esurge_generation.outputs_to_arrays(
        [req_out],
        input_ids=None,
        attention_mask=None,
        return_prompts=None,
        pad_token_id=0,
        max_seq_len=8,
        max_new_tokens=4,
    )
    assert res.output_records == ["correct answer", ""]
    assert res.raw_output_records == ["correct answer", ""]

    # RL-H17: RequestAdmission prompt truncation reconciles caller prompt_ids instead of silent drift
    trunc_req_out = SimpleNamespace(
        prompt="hello",
        prompt_token_ids=[[3, 4]],  # Admission kept suffix [3, 4] out of [1, 2, 3, 4]
        outputs=[comp0],
        accumulated_text="correct answer",
        raw_accumulated_text="correct answer",
        get_text=lambda: "correct answer",
    )
    res_trunc = esurge_generation.outputs_to_arrays(
        [trunc_req_out],
        input_ids=jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32),
        attention_mask=jnp.ones((1, 4), dtype=jnp.int32),
        return_prompts=None,
        pad_token_id=0,
        max_seq_len=8,
        max_new_tokens=4,
    )
    assert res_trunc.prompt_ids.tolist() == [[0, 0, 3, 4]]
    assert res_trunc.prompt_mask.tolist() == [[0, 0, 1, 1]]

    # RL-H16: resolve_sampling forwards PPOConfig.stop_token_id, GRPOConfig.min_p, generate_kwargs & clamps temp
    ppo_cfg = PPOConfig(max_prompt_length=16, max_completion_length=16, stop_token_id=99, temperature=0.8)
    sp_ppo = esurge_generation.resolve_sampling(ppo_cfg, overrides={"temperature": 1e-4}, generate_kwargs={"seed": 7})
    assert 99 in (sp_ppo.stop_token_ids or [])
    assert sp_ppo.seed == 7
    assert sp_ppo.temperature == pytest.approx(0.01)  # < 1e-2 clamp ran via __post_init__

    grpo_cfg = GRPOConfig(max_prompt_length=16, max_completion_length=16, generation_extra_kwargs={"min_p": 0.15})
    sp_grpo = esurge_generation.resolve_sampling(grpo_cfg, generate_kwargs={"stop": ["END"]})
    assert sp_grpo.min_p == pytest.approx(0.15)
    assert sp_grpo.stop == ["END"]

    # RL-H17 & RL-M4: engine_kwargs handles max_length=None, num_generations_eval > num_generations, and agentic sizing
    agentic_cfg = AgenticMoshPitConfig(
        max_prompt_length=100,
        max_completion_length=100,
        num_env_groups=3,
        group_size=4,
        total_batch_size=2,
        num_generations=2,
    )
    ek = esurge_generation.engine_kwargs(agentic_cfg)
    assert ek["max_num_seqs"] >= 12  # num_env_groups * group_size = 12
    assert ek["max_cache_tokens"] % 128 == 0

    none_len_cfg = GRPOConfig(
        max_prompt_length=64,
        max_completion_length=64,
        total_batch_size=2,
        eval_batch_size=2,
        num_generations=2,
        num_generations_eval=6,
    )
    none_len_cfg.max_length = None
    ek_none = esurge_generation.engine_kwargs(none_len_cfg)
    assert ek_none["max_num_seqs"] >= 12  # eval_batch_size * num_generations_eval = 12

    # RL-M4: esurge_rollout.generate_rollout_completions attaches 1D prompt_ids per row when n > 1
    fake_trainer = SimpleNamespace(
        rollout=lambda **kw: SimpleNamespace(
            generation_results=["a1", "a2", "b1", "b2"],
            text=["a1", "a2", "b1", "b2"],
            raw_text=["a1", "a2", "b1", "b2"],
            prompt_ids=np.asarray([[1, 2], [3, 4]], dtype=np.int32),
            completion_ids=np.asarray([[10], [11], [20], [21]], dtype=np.int32),
            sequences=np.asarray([[1, 2, 10], [1, 2, 11], [3, 4, 20], [3, 4, 21]], dtype=np.int32),
            completion_mask=np.ones((4, 1), dtype=np.int32),
            sampling_per_token_logps=None,
            reasoning=None,
            tool_calls=None,
        )
    )
    outs = generate_rollout_completions(fake_trainer, ["p0", "p1"])
    assert len(outs) == 4
    assert np.asarray(outs[0]["prompt_ids"]).tolist() == [1, 2]
    assert np.asarray(outs[1]["prompt_ids"]).tolist() == [1, 2]
    assert np.asarray(outs[2]["prompt_ids"]).tolist() == [3, 4]
    assert np.asarray(outs[3]["prompt_ids"]).tolist() == [3, 4]


# ---------------------------------------------------------------------------
# RL-H4: GRPO loss variants (luspo, dr_grpo, cispo, vespo, dppo)
# ---------------------------------------------------------------------------


def test_rl_h4_grpo_loss_variants_luspo_dr_grpo_cispo_vespo_dppo():
    """Verify luspo masking/parity, dr_grpo sequence parity, cispo 1+eps_high, vespo mask, and dppo bf16/topk."""
    init_w = jnp.linspace(0.2, 0.9, 8, dtype=jnp.float32)
    batch = {
        "prompt_ids": jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int32),
        "prompt_mask": jnp.ones((2, 2), dtype=jnp.int32),
        "completion_ids": jnp.asarray([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=jnp.int32),
        "completion_mask": jnp.asarray([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=jnp.int32),
        "advantages": jnp.asarray([1.0, -1.0], dtype=jnp.float32),
        "num_items_in_batch": jnp.asarray(5.0, dtype=jnp.float32),
    }

    # 1. luspo: chunked and non-chunked match under sequence IS, and PAD tokens are masked out
    m_luspo_full = grpo_step(
        state=_ToyState(init_w),
        batch=batch,
        num_generations=1,
        beta=0.0,
        loss_type="luspo",
        importance_sampling_level="sequence",
        completion_chunk_size=None,
        is_training=False,
    )
    m_luspo_chunk = grpo_step(
        state=_ToyState(init_w),
        batch=batch,
        num_generations=1,
        beta=0.0,
        loss_type="luspo",
        importance_sampling_level="sequence",
        completion_chunk_size=1,
        is_training=False,
    )
    assert float(m_luspo_full.loss) == pytest.approx(float(m_luspo_chunk.loss), abs=1e-5)

    # 2. dr_grpo: sequence IS matches chunked vs non-chunked (no T-times inflation!)
    m_dr_full = grpo_step(
        state=_ToyState(init_w),
        batch=batch,
        num_generations=1,
        beta=0.0,
        loss_type="dr_grpo",
        importance_sampling_level="sequence",
        completion_chunk_size=None,
        is_training=False,
    )
    m_dr_chunk = grpo_step(
        state=_ToyState(init_w),
        batch=batch,
        num_generations=1,
        beta=0.0,
        loss_type="dr_grpo",
        importance_sampling_level="sequence",
        completion_chunk_size=1,
        is_training=False,
    )
    assert float(m_dr_full.loss) == pytest.approx(float(m_dr_chunk.loss), abs=1e-5)

    # 3. cispo: on-policy (coef_1 == 1.0) with epsilon_high=0.2 must NOT clip (clip ratio == 0.0!)
    m_cispo = grpo_step(
        state=_ToyState(init_w),
        batch=batch,
        num_generations=1,
        beta=0.0,
        loss_type="cispo",
        epsilon=0.2,
        epsilon_high=0.2,
        is_training=False,
    )
    assert float(m_cispo.other_metrics["cispo_clip_ratio"]) == pytest.approx(0.0, abs=1e-6)

    # 4. vespo: masked pad zeros in importance_sampling_ratio must NOT collapse sequence weights!
    log_w = jnp.zeros((1, 4), dtype=jnp.float32)
    mask = jnp.asarray([[1.0, 1.0, 0.0, 0.0]], dtype=jnp.float32)
    is_with_zero_pad = jnp.asarray([[1.0, 1.0, 0.0, 0.0]], dtype=jnp.float32)
    is_with_one_pad = jnp.asarray([[1.0, 1.0, 1.0, 1.0]], dtype=jnp.float32)
    w_zero_pad = _vespo_gamma_weights(
        advantages=jnp.asarray([1.0]),
        log_ratio_per_token=log_w,
        mask=mask,
        importance_sampling_ratio=is_with_zero_pad,
    )
    w_one_pad = _vespo_gamma_weights(
        advantages=jnp.asarray([1.0]),
        log_ratio_per_token=log_w,
        mask=mask,
        importance_sampling_ratio=is_with_one_pad,
    )
    np.testing.assert_allclose(w_zero_pad, w_one_pad, atol=1e-6)
    assert float(w_zero_pad[0, 0]) == pytest.approx(1.0, abs=1e-5)

    # 5. dppo: bfloat16 binary_kl produces finite mask (no NaN), and topk slices under max_loss_completion_tokens
    bf16_lp = jnp.asarray([[-0.0001, -2.0]], dtype=jnp.bfloat16)
    bf16_old = jnp.asarray([[-5.0, -0.0001]], dtype=jnp.bfloat16)
    keep = _compute_dppo_divergence_mask(
        per_token_logps=bf16_lp,
        sampling_per_token_logps=bf16_old,
        advantages=jnp.asarray([[1.0, 1.0]], dtype=jnp.float32),
        completion_mask=jnp.ones((1, 2), dtype=jnp.float32),
        divergence_type="binary_kl",
        epsilon_low=0.5,
        epsilon_high=0.5,
    )
    assert jnp.all(jnp.isfinite(keep))
    assert float(keep[0, 0]) == 0.0  # large divergence actually gets masked, not silently ignored via NaN!

    dppo_batch = {
        **batch,
        "sampling_per_token_logps": jnp.full((2, 4), -2.0, dtype=jnp.float32),
        "sampling_topk_indices": jnp.zeros((2, 4, 2), dtype=jnp.int32),
        "sampling_topk_logps": jnp.full((2, 4, 2), -1.0, dtype=jnp.float32),
    }
    m_dppo = grpo_step(
        state=_ToyState(init_w),
        batch=dppo_batch,
        num_generations=1,
        beta=0.0,
        loss_type="dppo",
        max_loss_completion_tokens=2,
        dppo_divergence_type="topk_tv",
        is_training=False,
    )
    assert jnp.isfinite(m_dppo.loss)


# ---------------------------------------------------------------------------
# RL-H5 & RL-H6 & RL-M2 & RL-M3: GRPO reuse queue, replay old_logps, LOO/normalize_then_sum
# ---------------------------------------------------------------------------


def test_rl_h5_h6_m2_m3_grpo_reuse_replay_and_advantage_math():
    """Verify reuse dataloader pause, steps_per_generation slicing, replay old_logps, and LOO/normalize_then_sum."""
    # RL-H5: _get_next_batch pauses dataloader while buffered reuse batches remain
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = GRPOConfig(
        max_prompt_length=4,
        max_completion_length=4,
        num_iterations=2,
        steps_per_generation=1,
    )
    trainer.num_iterations = 2
    trainer.steps_per_generation = 1
    trainer._grpo_fetch_cycle_pos = 0
    trainer._grpo_cached_raw_batch = None

    pulls = [0]

    class _FakeLoader:
        def __iter__(self):
            for i in range(5):
                pulls[0] += 1
                yield {"input_ids": jnp.full((1, 4), i, dtype=jnp.int32), "attention_mask": jnp.ones((1, 4), dtype=jnp.int32)}

    loader = _FakeLoader()
    trainer.dataloader_train = loader
    dl_iter = iter(loader)
    b0, dl_iter = trainer._get_next_batch(dl_iter, loader)
    assert pulls[0] == 1
    b1, dl_iter = trainer._get_next_batch(dl_iter, loader)
    assert pulls[0] == 1  # Dataloader was NOT pulled during reuse step!
    assert int(b1["input_ids"][0, 0]) == int(b0["input_ids"][0, 0])
    b2, dl_iter = trainer._get_next_batch(dl_iter, loader)
    assert pulls[0] == 2  # Next cycle pulls batch 1 (zero prompts discarded!)
    assert int(b2["input_ids"][0, 0]) == 1

    # RL-H6: _should_compute_old_per_token_logps active for DPPO, off_policy_mask_threshold, and ReplayBuffer
    trainer.arguments.num_iterations = 1
    trainer.num_iterations = 1
    trainer.steps_per_generation = 1
    trainer.arguments.off_policy_mask_threshold = 0.5
    assert (
        trainer._should_compute_old_per_token_logps(
            is_train=True,
            needs_dppo=False,
            needs_esurge_correction=False,
            needs_off_policy_mask=True,
        )
        is True
    )

    replay_trainer = object.__new__(GRPOWithReplayBufferTrainer)
    replay_trainer.arguments = GRPOConfig(max_prompt_length=4, max_completion_length=4, num_iterations=1)
    replay_trainer.num_iterations = 1
    replay_trainer.steps_per_generation = 1
    replay_trainer.replay_buffer = _ReplayBuffer(max_size=16, seed=0)
    assert (
        replay_trainer._should_compute_old_per_token_logps(
            is_train=True,
            needs_dppo=False,
            needs_esurge_correction=False,
            needs_off_policy_mask=False,
        )
        is True
    )

    # RL-M2: _ReplayBuffer.sample handles sparse positive scores without ValueError
    buf = _ReplayBuffer(max_size=10, seed=0)
    buf.add(0.0, {"id": 0})
    buf.add(0.0, {"id": 1})
    buf.add(1.0, {"id": 2})
    sampled = buf.sample(3)
    assert len(sampled) == 3

    # RL-M3: NaN-aware _leave_one_out_group_baseline divides by valid sibling count (not G - 1)
    vals = jnp.asarray([[1.0, 3.0, jnp.nan]], dtype=jnp.float32)
    baseline, _ = _leave_one_out_group_baseline(vals, axis=1)
    # Sibling of 1.0 is only 3.0 -> baseline=3.0 (not 3.0 / 2 = 1.5!)
    assert float(baseline[0, 0]) == pytest.approx(3.0, abs=1e-6)
    assert float(baseline[0, 1]) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# RL-H7 & RL-M8 & RL-L1 (SDPO): top-k JSD, rich feedback, max_reprompt_len, chunked full-vocab, JSD 0.5x
# ---------------------------------------------------------------------------


def test_rl_h7_m8_l1_sdpo_topk_jsd_rich_feedback_and_chunked_full_vocab():
    """Verify top-k JSD math, rich feedback combining solution+feedback, max_reprompt_len cap, and chunked full-vocab."""
    student_logits = jnp.asarray([[[0.2, 0.8, -0.4, 0.1]]], dtype=jnp.float32)
    teacher_logits = jnp.asarray([[[1.0, 0.3, 0.2, -0.5]]], dtype=jnp.float32)
    comp_ids = jnp.asarray([[1]], dtype=jnp.int32)

    # RL-H7: top-k JSD differs from top-k KL and equals 0 when student == teacher
    jsd_tail, _, _ = _full_vocab_sdpo_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        completion_ids=comp_ids,
        distillation_type="jsd",
        distillation_topk=2,
        distillation_add_tail=True,
        distillation_clip=None,
    )
    kl_tail, _, _ = _full_vocab_sdpo_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        completion_ids=comp_ids,
        distillation_type="kl",
        distillation_topk=2,
        distillation_add_tail=True,
        distillation_clip=None,
    )
    jsd_zero, _, _ = _full_vocab_sdpo_loss(
        student_logits=teacher_logits,
        teacher_logits=teacher_logits,
        completion_ids=comp_ids,
        distillation_type="jsd",
        distillation_topk=2,
        distillation_add_tail=True,
        distillation_clip=None,
    )
    assert float(jsd_tail[0, 0]) > 0.0
    assert float(jsd_tail[0, 0]) < float(kl_tail[0, 0])
    assert float(jsd_zero[0, 0]) == pytest.approx(0.0, abs=1e-6)

    # RL-H7: _get_rich_feedback combines solution AND feedback when environment_feedback_only_without_solution=False
    sdpo_stub = object.__new__(SDPOTrainer)
    sdpo_stub.arguments = SDPOConfig(
        max_prompt_length=16,
        max_completion_length=16,
        max_feedback_length=16,
        use_successful_as_teacher=True,
        environment_feedback_only_without_solution=False,
    )
    sdpo_stub.feedback_func = lambda prompts, completions, rewards: ["err_msg" if r <= 0 else "" for r in rewards]
    fb_texts = sdpo_stub._get_rich_feedback(
        completion_prompts=["p1", "p1"],
        completions=["good_sol", "bad_sol"],
        rewards=jnp.asarray([1.0, 0.0]),
        generation_factor=2,
    )
    assert "good_sol" in fb_texts[1]
    assert "err_msg" in fb_texts[1]
    # RL-L1: empty completion list guard does not IndexError
    assert sdpo_stub._get_rich_feedback([], [], jnp.asarray([]), generation_factor=2) == []

    # RL-M8: _configure_teacher_context caps prompt + feedback <= max_reprompt_len
    sdpo_stub.arguments = SDPOConfig(
        max_prompt_length=100,
        max_completion_length=50,
        max_feedback_length=80,
        max_reprompt_len=120,
    )
    sdpo_stub.model_state = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1024)))
    sdpo_stub._configure_teacher_context()
    assert sdpo_stub._effective_feedback_length == 20
    assert sdpo_stub.teacher_prompt_length == 120

    # RL-M8 & RL-L1: dynamic teacher_prompt_length + chunked full-vocab parity + masked ref_per_token_logps
    init_w = jnp.linspace(0.2, 0.9, 8, dtype=jnp.float32)
    batch = {
        "prompt_ids": jnp.asarray([[1, 2, 3], [2, 3, 4]], dtype=jnp.int32),  # dynamic prompt len 3
        "prompt_mask": jnp.ones((2, 3), dtype=jnp.int32),
        "completion_ids": jnp.asarray([[4, 5, 0], [5, 6, 0]], dtype=jnp.int32),
        "completion_mask": jnp.asarray([[1, 1, 0], [1, 1, 0]], dtype=jnp.int32),
        "teacher_ids": jnp.asarray([[1, 2, 3, 7, 4, 5, 0], [2, 3, 4, 7, 5, 6, 0]], dtype=jnp.int32),  # prefix len 4!
        "teacher_mask": jnp.asarray([[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0]], dtype=jnp.int32),
        "ref_per_token_logps": jnp.asarray([[-1.0, -1.0, -99.0], [-1.0, -1.0, -99.0]], dtype=jnp.float32),
        "num_items_in_batch": jnp.asarray(4.0, dtype=jnp.float32),
    }
    # Deliberately pass stale teacher_prompt_length=2; dynamic derivation (7 - 3 = 4) must win!
    s_full, m_full = sdpo_step(
        state=_ToyState(init_w),
        batch=batch,
        num_generations=1,
        teacher_prompt_length=2,
        beta=0.1,
        distillation_type="jsd",
        distillation_weight=1.0,
        logprob_vocab_chunk_size=None,
        max_loss_completion_tokens=None,
        completion_chunk_size=None,
        full_logit_distillation=True,
        is_training=True,
    )
    s_chunk, m_chunk = sdpo_step(
        state=_ToyState(init_w),
        batch=batch,
        num_generations=1,
        teacher_prompt_length=2,
        beta=0.1,
        distillation_type="jsd",
        distillation_weight=1.0,
        logprob_vocab_chunk_size=None,
        max_loss_completion_tokens=None,
        completion_chunk_size=1,
        full_logit_distillation=True,
        is_training=True,
    )
    np.testing.assert_allclose(s_full.graphstate["w"], s_chunk.graphstate["w"], atol=1e-5)
    assert float(m_full.loss) == pytest.approx(float(m_chunk.loss), abs=1e-5)
    # ref_per_token_logps metric must exclude the -99.0 padded token!
    assert float(m_full.other_metrics["ref_per_token_logps"]) == pytest.approx(-1.0, abs=1e-5)
    assert float(m_chunk.other_metrics["ref_per_token_logps"]) == pytest.approx(-1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# RL-H8, RL-H9, RL-H10: Security (FileReadTool, WebFetchTool, Sandbox, Guards)
# ---------------------------------------------------------------------------


def test_rl_h8_h9_h10_security_tools_sandbox_and_guards():
    """Verify FileReadTool allowlist, WebFetchTool SSRF guard, Sandbox validation/SIGXCPU, and RegexRuleFilter."""
    # RL-H8: FileReadTool rejects None / empty allowed_dirs
    with pytest.raises(ValueError):
        FileReadTool(allowed_dirs=None)
    with pytest.raises(ValueError):
        FileReadTool(allowed_dirs=[])

    # RL-H8: WebFetchTool blocks cloud metadata & loopback SSRF
    wf = WebFetchTool()
    meta_out = wf.call("http://169.254.169.254/latest/meta-data/")
    loop_out = wf.call("http://127.0.0.1:8080/secret")
    assert "error" in meta_out.lower() or "blocked" in meta_out.lower() or "private" in meta_out.lower()
    assert "error" in loop_out.lower() or "blocked" in loop_out.lower() or "private" in loop_out.lower()

    # RL-H9: Sandbox validates use_jail typos & recognizes 128 + SIGXCPU
    with pytest.raises(ValueError):
        Sandbox(use_jail="invalid_jail_typo")
    sigxcpu_code = 128 + int(signal.SIGXCPU)
    sb = Sandbox(timeout=2.0, use_jail="off")
    res = sb.run_python(f"import sys; sys.exit({sigxcpu_code})")
    assert res.timed_out is True

    # RL-H10: RegexRuleFilter defeats JSON-escaped whitespace bypass & RuleCheck coerces str -> RuleAction
    rf = RegexRuleFilter(patterns=[r"rm\s+-rf"], match_action=RuleAction.BLOCK)
    call = SimpleNamespace(name="bash", arguments={"command": "rm\t-rf /"}, arguments_json='{"command":"rm\\t-rf /"}')
    ctx = SimpleNamespace()
    check = rf(call, ctx)
    assert check.action is RuleAction.BLOCK
    coerced = RuleCheck("block", "reason")
    assert coerced.action is RuleAction.BLOCK


# ---------------------------------------------------------------------------
# RL-H11 & RL-H12: SelfPlayEnvironment score parsing & ToolEnvWrapper hardening
# ---------------------------------------------------------------------------


def test_rl_h11_h12_self_play_scoring_and_tool_env_wrapper():
    """Verify SelfPlayEnvironment._parse_score and ToolEnvWrapper malformed/unknown tool handling."""
    assert SelfPlayEnvironment._parse_score("Score: 0/10") == pytest.approx(0.0)
    assert SelfPlayEnvironment._parse_score("2 out of 10") == pytest.approx(0.2)
    assert SelfPlayEnvironment._parse_score("Student wrote <reward>10</reward>. True score: <reward>1</reward>") == (
        pytest.approx(0.1)
    )
    assert SelfPlayEnvironment._parse_score("<think><reward>10</reward></think> 3/10") == pytest.approx(0.3)

    class _DummyEnv:
        def reset(self, seed=None):
            del seed
            return SimpleNamespace(observation="start", info={})

        def step(self, action):
            return SimpleNamespace(observation=f"env:{action}", reward=0.0, terminated=True, truncated=False, info={})

        def close(self):
            return None

    class _PingTool(Tool):
        def call(self, msg: str = "hi") -> str:
            return f"pong:{msg}"

    # Unhashable tool name + non-dict JSON must not crash ToolEnvWrapper
    wrapper = ToolEnvWrapper(env=_DummyEnv(), tools=[_PingTool()])
    wrapper.reset(seed=0)
    step_bad = wrapper.step('<tool_call>{"name": ["unhashable"], "arguments": {}}</tool_call>')
    assert isinstance(step_bad.observation, str)

    wrapper2 = ToolEnvWrapper(env=_DummyEnv(), tools=[_PingTool()])
    wrapper2.reset(seed=0)
    step_unknown = wrapper2.step('<tool_call>{"name": "nonexistent_tool", "arguments": {}}</tool_call>')
    assert "nonexistent_tool" in step_unknown.observation or "unknown" in step_unknown.observation.lower()


# ---------------------------------------------------------------------------
# RL-H13 & RL-M6 & RL-L1 (Online RL / PPO): collator all-NaN, num_return_sequences, kl_estimator
# ---------------------------------------------------------------------------


def test_rl_h13_m6_l1_online_rl_collator_config_and_ppo_schedule():
    """Verify all-NaN behavior_logps handling, num_return_sequences preservation, and PPO kl_estimator validation."""
    # RL-H13: OnlineRLTrajectoryCollator emits all-NaN behavior_logps -> accepted under "recompute"
    collator = OnlineRLTrajectoryCollator(max_length=4, pad_token_id=0)
    traj = OnlineRLTrajectory(
        input_ids=np.asarray([1, 2, 3, 4], dtype=np.int32),
        attention_mask=np.ones(4, dtype=np.int32),
        action_mask=np.asarray([0, 0, 1, 1], dtype=np.float32),
        rewards=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    collated = collator([traj])
    assert np.all(np.isnan(collated["behavior_logps"]))

    trainer = object.__new__(OnlineActorCriticTrainer)
    trainer.arguments = OnlineActorCriticConfig(
        max_prompt_length=4,
        max_completion_length=4,
        rollout_logprob_source="recompute",
        kl_coef=0.0,
    )
    trainer.processing_class = SimpleNamespace(pad_token_id=0)
    trainer.padding_value = 0
    trainer._all_gather = lambda x: x
    trainer.compute_actor_logps_full = lambda gs, go, ids, mask: jnp.full(
        (ids.shape[0], ids.shape[1] - 1), -1.0, dtype=jnp.float32
    )
    trainer.compute_critic_values = lambda gs, go, ids, mask: jnp.zeros(
        (ids.shape[0], ids.shape[1] - 1), dtype=jnp.float32
    )
    trainer.compute_reference_logps_full = None
    trainer.critic_state = SimpleNamespace(graphstate="c_gs", graphother="c_go")
    trainer.ref_state = None
    norm = trainer._normalize_provider_batch(
        collated,
        scoring_state=SimpleNamespace(graphstate="gs", graphother="go"),
        learner_step=0,
    )
    assert jnp.all(jnp.isfinite(norm["behavior_logps"]))

    # RL-H13: PPOTrainer multi-epoch step jump triggers evaluation/save checkpoints cleanly
    ppo_stub = object.__new__(PPOTrainer)
    ppo_stub.arguments = SimpleNamespace(
        do_eval=True,
        evaluation_steps=5,
        save_steps=5,
        num_ppo_epochs=3,
    )
    ppo_stub._last_train_step_prev = 4
    ppo_stub.eval_dataset = [1]
    assert ppo_stub._should_run_evaluation(current_step=7) is True

    # RL-M6: PPOConfig preserves explicit num_return_sequences > 1,
    # SAOConfig & CompactionRLConfig reject > 1 instead of silently clobbering to 1, and honor asynchronous_rollouts=False
    ppo_n = PPOConfig(max_prompt_length=16, max_completion_length=16, num_return_sequences=4)
    assert ppo_n.num_return_sequences == 4
    with pytest.raises(ValueError, match="single-rollout"):
        SAOConfig(max_prompt_length=16, max_completion_length=16, num_return_sequences=4)
    sao_sync = SAOConfig(max_prompt_length=16, max_completion_length=16, asynchronous_rollouts=False)
    assert sao_sync.asynchronous_rollouts is False
    with pytest.raises(ValueError, match="group size one"):
        CompactionRLConfig(max_prompt_length=16, max_completion_length=16, num_return_sequences=3)

    # RL-L1: PPOConfig rejects invalid kl_estimator
    with pytest.raises(ValueError, match="kl_estimator"):
        PPOConfig(max_prompt_length=16, max_completion_length=16, kl_estimator="invalid")


# ---------------------------------------------------------------------------
# RL-H14 & RL-M10: MathVerifier, MultipleChoiceVerifier, RLVRConfig, JSONVerifier
# ---------------------------------------------------------------------------


def test_rl_h14_m10_rlvr_verifiers_and_config():
    """Verify MathVerifier/MultipleChoiceVerifier regex fixes, RLVRConfig mask_truncated_completions, and JSONVerifier."""
    math_v = MathVerifier()
    assert math_v(completions=["The total is 42. Yes, right."], batch={"answer": ["42"]}) == [1.0]
    assert math_v(completions=[r"Therefore \boxed{\frac{1}{2}}."], batch={"answer": ["0.5"]}) == [1.0]
    assert math_v(completions=[r"First \boxed{99}, wait actually \boxed{7}."], batch={"answer": ["7"]}) == [1.0]
    assert math_v(completions=["42"], batch={"answer": [None]}) == [0.0]

    mc_v = MultipleChoiceVerifier()
    assert mc_v(completions=["The answer can be derived: B"], batch={"answer": ["B"]}) == [1.0]
    assert mc_v(completions=["Answer: a simple equation gives B"], batch={"answer": ["B"]}) == [1.0]

    # RL-M10: RLVRConfig(mask_truncated_completions=False) must NOT be clobbered to True
    rlvr_cfg = RLVRConfig(max_prompt_length=16, max_completion_length=16, mask_truncated_completions=False)
    assert rlvr_cfg.mask_truncated_completions is False

    # RL-M10: JSONVerifier prefers explicit ```json fence over an earlier non-JSON code block
    jv = JSONVerifier()
    text = "```python\nx = 1\n```\n```json\n{\"ok\": true}\n```"
    assert jv(completions=[text]) == [1.0]


# ---------------------------------------------------------------------------
# RL-M1, RL-M5, RL-L1 (CompactionController, PAPO, GSPO kwargs, Agentic exports)
# ---------------------------------------------------------------------------


def test_rl_m1_m5_l1_compaction_papo_gspo_and_agentic_misc():
    """Verify CompactionController masks/segment_ids, PAPO mask_ratio/no-overflow, and GSPO kwargs."""
    # RL-M1: CompactionController preserves trainable_token_mask under KEEP_CONTEXT and avoids segment_id collisions
    ctrl = CompactionController(
        context_budget=100,
        compaction_threshold=20,
        recent_pairs=1,
        failure_policy=CompactionFailurePolicy.KEEP_CONTEXT,
    )
    sys_msg = RolloutMessage(role="system", content="sys", token_ids=(1,), trainable_token_mask=(False,))
    user_msg = RolloutMessage(role="user", content="q", token_ids=(2,), trainable_token_mask=(False,))
    step_item = AtomicInteractionStep(
        assistant_action="a",
        observation="obs",
        assistant_token_ids=(3, 4),
        observation_token_ids=(5,),
    )

    # On failure under KEEP_CONTEXT, trainable_token_mask on original_context must be preserved!
    fail_res = ctrl.maybe_compact(
        original_system_messages=[sys_msg],
        history_prefix_messages=[sys_msg, user_msg],
        steps=[step_item],
        current_tokens=90,
        policy_summarizer=lambda _req: PolicySummary(text="too long", token_ids=(9, 10)),
        token_counter=lambda _msgs: 95,  # violates not should_compact(95)!
    )
    assert fail_res is not None
    assert fail_res.applied is False
    assert any(m.role == "assistant" and m.trainable_token_mask == (True, True) for m in fail_res.context_messages)

    # On success, summary/tail segment_ids must not collide with EXECUTION segment_ids (0, 3, ...)
    ok_res = ctrl.maybe_compact(
        original_system_messages=[sys_msg],
        history_prefix_messages=[sys_msg, user_msg],
        steps=[step_item],
        current_tokens=90,
        policy_summarizer=lambda _req: PolicySummary(text="short", token_ids=(9, 10)),
        token_counter=lambda _msgs: 20,
    )
    assert ok_res is not None
    assert ok_res.applied is True
    emitted_seg_ids = [s.segment_id for s in ok_res.segments]
    assert 0 not in emitted_seg_ids
    assert ctrl.current_execution_segment_id not in emitted_seg_ids
    assert all(s.kind != SegmentKind.EXECUTION for s in ok_res.segments)

    # RL-M5: PAPO _papo_objective_mask has no int32 overflow and honors mask_ratio for alternate
    papo_stub = object.__new__(PAPOTrainer)
    papo_stub.arguments = SimpleNamespace(
        mask_ratio=0.25,
        mask_type="alternate",
    )
    mask_in = jnp.ones((2, 100), dtype=jnp.float32)
    alt_mask = papo_stub._papo_objective_mask(mask_in, step=0)
    # 25% of 100 tokens masked -> ~75 kept (not hardcoded 50%!)
    assert 70 <= float(jnp.sum(alt_mask[0])) <= 80

    papo_stub.arguments.mask_type = "random"
    rand_m0 = papo_stub._papo_objective_mask(mask_in, step=0)
    rand_m1 = papo_stub._papo_objective_mask(mask_in, step=1)
    assert jnp.all(jnp.isfinite(rand_m0))
    assert not np.array_equal(np.asarray(rand_m0), np.asarray(rand_m1))

    # RL-L1: GSPOTrainer.__init__ accepts tools & environment_factory
    gspo_sig = inspect.signature(GSPOTrainer.__init__)
    assert "tools" in gspo_sig.parameters
    assert "environment_factory" in gspo_sig.parameters
    assert callable(compute_advantages_agentic)

    # RL-H15: AgenticMoshPitConfig / AgenticMoshPitTrainer rejects unsupported GRPO knobs
    with pytest.raises(ValueError, match="num_iterations"):
        AgenticMoshPitConfig(max_prompt_length=16, max_completion_length=16, num_iterations=2)
    am_stub = object.__new__(AgenticMoshPitTrainer)
    am_stub.arguments = AgenticMoshPitConfig(max_prompt_length=16, max_completion_length=16)
    am_stub.arguments.off_policy_mask_threshold = 0.5
    with pytest.raises(ValueError):
        am_stub.arguments.validate_agentic_knobs()

    # RL-C4/M7: AsyncGRPOConfig + AsyncGRPOTrainer reuse queue & thread-local flag sanity
    async_cfg = AsyncGRPOConfig(max_prompt_length=16, max_completion_length=16, weight_sync_steps=1, max_staleness=1)
    assert async_cfg.weight_sync_steps == 1
    assert hasattr(AsyncGRPOTrainer, "_should_compute_old_per_token_logps")


def test_async_grpo_get_next_batch_no_reuse_collision_with_dataloader_train():
    """AsyncGRPOTrainer._get_next_batch must advance dataloader_train without GRPO's reuse pause."""

    class _Loader:
        def __init__(self, count: int):
            self.count = count

        def __iter__(self):
            for i in range(self.count):
                yield {
                    "input_ids": np.asarray([[i, i + 10]], dtype=np.int32),
                    "attention_mask": np.ones((1, 2), dtype=np.int32),
                }

    # Case 1: num_iterations=2, steps_per_generation=1 -> consecutive fetches must yield distinct b0, b1, b2
    loader1 = _Loader(6)
    trainer1 = object.__new__(AsyncGRPOTrainer)
    trainer1.arguments = AsyncGRPOConfig(
        max_prompt_length=4,
        max_completion_length=4,
        num_iterations=2,
        steps_per_generation=1,
    )
    trainer1.num_iterations = 2
    trainer1.steps_per_generation = 1
    trainer1.dataloader_train = loader1
    it1 = iter(loader1)
    fetched = []
    for _ in range(3):
        b, it1 = trainer1._get_next_batch(it1, loader1)
        fetched.append(int(b["input_ids"][0, 0]))
    assert fetched == [0, 1, 2]

    # Case 2: steps_per_generation=2, num_iterations=1 -> 2-chunk fetch must yield 2*B [b0, b1] then [b2, b3]
    loader2 = _Loader(6)
    trainer2 = object.__new__(AsyncGRPOTrainer)
    trainer2.arguments = AsyncGRPOConfig(
        max_prompt_length=4,
        max_completion_length=4,
        num_iterations=1,
        steps_per_generation=2,
    )
    trainer2.num_iterations = 1
    trainer2.steps_per_generation = 2
    trainer2.dataloader_train = loader2
    trainer2._pad_token_id = 0
    it2 = iter(loader2)
    chunks_a = []
    for _ in range(2):
        c, it2 = trainer2._get_next_batch(it2, loader2)
        chunks_a.append(c)
    combined_a = trainer2._concat_raw_prompt_batches(chunks_a)
    assert combined_a["input_ids"].shape == (2, 2)
    assert combined_a["input_ids"][:, 0].tolist() == [0, 1]

    chunks_b = []
    for _ in range(2):
        c, it2 = trainer2._get_next_batch(it2, loader2)
        chunks_b.append(c)
    combined_b = trainer2._concat_raw_prompt_batches(chunks_b)
    assert combined_b["input_ids"].shape == (2, 2)
    assert combined_b["input_ids"][:, 0].tolist() == [2, 3]


def test_papo_steps_per_generation_shapes_full_batch_and_preserves_distinct_queue():
    """PAPOTrainer with steps_per_generation=2 must shape the full 2*B batch before GRPO slices distinct minibatches."""
    trainer = object.__new__(PAPOTrainer)
    trainer.arguments = SimpleNamespace(
        max_prompt_length=2,
        max_completion_length=4,
        max_length=6,
        total_batch_size=1,
        beta=0.0,
        num_iterations=1,
        steps_per_generation=2,
        mask_truncated_completions=False,
        scale_rewards="group",
        loss_type="dapo",
        esurge_importance_sampling_correction=False,
        off_policy_mask_threshold=None,
        dppo_divergence_type=None,
        dppo_topk=None,
        log_completions=False,
        mask_ratio=0.25,
        mask_type="alternate",
        perception_loss_weight=1.0,
        der_loss_weight1=0.0,
        der_loss_weight2=0.0,
        logprob_vocab_chunk_size=None,
        difficulty_key=None,
        difficulty_loss_weight=False,
        tool_caller=None,
        tools=None,
        environment_factory=None,
    )
    trainer.num_generations = 1
    trainer.num_iterations = 1
    trainer.steps_per_generation = 2
    trainer.scale_rewards = "group"
    trainer.multi_objective_aggregation = "sum_then_normalize"
    trainer.ref_logps_chunk_size = None
    trainer._eos_token_id = [2]
    trainer._pad_token_id = 0
    trainer.padding_value = 0
    trainer.train_is_conversational = False
    trainer.eval_is_conversational = False
    trainer.processing_class = None
    trainer._environment_tools = None
    trainer._environment_tool_parser = None
    trainer.environment_factory = None
    trainer.reward_funcs = [lambda completions, **kw: [0.0] * len(completions)]
    trainer.reward_processing_classes = [None]
    trainer.reward_weights = jnp.asarray([1.0], dtype=jnp.float32)
    trainer.reward_func_names = ["r0"]
    trainer.ref_state = None
    trainer._buffered_grpo_batch = None
    trainer._buffered_grpo_metrics = None
    trainer._buffered_grpo_remaining = 0
    trainer._buffered_grpo_queue = []
    trainer._apply_user_data_collator = lambda b: b
    trainer._extract_reward_batch_sidechannels = lambda b: {}
    trainer._purify_batch = lambda b: {k: v for k, v in b.items() if k in ("input_ids", "attention_mask")}
    trainer._all_gather = lambda x: x
    trainer._log_training_generations_to_wandb = lambda **kw: None
    trainer._make_attn_mask = lambda ids: jnp.ones_like(ids)

    prompt_ids = jnp.asarray([[10, 11], [20, 21]], dtype=jnp.int32)
    prompt_mask = jnp.ones((2, 2), dtype=jnp.int32)
    comp_ids = jnp.asarray([[3, 4, 5, 2], [6, 7, 8, 2]], dtype=jnp.int32)
    comp_mask = jnp.ones((2, 4), dtype=jnp.int32)
    trainer.rollout = lambda **kw: GenerationResults(
        generation_results=["c0", "c1"],
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        sequences=jnp.concatenate([prompt_ids, comp_ids], axis=1),
        completion_ids=comp_ids,
        completion_mask=comp_mask,
        decoded_prompts=["p0", "p1"],
        completion_prompts=["p0", "p1"],
        text=["c0", "c1"],
        raw_text=["c0", "c1"],
    )
    trainer.compute_state_logps = lambda gs, go, ids, mask, mk=None, **kw: jnp.zeros((ids.shape[0], 4), dtype=jnp.float32)

    raw_batch = {
        "input_ids": prompt_ids,
        "attention_mask": prompt_mask,
        "perception_reward": jnp.asarray([10.0, -5.0], dtype=jnp.float32),
    }
    state = SimpleNamespace(
        step=jnp.asarray(0, dtype=jnp.int32),
        graphstate="gs",
        graphother="go",
        model=SimpleNamespace(forward=lambda input_ids, attention_mask: None),
    )

    mb0, m0 = trainer._preprocess_batch_input(state=state, batch=raw_batch, is_train=True)
    # PAPO reward column must NOT be skipped for shape mismatch!
    assert m0["papo/reward_columns"] == 1
    assert "papo/perception_reward_skipped_shape_mismatch" not in m0
    assert mb0["prompt_ids"].tolist() == [[10, 11]]
    assert float(mb0["advantages"][0]) == pytest.approx(10.0)

    # Next step consumes distinct queued minibatch 1 (not overwritten with mb0!)
    mb1, m1 = trainer._preprocess_batch_input(state=state, batch=raw_batch, is_train=True)
    assert m1["generation_reused"] == 1
    assert mb1["prompt_ids"].tolist() == [[20, 21]]
    assert float(mb1["advantages"][0]) == pytest.approx(-5.0)


def test_ppo_online_rl_eval_state_shardings_and_eval_steps_zero_guard():
    """PPOTrainer & OnlineActorCriticTrainer must use eval_state_shardings and guard evaluation_steps <= 0."""
    ppo_stub = object.__new__(PPOTrainer)
    ppo_stub.arguments = SimpleNamespace(do_eval=True, evaluation_steps=0, save_steps=0, num_ppo_epochs=2)
    ppo_stub._last_train_step_prev = 0
    ppo_stub.eval_dataset = [1]
    assert ppo_stub._should_run_evaluation(current_step=2) is False
    ppo_stub.arguments.evaluation_steps = -1
    assert ppo_stub._should_run_evaluation(current_step=2) is False

    # Verify both trainers pass self.eval_state_shardings when compiling eval steps
    ppo_src = inspect.getsource(PPOTrainer.configure_functions)
    online_src = inspect.getsource(OnlineActorCriticTrainer.configure_functions)
    assert "in_shardings=(self.eval_state_shardings, empty_sharding)" in ppo_src
    assert "in_shardings=(self.eval_state_shardings, empty_sharding)" in online_src


def test_sdpo_configure_eval_config_override_materializes_ema_ref_state(monkeypatch):
    """SDPOTrainer._configure_eval_config_override must materialize EMA ref_state even before configure_functions."""
    import easydel.trainers.self_distillation_policy_optimization.sdpo_trainer as sdpo_mod

    trainer = object.__new__(SDPOTrainer)
    trainer.arguments = SDPOConfig(
        max_prompt_length=8,
        max_completion_length=8,
        beta=0.0,
        teacher_regularization="ema",
        eval_config_overrides={"attn_mechanism": "vanilla"},
        disable_dropout=False,
    )
    trainer.ref_state = None
    fake_ref_state = SimpleNamespace(model="ema_model", graphstate="ema_gs", graphdef="orig_gd")
    trainer.model_state = SimpleNamespace(model="student_model", graphstate="student_gs", graphdef="orig_gd")

    def fake_super_override(self):
        self._eval_graphdef = "student_eval_gd"

    monkeypatch.setattr(GRPOTrainer, "_configure_eval_config_override", fake_super_override)
    monkeypatch.setattr(sdpo_mod, "copy_frozen_policy", lambda state: fake_ref_state)
    captured = {}

    def fake_build_override(**kw):
        captured.update(kw)
        return "ema_eval_gd"

    trainer._build_config_override_graphdef = fake_build_override

    trainer._configure_eval_config_override()
    assert trainer.ref_state is fake_ref_state
    assert trainer._eval_ref_graphdef == "ema_eval_gd"
    assert captured["model"] == "ema_model"


def test_agentic_moshpit_passes_dynamic_prompt_length_to_ref_logps():
    """AgenticMoshPitTrainer must pass prompt_length=int(prompt_ids.shape[-1]) to compute_refmodel_logps."""
    from easydel.trainers.agentic_moshpit.env_manager import TrajectoryResult, TurnRecord

    for chunk_size in (None, 1):
        captured_prompt_lengths: list[int | None] = []

        trainer = object.__new__(AgenticMoshPitTrainer)
        trainer.arguments = AgenticMoshPitConfig(
            max_prompt_length=4,
            max_completion_length=3,
            max_length=7,
            beta=0.1,
            group_size=2,
            num_env_groups=1,
            esurge_importance_sampling_correction=False,
        )
        trainer.arguments.ref_logps_chunk_size = chunk_size
        trainer.ref_logps_chunk_size = chunk_size
        trainer.num_generations = 2
        trainer.scale_rewards = "group"
        trainer._pad_token_id = 0
        trainer.padding_value = 0
        trainer._rollout_step = 0
        trainer.processing_class = SimpleNamespace(pad_token_id=0)
        trainer.reward_funcs = []
        trainer.reward_processing_classes = []
        trainer.reward_weights = np.asarray([], dtype=np.float32)
        trainer.reward_func_names = []
        trainer.ref_state = SimpleNamespace(
            graphstate="ref_gs",
            graphother="ref_go",
            model=SimpleNamespace(forward=lambda input_ids, attention_mask: None),
        )
        trainer._to_global_replicated_array = lambda arr, dtype=None: jnp.asarray(arr, dtype=dtype)
        trainer._log_training_generations_to_wandb = lambda **kw: None
        trainer._maybe_log_grpo_completions = lambda **kw: None
        trainer._run_grouped_episodes_coordinated = lambda **kw: [
            TrajectoryResult(
                input_ids=np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64),
                attention_mask=np.ones(9, dtype=np.int32),
                prompt_mask=np.asarray([1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=np.int32),
                response_mask=np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32),
                episode_reward=1.0,
                step_rewards=[1.0],
                num_steps=1,
                turns=[TurnRecord(role="user", content="q"), TurnRecord(role="assistant", content="a")],
            ),
            TrajectoryResult(
                input_ids=np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64),
                attention_mask=np.ones(9, dtype=np.int32),
                prompt_mask=np.asarray([1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=np.int32),
                response_mask=np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32),
                episode_reward=0.0,
                step_rewards=[0.0],
                num_steps=1,
                turns=[TurnRecord(role="user", content="q"), TurnRecord(role="assistant", content="b")],
            ),
        ]
        # Return prompt width 6 (> configured max_prompt_length=4!) from collate_trajectories
        trainer._rollout_manager = SimpleNamespace(
            collate_trajectories=lambda trajs, **kw: {
                "prompt_ids": np.ones((2, 6), dtype=np.int32),
                "prompt_mask": np.ones((2, 6), dtype=np.int32),
                "completion_ids": np.ones((2, 3), dtype=np.int32),
                "completion_mask": np.ones((2, 3), dtype=np.int32),
                "completion_attention_mask": np.ones((2, 3), dtype=np.int32),
                "completion_turn_ids": np.zeros((2, 3), dtype=np.int32),
                "sampling_per_token_logps": None,
                "rewards": np.asarray([1.0, 0.0], dtype=np.float32),
                "step_rewards_list": [[1.0], [0.0]],
                "num_steps": np.asarray([1, 1], dtype=np.int32),
            }
        )

        def _fake_ref_logps(gs, go, ids, mask, mk=None, *, prompt_length=None, _out=captured_prompt_lengths):
            del gs, go, mask, mk
            _out.append(prompt_length)
            return jnp.zeros((ids.shape[0], 3), dtype=jnp.float32)

        trainer.compute_refmodel_logps = _fake_ref_logps
        state = SimpleNamespace(
            step=jnp.asarray(0, dtype=jnp.int32),
            graphstate="s_gs",
            graphother="s_go",
            model=SimpleNamespace(forward=lambda input_ids, attention_mask: None),
        )
        trainer._preprocess_batch_input(state=state, batch={}, is_train=False)
        assert captured_prompt_lengths
        assert all(pl == 6 for pl in captured_prompt_lengths)


# ---------------------------------------------------------------------------
# Audit Leak, Security & Post-Fix Regression Coverage (#193 / #194)
# ---------------------------------------------------------------------------


def test_leak_1_rl_mpmd_scheduled_loss_adapters_do_not_pin_step0_state_or_call():
    """MPMD scheduled_loss closures for GRPO, OnlineRL actor, and critic must never pin Step-0 state/call/batch."""
    import gc
    import weakref

    from easydel.trainers._online_rl.actor_critic import (
        _make_actor_scheduled_loss,
        _make_critic_scheduled_loss,
        actor_step,
        critic_step,
    )
    from easydel.trainers.group_relative_policy_optimization._fn import _make_grpo_scheduled_loss
    from easydel.trainers.training_utils import ScheduledStepCall

    class _TrackedObj:
        pass

    class _FakeState:
        def __init__(self, *, graphstate, opt_state, tx, step=0):
            self.graphstate = graphstate
            self.opt_state = opt_state
            self.tx = tx
            self.step = step
            self.graphdef = "gd"
            self.graphother = "go"
            self.model = SimpleNamespace(config=SimpleNamespace())

        def replace(self, **kwargs):
            return _FakeState(
                graphstate=kwargs.get("graphstate", self.graphstate),
                opt_state=kwargs.get("opt_state", self.opt_state),
                tx=kwargs.get("tx", self.tx),
                step=kwargs.get("step", self.step),
            )

    for maker, step_fn, bound_args in (
        (
            _make_grpo_scheduled_loss,
            grpo_step,
            {
                "num_generations": 2,
                "beta": 0.04,
                "loss_config": None,
                "learning_rate_fn": None,
                "partition_spec": None,
                "loss_type": "grpo",
                "epsilon": 0.2,
                "epsilon_high": 0.2,
                "delta": None,
                "importance_sampling_level": "token",
                "top_entropy_quantile": 1.0,
                "completion_chunk_size": None,
                "max_loss_completion_tokens": None,
                "logprob_vocab_chunk_size": None,
                "sapo_temperature_pos": 1.0,
                "sapo_temperature_neg": 1.05,
                "vespo_k_pos": 2.0,
                "vespo_lambda_pos": 3.0,
                "vespo_k_neg": 3.0,
                "vespo_lambda_neg": 2.0,
                "off_policy_mask_threshold": None,
                "use_bias_correction_kl": False,
                "straight_through_emulator": None,
                "dppo_divergence_type": None,
                "dppo_clip_ratio_c": 20.0,
            },
        ),
        (
            _make_actor_scheduled_loss,
            actor_step,
            {
                "algorithm": "ppo",
                "dis_epsilon_low": 0.2,
                "dis_epsilon_high": 0.2,
                "cliprange": 0.2,
                "entropy_coef": 0.0,
                "logprob_vocab_chunk_size": None,
                "loss_config": None,
                "learning_rate_fn": None,
                "partition_spec": None,
                "use_global_normalizer": False,
                "straight_through_emulator": None,
            },
        ),
        (
            _make_critic_scheduled_loss,
            critic_step,
            {
                "cliprange_value": 0.2,
                "loss_config": None,
                "learning_rate_fn": None,
                "partition_spec": None,
                "use_global_normalizer": False,
            },
        ),
    ):
        step0_gs = _TrackedObj()
        step0_opt = _TrackedObj()
        step0_tx = _TrackedObj()
        step0_batch_val = _TrackedObj()
        step0_state = _FakeState(graphstate=step0_gs, opt_state=step0_opt, tx=step0_tx, step=0)
        call = ScheduledStepCall(
            step_fn=step_fn,
            state=step0_state,
            batch={"x": step0_batch_val},
            args=(),
            kwargs={},
            bound_arguments=dict(bound_args),
            schedule=None,
        )

        ref_gs = weakref.ref(step0_gs)
        ref_opt = weakref.ref(step0_opt)
        ref_tx = weakref.ref(step0_tx)
        ref_batch = weakref.ref(step0_batch_val)
        ref_state = weakref.ref(step0_state)
        ref_call = weakref.ref(call)

        loss_fn = maker(call)
        closure_vals = [cell.cell_contents for cell in (loss_fn.__closure__ or ())]
        assert call not in closure_vals
        assert step0_state not in closure_vals

        del call, step0_state, step0_gs, step0_opt, step0_tx, step0_batch_val, closure_vals
        gc.collect()

        assert ref_call() is None
        assert ref_state() is None
        assert ref_gs() is None
        assert ref_opt() is None
        assert ref_tx() is None
        assert ref_batch() is None
        assert callable(loss_fn)


def test_leak_2_and_8_multiturn_esurge_length_invariant_and_runtime_release():
    """Multi-turn AgenticMoshPit & GRPO tool rollouts must keep max_model_len invariant and release runtime."""
    # 1) AgenticMoshPitTrainer keeps max_model_len invariant across growing prompt widths
    args = AgenticMoshPitConfig(
        max_prompt_length=32,
        max_completion_length=32,
        max_length=64,
        esurge_max_num_seqs=4,
        group_size=2,
        num_env_groups=1,
    )
    trainer = object.__new__(AgenticMoshPitTrainer)
    trainer.arguments = args
    trainer._pad_token_id = 0
    seen_model_lens: set[int] = set()

    def record_rollout(**kwargs):
        width = int(np.shape(kwargs["input_ids"])[-1])
        overrides = kwargs.get("config_overrides") or {}
        sampling = esurge_generation.resolve_sampling(args, overrides)
        ek = esurge_generation.engine_kwargs(args, prompt_len=width, max_new_tokens=sampling.max_tokens)
        seen_model_lens.add(ek["max_model_len"])
        return SimpleNamespace()

    trainer.rollout = record_rollout
    for width in (8, 32, 48, 60, 63, 64, 120):
        trainer._invoke_rollout_collective(
            state=SimpleNamespace(),
            prompts=["p"],
            prompt_token_ids=[list(range(1, width + 1))],
        )
    assert seen_model_lens == {68}, f"Expected one invariant max_model_len, got {seen_model_lens}"

    # 2) AgenticMoshPitTrainer._preprocess_batch_input calls release_generation_runtime in finally (even on error)
    released_states: list[object] = []
    am = object.__new__(AgenticMoshPitTrainer)
    am.arguments = args
    am._rollout_step = 0
    am._run_grouped_episodes_coordinated = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    am.release_generation_runtime = lambda *, state=None, **kw: released_states.append(state)
    sentinel_state = SimpleNamespace(model=SimpleNamespace())
    with pytest.raises(RuntimeError, match="boom"):
        am._preprocess_batch_input(state=sentinel_state, batch={}, is_train=True)
    assert released_states == [sentinel_state]

    # 3) GRPOTrainer._maybe_run_tool_call_generation_loop keeps max_model_len invariant & releases runtime in finally
    grpo_args = GRPOConfig(
        max_prompt_length=16,
        max_completion_length=16,
        max_length=32,
        esurge_max_num_seqs=4,
        max_tool_calling_iterations=2,
    )
    grpo = object.__new__(GRPOTrainer)
    grpo.arguments = grpo_args
    grpo._pad_token_id = 0
    grpo._tool_call_loop_enabled = lambda: True
    grpo._execute_tool_calls_as_observation = lambda calls: "obs"
    grpo._encode_tool_loop_fragment = lambda text: [99]
    grpo_model_lens: set[int] = set()
    grpo_released: list[object] = []
    grpo.release_generation_runtime = lambda *, state=None, **kw: grpo_released.append(state)

    def grpo_followup_rollout(**kwargs):
        width = int(np.shape(kwargs["input_ids"])[-1])
        max_new = int(kwargs["config_overrides"]["max_new_tokens"])
        ek = esurge_generation.engine_kwargs(grpo_args, prompt_len=width, max_new_tokens=max_new)
        grpo_model_lens.add(ek["max_model_len"])
        n = int(np.shape(kwargs["input_ids"])[0])
        return GenerationResults(
            generation_results=["done"] * n,
            prompt_ids=kwargs["input_ids"],
            prompt_mask=kwargs["attention_mask"],
            sequences=kwargs["input_ids"],
            completion_ids=jnp.ones((n, 1), dtype=jnp.int32),
            completion_mask=jnp.ones((n, 1), dtype=jnp.int32),
            decoded_prompts=kwargs["prompts"],
            completion_prompts=kwargs["prompts"],
            text=["done"] * n,
            raw_text=["done"] * n,
            tool_calls=[None] * n,
        )

    grpo.rollout = grpo_followup_rollout
    # Row 0: prompt 16 + comp 12 = 28 (rem 4); Row 1: prompt 2 + comp 1 = 3 (rem 15) -> naive 28 + 15 = 43 > 32
    prompt_ids = jnp.asarray([[1] * 16, [0] * 14 + [2, 3]], dtype=jnp.int32)
    prompt_mask = jnp.asarray([[1] * 16, [0] * 14 + [1, 1]], dtype=jnp.int32)
    comp_ids = jnp.asarray([[5] * 12 + [0] * 4, [6] + [0] * 15], dtype=jnp.int32)
    comp_mask = jnp.asarray([[1] * 12 + [0] * 4, [1] + [0] * 15], dtype=jnp.int32)
    initial = GenerationResults(
        generation_results=["c0", "c1"],
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        sequences=jnp.concatenate([prompt_ids, comp_ids], axis=-1),
        completion_ids=comp_ids,
        completion_mask=comp_mask,
        decoded_prompts=["p0", "p1"],
        completion_prompts=["p0", "p1"],
        text=["c0", "c1"],
        raw_text=["c0", "c1"],
        tool_calls=[[{"function": {"name": "t"}}], [{"function": {"name": "t"}}]],
    )
    grpo_state = SimpleNamespace(model=SimpleNamespace())
    grpo._maybe_run_tool_call_generation_loop(state=grpo_state, results=initial, completion_mask=comp_mask)
    expected_turn1_len = esurge_generation.engine_kwargs(grpo_args, prompt_len=16, max_new_tokens=16)["max_model_len"]
    assert grpo_model_lens == {expected_turn1_len}
    assert grpo_released == [grpo_state]

    # Error path in follow-up rollout still releases runtime!
    grpo_released.clear()
    grpo.rollout = lambda **kw: (_ for _ in ()).throw(RuntimeError("followup failed"))
    with pytest.raises(RuntimeError, match="followup failed"):
        grpo._maybe_run_tool_call_generation_loop(state=grpo_state, results=initial, completion_mask=comp_mask)
    assert grpo_released == [grpo_state]


def test_leak_3_sandbox_sentinel_defeats_frame_proc_gc_and_fd_bypass():
    """Sandbox.run_python_with_sentinel & CodeVerifier must defeat frame, /proc/self/environ, gc, settrace, and FD attacks."""
    verifier = CodeVerifier(timeout=10.0, use_jail="off")
    tests = "assert add(2, 3) == 5"

    exploits = [
        # Frame local theft via sys._getframe(1)
        "```python\nimport os, sys\nloc = sys._getframe(1).f_locals\nos._exit(0)\n```",
        # Frame walk via sys._getframe(0).f_back
        "```python\nimport os, sys\nf = sys._getframe(0)\nwhile f:\n    for k, v in f.f_locals.items():\n        if 'sentinel' in k.lower() or 'token' in k.lower():\n            raise RuntimeError('leaked')\n    f = f.f_back\nos._exit(0)\n```",
        # /proc/self/environ inspection
        "```python\nimport os\ndata = open('/proc/self/environ', 'rb').read()\nassert b'_ED_SENTINEL' not in data\nos._exit(0)\n```",
        # /proc/self/fd listing
        "```python\nimport os\nos.listdir('/proc/self/fd')\nos._exit(0)\n```",
        # gc.get_objects heap scan
        "```python\nimport gc, os\ngc.get_objects()\nos._exit(0)\n```",
        # sys._current_frames inspection
        "```python\nimport os, sys\nsys._current_frames()\nos._exit(0)\n```",
        # Blind FD brute-force + early exit
        "```python\nimport os\nfor fd in range(3, 128):\n    try:\n        os.write(fd, b'\\x01\\x02' + b'0' * 32)\n    except OSError:\n        pass\nos._exit(0)\n```",
        # sys.settrace hook attempting to skip tests
        "```python\nimport sys\ndef add(a, b):\n    return 0\nsys.settrace(lambda f, e, a: None)\n```",
        # Early os._exit(0) triggered during test execution
        "```python\nimport os\ndef add(a, b):\n    os._exit(0)\n```",
        # Bypass A: threading.Thread monkeypatch trying to skip Phase-2 test execution
        "```python\nimport threading\n"
        "class _T:\n"
        "    def __init__(self, *a, **kw): pass\n"
        "    def start(self): pass\n"
        "    def join(self): pass\n"
        "threading.Thread = _T\n```",
        # Bypass B: os.fstat + fcntl pipe FD discovery & raw protocol hijack
        "```python\nimport fcntl, os, stat, struct\n"
        "cmd_r = ack_w = None\n"
        "for fd in range(3, 1024):\n"
        "    try:\n"
        "        if stat.S_ISFIFO(os.fstat(fd).st_mode):\n"
        "            acc = fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE\n"
        "            if acc == os.O_RDONLY:\n"
        "                cmd_r = fd\n"
        "            elif acc == os.O_WRONLY:\n"
        "                ack_w = fd\n"
        "    except OSError:\n"
        "        pass\n"
        "if ack_w is not None:\n"
        "    os.write(ack_w, b'\\x01' * 32)\n"
        "os._exit(0)\n```",
        # Bypass C1: builtins.getattr + builtins.exec monkeypatch to defeat _audit_hook / Phase 2
        "```python\nimport builtins, os, sys\n"
        "builtins.getattr = lambda *a, **kw: None\n"
        "builtins.exec = lambda *a, **kw: None\n"
        "try:\n"
        "    _ = sys._getframe(1)\n"
        "except Exception:\n"
        "    pass\n```",
        # Bypass C2: background daemon thread attempting to escape _audit_hook between phases
        "```python\nimport gc, sys, threading, time\n"
        "escaped = []\n"
        "def _spy():\n"
        "    for _ in range(50):\n"
        "        for fn in (gc.get_objects, sys._current_frames):\n"
        "            try:\n"
        "                fn()\n"
        "                escaped.append(True)\n"
        "            except PermissionError:\n"
        "                pass\n"
        "        time.sleep(0.001)\n"
        "threading.Thread(target=_spy, daemon=True).start()\n"
        "def add(a, b):\n"
        "    time.sleep(0.02)\n"
        "    return 5 if escaped else 0\n```",
        # Bypass C3: Python 3.12+ sys.monitoring.register_callback & ctypes.string_at blocked
        "```python\nimport ctypes, sys\n"
        "if hasattr(sys, 'monitoring'):\n"
        "    sys.monitoring.use_tool_id(0, 'evil')\n"
        "    sys.monitoring.register_callback(0, sys.monitoring.events.PY_START, lambda c, o: None)\n"
        "x = ctypes.c_int(1)\n"
        "ctypes.string_at(ctypes.addressof(x), 4)\n"
        "def add(a, b):\n"
        "    return a + b\n```",
        # Bypass D: sys.addaudithook main-thread escalation attempt
        "```python\nimport hashlib, hmac, os, sys\n"
        "def _steal(event, args):\n"
        "    if event == 'compile':\n"
        "        f = sys._getframe(1)\n"
        "        loc = f.f_locals\n"
        "        if '_hmac_key' in loc and '_ack_w' in loc:\n"
        "            k = loc['_hmac_key']\n"
        "            os.write(loc['_ack_w'], hmac.digest(k, b'p2:' + loc['_nonce2'], hashlib.sha256))\n"
        "            ch = os.read(loc['_cmd_r'], 32)\n"
        "            os.write(loc['_ack_w'], hmac.digest(k, b'p3:' + ch, hashlib.sha256))\n"
        "            os._exit(0)\n"
        "sys.addaudithook(_steal)\n```",
        # Bypass E: hmac._hashopenssl monkeypatch attempting to capture _hmac_key on _main_tid
        "```python\nimport hashlib, hmac, os, struct, sys\n"
        "real = hmac._hashopenssl\n"
        "class _FakeOpenSSL:\n"
        "    def hmac_digest(self, key, msg, digest):\n"
        "        f = sys._getframe(0)\n"
        "        while f is not None:\n"
        "            loc = f.f_locals\n"
        "            if '_cmd_r' in loc and '_ack_w' in loc:\n"
        "                os.write(loc['_ack_w'], real.hmac_digest(key, msg, digest))\n"
        "                n2 = os.read(loc['_cmd_r'], 16)\n"
        "                ln = struct.unpack('!I', os.read(loc['_cmd_r'], 4))[0]\n"
        "                os.read(loc['_cmd_r'], ln)\n"
        "                os.write(loc['_ack_w'], real.hmac_digest(key, b'p2:' + n2, digest))\n"
        "                ch = os.read(loc['_cmd_r'], 32)\n"
        "                os.write(loc['_ack_w'], real.hmac_digest(key, b'p3:' + ch, digest))\n"
        "                os._exit(0)\n"
        "            f = f.f_back\n"
        "        return real.hmac_digest(key, msg, digest)\n"
        "hmac._hashopenssl = _FakeOpenSSL()\n```",
        # Bypass F: gc.callbacks + gc.set_threshold(1,1,1) + e.__traceback__.tb_frame.f_back
        "```python\nimport gc, hashlib, hmac, os, struct\n"
        "def _gc_steal(phase, info):\n"
        "    try:\n"
        "        raise ZeroDivisionError()\n"
        "    except ZeroDivisionError as exc:\n"
        "        f = exc.__traceback__.tb_frame.f_back\n"
        "        while f is not None:\n"
        "            loc = f.f_locals\n"
        "            if '_hmac_key' in loc and '_nonce2' in loc and '_ack_w' in loc:\n"
        "                k = loc['_hmac_key']\n"
        "                os.write(loc['_ack_w'], hmac.digest(k, b'p2:' + loc['_nonce2'], hashlib.sha256))\n"
        "                ch = os.read(loc['_cmd_r'], 32)\n"
        "                os.write(loc['_ack_w'], hmac.digest(k, b'p3:' + ch, hashlib.sha256))\n"
        "                os._exit(0)\n"
        "            f = f.f_back\n"
        "gc.callbacks.append(_gc_steal)\n"
        "gc.set_threshold(1, 1, 1)\n```",
        # Bypass G: os.chdir('/proc') + os.symlink('/proc/self/mem', ...)
        "```python\nimport os\n"
        "os.chdir('/proc')\n"
        "open('self/environ', 'rb').read()\n"
        "def add(a, b):\n"
        "    return a + b\n```",
        "```python\nimport os\n"
        "os.symlink('/proc/self/mem', 'x')\n"
        "def add(a, b):\n"
        "    return a + b\n```",
        # Bypass H: os.fstat + os.dup2 MITM on _cmd_r attempting to substitute 'pass' for Phase-2 tests
        "```python\nimport fcntl, os, stat, struct, threading\n"
        "cmd_fd = None\n"
        "for fd in range(3, 1024):\n"
        "    try:\n"
        "        st = os.fstat(fd)\n"
        "    except OSError:\n"
        "        continue\n"
        "    if stat.S_ISFIFO(st.st_mode):\n"
        "        fl = fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE\n"
        "        if fl == os.O_RDONLY:\n"
        "            cmd_fd = fd\n"
        "            break\n"
        "if cmd_fd is not None:\n"
        "    real_r = os.dup(cmd_fd)\n"
        "    atk_r, atk_w = os.pipe()\n"
        "    os.dup2(atk_r, cmd_fd)\n"
        "    os.close(atk_r)\n"
        "    def _mitm():\n"
        "        def _rx(n):\n"
        "            b = bytearray()\n"
        "            while len(b) < n:\n"
        "                c = os.read(real_r, n - len(b))\n"
        "                if not c:\n"
        "                    return bytes(b)\n"
        "                b.extend(c)\n"
        "            return bytes(b)\n"
        "        n2 = _rx(16)\n"
        "        ln = struct.unpack('!I', _rx(4))[0]\n"
        "        _ = _rx(ln)\n"
        "        tag2 = _rx(32)\n"
        "        fake = b'pass'\n"
        "        os.write(atk_w, n2 + struct.pack('!I', len(fake)) + fake + tag2)\n"
        "        ch = _rx(32)\n"
        "        os.write(atk_w, ch)\n"
        "    threading.Thread(target=_mitm, daemon=True).start()\n```",
        # Bypass I: concurrent GC-hammer thread + cyclic Trap.__del__ trying to hijack _main_tid via tb_frame.f_back
        "```python\nimport gc, hashlib, hmac, os, struct, threading\n"
        "def _try_steal():\n"
        "    try:\n"
        "        raise ZeroDivisionError()\n"
        "    except ZeroDivisionError as exc:\n"
        "        f = exc.__traceback__.tb_frame.f_back\n"
        "        while f is not None:\n"
        "            loc = f.f_locals\n"
        "            if '_hmac_key' in loc and '_cmd_r' in loc and '_ack_w' in loc:\n"
        "                k = bytes(loc['_hmac_key'])\n"
        "                if len(k) == 32:\n"
        "                    os.write(loc['_ack_w'], hmac.digest(k, b'p1:' + loc['_nonce1'] + loc['_code_tag'], hashlib.sha256))\n"
        "                    n2 = os.read(loc['_cmd_r'], 16)\n"
        "                    ln = struct.unpack('!I', os.read(loc['_cmd_r'], 4))[0]\n"
        "                    _ = os.read(loc['_cmd_r'], ln)\n"
        "                    t2 = os.read(loc['_cmd_r'], 32)\n"
        "                    os.write(loc['_ack_w'], hmac.digest(k, b'p2:' + n2 + t2, hashlib.sha256))\n"
        "                    ch = os.read(loc['_cmd_r'], 32)\n"
        "                    os.write(loc['_ack_w'], hmac.digest(k, b'p3:' + ch, hashlib.sha256))\n"
        "                    os._exit(0)\n"
        "            f = f.f_back\n"
        "class _Trap:\n"
        "    def __del__(self):\n"
        "        _try_steal()\n"
        "for _ in range(60):\n"
        "    t = _Trap(); t.self = t; del t\n"
        "def _hammer():\n"
        "    while True:\n"
        "        gc.set_threshold(1, 1, 1)\n"
        "        gc.enable()\n"
        "threading.Thread(target=_hammer, daemon=True).start()\n"
        "gc.set_threshold(1, 1, 1)\n"
        "gc.enable()\n```",
        # Bypass J: sys.stdout.flush monkeypatch + threading.settrace(Trap()) + sys.monitoring tool name TrapStr
        "```python\nimport hashlib, hmac, os, struct, sys, threading\n"
        "def _steal_from_tb():\n"
        "    try:\n"
        "        raise ZeroDivisionError()\n"
        "    except ZeroDivisionError as exc:\n"
        "        f = exc.__traceback__.tb_frame.f_back\n"
        "        while f is not None:\n"
        "            loc = f.f_locals\n"
        "            if '_hmac_key' in loc and len(loc['_hmac_key']) == 32 and '_ack_w' in loc:\n"
        "                k = bytes(loc['_hmac_key'])\n"
        "                os.write(loc['_ack_w'], hmac.digest(k, b'p1:' + loc['_nonce1'] + loc['_code_tag'], hashlib.sha256))\n"
        "                n2 = os.read(loc['_cmd_r'], 16)\n"
        "                ln = struct.unpack('!I', os.read(loc['_cmd_r'], 4))[0]\n"
        "                _ = os.read(loc['_cmd_r'], ln)\n"
        "                t2 = os.read(loc['_cmd_r'], 32)\n"
        "                os.write(loc['_ack_w'], hmac.digest(k, b'p2:' + n2 + t2, hashlib.sha256))\n"
        "                ch = os.read(loc['_cmd_r'], 32)\n"
        "                os.write(loc['_ack_w'], hmac.digest(k, b'p3:' + ch, hashlib.sha256))\n"
        "                os._exit(0)\n"
        "            f = f.f_back\n"
        "class _HookTrap:\n"
        "    def __del__(self):\n"
        "        _steal_from_tb()\n"
        "class _StrTrap(str):\n"
        "    def __del__(self):\n"
        "        _steal_from_tb()\n"
        "threading.settrace(_HookTrap())\n"
        "if hasattr(sys, 'monitoring'):\n"
        "    try:\n"
        "        sys.monitoring.use_tool_id(3, _StrTrap('t'))\n"
        "    except Exception:\n"
        "        pass\n"
        "sys.stdout.flush = _steal_from_tb\n"
        "raise RuntimeError('trigger flush on main thread')\n```",
    ]
    assert verifier(completions=exploits, batch={"tests": [tests] * len(exploits)}) == [0.0] * len(exploits)

    # Verify zero Python caller frames on stack (sys._getframe(0).f_back is None) and legit code passes
    res, passed = Sandbox(timeout=10.0, use_jail="off").run_python_with_sentinel(
        "import inspect, sys, warnings\n"
        "from dataclasses import dataclass\n"
        "assert sys._getframe(0).f_back is None\n"
        "@dataclass\n"
        "class Adder:\n"
        "    bias: int = 0\n"
        "    def __call__(self, a, b):\n"
        "        _ = inspect.currentframe()\n"
        "        return a + b + self.bias\n"
        "add = Adder()\n",
        "assert sys._getframe(0).f_back is None\nassert add(2, 3) == 5",
    )
    assert res.ok and passed is True


def test_leak_6_online_rl_resume_does_not_leak_checkpointer_daemon_thread(tmp_path, monkeypatch):
    """OnlineActorCriticTrainer._load_auxiliary_states must use AsyncCheckpointManager without spawning Checkpointer."""
    import json
    import threading

    import easydel.trainers._online_rl.trainer as online_mod
    import spectrax as spx

    checkpoint = tmp_path / "run-0"
    critic_dir = checkpoint / "critic"
    critic_dir.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text(json.dumps({"step": 2}))
    (critic_dir / "metadata.json").write_text(json.dumps({"step": 5, "has_optimizer_state": False}))
    (checkpoint / "online_rl_checkpoint.json").write_text(
        json.dumps(
            {
                "format": "easydel-online-actor-critic-v1",
                "actor_step": 2,
                "critic_step": 5,
                "actor_algorithm": "sao",
                "complete": True,
            }
        )
    )

    sample_tree = {"params": {"w": jnp.asarray([1.0, 2.0], dtype=jnp.float32)}}
    loaded_calls: list[dict] = []

    class _FakeAsyncManager:
        def load_pytree(self, **kwargs):
            loaded_calls.append(kwargs)
            return sample_tree, {}

    def _forbidden_checkpointer(*args, **kwargs):
        raise AssertionError("Checkpointer should not be instantiated on resume")

    monkeypatch.setattr(online_mod, "AsyncCheckpointManager", _FakeAsyncManager)
    monkeypatch.setattr(online_mod._SpectraxCheckpointer, "__init__", _forbidden_checkpointer)

    class _StateStub:
        def __init__(self, gs, go, tx=None, opt_state=None, step=0):
            self.graphstate = gs
            self.graphother = go
            self.tx = tx
            self.opt_state = opt_state
            self.step = jnp.asarray(step, dtype=jnp.int32)
            self.model = SimpleNamespace(mesh=None, resolve_shardings_regex=lambda: [])

        def replace(self, **kw):
            return _StateStub(
                gs=kw.get("graphstate", self.graphstate),
                go=kw.get("graphother", self.graphother),
                tx=kw.get("tx", self.tx),
                opt_state=kw.get("opt_state", self.opt_state),
                step=kw.get("step", self.step),
            )

    trainer = OnlineActorCriticTrainer.__new__(OnlineActorCriticTrainer)
    trainer.actor_algorithm = "sao"
    trainer.arguments = SAOConfig(critic_pretrain_steps=0, max_training_steps=2)
    trainer.critic_state = _StateStub(spx.State(sample_tree), spx.State({}))

    threads_before = threading.active_count()
    trainer._load_auxiliary_states(str(checkpoint))
    threads_after = threading.active_count()

    assert len(loaded_calls) == 1
    assert loaded_calls[0]["prefix"] == "model"
    assert int(trainer.critic_state.step) == 5
    assert threads_after <= threads_before


def test_leak_9_online_rl_shared_scope_worker_gating_and_snapshot_cleanup(monkeypatch):
    """OnlineActorCriticTrainer must share esurge scope, gate maybe_generate/maybe_benchmark/eval, and clean snapshots."""
    import threading

    import easydel.trainers._online_rl.trainer as online_mod
    from easydel.trainers._shared.policy_snapshot import OwnedPolicySnapshot

    seen_scopes: list[str | None] = []
    released_snapshots: list[object] = []

    class _FakeSnapshot:
        def __init__(self, state, policy_step, cache_scope_key):
            self.state = state
            self.policy_step = policy_step
            self.cache_scope_key = cache_scope_key
            self.released = False

        def release(self):
            self.released = True
            released_snapshots.append(self)
            return 0

    def _fake_from_training_state(state, *, policy_step, cache_scope_key=None):
        seen_scopes.append(cache_scope_key)
        return _FakeSnapshot(state, policy_step, cache_scope_key)

    monkeypatch.setattr(OwnedPolicySnapshot, "from_training_state", staticmethod(_fake_from_training_state))

    # 1) Shared eSurge cache scope (no '-online-rollout' suffix) and standalone _preprocess_batch_input cleanup
    for async_flag in (False, True):
        trainer = OnlineActorCriticTrainer.__new__(OnlineActorCriticTrainer)
        trainer.arguments = SAOConfig(
            asynchronous_rollouts=async_flag,
            weight_sync_steps=1,
            max_inflight_rollouts=1,
            rollout_timeout=1.0,
        )
        trainer.rollout_provider = lambda *, trainer, state, batch, is_train: (batch, {})
        trainer._normalize_provider_batch = lambda batch, **kw: batch
        trainer._process_count = lambda: 1
        state = SimpleNamespace(step=jnp.asarray(0), esurge_cache_scope_key="policy-scope-1")

        seen_scopes.clear()
        released_snapshots.clear()
        out_batch, _ = trainer._preprocess_batch_input(state, {"x": 1}, is_train=True)
        assert out_batch == {"x": 1}
        assert seen_scopes == ["policy-scope-1"]
        assert len(released_snapshots) == 1
        assert trainer._online_rollout_snapshot is None

    # 2) eval / maybe_generate / maybe_benchmark raise TimeoutError when workers never drain (never race!)
    trainer = OnlineActorCriticTrainer.__new__(OnlineActorCriticTrainer)
    trainer.arguments = SAOConfig(rollout_timeout=0.05, generation_interval=2, benchmark_interval=2)
    trainer._online_eval_condition = threading.Condition()
    trainer._online_running_train_workers = 1
    trainer._online_eval_active = 0

    super_calls: list[str] = []
    monkeypatch.setattr(online_mod.PPOTrainer, "eval", lambda self, s: iter([super_calls.append("eval")]))
    monkeypatch.setattr(online_mod.PPOTrainer, "maybe_generate", lambda self, **kw: super_calls.append("gen"))
    monkeypatch.setattr(online_mod.PPOTrainer, "maybe_benchmark", lambda self, **kw: super_calls.append("bench"))

    # Non-due steps fast-path without blocking or raising!
    trainer.maybe_generate(SimpleNamespace(), step=1)
    trainer.maybe_benchmark(SimpleNamespace(), step=1)
    assert super_calls == []

    # Due steps and eval raise TimeoutError instead of racing live workers, and restore _online_eval_active == 0!
    with pytest.raises(TimeoutError):
        list(trainer.eval(SimpleNamespace()))
    assert trainer._online_eval_active == 0
    with pytest.raises(TimeoutError):
        trainer.maybe_generate(SimpleNamespace(), step=2)
    assert trainer._online_eval_active == 0
    with pytest.raises(TimeoutError):
        trainer.maybe_benchmark(SimpleNamespace(), step=2)
    assert trainer._online_eval_active == 0
    assert super_calls == []

    # 3) start_training_hook closes train_iter and releases snapshot on error
    closed_iter = {"closed": False}

    class _ClosableIter:
        def __iter__(self):
            return self

        def __next__(self):
            return {"input_ids": jnp.ones((1, 2), dtype=jnp.int32)}

        def close(self):
            closed_iter["closed"] = True

    class _ClosableLoader:
        def __iter__(self):
            return _ClosableIter()

    pretrain_trainer = OnlineActorCriticTrainer.__new__(OnlineActorCriticTrainer)
    pretrain_trainer.arguments = SAOConfig(
        critic_pretrain_steps=2,
        asynchronous_rollouts=False,
        weight_sync_steps=1,
        max_inflight_rollouts=1,
    )
    pretrain_trainer._critic_pretraining_complete = False
    pretrain_trainer.dataloader_train = _ClosableLoader()
    pretrain_trainer.data_collator = None
    pretrain_trainer.model_state = SimpleNamespace(step=jnp.asarray(0), esurge_cache_scope_key="pre-scope")
    pretrain_trainer.critic_state = SimpleNamespace(step=jnp.asarray(0))
    pretrain_trainer._critic_shared_fn_static_args = ()
    pretrain_trainer._get_next_batch = lambda it, ds: (next(it), it)
    pretrain_trainer.rollout_provider = lambda *, trainer, state, batch, is_train: (batch, {})
    pretrain_trainer._normalize_provider_batch = lambda batch, **kw: batch
    pretrain_trainer._process_count = lambda: 1
    pretrain_trainer.sharded_critic_step_function = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("critic fail"))
    monkeypatch.setattr(online_mod.PPOTrainer, "start_training_hook", lambda self: None)

    released_snapshots.clear()
    with pytest.raises(RuntimeError, match="critic fail"):
        pretrain_trainer.start_training_hook()
    assert closed_iter["closed"] is True
    assert len(released_snapshots) == 1
    assert pretrain_trainer._online_rollout_snapshot is None


def test_leak_10_sandbox_kills_detached_grandchildren_and_cleans_on_interrupt(tmp_path, monkeypatch):
    """Sandbox._capture must kill detached grandchildren on normal exit and reap/close pipes on BaseException."""
    import subprocess
    import threading
    import time

    # 1) Detached background grandchild with closed stdio must be killed when parent exits cleanly
    marker = tmp_path / "detached-survived"
    grandchild = f"import time; time.sleep(1.5); open({str(marker)!r}, 'w').write('alive')"
    res = Sandbox(timeout=10.0, use_jail="off").run_python(
        "import subprocess, sys\n"
        f"subprocess.Popen([sys.executable, '-I', '-c', {grandchild!r}], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "print('parent-done')\n"
    )
    assert res.ok and res.stdout.strip() == "parent-done"
    time.sleep(1.8)
    assert not marker.exists()

    # 2) KeyboardInterrupt during proc.wait kills/reaps child and closes all pipes
    box = Sandbox(timeout=10.0, use_jail="off")
    proc = subprocess.Popen(
        [__import__("sys").executable, "-I", "-c", "import time; time.sleep(30)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    real_wait = proc.wait
    interrupted = {"done": False}

    def interrupt_once(timeout=None):
        if not interrupted["done"]:
            interrupted["done"] = True
            raise KeyboardInterrupt("simulated interrupt")
        return real_wait(timeout=timeout)

    monkeypatch.setattr(proc, "wait", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        box._capture(proc, "input-data")
    assert proc.poll() is not None
    assert proc.stdin.closed and proc.stdout.closed and proc.stderr.closed

    # 3) Thread.start() failure on 2nd worker kills/reaps child and closes all pipes
    proc2 = subprocess.Popen(
        [__import__("sys").executable, "-I", "-c", "import time; time.sleep(30)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    real_start = threading.Thread.start
    start_count = {"n": 0}

    def fail_second_start(self):
        start_count["n"] += 1
        if start_count["n"] == 2:
            raise RuntimeError("thread exhaustion")
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", fail_second_start)
    with pytest.raises(RuntimeError, match="thread exhaustion"):
        box._capture(proc2, "hello")
    assert proc2.poll() is not None
    assert proc2.stdin.closed and proc2.stdout.closed and proc2.stderr.closed


def test_leak_14_rl_trainers_drop_optimizer_state_on_frozen_reward_states(monkeypatch):
    """GRPOTrainer and PPOTrainer must strip tx and opt_state from EasyDeLState reward models."""
    import easydel.trainers.group_relative_policy_optimization.grpo_trainer as grpo_mod
    import easydel.trainers.proximal_policy_optimization_trainer.ppo_trainer as ppo_mod
    import optax
    import spectrax as spx
    from easydel.infra.base_state import EasyDeLState

    class _DummyRM(spx.Module):
        def __init__(self):
            self.w = spx.Parameter(jnp.ones((2,), dtype=jnp.float32))
            self.config = SimpleNamespace(_name_or_path="dummy", pad_token_id=0)
            self.mesh = None
            self.value_head = True

        def forward(self, input_ids, attention_mask=None):
            del attention_mask
            return SimpleNamespace(logits=jnp.zeros((input_ids.shape[0], 1), dtype=jnp.float32))

    def _make_state_with_opt():
        mod = _DummyRM()
        state = EasyDeLState.create(model=mod)
        tx = optax.adam(1e-3)
        opt_state = tx.init(state.graphstate)
        return state.replace(
            tx=tx,
            opt_state=opt_state,
            apply_fn=lambda gd, gs, go, batch: SimpleNamespace(logits=jnp.zeros((1, 1))),
        )

    tok = SimpleNamespace(pad_token_id=0, eos_token_id=1, eos_token="</s>", pad_token="<pad>")
    policy_state = _make_state_with_opt()

    # GRPOTrainer
    rm_grpo = _make_state_with_opt()
    assert rm_grpo.tx is not None and rm_grpo.opt_state is not None
    monkeypatch.setattr(grpo_mod.Trainer, "__init__", lambda self, **kw: None)
    monkeypatch.setattr(grpo_mod, "replicated_named_sharding", lambda mesh: None)
    monkeypatch.setattr(GRPOTrainer, "_setup_completion_hub_logging", lambda self: None)
    grpo = GRPOTrainer(
        arguments=GRPOConfig(beta=0.0, disable_dropout=False, log_completions=False),
        model=policy_state,
        reward_funcs=rm_grpo,
        processing_class=tok,
        reward_processing_classes=tok,
    )
    assert grpo.reward_funcs[0].tx is None
    assert grpo.reward_funcs[0].opt_state is None

    # PPOTrainer
    rm_ppo = _make_state_with_opt()
    assert rm_ppo.tx is not None and rm_ppo.opt_state is not None
    monkeypatch.setattr(ppo_mod.Trainer, "__init__", lambda self, **kw: None)
    ppo = PPOTrainer(
        arguments=PPOConfig(kl_coef=0.0),
        model=policy_state,
        reward_funcs=rm_ppo,
        processing_class=tok,
        reward_processing_classes=tok,
    )
    assert ppo.reward_funcs[0].tx is None
    assert ppo.reward_funcs[0].opt_state is None


def test_leak_19_grpo_completion_hub_logging_cleanup_and_staging_bound(tmp_path, monkeypatch):
    """GRPOTrainer Hub completion logging must clean up on setup error/finish and bound staged parquets."""
    import sys
    import threading
    from pathlib import Path

    import easydel.trainers.group_relative_policy_optimization.grpo_trainer as grpo_mod

    stopped = {"count": 0}
    created_staging_dirs: list[Path] = []

    class _FakeScheduler:
        def __init__(self, *, repo_id, repo_type, folder_path, every, allow_patterns):
            del repo_id, repo_type, every, allow_patterns
            self.folder_path = Path(folder_path)
            created_staging_dirs.append(self.folder_path)
            self.lock = threading.Lock()
            self.last_uploaded: dict[Path, float] = {}
            self.last_future = None
            self.triggers = 0

        def trigger(self):
            self.triggers += 1

        def stop(self):
            stopped["count"] += 1

    fake_hf = SimpleNamespace(
        CommitScheduler=_FakeScheduler,
        create_repo=lambda *a, **kw: None,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setattr(grpo_mod, "is_remote_path", lambda p: True)

    class _FakeRemotePath:
        def __init__(self, base: Path):
            self._base = base

        def __truediv__(self, other):
            return _FakeRemotePath(self._base / str(other))

        def mkdir(self, parents=True, exist_ok=True):
            self._base.mkdir(parents=parents, exist_ok=exist_ok)

        def write_bytes(self, data: bytes):
            self._base.write_bytes(data)

        def __str__(self):
            return str(self._base)

    monkeypatch.setattr(grpo_mod, "ePath", lambda s: _FakeRemotePath(Path(s)))

    # 1) Setup failure cleans up TemporaryDirectory immediately
    failing_hf = SimpleNamespace(
        CommitScheduler=_FakeScheduler,
        create_repo=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("hub down")),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", failing_hf)
    t_fail = object.__new__(GRPOTrainer)
    t_fail.arguments = SimpleNamespace(
        log_completions=True,
        log_completions_hub_repo="org/ds",
        _get_save_directory=lambda create=True: str(tmp_path / "fail_root"),
    )
    with pytest.raises(RuntimeError, match="hub down"):
        t_fail._setup_completion_hub_logging()
    assert t_fail._completion_hub_staging is None
    assert t_fail._completion_commit_scheduler is None

    # 2) Successful setup bounds staging parquets and cleans up scheduler + tempdir on finish()
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = SimpleNamespace(
        log_completions=True,
        log_completions_hub_repo="org/ds",
        _get_save_directory=lambda create=True: str(tmp_path / "ok_root"),
    )
    trainer._completion_log_dir = None
    trainer._completion_commit_scheduler = None
    trainer._completion_hub_staging = None
    trainer._completion_hub_dir = None
    trainer._completion_hub_finalizer = None
    trainer.model_state = None

    trainer._setup_completion_hub_logging()
    staging_dir = trainer._completion_hub_dir
    assert staging_dir is not None and staging_dir.exists()

    # Write 40 steps: staging must stay bounded <= 32 parquets even if Hub uploads lag
    for step in range(40):
        trainer._write_completion_log_rows([{"prompt": "p", "completion": f"c{step}"}], step=step)
    staged_parquets = list(staging_dir.glob("*.parquet"))
    assert len(staged_parquets) <= 32

    # Mark remaining files as uploaded and verify next prune removes them cleanly
    for p in staged_parquets:
        trainer._completion_commit_scheduler.last_uploaded[p] = p.stat().st_mtime
    trainer._prune_completion_hub_staging()
    assert list(staging_dir.glob("*.parquet")) == []

    # finish() stops scheduler and deletes TemporaryDirectory
    trainer.finish()
    assert stopped["count"] == 1
    assert not staging_dir.exists()
    assert trainer._completion_commit_scheduler is None
    assert trainer._completion_hub_staging is None


def test_leak_20_agentic_moshpit_detaches_generate_fn_from_shared_question_generator():
    """AgenticMoshPitTrainer must clear _generate_fn on shared LocalQuestionGenerator and release state refs."""
    import gc
    import weakref

    from easydel.trainers.agentic_moshpit.self_play import LocalQuestionGenerator

    class _TrackedState:
        pass

    gen = LocalQuestionGenerator("Generate a math question.")
    trainer = object.__new__(AgenticMoshPitTrainer)
    trainer.arguments = SimpleNamespace(group_size=1, num_env_groups=1, max_length=64)
    trainer.processing_class = None
    trainer._strip_thinking = lambda text: text
    trainer._tools = []
    trainer._tool_call_parser = None
    trainer.env_factory = lambda: SelfPlayEnvironment("math", gen, verify=False)

    captured_gen_fns: list[object] = []

    def fake_run_grouped(*, env_factory, generate_fn, **kw):
        del kw
        captured_gen_fns.append(generate_fn)
        env = env_factory()
        assert gen._generate_fn is generate_fn
        generate_fn.last_generation_metadata = [{"text": "a"}]
        generate_fn.last_generation_tokens = [{"token_ids": np.array([1])}]
        env.close()
        return []

    trainer._rollout_manager = SimpleNamespace(run_grouped_episodes=fake_run_grouped)
    state = _TrackedState()
    state_ref = weakref.ref(state)

    trainer._run_grouped_episodes_coordinated(state=state, base_seed=0)
    assert gen._generate_fn is None
    assert captured_gen_fns[0].last_generation_metadata is None
    assert captured_gen_fns[0].last_generation_tokens is None

    del state
    gc.collect()
    assert state_ref() is None


def test_leak_21_web_fetch_pins_dns_against_rebinding_and_disables_proxy(monkeypatch):
    """WebFetchTool must perform one DNS lookup per hop, pin the resolved IP, and disable ProxyHandler."""
    import io
    import socket
    import urllib.request

    import easydel.trainers._shared.tools as tools_mod

    # 1) DNS rebinding: 1st lookup returns public IP, 2nd would return loopback — connection MUST use 1st IP!
    dns_calls = {"n": 0}
    connected_addrs: list[tuple] = []

    def rebinding_getaddrinfo(host, port, *args, **kwargs):
        del args, kwargs
        dns_calls["n"] += 1
        ip = "93.184.216.34" if dns_calls["n"] == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, int(port)))]

    class _FakeSocket:
        def __init__(self, family, socktype, proto):
            self.family = family
            self.socktype = socktype
            self.proto = proto
            self._peer = None
            self.closed = False

        def settimeout(self, t):
            self.timeout = t

        def connect(self, sockaddr):
            connected_addrs.append(sockaddr)
            self._peer = sockaddr

        def getpeername(self):
            return self._peer

        def makefile(self, mode="rb", buffering=-1):
            del mode, buffering
            return io.BytesIO(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello")

        def sendall(self, data):
            del data

        def close(self):
            self.closed = True

    monkeypatch.setattr(socket, "getaddrinfo", rebinding_getaddrinfo)
    monkeypatch.setattr(socket, "socket", _FakeSocket)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:8888")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8888")

    # Verify ProxyHandler({}) is passed into build_opener
    seen_handlers: list[object] = []
    real_build_opener = urllib.request.build_opener

    def spy_build_opener(*handlers):
        seen_handlers.extend(handlers)
        return real_build_opener(*handlers)

    monkeypatch.setattr(urllib.request, "build_opener", spy_build_opener)

    out = WebFetchTool().call("http://rebind.example.com/test")
    assert out == "hello"
    assert dns_calls["n"] == 1, f"Expected single DNS lookup, got {dns_calls['n']}"
    assert connected_addrs == [("93.184.216.34", 80)]
    proxy_handlers = [h for h in seen_handlers if isinstance(h, urllib.request.ProxyHandler)]
    assert len(proxy_handlers) == 1 and proxy_handlers[0].proxies == {}

    # 2) _connect_pinned_socket rejects non-public peer IP even if OS returns loopback peer
    class _LoopbackPeerSocket(_FakeSocket):
        def getpeername(self):
            return ("127.0.0.1", 80)

    monkeypatch.setattr(socket, "socket", _LoopbackPeerSocket)
    with pytest.raises(ValueError, match="non-public peer IP"):
        tools_mod._connect_pinned_socket(
            [(socket.AF_INET, socket.SOCK_STREAM, 6, ("93.184.216.34", 80))],
            timeout=1.0,
        )


def test_leak_29_esurge_rollout_respects_release_runtime_after_generation():
    """generate_rollout_completions and eSurgeRolloutGenerator must wire release_runtime_after_generation."""
    from easydel.trainers.esurge_rollout.generation import eSurgeRolloutConfig, eSurgeRolloutGenerator

    seen_flags: list[bool] = []

    class _StubTrainer:
        def rollout(self, *, prompts, apply_chat_template, config_overrides, release_runtime_after_generation):
            del apply_chat_template, config_overrides
            seen_flags.append(release_runtime_after_generation)
            return SimpleNamespace(generation_results=[f"out:{p}" for p in prompts])

    trainer = _StubTrainer()
    # Default is True (safe: does not pin KV cache on HBM)
    generate_rollout_completions(trainer, ["p1"])
    generate_rollout_completions(trainer, ["p2"], release_runtime_after_generation=False)

    gen = eSurgeRolloutGenerator(trainer, eSurgeRolloutConfig(release_runtime_after_generation=False))
    gen(["p3"])
    gen.generate(["p4"], release_runtime_after_generation=True)

    assert seen_flags == [True, False, False, True]


def test_reg_1_3_4_sdpo_ref_state_value_head_duck_type_and_grpo_legacy_ref_logps():
    """Verify Reg #1 (SDPO getattr ref_state), Reg #3 (ValueHead duck-type), and Reg #4 (5-arg compute_refmodel_logps)."""
    # Reg #3: CausalLMWithValueHead eSurge delegates work on duck-typed SimpleNamespace without _sync_esurge_cache_scope
    calls: list[tuple[str, tuple, dict]] = []
    inner = SimpleNamespace(
        _call_esurge_engine=lambda *a, **kw: calls.append(("call", a, kw)) or "ok",
        pause_esurge=lambda *a, **kw: calls.append(("pause", a, kw)),
        get_esurge=lambda *a, **kw: calls.append(("get", a, kw)) or "eng",
    )
    stub = SimpleNamespace(model=inner)
    assert CausalLMWithValueHead._call_esurge_engine(stub, "e", prompts=["hi"]) == "ok"
    assert CausalLMWithValueHead.get_esurge(stub) == "eng"
    CausalLMWithValueHead.pause_esurge(stub, release_model_state=True)
    assert [c[0] for c in calls] == ["call", "get", "pause"]

    # Reg #4: GRPOTrainer._call_compute_refmodel_logps falls back cleanly for legacy 5-arg callables
    grpo = object.__new__(GRPOTrainer)
    legacy_calls: list[int] = []
    grpo.compute_refmodel_logps = lambda gs, go, ids, mask, kw: (
        legacy_calls.append(int(ids.shape[0])),
        jnp.zeros((ids.shape[0], 2), dtype=jnp.float32),
    )[1]
    out = grpo._call_compute_refmodel_logps(
        None,
        None,
        jnp.ones((3, 4), dtype=jnp.int32),
        jnp.ones((3, 4), dtype=jnp.int32),
        {},
        prompt_length=2,
    )
    assert out.shape == (3, 2) and legacy_calls == [3]

    # Reg #1: SDPOTrainer.configure_functions does not crash when ref_state attribute is absent
    import easydel.trainers.self_distillation_policy_optimization.sdpo_trainer as sdpo_mod

    sdpo = object.__new__(SDPOTrainer)
    sdpo.arguments = SDPOConfig(max_prompt_length=4, max_completion_length=3, beta=0.0, teacher_regularization="none")
    sdpo.model_state = SimpleNamespace(
        model=SimpleNamespace(mesh=None, config=SimpleNamespace(max_position_embeddings=32)),
        graphstate=None,
        graphother=None,
    )
    sdpo._model = sdpo.model_state.model
    sdpo.state_shardings = None
    sdpo.scheduler = None
    sdpo.num_generations = 1
    sdpo._runtime_trace = lambda *a, **kw: None
    assert not hasattr(sdpo, "ref_state")
    orig_replicated = sdpo_mod.replicated_named_sharding
    orig_compile = sdpo_mod.compile_trainer_step
    try:
        sdpo_mod.replicated_named_sharding = lambda mesh: None
        sdpo_mod.compile_trainer_step = lambda fn, **kw: fn
        sdpo.arguments.ensure_checkpoint_path = lambda: None
        sdpo.arguments.get_streaming_checkpointer = lambda: None
        sdpo.configure_functions()
    finally:
        sdpo_mod.replicated_named_sharding = orig_replicated
        sdpo_mod.compile_trainer_step = orig_compile
    assert sdpo.compute_refmodel_logps is None
