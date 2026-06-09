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
"""MiniLLM reverse-KL trainer helpers."""

from __future__ import annotations

import typing as tp

import jax
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.utils import Registry
from easydel.utils.helpers import capture_time

from ..group_relative_policy_optimization import GRPOTrainer
from ..model_loading import disable_state_dropout, reject_string_model_id
from ..sdft_trainer import _zero_reward_func
from .minillm_config import MiniLLMConfig


@Registry.register("trainer", "minillm")
class MiniLLMTrainer(GRPOTrainer):
    """MiniLLM trainer with native JAX reverse-KL teacher advantages.

    The trainer uses the standard GRPO rollout pipeline, then computes
    sampled-token reverse-KL advantages from a teacher state over the same
    generated completions. The teacher must already be an EasyDeL module/state;
    string model identifiers are rejected to avoid implicit loading.
    """

    arguments: MiniLLMConfig

    def __init__(
        self,
        arguments: MiniLLMConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        teacher_model: EasyDeLBaseModule | EasyDeLState | None = None,
        reward_funcs: object | list[object] | None = None,
        train_dataset: object | None = None,
        eval_dataset: object | dict[str, object] | None = None,
        processing_class: object | None = None,
        reward_processing_classes: object | None = None,
        data_tokenize_fn: tp.Callable[..., object] | None = None,
    ) -> None:
        """Initialize MiniLLM with an explicit teacher state.

        Args:
            arguments: MiniLLM config controlling GRPO rollout behavior and
                reverse-KL advantage construction.
            model: Trainable EasyDeL policy module/state.
            teacher_model: Optional initialized teacher module/state. When
                omitted, the frozen GRPO reference state is reused as teacher.
            reward_funcs: Optional reward functions for the inherited GRPO
                rollout path. ``None`` installs a zero-reward fallback so
                MiniLLM can run as pure distillation.
            train_dataset: Prompt dataset for rollout generation.
            eval_dataset: Optional evaluation prompts.
            processing_class: Tokenizer/processor used by generation.
            reward_processing_classes: Optional reward model processors.
            data_tokenize_fn: Optional custom dataset tokenizer.

        Raises:
            TypeError: If ``arguments`` is not ``MiniLLMConfig``.
            ValueError: If ``teacher_model`` is a string id; EasyDeL expects
                callers to pass initialized models/states.
        """
        if not isinstance(arguments, MiniLLMConfig):
            raise TypeError(f"arguments must be MiniLLMConfig, got {type(arguments)}")
        if isinstance(teacher_model, str):
            reject_string_model_id(teacher_model, role="teacher model")
        super().__init__(
            arguments=arguments,
            model=model,
            reward_funcs=reward_funcs if reward_funcs is not None else _zero_reward_func,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            data_tokenize_fn=data_tokenize_fn,
        )
        if teacher_model is None:
            self.teacher_state = self.ref_state
        elif isinstance(teacher_model, EasyDeLState):
            self.teacher_state = teacher_model
        else:
            self.teacher_state = teacher_model.to_state(trainable_selector=arguments.trainable_selector)
        if arguments.disable_dropout:
            self.teacher_state = disable_state_dropout(self.teacher_state)

    @staticmethod
    def _compute_reverse_kl_advantage(
        *,
        student_logps: jax.Array,
        teacher_logps: jax.Array,
        completion_mask: jax.Array,
        gamma: float,
        length_normalization: bool,
        kd_temperature: float = 1.0,
        single_step_decomposition: bool = False,
    ) -> jax.Array:
        """Compute sampled-token reverse-KL advantages from teacher/student log-probs.

        The returned tensor is masked to completion tokens. When ``gamma`` is
        positive, rewards are accumulated backward with geometric discounting;
        otherwise the immediate teacher-minus-student reward is used directly.
        ``single_step_decomposition=True`` keeps the immediate per-token signal
        instead of a suffix return, matching MiniLLM's step-local update mode.
        """
        temperature = jnp.asarray(kd_temperature, dtype=jnp.float32)
        rewards = ((teacher_logps - student_logps) / temperature) * completion_mask
        if single_step_decomposition or gamma <= 0.0:
            return rewards
        positions = jnp.arange(rewards.shape[1], dtype=jnp.float32)
        gamma_pow = jnp.power(jnp.asarray(gamma, dtype=jnp.float32), positions)
        discounted = rewards * gamma_pow[None, :]
        advantages = jnp.flip(jnp.cumsum(jnp.flip(discounted, axis=1), axis=1), axis=1)
        if length_normalization:
            lengths = completion_mask.astype(jnp.float32) * gamma_pow[None, :]
            lengths = jnp.flip(jnp.cumsum(jnp.flip(lengths, axis=1), axis=1), axis=1)
            advantages = advantages / jnp.maximum(lengths, 1e-4)
        return advantages * completion_mask

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Attach MiniLLM reverse-KL advantages to the GRPO model batch.

        Base GRPO preprocessing builds prompt/completion tensors and initial
        metrics. This override scores those tensors with the teacher and current
        student, replaces ``advantages`` with reverse-KL advantages when
        enabled, and records timing for both extra forwards.
        """
        model_batch, metrics = super()._preprocess_batch_input(state=state, batch=batch, is_train=is_train)
        if not self.arguments.rkl_advantage:
            return model_batch, metrics

        prompt_ids = model_batch["prompt_ids"]
        prompt_mask = model_batch["prompt_mask"]
        completion_ids = model_batch["completion_ids"]
        completion_mask = model_batch["completion_mask"]
        generation_factor = int(completion_ids.shape[0]) // max(int(prompt_ids.shape[0]), 1)
        generation_factor = max(generation_factor, 1)
        input_ids = jnp.concatenate([prompt_ids.repeat(generation_factor, 0), completion_ids], axis=-1)
        attention_mask = jnp.concatenate([prompt_mask.repeat(generation_factor, 0), completion_mask], axis=-1)
        prompt_len = int(prompt_ids.shape[-1])
        with capture_time() as minillm_teacher_time_fn:
            teacher_logps = self.compute_state_logps(
                self.teacher_state,
                input_ids,
                attention_mask,
                None,
                prompt_length=prompt_len,
                logprob_vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
            )
        with capture_time() as minillm_student_time_fn:
            student_logps = self.compute_state_logps(
                state,
                input_ids,
                attention_mask,
                None,
                prompt_length=prompt_len,
                logprob_vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
            )
        rkl_advantage = self._compute_reverse_kl_advantage(
            student_logps=student_logps,
            teacher_logps=teacher_logps,
            completion_mask=completion_mask,
            gamma=float(self.arguments.gamma),
            length_normalization=bool(self.arguments.length_normalization),
            kd_temperature=float(self.arguments.kd_temperature),
            single_step_decomposition=bool(self.arguments.single_step_decomposition),
        )
        advantages = model_batch["advantages"]
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        model_batch = {
            **model_batch,
            "advantages": advantages + rkl_advantage,
            "minillm_teacher_per_token_logps": teacher_logps,
        }
        metrics = {
            **metrics,
            "minillm/rkl_advantage": float(jnp.mean(rkl_advantage * completion_mask)),
            "minillm/teacher_logps_time": minillm_teacher_time_fn(),
            "minillm/student_logps_time": minillm_student_time_fn(),
        }
        return model_batch, metrics
