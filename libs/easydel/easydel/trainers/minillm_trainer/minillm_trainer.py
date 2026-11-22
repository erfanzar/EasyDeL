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

import collections.abc
import typing as tp

import jax
from jax import numpy as jnp
from spectrax import with_sharding_constraint

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.sharding import replicated_named_sharding
from easydel.utils import Registry
from easydel.utils.helpers import capture_time

from .._logprob_utils import (
    compute_per_token_logps_and_entropies_from_hidden_states,
    compute_token_logps_and_entropies_chunked,
    resolve_lmhead_chunksize,
)
from .._shared import drop_optimizer_state
from ..group_relative_policy_optimization import GRPOTrainer
from ..group_relative_policy_optimization._fn import _maybe_extend_inputs_embeds_for_scoring
from ..metrics import _host_scalar_float
from ..model_loading import disable_state_dropout, reject_string_model_id, zero_reward_func
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_utils import (
    compact_generation_model_kwargs,
    compile_trainer_step,
    extract_generation_model_kwargs,
    normalize_generation_model_kwargs,
    prepare_generation_model_kwargs_for_call,
    repeat_prompt_aligned_model_kwargs,
)
from .minillm_config import MiniLLMConfig


def _compute_logps_with_temperature(
    model: tp.Any,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    prompt_length: int,
    model_kwargs: dict[str, tp.Any] | None = None,
    logprob_vocab_chunk_size: int | None = None,
    temperature: float = 1.0,
) -> jax.Array:
    """Compute per-token log-probabilities with logit-level temperature scaling.

    Applies ``logits / temperature`` *before* ``log_softmax`` so the partition
    function reflects the softened distribution (unlike dividing T=1 logps).
    """
    model_kwargs = compact_generation_model_kwargs(
        normalize_generation_model_kwargs(model_kwargs, model_callable=getattr(model, "forward", model)),
    )
    model_kwargs = prepare_generation_model_kwargs_for_call(
        model_kwargs,
        target_sequence_length=input_ids.shape[-1],
        prompt_length=prompt_length,
    )
    model_kwargs = _maybe_extend_inputs_embeds_for_scoring(
        model,
        input_ids,
        model_kwargs,
        prompt_length=prompt_length,
    )
    call_kwargs = {
        "attention_mask": attention_mask,
        **model_kwargs,
    }
    if model_kwargs.get("inputs_embeds", None) is None:
        call_kwargs["input_ids"] = input_ids
    lmhead_chunksize = resolve_lmhead_chunksize(model)
    if lmhead_chunksize is not None:
        call_kwargs["apply_lm_head"] = False
    outputs = model(**call_kwargs)
    targets = input_ids[:, prompt_length:]
    inv_temp = jnp.asarray(1.0 / float(temperature), dtype=jnp.float32)

    if outputs.logits is None and lmhead_chunksize is not None:
        hidden_states = outputs.last_hidden_state
        if hidden_states is None:
            raise TypeError(
                f"{type(model).__name__} was called with `apply_lm_head=False` but did not return `last_hidden_state`."
            )
        hidden_states = hidden_states[:, prompt_length - 1 : -1, :]
        token_log_probs, _ = compute_per_token_logps_and_entropies_from_hidden_states(
            model=model,
            hidden_states=hidden_states,
            targets=targets,
            token_chunk_size=lmhead_chunksize,
            vocab_chunk_size=logprob_vocab_chunk_size,
            return_entropy=False,
            temperature=temperature,
        )
        return token_log_probs

    logits = outputs.logits
    if logits is None:
        raise TypeError(f"{type(model).__name__} did not return logits.")
    logits = logits[:, prompt_length - 1 : -1, :].astype(jnp.float32) * inv_temp
    token_log_probs, _ = compute_token_logps_and_entropies_chunked(
        logits,
        targets,
        return_entropy=False,
        chunk_size=logprob_vocab_chunk_size,
    )
    return token_log_probs


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
        if reward_funcs is not None:
            if isinstance(reward_funcs, collections.abc.Sequence) and not isinstance(reward_funcs, (str, bytes)):
                reward_funcs = [
                    drop_optimizer_state(rf) if isinstance(rf, EasyDeLState) else rf
                    for rf in reward_funcs
                ]
            elif isinstance(reward_funcs, EasyDeLState):
                reward_funcs = drop_optimizer_state(reward_funcs)
        super().__init__(
            arguments=arguments,
            model=model,
            reward_funcs=reward_funcs if reward_funcs is not None else zero_reward_func,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            data_tokenize_fn=data_tokenize_fn,
        )
        if teacher_model is None:
            if self.ref_state is None:
                raise ValueError(
                    "MiniLLM distills against a frozen teacher, but no `teacher_model` was given and "
                    f"`beta={arguments.beta}` disables the inherited GRPO reference model. Pass a "
                    "`teacher_model` or set a non-zero `beta` so a reference state is allocated."
                )
            self.teacher_state = self.ref_state
        elif isinstance(teacher_model, EasyDeLState):
            self.teacher_state = teacher_model
        else:
            teacher_model.eval()
            self.teacher_state = teacher_model.to_state()
        self.teacher_state = drop_optimizer_state(self.teacher_state)
        self.teacher_state = disable_state_dropout(self.teacher_state)
        self.teacher_state = self._share_esurge_cache_scope_if_compatible(
            self.teacher_state,
            primary_state=self.model_state,
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Compile GRPO step functions plus temperature-scaled MiniLLM logp scorer."""
        output = super().configure_functions()
        mesh = self.model.mesh
        empty_sharding = replicated_named_sharding(mesh)

        def _compute_state_logps_temperature(
            model_state: EasyDeLState,
            ids,
            mask,
            model_kwargs=None,
            *,
            prompt_length: int,
            logprob_vocab_chunk_size: int | None,
            temperature: float,
        ):
            module = model_state.model
            with module.mesh:
                ids = with_sharding_constraint(
                    ids,
                    self.arguments.step_partition_spec,
                    mesh=module.mesh,
                    ignore_mpmd=True,
                )
                mask = with_sharding_constraint(
                    mask,
                    self.arguments.step_partition_spec,
                    mesh=module.mesh,
                    ignore_mpmd=True,
                )
                return _compute_logps_with_temperature(
                    module,
                    ids,
                    mask,
                    prompt_length,
                    model_kwargs=model_kwargs,
                    logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                    temperature=temperature,
                )

        self.compute_state_logps_temperature = compile_trainer_step(
            _compute_state_logps_temperature,
            mesh=mesh,
            static_argnames=("prompt_length", "logprob_vocab_chunk_size", "temperature"),
            out_shardings=empty_sharding,
        )
        return output

    def _score_state_logps(
        self,
        model_state: EasyDeLState,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        model_kwargs: dict[str, tp.Any] | None,
        *,
        prompt_length: int,
        temperature: float,
    ) -> jax.Array:
        """Score per-token log-probabilities, applying ``temperature`` at the logit level."""
        if temperature == 1.0:
            return self.compute_state_logps(
                model_state,
                input_ids,
                attention_mask,
                model_kwargs,
                prompt_length=prompt_length,
                logprob_vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
            )
        compiled_fn = getattr(self, "compute_state_logps_temperature", None)
        if compiled_fn is not None:
            return compiled_fn(
                model_state,
                input_ids,
                attention_mask,
                model_kwargs,
                prompt_length=prompt_length,
                logprob_vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
                temperature=float(temperature),
            )
        return _compute_logps_with_temperature(
            model_state.model,
            input_ids,
            attention_mask,
            prompt_length,
            model_kwargs=model_kwargs,
            logprob_vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
            temperature=float(temperature),
        )

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

        Expects ``student_logps`` and ``teacher_logps`` to already reflect
        logit-level ``kd_temperature`` scaling (``log_softmax(logits / T)``).
        The returned tensor is masked to completion tokens. When ``gamma`` is
        positive and ``single_step_decomposition`` is ``False``, rewards are
        accumulated backward with geometric discounting; otherwise immediate
        rewards are used, still honoring ``length_normalization`` when enabled.
        """
        del kd_temperature
        mask_f32 = completion_mask.astype(jnp.float32)
        rewards = (teacher_logps.astype(jnp.float32) - student_logps.astype(jnp.float32)) * mask_f32
        if single_step_decomposition or gamma <= 0.0:
            if length_normalization:
                seq_lengths = jnp.maximum(jnp.sum(mask_f32, axis=-1, keepdims=True), 1e-4)
                return (rewards / seq_lengths) * mask_f32
            return rewards

        def discounted_suffix(values):
            def accumulate(carry, value):
                total = value + jnp.asarray(gamma, dtype=values.dtype) * carry
                return total, total

            _, suffix = jax.lax.scan(
                accumulate, jnp.zeros((values.shape[0],), dtype=values.dtype), values.T, reverse=True
            )
            return suffix.T

        advantages = discounted_suffix(rewards)
        if length_normalization:
            lengths = discounted_suffix(mask_f32)
            advantages = advantages / jnp.maximum(lengths, 1e-4)
        return advantages * mask_f32

    def _store_buffered_grpo_batch(
        self,
        model_batch: dict[str, jax.Array],
        metrics: dict[str, float | int | str],
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Defer GRPO rollout buffering until MiniLLM attaches reverse-KL advantages."""
        if getattr(self, "_minillm_defer_buffer_store", False):
            return model_batch, metrics
        return super()._store_buffered_grpo_batch(model_batch, metrics)

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, jax.Array],
        is_train: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, float | int | str]]:
        """Attach MiniLLM reverse-KL advantages to the GRPO model batch.

        Checks the GRPO rollout reuse buffer *before* running any teacher or
        student scoring so multi-iteration steps (``num_iterations > 1`` or
        ``steps_per_generation > 1``) reuse the cached off-policy advantages
        from the rollout step instead of rescoring with updated student weights.
        On fresh rollouts, defers GRPO buffer storage until after ``rkl_advantage``
        is attached (reusing ``old_per_token_logps`` when ``kd_temperature == 1.0``),
        then stores the completed batch into GRPO's reuse queue.
        """
        if is_train:
            cached = self._take_buffered_grpo_batch()
            if cached is not None:
                return cached

        self._minillm_defer_buffer_store = True
        try:
            model_batch, metrics = super()._preprocess_batch_input(state=state, batch=batch, is_train=is_train)
        finally:
            self._minillm_defer_buffer_store = False

        if not self.arguments.rkl_advantage:
            if is_train:
                return super()._store_buffered_grpo_batch(model_batch, metrics)
            return model_batch, metrics

        prompt_ids = model_batch["prompt_ids"]
        prompt_mask = model_batch["prompt_mask"]
        completion_ids = model_batch["completion_ids"]
        completion_mask = model_batch["completion_mask"]
        generation_factor = int(completion_ids.shape[0]) // max(int(prompt_ids.shape[0]), 1)
        generation_factor = max(generation_factor, 1)
        input_ids = jnp.concatenate([prompt_ids.repeat(generation_factor, 0), completion_ids], axis=-1)
        completion_attention_mask = model_batch.get("completion_attention_mask", completion_mask)
        attention_mask = jnp.concatenate([prompt_mask.repeat(generation_factor, 0), completion_attention_mask], axis=-1)
        prompt_len = int(prompt_ids.shape[-1])
        kd_temperature = float(self.arguments.kd_temperature)
        scoring_model_kwargs = repeat_prompt_aligned_model_kwargs(
            extract_generation_model_kwargs(model_batch),
            generation_factor,
            prompt_batch_size=prompt_ids.shape[0],
        )
        with capture_time() as minillm_teacher_time_fn:
            teacher_logps = self._score_state_logps(
                self.teacher_state,
                input_ids,
                attention_mask,
                scoring_model_kwargs,
                prompt_length=prompt_len,
                temperature=kd_temperature,
            )
        with capture_time() as minillm_student_time_fn:
            cached_old_logps = model_batch.get("old_per_token_logps")
            if kd_temperature == 1.0 and cached_old_logps is not None:
                student_logps = cached_old_logps
            else:
                student_logps = self._score_state_logps(
                    state,
                    input_ids,
                    attention_mask,
                    scoring_model_kwargs,
                    prompt_length=prompt_len,
                    temperature=kd_temperature,
                )
        rkl_advantage = self._compute_reverse_kl_advantage(
            student_logps=student_logps,
            teacher_logps=teacher_logps,
            completion_mask=completion_mask,
            gamma=float(self.arguments.gamma),
            length_normalization=bool(self.arguments.length_normalization),
            kd_temperature=kd_temperature,
            single_step_decomposition=bool(self.arguments.single_step_decomposition),
        )
        advantages = model_batch["advantages"]
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        model_batch = {
            **model_batch,
            "advantages": advantages + rkl_advantage,
        }
        mask_f32 = completion_mask.astype(jnp.float32)
        masked_rkl_mean = jnp.sum(rkl_advantage * mask_f32) / jnp.maximum(jnp.sum(mask_f32), 1.0)
        metrics = {
            **metrics,
            "minillm/rkl_advantage": _host_scalar_float(masked_rkl_mean),
            "minillm/teacher_logps_time": minillm_teacher_time_fn(),
            "minillm/student_logps_time": minillm_student_time_fn(),
        }
        if is_train:
            return super()._store_buffered_grpo_batch(model_batch, metrics)
        return model_batch, metrics
