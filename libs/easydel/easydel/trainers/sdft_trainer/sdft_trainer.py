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
"""Self-distillation fine-tuning trainer aliases."""

from __future__ import annotations

import collections.abc
import typing as tp

import jax
from jax import numpy as jnp

from easydel.infra.base_state import EasyDeLState
from easydel.utils import Registry

from .._shared import drop_optimizer_state
from ..self_distillation_policy_optimization import SDPOTrainer
from ..self_distillation_policy_optimization._fn import _apply_completion_loss_token_skip
from ._fn import _zero_reward_func
from .sdft_config import SDFTConfig


@Registry.register("trainer", "self_distillation")
class SelfDistillationTrainer(SDPOTrainer):
    """Alias to EasyDeL SDPO for the self-distillation trainer name.

    Runtime behavior is intentionally inherited from :class:`SDPOTrainer`. The
    class exists so registry lookup can distinguish the public trainer name
    without duplicating the SDPO implementation.
    """


@Registry.register("trainer", "sdft")
class SDFTTrainer(SDPOTrainer):
    """SDFT-compatible trainer backed by EasyDeL's native SDPO loss path.

    The trainer adds the SDFT-facing constructor defaults and zero-reward
    fallback while keeping generation, preprocessing, and optimization in the
    native SDPO code path.
    """

    arguments: SDFTConfig

    def __init__(
        self,
        arguments: SDFTConfig,
        model: object,
        reward_funcs: object | list[object] | None = None,
        train_dataset: object | None = None,
        eval_dataset: object | dict[str, object] | None = None,
        processing_class: object | None = None,
        reward_processing_classes: object | None = None,
        data_tokenize_fn: tp.Callable[..., object] | None = None,
        feedback_func: tp.Callable[..., list[str]] | None = None,
    ) -> None:
        """Create an SDFT trainer using the SDPO rollout and loss implementation.

        Args:
            arguments: SDFT config controlling teacher-template formatting,
                self-distillation generation, and inherited SDPO loss settings.
            model: Initialized EasyDeL policy module or state accepted by the
                SDPO base trainer.
            reward_funcs: Optional reward functions. When omitted, SDFT uses a
                zero-reward function because template feedback is the primary
                self-distillation signal.
            train_dataset: Dataset that may include ``context`` or
                ``privileged_context`` side-channel columns.
            eval_dataset: Optional evaluation dataset or named evaluation
                mapping.
            processing_class: Tokenizer or processor used by SDPO generation
                and preprocessing.
            reward_processing_classes: Optional reward processors forwarded to
                SDPO.
            data_tokenize_fn: Optional dataset tokenization override.
            feedback_func: Optional callable producing extra feedback strings
                for the teacher prompt template.
        """
        if not isinstance(arguments, SDFTConfig):
            raise TypeError(f"arguments must be SDFTConfig, got {type(arguments)}")
        self._sdft_user_feedback_func = feedback_func
        self._sdft_privileged_context: object | None = None
        self._sdft_active_generation_state: object | None = None
        self._sdft_rendered_teacher_prefixes: list[str] | None = None
        self._sdft_last_rollout_completion_mask: jax.Array | None = None
        self._sdft_in_preprocess: bool = False
        self._sdft_is_train: bool = True
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
            reward_funcs=reward_funcs if reward_funcs is not None else _zero_reward_func,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            data_tokenize_fn=data_tokenize_fn,
            feedback_func=self._sdft_template_feedback
            if arguments.teacher_prompt_template is not None
            else feedback_func,
        )

    @staticmethod
    def _coerce_template_values(value: object, target_len: int) -> list[str]:
        """Convert batch side-channel values into prompt-aligned strings.

        SDFT datasets commonly store privileged context as Python strings,
        lists of strings, NumPy arrays, or token tensors. This helper keeps the
        template path tolerant without changing the SDPO tensor batch consumed
        by the compiled training step.
        """
        if value is None:
            return [""] * target_len
        if isinstance(value, str):
            return [value] * target_len
        if isinstance(value, jax.Array):
            value = jax.device_get(value)
        if isinstance(value, (list, tuple)):
            values = ["" if item is None else str(item) for item in value]
        else:
            try:
                values = ["" if item is None else str(item) for item in value]  # type: ignore[operator]
            except TypeError:
                values = [str(value)]
        if not values:
            values = [""]
        if target_len == 0:
            return []
        if len(values) > target_len or target_len % len(values) != 0:
            raise ValueError(
                f"SDFT template values have {len(values)} entries, which cannot align "
                f"prompt-major with {target_len} rows."
            )
        repeats = target_len // len(values)
        return [value for value in values for _ in range(repeats)]

    def _sdft_template_feedback(
        self,
        prompts: list[str],
        completions: list[str],
        rewards: list[float],
    ) -> list[str]:
        """Build teacher-context separators from SDFT privileged-context rows.

        The returned strings are inserted between prompt and completion by the
        inherited SDPO batch builder. When a user feedback callable is provided,
        its output is also available to the template as ``{feedback}``.
        """
        template = self.arguments.teacher_prompt_template
        if template is None:
            return ["" for _ in completions]
        user_feedbacks = [""] * len(completions)
        if self._sdft_user_feedback_func is not None:
            raw = self._sdft_user_feedback_func(prompts=prompts, completions=completions, rewards=rewards)
            user_feedbacks = self._coerce_template_values(raw, len(completions))
        privileged_context = self._coerce_template_values(self._sdft_privileged_context, len(completions))
        prompt_values = self._coerce_template_values(prompts, len(completions))
        return [
            template.format(
                prompt=prompt,
                privileged_context=context,
                completion=completion,
                feedback=feedback,
            )
            for prompt, context, completion, feedback in zip(
                prompt_values,
                privileged_context,
                completions,
                user_feedbacks,
                strict=False,
            )
        ]

    def _get_rich_feedback(
        self,
        completion_prompts: list,
        completions: list,
        rewards: jax.Array,
        generation_factor: int | None = None,
    ) -> list[str]:
        """Render SDFT teacher prefixes directly without SDPO failure framing.

        When ``teacher_prompt_template`` is configured, SDFT formats the full
        teacher prefix directly via :meth:`_sdft_template_feedback` rather than
        passing it through SDPO's ``"unsuccessful earlier attempt"`` wrapper or
        skipping context whenever ``reward > success_reward_threshold``.
        """
        if getattr(self.arguments, "teacher_prompt_template", None) is None:
            return super()._get_rich_feedback(
                completion_prompts=completion_prompts,
                completions=completions,
                rewards=rewards,
                generation_factor=generation_factor,
            )
        if not completions:
            self._sdft_rendered_teacher_prefixes = []
            return []
        rewards_list = [float(r) for r in jax.device_get(rewards)]
        prompt_strs = (
            completion_prompts if isinstance(completion_prompts[0], str) else [str(p) for p in completion_prompts]
        )
        completion_strs = completions if isinstance(completions[0], str) else [str(c) for c in completions]
        rendered = self._sdft_template_feedback(
            prompts=prompt_strs,
            completions=completion_strs,
            rewards=rewards_list,
        )
        self._sdft_rendered_teacher_prefixes = rendered
        return rendered

    def _tokenize_feedback_separators(
        self,
        feedback_texts: list[str],
    ) -> tuple[jax.Array, jax.Array]:
        """Avoid truncating full SDFT teacher prefixes to ``_effective_feedback_length``."""
        if (
            getattr(self, "_sdft_in_preprocess", False)
            and getattr(self.arguments, "teacher_prompt_template", None) is not None
        ):
            width = max(0, int(getattr(self, "_effective_feedback_length", 0)))
            empty = jnp.zeros((len(feedback_texts), width), dtype=jnp.int32)
            return empty, empty
        return super()._tokenize_feedback_separators(feedback_texts)

    def _tokenize_teacher_prefixes(
        self,
        teacher_prefixes: list[str],
        *,
        target_length: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Left-pad rendered SDFT teacher prefixes to ``target_length``."""
        if target_length <= 0:
            empty = jnp.zeros((len(teacher_prefixes), 0), dtype=jnp.int32)
            return empty, empty
        call_kwargs: dict[str, tp.Any] = {
            "add_special_tokens": False,
            "max_length": int(target_length),
            "truncation": True,
            "padding": "max_length",
            "padding_side": "left",
            "return_tensors": "np",
        }
        try:
            encoded = self.processing_class(teacher_prefixes, **call_kwargs)
        except TypeError:
            call_kwargs.pop("padding_side", None)
            tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
            had_padding_side = hasattr(tokenizer, "padding_side")
            previous_padding_side = getattr(tokenizer, "padding_side", None)
            if had_padding_side:
                tokenizer.padding_side = "left"
            try:
                encoded = self.processing_class(teacher_prefixes, **call_kwargs)
            finally:
                if had_padding_side:
                    tokenizer.padding_side = previous_padding_side
        return jnp.asarray(encoded["input_ids"]), jnp.asarray(encoded["attention_mask"])

    def rollout(self, *args, **kwargs):
        """Route generation to the teacher state when ``generate_from_teacher=True``."""
        gen_state = getattr(self, "_sdft_active_generation_state", None)
        if gen_state is not None:
            kwargs["state"] = gen_state
        results = super().rollout(*args, **kwargs)
        if getattr(self, "_sdft_in_preprocess", False):
            self._sdft_last_rollout_completion_mask = getattr(results, "completion_mask", None)
        return results

    def _log_training_generations_to_wandb(self, *args, **kwargs) -> None:
        """Preserve live student step while tagging teacher-generated rollouts accurately."""
        if getattr(self, "_sdft_in_preprocess", False) and not getattr(self, "_sdft_is_train", True):
            return
        if getattr(self, "_sdft_active_generation_state", None) is not None:
            kwargs["source"] = "teacher"
        return super()._log_training_generations_to_wandb(*args, **kwargs)

    def _preprocess_batch_input(
        self,
        state,
        batch: dict[str, object],
        is_train: bool,
    ):
        """Capture SDFT context columns around inherited SDPO preprocessing.

        Renders SDFT ``teacher_prompt_template`` prefixes directly into the
        teacher prefix window (without duplicating ``{prompt}`` or applying
        SDPO's failure wrapper), keeps the live student ``state`` for W&B step
        logging even when ``generate_from_teacher=True``, and computes
        ``num_items_in_batch`` after applying ``num_loss_tokens_to_skip``.
        """
        batch = self._apply_user_data_collator(batch)
        context = batch.get("privileged_context")
        if context is None:
            context = batch.get("context")
        self._sdft_privileged_context = context
        self._sdft_in_preprocess = True
        self._sdft_is_train = is_train
        self._sdft_rendered_teacher_prefixes = None
        self._sdft_last_rollout_completion_mask = None
        self._sdft_active_generation_state = (
            self.ref_state
            if getattr(self.arguments, "generate_from_teacher", False) and getattr(self, "ref_state", None) is not None
            else None
        )
        try:
            out_batch, metrics_dict = super()._preprocess_batch_input(state=state, batch=batch, is_train=is_train)
            if isinstance(out_batch, dict):
                out_batch = dict(out_batch)
                rollout_mask = self._sdft_last_rollout_completion_mask
                if rollout_mask is not None and "completion_mask" in out_batch:
                    gathered_rollout_mask = self._all_gather(rollout_mask)
                    if gathered_rollout_mask.shape == out_batch["completion_mask"].shape:
                        updated_completion_mask = out_batch["completion_mask"] * gathered_rollout_mask.astype(
                            out_batch["completion_mask"].dtype
                        )
                        out_batch["completion_mask"] = updated_completion_mask
                        if "teacher_mask" in out_batch and "completion_ids" in out_batch:
                            prefix_len = int(out_batch["teacher_mask"].shape[1]) - int(
                                out_batch["completion_ids"].shape[1]
                            )
                            out_batch["teacher_mask"] = jnp.concatenate(
                                [out_batch["teacher_mask"][:, :prefix_len], updated_completion_mask],
                                axis=1,
                            )

                rendered_prefixes = self._sdft_rendered_teacher_prefixes
                if (
                    rendered_prefixes is not None
                    and "teacher_ids" in out_batch
                    and "completion_ids" in out_batch
                    and "completion_mask" in out_batch
                ):
                    teacher_prefix_len = int(out_batch["teacher_ids"].shape[1]) - int(
                        out_batch["completion_ids"].shape[1]
                    )
                    prefix_ids, prefix_mask = self._tokenize_teacher_prefixes(
                        rendered_prefixes,
                        target_length=teacher_prefix_len,
                    )
                    prefix_ids = self._all_gather(prefix_ids)
                    prefix_mask = self._all_gather(prefix_mask)
                    out_batch["teacher_ids"] = jnp.concatenate([prefix_ids, out_batch["completion_ids"]], axis=1)
                    out_batch["teacher_mask"] = jnp.concatenate([prefix_mask, out_batch["completion_mask"]], axis=1)

                if "completion_mask" in out_batch:
                    skip_tokens = int(getattr(self.arguments, "num_loss_tokens_to_skip", 0) or 0)
                    effective_mask = _apply_completion_loss_token_skip(out_batch["completion_mask"], skip_tokens)
                    out_batch["num_items_in_batch"] = jnp.sum(effective_mask)
            return out_batch, metrics_dict
        finally:
            self._sdft_privileged_context = None
            self._sdft_active_generation_state = None
            self._sdft_rendered_teacher_prefixes = None
            self._sdft_last_rollout_completion_mask = None
            self._sdft_in_preprocess = False
