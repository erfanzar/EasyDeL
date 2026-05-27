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

import typing as tp

import jax

from easydel.utils import Registry

from ..self_distillation_policy_optimization import SDPOTrainer
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
        self._sdft_privileged_context: list[str] | None = None
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
        if len(values) >= target_len:
            return values[:target_len]
        repeats = (target_len + len(values) - 1) // len(values)
        return (values * repeats)[:target_len]

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
        del rewards
        template = self.arguments.teacher_prompt_template
        if template is None:
            return ["" for _ in completions]
        user_feedbacks = [""] * len(completions)
        if self._sdft_user_feedback_func is not None:
            raw = self._sdft_user_feedback_func(prompts=prompts, completions=completions, rewards=[0.0] * len(prompts))
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

    def _preprocess_batch_input(
        self,
        state,
        batch: dict[str, object],
        is_train: bool,
    ):
        """Capture SDFT context columns around inherited SDPO preprocessing.

        The SDPO base path consumes tensorized prompt/completion fields. SDFT
        additionally needs raw privileged context strings while formatting
        teacher feedback, so this method stores a prompt-aligned context list
        for the duration of preprocessing and clears it in ``finally`` to avoid
        leaking context between batches. When ``generate_from_teacher=True``,
        rollout generation is delegated to the frozen reference state while the
        returned batch is still optimized by the normal training step.
        """
        context = batch.get("privileged_context")
        if context is None:
            context = batch.get("context")
        self._sdft_privileged_context = self._coerce_template_values(context, int(batch["input_ids"].shape[0]))
        try:
            generation_state = (
                self.ref_state if self.arguments.generate_from_teacher and self.ref_state is not None else state
            )
            return super()._preprocess_batch_input(state=generation_state, batch=batch, is_train=is_train)
        finally:
            self._sdft_privileged_context = None
