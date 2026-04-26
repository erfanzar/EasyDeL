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

"""Sparse (gray-box) knowledge distillation trainer.

The student generates completions on-policy, the teacher scores them with
top-k logprobs, and the student minimises partial KL divergence between
its own distribution and the teacher's sparse top-k distribution.
"""

from __future__ import annotations

import typing as tp

import jax
from eformer.loggings import get_logger
from jax import numpy as jnp

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.sharding import replicated_named_sharding
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.helpers import capture_time

from ..prompt_transforms import GRPOPreprocessTransform
from ..trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_utils import (
    compile_trainer_step,
    filter_kwargs_for_callable,
    resolve_straight_through_emulator,
    sanitize_model_call_kwargs,
)
from ._fn import sparse_distillation_step
from .sparse_distillation_config import SparseDistillationConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource

SparseTeacherFn = tp.Callable[
    [jax.Array, jax.Array, list[str]],
    tuple[jax.Array, jax.Array],
]

logger = get_logger(__name__)


@Registry.register("trainer", "sparse_distillation")
class SparseDistillationTrainer(Trainer):
    """Sparse (gray-box) knowledge distillation trainer.

    The student generates completions on-policy, the teacher scores them with
    top-k logprobs, and the student minimises partial KL divergence.  This
    works with API-based teachers that expose limited logprob information.

    Two teacher modes are supported:
        1. **Local teacher model** (``teacher_model``): An EasyDeL model whose
           full logits are computed, then only the top-k are retained.
        2. **External teacher function** (``teacher_fn``): A callable
           ``(input_ids, attention_mask, prompt_texts) -> (top_k_indices, top_k_logprobs)``
           for API-based teachers.  ``prompt_texts`` is a ``list[str]``
           of decoded prompts so the function can forward raw text to an
           external API when the teacher uses a different tokenizer.

    Training loop:
        1. Sample prompts from the dataset
        2. Student generates completions (on-policy)
        3. Teacher scores the full sequences → top-k logprobs
        4. Student minimises partial KL on the generated tokens

    Example:
        >>> config = SparseDistillationConfig(
        ...     top_k_teacher=20,
        ...     temperature=3.0,
        ...     alpha=0.9,
        ...     max_prompt_length=512,
        ...     max_completion_length=256,
        ...     learning_rate=2e-5,
        ... )
        >>> trainer = SparseDistillationTrainer(
        ...     arguments=config,
        ...     student_model=student,
        ...     teacher_model=teacher,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer,
        ... )
        >>> trainer.train()
    """

    teacher_state: EasyDeLState | None
    teacher_fn: SparseTeacherFn | None
    arguments: SparseDistillationConfig

    def __init__(
        self,
        arguments: SparseDistillationConfig,
        processing_class: ProcessingClassType,
        student_model: EasyDeLBaseModule | EasyDeLState | None = None,
        teacher_model: EasyDeLBaseModule | EasyDeLState | None = None,
        teacher_fn: SparseTeacherFn | None = None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
    ):
        tokenizer = processing_class
        if hasattr(processing_class, "tokenizer"):
            tokenizer = processing_class.tokenizer
        if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
            tokenizer.pad_token = tokenizer.eos_token

        if not isinstance(arguments, SparseDistillationConfig):
            raise TypeError("passed argument must be a `SparseDistillationConfig`.")
        if teacher_model is None and teacher_fn is None:
            raise ValueError("Either `teacher_model` or `teacher_fn` must be provided.")

        self.arguments = arguments

        if not isinstance(student_model, EasyDeLState):
            student_model = student_model.to_state(trainable_selector=arguments.trainable_selector)

        if teacher_model is not None and not isinstance(teacher_model, EasyDeLState):
            teacher_model = teacher_model.to_state(trainable_selector=arguments.trainable_selector)

        self.teacher_state = teacher_model
        self.teacher_fn = teacher_fn
        self.processing_class = processing_class

        pad_token_id = getattr(processing_class, "pad_token_id", None)
        if pad_token_id is None and hasattr(processing_class, "tokenizer"):
            pad_token_id = getattr(processing_class.tokenizer, "pad_token_id", None)
        self.padding_value = 0 if pad_token_id is None else int(pad_token_id)

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=student_model,
            data_collator=None,
            processing_class=processing_class,
        )

    def _get_preprocess_transform(self) -> GRPOPreprocessTransform | None:
        """Get preprocessing transform for prompt-only datasets."""
        if self._is_pretokenized():
            return None
        return GRPOPreprocessTransform(
            tokenizer=self.processing_class,
            max_prompt_length=self.arguments.max_prompt_length,
            skip_apply_chat_template=self.arguments.skip_apply_chat_template,
        )

    def _is_pretokenized(self) -> bool:
        """Check whether the source already yields token IDs."""
        if self._train_source is None:
            return False
        try:
            sample = next(iter(self._train_source.open_shard(self._train_source.shard_names[0])))
            return "input_ids" in sample
        except (StopIteration, IndexError):
            return False

    def create_grain_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create Grain data collator for prompt-only batches."""
        from ..utils import GRPODataCollatorGrain

        return GRPODataCollatorGrain(
            max_prompt_length=self.arguments.max_prompt_length,
            pad_token_id=self.padding_value,
        )

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create TFDS data collator for prompt-only batches."""
        from ..utils import GRPODataCollatorTFDS

        return GRPODataCollatorTFDS(
            max_prompt_length=self.arguments.max_prompt_length,
            pad_token_id=self.padding_value,
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Configure and JIT-compile training and evaluation step functions."""
        mesh = self.model.mesh
        empty_sharding = replicated_named_sharding(mesh)

        straight_through_emulator = resolve_straight_through_emulator(
            quantization_mode=self.arguments.quantization_mode,
            quantization_group_size=self.arguments.quantization_group_size,
            quantization_bits=self.arguments.quantization_bits,
            tensor_straight_through=self.arguments.tensor_straight_through,
            straight_through_emulator=self.arguments.straight_through_emulator,
        )

        self._train_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,  # is_training
            self.arguments.temperature,
            self.arguments.alpha,
            straight_through_emulator,
        )

        static_argnames = tuple(range(2, 10))

        sharded_training_step_function = compile_trainer_step(
            sparse_distillation_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnames,
        )

        self._eval_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            False,  # is_training
            self.arguments.temperature,
            self.arguments.alpha,
            None,  # straight_through_emulator
        )

        sharded_evaluation_step_function = compile_trainer_step(
            sparse_distillation_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnames,
        )

        if self.teacher_state is not None:
            flops_per_tkn = self.teacher_state.model.flops_per_token(include_loss=True, include_backward=False)
            self._extra_forward_flops_per_token = flops_per_tkn

        self.arguments.ensure_checkpoint_path()
        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=self.arguments.get_streaming_checkpointer(),
        )

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, tp.Any],
        is_train: bool,
    ) -> tuple[dict[str, tp.Any], dict[str, float | int | str]]:
        """Generate completions and score with teacher to get top-k logprobs.

        1. Student generates completions (on-policy).
        2. Full sequences (prompt + completion) are assembled.
        3. Teacher scores the sequences → top-k logprobs are extracted.
        4. The batch returned to the step function contains the pre-computed
           teacher top-k data alongside student inputs.
        """
        batch = self._purify_batch(batch)

        scoring_time: float = 0.0
        with capture_time() as preprocessing_time_fn:
            prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]

            with capture_time() as generation_time_fn:
                results = self.generate_unified(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    state=state,
                    apply_chat_template=False,
                    shard_inputs=False,
                    all_gather=False,
                )
                prompt_ids = results.prompt_ids
                prompt_mask = results.prompt_mask
                completion_ids = results.completion_ids
            generation_time = generation_time_fn()

            completion_mask = self._make_attn_mask(completion_ids)

            # Handle generation factor (num_generations_per_prompt > 1)
            generation_factor = completion_ids.shape[0] // max(prompt_mask.shape[0], 1)
            generation_factor = max(generation_factor, 1)
            expanded_prompt_ids = prompt_ids.repeat(generation_factor, 0) if generation_factor > 1 else prompt_ids
            expanded_prompt_mask = prompt_mask.repeat(generation_factor, 0) if generation_factor > 1 else prompt_mask

            input_ids_full = jnp.concatenate([expanded_prompt_ids, completion_ids], axis=1)
            attention_mask_full = jnp.concatenate([expanded_prompt_mask, completion_mask], axis=1)

            # Loss mask: zero for prompt, completion_mask for generated tokens
            prompt_zero_mask = jnp.zeros_like(expanded_prompt_mask)
            loss_mask = jnp.concatenate([prompt_zero_mask, completion_mask], axis=1)

            with capture_time() as scoring_time_fn:
                if self.teacher_fn is not None:
                    prompt_texts = self.processing_class.batch_decode(
                        expanded_prompt_ids,
                        skip_special_tokens=True,
                    )
                    top_k_indices, top_k_logprobs = self.teacher_fn(
                        input_ids_full,
                        attention_mask_full,
                        prompt_texts,
                    )
                    top_k_indices = jnp.asarray(top_k_indices)
                    top_k_logprobs = jnp.asarray(top_k_logprobs)
                else:
                    teacher_kwargs: dict[str, tp.Any] = {
                        "input_ids": input_ids_full,
                        "attention_mask": attention_mask_full,
                    }
                    teacher_kwargs = filter_kwargs_for_callable(
                        self.teacher_state.model.forward,
                        teacher_kwargs,
                    )
                    teacher_kwargs = sanitize_model_call_kwargs(teacher_kwargs)
                    teacher_outputs = self.teacher_state.model(**teacher_kwargs)

                    # Extract top-k from full logits
                    k = self.arguments.top_k_teacher
                    teacher_log_probs = jax.nn.log_softmax(
                        teacher_outputs.logits.astype(jnp.float32),
                        axis=-1,
                    )
                    top_k_logprobs, top_k_indices = jax.lax.top_k(teacher_log_probs, k)  # both [B, L, K]
            scoring_time = scoring_time_fn()

            # Gather to replicated sharding
            input_ids_full = self._all_gather(input_ids_full)
            attention_mask_full = self._all_gather(attention_mask_full)
            loss_mask = self._all_gather(loss_mask)
            top_k_indices = self._all_gather(top_k_indices)
            top_k_logprobs = self._all_gather(top_k_logprobs)

        preprocessing_time = preprocessing_time_fn()

        completion_length = jnp.sum(completion_mask, -1)
        metrics_dict: dict[str, float | int | str] = {
            "completion_length": float(jnp.mean(completion_length)),
            "generation_time": generation_time,
            "scoring_time": scoring_time,
            "preprocessing_time": preprocessing_time,
        }
        self._log_training_generations_to_wandb(
            state=state,
            prompts=expanded_prompt_ids,
            prompt_mask=expanded_prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            generation_time=generation_time,
            source="student",
        )

        return (
            {
                "input_ids": input_ids_full,
                "attention_mask": attention_mask_full,
                "completion_mask": loss_mask,
                "teacher_top_k_indices": top_k_indices,
                "teacher_top_k_logprobs": top_k_logprobs,
            },
            metrics_dict,
        )
