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
from __future__ import annotations

import typing as tp

from eformer.loggings import get_logger
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.compiling_utils import ejit
from easydel.utils.helpers import capture_time

from ..prompt_transforms import GRPOPreprocessTransform
from ..trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_utils import resolve_straight_through_emulator
from ._fn import on_policy_distillation_step
from .on_policy_distillation_config import OnPolicyDistillationConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)


@Registry.register("trainer", "on_policy_distillation")
class OnPolicyDistillationTrainer(Trainer):
    """On-policy knowledge distillation trainer.

    Unlike standard (offline) distillation which distills on a fixed dataset,
    this trainer generates completions from prompts during training and then
    distills teacher knowledge on the generated sequences. This ensures the
    distillation signal is always on-policy with respect to the student.

    Training loop:
        1. Sample prompts from the dataset
        2. Student (or teacher) generates completions from prompts
        3. Both teacher and student score the generated sequences (forward pass)
        4. KL divergence loss is computed on the generated tokens
        5. Student is updated via gradient descent

    Attributes:
        teacher_state: State of the teacher model (frozen during training).
        arguments: OnPolicyDistillationConfig with training hyperparameters.

    Example:
        >>> config = OnPolicyDistillationConfig(
        ...     temperature=3.0,
        ...     alpha=0.9,
        ...     max_prompt_length=512,
        ...     max_completion_length=256,
        ...     learning_rate=2e-5,
        ... )
        >>> trainer = OnPolicyDistillationTrainer(
        ...     arguments=config,
        ...     student_model=student,
        ...     teacher_model=teacher,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer,
        ... )
        >>> trainer.train()
    """

    teacher_state: EasyDeLState
    arguments: OnPolicyDistillationConfig  # type hinting

    def __init__(
        self,
        arguments: OnPolicyDistillationConfig,
        processing_class: ProcessingClassType,
        student_model: EasyDeLBaseModule | EasyDeLState | None = None,
        teacher_model: EasyDeLBaseModule | EasyDeLState | None = None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
    ):
        tokenizer = processing_class
        if hasattr(processing_class, "tokenizer"):
            tokenizer = processing_class.tokenizer
        if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
            tokenizer.pad_token = tokenizer.eos_token
        assert isinstance(
            arguments, OnPolicyDistillationConfig
        ), "passed argument must be an `OnPolicyDistillationConfig`."

        self.arguments = arguments

        if not isinstance(student_model, EasyDeLState):
            student_model = student_model.to_state()
        if not isinstance(teacher_model, EasyDeLState):
            teacher_model = teacher_model.to_state()

        self.teacher_state = teacher_model
        self.processing_class = processing_class

        pad_token_id = getattr(processing_class, "pad_token_id", None)
        if pad_token_id is None and hasattr(processing_class, "tokenizer"):
            pad_token_id = getattr(processing_class.tokenizer, "pad_token_id", None)
        self.padding_value = 0 if pad_token_id is None else int(pad_token_id)

        # Wire generation config into arguments for generate_unified
        if getattr(self.arguments, "generation_num_return_sequences", None) is None:
            self.arguments.generation_num_return_sequences = arguments.num_generations_per_prompt
        if getattr(self.arguments, "generation_top_p", None) is None:
            self.arguments.generation_top_p = arguments.top_p
        if getattr(self.arguments, "generation_top_k", None) is None:
            self.arguments.generation_top_k = arguments.top_k
        if getattr(self.arguments, "generation_temperature", None) is None:
            self.arguments.generation_temperature = arguments.temperature_sampling
        if getattr(self.arguments, "generation_extra_kwargs", None) is None:
            self.arguments.generation_extra_kwargs = {}

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
        """Configure and JIT-compile the training and evaluation step functions.

        Returns:
            TrainerConfigureFunctionOutput with compiled step functions.
        """
        mesh = self.model.mesh
        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)

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
            int(self.arguments.logits_chunk_size),
        )

        static_argnames = tuple(range(3, 12))

        sharded_training_step_function = ejit(
            on_policy_distillation_step,
            in_shardings=(self.state_shardings, empty_sharding, self.teacher_state.shardings),
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
            int(self.arguments.logits_chunk_size),
        )

        sharded_evaluation_step_function = ejit(
            on_policy_distillation_step,
            in_shardings=(self.state_shardings, empty_sharding, self.teacher_state.shardings),
            out_shardings=empty_sharding,
            static_argnums=static_argnames,
        )

        flops_per_tkn = self.teacher_state.model.flops_per_token(include_loss=True, include_backward=True)
        self._extra_forward_flops_per_token = flops_per_tkn
        self._extra_backward_flops_per_token = flops_per_tkn

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
        """Generate completions from prompts and prepare the distillation batch.

        This is the core on-policy step: prompts are extracted from the batch,
        completions are generated (outside gradient computation), and the full
        sequences are assembled for the distillation step function.
        """
        batch = self._purify_batch(batch)

        with capture_time() as preprocessing_time_fn:
            prompt_ids, prompt_mask = batch["input_ids"], batch["attention_mask"]

            # Choose which model generates completions
            if self.arguments.generate_with_teacher:
                gen_state = self.teacher_state
            else:
                gen_state = state

            with capture_time() as generation_time_fn:
                results = self.generate_unified(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    state=gen_state,
                    apply_chat_template=False,
                    shard_inputs=False,
                    all_gather=False,
                )
                prompt_ids = results.prompt_ids
                prompt_mask = results.prompt_mask
                completion_ids = results.completion_ids

            generation_time = generation_time_fn()

            # Build completion mask (mask up to first EOS, inclusive)
            completion_mask = self._make_attn_mask(completion_ids)

            # Derive generation factor from actual shapes
            generation_factor = completion_ids.shape[0] // max(prompt_mask.shape[0], 1)
            generation_factor = max(generation_factor, 1)

            # Repeat prompt tensors to match the number of generations
            expanded_prompt_ids = prompt_ids.repeat(generation_factor, 0) if generation_factor > 1 else prompt_ids
            expanded_prompt_mask = prompt_mask.repeat(generation_factor, 0) if generation_factor > 1 else prompt_mask

            # Concatenate prompt + completion into full sequences
            input_ids_full = jnp.concatenate([expanded_prompt_ids, completion_ids], axis=1)
            attention_mask_full = jnp.concatenate([expanded_prompt_mask, completion_mask], axis=1)

            # Build a loss mask that is zero for prompt tokens and uses completion_mask for generated tokens
            prompt_zero_mask = jnp.zeros_like(expanded_prompt_mask)
            loss_mask = jnp.concatenate([prompt_zero_mask, completion_mask], axis=1)

            # Gather to replicated sharding
            input_ids_full = self._all_gather(input_ids_full)
            attention_mask_full = self._all_gather(attention_mask_full)
            loss_mask = self._all_gather(loss_mask)

        preprocessing_time = preprocessing_time_fn()

        completion_length = jnp.sum(completion_mask, -1)
        metrics_dict: dict[str, float | int | str] = {
            "completion_length": float(jnp.mean(completion_length)),
            "generation_time": generation_time,
            "preprocessing_time": preprocessing_time,
        }

        return (
            {
                "input_ids": input_ids_full,
                "attention_mask": attention_mask_full,
                "completion_mask": loss_mask,
            },
            metrics_dict,
        )

    @property
    def _train_shared_fn_extra_args(self) -> tuple[tp.Any]:
        return (self.teacher_state,)

    @property
    def _eval_shared_fn_extra_args(self) -> tuple[tp.Any]:
        return (self.teacher_state,)
