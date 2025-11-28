# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
from collections import defaultdict
from functools import partial

import jax
from eformer.loggings import get_logger
from jax import jit
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.compiling_utils import ejit

from ..base_trainer import TrainerConfigureFunctionOutput
from ..prompt_transforms import ORPOPreprocessTransform
from ..trainer.trainer import Trainer
from ..utils import DPODataCollatorWithPaddingGrain, DPODataCollatorWithPaddingTFDS
from ._fn import concatenated_forward, orpo_step
from .orpo_config import ORPOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)


@Registry.register("trainer", "orpo")
class ORPOTrainer(Trainer):
    """Odds Ratio Preference Optimization trainer.

    ORPO is a reference-free preference optimization method that directly
    optimizes the odds ratio between preferred and rejected responses.
    Unlike DPO, ORPO doesn't require a reference model, making it more
    memory-efficient while maintaining competitive performance.

    The trainer uses lazy preprocessing transforms that are applied during
    iteration, providing better performance than eager HF .map() calls.

    Attributes:
        arguments: ORPOConfig with training hyperparameters
        processing_class: Tokenizer or processor for text encoding
        padding_value: Token ID used for padding

    Example:
        >>> config = ORPOConfig(
        ...     per_device_train_batch_size=4,
        ...     orpo_beta=0.1,
        ...     learning_rate=5e-6,
        ...     max_prompt_length=512,
        ...     max_completion_length=512
        ... )
        >>> trainer = ORPOTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     train_dataset=preference_dataset,
        ...     processing_class=tokenizer
        ... )
        >>> trainer.train()
    """

    arguments: ORPOConfig

    def __init__(
        self,
        arguments: ORPOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None = None,
        data_collator: DPODataCollatorWithPaddingTFDS | DPODataCollatorWithPaddingGrain | None = None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType = None,
    ):
        if arguments is None:
            raise ValueError("arguments cannot be None")
        if not isinstance(arguments, ORPOConfig):
            raise TypeError(f"arguments must be ORPOConfig, got {type(arguments)}")
        if processing_class is None:
            raise ValueError("processing_class must be specified to tokenize an ORPO dataset.")

        self.arguments = arguments
        self.truncation_mode = arguments.truncation_mode
        self.processing_class = processing_class
        self.is_encoder_decoder = arguments.is_encoder_decoder

        # Determine padding value
        if arguments.padding_value is not None:
            self.padding_value = arguments.padding_value
        else:
            if hasattr(processing_class, "pad_token_id") and processing_class.pad_token_id is not None:
                self.padding_value = processing_class.pad_token_id
            elif hasattr(processing_class, "tokenizer") and processing_class.tokenizer.pad_token_id is not None:
                self.padding_value = processing_class.tokenizer.pad_token_id
            else:
                raise ValueError(
                    "`padding_value` is not specified in `ORPOConfig`, and `pad_token_id` is missing in the "
                    "`processing_class`. Please set `tokenizer.pad_token` before instantiating the trainer."
                )
        arguments.padding_value = self.padding_value

        # Setup data collators
        self.input_data_collator_tfds = (
            DPODataCollatorWithPaddingTFDS(
                max_prompt_length=arguments.max_prompt_length,
                max_completion_length=arguments.max_completion_length,
                pad_token_id=self.padding_value,
                label_pad_token_id=arguments.label_pad_token_id,
                is_encoder_decoder=arguments.is_encoder_decoder,
                prepadded=True,
            )
            if data_collator is None
            else data_collator
        )
        self.input_data_collator_grain = (
            DPODataCollatorWithPaddingGrain(
                max_prompt_length=arguments.max_prompt_length,
                max_completion_length=arguments.max_completion_length,
                pad_token_id=self.padding_value,
                label_pad_token_id=arguments.label_pad_token_id,
                is_encoder_decoder=arguments.is_encoder_decoder,
                prepadded=True,
            )
            if data_collator is None
            else data_collator
        )

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        if not isinstance(model, EasyDeLState):
            model = model.to_state()

        super().__init__(
            model_state=model,
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            data_collator=None,
            processing_class=processing_class,
        )

    def _get_preprocess_transform(self) -> ORPOPreprocessTransform | None:
        """Get ORPO preprocessing transform for ShardedDataSource."""

        if self._is_pretokenized():
            return None

        return ORPOPreprocessTransform(
            tokenizer=self.processing_class,
            max_prompt_length=self.arguments.max_prompt_length,
            max_completion_length=self.arguments.max_completion_length,
            label_pad_token_id=self.arguments.label_pad_token_id,
        )

    def _is_pretokenized(self) -> bool:
        """Check if dataset already has tokenized fields."""
        if self._train_source is None:
            return False
        try:
            sample = next(iter(self._train_source.open_shard(self._train_source.shard_names[0])))
            return "prompt_input_ids" in sample
        except (StopIteration, IndexError):
            return False

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Configure and JIT-compile training and evaluation step functions."""
        mesh = self.model.mesh
        empty_sharding = jax.sharding.NamedSharding(spec=PartitionSpec(), mesh=mesh)

        partial_concatenated_forward = partial(
            concatenated_forward,
            is_encoder_decoder=self.arguments.is_encoder_decoder,
            label_pad_token_id=self.arguments.label_pad_token_id,
            padding_value=self.arguments.padding_value,
            max_length=self.arguments.max_length,
        )
        jited_concatenated_forward = ejit(
            partial_concatenated_forward,
            static_argnames=("is_encoder_decoder", "label_pad_token_id", "padding_value", "max_length"),
        )

        self._train_shared_fn_static_args = (
            partial_concatenated_forward,
            self.arguments.beta,
            self.scheduler,
            "train",
            self.arguments.loss_config,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
        )

        static_argnums = (2, 3, 4, 5, 6, 7, 8)

        sharded_training_step_function = jit(
            orpo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=static_argnums,
        )

        self._eval_shared_fn_static_args = (
            partial_concatenated_forward,
            self.arguments.beta,
            self.scheduler,
            "eval",
            self.arguments.loss_config,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
        )

        sharded_evaluation_step_function = jit(
            orpo_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=static_argnums,
        )

        self._extra_forward_flops_per_token = 0
        self._extra_backward_flops_per_token = 0

        self.arguments.ensure_checkpoint_path()
        self.concatenated_forward = jited_concatenated_forward
        checkpoint_manager = self.arguments.get_streaming_checkpointer()

        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
        )

    def create_grain_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create data collection function for Grain batching."""
        return self.input_data_collator_grain

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create data collection function for TFDS batching."""
        return self.input_data_collator_tfds
