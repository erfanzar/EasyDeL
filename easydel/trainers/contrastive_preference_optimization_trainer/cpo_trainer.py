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
from functools import partial

import jax
from eformer.loggings import get_logger
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.compiling_utils import ejit

from ..base_trainer import TrainerConfigureFunctionOutput
from ..prompt_transforms import CPOPreprocessTransform
from ..trainer.trainer import Trainer
from ..training_configurations import MetricsType
from ..training_utils import resolve_straight_through_emulator
from ..utils import (
    DataCollatorForPreferenceGrain,
    DataCollatorForPreferenceTFDS,
)
from ._fn import concatenated_forward, evaluation_step, training_step
from .cpo_config import CPOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)


@Registry.register("trainer", "cpo")
class CPOTrainer(Trainer):
    """Contrastive Preference Optimization (CPO) trainer.

    Implements CPO training which aligns language models using preference pairs.
    Supports multiple loss variants including sigmoid, hinge, IPO, SimPO, and AlphaPO.

    Args:
        arguments: CPO-specific training configuration.
        model: Policy model to train.
        processing_class: Tokenizer or processor.
        train_dataset: Training dataset with prompt, chosen, and rejected fields.
        eval_dataset: Optional evaluation dataset.
        data_collator: Optional custom data collator.
    """

    arguments: CPOConfig

    def __init__(
        self,
        arguments: CPOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        processing_class: ProcessingClassType,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        data_collator: DataCollatorForPreferenceTFDS | DataCollatorForPreferenceGrain | None = None,
    ):
        if not isinstance(arguments, CPOConfig):
            raise TypeError(f"`arguments` must be a `CPOConfig`, received {type(arguments)}.")
        if processing_class is None:
            raise ValueError("`processing_class` must be provided to tokenize a CPO dataset.")
        if model is None:
            raise ValueError("A policy model must be supplied to the CPO trainer.")

        self.arguments = arguments
        self.processing_class = processing_class
        self.truncation_mode = arguments.truncation_mode
        self._stored_metrics = {}

        if isinstance(model, EasyDeLState):
            model_state = model
        else:
            model_state = model.to_state()

        if arguments.is_encoder_decoder is not None:
            self.is_encoder_decoder = arguments.is_encoder_decoder
        else:
            self.is_encoder_decoder = getattr(model_state.model.config, "is_encoder_decoder", False)
            self.arguments.is_encoder_decoder = self.is_encoder_decoder

        if getattr(processing_class, "pad_token_id", None) is None and hasattr(processing_class, "eos_token"):
            processing_class.pad_token = processing_class.eos_token

        if arguments.padding_value is not None:
            self.padding_value = arguments.padding_value
        else:
            pad_token_id = getattr(processing_class, "pad_token_id", None)
            if pad_token_id is None and hasattr(processing_class, "tokenizer"):
                pad_token_id = getattr(processing_class.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                raise ValueError(
                    "`padding_value` is not specified in `CPOConfig`, and the tokenizer does not provide "
                    "a padding token. Please set `processing_class.pad_token` (e.g. to the eos token)."
                )
            self.padding_value = pad_token_id
            self.arguments.padding_value = pad_token_id

        if data_collator is None:
            self.input_data_collator_tfds = DataCollatorForPreferenceTFDS(
                max_prompt_length=arguments.max_prompt_length,
                max_completion_length=arguments.max_completion_length,
                pad_token_id=self.padding_value,
                label_pad_token_id=arguments.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )
            self.input_data_collator_grain = DataCollatorForPreferenceGrain(
                max_prompt_length=arguments.max_prompt_length,
                max_completion_length=arguments.max_completion_length,
                pad_token_id=self.padding_value,
                label_pad_token_id=arguments.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )
        else:
            self.input_data_collator_tfds = data_collator
            self.input_data_collator_grain = data_collator

        if arguments.disable_dropout:
            model_state.model.eval()

        self.model_state = model_state

        super().__init__(
            model_state=model_state,
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            data_collator=None,
            processing_class=processing_class,
        )

    def _get_preprocess_transform(self) -> CPOPreprocessTransform | None:
        """Get CPO preprocessing transform for ShardedDataSource."""

        if self._is_pretokenized():
            return None
        return CPOPreprocessTransform(
            tokenizer=self.processing_class,
            max_prompt_length=self.arguments.max_prompt_length,
            max_completion_length=self.arguments.max_completion_length,
            tools=getattr(self.arguments, "tools", None),
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
        """Configure JIT-compiled training and evaluation functions.

        Returns:
            Configuration containing compiled step functions and mesh.
        """
        mesh = self.model.mesh
        empty_sharding = jax.sharding.NamedSharding(spec=PartitionSpec(), mesh=mesh)
        straight_through_emulator = resolve_straight_through_emulator(
            quantization_mode=self.arguments.quantization_mode,
            quantization_block=self.arguments.quantization_block,
            tensor_straight_through=self.arguments.tensor_straight_through,
            straight_through_emulator=self.arguments.straight_through_emulator,
        )

        partial_concatenated_forward = partial(
            concatenated_forward,
            is_encoder_decoder=self.arguments.is_encoder_decoder,
            label_pad_token_id=self.arguments.label_pad_token_id,
            padding_value=self.padding_value,
            max_length=self.arguments.max_length,
            truncation_mode=self.arguments.truncation_mode,
            aux_loss_enabled=self.arguments.aux_loss_enabled,
            loss_type=self.arguments.loss_type,
        )
        self.concatenated_forward = ejit(
            partial_concatenated_forward,
            static_argnames=(
                "is_encoder_decoder",
                "label_pad_token_id",
                "padding_value",
                "max_length",
                "truncation_mode",
                "aux_loss_enabled",
                "loss_type",
            ),
        )

        self._train_shared_fn_static_args = (
            self.scheduler,
            partial_concatenated_forward,
            self.arguments.beta,
            self.arguments.label_smoothing,
            self.arguments.loss_type,
            self.arguments.cpo_alpha,
            self.arguments.simpo_gamma,
            self.arguments.alpha,
            self.arguments.loss_config,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            straight_through_emulator,
        )

        training_static_argnums = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
        sharded_training_step_function = ejit(
            training_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=training_static_argnums,
        )
        sharded_training_step_function.static_argnums_ = training_static_argnums

        self._eval_shared_fn_static_args = (
            partial_concatenated_forward,
            self.arguments.beta,
            self.arguments.label_smoothing,
            self.arguments.loss_type,
            self.arguments.cpo_alpha,
            self.arguments.simpo_gamma,
            self.arguments.alpha,
            self.arguments.step_partition_spec,
        )

        evaluation_static_argnums = (2, 3, 4, 5, 6, 7, 8, 9)
        sharded_evaluation_step_function = ejit(
            evaluation_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=evaluation_static_argnums,
        )
        sharded_evaluation_step_function.static_argnums_ = evaluation_static_argnums

        self.arguments.ensure_checkpoint_path()
        self.sharded_training_step_function = sharded_training_step_function
        self.sharded_evaluation_step_function = sharded_evaluation_step_function
        checkpoint_manager = self.arguments.get_streaming_checkpointer()

        self._extra_forward_flops_per_token = 0
        self._extra_backward_flops_per_token = 0

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
        """Create data collator for Grain data loading.

        Args:
            max_sequence_length: Maximum sequence length.
            truncation_mode: How to truncate sequences.

        Returns:
            Grain-compatible data collator.
        """
        return self.input_data_collator_grain

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """Create data collator for TFDS data loading.

        Args:
            max_sequence_length: Maximum sequence length.
            truncation_mode: How to truncate sequences.

        Returns:
            TFDS-compatible data collator.
        """
        return self.input_data_collator_tfds

    @property
    def _train_shared_fn_extra_args(self) -> tuple[tp.Any]:
        return ()

    @property
    def _eval_shared_fn_extra_args(self) -> tuple[tp.Any]:
        return ()

    def on_step_end(
        self,
        state: EasyDeLState,
        metrics: MetricsType,
        step: int,
    ) -> tuple[EasyDeLState, MetricsType]:
        """Called at the end of each training step.

        Args:
            state: Current model state.
            metrics: Step metrics.
            step: Current step number.

        Returns:
            Potentially modified state and metrics.
        """
        return state, metrics
