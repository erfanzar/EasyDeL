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

import jax
from eformer.loggings import get_logger
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.compiling_utils import ejit
from easydel.utils.traversals import deepcopy_model

from ..binary_classifier_optimization_trainer._fn import concatenated_forward
from ..prompt_transforms import KTOPreprocessTransform
from ..prompt_utils import (
    maybe_apply_chat_template,
    maybe_extract_prompt,
    maybe_unpair_preference_dataset,
)
from ..trainer.trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..training_utils import resolve_straight_through_emulator
from ..utils import BCODataCollatorGrain, BCODataCollatorTFDS
from ._fn import evaluation_step, training_step
from .kto_config import KTOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)


@Registry.register("trainer", "kto")
class KTOTrainer(Trainer):
    """Kahneman-Tversky Optimization trainer.

    Implements KTO training which uses human binary feedback (desirable/undesirable)
    to align language models. Unlike DPO which requires paired preferences, KTO
    works with unpaired binary labels.

    Args:
        arguments: KTO-specific training configuration.
        model: Policy model to train (EasyDeLBaseModule or EasyDeLState).
        reference_model: Reference model for KL penalty computation.
        processing_class: Tokenizer or processor for text encoding.
        train_dataset: Training dataset with prompt, completion, and label fields.
        eval_dataset: Optional evaluation dataset.
        data_collator: Optional custom data collator.
    """

    arguments: KTOConfig

    def __init__(
        self,
        arguments: KTOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reference_model: EasyDeLBaseModule | EasyDeLState | None = None,
        processing_class: ProcessingClassType | None = None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        data_collator: BCODataCollatorTFDS | BCODataCollatorGrain | None = None,
    ):
        if not isinstance(arguments, KTOConfig):
            raise TypeError(f"`arguments` must be a `KTOConfig`, received {type(arguments)}")
        if processing_class is None:
            raise ValueError("`processing_class` must be provided for KTO training.")
        if model is None:
            raise ValueError("`model` must be supplied to the KTO trainer.")
        if train_dataset is None:
            raise ValueError("`train_dataset` must be provided for KTOTrainer.")

        if getattr(processing_class, "pad_token_id", None) is None and hasattr(processing_class, "eos_token"):
            processing_class.pad_token = processing_class.eos_token

        if isinstance(model, EasyDeLState):
            model_state = model
        else:
            model_state = model.to_state()

        if reference_model is None:
            reference_state = deepcopy_model(model_state)
        elif isinstance(reference_model, EasyDeLState):
            reference_state = reference_model
        elif isinstance(reference_model, EasyDeLBaseModule):
            reference_state = reference_model.to_state()
        else:
            reference_state = deepcopy_model(model_state)

        self.arguments = arguments
        if self.arguments.is_encoder_decoder is not None:
            self.is_encoder_decoder = self.arguments.is_encoder_decoder
        else:
            self.is_encoder_decoder = getattr(model_state.model.config, "is_encoder_decoder", False)
            self.arguments.is_encoder_decoder = self.is_encoder_decoder

        if arguments.padding_value is not None:
            self.padding_value = arguments.padding_value
        else:
            pad_token_id = getattr(processing_class, "pad_token_id", None)
            if pad_token_id is None and hasattr(processing_class, "tokenizer"):
                pad_token_id = getattr(processing_class.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                raise ValueError("Tokenizer must expose a pad token or `padding_value` must be specified.")
            self.padding_value = pad_token_id
            self.arguments.padding_value = pad_token_id

        if arguments.max_completion_length is None and arguments.max_length is not None:
            arguments.max_completion_length = max(arguments.max_length - arguments.max_prompt_length, 1)

        if data_collator is None:
            input_data_collator_tfds = BCODataCollatorTFDS(
                max_prompt_length=arguments.max_prompt_length,
                max_completion_length=arguments.max_completion_length,
                pad_token_id=self.padding_value,
                label_pad_token_id=arguments.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )
            input_data_collator_grain = BCODataCollatorGrain(
                max_prompt_length=arguments.max_prompt_length,
                max_completion_length=arguments.max_completion_length,
                pad_token_id=self.padding_value,
                label_pad_token_id=arguments.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )
        else:
            input_data_collator_tfds = data_collator
            input_data_collator_grain = data_collator

        self.input_data_collator_tfds = input_data_collator_tfds
        self.input_data_collator_grain = input_data_collator_grain

        self.processing_class = processing_class
        self.reference_state = reference_state
        self.calculate_kl = arguments.loss_type == "kto"
        self.aux_loss_enabled = getattr(model_state.model, "output_router_logits", False)
        self.aux_loss_coef = getattr(model_state.model, "router_aux_loss_coef", 0.0)

        # Preprocess datasets: extract prompts, unpair preference data, apply chat template
        train_dataset = self._preprocess_kto_dataset(train_dataset, processing_class, arguments)
        if eval_dataset is not None:
            eval_dataset = self._preprocess_kto_dataset(eval_dataset, processing_class, arguments)

        super().__init__(
            arguments=arguments,
            model_state=model_state,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            data_collator=None,
            processing_class=processing_class,
        )

        if arguments.disable_dropout:
            self.model_state.model.eval()
            self.reference_state.model.eval()

    @staticmethod
    def _preprocess_kto_dataset(dataset, processing_class, arguments):
        """Preprocess dataset for KTO training.

        Handles raw preference datasets (chosen/rejected format) by:
        1. Extracting shared prompts from chosen/rejected conversations
        2. Unpairing preference data (1 pair → 2 examples with labels)
        3. Applying chat template to convert conversations to text

        Args:
            dataset: Raw dataset with chosen/rejected or prompt/completion/label fields.
            processing_class: Tokenizer for chat template application.
            arguments: KTO training arguments.

        Returns:
            Preprocessed dataset with prompt, completion, label fields.
        """
        if dataset is None:
            return None

        # Check if this is a dict of datasets
        if isinstance(dataset, dict):
            return {k: KTOTrainer._preprocess_kto_dataset(v, processing_class, arguments) for k, v in dataset.items()}

        # Map kwargs for dataset processing
        map_kwargs = {"writer_batch_size": 10}
        try:
            from datasets import Dataset

            if isinstance(dataset, Dataset):
                map_kwargs["num_proc"] = getattr(arguments, "dataset_num_proc", None)
        except ImportError:
            pass

        # Step 1: Extract shared prompts from chosen/rejected if needed
        dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

        # Step 2: Unpair preference data (chosen/rejected → prompt/completion/label)
        dataset = maybe_unpair_preference_dataset(dataset, num_proc=map_kwargs.get("num_proc"))

        # Step 3: Apply chat template if data is conversational
        if not getattr(arguments, "skip_apply_chat_template", False):
            dataset = dataset.map(
                maybe_apply_chat_template,
                fn_kwargs={"tokenizer": processing_class, "tools": getattr(arguments, "tools", None)},
                **map_kwargs,
            )

        return dataset

    def _get_preprocess_transform(self) -> KTOPreprocessTransform | None:
        """Get KTO preprocessing transform for ShardedDataSource."""

        if self._is_pretokenized():
            return None
        return KTOPreprocessTransform(
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

        def forward_fn(model, batch):
            return concatenated_forward(
                model,
                batch,
                is_encoder_decoder=self.arguments.is_encoder_decoder,
                label_pad_token_id=self.arguments.label_pad_token_id,
                padding_value=self.padding_value,
                max_length=self.arguments.max_length,
                truncation_mode=self.arguments.truncation_mode,
                aux_loss_enabled=self.aux_loss_enabled,
            )

        self.concatenated_forward = ejit(forward_fn, static_argnames=())

        self._train_shared_fn_static_args = (
            self.scheduler,
            forward_fn,
            self.arguments.beta,
            self.arguments.desirable_weight,
            self.arguments.undesirable_weight,
            self.arguments.loss_type,
            self.calculate_kl,
            self.aux_loss_coef,
            self.arguments.loss_config,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            straight_through_emulator,
        )

        ref_sharding = self.reference_state.shardings if self.reference_state is not None else empty_sharding

        train_static_argnums = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
        sharded_training_step_function = ejit(
            training_step,
            in_shardings=(self.state_shardings, empty_sharding, ref_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=train_static_argnums,
        )

        self._eval_shared_fn_static_args = (
            forward_fn,
            self.arguments.beta,
            self.arguments.desirable_weight,
            self.arguments.undesirable_weight,
            self.arguments.loss_type,
            self.calculate_kl,
            self.aux_loss_coef,
            self.arguments.step_partition_spec,
        )

        eval_static_argnums = (3, 4, 5, 6, 7, 8, 9, 10)
        sharded_evaluation_step_function = ejit(
            evaluation_step,
            in_shardings=(self.state_shardings, empty_sharding, ref_sharding),
            out_shardings=empty_sharding,
            static_argnums=eval_static_argnums,
        )

        self.sharded_training_step_function = sharded_training_step_function
        self.sharded_evaluation_step_function = sharded_evaluation_step_function
        self._train_shared_fn_extra_args = (self.reference_state,)
        self._eval_shared_fn_extra_args = (self.reference_state,)

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
