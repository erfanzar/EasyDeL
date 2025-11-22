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
from ..prompt_utils import maybe_apply_chat_template, maybe_extract_prompt
from ..trainer.trainer import Trainer
from ..training_configurations import MetricsType
from ..utils import (
    DataCollatorForPreferenceGrain,
    DataCollatorForPreferenceTFDS,
)
from ._fn import concatenated_forward, evaluation_step, training_step
from .cpo_config import CPOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import BaseImageProcessor, FeatureExtractionMixin, PreTrainedTokenizerBase, ProcessorMixin

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
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | IterableDataset | None = None,
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

        prepared_train = (
            self._prepare_dataset(
                train_dataset,
                processing_class,
                arguments,
                dataset_name="train",
            )
            if train_dataset is not None
            else None
        )

        prepared_eval: Dataset | IterableDataset | dict[str, Dataset] | None
        if eval_dataset is None:
            prepared_eval = None
        elif isinstance(eval_dataset, dict):
            prepared_eval = {
                key: self._prepare_dataset(dataset, processing_class, arguments, dataset_name=key)
                for key, dataset in eval_dataset.items()
            }
        else:
            prepared_eval = self._prepare_dataset(
                eval_dataset,
                processing_class,
                arguments,
                dataset_name="eval",
            )

        self.train_dataset = prepared_train
        self.eval_dataset = prepared_eval
        self.model_state = model_state

        super().__init__(
            model_state=model_state,
            arguments=arguments,
            dataset_train=prepared_train,
            dataset_eval=prepared_eval,
            data_collator=None,
            processing_class=processing_class,
        )

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin,
        arguments: CPOConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        """Prepare dataset by extracting prompts, applying templates, and tokenizing.

        Args:
            dataset: Raw dataset to process.
            processing_class: Tokenizer or processor.
            arguments: Training configuration.
            dataset_name: Name for logging.

        Returns:
            Processed dataset with tokenized fields.
        """
        map_kwargs: dict[str, tp.Any] = {}
        from datasets import Dataset

        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = arguments.dataset_num_proc

        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"Extracting prompt for {dataset_name}"
        dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"Applying chat template to {dataset_name}"
        dataset = dataset.map(
            maybe_apply_chat_template,
            fn_kwargs={"tokenizer": processing_class, "tools": getattr(arguments, "tools", None)},
            **map_kwargs,
        )

        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = f"Tokenising {dataset_name}"
        dataset = dataset.map(
            self.tokenize_row,
            remove_columns=["prompt", "chosen", "rejected"],
            fn_kwargs={
                "processing_class": processing_class,
                "max_prompt_length": arguments.max_prompt_length,
                "max_completion_length": arguments.max_completion_length,
            },
            **map_kwargs,
        )
        return dataset

    @staticmethod
    def tokenize_row(
        features: dict[str, str],
        processing_class,
        max_prompt_length: int | None,
        max_completion_length: int | None,
    ) -> dict[str, list[int]]:
        """Tokenize a single row with prompt, chosen, and rejected completions.

        Args:
            features: Dictionary with 'prompt', 'chosen', and 'rejected' fields.
            processing_class: Tokenizer.
            max_prompt_length: Maximum prompt length.
            max_completion_length: Maximum completion length.

        Returns:
            Dictionary with tokenized fields.
        """
        tokenizer = processing_class
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

        if tokenizer.bos_token_id is not None:
            prompt_input_ids = [tokenizer.bos_token_id, *prompt_input_ids]
        if tokenizer.eos_token_id is not None:
            prompt_input_ids = [*prompt_input_ids, tokenizer.eos_token_id]

        chosen_input_ids = [*chosen_input_ids, tokenizer.eos_token_id]
        rejected_input_ids = [*rejected_input_ids, tokenizer.eos_token_id]

        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Configure JIT-compiled training and evaluation functions.

        Returns:
            Configuration containing compiled step functions and mesh.
        """
        mesh = self.model.mesh
        empty_sharding = jax.sharding.NamedSharding(spec=PartitionSpec(), mesh=mesh)

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
        )

        training_static_argnums = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
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
