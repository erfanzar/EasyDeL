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

import dataclasses
import typing as tp
import warnings

import jax
from eformer.loggings import get_logger
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.trainers.prompt_utils import maybe_apply_chat_template
from easydel.trainers.trainer_protocol import TrainerConfigureFunctionOutput
from easydel.utils.compiling_utils import ejit

from ..trainer import Trainer
from ..utils import RewardDataCollatorWithPaddingGrain, RewardDataCollatorWithPaddingTFDS
from ._fn import evaluation_step, training_step
from .reward_config import RewardConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset
logger = get_logger(__name__)


def _tokenize(batch: dict[str, list[tp.Any]], tokenizer: ProcessingClassType) -> dict[str, list[tp.Any]]:
    """Tokenize a batch from a reward modelling dataset."""
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(batch["chosen"], batch["rejected"], strict=False):
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples


class RewardTrainer(Trainer):
    """
    This trainer extends the `Trainer` and provides functionalities.
    """

    def __init__(
        self,
        arguments: RewardConfig,
        processing_class: ProcessingClassType,
        model: EasyDeLBaseModule | EasyDeLState | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        data_collator: RewardDataCollatorWithPaddingTFDS | RewardDataCollatorWithPaddingGrain | None = None,
    ):
        if getattr(processing_class, "pad_token", None) is None:
            processing_class.pad_token = processing_class.eos_token
        assert isinstance(arguments, RewardConfig), "passed argument must be a `RewardConfig`."
        self.input_data_collator_tfds = data_collator
        self.input_data_collator_grain = data_collator
        if data_collator is None:
            if processing_class is None:
                raise ValueError(
                    "A processing_class must be specified when using the default RewardDataCollatorWithPadding"
                )

            max_sequence_length = arguments.max_sequence_length
            self.input_data_collator_tfds = RewardDataCollatorWithPaddingTFDS(
                processing_class,
                max_length=arguments.max_sequence_length,
                truncation_mode=arguments.truncation_mode,
            )
            self.input_data_collator_grain = RewardDataCollatorWithPaddingTFDS(
                processing_class,
                max_length=arguments.max_sequence_length,
                truncation_mode=arguments.truncation_mode,
            )
            if arguments.remove_unused_columns:
                try:
                    arguments.remove_unused_columns = False
                except dataclasses.FrozenInstanceError:
                    arguments = dataclasses.replace(arguments, remove_unused_columns=False)
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` "
                    "in your RewardConfig we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                    stacklevel=1,
                )

                self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False

        if "input_ids_chosen" not in train_dataset.column_names:
            fn_kwargs = {"tokenizer": processing_class}
            train_dataset = train_dataset.map(maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class})
            train_dataset = train_dataset.map(
                _tokenize,
                batched=True,
                fn_kwargs=fn_kwargs,
                num_proc=arguments.dataset_num_proc,
            )
            train_dataset = train_dataset.filter(
                lambda x: len(x["input_ids_chosen"]) <= max_sequence_length
                and len(x["input_ids_rejected"]) <= max_sequence_length,
                num_proc=arguments.dataset_num_proc,
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    maybe_apply_chat_template,
                    fn_kwargs={"tokenizer": processing_class},
                )
                eval_dataset = eval_dataset.map(
                    _tokenize,
                    fn_kwargs=fn_kwargs,
                    batched=True,
                    num_proc=arguments.dataset_num_proc,
                )
                eval_dataset = eval_dataset.filter(
                    lambda x: len(x["input_ids_chosen"]) <= max_sequence_length
                    and len(x["input_ids_rejected"]) <= max_sequence_length,
                    num_proc=arguments.dataset_num_proc,
                )
        if not isinstance(model, EasyDeLState):
            model = model.to_state()
        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=model,
            data_collator=None,
        )

    def create_collect_function(self, max_sequence_length, truncation_mode="keep_end"):
        if self.org_data_collator is not None:
            return self.org_data_collator
        return super().create_collect_function(max_sequence_length, truncation_mode)

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """
        Configures and JIT-compiles the training and evaluation step functions.

        This method prepares the functions that will be used during training and evaluation.
        It sets up sharding for the model parameters and optimizer state, JIT-compiles the
        training and evaluation functions with the appropriate static arguments and sharding
        constraints, and also sets up the checkpoint manager.

        Returns:
            TrainerConfigureFunctionOutput: An object containing:
                - sharded_training_step_function: The compiled training step function.
                - sharded_evaluation_step_function: The compiled evaluation step function.
                - mesh: The device mesh used for computation.
                - checkpoint_manager: The checkpointer for saving/loading model state.
        """
        empty_sharding = jax.sharding.NamedSharding(
            spec=PartitionSpec(),
            mesh=self.model.mesh,
        )
        self._train_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            self.arguments.center_rewards_coefficient,
        )

        sharded_training_static_argnums = (2, 3, 4, 5, 6)
        sharded_training_step_function = ejit(
            training_step,
            static_argnums=sharded_training_static_argnums,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
        )

        self._eval_shared_fn_static_args = (
            self.arguments.loss_config,
            self.arguments.step_partition_spec,
            self.arguments.center_rewards_coefficient,
        )

        sharded_evaluation_static_argnums = (2, 3, 4)
        sharded_evaluation_step_function = ejit(
            evaluation_step,
            static_argnums=sharded_evaluation_static_argnums,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
        )

        sharded_training_step_function.static_argnums_ = sharded_training_static_argnums
        sharded_evaluation_step_function.static_argnums_ = sharded_evaluation_static_argnums

        mesh = self.model.mesh
        self.arguments.ensure_checkpoint_path()
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
        """
        Creates a data collection function for batching.

        For DPO training, this method simply returns the pre-configured `data_collator`.

        Args:
            max_sequence_length (int): The maximum sequence length (not used in this implementation).
            truncation_mode (tp.Literal["keep_end", "keep_start"], optional):
                The truncation mode (not used in this implementation). Defaults to "keep_end".

        Returns:
            tp.Callable: The data collator function.
        """
        return self.input_data_collator_grain

    def create_tfds_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> tp.Callable:
        """
        Creates a data collection function for batching.

        For DPO training, this method simply returns the pre-configured `data_collator`.

        Args:
            max_sequence_length (int): The maximum sequence length (not used in this implementation).
            truncation_mode (tp.Literal["keep_end", "keep_start"], optional):
                The truncation mode (not used in this implementation). Defaults to "keep_end".

        Returns:
            tp.Callable: The data collator function.
        """
        return self.input_data_collator_tfds
