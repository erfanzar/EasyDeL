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
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from tqdm.autonotebook import tqdm

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry
from easydel.utils.compiling_utils import ejit
from easydel.utils.traversals import deepcopy_model

from ..base_trainer import TrainerConfigureFunctionOutput
from ..prompt_transforms import DPOPreprocessTransform
from ..trainer.trainer import Trainer
from ..training_configurations import MetricsType
from ..training_utils import resolve_straight_through_emulator
from ..utils import DataCollatorForPreferenceGrain, DataCollatorForPreferenceTFDS
from ._fn import concatenated_forward, evaluation_step, training_step
from .dpo_config import DPOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from easydel.data.core.protocols import ShardedDataSource

logger = get_logger(__name__)


@Registry.register("trainer", "dpo")
class DPOTrainer(Trainer):
    """Trainer for Direct Preference Optimization (DPO).

    This trainer implements the Direct Preference Optimization algorithm for training
    language models from human preferences without requiring a separate reward model.
    DPO directly optimizes the policy to match human preferences by maximizing the
    likelihood of preferred completions relative to rejected ones.

    The trainer uses lazy preprocessing transforms that are applied during iteration,
    providing better performance than eager HF .map() calls.

    Attributes:
        arguments (DPOConfig): Configuration object containing all training parameters.
        processing_class: Tokenizer or processor for data preprocessing.
        reference_state (EasyDeLState): Reference model state for KL divergence computation.
        padding_value (int): Token ID used for padding sequences.

    Example:
        >>> config = DPOConfig(
        ...     beta=0.1,
        ...     loss_type="sigmoid",
        ...     max_length=512,
        ...     learning_rate=5e-6
        ... )
        >>> trainer = DPOTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     reference_model=reference_model,
        ...     processing_class=tokenizer,
        ...     train_dataset=preference_dataset
        ... )
        >>> trainer.train()

    Note:
        The trainer expects datasets with 'prompt', 'chosen', and 'rejected' columns.
        These will be automatically tokenized via lazy transforms during iteration.
    """

    arguments: DPOConfig

    def __init__(
        self,
        arguments: DPOConfig,
        model: EasyDeLBaseModule | EasyDeLState,
        reference_model: EasyDeLBaseModule | EasyDeLState | None = None,
        processing_class: ProcessingClassType = None,
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        data_collator: tp.Callable | None = None,
    ):
        if arguments is None:
            raise ValueError("arguments cannot be None")
        if not isinstance(arguments, DPOConfig):
            raise TypeError(f"arguments must be DPOConfig, got {type(arguments)}")
        if processing_class is None:
            raise ValueError("processing_class must be specified to tokenize a DPO dataset.")

        self.arguments = arguments
        self.truncation_mode = arguments.truncation_mode
        self.processing_class = processing_class
        self.is_encoder_decoder = arguments.is_encoder_decoder
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

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
                    "`padding_value` is not specified in `DPOConfig`, and `pad_token_id` is missing in the "
                    "`processing_class`. Please either set the `padding_value` argument in `DPOConfig`, or set "
                    "`tokenizer.pad_token` (e.g., `tokenizer.pad_token = tokenizer.eos_token`) before instantiating "
                    "the trainer."
                )
        arguments.padding_value = self.padding_value

        # Setup data collators
        self.input_data_collator_tfds = (
            DataCollatorForPreferenceTFDS(
                max_prompt_length=arguments.max_prompt_length,
                max_completion_length=arguments.max_completion_length,
                pad_token_id=self.padding_value,
                label_pad_token_id=arguments.label_pad_token_id,
                is_encoder_decoder=arguments.is_encoder_decoder,
            )
            if data_collator is None
            else data_collator
        )
        self.input_data_collator_grain = (
            DataCollatorForPreferenceGrain(
                max_prompt_length=arguments.max_prompt_length,
                max_completion_length=arguments.max_completion_length,
                pad_token_id=self.padding_value,
                label_pad_token_id=arguments.label_pad_token_id,
                is_encoder_decoder=arguments.is_encoder_decoder,
            )
            if data_collator is None
            else data_collator
        )

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # Setup models
        if not isinstance(model, EasyDeLState):
            model = model.to_state()
        if reference_model is None:
            reference_model = deepcopy_model(model)
        if not isinstance(reference_model, EasyDeLState):
            reference_model = reference_model.to_state()

        self.reference_state = reference_model

        super().__init__(
            model_state=model,
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            data_collator=None,
            processing_class=processing_class,
        )

    def _get_preprocess_transform(self) -> DPOPreprocessTransform | None:
        """Get DPO preprocessing transform for ShardedDataSource.

        Returns a transform that handles:
        - Prompt extraction from chosen/rejected
        - Chat template application
        - Triple tokenization (prompt, chosen, rejected)

        Returns:
            DPOPreprocessTransform or None if data is already tokenized.
        """

        if self._is_pretokenized():
            return None

        return DPOPreprocessTransform(
            tokenizer=self.processing_class,
            max_prompt_length=self.arguments.max_prompt_length,
            max_completion_length=self.arguments.max_completion_length,
            tools=getattr(self.arguments, "tools", None),
            label_pad_token_id=self.arguments.label_pad_token_id,
        )

    def _is_pretokenized(self) -> bool:
        """Check if dataset already has DPO tokenized fields."""
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

        jited_concatenated_forward = ejit(
            partial_concatenated_forward,
            out_shardings=(empty_sharding,),
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
            self.arguments.reference_free,
            self.arguments.loss_config,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            straight_through_emulator,
        )

        sharded_training_static_argnums = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        sharded_training_step_function = ejit(
            training_step,
            in_shardings=(
                self.state_shardings,
                empty_sharding,
                self.reference_state.shardings,
            ),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=sharded_training_static_argnums,
        )

        self._eval_shared_fn_static_args = (
            partial_concatenated_forward,
            self.arguments.beta,
            self.arguments.label_smoothing,
            self.arguments.loss_type,
            self.arguments.reference_free,
            self.arguments.step_partition_spec,
        )

        sharded_evaluation_static_argnums = (3, 4, 5, 6, 7)
        sharded_evaluation_step_function = ejit(
            evaluation_step,
            in_shardings=(
                self.state_shardings,
                empty_sharding,
                self.reference_state.shardings,
            ),
            out_shardings=empty_sharding,
            static_argnums=sharded_evaluation_static_argnums,
        )

        sharded_training_step_function.static_argnums_ = sharded_training_static_argnums
        sharded_evaluation_step_function.static_argnums_ = sharded_evaluation_static_argnums

        self.arguments.ensure_checkpoint_path()
        self.concatenated_forward = jited_concatenated_forward
        checkpoint_manager = self.arguments.get_streaming_checkpointer()

        flops_per_tkn = self.reference_state.model.flops_per_token(include_loss=True, include_backward=True)
        self._extra_forward_flops_per_token = flops_per_tkn
        self._extra_backward_flops_per_token = flops_per_tkn

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

    def configure_dataloaders(self):
        """Configure dataloaders with optional precomputed reference log probs."""
        if self.dataset_train is not None:
            if self.arguments.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
                reference_chosen_log_probs = []
                ref_rejected_logps = []

                for padded_batch in tqdm(iterable=self.dataset_train, desc="Train dataset reference log probs"):
                    reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(
                        self.model_state,
                        padded_batch,
                    )
                    reference_chosen_log_probs.append(reference_chosen_logp)
                    ref_rejected_logps.append(reference_rejected_logp)

                all_reference_chosen_log_probs = jnp.concatenate(reference_chosen_log_probs)
                all_ref_rejected_logps = jnp.concatenate(ref_rejected_logps)

                self.dataset_train = self.dataset_train.add_column(
                    name="reference_chosen_log_probs",
                    column=all_reference_chosen_log_probs,
                )
                self.dataset_train = self.dataset_train.add_column(
                    name="ref_rejected_logps",
                    column=all_ref_rejected_logps,
                )
                self._precomputed_train_ref_log_probs = True

        if self.dataset_eval is not None:
            if self.arguments.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
                reference_chosen_log_probs = []
                ref_rejected_logps = []

                for padded_batch in tqdm(iterable=self.dataset_eval, desc="Eval dataset reference log probs"):
                    reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(
                        self.model_state,
                        padded_batch,
                    )
                    reference_chosen_log_probs.append(reference_chosen_logp)
                    ref_rejected_logps.append(reference_rejected_logp)

                all_reference_chosen_log_probs = jnp.concatenate(reference_chosen_log_probs)
                all_ref_rejected_logps = jnp.concatenate(ref_rejected_logps)

                self.dataset_eval = self.dataset_eval.add_column(
                    name="reference_chosen_log_probs", column=all_reference_chosen_log_probs
                )
                self.dataset_eval = self.dataset_eval.add_column(
                    name="ref_rejected_logps", column=all_ref_rejected_logps
                )
                self._precomputed_eval_ref_log_probs = True

        return super().configure_dataloaders()

    def compute_reference_log_probs(
        self,
        state: EasyDeLState,
        padded_batch: dict,
    ) -> tuple[tp.Any, tp.Any]:
        """Compute log probabilities of the reference model for a batch."""
        if self.reference_state is None:
            outs = self.concatenated_forward(state.model, batch=padded_batch)
        else:
            outs = self.concatenated_forward(self.reference_state.model, batch=padded_batch)
        return outs["chosen_logps"], outs["rejected_logps"]

    @property
    def _train_shared_fn_extra_args(self) -> tuple[tp.Any]:
        return (self.reference_state,)

    @property
    def _eval_shared_fn_extra_args(self) -> tuple[tp.Any]:
        return (self.reference_state,)

    def on_step_end(
        self,
        state: EasyDeLState,
        metrics: MetricsType,
        step: int,
    ) -> tuple[EasyDeLState, MetricsType]:
        """Hook called at the end of each step for reference model sync."""
        if (
            self.arguments.sync_ref_model
            and self.reference_state is not None
            and (step % self.arguments.ref_model_sync_steps == 0)
        ):
            self.reference_state = self.reference_state.replace(graphstate=deepcopy_model(state.graphstate))
        return state, metrics
