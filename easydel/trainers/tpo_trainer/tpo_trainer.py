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
"""Triple preference optimization trainer and loss functions."""

from __future__ import annotations

import typing as tp
from functools import partial

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.sharding import replicated_named_sharding
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry

from ..base_trainer import TrainerConfigureFunctionOutput
from ..direct_preference_optimization_trainer import DPOTrainer
from ..model_loading import reject_string_model_id
from ..training_utils import (
    compile_trainer_auxiliary,
    compile_trainer_step,
)
from ._fn import tpo_concatenated_forward, tpo_evaluation_step, tpo_training_step
from .tpo_config import TPOConfig
from .tpo_preprocess import (
    DataCollatorForTriplePreferenceGrain,
    DataCollatorForTriplePreferenceTFDS,
    TPOPreprocessTransform,
)

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset  # pyright: ignore[reportMissingTypeStubs]

    from easydel.data.core.protocols import ShardedDataSource


@Registry.register("trainer", "tpo")
class TPOTrainer(DPOTrainer):
    """DPO-style trainer for triple preference optimization batches.

    TPO reuses DPO model/state setup but swaps preprocessing, collation, forward
    scoring, and train/eval steps for chosen/rejected/reference triples. It
    remains reference-free with respect to policy-vs-reference log-ratios.
    """

    arguments: TPOConfig

    def __init__(
        self,
        arguments: TPOConfig | None,
        model: EasyDeLBaseModule | EasyDeLState,
        reference_model: EasyDeLBaseModule | EasyDeLState | None = None,
        processing_class: ProcessingClassType = None,
        train_dataset: "Dataset | IterableDataset | ShardedDataSource | None" = None,
        eval_dataset: "Dataset | IterableDataset | ShardedDataSource | None" = None,
        data_collator: tp.Callable | None = None,
    ) -> None:
        """Initialize TPO with DPO model setup and triple-preference collators.

        Args:
            arguments: TPO config with triple-objective loss settings and DPO
                base trainer options.
            model: Initialized EasyDeL policy module or state.
            reference_model: Optional initialized reference model/state accepted
                by the DPO base constructor, though TPO itself is configured as
                reference-free.
            processing_class: Tokenizer or processor used for triple
                preference preprocessing.
            train_dataset: Raw or pretokenized triple-preference training data.
            eval_dataset: Optional raw or pretokenized triple-preference eval
                data.
            data_collator: Optional external collator passed through to DPO
                setup before TPO installs its default triple collators.

        Raises:
            ValueError: If ``arguments`` is missing.
            TypeError: If ``arguments`` is not a ``TPOConfig``.
        """
        if arguments is None:
            raise ValueError("arguments cannot be None")
        if not isinstance(arguments, TPOConfig):
            raise TypeError(f"arguments must be TPOConfig, got {type(arguments)}")
        if isinstance(model, str):
            reject_string_model_id(model, role="policy model")
        super().__init__(
            arguments=arguments,
            model=model,
            reference_model=reference_model,
            processing_class=processing_class,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        include_reference = arguments.tpo_alpha != 0.0
        self.input_data_collator_tfds = DataCollatorForTriplePreferenceTFDS(
            max_prompt_length=arguments.max_prompt_length,
            max_completion_length=arguments.max_completion_length,
            pad_token_id=self.padding_value,
            label_pad_token_id=arguments.label_pad_token_id,
            is_encoder_decoder=arguments.is_encoder_decoder,
            pad_to_multiple_of=arguments.pad_to_multiple_of,
            include_reference=include_reference,
        )
        self.input_data_collator_grain = DataCollatorForTriplePreferenceGrain(
            max_prompt_length=arguments.max_prompt_length,
            max_completion_length=arguments.max_completion_length,
            pad_token_id=self.padding_value,
            label_pad_token_id=arguments.label_pad_token_id,
            is_encoder_decoder=arguments.is_encoder_decoder,
            pad_to_multiple_of=arguments.pad_to_multiple_of,
            include_reference=include_reference,
        )

    def _get_preprocess_transform(self) -> TPOPreprocessTransform | None:
        """Return TPO preprocessing unless the source is already tokenized.

        Raw datasets receive a transform that builds prompt, chosen, rejected,
        and reference token fields. Pretokenized datasets are left untouched so
        callers can provide custom triple-preference tensors.
        """
        if self._is_pretokenized():
            return None
        return TPOPreprocessTransform(
            tokenizer=self.processing_class,
            max_prompt_length=self.arguments.max_prompt_length,
            max_completion_length=self.arguments.max_completion_length,
            tools=getattr(self.arguments, "tools", None),
            label_pad_token_id=self.arguments.label_pad_token_id,
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """Compile TPO train/eval functions and auxiliary concatenated forward.

        The auxiliary forward is compiled for the model mesh and reused by both
        train and eval steps. Static arguments capture TPO loss parameters,
        sharding specs, gradient accumulation, and MPMD scheduling choices.
        """
        mesh = self.model.mesh
        empty_sharding = replicated_named_sharding(mesh)
        partial_concatenated_forward = partial(
            tpo_concatenated_forward,
            is_encoder_decoder=self.arguments.is_encoder_decoder,
            label_pad_token_id=self.arguments.label_pad_token_id,
            padding_value=self.padding_value,
            max_length=self.arguments.max_length,
            truncation_mode=self.arguments.truncation_mode,
            aux_loss_enabled=self.arguments.aux_loss_enabled,
            loss_type=self.arguments.loss_type,
            tpo_alpha=self.arguments.tpo_alpha,
            logprob_vocab_chunk_size=self.arguments.logprob_vocab_chunk_size,
        )
        jited_concatenated_forward = compile_trainer_auxiliary(
            partial_concatenated_forward,
            mesh=mesh,
            out_shardings=(empty_sharding,),
            static_argnames=(
                "is_encoder_decoder",
                "label_pad_token_id",
                "padding_value",
                "max_length",
                "truncation_mode",
                "aux_loss_enabled",
                "loss_type",
                "tpo_alpha",
            ),
        )
        self._train_shared_fn_static_args = (
            self.scheduler,
            partial_concatenated_forward,
            self.arguments.beta,
            self.arguments.label_smoothing,
            self.arguments.loss_type,
            self.arguments.tpo_alpha,
            self.arguments.tpo_l_gamma,
            self.arguments.loss_config,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
        )
        self._eval_shared_fn_static_args = (
            partial_concatenated_forward,
            self.arguments.beta,
            self.arguments.label_smoothing,
            self.arguments.loss_type,
            self.arguments.tpo_alpha,
            self.arguments.tpo_l_gamma,
            self.arguments.step_partition_spec,
        )
        sharded_training_step_function = compile_trainer_step(
            tpo_training_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=(self.state_shardings, empty_sharding),
            donate_argnums=(0,),
            static_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
            mesh=mesh,
            schedule=self.arguments.mpmd_scheduler,
        )
        sharded_evaluation_step_function = compile_trainer_step(
            tpo_evaluation_step,
            in_shardings=(self.state_shardings, empty_sharding),
            out_shardings=empty_sharding,
            static_argnums=(2, 3, 4, 5, 6, 7, 8),
            mesh=mesh,
            schedule=self.arguments.mpmd_scheduler,
        )
        sharded_training_step_function.static_argnums_ = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        sharded_evaluation_step_function.static_argnums_ = (2, 3, 4, 5, 6, 7, 8)

        self.arguments.ensure_checkpoint_path()
        self.concatenated_forward = jited_concatenated_forward
        return TrainerConfigureFunctionOutput(
            sharded_training_step_function=sharded_training_step_function,
            sharded_evaluation_step_function=sharded_evaluation_step_function,
            mesh=mesh,
            checkpoint_manager=self.arguments.get_streaming_checkpointer(),
        )

    @property
    def _train_shared_fn_extra_args(self) -> tuple[()]:
        """Return extra runtime args passed to TPO training steps.

        TPO stores all required state in static args and the batch, so the base
        trainer should pass no additional dynamic positional arguments.
        """
        return ()

    @property
    def _eval_shared_fn_extra_args(self) -> tuple[()]:
        """Return extra runtime args passed to TPO evaluation steps.

        Evaluation mirrors training with no extra dynamic positional arguments;
        this keeps the base trainer call convention explicit.
        """
        return ()
