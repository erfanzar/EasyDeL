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

from eformer.loggings import get_logger
from jax.sharding import NamedSharding, PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils.compiling_utils import ejit

from ..trainer import Trainer
from ..trainer_protocol import TrainerConfigureFunctionOutput
from ..utils import DataCollatorForCompletionOnlyLM
from ._fn import distillation_step
from .distillation_config import DistillationConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset

logger = get_logger(__name__)


class DistillationTrainer(Trainer):
    teacher_state: EasyDeLState
    arguments: DistillationConfig  # type hinting

    def __init__(
        self,
        arguments: DistillationConfig,
        processing_class: ProcessingClassType,
        student_model: EasyDeLBaseModule | EasyDeLState | None = None,
        teacher_model: EasyDeLBaseModule | EasyDeLState | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        data_collator: DataCollatorForCompletionOnlyLM | None = None,
    ):
        tokenizer = processing_class
        if hasattr(processing_class, "tokenizer"):
            tokenizer = processing_class.tokenizer
        if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
            tokenizer.pad_token = tokenizer.eos_token
        assert isinstance(arguments, DistillationConfig), "passed argument must be a `DistillationConfig`."

        self.arguments = arguments

        if not isinstance(student_model, EasyDeLState):
            student_model = student_model.to_state()
        if not isinstance(teacher_model, EasyDeLState):
            teacher_model = teacher_model.to_state()

        self.teacher_state = teacher_model

        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=student_model,
            data_collator=data_collator,
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """
        Configures and JIT-compiles the training and evaluation step functions.

        This method sets up the necessary functions for training and evaluation, including:
            - Initialization of the model state.
            - Sharding of the model parameters and optimizer state.
            - JIT-compilation of the training and evaluation step functions.

        Returns:
            TrainerConfigureFunctionOutput: An object containing the configured functions and other relevant information.
        """
        mesh = self.model.mesh

        empty_sharding = NamedSharding(spec=PartitionSpec(), mesh=mesh)

        self._train_shared_fn_static_args = (
            self.arguments.loss_config,
            self.scheduler,
            self.arguments.step_partition_spec,
            self.arguments.gradient_accumulation_steps,
            True,  # is_train
            self.arguments.temperature,
            self.arguments.alpha,
        )

        static_argnames = (3, 4, 5, 6, 7, 8, 9)
        sharded_training_step_function = ejit(
            distillation_step,
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
            False,  # is_train
            self.arguments.temperature,
            self.arguments.alpha,
        )

        sharded_evaluation_step_function = ejit(
            distillation_step,
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

    @property
    def _train_shared_fn_extra_args(self) -> tuple[tp.Any]:
        return (self.teacher_state,)

    @property
    def _eval_shared_fn_extra_args(self) -> tuple[tp.Any]:
        return (self.teacher_state,)
