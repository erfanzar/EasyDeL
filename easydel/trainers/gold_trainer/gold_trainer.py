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
"""GOLD offline distillation trainer mapping."""

from __future__ import annotations

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.utils import Registry

from ..distillation_trainer import DistillationTrainer
from .gold_config import GOLDConfig


@Registry.register("trainer", "gold")
class GOLDTrainer(DistillationTrainer):
    """GOLD compatibility trainer backed by EasyDeL offline distillation.

    GOLD runs on top of EasyDeL's initialized-student/initialized-teacher
    distillation path. GOLD-specific config knobs are normalized by
    :class:`GOLDConfig` before the shared distillation step is compiled.
    """

    arguments: GOLDConfig

    def __init__(
        self,
        arguments: GOLDConfig,
        model: EasyDeLBaseModule | EasyDeLState | None = None,
        *,
        student_model: EasyDeLBaseModule | EasyDeLState | None = None,
        teacher_model: EasyDeLBaseModule | EasyDeLState | None = None,
        train_dataset: object | None = None,
        eval_dataset: object | dict[str, object] | None = None,
        processing_class: object | None = None,
        data_collator: object | None = None,
    ) -> None:
        """Initialize GOLD with EasyDeL's explicit student/teacher contract.

        Args:
            arguments: GOLD configuration. It must already be a ``GOLDConfig`` so
                GOLD-specific ULD/SeqKD compatibility fields have been validated.
            model: Convenience alias for ``student_model``. It is accepted for
                TRL-style constructor parity but cannot be supplied together with
                ``student_model``.
            student_model: Initialized EasyDeL module/state to optimize.
            teacher_model: Optional initialized teacher. When omitted, GOLD uses
                the resolved student as the teacher, matching the self-teacher
                offline distillation fallback.
            train_dataset: Training dataset forwarded to ``DistillationTrainer``.
            eval_dataset: Optional evaluation dataset or named split mapping.
            processing_class: Tokenizer/processor used by the inherited
                distillation preprocessing path.
            data_collator: Optional collator override.

        Raises:
            TypeError: If ``arguments`` is not ``GOLDConfig``.
            ValueError: If both ``model`` and ``student_model`` are provided.
        """
        if not isinstance(arguments, GOLDConfig):
            raise TypeError(f"arguments must be GOLDConfig, got {type(arguments)}")
        if model is not None and student_model is not None:
            raise ValueError("Pass either `model` or `student_model`, not both.")
        resolved_student = student_model if student_model is not None else model
        resolved_teacher = teacher_model if teacher_model is not None else resolved_student
        super().__init__(
            arguments=arguments,
            processing_class=processing_class,
            student_model=resolved_student,
            teacher_model=resolved_teacher,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
