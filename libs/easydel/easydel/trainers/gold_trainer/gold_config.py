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

from dataclasses import dataclass, field

from easydel.utils import Registry

from ..distillation_trainer import DistillationConfig


@Registry.register("trainer-arguments", "gold")
@dataclass
class GOLDConfig(DistillationConfig):
    """Configuration for EasyDeL's GOLD-compatible offline distillation path.

    EasyDeL's supported GOLD path is explicit offline teacher-logit
    distillation through :class:`DistillationConfig`. GOLD's ULD,
    cross-tokenizer, and sequence-KD variants need different batch schemas and
    loss functions, so those switches are rejected at config construction
    instead of being accepted and ignored at runtime.
    """

    trainer_prefix: str | None = field(default="GOLD")
    use_uld_loss: bool = field(default=False)
    use_extended_uld: bool = field(default=False)
    teacher_tokenizer_name_or_path: str | None = field(default=None)
    seq_kd: bool = field(default=False)
    uld_use_hybrid_loss: bool = field(default=False)
    uld_hybrid_matched_weight: float | None = field(default=None)
    uld_hybrid_unmatched_weight: float | None = field(default=None)
    uld_crossentropy_weight: float = field(default=0.0)
    uld_distillation_weight: float = field(default=1.0)
    uld_student_temperature: float = field(default=1.0)
    uld_teacher_temperature: float = field(default=1.0)
    uld_skip_student_eos: bool = field(default=True)
    uld_skip_teacher_eos: bool = field(default=True)

    def __post_init__(self, max_sequence_length: int | None, quantization_block: int | None) -> None:
        """Validate GOLD/ULD compatibility fields after base distillation setup.

        The inherited distillation config normalizes common KD settings such as
        temperature, alpha, dropout handling, and optional top-k loss. This hook
        validates GOLD-specific weighting knobs so ULD and hybrid ULD modes do
        not enter runtime with partial or invalid coefficients. Matched and
        unmatched hybrid weights must be supplied together, all loss weights must
        be non-negative, and student/teacher ULD temperatures must be positive.
        """
        super().__post_init__(max_sequence_length=max_sequence_length, quantization_block=quantization_block)
        if (self.uld_hybrid_matched_weight is None) != (self.uld_hybrid_unmatched_weight is None):
            raise ValueError("`uld_hybrid_matched_weight` and `uld_hybrid_unmatched_weight` must be set together.")
        if self.uld_hybrid_matched_weight is not None and self.uld_hybrid_matched_weight < 0.0:
            raise ValueError("`uld_hybrid_matched_weight` must be non-negative.")
        if self.uld_hybrid_unmatched_weight is not None and self.uld_hybrid_unmatched_weight < 0.0:
            raise ValueError("`uld_hybrid_unmatched_weight` must be non-negative.")
        if self.uld_crossentropy_weight < 0.0:
            raise ValueError("`uld_crossentropy_weight` must be non-negative.")
        if self.uld_distillation_weight < 0.0:
            raise ValueError("`uld_distillation_weight` must be non-negative.")
        if self.uld_student_temperature <= 0.0:
            raise ValueError("`uld_student_temperature` must be positive.")
        if self.uld_teacher_temperature <= 0.0:
            raise ValueError("`uld_teacher_temperature` must be positive.")
        unsupported: list[str] = []
        if self.use_uld_loss:
            unsupported.append("use_uld_loss")
        if self.use_extended_uld:
            unsupported.append("use_extended_uld")
        if self.teacher_tokenizer_name_or_path is not None:
            unsupported.append("teacher_tokenizer_name_or_path")
        if self.seq_kd:
            unsupported.append("seq_kd")
        if self.uld_use_hybrid_loss:
            unsupported.append("uld_use_hybrid_loss")
        if self.uld_hybrid_matched_weight is not None:
            unsupported.append("uld_hybrid_matched_weight")
        if self.uld_hybrid_unmatched_weight is not None:
            unsupported.append("uld_hybrid_unmatched_weight")
        if self.uld_crossentropy_weight != 0.0:
            unsupported.append("uld_crossentropy_weight")
        if self.uld_distillation_weight != 1.0:
            unsupported.append("uld_distillation_weight")
        if self.uld_student_temperature != 1.0:
            unsupported.append("uld_student_temperature")
        if self.uld_teacher_temperature != 1.0:
            unsupported.append("uld_teacher_temperature")
        if not self.uld_skip_student_eos:
            unsupported.append("uld_skip_student_eos")
        if not self.uld_skip_teacher_eos:
            unsupported.append("uld_skip_teacher_eos")
        if unsupported:
            fields = ", ".join(unsupported)
            raise ValueError(
                "EasyDeL GOLD currently supports the offline teacher-logit distillation path only; "
                f"the following GOLD-specific modes are not runtime-backed: {fields}."
            )
