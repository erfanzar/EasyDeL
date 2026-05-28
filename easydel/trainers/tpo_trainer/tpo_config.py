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

from dataclasses import dataclass, field

from easydel.utils import Registry

from ..direct_preference_optimization_trainer import DPOConfig


@Registry.register("trainer-arguments", "tpo")
@dataclass
class TPOConfig(DPOConfig):
    """Configuration for triple preference optimization.

    TPO consumes chosen, rejected, and reference completions in one preference
    example, then trains reference-free with one of the supported TPO/DPO-style
    scalar objectives.
    """

    trainer_prefix: str | None = field(default="TPO")
    tpo_alpha: float = field(default=1.0)
    tpo_l_gamma: float = field(default=0.5)

    def __post_init__(self, max_sequence_length: int | None, quantization_block: int | None) -> None:
        """Validate TPO scalar loss settings and force reference-free mode.

        TPO accepts exactly one scalar loss type because the triple objective
        combines chosen, rejected, and reference completions inside a dedicated
        loss function. Reference log-prob precompute is disabled after DPO
        normalization because the TPO reference branch is a dataset completion,
        not a separate reference model score.
        """
        super().__post_init__(max_sequence_length=max_sequence_length, quantization_block=quantization_block)
        if isinstance(self.loss_type, tuple | list):
            if len(self.loss_type) != 1:
                raise ValueError("TPO supports exactly one `loss_type`, not DPO-style multi-loss combinations.")
            self.loss_type = self.loss_type[0]
        if self.loss_type not in {"sigmoid", "hinge", "ipo", "tpo-l"}:
            raise ValueError("TPO `loss_type` must be one of 'sigmoid', 'hinge', 'ipo', or 'tpo-l'.")
        if self.tpo_alpha < 0.0:
            raise ValueError("`tpo_alpha` must be non-negative.")
        if self.tpo_l_gamma < 0.0:
            raise ValueError("`tpo_l_gamma` must be non-negative.")
        self.reference_free = True
        self.precompute_ref_log_probs = False
