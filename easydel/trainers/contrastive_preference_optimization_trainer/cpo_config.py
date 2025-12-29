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
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments

LOSS_TYPES = tp.Literal["sigmoid", "hinge", "ipo", "simpo", "alphapo"]


@Registry.register("trainer-arguments", "cpo")
@dataclass
class CPOConfig(TrainingArguments):
    """Configuration class for Contrastive Preference Optimization (CPO) training.

    This dataclass extends :class:`TrainingArguments` with the knobs required to reproduce
    the behaviour of the TRL CPO trainer while keeping the EasyDeL runtime defaults.

    Key parameters:
        beta: Temperature controlling how far the updated policy may drift.
        loss_type: Choice of CPO loss formulation (sigmoid, hinge, ipo, simpo, alphapo).
        cpo_alpha: Weight for the behaviour cloning regulariser.
        simpo_gamma: Margin used by SimPO when ``loss_type == "simpo"``.
        alpha: AlphaPO reward shaping parameter (alpha == 0 disables the transform).
        label_pad_token_id: Token id ignored by the NLL term.
        padding_value: Explicit padding token id for collators (defaults to tokenizer pad).
        max_length / max_prompt_length / max_completion_length: Sequence length controls.
    """

    trainer_prefix: str | None = field(
        default="cpotrainer",
        metadata={"help": "Default prefix used when generating checkpoint names or logs."},
    )
    beta: float = field(
        default=0.1,
        metadata={
            "help": (
                "Temperature parameter controlling the deviation from the original policy. "
                "Higher values keep the new policy closer to the old one."
            )
        },
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "Optional label smoothing applied to the CPO loss."},
    )
    loss_type: LOSS_TYPES = field(
        default="sigmoid",
        metadata={
            "help": "Variant of the CPO loss to optimise.",
            "choices": ["sigmoid", "hinge", "ipo", "simpo", "alphapo"],
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Disable dropout layers during training for more stable updates."},
    )
    cpo_alpha: float = field(
        default=1.0,
        metadata={
            "help": (
                "Weight of the behaviour cloning (supervised) term. "
                "Setting this to zero recovers the reference-free AlphaPO variant."
            )
        },
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "Target reward margin used by SimPO when `loss_type == 'simpo'`."},
    )
    alpha: float = field(
        default=0.0,
        metadata={
            "help": (
                "AlphaPO reward shaping parameter. Alpha == 0 uses log-prob rewards; "
                "non-zero alpha applies (1 - p^(-alpha)) / alpha to the token probabilities."
            )
        },
    )
    label_pad_token_id: int = field(
        default=-100,
        metadata={"help": "Token id used to mask prompt tokens in the NLL objective."},
    )
    padding_value: int | None = field(
        default=None,
        metadata={"help": "Explicit padding value for completions. Falls back to tokenizer pad token when `None`."},
    )
    truncation_mode: tp.Literal["keep_end", "keep_start"] = field(
        default="keep_end",
        metadata={
            "help": "How to truncate sequences that exceed `max_length`.",
            "choices": ["keep_end", "keep_start"],
        },
    )
    max_length: int | None = field(
        default=1024,
        metadata={"help": "Maximum combined length (prompt + completion) considered during training."},
    )
    max_prompt_length: int | None = field(
        default=512,
        metadata={"help": "Maximum number of prompt tokens kept after preprocessing."},
    )
    max_completion_length: int | None = field(
        default=None,
        metadata={
            "help": (
                "Maximum number of completion tokens. Defaults to `max_length - max_prompt_length` when not specified."
            )
        },
    )
    is_encoder_decoder: bool | None = field(
        default=None,
        metadata={"help": "Override automatic detection for encoder-decoder architectures."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes used when tokenising datasets (datasets.map `num_proc`)."},
    )

    def __post_init__(self, max_sequence_length: int | None):
        self._handle_deprecated_max_sequence_length(max_sequence_length)
        if self.max_length is not None and self.max_prompt_length is not None:
            if self.max_completion_length is None:
                self.max_completion_length = max(self.max_length - self.max_prompt_length, 0)

        # AlphaPO syntactic sugar: switch to SimPO loss with zero BC regularisation.
        if self.loss_type == "alphapo":
            self.loss_type = "simpo"
            self.cpo_alpha = 0.0

        if hasattr(super(), "__post_init__"):
            super().__post_init__(max_sequence_length=None)

    __hash__ = hash_fn
