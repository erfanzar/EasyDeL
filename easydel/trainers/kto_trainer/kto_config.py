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

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@Registry.register("trainer-arguments", "kto")
@dataclass
class KTOConfig(TrainingArguments):
    """Configuration for the :class:`~easydel.trainers.KTOTrainer`."""

    trainer_prefix: str | None = field(
        default="ktotrainer",
        metadata={"help": "Default prefix used for checkpoints and experiment tracking."},
    )
    max_length: int | None = field(
        default=1024,
        metadata={"help": "Maximum total sequence length (prompt + completion)."},
    )
    max_prompt_length: int | None = field(
        default=512,
        metadata={"help": "Maximum length for prompt tokens."},
    )
    max_completion_length: int | None = field(
        default=None,
        metadata={"help": "Maximum length for completion tokens. Defaults to max_length - max_prompt_length."},
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "Scaling factor controlling deviation from the reference model."},
    )
    desirable_weight: float = field(
        default=1.0,
        metadata={"help": "Weight applied to losses from desirable (label=True) samples."},
    )
    undesirable_weight: float = field(
        default=1.0,
        metadata={"help": "Weight applied to losses from undesirable (label=False) samples."},
    )
    loss_type: str = field(
        default="kto",
        metadata={"help": "Loss variant to use.", "choices": ["kto", "apo_zero_unpaired"]},
    )
    label_pad_token_id: int = field(
        default=-100,
        metadata={"help": "Padding value used for completion labels."},
    )
    padding_value: int | None = field(
        default=None,
        metadata={"help": "Pad token ID to use. When None, tokenizer pad token is used."},
    )
    truncation_mode: str = field(
        default="keep_end",
        metadata={"help": "Prompt truncation strategy.", "choices": ["keep_end", "keep_start"]},
    )
    is_encoder_decoder: bool | None = field(
        default=None,
        metadata={"help": "Explicitly set when the model is encoder-decoder. Auto-detected when None."},
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Disable dropout in both policy and reference models during training."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for dataset preprocessing."},
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={"help": "Whether to precompute reference log probabilities into the dataset."},
    )

    def __post_init__(self, max_sequence_length: int | None):
        self._handle_deprecated_max_sequence_length(max_sequence_length)
        if self.max_completion_length is None and self.max_length is not None and self.max_prompt_length is not None:
            self.max_completion_length = max(self.max_length - self.max_prompt_length, 1)
        if hasattr(super(), "__post_init__"):
            super().__post_init__(max_sequence_length=None)

    __hash__ = hash_fn
