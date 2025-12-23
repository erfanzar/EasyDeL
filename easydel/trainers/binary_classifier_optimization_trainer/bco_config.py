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


@Registry.register("trainer-arguments", "bco")
@dataclass
class BCOConfig(TrainingArguments):
    """Configuration container for Binary Classifier Optimisation (BCO) training."""

    trainer_prefix: str | None = field(
        default="bcotrainer",
        metadata={"help": "Default prefix used when generating checkpoints or logging artifacts."},
    )
    beta: float = field(
        default=0.1,
        metadata={
            "help": "Controls deviation from reference model. Larger values keep the updated policy closer to the reference."
        },
    )
    label_pad_token_id: int = field(
        default=-100,
        metadata={"help": "Pad token id used for completion labels."},
    )
    padding_value: int | None = field(
        default=None,
        metadata={"help": "Explicit padding token id. Falls back to tokenizer pad token when not provided."},
    )
    max_length: int | None = field(
        default=1024,
        metadata={"help": "Maximum combined sequence length (prompt + completion)."},
    )
    max_prompt_length: int | None = field(
        default=512,
        metadata={"help": "Maximum number of prompt tokens retained after preprocessing."},
    )
    max_completion_length: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of completion tokens. Defaults to `max_length - max_prompt_length` if omitted."
        },
    )
    truncation_mode: tp.Literal["keep_end", "keep_start"] = field(
        default="keep_end",
        metadata={"help": "How to truncate sequences that exceed `max_length`.", "choices": ["keep_end", "keep_start"]},
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Disable dropout layers for both policy and reference models."},
    )
    generate_during_eval: bool = field(
        default=False,
        metadata={"help": "Generate and log completions during evaluation if True."},
    )
    is_encoder_decoder: bool | None = field(
        default=None,
        metadata={"help": "Override automatic detection for encoder-decoder architectures."},
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={
            "help": "Whether to precompute reference log probabilities and store them inside the dataset to save compute."
        },
    )
    model_init_kwargs: dict[str, tp.Any] | None = field(
        default=None,
        metadata={"help": "Keyword arguments passed to `AutoModelForCausalLM.from_pretrained` when loading the policy."},
    )
    ref_model_init_kwargs: dict[str, tp.Any] | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments passed to `AutoModelForCausalLM.from_pretrained` when loading the reference model."
        },
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes used when mapping over datasets."},
    )
    prompt_sample_size: int = field(
        default=1024,
        metadata={"help": "Number of prompts sampled for training the density ratio classifier (UDM)."},
    )
    min_density_ratio: float = field(
        default=0.5,
        metadata={"help": "Minimum clamp applied to the estimated density ratio when UDM is enabled."},
    )
    max_density_ratio: float = field(
        default=10.0,
        metadata={"help": "Maximum clamp applied to the estimated density ratio when UDM is enabled."},
    )

    def __post_init__(self, max_sequence_length: int | None):
        self._handle_deprecated_max_sequence_length(max_sequence_length)
        if self.max_length is not None and self.max_prompt_length is not None:
            if self.max_completion_length is None:
                self.max_completion_length = max(self.max_length - self.max_prompt_length, 0)

        if hasattr(super(), "__post_init__"):
            super().__post_init__(max_sequence_length=None)

    __hash__ = hash_fn
