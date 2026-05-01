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
"""Configuration dataclass for the Binary Classifier Optimization (BCO) trainer.

BCO -- introduced as an extension of KTO that handles unpaired desirable
/ undesirable completions -- minimises a logistic loss against a
reference policy.  This module defines :class:`BCOConfig`, the
algorithm-specific knobs that ride on top of :class:`TrainingArguments`.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from .._shared import normalize_logprob_vocab_chunk_size
from ..training_configurations import TrainingArguments


@Registry.register("trainer-arguments", "bco")
@dataclass
class BCOConfig(TrainingArguments):
    """Configuration container for Binary Classifier Optimization (BCO) training.

    BCO (Jung et al. 2024) is a KTO-style alignment objective designed
    for *unpaired* preference data: each example has a binary
    desirable/undesirable label rather than a paired
    chosen/rejected ranking. The trainer fits an implicit reward
    classifier whose decision boundary is anchored at the in-batch
    *underlying density model* (UDM) reference reward, and the loss is
    a logistic surrogate against that boundary.

    This config layers BCO-specific knobs (the reference KL temperature
    ``beta``, length budgets, optional UDM density-ratio bounds) on top
    of every field exposed by :class:`TrainingArguments`. Construct
    using dict-literal kwargs:

    >>> cfg = BCOConfig(beta=0.1, max_length=2048, learning_rate=5e-7)

    Attributes:
        trainer_prefix: Default prefix used when generating checkpoints
            or logging artifacts (``"BCO"``).
        beta: Inverse-temperature on the implicit-reward log-ratio.
            Larger values keep the updated policy closer to the
            reference; smaller values allow more aggressive shaping.
        label_pad_token_id: Token id placed in completion ``labels``
            tensors at positions that should not contribute to the
            loss (default ``-100``).
        padding_value: Explicit padding token id used by the data
            collator. ``None`` falls back to the tokenizer's pad token.
        max_length: Maximum combined ``prompt + completion`` sequence
            length. Defaults to 1024.
        max_prompt_length: Maximum prompt-only token budget. Defaults
            to 512.
        max_completion_length: Maximum completion token budget.
            Defaults to ``max_length - max_prompt_length`` when not
            set.
        logprob_vocab_chunk_size: Vocab-axis chunk size used by
            :func:`compute_token_logps_and_entropies_chunked` when
            scoring sequences. ``None`` disables chunking.
        truncation_mode: ``"keep_end"`` keeps the latest tokens when
            truncating; ``"keep_start"`` keeps the earliest.
        disable_dropout: When ``True``, dropout layers are disabled on
            both policy and reference models for deterministic logp
            computation.
        generate_during_eval: When ``True``, sample completions during
            evaluation steps for qualitative monitoring.
        is_encoder_decoder: Override the automatic
            encoder-decoder/causal-LM detection.
        precompute_ref_log_probs: When ``True``, compute reference
            logps once at dataset-prep time and stash them in the
            dataset to skip the reference forward during training.
        model_init_kwargs: Extra kwargs forwarded to the policy model
            loader.
        ref_model_init_kwargs: Extra kwargs forwarded to the reference
            model loader.
        dataset_num_proc: Worker count used when ``Dataset.map``-ing
            over the dataset during preprocessing.
        prompt_sample_size: Number of prompts sampled to train the
            UDM density-ratio classifier.
        min_density_ratio: Lower clamp on the UDM density ratio (used
            to stabilise the reweighted loss). Default ``0.5``.
        max_density_ratio: Upper clamp on the UDM density ratio.
            Default ``10.0``.
    """

    trainer_prefix: str | None = field(
        default="BCO",
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
    logprob_vocab_chunk_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Vocabulary chunk size used when computing selected-token log probabilities. "
                "Set to `None` to disable chunking."
            )
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

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        """Finalize BCO-specific config invariants.

        Resolves the legacy ``max_sequence_length`` alias, derives
        ``max_completion_length`` from ``max_length - max_prompt_length``
        when not explicitly set, normalizes
        ``logprob_vocab_chunk_size``, and then defers to the base
        :class:`TrainingArguments.__post_init__`.

        Args:
            max_sequence_length: Legacy alias for ``max_length`` (only
                used when no explicit ``max_length`` is provided).
            quantization_block: Legacy alias for the quantization group
                size; forwarded to the base ``__post_init__``.
        """
        self._handle_deprecated_max_sequence_length(max_sequence_length)
        if self.max_length is not None and self.max_prompt_length is not None:
            if self.max_completion_length is None:
                self.max_completion_length = max(self.max_length - self.max_prompt_length, 0)
        self.logprob_vocab_chunk_size = normalize_logprob_vocab_chunk_size(self.logprob_vocab_chunk_size)

        if hasattr(super(), "__post_init__"):
            super().__post_init__(
                max_sequence_length=None,
                quantization_block=quantization_block,
            )

    __hash__ = hash_fn
