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
"""Configuration dataclass for the KTO trainer.

Defines :class:`KTOConfig`, including the temperature ``beta``,
desirable/undesirable loss weights, and the legacy aliases shared with
other preference trainers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from .._shared import normalize_logprob_vocab_chunk_size
from ..training_configurations import TrainingArguments


@Registry.register("trainer-arguments", "kto")
@dataclass
class KTOConfig(TrainingArguments):
    """Configuration for the :class:`~easydel.trainers.KTOTrainer`.

    KTO (Kahneman-Tversky Optimisation, Ethayarajh et al. 2024) is an
    unpaired preference-learning objective rooted in prospect theory.
    Each example carries a single binary ``label`` (desirable vs.
    undesirable) instead of a paired chosen/rejected ranking, and the
    loss shapes the implicit reward
    ``r = beta * (log pi(y|x) - log pi_ref(y|x))`` according to a
    KL-regularised value function:

    * Desirable rows are pushed *above* the in-batch reference KL
      ``z`` weighted by ``desirable_weight``.
    * Undesirable rows are pushed *below* ``z`` weighted by
      ``undesirable_weight``.

    Setting ``loss_type = "apo_zero_unpaired"`` switches to the
    APO-zero-style unpaired surrogate (an alternative shaping
    function with equivalent KL anchoring).

    Construct with dict-literal kwargs:

    >>> cfg = KTOConfig(beta=0.1, desirable_weight=1.0,
    ...                 undesirable_weight=1.0, max_length=1024)

    Attributes:
        trainer_prefix: Default prefix used for checkpoints/logs
            (``"KTO"``).
        max_length: Maximum combined ``prompt + completion`` sequence
            length. Default ``1024``.
        max_prompt_length: Maximum prompt-only token budget. Default
            ``512``.
        max_completion_length: Maximum completion token budget;
            defaults to ``max_length - max_prompt_length`` when not
            set explicitly.
        logprob_vocab_chunk_size: Vocab-axis chunk size for
            :func:`compute_token_logps_and_entropies_chunked`. ``None``
            disables chunking.
        beta: Inverse-temperature on the implicit reward log-ratio
            against the reference. Default ``0.1``.
        desirable_weight: Loss multiplier on desirable rows
            (``label=True``).
        undesirable_weight: Loss multiplier on undesirable rows
            (``label=False``).
        loss_type: ``"kto"`` (default) or ``"apo_zero_unpaired"``.
        label_pad_token_id: Sentinel token id used to mask label
            positions excluded from the loss (default ``-100``).
        padding_value: Explicit padding token id; falls back to the
            tokenizer pad token when ``None``.
        truncation_mode: ``"keep_end"`` (default) or ``"keep_start"``
            truncation policy.
        is_encoder_decoder: Override the automatic encoder-decoder
            detection.
        disable_dropout: Disable dropout layers on policy *and*
            reference for deterministic logp computation.
        dataset_num_proc: Worker count for ``Dataset.map``.
        precompute_ref_log_probs: When ``True``, the trainer caches
            reference logps during dataset preparation so the
            reference forward is skipped at training time.
    """

    trainer_prefix: str | None = field(
        default="KTO",
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
    logprob_vocab_chunk_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Vocabulary chunk size used when computing selected-token log probabilities. "
                "Set to `None` to disable chunking."
            )
        },
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

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        """Finalize KTO-specific config invariants.

        Resolves the legacy ``max_sequence_length`` alias, derives
        ``max_completion_length`` from the prompt/completion budget,
        normalises ``logprob_vocab_chunk_size``, and forwards to the
        base ``__post_init__``.

        Args:
            max_sequence_length: Legacy alias for ``max_length``.
            quantization_block: Legacy alias for the quantization group
                size.
        """
        self._handle_deprecated_max_sequence_length(max_sequence_length)
        if self.max_completion_length is None and self.max_length is not None and self.max_prompt_length is not None:
            self.max_completion_length = max(self.max_length - self.max_prompt_length, 1)
        self.logprob_vocab_chunk_size = normalize_logprob_vocab_chunk_size(self.logprob_vocab_chunk_size)
        if hasattr(super(), "__post_init__"):
            super().__post_init__(
                max_sequence_length=None,
                quantization_block=quantization_block,
            )

    __hash__ = hash_fn
