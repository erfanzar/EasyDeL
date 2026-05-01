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
"""Configuration dataclass for the CPO trainer.

Contrastive Preference Optimization (CPO) is a reference-free
preference-learning method that combines a max-margin preference loss
with an auxiliary supervised log-likelihood term on the chosen
response, scaled by ``cpo_alpha``.  This module defines the
:class:`CPOConfig` dataclass holding the algorithm knobs.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from .._shared import normalize_logprob_vocab_chunk_size
from ..training_configurations import TrainingArguments

LOSS_TYPES = tp.Literal["sigmoid", "hinge", "ipo", "simpo", "alphapo"]


@Registry.register("trainer-arguments", "cpo")
@dataclass
class CPOConfig(TrainingArguments):
    """Configuration class for Contrastive Preference Optimization (CPO) training.

    CPO (Xu et al. 2024) is a *reference-free* preference-learning
    objective: it combines a max-margin contrastive loss between
    chosen and rejected completions with an auxiliary supervised
    log-likelihood term on the chosen completion, weighted by
    ``cpo_alpha``. Because no reference model is required, CPO sidesteps
    the reference forward and the associated KL drift accounting that
    DPO uses, at the cost of a stronger anchoring on the supervised
    objective.

    The ``loss_type`` field selects between several variants:

    * ``"sigmoid"`` -- the canonical contrastive logistic surrogate.
    * ``"hinge"`` -- max-margin hinge form.
    * ``"ipo"`` -- IPO-style squared-error contrastive surrogate.
    * ``"simpo"`` -- SimPO (Meng et al. 2024); length-normalized
      log-probs with explicit margin ``simpo_gamma``.
    * ``"alphapo"`` -- AlphaPO syntactic sugar that resolves to
      ``simpo`` with ``cpo_alpha = 0.0`` after ``__post_init__``,
      using the AlphaPO probability-power transform when
      ``alpha != 0``.

    Construct using dict-literal kwargs, e.g.:

    >>> cfg = CPOConfig(beta=0.1, loss_type="simpo", simpo_gamma=0.5)

    Attributes:
        trainer_prefix: Default prefix used for checkpoints/logs
            (``"CPO"``).
        beta: Inverse-temperature on the policy-vs-reference log-ratio.
            Larger values keep the policy closer to the reference (or,
            in the reference-free case, to the supervised target).
        label_smoothing: Optional cDPO-style smoothing applied to the
            contrastive loss. Default ``0.0``.
        loss_type: One of ``"sigmoid"``, ``"hinge"``, ``"ipo"``,
            ``"simpo"``, ``"alphapo"``. Default ``"sigmoid"``.
        disable_dropout: When ``True``, dropout is disabled on the
            policy for deterministic logp computation.
        cpo_alpha: Weight on the behaviour-cloning (supervised) term
            added to the contrastive loss. ``0.0`` recovers the pure
            contrastive objective (AlphaPO/SimPO variants).
        simpo_gamma: Target reward margin used by SimPO. Only consulted
            when ``loss_type == "simpo"``. Default ``0.5``.
        alpha: AlphaPO reward shaping parameter. ``0.0`` uses log-prob
            rewards; non-zero applies ``(1 - p**(-alpha)) / alpha``
            to the token probabilities.
        label_pad_token_id: Token id used to mask prompt tokens in the
            supervised NLL term (default ``-100``).
        padding_value: Explicit padding token id for completions.
            Falls back to the tokenizer pad token when ``None``.
        truncation_mode: ``"keep_end"`` or ``"keep_start"`` truncation
            policy.
        max_length: Maximum combined ``prompt + completion`` sequence
            length. Default ``1024``.
        max_prompt_length: Maximum prompt-only token budget. Default
            ``512``.
        max_completion_length: Maximum completion token budget;
            defaults to ``max_length - max_prompt_length`` when not
            explicitly set.
        logprob_vocab_chunk_size: Vocab-axis chunk size for
            :func:`compute_token_logps_and_entropies_chunked`. ``None``
            disables chunking.
        is_encoder_decoder: Override automatic encoder-decoder
            detection.
        dataset_num_proc: Worker count for ``Dataset.map`` calls
            during preprocessing.
    """

    trainer_prefix: str | None = field(
        default="CPO",
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
    logprob_vocab_chunk_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Vocabulary chunk size used when computing selected-token log probabilities. "
                "Set to `None` to disable chunking."
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

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        """Finalize CPO-specific config invariants.

        Resolves the legacy ``max_sequence_length`` alias, derives
        ``max_completion_length`` from
        ``max_length - max_prompt_length`` when omitted, applies the
        ``"alphapo"`` syntactic-sugar (which becomes SimPO with
        ``cpo_alpha=0.0``), normalizes ``logprob_vocab_chunk_size``,
        and finally defers to the base ``__post_init__``.

        Args:
            max_sequence_length: Legacy alias for ``max_length``.
            quantization_block: Legacy alias for the quantization group
                size; forwarded to the base class.
        """
        self._handle_deprecated_max_sequence_length(max_sequence_length)
        if self.max_length is not None and self.max_prompt_length is not None:
            if self.max_completion_length is None:
                self.max_completion_length = max(self.max_length - self.max_prompt_length, 0)

        # AlphaPO syntactic sugar: switch to SimPO loss with zero BC regularisation.
        if self.loss_type == "alphapo":
            self.loss_type = "simpo"
            self.cpo_alpha = 0.0
        self.logprob_vocab_chunk_size = normalize_logprob_vocab_chunk_size(self.logprob_vocab_chunk_size)

        if hasattr(super(), "__post_init__"):
            super().__post_init__(
                max_sequence_length=None,
                quantization_block=quantization_block,
            )

    __hash__ = hash_fn
