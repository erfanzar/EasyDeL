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
"""Configuration dataclass for the offline distillation trainer.

Defines :class:`DistillationConfig`, including the temperature, the
KL/CE mixing weight ``alpha``, and optional projection-head configs
for hidden-state and routing-logit matching.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@Registry.register("trainer-arguments", "distillation")
@dataclass
class DistillationConfig(TrainingArguments):
    """Configuration class for offline knowledge distillation training.

    Trains a student model against a frozen teacher model using a
    convex combination of:

    * a temperature-softened KL on the next-token distributions
      ``KL(softmax(student / T) || softmax(teacher / T)) * T**2``, and
    * the standard supervised cross-entropy on the ground-truth labels
      (when present).

    Optionally adds two further matching terms when the corresponding
    weights are positive:

    * **Hidden-state matching** -- MSE between selected layers of the
      student and teacher hidden states (after a learned projection
      when shapes differ).
    * **Attention matching** -- distance between selected layers'
      attention probability tensors (with optional L1 renormalization
      when one model emits unnormalized attention).

    Construct using dict-literal kwargs:

    >>> cfg = DistillationConfig(temperature=5.0, alpha=0.7,
    ...                          learning_rate=1e-4)

    The total loss is
    ``alpha * KD + (1 - alpha) * CE + hidden_w * hidden_loss
    + attn_w * attn_loss``.

    Attributes:
        trainer_prefix: Default prefix used for checkpoints/logs
            (``"Distillation"``).
        temperature: Softmax temperature applied to both student and
            teacher logits before computing the KL term. Larger values
            (3-10) reveal more of the teacher's relative confidence
            ordering. Default ``2.0``.
        alpha: Mixing weight on the distillation term. ``1.0`` is pure
            distillation; ``0.0`` is pure supervised CE. Must be in
            ``[0, 1]`` (validated in ``__post_init__``). Default ``0.9``.
        dataset_text_field: Field name read by the SFT-style
            tokenization fallback when the dataset is plain text.
            Default ``"text"``.
        assistant_only_loss: When ``True``, the supervised CE term is
            masked to assistant/completion tokens only (requires a
            chat-style tokenization that emits an assistant mask).
        completion_only_loss: Deprecated alias for
            ``assistant_only_loss``; if explicitly set, it overrides.
        hidden_state_loss_weight: Coefficient on the hidden-state
            matching term. ``None`` (or non-positive) disables that
            term.
        hidden_state_layers: Tuple of layer indices to match in the
            hidden-state term. Negative indices follow Python
            convention. ``None`` matches the last layer only.
        hidden_state_loss: Distance function for hidden-state
            matching. Currently only ``"mse"``.
        attention_loss_weight: Coefficient on the attention matching
            term. ``None`` (or non-positive) disables it.
        attention_layers: Tuple of attention-layer indices to match.
            ``None`` matches all available layers.
        attention_normalize: When ``True``, L1-normalises attention
            tensors before computing the distance (use when one of
            the models exposes unnormalized weights).
        logits_chunk_size: When set, computes the KL term in
            sequence-axis chunks of this size instead of materialising
            the full ``[B, L, V]`` student logits tensor at once.
            Trades a small amount of extra compute (LM head recomputed
            per chunk in the backward pass) for an ``O(L)`` -> ``O(chunk)``
            peak memory reduction. Recommended values 128-512 for
            large vocabularies; ``None`` disables chunking.
    """

    trainer_prefix: str | None = field(
        default="Distillation", metadata={"help": "Prefix used for trainer logs, checkpoints, and wandb runs."}
    )
    temperature: float = field(
        default=2.0,
        metadata={
            "help": "Temperature for softening probability distributions. Higher values "
            "create softer distributions, revealing more about teacher's confidence."
        },
    )
    alpha: float = field(
        default=0.9,
        metadata={
            "help": "Weight for distillation loss vs supervised loss. "
            "1.0 = pure distillation, 0.0 = pure supervised learning."
        },
    )
    dataset_text_field: str | None = field(
        default="text",
        metadata={"help": "Name of the text field used when tokenizing raw text datasets."},
    )
    assistant_only_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to compute supervised CE only on assistant/completion tokens. "
                "Requires conversational tokenization that can emit assistant masks."
            )
        },
    )
    completion_only_loss: bool | None = field(
        default=None,
        metadata={"help": ("Deprecated alias for `assistant_only_loss`. If set, it overrides `assistant_only_loss`.")},
    )
    hidden_state_loss_weight: float | None = field(
        default=None,
        metadata={
            "help": (
                "Optional coefficient for matching student and teacher hidden states. "
                "Set to `None` to disable hidden-state distillation; `0` is accepted for backward compatibility."
            )
        },
    )
    hidden_state_layers: tuple[int, ...] | None = field(
        default=None,
        metadata={
            "help": (
                "Indices of transformer layers whose hidden states should be distilled. "
                "Negative indices follow Python semantics. Defaults to the final layer when omitted."
            )
        },
    )
    hidden_state_loss: tp.Literal["mse"] = field(
        default="mse",
        metadata={"help": "Distance function used for hidden-state distillation. Currently only 'mse' is supported."},
    )
    attention_loss_weight: float | None = field(
        default=None,
        metadata={
            "help": (
                "Optional coefficient for matching attention probability tensors. "
                "Set to `None` to disable attention-head distillation; `0` is accepted for backward compatibility."
            )
        },
    )
    attention_layers: tuple[int, ...] | None = field(
        default=None,
        metadata={
            "help": (
                "Indices of attention layers whose probability matrices should be distilled. "
                "Negative indices follow Python semantics. Defaults to all available layeatrs when omitted."
            )
        },
    )
    attention_normalize: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to L1-normalize attention matrices before computing the distillation loss. "
                "Useful when working with models that emit un-normalized attention weights."
            )
        },
    )
    logits_chunk_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "When > 0, compute the KL-divergence distillation loss in chunks of this many "
                "tokens instead of materialising the full [B, L, V] logits tensor. This trades "
                "a small amount of extra compute (lm_head is recomputed per chunk during "
                "backward) for a massive memory saving — peak logit memory drops from "
                "O(B*L*V) to O(B*chunk_size*V). Recommended values: 128-512 for large vocabs. "
                "Set to `None` to disable chunking; `0` is accepted for backward compatibility."
            )
        },
    )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        """Finalize distillation-specific config invariants.

        Mirrors the legacy ``completion_only_loss`` flag onto the
        canonical ``assistant_only_loss``, normalises the optional
        layer-index lists into tuples for hashing, and forwards
        ``max_sequence_length`` / ``quantization_block`` to the base
        :class:`TrainingArguments.__post_init__`.

        Args:
            max_sequence_length: Legacy alias for ``max_length``.
            quantization_block: Legacy alias for the quantization group
                size.
        """
        if self.completion_only_loss is not None:
            self.assistant_only_loss = bool(self.completion_only_loss)
        self.completion_only_loss = bool(self.assistant_only_loss)
        if self.hidden_state_layers is not None:
            self.hidden_state_layers = tuple(int(i) for i in self.hidden_state_layers)
        if self.attention_layers is not None:
            self.attention_layers = tuple(int(i) for i in self.attention_layers)
        if self.hidden_state_loss_weight is not None:
            normalized_hidden_state_loss_weight = float(self.hidden_state_loss_weight)
            self.hidden_state_loss_weight = (
                normalized_hidden_state_loss_weight if normalized_hidden_state_loss_weight > 0.0 else None
            )
        if self.attention_loss_weight is not None:
            normalized_attention_loss_weight = float(self.attention_loss_weight)
            self.attention_loss_weight = (
                normalized_attention_loss_weight if normalized_attention_loss_weight > 0.0 else None
            )
        if self.logits_chunk_size is not None:
            normalized_logits_chunk_size = int(self.logits_chunk_size)
            self.logits_chunk_size = normalized_logits_chunk_size if normalized_logits_chunk_size > 0 else None
        if not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError("`alpha` must be within [0, 1].")
        if float(self.temperature) <= 0.0:
            raise ValueError("`temperature` must be strictly positive.")
        if hasattr(super(), "__post_init__"):
            super().__post_init__(
                max_sequence_length=max_sequence_length,
                quantization_block=quantization_block,
            )

    __hash__ = hash_fn
