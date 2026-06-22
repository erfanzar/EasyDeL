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
        checkpoint_kl_loss: When ``True`` (default) and the chunked
            path is active, wrap each chunk's KL/CE body in
            ``jax.checkpoint`` so its vocab-sized logits are
            recomputed during the backward pass instead of being kept
            live. Set ``False`` for a faster backward at the cost of
            holding every chunk's logits simultaneously -- only viable
            for small effective batch / chunk_size.
        log_top1_agreement: When ``True``, additionally logs
            ``top1_agreement`` -- the masked fraction of tokens where the
            student and teacher argmax agree. A precision-robust hard
            agreement signal that complements ``kl_loss``; computed only
            on the chunked KL path. Default ``False``.
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
    teacher_model_revision: str | None = field(
        default=None,
        metadata={"help": "Reserved teacher revision metadata for externally loaded teacher modules."},
    )
    lmbda: float = field(
        default=0.0,
        metadata={"help": "TRL on-policy sampling probability. EasyDeL offline distillation requires 0.0."},
    )
    beta: float | None = field(
        default=None,
        metadata={"help": "Optional generalized Jensen-Shannon interpolation coefficient in [0, 1]."},
    )
    reverse_kl_top_1_mode: tp.Literal["sampled", "argmax"] = field(
        default="sampled",
        metadata={"help": "TRL reverse-KL support selector. Only the default is accepted in EasyDeL offline mode."},
    )
    loss_top_k: int = field(
        default=0,
        metadata={"help": "If positive, distill only the teacher top-k vocabulary support."},
    )
    loss_add_tail: bool = field(
        default=False,
        metadata={"help": "When using `loss_top_k`, add one bucket for all non-top-k teacher/student probability mass."},
    )
    max_prompt_length: int | None = field(
        default=None,
        metadata={"help": "TRL on-policy prompt length budget. EasyDeL offline distillation uses `max_length`."},
    )
    max_completion_length: int | None = field(
        default=None,
        metadata={"help": "TRL on-policy completion length budget. Not used by EasyDeL offline distillation."},
    )
    disable_dropout: bool = field(
        default=False,
        metadata={"help": "Put both student and teacher states in eval mode before training."},
    )
    num_generations: int = field(
        default=1,
        metadata={"help": "TRL on-policy generations per prompt. EasyDeL offline distillation requires 1."},
    )
    generation_batch_size: int | None = field(
        default=None,
        metadata={"help": "TRL on-policy generation batch size. Not used by EasyDeL offline distillation."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "TRL on-policy top-p sampling. Not used by EasyDeL offline distillation."},
    )
    top_k: int = field(
        default=0,
        metadata={"help": "TRL on-policy top-k sampling. Not used by EasyDeL offline distillation."},
    )
    wandb_entity: str | None = field(default=None, metadata={"help": "TRL W&B entity."})
    wandb_project: str | None = field(default=None, metadata={"help": "TRL W&B project."})
    wandb_run_group: str | None = field(default=None, metadata={"help": "TRL W&B run group."})
    log_completions: bool = field(default=False, metadata={"help": "TRL generated-completion logging toggle."})
    log_completions_steps: int = field(default=100, metadata={"help": "TRL generated-completion logging interval."})
    num_completions_to_print: int | None = field(default=None, metadata={"help": "Number of completions to print."})
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
    checkpoint_kl_loss: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to wrap the per-chunk KL/CE body of the chunked distillation loss in "
                "`jax.checkpoint` so each chunk's vocab-sized logits are recomputed during the "
                "backward pass instead of being kept live. `True` (default) keeps peak memory "
                "≈ O(B*chunk_size*V) regardless of sequence length. Set `False` to skip the "
                "recompute (faster backward) at the cost of holding every chunk's logits in "
                "memory simultaneously — only viable for small effective batch / chunk_size."
            )
        },
    )
    log_top1_agreement: bool = field(
        default=False,
        metadata={
            "help": (
                "When True, additionally log `top1_agreement`: the masked fraction of tokens where "
                "argmax(student_logits) == argmax(teacher_logits). An argmax comparison involves no "
                "near-equal subtraction, so the metric is insensitive to softmax-normalizer precision "
                "and complements `kl_loss` as a hard agreement signal (useful when the student is "
                "heavily compressed and per-token distribution metrics sit near their numeric floor). "
                "Computed only on the chunked KL path (`logits_chunk_size` set); costs two extra "
                "vocab-axis reductions per chunk over logits already in hand (fused by XLA, ~zero memory)."
            )
        },
    )
    mtp_distillation: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable Multi-Token-Prediction (MTP) knowledge distillation. When the student has "
                "an MTP head (e.g. Qwen3.5 with `mtp_num_hidden_layers > 0` and `mtp_loss_coef > 0`), "
                "adds a soft-KD term that supervises the student's MTP head (predicting token t+2) "
                "with the teacher's own next-token distribution at position t+1 — the same conditional "
                "P(x_{t+2} | x_<=t+1). The teacher needs NO MTP head; its ordinary logits are reused. "
                "The student's self-supervised MTP CE (folded into `aux_loss` via `mtp_loss_coef`) is "
                "always included on top. Incompatible with `logits_chunk_size`."
            )
        },
    )
    mtp_kd_weight: float = field(
        default=0.3,
        metadata={
            "help": (
                "Weight on the soft MTP-KD term added to the total loss when `mtp_distillation=True`. "
                "Analogous to the model's `mtp_loss_coef` (which weights the self-supervised MTP CE)."
            )
        },
    )
    mtp_draft_tokens: int = field(
        default=1,
        metadata={
            "help": (
                "Number of tokens to draft ahead and distill through the MTP head (requires "
                "`mtp_distillation=True`). With 1 (default) the depth-1 head is distilled on its single "
                "t+2 prediction. With K>1 the head is recursively applied K times (teacher-forced, "
                "FastMTP-style) and each step k is distilled against the teacher's next-token "
                "distribution at offset k — training the head to draft K tokens ahead, matching how the "
                "inference drafter (`num_draft_tokens=K`) recursively re-applies it. Costs K large-vocab "
                "projections per step, so keep it modest (e.g. 4-8)."
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
        if self.teacher_model_revision is not None:
            raise ValueError("`teacher_model_revision` is not used by EasyDeL trainers; pass an initialized teacher.")
        if self.lmbda != 0.0:
            raise ValueError("EasyDeL `DistillationTrainer` is offline-only; set `lmbda=0.0`.")
        if self.beta is not None and not 0.0 <= float(self.beta) <= 1.0:
            raise ValueError("`beta` must be within [0, 1] when set.")
        if self.reverse_kl_top_1_mode != "sampled":
            raise ValueError("`reverse_kl_top_1_mode` is only meaningful for TRL GJSD distillation.")
        self.loss_top_k = int(self.loss_top_k)
        if self.loss_top_k < 0:
            raise ValueError("`loss_top_k` must be non-negative.")
        if self.loss_add_tail and self.loss_top_k <= 0:
            raise ValueError("`loss_add_tail=True` requires `loss_top_k > 0`.")
        if self.max_prompt_length is not None:
            raise ValueError("`max_prompt_length` is not used by EasyDeL offline distillation; use `max_length`.")
        if self.max_completion_length is not None:
            raise ValueError("`max_completion_length` is not used by EasyDeL offline distillation.")
        if self.num_generations != 1:
            raise ValueError("EasyDeL offline distillation requires `num_generations=1`.")
        if self.generation_batch_size is not None:
            raise ValueError("`generation_batch_size` is not used by EasyDeL offline distillation.")
        if self.top_p != 1.0 or self.top_k != 0:
            raise ValueError("On-policy sampling knobs are not used by EasyDeL offline distillation.")
        self._validate_logging_fields()
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
        self.log_top1_agreement = bool(self.log_top1_agreement)
        self.mtp_distillation = bool(self.mtp_distillation)
        self.mtp_kd_weight = float(self.mtp_kd_weight)
        if self.mtp_kd_weight < 0.0:
            raise ValueError("`mtp_kd_weight` must be non-negative.")
        self.mtp_draft_tokens = int(self.mtp_draft_tokens)
        if self.mtp_draft_tokens < 1:
            raise ValueError("`mtp_draft_tokens` must be >= 1.")
        if self.mtp_draft_tokens > 1 and not self.mtp_distillation:
            raise ValueError("`mtp_draft_tokens > 1` requires `mtp_distillation=True`.")
        if self.mtp_distillation and self.logits_chunk_size is not None:
            raise ValueError(
                "`mtp_distillation` is incompatible with `logits_chunk_size` (the MTP soft-KD term "
                "needs the full teacher/student logits). Set `logits_chunk_size=None`."
            )
        if self.mtp_distillation and getattr(self, "mpmd_scheduler", None) is not None:
            raise ValueError(
                "`mtp_distillation` is not yet supported on the MPMD scheduled-loss path "
                "(the stage-local loss does not include the MTP-KD term). Set `mpmd_scheduler=None`."
            )
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

    def _validate_logging_fields(self) -> None:
        """Reject TRL logging-only fields that EasyDeL distillation does not use."""
        if self.log_completions_steps <= 0:
            raise ValueError("`log_completions_steps` must be positive.")
        if self.num_completions_to_print is not None and self.num_completions_to_print <= 0:
            raise ValueError("`num_completions_to_print` must be positive when set.")
