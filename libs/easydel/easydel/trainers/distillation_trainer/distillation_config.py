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
        cakld_gamma: Optional Confidence-Aware KL coefficient. When set,
            enables ``gamma * KL(student || teacher) + (1 - gamma) *
            KL(teacher || student)``. This should be the precomputed
            averaged teacher-token confidence from the CAKLD paper.
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
    teacher_matched_compilation: bool = field(
        default=False,
        metadata={
            "help": "Run the teacher forward as a jax.vjp primal so it compiles with the same "
            "structure as the student's differentiated forward, removing the bf16 grad-vs-nograd "
            "KL floor (~0.5 nats at 60 layers; identical weights then match bitwise). Costly at "
            "long context: custom-VJP kernels take their train cores, whose fused Pallas kernels "
            "always materialize backward residuals (not DCE-able), which can spill HBM and slow "
            "steps by an order of magnitude at 131k tokens. Enable only when the exact KL "
            "reading is worth it or at scales where the residuals fit."
        },
    )
    lmbda: float = field(
        default=0.0,
        metadata={"help": "TRL on-policy sampling probability. EasyDeL offline distillation requires 0.0."},
    )
    beta: float | None = field(
        default=None,
        metadata={"help": "Optional generalized Jensen-Shannon interpolation coefficient in [0, 1]."},
    )
    cakld_gamma: float | None = field(
        default=None,
        metadata={
            "help": (
                "Optional Confidence-Aware KL gamma in [0, 1]. When set, the soft distillation term is "
                "gamma * KL(student || teacher) + (1 - gamma) * KL(teacher || student). This value is "
                "normally precomputed as the averaged teacher-token probability on a calibration subset."
            )
        },
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
    teacher_from_cache: bool = field(
        default=False,
        metadata={
            "help": (
                "Offline teacher: the step runs NO teacher forward. Each batch is joined (by row content) with "
                "records of an offline teacher cache (`easydel.data.teacher_cache`), adding `teacher_topk_idx` "
                "[B,S,k] / `teacher_topk_logp` [B,S,k] / `teacher_tail_logp` [B,S], and the chunked KL becomes the "
                "cached top-k + tail KL. Requires `teacher_cache_path`, `logits_chunk_size > 0`, `temperature == "
                "1.0` (the cache stores T=1 log-probs), and no objective that needs the teacher's full "
                "distribution or hidden states (beta / cakld_gamma / loss_top_k / MTP / hidden / attention). "
                "`teacher_model=None` is accepted only in this mode. A batch without a cache record raises."
            )
        },
    )
    teacher_cache_topk: int = field(
        default=0,
        metadata={
            "help": (
                "With `teacher_from_cache`: use only the first k cached ids (the tail is re-derived from the kept "
                "log-probs). 0 = the cache's own k and stored tail."
            )
        },
    )
    teacher_cache_add_tail: bool = field(
        default=True,
        metadata={
            "help": (
                "With `teacher_from_cache`: True trains the tail-bucket form (top-k ids get the full-KL gradient, "
                "the remaining mass is one bucket); False renormalises both distributions over the k support."
            )
        },
    )
    teacher_cache_path: str | None = field(
        default=None,
        metadata={"help": "With `teacher_from_cache`: the cache root (local or gs://) holding MANIFEST.json."},
    )
    teacher_cache_teacher_uri: str | None = field(
        default=None,
        metadata={"help": "With `teacher_from_cache`: required manifest `teacher_uri` (None = not checked)."},
    )
    teacher_cache_tokenizer: str | None = field(
        default=None,
        metadata={"help": "With `teacher_from_cache`: required manifest `tokenizer` (None = not checked)."},
    )
    teacher_cache_segment_align: int = field(
        default=1,
        metadata={
            "help": (
                "With `teacher_from_cache`: the document-start alignment the cache was dumped with (the manifest's "
                "`segment_align` must equal it); also the alignment used when the trainer derives the layout "
                "sidecar of a packed batch from its `segment_ids`."
            )
        },
    )
    loss_group_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": (
                "Per-group loss weights: optimise sum_g w_g * mean_g (each group's supervised-token mean KL) instead "
                "of the pooled token mean. Each supervised token of group g is scaled by w_g * N / N_g on the step "
                "batch; groups absent from a step drop out and the rest renormalise. Groups come from the per-token "
                "`dataset_ids` (else per-row `_dataset_id`) through `loss_group_dataset_map`. Normalised to sum 1."
            )
        },
    )
    loss_group_dataset_map: list[int] | None = field(
        default=None,
        metadata={
            "help": (
                "With `loss_group_weights`: entry i is the loss group of dataset id i (-1 = no group); None maps "
                "dataset i to group i. A supervised token without a group is refused."
            )
        },
    )
    loss_group_max_multiplier: float | None = field(
        default=None,
        metadata={
            "help": "With `loss_group_weights`: cap on a group's per-token multiplier w_g * N / N_g (None = no cap)."
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
        if self.cakld_gamma is not None:
            self.cakld_gamma = float(self.cakld_gamma)
            if not 0.0 <= self.cakld_gamma <= 1.0:
                raise ValueError("`cakld_gamma` must be within [0, 1] when set.")
            if self.beta is not None:
                raise ValueError("`cakld_gamma` is mutually exclusive with GJSD `beta`.")
        if self.reverse_kl_top_1_mode != "sampled":
            raise ValueError("`reverse_kl_top_1_mode` is only meaningful for TRL GJSD distillation.")
        self.loss_top_k = int(self.loss_top_k)
        if self.loss_top_k < 0:
            raise ValueError("`loss_top_k` must be non-negative.")
        if self.loss_add_tail and self.loss_top_k <= 0:
            raise ValueError("`loss_add_tail=True` requires `loss_top_k > 0`.")
        if self.cakld_gamma is not None and self.loss_top_k > 0:
            raise ValueError("`cakld_gamma` is not supported with `loss_top_k` / `loss_add_tail`.")
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
        self._validate_teacher_cache_fields()
        self._validate_loss_group_fields()
        if hasattr(super(), "__post_init__"):
            super().__post_init__(
                max_sequence_length=max_sequence_length,
                quantization_block=quantization_block,
            )

    __hash__ = hash_fn

    def _validate_teacher_cache_fields(self) -> None:
        """Validate the offline-teacher-cache fields.

        Every refusal names the field: the cache holds the teacher's T=1 top-k + tail and nothing else, so any
        objective reading the full teacher distribution or hidden states is refused here.

        Raises:
            ValueError: On invalid values or an objective the cache cannot serve.
        """
        self.teacher_from_cache = bool(self.teacher_from_cache)
        self.teacher_cache_topk = int(self.teacher_cache_topk)
        self.teacher_cache_add_tail = bool(self.teacher_cache_add_tail)
        self.teacher_cache_segment_align = int(self.teacher_cache_segment_align)
        if self.teacher_cache_topk < 0:
            raise ValueError("`teacher_cache_topk` must be non-negative.")
        if self.teacher_cache_segment_align < 1:
            raise ValueError("`teacher_cache_segment_align` must be >= 1.")
        if not self.teacher_from_cache:
            return
        problems = []
        if not self.teacher_cache_path:
            problems.append("teacher_cache_path must name the cache root")
        # Checked on the raw value: a non-positive chunk size means "no chunking", which the cached KL cannot use
        # (an earlier version only rejected None and let 0 / negatives through to a dense path with no teacher).
        if self.logits_chunk_size is None or int(self.logits_chunk_size) <= 0:
            problems.append("logits_chunk_size must be > 0 (the cached KL lives in the chunked scan)")
        if float(self.temperature) != 1.0:
            problems.append(f"temperature must be 1.0 (the cache stores T=1 log-probs), got {self.temperature}")
        if self.beta is not None:
            problems.append("beta (GJSD) needs the teacher's full distribution")
        if self.cakld_gamma is not None:
            problems.append("cakld_gamma needs the teacher's full distribution")
        if self.loss_top_k > 0 or self.loss_add_tail:
            problems.append("loss_top_k / loss_add_tail select the dense path; use teacher_cache_topk / _add_tail")
        if self.mtp_distillation:
            problems.append("mtp_distillation needs the teacher's full logits")
        if self.hidden_state_loss_weight is not None or self.attention_loss_weight is not None:
            problems.append("hidden_state_loss_weight / attention_loss_weight need a live teacher")
        if getattr(self, "mpmd_scheduler", None) is not None:
            problems.append("mpmd_scheduler is not wired for the cached teacher")
        if problems:
            raise ValueError("`teacher_from_cache=True` refused: " + "; ".join(problems) + ".")

    def _validate_loss_group_fields(self) -> None:
        """Validate and normalise the loss-group weighting fields.

        Raises:
            ValueError: On invalid weights / map / cap, or a combination the weighting cannot serve.
        """
        if self.loss_group_weights is None:
            if self.loss_group_dataset_map is not None or self.loss_group_max_multiplier is not None:
                raise ValueError("`loss_group_dataset_map` / `loss_group_max_multiplier` need `loss_group_weights`.")
            return
        weights = [float(w) for w in self.loss_group_weights]
        if not weights or any(not w >= 0.0 for w in weights) or sum(weights) <= 0.0:
            raise ValueError(f"`loss_group_weights` must be non-negative with a positive sum, got {weights}.")
        self.loss_group_weights = [w / sum(weights) for w in weights]
        if self.loss_group_dataset_map is not None:
            self.loss_group_dataset_map = [int(g) for g in self.loss_group_dataset_map]
            bad = [g for g in self.loss_group_dataset_map if not -1 <= g < len(weights)]
            if bad:
                raise ValueError(
                    f"`loss_group_dataset_map` entries must be -1 or a group index < {len(weights)}, got {bad}."
                )
        if self.loss_group_max_multiplier is not None:
            self.loss_group_max_multiplier = float(self.loss_group_max_multiplier)
            if self.loss_group_max_multiplier <= 0.0:
                raise ValueError("`loss_group_max_multiplier` must be positive when set.")
        if getattr(self, "mpmd_scheduler", None) is not None:
            raise ValueError("`loss_group_weights` is not wired for the mpmd_scheduler (scheduled loss) path.")

    def _validate_logging_fields(self) -> None:
        """Reject TRL logging-only fields that EasyDeL distillation does not use."""
        if self.log_completions_steps <= 0:
            raise ValueError("`log_completions_steps` must be positive.")
        if self.num_completions_to_print is not None and self.num_completions_to_print <= 0:
            raise ValueError("`num_completions_to_print` must be positive when set.")
