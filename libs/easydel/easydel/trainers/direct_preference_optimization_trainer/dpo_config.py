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
"""Configuration dataclass for the Direct Preference Optimization trainer.

DPO learns from pair-wise preference data (chosen / rejected
completions) by maximising the likelihood-ratio margin between the
policy and a frozen reference model.  This module defines
:class:`DPOConfig`, which selects the loss variant
(sigmoid/IPO/Hinge/...), the temperature ``beta``, length normalisation,
and the various preprocessing/precomputation knobs.
"""

import typing as tp
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from .._shared import normalize_logprob_vocab_chunk_size
from ..training_configurations import TrainingArguments

LOSS_FN_VARIANTS = tp.Literal[
    "sigmoid",
    "hinge",
    "ipo",
    "exo_pair",
    "nca_pair",
    "robust",
    "bco_pair",
    "sppo_hard",
    "aot",
    "aot_pair",
    "aot_unpaired",
    "apo_zero",
    "apo_down",
    "discopop",
    "sft",
    "sigmoid_norm",
]

F_DIVERGENCE_VARIANTS = tp.Literal[
    "reverse_kl",
    "forward_kl",
    "js_divergence",
    "alpha_divergence",
]


@Registry.register("trainer-arguments", "dpo")
@dataclass
class DPOConfig(TrainingArguments):
    """Configuration class for Direct Preference Optimization (DPO) training.

    Inherits from TrainingArguments and adds parameters specific to DPO training
    as described in https://arxiv.org/abs/2305.18290. This configuration controls
    various aspects of the DPO training process including loss computation,
    model architecture, and dataset processing.

    Attributes:
        beta (float): Temperature parameter (β) controlling deviation from reference model.
            Higher values make training focus more on preference matching. Default: 0.1
        label_smoothing (float): Smoothing factor for labels in loss calculation.
            Helps prevent overconfidence. 0.0 means no smoothing. Default: 0.0
        loss_type (LOSS_FN_VARIANTS | Sequence[LOSS_FN_VARIANTS]): Type of contrastive loss function to use.
            Valid options: 'sigmoid', 'hinge', 'ipo', 'exo_pair', 'nca_pair', 'robust',
            'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'aot_unpaired', 'apo_zero',
            'apo_down', 'discopop', 'sft', 'sigmoid_norm'. Multiple losses are
            combined using ``loss_weights``.
            Default: 'sigmoid'
        loss_weights (tuple[float, ...] | None): Optional weights for multi-loss
            combinations. Defaults to 1.0 for each selected loss.
        ld_alpha (float | None): Length-debiased DPO tail-token weight.
            ``None`` disables length debiasing. Values must be in
            ``[0.0, 1.0]``; ``1.0`` is equivalent to standard DPO while
            ``0.0`` keeps only the chosen/rejected shared completion prefix.
        f_divergence_type (str): f-DPO divergence transform for pair scores.
        f_alpha_divergence_coef (float): Alpha coefficient used when
            ``f_divergence_type="alpha_divergence"``.
        discopop_tau (float): Temperature for the DiscoPOP modulation.
        use_weighting (bool): Whether to apply example weighting in loss calculation.
            Default: False
        label_pad_token_id (int): Token ID used for padding labels. Default: -100
        padding_value (int | None): Value used for padding sequences. If None,
            uses model's default padding token. Default: None
        pad_token (str | None): Optional pad token override applied to
            the tokenizer before resolving ``padding_value``.
        max_length (int | None): Maximum total sequence length (prompt + completion).
            Default: 512
        max_prompt_length (int | None): Maximum length for prompt sequences.
            Default: 256
        max_completion_length (int | None): Maximum length for completion sequences.
            Auto-calculated as max_length - max_prompt_length if None. Default: None
        pad_to_multiple_of (int | None): If set, preference collators pad prompt
            and completion arrays to a multiple of this value.
        padding_free (bool): Compatibility flag for TRL's flattened
            preference-batch mode. Currently disabled at trainer init,
            matching TRL's temporary fallback behavior.
        is_encoder_decoder (bool | None): Explicitly set if model is encoder-decoder.
            Auto-detected if None. Default: None
        disable_dropout (bool): Whether to disable dropout during training for
            deterministic behavior. Default: True
        generate_during_eval (bool): TRL compatibility flag for preview generation.
            When enabled, EasyDeL schedules evaluation-time generation with its
            native eSurge generation path.
        precompute_ref_log_probs (bool): Whether to precompute reference model
            log probabilities before training. Default: False
        precompute_ref_batch_size (int | None): Optional batch size used
            for reference-logp precomputation. When ``None``, the
            train/eval dataloader batch sizes are used.
        dataset_num_proc (int | None): Number of processes for dataset preprocessing.
            Default: None (sequential processing)
        reference_free (bool): Whether to use reference-free variant of DPO.
            Default: False
        force_use_ref_model (bool): Force use reference model even when reference_free=True.
            Default: False
        sync_ref_model (bool): Whether to periodically sync reference model with
            training model. Default: False
        learning_rate (float): Optimizer learning rate. Default: 1e-6
        ref_model_mixup_alpha (float): Alpha parameter for mixup between policy
            and reference models. Default: 0.9
        ref_model_sync_steps (int): Number of steps between reference model syncs.
            Default: 64
        rpo_alpha (float | None): Reserved compatibility field for Relative
            Preference Optimization. EasyDeL DPO does not currently implement
            this loss term; leave as ``None``.
        logprob_vocab_chunk_size (int | None): Vocabulary chunk size used when
            computing selected-token log probabilities for DPO. Set to ``None``
            to disable chunking and use the full vocab in one pass.
            Normalised via :func:`normalize_logprob_vocab_chunk_size` in
            ``__post_init__``. Default: ``None``.
        tools (list[dict | Callable] | None): Additional tools for training process
            (e.g. function-calling schemas threaded through the prompt
            transform). ``None`` disables tool injection. Default: ``None``.

    Example:
        >>> config = DPOConfig(
        ...   beta=0.2, loss_type="ipo", max_length=1024, learning_rate=5e-6
        ... )
    """

    trainer_prefix: str | None = field(
        default="DPO",
        metadata={"help": "default prefix name for trainer."},
    )
    beta: float = field(
        default=0.1,
        metadata={
            "help": (
                "Temperature parameter (β) controlling deviation from reference model. Higher values make training"
                " focus more on preference matching."
            )
        },
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={
            "help": (
                "Smoothing factor for labels in loss calculation. Helps prevent overconfidence. 0.0 means no smoothing."
            )
        },
    )
    loss_type: LOSS_FN_VARIANTS | tuple[LOSS_FN_VARIANTS, ...] | list[LOSS_FN_VARIANTS] = field(
        default="sigmoid",
        metadata={
            "help": (
                "Type of contrastive loss function to use. Valid options: 'sigmoid', 'hinge', 'ipo', 'exo_pair', "
                "'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'aot_unpaired', 'apo_zero', "
                "'apo_down', 'discopop', 'sft', 'sigmoid_norm'. Pass a sequence to combine multiple losses."
            )
        },
    )
    loss_weights: tuple[float, ...] | list[float] | None = field(
        default=None,
        metadata={"help": "Optional weights for multi-loss combinations. Defaults to 1.0 per loss."},
    )
    ld_alpha: float | None = field(
        default=None,
        metadata={
            "help": (
                "Length-debiased DPO tail-token weight. None disables LD-DPO; values in [0, 1] downweight tokens "
                "after the shared chosen/rejected completion length."
            )
        },
    )
    f_divergence_type: F_DIVERGENCE_VARIANTS = field(
        default="reverse_kl",
        metadata={
            "help": (
                "f-divergence transform used for DPO pair scores. One of 'reverse_kl', 'forward_kl', "
                "'js_divergence', or 'alpha_divergence'."
            )
        },
    )
    f_alpha_divergence_coef: float = field(
        default=0.5,
        metadata={"help": "Alpha coefficient used when f_divergence_type='alpha_divergence'."},
    )
    discopop_tau: float = field(
        default=0.05,
        metadata={"help": "DiscoPOP temperature controlling the log-ratio modulation."},
    )
    use_weighting: bool = field(
        default=False,
        metadata={"help": "Whether to apply example weighting in loss calculation."},
    )
    label_pad_token_id: int = field(
        default=-100,
        metadata={"help": "Token ID used for padding labels."},
    )
    padding_value: int | None = field(
        default=None,
        metadata={"help": "Value used for padding sequences. If None, uses model's default padding token."},
    )
    pad_token: str | None = field(
        default=None,
        metadata={"help": "Optional pad token override applied to the tokenizer before resolving padding ids."},
    )
    max_length: int | None = field(
        default=512,
        metadata={"help": "Maximum total sequence length (prompt + completion)."},
    )
    max_prompt_length: int | None = field(
        default=256,
        metadata={"help": "Maximum length for prompt sequences."},
    )
    max_completion_length: int | None = field(
        default=None,
        metadata={
            "help": "Maximum length for completion sequences. Auto-calculated as max_length - max_prompt_length if None."
        },
    )
    pad_to_multiple_of: int | None = field(
        default=None,
        metadata={"help": "If set, preference batches are padded to a multiple of this value."},
    )
    padding_free: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to request padding-free flattened preference batches. Currently disabled at trainer init "
                "to match TRL's temporary fallback behavior."
            )
        },
    )
    is_encoder_decoder: bool | None = field(
        default=None,
        metadata={"help": "Explicitly set if model is encoder-decoder. Auto-detected if None."},
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout during training for deterministic behavior."},
    )
    generate_during_eval: bool = field(
        default=False,
        metadata={
            "help": (
                "TRL compatibility alias for EasyDeL preview generation. When True, preview generation runs on "
                "evaluation steps using the configured EasyDeL/eSurge generation settings."
            )
        },
    )
    activation_offloading: bool = field(
        default=False,
        metadata={"help": "TRL compatibility field. EasyDeL DPO does not currently support activation offloading."},
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={"help": "Whether to precompute reference model log probabilities before training."},
    )
    precompute_ref_batch_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Batch size used when precomputing reference model log probabilities. "
                "None defaults to the split's normal train/eval batch size."
            )
        },
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes for dataset preprocessing. Default: None (sequential processing)"},
    )
    reference_free: bool = field(
        default=False,
        metadata={"help": "Whether to use reference-free variant of DPO."},
    )
    force_use_ref_model: bool = field(
        default=False,
        metadata={"help": "Force use reference model even when reference_free=True."},
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={"help": "Whether to periodically sync reference model with training model."},
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Optimizer learning rate."},
    )
    ref_model_mixup_alpha: float = field(
        default=0.9,
        metadata={"help": "Alpha parameter for mixup between policy and reference models."},
    )
    ref_model_sync_steps: int = field(
        default=64,
        metadata={"help": "Number of steps between reference model syncs."},
    )
    rpo_alpha: float | None = field(
        default=None,
        metadata={
            "help": (
                "Reserved compatibility field for Relative Preference Optimization. "
                "EasyDeL DPO does not currently implement this loss term."
            )
        },
    )
    logprob_vocab_chunk_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "Vocabulary chunk size used when computing selected-token log probabilities for DPO. "
                "Set to `None` to disable chunking."
            )
        },
    )
    tools: list[dict | tp.Callable] | None = field(
        default=None,
        metadata={"help": "Additional tools for training process."},
    )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        """Finalize DPO-specific config invariants.

        Performs the following derivations and validations:

        * Maps the deprecated ``max_sequence_length`` alias onto
          ``max_length`` via :meth:`_handle_deprecated_max_sequence_length`.
        * Defaults ``max_completion_length`` to
          ``max_length - max_prompt_length`` when omitted.
        * Normalises ``logprob_vocab_chunk_size`` (``0`` / negative / very
          small values are coerced according to
          :func:`normalize_logprob_vocab_chunk_size`).
        * Delegates to the base :class:`TrainingArguments.__post_init__`
          for any remaining shared housekeeping.

        Args:
            max_sequence_length: Deprecated alias for ``max_length``;
                forwarded to the legacy handler.
            quantization_block: Legacy alias for the quantization
                group/block size, forwarded to the base class.
        """
        self._handle_deprecated_max_sequence_length(max_sequence_length)
        if self.max_completion_length is None and self.max_length is not None:
            self.max_completion_length = self.max_length - self.max_prompt_length
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of <= 0:
            raise ValueError("`pad_to_multiple_of` must be a positive integer when set.")
        if self.precompute_ref_batch_size is not None and self.precompute_ref_batch_size <= 0:
            raise ValueError("`precompute_ref_batch_size` must be a positive integer when set.")
        if isinstance(self.loss_type, str):
            self.loss_type = (self.loss_type,)
        else:
            self.loss_type = tuple(self.loss_type)
        if self.loss_weights is not None:
            self.loss_weights = tuple(float(weight) for weight in self.loss_weights)
            if len(self.loss_weights) != len(self.loss_type):
                raise ValueError(
                    "`loss_weights` must have the same length as `loss_type` when combining DPO losses. "
                    f"Got {len(self.loss_weights)} weights for {len(self.loss_type)} loss types."
                )
        if self.ld_alpha is not None and not (0.0 <= self.ld_alpha <= 1.0):
            raise ValueError(f"`ld_alpha` must be in [0.0, 1.0] when set; got {self.ld_alpha}.")
        if not (0.0 <= self.ref_model_mixup_alpha <= 1.0):
            raise ValueError(f"`ref_model_mixup_alpha` must be in [0.0, 1.0]; got {self.ref_model_mixup_alpha}.")
        if self.ref_model_sync_steps <= 0:
            raise ValueError(f"`ref_model_sync_steps` must be positive; got {self.ref_model_sync_steps}.")
        if self.f_divergence_type not in ("reverse_kl", "forward_kl", "js_divergence", "alpha_divergence"):
            raise ValueError(
                "`f_divergence_type` must be one of 'reverse_kl', 'forward_kl', 'js_divergence', "
                f"or 'alpha_divergence'; got {self.f_divergence_type!r}."
            )
        if "exo_pair" in self.loss_type and self.label_smoothing == 0.0:
            raise ValueError(
                "Label smoothing must be greater than 0.0 when using 'exo_pair' loss. "
                "The EXO paper recommends a value of 1e-3."
            )
        if "robust" in self.loss_type and not (0.0 <= self.label_smoothing < 0.5):
            raise ValueError(
                "The `label_smoothing` parameter should lie in [0.0, 0.5) for the 'robust' loss. "
                f"You provided {self.label_smoothing}."
            )
        if self.use_weighting and any(loss in {"aot", "aot_pair", "aot_unpaired"} for loss in self.loss_type):
            raise ValueError("`use_weighting=True` is not supported with AOT losses.")
        if self.rpo_alpha is not None and self.rpo_alpha < 0.0:
            raise ValueError("`rpo_alpha` must be non-negative when set.")
        if self.generate_during_eval:
            self.generation_interval = self.generation_interval or self.evaluation_steps or 1
            self.use_esurge_generation = True
        self.logprob_vocab_chunk_size = normalize_logprob_vocab_chunk_size(self.logprob_vocab_chunk_size)
        # Call the post_init of the parent class if it exists. Important for inheritance
        if hasattr(super(), "__post_init__"):
            super().__post_init__(
                max_sequence_length=None,
                quantization_block=quantization_block,
            )

    __hash__ = hash_fn
