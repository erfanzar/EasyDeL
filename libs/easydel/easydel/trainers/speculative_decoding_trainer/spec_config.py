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

"""Configuration surface for speculative-decoding drafter training.

The config describes how to call the frozen target, how to read or construct
drafter logits, and how to mix target KL with supervised CE.  It intentionally
expects models to be loaded outside the trainer so checkpoint conversion and
mesh placement stay explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments

_DEFAULT_SPEC_METRICS_TO_SHOW = [
    "loss",
    "draft_accept_rate",
    "tau_probabilistic",
    "deepspec_block_ce_loss",
    "deepspec_block_l1_loss",
]


@Registry.register("trainer-arguments", ["speculative_decoding", "spec_decoding", "spec"])
@dataclass
class SpeculativeDecodingConfig(TrainingArguments):
    """Training arguments for fitting a speculative-decoding drafter.

    The trainer keeps ``target_model`` frozen and updates ``drafter_model`` so
    the drafter's next-token or multi-token draft distributions match the
    target model.  The loss is

    ``alpha * KL(target || drafter) + (1 - alpha) * CE(drafter, labels)``.

    Normal causal-LM drafters use their ``logits`` output and train the
    next-token distribution.  Models with explicit draft heads can expose
    ``draft_logits`` / ``mtp_logits`` or a configurable chain method returning
    ``[draft_steps, batch, sequence, vocab]`` logits.  DSpark/DFlash-style
    block drafters can instead return block logits, target IDs, and eval masks;
    the trainer then uses the native DeepSpec CE/L1/confidence loss.
    """

    trainer_prefix: str | None = field(
        default="SpeculativeDecoding",
        metadata={"help": "Prefix used for trainer logs, checkpoints, and wandb runs."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature used for the soft target KL term."},
    )
    alpha: float = field(
        default=1.0,
        metadata={"help": "Weight on target-distribution KL. 1.0 is pure KL; 0.0 is pure supervised CE."},
    )
    num_draft_tokens: int = field(
        default=1,
        metadata={
            "help": (
                "Maximum number of explicit draft-token heads to consume when the drafter returns "
                "multi-step logits. Ordinary causal-LM drafters still train the next-token step only."
            )
        },
    )
    draft_target_offset: int = field(
        default=0,
        metadata={
            "help": (
                "Offset between drafter position t and target logits. Use 0 for normal next-token "
                "draft LMs; use 1 for an MTP head whose first draft logit predicts t+2."
            )
        },
    )
    target_logits_attr: str = field(
        default="logits",
        metadata={"help": "Attribute or dotted path used to read target logits from target outputs."},
    )
    draft_logits_attr: str | None = field(
        default=None,
        metadata={
            "help": (
                "Attribute or dotted path used to read drafter logits. When unset, the trainer tries "
                "`draft_logits`, then `mtp_logits`, then `logits`, then an array return value."
            )
        },
    )
    draft_chain_method: str | None = field(
        default="auto",
        metadata={
            "help": (
                "Optional method on the drafter that returns multi-step draft logits from the normal "
                "forward output. `auto` uses `compute_mtp_chain` when present and `num_draft_tokens > 1`; "
                "set to None to disable chain extraction."
            )
        },
    )
    drafter_forward_method: str | None = field(
        default=None,
        metadata={"help": "Optional drafter method name to call instead of the normal module forward."},
    )
    target_forward_method: str | None = field(
        default=None,
        metadata={"help": "Optional target method name to call instead of the normal module forward."},
    )
    target_output_hidden_states: bool = field(
        default=False,
        metadata={"help": "Request hidden states from the frozen target forward."},
    )
    pass_target_outputs: bool = field(
        default=False,
        metadata={
            "help": (
                "Pass target logits/hidden states/outputs into the drafter call when the drafter "
                "forward or custom method accepts those kwargs."
            )
        },
    )
    target_model_revision: str | None = field(
        default=None,
        metadata={"help": "Reserved target revision metadata for externally loaded target modules."},
    )
    disable_dropout: bool = field(
        default=False,
        metadata={"help": "Put both drafter and target states in eval mode before training."},
    )
    dataset_text_field: str | None = field(
        default="text",
        metadata={"help": "Name of the text field used when tokenizing raw text datasets."},
    )
    assistant_only_loss: bool = field(
        default=False,
        metadata={"help": "Whether to compute CE/KL only on assistant/completion tokens when masks exist."},
    )
    completion_only_loss: bool | None = field(
        default=None,
        metadata={"help": "Deprecated alias for `assistant_only_loss`; if set, it overrides it."},
    )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        """Normalize aliases and validate speculative-decoding options.

        Empty optional method names become ``None`` so the step code does not
        compile distinct cache entries for equivalent configs.  Numerical
        fields are cast to their runtime types before validation, and target
        revision strings are rejected because this trainer only accepts
        already loaded target models.

        Args:
            max_sequence_length (int | None): Maximum sequence length forwarded
                to the base class ``__post_init__``.
            quantization_block (int | None): Quantization block size forwarded
                to the base class ``__post_init__``.

        Returns:
            None

        Raises:
            ValueError: If ``target_model_revision`` is not ``None``,
                ``temperature`` is not strictly positive, ``alpha`` is outside
                ``[0, 1]``, ``num_draft_tokens`` is less than ``1``,
                ``draft_target_offset`` is negative, or ``target_logits_attr``
                is empty.
        """
        if self.completion_only_loss is not None:
            self.assistant_only_loss = bool(self.completion_only_loss)
        self.completion_only_loss = bool(self.assistant_only_loss)

        if self.target_model_revision is not None:
            raise ValueError("`target_model_revision` is metadata only; pass an initialized target model/state.")
        self.temperature = float(self.temperature)
        if self.temperature <= 0.0:
            raise ValueError("`temperature` must be strictly positive.")
        self.alpha = float(self.alpha)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("`alpha` must be within [0, 1].")
        self.num_draft_tokens = int(self.num_draft_tokens)
        if self.num_draft_tokens < 1:
            raise ValueError("`num_draft_tokens` must be >= 1.")
        self.draft_target_offset = int(self.draft_target_offset)
        if self.draft_target_offset < 0:
            raise ValueError("`draft_target_offset` must be non-negative.")
        if not self.target_logits_attr:
            raise ValueError("`target_logits_attr` must be a non-empty attribute path.")
        if self.metrics_to_show_in_rich_pbar is None:
            self.metrics_to_show_in_rich_pbar = list(_DEFAULT_SPEC_METRICS_TO_SHOW)
        if self.draft_logits_attr == "":
            self.draft_logits_attr = None
        if self.draft_chain_method == "":
            self.draft_chain_method = None
        if self.drafter_forward_method == "":
            self.drafter_forward_method = None
        if self.target_forward_method == "":
            self.target_forward_method = None
        if hasattr(super(), "__post_init__"):
            super().__post_init__(
                max_sequence_length=max_sequence_length,
                quantization_block=quantization_block,
            )

    __hash__ = hash_fn
