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
"""Configuration dataclass for the supervised fine-tuning (SFT) trainer.

Defines :class:`SFTConfig` and the SFT-specific knobs layered on top of
:class:`TrainingArguments`: dataset text field selection, optional
sequence packing (``bfd`` / ``wrapped``), assistant-only loss masking,
and dataset-preprocessing worker controls.
"""

import typing as tp
from dataclasses import dataclass, field

from easydel.infra.loss_utils import LossConfig
from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@Registry.register("trainer-arguments", "sft")
@dataclass
class SFTConfig(TrainingArguments):
    r"""Hyperparameters and dataset knobs for :class:`SFTTrainer`.

    Supervised fine-tuning runs standard causal-LM cross-entropy on
    chat-templated text, optionally restricted to assistant turns
    (``assistant_only_loss``) and/or accelerated by sequence packing
    (``packing``). The fields below extend
    :class:`TrainingArguments` with SFT-specific dataset and packing
    controls; everything else (optimiser, scheduler, sharding,
    quantisation) is inherited unchanged.

    Attributes:
        trainer_prefix: Prefix used when naming logs / checkpoints /
            W&B runs. Default: ``"SFT"``.
        dataset_text_field: Name of the column carrying the chat-
            templatable / pre-rendered text. Used by the SFT
            preprocessing transform when assembling the loss target.
            Default: ``"text"``.
        chat_template_path: Optional local Jinja template path or
            tokenizer repo/local directory whose chat template should be
            copied onto the active tokenizer.
        eos_token: Optional EOS token override applied to the tokenizer
            before preprocessing.
        pad_token: Optional pad token override applied to the tokenizer
            before preprocessing and batching.
        shuffle_dataset: TRL-compatible alias for
            ``shuffle_train_dataset``. ``None`` leaves the base setting
            unchanged.
        add_special_tokens: When ``True`` the tokenizer is allowed to
            inject BOS/EOS tokens during chat-template rendering. Most
            chat templates already produce their own delimiters, so the
            default is ``False`` to avoid double-BOS issues.
        packing: When ``True`` short sequences are concatenated into
            fixed-length blocks of ``max_length`` tokens to reduce
            padding waste; concatenation respects per-sequence
            attention boundaries via the document-id mask emitted by
            the packing strategy.
        packing_strategy: Selects the packing algorithm. ``"bfd"``
            (best-fit decreasing, the default) sorts sequences by
            descending length and greedily places them in the smallest
            block that still has room. ``"wrapped"`` simply concatenates
            sequences in order and wraps when the block fills up.
        assistant_only_loss: When ``True`` the loss mask is restricted
            to assistant turns produced by the chat template; prompt
            and tool turns receive ``-100`` labels and contribute zero
            gradient. Requires the dataset to be conversational (i.e.
            chat-template renders messages with ``role`` markers).
        learning_rate: Initial learning rate for the optimiser
            constructed by :class:`TrainingArguments`. Default
            ``2e-5`` overrides the base default.
        dataset_num_proc: Number of worker processes used by the
            dataset preprocessing pipeline. ``None`` runs sequentially.
            Only honoured when ``packing=False``.
        dataset_batch_size: Number of rows tokenised per worker call.
            Set to ``<= 0`` or ``None`` to tokenise the full dataset
            as a single batch.
        dataset_kwargs: TRL-compatible dataset preparation options.
            Currently only ``{"skip_prepare_dataset": True}`` is
            supported, which tells the trainer to trust preprocessed
            rows and skip the SFT tokenization transform.
        eval_packing: Optional eval-time override for ``packing``. When
            ``None`` the eval pipeline mirrors the train-time setting.
        num_of_sequences: Number of sequences buffered by the constant-
            length dataset wrapper that backs sequence packing.
    """

    trainer_prefix: str | None = field(
        default="SFT",
        metadata={"help": "default prefix name for trainer."},
    )
    dataset_text_field: str | None = field(
        default="text",
        metadata={"help": "Name of the text field of the dataset."},
    )
    add_special_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Deprecated compatibility field. EasyDeL SFT does not currently expose a tokenizer special-token "
                "insertion switch; leave as False."
            )
        },
    )
    truncation_mode: tp.Literal["keep_end", "keep_start"] = field(
        default="keep_start",
        metadata={"help": "Truncation mode used during SFT tokenization."},
    )
    packing: bool = field(
        default=False,
        metadata={
            "help": "Whether to group multiple sequences into fixed-length blocks to improve computational efficiency "
            "and reduce padding. Uses `max_length` to define sequence length."
        },
    )
    packing_strategy: str = field(
        default="bfd",
        metadata={
            "help": "Strategy for packing sequences. Can be either `'bfd'` (best-fit decreasing, default), or "
            "`'wrapped'`."
        },
    )
    assistant_only_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to compute loss only on the assistant part of the sequence. If set to `True`, loss is "
                "computed only on the assistant responses, which is supported only for conversational datasets."
                " If `False`, loss is computed on the entire sequence."
            )
        },
    )
    completion_only_loss: bool | None = field(
        default=None,
        metadata={
            "help": (
                "Explicit TRL-compatible completion-only loss knob. When set, it overrides assistant_only_loss "
                "for prompt/completion style datasets."
            )
        },
    )
    padding_free: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable padding-free style preprocessing where supported. In EasyDeL this opts into packed sources "
                "so batches avoid padding waste while preserving current model call contracts."
            )
        },
    )
    chat_template_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional local Jinja file path or tokenizer repo/local directory whose chat template should be "
                "assigned to the active tokenizer before preprocessing."
            )
        },
    )
    eos_token: str | None = field(
        default=None,
        metadata={"help": "Optional EOS token override applied to the tokenizer before preprocessing."},
    )
    pad_token: str | None = field(
        default=None,
        metadata={"help": "Optional pad token override applied to the tokenizer before preprocessing."},
    )
    shuffle_dataset: bool | None = field(
        default=None,
        metadata={"help": "TRL-compatible alias for `shuffle_train_dataset`."},
    )
    pad_to_multiple_of: int | None = field(
        default=None,
        metadata={"help": "If set, tokenized SFT rows are padded to a multiple of this value."},
    )
    loss_type: tp.Literal["nll", "dft", "chunked_nll"] = field(
        default="nll",
        metadata={"help": "SFT loss variant: standard NLL, DFT, or chunked NLL."},
    )
    completion_loss_chunk_size: int | None = field(
        default=None,
        metadata={
            "help": "Token chunk size used when loss_type='chunked_nll'. None uses the model/default loss config."
        },
    )
    activation_offloading: bool = field(
        default=False,
        metadata={"help": "TRL compatibility field. EasyDeL SFT does not currently support activation offloading."},
    )
    learning_rate: float = field(
        default=2.0e-5,
        metadata={"help": "Initial learning rate for the AdamW optimizer."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    dataset_batch_size: int = field(
        default=1000,
        metadata={
            "help": (
                "Deprecated compatibility field from HF Dataset.map preprocessing. EasyDeL lazy transforms do not "
                "batch-tokenize through this config; leave at the default."
            )
        },
    )
    dataset_kwargs: dict[str, object] | None = field(
        default=None,
        metadata={"help": "Dataset preparation options. Supported key: `skip_prepare_dataset`."},
    )
    eval_packing: bool | None = field(
        default=None,
        metadata={"help": "Whether to pack the eval dataset. If None, uses the same value as packing."},
    )
    num_of_sequences: int = field(
        default=1024,
        metadata={
            "help": (
                "Deprecated compatibility field from ConstantLengthDataset-style packing. EasyDeL sharded packing "
                "does not use this value; leave at the default."
            )
        },
    )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        if self.loss_type not in ("nll", "dft", "chunked_nll"):
            raise ValueError("`loss_type` must be one of 'nll', 'dft', or 'chunked_nll'.")
        if self.truncation_mode not in ("keep_end", "keep_start"):
            raise ValueError("`truncation_mode` must be either 'keep_end' or 'keep_start'.")
        if self.add_special_tokens:
            raise ValueError(
                "`add_special_tokens=True` is not supported by EasyDeL SFT; leave it False to avoid silently ignored "
                "tokenizer special-token config."
            )
        if self.dataset_batch_size != 1000:
            raise ValueError(
                "`dataset_batch_size` is not used by EasyDeL SFT's lazy preprocessing pipeline; leave it at 1000."
            )
        if self.num_of_sequences != 1024:
            raise ValueError(
                "`num_of_sequences` is not used by EasyDeL SFT's sharded packing pipeline; leave it at 1024."
            )
        if self.completion_only_loss is not None:
            self.assistant_only_loss = bool(self.completion_only_loss)
        if self.shuffle_dataset is not None:
            self.shuffle_train_dataset = bool(self.shuffle_dataset)
        # The shared `sequence_packing` flag (base TrainingArguments) feeds SFT's existing
        # `packing` machinery, which already emits segment_ids + completion masks.
        if getattr(self, "sequence_packing", False) and not self.packing:
            self.packing = True
        if self.padding_free and not self.packing:
            self.packing = True
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of <= 0:
            raise ValueError("`pad_to_multiple_of` must be a positive integer when set.")
        if self.packing_strategy == "bfd_split":
            self.packing_strategy = "bfd"
        if self.dataset_kwargs is not None:
            unsupported_dataset_kwargs = set(self.dataset_kwargs) - {"skip_prepare_dataset"}
            if unsupported_dataset_kwargs:
                raise ValueError(
                    "`dataset_kwargs` only supports `skip_prepare_dataset`; "
                    f"got unsupported keys: {sorted(unsupported_dataset_kwargs)}."
                )
        if self.loss_type == "chunked_nll" and self.completion_loss_chunk_size is not None:
            loss_config = self.loss_config or LossConfig()
            loss_config.chunk_token_size = int(self.completion_loss_chunk_size)
            self.loss_config = loss_config
        if hasattr(super(), "__post_init__"):
            super().__post_init__(
                max_sequence_length=max_sequence_length,
                quantization_block=quantization_block,
            )

    __hash__ = hash_fn
