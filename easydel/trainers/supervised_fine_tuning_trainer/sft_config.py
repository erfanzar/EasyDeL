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
import typing as tp
from dataclasses import dataclass, field

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
        dataset_kwargs: Extra keyword arguments forwarded to the
            (packed or unpacked) dataset constructor; useful for opting
            into experimental packing knobs without growing the config.
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
        metadata={"help": "Whether to add special tokens."},
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
        metadata={"help": "Number of examples to tokenize per batch."},
    )
    dataset_kwargs: dict[str, tp.Any] | None = field(
        default=None,
        metadata={"help": "Dictionary of optional keyword arguments to pass when creating datasets."},
    )
    eval_packing: bool | None = field(
        default=None,
        metadata={"help": "Whether to pack the eval dataset. If None, uses the same value as packing."},
    )
    num_of_sequences: int = field(
        default=1024,
        metadata={"help": "Number of sequences to use for the ConstantLengthDataset."},
    )

    __hash__ = hash_fn
