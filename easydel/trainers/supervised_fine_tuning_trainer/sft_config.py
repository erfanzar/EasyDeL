# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
from dataclasses import field

from eformer.pytree import auto_pytree

from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@auto_pytree
class SFTConfig(TrainingArguments):
    r"""
    Configuration class for the [`SFTTrainer`].

    Parameters:
        model_name (str): The name of the model. Defaults to "SFTTrainer".
        dataset_text_field (str, optional): Name of the text field of the dataset. If provided, the trainer will
            automatically create a [`ConstantLengthDataset`] based on `dataset_text_field`. Defaults to None.
        packing (bool, optional): Controls whether the [`ConstantLengthDataset`] packs the sequences of the
            dataset. Defaults to False.
        learning_rate (float, optional): Initial learning rate for [`AdamW`] optimizer. The default value replaces
            that of [`~transformers.TrainingArguments`]. Defaults to 2e-5.
        dataset_num_proc (int, optional): Number of processes to use for processing the dataset. Only used when
            `packing=False`. Defaults to None.
        dataset_batch_size (int, optional): Number of examples to tokenize per batch. If
            `dataset_batch_size <= 0` or `dataset_batch_size is None`, tokenizes the full dataset as a single
            batch. Defaults to 1000.
        dataset_kwargs (dict[str, Any], optional): Dictionary of optional keyword arguments to pass when creating
            packed or non-packed datasets. Defaults to None.
        eval_packing (bool, optional): Whether to pack the eval dataset. If `None`, uses the same value as
            `packing`. Defaults to None.
        num_of_sequences (int, optional): Number of sequences to use for the [`ConstantLengthDataset`].
            Defaults to 1024.
        chars_per_token (float, optional): Number of characters per token to use for the
            [`ConstantLengthDataset`]. See
            [chars_token_ratio](https://github.com/huggingface/trl/blob/08f550674c553c36c51d1027613c29f14f3676a5/examples/stack_llama/scripts/supervised_finetuning.py#L53)
            for more details. Defaults to 3.6.
    """

    trainer_prefix: str | None = field(
        default="sfttrainer",
        metadata={"help": "default prefix name for trainer."},
    )
    dataset_text_field: str | None = field(
        default=None,
        metadata={"help": "Name of the text field of the dataset."},
    )
    add_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether to add special tokens."},
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Controls whether the sequences of the dataset are packed."},
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
    chars_per_token: float = field(
        default=3.6,
        metadata={"help": "Number of characters per token to use for the ConstantLengthDataset."},
    )

    __hash__ = hash_fn
