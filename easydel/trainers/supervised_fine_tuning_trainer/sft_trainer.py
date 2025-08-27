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
from __future__ import annotations

import typing as tp
import warnings

from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType

from ..trainer import Trainer
from ..utils import DataCollatorForCompletionOnlyLM, create_constant_length_dataset, get_formatting_func_from_dataset
from .sft_config import SFTConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset
logger = get_logger(__name__)


class SFTTrainer(Trainer):
    """
    Trainer class for Supervised Fine-Tuning (SFT) of language models.

    This trainer extends the `Trainer` and provides functionalities
    specific to supervised fine-tuning tasks.
    """

    def __init__(
        self,
        arguments: SFTConfig,
        processing_class: ProcessingClassType,
        model: EasyDeLBaseModule | EasyDeLState | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        formatting_func: tp.Callable | None = None,
        data_collator: DataCollatorForCompletionOnlyLM | None = None,
    ):
        tokenizer = processing_class
        if hasattr(processing_class, "tokenizer"):
            tokenizer = processing_class.tokenizer
        if getattr(
            tokenizer,
            "pad_token",
            None,
        ) is None and hasattr(
            tokenizer,
            "eos_token",
        ):
            tokenizer.pad_token = tokenizer.eos_token
        assert isinstance(arguments, SFTConfig), "passed argument must be a `SFTConfig`."

        if formatting_func is None and arguments.dataset_text_field is None:
            formatting_func = get_formatting_func_from_dataset(train_dataset, processing_class)

        if not arguments.packing:
            if data_collator:
                raise ValueError(
                    "You passed `packing=False` to the SFTTrainer, but you didn't pass a "
                    "`dataset_text_field` or `formatting_func` argument."
                )

        self.dataset_num_proc = arguments.dataset_num_proc
        self.dataset_batch_size = arguments.dataset_batch_size
        self.arguments = arguments
        if arguments.dataset_kwargs is None:
            arguments.dataset_kwargs = {}
        if train_dataset is not None:
            train_dataset = self._prepare_dataset(
                train_dataset,
                processing_class,
                arguments.packing,
                arguments.dataset_text_field,
                arguments.max_sequence_length,
                formatting_func,
                arguments.num_of_sequences,
                arguments.chars_per_token,
                remove_unused_columns=arguments.remove_unused_columns,
                add_special_tokens=arguments.add_special_tokens,
                **arguments.dataset_kwargs,
            )
        if eval_dataset is not None:
            _multiple = isinstance(eval_dataset, dict)
            _eval_datasets = eval_dataset if _multiple else {"singleton": eval_dataset}

            eval_packing = arguments.packing if arguments.eval_packing is None else arguments.eval_packing

            for _eval_dataset_name, _eval_dataset in _eval_datasets.items():
                _eval_datasets[_eval_dataset_name] = self._prepare_dataset(
                    _eval_dataset,
                    processing_class,
                    eval_packing,
                    arguments.dataset_text_field,
                    arguments.max_sequence_length,
                    formatting_func,
                    arguments.num_of_sequences,
                    arguments.chars_per_token,
                    remove_unused_columns=arguments.remove_unused_columns,
                    add_special_tokens=arguments.add_special_tokens,
                    **arguments.dataset_kwargs,
                )
            if not _multiple:
                eval_dataset = _eval_datasets["singleton"]
        if hasattr(tokenizer, "padding_side"):
            if tokenizer.padding_side is not None and tokenizer.padding_side != "left":
                warnings.warn(
                    "You passed a processing_class with `padding_side` not equal to `left` to the SFTTrainer. "
                    "This might lead to some unexpected behaviour due to overflow issues when training a "
                    "model in half-precision. You might consider adding `processing_class.padding_side = 'left'`"
                    " to your code.",
                    stacklevel=1,
                )
        if not isinstance(model, EasyDeLState):
            model = model.to_state()
        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            model_state=model,
            data_collator=data_collator,
        )

    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        packing,
        dataset_text_field,
        max_sequence_length,
        formatting_func,
        num_of_sequences,
        chars_per_token,
        remove_unused_columns=True,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        """
        Prepares the dataset for training by applying tokenization and packing (if enabled).

        Args:
            dataset (Dataset): The dataset to prepare.
            processing_class (ProcessingClassType): The processing_class to use.
            packing (bool): Whether to pack multiple sequences into a single sample.
            dataset_text_field (str): The name of the text field in the dataset.
            max_sequence_length (int): The maximum sequence length.
            formatting_func (tp.Callable): A formatting function to apply to each sample.
            num_of_sequences (int): Number of sequences to pack in each sample (if packing is enabled).
            chars_per_token (float): Average number of characters per token.
            remove_unused_columns (bool, optional): Whether to remove unused columns. Defaults to True.
            append_concat_token (bool, optional): Whether to append a concat token for packing. Defaults to True.
            add_special_tokens (bool, optional): Whether to add special tokens during tokenization. Defaults to True.

        Returns:
            Dataset: The processed dataset ready for training.

        Raises:
            ValueError: If the dataset is None or if packing is enabled without a
                `dataset_text_field` or `formatting_func`.
        """
        if dataset is None:
            raise ValueError("The dataset should not be None")

        if not packing:
            return self._prepare_non_packed_dataloader(
                processing_class,
                dataset,
                dataset_text_field,
                max_sequence_length,
                formatting_func,
                add_special_tokens,
                remove_unused_columns,
            )

        else:
            return self._prepare_packed_dataloader(
                processing_class,
                dataset,
                dataset_text_field,
                max_sequence_length,
                num_of_sequences,
                chars_per_token,
                formatting_func,
                append_concat_token,
                add_special_tokens,
            )

    def _prepare_non_packed_dataloader(
        self,
        processing_class: ProcessingClassType,
        dataset,
        dataset_text_field,
        max_sequence_length,
        formatting_func=None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        """
        Prepares a non-packed dataloader from the given dataset.

        This method tokenizes the text data in the dataset, truncates or pads sequences to a fixed length,
        and removes unused columns as specified. It's suitable for datasets where each sample represents
        a single sequence.

        Args:
            processing_class: The processing_class to use for text encoding.
            dataset (Dataset): The dataset to prepare.
            dataset_text_field (str): The name of the text field in the dataset.
            max_sequence_length (int): The maximum sequence length.
            formatting_func (tp.Callable, optional): A formatting function to apply to each sample before tokenization.
                Defaults to None.
            add_special_tokens (bool, optional): Whether to add special tokens during tokenization. Defaults to True.
            remove_unused_columns (bool, optional): Whether to remove unused columns from the dataset. Defaults to True.

        Returns:
            Dataset: The processed dataset ready for training.
        """
        from datasets import Dataset

        if dataset_text_field is None and formatting_func is None:
            raise ValueError("please provide `dataset_text_field` or `formatting_func`.")

        def tokenize(element):
            inputs = element[dataset_text_field] if formatting_func is None else formatting_func(element)

            outputs = processing_class(
                text=inputs,
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding="max_length",
                max_length=max_sequence_length,
                return_overflowing_tokens=False,
                return_attention_mask=True,
                return_length=False,
            )

            if formatting_func is not None and not isinstance(formatting_func(element), list):
                raise ValueError(
                    "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                )

            return outputs

        signature_columns = ["input_ids", "labels", "attention_mask"]

        if dataset.column_names is not None:
            extra_columns = list(set(dataset.column_names) - set(signature_columns))
        else:
            extra_columns = []

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with "
                "the default collator and yield to errors. If you want to inspect dataset other columns (in this "
                f"case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the "
                "default collator and create your own data collator in order to inspect the unused dataset columns.",
                UserWarning,
                stacklevel=1,
            )

        map_kwargs = {
            "batched": True,
            "remove_columns": dataset.column_names if remove_unused_columns else None,
            "batch_size": self.dataset_batch_size,
        }
        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = self.dataset_num_proc
        tokenized_dataset = dataset.map(tokenize, **map_kwargs)
        return tokenized_dataset

    @staticmethod
    def _prepare_packed_dataloader(
        processing_class,
        dataset,
        dataset_text_field,
        max_sequence_length,
        num_of_sequences,
        chars_per_token,
        formatting_func=None,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        """
        Prepares a packed dataloader from the given dataset.

        This method is designed for efficient training of language models by packing multiple
        sequences from the dataset into a single sample. This can be particularly beneficial
        for handling long sequences and optimizing GPU/TPU utilization.

        Args:
            processing_class: The processing_class used for text encoding.
            dataset (Dataset): The dataset to prepare.
            dataset_text_field (str): The name of the text field in the dataset.
            max_sequence_length (int): The maximum length of each packed sequence.
            num_of_sequences (int): The number of sequences to pack into a single sample.
            chars_per_token (float): The average number of characters per token, used for estimating
                the number of tokens in a text sequence.
            formatting_func (tp.Callable, optional): A function to format each sample from the dataset
                before packing. It should take a sample as input and return a dictionary with a "text"
                key containing the processed text. Defaults to None.
            append_concat_token (bool, optional): Whether to append a special concatenation token
                between packed sequences. Defaults to True.
            add_special_tokens (bool, optional): Whether to add special tokens (like BOS, EOS)
                during tokenization. Defaults to True.

        Returns:
            Dataset: The processed dataset with packed sequences.

        Raises:
            ValueError: If both `dataset_text_field` and `formatting_func` are None, or if there's
                an error during dataset packing.
        """
        if dataset_text_field is not None or formatting_func is not None:
            tokenizer = processing_class

            if hasattr(processing_class, "tokenizer"):
                tokenizer = processing_class.tokenizer

            if getattr(tokenizer, "pad_token_id", None) is None and hasattr(tokenizer, "eos_token_id"):
                tokenizer.pad_token_id = tokenizer.eos_token_id
            if processing_class is None:
                raise ValueError(
                    "You need to pass a processing_class when using `dataset_text_field` with `SFTTrainer`."
                )
            processing_class.eos_token_id = tokenizer.eos_token_id
            processing_class.pad_token_id = tokenizer.pad_token_id
            constant_length_iterator = create_constant_length_dataset(
                processing_class=processing_class,
                dataset=dataset,
                dataset_text_field=dataset_text_field,
                formatting_func=formatting_func,
                seq_length=max_sequence_length,
                infinite=False,
                num_of_sequences=num_of_sequences,
                chars_per_token=chars_per_token,
                eos_token_id=tokenizer.eos_token_id,
                append_concat_token=append_concat_token,
                add_special_tokens=add_special_tokens,
            )

            def data_generator(inner_constant_length_iterator):
                yield from inner_constant_length_iterator()

            # Import Only and Only when needed, don't dst the runtime.
            try:
                from datasets import Dataset
                from datasets.arrow_writer import SchemaInferenceError
                from datasets.builder import DatasetGenerationError
            except ImportError as exc:
                raise ImportError(
                    "Could not import `datasets` from Hugging Face. Make sure to install the "
                    "library using `pip install datasets`."
                ) from exc
            try:
                packed_dataset = Dataset.from_generator(
                    data_generator,
                    gen_kwargs={"inner_constant_length_iterator": constant_length_iterator},
                )
            except (DatasetGenerationError, SchemaInferenceError) as exc:
                raise ValueError(
                    "Error occurred while packing the dataset. "
                    "Make sure that your dataset has enough samples to at least yield one packed sequence.\n"
                    f"External Information : {exc}"
                ) from exc
            return packed_dataset
        else:
            raise ValueError(
                "You need to pass a `dataset_text_field` or `formatting_func` argument to the SFTTrainer if you want "
                "to use the `ConstantLengthDataset`."
            )
