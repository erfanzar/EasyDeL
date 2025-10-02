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

import json
import typing as tp
from functools import partial

from eformer.loggings import get_logger

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.trainers.prompt_utils import (
    is_conversational,
    is_conversational_from_value,
    keep_arrays_map,
    maybe_convert_to_chatml,
    pack_dataset,
    pad_and_truncate_dataset,
    remove_none_values,
)
from easydel.utils import Registry

from ..trainer import Trainer
from ..utils import DataCollatorForCompletionOnlyLM, get_formatting_func_from_dataset
from .sft_config import SFTConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset
logger = get_logger(__name__)


@Registry.register("trainer", "sft")
class SFTTrainer(Trainer):
    """Supervised Fine-Tuning trainer for language models.

    Implements standard supervised fine-tuning for both base and instruction-tuned
    models. Supports various data formats including conversational datasets,
    completion-only training, and packed sequences for efficient training.

    Key features:
    - Automatic dataset formatting and tokenization
    - Support for conversational/chat templates
    - Sequence packing for improved efficiency
    - Completion-only loss (ignore prompt tokens)
    - Multi-turn conversation handling

    The trainer automatically handles:
    - Chat template application for conversational data
    - Proper padding and truncation strategies
    - Efficient packing of multiple sequences
    - Loss masking for prompt tokens when needed

    Attributes:
        arguments: SFTConfig with training hyperparameters
        tokenizer: Tokenizer for text processing
        dataset_num_proc: Number of processes for dataset operations
        dataset_batch_size: Batch size for dataset mapping

    Example:
        >>> config = SFTConfig(
        ...     per_device_train_batch_size=4,
        ...     learning_rate=2e-5,
        ...     packing=True,
        ...     max_sequence_length=2048
        ... )
        >>> trainer = SFTTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer,
        ...     formatting_func=lambda x: x["text"]  # Optional
        ... )
        >>> trainer.train()

    Note:
        For conversational datasets, the trainer expects either:
        - A 'messages' column with chat format
        - A custom formatting_func to extract text
        - A dataset_text_field pointing to the text column
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
        if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
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
        self.tokenizer = tokenizer
        if arguments.dataset_kwargs is None:
            arguments.dataset_kwargs = {}
        if train_dataset is not None:
            train_dataset = self._prepare_dataset(
                processing_class,
                train_dataset,
                arguments.dataset_text_field,
                formatting_func,
                arguments.packing,
            )
        if eval_dataset is not None:
            _multiple = isinstance(eval_dataset, dict)
            _eval_datasets = eval_dataset if _multiple else {"singleton": eval_dataset}

            eval_packing = arguments.packing if arguments.eval_packing is None else arguments.eval_packing

            for _eval_dataset_name, _eval_dataset in _eval_datasets.items():
                _eval_datasets[_eval_dataset_name] = self._prepare_dataset(
                    processing_class, _eval_dataset, arguments.dataset_text_field, formatting_func, eval_packing
                )
            if not _multiple:
                eval_dataset = _eval_datasets["singleton"]

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
        processing_class: ProcessingClassType,
        dataset,
        dataset_text_field,
        formatting_func=None,
        do_packing: bool = False,
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

        if isinstance(dataset, Dataset):  # IterableDataset does not support `with_transform`
            dataset = dataset.with_transform(remove_none_values)

        column_names = list(next(iter(dataset)).keys())
        is_processed = "input_ids" in column_names

        map_kwargs = {}
        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = self.dataset_num_proc

        if formatting_func is not None and is_processed:
            logger.warning(
                "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                "`formatting_func` or pass a dataset that is not already processed.",
            )

        if formatting_func is not None and not is_processed:
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = "Applying formatting function to dataset"

            def _func(example):
                return {"text": formatting_func(example)}

            dataset = dataset.map(_func, batched=False, **map_kwargs)

        if not is_processed:
            first_example = next(iter(dataset))
            if is_conversational_from_value(first_example):
                if isinstance(dataset, Dataset):
                    map_kwargs["desc"] = "Converting dataset to ChatML"
                column_names = next(iter(dataset)).keys()
                dataset = dataset.map(
                    maybe_convert_to_chatml,
                    remove_columns="conversations" if "conversations" in column_names else None,
                    **map_kwargs,
                )

            first_example = next(iter(dataset))
            if not is_conversational(first_example):
                if isinstance(dataset, Dataset):
                    map_kwargs["desc"] = "Adding EOS to dataset"

                def add_eos(example, eos_token):
                    if "text" in example and not example["text"].endswith(eos_token):
                        example["text"] = example["text"] + eos_token
                    elif "completion" in example and not example["completion"].endswith(eos_token):
                        example["completion"] = example["completion"] + eos_token
                    return example

                dataset = dataset.map(
                    add_eos,
                    fn_kwargs={"eos_token": processing_class.eos_token},
                    remove_columns="messages" if "messages" in column_names else None,
                    **map_kwargs,
                )

            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = "Tokenizing dataset"

            def tokenize(example, processing_class, dataset_text_field, assistant_only_loss):
                if "prompt" in example:
                    output = {}
                    if is_conversational(example):
                        prompt_ids = processing_class.apply_chat_template(
                            example["prompt"],
                            tools=example.get("tools"),
                            **example.get("chat_template_kwargs", {}),
                        )
                        prompt_completion_processed = processing_class.apply_chat_template(
                            example["prompt"] + example["completion"],
                            return_dict=True,
                            return_assistant_tokens_mask=assistant_only_loss,
                            tools=example.get("tools"),
                            **example.get("chat_template_kwargs", {}),
                        )
                        prompt_completion_ids = prompt_completion_processed["input_ids"]
                        if "assistant_masks" in prompt_completion_processed:
                            output["assistant_masks"] = prompt_completion_processed["assistant_masks"]
                    else:
                        prompt_ids = processing_class(text=example["prompt"])["input_ids"]
                        prompt_completion_ids = processing_class(text=example["prompt"] + example["completion"])[
                            "input_ids"
                        ]

                    if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                        logger.warning(
                            "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                            "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                            "token handling. Verify that the tokenizer is processing text consistently."
                        )

                    completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
                    output["input_ids"] = prompt_completion_ids
                    output["completion_mask"] = completion_mask

                else:
                    if is_conversational(example):
                        tools = example.get("tools")
                        if isinstance(tools, str):
                            tools = json.loads(tools)
                        elif isinstance(tools, list):
                            if isinstance(tools[0], str):
                                tools = json.loads(tools)
                        processed = processing_class.apply_chat_template(
                            example["messages"],
                            return_dict=True,
                            return_assistant_tokens_mask=False,
                            return_attention_mask=True,
                            tools=tools,
                            truncation=True,
                            max_length=self.arguments.max_sequence_length,
                            **example.get("chat_template_kwargs", {}),
                        )
                        if "assistant_masks" in processed and 1 not in processed["assistant_masks"]:
                            raise RuntimeError(
                                "You're using `assistant_only_loss=True`, but at least one example has no "
                                "assistant tokens. This usually means the tokenizer's chat template doesn't "
                                "generate assistant masks — it may be missing the `{% generation %}` keyword. Please "
                                "check the template and ensure it's correctly configured to support assistant "
                                "masking."
                            )
                        output = processed
                    else:
                        output = processing_class(
                            text=example[dataset_text_field],
                            return_dict=True,
                            return_attention_mask=True,
                            truncation=True,
                            max_length=self.arguments.max_sequence_length,
                        )
                return output

            dataset = dataset.map(
                tokenize,
                fn_kwargs={
                    "processing_class": processing_class,
                    "dataset_text_field": dataset_text_field,
                    "assistant_only_loss": False,
                },
                **map_kwargs,
            )

        if do_packing:
            columns_names = next(iter(dataset)).keys()
            if self.arguments.max_sequence_length is None:
                raise ValueError("When packing is enabled, `max_sequence_length` can't be `None`.")
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = "Packing dataset"

            columns = ["input_ids"]
            if "completion_mask" in columns_names:
                columns.append("completion_mask")
            if "assistant_masks" in columns_names:
                columns.append("assistant_masks")
            if "attention_mask" in columns_names:
                columns.append("attention_mask")

            dataset = dataset.select_columns(columns)

            dataset = pack_dataset(
                dataset,
                self.arguments.max_sequence_length,
                self.arguments.packing_strategy,
                map_kwargs,
            )
        if isinstance(dataset, Dataset):
            map_kwargs["desc"] = "Truncating dataset"
        columns_names = next(iter(dataset)).keys()
        dataset = dataset.map(
            partial(
                keep_arrays_map,
                array_fields=["input_ids", "attention_mask", "position_ids", "assistant_masks", "completion_mask"],
                drop_fields=["seq_lengths"],
            ),
            remove_columns=columns_names,
        )
        dataset = pad_and_truncate_dataset(
            dataset,
            max_length=self.arguments.max_sequence_length,
            padding_token_id=self.tokenizer.pad_token_id,
            padding=True,
            truncate=True,
            map_kwargs=map_kwargs,
        )
        return dataset
