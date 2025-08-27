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
import logging
import random
import typing as tp
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from threading import current_thread

import chex
import jax
import numpy as np
from datasets import Dataset, IterableDataset
from datasets.distributed import split_dataset_by_node
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from grain import python as pygrain
from jax import numpy as jnp
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder

from easydel.infra.utils import ProcessingClassType

logger = get_logger(__name__)


class JaxDistributedConfig:
    """
    From EasyLM
    Utility class for initializing JAX distributed.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.initialize_jax_distributed = False
        config.coordinator_address = placeholder(str)
        config.num_processes = placeholder(int)
        config.process_id = placeholder(int)
        config.local_device_ids = placeholder(str)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def initialize(cls, config=None):
        config = cls.get_default_config(config)
        if config.initialize_jax_distributed:
            if config.local_device_ids is not None:
                local_device_ids = [int(x) for x in config.local_device_ids.split(",")]
            else:
                local_device_ids = None

            jax.distributed.initialize(
                coordinator_address=config.coordinator_address,
                num_processes=config.num_processes,
                process_id=config.process_id,
                local_device_ids=local_device_ids,
            )


def create_prompt_creator(processing_class):
    def to_role_and_content(field):
        return {
            "conversation": [
                {"role": "user", "content": field["conversation"][0]["input"]},
                {"role": "assistant", "content": field["conversation"][0]["output"]},
            ]
        }

    def _pc(sample):
        return conversations_formatting_function(processing_class, messages_field="conversation")(
            to_role_and_content(sample)
        )

    return _pc


def create_constant_length_dataset(
    processing_class,
    dataset,
    dataset_text_field: str | None = None,
    formatting_func: tp.Callable | None = None,
    infinite: bool = False,
    seq_length: int = 1024,
    num_of_sequences: int = 1024,
    chars_per_token: float = 3.6,
    eos_token_id: int = 0,
    shuffle: bool = True,
    append_concat_token: bool = True,
    add_special_tokens: bool = True,
) -> tp.Callable[[], tp.Iterator[dict[str, jnp.ndarray]]]:
    """
    Creates a generator function that yields constant length chunks of tokens from a stream of text files.

    Args:
        processing_class: The processor used for processing the data.
        dataset: Dataset with text files.
        dataset_text_field: Name of the field in the dataset that contains the text.
        formatting_func: Function that formats the text before tokenization.
        infinite: If True the iterator is reset after dataset reaches end else stops.
        seq_length: Length of token sequences to return.
        num_of_sequences: Number of token sequences to keep in buffer.
        chars_per_token: Number of characters per token used to estimate number of tokens in text buffer.
        eos_token_id: Id of the end of sequence token if the passed processing_class does not have an EOS token.
        shuffle: Shuffle the examples before they are returned.
        append_concat_token: If true, appends eos_token_id at the end of each sample being packed.
        add_special_tokens: If true, processing_class adds special tokens to each sample being packed.

    Returns:
        A generator function that yields dictionaries containing input_ids and attention_mask as jnp.arrays
    """
    if processing_class.eos_token_id is None:
        warnings.warn(
            "The passed processing_class does not have an EOS token. We will use the passed eos_token_id instead which "
            f"corresponds to {eos_token_id}. If this is not the correct EOS token, "
            "make sure to pass the correct eos_token_id.",
            stacklevel=1,
        )

    concat_token_id = processing_class.eos_token_id if processing_class.eos_token_id else eos_token_id
    max_buffer_size = seq_length * chars_per_token * num_of_sequences

    # Input validation and formatting function setup
    if dataset_text_field is not None and formatting_func is not None:
        warnings.warn(
            "Only one of `dataset_text_field` and `formatting_func` should be provided. "
            "Ignoring `dataset_text_field` and using `formatting_func`.",
            stacklevel=1,
        )

    if formatting_func is not None:
        if formatting_func.__code__.co_argcount > 1:
            warnings.warn(
                "The passed formatting_func has more than one argument. Usually that "
                "function should have a single argument `example` which corresponds "
                "to the dictionary returned by each element of the dataset. Make sure you know what you are doing.",
                stacklevel=1,
            )
    elif dataset_text_field is not None:
        formatting_func = lambda x: x[dataset_text_field]  # noqa
    else:
        raise ValueError("Either `dataset_text_field` or `formatting_func` should be provided.")

    def constant_length_generator() -> tp.Iterator[dict[str, jnp.ndarray]]:
        iterator = iter(dataset)
        more_examples = True

        while more_examples:
            buffer, buffer_len = [], 0

            # Fill the buffer
            while True:
                if buffer_len >= max_buffer_size:
                    break
                try:
                    prompt = formatting_func(next(iterator))
                    if isinstance(prompt, list):
                        prompt = "".join(p for p in prompt)
                    buffer.append(prompt)
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if infinite:
                        iterator = iter(dataset)
                        warnings.warn(
                            "The dataset reached end and the iterator is reset to the start.",
                            stacklevel=1,
                        )
                    else:
                        more_examples = False
                        break

            if shuffle:
                random.shuffle(buffer)

            # Tokenize all texts in the buffer
            tokens = processing_class(
                text=buffer,
                add_special_tokens=add_special_tokens,
                truncation=False,
            )
            tokenized_inputs = tokens["input_ids"]
            attention_masks = tokens["attention_mask"]
            # Concatenate all tokens and attention masks
            all_token_ids = []
            all_attention_masks = []
            for tokenized_input, attention_mask in zip(tokenized_inputs, attention_masks, strict=False):
                if append_concat_token:
                    tokenized_input = [*tokenized_input, concat_token_id]
                    attention_mask = [*attention_mask, 1]
                all_token_ids.extend(tokenized_input)
                all_attention_masks.extend(attention_mask)

            # Create fixed-length examples
            examples = []
            examples_attention_masks = []
            for i in range(0, len(all_token_ids), seq_length):
                input_ids = all_token_ids[i : i + seq_length]
                org_attention_masks = all_attention_masks[i : i + seq_length]
                if len(input_ids) == seq_length:
                    examples.append(input_ids)
                    examples_attention_masks.append(org_attention_masks)

            if shuffle:
                # Shuffle examples while keeping pairs together
                combined = list(zip(examples, examples_attention_masks, strict=False))
                random.shuffle(combined)
                examples, examples_attention_masks = zip(*combined, strict=False)

            # Yield examples
            for example, example_attention_mask in zip(examples, examples_attention_masks, strict=False):
                yield {
                    "input_ids": jnp.asarray(example, dtype="i4"),
                    "attention_mask": jnp.asarray(example_attention_mask, dtype="i4"),
                }

    return constant_length_generator


def _collate_batch(
    examples,
    processing_class,
    pad_to_multiple_of: int | None = None,
):
    if isinstance(examples[0], list | tuple):
        examples = [jnp.array(e, dtype=jnp.int64) for e in examples]

    length_of_first = len(examples[0])
    are_tensors_same_length = all(len(x) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return jnp.stack(examples, axis=0)

    if processing_class._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the processing_class you are using"
            f" ({processing_class.__class__.__name__}) does not have a pad token."
        )

    max_length = max(len(x) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = jnp.full(
        shape=(len(examples), max_length),
        fill_value=processing_class.pad_token_id,
        dtype=examples[0].dtype,
    )
    for i, example in enumerate(examples):
        if processing_class.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


def tolist(x):
    """from HF
    Args:
        x:

    Returns: X as tp.List

    """
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):
        x = x.numpy()
    return x.tolist()


class DataCollatorForCompletionOnlyLM:
    """Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensures that the loss is only
    calculated on the completion made by the assistant.
    """

    def __init__(
        self,
        processing_class: tp.Union[str, "PreTrainedTokenizerBase"],  # type:ignore #noqa
        response_template: str | list[int],
        instruction_template: str | list[int] | None = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        from transformers import AutoTokenizer

        if isinstance(processing_class, str):
            processing_class = AutoTokenizer.from_pretrained(processing_class)
            self.processing_class = processing_class
        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            self.instruction_token_ids = self.processing_class.encode(
                self.instruction_template, add_special_tokens=False
            )
        else:
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            self.response_token_ids = self.processing_class.encode(self.response_template, add_special_tokens=False)
        else:
            self.response_token_ids = response_template

        if (
            not mlm
            and self.instruction_template
            and self.processing_class.pad_token_id == self.processing_class.eos_token_id
        ):
            warnings.warn(
                "The pad_token_id and eos_token_id values of this processing_class are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value.",
                stacklevel=1,
            )

        self.ignore_index = ignore_index

    def _whole_word_mask(self, input_tokens: list[str], max_predictions=512):
        from transformers import BertTokenizer, BertTokenizerFast

        if not isinstance(self.processing_class, BertTokenizer | BertTokenizerFast):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information.",
                stacklevel=1,
            )

        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, round(len(input_tokens) * 0.15)))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def jax_mask_tokens(self, inputs: tp.Any, special_tokens_mask: tp.Any | None = None) -> tuple[tp.Any, tp.Any]:
        """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
        labels = np.copy(inputs)
        probability_matrix = np.full(labels.shape, 0.15)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.processing_class.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
            ]
            special_tokens_mask = np.array(special_tokens_mask, dtype=bool)
        else:
            special_tokens_mask = special_tokens_mask.astype(bool)

        probability_matrix[special_tokens_mask] = 0
        masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(bool)  # noqa
        labels[~masked_indices] = -100
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices  # noqa
        inputs[indices_replaced] = self.processing_class.mask_token_id
        indices_random = np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced  # noqa
        random_words = np.random.randint(  # noqa
            low=0,
            high=len(self.processing_class),
            size=np.count_nonzero(indices_random),
            dtype=np.int64,
        )
        inputs[indices_random] = random_words
        return inputs, labels

    def jax_call(self, examples: list[list[int] | tp.Any | dict[str, tp.Any]]) -> dict[str, tp.Any]:
        if isinstance(examples[0], tp.Mapping):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _collate_batch(
            input_ids,
            self.processing_class,
        )

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for ida in tolist(e["input_ids"]):
                token = self.processing_class._convert_id_to_token(ida)
                ref_tokens.append(token)

            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]  # noqa
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            mask_labels.append(self._whole_word_mask(ref_tokens))
        batch_mask = _collate_batch(
            mask_labels,
            self.processing_class,
        )
        inputs, labels = self.jax_mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}

    def __call__(self, examples: list[list[int] | tp.Any | dict[str, tp.Any]]) -> dict[str, tp.Any]:
        batch = self.jax_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in jnp.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    if self.response_token_ids == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist():
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f"following instance: {self.processing_class.decode(batch['input_ids'][i])} "
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`.",
                        stacklevel=1,
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in jnp.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f"following instance: {self.processing_class.decode(batch['input_ids'][i])} "
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`.",
                        stacklevel=1,
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in jnp.where(batch["labels"][i] == human_token_ids[0])[0]:
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the "
                        f"following instance: {self.processing_class.decode(batch['input_ids'][i])} "
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`.",
                        stacklevel=1,
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0, *human_token_ids_idxs]

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs, strict=False)):
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        return batch


@auto_pytree
class RewardDataCollatorWithPaddingTFDS:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.

    Args:
        tokenizer (`ProcessingClassType`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`int` or `None`, `optional`, defaults to `None`):
            If set will pad the sequence to a maximum provided value.
    """

    tokenizer: ProcessingClassType
    padding: bool | str = "max_length"
    max_length: int | None = None
    truncation_mode: str = "keep_end"

    def __call__(self, features: list[dict[str, tp.Any]]) -> dict[str, tp.Any]:
        features_chosen = []
        features_rejected = []
        margin = []
        has_margin = "margin" in features[0]
        for feature in features:
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, "
                    "`input_ids_rejected` and `attention_mask_rejected`"
                )

            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            if has_margin:
                margin.append(feature["margin"])
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="jax",
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="jax",
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
        }
        if has_margin:
            margin = jnp.array(margin, dtype="f4")
            batch["margin"] = margin
        return batch


@auto_pytree
class RewardDataCollatorWithPaddingGrain:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.

    Args:
        tokenizer (`ProcessingClassType`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`int` or `None`, `optional`, defaults to `None`):
            If set will pad the sequence to a maximum provided value.
    """

    tokenizer: ProcessingClassType
    padding: bool | str = "max_length"
    max_length: int | None = None
    truncation_mode: str = "keep_end"

    def __call__(self, features: dict[str, tp.Any]) -> dict[str, tp.Any]:
        features_chosen = []
        features_rejected = []
        margin = []
        has_margin = "margin" in features.keys()

        if (
            "input_ids_chosen" not in features.keys()
            or "input_ids_rejected" not in features.keys()
            or "attention_mask_chosen" not in features.keys()
            or "attention_mask_rejected" not in features.keys()
        ):
            raise ValueError(
                "The features should include `input_ids_chosen`, `attention_mask_chosen`, "
                "`input_ids_rejected` and `attention_mask_rejected`"
            )

        features_chosen.append(
            {
                "input_ids": features["input_ids_chosen"],
                "attention_mask": features["attention_mask_chosen"],
            }
        )
        features_rejected.append(
            {
                "input_ids": features["input_ids_rejected"],
                "attention_mask": features["attention_mask_rejected"],
            }
        )
        if has_margin:
            margin.append(features["margin"])
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="np",
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="np",
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
        }
        if has_margin:
            margin = np.array(margin, dtype="f4")
            batch["margin"] = margin
        return batch


@auto_pytree
class DataCollatorForPreferenceTFDS:
    r"""DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch."""

    max_prompt_length: int
    max_completion_length: int
    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: bool | None = False

    def __call__(self, features: list[dict[str, tp.Any]]) -> dict[str, tp.Any]:
        prompt_input_ids = [jnp.array(feature["prompt_input_ids"]) for feature in features]
        prompt_attention_mask = [jnp.ones_like(input_ids) for input_ids in prompt_input_ids]
        chosen_input_ids = [jnp.array(feature["chosen_input_ids"]) for feature in features]
        chosen_attention_mask = [jnp.ones_like(input_ids) for input_ids in chosen_input_ids]
        rejected_input_ids = [jnp.array(feature["rejected_input_ids"]) for feature in features]
        rejected_attention_mask = [jnp.ones_like(input_ids) for input_ids in rejected_input_ids]

        pixel_values = None
        pixel_attention_mask = None
        if "pixel_values" in features[0]:
            pixel_values = [jnp.array(feature["pixel_values"]) for feature in features]
        if "pixel_attention_mask" in features[0]:
            pixel_attention_mask = [jnp.array(feature["pixel_attention_mask"]) for feature in features]

        ref_chosen_logps = None
        ref_rejected_logps = None
        if "ref_chosen_logps" in features[0] and "ref_rejected_logps" in features[0]:
            ref_chosen_logps = jnp.array([feature["ref_chosen_logps"] for feature in features])
            ref_rejected_logps = jnp.array([feature["ref_rejected_logps"] for feature in features])

        # Pad sequences
        output = {
            "prompt_input_ids": pad(
                prompt_input_ids,
                self.max_prompt_length,
                padding_value=self.pad_token_id,
                padding_side="left",
            ),
            "prompt_attention_mask": pad(
                prompt_attention_mask,
                self.max_prompt_length,
                padding_value=0,
                padding_side="left",
            ),
            "chosen_input_ids": pad(chosen_input_ids, self.max_completion_length, padding_value=self.pad_token_id),
            "chosen_attention_mask": pad(chosen_attention_mask, self.max_completion_length, padding_value=0),
            "rejected_input_ids": pad(rejected_input_ids, self.max_completion_length, padding_value=self.pad_token_id),
            "rejected_attention_mask": pad(rejected_attention_mask, self.max_completion_length, padding_value=0),
        }
        if pixel_values is not None:
            output["pixel_values"] = pad(pixel_values, self.max_prompt_length, padding_value=0.0)
        if pixel_attention_mask is not None:
            output["pixel_attention_mask"] = pad(pixel_attention_mask, self.max_prompt_length, padding_value=0)
        if "image_sizes" in features[0]:
            output["image_sizes"] = jnp.array([feature["image_sizes"] for feature in features])
        if ref_chosen_logps is not None and ref_rejected_logps is not None:
            output["ref_chosen_logps"] = ref_chosen_logps
            output["ref_rejected_logps"] = ref_rejected_logps
        return output


@auto_pytree
class DataCollatorForPreferenceGrain:
    r"""DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch."""

    max_prompt_length: int
    max_completion_length: int
    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: bool | None = False

    def __call__(self, features: dict[str, tp.Any]) -> dict[str, tp.Any]:
        prompt_input_ids = np.array(features["prompt_input_ids"])
        prompt_attention_mask = np.ones_like(prompt_input_ids)
        chosen_input_ids = np.array(features["chosen_input_ids"])
        chosen_attention_mask = np.ones_like(chosen_input_ids)
        rejected_input_ids = np.array(features["rejected_input_ids"])
        rejected_attention_mask = np.ones_like(rejected_input_ids)
        pixel_values = None
        pixel_attention_mask = None
        if "pixel_values" in features.keys():
            pixel_values = np.array(features["pixel_values"])
        if "pixel_attention_mask" in features.keys():
            pixel_attention_mask = np.array(features["pixel_attention_mask"])

        ref_chosen_logps = None
        ref_rejected_logps = None
        if "ref_chosen_logps" in features.keys() and "ref_rejected_logps" in features.keys():
            ref_chosen_logps = np.array(features["ref_chosen_logps"])
            ref_rejected_logps = np.array(features["ref_rejected_logps"])

        # Pad sequences
        output = {
            "prompt_input_ids": pad_single(
                prompt_input_ids,
                self.max_prompt_length,
                padding_value=self.pad_token_id,
                padding_side="left",
            ),
            "prompt_attention_mask": pad_single(
                prompt_attention_mask,
                self.max_prompt_length,
                padding_value=0,
                padding_side="left",
            ),
            "chosen_input_ids": pad_single(
                chosen_input_ids,
                self.max_completion_length,
                padding_value=self.pad_token_id,
            ),
            "chosen_attention_mask": pad_single(
                chosen_attention_mask,
                self.max_completion_length,
                padding_value=0,
            ),
            "rejected_input_ids": pad_single(
                rejected_input_ids,
                self.max_completion_length,
                padding_value=self.pad_token_id,
            ),
            "rejected_attention_mask": pad_single(
                rejected_attention_mask,
                self.max_completion_length,
                padding_value=0,
            ),
        }

        # Add optional outputs
        if pixel_values is not None:
            output["pixel_values"] = pad_single(pixel_values, self.max_prompt_length, padding_value=0.0)
        if pixel_attention_mask is not None:
            output["pixel_attention_mask"] = pad_single(pixel_attention_mask, self.max_prompt_length, padding_value=0)
        if "image_sizes" in features.keys():
            output["image_sizes"] = np.array(features["image_sizes"])
        if ref_chosen_logps is not None and ref_rejected_logps is not None:
            output["ref_chosen_logps"] = ref_chosen_logps
            output["ref_rejected_logps"] = ref_rejected_logps
        return output


@auto_pytree
class DPODataCollatorWithPaddingTFDS:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    """

    max_prompt_length: int
    max_completion_length: int
    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: bool | None = False
    output_arrays_only: bool = True
    prepadded: bool = True

    def __call__(self, features: list[dict[str, tp.Any]]) -> dict[str, tp.Any]:
        camax_length = self.max_completion_length + self.max_prompt_length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith(("_input_ids", "_attention_mask", "_labels", "_pixel_values")):
                match k.split("_")[0]:
                    case "rejected":
                        max_length = self.max_completion_length
                    case "chosen":
                        max_length = self.max_completion_length
                    case "prompt":
                        max_length = self.max_prompt_length
                    case _:
                        max_length = camax_length

                if self.is_encoder_decoder:
                    to_pad = [jnp.array(ex[k], dtype="i4") for ex in features]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` "
                                "(e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(
                        to_pad,
                        batch_first=False,
                        padding_value=padding_value,
                        max_len=None if self.prepadded else max_length,
                    )
                else:
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` "
                                "(e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.endswith("_pixel_values"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    if k in ["prompt_input_ids", "prompt_attention_mask"]:
                        padding_side = "left"
                    else:
                        padding_side = "right"
                    if k.endswith("_pixel_values"):
                        dtype = jnp.float32
                    else:
                        dtype = jnp.int32

                    to_pad = [jnp.array(ex[k], dtype=dtype) for ex in features]

                    padded_batch[k] = pad(
                        to_pad,
                        None if self.prepadded else max_length,
                        padding_value=padding_value,
                        padding_side=padding_side,
                    )
            elif k.endswith("_logps"):
                padded_batch[k] = jnp.array([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]
            if self.output_arrays_only:
                val = padded_batch.get(k)
                if hasattr(val, "dtype"):
                    if val.dtype not in [jnp.float64, jnp.float32, jnp.float16, jnp.int32, jnp.int16, jnp.int8]:
                        padded_batch.pop(k)
                else:
                    padded_batch.pop(k)
        return padded_batch


@auto_pytree
class DPODataCollatorWithPaddingGrain:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    """

    max_prompt_length: int
    max_completion_length: int
    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: bool | None = False
    output_arrays_only: bool = True
    prepadded: bool = True

    def __call__(self, features: dict[str, tp.Any]) -> dict[str, tp.Any]:
        camax_length = self.max_completion_length + self.max_prompt_length
        padded_batch = {}
        for k in features.keys():
            if k.endswith(("_input_ids", "_attention_mask", "_labels", "_pixel_values")):
                match k.split("_")[0]:
                    case "rejected":
                        max_length = self.max_completion_length
                    case "chosen":
                        max_length = self.max_completion_length
                    case "prompt":
                        max_length = self.max_prompt_length
                    case _:
                        max_length = camax_length

                if self.is_encoder_decoder:
                    to_pad = np.array(features[k], dtype="i4")

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` "
                                "(e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(
                        to_pad,
                        batch_first=False,
                        padding_value=padding_value,
                        max_len=None if self.prepadded else max_length,
                    )
                else:
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` "
                                "(e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.endswith("_pixel_values"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    if k in ["prompt_input_ids", "prompt_attention_mask"]:
                        padding_side = "left"
                    else:
                        padding_side = "right"
                    if k.endswith("_pixel_values"):
                        dtype = np.float32
                    else:
                        dtype = np.int32

                    to_pad = np.array(features[k], dtype=dtype)

                    padded_batch[k] = pad_single(
                        to_pad,
                        None if self.prepadded else max_length,
                        padding_value=padding_value,
                        padding_side=padding_side,
                    )
            elif k.endswith("_logps"):
                padded_batch[k] = np.array(features[k])
            else:
                padded_batch[k] = features[k]
            if self.output_arrays_only:
                val = padded_batch.get(k)
                if hasattr(val, "dtype"):
                    if val.dtype not in [np.float64, np.float32, np.float16, np.int32, np.int16, np.int8]:
                        padded_batch.pop(k)
                else:
                    padded_batch.pop(k)
        return padded_batch


class HFDataSource(pygrain.RandomAccessDataSource):
    """A Grain DataSource for Hugging Face IterableDatasets."""

    def __init__(
        self,
        dataset: IterableDataset,
        shard_options: pygrain.ShardOptions,
        num_threads: int = 1,
    ):
        self.dataset = dataset
        self.shard_options = shard_options
        self.num_threads = num_threads

        if not hasattr(dataset, "n_shards"):
            if dataset.n_shards < self.shard_options.shard_count:
                warnings.warn(
                    f"Dataset has {getattr(dataset, 'n_shards', 1)} shards, but {self.shard_options.shard_count} "
                    "are expected by the dataloader. This may lead to inefficient loading or errors.",
                    stacklevel=1,
                )
            self.n_shards = dataset.n_shards
        else:
            self.n_shards = 1
        self.dataset_shards = [(self.shard_options.shard_index * self.num_threads) + i for i in range(self.num_threads)]
        self.datasets = [
            split_dataset_by_node(dataset, world_size=self.n_shards, rank=shard_rank)
            for shard_rank in self.dataset_shards
        ]
        self.data_iters = []

    def __len__(self):
        return 10_000_000_000

    def __getitem__(self, index):
        if not self.data_iters:
            self.data_iters = [iter(ds) for ds in self.datasets]
        thread_id_str = current_thread().name.split("_")[-1]
        if not thread_id_str.isdigit():
            worker_idx = 0
        else:
            worker_idx = int(thread_id_str) % self.num_threads
        try:
            return next(self.data_iters[worker_idx])
        except StopIteration as e:
            raise IndexError(f"Iterator for worker {worker_idx} is exhausted.") from e


@dataclass
class CollateMapTransform(pygrain.MapTransform):
    """A Grain MapTransform to apply a user-defined collation function."""

    collate_fn: callable

    def map(self, element):
        return self.collate_fn(element)


@dataclass
class ToNumpy(pygrain.MapTransform):
    """A Grain MapTransform to apply a user-defined collation function."""

    def map(self, element):
        for name, value in element.items():
            element[name] = np.asarray(value)
        return element


def shift_and_pad(mask, *tensors):
    for i in range(mask.shape[0]):
        first_one_idx = np.nonzero(mask[i])[0][0].item()
        mask[i] = np.roll(mask[i], shift=-first_one_idx)
        for tensor in tensors:
            tensor[i] = np.roll(tensor[i], shift=-first_one_idx)

    if not tensors:
        return mask
    else:
        return mask, *tensors


def pad(
    tensors: list[jnp.ndarray],
    max_lenght: int | None,
    padding_value: int = 0,
    padding_side: str = "right",
) -> jnp.ndarray:
    """
    Pads a list of tensors to the same shape along the first dimension.
    """
    output_shape = tensors[0].shape[:-1]
    current_max = tensors[0].shape[-1]
    if max_lenght is None:
        max_lenght = current_max
    x_lenght = max(current_max, max_lenght)
    output_shape += (x_lenght,)
    output = jnp.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (i, seq_slice, *tuple(slice(0, s) for s in t.shape[1:]))
        output = output.at[slices].set(t)

    if padding_side == "left":
        output = output[..., -max_lenght:]
    elif padding_side == "right":
        output = output[..., :max_lenght]
    else:
        raise ValueError("padding_side must be 'left' or 'right'")
    return output


def pad_single(
    tensor: np.ndarray,
    max_length: int | None = None,
    padding_value: int = 0,
    padding_side: str = "right",
) -> np.ndarray:
    """
    Pads a single tensor to a specified length along the last dimension.
    """
    current_length = tensor.shape[-1]

    if max_length is None:
        return tensor

    if current_length >= max_length:
        if padding_side == "left":
            return tensor[..., -max_length:]
        else:
            return tensor[..., :max_length]

    pad_amount = max_length - current_length

    if padding_side == "left":
        pad_config = [(0, 0)] * (tensor.ndim - 1) + [(pad_amount, 0)]
    elif padding_side == "right":
        pad_config = [(0, 0)] * (tensor.ndim - 1) + [(0, pad_amount)]
    else:
        raise ValueError("padding_side must be 'left' or 'right'")

    return np.pad(tensor, pad_config, mode="constant", constant_values=padding_value)


def np_pad(
    tensors: list[np.ndarray],
    max_lenght: int | None,
    padding_value: int = 0,
    padding_side: str = "right",
) -> np.ndarray:
    """
    Pads a list of tensors to the same shape along the first dimension.
    """
    output_shape = tensors[0].shape[:-1]
    current_max = tensors[0].shape[-1]
    if max_lenght is None:
        max_lenght = current_max
    x_lenght = max(current_max, max_lenght)
    output_shape += (x_lenght,)
    output = np.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (i, seq_slice, *tuple(slice(0, s) for s in t.shape[1:]))
        output[slices] = t

    if padding_side == "left":
        output = output[..., -max_lenght:]
    elif padding_side == "right":
        output = output[..., :max_lenght]
    else:
        raise ValueError("padding_side must be 'left' or 'right'")
    return output


def pad_to_length(
    tensor: chex.Array,
    length: int,
    pad_value: int | float,
    axis: int = -1,
) -> chex.Array:
    if tensor.shape[axis] >= length:
        if tensor.ndim == 2:
            tensor = tensor[:, :length]
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[axis] = length - tensor.shape[axis]
        return jax.numpy.concatenate(
            [
                tensor,
                pad_value * jax.numpy.ones(pad_size, dtype=tensor.dtype),
            ],
            axis=axis,
        )


def pad_sequence(
    sequences,
    batch_first=False,
    padding_value=0,
    max_len: int | None = None,
):
    max_len = max(seq.shape[-1] for seq in sequences) if max_len is None else max_len
    padding_value = jnp.array(padding_value).reshape(1)
    if batch_first:
        padded_seqs = [
            (
                jnp.concatenate(
                    [
                        seq.reshape(1, -1),
                        jnp.ones((1, max_len - seq.shape[-1])) * padding_value,
                    ],
                    axis=1,
                )
                if seq.shape[-1] < max_len
                else seq.reshape(1, -1)
            )
            for seq in sequences
        ]
    else:
        padded_seqs = [
            (
                jnp.concatenate(
                    [
                        jnp.ones((1, max_len - seq.shape[-1])) * padding_value,
                        seq.reshape(1, -1),
                    ],
                    axis=1,
                )
                if seq.shape[-1] < max_len
                else seq.reshape(1, -1)
            )
            for seq in sequences
        ]

    return jnp.array(padded_seqs)


@contextmanager
def leave_alone_context_manager():
    # Perform setup actions (none in this case)
    yield


def conversations_formatting_function(
    processing_class: "AutoTokenizer",  # type:ignore #noqa
    messages_field: tp.Literal["messages", "conversations"],
    tools: list | None = None,
):
    r"""
    return a callable function that takes in a "messages"
    dataset and returns a formatted dataset, based on the processing_class
    apply chat template to the dataset
    """

    def format_dataset(examples):
        if isinstance(examples[messages_field][0], list):
            output_texts = []
            for i in range(len(examples[messages_field])):
                output_texts.append(
                    processing_class.apply_chat_template(
                        examples[messages_field][i],
                        tokenize=False,
                        tools=tools,
                    )
                )
            return output_texts
        else:
            return processing_class.apply_chat_template(
                examples[messages_field],
                tokenize=False,
                tools=tools,
            )

    return format_dataset


def instructions_formatting_function(processing_class: "AutoTokenizer"):  # type:ignore #noqa
    r"""from TRL
    return a callable function that takes in an "instructions" dataset and returns a
    formatted dataset, based on the processing_class apply chat template to the dataset
    """

    def format_dataset(examples):
        if isinstance(examples["prompt"], list):
            output_texts = []
            for i in range(len(examples["prompt"])):
                converted_sample = [
                    {"role": "user", "content": examples["prompt"][i]},
                    {"role": "assistant", "content": examples["completion"][i]},
                ]
                output_texts.append(processing_class.apply_chat_template(converted_sample, tokenize=False))  # type: ignore
            return output_texts
        else:
            converted_sample = [
                {"role": "user", "content": examples["prompt"]},
                {"role": "assistant", "content": examples["completion"]},
            ]
            return processing_class.apply_chat_template(converted_sample, tokenize=False)  # type: ignore

    return format_dataset


def get_formatting_func_from_dataset(
    dataset: tp.Union["Dataset", "ConstantLengthDataset"],  # type: ignore # noqa
    processing_class: "AutoTokenizer",  # type:ignore #noqa
    tools: list | None = None,
) -> tp.Callable | None:
    try:
        from datasets import Dataset, Value
    except ImportError as e:
        raise ImportError("Please install the datasets library to use this function.") from e
    FORMAT_MAPPING = {
        "chatml": [
            {
                "content": Value(dtype="string", id=None),
                "role": Value(dtype="string", id=None),
            }
        ],
        "instruction": {
            "completion": Value(dtype="string", id=None),
            "prompt": Value(dtype="string", id=None),
        },
    }

    if isinstance(dataset, Dataset):
        if "messages" in dataset.features:
            if dataset.features["messages"] == FORMAT_MAPPING["chatml"]:
                logging.info("Formatting dataset with chatml format")
                return conversations_formatting_function(
                    processing_class,
                    "messages",
                    tools,
                )
        if "conversations" in dataset.features:
            if dataset.features["conversations"] == FORMAT_MAPPING["chatml"]:
                logging.info("Formatting dataset with chatml format")
                return conversations_formatting_function(
                    processing_class,
                    "conversations",
                    tools,
                )
        elif dataset.features == FORMAT_MAPPING["instruction"]:
            logging.info("Formatting dataset with instruction format")
            return instructions_formatting_function(processing_class)

    return None


def add_bos_token_if_needed(
    bos_token_id: int | None,
    prompt_len_input_ids: int,
    prompt_tokens: dict[str, list[int]],
    chosen_prompt_len_input_ids: int,
    chosen_tokens: dict[str, list[int]],
    rejected_prompt_len_input_ids: int,
    rejected_tokens: dict[str, list[int]],
):
    if bos_token_id is not None:
        if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
            prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
            chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
            rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]
    return prompt_tokens, chosen_tokens, rejected_tokens


def add_eos_token_if_needed(
    eos_token_id: int,
    chosen_tokens: dict[str, list[int]],
    rejected_tokens: dict[str, list[int]],
):
    if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
        chosen_tokens["input_ids"].append(eos_token_id)
        chosen_tokens["attention_mask"].append(1)
    if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
        rejected_tokens["input_ids"].append(eos_token_id)
        rejected_tokens["attention_mask"].append(1)
    return chosen_tokens, rejected_tokens


def first_true_indices(bools, dtype=jnp.int32):
    """
    Takes an N-dimensional bool array and returns an (N-1)-dimensional array of integers giving
    the position of the first True in each "row".
    """
    row_len = bools.shape[-1]
    zero_or_index = row_len * (~bools).astype(dtype) + jnp.arange(row_len, dtype=dtype)
    return jnp.min(zero_or_index, axis=-1)


def truncate_right(input_ids, stop_token_id, pad_token_id):
    """
    Truncates the input array from the right side after the first occurrence of the stop token.
    """
    trunc_idxs = first_true_indices(input_ids == stop_token_id).reshape((-1, 1))
    idxs = jnp.arange(input_ids.shape[1]).reshape((1, -1))
    output_ids = jnp.where(idxs > trunc_idxs, pad_token_id, input_ids)
    mask = jnp.where(idxs > trunc_idxs, 0, 1)
    return output_ids, mask
