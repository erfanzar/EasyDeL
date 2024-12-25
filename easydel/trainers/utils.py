# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import jax
import numpy as np
from jax import numpy as jnp
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder

from easydel.etils.etils import get_logger

logger = get_logger(__name__)


class JaxDistributedConfig(object):
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
	def initialize(cls, config):
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


# fmt:off
def create_prompt_creator(processing_class):
	def to_role_and_content(field):
		return {
			"conversation": [
				{"role": "user", "content": field["conversation"][0]["input"]},
				{"role": "assistant", "content": field["conversation"][0]["output"]},
			]
		}
	def _pc(sample):
		return conversations_formatting_function(processing_class, messages_field="conversation")(to_role_and_content(sample))
	return _pc
# fmt:on


def create_constant_length_dataset(
	processing_class,
	dataset,
	dataset_text_field: tp.Optional[str] = None,
	formatting_func: tp.Optional[tp.Callable] = None,
	infinite: bool = False,
	seq_length: int = 1024,
	num_of_sequences: int = 1024,
	chars_per_token: float = 3.6,
	eos_token_id: int = 0,
	shuffle: bool = True,
	append_concat_token: bool = True,
	add_special_tokens: bool = True,
) -> tp.Callable[[], tp.Iterator[tp.Dict[str, jnp.ndarray]]]:
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
			f"corresponds to {eos_token_id}. If this is not the correct EOS token, make sure to pass the correct eos_token_id.",
			stacklevel=1,
		)

	concat_token_id = (
		processing_class.eos_token_id if processing_class.eos_token_id else eos_token_id
	)
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
				"The passed formatting_func has more than one argument. Usually that function should have a single argument "
				"`example` which corresponds to the dictionary returned by each element of the dataset. Make sure you know "
				"what you are doing.",
				stacklevel=1,
			)
	elif dataset_text_field is not None:
		formatting_func = lambda x: x[dataset_text_field]  # noqa
	else:
		raise ValueError(
			"Either `dataset_text_field` or `formatting_func` should be provided."
		)

	def constant_length_generator() -> tp.Iterator[tp.Dict[str, jnp.ndarray]]:
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
				buffer,
				add_special_tokens=add_special_tokens,
				truncation=False,
			)
			tokenized_inputs = tokens["input_ids"]
			attention_masks = tokens["attention_mask"]
			# Concatenate all tokens and attention masks
			all_token_ids = []
			all_attention_masks = []
			for tokenized_input, attention_mask in zip(tokenized_inputs, attention_masks):
				if append_concat_token:
					tokenized_input = tokenized_input + [concat_token_id]
					attention_mask = attention_mask + [1]
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
				combined = list(zip(examples, examples_attention_masks))
				random.shuffle(combined)
				examples, examples_attention_masks = zip(*combined)

			# Yield examples
			for example, example_attention_mask in zip(examples, examples_attention_masks):
				yield {
					"input_ids": jnp.asarray(example, dtype="i4"),
					"attention_mask": jnp.asarray(example_attention_mask, dtype="i4"),
				}

	return constant_length_generator


def _collate_batch(
	examples, processing_class, pad_to_multiple_of: tp.Optional[int] = None
):
	if isinstance(examples[0], (list, tuple)):
		examples = [jnp.array(e, dtype=jnp.int64) for e in examples]

	length_of_first = len(examples[0])
	are_tensors_same_length = all(len(x) == length_of_first for x in examples)
	if are_tensors_same_length and (
		pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0
	):
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
		response_template: tp.Union[str, tp.List[int]],
		instruction_template: tp.Optional[tp.Union[str, tp.List[int]]] = None,
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
			self.response_token_ids = self.processing_class.encode(
				self.response_template, add_special_tokens=False
			)
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

	def _whole_word_mask(self, input_tokens: tp.List[str], max_predictions=512):
		from transformers import (
			BertTokenizer,
			BertTokenizerFast,
		)

		if not isinstance(self.processing_class, (BertTokenizer, BertTokenizerFast)):
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
		num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * 0.15))))
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
			raise ValueError(
				"Length of covered_indexes is not equal to length of masked_lms."
			)
		mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
		return mask_labels

	def jax_mask_tokens(
		self, inputs: tp.Any, special_tokens_mask: tp.Optional[tp.Any] = None
	) -> tp.Tuple[tp.Any, tp.Any]:
		"""Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
		labels = np.copy(inputs)
		probability_matrix = np.full(labels.shape, 0.15)
		if special_tokens_mask is None:
			special_tokens_mask = [
				self.processing_class.get_special_tokens_mask(
					val, already_has_special_tokens=True
				)
				for val in labels.tolist()
			]
			special_tokens_mask = np.array(special_tokens_mask, dtype=bool)
		else:
			special_tokens_mask = special_tokens_mask.astype(bool)

		probability_matrix[special_tokens_mask] = 0
		masked_indices = np.random.binomial(
			1, probability_matrix, size=probability_matrix.shape
		).astype(bool)
		labels[~masked_indices] = -100
		indices_replaced = (
			np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
		)
		inputs[indices_replaced] = self.processing_class.mask_token_id
		indices_random = (
			np.random.binomial(1, 0.5, size=labels.shape).astype(bool)
			& masked_indices
			& ~indices_replaced
		)
		random_words = np.random.randint(
			low=0,
			high=len(self.processing_class),
			size=np.count_nonzero(indices_random),
			dtype=np.int64,
		)
		inputs[indices_random] = random_words
		return inputs, labels

	def jax_call(
		self, examples: tp.List[tp.Union[tp.List[int], tp.Any, tp.Dict[str, tp.Any]]]
	) -> tp.Dict[str, tp.Any]:
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

			# For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
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

	def __call__(
		self, examples: tp.List[tp.Union[tp.List[int], tp.Any, tp.Dict[str, tp.Any]]]
	) -> tp.Dict[str, tp.Any]:
		batch = self.jax_call(examples)

		if self.instruction_template is None:
			for i in range(len(examples)):
				response_token_ids_start_idx = None

				for idx in jnp.where(batch["labels"][i] == self.response_token_ids[0])[0]:
					if (
						self.response_token_ids
						== batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
					):
						response_token_ids_start_idx = idx

				if response_token_ids_start_idx is None:
					warnings.warn(
						f"Could not find response key `{self.response_template}` in the "
						f'following instance: {self.processing_class.decode(batch["input_ids"][i])} '
						f"This instance will be ignored in loss calculation. "
						f"Note, if this happens often, consider increasing the `max_seq_length`.",
						stacklevel=1,
					)
					batch["labels"][i, :] = self.ignore_index
				else:
					response_token_ids_end_idx = response_token_ids_start_idx + len(
						self.response_token_ids
					)
					batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

		else:
			for i in range(len(examples)):
				response_token_ids_idxs = []
				human_token_ids_idxs = []

				for assistant_idx in jnp.where(
					batch["labels"][i] == self.response_token_ids[0]
				)[0]:
					if (
						self.response_token_ids
						== batch["labels"][i][
							assistant_idx : assistant_idx + len(self.response_token_ids)
						].tolist()
					):
						response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

				if len(response_token_ids_idxs) == 0:
					warnings.warn(
						f"Could not find response key `{self.response_template}` in the "
						f'following instance: {self.processing_class.decode(batch["input_ids"][i])} '
						f"This instance will be ignored in loss calculation. "
						f"Note, if this happens often, consider increasing the `max_seq_length`.",
						stacklevel=1,
					)
					batch["labels"][i, :] = self.ignore_index

				human_token_ids = self.instruction_token_ids
				for human_idx in jnp.where(batch["labels"][i] == human_token_ids[0])[0]:
					if (
						human_token_ids
						== batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist()
					):
						human_token_ids_idxs.append(human_idx)

				if len(human_token_ids_idxs) == 0:
					warnings.warn(
						f"Could not find instruction key `{self.instruction_template}` in the "
						f'following instance: {self.processing_class.decode(batch["input_ids"][i])} '
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
					human_token_ids_idxs = [0] + human_token_ids_idxs

				for idx, (start, end) in enumerate(
					zip(human_token_ids_idxs, response_token_ids_idxs)
				):
					if idx != 0:
						batch["labels"][i, start:end] = self.ignore_index
					else:
						batch["labels"][i, :end] = self.ignore_index

				if len(response_token_ids_idxs) < len(human_token_ids_idxs):
					batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

		return batch


def conversations_formatting_function(
	processing_class: "AutoTokenizer",  # type:ignore #noqa
	messages_field: tp.Literal["messages", "conversations"],
):
	r"""
	return a callable function that takes in a "messages" dataset and returns a formatted dataset, based on the processing_class
	apply chat template to the dataset
	"""

	def format_dataset(examples):
		if isinstance(examples[messages_field][0], list):
			output_texts = []
			for i in range(len(examples[messages_field])):
				output_texts.append(
					processing_class.apply_chat_template(
						examples[messages_field][i], tokenize=False
					)
				)  # type: ignore
			return output_texts
		else:
			return processing_class.apply_chat_template(
				examples[messages_field], tokenize=False
			)  # type: ignore

	return format_dataset


def instructions_formatting_function(processing_class: "AutoTokenizer"):  # type:ignore #noqa
	r"""from TRL
	return a callable function that takes in an "instructions" dataset and returns a formatted dataset, based on the processing_class
	apply chat template to the dataset
	"""

	def format_dataset(examples):
		if isinstance(examples["prompt"], list):
			output_texts = []
			for i in range(len(examples["prompt"])):
				converted_sample = [
					{"role": "user", "content": examples["prompt"][i]},
					{"role": "assistant", "content": examples["completion"][i]},
				]
				output_texts.append(
					processing_class.apply_chat_template(converted_sample, tokenize=False)
				)  # type: ignore
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
) -> tp.Optional[tp.Callable]:
	from datasets import Dataset, Value

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
				return conversations_formatting_function(processing_class, "messages")
		if "conversations" in dataset.features:
			if dataset.features["conversations"] == FORMAT_MAPPING["chatml"]:
				logging.info("Formatting dataset with chatml format")
				return conversations_formatting_function(processing_class, "conversations")
		elif dataset.features == FORMAT_MAPPING["instruction"]:
			logging.info("Formatting dataset with instruction format")
			return instructions_formatting_function(processing_class)

	return None


def add_bos_token_if_needed(
	bos_token_id: tp.Optional[int],
	prompt_len_input_ids: int,
	prompt_tokens: tp.Dict[str, tp.List[int]],
	chosen_prompt_len_input_ids: int,
	chosen_tokens: tp.Dict[str, tp.List[int]],
	rejected_prompt_len_input_ids: int,
	rejected_tokens: tp.Dict[str, tp.List[int]],
):
	if bos_token_id is not None:
		if (
			prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]
		):
			prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens[
				"prompt_input_ids"
			]
			prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens[
				"prompt_attention_mask"
			]
		if (
			chosen_prompt_len_input_ids == 0
			or bos_token_id != chosen_tokens["prompt_input_ids"][0]
		):
			chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens[
				"prompt_input_ids"
			]
			chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens[
				"prompt_attention_mask"
			]
		if (
			rejected_prompt_len_input_ids == 0
			or bos_token_id != rejected_tokens["prompt_input_ids"][0]
		):
			rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens[
				"prompt_input_ids"
			]
			rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens[
				"prompt_attention_mask"
			]
	return prompt_tokens, chosen_tokens, rejected_tokens


def add_eos_token_if_needed(
	eos_token_id: int,
	chosen_tokens: tp.Dict[str, tp.List[int]],
	rejected_tokens: tp.Dict[str, tp.List[int]],
):
	if (
		len(chosen_tokens["input_ids"]) == 0
		or eos_token_id != chosen_tokens["input_ids"][-1]
	):
		chosen_tokens["input_ids"].append(eos_token_id)
		chosen_tokens["attention_mask"].append(1)
	if (
		len(rejected_tokens["input_ids"]) == 0
		or eos_token_id != rejected_tokens["input_ids"][-1]
	):
		rejected_tokens["input_ids"].append(eos_token_id)
		rejected_tokens["attention_mask"].append(1)
	return chosen_tokens, rejected_tokens


def first_true_indices(bools, dtype=jnp.int32):
	"""
	Takes an N-dimensional bool array and returns an (N-1)-dimensional array of integers giving
	the position of the first True in each "row".

	Returns the length of the rows (bools.shape[-1]) if no element is True in a given row.

	Args:
	    bools (jax.Array):
	        An N-dimensional boolean array.
	    dtype (jnp.dtype, optional):
	        The desired data type of the output array. Defaults to `jnp.int32`.

	Returns:
	    jax.Array:
	        An (N-1)-dimensional array of integers indicating the position of the first True
	        in each row. If no True value is found in a row, returns the length of the row.
	"""
	row_len = bools.shape[-1]
	zero_or_index = row_len * (~bools).astype(dtype) + jnp.arange(row_len, dtype=dtype)
	return jnp.min(zero_or_index, axis=-1)


def truncate_right(input_ids, stop_token_id, pad_token_id):
	"""
	Truncates the input array from the right side after the first occurrence of the stop token.

	Args:
	    input_ids (jax.Array):
	        The array containing the responses to be truncated
	    stop_token_id (int):
	        The token ID representing the stop token where truncation occurs
	    pad_token_id (int):
	        The token ID representing the pad token used to fill the truncated responses

	Returns:
	    tuple:
	        - output_ids (jax.Array):
	            The truncated responses array with pad tokens filled after the stop token
	        - mask (jax.Array):
	            The mask array to indicate the padding tokens
	"""
	trunc_idxs = first_true_indices(input_ids == stop_token_id).reshape((-1, 1))
	idxs = jnp.arange(input_ids.shape[1]).reshape((1, -1))
	output_ids = jnp.where(idxs > trunc_idxs, pad_token_id, input_ids)
	mask = jnp.where(idxs > trunc_idxs, 0, 1)
	return output_ids, mask
