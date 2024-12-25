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
import inspect
import typing as tp
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import chex
import jax
from jax import numpy as jnp

from easydel.infra.utils import ProcessingClassType
from easydel.modules import EasyDeLBaseModule

from ..utils import add_bos_token_if_needed, add_eos_token_if_needed
from .dpo_config import DPOConfig


@dataclass
class DPODataCollatorWithPadding:
	r"""DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.

	Args:
			pad_token_id: int: The tokenizers pad_token_id.
			label_pad_token_id: int: The label used for masking.
			is_encoder_decoder: tp.Optional[bool]: Whether you model has an
					encoder_decoder architecture
	"""

	max_prompt_length: int
	max_completion_length: int
	pad_token_id: int = 0
	label_pad_token_id: int = -100
	is_encoder_decoder: tp.Optional[bool] = False
	ids_to_pop_from_dataset: tp.Optional[dict] = None
	auto_fix_data: bool = True

	def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]) -> tp.Dict[str, tp.Any]:
		padded_batch = {}
		for k in features[0].keys():
			if (
				k.endswith("_input_ids")
				or k.endswith("_attention_mask")
				or k.endswith("_labels")
			):
				if self.is_encoder_decoder:
					to_pad = [jnp.array(ex[k], dtype="i4") for ex in features]

					if (k.startswith("prompt")) and (k.endswith("input_ids")):
						padding_value = self.pad_token_id
					elif k.endswith("_attention_mask"):
						padding_value = 0
					elif (
						(k.startswith("chosen")) or (k.startswith("rejected")) or ("decoder" in k)
					):
						padding_value = self.label_pad_token_id
					else:
						raise ValueError(f"Unexpected key in batch '{k}'")
					padded_batch[k] = pad_sequence(
						to_pad, batch_first=True, padding_value=padding_value
					).astype("i4")
				else:
					if "prompt" in k:
						to_pad = [jnp.array(ex[k][::-1], dtype="i4") for ex in features]
					else:
						to_pad = [jnp.array(ex[k], dtype="i4") for ex in features]
					if k.endswith("_input_ids"):
						padding_value = self.pad_token_id
					elif k.endswith("_labels"):
						padding_value = self.label_pad_token_id
					elif k.endswith("_attention_mask"):
						padding_value = 0
					else:
						raise ValueError(f"Unexpected key in batch '{k}'")
					padded_batch[k] = pad_sequence(
						to_pad, batch_first=True, padding_value=padding_value
					).astype("i4")
					if "prompt" in k:
						padded_batch[k] = jnp.flip(padded_batch[k], axis=[1])
			elif k.endswith("_logps"):
				padded_batch[k] = jnp.array([ex[k] for ex in features])
			else:
				padded_batch[k] = [ex[k] for ex in features]
		if self.ids_to_pop_from_dataset:
			for key in self.ids_to_pop_from_dataset:
				_ = padded_batch.pop(key, None)
		for key in list(padded_batch.keys()):
			if not (
				key.endswith("_input_ids")
				or key.endswith("_attention_mask")
				or key.endswith("_labels")
				or key.endswith("_log_probs")
			):
				_ = padded_batch.pop(key, None)
		for k in list(padded_batch.keys()):
			v = padded_batch[k]
			padded_batch[k] = v.reshape(v.shape[0], -1)
		if self.auto_fix_data:
			padded_batch["rejected_input_ids"] = padded_batch["rejected_input_ids"][
				..., : self.max_completion_length
			]
			padded_batch["rejected_attention_mask"] = padded_batch["rejected_attention_mask"][
				..., : self.max_completion_length
			]
			padded_batch["rejected_labels"] = padded_batch["rejected_labels"][
				..., : self.max_completion_length
			]

			padded_batch["chosen_input_ids"] = padded_batch["chosen_input_ids"][
				..., : self.max_completion_length
			]
			padded_batch["chosen_attention_mask"] = padded_batch["chosen_attention_mask"][
				..., : self.max_completion_length
			]
			padded_batch["chosen_labels"] = padded_batch["chosen_labels"][
				..., : self.max_completion_length
			]

			padded_batch["prompt_input_ids"] = padded_batch["prompt_input_ids"][
				..., : self.max_prompt_length
			]
			padded_batch["prompt_attention_mask"] = padded_batch["prompt_attention_mask"][
				..., : self.max_prompt_length
			]
		return {k: jnp.array(v) for k, v in padded_batch.items()}


def pad_to_length(
	tensor: chex.Array,
	length: int,
	pad_value: tp.Union[int, float],
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


def build_tokenize(
	model: tp.Optional[EasyDeLBaseModule] = None,
	args: tp.Optional[DPOConfig] = None,
):
	def _tokenize(
		features: tp.Dict[str, tp.List],
		processing_class: ProcessingClassType,
		processor: tp.Optional[tp.Callable] = None,
	) -> tp.Dict[str, tp.List]:
		"""
		Tokenizes and processes a batch of input features using the provided processing_class and processor.
		"""
		batch = defaultdict(list)

		if model is None:
			prompt = features["prompt"]
			images = features.get("images", [None] * len(features["prompt"]))

			prompt_tokens = _process_prompt(
				prompt,
				processor,
				processing_class,
				images,
			)
			chosen_tokens = _process_answer(
				prompt,
				features["chosen"],
				processor,
				processing_class,
				images,
			)
			rejected_tokens = _process_answer(
				prompt,
				features["rejected"],
				processor,
				processing_class,
				images,
			)

			prompt_len_input_ids = _adjust_prompt_length(
				prompt_tokens,
				chosen_tokens,
				rejected_tokens,
			)

			prompt_tokens, chosen_tokens, rejected_tokens = _add_special_tokens(
				processing_class,
				prompt_len_input_ids,
				prompt_tokens,
				chosen_tokens,
				rejected_tokens,
			)

			_truncate_tokens(chosen_tokens, rejected_tokens, prompt_tokens, args)
			_build_sequence_tokens(batch, chosen_tokens, args, "chosen")
			_build_sequence_tokens(batch, rejected_tokens, args, "rejected")
			_append_prompt_tokens_to_batch(batch, prompt_tokens, args)

		else:
			_tokenize_encoder_decoder(
				batch,
				processing_class,
				features["prompt"],
				features["chosen"],
				features["rejected"],
				args,
			)

		return dict(batch)

	return _tokenize


def _process_prompt(
	prompts: tp.List[str],
	processor: tp.Optional[tp.Callable],
	processing_class: ProcessingClassType,
	images: tp.List[tp.Optional[tp.Any]],
) -> tp.List[tp.Dict[str, tp.List[int]]]:
	"""
	Processes a list of prompts by tokenizing them, optionally using a processor for additional processing.
	"""
	if processor:
		processor_kwargs = (
			{"add_special_tokens": False}
			if "add_special_tokens" in inspect.signature(processor).parameters
			else {}
		)
		prompt_tokens = []
		for prompt, image in zip(prompts, images):
			tokens = processor(images=image, text=prompt, **processor_kwargs)
			tokens = {k: v[0] for k, v in tokens.items()}
			if not isinstance(tokens["input_ids"], list):
				tokens["input_ids"] = tokens["input_ids"].tolist()
				tokens["attention_mask"] = tokens["attention_mask"].tolist()
			prompt_tokens.append(tokens)
	else:
		prompt_tokens = [
			processing_class(prompt, add_special_tokens=False) for prompt in prompts
		]
	return [{f"prompt_{k}": v for k, v in tokens.items()} for tokens in prompt_tokens]


def _process_answer(
	prompts: tp.List[str],
	answers: tp.List[str],
	processor: tp.Optional[tp.Callable],
	processing_class: ProcessingClassType,
	images: tp.List[tp.Optional[tp.Any]],
) -> tp.List[tp.Dict[str, tp.Any]]:
	return [
		_build_tokenized_answer(
			prompt,
			answer,
			image,
			processor=processor,
			processing_class=processing_class,
		)
		for prompt, answer, image in zip(prompts, answers, images)
	]


def _adjust_prompt_length(
	prompt_tokens: tp.List[tp.Dict[str, tp.List[int]]],
	chosen_tokens: tp.List[tp.Dict[str, tp.List[int]]],
	rejected_tokens: tp.List[tp.Dict[str, tp.List[int]]],
) -> tp.List[int]:
	prompt_len_input_ids = []
	for p_tokens, c_tokens, r_tokens in zip(
		prompt_tokens, chosen_tokens, rejected_tokens
	):
		c_len = len(c_tokens["prompt_input_ids"])
		r_len = len(r_tokens["prompt_input_ids"])
		min_len = min(c_len, r_len)

		for k, v in p_tokens.items():
			p_tokens[k] = v[:min_len]

		num_diff_tokens = sum(
			[
				a != b
				for a, b in zip(c_tokens["prompt_input_ids"], r_tokens["prompt_input_ids"])
			]
		)
		num_diff_len = abs(c_len - r_len)
		if num_diff_tokens > 1 or num_diff_len > 1:
			raise ValueError(
				"Chosen and rejected prompt_input_ids might only differ on the last token due to processing_class merge ops."
			)
		prompt_len_input_ids.append(min_len)
	return prompt_len_input_ids


def _add_special_tokens(
	processing_class: ProcessingClassType,
	prompt_len_input_ids: tp.List[int],
	prompt_tokens: tp.List[tp.Dict[str, tp.List[int]]],
	chosen_tokens: tp.List[tp.Dict[str, tp.List[int]]],
	rejected_tokens: tp.List[tp.Dict[str, tp.List[int]]],
) -> tp.Tuple[
	tp.List[tp.Dict[str, tp.List[int]]],
	tp.List[tp.Dict[str, tp.List[int]]],
	tp.List[tp.Dict[str, tp.List[int]]],
]:
	for i in range(len(prompt_tokens)):
		prompt_tokens[i], chosen_tokens[i], rejected_tokens[i] = add_bos_token_if_needed(
			processing_class.bos_token_id,
			prompt_len_input_ids[i],
			prompt_tokens[i],
			len(chosen_tokens[i]["prompt_input_ids"]),
			chosen_tokens[i],
			len(rejected_tokens[i]["prompt_input_ids"]),
			rejected_tokens[i],
		)

		chosen_tokens[i], rejected_tokens[i] = add_eos_token_if_needed(
			processing_class.eos_token_id, chosen_tokens[i], rejected_tokens[i]
		)
	return prompt_tokens, chosen_tokens, rejected_tokens


def _truncate_tokens(
	chosen_tokens: tp.List[tp.Dict[str, tp.List[int]]],
	rejected_tokens: tp.List[tp.Dict[str, tp.List[int]]],
	prompt_tokens: tp.List[tp.Dict[str, tp.List[int]]],
	args: DPOConfig,
) -> None:
	"""
	Truncates the tokens in chosen, rejected, and prompt sequences to ensure they fit within the maximum length constraints.
	"""
	if args.truncation_mode not in ["keep_start", "keep_end"]:
		raise ValueError(f"Invalid truncation mode: {args.truncation_mode}")

	for c_tokens, r_tokens, p_tokens in zip(
		chosen_tokens, rejected_tokens, (prompt_tokens)
	):
		longer_response_length = max(len(c_tokens["input_ids"]), len(r_tokens["input_ids"]))

		# if combined sequence is too long, truncate the prompt
		for answer_tokens in [c_tokens, r_tokens, p_tokens]:
			if (
				len(answer_tokens["prompt_input_ids"]) + longer_response_length
				> args.max_length
			):
				if args.truncation_mode == "keep_start":
					for k in ["prompt_input_ids", "prompt_attention_mask"]:
						answer_tokens[k] = answer_tokens[k][: args.max_prompt_length]
				elif args.truncation_mode == "keep_end":
					for k in ["prompt_input_ids", "prompt_attention_mask"]:
						answer_tokens[k] = answer_tokens[k][-args.max_prompt_length :]

		for answer_tokens in [c_tokens, r_tokens]:
			if (
				len(answer_tokens["prompt_input_ids"]) + longer_response_length
				> args.max_length
			):
				for k in ["input_ids", "attention_mask"]:
					answer_tokens[k] = answer_tokens[k][
						: args.max_length - args.max_prompt_length
					]


def _build_sequence_tokens(
	batch: tp.Dict[str, tp.List[int]],
	tokens: tp.List[tp.Dict[str, tp.List[int]]],
	args: DPOConfig,
	prefix: str,
) -> None:
	for token in tokens:
		sequence_tokens = {
			f"{prefix}_{k}": token[f"prompt_{k}"] + token[k]
			for k in ["input_ids", "attention_mask"]
		}
		sequence_tokens[f"{prefix}_labels"] = sequence_tokens[f"{prefix}_input_ids"][:]
		sequence_tokens[f"{prefix}_labels"][: len(token["prompt_input_ids"])] = [
			args.label_pad_token_id
		] * len(token["prompt_input_ids"])
		for k, v in sequence_tokens.items():
			tobe_added = args.max_completion_length - len(v)
			if tobe_added > 0:
				if k.endswith("attention_mask"):
					v = v + ([0] * tobe_added)
				elif k.endswith("input_ids") or k.endswith("labels"):
					v = v + ([args.padding_value] * tobe_added)
			batch[k].append(v)


def _append_prompt_tokens_to_batch(
	batch: tp.Dict[str, tp.List[int]],
	prompt_tokens: tp.List[tp.Dict[str, tp.List[int]]],
	args: DPOConfig,
) -> None:
	for p_tokens in prompt_tokens:
		for k, v in p_tokens.items():
			tobe_added = args.max_completion_length - len(v)
			if tobe_added > 0:
				if k.endswith("attention_mask"):
					v = v + ([0] * tobe_added)
				elif k.endswith("input_ids") or k.endswith("labels"):
					v = v + ([args.padding_value] * tobe_added)
			batch[k].append(v)


def _tokenize_encoder_decoder(
	batch: tp.Dict[str, tp.List[int]],
	processing_class: ProcessingClassType,
	prompt: tp.List[str],
	chosen: tp.List[str],
	rejected: tp.List[str],
	args: DPOConfig,
) -> None:
	chosen_tokens = processing_class(
		chosen,
		truncation=True,
		max_length=args.max_completion_length,
		padding="max_lenght",
		add_special_tokens=True,
	)
	rejected_tokens = processing_class(
		rejected,
		truncation=True,
		max_length=args.max_completion_length,
		padding="max_lenght",
		add_special_tokens=True,
	)
	prompt_tokens = processing_class(
		prompt,
		truncation=True,
		max_length=args.max_prompt_length,
		add_special_tokens=True,
	)

	batch["chosen_labels"] = chosen_tokens["input_ids"]
	batch["rejected_labels"] = rejected_tokens["input_ids"]
	batch["prompt_input_ids"] = prompt_tokens["input_ids"]
	batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]


def _build_tokenized_answer(
	prompt: str,
	answer: str,
	images: tp.Optional[tp.List[tp.Any]] = None,
	processor: tp.Optional[tp.Callable] = None,
	processing_class: tp.Optional[ProcessingClassType] = None,
) -> tp.Dict[str, tp.Any]:
	"""
	Build tokenized response, handling vision models and different tokenizers.
	"""

	def tokenize(text, images=None):
		if processor:
			processor_kwargs = (
				{"add_special_tokens": False}
				if "add_special_tokens" in inspect.signature(processor).parameters
				else {}
			)
			tokenized = processor(images=images, text=text, **processor_kwargs)
			tokenized = {k: v[0] for k, v in tokenized.items()}
			if not isinstance(tokenized["input_ids"], list):
				tokenized["input_ids"] = tokenized["input_ids"].tolist()
				tokenized["attention_mask"] = tokenized["attention_mask"].tolist()
		else:
			tokenized = processing_class(text, add_special_tokens=False)
		return tokenized

	full_tokenized = tokenize(prompt + answer, images)
	prompt_tokenized = tokenize(prompt, images)

	prompt_input_ids = prompt_tokenized["input_ids"]
	answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
	answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

	if len(full_tokenized["input_ids"]) != len(prompt_input_ids + answer_input_ids):
		raise ValueError(
			"Prompt input ids and answer input ids should have the same length."
		)

	response_token_ids_start_idx = len(prompt_input_ids)

	if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
		response_token_ids_start_idx -= 1

	prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
	prompt_attention_mask = full_tokenized["attention_mask"][
		:response_token_ids_start_idx
	]

	if len(prompt_input_ids) != len(prompt_attention_mask):
		raise ValueError("Prompt input ids and attention mask should have the same length.")

	return_dict = {
		"prompt_input_ids": prompt_input_ids,
		"prompt_attention_mask": prompt_attention_mask,
		"input_ids": answer_input_ids,
		"attention_mask": answer_attention_mask,
	}
	if "pixel_values" in full_tokenized:
		return_dict["prompt_pixel_values"] = full_tokenized["pixel_values"]
	if "pixel_attention_mask" in full_tokenized:
		return_dict["prompt_pixel_attention_mask"] = full_tokenized["pixel_attention_mask"]

	return return_dict
