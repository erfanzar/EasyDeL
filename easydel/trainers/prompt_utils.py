# FROM HF TRL

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing as tp

from datasets import Dataset, DatasetDict

from easydel.infra.utils import ProcessingClassType

DatasetType = tp.TypeVar("DatasetType", Dataset, DatasetDict)


def is_conversational(example: dict[str, tp.Any]) -> bool:
	"""
	Check if the example is in a conversational format.
	"""
	supported_keys = ["prompt", "chosen", "rejected", "completion", "messages"]
	example_keys = {key for key in example.keys() if key in supported_keys}

	if example_keys:
		key = example_keys.pop()
		maybe_messages = example[key]
		if isinstance(maybe_messages, list):
			maybe_message = maybe_messages[0]
			if (
				isinstance(maybe_message, dict)
				and "role" in maybe_message
				and "content" in maybe_message
			):
				return True

	return False


def apply_chat_template(
	example: dict[str, list[dict[str, str]]],
	tokenizer: ProcessingClassType,
	tools: tp.Optional[list[tp.Union[dict, tp.Callable]]] = None,
) -> dict[str, str]:
	r"""
	Apply a chat template to a conversational example along with the schema for a list of functions in `tools`.

	For more details, see [`maybe_apply_chat_template`].
	"""
	supported_keys = ["prompt", "chosen", "rejected", "completion", "messages", "label"]
	example_keys = {key for key in example.keys() if key in supported_keys}
	if example_keys not in [
		{"messages"},
		{"prompt"},
		{"prompt", "completion"},
		{"prompt", "chosen", "rejected"},
		{"chosen", "rejected"},
		{"prompt", "completion", "label"},
	]:
		raise KeyError(f"Invalid keys in the example: {example_keys}")

	if "messages" in example:
		messages = tokenizer.apply_chat_template(
			example["messages"], tools=tools, tokenize=False
		)

	if "prompt" in example:
		prompt = tokenizer.apply_chat_template(
			example["prompt"], tools=tools, tokenize=False, add_generation_prompt=True
		)
	if "prompt" in example:
		if "chosen" in example:
			prompt_chosen = tokenizer.apply_chat_template(
				example["prompt"] + example["chosen"],
				tools=tools,
				tokenize=False,
			)
			chosen = prompt_chosen[len(prompt) :]
		if "rejected" in example and "prompt" in example:
			prompt_rejected = tokenizer.apply_chat_template(
				example["prompt"] + example["rejected"],
				tools=tools,
				tokenize=False,
			)
			rejected = prompt_rejected[len(prompt) :]
		if "completion" in example:
			prompt_completion = tokenizer.apply_chat_template(
				example["prompt"] + example["completion"],
				tools=tools,
				tokenize=False,
			)
			completion = prompt_completion[len(prompt) :]
	else:
		if "chosen" in example:
			chosen = tokenizer.apply_chat_template(
				example["chosen"],
				tools=tools,
				tokenize=False,
			)
		if "rejected" in example:
			rejected = tokenizer.apply_chat_template(
				example["rejected"],
				tools=tools,
				tokenize=False,
			)

	if "prompt" in example:
		error_message = (
			"The chat template applied to the prompt + completion does not start with the chat template applied to "
			"the prompt alone."
			"\n**Prompt**:\n{}\n\n**Prompt + Completion**:\n{}"
		)
		if "chosen" in example and not prompt_chosen.startswith(prompt):
			raise ValueError(error_message.format(prompt, prompt_chosen))
		if "rejected" in example and not prompt_rejected.startswith(prompt):
			raise ValueError(error_message.format(prompt, prompt_rejected))
		if "completion" in example and not prompt_completion.startswith(prompt):
			raise ValueError(error_message.format(prompt, prompt_completion))

	output = {}
	if "messages" in example:
		output["text"] = messages
	if "prompt" in example:
		output["prompt"] = prompt
	if "chosen" in example:
		output["chosen"] = chosen
	if "rejected" in example:
		output["rejected"] = rejected
	if "completion" in example:
		output["completion"] = completion
	if "label" in example:
		output["label"] = example["label"]

	return output


def maybe_apply_chat_template(
	example: dict[str, list[dict[str, str]]],
	tokenizer: ProcessingClassType,
	tools: tp.Optional[list[tp.Union[dict, tp.Callable]]] = None,
) -> dict[str, str]:
	"""
	If the example is in a conversational format, apply a chat template to it.
	"""
	if is_conversational(example):
		return apply_chat_template(example, tokenizer, tools)
	else:
		return example


def _unpair_row(
	examples: list[dict[str, list[dict[str, str]]]],
) -> list[dict[str, list[dict[str, str]]]]:
	batch_size = len(examples["chosen"])
	new_rows = {
		"completion": examples["chosen"] + examples["rejected"],
		"label": [True] * batch_size + [False] * batch_size,
	}
	if "prompt" in examples:
		new_rows["prompt"] = examples["prompt"] + examples["prompt"]
	return new_rows


def unpair_preference_dataset(
	dataset: DatasetType,
	num_proc: tp.Optional[int] = None,
	desc: tp.Optional[str] = None,
) -> DatasetType:
	"""
	Unpair a preference dataset.
	"""
	return dataset.map(
		_unpair_row,
		batched=True,
		remove_columns=["chosen", "rejected"],
		num_proc=num_proc,
		desc=desc,
	)


def maybe_unpair_preference_dataset(
	dataset: DatasetType,
	num_proc: tp.Optional[int] = None,
	desc: tp.Optional[str] = None,
) -> DatasetType:
	"""
	Unpair a preference dataset if it is paired.
	"""
	if isinstance(dataset, DatasetDict):
		column_names = dataset[list(dataset.keys())[0]].column_names
	else:
		column_names = dataset.column_names
	if "chosen" in column_names and "rejected" in column_names:
		return unpair_preference_dataset(dataset, num_proc=num_proc, desc=desc)
	else:
		return dataset


def extract_prompt(example: dict[str, tp.Sequence]) -> dict[str, tp.Sequence]:
	r"""
	Extracts the shared prompt from a preference data example, where the prompt is implicit within both
	the chosen and rejected completions.
	"""
	for idx in range(min(len(example["chosen"]), len(example["rejected"]))):
		if example["chosen"][idx] != example["rejected"][idx]:
			if example["chosen"][idx - 1] == " ":
				idx -= 1
			break
	return {
		"prompt": example["chosen"][:idx],
		"chosen": example["chosen"][idx:],
		"rejected": example["rejected"][idx:],
	}


def maybe_extract_prompt(example: dict[str, list]) -> dict[str, list]:
	r"""
	Extracts the shared prompt from a preference data example, where the prompt is implicit within both
	the chosen and rejected completions.
	"""
	if "chosen" not in example or "rejected" not in example:
		return example
	if "prompt" in example:
		chosen_conv = is_conversational({"chosen": example["chosen"]})
		prompt_conv = is_conversational({"prompt": example["prompt"]})
		if (chosen_conv and prompt_conv) or (not chosen_conv and not prompt_conv):
			return example
	return extract_prompt({"chosen": example["chosen"], "rejected": example["rejected"]})
