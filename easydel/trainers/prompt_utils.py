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

import copy
import typing as tp

from datasets import Dataset, DatasetDict

from easydel.infra.utils import ProcessingClassType

DatasetType = tp.TypeVar("DatasetType", Dataset, DatasetDict)

InputDict = dict[str, str]
InputListDict = list[InputDict]
InputListListDict = list[list[InputDict]]
InputType = tp.Union[InputListListDict, InputListDict, InputDict]  # noqa:UP007
OpenAIMessageContentPart = dict[str, str]
OpenAIMessage = dict[str, str | list[OpenAIMessageContentPart]]
OutputDict = dict[str, str]
OutputListDict = list[OutputDict]
OutputType = tp.Union[OutputDict, OutputListDict, None]  # noqa:UP007
OpenAIMessageList = list[OpenAIMessage]


def _is_valid_openai_message_list(data: tp.Any) -> bool:
    """
    Checks if the input data strictly conforms to the OpenAIMessageList format
    where content is specifically a list of parts (e.g., [{"type": "text", ...}]).
    """
    if not isinstance(data, list):
        return False

    if not data:
        return True
    for item in data:
        if not isinstance(item, dict):
            return False
        if "role" not in item or "content" not in item:
            return False
        if not isinstance(item.get("role"), str):
            return False
        content = item.get("content")
        if not isinstance(content, list):
            return False
        for part in content:
            if not isinstance(part, dict):
                return False
            if "type" not in part:
                return False
    return True


def _convert_single_dict(source_dict: InputDict) -> OpenAIMessage | None:
    """
    Converts a single source dictionary into the target OpenAI message format.
    Handles variations in keys like 'content', 'text', 'message'.
    """
    if not isinstance(source_dict, dict):
        print(f"Warning: Expected a dictionary, but got {type(source_dict)}. Skipping.")
        return None

    working_dict = copy.deepcopy(source_dict)

    role = "user"
    content_text = ""
    role_key_found = None
    for key in working_dict:
        if key.lower() == "role":
            role_value = working_dict[key]
            if isinstance(role_value, str):
                role = role_value.lower()
                if role not in ["user", "assistant", "system", "tool"]:
                    print(f"Warning: Non-standard role '{role}' found. Using it.")
                role_key_found = key
                break
            else:
                print(f"Warning: 'role' value is not a string ({role_value}). Using default 'user'.")
                role_key_found = key
                break
    if role_key_found:
        del working_dict[role_key_found]
    content_keys_priority = ["content", "text", "message"]
    content_key_found = None
    for priority_key in content_keys_priority:
        for key in working_dict:
            if key.lower() == priority_key:
                content_value = working_dict[key]
                if isinstance(content_value, str):
                    content_text = content_value
                    content_key_found = key
                    break
                else:
                    print(
                        f"Warning: Found content key '{key}' but value is not a string ({content_value}). "
                        "Trying other keys or defaulting to empty."
                    )
        if content_key_found:
            break
    target_message: OpenAIMessage = {
        "role": role,
        "content": [{"type": "text", "text": content_text}],
    }
    return target_message


def reverse_openai_format(
    openai_messages: OpenAIMessageList,
    content_key_name: str = "content",
) -> OutputType | None:
    """
    Converts a list of OpenAI Chat Completion messages back into simpler formats.

    Input Format Example:
    [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello AI."}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello User!"}]
        }
    ]

    Output Format Examples:
    - If input has 1 message: {"role": "user", "content": "Hello AI."}
    - If input has >1 message: [
          {"role": "user", "content": "Hello AI."},
          {"role": "assistant", "content": "Hello User!"}
      ]
    - If input is empty: []

    Args:
        openai_messages: A list of messages in the OpenAI format.
        content_key_name: The key name to use for the message text in the
                          output dictionaries (e.g., "content", "text"). Defaults to "content".

    Returns:
        A single dictionary if only one message was processed,
        a list of dictionaries if multiple messages were processed,
        an empty list if the input was empty,
        or None if the input list structure is invalid.
    """
    if not isinstance(openai_messages, list):
        print(f"Error: Input must be a list, but got {type(openai_messages)}.")
        return None

    if not openai_messages:
        return []
    simple_messages: OutputListDict = []

    for i, message in enumerate(openai_messages):
        if not isinstance(message, dict):
            print(f"Warning: Item at index {i} is not a dictionary. Skipping.")
            continue

        role = message.get("role")
        content_list = message.get("content")

        if not isinstance(role, str) or not role:
            print(f"Warning: Message at index {i} is missing or has invalid 'role'. Skipping.")
            continue

        if not isinstance(content_list, list):
            print(f"Warning: Message at index {i} is missing or has invalid 'content' (must be a list). Skipping.")
            continue
        message_text = ""
        found_text = False
        for part in content_list:
            if isinstance(part, dict) and part.get("type") == "text":
                text_val = part.get("text")
                if isinstance(text_val, str):
                    if not found_text:
                        message_text = text_val
                        found_text = True
        if not found_text:
            print(
                f"Warning: Message at index {i} (role: {role}) has no 'content' "
                "part with type 'text'. Using empty string."
            )
        simple_dict: OutputDict = {"role": role, content_key_name: message_text}
        simple_messages.append(simple_dict)
    if len(simple_messages) == 0:
        print("Warning: Input list contained messages, but none could be processed successfully.")
        return []
    elif len(simple_messages) == 1:
        return simple_messages[0]
    else:
        return simple_messages


def convert_to_openai_format(input_data: InputType) -> OpenAIMessageList:
    """
    Converts various input formats (list[list[dict]], list[dict], dict)
    into the OpenAI Chat Completions message list format.

    If the input_data already conforms to the target OpenAIMessageList format
    (specifically with content as list of parts), it is returned directly.

    Target Format Example for one message:
    {
        "role": "user",
        "content": [{"type": "text", "text": "message content here"}]
    }

    Args:
        input_data: Data in one of the supported formats or already in the
                    target OpenAIMessageList format. Keys like 'role',
                    'content', 'text', 'message' are searched case-insensitively
                    within dictionaries during conversion.

    Returns:
        A list of messages in the target OpenAI format. Returns an empty list
        if the input is invalid, cannot be parsed, results in no valid messages,
        or is an unsupported type. Returns the input directly if it already
        matches the target format.
    """
    if _is_valid_openai_message_list(input_data):
        return tp.cast(OpenAIMessageList, input_data)

    output_messages: OpenAIMessageList = []
    items_to_process: list[InputDict] = []
    if isinstance(input_data, list):
        is_list_of_lists = False
        if input_data and all(isinstance(sub_item, list) for sub_item in input_data):
            is_list_of_lists = True

        if is_list_of_lists:
            for sublist in input_data:
                if isinstance(sublist, list):
                    for item in sublist:
                        if isinstance(item, dict):
                            items_to_process.append(item)
        else:
            for item in input_data:
                if isinstance(item, dict):
                    items_to_process.append(item)
                elif isinstance(item, list):
                    for sub_item in item:
                        if isinstance(sub_item, dict):
                            items_to_process.append(sub_item)

    elif isinstance(input_data, dict):
        items_to_process.append(input_data)

    else:
        return []
    for source_dict in items_to_process:
        converted_message = _convert_single_dict(source_dict)
        if converted_message:
            output_messages.append(converted_message)

    return output_messages


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
            if isinstance(maybe_message, dict) and "role" in maybe_message and "content" in maybe_message:
                return True

    return False


def apply_chat_template(
    example: dict[str, list[dict[str, str]]],
    tokenizer: ProcessingClassType,
    tools: list[dict | tp.Callable] | None = None,
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
        messages = tokenizer.apply_chat_template(example["messages"], tools=tools, tokenize=False)

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
    tools: list[dict | tp.Callable] | None = None,
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
    num_proc: int | None = None,
    desc: str | None = None,
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
    num_proc: int | None = None,
    desc: str | None = None,
) -> DatasetType:
    """
    Unpair a preference dataset if it is paired.
    """
    if isinstance(dataset, DatasetDict):
        column_names = dataset[next(iter(dataset.keys()))].column_names
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
