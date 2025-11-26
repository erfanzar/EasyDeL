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


"""Prompt formatting and chat template utilities.

This module provides utilities for converting between different conversation
formats, applying chat templates, and handling various prompt structures.
Originally from HuggingFace TRL, adapted for EasyDeL.

Key functionality:
- Convert between OpenAI format and simpler dictionary formats
- Apply chat templates to conversational datasets
- Detect conversational vs instruction formats
- Handle multi-turn conversations and function calling
"""

import copy
import typing as tp
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence

import jax
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset, DatasetDict
from jax import numpy as jnp

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
TListOrMapping = tp.TypeVar("TListOrMapping", list, Mapping)
DatasetLike = Dataset | DatasetDict


def _is_valid_openai_message_list(data: tp.Any) -> bool:
    """Check if data conforms to OpenAI message list format.

    Validates that the input data strictly follows the OpenAI Chat Completions
    message format where content is a list of content parts.

    Args:
        data: Data to validate.

    Returns:
        bool: True if data is valid OpenAI message list, False otherwise.

    Note:
        Expected format:
        [{"role": "user", "content": [{"type": "text", "text": "..."}]}]
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
    """Convert a single dictionary to OpenAI message format.

    Handles various input formats with flexible key names, converting them
    to the standardized OpenAI message structure.

    Args:
        source_dict: Dictionary with message data. Searches for keys like
                    'role', 'content', 'text', 'message' (case-insensitive).

    Returns:
        OpenAIMessage | None: Converted message in OpenAI format, or None if
                             conversion fails.

    Note:
        - Defaults to 'user' role if not specified
        - Handles non-standard roles with warnings
        - Prioritizes 'content' > 'text' > 'message' for content extraction
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
    """Check if an example is in conversational format.

    Detects whether the example contains conversation-style data with
    role and content fields.

    Args:
        example: Dictionary to check. Looks for keys like 'prompt',
                'chosen', 'rejected', 'completion', or 'messages'.

    Returns:
        bool: True if example contains conversational data with role/content
             structure, False otherwise.

    Note:
        Used to determine whether to apply chat templates during processing.
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
    **template_kwargs,
) -> dict[str, str]:
    """Apply chat template to conversational examples.

    Formats conversation data using the tokenizer's chat template,
    handling various input formats and optionally including tool schemas.

    Args:
        example: Dictionary containing conversation data. Supported keys:
                'prompt', 'chosen', 'rejected', 'completion', 'messages', 'label'.
        tokenizer: Tokenizer with chat template support.
        tools: Optional list of tool/function schemas for function calling.

    Returns:
        dict: Formatted example with chat template applied to text fields.

    Raises:
        ValueError: If example format is not supported.

    Note:
        Handles both single and multi-turn conversations.
        Preserves original structure while applying templates.
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
            example["messages"],
            tools=tools,
            tokenize=False,
            **template_kwargs,
        )

    if "prompt" in example:
        prompt = tokenizer.apply_chat_template(
            example["prompt"],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
    if "prompt" in example:
        if "chosen" in example:
            prompt_chosen = tokenizer.apply_chat_template(
                example["prompt"] + example["chosen"],
                tools=tools,
                tokenize=False,
                **template_kwargs,
            )
            chosen = prompt_chosen[len(prompt) :]
        if "rejected" in example and "prompt" in example:
            prompt_rejected = tokenizer.apply_chat_template(
                example["prompt"] + example["rejected"],
                tools=tools,
                tokenize=False,
                **template_kwargs,
            )
            rejected = prompt_rejected[len(prompt) :]
        if "completion" in example:
            prompt_completion = tokenizer.apply_chat_template(
                example["prompt"] + example["completion"],
                tools=tools,
                tokenize=False,
                **template_kwargs,
            )
            completion = prompt_completion[len(prompt) :]
    else:
        if "chosen" in example:
            chosen = tokenizer.apply_chat_template(
                example["chosen"],
                tools=tools,
                tokenize=False,
                **template_kwargs,
            )
        if "rejected" in example:
            rejected = tokenizer.apply_chat_template(
                example["rejected"],
                tools=tools,
                tokenize=False,
                **template_kwargs,
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
    """Conditionally apply chat template to conversational examples.

    Checks if the example is in conversational format and applies the
    chat template if needed, otherwise returns the example unchanged.

    Args:
        example: Dictionary that may contain conversation data.
        tokenizer: Tokenizer with chat template support.
        tools: Optional list of tool/function schemas.

    Returns:
        dict: Example with chat template applied if conversational,
             otherwise unchanged.

    Note:
        Useful for datasets that may contain mixed formats.
    """
    if is_conversational(example):
        return apply_chat_template(example, tokenizer, tools)
    else:
        return example


def maybe_convert_to_chatml(example: dict[str, list]) -> dict[str, list]:
    """
    Convert a conversational dataset with fields `from` and `value` to ChatML format.

    This function modifies conversational data to align with OpenAI's ChatML format:
    - Replaces the key `"from"` with `"role"` in message dictionaries.
    - Replaces the key `"value"` with `"content"` in message dictionaries.
    - Renames `"conversations"` to `"messages"` for consistency with ChatML.

    Args:
        example (`dict[str, list]`):
            A single data entry containing a list of messages.

    Returns:
        `dict[str, list]`:
            Example reformatted to ChatML style.

    Example:
    ```python
    >>> example = {
    ...     "conversations": [
    ...         {"from": "user", "value": "What color is the sky?"},
    ...         {"from": "assistant", "value": "It is blue."},
    ...     ]
    ... }
    >>> maybe_convert_to_chatml(example)
    {'messages': [{'role': 'user', 'content': 'What color is the sky?'},
                  {'role': 'assistant', 'content': 'It is blue.'}]}
    ```
    """
    for key in ["prompt", "completion", "chosen", "rejected", "messages", "conversations", "conversation"]:
        if key in example and isinstance(example[key], list):
            messages = example[key]
            for message in messages:
                if isinstance(message, dict):
                    if "from" in message:
                        message["role"] = message.pop("from")
                    if "value" in message:
                        message["content"] = message.pop("value")

    if "conversations" in example:
        example["messages"] = example.pop("conversations")

    if "conversation" in example:
        example["messages"] = example.pop("conversation")

    return example


def remove_none_values(example: TListOrMapping) -> TListOrMapping:
    """
    Recursively removes entries with `None` values from a nested structure (list or dictionary).

    Args:
        example (`list` or `Mapping`):
            Input nested structure (list or dictionary) from which to remove `None`.

    Example:
    ```python
    >>> [
    ...     {
    ...         "a": {"aa": None, "ab": 1},
    ...         "b": "my_string",
    ...     }
    ... ]
    >>> remove_none_values(example)
    [{'a': {'ab': 1}, 'b': 'my_string'}]
    ```
    """
    if isinstance(example, list):
        return [remove_none_values(value) if isinstance(value, dict | list) else value for value in example]
    elif isinstance(example, Mapping):
        return {
            key: remove_none_values(value) if isinstance(value, dict | list) else value
            for key, value in example.items()
            if value is not None
        }
    else:
        raise TypeError("Input must be a list or a dictionary.")


def keep_array_and_primitives(example: TListOrMapping) -> TListOrMapping:
    """
    Recursively keeps only numpy/jax arrays, ints, floats, and bools from a nested structure.

    Args:
        example (`list` or `Mapping`):
            Input nested structure (list or dictionary) to filter.

    Returns:
        Filtered structure containing only arrays and primitive types.

    Example:
    ```python
    >>> import numpy as np
    >>> example = {
    ...     "array": np.array([1, 2, 3]),
    ...     "int_val": 42,
    ...     "float_val": 3.14,
    ...     "bool_val": True,
    ...     "string": "remove_me",
    ...     "nested": {"keep": 1, "remove": "text"}
    ... }
    >>> keep_array_and_primitives(example)
    {'array': array([1, 2, 3]), 'int_val': 42, 'float_val': 3.14, 'bool_val': True, 'nested': {'keep': 1}}
    ```
    """
    import numpy as np

    def is_valid_type(value):
        """Check if value is numpy/jax array or primitive type."""
        if isinstance(value, int | float | bool | np.ndarray | jax.Array | jnp.ndarray):
            return True
        return False

    if isinstance(example, list):
        filtered = []
        for value in example:
            if isinstance(value, dict | list):
                nested = keep_array_and_primitives(value)
                if nested:
                    filtered.append(nested)
            elif is_valid_type(value):
                filtered.append(value)
        return filtered
    elif isinstance(example, Mapping):
        filtered = {}
        for key, value in example.items():
            if isinstance(value, dict | list):
                nested = keep_array_and_primitives(value)
                if nested:
                    filtered[key] = nested
            elif is_valid_type(value):
                filtered[key] = value
        return filtered
    else:
        raise TypeError("Input must be a list or a dictionary.")


def keep_arrays_map(
    example: dict[str, tp.Any],
    array_fields: list[str] | None = None,
    drop_fields: list[str] | None = None,
) -> dict[str, tp.Any]:
    """Keep only array fields and convert them to numpy arrays for HF datasets compatibility."""
    results = {}
    if array_fields is None:
        array_fields = []
    if drop_fields is None:
        drop_fields = []
    for k, v in example.items():
        if k in array_fields:
            results[k] = np.asarray(v)
        if k in drop_fields:
            continue
        elif isinstance(v, list | np.ndarray | jax.Array):
            if isinstance(v, list):
                try:
                    el = v[0]
                    if isinstance(el, dict):
                        continue
                except Exception:
                    ...
            results[k] = np.asarray(v)
    return results


def _unpair_row(examples: dict[str, list[tp.Any]]) -> dict[str, list[tp.Any]]:
    """Convert a batch of paired preference rows into unpaired rows."""

    batch_size = len(examples["chosen"])
    new_rows = {
        "completion": examples["chosen"] + examples["rejected"],
        "label": [True] * batch_size + [False] * batch_size,
    }
    if "prompt" in examples:
        new_rows["prompt"] = examples["prompt"] + examples["prompt"]
    return new_rows


def unpair_preference_dataset(dataset: DatasetType, num_proc: int | None = None, desc: str | None = None) -> DatasetType:
    r"""
    Unpair a preference dataset.

    Args:
        dataset (`Dataset` or `DatasetDict`):
            Preference dataset to unpair. The dataset must have columns `"chosen"`, `"rejected"` and optionally
            `"prompt"`.
        num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset. (Unused in the current implementation.)
        desc (`str`, *optional*):
            Meaningful description to be displayed alongside with the progress bar while mapping examples.

    Returns:
        `Dataset`: The unpaired preference dataset.

    Example:

    ```python
    >>> from datasets import Dataset

    >>> dataset_dict = {
    ...     "prompt": ["The sky is", "The sun is"],
    ...     "chosen": [" blue.", "in the sky."],
    ...     "rejected": [" green.", " in the sea."],
    ... }
    >>> dataset = Dataset.from_dict(dataset_dict)
    >>> dataset = unpair_preference_dataset(dataset)
    >>> dataset
    Dataset({
        features: ['prompt', 'completion', 'label'],
        num_rows: 4
    })

    >>> dataset[0]
    {'prompt': 'The sky is', 'completion': ' blue.', 'label': True}
    ```
    """

    if isinstance(dataset, DatasetDict):
        return DatasetDict(
            {key: unpair_preference_dataset(subset, num_proc=num_proc, desc=desc) for key, subset in dataset.items()}
        )

    column_names = dataset.column_names

    remove_columns = ["chosen", "rejected"]

    try:
        return dataset.map(
            _unpair_row,
            batched=True,
            remove_columns=remove_columns,
            num_proc=num_proc,
            desc=desc,
        )
    except pa.ArrowInvalid:
        data = {"completion": [], "label": []}
        if "prompt" in column_names:
            data["prompt"] = []
        for example in dataset:
            prompt_value = example.get("prompt")
            for completion, label in ((example["chosen"], True), (example["rejected"], False)):
                data["completion"].append(completion)
                data["label"].append(label)
                if "prompt" in column_names:
                    data["prompt"].append(prompt_value)
        return Dataset.from_dict(data)


def maybe_unpair_preference_dataset(
    dataset: DatasetType,
    num_proc: int | None = None,
    desc: str | None = None,
) -> DatasetType:
    r"""
    Unpair a preference dataset if it is paired.

    Args:
        dataset (`Dataset` or `DatasetDict`):
            Preference dataset to unpair. The dataset must have columns `"chosen"`, `"rejected"` and optionally
            `"prompt"`.
        num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        desc (`str`, *optional*):
            Meaningful description to be displayed alongside with the progress bar while mapping examples.

    Returns:
        `Dataset` or `DatasetDict`: The unpaired preference dataset if it was paired, otherwise the original dataset.

    Example:

    ```python
    >>> from datasets import Dataset

    >>> dataset_dict = {
    ...     "prompt": ["The sky is", "The sun is"],
    ...     "chosen": [" blue.", "in the sky."],
    ...     "rejected": [" green.", " in the sea."],
    ... }
    >>> dataset = Dataset.from_dict(dataset_dict)
    >>> dataset = unpair_preference_dataset(dataset)
    >>> dataset
    Dataset({
        features: ['prompt', 'completion', 'label'],
        num_rows: 4
    })

    >>> dataset[0]
    {'prompt': 'The sky is', 'completion': ' blue.', 'label': True}
    ```
    """
    if isinstance(dataset, DatasetDict):
        column_names = dataset[next(iter(dataset.keys()))].column_names
    else:
        column_names = dataset.column_names
    if "chosen" in column_names and "rejected" in column_names:
        return unpair_preference_dataset(dataset, num_proc=num_proc, desc=desc)
    else:
        return dataset


def extract_prompt(example: dict[str, Sequence]) -> dict[str, Sequence]:
    r"""
    Extracts the shared prompt from a preference data example, where the prompt is implicit within both the chosen and
    rejected completions.

    For more details, see [`maybe_extract_prompt`].
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
    Extracts the shared prompt from a preference data example, where the prompt is implicit within both the chosen and
    rejected completions.

    If the example already contains a `"prompt"` key, the function returns the example as is. Else, the function
    identifies the longest common sequence (prefix) of conversation turns between the "chosen" and "rejected"
    completions and extracts this as the prompt. It then removes this prompt from the respective "chosen" and
    "rejected" completions.

    Args:
        example (`dict[str, list]`):
            A dictionary representing a single data entry in the preference dataset. It must contain the keys
            `"chosen"` and `"rejected"`, where each value is either conversational or standard (`str`).

    Returns:
        `dict[str, list]`: A dictionary containing:
            - `"prompt"`: The longest common prefix between the "chosen" and "rejected" completions.
            - `"chosen"`: The remainder of the "chosen" completion, with the prompt removed.
            - `"rejected"`: The remainder of the "rejected" completion, with the prompt removed.

    Examples:

    ```python
    >>> example = {
    ...     "chosen": [
    ...         {"role": "user", "content": "What color is the sky?"},
    ...         {"role": "assistant", "content": "It is blue."},
    ...     ],
    ...     "rejected": [
    ...         {"role": "user", "content": "What color is the sky?"},
    ...         {"role": "assistant", "content": "It is green."},
    ...     ],
    ... }
    >>> extract_prompt(example)
    {'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
     'chosen': [{'role': 'assistant', 'content': 'It is blue.'}],
     'rejected': [{'role': 'assistant', 'content': 'It is green.'}]}
    ```

    Or, with the `map` method of `datasets.Dataset`:

    ```python
    >>> from trl import extract_prompt
    >>> from datasets import Dataset

    >>> dataset_dict = {
    ...     "chosen": [
    ...         [
    ...             {"role": "user", "content": "What color is the sky?"},
    ...             {"role": "assistant", "content": "It is blue."},
    ...         ],
    ...         [
    ...             {"role": "user", "content": "Where is the sun?"},
    ...             {"role": "assistant", "content": "In the sky."},
    ...         ],
    ...     ],
    ...     "rejected": [
    ...         [
    ...             {"role": "user", "content": "What color is the sky?"},
    ...             {"role": "assistant", "content": "It is green."},
    ...         ],
    ...         [
    ...             {"role": "user", "content": "Where is the sun?"},
    ...             {"role": "assistant", "content": "In the sea."},
    ...         ],
    ...     ],
    ... }
    >>> dataset = Dataset.from_dict(dataset_dict)
    >>> dataset = dataset.map(extract_prompt)
    >>> dataset[0]
    {'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
     'chosen': [{'role': 'assistant', 'content': 'It is blue.'}],
     'rejected': [{'role': 'assistant', 'content': 'It is green.'}]}
    ```
    """
    if "chosen" not in example or "rejected" not in example:
        return example
    if "prompt" in example:
        chosen_conv = is_conversational({"chosen": example["chosen"]})
        prompt_conv = is_conversational({"prompt": example["prompt"]})
        if (chosen_conv and prompt_conv) or (not chosen_conv and not prompt_conv):
            return example
    return extract_prompt({"chosen": example["chosen"], "rejected": example["rejected"]})


class _SegmentTree:
    """
    A segment tree data structure that, when initialized as `_SegmentTree(maxval)`, efficiently finds the next larger
    value for a given input within the range [1, maxval].
    """

    def __init__(self, maxval: int):
        self.maxval = maxval

        self.tree_size = 1 << (maxval - 1).bit_length()
        self.tree = [0] * (2 * self.tree_size)

    def add(self, val):
        assert 0 < val <= self.maxval
        i = self.tree_size + val - 1
        self.tree[i] = val
        while i > 1:
            i >>= 1
            left, right = self.tree[i << 1], self.tree[(i << 1) + 1]

            self.tree[i] = left if left >= right else right

    def remove(self, val):
        assert 0 < val <= self.maxval
        i = self.tree_size + val - 1
        self.tree[i] = 0
        while i > 1:
            i >>= 1
            left, right = self.tree[i << 1], self.tree[(i << 1) + 1]

            self.tree[i] = left if left >= right else right

    def search(self, val):
        assert 0 < val <= self.maxval
        i = 1
        while i < self.tree_size:
            if self.tree[i << 1] >= val:
                i = i << 1
            else:
                i = (i << 1) + 1
        return self.tree[i]


def _pack_bfd(examples: pa.Table, seq_length: int) -> pa.Table:
    """Pack sequences in a pyarrow Table using Best Fit Decreasing strategy."""
    columns = []
    list_column_idx = None
    for idx, column in enumerate(examples.columns):
        if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
            column = pc.list_slice(column, 0, seq_length)
            if list_column_idx is None:
                list_column_idx = idx
        columns.append(column)
    examples = pa.Table.from_arrays(columns, names=examples.column_names)

    ids = np.arange(len(examples))
    assert list_column_idx is not None
    lengths = pc.list_value_length(examples[list_column_idx]).combine_chunks()
    examples = examples.append_column("seq_lengths", lengths)
    lengths = pc.make_struct(lengths, ids)
    lengths = lengths.sort("descending", by=0)

    segment_tree = _SegmentTree(seq_length)
    segment_tree.add(seq_length)
    space_to_bin = defaultdict(deque)

    bins: list[dict] = []
    for length, idx in zip(lengths.field(0).to_numpy(), lengths.field(1).to_numpy(), strict=False):
        space = segment_tree.search(length)

        if space < seq_length:
            bin = space_to_bin[space].popleft()  # noqa
        else:
            bin = {"ids": [], "length": 0}  # noqa
            bins.append(bin)

        bin["ids"].append(idx)
        bin["length"] += length
        if space < seq_length and not space_to_bin[space]:
            segment_tree.remove(space)

        space = space - length
        space_to_bin[space].append(bin)
        if space > 0:
            segment_tree.add(space)

    examples = pc.take(examples, [id_ for bin in bins for id_ in bin["ids"]])  # noqa
    offsets = np.array([0] + [bin["length"] for bin in bins])  # noqa
    offsets = np.cumsum(offsets)

    assert all(column.num_chunks == 1 for column in examples.columns)

    lengths = examples["seq_lengths"].chunks[0]
    examples = examples.drop_columns("seq_lengths")
    lengths = pa.ListArray.from_arrays(np.cumsum([0] + [len(bin["ids"]) for bin in bins], dtype=np.int32), lengths)  # noqa

    columns = []
    for column in examples.columns:
        column = column.chunks[0]
        if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
            dtype = column.offsets.type.to_pandas_dtype()
            column = type(column).from_arrays(offsets.astype(dtype), column.values)
        columns.append(column)
    return pa.Table.from_arrays([*columns, lengths], names=[*examples.column_names, "seq_lengths"])


def _pack_wrapped(examples: pa.Table, seq_length: int) -> pa.Table:
    """Pack sequences in a pyarrow Table using a wrapped strategy."""
    columns = []
    for column in examples.columns:
        if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
            if isinstance(column, pa.ChunkedArray):
                column = column.combine_chunks()
            offsets, values = column.offsets, column.values
            values = values[offsets[0].as_py() : offsets[-1].as_py()]
            num_elements = len(values)
            dtype = offsets.type.to_pandas_dtype()
            offsets = np.arange(0, num_elements, seq_length, dtype=dtype)
            offsets = np.concatenate((offsets, [num_elements]))
            column = type(column).from_arrays(offsets, values)
        columns.append(column)
    return pa.Table.from_arrays(columns, names=examples.column_names)


def pack_dataset(
    dataset: DatasetType,
    seq_length: int,
    strategy: str = "bfd",
    map_kwargs: dict[str, tp.Any] | None = None,
) -> DatasetType:
    r"""
    Pack sequences in a dataset into chunks of size `seq_length`.

    Args:
        dataset (`Dataset` or `DatasetDict`):
            Dataset to pack
        seq_length (`int`):
            Target sequence length to pack to.
        strategy (`str`, *optional*, defaults to `"bfd"`):
            Packing strategy to use. Can be either:

            - `"bfd"` (Best Fit Decreasing): Slower but preserves sequence boundaries. Sequences are never cut in the
                middle.
            - `"wrapped"`: Faster but more aggressive. Ignores sequence boundaries and will cut sequences in the middle
                to completely fill each packed sequence with data.
        map_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the dataset's map method when packing examples.

    Returns:
        `Dataset` or `DatasetDict`: The dataset with packed sequences. The number of examples may decrease as sequences
        are combined.

    Example:
    ```python
    >>> from datasets import Dataset
    >>> from trl import pack_dataset

    >>> examples = {
    ...     "input_ids": [[1, 2, 3], [4, 5], [6, 7, 8], [9]],
    ...     "attention_mask": [[1, 1, 0], [1, 0], [1, 0, 0], [1]],
    ... }
    >>> dataset = Dataset.from_dict(examples)
    >>> packed_dataset = pack_dataset(dataset, seq_length=4, strategy="bfd")
    >>> packed_dataset[:]
    {'input_ids': [[1, 2, 3, 9], [6, 7, 8], [4, 5]],
    'attention_mask': [[1, 1, 0, 1], [1, 0, 0], [1, 0]],
    'seq_lengths': [[3, 1], [3], [2]]}
    ```
    """
    if map_kwargs is None:
        map_kwargs = {}

    dataset = dataset.with_format("arrow")
    if strategy == "bfd":
        dataset = dataset.map(_pack_bfd, batched=True, fn_kwargs={"seq_length": seq_length}, **map_kwargs)
    elif strategy == "wrapped":
        dataset = dataset.map(_pack_wrapped, batched=True, fn_kwargs={"seq_length": seq_length}, **map_kwargs)
    else:
        raise ValueError(f"Invalid packing strategy: {strategy}. Use 'bfd' or 'wrapped'.")
    dataset = dataset.with_format(None)
    return dataset


def truncate_dataset(dataset: DatasetType, max_length: int, map_kwargs: dict[str, tp.Any] | None = None) -> DatasetType:
    r"""
    Truncate sequences in a dataset to a specified `max_length`.

    Args:
        dataset (`Dataset` or `DatasetDict`):
            Dataset to truncate.
        max_length (`int`):
            Maximum sequence length to truncate to.
        map_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the dataset's map method when truncating examples.

    Returns:
        `Dataset` or `DatasetDict`: The dataset with truncated sequences.

    Example:
    ```python
    >>> from datasets import Dataset

    >>> examples = {
    ...     "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
    ...     "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
    ... }
    >>> dataset = Dataset.from_dict(examples)
    >>> truncated_dataset = truncate_dataset(dataset, max_length=2)
    >>> truncated_dataset[:]
    {'input_ids': [[1, 2], [4, 5], [8]],
     'attention_mask': [[0, 1], [0, 0], [1]]}
    ```
    """
    if map_kwargs is None:
        map_kwargs = {}
    if isinstance(dataset, Dataset):

        def truncate(examples):
            truncated_columns = []
            for column in examples.columns:
                if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
                    column = pc.list_slice(column, 0, max_length)
                truncated_columns.append(column)
            return pa.Table.from_arrays(truncated_columns, names=examples.column_names)

        dataset = dataset.with_format("arrow")
        dataset = dataset.map(truncate, batched=True, **map_kwargs)
        dataset = dataset.with_format(None)
    else:

        def truncate(examples):
            truncated_examples = {}
            for key, column in examples.items():
                if column and isinstance(column[0], list):
                    column = [val[:max_length] for val in column]
                truncated_examples[key] = column
            return truncated_examples

        dataset = dataset.map(truncate, batched=True, **map_kwargs)
    return dataset


def pad_and_truncate_dataset(
    dataset: DatasetLike,
    max_length: int,
    padding_token_id: int | None = None,
    padding_values: dict[str, tp.Any] | None = None,
    truncate: bool = True,
    padding: bool = True,
    side: tp.Literal["left", "right"] = "left",
    map_kwargs: dict[str, tp.Any] | None = None,
    make_it_1d: bool = True,
) -> DatasetLike:
    """
    Pad and/or truncate sequences in a dataset to a specified `max_length`.

    Preserves array backends:
      - If a column's sequences are numpy arrays, outputs numpy arrays.
      - If a column's sequences are JAX arrays, outputs JAX arrays.
      - If a column's sequences are Python lists, outputs lists.

    Special handling:
      - Columns ending with '_ids' or named 'labels' use `padding_token_id` (required if padding such columns)
      - 'attention_mask' columns use 0 for padding
      - 'position_ids' columns are continued sequentially when padding
      - Custom padding values can be specified via `padding_values`, which overrides defaults.

    Notes:
      - If an entire batch column is None, backend cannot be inferred; it falls back to Python lists for that batch.
      - Hugging Face Datasets stores data in Arrow; on retrieval, types may depend on dataset.set_format().
        This function preserves types within the map, but downstream representation may vary unless you set a format.
    """
    if map_kwargs is None:
        map_kwargs = {}
    if padding_values is None:
        padding_values = {}

    def get_padding_value(column_name: str) -> tp.Any:
        if column_name in padding_values:
            return padding_values[column_name]

        if column_name == "attention_mask":
            return 0
        elif column_name == "position_ids":
            return None
        elif column_name.endswith("_ids") or column_name == "labels":
            if padding_token_id is None:
                raise ValueError(
                    f"padding_token_id must be provided for column '{column_name}'. "
                    f"Alternatively, specify a custom value in padding_values."
                )
            return padding_token_id
        elif column_name.endswith("_mask"):
            return 0
        else:
            return 0

    def process_batch(batch: dict[str, list[tp.Any]]) -> dict[str, list[tp.Any]]:
        processed: dict[str, list[tp.Any]] = {}
        for k, v in batch.items():
            # Ensure v is an array (handle cases where HF datasets returns lists)
            if not hasattr(v, "shape"):
                v = jnp.asarray(v)

            pad_val = get_padding_value(k)
            pad = max_length - v.shape[-1]
            if pad < 0 and truncate:
                v = v[..., -max_length:] if side == "left" else v[..., :max_length]
            elif padding and pad > 0:
                pad_width = [(0, 0)] * v.ndim
                pad_width[-1] = (pad, 0) if side == "left" else (0, pad)
                v = jnp.pad(v, tuple(pad_width), mode="constant", constant_values=pad_val)
            if make_it_1d:
                v = v.reshape(-1)
            processed[k] = v
        return processed

    return dataset.map(process_batch, batched=False, **map_kwargs)


def is_conversational_from_value(example: dict[str, tp.Any]) -> bool:
    r"""
    Check if the example is in a conversational format (from/value). Note that this format isn't recommended. Prefer
    the ChatML format (role/content)

    Args:
        example (`dict[str, Any]`):
            A single data entry of a dataset. The example can have different keys depending on the dataset type.

    Returns:
        `bool`:
            `True` if the data is in a conversational Chatformat, `False` otherwise.

    Examples:

    ```python
    >>> example = {"conversations": [{"from": "user", "value": "What color is the sky?"}]}
    >>> is_conversational_from_value(example)
    True

    >>> example = {"conversations": [{"role": "user", "content": "What color is the sky?"}]}
    >>> is_conversational_from_value(example)
    False

    >>> example = {"conversations": "The sky is"}
    >>> is_conversational_from_value(example)
    False
    ```
    """
    maybe_messages = example.get("conversations")
    if maybe_messages is None:
        maybe_messages = example.get("conversation")

    if isinstance(maybe_messages, list):
        maybe_message = maybe_messages[0]

        if isinstance(maybe_message, dict) and "from" in maybe_message and "value" in maybe_message:
            return True

        if isinstance(maybe_message, dict) and "role" in maybe_message and "content" in maybe_message:
            return True

    return False
