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
import json
import typing as tp
from collections.abc import Mapping, Sequence

import jax
import numpy as np
import pyarrow as pa  # pyright: ignore[reportMissingTypeStubs]
from datasets import Dataset, DatasetDict  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp

from easydel.infra.utils import ProcessingClassType

DatasetType = tp.TypeVar("DatasetType", Dataset, DatasetDict)

InputDict = dict[str, str]
InputListDict = list[InputDict]
InputListListDict = list[list[InputDict]]
InputType = InputListListDict | InputListDict | InputDict
OpenAIMessageContentPart = dict[str, str]
OpenAIMessage = dict[str, str | list[OpenAIMessageContentPart]]
OutputDict = dict[str, str]
OutputListDict = list[OutputDict]
OutputType = OutputDict | OutputListDict | None
OpenAIMessageList = list[OpenAIMessage]
TListOrMapping = tp.TypeVar("TListOrMapping", list, Mapping)
DatasetLike = Dataset | DatasetDict

_CHATML_ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
}


def _maybe_json_load(value: str) -> tp.Any:
    """Best-effort JSON decode of a string, returning the original on failure.

    Args:
        value: A string that may be a JSON document.

    Returns:
        The parsed JSON object when ``value`` looks like JSON (begins with
        ``{`` or ``[``) and decodes cleanly; otherwise the original
        ``value``.
    """
    stripped = value.strip()
    if not stripped or stripped[0] not in {"{", "["}:
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def normalize_message_payload(
    payload: tp.Any,
    *,
    allow_plain_text: bool = False,
) -> list[dict[str, tp.Any]] | None:
    """Normalize chat payloads into ``[{role, content, ...}, ...]`` form."""

    def _normalize_single(item: tp.Any) -> list[dict[str, tp.Any]] | None:
        """Normalize a single message payload to a list of role/content dicts.

        Handles ChatML ``from``/``value`` pairs, OpenAI ``role``/``content``
        dicts, JSON strings, and (when ``allow_plain_text`` is set) bare
        strings interpreted as a user turn.

        Args:
            item: A single message-shaped value (string/dict/list).

        Returns:
            A one-element list of normalized message dicts, or ``None``
            when the payload cannot be turned into a chat message.
        """
        if isinstance(item, str):
            parsed = _maybe_json_load(item)
            if parsed is not item:
                return normalize_message_payload(parsed, allow_plain_text=allow_plain_text)
            if allow_plain_text:
                return [{"role": "user", "content": item}]
            return None

        if isinstance(item, dict):
            if "from" in item and "value" in item:
                normalized = dict(item)
                normalized["role"] = _CHATML_ROLE_MAPPING.get(item.get("from", "user"), item.get("from", "user"))
                normalized["content"] = item.get("value", "")
                normalized.pop("from", None)
                normalized.pop("value", None)
                return [normalized]

            if "role" in item:
                normalized = dict(item)
                normalized.setdefault("content", "")
                return [normalized]

            if "content" in item and allow_plain_text:
                normalized = dict(item)
                normalized.setdefault("role", "user")
                return [normalized]
            return None

        if isinstance(item, list):
            return normalize_message_payload(item, allow_plain_text=allow_plain_text)

        return None

    if payload is None:
        return None

    if isinstance(payload, list):
        normalized: list[dict[str, tp.Any]] = []
        for item in payload:
            turns = _normalize_single(item)
            if turns is None:
                return None
            normalized.extend(turns)
        return normalized

    return _normalize_single(payload)


def normalize_tool_payload(payload: tp.Any) -> tp.Any:
    """Normalize stringified JSON tool payloads into dict/list form."""
    if isinstance(payload, str):
        parsed = _maybe_json_load(payload)
        if parsed is not payload:
            return normalize_tool_payload(parsed)
        return payload

    if isinstance(payload, list):
        normalized: list[tp.Any] = []
        for item in payload:
            parsed = normalize_tool_payload(item)
            if isinstance(parsed, list):
                normalized.extend(parsed)
            else:
                normalized.append(parsed)
        return normalized

    if isinstance(payload, tuple):
        return normalize_tool_payload(list(payload))

    if isinstance(payload, dict):
        return dict(payload)

    return payload


def resolve_example_tools(
    example: dict[str, tp.Any],
    fallback_tools: list | None = None,
) -> list | None:
    """Return per-example tool schemas when available, otherwise ``fallback_tools``."""

    example_tools = normalize_tool_payload(example.get("tools"))
    if isinstance(example_tools, dict):
        example_tools = [example_tools]
    if example_tools is not None and example.get("tools") is not example_tools:
        example["tools"] = example_tools
    return example_tools if example_tools is not None else fallback_tools


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
    for key in ["prompt", "chosen", "rejected", "completion", "messages", "conversations", "conversation"]:
        maybe_messages = normalize_message_payload(example.get(key), allow_plain_text=(key == "messages"))
        if maybe_messages:
            return True

    return False


def _normalize_chat_suffix(
    value: list[dict[str, tp.Any]] | str,
    field_name: str,
) -> list[dict[str, tp.Any]]:
    """Normalize a prompt or completion suffix value into a conversational message list.

    This helper ensures that suffix values (which may come from dataset
    columns or trainer configuration) are always represented as a list of
    chat-style message dictionaries suitable for passing to a chat template.

    If the value is already a list, it is returned unchanged (assumed to be
    a well-formed list of ``{"role": ..., "content": ...}`` dicts).  If it
    is a plain string, it is wrapped as a single assistant message:
    ``[{"role": "assistant", "content": value}]``.

    Args:
        value: The suffix to normalize. Either a string (interpreted as
            assistant-role content) or a pre-formed list of message dicts.
        field_name: Name of the field being normalized, used in the error
            message when the type is unsupported.

    Returns:
        A list of message dictionaries representing the suffix.

    Raises:
        TypeError: If ``value`` is neither a string nor a list.
    """
    normalized = normalize_message_payload(value, allow_plain_text=False)
    if normalized is not None:
        return normalized
    if isinstance(value, str):
        return [{"role": "assistant", "content": value}]
    raise TypeError(f"`{field_name}` must be a string or a list of chat messages, received {type(value)}.")


def render_prompt_with_suffix(
    prompt_messages: list[dict[str, str]],
    suffix: list[dict[str, str]] | str,
    tokenizer: ProcessingClassType,
    *,
    field_name: str,
    tools: list[dict | tp.Callable] | None = None,
    **template_kwargs,
) -> tuple[str, str, str]:
    """Render a conversational prompt using a chat template and derive the rendered suffix.

    The function first renders the prompt messages alone (with
    ``add_generation_prompt=True``) to obtain the prompt text.  It then
    renders the full conversation (prompt messages + normalized suffix
    messages) to obtain the complete text.  The suffix text is computed as
    the difference between the full rendering and the prompt-only
    rendering, i.e. ``full_text[len(prompt_text):]``.

    This two-step approach ensures that the suffix text is exactly what the
    chat template would produce for the suffix portion, including any
    special tokens or formatting that the template inserts between turns.

    Args:
        prompt_messages: List of message dicts forming the prompt portion of
            the conversation (e.g. ``[{"role": "user", "content": "..."}]``).
        suffix: The suffix to append after the prompt. Can be a plain
            string (treated as assistant content) or a list of message
            dicts. Normalized via ``_normalize_chat_suffix``.
        tokenizer: Tokenizer or processor with an ``apply_chat_template``
            method.
        field_name: Identifier for the field being rendered, used in error
            messages and passed through to ``_normalize_chat_suffix``.
        tools: Optional list of tool/function schemas to pass to the chat
            template (for function-calling models).
        **template_kwargs: Additional keyword arguments forwarded to
            ``tokenizer.apply_chat_template``.

    Returns:
        A 3-tuple of ``(prompt_text, suffix_text, full_text)`` where:
            - ``prompt_text`` is the rendered prompt with generation prompt.
            - ``suffix_text`` is the rendered suffix (completion) portion.
            - ``full_text`` is the complete rendered conversation.

    Raises:
        ValueError: If the full conversation rendering does not start with
            the prompt-only rendering, which indicates an incompatible chat
            template.
    """
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    full_conversation = tokenizer.apply_chat_template(
        prompt_messages + _normalize_chat_suffix(suffix, field_name),
        tools=tools,
        tokenize=False,
        **template_kwargs,
    )
    if not full_conversation.startswith(prompt_text):
        raise ValueError(
            "The chat template applied to the prompt + suffix does not start with the chat template "
            f"applied to the prompt alone for `{field_name}`."
            f"\n**Prompt**:\n{prompt_text}\n\n**Prompt + Suffix**:\n{full_conversation}"
        )
    return prompt_text, full_conversation[len(prompt_text) :], full_conversation


def apply_chat_template(
    example: dict[str, tp.Any],
    tokenizer: ProcessingClassType,
    tools: list[dict | tp.Callable] | None = None,
    **template_kwargs,
) -> dict[str, tp.Any]:
    """Apply chat template to conversational examples.

    Formats conversation data using the tokenizer's chat template,
    handling various input formats and optionally including tool schemas.

    Args:
        example: Dictionary containing conversation data. Supported keys:
                'prompt', 'chosen', 'rejected', 'completion', 'messages', 'label'.
        tokenizer: Tokenizer with chat template support.
        tools: Optional list of tool/function schemas for function calling.

    Returns:
        dict: Formatted example with chat template applied to text fields
            while preserving unrelated payload columns.

    Raises:
        ValueError: If example format is not supported.

    Note:
        Handles both single and multi-turn conversations.
        Preserves original structure while applying templates.
    """
    example = maybe_convert_to_chatml(dict(example))
    tools = resolve_example_tools(example, tools)

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

    # Initialize variables that may be conditionally assigned
    messages: str = ""
    prompt: str = ""
    chosen: str = ""
    rejected: str = ""
    completion: str = ""

    if "messages" in example:
        messages = tokenizer.apply_chat_template(
            example["messages"],
            tools=tools,
            tokenize=False,
            **template_kwargs,
        )

    prompt_value = example.get("prompt")
    if isinstance(prompt_value, list):
        prompt = tokenizer.apply_chat_template(
            prompt_value,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        if "chosen" in example:
            prompt, chosen, _prompt_chosen = render_prompt_with_suffix(
                prompt_value,
                example["chosen"],
                tokenizer,
                field_name="chosen",
                tools=tools,
                **template_kwargs,
            )
        if "rejected" in example:
            prompt, rejected, _prompt_rejected = render_prompt_with_suffix(
                prompt_value,
                example["rejected"],
                tokenizer,
                field_name="rejected",
                tools=tools,
                **template_kwargs,
            )
        if "completion" in example:
            prompt, completion, _prompt_completion = render_prompt_with_suffix(
                prompt_value,
                example["completion"],
                tokenizer,
                field_name="completion",
                tools=tools,
                **template_kwargs,
            )
    elif "prompt" in example:
        prompt = example["prompt"]
        if any(isinstance(example.get(key), list) for key in ("chosen", "rejected", "completion")):
            raise ValueError(
                "Conversational chosen/rejected/completion values require `prompt` to be a conversational "
                "message list so the full conversation can be rendered safely."
            )
        if "chosen" in example:
            chosen = example["chosen"]
        if "rejected" in example:
            rejected = example["rejected"]
        if "completion" in example:
            completion = example["completion"]
    else:
        if "chosen" in example:
            chosen = (
                tokenizer.apply_chat_template(
                    example["chosen"],
                    tools=tools,
                    tokenize=False,
                    **template_kwargs,
                )
                if isinstance(example["chosen"], list)
                else example["chosen"]
            )
        if "rejected" in example:
            rejected = (
                tokenizer.apply_chat_template(
                    example["rejected"],
                    tools=tools,
                    tokenize=False,
                    **template_kwargs,
                )
                if isinstance(example["rejected"], list)
                else example["rejected"]
            )

    output = {
        key: value
        for key, value in example.items()
        if key not in {"messages", "prompt", "chosen", "rejected", "completion"}
    }
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
    example: dict[str, tp.Any],
    tokenizer: ProcessingClassType,
    tools: list[dict | tp.Callable] | None = None,
) -> dict[str, tp.Any]:
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
        return example  # pyright: ignore[reportReturnType]


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
    result = dict(example)

    if "conversations" in result:
        result["messages"] = result.pop("conversations")

    if "conversation" in result:
        result["messages"] = result.pop("conversation")

    for key in ["prompt", "completion", "chosen", "rejected", "messages"]:
        if key not in result:
            continue
        normalized = normalize_message_payload(result[key], allow_plain_text=(key == "messages"))
        if normalized is not None:
            result[key] = normalized

    resolve_example_tools(result)
    return result


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
        return filtered  # pyright: ignore[reportReturnType]
    elif isinstance(example, Mapping):
        filtered = {}
        for key, value in example.items():
            if isinstance(value, dict | list):
                nested = keep_array_and_primitives(value)
                if nested:
                    filtered[key] = nested
            elif is_valid_type(value):
                filtered[key] = value
        return filtered  # pyright: ignore[reportReturnType]
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
        return Dataset.from_dict(data)  # pyright: ignore[reportReturnType]


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
    idx = 0
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
    return extract_prompt({"chosen": example["chosen"], "rejected": example["rejected"]})  # pyright: ignore[reportReturnType]


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
