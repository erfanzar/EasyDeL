# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainer-independent conversion of message dictionaries to OpenAI content parts."""

import copy
import typing as tp

InputDict = dict[str, tp.Any]
InputListDict = list[InputDict]
InputListListDict = list[list[InputDict]]
InputType = InputListListDict | InputListDict | InputDict
OpenAIMessageContentPart = dict[str, tp.Any]
OpenAIMessage = dict[str, tp.Any]
OpenAIMessageList = list[OpenAIMessage]


def _is_valid_openai_message_list(data: tp.Any) -> bool:
    """Check for a message list whose content consists of typed content parts.

    Args:
        data: Possible OpenAI message list. Empty lists are valid.

    Returns:
        Whether every item has a string role and a list of dictionaries with
        a ``type`` field. Multimodal parts are accepted without modification.
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
    """Convert a dictionary to a message without mutating the source.

    Args:
        source_dict: Message dictionary. Keys are case-insensitive; content
            is selected in ``content``, ``text``, ``message`` order. Typed content
            parts and ancillary fields such as tool calls are preserved.

    Returns:
        A normalized message, or ``None`` for a non-dictionary. Strings become
        text parts; explicit null content and tool-call-only messages retain
        null content. Missing or non-string roles default to user; nonstandard
        string roles are lowercased and retained with a printed warning.
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
    content: list[OpenAIMessageContentPart] | None = [{"type": "text", "text": content_text}]
    has_null_content = False
    for priority_key in content_keys_priority:
        for key in working_dict:
            if key.lower() == priority_key:
                content_value = working_dict[key]
                if isinstance(content_value, str):
                    content = [{"type": "text", "text": content_value}]
                    content_key_found = key
                    break
                elif isinstance(content_value, list) and all(
                    isinstance(part, dict) and "type" in part for part in content_value
                ):
                    content = content_value
                    content_key_found = key
                    break
                elif content_value is None and priority_key == "content":
                    has_null_content = True
                else:
                    print(
                        f"Warning: Found content key '{key}' but value is not a string or typed content list "
                        f"({content_value}). Trying other keys or defaulting to empty."
                    )
        if content_key_found:
            break
    if content_key_found is None and (
        has_null_content or "tool_calls" in working_dict or "function_call" in working_dict
    ):
        content = None
    # Retain every ancillary OpenAI field while replacing legacy role/content aliases
    # with their canonical names. The source was deep-copied, including nested media.
    target_message: OpenAIMessage = {
        key: value for key, value in working_dict.items() if key.lower() not in {"role", *content_keys_priority}
    }
    target_message.update(role=role, content=content)
    return target_message


def convert_to_openai_format(input_data: InputType) -> OpenAIMessageList:
    """Convert message dictionaries or nested lists into OpenAI content parts.

    Args:
        input_data: A dictionary, a list of dictionaries, or nested lists.
            Already-formatted OpenAI message lists are returned by identity,
            including multimodal content parts and additional message fields.

    Returns:
        A flattened list of converted messages, or an empty list for unsupported
        inputs. Non-dictionary entries are ignored. Role handling and content-key
        precedence follow ``_convert_single_dict``.
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
