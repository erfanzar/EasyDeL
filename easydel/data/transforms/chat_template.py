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

"""Chat template transforms for conversational data.

This module provides:
- ChatTemplateTransform: Apply chat template to convert messages to formatted text
- MaybeApplyChatTemplate: Conditionally apply chat template if data is conversational
- ConvertToChatML: Convert from/value format to role/content (ChatML) format
"""

from __future__ import annotations

import typing as tp

from .base import Example, Transform


def is_conversational(example: dict) -> bool:
    """Check if an example is in conversational format.

    Detects if the example contains messages/conversations with role/content structure.

    Args:
        example: Example dictionary to check.

    Returns:
        True if example appears to be conversational format.
    """
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages", "conversations"]

    for key in supported_keys:
        value = example.get(key)
        if value is None:
            continue

        # Check if it's a list of dicts with role/content
        if isinstance(value, list) and value:
            first_item = value[0]
            if isinstance(first_item, dict):
                if "role" in first_item or "from" in first_item:
                    return True

    return False


def convert_to_chatml(messages: list[dict]) -> list[dict]:
    """Convert from/value format to role/content (ChatML) format.

    Args:
        messages: List of messages, possibly in from/value format.

    Returns:
        Messages in role/content format.
    """
    result = []
    for msg in messages:
        if "from" in msg and "value" in msg:
            # Convert from/value to role/content
            result.append(
                {
                    "role": msg["from"],
                    "content": msg["value"],
                }
            )
        else:
            result.append(msg)
    return result


class ChatTemplateTransform(Transform):
    """Apply chat template to convert messages to formatted text.

    This is the primary transform for converting conversational data:
        {"messages": [{"role": "user", "content": "Hello!"}, ...]}
    to:
        {"text": "<formatted conversation>"}

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        >>> transform = ChatTemplateTransform(tokenizer)
        >>> result = transform({
        ...     "messages": [
        ...         {"role": "user", "content": "Hello!"},
        ...         {"role": "assistant", "content": "Hi there!"}
        ...     ]
        ... })
        >>> # {"text": "<s>[INST] Hello! [/INST] Hi there! </s>"}
    """

    def __init__(
        self,
        tokenizer: tp.Any,
        messages_field: str = "messages",
        output_field: str = "text",
        tools: list[dict | tp.Callable] | None = None,
        convert_from_value_format: bool = True,
        drop_messages: bool = True,
        **template_kwargs,
    ):
        """Initialize ChatTemplateTransform.

        Args:
            tokenizer: HuggingFace tokenizer with chat template.
            messages_field: Field containing the messages list. Also checks
                "conversations" and "conversation" as fallbacks.
            output_field: Field to store the formatted text.
            tools: Optional tools/functions for function calling templates.
            convert_from_value_format: Auto-convert from/value to role/content format.
            drop_messages: Remove the original messages field after conversion.
            **template_kwargs: Additional kwargs passed to apply_chat_template.
        """
        self._tokenizer = tokenizer
        self._messages_field = messages_field
        self._output_field = output_field
        self._tools = tools
        self._convert_from_value = convert_from_value_format
        self._drop_messages = drop_messages
        self._template_kwargs = template_kwargs

    def __call__(self, example: Example) -> Example:
        """Apply chat template to the example."""
        result = example.copy()

        # Handle alternate field names
        messages = None
        source_field = None
        for field in [self._messages_field, "conversations", "conversation"]:
            if field in result:
                messages = result[field]
                source_field = field
                break

        if messages is None:
            return result

        # Convert from/value format if needed
        if self._convert_from_value and messages:
            first_msg = messages[0] if isinstance(messages, list) and messages else None
            if isinstance(first_msg, dict) and "from" in first_msg and "value" in first_msg:
                messages = convert_to_chatml(messages)

        # Apply chat template
        try:
            formatted_text = self._tokenizer.apply_chat_template(
                messages,
                tools=self._tools,
                tokenize=False,
                **self._template_kwargs,
            )
        except Exception as e:
            # If chat template fails, fall back to simple formatting
            import warnings

            warnings.warn(f"Chat template failed: {e}. Using simple formatting.", stacklevel=2)
            formatted_text = self._simple_format(messages)

        result[self._output_field] = formatted_text

        if self._drop_messages and source_field:
            result.pop(source_field, None)

        return result

    def _simple_format(self, messages: list[dict]) -> str:
        """Simple fallback formatting if chat template fails."""
        parts = []
        for msg in messages:
            role = msg.get("role", msg.get("from", "unknown"))
            content = msg.get("content", msg.get("value", ""))
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def __repr__(self) -> str:
        return f"ChatTemplateTransform(messages_field={self._messages_field!r}, output_field={self._output_field!r})"


class MaybeApplyChatTemplate(Transform):
    """Conditionally apply chat template only if example is conversational.

    This is useful when processing datasets that may have mixed formats.

    Example:
        >>> transform = MaybeApplyChatTemplate(tokenizer)
        >>> # Conversational example - template applied
        >>> result = transform({"messages": [{"role": "user", "content": "Hi"}]})
        >>> # Non-conversational example - passed through unchanged
        >>> result = transform({"text": "Hello world"})
    """

    def __init__(self, tokenizer: tp.Any, **chat_template_kwargs):
        """Initialize MaybeApplyChatTemplate.

        Args:
            tokenizer: HuggingFace tokenizer with chat template.
            **chat_template_kwargs: Arguments passed to ChatTemplateTransform.
        """
        self._tokenizer = tokenizer
        self._chat_transform = ChatTemplateTransform(tokenizer, **chat_template_kwargs)

    def __call__(self, example: Example) -> Example:
        """Apply chat template if example is conversational."""
        if is_conversational(example):
            return self._chat_transform(example)
        return example

    def __repr__(self) -> str:
        return "MaybeApplyChatTemplate()"


class ConvertInputOutputToChatML(Transform):
    """Convert input/output conversation format to ChatML messages format.

    This handles datasets like Capybara and Pure-Dove that use:
        {"conversation": [{"input": "...", "output": "..."}, ...]}

    Converts to standard ChatML format:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}

    Example:
        >>> transform = ConvertInputOutputToChatML()
        >>> result = transform({
        ...     "conversation": [
        ...         {"input": "What is 2+2?", "output": "2+2 equals 4."},
        ...         {"input": "Thanks!", "output": "You're welcome!"}
        ...     ]
        ... })
        >>> # {"messages": [
        >>> #     {"role": "user", "content": "What is 2+2?"},
        >>> #     {"role": "assistant", "content": "2+2 equals 4."},
        >>> #     {"role": "user", "content": "Thanks!"},
        >>> #     {"role": "assistant", "content": "You're welcome!"}
        >>> # ]}
    """

    def __init__(
        self,
        input_field: str = "conversation",
        output_field: str = "messages",
        user_role: str = "user",
        assistant_role: str = "assistant",
    ):
        """Initialize ConvertInputOutputToChatML.

        Args:
            input_field: Field containing the conversations (also checks "conversations").
            output_field: Field to store the converted messages.
            user_role: Role name for user turns (default: "user").
            assistant_role: Role name for assistant turns (default: "assistant").
        """
        self._input_field = input_field
        self._output_field = output_field
        self._user_role = user_role
        self._assistant_role = assistant_role

    def __call__(self, example: Example) -> Example:
        """Convert the example to ChatML format."""
        result = example.copy()

        # Find conversation field
        conversation = None
        source_field = None
        for field in [self._input_field, "conversation", "conversations"]:
            if field in result:
                conversation = result[field]
                source_field = field
                break

        if conversation is None:
            return result

        # Convert input/output pairs to messages
        messages = []
        for turn in conversation:
            if "input" in turn:
                messages.append({"role": self._user_role, "content": turn["input"]})
            if "output" in turn:
                messages.append({"role": self._assistant_role, "content": turn["output"]})

        # Remove old field if different from output
        if source_field and source_field != self._output_field:
            result.pop(source_field, None)

        result[self._output_field] = messages
        return result

    def __repr__(self) -> str:
        return f"ConvertInputOutputToChatML({self._input_field!r} -> {self._output_field!r})"


# Default role mapping for common dataset formats to ChatML standard roles
DEFAULT_ROLE_MAPPING: dict[str, str] = {
    # Common variations -> ChatML standard
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}


class ConvertToChatML(Transform):
    """Convert from/value format to ChatML messages format.

    ChatML format is the standard format for chat models:
        {"messages": [{"role": "user"|"assistant"|"system"|"tool", "content": "..."}]}

    This transform handles common variations:
    - "from" -> "role" (with normalization to user/assistant/system/tool)
    - "value" -> "content"
    - "conversations" -> "messages"

    Role normalization (default):
    - "human" -> "user"
    - "gpt", "bot", "model" -> "assistant"
    - "system" -> "system"
    - "function", "tool" -> "tool"

    Example:
        >>> transform = ConvertToChatML()
        >>> result = transform({
        ...     "conversations": [
        ...         {"from": "human", "value": "Hello!"},
        ...         {"from": "gpt", "value": "Hi there!"}
        ...     ]
        ... })
        >>> # {"messages": [
        >>> #     {"role": "user", "content": "Hello!"},
        >>> #     {"role": "assistant", "content": "Hi there!"}
        >>> # ]}
    """

    def __init__(
        self,
        input_field: str = "conversations",
        output_field: str = "messages",
        role_mapping: dict[str, str] | None = None,
        use_default_mapping: bool = True,
    ):
        """Initialize ConvertToChatML.

        Args:
            input_field: Field containing the conversations (also checks "messages").
            output_field: Field to store the converted messages.
            role_mapping: Custom mapping to normalize role names. If use_default_mapping
                is True, these are merged with defaults (custom takes precedence).
            use_default_mapping: Whether to use default role mapping (human->user, etc).
                Set to False to only use custom role_mapping.
        """
        self._input_field = input_field
        self._output_field = output_field

        # Build role mapping
        if use_default_mapping:
            self._role_mapping = DEFAULT_ROLE_MAPPING.copy()
            if role_mapping:
                self._role_mapping.update(role_mapping)
        else:
            self._role_mapping = role_mapping or {}

    def __call__(self, example: Example) -> Example:
        """Convert the example to ChatML format."""
        result = example.copy()

        # Find messages field
        messages = None
        source_field = None
        for field in [self._input_field, "conversations", "messages"]:
            if field in result:
                messages = result[field]
                source_field = field
                break

        if messages is None:
            return result

        # Convert messages
        converted = []
        for msg in messages:
            new_msg = {}

            # Handle role/from
            if "from" in msg:
                role = msg["from"]
            elif "role" in msg:
                role = msg["role"]
            else:
                role = "user"  # Default to user if unknown

            # Apply role mapping (normalize to ChatML standard roles)
            role = self._role_mapping.get(role, role)
            new_msg["role"] = role

            # Handle content/value
            if "value" in msg:
                new_msg["content"] = msg["value"]
            elif "content" in msg:
                new_msg["content"] = msg["content"]
            else:
                new_msg["content"] = ""

            converted.append(new_msg)

        # Remove old field if different from output
        if source_field and source_field != self._output_field:
            result.pop(source_field, None)

        result[self._output_field] = converted
        return result

    def __repr__(self) -> str:
        return f"ConvertToChatML({self._input_field!r} -> {self._output_field!r})"
