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
    """Heuristic check that distinguishes chat-format rows from plain-text rows.

    Looks for any of the recognised chat columns
    (``"prompt"``, ``"chosen"``, ``"rejected"``, ``"completion"``,
    ``"messages"``, ``"conversations"``) and accepts the row as
    conversational when at least one of those columns holds a
    non-empty list whose first element is a dict with a ``"role"``
    or ``"from"`` key. The check is intentionally cheap (constant
    time per row) so it can be applied per-row by
    :class:`MaybeApplyChatTemplate`.

    Args:
        example: Row dict to inspect.

    Returns:
        bool: ``True`` if the row's shape matches a chat-style
        schema, ``False`` for plain-text and other schemas.
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
    """Re-key from/value-style messages into ChatML's role/content shape.

    Many open-source datasets follow the ShareGPT convention
    (``{"from": ..., "value": ...}``) instead of the OpenAI/ChatML
    convention (``{"role": ..., "content": ...}``). This helper
    rewrites the keys without touching message order or values.
    Messages already in role/content shape are passed through
    unchanged so the function is safe to call defensively.

    Args:
        messages: Sequence of message dicts; entries may be in either
            shape and are converted independently.

    Returns:
        list[dict]: New list of messages in ``role``/``content``
        shape; the originals are not mutated.
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
    """Render conversational rows to model-ready text using the tokenizer's chat template.

    Reads the configured messages field
    (``messages_field``, with fallbacks to ``"conversations"`` /
    ``"conversation"``), optionally re-keys ShareGPT-style
    ``from``/``value`` entries to ChatML ``role``/``content``, and
    invokes ``tokenizer.apply_chat_template(messages, tokenize=False, ...)``
    to produce a single rendered string. The result is written to
    ``output_field`` (default ``"text"``); when the template raises,
    a minimal ``"role: content"`` fallback string is produced so
    downstream tokenization can still proceed (with a warning).

    Use this transform once chat data has already been normalised to
    role/content shape. For mixed-format datasets, wrap with
    :class:`MaybeApplyChatTemplate`. For ShareGPT/OpenAI variant
    inputs, prepend :class:`ConvertToChatML`.

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
        """Capture template settings without invoking the tokenizer.

        Args:
            tokenizer: HuggingFace ``PreTrainedTokenizerBase`` (or
                compatible) exposing ``apply_chat_template``. The
                tokenizer must define a chat template — the transform
                falls back to a simple formatter when it raises but
                logs a warning.
            messages_field: Primary row key holding the message list.
                ``"conversations"`` and ``"conversation"`` are tried
                as fallbacks before giving up.
            output_field: Row key under which the rendered string is
                stored. Defaults to ``"text"`` so downstream
                tokenization picks it up without configuration.
            tools: Optional tools/functions list forwarded as
                ``tools=`` to ``apply_chat_template`` (for tokenizers
                that support function-calling templates). ``None``
                disables tool rendering.
            convert_from_value_format: When ``True``, ShareGPT-style
                ``from``/``value`` messages are re-keyed via
                :func:`convert_to_chatml` before rendering; when
                ``False``, messages are passed through verbatim.
            drop_messages: When ``True``, removes the original source
                field after rendering so downstream stages do not see
                two copies of the conversation.
            **template_kwargs: Extra keyword arguments forwarded
                verbatim to ``apply_chat_template`` —
                ``add_generation_prompt``, ``continue_final_message``,
                custom template flags, etc.
        """
        self._tokenizer = tokenizer
        self._messages_field = messages_field
        self._output_field = output_field
        self._tools = tools
        self._convert_from_value = convert_from_value_format
        self._drop_messages = drop_messages
        self._template_kwargs = template_kwargs

    def __call__(self, example: Example) -> Example:
        """Render the example's messages through the tokenizer's chat template.

        Walks the source field fallbacks (``messages_field`` then
        ``"conversations"`` then ``"conversation"``) and stops at the
        first one present. Optionally normalises ShareGPT-style
        messages, then invokes ``apply_chat_template``. On failure,
        falls back to :meth:`_simple_format` and emits a warning.

        Args:
            example: Row dict containing the message list under one
                of the recognised keys.

        Returns:
            dict: A copy of ``example`` with the rendered string in
            ``output_field`` and the original messages key removed
            when ``drop_messages`` is ``True``. Rows with no
            recognised messages key are returned unchanged.
        """
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
        """Render messages with a minimal ``role: content`` formatter.

        Used as a fallback when the tokenizer's chat template raises.

        Args:
            messages: List of role/content message dicts (from/value also
                supported).

        Returns:
            Newline-joined string of ``"<role>: <content>"`` lines.
        """
        parts = []
        for msg in messages:
            role = msg.get("role", msg.get("from", "unknown"))
            content = msg.get("content", msg.get("value", ""))
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            ``"ChatTemplateTransform(messages_field=..., output_field=...)"``.
        """
        return f"ChatTemplateTransform(messages_field={self._messages_field!r}, output_field={self._output_field!r})"


class MaybeApplyChatTemplate(Transform):
    """Per-row guard that applies a chat template only when the row looks conversational.

    Useful when a single dataset is heterogeneous — e.g. an SFT mix
    that combines plain ``text`` rows with chat ``messages`` rows.
    The transform tests :func:`is_conversational` on each row and
    forwards plain rows untouched, applying the wrapped
    :class:`ChatTemplateTransform` only when the row matches the
    chat shape.

    Example:
        >>> transform = MaybeApplyChatTemplate(tokenizer)
        >>> # Conversational example - template applied
        >>> result = transform({"messages": [{"role": "user", "content": "Hi"}]})
        >>> # Non-conversational example - passed through unchanged
        >>> result = transform({"text": "Hello world"})
    """

    def __init__(self, tokenizer: tp.Any, **chat_template_kwargs):
        """Build the wrapped :class:`ChatTemplateTransform` once at construction time.

        Args:
            tokenizer: HuggingFace tokenizer with a chat template;
                forwarded to :class:`ChatTemplateTransform`.
            **chat_template_kwargs: Forwarded verbatim to the
                wrapped transform — same vocabulary as
                :class:`ChatTemplateTransform.__init__`.
        """
        self._tokenizer = tokenizer
        self._chat_transform = ChatTemplateTransform(tokenizer, **chat_template_kwargs)

    def __call__(self, example: Example) -> Example:
        """Run the chat template only for chat-shaped rows; pass plain rows through.

        Args:
            example: Row dict; may be plain text or chat shape.

        Returns:
            dict: The result of the wrapped chat-template transform
            for chat-shaped rows, or the original ``example``
            object (no copy) for non-chat rows.
        """
        if is_conversational(example):
            return self._chat_transform(example)
        return example

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            ``"MaybeApplyChatTemplate()"``.
        """
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
        """Capture the schema field names and role labels used during conversion.

        Args:
            input_field: Primary row key holding the input/output
                conversation turns. Falls back to ``"conversation"``
                and ``"conversations"`` when the primary key is absent.
            output_field: Row key under which the produced messages
                list is stored. Defaults to ``"messages"`` to align
                with ChatML conventions.
            user_role: Role tag emitted for ``input`` text. Defaults
                to ``"user"`` to match ChatML; some templates may
                expect alternates such as ``"human"``.
            assistant_role: Role tag emitted for ``output`` text.
                Defaults to ``"assistant"``.
        """
        self._input_field = input_field
        self._output_field = output_field
        self._user_role = user_role
        self._assistant_role = assistant_role

    def __call__(self, example: Example) -> Example:
        """Expand each ``{input, output}`` turn into a ``user`` + ``assistant`` message pair.

        Walks the configured input field (with fallbacks) and
        translates every turn into one or two ChatML messages
        (``user`` from ``input``, ``assistant`` from ``output``);
        turns missing one side produce only the present side. The
        original input field is removed when it differs from
        ``output_field``.

        Args:
            example: Row dict expected to carry an
                ``input``/``output`` turn list under one of the
                recognised keys.

        Returns:
            dict: Copy of ``example`` with the rendered messages list
            in ``output_field``. Rows with no recognised conversation
            field are returned unchanged.
        """
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
        """Return a developer-friendly representation.

        Returns:
            ``"ConvertInputOutputToChatML('input_field' -> 'output_field')"``.
        """
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
        """Capture the schema field names and assemble the role-normalisation table.

        Args:
            input_field: Primary row key holding the conversation
                list; ``"conversations"`` and ``"messages"`` are
                tried as fallbacks during ``__call__``.
            output_field: Row key under which the converted ChatML
                messages are stored. Defaults to ``"messages"``.
            role_mapping: Per-call role rename map (e.g.
                ``{"human": "user", "gpt": "assistant"}``). When
                ``use_default_mapping`` is ``True`` the map is merged
                on top of :data:`DEFAULT_ROLE_MAPPING` with custom
                entries taking precedence; otherwise only the custom
                map is consulted.
            use_default_mapping: When ``True`` (default), include the
                shipped :data:`DEFAULT_ROLE_MAPPING` entries
                (``human -> user``, ``gpt -> assistant`` etc.) so
                common ShareGPT corpora work out-of-the-box.
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
        """Re-key every message to ``role``/``content`` and normalise role names.

        For each message in the source list, picks the role from
        ``"from"`` or ``"role"`` (defaulting to ``"user"`` if neither
        is present), runs it through the role rename map, and
        copies the body from ``"value"`` or ``"content"``
        (defaulting to ``""``). The original input field is removed
        when it differs from ``output_field`` so downstream stages
        do not see two copies of the conversation.

        Args:
            example: Row dict expected to carry messages under one
                of ``input_field``, ``"conversations"``, or
                ``"messages"``.

        Returns:
            dict: Copy of ``example`` with the converted messages
            list under ``output_field``. Rows with no recognised
            conversation field are returned unchanged.
        """
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
        """Return a developer-friendly representation.

        Returns:
            ``"ConvertToChatML('input_field' -> 'output_field')"``.
        """
        return f"ConvertToChatML({self._input_field!r} -> {self._output_field!r})"
