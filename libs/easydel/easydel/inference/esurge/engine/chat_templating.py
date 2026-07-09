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

"""Pure chat-template and stop-sequence normalizers for the eSurge engine.

Every function here is stateless: they take OpenAI-style payloads (messages,
tools, stop sequences) and massage them into shapes that HF tokenizer chat
templates can render. :func:`format_chat_prompt` is the composition point
that renders messages through ``tokenizer.apply_chat_template`` with the
shape-mismatch retry ladder.
"""

from __future__ import annotations

import copy
import json
import typing
from typing import Any

from ..logger import logger


def coerce_mapping_like(value: Any) -> Any:
    """Best-effort JSON-decode a string ``value``, returning the original on failure.

    OpenAI-compatible clients sometimes send tool-call ``arguments`` (and
    similar fields) as JSON strings instead of nested objects. The
    chat-template normalizer wants real dicts, so this helper attempts
    ``json.loads`` and silently returns the input when it isn't a
    valid JSON string. Non-string inputs pass through unchanged.

    Args:
        value: Anything; only ``str`` values are inspected.

    Returns:
        The decoded JSON object when ``value`` is a parseable JSON
        string, otherwise ``value`` itself.
    """

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return value
        return parsed
    return value


def normalize_chat_template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Massage OpenAI-style messages into a shape every HF chat template can render.

    Tokenizer Jinja templates assume ``content`` is a string or list,
    ``tool_calls[*].function.arguments`` is a dict, and there is at most
    one leading system message. Real-world OpenAI clients break all
    three regularly: ``content=None`` for assistant tool-call messages,
    ``arguments`` as a JSON string, and multi-turn system messages.
    This pass deep-copies and rewrites the input so the template can
    iterate without throwing — defaulting ``content`` to ``""``,
    json-decoding ``arguments`` into dicts (with a ``{"value": ...}``
    wrapper when decode produces a non-dict), and finally folding all
    system messages into a single leading turn via
    :func:`collapse_system_messages`.

    Args:
        messages: List of OpenAI-style message dicts as accepted by
            :meth:`eSurge.chat`.

    Returns:
        New list of normalized message dicts safe to pass to
        ``tokenizer.apply_chat_template``.
    """

    normalized_messages: list[dict[str, Any]] = []
    for message in messages:
        msg = dict(message)
        if msg.get("content") is None:
            msg["content"] = ""

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized_calls: list[dict[str, Any]] = []
            for raw_call in tool_calls:
                if not isinstance(raw_call, dict):
                    continue
                call = dict(raw_call)
                function_payload = call.get("function")
                if isinstance(function_payload, dict):
                    function_dict = dict(function_payload)
                    arguments = coerce_mapping_like(function_dict.get("arguments"))
                    if arguments is None:
                        arguments = {}
                    if not isinstance(arguments, dict):
                        arguments = {"value": str(arguments)}
                    function_dict["arguments"] = arguments
                    call["function"] = function_dict
                elif isinstance(function_payload, str):
                    coerced = coerce_mapping_like(function_payload)
                    if isinstance(coerced, dict):
                        call["function"] = coerced
                normalized_calls.append(call)
            msg["tool_calls"] = normalized_calls

        function_call = msg.get("function_call")
        if isinstance(function_call, dict):
            fc = dict(function_call)
            arguments = coerce_mapping_like(fc.get("arguments"))
            if isinstance(arguments, dict):
                fc["arguments"] = arguments
            msg["function_call"] = fc

        normalized_messages.append(msg)

    return collapse_system_messages(normalized_messages)


def content_to_text_parts(content: Any) -> list[dict[str, Any]]:
    """Lift any message ``content`` into the structured-parts list shape.

    Strict chat templates (e.g. Gemma's) iterate over ``message["content"]``
    as a list of typed parts (``[{"type": "text", "text": ...}, ...]``)
    rather than a bare string. This helper produces that shape regardless
    of input flavour: ``None`` becomes ``[]``, scalars become a single
    ``"text"`` part, lists are deep-copied (with stringification of any
    non-dict, non-None entries), and existing typed parts pass through
    verbatim.

    Args:
        content: Message ``content`` value as it appeared in the input.

    Returns:
        List of typed-part dicts ready to be merged with other turns.
    """

    if content is None:
        return []
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(copy.deepcopy(item))
            elif item is not None:
                parts.append({"type": "text", "text": str(item)})
        return parts
    return [{"type": "text", "text": str(content)}]


def merge_system_content(existing: Any, new_content: Any) -> Any:
    """Combine two system-message ``content`` values without losing structure.

    Two cases:

    * Either side is a list of typed parts → both are normalized via
      :func:`content_to_text_parts` and concatenated, preserving
      all parts in order.
    * Both sides are scalar strings → joined with ``"\\n\\n"`` (or
      falling back to the non-empty side) so simple text-only system
      messages remain rendered as a single readable turn.

    Args:
        existing: Already-folded system content (or ``None``).
        new_content: System content from the next system turn being
            folded in.

    Returns:
        Merged content compatible with the input shape — list of
        parts, joined string, or ``None`` when both inputs were empty.
    """

    if isinstance(existing, list) or isinstance(new_content, list):
        return content_to_text_parts(existing) + content_to_text_parts(new_content)

    existing_text = "" if existing is None else str(existing)
    new_text = "" if new_content is None else str(new_content)
    if existing_text and new_text:
        return f"{existing_text}\n\n{new_text}"
    return existing_text or new_text


def collapse_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge every ``role == "system"`` turn into a single leading system message.

    Many tokenizer chat templates (Llama, Gemma, Qwen, …) require
    exactly one system message at the start of the conversation and
    will silently drop or misrender extra system turns. This helper
    scans the input, returns it unchanged when at most one system
    message already sits in position 0, and otherwise builds a single
    merged system message via :func:`merge_system_content`, drops
    ``tool_calls`` / ``function_call`` from system turns (templates
    do not expect them), and prepends the merged turn before all
    non-system messages in their original order.

    Args:
        messages: Already field-normalized message list.

    Returns:
        Either the original list or a new list with system messages
        collapsed; logs a WARNING when a collapse actually happened so
        users notice that their multi-system input was rewritten.
    """

    if not messages:
        return messages

    system_indices = [idx for idx, msg in enumerate(messages) if msg.get("role") == "system"]
    if len(system_indices) <= 1 and (not system_indices or system_indices[0] == 0):
        return messages

    merged_system: dict[str, Any] | None = None
    ordered_messages: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "system":
            ordered_messages.append(message)
            continue

        msg_copy = dict(message)
        msg_copy.pop("tool_calls", None)
        msg_copy.pop("function_call", None)
        if merged_system is None:
            merged_system = msg_copy
            continue
        merged_system["content"] = merge_system_content(
            merged_system.get("content"),
            msg_copy.get("content"),
        )

    if merged_system is None:
        return ordered_messages

    if system_indices != [0]:
        logger.warning(
            "Collapsing %d system messages into the first position for chat-template compatibility.",
            len(system_indices),
        )
    return [merged_system, *ordered_messages]


def normalize_chat_template_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Normalize tool definitions for HF chat templates.

    Some templates iterate on nested mapping fields (e.g. ``parameters.items()``).
    This helper hardens tool payloads by ensuring those fields are dictionaries.

    Args:
        tools: Raw tool definitions, either OpenAI-wrapped
            (``{"type": "function", "function": {...}}``) or bare
            function payloads. ``None`` and empty inputs pass through.

    Returns:
        Sanitized list of *unwrapped* function-payload dicts ready for
        templates that iterate them directly, or ``None`` when there
        is nothing usable to return.
    """

    if not tools:
        return None

    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        candidate = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if not isinstance(candidate, dict):
            continue

        payload = dict(candidate)
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            continue

        description = payload.get("description")
        if description is not None and not isinstance(description, str):
            payload["description"] = str(description)

        parameters = payload.get("parameters", {})
        if isinstance(parameters, str):
            try:
                parsed = json.loads(parameters)
            except Exception:
                parsed = {}
            parameters = parsed if isinstance(parsed, dict) else {}
        elif not isinstance(parameters, dict):
            parameters = {}

        properties = parameters.get("properties")
        if isinstance(properties, str):
            try:
                parsed_properties = json.loads(properties)
            except Exception:
                parsed_properties = {}
            properties = parsed_properties if isinstance(parsed_properties, dict) else {}
        elif properties is not None and not isinstance(properties, dict):
            properties = {}
        if properties is not None:
            parameters["properties"] = properties

        required = parameters.get("required")
        if isinstance(required, str):
            parameters["required"] = [required]
        elif required is not None and not isinstance(required, list):
            parameters["required"] = []

        payload["parameters"] = parameters
        normalized.append(payload)

    return normalized or None


def normalize_wrapped_chat_template_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Normalize tools into OpenAI-style ``{"type":"function","function":...}`` wrappers.

    Some tokenizer chat templates, including Gemma-style templates, expect
    each tool entry to expose a ``function`` mapping rather than a bare
    function payload. This helper sanitizes the inner function schema using
    :func:`normalize_chat_template_tools` and wraps it back into the expected
    outer structure.

    Args:
        tools: Raw tool definitions as supplied by the caller.

    Returns:
        List of wrapped tool dicts of the form
        ``{"type": "function", "function": {...}}`` ready for templates
        that expect the OpenAI layout, or ``None`` when nothing usable
        remained after sanitization.
    """

    normalized = normalize_chat_template_tools(tools)
    if not normalized:
        return None

    wrapped: list[dict[str, Any]] = []
    source_tools = [tool for tool in (tools or []) if isinstance(tool, dict)]
    for idx, payload in enumerate(normalized):
        source = source_tools[idx] if idx < len(source_tools) else {}
        tool_type = source.get("type") if isinstance(source.get("type"), str) and source.get("type") else "function"
        wrapped.append({"type": tool_type, "function": payload})
    return wrapped or None


def is_recoverable_chat_template_tool_error(exc: Exception) -> bool:
    """Whether ``exc`` looks like a tool-shape mismatch worth retrying.

    ``apply_chat_template`` failures come in two flavours: configuration
    bugs (missing template, undefined variable) and shape mismatches
    between our normalized tools and a template's expectations
    (e.g. Gemma's wrapped vs unwrapped tool layout). The second class
    is recoverable by reformatting tools and retrying; this predicate
    identifies them by inspecting the exception type and message
    substrings (``KeyError`` / ``TypeError`` referencing
    ``parameters`` / ``properties`` / ``arguments``, plus
    ``UndefinedError`` references to ``name``-style attributes).

    Args:
        exc: Exception raised by ``tokenizer.apply_chat_template``.

    Returns:
        ``True`` for template/tool shape mismatches the caller should
        retry with a different normalization, ``False`` for hard
        errors that should propagate.
    """

    if isinstance(exc, KeyError):
        missing_key = exc.args[0] if len(exc.args) > 0 else None
        if missing_key in {"items", "function"}:
            return True

    exc_text = str(exc)
    if exc_text.strip("'\"") in {"items", "function"}:
        return True

    return (
        "has no attribute 'items'" in exc_text
        or "items" in exc_text
        or "has no attribute 'function'" in exc_text
        or "tool_data['function']" in exc_text
    )


def to_structured_text_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite every message's ``content`` into the typed-parts list shape.

    Used as a retry rewrite when a chat template fails on bare-string
    content (e.g. Gemma's template insists on
    ``content=[{"type":"text","text":"..."}]``). Returns a new list of
    deep-copied message dicts whose ``content`` field has been routed
    through :func:`content_to_text_parts`; ``None`` and empty lists
    are normalized to ``[]`` so the template can still iterate without
    special-casing.

    Args:
        messages: Already field-normalized messages.

    Returns:
        New list of messages with structured-parts content.
    """

    normalized: list[dict[str, Any]] = []
    for message in messages:
        msg = dict(message)
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [{"type": "text", "text": content}]
        elif content is None:
            msg["content"] = []
        elif isinstance(content, dict):
            msg["content"] = [content]
        elif isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for part in content:
                if isinstance(part, str):
                    parts.append({"type": "text", "text": part})
                    continue
                if not isinstance(part, dict):
                    parts.append({"type": "text", "text": str(part)})
                    continue
                part_type = part.get("type")
                if part_type in ("text", "input_text", "output_text"):
                    text = part.get("text", part.get("content", ""))
                    parts.append({"type": "text", "text": "" if text is None else str(text)})
                else:
                    parts.append(part)
            msg["content"] = parts
        normalized.append(msg)
    return normalized


def normalize_stop_sequences(stop: typing.Any) -> list[str]:
    """Coerce a heterogeneous stop-sequence input into a clean ``list[str]``.

    Accepts a single string, an iterable of strings, or ``None``, and
    produces a deduplicated list (preserving insertion order) of
    non-empty strings. ``None`` and falsy inputs round-trip to ``[]``.
    Used both to merge engine-level ``extra_stops`` into per-request
    sampling params and to normalize user-supplied ``stop`` arguments.

    Args:
        stop: Raw stop-sequence input — string, iterable of strings,
            or ``None``.

    Returns:
        Deduplicated, order-preserving list of non-empty stop strings.
    """

    if stop is None:
        return []
    if isinstance(stop, str):
        candidates = [stop]
    elif isinstance(stop, (list, tuple, set)):
        candidates = list(stop)
    else:
        candidates = [stop]

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        value = candidate if isinstance(candidate, str) else str(candidate)
        if value == "" or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def format_chat_prompt(
    tokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
    chat_template: str | None = None,
    tools: list[dict] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Format chat messages into a prompt string using the tokenizer's chat template.

    Converts a list of chat messages into a formatted prompt string that can be
    passed to the model for generation. Uses the tokenizer's built-in chat template
    or a custom template if provided.

    Args:
        tokenizer: Tokenizer exposing ``apply_chat_template``.
        messages: List of message dictionaries with 'role' and 'content' keys.
            Roles can be 'system', 'user', 'assistant', etc.
        add_generation_prompt: Whether to add the generation prompt token/string
            at the end to indicate the model should generate a response.
        chat_template: Optional custom chat template to override the tokenizer's
            default template. Should be a Jinja2 template string.
        tools: Optional list of tool/function definitions that the model can use.
            Format depends on the specific model's tool calling conventions.
        chat_template_kwargs: Optional extra keyword arguments forwarded
            to ``tokenizer.apply_chat_template`` (e.g. template-specific
            toggles like ``enable_thinking``).

    Returns:
        Formatted prompt string ready for tokenization and generation.

    Raises:
        Exception: Propagates :meth:`tokenizer.apply_chat_template`
            failures that remain unrecoverable after the tool/messages
            shape fallbacks have been exhausted.

    Note:
        The exact format depends on the tokenizer's chat template. Different models
        use different special tokens and formatting conventions.
    """

    if chat_template_kwargs is None:
        chat_template_kwargs = {}
    normalized_messages = normalize_chat_template_messages(messages)
    normalized_tools = normalize_chat_template_tools(tools)
    normalized_wrapped_tools = normalize_wrapped_chat_template_tools(tools)
    try:
        return tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=False,
            tools=normalized_tools,
            add_generation_prompt=add_generation_prompt,
            chat_template=chat_template,
            **chat_template_kwargs,
        )
    except Exception as exc:
        # Some tokenizer chat templates expect `.items()` on nested mappings,
        # while others expect OpenAI-style tool wrappers with `.function`.
        recoverable_tool_error = is_recoverable_chat_template_tool_error(exc)
        if tools is None or not recoverable_tool_error:
            raise

        structured_messages = to_structured_text_messages(normalized_messages)

        retries = [
            ("wrapped_tools", normalized_messages, normalized_wrapped_tools),
            ("structured_messages", structured_messages, normalized_tools),
            ("structured_messages+wrapped_tools", structured_messages, normalized_wrapped_tools),
            ("structured_messages_no_tools", structured_messages, None),
        ]
        if len(normalized_tools or []) != len(tools or []):
            logger.warning(
                "Malformed tool entries detected in chat template tools (kept=%d dropped=%d); trying sanitized fallback.",
                len(normalized_tools or []),
                len(tools or []) - len(normalized_tools or []),
            )

        last_error: Exception = exc
        for label, retry_messages, retry_tools in retries:
            try:
                prompt = tokenizer.apply_chat_template(
                    retry_messages,
                    tokenize=False,
                    tools=retry_tools,
                    add_generation_prompt=add_generation_prompt,
                    chat_template=chat_template,
                    **chat_template_kwargs,
                )
                logger.warning("Recovered chat template rendering via %s fallback.", label)
                return prompt
            except Exception as retry_exc:
                if not is_recoverable_chat_template_tool_error(retry_exc):
                    raise
                last_error = retry_exc

        raise last_error from exc
