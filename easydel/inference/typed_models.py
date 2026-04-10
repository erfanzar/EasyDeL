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

"""Typed Pydantic models for the OpenAI Responses API surface.

This module defines the data-model layer that backs EasyDeL's implementation of
the OpenAI *Responses* API (``/v1/responses``).  Unlike the older Chat
Completions API — which returns a single message per choice — the Responses API
streams a sequence of **output items** (reasoning summaries, function calls,
assistant messages) wrapped in typed **event frames**.

The module is split into three logical sections:

1. **Output-item models** — ``ResponseReasoningItem``, ``ResponseFunctionCallItem``,
   ``ResponseMessageItem`` — which are the building blocks of a response's
   ``output`` list.  They share a discriminated-union type alias
   ``ResponsesOutputItem`` so Pydantic can round-trip them through JSON.

2. **Stream-event payload models** — one model per server-sent event type
   (``response.created``, ``response.output_text.delta``, etc.).  Each payload
   is paired with an event name inside a ``StreamEventFrame`` dataclass.

3. **Accumulator state dataclasses** — ``ReasoningStreamState``,
   ``FunctionCallStreamState``, ``MessageStreamState`` — which the
   ``ResponsesStreamAccumulator`` (in ``stream_protocol.py``) uses to track
   partially-built items across multiple ``add_output`` calls.

All models inherit from ``OpenAIBaseModel`` so they serialize with the same
``model_dump`` / ``model_validate`` conventions used in the rest of the
inference server.
"""

from __future__ import annotations

import typing as tp
import uuid
from dataclasses import dataclass

from pydantic import Field

from .openai_api_modules import (
    ChatMessage,
    FunctionCall,
    FunctionDefinition,
    OpenAIBaseModel,
    ToolCall,
    ToolChoiceOption,
    ToolDefinition,
)


class ResponseSummaryText(OpenAIBaseModel):
    """A single block of summary text inside a ``ResponseReasoningItem``.

    The Responses API allows a reasoning item to carry one or more summary
    blocks (indexed by ``summary_index`` in the delta events).  In practice
    eSurge always emits exactly one summary per reasoning item, but the list
    structure is kept for forward-compatibility with the OpenAI spec.
    """

    type: tp.Literal["summary_text"] = "summary_text"
    text: str = ""


class ResponseOutputTextPart(OpenAIBaseModel):
    """A single text content-part within a ``ResponseMessageItem``.

    Maps to the ``output_text`` discriminator in the Responses API.
    ``annotations`` and ``logprobs`` are placeholder lists reserved for
    future use (e.g. citation annotations, per-token log-probabilities).
    """

    type: tp.Literal["output_text"] = "output_text"
    annotations: list[tp.Any] = Field(default_factory=list)
    logprobs: list[tp.Any] = Field(default_factory=list)
    text: str = ""


class ResponseReasoningItem(OpenAIBaseModel):
    """Output item representing chain-of-thought reasoning from the model.

    This item is emitted into the ``output`` list of a ``ResponsesResponse``
    when the engine detects a reasoning section (e.g. ``<think>…</think>``
    blocks) and ``include_reasoning_summary`` is enabled on the request.
    The reasoning text is stored inside a ``summary`` list of
    ``ResponseSummaryText`` blocks rather than as a flat string so that
    streaming can address individual blocks by index.

    During streaming the item starts with ``status="in_progress"`` and
    transitions to ``"completed"`` once the reasoning boundary is closed.
    """

    id: str = Field(default_factory=lambda: f"rs_{uuid.uuid4().hex}")
    type: tp.Literal["reasoning"] = "reasoning"
    summary: list[ResponseSummaryText] = Field(default_factory=list)
    status: str | None = None


class ResponseFunctionCallItem(OpenAIBaseModel):
    """Output item representing a single function (tool) call.

    Each tool call extracted by the tool parser becomes one of these items
    in the response's ``output`` list.  ``call_id`` is the unique identifier
    the client uses to match the tool result back, ``name`` is the function
    name, and ``arguments`` is the JSON-serialized argument string.

    During streaming the item starts with ``status="in_progress"`` while
    arguments are still arriving, then transitions to ``"completed"`` when
    the ``response.function_call_arguments.done`` event is emitted.
    """

    id: str = Field(default_factory=lambda: f"fc_{uuid.uuid4().hex}")
    type: tp.Literal["function_call"] = "function_call"
    call_id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex}")
    name: str = ""
    arguments: str = ""
    status: str = "completed"


class ResponseMessageItem(OpenAIBaseModel):
    """Output item representing an assistant text message.

    This is the most common item type — it holds the visible text the
    model produced after reasoning and tool-call markup have been stripped.
    ``content`` is a list of ``ResponseOutputTextPart`` (always length 1
    in the current implementation) to match the OpenAI spec's multi-part
    structure.

    A message item is emitted whenever there is visible text output *or*
    when no tool calls were made (so the client always receives at least
    one output item).
    """

    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    type: tp.Literal["message"] = "message"
    role: str = "assistant"
    content: list[ResponseOutputTextPart] = Field(default_factory=list)
    status: str = "completed"


ResponsesOutputItem = tp.Annotated[
    ResponseReasoningItem | ResponseFunctionCallItem | ResponseMessageItem,
    Field(discriminator="type"),
]


class ResponsesUsage(OpenAIBaseModel):
    """Token-level usage statistics attached to a ``ResponsesResponse``.

    Mirrors the ``usage`` block from the OpenAI Responses API.  Unlike the
    Chat Completions ``UsageInfo`` (which also carries throughput metrics),
    this model is intentionally minimal — just raw token counts — because
    throughput data is not part of the Responses spec.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ResponsesTextFormat(OpenAIBaseModel):
    """Text-format discriminator used inside ``ResponsesTextConfig``.

    Currently only ``"text"`` is supported.  The field exists so the
    serialized JSON matches the ``text.format.type`` nesting that the
    OpenAI Responses API returns, allowing clients to parse responses
    without special-casing EasyDeL output.
    """

    type: tp.Literal["text"] = "text"


class ResponsesTextConfig(OpenAIBaseModel):
    """Configuration block describing the text format of a response.

    Wraps a ``ResponsesTextFormat`` and is attached to every
    ``ResponsesResponse``.  This mirrors the ``text`` field in the OpenAI
    spec and is present primarily for wire-format compatibility.
    """

    format: ResponsesTextFormat = Field(default_factory=ResponsesTextFormat)


class ResponsesResponse(OpenAIBaseModel):
    """Top-level object returned by the ``/v1/responses`` endpoint.

    Represents a complete (or in-progress) response from the model.
    The ``output`` list contains an ordered sequence of
    ``ResponsesOutputItem`` instances — reasoning summaries, function calls,
    and the final assistant message — that together describe everything the
    model produced.

    During non-streaming requests this object is returned once in its
    entirety.  During streaming it is first sent inside a
    ``response.created`` event (with ``status="in_progress"`` and an empty
    ``output``) and then again inside the final ``response.completed``
    event with all fields populated.

    Fields like ``instructions``, ``temperature``, ``tool_choice``, etc.
    are echo-back fields: they reflect the parameters the caller set on
    the request so that downstream consumers can inspect them without
    needing access to the original request payload.
    """

    id: str
    object: str = "response"
    created_at: int
    model: str
    status: str
    output: list[ResponsesOutputItem] = Field(default_factory=list)
    usage: ResponsesUsage | None = None
    error: tp.Any = None
    incomplete_details: tp.Any = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    previous_response_id: str | None = None
    store: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    truncation: str | None = None
    tool_choice: str | ToolChoiceOption | None = None
    tools: list[ToolDefinition | FunctionDefinition | tp.Any] = Field(default_factory=list)
    parallel_tool_calls: bool | None = None
    metadata: dict[str, tp.Any] = Field(default_factory=dict)
    text: ResponsesTextConfig = Field(default_factory=ResponsesTextConfig)


class ResponsesFinalizationOptions(OpenAIBaseModel):
    """Optional overrides applied to the final ``ResponsesResponse`` at stream close.

    When the ``ResponsesStreamAccumulator`` (in ``stream_protocol.py``)
    builds the terminal ``response.completed`` event it merges these
    overrides into the response object via ``model_copy(update=…)``.
    This allows the server to inject request-echo fields (``instructions``,
    ``tool_choice``, ``metadata``, etc.) that are not known until the
    request handler processes the original payload.

    All fields default to ``None`` / empty so that only explicitly-set
    values overwrite the response.
    """

    error: tp.Any = None
    incomplete_details: tp.Any = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    previous_response_id: str | None = None
    store: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    truncation: str | None = None
    tool_choice: str | ToolChoiceOption | None = None
    tools: list[ToolDefinition | FunctionDefinition | tp.Any] = Field(default_factory=list)
    parallel_tool_calls: bool | None = None
    metadata: dict[str, tp.Any] = Field(default_factory=dict)
    text: ResponsesTextConfig = Field(default_factory=ResponsesTextConfig)

    def as_update_dict(self) -> dict[str, tp.Any]:
        """Serialize all fields into a plain dict for ``ResponsesResponse.model_copy(update=…)``.

        Mutable containers (``tools``, ``metadata``) are shallow-copied so
        that callers cannot accidentally mutate the finalization options
        through the response object.
        """
        return {
            "error": self.error,
            "incomplete_details": self.incomplete_details,
            "instructions": self.instructions,
            "max_output_tokens": self.max_output_tokens,
            "previous_response_id": self.previous_response_id,
            "store": self.store,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "truncation": self.truncation,
            "tool_choice": self.tool_choice,
            "tools": list(self.tools),
            "parallel_tool_calls": self.parallel_tool_calls,
            "metadata": dict(self.metadata),
            "text": self.text,
        }


class ResponseCreatedEvent(OpenAIBaseModel):
    """Payload for the ``response.created`` SSE event.

    This is always the first event in a Responses API stream.  It carries
    a skeleton ``ResponsesResponse`` with ``status="in_progress"`` and an
    empty ``output`` list so the client knows the response ID and model
    before any content arrives.
    """

    type: tp.Literal["response.created"] = "response.created"
    response: ResponsesResponse


class ResponseOutputItemAddedEvent(OpenAIBaseModel):
    """Payload for the ``response.output_item.added`` SSE event.

    Emitted once for each new output item (reasoning, function call, or
    message) as soon as it is created.  ``output_index`` gives the item's
    position in the response ``output`` list so the client can build its
    own mirror.
    """

    type: tp.Literal["response.output_item.added"] = "response.output_item.added"
    output_index: int
    item: ResponsesOutputItem


class ResponseReasoningSummaryTextDeltaEvent(OpenAIBaseModel):
    """Payload for the ``response.reasoning_summary_text.delta`` SSE event.

    Carries an incremental chunk of reasoning-summary text.  The client
    should concatenate ``delta`` values to reconstruct the full summary.
    ``summary_index`` identifies which summary block within the reasoning
    item is being extended (always 0 in the current implementation).
    """

    type: tp.Literal["response.reasoning_summary_text.delta"] = "response.reasoning_summary_text.delta"
    output_index: int
    item_id: str
    summary_index: int
    delta: str


class ResponseReasoningSummaryTextDoneEvent(OpenAIBaseModel):
    """Payload for the ``response.reasoning_summary_text.done`` SSE event.

    Emitted once when the reasoning summary text is finalized.  ``text``
    contains the complete summary so the client can verify its accumulated
    value and discard partial state.
    """

    type: tp.Literal["response.reasoning_summary_text.done"] = "response.reasoning_summary_text.done"
    output_index: int
    item_id: str
    summary_index: int
    text: str


class ResponseFunctionCallArgumentsDeltaEvent(OpenAIBaseModel):
    """Payload for the ``response.function_call_arguments.delta`` SSE event.

    Streams an incremental chunk of the JSON-serialized function-call
    arguments for one tool call.  The client concatenates ``delta``
    strings and JSON-parses the result once the corresponding ``done``
    event arrives.
    """

    type: tp.Literal["response.function_call_arguments.delta"] = "response.function_call_arguments.delta"
    output_index: int
    item_id: str
    delta: str


class ResponseFunctionCallArgumentsDoneEvent(OpenAIBaseModel):
    """Payload for the ``response.function_call_arguments.done`` SSE event.

    Emitted once per function call when its arguments are complete.
    ``arguments`` contains the full JSON string so the client can
    validate against its accumulated deltas.
    """

    type: tp.Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"
    output_index: int
    item_id: str
    arguments: str


class ResponseContentPartAddedEvent(OpenAIBaseModel):
    """Payload for the ``response.content_part.added`` SSE event.

    Signals that a new content part (text block) has been opened inside a
    message output item.  The ``part`` field carries an empty
    ``ResponseOutputTextPart`` so the client can create its local
    representation before text deltas start flowing.
    """

    type: tp.Literal["response.content_part.added"] = "response.content_part.added"
    output_index: int
    item_id: str
    content_index: int
    part: ResponseOutputTextPart


class ResponseOutputTextDeltaEvent(OpenAIBaseModel):
    """Payload for the ``response.output_text.delta`` SSE event.

    Carries an incremental chunk of visible assistant text.  This is the
    Responses API equivalent of ``choices[0].delta.content`` in the Chat
    Completions streaming format.  ``content_index`` identifies which
    content part the delta belongs to (always 0 today).
    """

    type: tp.Literal["response.output_text.delta"] = "response.output_text.delta"
    output_index: int
    item_id: str
    content_index: int
    delta: str


class ResponseOutputTextDoneEvent(OpenAIBaseModel):
    """Payload for the ``response.output_text.done`` SSE event.

    Emitted when all text for a content part has been sent.  ``text``
    contains the full accumulated string so the client can replace its
    delta-assembled buffer with an authoritative copy.
    """

    type: tp.Literal["response.output_text.done"] = "response.output_text.done"
    output_index: int
    item_id: str
    content_index: int
    text: str


class ResponseOutputItemDoneEvent(OpenAIBaseModel):
    """Payload for the ``response.output_item.done`` SSE event.

    Marks a single output item (reasoning, function call, or message) as
    finalized.  ``item`` carries the complete, up-to-date item model so the
    client can reconcile its local state.  Every item that was opened with
    ``response.output_item.added`` will eventually receive a corresponding
    ``done`` event.
    """

    type: tp.Literal["response.output_item.done"] = "response.output_item.done"
    output_index: int
    item: ResponsesOutputItem


class ResponseCompletedEvent(OpenAIBaseModel):
    """Payload for the ``response.completed`` SSE event.

    This is always the last event in a Responses API stream.  It carries
    the fully-populated ``ResponsesResponse`` (with usage, all output
    items, and finalization overrides applied) so the client has an
    authoritative snapshot identical to what a non-streaming request
    would have returned.
    """

    type: tp.Literal["response.completed"] = "response.completed"
    response: ResponsesResponse


ResponsesStreamEventPayload = (
    ResponseCreatedEvent
    | ResponseOutputItemAddedEvent
    | ResponseReasoningSummaryTextDeltaEvent
    | ResponseReasoningSummaryTextDoneEvent
    | ResponseFunctionCallArgumentsDeltaEvent
    | ResponseFunctionCallArgumentsDoneEvent
    | ResponseContentPartAddedEvent
    | ResponseOutputTextDeltaEvent
    | ResponseOutputTextDoneEvent
    | ResponseOutputItemDoneEvent
    | ResponseCompletedEvent
)


@dataclass
class StreamEventFrame:
    """A single server-sent event in the Responses API stream.

    Pairs an event name (e.g. ``"response.output_text.delta"``) with its
    typed payload model.  The ``eSurgeApiServer`` serializes each frame
    as an SSE line: ``event: {event}\\ndata: {json(payload)}\\n\\n``.
    """

    event: str
    payload: ResponsesStreamEventPayload


@dataclass
class ReasoningStreamState:
    """Mutable accumulator state for a reasoning output item being streamed.

    Created by ``ResponsesStreamAccumulator._ensure_reasoning_item`` the
    first time a reasoning delta arrives.  The accumulator appends text
    to ``item.summary[0].text`` on each delta and flips ``done=True``
    after emitting the ``response.reasoning_summary_text.done`` event so
    that finalize does not duplicate it.
    """

    item: ResponseReasoningItem
    output_index: int
    done: bool = False

    @property
    def item_id(self) -> str:
        """Stable unique identifier for this reasoning item across all delta events."""
        return self.item.id


@dataclass
class FunctionCallStreamState:
    """Mutable accumulator state for a function-call output item being streamed.

    One instance is created per tool call (keyed by ``call_id`` or
    positional index) when the first arguments delta arrives.  The
    accumulator concatenates argument chunks into ``item.arguments``
    and marks ``done=True`` after the ``response.function_call_arguments.done``
    event is emitted during finalization.
    """

    item: ResponseFunctionCallItem
    output_index: int
    done: bool = False

    @property
    def item_id(self) -> str:
        """Stable unique identifier for this function-call item across all delta events."""
        return self.item.id


@dataclass
class MessageStreamState:
    """Mutable accumulator state for a message output item being streamed.

    Created the first time visible assistant text arrives.  The
    accumulator appends text deltas and, during finalization, replaces
    the item's content list with a single ``ResponseOutputTextPart``
    containing the complete text.  ``content_index`` is always 0 in the
    current implementation but is kept for spec-compatibility with
    multi-part messages.
    """

    item: ResponseMessageItem
    output_index: int
    done: bool = False
    content_index: int = 0

    @property
    def item_id(self) -> str:
        """Stable unique identifier for this message item across all delta events."""
        return self.item.id


def assistant_message_from_output_items(output_items: list[ResponsesOutputItem]) -> ChatMessage:
    """Collapse a list of Responses API output items into a single ``ChatMessage``.

    This is needed when the server stores a completed Responses API
    response in conversation history and later needs to inject it as
    an ``assistant`` message in a follow-up Chat Completions request.

    The function iterates over each output item:

    - ``ResponseMessageItem``: text parts are concatenated into the
      message's ``content`` string.
    - ``ResponseFunctionCallItem``: converted into a ``ToolCall`` and
      appended to the message's ``tool_calls`` list.
    - ``ResponseReasoningItem``: skipped (reasoning is not part of the
      visible conversation history).

    Raw dicts are accepted alongside model instances so that items
    round-tripped through JSON (e.g. from a response store) work without
    an explicit deserialization step.
    """
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for raw_item in output_items:
        item: ResponsesOutputItem | None
        if isinstance(raw_item, (ResponseReasoningItem, ResponseFunctionCallItem, ResponseMessageItem)):
            item = raw_item
        elif isinstance(raw_item, dict):
            item_type = raw_item.get("type")
            if item_type == "reasoning":
                item = ResponseReasoningItem.model_validate(raw_item)
            elif item_type == "function_call":
                item = ResponseFunctionCallItem.model_validate(raw_item)
            elif item_type == "message":
                item = ResponseMessageItem.model_validate(raw_item)
            else:
                item = None
        else:
            item = None

        if item is None:
            continue
        if isinstance(item, ResponseMessageItem):
            text_parts.extend(part.text for part in item.content)
            continue
        if isinstance(item, ResponseFunctionCallItem) and item.name.strip():
            tool_calls.append(
                ToolCall(
                    id=item.call_id,
                    type="function",
                    function=FunctionCall(name=item.name, arguments=item.arguments),
                )
            )

    assistant_content = "".join(text_parts)
    return ChatMessage(
        role="assistant",
        content=assistant_content if assistant_content or not tool_calls else None,
        tool_calls=tool_calls or None,
    )
