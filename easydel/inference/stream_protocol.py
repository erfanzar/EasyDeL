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

"""Streaming protocol helpers for OpenAI-compatible and Responses API output.

The eSurge inference engine produces a stream of ``RequestOutput`` snapshots
(one per generated token or micro-batch).  This module provides the
conversion layer that turns those snapshots into the two streaming wire
formats EasyDeL exposes:

1. **OpenAI Chat Completions SSE** — ``iter_chat_completion_stream_responses``
   yields ``ChatCompletionStreamResponse`` objects that serialize to the
   ``data: {json}`` lines expected by ``/v1/chat/completions?stream=true``.

2. **OpenAI Responses API SSE** — ``iter_responses_stream_frames`` (and its
   underlying ``ResponsesStreamAccumulator``) yield ``StreamEventFrame``
   objects for the ``/v1/responses`` streaming protocol, which uses typed
   event names (``response.output_text.delta``, ``response.completed``, …).

Supporting both formats requires a shared set of primitive operations —
computing safe text deltas, normalizing heterogeneous tool-call
representations, coercing delta messages — that are collected here as
module-level functions so they can be reused by the API server, the
engine parsing mixin, and the delegating parser.
"""

from __future__ import annotations

import json
import time
import typing as tp
from typing import Protocol

from .openai_api_modules import (
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    ChatMessage,
    DeltaMessage,
    DeltaToolCall,
    ToolCall,
    UsageInfo,
)
from .typed_models import (
    FunctionCallStreamState,
    MessageStreamState,
    ReasoningStreamState,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionCallItem,
    ResponseMessageItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseOutputTextPart,
    ResponseReasoningItem,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponsesFinalizationOptions,
    ResponsesOutputItem,
    ResponsesResponse,
    ResponsesTextConfig,
    ResponseSummaryText,
    ResponsesUsage,
    StreamEventFrame,
    assistant_message_from_output_items,
)


class CompletionOutputLike(Protocol):
    """Structural protocol describing a single completion sample from the engine.

    Each ``RequestOutputLike`` contains one or more ``CompletionOutputLike``
    entries in its ``outputs`` list (one per ``n`` or beam).  The protocol
    is intentionally duck-typed so that both the real ``CompletionOutput``
    dataclass and lightweight test stubs satisfy it without inheritance.
    """

    finish_reason: str | None
    tool_calls: list[ToolCall] | None
    reasoning_content: str | None


class RequestOutputLike(Protocol):
    """Structural protocol describing an engine output snapshot.

    The eSurge engine emits one of these per decoding step.  It carries
    both *accumulated* state (``accumulated_text``, ``tool_calls``) and
    *delta* state (``delta_text``, ``delta_tool_calls``) so that
    consumers can choose between snapshot-based and incremental
    processing.  Prompt token IDs may be flat or segmented (for
    multi-segment prompts), so both shapes are accepted.
    """

    prompt_token_ids: list[list[int]] | list[int]
    outputs: list[CompletionOutputLike]
    num_generated_tokens: int
    tokens_per_second: float
    processing_time: float
    first_token_time: float | None
    delta_text: str
    delta_reasoning_content: str | None
    delta_tool_calls: list[tp.Any] | None
    reasoning_content: str | None
    accumulated_text: str
    tool_calls: list[ToolCall] | None


def compute_stream_delta_text(current_text: str, previous_text: str, fallback_delta: str) -> str:
    """Compute a safe streaming delta from two accumulated text snapshots.

    The primary strategy is a simple prefix check: if ``current_text``
    starts with ``previous_text``, the delta is the tail.  When that
    fails (e.g. the parser rewrote a boundary, or reasoning extraction
    shifted indices), the function walks backwards looking for the
    longest suffix of ``previous_text`` that matches a prefix of
    ``current_text`` and returns the non-overlapping tail.

    If no reliable delta can be determined, ``fallback_delta`` is
    returned (typically the raw ``delta_text`` from the engine) as a
    last resort, or an empty string if even that is unsafe.
    """

    current_text = current_text or ""
    previous_text = previous_text or ""
    fallback_delta = fallback_delta or ""

    if current_text.startswith(previous_text):
        return current_text[len(previous_text) :]

    if not current_text and previous_text and not fallback_delta:
        return ""

    max_overlap = min(len(previous_text), len(current_text))
    for overlap in range(max_overlap, 0, -1):
        if previous_text.endswith(current_text[:overlap]):
            return current_text[overlap:]

    if len(current_text) <= len(previous_text):
        if fallback_delta and not previous_text.endswith(fallback_delta):
            return fallback_delta
        return ""

    if fallback_delta and (not previous_text or not previous_text.endswith(fallback_delta)):
        return fallback_delta
    return current_text if not previous_text else ""


def prompt_token_count_from_output(output: RequestOutputLike) -> int:
    """Extract the total prompt token count from a ``RequestOutputLike``.

    Handles both flat (``list[int]``) and segmented (``list[list[int]]``)
    prompt token ID layouts.  Segmented layouts occur when the prompt was
    split across multiple encoder segments (e.g. for multimodal inputs).
    """

    prompt_ids = output.prompt_token_ids
    if not prompt_ids:
        return 0
    if prompt_ids and isinstance(prompt_ids[0], list):
        return sum(len(seg) for seg in tp.cast(list[list[int]], prompt_ids))
    return len(tp.cast(list[int], prompt_ids))


def normalize_tool_calls(tool_calls: tp.Any) -> list[ToolCall] | None:
    """Validate and coerce heterogeneous tool-call data into ``ToolCall`` instances.

    Accepts a list that may contain ``ToolCall`` model instances, raw dicts,
    or other Pydantic-compatible representations.  Each element is passed
    through ``ToolCall.model_validate``; items that fail validation are
    silently dropped.  Returns ``None`` (not an empty list) when no valid
    calls remain, so callers can use a simple truthiness check.
    """
    if not tool_calls:
        return None
    if not isinstance(tool_calls, list):
        return None
    normalized: list[ToolCall] = []
    for call in tool_calls:
        if isinstance(call, ToolCall):
            normalized.append(call)
            continue
        try:
            normalized.append(ToolCall.model_validate(call))
        except Exception:
            continue
    return normalized or None


def normalize_delta_tool_calls(tool_calls: tp.Any) -> list[DeltaToolCall] | None:
    """Validate and coerce heterogeneous delta-tool-call data into ``DeltaToolCall`` instances.

    Same contract as ``normalize_tool_calls`` but for the streaming delta
    variant which carries partial function names and argument chunks
    rather than complete calls.
    """
    if not tool_calls:
        return None
    if not isinstance(tool_calls, list):
        return None
    normalized: list[DeltaToolCall] = []
    for call in tool_calls:
        if isinstance(call, DeltaToolCall):
            normalized.append(call)
            continue
        try:
            normalized.append(DeltaToolCall.model_validate(call))
        except Exception:
            continue
    return normalized or None


def jsonify_tool_calls(tool_calls: tp.Any) -> list[tp.Any] | None:
    """Normalize tool calls and serialize each to a JSON-safe dict.

    Combines ``normalize_tool_calls`` with ``model_dump`` so the result
    can be directly embedded in a JSON response body without further
    processing.  ``exclude_unset`` and ``exclude_none`` keep the output
    compact.
    """
    normalized = normalize_tool_calls(tool_calls)
    if normalized is None:
        return None
    return [tool_call.model_dump(exclude_unset=True, exclude_none=True) for tool_call in normalized]


def coerce_stream_delta_message(
    delta_message: tp.Any,
    *,
    fallback_text: str = "",
    default_role: str | None = None,
) -> DeltaMessage | None:
    """Normalize an engine/parser streaming delta into a safe ``DeltaMessage``.

    The engine and various tool/reasoning parsers return deltas in
    different shapes — ``DeltaMessage`` instances, plain strings, raw
    dicts, or even ``None``.  This function coerces all of them into a
    canonical ``DeltaMessage``, applying three fixups:

    1. If the input is ``None`` and ``fallback_text`` is non-empty, a
       text-only delta is synthesized.
    2. ``default_role`` is applied when the delta lacks a role.
    3. When ``tool_calls`` are present, ``content`` is forced to
       ``None`` to match the OpenAI streaming spec (which does not
       allow both in the same chunk).
    """

    if delta_message is None:
        return None

    normalized: DeltaMessage | None = None
    if isinstance(delta_message, DeltaMessage):
        normalized = delta_message
    elif isinstance(delta_message, str):
        normalized = DeltaMessage(content=delta_message)
    else:
        try:
            normalized = DeltaMessage.model_validate(delta_message)
        except Exception:
            normalized = None

    if normalized is None:
        if fallback_text:
            normalized = DeltaMessage(content=fallback_text)
        else:
            return None

    if default_role and not normalized.role:
        normalized.role = default_role

    if normalized.content is not None and not isinstance(normalized.content, (str, list)):
        normalized.content = str(normalized.content)
    if isinstance(normalized.content, list):
        normalized.content = [part for part in normalized.content if isinstance(part, dict)] or None

    normalized_delta_tool_calls = normalize_delta_tool_calls(normalized.tool_calls)
    normalized.tool_calls = normalized_delta_tool_calls
    if normalized_delta_tool_calls:
        normalized.content = None
    return normalized


def build_responses_reasoning_item(reasoning_text: str) -> ResponseReasoningItem:
    """Create a ``ResponseReasoningItem`` wrapping the given reasoning text.

    The text is stored inside a single ``ResponseSummaryText`` block at
    ``summary[0]``.  During streaming the accumulator later updates this
    block in-place as more reasoning tokens arrive.
    """
    return ResponseReasoningItem(summary=[ResponseSummaryText(text=reasoning_text)])


def build_responses_function_call_items(tool_calls: list[tp.Any] | None) -> list[ResponseFunctionCallItem]:
    """Convert a list of raw/normalized tool calls into ``ResponseFunctionCallItem`` instances.

    Each valid tool call becomes one item with ``call_id``, ``name``, and
    ``arguments`` populated from the corresponding ``ToolCall``.  Calls
    with an empty function name are silently skipped because they usually
    indicate an incomplete parse artifact.
    """
    normalized_tool_calls = normalize_tool_calls(tool_calls)
    if not normalized_tool_calls:
        return []

    items: list[ResponseFunctionCallItem] = []
    for tool_call in normalized_tool_calls:
        function = tool_call.function
        if not function.name.strip():
            continue
        items.append(
            ResponseFunctionCallItem(
                call_id=tool_call.id,
                name=function.name,
                arguments=function.arguments,
                status="completed",
            )
        )
    return items


def build_responses_message_item(output_text: str) -> ResponseMessageItem:
    """Create a ``ResponseMessageItem`` containing the given assistant text.

    The text is wrapped in a single ``ResponseOutputTextPart`` inside the
    item's ``content`` list.  Status is set to ``"completed"`` because
    this builder is used for non-streaming (batch) responses.
    """
    return ResponseMessageItem(
        content=[ResponseOutputTextPart(text=output_text)],
        status="completed",
    )


def should_emit_responses_message_item(
    output_text: str,
    tool_calls: list[tp.Any] | None = None,
) -> bool:
    """Decide whether a message output item should be included in the response.

    A message item is emitted when either (a) there is visible assistant
    text, or (b) no tool calls were extracted.  The second case ensures
    that the client always receives at least one output item — an empty
    message — even if the model produced no text and no tool calls
    (e.g. an early abort).
    """
    return bool(output_text) or not tool_calls


def build_responses_output_items(
    *,
    output_text: str,
    tool_calls: list[tp.Any] | None = None,
    reasoning_text: str | None = None,
    include_reasoning_summary: bool = False,
) -> list[ResponsesOutputItem]:
    """Assemble the complete ``output`` list for a finished ``ResponsesResponse``.

    Items are appended in display order:
    1. A reasoning summary (if enabled and non-empty reasoning text exists).
    2. One function-call item per extracted tool call.
    3. A message item (unless suppressed by ``should_emit_responses_message_item``).

    This mirrors the ordering the OpenAI Responses API uses.
    """
    items: list[ResponsesOutputItem] = []
    if include_reasoning_summary and isinstance(reasoning_text, str) and reasoning_text.strip():
        items.append(build_responses_reasoning_item(reasoning_text))
    items.extend(build_responses_function_call_items(tool_calls))
    if should_emit_responses_message_item(output_text, tool_calls):
        items.append(build_responses_message_item(output_text))
    return items


def responses_assistant_message_from_output_items(
    output_items: list[ResponsesOutputItem],
) -> ChatMessage:
    """Convert Responses API output items back into a ``ChatMessage``.

    Thin wrapper around ``assistant_message_from_output_items`` (defined
    in ``typed_models``) exposed here for convenience so that callers in
    the server layer do not need to import from both modules.
    """
    return assistant_message_from_output_items(output_items)


def build_responses_object(
    *,
    response_id: str,
    model: str,
    output_text: str,
    prompt_tokens: int,
    completion_tokens: int,
    tool_calls: list[tp.Any] | None = None,
    reasoning_text: str | None = None,
    include_reasoning_summary: bool = False,
    output_items: list[ResponsesOutputItem] | None = None,
    created_at: int | None = None,
) -> ResponsesResponse:
    """Build a complete, non-streaming ``ResponsesResponse`` object.

    If ``output_items`` is not provided, the function assembles them
    automatically via ``build_responses_output_items``.  ``created_at``
    defaults to the current Unix timestamp.  The returned object is
    ready to be serialized as the response body of a non-streaming
    ``/v1/responses`` request.
    """
    created_at_value = int(created_at if created_at is not None else time.time())
    if output_items is None:
        output_items = build_responses_output_items(
            output_text=output_text,
            tool_calls=tool_calls,
            reasoning_text=reasoning_text,
            include_reasoning_summary=include_reasoning_summary,
        )

    return ResponsesResponse(
        id=response_id,
        object="response",
        created_at=created_at_value,
        model=model,
        status="completed",
        output=output_items,
        usage=ResponsesUsage(
            input_tokens=int(prompt_tokens),
            output_tokens=int(completion_tokens),
            total_tokens=int(prompt_tokens) + int(completion_tokens),
        ),
        text=ResponsesTextConfig(),
    )


def iter_chat_completion_stream_responses(
    outputs: tp.Iterator[RequestOutputLike],
    *,
    model: str,
) -> tp.Iterator[ChatCompletionStreamResponse]:
    """Convert a stream of engine output snapshots into OpenAI Chat Completion SSE chunks.

    Each yielded ``ChatCompletionStreamResponse`` corresponds to one
    ``data:`` line in the SSE stream.  The final chunk carries a
    non-null ``finish_reason`` (``"stop"``, ``"tool_calls"``, or
    ``"length"``) and an empty/null content delta to signal end-of-stream.

    Tool-call deltas are passed through as-is; the ``finish_reason``
    is upgraded to ``"tool_calls"`` whenever at least one delta or
    batch tool call was detected during the stream.
    """

    prompt_tokens = 0
    total_generated = 0
    generation_time = 0.0
    tokens_per_second = 0.0
    last_output = None
    saw_tool_call_delta = False

    for output in outputs:
        last_output = output
        if not prompt_tokens:
            prompt_tokens = prompt_token_count_from_output(output)

        current_completion_tokens = int(output.num_generated_tokens or 0)
        current_tps = output.tokens_per_second
        elapsed_time = output.processing_time

        delta_message = coerce_stream_delta_message(
            DeltaMessage(
                role="assistant",
                content=output.delta_text or None,
                tool_calls=normalize_delta_tool_calls(output.delta_tool_calls),
                reasoning_content=output.delta_reasoning_content,
            ),
            fallback_text=output.delta_text or "",
            default_role="assistant",
        )
        if delta_message is None:
            total_generated = current_completion_tokens
            generation_time = elapsed_time
            tokens_per_second = current_tps
            continue

        if delta_message.tool_calls:
            saw_tool_call_delta = True

        yield ChatCompletionStreamResponse(
            model=model,
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=delta_message,
                    finish_reason=None,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=current_completion_tokens,
                total_tokens=prompt_tokens + current_completion_tokens,
                tokens_per_second=current_tps,
                processing_time=elapsed_time,
                first_token_time=output.first_token_time,
            ),
        )
        total_generated = current_completion_tokens
        generation_time = elapsed_time
        tokens_per_second = current_tps

    if last_output is None:
        raise RuntimeError("Streaming finished without any output")

    primary_output = last_output.outputs[0] if last_output.outputs else None
    final_tool_calls = normalize_tool_calls(
        last_output.tool_calls or (primary_output.tool_calls if primary_output else None)
    )
    has_tool_calls = saw_tool_call_delta or bool(final_tool_calls)
    finish_reason = primary_output.finish_reason if primary_output is not None else None
    if has_tool_calls:
        finish_reason = "tool_calls"
    elif finish_reason == "finished":
        finish_reason = "stop"
    elif finish_reason is None:
        finish_reason = "stop"

    yield ChatCompletionStreamResponse(
        model=model,
        choices=[
            ChatCompletionStreamResponseChoice(
                index=0,
                delta=DeltaMessage(content=None if has_tool_calls else "", role="assistant"),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=total_generated,
            total_tokens=prompt_tokens + total_generated,
            tokens_per_second=tokens_per_second,
            processing_time=generation_time,
            first_token_time=last_output.first_token_time,
        ),
    )


class ResponsesStreamAccumulator:
    """Stateful builder that converts engine output snapshots into Responses API SSE frames.

    The accumulator is the core of the Responses API streaming pipeline.
    It maintains three pieces of mutable state — one each for reasoning,
    function calls, and the assistant message — and exposes a simple
    three-phase lifecycle:

    1. **``initial_frames()``** — returns the opening ``response.created``
       event with a skeleton response (``status="in_progress"``, empty
       ``output``).

    2. **``add_output(output)``** — called once per engine snapshot.
       Inspects the delta fields to decide which output items to create
       or extend, emitting the appropriate ``*.added``, ``*.delta``, etc.
       events.  Reasoning deltas, function-call argument chunks, and
       visible text deltas are routed independently.

    3. **``finalize(last_output, prompt_tokens)``** — called once after
       the engine signals completion.  Emits ``*.done`` events for every
       open item, builds the final ``ResponsesResponse`` (with usage
       stats and finalization overrides applied), and returns the
       closing ``response.completed`` frame plus the response object.

    The accumulator is single-use: after ``finalize`` is called the
    internal state should not be reused.
    """

    def __init__(
        self,
        *,
        response_id: str,
        model: str,
        include_reasoning_summary: bool = False,
        final_response_overrides: ResponsesFinalizationOptions | dict[str, tp.Any] | None = None,
        created_at: int | None = None,
    ):
        self.response_id = response_id
        self.model = model
        self.include_reasoning_summary = include_reasoning_summary
        if isinstance(final_response_overrides, ResponsesFinalizationOptions):
            self.final_response_overrides = final_response_overrides
        elif isinstance(final_response_overrides, dict):
            self.final_response_overrides = ResponsesFinalizationOptions.model_validate(final_response_overrides)
        else:
            self.final_response_overrides = ResponsesFinalizationOptions()
        self.created_at = int(created_at if created_at is not None else time.time())

        self.output_items_stream: list[ResponsesOutputItem] = []
        self.next_output_index = 0

        self.reasoning_state: ReasoningStreamState | None = None
        self.reasoning_text_accum = ""

        self.function_states: dict[str, FunctionCallStreamState] = {}
        self.function_order: list[str] = []
        self.saw_function_call_delta = False

        self.message_state: MessageStreamState | None = None
        self.message_text_accum = ""

    @staticmethod
    def _primary_output(output: RequestOutputLike) -> CompletionOutputLike | None:
        return output.outputs[0] if output.outputs else None

    @staticmethod
    def _json_dump_arguments(arguments: tp.Any) -> str:
        if isinstance(arguments, str):
            return arguments
        if isinstance(arguments, (dict, list)):
            return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        return "" if arguments is None else str(arguments)

    def _frame(self, event: str, payload: tp.Any) -> StreamEventFrame:
        return StreamEventFrame(event=event, payload=payload)

    def initial_frames(self) -> list[StreamEventFrame]:
        """Return the opening ``response.created`` event that starts the SSE stream.

        The skeleton response has ``status="in_progress"`` and an empty
        ``output`` list.  The client uses the ``id`` and ``model`` fields
        to associate subsequent delta events with this response.
        """
        response = ResponsesResponse(
            id=self.response_id,
            object="response",
            created_at=self.created_at,
            model=self.model,
            status="in_progress",
            output=[],
            usage=None,
        )
        return [self._frame("response.created", ResponseCreatedEvent(response=response))]

    def _ensure_reasoning_item(self, initial_text: str = "") -> list[StreamEventFrame]:
        if self.reasoning_state is not None:
            return []

        reasoning_item = build_responses_reasoning_item(initial_text)
        reasoning_item.status = "in_progress"
        self.reasoning_state = ReasoningStreamState(item=reasoning_item, output_index=self.next_output_index)
        self.next_output_index += 1
        self.output_items_stream.append(reasoning_item)
        return [
            self._frame(
                "response.output_item.added",
                ResponseOutputItemAddedEvent(
                    output_index=self.reasoning_state.output_index,
                    item=reasoning_item,
                ),
            )
        ]

    def _ensure_message_item(self) -> list[StreamEventFrame]:
        if self.message_state is not None:
            return []

        message_item = ResponseMessageItem(content=[], status="in_progress")
        self.message_state = MessageStreamState(item=message_item, output_index=self.next_output_index)
        self.next_output_index += 1
        self.output_items_stream.append(message_item)
        return [
            self._frame(
                "response.output_item.added",
                ResponseOutputItemAddedEvent(
                    output_index=self.message_state.output_index,
                    item=message_item,
                ),
            ),
            self._frame(
                "response.content_part.added",
                ResponseContentPartAddedEvent(
                    output_index=self.message_state.output_index,
                    item_id=self.message_state.item_id,
                    content_index=self.message_state.content_index,
                    part=ResponseOutputTextPart(text=""),
                ),
            ),
        ]

    def _resolve_function_state(
        self,
        *,
        call_index: int,
        call_id: str | None,
    ) -> tuple[FunctionCallStreamState, list[StreamEventFrame]]:
        if isinstance(call_id, str) and call_id:
            call_key = call_id
            resolved_call_id = call_id
        else:
            call_key = f"idx:{call_index}"
            resolved_call_id = f"call_{self.response_id}_{call_index}"

        state = self.function_states.get(call_key)
        if state is not None:
            return state, []

        function_item = ResponseFunctionCallItem(
            call_id=resolved_call_id,
            name="",
            arguments="",
            status="in_progress",
        )
        state = FunctionCallStreamState(item=function_item, output_index=self.next_output_index)
        self.function_states[call_key] = state
        self.function_order.append(call_key)
        self.output_items_stream.append(function_item)
        self.next_output_index += 1
        return state, [
            self._frame(
                "response.output_item.added",
                ResponseOutputItemAddedEvent(output_index=state.output_index, item=function_item),
            )
        ]

    def add_output(self, output: RequestOutputLike) -> list[StreamEventFrame]:
        """Process one engine output snapshot and emit the corresponding SSE frames.

        Inspects the snapshot's delta fields to route data to the correct
        output item:

        - **Reasoning deltas** are appended to the reasoning summary text
          and emitted as ``response.reasoning_summary_text.delta`` events.
        - **Tool-call deltas** create or extend function-call items, emitting
          ``response.output_item.added`` (on first sight) and
          ``response.function_call_arguments.delta`` events.
        - **Text deltas** are appended to the message item and emitted as
          ``response.output_text.delta`` events.  Text is suppressed once
          a tool-call delta has been seen (matching OpenAI behavior where
          content and tool calls are mutually exclusive in a single chunk).

        Returns a list of ``StreamEventFrame`` objects to be sent to the
        client.  May return an empty list if the snapshot carried no
        actionable deltas (e.g. a control-token-only step).
        """
        frames: list[StreamEventFrame] = []
        primary_output = self._primary_output(output)
        delta_text = output.delta_text or ""
        delta_reasoning = output.delta_reasoning_content or ""
        delta_tool_calls = normalize_delta_tool_calls(output.delta_tool_calls) or []

        current_reasoning = output.reasoning_content or (primary_output.reasoning_content if primary_output else "")
        if self.include_reasoning_summary and isinstance(current_reasoning, str):
            delta_reasoning = compute_stream_delta_text(
                current_reasoning,
                self.reasoning_text_accum,
                delta_reasoning,
            )

        if self.include_reasoning_summary and delta_reasoning:
            frames.extend(self._ensure_reasoning_item(""))
            assert self.reasoning_state is not None
            self.reasoning_text_accum += delta_reasoning
            self.reasoning_state.item.summary[0].text = self.reasoning_text_accum
            frames.append(
                self._frame(
                    "response.reasoning_summary_text.delta",
                    ResponseReasoningSummaryTextDeltaEvent(
                        output_index=self.reasoning_state.output_index,
                        item_id=self.reasoning_state.item_id,
                        summary_index=0,
                        delta=delta_reasoning,
                    ),
                )
            )

        if delta_tool_calls:
            self.saw_function_call_delta = True

        for position, delta_call in enumerate(delta_tool_calls):
            call_index = delta_call.index if isinstance(delta_call.index, int) else position
            state, added_frames = self._resolve_function_state(
                call_index=call_index,
                call_id=delta_call.id,
            )
            frames.extend(added_frames)

            function_payload = delta_call.function
            if function_payload and function_payload.name:
                state.item.name = function_payload.name

            arguments_delta = function_payload.arguments if function_payload is not None else None
            if arguments_delta is None:
                continue

            arguments_delta_text = self._json_dump_arguments(arguments_delta)
            state.item.arguments += arguments_delta_text
            frames.append(
                self._frame(
                    "response.function_call_arguments.delta",
                    ResponseFunctionCallArgumentsDeltaEvent(
                        output_index=state.output_index,
                        item_id=state.item_id,
                        delta=arguments_delta_text,
                    ),
                )
            )

        if self.saw_function_call_delta:
            delta_text = ""

        if delta_text:
            frames.extend(self._ensure_message_item())
            assert self.message_state is not None
            self.message_text_accum += delta_text
            frames.append(
                self._frame(
                    "response.output_text.delta",
                    ResponseOutputTextDeltaEvent(
                        output_index=self.message_state.output_index,
                        item_id=self.message_state.item_id,
                        content_index=self.message_state.content_index,
                        delta=delta_text,
                    ),
                )
            )

        return frames

    def finalize(
        self,
        *,
        last_output: RequestOutputLike,
        prompt_tokens: int,
    ) -> tuple[list[StreamEventFrame], ResponsesResponse]:
        """Close the stream by emitting all ``*.done`` events and ``response.completed``.

        This method performs several finalization steps in order:

        1. **Reasoning close** — if a reasoning item is open and has
           accumulated text, emits ``response.reasoning_summary_text.done``
           and ``response.output_item.done``.

        2. **Function-call reconciliation** — for each tool call in the
           final output, ensures a function-call item exists (creating one
           if the tool call was never seen during streaming, e.g. when the
           entire call arrived in a single batch extraction).  Emits any
           missing ``response.function_call_arguments.delta`` and then
           ``response.function_call_arguments.done`` +
           ``response.output_item.done`` for each.

        3. **Message close** — if a message item should be emitted (see
           ``should_emit_responses_message_item``), emits
           ``response.output_text.done`` and ``response.output_item.done``
           with the final text.

        4. **Response close** — builds the complete ``ResponsesResponse``
           with usage stats, applies ``final_response_overrides``, and
           emits ``response.completed``.

        Returns:
            A tuple of (list of final frames, the completed ResponsesResponse).
        """
        frames: list[StreamEventFrame] = []
        primary_output = self._primary_output(last_output)
        full_text = last_output.accumulated_text or ""
        reasoning_text_final = last_output.reasoning_content or (
            primary_output.reasoning_content if primary_output else None
        )

        tool_calls_payload = normalize_tool_calls(
            last_output.tool_calls or (primary_output.tool_calls if primary_output else None)
        )
        if tool_calls_payload or self.saw_function_call_delta:
            full_text = ""

        if (
            self.include_reasoning_summary
            and not self.reasoning_text_accum
            and isinstance(reasoning_text_final, str)
            and reasoning_text_final.strip()
        ):
            self.reasoning_text_accum = reasoning_text_final

        if (
            self.include_reasoning_summary
            and self.reasoning_text_accum
            and not (self.reasoning_state and self.reasoning_state.done)
        ):
            frames.extend(self._ensure_reasoning_item(self.reasoning_text_accum))
            assert self.reasoning_state is not None
            self.reasoning_state.item.summary[0].text = self.reasoning_text_accum
            self.reasoning_state.item.status = "completed"
            frames.append(
                self._frame(
                    "response.reasoning_summary_text.done",
                    ResponseReasoningSummaryTextDoneEvent(
                        output_index=self.reasoning_state.output_index,
                        item_id=self.reasoning_state.item_id,
                        summary_index=0,
                        text=self.reasoning_text_accum,
                    ),
                )
            )
            frames.append(
                self._frame(
                    "response.output_item.done",
                    ResponseOutputItemDoneEvent(
                        output_index=self.reasoning_state.output_index,
                        item=self.reasoning_state.item,
                    ),
                )
            )
            self.reasoning_state.done = True

        normalized_tool_calls = tool_calls_payload or []
        for idx, tool_call in enumerate(normalized_tool_calls):
            state, added_frames = self._resolve_function_state(
                call_index=idx,
                call_id=tool_call.id,
            )
            frames.extend(added_frames)

            if tool_call.function.name:
                state.item.name = tool_call.function.name

            arguments_text = self._json_dump_arguments(tool_call.function.arguments)
            if arguments_text and not state.item.arguments:
                state.item.arguments = arguments_text
                frames.append(
                    self._frame(
                        "response.function_call_arguments.delta",
                        ResponseFunctionCallArgumentsDeltaEvent(
                            output_index=state.output_index,
                            item_id=state.item_id,
                            delta=arguments_text,
                        ),
                    )
                )
            elif arguments_text:
                state.item.arguments = arguments_text

        for call_key in self.function_order:
            state = self.function_states.get(call_key)
            if state is None or state.done:
                continue
            state.item.status = "completed"
            frames.append(
                self._frame(
                    "response.function_call_arguments.done",
                    ResponseFunctionCallArgumentsDoneEvent(
                        output_index=state.output_index,
                        item_id=state.item_id,
                        arguments=state.item.arguments,
                    ),
                )
            )
            frames.append(
                self._frame(
                    "response.output_item.done",
                    ResponseOutputItemDoneEvent(output_index=state.output_index, item=state.item),
                )
            )
            state.done = True

        if should_emit_responses_message_item(full_text, normalized_tool_calls):
            frames.extend(self._ensure_message_item())
            assert self.message_state is not None
            self.message_text_accum = full_text
            frames.append(
                self._frame(
                    "response.output_text.done",
                    ResponseOutputTextDoneEvent(
                        output_index=self.message_state.output_index,
                        item_id=self.message_state.item_id,
                        content_index=self.message_state.content_index,
                        text=full_text,
                    ),
                )
            )
            self.message_state.item.content = [ResponseOutputTextPart(text=full_text)]
            self.message_state.item.status = "completed"
            if not self.message_state.done:
                frames.append(
                    self._frame(
                        "response.output_item.done",
                        ResponseOutputItemDoneEvent(
                            output_index=self.message_state.output_index,
                            item=self.message_state.item,
                        ),
                    )
                )
                self.message_state.done = True

        completion_tokens = int(last_output.num_generated_tokens or 0)
        final_obj = build_responses_object(
            response_id=self.response_id,
            model=self.model,
            output_text=full_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tool_calls=tool_calls_payload,
            reasoning_text=self.reasoning_text_accum or None,
            include_reasoning_summary=self.include_reasoning_summary,
            output_items=self.output_items_stream,
            created_at=self.created_at,
        )
        final_obj = final_obj.model_copy(update=self.final_response_overrides.as_update_dict())
        frames.append(
            self._frame(
                "response.completed",
                ResponseCompletedEvent(response=final_obj),
            )
        )
        return frames, final_obj


def iter_responses_stream_frames(
    outputs: tp.Iterator[RequestOutputLike],
    *,
    response_id: str,
    model: str,
    include_reasoning_summary: bool = False,
    final_response_overrides: ResponsesFinalizationOptions | dict[str, tp.Any] | None = None,
    created_at: int | None = None,
) -> tp.Iterator[StreamEventFrame]:
    """Convert a stream of engine output snapshots into Responses API SSE event frames.

    This is the top-level entry point for Responses API streaming.  It
    creates a ``ResponsesStreamAccumulator``, feeds each snapshot through
    ``add_output``, and calls ``finalize`` after the last snapshot.  The
    caller (typically the API server) iterates over the yielded
    ``StreamEventFrame`` objects and serializes each as an SSE line.
    """

    accumulator = ResponsesStreamAccumulator(
        response_id=response_id,
        model=model,
        include_reasoning_summary=include_reasoning_summary,
        final_response_overrides=final_response_overrides,
        created_at=created_at,
    )
    for frame in accumulator.initial_frames():
        yield frame

    prompt_tokens = 0
    last_output = None
    for output in outputs:
        last_output = output
        if not prompt_tokens:
            prompt_tokens = prompt_token_count_from_output(output)
        for frame in accumulator.add_output(output):
            yield frame

    if last_output is None:
        raise RuntimeError("Streaming finished without any output")

    final_frames, _final_obj = accumulator.finalize(last_output=last_output, prompt_tokens=prompt_tokens)
    for frame in final_frames:
        yield frame
