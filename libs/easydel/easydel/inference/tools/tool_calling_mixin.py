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

"""Mixin class for tool calling functionality in inference servers.

This module provides the ToolCallingMixin class which adds tool/function calling
capabilities to inference API servers. It handles the initialization of tool parsers,
extraction of tool calls from both batch and streaming responses, and provides
utility methods for tool-related API endpoints.

The mixin is designed to be used with FastAPI-based inference servers that
implement OpenAI-compatible chat completion APIs.

Example:
    >>> from easydel.inference.tools import ToolCallingMixin
    >>>
    >>> class ChatCompletionServer(ToolCallingMixin):
    ...     def __init__(self, model_processors, tool_parser_name="hermes"):
    ...         self.tool_parsers = self.initialize_tool_parsers(
    ...             model_processors,
    ...             tool_parser_name,
    ...             enable_function_calling=True
    ...         )
    ...
    ...     def generate_response(self, request, response_text):
    ...         message, finish_reason = self.extract_tool_calls_batch(
    ...             response_text, request, "my_model"
    ...         )
    ...         return message, finish_reason

See Also:
    - easydel.inference.tools.abstract_tool.ToolParser: Base parser class
    - easydel.inference.tools.abstract_tool.ToolParserManager: Parser registry
"""

from __future__ import annotations

import dataclasses
import json
import typing as tp
from collections import OrderedDict
from http import HTTPStatus

from eformer.loggings import get_logger
from fastapi.responses import JSONResponse

from ..openai_api_modules import (
    ChatCompletionRequest,
    ChatMessage,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from . import ToolParser, ToolParserManager

logger = get_logger("ToolCallingMixin")

_MAX_TRACKED_TOOL_STREAMS = 512
"""Soft bound on tracked streams: past it, streams that have emitted nothing are evicted first."""

_HARD_MAX_TRACKED_TOOL_STREAMS_FACTOR = 4
"""Past ``factor * _MAX_TRACKED_TOOL_STREAMS`` the least recently used streams are dropped even if they
emitted output, so streams abandoned without a ``finished=True`` call cannot grow the table forever."""


@dataclasses.dataclass
class _ToolStreamState:
    """Per-stream bookkeeping for :meth:`ToolCallingMixin.extract_tool_calls_streaming`.

    Attributes:
        parser: The parser instance that owns this stream's parsing state. A fresh
            instance per stream when the registered parser can be rebuilt from its
            tokenizer, otherwise the shared registered instance.
        arguments: Tool index to the concatenation of every argument delta emitted
            to the client so far (parser-independent).
        named: Tool indices whose function name has been emitted.
        call_ids: Tool index to the call id emitted for it, if any.
        call_types: Tool index to the call type emitted for it, if any.
    """

    parser: tp.Any
    arguments: dict[int, str] = dataclasses.field(default_factory=dict)
    named: set[int] = dataclasses.field(default_factory=set)
    call_ids: dict[int, str] = dataclasses.field(default_factory=dict)
    call_types: dict[int, str] = dataclasses.field(default_factory=dict)
    emitted_content: bool = False

    @property
    def emitted(self) -> bool:
        """Whether anything (content or tool output) reached the client; such a stream cannot restart."""
        return bool(self.emitted_content or self.arguments or self.named or self.call_ids or self.call_types)

    def record(self, tool_calls: tp.Iterable[DeltaToolCall]) -> None:
        """Fold emitted tool-call deltas into the accumulators.

        Args:
            tool_calls: Tool-call deltas that were just returned to the caller.
        """
        for call in tool_calls:
            index = call.index if isinstance(call.index, int) else 0
            if call.id:
                self.call_ids[index] = call.id
            if call.type:
                self.call_types[index] = call.type
            function = call.function
            if function is None:
                continue
            if function.name:
                self.named.add(index)
            if function.arguments:
                self.arguments[index] = self.arguments.get(index, "") + function.arguments


def _json_payloads_equal(first: str, second: str) -> bool:
    """Compare two JSON argument strings by decoded value.

    Strings that fail to decode are never considered equal, so malformed
    streams still fall back to text-level catch-up.

    Args:
        first: First argument JSON string (final, batch-extracted).
        second: Second argument JSON string (accumulated stream).

    Returns:
        ``True`` when both decode to equal payloads.
    """
    try:
        return json.loads(first) == json.loads(second)
    except Exception:
        return False


def build_tool_parser_or_none(
    parser_class: type[ToolParser] | None,
    tokenizer: tp.Any,
    *,
    context: str = "",
) -> ToolParser | None:
    """Instantiate a tool parser, degrading to ``None`` on any failure.

    Several tool parsers validate tokenizer capabilities in ``__init__`` and
    raise when the requirement is not met (e.g. :class:`MistralToolParser`
    raises ``RuntimeError`` when the tokenizer vocabulary lacks the
    ``[TOOL_CALLS]`` delimiter token). When such a parser is auto-selected by
    model type (``mistral4`` -> ``mistral``) but paired with a tokenizer that
    does not carry the delimiter, constructing it at request-add time would
    raise inside the engine's request path and take the request (and, via the
    scheduler retry budget, the engine) down.

    This helper contains that failure: it returns a live parser when
    construction succeeds and ``None`` (with a warning) when it does not, so
    callers can disable function calling gracefully instead of crashing.

    Args:
        parser_class: The resolved tool-parser class, or ``None`` when tool
            parsing is disabled.
        tokenizer: Tokenizer/processor passed to the parser constructor.
        context: Optional human-readable context (model or request id) folded
            into the warning message.

    Returns:
        A constructed :class:`ToolParser`, or ``None`` when ``parser_class`` is
        ``None`` or its construction raised.
    """
    if parser_class is None:
        return None
    try:
        return parser_class(tokenizer)
    except Exception as exc:
        # Degrade gracefully on any parser-init failure (missing delimiter
        # token, unsupported tokenizer, etc.) so callers can disable function
        # calling instead of crashing.
        suffix = f" for {context}" if context else ""
        logger.warning(
            "Tool parser %s is incompatible with the tokenizer%s; function calling disabled. Reason: %s",
            getattr(parser_class, "__name__", parser_class),
            suffix,
            exc,
        )
        return None


class ToolCallingMixin:
    """Mixin class providing tool calling functionality for inference API servers.

    This mixin provides comprehensive tool calling support for inference servers,
    including parser initialization, tool call extraction for both batch and
    streaming responses, and utility methods for tool-related API endpoints.

    The mixin is designed to be inherited by inference server classes that need
    to support OpenAI-compatible function/tool calling APIs.

    Attributes:
        tool_parsers (dict[str, ToolParser]): Dictionary mapping model names to
            their corresponding tool parser instances. This attribute should be
            set by calling initialize_tool_parsers() during server initialization.

    Note:
        Classes using this mixin should have the following attributes available:
            - self.tool_parsers: dict[str, ToolParser] - Set via initialize_tool_parsers()
            - self.tool_parser_name: str - Name of the parser being used (optional)
            - self.enable_function_calling: bool - Whether function calling is enabled (optional)

    Example:
        >>> class MyServer(ToolCallingMixin):
        ...     def __init__(self, tokenizers, parser_name):
        ...         self.tool_parser_name = parser_name
        ...         self.enable_function_calling = True
        ...         self.tool_parsers = self.initialize_tool_parsers(
        ...             tokenizers, parser_name, True
        ...         )
        ...
        ...     def handle_completion(self, request, model_output, model_name):
        ...         return self.extract_tool_calls_batch(
        ...             model_output, request, model_name
        ...         )
    """

    tool_parsers: dict[str, ToolParser]

    def initialize_tool_parsers(
        self,
        model_processors: dict[str, tp.Any],
        tool_parser_name: str | dict[str, str | None],
        enable_function_calling: bool,
    ) -> dict[str, ToolParser]:
        """Initialize tool parsers for all registered models.

        Creates tool parser instances for each model using the specified parser
        implementation. This method should be called during server initialization
        to set up tool calling capabilities.

        Args:
            model_processors (dict[str, tp.Any]): Dictionary mapping model names
                to their tokenizers/processors. Each processor should be compatible
                with the tool parser's requirements.
            tool_parser_name (str | dict[str, str | None]): Name of the tool
                parser to use for every model, or a per-model mapping of model
                name to parser name.
            enable_function_calling (bool): Whether to enable function calling.
                If False, returns an empty dictionary and no parsers are initialized.

        Returns:
            dict[str, ToolParser]: Dictionary mapping model names to their
                initialized tool parser instances. Returns an empty dictionary
                if enable_function_calling is False or if parser initialization
                fails for all models.

        Example:
            >>> processors = {"gpt-model": tokenizer1, "llama-model": tokenizer2}
            >>> parsers = self.initialize_tool_parsers(
            ...     processors, "hermes", enable_function_calling=True
            ... )
            >>> print(parsers.keys())
            dict_keys(['gpt-model', 'llama-model'])

        Note:
            If initialization fails for a specific model (e.g., parser not found
            or incompatible tokenizer), a warning is logged and that model is
            skipped. Other models will still be initialized.
        """
        tool_parsers = {}

        if not enable_function_calling:
            return tool_parsers

        for model_name, processor in model_processors.items():
            parser_name: str | None = None
            try:
                parser_name = (
                    tool_parser_name.get(model_name) if isinstance(tool_parser_name, dict) else tool_parser_name
                )
                if not isinstance(parser_name, str) or not parser_name.strip():
                    logger.warning(f"No tool parser resolved for model {model_name}, function calling disabled")
                    continue
                parser_class = ToolParserManager.get_tool_parser(parser_name)
                parser = build_tool_parser_or_none(parser_class, processor, context=f"model {model_name}")
                if parser is None:
                    # Incompatible tokenizer (e.g. missing delimiter token): skip
                    # this model so function calling degrades gracefully.
                    continue
                tool_parsers[model_name] = parser
                logger.info(f"Initialized {parser_name} tool parser for model {model_name}")
            except KeyError:
                logger.warning(f"Tool parser '{parser_name}' not found, function calling disabled for {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize tool parser for {model_name}: {e}")

        return tool_parsers

    def extract_tool_calls_batch(
        self,
        response_text: str,
        request: ChatCompletionRequest,
        model_name: str,
    ) -> tuple[ChatMessage, str]:
        """Extract tool calls from a complete (non-streaming) response.

        Parses the complete model response to identify and extract any tool/function
        calls. This method is used for batch (non-streaming) completions where the
        entire response is available at once.

        Args:
            response_text (str): The complete generated text response from the model.
            request (ChatCompletionRequest): The original chat completion request
                containing tool definitions and other parameters.
            model_name (str): The name of the model that generated the response.
                Used to look up the appropriate parser.

        Returns:
            tuple[ChatMessage, str]: A tuple containing:
                - ChatMessage: The assistant message with role, content, and
                  optional tool_calls field populated if tools were detected.
                - str: The finish_reason - "tool_calls" if tools were called,
                  or "stop" for regular text completion.

        Example:
            >>> response = '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>'
            >>> message, reason = self.extract_tool_calls_batch(
            ...     response, request, "my_model"
            ... )
            >>> print(reason)
            'tool_calls'
            >>> print(message.tool_calls[0].function.name)
            'get_weather'

        Note:
            If no tool parser is available for the model or if no tools are
            detected in the response, returns the original text as content
            with finish_reason "stop".
        """
        if not hasattr(self, "tool_parsers") or model_name not in self.tool_parsers:
            return ChatMessage(role="assistant", content=response_text), "stop"

        tool_parser = self.tool_parsers[model_name]
        extracted = tool_parser.extract_tool_calls(response_text, request)

        if extracted.tools_called and extracted.tool_calls:
            message = ChatMessage(
                role="assistant",
                content=extracted.content,
                tool_calls=extracted.tool_calls,
            )
            finish_reason = "tool_calls"
        else:
            message = ChatMessage(
                role="assistant",
                content=extracted.content if extracted.content else response_text,
            )
            finish_reason = "stop"

        return message, finish_reason

    def extract_tool_calls_streaming(
        self,
        model_name: str,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: list[int] | None = None,
        current_token_ids: list[int] | None = None,
        delta_token_ids: list[int] | None = None,
        request: ChatCompletionRequest | None = None,
        finished: bool = False,
        stream_id: str | None = None,
    ) -> DeltaMessage | None:
        """Extract tool calls from streaming response chunks.

        Processes incremental model output to identify partial tool calls and emit
        appropriate streaming updates. This method maintains state across calls
        (via the parser instance) to handle incomplete JSON/XML structures as
        they are being generated.

        Args:
            model_name (str): The name of the model generating the response.
                Used to look up the appropriate parser.
            previous_text (str): The accumulated text up to the previous call.
            current_text (str): The accumulated text including the current chunk.
            delta_text (str): The new text in the current streaming chunk.
            previous_token_ids (list[int] | None, optional): Token IDs accumulated
                up to the previous call. Defaults to None (treated as empty list).
            current_token_ids (list[int] | None, optional): Token IDs including
                the current chunk. Defaults to None (treated as empty list).
            delta_token_ids (list[int] | None, optional): New token IDs in the
                current chunk. Defaults to None (treated as empty list).
            request (ChatCompletionRequest | None, optional): The original request
                containing tool definitions. Defaults to None.
            finished (bool, optional): Set to ``True`` on the final call of a
                stream so any argument tail withheld by the parser (trailing
                JSON closers) is delivered as a catch-up delta. Defaults to
                False. Requires ``request`` to be provided for the batch
                comparison.
            stream_id (str | None, optional): Identifies the stream this chunk
                belongs to so concurrent streams on the same model do not share
                parser or catch-up state. Defaults to ``id(request)`` when a
                request is given, otherwise all streams of the model share one
                state.

        Returns:
            DeltaMessage | None: A DeltaMessage containing incremental tool call
                information (tool call index, function name delta, arguments delta),
                or None if no tool call update is available for this chunk.

        Example:
            >>> # First chunk with tool call start
            >>> delta1 = self.extract_tool_calls_streaming(
            ...     "model", "", "<tool_call>", "<tool_call>"
            ... )
            >>> # Subsequent chunk with function name
            >>> delta2 = self.extract_tool_calls_streaming(
            ...     "model", "<tool_call>", '<tool_call>{"name":', '{"name":'
            ... )

        Note:
            This method is stateful per stream: each stream (see ``stream_id``)
            gets its own parser instance and emitted-delta bookkeeping, dropped
            when ``finished=True`` is seen. Use the same model_name and stream
            identity for every chunk of one streaming session.
        """
        if not hasattr(self, "tool_parsers") or model_name not in self.tool_parsers:
            return None

        stream_key = stream_id if stream_id is not None else (id(request) if request is not None else None)
        # An empty previous_text marks the first chunk of a stream: start from clean state.
        state = self._tool_stream_state(
            model_name,
            stream_key,
            self.tool_parsers[model_name],
            restart=not previous_text,
        )
        tool_parser = state.parser

        previous_token_ids = previous_token_ids or []
        current_token_ids = current_token_ids or []
        delta_token_ids = delta_token_ids or []

        try:
            delta = tool_parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=delta_token_ids,
                request=request,
            )
        except Exception:
            logger.debug(
                "Tool parser streaming extraction failed for model %s; falling back to raw text delta.",
                model_name,
                exc_info=True,
            )
            # Fall through (no early return): a terminal call must still flush, end the stream and run
            # the catch-up, or the state leaks and the withheld argument tail is lost (r15 S12-1).
            normalized = DeltaMessage(content=delta_text)
        else:
            normalized = self._normalize_streaming_delta(delta, delta_text)

        if normalized is not None:
            if normalized.tool_calls:
                state.record(normalized.tool_calls)
            if normalized.content:
                state.emitted_content = True

        if finished:
            flush = getattr(tool_parser, "flush_streaming_content", None)
            flushed = ""
            if callable(flush):
                try:
                    flushed = flush() or ""
                except Exception:
                    logger.debug("Tool parser content flush failed for model %s.", model_name, exc_info=True)
            if flushed:
                if normalized is None:
                    normalized = DeltaMessage(content=flushed)
                else:
                    normalized.content = (normalized.content or "") + flushed
            self._end_tool_stream(model_name, stream_key)
        if not finished or request is None:
            return normalized

        catch_up_calls = self._build_final_tool_catch_up(tool_parser, current_text, request, state)
        if not catch_up_calls:
            return normalized

        base_calls = list(normalized.tool_calls) if normalized is not None and normalized.tool_calls else []
        merged = base_calls + catch_up_calls
        if normalized is None:
            return DeltaMessage(tool_calls=merged)
        normalized.tool_calls = merged
        return normalized

    @staticmethod
    def _normalize_streaming_delta(delta: tp.Any, delta_text: str) -> DeltaMessage | None:
        """Convert whatever a parser returned into a ``DeltaMessage`` (raw ``delta_text`` if it cannot).

        Args:
            delta: The parser's streaming return value.
            delta_text: The raw text delta, used as the fallback content.

        Returns:
            The normalized delta, or ``None`` when the parser withheld this chunk.
        """
        if delta is None or isinstance(delta, DeltaMessage):
            return delta
        if isinstance(delta, str):
            return DeltaMessage(content=delta)
        payload = delta
        if not isinstance(delta, dict) and hasattr(delta, "model_dump"):
            payload = delta.model_dump(exclude_unset=True, exclude_none=True)
        if isinstance(payload, dict):
            try:
                return DeltaMessage(**payload)
            except Exception:
                pass
        return DeltaMessage(content=delta_text)

    def abort_tool_stream(
        self,
        model_name: str,
        stream_id: tp.Hashable | None = None,
        request: ChatCompletionRequest | None = None,
    ) -> None:
        """Drop a stream's state without the terminal ``finished=True`` call (e.g. client disconnect).

        Args:
            model_name: Model the stream belongs to.
            stream_id: The ``stream_id`` passed to :meth:`extract_tool_calls_streaming`.
            request: The request, when the stream was keyed by request identity (no ``stream_id``).
        """
        stream_key = stream_id if stream_id is not None else (id(request) if request is not None else None)
        self._end_tool_stream(model_name, stream_key)

    @staticmethod
    def _new_stream_parser(registered: tp.Any) -> tp.Any:
        """Build an isolated parser instance for one stream.

        Tool parsers keep per-stream parsing state on the instance, so streams
        must not share the registered one. Parsers that cannot be rebuilt from
        their tokenizer (no ``model_tokenizer`` or a failing constructor) fall
        back to the shared registered instance.

        Args:
            registered: The parser registered for the model.

        Returns:
            A fresh parser of the same class, or ``registered`` when a fresh one
            cannot be built.
        """
        tokenizer = getattr(registered, "model_tokenizer", None)
        if tokenizer is None:
            return registered
        fresh = build_tool_parser_or_none(type(registered), tokenizer, context="a new tool stream")
        return fresh if fresh is not None else registered

    def _tool_stream_state(
        self,
        model_name: str,
        stream_key: tp.Hashable | None,
        registered: tp.Any,
        *,
        restart: bool,
    ) -> _ToolStreamState:
        """Look up (or create) the state of one stream.

        Args:
            model_name: Model the stream belongs to.
            stream_key: Stream identity within the model (``None`` shares one
                state per model).
            registered: The parser registered for the model, used as the
                template for a fresh per-stream parser.
            restart: Discard any existing state and start clean (first chunk of
                a stream).

        Returns:
            The mutable :class:`_ToolStreamState` for this stream.
        """
        streams = getattr(self, "_mixin_tool_streams", None)
        if not isinstance(streams, OrderedDict):
            streams = OrderedDict()
            self._mixin_tool_streams = streams
        key = (model_name, stream_key)
        state = None if restart else streams.get(key)
        if state is None:
            state = _ToolStreamState(parser=self._new_stream_parser(registered))
            streams[key] = state
            # Re-assigning an existing key keeps its old position; a restarted stream is the newest (r15 S12-3).
            streams.move_to_end(key)
            if len(streams) > _MAX_TRACKED_TOOL_STREAMS:
                self._evict_idle_tool_streams(streams, keep=key)
        else:
            streams.move_to_end(key)
        return state

    @staticmethod
    def _evict_idle_tool_streams(streams: OrderedDict, keep: tp.Hashable) -> None:
        """Evict the oldest streams that have not emitted any tool output yet.

        A stream that already sent content, a name, id, type or argument prefix cannot be restarted: a
        fresh parser would re-announce its tools or re-emit delivered text, and the final catch-up would
        resend complete arguments (r14 S12-1, r15 S12-3). Such streams are kept past the soft cap; only
        idle streams (nothing emitted, so a restart is invisible) are evicted. Past the hard cap the
        least recently used streams are dropped regardless (r15 S12-2).

        Args:
            streams: The ordered per-stream state table (oldest first).
            keep: Key of the stream just created; never evicted.
        """
        excess = len(streams) - _MAX_TRACKED_TOOL_STREAMS
        for key in list(streams):
            if excess <= 0:
                return
            if key == keep:
                continue
            if streams[key].emitted:
                continue
            del streams[key]
            excess -= 1
        if excess <= 0:
            return
        hard_cap = _HARD_MAX_TRACKED_TOOL_STREAMS_FACTOR * _MAX_TRACKED_TOOL_STREAMS
        if len(streams) <= hard_cap:
            logger.warn_once(
                f"More than {_MAX_TRACKED_TOOL_STREAMS} tool streams have emitted output and are still open; "
                "keeping them until they finish instead of evicting live streams."
            )
            return
        # Streams abandoned without ``finished=True`` (and without abort_tool_stream) would otherwise
        # grow the table forever (r15 S12-2): drop the least recently used ones past the hard cap.
        logger.warn_once(
            f"More than {hard_cap} tool streams are open; dropping the least recently used. "
            "Call abort_tool_stream() for streams that end without a finished=True call."
        )
        for key in list(streams):
            if len(streams) <= hard_cap:
                return
            if key != keep:
                del streams[key]

    def _end_tool_stream(self, model_name: str, stream_key: tp.Hashable | None) -> None:
        """Drop the state of a finished stream.

        Args:
            model_name: Model the stream belongs to.
            stream_key: Stream identity used when the state was created.
        """
        streams = getattr(self, "_mixin_tool_streams", None)
        if isinstance(streams, OrderedDict):
            streams.pop((model_name, stream_key), None)

    @staticmethod
    def _build_final_tool_catch_up(
        tool_parser: ToolParser,
        current_text: str,
        request: ChatCompletionRequest,
        state: _ToolStreamState,
    ) -> list[DeltaToolCall]:
        """Synthesize final tool-call deltas for data the streaming parser withheld.

        Several tool parsers deliberately withhold trailing JSON closing
        characters while a call is in flight and rely on a terminal catch-up to
        deliver the completing tail. This helper mirrors the production
        :class:`~easydel.inference.parsing.delegating_parser.DelegatingParser`
        behavior for direct mixin users: it compares the batch extraction of the
        final text against the mixin's own accumulator of the argument deltas
        actually emitted for this stream (independent of any parser-internal
        bookkeeping) and emits the missing remainder.

        Args:
            tool_parser: The stream's parser, used for batch extraction.
            current_text: The complete generated text for the stream.
            request: The original chat completion request.
            state: Per-stream record of emitted names, ids, types and arguments.
                A name is only included in the catch-up when it was never
                sent, and then together with the call id and type.

        Returns:
            A list of catch-up :class:`DeltaToolCall` objects (possibly empty).
        """
        try:
            extracted = tool_parser.extract_tool_calls(current_text, request)
        except Exception:
            logger.debug("Final tool batch extraction failed; skipping tool stream catch-up.", exc_info=True)
            return []
        if not (extracted.tools_called and extracted.tool_calls):
            return []

        synthesized: list[DeltaToolCall] = []
        for index, tool_call in enumerate(extracted.tool_calls):
            final_args = tool_call.function.arguments or ""
            streamed = state.arguments.get(index, "")

            if final_args == streamed:
                missing = ""
            elif streamed and final_args.startswith(streamed):
                missing = final_args[len(streamed) :]
            elif streamed and _json_payloads_equal(final_args, streamed):
                # Same payload in a different serialization; re-sending would
                # corrupt the concatenated argument stream.
                missing = ""
            else:
                missing = final_args

            include_name = index not in state.named and bool(tool_call.function.name)
            if not include_name and not missing:
                continue
            # A call announced for the first time needs its id and type too, unless
            # an id/type already went out with earlier deltas for this index.
            synthesized.append(
                DeltaToolCall(
                    index=index,
                    id=(state.call_ids.get(index) or tool_call.id)
                    if include_name and index not in state.call_ids
                    else None,
                    type=(state.call_types.get(index) or tool_call.type)
                    if include_name and index not in state.call_types
                    else None,
                    function=DeltaFunctionCall(
                        name=tool_call.function.name if include_name else None,
                        arguments=missing or None,
                    ),
                )
            )
        return synthesized

    def get_tool_parser_for_model(self, model_name: str) -> ToolParser | None:
        """Get the tool parser instance for a specific model.

        Retrieves the tool parser that was initialized for the given model name.
        Useful for direct access to parser methods or configuration.

        Args:
            model_name (str): The name of the model to get the parser for.

        Returns:
            ToolParser | None: The ToolParser instance for the model, or None
                if no parser is available (either tool_parsers not initialized
                or model not found).

        Example:
            >>> parser = self.get_tool_parser_for_model("my_model")
            >>> if parser:
            ...     adjusted_request = parser.adjust_request(original_request)
        """
        if not hasattr(self, "tool_parsers"):
            return None
        return self.tool_parsers.get(model_name)

    def create_tools_response(self, model_names: list[str]) -> dict[str, tp.Any]:
        """Create a standardized tools response for API listing endpoints.

        Generates a response object containing tool information for each model,
        suitable for returning from a /tools or /models endpoint. Includes
        example tool definitions and supported formats.

        Args:
            model_names (list[str]): List of model names to include in the response.

        Returns:
            dict[str, tp.Any]: A dictionary containing:
                - "models": dict mapping model names to their tool configurations,
                  where each configuration includes:
                    - "tools": list of example tool definitions
                    - "tool_parser": name of the parser being used (or None)
                    - "formats_supported": list of all registered parser names
                    - "parallel_calls": bool indicating parallel call support
                - "default_format": the default tool format ("openai")

        Example:
            >>> response = self.create_tools_response(["model1", "model2"])
            >>> print(response["default_format"])
            'openai'
            >>> print(response["models"]["model1"]["formats_supported"])
            ['hermes', 'qwen', 'mistral', ...]
        """
        tools_by_model = {}

        for model_name in model_names:
            model_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "example_function",
                        "description": "An example function for demonstration",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "param1": {
                                    "type": "string",
                                    "description": "First parameter",
                                },
                                "param2": {
                                    "type": "number",
                                    "description": "Second parameter",
                                },
                            },
                            "required": ["param1"],
                        },
                    },
                }
            ]

            has_parser = hasattr(self, "tool_parsers") and model_name in self.tool_parsers

            tools_by_model[model_name] = {
                "tools": model_tools,
                "tool_parser": getattr(self, "tool_parser_name", None) if has_parser else None,
                "formats_supported": list(ToolParserManager.tool_parsers.keys()),
                "parallel_calls": True,
            }

        return {"models": tools_by_model, "default_format": "openai"}

    def create_tool_execution_placeholder(self) -> JSONResponse:
        """Create a placeholder response for tool execution endpoints.

        Returns a NOT_IMPLEMENTED response indicating that the tool execution
        endpoint needs to be implemented by the user. This is a placeholder
        that should be replaced with actual tool execution logic.

        Returns:
            JSONResponse: A FastAPI JSONResponse with HTTP 501 NOT_IMPLEMENTED
                status and an error message explaining that the endpoint needs
                implementation.

        Example:
            >>> @app.post("/tools/execute")
            ... def execute_tool(self, request: ToolExecuteRequest):
            ...     # Replace this with actual implementation
            ...     return self.create_tool_execution_placeholder()
        """
        error_response = {
            "error": {
                "message": "Tool execution endpoint is a placeholder. Implement based on your needs.",
                "type": HTTPStatus.NOT_IMPLEMENTED.name,
            }
        }
        return JSONResponse(content=error_response, status_code=HTTPStatus.NOT_IMPLEMENTED.value)
