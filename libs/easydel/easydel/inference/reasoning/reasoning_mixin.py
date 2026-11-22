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

"""Mixin class for reasoning extraction functionality in inference servers.

Provides :class:`ReasoningMixin`, a per-server bridge between OpenAI-compatible
API surfaces and the registered :class:`~easydel.inference.reasoning.abstract_reasoning.ReasoningParser`
implementations. The mixin owns a ``reasoning_parsers`` dict keyed by model
name and exposes:

* :meth:`ReasoningMixin.initialize_reasoning_parsers` — bulk construction of
  per-model parsers from a tokenizer/processor map.
* :meth:`ReasoningMixin.extract_reasoning_batch` — non-streaming split into
  reasoning / visible content.
* :meth:`ReasoningMixin.extract_reasoning_streaming` — streaming split that
  emits :class:`~easydel.inference.openai_api_modules.DeltaMessage` payloads.
* :meth:`ReasoningMixin.get_reasoning_parser_for_model` — direct lookup.

Mirrors the shape of ``ToolCallingMixin`` and is intended to be combined with
it on an inference server class.
"""

from __future__ import annotations

import dataclasses
import threading
import typing as tp
from collections import OrderedDict

from eformer.loggings import get_logger

from ..openai_api_modules import DeltaMessage
from .abstract_reasoning import ReasoningParser, ReasoningParserManager

logger = get_logger("ReasoningMixin")

_MAX_TRACKED_REASONING_STREAMS = 512
_HARD_MAX_TRACKED_REASONING_STREAMS_FACTOR = 4
_REASONING_LOCK_INIT = threading.Lock()


@dataclasses.dataclass
class _ReasoningStreamState:
    """Per-stream bookkeeping for :meth:`ReasoningMixin.extract_reasoning_streaming`."""

    parser: tp.Any
    emitted: bool = False


class ReasoningMixin:
    """Mixin class providing reasoning extraction for inference API servers.

    Follows the same pattern as ToolCallingMixin.
    """

    reasoning_parsers: dict[str, ReasoningParser]

    def initialize_reasoning_parsers(
        self,
        model_processors: dict[str, tp.Any],
        reasoning_parser_name: str,
        enable_reasoning: bool,
    ) -> dict[str, ReasoningParser]:
        """Initialize reasoning parsers for all registered models.

        Args:
            model_processors: Mapping of model name to its tokenizer/processor.
            reasoning_parser_name: Name of the parser registered with
                :class:`ReasoningParserManager`.
            enable_reasoning: When ``False``, returns an empty mapping
                without instantiating parsers.

        Returns:
            Dictionary mapping model name to a configured
            :class:`ReasoningParser`. Models for which initialization fails
            are silently dropped (with a warning logged).
        """
        reasoning_parsers = {}
        if not enable_reasoning:
            return reasoning_parsers

        for model_name, processor in model_processors.items():
            try:
                parser_class = ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)
                reasoning_parsers[model_name] = parser_class(processor)
                logger.info(f"Initialized {reasoning_parser_name} reasoning parser for model {model_name}")
            except KeyError:
                logger.warning(
                    f"Reasoning parser '{reasoning_parser_name}' not found, reasoning disabled for {model_name}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize reasoning parser for {model_name}: {e}")

        return reasoning_parsers

    def extract_reasoning_batch(
        self,
        response_text: str,
        model_name: str,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning from a complete response.

        Args:
            response_text: Full assistant response text.
            model_name: Model identifier used to look up the registered parser.

        Returns:
            Tuple of (reasoning_content, content_without_reasoning). Either
            element may be ``None`` (or the original text when no parser is
            available for ``model_name``).
        """
        if not hasattr(self, "reasoning_parsers") or model_name not in self.reasoning_parsers:
            return None, response_text

        parser = self.reasoning_parsers[model_name]
        return parser.extract_reasoning(response_text)

    def extract_reasoning_streaming(
        self,
        model_name: str,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: list[int] | None = None,
        current_token_ids: list[int] | None = None,
        delta_token_ids: list[int] | None = None,
        request=None,
        final: bool = False,
        stream_id: tp.Hashable | None = None,
    ) -> DeltaMessage | None:
        """Extract reasoning from streaming response chunks with per-stream isolation.

        Each active stream (identified by ``stream_id`` or ``id(request)``) receives
        its own fresh parser instance so concurrent or consecutive requests never
        share mutable parser state. An empty ``previous_text`` starts a clean stream
        state; setting ``final=True`` flushes withheld delimiter prefixes and drops
        the stream's state.
        """
        if not hasattr(self, "reasoning_parsers") or model_name not in self.reasoning_parsers:
            return None

        stream_key = stream_id if stream_id is not None else (id(request) if request is not None else None)
        if final and not delta_text:
            with self._get_reasoning_streams_lock():
                streams = getattr(self, "_mixin_reasoning_streams", None)
                if not isinstance(streams, OrderedDict) or (model_name, stream_key) not in streams:
                    return None
        restart = not previous_text and not (final and not delta_text)
        state = self._reasoning_stream_state(
            model_name,
            stream_key,
            self.reasoning_parsers[model_name],
            restart=restart,
        )
        parser = state.parser

        normalized: DeltaMessage | None = None
        if delta_text:
            try:
                raw_delta = parser.extract_reasoning_streaming(
                    previous_text=previous_text,
                    current_text=current_text,
                    delta_text=delta_text,
                    previous_token_ids=previous_token_ids or [],
                    current_token_ids=current_token_ids or [],
                    delta_token_ids=delta_token_ids or [],
                    request=request,
                )
            except Exception:
                normalized = DeltaMessage(content=delta_text)
            else:
                normalized = self._normalize_reasoning_delta(raw_delta, delta_text)
            if normalized is not None and (normalized.content or normalized.reasoning_content):
                with self._get_reasoning_streams_lock():
                    state.emitted = True

        if final:
            try:
                tail = parser.finalize_streaming(current_text, current_token_ids or [], request)
            except Exception:
                tail = None
            self._end_reasoning_stream(model_name, stream_key)
            if tail is not None:
                if normalized is None:
                    return tail
                return normalized.model_copy(
                    update={
                        "content": (normalized.content or "") + (tail.content or "") or None,
                        "reasoning_content": (normalized.reasoning_content or "") + (tail.reasoning_content or "")
                        or None,
                    }
                )
            return normalized

        return normalized

    @staticmethod
    def _normalize_reasoning_delta(delta: tp.Any, delta_text: str) -> DeltaMessage | None:
        """Convert a parser's streaming output into a normalized ``DeltaMessage``."""
        if delta is None or isinstance(delta, DeltaMessage):
            return delta
        if isinstance(delta, str):
            return DeltaMessage(reasoning_content=delta)
        if isinstance(delta, dict):
            try:
                return DeltaMessage(**delta)
            except Exception:
                return DeltaMessage(content=delta_text)
        if hasattr(delta, "model_dump"):
            payload = delta.model_dump(exclude_unset=True, exclude_none=True)
            if isinstance(payload, dict):
                try:
                    return DeltaMessage(**payload)
                except Exception:
                    return DeltaMessage(content=delta_text)
        return DeltaMessage(content=delta_text)

    def abort_reasoning_stream(
        self,
        model_name: str,
        stream_id: tp.Hashable | None = None,
        request=None,
    ) -> None:
        """Drop a stream's isolated reasoning parser state without calling ``final=True``."""
        stream_key = stream_id if stream_id is not None else (id(request) if request is not None else None)
        self._end_reasoning_stream(model_name, stream_key)

    @staticmethod
    def _new_reasoning_stream_parser(registered: tp.Any) -> tp.Any:
        """Construct a fresh parser instance isolated to one stream."""
        tokenizer = getattr(registered, "model_tokenizer", None)
        try:
            fresh = type(registered)(tokenizer)
        except Exception:
            return registered
        if hasattr(registered, "assume_reasoning"):
            fresh.assume_reasoning = registered.assume_reasoning
        return fresh

    def _get_reasoning_streams_lock(self) -> threading.Lock:
        """Return the per-instance lock protecting ``_mixin_reasoning_streams``."""
        lock = getattr(self, "_mixin_reasoning_streams_lock", None)
        if lock is None:
            with _REASONING_LOCK_INIT:
                lock = getattr(self, "_mixin_reasoning_streams_lock", None)
                if lock is None:
                    lock = threading.Lock()
                    self._mixin_reasoning_streams_lock = lock
        return lock

    def _reasoning_stream_state(
        self,
        model_name: str,
        stream_key: tp.Hashable | None,
        registered: tp.Any,
        *,
        restart: bool,
    ) -> _ReasoningStreamState:
        """Look up or create per-stream reasoning parser state."""
        key = (model_name, stream_key)
        with self._get_reasoning_streams_lock():
            streams = getattr(self, "_mixin_reasoning_streams", None)
            if not isinstance(streams, OrderedDict):
                streams = OrderedDict()
                self._mixin_reasoning_streams = streams
            state = None if restart else streams.get(key)
            if state is None:
                state = _ReasoningStreamState(parser=self._new_reasoning_stream_parser(registered))
                streams[key] = state
                streams.move_to_end(key)
                if len(streams) > _MAX_TRACKED_REASONING_STREAMS:
                    self._evict_idle_reasoning_streams(streams, keep=key)
            else:
                streams.move_to_end(key)
            return state

    @staticmethod
    def _evict_idle_reasoning_streams(streams: OrderedDict, keep: tp.Hashable) -> None:
        """Evict oldest idle streams first, then LRU entries past the hard cap."""
        excess = len(streams) - _MAX_TRACKED_REASONING_STREAMS
        for key in list(streams):
            if excess <= 0:
                return
            if key == keep or streams[key].emitted:
                continue
            del streams[key]
            excess -= 1
        hard_cap = _HARD_MAX_TRACKED_REASONING_STREAMS_FACTOR * _MAX_TRACKED_REASONING_STREAMS
        if len(streams) <= hard_cap:
            return
        for key in list(streams):
            if len(streams) <= hard_cap:
                return
            if key != keep:
                del streams[key]

    def _end_reasoning_stream(self, model_name: str, stream_key: tp.Hashable | None) -> None:
        """Remove a finished or aborted stream from the per-stream table."""
        with self._get_reasoning_streams_lock():
            streams = getattr(self, "_mixin_reasoning_streams", None)
            if isinstance(streams, OrderedDict):
                streams.pop((model_name, stream_key), None)

    def get_reasoning_parser_for_model(self, model_name: str) -> ReasoningParser | None:
        """Get the reasoning parser instance for a specific model.

        Args:
            model_name: Name of the model to look up.

        Returns:
            The ReasoningParser for the model, or None if not found or
            reasoning parsers are not initialized.
        """
        if not hasattr(self, "reasoning_parsers"):
            return None
        return self.reasoning_parsers.get(model_name)
