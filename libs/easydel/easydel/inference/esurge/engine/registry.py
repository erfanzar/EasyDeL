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

"""Request bookkeeping for the eSurge engine.

:class:`RequestRecord` is the typed per-request tracking state (replacing an
untyped dict-of-string-keys), and :class:`RequestRegistry` owns every
request-scoped container and synchronization primitive: active records,
client-facing outputs, per-request wake events, the finished-id ring, and
the request/output locks. Producers (admission, output pipeline) and
consumers (generate/stream waiters) share exactly this object instead of a
bag of engine attributes.
"""

from __future__ import annotations

import threading
import typing
from collections import deque
from dataclasses import dataclass, field
from typing import Any

if typing.TYPE_CHECKING:
    from ..esurge_engine import RequestOutput


@dataclass
class RequestRecord:
    """Mutable per-request tracking state used by admission and streaming.

    One record exists per engine request id (child requests of an ``n>1``
    parent get their own records). All mutation happens under the registry's
    request lock.

    Attributes:
        prompt: Prompt text as submitted to the engine (post-truncation).
        prompt_token_ids: Tokenized prompt ids.
        sampling_params: The request's ``SamplingParams``.
        generated_tokens: Exact engine-emitted token ids (EOS included).
        decodable_tokens: Incrementally-filtered tokens for detokenization.
        reported_generated_count: Last generated-count reported to metrics.
        last_decoded_index: Index into decodable tokens of the last decode.
        start_time: ``perf_counter`` timestamp at admission.
        first_token_time: Seconds from admission to the first token, once known.
        last_decode_time: ``perf_counter`` timestamp of the last decode flush.
        decoder_visible_text: Client-visible accumulated text (stop-trimmed).
        truncated: Whether the prompt was truncated at admission.
        tokens_dropped: Number of prompt tokens dropped by truncation.
        requested_new_tokens_original: ``max_tokens`` as requested.
        requested_new_tokens_final: ``max_tokens`` after context capping.
        reserve_tokens: Engine reserve-token setting at admission.
        max_model_len: Engine context limit at admission.
        delegating_parser: Per-request tool/reasoning parser pipeline.
        parser_previous_text: Parser's last-seen accumulated text.
        parser_previous_token_ids: Parser's last-seen token-id prefix.
        sample_index: Sample index for ``n>1`` child requests.
        parent_request_id: Parent id for ``n>1`` child requests, else ``None``.
    """

    prompt: str = ""
    prompt_token_ids: list[int] = field(default_factory=list)
    sampling_params: Any = None
    generated_tokens: list[int] = field(default_factory=list)
    decodable_tokens: list[int] = field(default_factory=list)
    reported_generated_count: int = 0
    last_decoded_index: int = 0
    start_time: float = 0.0
    first_token_time: float | None = None
    last_decode_time: float = 0.0
    decoder_visible_text: str = ""
    truncated: bool = False
    tokens_dropped: int = 0
    requested_new_tokens_original: int | None = None
    requested_new_tokens_final: int | None = None
    reserve_tokens: int = 0
    max_model_len: int = 0
    delegating_parser: Any = None
    parser_previous_text: str = ""
    parser_previous_token_ids: Any = field(default_factory=list)
    sample_index: int = 0
    parent_request_id: str | None = None


class RequestRegistry:
    """Owner of request-scoped state shared between engine components.

    Consolidates what used to be seven separate engine attributes
    (``_active_requests``, ``_request_outputs``, ``_request_events``,
    ``_finished_request_ids``, ``_request_lock``, ``_output_lock``,
    ``_output_event``) behind one object with an explicit API. The lock
    ordering contract is: ``request_lock`` before ``output_lock``; never the
    reverse.

    Args:
        max_outputs: Retention cap for finished request outputs; ``None``
            keeps everything until explicitly removed.
    """

    def __init__(self, *, max_outputs: int | None = None) -> None:
        self.records: dict[str, RequestRecord] = {}
        self.outputs: dict[str, RequestOutput] = {}
        self.events: dict[str, threading.Event] = {}
        self.finished_ids: deque[str] = deque()
        self.request_lock = threading.RLock()
        self.output_lock = threading.RLock()
        self.output_event = threading.Event()
        self.max_outputs = max_outputs

    def register(self, request_id: str, record: RequestRecord, output: RequestOutput) -> threading.Event:
        """Register a new request's record, output, and wake event.

        Args:
            request_id: Engine request id.
            record: The request's tracking record.
            output: The client-facing output container.

        Returns:
            The per-request wake event streaming waiters block on.
        """
        event = threading.Event()
        with self.request_lock:
            self.events[request_id] = event
            self.records[request_id] = record
        with self.output_lock:
            self.outputs[request_id] = output
        return event

    def get_record(self, request_id: str) -> RequestRecord | None:
        """Return the record for ``request_id``, or ``None``."""
        return self.records.get(request_id)

    def get_output(self, request_id: str) -> RequestOutput | None:
        """Return the output for ``request_id``, or ``None``."""
        return self.outputs.get(request_id)

    def signal(self, request_id: str) -> None:
        """Wake the request's waiter (if any) and the global output event."""
        event = self.events.get(request_id)
        if event is not None:
            event.set()
        self.output_event.set()

    def signal_all(self) -> None:
        """Wake every registered waiter and the global output event."""
        with self.request_lock:
            events = list(self.events.values())
        for event in events:
            event.set()
        self.output_event.set()

    def wait_any(self, timeout: float) -> bool:
        """Block until any request output changes (or the timeout elapses)."""
        result = self.output_event.wait(timeout)
        if result:
            self.output_event.clear()
        return result

    def wait_request(self, request_id: str, timeout: float) -> bool:
        """Block until ``request_id``'s output changes (or timeout elapses)."""
        event = self.events.get(request_id)
        if event is None:
            return False
        result = event.wait(timeout)
        if result:
            event.clear()
        return result

    def remove(self, request_id: str) -> None:
        """Drop the record and event for ``request_id`` (output retained)."""
        with self.request_lock:
            self.records.pop(request_id, None)
            self.events.pop(request_id, None)

    def track_finished(self, request_id: str) -> None:
        """Record a finished request id and evict the oldest retained outputs."""
        with self.output_lock:
            self.finished_ids.append(request_id)
            if self.max_outputs is not None:
                while len(self.finished_ids) > self.max_outputs:
                    evicted = self.finished_ids.popleft()
                    self.outputs.pop(evicted, None)
