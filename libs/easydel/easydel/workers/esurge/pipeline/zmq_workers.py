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

"""ZeroMQ REQ-socket clients for the tokenizer / detokenizer workers.

Provides the in-process companions to the worker loops defined in
``easydel.workers.esurge.pipeline.worker_main``. Each client owns a
single ``zmq.REQ`` socket guarded by a per-client lock so concurrent
FastAPI request handlers can share one client instance without
interleaving send / receive cycles.

Module exports:
    - :class:`DetokenizerResult`: streaming detokenization payload
      returned by :meth:`DetokenizerWorkerClient.decode`.
    - :class:`TokenizerWorkerClient`: thin wrapper around
      ``tokenize`` / ``drain`` / ``shutdown`` commands on the
      tokenizer worker.
    - :class:`DetokenizerWorkerClient`: incremental detokenization
      client that tracks per-request streamed length to send only the
      new token delta on subsequent calls.

Note:
    This module is for internal use only and is not part of EasyDeL's
    public API.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import zmq


@dataclass
class DetokenizerResult:
    """Result from a detokenization operation.

    Attributes:
        accumulated_text: The full decoded text accumulated so far for the
            request.
        delta_text: The newly decoded text since the last decode call.
        last_decoded_index: Index of the last token that was decoded.
        finished: Whether this is the final decode for the request.
        detoktook: Wall-clock time spent inside the worker on this decode
            step (seconds), used by the engine for telemetry.
    """

    accumulated_text: str
    delta_text: str
    last_decoded_index: int
    finished: bool
    detoktook: float


class _BaseWorkerClient:
    """Shared REQ-socket plumbing for tokenizer and detokenizer ZMQ clients.

    Connects a single ``zmq.REQ`` socket to the worker endpoint and
    serializes all sends/receives behind a lock. Subclasses
    (:class:`TokenizerWorkerClient`, :class:`DetokenizerWorkerClient`)
    layer command-specific helpers on top of :meth:`_request` and inherit
    :meth:`close` for orderly socket teardown.

    Instances are safe to share across worker threads because every
    request acquires :attr:`_lock` for the full ``send_pyobj`` /
    ``recv_pyobj`` round-trip; concurrent traffic to the same client
    simply queues on the lock.
    """

    def __init__(self, endpoint: str):
        """Connect a REQ socket to ``endpoint`` under a thread lock.

        Args:
            endpoint: ZMQ endpoint URI of the worker process.
        """
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(endpoint)
        self._lock = threading.Lock()

    def _request(self, payload: dict) -> dict:
        """Send one REQ/REP round-trip under the client lock.

        Args:
            payload: Command dictionary; must include a ``cmd`` key the
                worker recognises.

        Returns:
            dict: The full Pyobj-decoded worker reply. Subclasses are
            responsible for inspecting ``status`` and raising on errors.
        """
        with self._lock:
            self._socket.send_pyobj(payload)
            return self._socket.recv_pyobj()

    def close(self):
        """Close the local REQ socket without sending a shutdown command.

        Used in the attached (non-owning) shutdown path where the
        upstream worker is managed by an external lifecycle.
        """
        self._socket.close(0)


class TokenizerWorkerClient(_BaseWorkerClient):
    """ZMQ client for the bundled HuggingFace ``AutoTokenizer`` worker.

    Wraps the tokenizer worker spawned by :class:`WorkerManager` so the
    eSurge engine can delegate ``str -> token_ids`` conversion to a
    sibling process (avoiding GIL contention with the JAX hot path).
    Provides :meth:`tokenize` (one prompt per call), plus :meth:`drain`
    and :meth:`shutdown` for lifecycle control.
    """

    def __init__(self, endpoint: str):
        """Initialize the tokenizer client.

        Args:
            endpoint: ZMQ endpoint URI of the tokenizer worker.

        Raises:
            ValueError: When ``endpoint`` is empty.
        """
        if not endpoint:
            raise ValueError("Tokenizer worker endpoint must be provided.")
        super().__init__(endpoint)

    def tokenize(self, request_id: str, prompt: str) -> list[int]:
        """Encode a single text prompt into token ids on the worker.

        Args:
            request_id: Unique identifier for this request; currently
                forwarded but unused by the tokenizer worker (kept for
                future telemetry / cancellation hooks).
            prompt: Raw input string to encode.

        Returns:
            list[int]: Flat list of token ids produced by the worker's
            HuggingFace ``AutoTokenizer`` for ``prompt``.

        Raises:
            RuntimeError: When the worker returns a ``status != "ok"``
                payload (typically caused by an unrecognised command on
                a mismatched worker version).
        """
        resp = self._request({"cmd": "tokenize", "request_id": request_id, "prompt": prompt})
        if resp.get("status") != "ok":
            raise RuntimeError(resp.get("message", "Tokenizer worker failed"))
        return resp["tokens"]

    def drain(self) -> None:
        """Send ``drain`` to the tokenizer worker.

        Tokenization is stateless, so this is effectively a no-op
        acknowledgement included for symmetry with the detokenizer
        client's pause / resume flow.
        """
        self._request({"cmd": "drain"})

    def shutdown(self) -> None:
        """Send ``shutdown`` to the worker then close the local socket.

        Errors from the round-trip (e.g. when the worker has already
        exited) are swallowed so the local socket is always released.
        """
        try:
            self._request({"cmd": "shutdown"})
        except Exception:
            pass
        finally:
            self.close()


class DetokenizerWorkerClient(_BaseWorkerClient):
    """ZMQ client for the bundled incremental detokenizer worker.

    Talks to the detokenizer worker spawned by :class:`WorkerManager`
    which runs :class:`FastIncrementalDecoder` on top of the same
    HuggingFace tokenizer used for prompt encoding. The worker keeps a
    per-request decoder state (handling buffered partial UTF-8 sequences
    and SentencePiece prompt context) so this client only needs to send
    the latest token slice per call. Returns :class:`DetokenizerResult`
    instances that include both the cumulative ``accumulated_text`` and
    the streaming ``delta_text``.
    """

    def __init__(self, endpoint: str):
        """Initialize the detokenizer client.

        Args:
            endpoint: ZMQ endpoint URI of the detokenizer worker.

        Raises:
            ValueError: When ``endpoint`` is empty.
        """
        if not endpoint:
            raise ValueError("Detokenizer worker endpoint must be provided.")
        super().__init__(endpoint)
        self._sent_lengths: dict[str, int] = {}

    def decode(
        self,
        request_id: str,
        generated_tokens: list[int],
        *,
        finished: bool,
        skip_special_tokens: bool,
        spaces_between_special_tokens: bool = True,
        prompt_context: list[int] | None = None,
    ) -> DetokenizerResult:
        """Stream a detokenization step for a request to the worker.

        Tracks the number of tokens previously sent for ``request_id``
        in ``self._sent_lengths`` so only the *delta* slice plus a
        ``token_offset`` is forwarded on subsequent calls. The first
        call (and any ``finished=True`` call) also sends the full
        ``tokens`` list so the worker can recover from an out-of-sync
        offset. ``prompt_context`` is forwarded only once per request;
        the worker caches it for later calls.

        Args:
            request_id: Stable identifier shared with the worker's
                per-request decoder state.
            generated_tokens: Full list of generated token ids so far.
                The client computes the delta against the last call.
            finished: Whether this is the final detokenization step
                for the request; on ``True`` the worker re-decodes the
                full token list once for consistency and drops its
                state, and the client clears its sent-length record.
            skip_special_tokens: Forwarded to the tokenizer.
            spaces_between_special_tokens: Forwarded when supported by
                the underlying tokenizer.
            prompt_context: Optional last-N prompt token ids used as
                context on the very first detokenization step;
                SentencePiece-style tokenizers need this to avoid
                spurious leading spaces.

        Returns:
            DetokenizerResult: Wire reply mapped onto the
            :class:`DetokenizerResult` dataclass.

        Raises:
            RuntimeError: When the worker reports ``status != "ok"``.
        """
        prev_sent = int(self._sent_lengths.get(request_id, 0))
        total_tokens = len(generated_tokens)
        if prev_sent < 0 or prev_sent > total_tokens:
            prev_sent = 0
        token_delta = generated_tokens[prev_sent:]

        msg = {
            "cmd": "decode",
            "request_id": request_id,
            "tokens_delta": token_delta,
            "token_offset": prev_sent,
            "total_tokens": total_tokens,
            "finished": finished,
            "skip_special_tokens": skip_special_tokens,
            "spaces_between_special_tokens": spaces_between_special_tokens,
        }
        if prev_sent == 0 or finished:
            msg["tokens"] = generated_tokens
        if prompt_context:
            msg["prompt_context"] = prompt_context
        resp = self._request(msg)
        if resp.get("status") != "ok":
            raise RuntimeError(resp.get("message", "Detokenizer worker failed"))
        if finished:
            self._sent_lengths.pop(request_id, None)
        else:
            self._sent_lengths[request_id] = total_tokens
        result_payload = resp["result"]
        return DetokenizerResult(**result_payload)

    def reset(self, request_id: str) -> None:
        """Drop per-request state on both client and worker.

        Used when a request is cancelled or restarted so the worker
        does not retain stale decoder state for the same id.

        Args:
            request_id: The request id whose state should be cleared.
        """
        self._sent_lengths.pop(request_id, None)
        self._request({"cmd": "reset", "request_id": request_id})

    def drain(self) -> None:
        """Flush every per-request decoder state on the worker.

        Used during pause / resume cycles when the engine needs the
        worker to forget all in-flight decoding before reconfiguring.
        """
        self._request({"cmd": "drain"})

    def shutdown(self) -> None:
        """Send ``shutdown`` to the worker then close the local socket.

        Clears the local sent-length bookkeeping before contacting the
        worker so a subsequent reuse of the same client (in tests) is
        safe. Errors from the round-trip are swallowed so the local
        socket is always released.
        """
        self._sent_lengths.clear()
        try:
            self._request({"cmd": "shutdown"})
        except Exception:
            pass
        finally:
            self.close()
