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

"""ZeroMQ REQ-socket client for the tokenizer worker.

Streaming detokenization is owned by :class:`surgecore.Detokenizer` in the engine process and has no ZMQ client or
fallback transport. This module retains only the stateless prompt-tokenization client.
"""

from __future__ import annotations

import threading

import zmq

_BATCH_MAX_PROMPTS = 64
"""Most prompts in one ``tokenize_batch`` request."""

_BATCH_MAX_CHARS = 4_000_000
"""Character budget of one ``tokenize_batch`` request (about 0.7M tokens of source text); a longer prompt goes alone."""


def _prompt_chunks(prompts: list[str]) -> list[list[str]]:
    """Split ``prompts``, in order, into chunks within ``_BATCH_MAX_PROMPTS`` and ``_BATCH_MAX_CHARS``."""
    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for prompt in prompts:
        if current and (len(current) >= _BATCH_MAX_PROMPTS or size + len(prompt) > _BATCH_MAX_CHARS):
            chunks.append(current)
            current, size = [], 0
        current.append(prompt)
        size += len(prompt)
    if current:
        chunks.append(current)
    return chunks


class TokenizerWorkerClient:
    """Thread-safe client for the bundled HuggingFace ``AutoTokenizer`` worker."""

    def __init__(self, endpoint: str, request_timeout: float = 30.0):
        """Connect to the tokenizer worker with a bounded request deadline in seconds."""
        if not endpoint:
            raise ValueError("Tokenizer worker endpoint must be provided.")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        self._context = zmq.Context.instance()
        self._endpoint = endpoint
        self._request_timeout_ms = max(1, int(request_timeout * 1000))
        self._lock = threading.RLock()
        self._socket = self._connect_socket()

    def _connect_socket(self):
        """Create a fresh REQ socket, including after an interrupted round trip."""
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.SNDTIMEO, self._request_timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, self._request_timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self._endpoint)
        return socket

    def _request(self, payload: dict) -> dict:
        """Send one REQ/REP round-trip under the client lock without replaying timed-out commands."""
        with self._lock:
            try:
                self._socket.send_pyobj(payload)
                return self._socket.recv_pyobj()
            except zmq.Again as error:
                self._socket.close(0)
                self._socket = self._connect_socket()
                raise TimeoutError(f"Worker request timed out: {payload.get('cmd')}") from error

    def close(self) -> None:
        """Close the local REQ socket without stopping an externally managed worker."""
        self._socket.close(0)

    def tokenize(self, request_id: str, prompt: str) -> list[int]:
        """Encode one text prompt into a flat token-id list."""
        resp = self._request({"cmd": "tokenize", "request_id": request_id, "prompt": prompt})
        if resp.get("status") != "ok":
            raise RuntimeError(resp.get("message", "Tokenizer worker failed"))
        return resp["tokens"]

    def tokenize_batch(self, prompts: list[str]) -> list[list[int]]:
        """Encode several prompts, in order, in as few round trips as the per-request size budget allows.

        Prompts go to the worker in chunks of at most ``_BATCH_MAX_PROMPTS`` prompts and about ``_BATCH_MAX_CHARS``
        characters (a chunk always holds at least one prompt), so one request stays far inside the request deadline.
        Each prompt's ids equal those of :meth:`tokenize`.
        """
        tokens: list[list[int]] = []
        for chunk in _prompt_chunks(prompts):
            resp = self._request({"cmd": "tokenize_batch", "prompts": chunk})
            if resp.get("status") != "ok":
                raise RuntimeError(resp.get("message", "Tokenizer worker failed"))
            if len(resp["tokens"]) != len(chunk):
                raise RuntimeError(f"Tokenizer worker returned {len(resp['tokens'])} encodings for {len(chunk)} prompts")
            tokens.extend(resp["tokens"])
        return tokens

    def drain(self) -> None:
        """Acknowledge a drain; prompt tokenization is stateless."""
        self._request({"cmd": "drain"})

    def shutdown(self) -> None:
        """Send ``shutdown`` and always release the local socket."""
        try:
            self._request({"cmd": "shutdown"})
        except Exception:
            pass
        finally:
            self.close()
