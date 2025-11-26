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

"""ZeroMQ-powered tokenizer and detokenizer clients.

This module provides client classes for communicating with tokenizer and
detokenizer worker processes via ZeroMQ. The clients handle request/response
serialization and provide a clean API for tokenization and detokenization operations.

Note:
    This module is for internal use only and is not part of EasyDeL's public API.
    It is only accessible to EasyDeL modules that require external worker processes
    to handle specific tasks.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import zmq


@dataclass
class DetokenizerResult:
    """Result from a detokenization operation.

    Attributes:
        accumulated_text: The full decoded text accumulated so far.
        delta_text: The newly decoded text since the last decode call.
        last_decoded_index: Index of the last token that was decoded.
        finished: Whether this is the final decode for the request.
    """

    accumulated_text: str
    delta_text: str
    last_decoded_index: int
    finished: bool


class _BaseWorkerClient:
    """Base class for ZeroMQ worker clients.

    Provides common functionality for thread-safe communication with worker processes.

    Args:
        endpoint: The ZeroMQ endpoint to connect to.
    """

    def __init__(self, endpoint: str):
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(endpoint)
        self._lock = threading.Lock()

    def _request(self, payload: dict) -> dict:
        """Send a request to the worker and return the response.

        Args:
            payload: The request payload to send.

        Returns:
            The response from the worker.
        """
        with self._lock:
            self._socket.send_pyobj(payload)
            return self._socket.recv_pyobj()

    def close(self):
        """Close the ZeroMQ socket."""
        self._socket.close(0)


class TokenizerWorkerClient(_BaseWorkerClient):
    """Client for communicating with a tokenizer worker process.

    Args:
        endpoint: The ZeroMQ endpoint of the tokenizer worker.

    Raises:
        ValueError: If endpoint is not provided.
    """

    def __init__(self, endpoint: str):
        if not endpoint:
            raise ValueError("Tokenizer worker endpoint must be provided.")
        super().__init__(endpoint)

    def tokenize(self, request_id: str, prompt: str) -> list[int]:
        """Tokenize a text prompt.

        Args:
            request_id: Unique identifier for this request.
            prompt: The text to tokenize.

        Returns:
            List of token IDs.

        Raises:
            RuntimeError: If the worker returns an error.
        """
        resp = self._request({"cmd": "tokenize", "request_id": request_id, "prompt": prompt})
        if resp.get("status") != "ok":
            raise RuntimeError(resp.get("message", "Tokenizer worker failed"))
        return resp["tokens"]

    def drain(self) -> None:
        """Ensure all tokenizer-side buffers are flushed."""
        self._request({"cmd": "drain"})

    def shutdown(self) -> None:
        """Shutdown the tokenizer worker and close the connection."""
        try:
            self._request({"cmd": "shutdown"})
        except Exception:
            pass
        finally:
            self.close()


class DetokenizerWorkerClient(_BaseWorkerClient):
    """Client for communicating with a detokenizer worker process.

    Args:
        endpoint: The ZeroMQ endpoint of the detokenizer worker.

    Raises:
        ValueError: If endpoint is not provided.
    """

    def __init__(self, endpoint: str):
        if not endpoint:
            raise ValueError("Detokenizer worker endpoint must be provided.")
        super().__init__(endpoint)

    def decode(
        self,
        request_id: str,
        generated_tokens: list[int],
        *,
        finished: bool,
        skip_special_tokens: bool,
    ) -> DetokenizerResult:
        """Decode tokens incrementally.

        Args:
            request_id: Unique identifier for this request.
            generated_tokens: The list of tokens generated so far.
            finished: Whether generation is complete for this request.
            skip_special_tokens: Whether to skip special tokens in the final decode.

        Returns:
            DetokenizerResult containing the decoded text.

        Raises:
            RuntimeError: If the worker returns an error.
        """
        resp = self._request(
            {
                "cmd": "decode",
                "request_id": request_id,
                "tokens": generated_tokens,
                "finished": finished,
                "skip_special_tokens": skip_special_tokens,
            }
        )
        if resp.get("status") != "ok":
            raise RuntimeError(resp.get("message", "Detokenizer worker failed"))
        result_payload = resp["result"]
        return DetokenizerResult(**result_payload)

    def reset(self, request_id: str) -> None:
        """Reset the decoding state for a request.

        Args:
            request_id: The request ID to reset.
        """
        self._request({"cmd": "reset", "request_id": request_id})

    def drain(self) -> None:
        """Flush all detokenizer state (used during pause/resume)."""
        self._request({"cmd": "drain"})

    def shutdown(self) -> None:
        """Shutdown the detokenizer worker and close the connection."""
        try:
            self._request({"cmd": "shutdown"})
        except Exception:
            pass
        finally:
            self.close()
