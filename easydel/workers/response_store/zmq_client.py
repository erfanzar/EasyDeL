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

"""ZMQ REQ-socket client for the Responses API state store worker.

Exposes :class:`ResponseStoreWorkerClient`, the thin wrapper used by
the API server to read and write OpenAI Responses API records
(``response_id`` payloads and ``conversation_id`` histories) against
the persistent :class:`FileResponseStore` running in the worker
subprocess. Every public method maps 1:1 to a worker command and
serialises the underlying ZMQ round-trip behind a lock so concurrent
request handlers can share one client safely.
"""

from __future__ import annotations

import threading
import typing as tp

import zmq


class ResponseStoreWorkerClient:
    """Thread-safe ZMQ client for communicating with a response store worker.

    Provides methods to get, put, and delete responses and conversations
    from a remote ``FileResponseStore`` running in a separate process.

    Args:
        endpoint: ZeroMQ endpoint of the response store worker.

    Raises:
        ValueError: If ``endpoint`` is empty.
    """

    def __init__(self, endpoint: str):
        """Connect a REQ socket to ``endpoint``.

        Args:
            endpoint: ZMQ endpoint URI to connect to. Must be non-empty.

        Raises:
            ValueError: When ``endpoint`` is the empty string.
        """
        if not endpoint:
            raise ValueError("Response store worker endpoint must be provided.")
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(endpoint)
        self._lock = threading.Lock()
        self._endpoint = endpoint

    def _request(self, payload: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Send a request and return the worker's response under a lock.

        Args:
            payload: Command dict consumed by ``worker_main``.

        Returns:
            dict[str, tp.Any]: The worker's response dict.

        Raises:
            RuntimeError: If the worker reported ``status == "error"``.
        """
        with self._lock:
            self._socket.send_pyobj(payload)
            resp = tp.cast(dict[str, tp.Any], self._socket.recv_pyobj())
            if resp.get("status") == "error":
                raise RuntimeError(resp.get("message", "Response store worker failed"))
            return resp

    def get_response(self, response_id: str) -> dict[str, tp.Any] | None:
        """Retrieve a response record by ID from the remote store.

        Args:
            response_id: Unique response identifier.

        Returns:
            The response record dict, or ``None`` if not found.
        """
        resp = self._request({"cmd": "get_response", "response_id": response_id})
        record = resp.get("record")
        return tp.cast(dict[str, tp.Any], record) if isinstance(record, dict) else None

    def put_response(self, response_id: str, record: dict[str, tp.Any]) -> None:
        """Store or update a response record in the remote store.

        Args:
            response_id: Unique response identifier.
            record: The response data to persist.
        """
        self._request({"cmd": "put_response", "response_id": response_id, "record": record})

    def delete_response(self, response_id: str) -> bool:
        """Delete a response record from the remote store.

        Args:
            response_id: Unique response identifier.

        Returns:
            ``True`` if the record was found and deleted, ``False`` otherwise.
        """
        resp = self._request({"cmd": "delete_response", "response_id": response_id})
        return bool(resp.get("success"))

    def get_conversation(self, conversation_id: str) -> list[dict[str, tp.Any]] | None:
        """Retrieve a conversation history by ID from the remote store.

        Args:
            conversation_id: Unique conversation identifier.

        Returns:
            List of message dicts, or ``None`` if not found.
        """
        resp = self._request({"cmd": "get_conversation", "conversation_id": conversation_id})
        history = resp.get("history")
        return tp.cast(list[dict[str, tp.Any]], history) if isinstance(history, list) else None

    def put_conversation(self, conversation_id: str, history: list[dict[str, tp.Any]]) -> None:
        """Store or update a conversation history in the remote store.

        Args:
            conversation_id: Unique conversation identifier.
            history: List of message dicts representing the conversation.
        """
        self._request({"cmd": "put_conversation", "conversation_id": conversation_id, "history": history})

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation record from the remote store.

        Args:
            conversation_id: Unique conversation identifier.

        Returns:
            ``True`` if the record was found and deleted, ``False`` otherwise.
        """
        resp = self._request({"cmd": "delete_conversation", "conversation_id": conversation_id})
        return bool(resp.get("success"))

    def stats(self) -> dict[str, tp.Any]:
        """Retrieve store statistics from the remote worker.

        Returns:
            Dictionary with counts of stored responses/conversations,
            their capacity limits, and storage directory path.
        """
        resp = self._request({"cmd": "stats"})
        stats = resp.get("stats")
        return tp.cast(dict[str, tp.Any], stats) if isinstance(stats, dict) else {}

    def shutdown(self) -> None:
        """Send a shutdown command to the worker and close the connection."""
        try:
            self._request({"cmd": "shutdown"})
        finally:
            self.close()

    def close(self) -> None:
        """Close the ZeroMQ socket without sending a shutdown command."""
        self._socket.close(0)

    @property
    def endpoint(self) -> str:
        """Return the ZMQ endpoint this client is connected to.

        Returns:
            str: The endpoint URI passed to the constructor.
        """
        return self._endpoint

    @property
    def enabled(self) -> bool:
        """Whether the client is active.

        Returns:
            bool: Always ``True`` for this client; provided for interface
            compatibility with stub clients used when persistence is
            disabled.
        """
        return True
