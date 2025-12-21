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

"""ZeroMQ client for communicating with the response store worker process."""

from __future__ import annotations

import threading
import typing as tp

import zmq


class ResponseStoreWorkerClient:
    """Client for communicating with a response store worker process via ZMQ."""

    def __init__(self, endpoint: str):
        if not endpoint:
            raise ValueError("Response store worker endpoint must be provided.")
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(endpoint)
        self._lock = threading.Lock()
        self._endpoint = endpoint

    def _request(self, payload: dict[str, tp.Any]) -> dict[str, tp.Any]:
        with self._lock:
            self._socket.send_pyobj(payload)
            resp = tp.cast(dict[str, tp.Any], self._socket.recv_pyobj())
            if resp.get("status") == "error":
                raise RuntimeError(resp.get("message", "Response store worker failed"))
            return resp

    def get_response(self, response_id: str) -> dict[str, tp.Any] | None:
        resp = self._request({"cmd": "get_response", "response_id": response_id})
        record = resp.get("record")
        return tp.cast(dict[str, tp.Any], record) if isinstance(record, dict) else None

    def put_response(self, response_id: str, record: dict[str, tp.Any]) -> None:
        self._request({"cmd": "put_response", "response_id": response_id, "record": record})

    def delete_response(self, response_id: str) -> bool:
        resp = self._request({"cmd": "delete_response", "response_id": response_id})
        return bool(resp.get("success"))

    def get_conversation(self, conversation_id: str) -> list[dict[str, tp.Any]] | None:
        resp = self._request({"cmd": "get_conversation", "conversation_id": conversation_id})
        history = resp.get("history")
        return tp.cast(list[dict[str, tp.Any]], history) if isinstance(history, list) else None

    def put_conversation(self, conversation_id: str, history: list[dict[str, tp.Any]]) -> None:
        self._request({"cmd": "put_conversation", "conversation_id": conversation_id, "history": history})

    def delete_conversation(self, conversation_id: str) -> bool:
        resp = self._request({"cmd": "delete_conversation", "conversation_id": conversation_id})
        return bool(resp.get("success"))

    def stats(self) -> dict[str, tp.Any]:
        resp = self._request({"cmd": "stats"})
        stats = resp.get("stats")
        return tp.cast(dict[str, tp.Any], stats) if isinstance(stats, dict) else {}

    def shutdown(self) -> None:
        try:
            self._request({"cmd": "shutdown"})
        finally:
            self.close()

    def close(self) -> None:
        self._socket.close(0)

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def enabled(self) -> bool:
        return True
