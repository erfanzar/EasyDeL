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

"""Responses store worker process for handling persistent Responses API state via ZMQ."""

from __future__ import annotations

import argparse
import os
import traceback
import typing as tp
from pathlib import Path

import zmq

from easydel.workers.loggers import get_logger

from .file_store import FileResponseStore

logger = get_logger("ResponseStoreWorker")


def _worker(
    *,
    endpoint: str,
    storage_dir: str | None,
    max_stored_responses: int,
    max_stored_conversations: int,
    compression_level: int,
) -> None:
    if storage_dir is None:
        storage_dir = str(Path.home() / ".cache" / "easydel-response-store")
    store = FileResponseStore(
        storage_dir,
        max_stored_responses=max_stored_responses,
        max_stored_conversations=max_stored_conversations,
        compression_level=compression_level,
    )

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(endpoint)
    logger.info(f"Response store worker bound to {endpoint}")

    try:
        while True:
            message = socket.recv_pyobj()
            cmd = message.get("cmd")

            try:
                if cmd == "get_response":
                    response_id = message.get("response_id")
                    record = store.get_response(str(response_id) if response_id is not None else "")
                    socket.send_pyobj({"status": "ok", "record": record})

                elif cmd == "put_response":
                    response_id = message.get("response_id")
                    record = message.get("record")
                    if not isinstance(record, dict):
                        record = {}
                    store.put_response(
                        str(response_id) if response_id is not None else "", tp.cast(dict[str, tp.Any], record)
                    )
                    socket.send_pyobj({"status": "ok"})

                elif cmd == "delete_response":
                    response_id = message.get("response_id")
                    success = store.delete_response(str(response_id) if response_id is not None else "")
                    socket.send_pyobj({"status": "ok", "success": success})

                elif cmd == "get_conversation":
                    conversation_id = message.get("conversation_id")
                    history = store.get_conversation(str(conversation_id) if conversation_id is not None else "")
                    socket.send_pyobj({"status": "ok", "history": history})

                elif cmd == "put_conversation":
                    conversation_id = message.get("conversation_id")
                    history = message.get("history")
                    if not isinstance(history, list):
                        history = []
                    store.put_conversation(
                        str(conversation_id) if conversation_id is not None else "",
                        tp.cast(list[dict[str, tp.Any]], history),
                    )
                    socket.send_pyobj({"status": "ok"})

                elif cmd == "delete_conversation":
                    conversation_id = message.get("conversation_id")
                    success = store.delete_conversation(str(conversation_id) if conversation_id is not None else "")
                    socket.send_pyobj({"status": "ok", "success": success})

                elif cmd == "stats":
                    socket.send_pyobj({"status": "ok", "stats": store.stats()})

                elif cmd == "shutdown":
                    socket.send_pyobj({"status": "ok"})
                    break

                else:
                    socket.send_pyobj({"status": "error", "message": f"Unknown cmd {cmd}"})

            except Exception as exc:
                logger.error(f"Response store worker error: {exc}")
                logger.error(traceback.format_exc())
                socket.send_pyobj({"status": "error", "message": str(exc)})

    finally:
        store.close()
        socket.close(0)
        ctx.term()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--storage-dir", default=None)
    parser.add_argument("--max-stored-responses", type=int, default=10_000)
    parser.add_argument("--max-stored-conversations", type=int, default=1_000)
    parser.add_argument("--compression-level", type=int, default=3)
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    _worker(
        endpoint=args.endpoint,
        storage_dir=args.storage_dir,
        max_stored_responses=args.max_stored_responses,
        max_stored_conversations=args.max_stored_conversations,
        compression_level=args.compression_level,
    )


if __name__ == "__main__":
    main()
