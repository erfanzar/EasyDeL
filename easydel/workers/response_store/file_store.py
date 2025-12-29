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

from __future__ import annotations

import hashlib
import json
import time
import typing as tp
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from easydel.workers.loggers import get_logger

logger = get_logger("ResponseStore")


@dataclass
class _Entry:
    file: str
    created_at: float
    touched_at: float


class FileResponseStore:
    """File-backed persistent store for Responses API state.

    Storage layout:
        <storage_dir>/
            index.json
            responses/
                <response_id>.bin
            conversations/
                <sha256(conversation_id)>.bin
    """

    def __init__(
        self,
        storage_dir: str | Path,
        *,
        max_stored_responses: int = 10_000,
        max_stored_conversations: int = 1_000,
        compression_level: int = 3,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.responses_dir = self.storage_dir / "responses"
        self.conversations_dir = self.storage_dir / "conversations"
        self.index_file = self.storage_dir / "index.json"

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

        self._max_stored_responses = max(0, int(max_stored_responses))
        self._max_stored_conversations = max(0, int(max_stored_conversations))
        self._compression_level = max(0, min(int(compression_level), 9))

        self._responses: OrderedDict[str, _Entry] = OrderedDict()
        self._conversations: OrderedDict[str, _Entry] = OrderedDict()
        self._dirty = False
        self._load_index()

        logger.info(f"Response store initialized at: {self.storage_dir}")

    def _encode(self, obj: tp.Any) -> bytes:
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if self._compression_level <= 0:
            return data
        return zlib.compress(data, level=self._compression_level)

    def _decode(self, blob: bytes) -> tp.Any:
        try:
            data = zlib.decompress(blob)
        except zlib.error:
            data = blob
        return json.loads(data.decode("utf-8"))

    def _atomic_write(self, path: Path, data: bytes) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        backup_path = path.with_suffix(path.suffix + ".bak")

        temp_path.write_bytes(data)

        if path.exists():
            if backup_path.exists():
                backup_path.unlink()
            path.rename(backup_path)

        temp_path.rename(path)

    def _conversation_file_name(self, conversation_id: str) -> str:
        digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()
        return f"{digest}.bin"

    def _load_index(self) -> None:
        if not self.index_file.exists():
            self._dirty = True
            self._flush_index()
            return

        try:
            raw = json.loads(self.index_file.read_text(encoding="utf-8"))
        except Exception:
            backup = self.index_file.with_suffix(".json.bak")
            if backup.exists():
                raw = json.loads(backup.read_text(encoding="utf-8"))
            else:
                raw = {}

        responses = tp.cast(dict[str, tp.Any], raw.get("responses") or {})
        conversations = tp.cast(dict[str, tp.Any], raw.get("conversations") or {})

        def _load_entries(items: dict[str, tp.Any], base_dir: Path) -> list[tuple[str, _Entry]]:
            loaded: list[tuple[str, _Entry]] = []
            for key, meta in items.items():
                if not isinstance(meta, dict):
                    continue
                file = meta.get("file")
                if not isinstance(file, str) or not file:
                    continue
                path = base_dir / file
                if not path.exists():
                    continue
                try:
                    created_at = float(meta.get("created_at") or 0.0)
                except (TypeError, ValueError):
                    created_at = 0.0
                try:
                    touched_at = float(meta.get("touched_at") or created_at or 0.0)
                except (TypeError, ValueError):
                    touched_at = created_at or 0.0
                loaded.append((key, _Entry(file=file, created_at=created_at, touched_at=touched_at)))
            loaded.sort(key=lambda item: item[1].touched_at)
            return loaded

        self._responses = OrderedDict(
            _load_entries(tp.cast(dict[str, tp.Any], responses.get("items") or {}), self.responses_dir)
        )
        self._conversations = OrderedDict(
            _load_entries(tp.cast(dict[str, tp.Any], conversations.get("items") or {}), self.conversations_dir)
        )

    def _flush_index(self) -> None:
        if not self._dirty and self.index_file.exists():
            return

        payload = {
            "version": 1,
            "saved_at": time.time(),
            "responses": {
                "max": self._max_stored_responses,
                "items": {
                    key: {"file": entry.file, "created_at": entry.created_at, "touched_at": entry.touched_at}
                    for key, entry in self._responses.items()
                },
            },
            "conversations": {
                "max": self._max_stored_conversations,
                "items": {
                    key: {"file": entry.file, "created_at": entry.created_at, "touched_at": entry.touched_at}
                    for key, entry in self._conversations.items()
                },
            },
        }

        self._atomic_write(self.index_file, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
        self._dirty = False

    def _touch(self, store: OrderedDict[str, _Entry], key: str) -> None:
        entry = store.get(key)
        if entry is None:
            return
        entry.touched_at = time.time()
        store.move_to_end(key)
        self._dirty = True

    def _evict(self) -> None:
        if self._max_stored_responses == 0:
            for entry in self._responses.values():
                try:
                    (self.responses_dir / entry.file).unlink(missing_ok=True)
                except Exception:
                    pass
            self._responses.clear()

        while self._max_stored_responses > 0 and len(self._responses) > self._max_stored_responses:
            _key, entry = self._responses.popitem(last=False)
            try:
                (self.responses_dir / entry.file).unlink(missing_ok=True)
            except Exception:
                pass
            self._dirty = True

        if self._max_stored_conversations == 0:
            for entry in self._conversations.values():
                try:
                    (self.conversations_dir / entry.file).unlink(missing_ok=True)
                except Exception:
                    pass
            self._conversations.clear()

        while self._max_stored_conversations > 0 and len(self._conversations) > self._max_stored_conversations:
            _key, entry = self._conversations.popitem(last=False)
            try:
                (self.conversations_dir / entry.file).unlink(missing_ok=True)
            except Exception:
                pass
            self._dirty = True

    def get_response(self, response_id: str) -> dict[str, tp.Any] | None:
        if not response_id:
            return None
        entry = self._responses.get(response_id)
        if entry is None:
            return None

        path = self.responses_dir / entry.file
        if not path.exists():
            del self._responses[response_id]
            self._dirty = True
            return None

        record = self._decode(path.read_bytes())
        self._touch(self._responses, response_id)
        return tp.cast(dict[str, tp.Any], record) if isinstance(record, dict) else None

    def put_response(self, response_id: str, record: dict[str, tp.Any]) -> None:
        if not response_id or self._max_stored_responses == 0:
            return

        now = time.time()
        file_name = f"{response_id}.bin"
        path = self.responses_dir / file_name
        self._atomic_write(path, self._encode(record))

        created_at = record.get("created_at")
        try:
            created_at_f = float(created_at) if created_at is not None else now
        except (TypeError, ValueError):
            created_at_f = now

        self._responses[response_id] = _Entry(file=file_name, created_at=created_at_f, touched_at=now)
        self._responses.move_to_end(response_id)
        self._dirty = True
        self._evict()
        self._flush_index()

    def delete_response(self, response_id: str) -> bool:
        if not response_id:
            return False
        entry = self._responses.pop(response_id, None)
        if entry is None:
            return False
        try:
            (self.responses_dir / entry.file).unlink(missing_ok=True)
        except Exception:
            pass
        self._dirty = True
        self._flush_index()
        return True

    def get_conversation(self, conversation_id: str) -> list[dict[str, tp.Any]] | None:
        if not conversation_id:
            return None
        entry = self._conversations.get(conversation_id)
        if entry is None:
            return None

        path = self.conversations_dir / entry.file
        if not path.exists():
            del self._conversations[conversation_id]
            self._dirty = True
            return None

        payload = self._decode(path.read_bytes())
        history = payload.get("history") if isinstance(payload, dict) else None
        self._touch(self._conversations, conversation_id)
        return tp.cast(list[dict[str, tp.Any]], history) if isinstance(history, list) else None

    def put_conversation(self, conversation_id: str, history: list[dict[str, tp.Any]]) -> None:
        if not conversation_id or self._max_stored_conversations == 0:
            return

        now = time.time()
        file_name = self._conversation_file_name(conversation_id)
        path = self.conversations_dir / file_name
        self._atomic_write(path, self._encode({"id": conversation_id, "history": history}))

        entry = self._conversations.get(conversation_id)
        created_at = entry.created_at if entry is not None else now
        self._conversations[conversation_id] = _Entry(file=file_name, created_at=created_at, touched_at=now)
        self._conversations.move_to_end(conversation_id)
        self._dirty = True
        self._evict()
        self._flush_index()

    def delete_conversation(self, conversation_id: str) -> bool:
        if not conversation_id:
            return False
        entry = self._conversations.pop(conversation_id, None)
        if entry is None:
            return False
        try:
            (self.conversations_dir / entry.file).unlink(missing_ok=True)
        except Exception:
            pass
        self._dirty = True
        self._flush_index()
        return True

    def stats(self) -> dict[str, tp.Any]:
        return {
            "responses": len(self._responses),
            "conversations": len(self._conversations),
            "max_stored_responses": self._max_stored_responses,
            "max_stored_conversations": self._max_stored_conversations,
            "storage_dir": str(self.storage_dir),
        }

    def close(self) -> None:
        try:
            self._flush_index()
        except Exception:
            pass
