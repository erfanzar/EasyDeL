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

"""``FileResponseStore`` must not leave orphan ``.bak`` files behind (pure Python)."""

import pytest
from easydel.workers.response_store.file_store import FileResponseStore

pytestmark = pytest.mark.cpu_ok


def _record_files(directory):
    return sorted(p.name for p in directory.iterdir())


def test_overwrite_leaves_no_backup_files(tmp_path):
    store = FileResponseStore(tmp_path)
    for turn in range(3):
        store.put_conversation("conv-1", [{"role": "user", "content": str(turn)}])
        store.put_response("resp_1", {"id": "resp_1", "turn": turn})
    assert len(_record_files(store.conversations_dir)) == 1
    assert _record_files(store.responses_dir) == ["resp_1.bin"]
    assert store.get_response("resp_1") == {"id": "resp_1", "turn": 2}
    assert store.get_conversation("conv-1") == [{"role": "user", "content": "2"}]


def test_delete_and_eviction_remove_all_files(tmp_path):
    store = FileResponseStore(tmp_path, max_stored_responses=1, max_stored_conversations=1)
    store.put_response("a", {"v": 1})
    store.put_response("a", {"v": 2})
    store.put_response("b", {"v": 1})  # evicts "a"
    assert _record_files(store.responses_dir) == ["b.bin"]
    store.put_conversation("c1", [{"x": 1}])
    store.put_conversation("c1", [{"x": 2}])
    store.put_conversation("c2", [{"x": 1}])  # evicts "c1"
    assert len(_record_files(store.conversations_dir)) == 1
    assert store.delete_response("b")
    assert store.delete_conversation("c2")
    assert _record_files(store.responses_dir) == []
    assert _record_files(store.conversations_dir) == []


def test_index_still_keeps_its_recovery_backup(tmp_path):
    store = FileResponseStore(tmp_path)
    store.put_response("a", {"v": 1})
    store.put_response("b", {"v": 2})
    assert (tmp_path / "index.json.bak").exists()
