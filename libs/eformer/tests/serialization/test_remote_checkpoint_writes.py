# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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
import datetime as dt
import json

import numpy as np
import pytest

from eformer.serialization import async_manager as async_mod
from eformer.serialization import checkpointer as checkpointer_mod
from eformer.serialization import serialization as serialization_mod


class FakePath:
    def __init__(self, path: str, store: dict[str, str], mkdir_calls: list[str]):
        self.path = str(path)
        self.store = store
        self.mkdir_calls = mkdir_calls

    def __truediv__(self, other):
        return FakePath(f"{self.path.rstrip('/')}/{other}", self.store, self.mkdir_calls)

    def exists(self) -> bool:
        return self.path in self.store

    def read_text(self, encoding: str = "utf-8") -> str:
        del encoding
        return self.store[self.path]

    def write_text(self, data: str, encoding: str = "utf-8") -> None:
        del encoding
        self.store[self.path] = data

    def mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        del parents, exist_ok
        self.mkdir_calls.append(self.path)

    def __str__(self) -> str:
        return self.path


def make_fake_epath(store: dict[str, str], mkdir_calls: list[str]):
    def factory(path):
        return FakePath(str(path), store, mkdir_calls)

    return factory


class RecordingSerializeManager:
    def __init__(self):
        self.calls = []

    def serialize_with_paths(self, arrays, paths, on_commit_callback=None):
        self.calls.append({"arrays": arrays, "paths": paths})
        if on_commit_callback is not None:
            on_commit_callback()

    def wait_until_finished(self):
        pass


class RecordingGlobalManager:
    def __init__(self):
        self.wait_calls = 0

    def wait_until_finished(self):
        self.wait_calls += 1


class RecordingCheckpointManager:
    def __init__(self):
        self.calls = []

    def save_pytree(self, **kwargs):
        self.calls.append(kwargs)


def build_checkpointer(base_path: str, manager: RecordingCheckpointManager):
    checkpointer = object.__new__(checkpointer_mod.Checkpointer)
    checkpointer.base_path = base_path
    checkpointer._manager = manager
    checkpointer._dt_now_injection = lambda: dt.datetime(2026, 1, 1)
    checkpointer._last_save_time = checkpointer._dt_now_injection()
    checkpointer._last_save_step = 0
    return checkpointer


def test_tree_serialize_leaves_remote_nonzero_skips_shared_index_write(monkeypatch):
    store = {}
    mkdir_calls = []
    manager = RecordingSerializeManager()

    monkeypatch.setattr(serialization_mod, "ePath", make_fake_epath(store, mkdir_calls))
    monkeypatch.setattr(serialization_mod.fsspec_utils.jax, "process_index", lambda: 1)

    pytree = {"params": np.arange(4)}
    serialization_mod.tree_serialize_leaves(
        checkpoint_dir="gs://bucket/run-1",
        pytree=pytree,
        manager=manager,
        prefix="model",
        write_index=True,
    )

    assert len(manager.calls) == 1
    assert manager.calls[0]["paths"] == ["gs://bucket/run-1/model/params"]
    assert "gs://bucket/run-1/tensorstore_index.json" not in store
    assert mkdir_calls == []


def test_tree_serialize_leaves_local_nonzero_still_writes_index(monkeypatch):
    store = {}
    mkdir_calls = []
    manager = RecordingSerializeManager()

    monkeypatch.setattr(serialization_mod, "ePath", make_fake_epath(store, mkdir_calls))
    monkeypatch.setattr(serialization_mod.fsspec_utils.jax, "process_index", lambda: 1)

    serialization_mod.tree_serialize_leaves(
        checkpoint_dir="/tmp/run-local",
        pytree={"params": np.arange(4)},
        manager=manager,
        prefix="model",
        write_index=True,
    )

    assert len(manager.calls) == 1
    index = json.loads(store["/tmp/run-local/tensorstore_index.json"])
    assert index["prefixes"]["model"][0]["path"] == "model/params"
    assert mkdir_calls == []


def test_async_save_pytree_remote_nonzero_skips_shared_writes_but_still_serializes(monkeypatch):
    store = {}
    mkdir_calls = []
    serialize_calls = []
    sync_calls = []
    manager = async_mod.AsyncCheckpointManager(use_tensorstore=True)
    manager._global_manager = RecordingGlobalManager()

    def fake_tree_serialize_leaves(**kwargs):
        serialize_calls.append(kwargs)

    monkeypatch.setattr(async_mod, "ePath", make_fake_epath(store, mkdir_calls))
    monkeypatch.setattr(async_mod, "tree_serialize_leaves", fake_tree_serialize_leaves)
    monkeypatch.setattr(
        async_mod,
        "_sync_remote_checkpoint_visibility",
        lambda path, *, scope: sync_calls.append((str(path), scope)),
    )
    monkeypatch.setattr(async_mod.fsspec_utils.jax, "process_index", lambda: 1)

    path = manager.save_pytree(
        pytree={"params": np.arange(4)},
        path="gs://bucket/run-2",
        prefix="model",
    )

    assert path == "gs://bucket/run-2"
    assert len(serialize_calls) == 1
    assert serialize_calls[0]["write_index"] is False
    assert manager._global_manager.wait_calls == 1
    assert sync_calls == [("gs://bucket/run-2", "save-pytree:model")]
    assert mkdir_calls == []
    assert store == {}


def test_async_save_pytree_remote_primary_writes_shared_metadata(monkeypatch):
    store = {}
    mkdir_calls = []
    serialize_calls = []
    sync_calls = []
    manager = async_mod.AsyncCheckpointManager(use_tensorstore=True)
    manager._global_manager = RecordingGlobalManager()

    def fake_tree_serialize_leaves(**kwargs):
        serialize_calls.append(kwargs)
        if kwargs["write_index"]:
            store["gs://bucket/run-3/tensorstore_index.json"] = json.dumps(
                {
                    "format": "tensorstore",
                    "prefixes": {
                        "model": [
                            {
                                "path": "model/params",
                                "shape": [4],
                                "dtype": str(np.arange(4).dtype),
                            }
                        ]
                    },
                }
            )

    monkeypatch.setattr(async_mod, "ePath", make_fake_epath(store, mkdir_calls))
    monkeypatch.setattr(async_mod, "tree_serialize_leaves", fake_tree_serialize_leaves)
    monkeypatch.setattr(
        async_mod,
        "_sync_remote_checkpoint_visibility",
        lambda path, *, scope: sync_calls.append((str(path), scope)),
    )
    monkeypatch.setattr(async_mod.fsspec_utils.jax, "process_index", lambda: 0)

    path = manager.save_pytree(
        pytree={"params": np.arange(4)},
        path="gs://bucket/run-3",
        prefix="model",
        extras={"step": 3},
    )

    assert path == "gs://bucket/run-3"
    assert len(serialize_calls) == 1
    assert serialize_calls[0]["write_index"] is True
    assert sync_calls == [("gs://bucket/run-3", "save-pytree:model")]
    assert mkdir_calls == ["gs://bucket/run-3"]
    assert "gs://bucket/run-3/model_structure.json" in store
    assert "gs://bucket/run-3/checkpoint_metadata.json" in store

    structure = json.loads(store["gs://bucket/run-3/model_structure.json"])
    assert structure["prefix"] == "model"
    assert structure["array_relpaths"] == ["model/params"]


def test_async_save_tree_remote_nonzero_syncs_before_callback(monkeypatch):
    store = {}
    mkdir_calls = []
    serialize_calls = []
    event_order = []
    manager = async_mod.AsyncCheckpointManager(use_tensorstore=True)
    manager._global_manager = RecordingGlobalManager()

    def fake_tree_serialize_leaves(**kwargs):
        serialize_calls.append(kwargs)

    monkeypatch.setattr(async_mod, "ePath", make_fake_epath(store, mkdir_calls))
    monkeypatch.setattr(async_mod, "tree_serialize_leaves", fake_tree_serialize_leaves)
    monkeypatch.setattr(
        async_mod,
        "_sync_remote_checkpoint_visibility",
        lambda path, *, scope: event_order.append(("sync", str(path), scope)),
    )
    monkeypatch.setattr(async_mod.fsspec_utils.jax, "process_index", lambda: 1)

    path = manager.save_tree(
        tree={"params": np.arange(4)},
        path="gs://bucket/run-4",
        mesh=object(),
        metadata={"step": 4},
        callback=lambda saved_path: event_order.append(("callback", saved_path)),
    )

    assert path == "gs://bucket/run-4"
    assert len(serialize_calls) == 1
    assert serialize_calls[0]["write_index"] is False
    assert event_order == [
        ("sync", "gs://bucket/run-4", "save-tree:root"),
        ("callback", "gs://bucket/run-4"),
    ]
    assert "gs://bucket/run-4/checkpoint_metadata.json" not in store


@pytest.mark.parametrize(
    ("base_path", "process_index", "expected_mkdirs"),
    [
        ("gs://bucket/checkpoints", 1, []),
        ("gs://bucket/checkpoints", 0, ["gs://bucket/checkpoints/run-7"]),
        ("/tmp/checkpoints", 1, ["/tmp/checkpoints/run-7"]),
    ],
)
def test_checkpointer_save_pytree_only_gates_remote_shared_setup(
    monkeypatch,
    base_path,
    process_index,
    expected_mkdirs,
):
    mkdir_calls = []
    manager = RecordingCheckpointManager()
    checkpointer = build_checkpointer(base_path=base_path, manager=manager)

    monkeypatch.setattr(checkpointer_mod.fsspec_utils, "mkdirs", lambda path: mkdir_calls.append(path))
    monkeypatch.setattr(checkpointer_mod.fsspec_utils.jax, "process_index", lambda: process_index)
    monkeypatch.setattr(checkpointer_mod, "_write_checkpoint_metadata", lambda *args, **kwargs: None)

    path = checkpointer_mod.Checkpointer.save_pytree(
        checkpointer,
        tree={"params": np.arange(2)},
        prefix="model",
        step=7,
    )

    assert path == f"{base_path}/run-7"
    assert mkdir_calls == expected_mkdirs
    assert len(manager.calls) == 1
    assert manager.calls[0]["path"] == f"{base_path}/run-7"
    assert manager.calls[0]["prefix"] == "model"
