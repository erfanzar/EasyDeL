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

"""Tests for path utilities."""

import jax.numpy as jnp
import numpy as np

from eformer.paths import GCSPath, LocalPath, MLUtilPath, PathManager, is_local_path, is_remote_path, path_protocol


def test_local_path_read_write_and_stat(tmp_path):
    target = LocalPath(tmp_path / "nested" / "file.txt")
    target.write_text("hello")
    assert target.exists()
    assert target.is_file()
    assert target.read_text() == "hello"

    data = b"binary"
    binary_target = LocalPath(tmp_path / "nested" / "blob.bin")
    binary_target.write_bytes(data)
    assert binary_target.read_bytes() == data

    stats = target.stat()
    assert stats["size"] == len("hello")


def test_local_path_iterdir_glob_and_manipulation(tmp_path):
    base = LocalPath(tmp_path / "tree")
    base.mkdir()

    (base / "a.txt").write_text("a")
    (base / "b.log").write_text("b")
    sub = base / "subdir"
    sub.mkdir()
    (sub / "c.txt").write_text("c")

    names = {item.name for item in base.iterdir()}
    assert names == {"a.txt", "b.log", "subdir"}

    matches = {item.name for item in base.glob("*.txt")}
    assert matches == {"a.txt"}

    recursive_matches = {item.name for item in base.glob("**/*.txt", recursive=True)}
    assert recursive_matches == {"a.txt", "c.txt"}

    renamed = (base / "a.txt").with_name("renamed.txt")
    assert renamed.name == "renamed.txt"

    with_suffix = renamed.with_suffix(".data")
    assert with_suffix.suffix == ".data"

    with_stem = with_suffix.with_stem("stem")
    assert with_stem.stem() == "stem"

    rel = with_stem.relative_to(base)
    assert rel.parts()[-1] == "stem.data"


def test_local_path_rename_unlink(tmp_path):
    original = LocalPath(tmp_path / "orig.txt")
    original.write_text("content")
    target = LocalPath(tmp_path / "moved.txt")

    moved = original.rename(target)
    assert moved.exists()
    assert not original.exists()

    moved.unlink()
    assert not moved.exists()


def test_path_manager_local_and_gcs():
    fake_client = object()
    manager = PathManager(gcs_client=fake_client)

    local = manager("/tmp/example.txt")
    assert isinstance(local, LocalPath)

    gcs = manager("gs://bucket/path.txt")
    assert isinstance(gcs, GCSPath)
    assert gcs.client is fake_client


def test_path_protocol_helpers():
    assert path_protocol("/tmp/example.txt") == "file"
    assert path_protocol("file:///tmp/example.txt") == "file"
    assert path_protocol("gs://bucket/path.txt") == "gs"
    assert path_protocol("s3://bucket/path.txt") == "s3"

    assert is_local_path("/tmp/example.txt") is True
    assert is_local_path(LocalPath("/tmp/example.txt")) is True
    assert is_remote_path("gs://bucket/path.txt") is True
    assert is_remote_path("s3://bucket/path.txt") is True
    assert is_remote_path(GCSPath("gs://bucket/path.txt", client=object())) is True


def test_mlutilpath_save_load_and_copy(tmp_path):
    manager = MLUtilPath()

    array = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    npy_path = tmp_path / "array.npy"
    manager.save_jax_array(array, npy_path, format="npy")
    loaded_npy = manager.load_jax_array(npy_path, format="npy")
    assert np.array_equal(np.array(loaded_npy), np.array(array))

    pkl_path = tmp_path / "array.pkl"
    manager.save_jax_array(array, pkl_path, format="pickle")
    loaded_pkl = manager.load_jax_array(pkl_path, format="pickle")
    assert np.array_equal(np.array(loaded_pkl), np.array(array))

    data = {"values": array, "count": 2}
    json_path = tmp_path / "data.json"
    manager.save_dict(data, json_path, format="json")
    loaded_json = manager.load_dict(json_path, format="json")
    assert loaded_json["values"] == array.tolist()
    assert loaded_json["count"] == 2

    pickle_path = tmp_path / "data.pkl"
    manager.save_dict(data, pickle_path, format="pickle")
    loaded_pickle = manager.load_dict(pickle_path, format="pickle")
    assert np.array_equal(np.array(loaded_pickle["values"]), np.array(array))

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "file.txt").write_text("payload", encoding="utf-8")
    manager.copy_tree(str(src), str(dst))
    assert (dst / "sub" / "file.txt").read_text(encoding="utf-8") == "payload"
