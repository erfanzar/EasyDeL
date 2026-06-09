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

"""Tests for serialization helpers."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from eformer.paths import LocalPath
from eformer.serialization import serialization as ser
from eformer.serialization import utils as ser_utils


def test_join_key_and_leaf_key_paths():
    assert ser.join_key(None, None) == ""
    assert ser.join_key("a", None) == "a"
    assert ser.join_key(None, "b") == "b"
    assert ser.join_key("a", "b") == "a.b"

    tree = {"a": jnp.array(1), "b": [jnp.array(2), {"c": jnp.array(3)}]}
    paths = ser.leaf_key_paths(tree, prefix="root")
    assert paths["a"] == "root.a"
    assert paths["b"][0] == "root.b.0"
    assert paths["b"][1]["c"] == "root.b.1.c"


def test_is_array_like():
    assert ser.is_array_like(jnp.ones((2,)))
    assert ser.is_array_like(SimpleNamespace(shape=(2,), dtype="f"))
    assert not ser.is_array_like(123)


def test_path_helpers():
    assert ser_utils.derive_base_prefix_from_path("/x/model.safetensors") == "/x/model"
    assert ser_utils.derive_base_prefix_from_path("/x/model.safetensors.index.json") == "/x/model"
    assert ser_utils.derive_base_prefix_from_path("/x/model-00001-of-00004.safetensors") == "/x/model"

    assert ser_utils.shard_filename("/x/model", 1, 4) == "/x/model-00001-of-00004.safetensors"
    assert ser_utils.index_filename("/x/model") == "/x/model.safetensors.index.json"


def test_gcs_path_helpers():
    bucket, blob = ser_utils.parse_gcs_path("gs://bucket/path/to/file")
    assert bucket == "bucket"
    assert blob == "path/to/file"

    bucket, blob = ser_utils.parse_gcs_path("gs://bucket")
    assert bucket == "bucket"
    assert blob == ""

    with pytest.raises(ValueError):
        ser_utils.parse_gcs_path("/not/gcs")

    assert ser_utils.is_gcs_path("gs://bucket/file") is True
    assert ser_utils.is_gcs_path(LocalPath("/tmp/file")) is False


def test_group_keys_by_shard_size():
    arrays = {
        "a": jnp.ones((2,), dtype=jnp.float32),  # 8 bytes
        "b": jnp.ones((2,), dtype=jnp.float32),  # 8 bytes
        "c": jnp.ones((2,), dtype=jnp.float32),  # 8 bytes
    }
    shards = ser_utils.group_keys_by_shard_size(arrays, max_shard_size_bytes=16)
    assert shards == [["a", "b"], ["c"]]
