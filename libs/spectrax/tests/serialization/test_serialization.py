# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for tree_serialize_leaves / tree_deserialize_leaves primitives."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import NamedSharding, PartitionSpec

from spectrax import State
from spectrax.serialization.serialization import (
    _fs_paths_from_key_paths,
    _fully_replicated_sharding,
    _sharding_from_leaf,
    is_array_like,
    join_key,
    leaf_key_paths,
    tree_deserialize_leaves,
    tree_serialize_leaves,
)


class TestTreeSerializeLeaves:
    """Tests for tree_serialize_leaves."""

    def test_writes_tensorstore_index(self, tmp_checkpoint_dir, mesh):
        """Serialization produces tensorstore_index.json."""
        tree = {"w": jnp.ones((4, 4)), "b": jnp.zeros(4)}
        tree_serialize_leaves(tmp_checkpoint_dir, tree, prefix="model")

        index_path = Path(tmp_checkpoint_dir) / "tensorstore_index.json"
        assert index_path.exists()
        data = json.loads(index_path.read_text())
        assert data["format"] == "tensorstore"
        assert "model" in data.get("prefixes", {})
        assert len(data["prefixes"]["model"]) == 2

    def test_writes_arrays_without_prefix(self, tmp_checkpoint_dir, mesh):
        """Serialization without prefix stores arrays at top level."""
        tree = {"x": jnp.ones((2, 2))}
        tree_serialize_leaves(tmp_checkpoint_dir, tree)

        index_path = Path(tmp_checkpoint_dir) / "tensorstore_index.json"
        data = json.loads(index_path.read_text())
        assert "arrays" in data
        assert len(data["arrays"]) == 1

    def test_non_array_leaves_ignored(self, tmp_checkpoint_dir, mesh):
        """Non-array leaves are not written to TensorStore."""
        tree = {"arr": jnp.ones(2), "scalar": 42, "text": "hello"}
        tree_serialize_leaves(tmp_checkpoint_dir, tree, prefix="model")

        index_path = Path(tmp_checkpoint_dir) / "tensorstore_index.json"
        data = json.loads(index_path.read_text())
        assert len(data["prefixes"]["model"]) == 1


class TestTreeDeserializeLeaves:
    """Tests for tree_deserialize_leaves."""

    def test_basic_deserialize(self, tmp_checkpoint_dir, mesh):
        """Roundtrip through serialize/deserialize preserves arrays."""
        sh = NamedSharding(mesh, PartitionSpec("x", "y"))
        arr = jax.device_put(jnp.arange(8).reshape(2, 4), sh)
        tree = {"w": arr, "b": jnp.ones(4)}
        tree_serialize_leaves(tmp_checkpoint_dir, tree, prefix="model")

        result = tree_deserialize_leaves(
            tmp_checkpoint_dir,
            mesh,
            prefix="model",
            shardings={"w": sh, "b": NamedSharding(mesh, PartitionSpec())},
        )
        assert jnp.allclose(result["w"], tree["w"])
        assert jnp.allclose(result["b"], tree["b"])

    def test_deserialize_without_prefix(self, tmp_checkpoint_dir, mesh):
        """Deserialize arrays saved without a prefix."""
        tree = {"x": jnp.ones((2, 2)), "y": jnp.zeros(2)}
        tree_serialize_leaves(tmp_checkpoint_dir, tree)

        result = tree_deserialize_leaves(
            tmp_checkpoint_dir,
            mesh,
            shardings={
                "x": NamedSharding(mesh, PartitionSpec()),
                "y": NamedSharding(mesh, PartitionSpec()),
            },
        )
        assert jnp.allclose(result["x"], tree["x"])
        assert jnp.allclose(result["y"], tree["y"])

    def test_sharding_rules_matching(self, tmp_checkpoint_dir, mesh):
        """sharding_rules assigns shardings by regex match."""
        sh = NamedSharding(mesh, PartitionSpec("x", "y"))
        tree = {"weight": jnp.arange(8).reshape(2, 4), "bias": jnp.ones(4)}
        tree_serialize_leaves(tmp_checkpoint_dir, tree, prefix="model")

        result = tree_deserialize_leaves(
            tmp_checkpoint_dir,
            mesh,
            prefix="model",
            sharding_rules=[(".*weight.*", sh)],
        )
        assert jnp.allclose(result["weight"], tree["weight"])
        assert jnp.allclose(result["bias"], tree["bias"])

    def test_prefix_not_found_raises(self, tmp_checkpoint_dir, mesh):
        """Requesting a missing prefix raises ValueError listing available."""
        tree = {"a": jnp.ones(2)}
        tree_serialize_leaves(tmp_checkpoint_dir, tree, prefix="model")

        with pytest.raises(ValueError, match="Prefix 'optimizer' not found"):
            tree_deserialize_leaves(tmp_checkpoint_dir, mesh, prefix="optimizer")

    def test_chunked_deserialize(self, tmp_checkpoint_dir, mesh):
        """chunk_size splits deserialization into batches."""
        tree = {f"p{i}": jnp.ones((4, 4)) for i in range(5)}
        tree_serialize_leaves(tmp_checkpoint_dir, tree, prefix="model")

        sh = NamedSharding(mesh, PartitionSpec())
        result = tree_deserialize_leaves(
            tmp_checkpoint_dir,
            mesh,
            prefix="model",
            shardings={k: sh for k in tree},
            chunk_size=2,
        )
        assert set(result.keys()) == set(tree.keys())
        for k in tree:
            assert jnp.allclose(result[k], tree[k])

    def test_callback_applied(self, tmp_checkpoint_dir, mesh):
        """Per-array callback is applied during deserialization."""
        tree = {"a": jnp.ones((2, 2))}
        tree_serialize_leaves(tmp_checkpoint_dir, tree, prefix="model")

        def double(arr, key):
            """Double the input."""
            return arr * 2

        result = tree_deserialize_leaves(
            tmp_checkpoint_dir,
            mesh,
            prefix="model",
            shardings={"a": NamedSharding(mesh, PartitionSpec())},
            callback=double,
        )
        assert jnp.allclose(result["a"], tree["a"] * 2)


class TestSerializationHelpers:
    """Tests for individual helper functions in serialization.py."""

    def test_join_key_both_none(self):
        """Join key both none."""
        assert join_key(None, None) == ""

    def test_join_key_prefix_only(self):
        """Join key prefix only."""
        assert join_key("model", None) == "model"

    def test_join_key_key_only(self):
        """Join key key only."""
        assert join_key(None, "weight") == "weight"

    def test_join_key_both(self):
        """Join key both."""
        assert join_key("model", "weight") == "model.weight"

    def test_is_array_like_array(self):
        """Is array like array."""
        assert is_array_like(jnp.ones(2)) is True

    def test_is_array_like_scalar(self):
        """Is array like scalar."""
        assert is_array_like(42) is False

    def test_is_array_like_string(self):
        """Is array like string."""
        assert is_array_like("hello") is False

    def test_leaf_key_paths_simple_dict(self):
        """Leaf key paths simple dict."""
        tree = {"a": 1, "b": 2}
        result = leaf_key_paths(tree, prefix="model")
        assert result == {"a": "model.a", "b": "model.b"}

    def test_leaf_key_paths_nested(self):
        """Leaf key paths nested."""
        tree = {"layer": {"w": 1, "b": 2}}
        result = leaf_key_paths(tree, prefix="model")
        assert result == {"layer": {"w": "model.layer.w", "b": "model.layer.b"}}

    def test_leaf_key_paths_empty_prefix(self):
        """Leaf key paths empty prefix."""
        tree = {"a": 1}
        result = leaf_key_paths(tree, prefix="")
        assert result == {"a": "a"}

    def test_leaf_key_paths_state_custom_tuple_keys(self):
        """State key-path tuples are flattened into stable dotted keys."""
        state = State({"parameters": {"model": {"lm_head": {"weight": jnp.ones(())}}}})

        result = leaf_key_paths(state, prefix="tx")

        assert jax.tree_util.tree_leaves(result) == ["tx.parameters.model.lm_head.weight"]

    def test_fs_paths_from_key_paths(self):
        """Fs paths from key paths."""
        tree = {"a": "model.a", "b": {"c": "model.b.c"}}
        result = _fs_paths_from_key_paths("/ckpt", tree)
        assert result == {"a": "/ckpt/model/a", "b": {"c": "/ckpt/model/b/c"}}

    def test_fully_replicated_sharding_with_mesh(self, mesh):
        """Fully replicated sharding with mesh."""
        sh = _fully_replicated_sharding(mesh)
        assert isinstance(sh, NamedSharding)
        assert sh.spec == PartitionSpec()

    def test_fully_replicated_sharding_no_mesh(self):
        """Fully replicated sharding no mesh."""
        sh = _fully_replicated_sharding(None)
        from jax.sharding import SingleDeviceSharding

        assert isinstance(sh, SingleDeviceSharding)

    def test_sharding_from_leaf_with_sharding(self, mesh):
        """Sharding from leaf with sharding."""
        arr = jax.device_put(jnp.ones(2), NamedSharding(mesh, PartitionSpec()))
        sh = _sharding_from_leaf(arr, mesh)
        assert sh == arr.sharding

    def test_sharding_from_leaf_array(self, mesh):
        """Sharding from leaf array."""
        arr = jnp.ones(2)
        sh = _sharding_from_leaf(arr, mesh)
        from jax.sharding import SingleDeviceSharding

        assert isinstance(sh, (NamedSharding, SingleDeviceSharding))

    def test_sharding_from_leaf_scalar(self, mesh):
        """Sharding from leaf scalar."""
        sh = _sharding_from_leaf(3.14, mesh)
        assert isinstance(sh, NamedSharding)
        assert sh.spec == PartitionSpec()

    def test_sharding_from_leaf_unknown(self, mesh):
        """Sharding from leaf unknown."""
        sh = _sharding_from_leaf("hello", mesh)
        assert sh is None

    def test_tree_deserialize_leaves_no_mesh(self, tmp_checkpoint_dir):
        """Deserialize without providing a mesh uses SingleDeviceSharding."""
        from jax.sharding import SingleDeviceSharding

        tree = {"x": jnp.ones((2, 2))}
        tree_serialize_leaves(tmp_checkpoint_dir, tree)
        result = tree_deserialize_leaves(
            tmp_checkpoint_dir,
            mesh=None,
            shardings={"x": SingleDeviceSharding(jax.devices()[0])},
        )
        assert jnp.allclose(result["x"], tree["x"])
