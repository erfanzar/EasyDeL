# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.core.state`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from spectrax.core.state import State


def test_empty_state_len_and_collections():
    """A default :class:`State` is empty."""
    s = State()
    assert len(s) == 0
    assert s.collections() == set()


def test_construct_from_nested_mapping():
    """Construct from a nested ``{coll: {path: val}}`` mapping."""
    s = State({"parameters": {"w": jnp.zeros(3), "b": jnp.ones(1)}})
    assert "parameters" in s
    assert len(s) == 2


def test_getitem_auto_creates_inner_dict():
    """Indexing a missing collection returns a writable inner dict."""
    s = State()
    d = s["parameters"]
    d["w"] = jnp.zeros(2)
    assert "w" in s["parameters"]


def test_setitem_replaces_collection():
    """``state[collection] = mapping`` overwrites the inner dict."""
    s = State({"parameters": {"w": jnp.zeros(3)}})
    s["parameters"] = {"b": jnp.ones(1)}
    assert "w" not in s["parameters"]
    assert "b" in s["parameters"]


def test_contains_only_for_nonempty_string():
    """``collection in state`` is ``True`` only for non-empty string keys."""
    s = State({"parameters": {"w": jnp.zeros(1)}})
    assert "parameters" in s
    assert "empty" not in s
    assert 42 not in s


def test_iter_and_keys_order():
    """Iteration yields collection names."""
    s = State({"a": {"x": jnp.zeros(1)}, "b": {"y": jnp.zeros(1)}})
    assert set(iter(s)) == {"a", "b"}


def test_len_counts_total_leaves():
    """``len`` sums leaves across all collections."""
    s = State({"a": {"x": 0, "y": 0}, "b": {"z": 0}})
    assert len(s) == 3


def test_items_yields_triples():
    """``items`` yields ``(collection, path, leaf)`` triples."""
    s = State({"a": {"x": 1}, "b": {"y": 2}})
    triples = set(s.items())
    assert ("a", "x", 1) in triples
    assert ("b", "y", 2) in triples


def test_paths_unfiltered():
    """``paths()`` returns every ``(collection, path)`` pair."""
    s = State({"a": {"x": 1}, "b": {"y": 2}})
    assert set(s.paths()) == {("a", "x"), ("b", "y")}


def test_paths_filtered():
    """``paths(collection=...)`` filters to one collection."""
    s = State({"a": {"x": 1}, "b": {"y": 2}})
    assert set(s.paths("a")) == {("a", "x")}


def test_filter_keeps_only_named():
    """``filter`` mutates in place by default."""
    s = State({"a": {"x": 1}, "b": {"y": 2}})
    out = s.filter("a")
    assert out is s
    assert "a" in s
    assert "b" not in s


def test_exclude_removes_named():
    """``exclude`` mutates in place by default."""
    s = State({"a": {"x": 1}, "b": {"y": 2}})
    out = s.exclude("a")
    assert out is s
    assert "a" not in s
    assert "b" in s


def test_filter_copy_returns_detached_state():
    """``filter(copy=True)`` preserves the original state."""
    s = State({"a": {"x": 1}, "b": {"y": 2}})
    out = s.filter("a", copy=True)
    assert out is not s
    assert set(s.paths()) == {("a", "x"), ("b", "y")}
    assert set(out.paths()) == {("a", "x")}


def test_merge_other_wins_on_collision():
    """``merge`` returns a fresh state and prefers ``other`` on collision."""
    a = State({"parameters": {"w": 1}})
    b = State({"parameters": {"w": 2, "b": 3}})
    out = a.merge(b)
    assert out is not a
    assert a["parameters"]["w"] == 1
    assert out["parameters"]["w"] == 2
    assert out["parameters"]["b"] == 3


def test_merge_copy_returns_fresh_instance():
    """``merge(copy=True)`` produces a detached :class:`State`."""
    a = State({"x": {"p": 1}})
    b = State({"y": {"q": 2}})
    c = a.merge(b, copy=True)
    c["x"]["p"] = 99
    assert a["x"]["p"] == 1


def test_overlay_returns_fresh_structure_without_mutating_inputs():
    """``overlay`` builds a temporary merged state while leaving inputs alone."""
    leaf = object()
    a = State({"x": {"p": leaf, "same": 1}})
    b = State({"x": {"same": 2}, "y": {"q": 3}})
    out = a.overlay(b)
    assert out is not a
    assert out is not b
    assert a["x"]["same"] == 1
    assert b["x"]["same"] == 2
    assert out["x"]["same"] == 2
    assert out["x"]["p"] is leaf
    out["x"]["p"] = "changed"
    assert a["x"]["p"] is leaf


def test_map_all_collections():
    """``map`` without filter mutates every leaf in place by default."""
    s = State({"a": {"x": 1, "y": 2}, "b": {"z": 3}})
    out = s.map(lambda v: v * 10)
    assert out is s
    assert out["a"]["x"] == 10
    assert out["b"]["z"] == 30


def test_map_path_aware_callback_receives_dotted_path():
    """``map`` passes dotted paths to two-argument callbacks."""
    s = State({"parameters": {"layer": {"weight": 1, "bias": 2}}})
    seen: list[str] = []

    def annotate(path, value):
        """Annotation helper."""
        seen.append(path)
        return f"{path}={value}"

    out = s.map(annotate, "parameters")
    assert out["parameters"]["layer"]["weight"] == "layer.weight=1"
    assert out["parameters"]["layer"]["bias"] == "layer.bias=2"
    assert set(seen) == {"layer.bias", "layer.weight"}


def test_map_optional_path_callback_receives_dotted_path():
    """Path-aware callbacks may keep ``value`` optional for ergonomic partials."""
    s = State({"parameters": {"layer": {"weight": 1}}})
    seen: list[tuple[str, int | None]] = []

    def annotate(path, value=None):
        """Annotation helper."""
        seen.append((path, value))
        return f"{path}={value}"

    out = s.map(annotate, "parameters")
    assert out["parameters"]["layer"]["weight"] == "layer.weight=1"
    assert seen == [("layer.weight", 1)]


def test_map_collection_path_callback_receives_collection_and_path():
    """``map`` passes ``(path, value, collection)`` to three-arg callbacks."""
    s = State({"parameters": {"w": 1}, "buffers": {"w": 2}})
    out = s.map(lambda path, value, collection: f"{collection}/{path}={value}")
    assert out["parameters"]["w"] == "parameters/w=1"
    assert out["buffers"]["w"] == "buffers/w=2"


def test_map_restricted_to_collections():
    """``map(fn, *collections)`` only touches the named collections."""
    s = State({"a": {"x": 1}, "b": {"y": 2}})
    out = s.map(lambda v: v + 100, "a")
    assert out is s
    assert out["a"]["x"] == 101
    assert out["b"]["y"] == 2


def test_map_copy_returns_new_state():
    """``map(copy=True)`` returns a detached state."""
    s = State({"a": {"x": 1}})
    out = s.map(lambda v: v + 1, copy=True)
    assert out is not s
    assert s["a"]["x"] == 1
    assert out["a"]["x"] == 2


def test_set_mutates_state_by_default():
    """``set`` mutates the state in place by default."""
    s = State()
    out = s.set("parameters", "w", jnp.zeros(3))
    assert out is s
    assert "parameters" in s


def test_set_copy_returns_new_state():
    """``set(copy=True)`` preserves the original state."""
    s = State()
    new = s.set("parameters", "w", jnp.zeros(3), copy=True)
    assert "parameters" in new
    assert "parameters" not in s


def test_set_with_tuple_path_is_encoded():
    """Tuple paths become nested dict entries."""
    s = State().set("parameters", ("layers", 0, "w"), jnp.zeros(1))
    assert s["parameters"]["layers"][0]["w"].shape == (1,)


def test_get_with_default():
    """``get`` returns ``default`` on missing keys."""
    s = State({"a": {"x": 42}})
    assert s.get("a", "x") == 42
    assert s.get("a", "missing", default="fallback") == "fallback"
    assert s.get("missing", "path", default=None) is None


def test_get_with_tuple_path():
    """``get`` accepts a tuple path."""
    s = State().set("a", ("x", 1), 42)
    assert s.get("a", ("x", 1)) == 42


def test_repr_is_summary():
    """``repr`` includes leaf counts."""
    s = State({"a": {"x": 1, "y": 2}, "b": {"z": 3}})
    r = repr(s)
    assert "State(" in r
    assert "3 leaves" in r


def test_pytree_flatten_unflatten_roundtrip():
    """:class:`State` is registered as a JAX pytree with stable leaf order."""
    s = State({"b": {"q": jnp.asarray(2.0)}, "a": {"p": jnp.asarray(1.0)}})
    leaves, treedef = jax.tree_util.tree_flatten(s)
    assert leaves == [jnp.asarray(1.0), jnp.asarray(2.0)]
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt["a"]["p"] == 1.0
    assert rebuilt["b"]["q"] == 2.0


def test_tree_map_with_path_emits_flat_key_paths():
    """``tree_map_with_path`` reports collection/path entries as a flat tuple."""
    s = State({"parameters": {"layer": {7: {"weight": jnp.ones(())}}}})
    paths = []

    mapped = jax.tree_util.tree_map_with_path(lambda path, leaf: paths.append(path) or leaf, s)
    _path_leaves, path_treedef = jax.tree_util.tree_flatten_with_path(s)
    _plain_leaves, plain_treedef = jax.tree_util.tree_flatten(s)

    assert isinstance(mapped, State)
    assert path_treedef == plain_treedef
    assert paths == [
        (
            jax.tree_util.DictKey("parameters"),
            jax.tree_util.DictKey("layer"),
            jax.tree_util.DictKey(7),
            jax.tree_util.DictKey("weight"),
        )
    ]


def test_tree_map_with_path_skips_empty_collections_like_tree_flatten():
    """Empty collections should not create phantom keyed children."""
    s = State({"parameters": {"weight": jnp.ones(())}, "empty": {}})
    paths = []

    mapped = jax.tree_util.tree_map_with_path(lambda path, leaf: paths.append(path) or leaf, s)

    assert isinstance(mapped, State)
    assert paths == [(jax.tree_util.DictKey("parameters"), jax.tree_util.DictKey("weight"))]
    assert "empty" not in mapped


def test_pytree_treedef_preserves_keys():
    """``tree_unflatten`` restores the original ``(collection, path)`` shape."""
    original = State({"parameters": {"w": jnp.zeros(2)}, "buffers": {"mu": jnp.ones(2)}})
    leaves, treedef = jax.tree_util.tree_flatten(original)
    scaled = [leaf * 2 for leaf in leaves]
    rebuilt = jax.tree_util.tree_unflatten(treedef, scaled)
    assert set(rebuilt.paths()) == set(original.paths())


def test_pytree_unflatten_rejects_leaf_count_mismatch():
    """State unflattening should fail loudly instead of truncating leaves."""
    original = State({"a": {"x": jnp.asarray(1.0)}, "b": {"y": jnp.asarray(2.0)}})
    leaves, treedef = jax.tree_util.tree_flatten(original)

    with pytest.raises(ValueError):
        jax.tree_util.tree_unflatten(treedef, leaves[:1])


def test_raw_returns_nested_dict():
    """``raw`` exposes the backing nested dict."""
    s = State({"a": {"x": 1}})
    assert s.raw() == {"a": {"x": 1}}


def test_collections_skips_empty():
    """``collections()`` only reports non-empty collections."""
    s = State({"parameters": {"w": 1}, "empty": {}})
    assert s.collections() == {"parameters"}
