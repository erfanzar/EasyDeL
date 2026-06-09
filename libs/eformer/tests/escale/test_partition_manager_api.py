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

"""Additional API and edge-case tests for partition manager helpers."""

import threading

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from eformer.common_types import (
    BATCH,
    MODE_DECODE,
    MODE_TRAIN,
    NOT_GIVEN,
    DynamicShardingAxes,
    HiddenStateSharding,
)
from eformer.escale.partition import (
    PartitionAxis,
    PartitionManager,
    apply_logical_sharding,
    get_current_partition_manager,
    get_partition_manager,
)
from eformer.escale.partition.auto_spec import auto_partition_spec
from eformer.escale.partition.manager import get_safe_hash_int


@pytest.fixture(autouse=True)
def _clear_custom_registry():
    PartitionAxis.clear_registered_axes()
    yield
    PartitionAxis.clear_registered_axes()


def test_get_safe_hash_int_is_deterministic():
    assert get_safe_hash_int("abc") == get_safe_hash_int("abc")
    assert get_safe_hash_int("abc", algorithm="sha256") == get_safe_hash_int("abc", algorithm="sha256")


def test_get_safe_hash_int_rejects_unknown_algorithm():
    with pytest.raises(ValueError, match="Unsupported hash algorithm"):
        get_safe_hash_int("abc", algorithm="not_a_real_hash")


def test_register_rejects_empty_semantic_axis():
    with pytest.raises(ValueError, match="non-empty string"):
        PartitionAxis.register("", "head_axis")


def test_unregister_rejects_empty_semantic_axis():
    with pytest.raises(ValueError, match="non-empty string"):
        PartitionAxis.unregister(" ")


def test_unregister_raises_key_error_when_missing_and_missing_ok_false():
    with pytest.raises(KeyError, match="is not registered"):
        PartitionAxis.unregister("__DOES_NOT_EXIST__", missing_ok=False)


def test_register_infers_generation_mapping_for_known_standard_axis():
    semantic = "__CUSTOM_QUERY_AXIS__"
    PartitionAxis.register(semantic, "query_sequence_axis")
    registered = PartitionAxis.get_registered_axes()
    assert registered[semantic]["axis_rule"] == "query_sequence_axis"
    assert registered[semantic]["generation_axis_rule"] == "decode_query_sequence_axis"


def test_register_uses_explicit_generation_mapping_when_provided():
    semantic = "__CUSTOM_AXIS_EXPLICIT_GEN__"
    PartitionAxis.register(
        semantic,
        "query_sequence_axis",
        generation_axis_rule="decode_head_axis",
    )
    registered = PartitionAxis.get_registered_axes()
    assert registered[semantic]["axis_rule"] == "query_sequence_axis"
    assert registered[semantic]["generation_axis_rule"] == "decode_head_axis"


def test_register_nonstandard_axis_has_not_given_generation_mapping():
    semantic = "__CUSTOM_MESH_LITERAL__"
    PartitionAxis.register(semantic, "tp")
    registered = PartitionAxis.get_registered_axes()
    assert registered[semantic]["generation_axis_rule"] is NOT_GIVEN


def test_clear_registered_axes_removes_custom_mappings():
    PartitionAxis.register("__A__", "head_axis")
    PartitionAxis.register("__B__", "query_sequence_axis")
    assert PartitionAxis.get_registered_axes()
    PartitionAxis.clear_registered_axes()
    assert PartitionAxis.get_registered_axes() == {}


def test_register_override_replaces_previous_custom_mapping():
    semantic = "__CUSTOM_OVERRIDE_AXIS__"
    PartitionAxis.register(semantic, "head_axis")
    PartitionAxis.register(semantic, "expert_axis", override=True)
    paxis = PartitionAxis()
    assert paxis.resolve_axis([semantic], mode=MODE_TRAIN) == [paxis.expert_axis]


def test_resolve_axis_raises_for_unknown_semantic_axis():
    with pytest.raises(ValueError, match="Unknown semantic axis name"):
        PartitionAxis().resolve_axis(["__UNKNOWN_SEMANTIC__"], mode=MODE_TRAIN)


def test_resolve_axis_treats_none_and_empty_symbol_as_unsharded():
    resolved = PartitionAxis().resolve_axis([None, "_"], mode=MODE_TRAIN)
    assert resolved == [None, None]


def test_resolve_spec_returns_partition_spec():
    spec = PartitionAxis().resolve_spec([BATCH], mode=MODE_TRAIN)
    assert spec == PartitionSpec(("fsdp", "dp"))


def test_custom_generation_rule_changes_output_in_generation_mode():
    semantic = "__CUSTOM_GEN_SWITCH__"
    PartitionAxis.register(
        semantic,
        "query_sequence_axis",
        generation_axis_rule="decode_query_sequence_axis",
    )
    paxis = PartitionAxis(decode_query_sequence_axis="dp")
    train_axis = paxis.resolve_axis([semantic], mode=MODE_TRAIN)[0]
    decode_axis = paxis.resolve_axis([semantic], mode=MODE_DECODE)[0]

    assert train_axis == "sp"
    assert decode_axis == "dp"


def test_true_cycle_in_custom_aliases_raises():
    axis_a = "__CUSTOM_CYCLE_A__"
    axis_b = "__CUSTOM_CYCLE_B__"
    PartitionAxis.register(axis_a, axis_b)
    PartitionAxis.register(axis_b, axis_a)
    with pytest.raises(ValueError, match="Cyclic semantic axis registration"):
        PartitionAxis().resolve_axis([axis_a], mode=MODE_TRAIN)


def test_partition_manager_rejects_invalid_partition_axis_type():
    with pytest.raises(TypeError, match="Expected PartitionAxis"):
        PartitionManager(paxis="not-a-partition-axis")


def test_nested_partition_manager_context_restores_outer_manager():
    outer = PartitionManager(PartitionAxis(data_parallel_axis="dp_outer"))
    inner = PartitionManager(PartitionAxis(data_parallel_axis="dp_inner"))

    with outer:
        assert get_current_partition_manager() is outer
        with inner:
            assert get_current_partition_manager() is inner
        assert get_current_partition_manager() is outer
    assert get_current_partition_manager() is None


def test_last_partition_manager_tracks_last_created_instance():
    first = PartitionManager(PartitionAxis(data_parallel_axis="dp1"))
    second = PartitionManager(PartitionAxis(data_parallel_axis="dp2"))
    assert get_partition_manager() is second
    assert get_partition_manager() is not first


def test_last_partition_manager_is_visible_from_other_threads():
    manager = PartitionManager(PartitionAxis(data_parallel_axis="dp"))
    seen = []

    def worker():
        seen.append(get_partition_manager())

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert seen == [manager]


def test_resolve_requires_shape_for_integer_mode_dynamic_axes():
    manager = PartitionManager(PartitionAxis())
    with pytest.raises(ValueError, match="shape should be provided"):
        manager.resolve(HiddenStateSharding)


def test_resolve_accepts_dynamic_axes_argument():
    manager = PartitionManager(PartitionAxis())
    spec = manager.resolve(dynamic_axes=DynamicShardingAxes([BATCH], MODE_TRAIN))
    assert spec == PartitionSpec(("fsdp", "dp"))


def test_apply_logical_sharding_with_none_manager_falls_back(monkeypatch):
    manager = PartitionManager(PartitionAxis())
    monkeypatch.setattr(manager, "shard", lambda *args, **kwargs: "fallback-called")
    output = apply_logical_sharding(
        x="arr",
        partition_manager=None,
        axes=[BATCH],
        mode=MODE_TRAIN,
    )
    assert output == "fallback-called"


def test_auto_partition_spec_deduplicates_repeated_names():
    mesh = Mesh(np.array(jax.devices()[:1]), ("dp",))
    spec = auto_partition_spec(np.ones((8, 8)), mesh=mesh, names=["dp", "dp"], min_sharding_size=1)

    assert spec == PartitionSpec("dp", None)
