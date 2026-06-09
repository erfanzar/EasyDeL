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

"""Tests for partition axis mappings."""

import pytest
from jax.sharding import PartitionSpec

import eformer.escale.partition.manager as partition_manager_module
from eformer.common_types import (
    BATCH,
    DATA_PARALLEL,
    KV_HEAD_DIM,
    MODE_TRAIN,
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


def test_kv_head_dim_maps_to_attention_kv_dim_axis():
    assert PartitionAxis._SEMANTIC_MAP[KV_HEAD_DIM] == "attention_kv_dim_axis"
    assert PartitionAxis._STANDARD_TO_GENERATION_ATTR_MAP["attention_kv_dim_axis"] == "decode_attention_kv_dim_axis"


def test_register_custom_semantic_axis_is_global():
    semantic_axis = "__TEST_CUSTOM_AXIS__"
    PartitionAxis.unregister(semantic_axis, missing_ok=True)
    try:
        PartitionAxis.register(semantic_axis, "expert_parallel_axis")
        paxis_a = PartitionAxis()
        paxis_b = PartitionAxis()
        assert paxis_a.resolve_axis([semantic_axis], mode=MODE_TRAIN)[0] == paxis_a.expert_parallel_axis
        assert paxis_b.resolve_axis([semantic_axis], mode=MODE_TRAIN)[0] == paxis_b.expert_parallel_axis
    finally:
        PartitionAxis.unregister(semantic_axis, missing_ok=True)


def test_register_requires_override_for_existing_semantics():
    with pytest.raises(ValueError):
        PartitionAxis.register(DATA_PARALLEL, "expert_parallel_axis")


def test_register_can_override_existing_semantic_axis_mapping():
    try:
        PartitionAxis.register(DATA_PARALLEL, "expert_parallel_axis", override=True)
        paxis = PartitionAxis()
        assert paxis.resolve_axis([DATA_PARALLEL], mode=MODE_TRAIN)[0] == paxis.expert_parallel_axis
    finally:
        PartitionAxis.unregister(DATA_PARALLEL, missing_ok=True)


def test_resolve_accepts_dynamic_sharding_axes_instance():
    manager = PartitionManager(PartitionAxis())
    spec = manager.resolve(DynamicShardingAxes([BATCH], MODE_TRAIN))
    assert spec == PartitionSpec(("fsdp", "dp"))


def test_resolve_accepts_dynamic_sharding_axes_class():
    manager = PartitionManager(PartitionAxis(decode_query_sequence_axis="dp"))
    decode_spec = manager.resolve(HiddenStateSharding, shape=(2, 1, 32))
    train_spec = manager.resolve(HiddenStateSharding, shape=(2, 8, 32))

    assert decode_spec == PartitionSpec(("fsdp", "dp"), "dp", "tp")
    assert train_spec == PartitionSpec(("fsdp", "dp"), "sp", "tp")


def test_nested_alias_with_repeated_semantics_does_not_false_cycle():
    semantic_a = "__TEST_NESTED_ALIAS_A__"
    semantic_b = "__TEST_NESTED_ALIAS_B__"
    for semantic in (semantic_a, semantic_b):
        PartitionAxis.unregister(semantic, missing_ok=True)

    try:
        PartitionAxis.register(semantic_b, [DATA_PARALLEL, DATA_PARALLEL])
        PartitionAxis.register(semantic_a, semantic_b)
        resolved = PartitionAxis().resolve_axis([semantic_a], mode=MODE_TRAIN)
        assert resolved == [["dp", "dp"]]
    finally:
        for semantic in (semantic_a, semantic_b):
            PartitionAxis.unregister(semantic, missing_ok=True)


def test_partition_manager_context_manager_updates_lookup_helpers():
    manager = PartitionManager(PartitionAxis())
    assert get_partition_manager() is manager
    assert get_current_partition_manager() is None

    with manager as active:
        assert active is manager
        assert get_current_partition_manager() is manager

    assert get_current_partition_manager() is None


def test_apply_logical_sharding_uses_context_when_manager_not_provided(monkeypatch):
    manager = PartitionManager(PartitionAxis())
    calls = {}

    def _fake_shard(self, x, axes, mode, dynamic_axes, auto_correct):
        calls["x"] = x
        calls["axes"] = axes
        calls["mode"] = mode
        calls["dynamic_axes"] = dynamic_axes
        calls["auto_correct"] = auto_correct
        return "ok"

    monkeypatch.setattr(manager, "shard", _fake_shard.__get__(manager, PartitionManager))

    with manager:
        output = apply_logical_sharding(x="arr", axes=[BATCH], mode=MODE_TRAIN)

    assert output == "ok"
    assert calls["x"] == "arr"
    assert calls["axes"] == [BATCH]
    assert calls["mode"] == MODE_TRAIN
    assert calls["auto_correct"] is True


def test_apply_logical_sharding_uses_last_created_manager(monkeypatch):
    manager = PartitionManager(PartitionAxis())

    def _fake_shard(self, x, axes, mode, dynamic_axes, auto_correct):
        return ("last", x, axes, mode, dynamic_axes, auto_correct)

    monkeypatch.setattr(manager, "shard", _fake_shard.__get__(manager, PartitionManager))
    output = apply_logical_sharding(x="arr", axes=[BATCH], mode=MODE_TRAIN, auto_correct=False)
    assert output == ("last", "arr", [BATCH], MODE_TRAIN, partition_manager_module.NOT_GIVEN, False)


def test_apply_logical_sharding_prefers_explicit_manager_over_context(monkeypatch):
    context_manager = PartitionManager(PartitionAxis())
    explicit_manager = PartitionManager(PartitionAxis())

    monkeypatch.setattr(context_manager, "shard", lambda *args, **kwargs: "context-manager")
    monkeypatch.setattr(explicit_manager, "shard", lambda *args, **kwargs: "explicit-manager")

    with context_manager:
        output = apply_logical_sharding(
            x="arr",
            partition_manager=explicit_manager,
            axes=[BATCH],
            mode=MODE_TRAIN,
        )

    assert output == "explicit-manager"


def test_apply_logical_sharding_raises_when_no_manager_available(monkeypatch):
    monkeypatch.setattr(partition_manager_module, "get_current_partition_manager", lambda: None)
    monkeypatch.setattr(partition_manager_module, "get_partition_manager", lambda: None)
    with pytest.raises(ValueError, match="No PartitionManager is available"):
        partition_manager_module.apply_logical_sharding(x="arr", axes=[BATCH], mode=MODE_TRAIN)


def test_hash_changes_when_partition_axis_fields_change():
    paxis_a = PartitionAxis(data_parallel_axis="dp")
    paxis_b = PartitionAxis(data_parallel_axis="dp_alt")
    assert hash(paxis_a) != hash(paxis_b)


def test_hash_changes_when_partition_manager_fields_change():
    manager_a = PartitionManager(PartitionAxis(data_parallel_axis="dp"))
    manager_b = PartitionManager(PartitionAxis(data_parallel_axis="dp_alt"))
    assert hash(manager_a) != hash(manager_b)
