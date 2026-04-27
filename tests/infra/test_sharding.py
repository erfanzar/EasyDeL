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

"""Tests for ``easydel.infra.sharding`` helpers and lightweight types.

The module is mostly thin adapters around spectrax types. We focus on the
pure-Python helpers that don't require a real device mesh, plus the
``TensorLayout``/``AxisPolicy`` round-trip semantics:

* ``_normalize_axis_entry`` and ``_normalize_axes`` -- spec normalization
* ``_normalize_logical_axis_rules`` -- rule mapping coercion
* ``mesh_partition_product`` -- multiplicity from a partition spec
* ``sanitize_partition_spec_for_shape`` -- drops non-divisible axes
* ``TensorLayout.from_any`` / ``to_dict`` / ``as_spectrax_sharding``
* ``sharding_for_layout`` -- only train-time layouts produce a sharding
* ``metadata_for_layout`` -- builds variable metadata with stage hint
* ``AxisPolicy.from_partition_axis`` / ``from_dict`` / ``to_dict`` round-trip
* ``coerce_axis_policy`` -- normalizes dict / PartitionAxis input
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import spectrax as spx
from spectrax import PartitionAxis
from spectrax.common_types import EMPTY, MODE_DECODE, MODE_TRAIN

from easydel.infra.sharding import (
    AxisPolicy,
    TensorLayout,
    _normalize_axes,
    _normalize_axis_entry,
    _normalize_logical_axis_rules,
    coerce_axis_policy,
    mesh_partition_product,
    metadata_for_layout,
    sanitize_partition_spec_for_shape,
    sharding_for_layout,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (EMPTY, None),
        ("_", None),
        ("", None),
        ("  ", None),
        ("dp", "dp"),
        ("  dp  ", "dp"),
        ((), None),
        (("dp",), ("dp",)),
        (("dp", "fsdp"), ("dp", "fsdp")),
        (("dp", None, "fsdp"), ("dp", "fsdp")),
        (("dp", EMPTY, "fsdp"), ("dp", "fsdp")),
        ([("dp", "fsdp"), "tp"], ("dp", "fsdp", "tp")),
    ],
)
def test_normalize_axis_entry(value, expected):
    assert _normalize_axis_entry(value) == expected


def test_normalize_axis_entry_rejects_invalid_type():
    with pytest.raises(TypeError, match="Unsupported axis entry"):
        _normalize_axis_entry(42)


def test_normalize_axes_applies_per_element():
    result = _normalize_axes(("dp", None, ("fsdp", "sp"), "_"))
    assert result == ("dp", None, ("fsdp", "sp"), None)


def test_normalize_logical_axis_rules_none_returns_empty_tuple():
    assert _normalize_logical_axis_rules(None) == ()


def test_normalize_logical_axis_rules_dict():
    result = _normalize_logical_axis_rules({"batch": "dp", "seq": None, "hidden": "tp"})

    assert dict(result) == {"batch": "dp", "seq": None, "hidden": "tp"}


def test_normalize_logical_axis_rules_iterable_of_pairs():
    result = _normalize_logical_axis_rules([("batch", "dp"), ("hidden", None)])
    assert result == (("batch", "dp"), ("hidden", None))


def test_normalize_logical_axis_rules_coerces_to_strings():
    """Non-str names get str()'d; None mesh axis stays None."""
    result = _normalize_logical_axis_rules({1: 2, "x": None})
    assert result == (("1", "2"), ("x", None))


def _stub_mesh(shape: dict[str, int]) -> SimpleNamespace:
    """Lightweight mesh stub exposing only ``mesh.shape[axis]``."""
    return SimpleNamespace(shape=shape)


def test_mesh_partition_product_none_returns_one():
    mesh = _stub_mesh({"dp": 4})
    assert mesh_partition_product(mesh, None) == 1


def test_mesh_partition_product_single_axis():
    mesh = _stub_mesh({"dp": 4, "tp": 2})
    assert mesh_partition_product(mesh, "dp") == 4
    assert mesh_partition_product(mesh, "tp") == 2


def test_mesh_partition_product_compound_axis_multiplies():
    """Tuple axis sums via product across all named axes."""
    mesh = _stub_mesh({"dp": 4, "fsdp": 2, "sp": 3})
    assert mesh_partition_product(mesh, ("dp", "fsdp", "sp")) == 24


def test_mesh_partition_product_compound_axis_skips_none():
    mesh = _stub_mesh({"dp": 4, "fsdp": 2})
    assert mesh_partition_product(mesh, ("dp", None, "fsdp")) == 8


def test_mesh_partition_product_unknown_axis_returns_one():
    """Missing axis falls back to 1 (per ``_mesh_axis_size``'s exception handler)."""
    mesh = _stub_mesh({"dp": 4})
    assert mesh_partition_product(mesh, "nonexistent") == 1


def test_sanitize_drops_axis_when_dim_not_divisible():
    """Spec axis ``"dp"`` (size 4) on a shape ``(7,)`` -> dropped (7 % 4 != 0)."""
    from jax.sharding import PartitionSpec

    mesh = _stub_mesh({"dp": 4})
    out = sanitize_partition_spec_for_shape(PartitionSpec("dp"), shape=(7,), mesh=mesh)
    assert out == PartitionSpec(None)


def test_sanitize_keeps_axis_when_divisible():
    from jax.sharding import PartitionSpec

    mesh = _stub_mesh({"dp": 4})
    out = sanitize_partition_spec_for_shape(PartitionSpec("dp"), shape=(8,), mesh=mesh)
    assert out == PartitionSpec("dp")


def test_sanitize_truncates_extra_axes_beyond_shape_rank():
    """A spec longer than the shape's rank gets truncated."""
    from jax.sharding import PartitionSpec

    mesh = _stub_mesh({"dp": 4, "tp": 2})
    out = sanitize_partition_spec_for_shape(
        PartitionSpec("dp", "tp", "tp"),
        shape=(8, 4),
        mesh=mesh,
    )
    assert len(tuple(out)) == 2


def test_sanitize_passes_through_when_no_change():
    from jax.sharding import PartitionSpec

    mesh = _stub_mesh({"dp": 4})
    spec = PartitionSpec(None)
    out = sanitize_partition_spec_for_shape(spec, shape=(7,), mesh=mesh)
    assert out is spec


def test_tensor_layout_normalizes_axes_in_post_init():
    layout = TensorLayout(axes=("dp", None, ("fsdp", "sp")))

    assert layout.axes == ("dp", None, ("fsdp", "sp"))


def test_tensor_layout_default_mode_is_train():
    layout = TensorLayout(axes=("dp",))
    assert layout.mode == MODE_TRAIN


def test_tensor_layout_from_any_passes_through_existing_layout():
    layout = TensorLayout(axes=("dp",))
    assert TensorLayout.from_any(layout) is layout


def test_tensor_layout_from_any_none_returns_none():
    assert TensorLayout.from_any(None) is None


def test_tensor_layout_from_any_dict():
    layout = TensorLayout.from_any({"axes": ("dp", "tp"), "mode": MODE_TRAIN})
    assert isinstance(layout, TensorLayout)
    assert layout.axes == ("dp", "tp")


def test_tensor_layout_from_any_dict_default_mode():
    """Missing 'mode' defaults to MODE_TRAIN."""
    layout = TensorLayout.from_any({"axes": ("dp",)})
    assert layout.mode == MODE_TRAIN


def test_tensor_layout_from_any_list():
    layout = TensorLayout.from_any(["dp", "tp"])
    assert layout.axes == ("dp", "tp")


def test_tensor_layout_from_any_object_with_axes_and_mode():
    """Anything with .axes and .mode duck-types as a TensorLayout."""
    duck = SimpleNamespace(axes=("dp",), mode=MODE_TRAIN)
    layout = TensorLayout.from_any(duck)
    assert layout is not None
    assert layout.axes == ("dp",)


def test_tensor_layout_to_dict_round_trip():
    layout = TensorLayout(axes=("dp", "tp"), mode=MODE_TRAIN)
    d = layout.to_dict()
    assert d == {"axes": ("dp", "tp"), "mode": MODE_TRAIN}
    restored = TensorLayout.from_any(d)
    assert restored == layout


def test_tensor_layout_as_spectrax_sharding():
    layout = TensorLayout(axes=("dp", "tp"))
    sharding = layout.as_spectrax_sharding()
    assert isinstance(sharding, spx.Sharding)
    assert sharding.axis_names == ("dp", "tp")


def test_sharding_for_layout_returns_none_for_none_input():
    assert sharding_for_layout(None) is None


def test_sharding_for_layout_returns_sharding_for_train_mode():
    layout = TensorLayout(axes=("dp",), mode=MODE_TRAIN)
    sharding = sharding_for_layout(layout)
    assert isinstance(sharding, spx.Sharding)
    assert sharding.axis_names == ("dp",)


def test_sharding_for_layout_skips_non_train_mode():
    """Decode-mode layouts intentionally produce no static sharding."""
    layout = TensorLayout(axes=("dp",), mode=MODE_DECODE)
    assert sharding_for_layout(layout) is None


def test_metadata_for_layout_train_mode_includes_sharding():
    layout = TensorLayout(axes=("dp",), mode=MODE_TRAIN)
    metadata = metadata_for_layout(layout)
    assert "sharding" in metadata
    assert isinstance(metadata["sharding"], spx.Sharding)


def test_metadata_for_layout_non_train_mode_uses_tensor_layout_key():
    layout = TensorLayout(axes=("dp",), mode=MODE_DECODE)
    metadata = metadata_for_layout(layout)
    assert "tensor_layout" in metadata
    assert metadata["tensor_layout"] is layout


def test_metadata_for_layout_explicit_pipeline_stage_overrides_active_context():
    """An explicit ``pipeline_stage`` arg lands in metadata."""
    metadata = metadata_for_layout(
        layout=TensorLayout(axes=("dp",)),
        pipeline_stage=(3, 8),
    )
    from spectrax.core.stage_assignment import PIPELINE_STAGE_METADATA_KEY

    assert metadata[PIPELINE_STAGE_METADATA_KEY] == (3, 8)


def test_metadata_for_layout_no_layout_returns_just_stage():
    """Passing layout=None still returns metadata containing only the stage hint."""
    metadata = metadata_for_layout(None, pipeline_stage=(0, 4))
    from spectrax.core.stage_assignment import PIPELINE_STAGE_METADATA_KEY

    assert PIPELINE_STAGE_METADATA_KEY in metadata
    assert "sharding" not in metadata
    assert "tensor_layout" not in metadata


def test_metadata_for_layout_uses_active_assign_stage_when_no_explicit_arg():
    """Inside an ``assign_stage`` context, ``current_stage_assignment`` populates the hint."""
    from spectrax import assign_stage
    from spectrax.core.stage_assignment import PIPELINE_STAGE_METADATA_KEY

    with assign_stage(total=4, current=2):
        metadata = metadata_for_layout(None)

    assert metadata[PIPELINE_STAGE_METADATA_KEY] == (2, 4)


def test_axis_policy_default_constructible():
    policy = AxisPolicy()
    assert isinstance(policy.partition_axis, PartitionAxis)


def test_axis_policy_from_partition_axis_round_trip():
    pa = PartitionAxis()
    policy = AxisPolicy.from_partition_axis(pa)
    assert isinstance(policy, AxisPolicy)

    out = policy.to_partition_axis()
    assert isinstance(out, PartitionAxis)


def test_axis_policy_from_dict_uses_default_when_empty():
    policy = AxisPolicy.from_dict({})
    assert isinstance(policy.partition_axis, PartitionAxis)


def test_axis_policy_from_any_passes_through_axis_policy():
    p1 = AxisPolicy()
    p2 = AxisPolicy.from_any(p1)
    assert p2 is p1


def test_axis_policy_from_any_none_returns_default_axis_policy():
    policy = AxisPolicy.from_any(None)
    assert isinstance(policy, AxisPolicy)


def test_axis_policy_to_dict_returns_field_dict():
    policy = AxisPolicy()
    d = policy.to_dict()
    assert isinstance(d, dict)

    import dataclasses

    expected_fields = {f.name for f in dataclasses.fields(policy.partition_axis)}
    assert set(d.keys()) == expected_fields


def test_axis_policy_getattr_forwards_to_partition_axis():
    """``policy.<attr>`` falls through to the underlying ``PartitionAxis``."""
    policy = AxisPolicy()

    assert callable(policy.resolve_axis)


def test_coerce_axis_policy_from_existing_policy_returns_same():
    policy = AxisPolicy()
    assert coerce_axis_policy(policy) is policy


def test_coerce_axis_policy_from_partition_axis_wraps():
    pa = PartitionAxis()
    policy = coerce_axis_policy(pa)
    assert isinstance(policy, AxisPolicy)


def test_coerce_axis_policy_from_none_returns_default():
    policy = coerce_axis_policy(None)
    assert isinstance(policy, AxisPolicy)


def test_axis_policy_logical_axis_rule_pairs_includes_canonical_axes():
    """The rule pair set always contains identity rules for canonical mesh axes."""
    policy = AxisPolicy()
    rules = dict(policy.logical_axis_rule_pairs())
    for canonical in ("dp", "fsdp", "ep", "tp", "sp"):
        assert canonical in rules
