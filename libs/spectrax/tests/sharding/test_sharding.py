# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :mod:`spectrax.sharding`."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import spectrax as spx
from spectrax import common_types as ct
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs
from spectrax.sharding import (
    PartitionAxis,
    PartitionManager,
    create_mesh,
    current_axis_rules,
    get_named_sharding,
    get_partition_spec,
    logical_axis_rules,
    with_partitioning,
    with_sharding_constraint_by_name,
)


def test_current_axis_rules_empty_outside_context():
    """Outside a context the mapping is empty."""
    assert dict(current_axis_rules()) == {}


def test_logical_axis_rules_pushes_and_pops():
    """Entering / leaving a context updates the mapping."""
    with logical_axis_rules([("in", "dp"), ("out", "mp")]):
        rules = dict(current_axis_rules())
        assert rules["in"] == "dp"
        assert rules["out"] == "mp"
    assert dict(current_axis_rules()) == {}


def test_logical_axis_rules_nesting_inner_overrides():
    """Inner rules override outer rules for the same logical name."""
    with logical_axis_rules([("in", "dp")]):
        with logical_axis_rules([("in", "mp")]):
            assert dict(current_axis_rules())["in"] == "mp"
        assert dict(current_axis_rules())["in"] == "dp"


def test_logical_axis_rules_preserve_compound_physical_targets():
    """Logical axis rules can map one semantic axis to multiple mesh axes."""
    with logical_axis_rules([("batch", ("fsdp", "dp")), ("hidden", "tp")]):
        rules = dict(current_axis_rules())
        assert rules["batch"] == ("fsdp", "dp")
        assert rules["hidden"] == "tp"
    assert dict(current_axis_rules()) == {}


def test_with_partitioning_stamps_sharding_on_output():
    """Wrapped initializer records the sharding on its output."""
    from spectrax.init import zeros

    init = with_partitioning(zeros, ("in", "out"))
    assert hasattr(init, "_spx_sharding")
    assert init._spx_sharding.axis_names == ("in", "out")


def test_get_partition_spec_returns_collection_dict():
    """Partition spec tree is keyed by ``collection`` then path."""
    m = Linear(4, 8, rngs=Rngs(0))
    specs = get_partition_spec(m)
    assert "parameters" in specs
    paths = list(specs["parameters"].keys())
    assert any(p.endswith("weight") for p in paths)


def test_partition_axis_resolves_builtin_semantics_and_generation_mode():
    """Partition axis resolves builtin semantics and generation mode."""
    paxis = PartitionAxis(data_parallel_axis="dp", tensor_parallel_axis="tp", sequence_parallel_axis="sp")

    assert paxis.resolve_axis([ct.BATCH, ct.HEAD, ct.EMPTY], mode=ct.MODE_TRAIN) == [
        ("fsdp", "dp"),
        "tp",
        None,
    ]
    assert paxis.resolve_axis([ct.BATCH, ct.HEAD], mode=ct.MODE_DECODE) == [("fsdp", "dp"), "tp"]


def test_partition_axis_rejects_unknown_composite_sub_axis():
    """Composite axes should validate every semantic sub-axis."""
    paxis = PartitionAxis(data_parallel_axis="dp", tensor_parallel_axis="tp")

    with pytest.raises(ValueError, match="unknown_sub_axis"):
        paxis.resolve_axis([(ct.BATCH, "unknown_sub_axis")], mode=ct.MODE_TRAIN)


def test_partition_axis_allows_empty_inside_composite_axis():
    """``EMPTY`` remains legal as a replicated slot inside fused axis specs."""
    paxis = PartitionAxis(data_parallel_axis="dp", tensor_parallel_axis="tp")

    resolved = paxis.resolve_axis([(ct.BATCH, ct.EMPTY)], mode=ct.MODE_TRAIN)

    assert resolved == [[paxis.batch_axis, None]]


def test_partition_axis_register_rejects_missing_axis_rule():
    """Typos should fail at registration time, not later during resolution."""
    with pytest.raises(ValueError, match="axis_rule"):
        PartitionAxis.register("custom_missing_rule", ct.NOT_GIVEN)


def test_partition_manager_context_drives_apply_logical_sharding():
    """Partition manager context drives apply logical sharding."""
    paxis = PartitionAxis(batch_axis=None, query_sequence_axis=None, hidden_state_axis=None)
    x = jnp.ones((2, 3, 4), dtype=jnp.float32)

    with PartitionManager(paxis):
        y = spx.apply_logical_sharding(x, dynamic_axes=ct.HiddenStateSharding)

    assert y.shape == x.shape


def test_partition_helpers_preserve_tree_structure_and_sanitize_missing_mesh_axes():
    """Partition helpers preserve tree structure and sanitize missing mesh axes."""
    tree = {"model": {"weight": jnp.ones((4, 8)), "bias": jnp.ones((8,))}}
    specs = spx.match_partition_rules(
        [(r"model/weight", jax.sharding.PartitionSpec("fsdp", "tp")), (r".*", jax.sharding.PartitionSpec())],
        tree,
    )
    assert tuple(specs["model"]["weight"]) == ("fsdp", "tp")

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape((1,)), ("tp",))
    shard_fns, gather_fns = spx.make_shard_and_gather_fns(specs, mesh=mesh)
    assert callable(shard_fns["model"]["weight"])
    placed = shard_fns["model"]["weight"](tree["model"]["weight"])
    assert tuple(placed.sharding.spec) == (None, "tp")
    gathered = gather_fns["model"]["weight"](placed)
    assert gathered.shape == tree["model"]["weight"].shape


def test_match_partition_rules_applies_min_size_when_not_strict():
    """``min_size`` is independent from strict path validation."""
    specs = spx.match_partition_rules(
        [(r".*", jax.sharding.PartitionSpec("tp"))],
        {"small": jnp.ones((2,)), "large": jnp.ones((2048,))},
        min_size=1024,
        strict=False,
    )

    assert tuple(specs["small"]) == ()
    assert tuple(specs["large"]) == ("tp",)


def test_apply_logical_sharding_resolve_fallback_honors_auto_correct(monkeypatch):
    """Resolve-only managers should still sanitize non-divisible specs."""
    from spectrax.sharding import partition as partition_mod

    seen = {}

    class ResolveOnly:
        """Fixture class for testing."""

        def resolve(self, **_kwargs):
            """Resolve helper."""
            return jax.sharding.PartitionSpec("tp")

    def fake_constraint(x, spec):
        """Fake constraint for testing."""
        seen["spec"] = spec
        return x

    mesh = create_mesh(axis_dims=(-1,), axis_names=("tp",))
    monkeypatch.setattr(partition_mod, "with_sharding_constraint", fake_constraint)
    with mesh:
        partition_mod.apply_logical_sharding(jnp.ones((3,)), partition_manager=ResolveOnly(), auto_correct=True)

    assert tuple(seen["spec"]) == (None,)


def test_with_sharding_constraint_propagates_jax_errors(monkeypatch):
    """Unexpected JAX failures should not become silent no-ops."""
    from spectrax.sharding import partition as partition_mod

    def boom(*_args, **_kwargs):
        """Helper that raises an error."""
        raise RuntimeError("constraint boom")

    monkeypatch.setattr(jax.lax, "with_sharding_constraint", boom)

    with pytest.raises(RuntimeError, match="constraint boom"):
        partition_mod.with_sharding_constraint(jnp.ones((2,)), jax.sharding.PartitionSpec())


def test_mesh_query_helpers_exclude_mpmd_axis_from_spmd_axes():
    """Mesh query helpers exclude mpmd axis from spmd axes."""
    mesh = create_mesh(axis_dims=(1, -1), axis_names=("pp", "tp"), mpmd_axis="pp")

    assert spx.names_in_current_mesh("tp", mesh=mesh)
    assert not spx.names_in_current_mesh("pp", mesh=mesh)
    assert spx.get_axes_size_in_mesh("pp", mesh=mesh) == 1
    assert spx.get_axes_size_in_mesh(("pp", "tp"), mesh=mesh) == len(jax.devices())

    with mesh:
        assert spx.names_in_current_mesh("tp")
        assert not spx.names_in_current_mesh("pp")


def test_mpmd_with_sharding_constraint_uses_explicit_stage_mesh():
    """Mpmd with sharding constraint uses explicit stage mesh."""
    mesh = create_mesh(axis_dims=(1, -1), axis_names=("pp", "tp"), mpmd_axis="pp")
    x = jnp.ones((2, 4), dtype=jnp.float32)

    y = spx.with_sharding_constraint(x, jax.sharding.PartitionSpec("pp", "tp"), mesh=mesh, stage=0)

    assert isinstance(y.sharding, jax.sharding.NamedSharding)
    assert tuple(y.sharding.spec) == (None, "tp")
    assert np.array_equal(y.sharding.mesh.devices, mesh.mpmd_mesh.submesh(0).devices)


def test_mpmd_with_sharding_constraint_promotes_active_raw_mesh():
    """Mpmd with sharding constraint promotes active raw mesh."""
    mesh = create_mesh(axis_dims=(1, -1), axis_names=("pp", "tp"), mpmd_axis="pp")
    x = jnp.ones((2, 4), dtype=jnp.float32)

    with mesh, spx.assign_stage(total=1, current=0):
        y = spx.with_sharding_constraint(
            x,
            jax.sharding.NamedSharding(mesh.jax_mesh, jax.sharding.PartitionSpec("pp", "tp")),
        )

    assert isinstance(y.sharding, jax.sharding.NamedSharding)
    assert tuple(y.sharding.spec) == (None, "tp")
    assert np.array_equal(y.sharding.mesh.devices, mesh.mpmd_mesh.submesh(0).devices)


def test_mpmd_with_sharding_constraint_can_be_ignored():
    """Mpmd with sharding constraint can be ignored."""
    mesh = create_mesh(axis_dims=(1, -1), axis_names=("pp", "tp"), mpmd_axis="pp")
    x = jnp.ones((2, 4), dtype=jnp.float32)

    y = spx.with_sharding_constraint(
        x,
        jax.sharding.PartitionSpec("pp", "tp"),
        mesh=mesh,
        ignore_mpmd=True,
    )

    assert y is x


def test_with_sharding_constraint_handles_pytrees():
    """With sharding constraint handles pytrees."""
    mesh = create_mesh(axis_dims=(1, -1), axis_names=("pp", "tp"), mpmd_axis="pp")
    batch = {"x": jnp.ones((2, 4), dtype=jnp.float32)}

    y = spx.with_sharding_constraint(
        batch,
        jax.sharding.PartitionSpec("pp", "tp"),
        mesh=mesh,
        stage=0,
    )
    ignored = spx.with_sharding_constraint(
        batch,
        jax.sharding.PartitionSpec("pp", "tp"),
        mesh=mesh,
        ignore_mpmd=True,
    )

    assert isinstance(y["x"].sharding, jax.sharding.NamedSharding)
    assert tuple(y["x"].sharding.spec) == (None, "tp")
    assert ignored is batch


def test_mpmd_with_sharding_constraint_uses_requested_stage_rank():
    """Mpmd with sharding constraint uses requested stage rank."""
    if len(jax.devices()) < 2:
        pytest.skip("need at least 2 devices for multi-stage MPMD constraint test")

    mesh = create_mesh(axis_dims=(-1, 1), axis_names=("pp", "tp"), mpmd_axis="pp")
    x = jnp.ones((2, 4), dtype=jnp.float32)

    y = spx.with_sharding_constraint(x, jax.sharding.PartitionSpec("pp", None), mesh=mesh, stage=1)

    assert isinstance(y.sharding, jax.sharding.NamedSharding)
    assert tuple(y.sharding.spec) == (None, None)
    assert np.array_equal(y.sharding.mesh.devices, mesh.mpmd_mesh.submesh(1).devices)


def test_lax_reshard_and_sharding_structure_are_mpmd_aware():
    """Lax reshard and sharding structure are mpmd aware."""
    mesh = create_mesh(axis_dims=(1, -1), axis_names=("pp", "tp"), mpmd_axis="pp")
    tree = {"x": jnp.ones((2, 4), dtype=jnp.float32), "meta": 1}

    out = spx.lax_reshard(tree, {"x": jax.sharding.PartitionSpec("pp", "tp"), "meta": None}, mesh=mesh, stage=0)
    sharding_tree = spx.extract_sharding_structure(out, mesh=mesh, stage=0)

    assert isinstance(out["x"].sharding, jax.sharding.NamedSharding)
    assert tuple(out["x"].sharding.spec) == (None, "tp")
    assert np.array_equal(out["x"].sharding.mesh.devices, mesh.mpmd_mesh.submesh(0).devices)
    assert tuple(sharding_tree["x"].spec) == (None, "tp")
    assert np.array_equal(sharding_tree["x"].mesh.devices, mesh.mpmd_mesh.submesh(0).devices)
    assert sharding_tree["meta"] is None


def test_lax_reshard_uses_active_spxmesh_context():
    """Lax reshard uses active spxmesh context."""
    mesh = create_mesh(axis_dims=(1, -1), axis_names=("pp", "tp"), mpmd_axis="pp")
    x = jnp.ones((2, 4), dtype=jnp.float32)

    with mesh:
        y = spx.lax_reshard(x, jax.sharding.PartitionSpec("pp", "tp"), stage=0)
        sharding = spx.extract_sharding_structure(y, stage=0)

    assert isinstance(y.sharding, jax.sharding.NamedSharding)
    assert tuple(y.sharding.spec) == (None, "tp")
    assert np.array_equal(y.sharding.mesh.devices, mesh.mpmd_mesh.submesh(0).devices)
    assert tuple(sharding.spec) == (None, "tp")
    assert np.array_equal(sharding.mesh.devices, mesh.mpmd_mesh.submesh(0).devices)


def test_lax_reshard_uses_active_assign_stage_context():
    """Lax reshard uses active assign stage context."""
    if len(jax.devices()) < 2:
        pytest.skip("need at least 2 devices for multi-stage MPMD constraint test")

    mesh = create_mesh(axis_dims=(-1, 1), axis_names=("pp", "tp"), mpmd_axis="pp")
    x = jnp.ones((2, 4), dtype=jnp.float32)

    with mesh, spx.assign_stage(total=mesh.mpmd_mesh.mpmd_dim, current=1):
        y = spx.lax_reshard(x, jax.sharding.PartitionSpec("pp", "tp"))
        sharding = spx.extract_sharding_structure(y)

    assert isinstance(y.sharding, jax.sharding.NamedSharding)
    assert tuple(y.sharding.spec) == (None, "tp")
    assert np.array_equal(y.sharding.mesh.devices, mesh.mpmd_mesh.submesh(1).devices)
    assert tuple(sharding.spec) == (None, "tp")
    assert np.array_equal(sharding.mesh.devices, mesh.mpmd_mesh.submesh(1).devices)


def test_apply_logical_sharding_uses_active_spxmesh_and_assign_stage():
    """Apply logical sharding uses active spxmesh and assign stage."""
    if len(jax.devices()) < 2:
        pytest.skip("need at least 2 devices for multi-stage MPMD constraint test")

    mesh = create_mesh(axis_dims=(-1, 1), axis_names=("pp", "tp"), mpmd_axis="pp")
    x = jnp.ones((2, 4, 8), dtype=jnp.float32)

    with mesh, spx.assign_stage(total=mesh.mpmd_mesh.mpmd_dim, current=1):
        with PartitionManager(PartitionAxis(hidden_state_axis="tp")):
            y = spx.apply_logical_sharding(x, dynamic_axes=ct.HiddenStateSharding)

    assert isinstance(y.sharding, jax.sharding.NamedSharding)
    assert tuple(y.sharding.spec) == (None, None, "tp")
    assert np.array_equal(y.sharding.mesh.devices, mesh.mpmd_mesh.submesh(1).devices)


def test_get_partition_spec_uses_active_rules():
    """Active axis-rules resolve logical axis names to mesh axes."""
    m = Linear(4, 8, rngs=Rngs(0))
    with logical_axis_rules([("in", "dp"), ("out", "mp")]):
        specs = get_partition_spec(m)
    weight_paths = [p for p in specs["parameters"] if p.endswith("weight")]
    assert len(weight_paths) == 1
    spec = specs["parameters"][weight_paths[0]]
    assert tuple(spec) == ("dp", "mp")


def test_get_partition_spec_preserves_compound_active_rules():
    """Active rules must not collapse fused physical mesh axes."""
    m = Linear(4, 8, rngs=Rngs(0))
    with logical_axis_rules([("in", ("fsdp", "dp")), ("out", "tp")]):
        specs = get_partition_spec(m)
    weight_paths = [p for p in specs["parameters"] if p.endswith("weight")]
    assert len(weight_paths) == 1
    spec = specs["parameters"][weight_paths[0]]
    assert tuple(spec) == (("fsdp", "dp"), "tp")


def test_get_named_sharding_wraps_in_namedsharding_for_single_device_mesh():
    """Against a 1-device mesh the NamedSharding tree is well-formed."""
    m = Linear(2, 4, rngs=Rngs(0))
    devices = jax.devices()[:1]
    mesh = jax.sharding.Mesh(devices, ("dp",))
    with logical_axis_rules([("in", "dp"), ("out", None)]):
        tree = get_named_sharding(m, mesh)
    for _p, ns in tree["parameters"].items():
        assert isinstance(ns, jax.sharding.NamedSharding)


def test_get_named_sharding_preserves_compound_active_rules():
    """NamedSharding resolution keeps compound logical-rule targets."""
    m = Linear(4, 8, rngs=Rngs(0))
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape((1, 1, 1)), ("fsdp", "dp", "tp"))

    with logical_axis_rules([("in", ("fsdp", "dp")), ("out", "tp")]):
        tree = get_named_sharding(m, mesh)

    weight_paths = [p for p in tree["parameters"] if p.endswith("weight")]
    assert len(weight_paths) == 1
    assert tuple(tree["parameters"][weight_paths[0]].spec) == (("fsdp", "dp"), "tp")


def test_parameter_named_sharding_accepts_compound_physical_axis_names():
    """Physical mesh axis names can be used directly, including compound axes."""

    class Model(spx.Module):
        """Fixture model module for testing."""

        def __init__(self):
            """Initialize with weight."""
            super().__init__()
            self.weight = spx.Parameter(
                jnp.ones((4, 4), dtype=jnp.float32),
                axis_names=(("fsdp", "sp"), "tp"),
            )

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape((1, 1, 1)), ("fsdp", "sp", "tp"))
    tree = get_named_sharding(Model(), mesh)

    assert tuple(tree["parameters"]["weight"].spec) == (("fsdp", "sp"), "tp")


def test_parameter_named_sharding_accepts_compound_logical_axis_names():
    """Compound entries can still resolve through logical-axis rules."""

    class Model(spx.Module):
        """Fixture model module for testing."""

        def __init__(self):
            """Initialize with weight."""
            super().__init__()
            self.weight = spx.Parameter(
                jnp.ones((4, 4), dtype=jnp.float32),
                axis_names=(("data", "sequence"), "model"),
            )

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape((1, 1, 1)), ("fsdp", "sp", "tp"))
    with logical_axis_rules([("data", "fsdp"), ("sequence", "sp"), ("model", "tp")]):
        tree = get_named_sharding(Model(), mesh)

    assert tuple(tree["parameters"]["weight"].spec) == (("fsdp", "sp"), "tp")


def test_stage_tagged_variable_resolves_to_stage_local_namedsharding():
    """Stage-tagged vars on an MPMD mesh resolve to their owning sub-mesh."""

    class StageTagged(spx.Module):
        """Fixture module for testing."""

        def __init__(self):
            """Initialize the instance."""
            super().__init__()
            with spx.assign_stage(total=4, current=3):
                self.weight = spx.Parameter(jnp.ones((2, 4)), sharding=("fsdp", "tp"))

    if len(jax.devices()) < 4:
        pytest.skip("need at least 4 devices for stage-local sharding test")

    mesh = create_mesh(
        axis_dims=(2, 2),
        axis_names=("pp", "tp"),
        mpmd_axis="pp",
    )
    model = StageTagged()
    with logical_axis_rules([("fsdp", None), ("tp", "tp")]):
        shards = get_named_sharding(model, mesh)

    weight_sharding = shards["parameters"]["weight"]
    assert model.weight.stage_index == 3
    assert model.weight.resolved_stage_index(mesh) == 1
    assert tuple(weight_sharding.spec) == (None, "tp")
    assert np.array_equal(weight_sharding.mesh.devices, mesh.mpmd_mesh.submesh(1).devices)
    assert np.array_equal(model.weight.stage_mesh(mesh).devices, mesh.mpmd_mesh.submesh(1).devices)


def test_mpmd_mesh_orders_pipeline_axis_by_topology():
    """MPMD stages should follow a nearest-neighbor device path when coordinates exist."""
    if len(jax.devices()) < 4:
        pytest.skip("need at least 4 devices for topology ordering test")
    if getattr(jax.devices()[0], "coords", None) is None:
        pytest.skip("device coordinates are required for topology ordering")

    mesh = create_mesh(axis_dims=(4,), axis_names=("pp",), mpmd_axis="pp")
    stage_devices = [mesh.mpmd_mesh.submesh(i).devices.reshape(-1)[0] for i in range(4)]
    coords = [tuple(float(v) for v in device.coords) for device in stage_devices]

    def dist(a, b):
        """Distribution helper."""
        return sum((x - y) ** 2 for x, y in zip(a, b, strict=True))

    nearest_upper_bound = max(min(dist(c, other) for other in coords if other != c) for c in coords)
    assert all(dist(a, b) <= nearest_upper_bound for a, b in itertools.pairwise(coords))


def test_parameter_applies_sharding_immediately_inside_mesh_context():
    """Constructing a parameter under ``with mesh`` applies its sharding."""
    mesh = create_mesh(axis_dims=(-1,), axis_names=("tp",))
    with mesh, logical_axis_rules([("row", "tp"), ("col", None)]):
        weight = spx.Parameter(jnp.arange(32.0).reshape(8, 4), sharding=("row", "col"))

    assert isinstance(weight.value.sharding, jax.sharding.NamedSharding)
    assert tuple(weight.value.sharding.spec) == ("tp", None)
    assert np.array_equal(weight.value.sharding.mesh.devices, mesh.jax_mesh.devices)


def test_variable_init_placement_hook_can_place_without_spx_mesh_context():
    """Runtime integrations can place freshly-created variables from metadata."""
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape((1,)), ("tp",))
    axis_rules = {"batch": None, "hidden": "tp"}
    seen = {}

    def place(value, metadata, explicit_sharding):
        """Placement helper."""
        seen["metadata"] = dict(metadata)
        seen["explicit_sharding"] = explicit_sharding
        axis_names = metadata.get("axis_names")
        if axis_names is None:
            return None
        spec = tuple(axis_rules.get(axis, axis) for axis in axis_names)
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))
        return jax.device_put(value, sharding)

    with spx.variable_init_placement(place):
        weight = spx.Parameter(jnp.arange(32.0).reshape(8, 4), axis_names=("batch", "hidden"))

    assert seen["metadata"]["axis_names"] == ("batch", "hidden")
    assert seen["explicit_sharding"] is False
    assert isinstance(weight.value.sharding, jax.sharding.NamedSharding)
    assert tuple(weight.value.sharding.spec) == (None, "tp")
    assert np.array_equal(weight.value.sharding.mesh.devices, mesh.devices)


def test_deferred_parameter_materialize_uses_variable_init_placement_hook():
    """Lazy variables also use the active init-placement hook at materialization."""
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape((1,)), ("tp",))

    def init(_rngs, shape, dtype):
        """Initialization helper."""
        return jnp.arange(np.prod(shape), dtype=dtype).reshape(shape)

    def place(value, metadata, _explicit_sharding):
        """Placement helper."""
        axis_names = metadata.get("axis_names")
        if axis_names != ("batch", "hidden"):
            return None
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "tp"))
        return jax.device_put(value, sharding)

    weight = spx.DeferredParameter(
        (None, 4),
        init,
        None,
        jnp.float32,
        axis_names=("batch", "hidden"),
    )
    weight.resolve_shape((8, 4))
    with spx.variable_init_placement(place):
        weight.materialize()

    assert isinstance(weight.value.sharding, jax.sharding.NamedSharding)
    assert tuple(weight.value.sharding.spec) == (None, "tp")


def test_stage_tagged_parameter_applies_stage_local_sharding_immediately():
    """Stage-tagged parameters built under an MPMD mesh land on the owner sub-mesh."""
    if len(jax.devices()) < 4:
        pytest.skip("need at least 4 devices for stage-local placement test")

    mesh = create_mesh(
        axis_dims=(2, 2),
        axis_names=("pp", "tp"),
        mpmd_axis="pp",
    )
    with mesh, logical_axis_rules([("fsdp", None), ("tp", "tp")]):
        with spx.assign_stage(total=4, current=3):
            weight = spx.Parameter(jnp.arange(32.0).reshape(8, 4), sharding=("fsdp", "tp"))

    assert isinstance(weight.value.sharding, jax.sharding.NamedSharding)
    assert tuple(weight.value.sharding.spec) == (None, "tp")
    assert np.array_equal(weight.value.sharding.mesh.devices, mesh.mpmd_mesh.submesh(1).devices)


def test_parameter_preserves_existing_value_sharding_without_explicit_override():
    """A pre-sharded input value stays as-is unless ``sharding=...`` is explicit."""
    mesh = create_mesh(axis_dims=(-1,), axis_names=("tp",))
    replicated = jax.sharding.NamedSharding(mesh.jax_mesh, jax.sharding.PartitionSpec(None, None))
    with mesh, logical_axis_rules([("row", "tp"), ("col", None)]):
        x = jax.device_put(jnp.arange(32.0).reshape(8, 4), replicated)
        weight = spx.Parameter(x, axis_names=("row", "col"))

    assert isinstance(weight.value.sharding, jax.sharding.NamedSharding)
    assert tuple(weight.value.sharding.spec) == (None, None)


def test_parameter_explicit_sharding_overrides_existing_value_sharding():
    """Explicit constructor sharding still re-places an already-sharded value."""
    mesh = create_mesh(axis_dims=(-1,), axis_names=("tp",))
    replicated = jax.sharding.NamedSharding(mesh.jax_mesh, jax.sharding.PartitionSpec(None, None))
    with mesh, logical_axis_rules([("row", "tp"), ("col", None)]):
        x = jax.device_put(jnp.arange(32.0).reshape(8, 4), replicated)
        weight = spx.Parameter(x, sharding=("row", "col"))

    assert isinstance(weight.value.sharding, jax.sharding.NamedSharding)
    assert tuple(weight.value.sharding.spec) == ("tp", None)


def test_with_sharding_constraint_by_name_is_identity_inside_mesh():
    """With a mesh in context the constraint is effectively a no-op for a 1-device mesh."""
    devices = jax.devices()[:1]
    mesh = jax.sharding.Mesh(devices, ("dp",))
    x = jnp.arange(6.0).reshape((2, 3))
    with mesh:
        y = with_sharding_constraint_by_name(x, ("batch", "features"))
    assert jnp.array_equal(x, y)


def test_with_sharding_constraint_by_name_inside_jit_works():
    """Inside a ``jit`` with a 1-device mesh the constraint is honoured."""
    devices = jax.devices()[:1]
    mesh = jax.sharding.Mesh(devices, ("dp",))

    def f(x):
        """Helper function."""
        with logical_axis_rules([("batch", "dp"), ("features", None)]):
            return with_sharding_constraint_by_name(x, ("batch", "features"))

    x = jnp.arange(6.0).reshape((2, 3))
    with mesh:
        out = jax.jit(f)(x)
    assert out.shape == x.shape


def test_with_sharding_constraint_by_name_accepts_compound_axis_names():
    """Constraints support one array dimension mapped over multiple mesh axes."""
    x = jnp.arange(16.0).reshape((4, 4))
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape((1, 1, 1)), ("fsdp", "sp", "tp"))

    with mesh, logical_axis_rules([("data", "fsdp"), ("sequence", "sp"), ("model", "tp")]):
        y = with_sharding_constraint_by_name(x, (("data", "sequence"), "model"))

    assert jnp.array_equal(x, y)


def test_with_sharding_constraint_by_name_accepts_compound_rule_values():
    """Constraint rules can map one logical axis to a fused physical spec."""
    x = jnp.arange(16.0).reshape((4, 4))
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape((1, 1, 1)), ("fsdp", "dp", "tp"))

    with mesh, logical_axis_rules([("batch", ("fsdp", "dp")), ("features", "tp")]):
        y = with_sharding_constraint_by_name(x, ("batch", "features"))

    assert jnp.array_equal(x, y)


@pytest.mark.parametrize(
    "axis_names",
    [
        ("in", "out"),
        ("batch",),
        (),
    ],
)
def test_with_partitioning_accepts_various_axis_shapes(axis_names):
    """``with_partitioning`` normalizes multiple tuple shapes."""
    from spectrax.init import zeros

    init = with_partitioning(zeros, axis_names)
    arr = init(jax.random.key(0), (2,) * (len(axis_names) or 1), jnp.float32)
    assert arr.shape == (2,) * (len(axis_names) or 1)
