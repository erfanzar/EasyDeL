import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from easydel.infra.mixins.sharding import EasyShardingMixin
from easydel.infra.sharding import device_put_if_sharding_mismatch, sharding_matches


def _single_axis_mesh() -> Mesh:
    return Mesh(np.asarray(jax.devices()[:1]), ("x",))


def test_device_put_if_sharding_mismatch_skips_exact_named_sharding(monkeypatch):
    mesh = _single_axis_mesh()
    sharding = NamedSharding(mesh, PartitionSpec())
    value = jax.device_put(jnp.arange(4), sharding)

    def unexpected_device_put(*_args, **_kwargs):
        raise AssertionError("already-sharded leaves should not be device_put again")

    monkeypatch.setattr(jax, "device_put", unexpected_device_put)

    assert sharding_matches(value, sharding)
    assert device_put_if_sharding_mismatch(value, sharding, donate=True) is value


def test_device_put_if_sharding_mismatch_replaces_different_spec(monkeypatch):
    mesh = _single_axis_mesh()
    source_sharding = NamedSharding(mesh, PartitionSpec())
    target_sharding = NamedSharding(mesh, PartitionSpec("x"))
    value = jax.device_put(jnp.arange(4), source_sharding)
    sentinel = object()

    def fake_device_put(leaf, sharding, *, donate=False):
        assert leaf is value
        assert sharding is target_sharding
        assert donate is False
        return sentinel

    monkeypatch.setattr(jax, "device_put", fake_device_put)

    assert not sharding_matches(value, target_sharding)
    assert device_put_if_sharding_mismatch(value, target_sharding) is sentinel


def test_device_put_if_sharding_mismatch_ignores_non_array_leaves(monkeypatch):
    def unexpected_device_put(*_args, **_kwargs):
        raise AssertionError("non-array leaves should not be device_put")

    monkeypatch.setattr(jax, "device_put", unexpected_device_put)

    assert device_put_if_sharding_mismatch("metadata", None) == "metadata"


def test_apply_sharding_for_tree_skips_already_named_sharded_tree():
    mesh = _single_axis_mesh()
    sharding = NamedSharding(mesh, PartitionSpec())
    value = jax.device_put(jnp.arange(4), sharding)

    class DummySharder(EasyShardingMixin):
        def resolve_sharding_for_tree(self, tree=None, *, mesh=None):
            del tree, mesh
            raise AssertionError("already-sharded trees should not resolve sharding rules")

    tree = {"w": value}

    assert DummySharder().apply_sharding_for_tree(tree) is tree


def test_apply_sharding_for_tree_honors_explicit_mesh_override():
    mesh = _single_axis_mesh()
    sharding = NamedSharding(mesh, PartitionSpec())
    value = jax.device_put(jnp.arange(4), sharding)
    calls = {"resolve": 0}

    class DummySharder(EasyShardingMixin):
        def resolve_sharding_for_tree(self, tree=None, *, mesh=None):
            assert mesh is not None
            calls["resolve"] += 1
            return {"w": sharding}

    tree = {"w": value}

    assert DummySharder().apply_sharding_for_tree(tree, mesh=mesh)["w"] is value
    assert calls["resolve"] == 1
