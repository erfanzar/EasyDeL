"""Tiny array-computation regressions. Run on an accelerator, never on CPU."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.mixins.generation import EasyGenerationMixin
from easydel.infra.mixins.sharding import EasyShardingMixin
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec


class StateModel(EasyShardingMixin):
    def __init__(self, mesh, state):
        self.config = SimpleNamespace(mesh=mesh)
        self.state = state

    def split_module(self):
        return "graph", self.state, {}

    def merge_module(self, graph, state, other):
        assert graph == "graph"
        self.state = state
        return self

    def resolve_shardings_regex(self, mesh=None):
        return (("weight", NamedSharding(mesh or self.mesh, PartitionSpec("tp"))),)


@pytest.fixture
def mesh():
    return Mesh(np.array(jax.devices()), ("tp",))


def test_named_sharded_arrays_are_resharded_to_target(mesh):
    if mesh.size < 2:
        pytest.skip("Requires two accelerator devices to distinguish replication from partitioning")
    value = jax.device_put(np.arange(mesh.size * 4, dtype=np.float32), NamedSharding(mesh, PartitionSpec()))
    model = StateModel(mesh, {"weight": value})
    assert model.apply_sharding_for_tree(model.state) is model.state
    placed = model.apply_sharding_for_tree(model.state, mesh=mesh)
    assert placed["weight"].sharding.spec == PartitionSpec("tp")
    np.testing.assert_array_equal(placed["weight"], np.arange(mesh.size * 4, dtype=np.float32))


def test_apply_out_shardings_accepts_real_split_tuple(mesh):
    state = {"weight": jnp.arange(mesh.size * 4, dtype=jnp.float32)}
    model = StateModel(mesh, state)
    result = model.apply_out_shardings(({"weight": NamedSharding(mesh, PartitionSpec())}, {}))
    assert result is model
    assert result.state["weight"].sharding.spec == PartitionSpec()
    np.testing.assert_array_equal(result.state["weight"], np.arange(mesh.size * 4, dtype=np.float32))


def test_gather_model_honors_mesh_and_keeps_values(mesh):
    values = np.arange(mesh.size * 4, dtype=np.float32)
    state = {"weight": jax.device_put(values, NamedSharding(mesh, PartitionSpec("tp")))}
    model = StateModel(mesh, state)
    other_mesh = Mesh(np.array(jax.devices()), ("gather",))
    result = model.gather_model(mesh=other_mesh)
    assert isinstance(result.state["weight"], np.ndarray)
    np.testing.assert_array_equal(result.state["weight"], values)


def test_fully_gather_keeps_public_feature(mesh):
    class Module(EasyShardingMixin, spx.Module):
        def __init__(self):
            self.config = SimpleNamespace(mesh=mesh)
            self.weight = spx.Parameter(
                jax.device_put(np.arange(mesh.size * 4, dtype=np.float32), NamedSharding(mesh, PartitionSpec("tp")))
            )

        def resolve_sharding_for_tree(self, tree=None, *, mesh=None):
            return jax.tree_util.tree_map(lambda _: NamedSharding(self.mesh, PartitionSpec("tp")), tree)

    gathered = Module().fully_gather()
    assert gathered.weight.value.sharding.spec == PartitionSpec()
    np.testing.assert_array_equal(gathered.weight.value, np.arange(mesh.size * 4, dtype=np.float32))


def test_generation_cache_shardings_are_applied(mesh):
    class Config(SimpleNamespace):
        def get_text_config(self):
            return self

        def get_mask_details(self):
            return None

    class Model(EasyGenerationMixin):
        def __init__(self):
            self.config = Config(
                mesh=mesh,
                num_hidden_layers=1,
                num_key_value_heads=1,
                head_dim=4,
                kvdtype=jnp.float32,
                kv_cache_quantization_config=None,
            )
            self._quant_class = lambda **_: None

        def get_inference_cache_type(self):
            return "hybrid"

        def init_operations_cache(self, **kwargs):
            return {"key": jnp.zeros((mesh.size, 4)), "value": jnp.ones((mesh.size, 4))}

        def prepare_inputs_for_call(self, **kwargs):
            return kwargs

    model = Model()
    target = NamedSharding(mesh, PartitionSpec("tp", None))
    cache = model.init_cache(batch_size=mesh.size, max_length=4, shardings={"key": target})
    assert cache["key"].sharding.spec == target.spec
    prepared = model.prepare_inputs_for_generation(
        jnp.ones((mesh.size, 1), dtype=jnp.int32),
        max_length=4,
        pad_token_id=0,
        starts=0,
        shardings={"key": target},
    )
    assert prepared["past_key_values"]["key"].sharding.spec == target.spec
