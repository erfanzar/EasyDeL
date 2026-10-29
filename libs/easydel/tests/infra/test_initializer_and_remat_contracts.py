"""Initializer metadata and independent checkpoint configurations."""

import pickle

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.utils import ArrayParam, auto_remat
from jax import numpy as jnp


@pytest.mark.cpu_ok
def test_array_param_initializer_metadata_survives_graph_and_pickle():
    class Holder(spx.Module):
        def __init__(self):
            super().__init__()
            self.weight = ArrayParam.bound(
                (4,), jnp.float32, "uniform", {"scale": 0.25}, value=jax.ShapeDtypeStruct((4,), jnp.float32)
            )

    holder = Holder()
    graph, state = spx.export(holder)
    restored = spx.bind(graph, state)
    for param in (holder.weight, restored.weight, pickle.loads(pickle.dumps(holder.weight))):
        assert param.init_method == "uniform"
        assert param.init_kwargs == {"scale": 0.25}
        param.init_method = "normal"
        param.init_kwargs = {"stddev": 0.5}
        assert param.metadata["init_method"] == "normal"
        assert param.metadata["init_kwargs"] == {"stddev": 0.5}


@pytest.mark.parametrize(("method", "kwargs"), [("zeros", {}), ("ones", {}), ("normal", {"stddev": 0.5})])
def test_array_param_reinitialization_matches_independent_initializer(method, kwargs):
    key = jax.random.PRNGKey(42)
    param = ArrayParam.bound((8,), jnp.float32, method, kwargs, value=jnp.full((8,), -5.0))
    graph, state = spx.export(_ParameterHolder(param))
    restored = spx.bind(graph, state).weight
    restored.resure(key)
    initializer = getattr(jax.nn.initializers, method)
    if method == "normal":
        initializer = initializer(**kwargs)
    expected = initializer(key, (8,), jnp.float32)
    np.testing.assert_array_equal(restored.value, expected)


class _ParameterHolder(spx.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight


@pytest.mark.cpu_ok
@pytest.mark.parametrize("policy", ["nothing_saveable", "everything_saveable"])
@pytest.mark.parametrize("prevent_cse", [False, True])
def test_auto_remat_preserves_shared_class_and_graph_type(policy, prevent_cse):
    class Block(spx.Module):
        def forward(self, x):
            return x * x

    original = Block.forward
    earlier = Block()
    assert auto_remat(Block, policy="none") is Block
    assert Block.forward is original
    rematted = auto_remat(Block, policy=policy, prevent_cse=prevent_cse)
    assert rematted is Block
    assert getattr(Block.forward, "_easydel_auto_remat_wrapped", False)
    assert earlier.forward.__func__ is Block.forward
    wrapped_forward = Block.forward
    for other_policy in (policy, "everything_saveable", "nothing_saveable", "none"):
        assert auto_remat(Block, policy=other_policy) is Block
        assert Block.forward is wrapped_forward
    graph, state = spx.export(rematted())
    assert type(spx.bind(graph, state)) is Block
    assert jax.eval_shape(lambda x: rematted()(x), jax.ShapeDtypeStruct((2,), jnp.float32)).shape == (2,)
    traced = jax.make_jaxpr(lambda x: rematted()(x))(jax.ShapeDtypeStruct((2,), jnp.float32))
    params = next(eqn.params for eqn in traced.jaxpr.eqns if eqn.primitive.name == "remat2")
    assert params["policy"] is getattr(jax.checkpoint_policies, policy)
    assert params["prevent_cse"] is prevent_cse
