import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx as nn

import easydel as ed
from easydel.infra.base_state import EasyDeLState


@pytest.fixture(scope="module")
def tiny_sharded_llama():
    module_config, module_class = ed.get_modules_by_type("llama", ed.TaskType.CAUSAL_LM)
    config = module_config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    config.add_basic_configurations(
        sharding_axis_dims=(1, 1, -1, 1, 1),
        use_sharding_constraint=False,
    )
    with config.mesh:
        model = module_class.sequential_init(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=nn.Rngs(0),
        )
        return model.shard_model()


def test_to_state_handles_meta_graphother_leaves(tiny_sharded_llama):
    state = tiny_sharded_llama.to_state()
    assert isinstance(state, EasyDeLState)


def test_state_gather_paths_handle_graphother_tree(tiny_sharded_llama):
    state = EasyDeLState.create(model=tiny_sharded_llama)
    gathered_model_state = state.gather_model()
    assert isinstance(gathered_model_state, EasyDeLState)
    gathered_state = state.gather_state()
    assert isinstance(gathered_state, EasyDeLState)


def test_optimizer_gather_works_without_mesh_context_and_create_validation(tiny_sharded_llama):
    state = EasyDeLState.create(model=tiny_sharded_llama).init_tx(optax.adam(1e-3))
    gathered_opt_state = state.gather_optimizer_state()
    assert isinstance(gathered_opt_state, EasyDeLState)

    graphdef, graphstate, _ = nn.split(tiny_sharded_llama, nn.Param, ...)
    with pytest.raises(ValueError):
        EasyDeLState.create(graphdef=graphdef, graphstate=graphstate, graphother=None)
