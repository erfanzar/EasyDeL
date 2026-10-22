"""SmolLM3 regressions (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.caching import RaggedPagesCache, RaggedPagesMetadata, TransformerCache, TransformerMetadata
from easydel.modules.smollm3.modeling_smollm3 import SmolLM3MLP, SmolLM3Model
from easydel.modules.smollm3.smollm3_configuration import SmolLM3Config
from jax import numpy as jnp


def _config(**kwargs):
    return SmolLM3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        **kwargs,
    )


@pytest.mark.cpu_ok
@pytest.mark.parametrize("cache_type", [TransformerCache, RaggedPagesCache])
def test_cache_shell_contract_accepts_layer_count_without_allocating_arrays(cache_type):
    cache = cache_type.init_empty(2)
    assert isinstance(cache, cache_type)
    assert cache.views == [None, None]
    assert cache_type.init_empty(0).views == []


def test_norms_follow_compute_dtype_not_param_dtype():
    model = SmolLM3Model(_config(), dtype=jnp.bfloat16, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    hidden = jnp.ones((1, 3, 16), dtype=jnp.bfloat16)
    for norm in (model.norm, model.layers[0].input_layernorm, model.layers[0].post_attention_layernorm):
        assert norm(hidden).dtype == jnp.bfloat16
        assert norm.weight.value.dtype == jnp.float32
    assert model(input_ids=jnp.array([[1, 2, 3]])).last_hidden_state.dtype == jnp.bfloat16


def test_mlp_honours_hidden_act():
    mlp = SmolLM3MLP(
        _config(hidden_act="relu"),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    hidden = jnp.linspace(-1.0, 1.0, 32).reshape(1, 2, 16)
    gate_up = np.asarray(hidden) @ np.asarray(mlp.gate_up_proj.weight.value)
    gate, up = np.split(gate_up, 2, axis=-1)
    expected = (np.maximum(gate, 0) * up) @ np.asarray(mlp.down_proj.weight.value)
    np.testing.assert_allclose(np.asarray(mlp(hidden)), expected, rtol=1e-4, atol=1e-6)


def test_cache_metadata_without_cache_does_not_crash():
    model = SmolLM3Model(_config(), dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    ids = jnp.array([[1, 2, 3]], jnp.int32)
    out = model(input_ids=ids, cache_metadata=TransformerMetadata())
    assert out.last_hidden_state.shape == (1, 3, 16)
    assert np.isfinite(np.asarray(out.last_hidden_state)).all()
    assert isinstance(out.past_key_values, TransformerCache)
    assert len(out.past_key_values.views) == 2


def test_paged_metadata_without_cache_returns_a_cache_shell():
    model = SmolLM3Model(_config(), dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    metadata = RaggedPagesMetadata.create_empty(num_tokens=3, max_num_reqs=1, max_pages=1)
    out = model(input_ids=jnp.array([[1, 2, 3]]), cache_metadata=metadata)
    assert isinstance(out.past_key_values, RaggedPagesCache)
    assert len(out.past_key_values.views) == 2
    assert out.last_hidden_state.shape == (1, 3, 16)


def test_first_scanned_forward_does_not_cache_a_tracer():
    model = SmolLM3Model(_config(scan_layers=True), dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    ids = jnp.array([[1, 2, 3]], jnp.int32)
    with jax.checking_leaks():
        first = model(input_ids=ids, trace=False).last_hidden_state
    second = model(input_ids=ids, trace=False).last_hidden_state
    np.testing.assert_allclose(np.asarray(first), np.asarray(second), rtol=1e-5, atol=1e-5)
