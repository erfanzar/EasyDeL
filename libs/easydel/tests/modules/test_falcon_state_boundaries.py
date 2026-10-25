"""Falcon recurrent-state regressions; computation tests for accelerator execution only."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.caching import HybridCacheView, RecurrentCacheView
from easydel.modules.falcon_h1 import FalconH1Config, FalconH1ForCausalLM, FalconH1Model
from easydel.modules.falcon_h1.modeling_falcon_h1 import FalconH1Mixer
from easydel.modules.falcon_mamba import FalconMambaConfig, FalconMambaForCausalLM
from easydel.modules.falcon_mamba.modeling_falcon_mamba import FalconMambaMixer
from ejkernel.types import MaskInfo
from jax import numpy as jnp


def _mamba_config(use_associative_scan=False):
    return FalconMambaConfig(
        vocab_size=16,
        hidden_size=8,
        state_size=2,
        num_hidden_layers=1,
        expand=2,
        conv_kernel=2,
        time_step_rank=2,
        use_associative_scan=use_associative_scan,
    )


def _recurrent_cache(conv, state):
    return RecurrentCacheView(
        conv_state=conv,
        recurrent_state=state,
        positions=jnp.zeros((conv.shape[0],), dtype=jnp.int32),
        seqlen_offset=jnp.asarray(0, dtype=jnp.int32),
        metadata=None,
        layer_index=0,
    )


@pytest.mark.parametrize("use_associative_scan", [False, True])
def test_falcon_mamba_packed_slots_match_independent_sequences_and_keep_idle_slot(use_associative_scan):
    config = _mamba_config(use_associative_scan)
    mixer = FalconMambaMixer(config, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    inputs = jax.random.normal(jax.random.key(1), (1, 5, config.hidden_size))
    conv = jnp.zeros((3, config.intermediate_size, config.conv_kernel)).at[2].set(0.25)
    state = jax.random.normal(jax.random.key(5), (3, config.intermediate_size, config.state_size)) * 0.1
    metadata = SimpleNamespace(query_start_loc=jnp.asarray([0, 2, 5, 5]), num_seqs=2)
    packed_output, packed_cache = mixer(inputs, cache_params=_recurrent_cache(conv, state), cache_metadata=metadata)
    for slot, (start, end) in enumerate(((0, 2), (2, 5))):
        independent_output, independent_cache = mixer(
            inputs[:, start:end],
            cache_params=_recurrent_cache(conv[slot : slot + 1], state[slot : slot + 1]),
        )
        np.testing.assert_allclose(packed_output[:, start:end], independent_output, atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(packed_cache.conv_state[slot], independent_cache.conv_state[0], atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(
            packed_cache.recurrent_state[slot], independent_cache.recurrent_state[0], atol=2e-4, rtol=2e-4
        )
    np.testing.assert_array_equal(packed_cache.conv_state[2], conv[2])
    np.testing.assert_array_equal(packed_cache.recurrent_state[2], state[2])


def _scan_count(graph):
    """Count sequential scan primitives, including nested compiled call graphs."""
    # Current JAX exposes Jaxpr.jaxpr as a self-alias; inspect equations
    # before unwrapping ClosedJaxpr to avoid recursing into that alias.
    if hasattr(graph, "eqns"):
        return sum((eqn.primitive.name == "scan") + _scan_count(eqn.params) for eqn in graph.eqns)
    if hasattr(graph, "jaxpr"):
        return _scan_count(graph.jaxpr)
    if isinstance(graph, dict):
        return sum(_scan_count(value) for value in graph.values())
    if isinstance(graph, (tuple, list)):
        return sum(_scan_count(value) for value in graph)
    return 0


@pytest.mark.parametrize("packed", [False, True])
def test_falcon_mamba_scan_selection_preserves_outputs_and_final_state(packed):
    """Check the selected execution graph as well as public output/cache parity.

    Parity alone cannot catch a dropped flag: running sequentially twice also
    produces matching outputs. The graph must lose the SSM sequential scan when
    associative composition is requested, in both regular and packed execution.
    """
    config = _mamba_config()
    mixer = FalconMambaMixer(config, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    inputs = jax.random.normal(jax.random.key(6), (1, 5, config.hidden_size))
    slots = 3 if packed else 1
    conv = jnp.zeros((slots, config.intermediate_size, config.conv_kernel))
    state = jax.random.normal(jax.random.key(7), (slots, config.intermediate_size, config.state_size)) * 0.1
    cache = _recurrent_cache(conv, state)
    metadata = SimpleNamespace(query_start_loc=jnp.asarray([0, 2, 5, 5]), num_seqs=2) if packed else None

    def forward(x):
        output, updated = mixer(x, cache_params=cache, cache_metadata=metadata)
        return output, updated.conv_state, updated.recurrent_state

    sequential = forward(inputs)
    sequential_graph = jax.make_jaxpr(forward)(inputs)
    config.use_associative_scan = True
    associative = forward(inputs)
    # A fresh trace avoids reusing a specialization of the mutable config flag.
    associative_graph = jax.make_jaxpr(lambda x: forward(x))(inputs)
    assert _scan_count(sequential_graph) == _scan_count(associative_graph) + 1
    for sequential_value, associative_value in zip(sequential, associative, strict=True):
        np.testing.assert_allclose(sequential_value, associative_value, atol=2e-4, rtol=2e-4)
    if packed:
        np.testing.assert_array_equal(associative[1][2], conv[2])
        np.testing.assert_array_equal(associative[2][2], state[2])


def test_falcon_mamba_lm_packed_documents_match_separate_documents():
    config = _mamba_config()
    model = FalconMambaForCausalLM(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    embeds = jax.random.normal(jax.random.key(2), (1, 4, config.hidden_size))
    mask = MaskInfo.from_segments(q_segment_ids=jnp.asarray([[0, 0, 1, 1]], dtype=jnp.int32))
    packed = model(inputs_embeds=embeds, mask_info=mask).logits
    for start in (0, 2):
        separate = model(inputs_embeds=embeds[:, start : start + 2]).logits
        np.testing.assert_allclose(packed[:, start : start + 2], separate, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("model_class", [FalconH1Model, FalconH1ForCausalLM])
@pytest.mark.parametrize("bounds", [(0.05, 0.05), (0.01, 0.1)])
def test_falcon_h1_sequential_init_preserves_configured_time_step_initialization(model_class, bounds):
    config = FalconH1Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        mamba_d_ssm=8,
        mamba_n_heads=2,
        mamba_d_head=4,
        mamba_d_state=2,
        mamba_d_conv=2,
        mamba_chunk_size=2,
        time_step_min=bounds[0],
        time_step_max=bounds[1],
    )
    model = model_class.sequential_init(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(4))
    backbone = model if model_class is FalconH1Model else model.model
    steps = []
    for layer in backbone.layers:
        bias = layer.mamba.dt_bias.value
        assert bias.shape == (config.mamba_n_heads,)
        assert bias.dtype == jnp.float32
        steps.append(np.asarray(jax.nn.softplus(bias)))
    steps = np.concatenate(steps)
    assert np.all(np.isfinite(steps))
    assert np.all(steps >= bounds[0] - 1e-6)
    assert np.all(steps <= bounds[1] + 1e-6)
    if bounds[0] == bounds[1]:
        np.testing.assert_allclose(steps, bounds[0], rtol=1e-5, atol=1e-6)
    else:
        assert np.ptp(steps) > 0


@pytest.mark.parametrize("packed", [False, True])
def test_falcon_h1_time_step_clamp_controls_recurrent_state(packed):
    config = FalconH1Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        mamba_d_ssm=8,
        mamba_n_heads=2,
        mamba_d_head=4,
        mamba_d_state=2,
        mamba_d_conv=2,
        mamba_chunk_size=2,
        time_step_min=0.05,
        time_step_max=0.05,
        time_step_limit=(0.2, 0.2),
    )
    mixer = FalconH1Mixer(config, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    np.testing.assert_allclose(jax.nn.softplus(mixer.dt_bias.value), 0.05, rtol=1e-5, atol=1e-6)
    inputs = jax.random.normal(jax.random.key(3), (1, 2, config.hidden_size))
    cache = HybridCacheView(
        key=None,
        value=None,
        conv_state=jnp.zeros((1, mixer.conv_dim, mixer.conv_kernel_size)),
        recurrent_state=jnp.ones((1, mixer.num_heads, mixer.head_dim, mixer.ssm_state_size)),
        positions=jnp.zeros((1,), dtype=jnp.int32),
        metadata=None,
        layer_index=0,
        layer_type="linear_attention",
    )
    metadata = SimpleNamespace(query_start_loc=jnp.asarray([0, 2]), num_seqs=1) if packed else None
    # With a fixed lower == upper clamp, opposite raw time steps must give the
    # same outputs and cache transition. Without the clamp they differ greatly.
    mixer.dt_bias.value = jnp.full_like(mixer.dt_bias.value, -30.0)
    low_output, low_cache = mixer(inputs, mask_info=None, cache_view=cache, cache_metadata=metadata)
    mixer.dt_bias.value = jnp.full_like(mixer.dt_bias.value, 30.0)
    high_output, high_cache = mixer(inputs, mask_info=None, cache_view=cache, cache_metadata=metadata)
    np.testing.assert_allclose(low_output, high_output, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(low_cache.recurrent_state, high_cache.recurrent_state, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(low_cache.conv_state, high_cache.conv_state)
