"""Shared Omni MLPs retain submodel widths, projection names, and gated outputs."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavMLP,
    Qwen3OmniMoeTalkerCodePredictorMLP,
    Qwen3OmniMoeTalkerTextMLP,
    Qwen3OmniMoeTextMLP,
)
from easydel.modules.qwen3_omni_moe.qwen3_omni_moe_configuration import (
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeTalkerTextConfig,
    Qwen3OmniMoeTextConfig,
)
from jax import numpy as jnp


@pytest.fixture(autouse=True)
def _highest_matmul_precision():
    # The shared fused composition inherits JAX's ambient dot precision.
    with jax.default_matmul_precision("highest"):
        yield


@pytest.mark.parametrize(
    ("module_class", "config_class", "width_override"),
    [
        (Qwen3OmniMoeTextMLP, Qwen3OmniMoeTextConfig, None),
        (Qwen3OmniMoeTalkerCodePredictorMLP, Qwen3OmniMoeTalkerCodePredictorConfig, None),
        (Qwen3OmniMoeTalkerTextMLP, Qwen3OmniMoeTalkerTextConfig, 24),
        (Qwen3OmniMoeCode2WavMLP, Qwen3OmniMoeCode2WavConfig, None),
    ],
)
def test_submodel_mlp_projection_layout_and_independent_gated_reference(module_class, config_class, width_override):
    config = config_class(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_act="silu",
        attn_mechanism="vanilla",
    )
    kwargs = {} if width_override is None else {"intermediate_size": width_override}
    model = module_class(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision="highest", rngs=spx.Rngs(0), **kwargs
    )
    width = width_override or config.intermediate_size
    assert model.gate_up_proj.weight.value.shape == (16, 2 * width)
    assert model.down_proj.weight.value.shape == (width, 16)
    assert model.gate_up_proj.weight.value.dtype == jnp.float32
    assert model.down_proj.weight.value.dtype == jnp.float32
    random = np.random.default_rng(5)
    gate_up = (random.normal(size=(16, 2 * width)) * 0.1).astype(np.float32)
    down = (random.normal(size=(width, 16)) * 0.1).astype(np.float32)
    model.gate_up_proj.weight.value = jnp.asarray(gate_up)
    model.down_proj.weight.value = jnp.asarray(down)
    inputs = random.normal(size=(2, 3, 16)).astype(np.float32)
    gate, up = np.split(inputs @ gate_up, 2, axis=-1)
    expected = ((gate / (1 + np.exp(-gate))) * up) @ down
    actual = model(jnp.asarray(inputs))
    assert actual.shape == inputs.shape
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-5)
