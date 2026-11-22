"""Gated-MLP wrapper contracts and accelerator-only initialization/output parity.

Only the explicitly marked metadata/signature tests may run on CPU.
"""

import inspect
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers import ColumnParallelLinear, RowParallelLinear, dense_gate_up_layout
from easydel.layers import mlp as shared_mlp
from easydel.modules.qwen3_omni_moe import modeling_qwen3_omni_moe as omni
from easydel.modules.qwen3_omni_moe.qwen3_omni_moe_configuration import (
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeTalkerTextConfig,
    Qwen3OmniMoeTextConfig,
)
from jax import numpy as jnp


@pytest.mark.cpu_ok
def test_public_wrapper_constructor_signatures():
    """Public positional order, keyword-only options, and defaults stay stable."""
    text_params = inspect.signature(omni.Qwen3OmniMoeTextMLP.__init__).parameters
    assert list(text_params) == [
        "self",
        "config",
        "dtype",
        "param_dtype",
        "precision",
        "rngs",
        "intermediate_size",
        "kernel_init",
    ]
    for name in ("rngs", "intermediate_size", "kernel_init"):
        assert text_params[name].kind == inspect.Parameter.KEYWORD_ONLY
    talker_params = inspect.signature(omni.Qwen3OmniMoeTalkerTextMLP.__init__).parameters
    assert list(talker_params) == ["self", "config", "intermediate_size", "dtype", "param_dtype", "precision", "rngs"]
    assert talker_params["intermediate_size"].default is None
    assert list(inspect.signature(omni.Qwen3OmniMoeCode2WavMLP.__init__).parameters) == [
        "self",
        "config",
        "dtype",
        "param_dtype",
        "precision",
        "rngs",
    ]
    predictor_params = inspect.signature(omni.Qwen3OmniMoeTalkerCodePredictorMLP.__init__).parameters
    assert list(predictor_params) == list(text_params)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("variant,width", [("text", 12), ("talker", 20), ("predictor", 12), ("code2wav", 12)])
def test_projection_metadata_and_checkpoint_contract(monkeypatch, variant, width):
    """Exercise real wrapper construction/reform rules without allocating arrays."""

    class MetadataLinear(spx.Module):
        build_reform_param = ColumnParallelLinear.build_reform_param

        def __init__(
            self, in_features, out_features, *, dtype, param_dtype, precision, kernel_init, use_bias, rngs, layout=None
        ):
            self.in_features = in_features
            self.out_features = sum(out_features) if isinstance(out_features, tuple) else out_features
            self.dtype = dtype
            self.param_dtype = param_dtype
            self.precision = precision
            self.kernel_init = kernel_init
            self.use_bias = use_bias
            self.layout = layout
            self.weight = spx.Parameter(jax.ShapeDtypeStruct((in_features, self.out_features), param_dtype))

    monkeypatch.setattr(shared_mlp, "ColumnParallelLinear", MetadataLinear)
    monkeypatch.setattr(shared_mlp, "RowParallelLinear", MetadataLinear)
    monkeypatch.setattr(jax.nn.initializers, "normal", lambda std: ("normal", std))
    monkeypatch.setattr(jax.nn.initializers, "lecun_normal", lambda: ("lecun_normal",))
    config = SimpleNamespace(hidden_size=8, intermediate_size=12, hidden_act="silu", initializer_range=0.017)
    classes = {
        "text": omni.Qwen3OmniMoeTextMLP,
        "talker": omni.Qwen3OmniMoeTalkerTextMLP,
        "predictor": omni.Qwen3OmniMoeTalkerCodePredictorMLP,
        "code2wav": omni.Qwen3OmniMoeCode2WavMLP,
    }
    kwargs = {"intermediate_size": width} if variant == "talker" else {}
    model = classes[variant](
        config, dtype=jnp.bfloat16, param_dtype=jnp.float32, precision="highest", rngs=None, **kwargs
    )
    assert model.gate_up_proj.weight.value.shape == (8, 2 * width)
    assert model.down_proj.weight.value.shape == (width, 8)
    for projection in (model.gate_up_proj, model.down_proj):
        assert projection.weight.value.dtype == jnp.float32
        assert projection.dtype == jnp.bfloat16
        assert projection.precision == "highest"
        assert not projection.use_bias
        expected_init = ("lecun_normal",) if variant == "code2wav" else ("normal", 0.017)
        assert projection.kernel_init == expected_init
    _, state = spx.export(model)
    assert set(state.flatten()) == {"parameters/gate_up_proj.weight", "parameters/down_proj.weight"}
    assert set(model.reform_param) == {"gate_up_proj.weight$"}
    assert model.reform_param["gate_up_proj.weight$"]["sources"] == ("gate_proj.weight", "up_proj.weight")


@pytest.mark.parametrize(
    "cls,config_cls,width,lecun",
    [
        (omni.Qwen3OmniMoeTextMLP, Qwen3OmniMoeTextConfig, None, False),
        (omni.Qwen3OmniMoeTalkerTextMLP, Qwen3OmniMoeTalkerTextConfig, 20, False),
        (omni.Qwen3OmniMoeTalkerCodePredictorMLP, Qwen3OmniMoeTalkerCodePredictorConfig, None, False),
        (omni.Qwen3OmniMoeCode2WavMLP, Qwen3OmniMoeCode2WavConfig, None, True),
    ],
)
def test_initialization_and_numpy_forward_reference(cls, config_cls, width, lecun):
    """Independent pre-refactor projection setup checks RNG order and numerics."""
    assert jax.default_backend() in {"tpu", "gpu", "cuda", "rocm"}, "Array-computation tests require an accelerator."
    config = config_cls(hidden_size=8, intermediate_size=12, hidden_act="silu", use_fused_mlp=False)
    kwargs = {"intermediate_size": width} if width is not None else {}
    model = cls(config, dtype=jnp.float32, param_dtype=jnp.float32, precision="highest", rngs=spx.Rngs(42), **kwargs)
    width = width or config.intermediate_size
    kernel_init = jax.nn.initializers.lecun_normal() if lecun else jax.nn.initializers.normal(config.initializer_range)
    rngs = spx.Rngs(42)
    gate_up = ColumnParallelLinear(
        config.hidden_size,
        (width, width),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        use_bias=False,
        kernel_init=kernel_init,
        precision="highest",
        rngs=rngs,
        layout=dense_gate_up_layout(width),
    )
    down = RowParallelLinear(
        width,
        config.hidden_size,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        use_bias=False,
        kernel_init=kernel_init,
        precision="highest",
        rngs=rngs,
    )
    np.testing.assert_array_equal(np.asarray(model.gate_up_proj.weight.value), np.asarray(gate_up.weight.value))
    np.testing.assert_array_equal(np.asarray(model.down_proj.weight.value), np.asarray(down.weight.value))
    states = np.random.default_rng(7).normal(size=(2, 3, 8)).astype(np.float32)
    gate, up = np.split(states @ np.asarray(gate_up.weight.value), 2, axis=-1)
    expected = ((gate / (1 + np.exp(-gate))) * up) @ np.asarray(down.weight.value)
    actual = model(jnp.asarray(states))
    assert actual.shape == states.shape
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-6)
