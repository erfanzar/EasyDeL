# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Model mHC adapters preserve equations, initialization, and checkpoint paths."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.glm5_next.glm5_next_configuration import Glm5NextTextConfig
from easydel.modules.glm5_next.modeling_glm5_next import Glm5NextHyperConnection, _hc_head_collapse
from jax import numpy as jnp


def _glm_config():
    # Deliberately nondefault settings catch dropped or swapped adapter fields.
    return Glm5NextTextConfig(
        hidden_size=5,
        hc_mult=3,
        hc_eps=2e-4,
        rms_norm_eps=3e-3,
        hc_sinkhorn_iters=4,
        initializer_range=0.07,
        num_hidden_layers=1,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )


def _reference_connection(x, fn, base, scale, config):
    """Independent NumPy equations, without shared mHC numerical helpers."""
    xf = np.asarray(x, np.float32)
    fn, base, scale = (np.asarray(value, np.float32) for value in (fn, base, scale))
    flat = xf.reshape(*xf.shape[:2], -1)
    flat = flat / np.sqrt(np.mean(flat * flat, axis=-1, keepdims=True) + config.rms_norm_eps)
    logits = np.stack([np.sum(flat * row, axis=-1) for row in fn], axis=-1)
    hc = config.hc_mult
    read = 1 / (1 + np.exp(-(logits[..., :hc] * scale[0] + base[:hc]))) + config.hc_eps
    post = 2 / (1 + np.exp(-(logits[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])))
    scores = (logits[..., 2 * hc :] * scale[2] + base[2 * hc :]).reshape(*xf.shape[:2], hc, hc)
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    comb = scores / scores.sum(axis=-1, keepdims=True) + config.hc_eps
    for axis in [-2] + [-1, -2] * (config.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(axis=axis, keepdims=True) + config.hc_eps)
    collapsed = sum(read[..., i, None] * xf[..., i, :] for i in range(hc))
    return post, comb, collapsed.astype(x.dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_glm_mhc_adapter_forward_and_mean_head(dtype):
    config = _glm_config()
    module = Glm5NextHyperConnection(
        config, dtype=dtype, param_dtype=dtype, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(17)
    )
    rng = np.random.default_rng(23)
    x = jnp.asarray(rng.normal(size=(2, 4, config.hc_mult, config.hidden_size)), dtype)
    for parameter in (module.fn, module.base, module.scale):
        parameter.value = jnp.asarray(rng.normal(size=parameter.value.shape) * 0.4 + 0.1, dtype)
    expected = _reference_connection(x, module.fn.value, module.base.value, module.scale.value, config)
    for outputs in (module(x), jax.jit(module)(x)):
        assert [output.dtype for output in outputs] == [jnp.float32, jnp.float32, dtype]
        for got, want in zip(outputs, expected, strict=True):
            tolerance = 8e-3 if got.dtype == jnp.bfloat16 else 3e-6
            np.testing.assert_allclose(
                np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=tolerance, atol=tolerance
            )
    mean = _hc_head_collapse(x)
    assert mean.dtype == dtype
    np.testing.assert_allclose(
        np.asarray(mean, np.float32),
        np.asarray(np.asarray(x, np.float32).mean(axis=2).astype(x.dtype), np.float32),
        rtol=2e-7,
        atol=2e-7,
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_glm_mhc_adapter_checkpoint_layout_rng_and_rebind(dtype):
    config = _glm_config()
    rngs = spx.Rngs(47)
    module = Glm5NextHyperConnection(config, dtype=dtype, param_dtype=dtype, rngs=rngs)
    graphdef, state = spx.export(module)
    leaves = state.flatten()
    hc = config.hc_mult
    mix = (2 + hc) * hc
    shapes = {"fn": (mix, hc * config.hidden_size), "base": (mix,), "scale": (3,)}
    assert set(leaves) == {f"parameters/{name}" for name in shapes}
    for name, shape in shapes.items():
        assert leaves[f"parameters/{name}"].shape == shape
        assert leaves[f"parameters/{name}"].dtype == dtype

    # Reference the original direct parameter initialization, not the shared
    # layer constructor, to detect key-order or nested checkpoint changes.
    expected_rngs = spx.Rngs(47)
    expected_fn = jax.nn.initializers.normal(config.initializer_range)(expected_rngs.param, shapes["fn"], dtype)
    np.testing.assert_array_equal(leaves["parameters/fn"], expected_fn)
    np.testing.assert_array_equal(leaves["parameters/base"], jnp.zeros(shapes["base"], dtype))
    np.testing.assert_array_equal(leaves["parameters/scale"], jnp.ones(shapes["scale"], dtype))
    _ = expected_rngs.param  # zero initializer still consumes a key
    _ = expected_rngs.param  # one initializer still consumes a key
    np.testing.assert_array_equal(jax.random.key_data(rngs.param), jax.random.key_data(expected_rngs.param))

    restored = spx.bind(graphdef, state)
    x = jnp.asarray(np.random.default_rng(5).normal(size=(2, 4, hc, config.hidden_size)), dtype)
    for got, want in zip(restored(x), module(x), strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_deepseek_mhc_adapter_and_learned_head(dtype):
    from easydel.modules.deepseek_v4.deepseek_v4_configuration import DeepseekV4Config
    from easydel.modules.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection, DeepseekV4HyperHead

    config = DeepseekV4Config(
        hidden_size=5,
        hc_mult=3,
        hc_eps=2e-4,
        rms_norm_eps=3e-3,
        hc_sinkhorn_iters=4,
        initializer_range=0.07,
        num_hidden_layers=1,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    # Match the full-fp32 NumPy reference, as in the GLM adapter test above.
    connection = DeepseekV4HyperConnection(
        config, dtype=dtype, param_dtype=dtype, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(17)
    )
    head = DeepseekV4HyperHead(
        config, dtype=dtype, param_dtype=dtype, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(29)
    )
    rng = np.random.default_rng(23)
    x = jnp.asarray(rng.normal(size=(2, 4, config.hc_mult, config.hidden_size)), dtype)
    for parameter in (connection.fn, connection.base, connection.scale, head.hc_fn, head.hc_base, head.hc_scale):
        parameter.value = jnp.asarray(rng.normal(size=parameter.value.shape) * 0.4 + 0.1, dtype)
    expected = _reference_connection(x, connection.fn.value, connection.base.value, connection.scale.value, config)
    for outputs in (connection(x), jax.jit(connection)(x)):
        for got, want in zip(outputs, expected, strict=True):
            tolerance = 8e-3 if got.dtype == jnp.bfloat16 else 3e-6
            np.testing.assert_allclose(
                np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=tolerance, atol=tolerance
            )
    xf = np.asarray(x, np.float32)
    flat = xf.reshape(2, 4, -1)
    flat = flat / np.sqrt(np.mean(flat**2, axis=-1, keepdims=True) + config.rms_norm_eps)
    logits = flat @ np.asarray(head.hc_fn.value, np.float32).T
    logits = logits * np.asarray(head.hc_scale.value, np.float32) + np.asarray(head.hc_base.value, np.float32)
    gates = 1 / (1 + np.exp(-logits)) + config.hc_eps
    expected_head = np.sum(gates[..., None] * np.asarray(x, np.float32), axis=2).astype(x.dtype)
    for got in (head(x), jax.jit(head)(x)):
        tolerance = 8e-3 if dtype == jnp.bfloat16 else 3e-6
        assert got.dtype == dtype
        np.testing.assert_allclose(
            np.asarray(got, np.float32), np.asarray(expected_head, np.float32), rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("head", [False, True])
def test_deepseek_mhc_direct_parameter_rng_and_rebind(head):
    from easydel.modules.deepseek_v4.deepseek_v4_configuration import DeepseekV4Config
    from easydel.modules.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection, DeepseekV4HyperHead

    config = DeepseekV4Config(
        hidden_size=5, hc_mult=3, initializer_range=0.07, num_hidden_layers=1, sharding_axis_dims=(1, 1, 1, 1, 1, 1)
    )
    rngs = spx.Rngs(47)
    cls = DeepseekV4HyperHead if head else DeepseekV4HyperConnection
    module = cls(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
    graphdef, state = spx.export(module)
    shapes = (
        {"hc_fn": (3, 15), "hc_base": (3,), "hc_scale": (1,)} if head else {"fn": (15, 15), "base": (15,), "scale": (3,)}
    )
    leaves = state.flatten()
    assert set(leaves) == {f"parameters/{name}" for name in shapes}
    expected_rngs = spx.Rngs(47)
    for (name, shape), initializer in zip(
        shapes.items(),
        (jax.nn.initializers.normal(0.07), jax.nn.initializers.zeros, jax.nn.initializers.ones),
        strict=True,
    ):
        expected = initializer(expected_rngs.param, shape, jnp.float32)
        np.testing.assert_array_equal(leaves[f"parameters/{name}"], expected)
    np.testing.assert_array_equal(jax.random.key_data(rngs.param), jax.random.key_data(expected_rngs.param))
    x = jnp.asarray(np.random.default_rng(5).normal(size=(2, 4, 3, 5)), jnp.float32)
    restored = spx.bind(graphdef, state)
    if head:
        np.testing.assert_array_equal(restored(x), module(x))
    else:
        for got, want in zip(restored(x), module(x), strict=True):
            np.testing.assert_array_equal(got, want)


def _fused_adapter(family, fused, dtype):
    from easydel.modules.deepseek_v4.deepseek_v4_configuration import DeepseekV4Config
    from easydel.modules.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection

    config_type, module_type = {
        "glm": (Glm5NextTextConfig, Glm5NextHyperConnection),
        "deepseek": (DeepseekV4Config, DeepseekV4HyperConnection),
    }[family]
    # Four streams reach the packed TPU coefficients kernel when fused.
    config = config_type(
        hidden_size=5,
        hc_mult=4,
        hc_eps=2e-4,
        rms_norm_eps=3e-3,
        hc_sinkhorn_iters=4,
        initializer_range=0.07,
        num_hidden_layers=1,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        hc_use_fused_coefficients=fused,
    )
    module = module_type(config, dtype=dtype, param_dtype=dtype, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(17))
    return config, module


@pytest.mark.parametrize("family", ["glm", "deepseek"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_fused_coefficients_flag_reaches_adapter_and_preserves_math_and_leaves(family, dtype):
    layouts = {}
    for fused in (False, True):
        config, module = _fused_adapter(family, fused, dtype)
        assert module.use_fused_coefficients is fused
        _, state = spx.export(module)
        layouts[fused] = {key: (value.shape, value.dtype) for key, value in state.flatten().items()}
        rng = np.random.default_rng(23)
        x = jnp.asarray(rng.normal(size=(2, 4, config.hc_mult, config.hidden_size)), dtype)
        for parameter in (module.fn, module.base, module.scale):
            parameter.value = jnp.asarray(rng.normal(size=parameter.value.shape) * 0.4 + 0.1, dtype)
        expected = _reference_connection(x, module.fn.value, module.base.value, module.scale.value, config)
        for got, want in zip(jax.jit(module)(x), expected, strict=True):
            tolerance = 8e-3 if got.dtype == jnp.bfloat16 else 3e-6
            np.testing.assert_allclose(
                np.asarray(got, np.float32), np.asarray(want, np.float32), rtol=tolerance, atol=tolerance
            )
    assert layouts[True] == layouts[False]


def test_fused_coefficients_flag_defaults_off():
    from easydel.modules.deepseek_v4.deepseek_v4_configuration import DeepseekV4Config

    assert Glm5NextTextConfig(hidden_size=5).hc_use_fused_coefficients is False
    assert DeepseekV4Config(hidden_size=5).hc_use_fused_coefficients is False


@pytest.mark.parametrize("fused", [False, True])
def test_selective_remat_can_retain_mhc_logits_and_coefficients(fused):
    """Retention names must keep the small tensors instead of replaying Sinkhorn."""
    from easydel.infra.etils import GRADIENT_CHECKPOINT_TARGETS
    from easydel.infra.utils import get_gradient_checkpoint_policy
    from jax._src.ad_checkpoint import saved_residuals

    assert {"mhc_logits", "mhc_coefficients", "indexer_topk"} <= set(GRADIENT_CHECKPOINT_TARGETS)
    config, module = _fused_adapter("glm", fused, jnp.float32)
    graphdef, state = spx.export(module)
    x = jnp.asarray(np.random.default_rng(3).normal(size=(2, 8, config.hc_mult, config.hidden_size)), jnp.float32)

    def connection(x):
        post, comb, collapsed = spx.bind(graphdef, state)(x)
        return jnp.sum(post) + jnp.sum(comb) + jnp.sum(collapsed)

    def named(policy):
        residuals = saved_residuals(jax.checkpoint(connection, policy=policy), x)
        return sorted(tuple(aval.shape) for aval, source in residuals if "named 'mhc_" in source)

    retain = get_gradient_checkpoint_policy("save_only_these_names", save_names=["mhc_logits", "mhc_coefficients"])
    hc = config.hc_mult
    # Backward needs the coefficient inputs (logits) and the read gates for the
    # collapsed stream; write gates and mixer are outputs consumed downstream.
    assert named(retain) == sorted([(2, 8, hc * (hc + 2)), (2, 8, hc)])
    assert named(get_gradient_checkpoint_policy("nothing_saveable")) == []
