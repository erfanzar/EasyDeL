# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Public vanilla-attention precision contracts, including the segment-only path.

Float32 storage alone does not select full-precision TPU matrix multiplication.
These multirow inputs exercise both contractions through the real registry. CPU
runs still check outputs, masks, and gradients, but only TPU runs establish that
DEFAULT and HIGHEST are numerically distinguishable on this workload.
"""

import jax
import numpy as np
import pytest
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.layers.attention import FlexibleAttentionModule
from ejkernel.types import MaskInfo
from jax import numpy as jnp


def _make_case(mask_kind):
    # Use the owning config's normal six-axis mesh and partition policies. Four
    # examples per device also keep this valid on the fake eight-device CPU mesh.
    batch = 4 * jax.device_count()
    rng = np.random.default_rng(42)
    query = rng.normal(size=(batch, 32, 2, 64)).astype(np.float32)
    key = rng.normal(size=(batch, 64, 2, 64)).astype(np.float32)
    value = rng.normal(size=(batch, 64, 2, 64)).astype(np.float32)
    q_segments = np.broadcast_to(np.arange(32, dtype=np.int32) // 16, (batch, 32))
    kv_segments = np.broadcast_to(np.arange(64, dtype=np.int32) // 32, (batch, 64))
    allowed = q_segments[:, None, :, None] == kv_segments[:, None, None, :]
    if mask_kind == "dense":
        mask_info = MaskInfo.from_attention_mask(jnp.asarray(allowed))
    else:
        mask_info = MaskInfo.from_segments(
            q_segment_ids=jnp.asarray(q_segments),
            kv_segment_ids=jnp.asarray(kv_segments),
        )
    config = EasyDeLBaseConfig(
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
    )
    module = FlexibleAttentionModule(config, softmax_scale=0.125, requires_cache=False)
    return module, mask_info, (query, key, value), allowed


def _public_forward(module, mask_info, precision):
    def forward(query, key, value):
        result = module(
            query_states=query,
            key_states=key,
            value_states=value,
            mode=None,
            mask_info=mask_info,
            causal=False,
            output_attentions=True,
            precision=precision,
        )
        return result.attention_outputs, result.attention_weights

    return forward


def _numpy_reference(inputs, allowed, cotangent=None):
    """Independent float64 attention and analytical Q/K/V vector-Jacobian product."""
    query, key, value = (np.transpose(x.astype(np.float64), (0, 2, 1, 3)) for x in inputs)
    logits = (query @ np.swapaxes(key, -1, -2)) * 0.125
    logits = np.where(allowed, logits, -np.inf)
    unnormalized = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights = unnormalized / unnormalized.sum(axis=-1, keepdims=True)
    outputs = np.transpose(weights @ value, (0, 2, 1, 3))
    if cotangent is None:
        return outputs, weights

    dout = np.transpose(cotangent.astype(np.float64), (0, 2, 1, 3))
    dvalue = np.swapaxes(weights, -1, -2) @ dout
    dweights = dout @ np.swapaxes(value, -1, -2)
    dlogits = weights * (dweights - (weights * dweights).sum(axis=-1, keepdims=True))
    dquery = (dlogits @ key) * 0.125
    dkey = (np.swapaxes(dlogits, -1, -2) @ query) * 0.125
    return tuple(np.transpose(x, (0, 2, 1, 3)) for x in (dquery, dkey, dvalue))


def _jax_reference(inputs, allowed, precision):
    """Independent batched-matmul reference for hardware DEFAULT arithmetic."""
    query, key, value = (jnp.transpose(x, (0, 2, 1, 3)) for x in inputs)
    logits = jnp.matmul(query * 0.125, jnp.swapaxes(key, -1, -2), precision=precision)
    logits = jnp.where(allowed, logits, -jnp.inf)
    weights = jax.nn.softmax(logits, axis=-1)
    outputs = jnp.matmul(weights, value, precision=precision)
    return jnp.transpose(outputs, (0, 2, 1, 3)), weights


def _assert_outputs(actual, expected):
    for result, reference in zip(actual, expected, strict=True):
        assert result is not None
        assert result.shape == reference.shape
        assert result.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(result), np.asarray(reference), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("mask_kind", ["dense", "segments"])
def test_vanilla_explicit_highest_matches_numpy_outputs_and_weights(mask_kind):
    module, mask_info, inputs, allowed = _make_case(mask_kind)
    expected = _numpy_reference(inputs, allowed)
    # Deliberately use a low ambient setting: explicit HIGHEST must win in
    # BOTH QK^T and probabilities @ V, not just in projection layers upstream.
    with jax.default_matmul_precision("bfloat16"):
        forward = jax.jit(_public_forward(module, mask_info, jax.lax.Precision.HIGHEST))
        actual = forward(*(jnp.asarray(x) for x in inputs))
        _assert_outputs(actual, expected)
    weights = np.asarray(actual[1])
    np.testing.assert_array_equal(np.where(allowed, 0.0, weights), np.zeros_like(weights))
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("mask_kind", ["dense", "segments"])
def test_vanilla_none_uses_ambient_precision_but_explicit_default_does_not(mask_kind):
    module, mask_info, inputs, allowed = _make_case(mask_kind)
    arrays = tuple(jnp.asarray(x) for x in inputs)
    # Reuse the same public callable across scopes to cover the ambient setting
    # as part of compilation, including the nested ejkernel dense-attention jit.
    ambient_forward = jax.jit(_public_forward(module, mask_info, None))
    with jax.default_matmul_precision("bfloat16"):
        ambient_low = ambient_forward(*arrays)
        jax.block_until_ready(ambient_low)
    with jax.default_matmul_precision("float32"):
        ambient_high = ambient_forward(*arrays)
        explicit_default = jax.jit(_public_forward(module, mask_info, jax.lax.Precision.DEFAULT))(*arrays)
        default_reference = jax.jit(
            lambda q, k, v: _jax_reference((q, k, v), jnp.asarray(allowed), jax.lax.Precision.DEFAULT)
        )(*arrays)
        _assert_outputs(ambient_high, _numpy_reference(inputs, allowed))
        _assert_outputs(explicit_default, default_reference)
        _assert_outputs(ambient_low, default_reference)

    if jax.default_backend() == "tpu":
        # Only TPU separates DEFAULT from HIGHEST here. Check the weights as
        # well as the output to catch QK precision.
        for high, low in zip(ambient_high, explicit_default, strict=True):
            assert np.max(np.abs(np.asarray(high) - np.asarray(low))) > 2e-5


@pytest.mark.parametrize("mask_kind", ["dense", "segments"])
def test_vanilla_explicit_highest_gradients_match_numpy_vjp(mask_kind):
    module, mask_info, inputs, allowed = _make_case(mask_kind)
    cotangent = np.random.default_rng(43).normal(size=inputs[0].shape).astype(np.float32)
    expected = _numpy_reference(inputs, allowed, cotangent=cotangent)
    forward = _public_forward(module, mask_info, jax.lax.Precision.HIGHEST)

    def loss(query, key, value, dout):
        outputs, _ = forward(query, key, value)
        return jnp.sum(outputs * dout)

    with jax.default_matmul_precision("bfloat16"):
        actual = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(*(jnp.asarray(x) for x in inputs), jnp.asarray(cotangent))
        for gradient, reference in zip(actual, expected, strict=True):
            assert gradient.shape == reference.shape
            assert gradient.dtype == jnp.float32
            np.testing.assert_allclose(np.asarray(gradient), reference, rtol=3e-5, atol=3e-6)
