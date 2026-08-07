# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Parity for the channelwise XLA quantized matmul against dequantized dense.

The reference is ``x @ (w_q * scale)`` computed in float32 — independent of
the implementation under test, so a broken fusion or a scale applied on the
wrong axis fails loudly rather than cancelling out.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.quantized_matmul import channelwise_quantized_matmul


def _quantize_channelwise(weight: np.ndarray, qdtype) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Symmetric per-output-channel quantization reference."""
    qmax = float(jnp.iinfo(qdtype).max)
    absmax = np.abs(weight).max(axis=0, keepdims=True)
    scale = absmax / qmax
    codes = np.clip(np.round(weight / np.where(scale == 0, 1, scale)), -qmax, qmax)
    return jnp.asarray(codes).astype(qdtype), jnp.asarray(scale, dtype=jnp.float32)


def _reference(x, w_q, scale):
    """Dense float32 reference of the quantized product."""
    return np.asarray(x, dtype=np.float32) @ (
        np.asarray(w_q, dtype=np.float32) * np.asarray(scale, dtype=np.float32)
    )


@pytest.mark.parametrize("qdtype", [jnp.int8, jnp.int4])
@pytest.mark.parametrize("tokens", [8, 512])
def test_upcast_path_matches_dense_reference(qdtype, tokens):
    """The W-A16 fused-upcast path reproduces the dequantized product."""
    rng = np.random.default_rng(0)
    weight = rng.normal(size=(256, 128)).astype(np.float32)
    w_q, scale = _quantize_channelwise(weight, qdtype)
    x = jnp.asarray(rng.normal(size=(tokens, 256)), dtype=jnp.bfloat16)

    got = np.asarray(channelwise_quantized_matmul(x, w_q, scale)).astype(np.float32)
    ref = _reference(x, w_q, scale)
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 0.01, f"upcast path diverged: relerr={rel}"


def test_prefill_int8_dot_matches_dense_reference():
    """The W8A8 integer-dot path stays within dynamic-activation error."""
    rng = np.random.default_rng(1)
    weight = rng.normal(size=(256, 128)).astype(np.float32)
    w_q, scale = _quantize_channelwise(weight, jnp.int8)
    x = jnp.asarray(rng.normal(size=(512, 256)), dtype=jnp.bfloat16)

    got = np.asarray(
        channelwise_quantized_matmul(x, w_q, scale, quantize_activations=True, prefill_threshold=256)
    ).astype(np.float32)
    ref = _reference(x, w_q, scale)
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 0.03, f"int8-dot path diverged: relerr={rel}"


def test_decode_stays_on_upcast_path_even_with_activations_enabled():
    """Below the threshold the exact upcast path is used, not the int dot."""
    rng = np.random.default_rng(2)
    weight = rng.normal(size=(256, 128)).astype(np.float32)
    w_q, scale = _quantize_channelwise(weight, jnp.int8)
    x = jnp.asarray(rng.normal(size=(8, 256)), dtype=jnp.bfloat16)

    with_flag = np.asarray(
        channelwise_quantized_matmul(x, w_q, scale, quantize_activations=True, prefill_threshold=256)
    )
    without_flag = np.asarray(channelwise_quantized_matmul(x, w_q, scale))
    np.testing.assert_array_equal(with_flag, without_flag)


def test_w4a4_requires_int4_weights():
    """4-bit activations without int4 weights is a configuration error."""
    rng = np.random.default_rng(3)
    weight = rng.normal(size=(64, 32)).astype(np.float32)
    w_q, scale = _quantize_channelwise(weight, jnp.int8)
    x = jnp.ones((512, 64), jnp.bfloat16)
    with pytest.raises(ValueError, match="requires int4"):
        channelwise_quantized_matmul(x, w_q, scale, quantize_activations=True, activation_bits=4)


def test_float_weight_rejected():
    """A float weight means the caller wanted the dense path — fail loudly."""
    x = jnp.ones((8, 64), jnp.bfloat16)
    w = jnp.ones((64, 32), jnp.bfloat16)
    with pytest.raises(ValueError, match="integer"):
        channelwise_quantized_matmul(x, w, jnp.ones((1, 32)))
