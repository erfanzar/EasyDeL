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

"""Packed depthwise convs shift in the input dtype and widen per tap, numerically identical to widening first."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.infra.sequence_packing import (
    _segmented_depthwise_causal_conv1d_scan,
    segmented_depthwise_causal_conv1d,
)
from easydel.modules.qwen3_next.modeling_qwen3_next import Qwen3NextLinearAttention
from easydel.modules.qwen4_exp.modeling_qwen4_exp import _packed_depthwise_causal_conv

BATCH, SEQ, CHANNELS, WIDTH = 2, 16, 8, 4
SEGMENTS = np.array([[0] * 5 + [1] * 6 + [2] * 5, [0] * 9 + [1] * 4 + [-1] * 3], dtype=np.int32)


def _inputs(dtype, seed=0):
    key_x, key_w = jax.random.split(jax.random.key(seed))
    x = jax.random.normal(key_x, (BATCH, SEQ, CHANNELS), jnp.float32).astype(dtype)
    kernel = jax.random.normal(key_w, (WIDTH, 1, CHANNELS), jnp.float32).astype(dtype)
    return x, kernel, jnp.asarray(SEGMENTS)


def _qwen3_next_widen_first(x, kernel, seg):
    """Previous formulation: widen the whole stream to fp32 before the shifts."""
    w = kernel[:, 0, :].astype(jnp.float32)
    xf = x.astype(jnp.float32)
    b, s, c = xf.shape
    acc = jnp.zeros_like(xf)
    for j in range(WIDTH):
        shift = WIDTH - 1 - j
        if shift == 0:
            shifted, same = xf, jnp.ones((b, s), jnp.bool_)
        else:
            shifted = jnp.concatenate([jnp.zeros((b, shift, c), xf.dtype), xf[:, :-shift]], axis=1)
            src = jnp.concatenate([jnp.full((b, shift), -2, seg.dtype), seg[:, :-shift]], axis=1)
            same = src == seg
        acc = acc + jnp.where(same[..., None], shifted * w[j][None, None, :], 0.0)
    return acc.astype(x.dtype)


def _qwen4_widen_first(x, kernel, seg, dilation):
    """Previous formulation: fp32 accumulator, fp32 taps."""
    b, s, c = x.shape
    pos = jnp.arange(s, dtype=jnp.int32)
    out = jnp.zeros((b, s, c), jnp.float32)
    for k in range(kernel.shape[0]):
        lag = dilation * (kernel.shape[0] - 1 - k)
        src = pos - lag
        gather = jnp.clip(src, 0, s - 1)[None, :, None]
        shifted = jnp.take_along_axis(x, jnp.broadcast_to(gather, x.shape), axis=1)
        src_seg = jnp.take_along_axis(seg, jnp.broadcast_to(gather[..., 0], seg.shape), axis=1)
        valid = (src[None, :] >= 0) & (seg >= 0) & (src_seg == seg)
        out = out + jnp.where(valid[..., None], shifted.astype(jnp.float32), 0.0) * kernel[k, 0].astype(jnp.float32)
    return out


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_qwen3_next_segment_conv_matches_widen_first(dtype):
    """Same values and output dtype as the fp32-copy formulation, for fp32 and bf16 streams."""
    x, kernel, seg = _inputs(dtype)
    layer = SimpleNamespace(conv1d=SimpleNamespace(weight=SimpleNamespace(value=kernel)))
    got = Qwen3NextLinearAttention._segment_aware_depthwise_conv(layer, x, seg)
    want = _qwen3_next_widen_first(x, kernel, seg)
    assert got.dtype == x.dtype
    np.testing.assert_array_equal(np.asarray(got.astype(jnp.float32)), np.asarray(want.astype(jnp.float32)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_qwen4_exp_packed_conv_matches_widen_first_and_keeps_fp32_output(dtype):
    """Callers still receive an fp32 accumulator for bf16 streams, with identical values."""
    x, kernel, seg = _inputs(dtype)
    kernel = kernel.astype(jnp.float32)
    got = _packed_depthwise_causal_conv(x, kernel, seg, dilation=1)
    want = _qwen4_widen_first(x, kernel, seg, dilation=1)
    assert got.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_segmented_conv_matches_serial_reference(dtype):
    """The closed form (now shifting in the stream dtype) still tracks the token-serial reference."""
    x, kernel, seg = _inputs(dtype, seed=1)
    taps = kernel[:, 0, :].T  # [channels, kernel_size]
    got, got_state = segmented_depthwise_causal_conv1d(x, taps, seg)
    want, want_state = _segmented_depthwise_causal_conv1d_scan(x, taps, seg)
    tol = 1e-5 if dtype == jnp.float32 else 5e-2
    np.testing.assert_allclose(
        np.asarray(got.astype(jnp.float32)), np.asarray(want.astype(jnp.float32)), rtol=tol, atol=tol
    )
    np.testing.assert_array_equal(np.asarray(got_state), np.asarray(want_state))
    assert got.dtype == want.dtype
