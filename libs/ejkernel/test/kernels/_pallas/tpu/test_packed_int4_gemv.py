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

"""On-device parity for the packed-int4 decode kernels.

TPU-required: the W4A16 kernel's Mosaic lowering (int32 nibble decode) and
the W4A4 kernel's bitcast-into-int4-MXU feed both exist only on TPU — the
CPU backend rejects int4 dot_generals outright, so interpret mode cannot
stand in for the W4A4 path.

References are dense float32 products of the *dequantized* codes, computed
independently of the kernels.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.quantized_matmul._packed_gemv import (
    pack_int4_adjacent,
    pack_int4_split_k,
    packed_int4_gemv,
    w4a4_gemv,
)

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "tpu",
    reason="packed gemv kernels lower through Mosaic; int4 MXU feed requires TPU",
)


def _quantize(rng, k_dim, n_dim):
    """Per-channel symmetric int4 quantization of a gaussian weight."""
    weight = rng.normal(size=(k_dim, n_dim)).astype(np.float32)
    scale = np.abs(weight).max(axis=0, keepdims=True) / 7.0
    codes = np.clip(np.round(weight / scale), -7, 7).astype(np.int32)
    return codes, scale


def test_w4a16_matches_dense_reference():
    """Split-K packed streaming reproduces the dequantized product."""
    rng = np.random.default_rng(0)
    k_dim, n_dim, tokens = 2048, 2048, 8
    codes, scale = _quantize(rng, k_dim, n_dim)
    x = jnp.asarray(rng.normal(size=(tokens, k_dim)), dtype=jnp.bfloat16)

    got = np.asarray(
        packed_int4_gemv(x, pack_int4_split_k(jnp.asarray(codes)), jnp.asarray(scale), tile_n=512, chunk=256),
        dtype=np.float32,
    )
    ref = np.asarray(x, dtype=np.float32) @ (codes.astype(np.float32) * scale)
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 0.02, f"W4A16 packed gemv diverged: relerr={rel}"


def test_w4a4_matches_quantized_reference():
    """The bitcast int4-MXU path is exact against its quantized semantics."""
    rng = np.random.default_rng(1)
    k_dim, n_dim, tokens = 2048, 2048, 8
    codes, scale = _quantize(rng, k_dim, n_dim)

    x_float = rng.normal(size=(tokens, k_dim)).astype(np.float32)
    x_scale = np.abs(x_float).max(axis=1, keepdims=True) / 7.0
    x_codes = np.clip(np.round(x_float / x_scale), -7, 7)
    x4 = jnp.asarray(x_codes, dtype=jnp.int4)

    got = (
        np.asarray(
            w4a4_gemv(x4, pack_int4_adjacent(jnp.asarray(codes)), jnp.asarray(scale), tile_n=512),
            dtype=np.float32,
        )
        * x_scale
    )
    ref = (x_codes * x_scale) @ (codes.astype(np.float32) * scale)
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 0.01, f"W4A4 gemv diverged from its quantized reference: relerr={rel}"


def test_w4a4_rejects_unquantized_activations():
    """Float activations mean the caller wanted W4A16 — fail loudly."""
    x = jnp.ones((8, 256), jnp.bfloat16)
    packed = jnp.zeros((128, 256), jnp.uint8)
    with pytest.raises(ValueError, match="int4 activations"):
        w4a4_gemv(x, packed, jnp.ones((1, 256)))
