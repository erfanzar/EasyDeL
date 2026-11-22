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

"""CPU parity for the XLA packed-int4 gemv references.

The references must invert the exact packing layouts the Pallas kernels
consume (``pack_int4_split_k`` / ``pack_int4_adjacent``, pure jnp, importable
anywhere), so parity here is against an independent numpy dequantization of
the *original* codes — not against the unpack helpers, which would be
circular.
"""

from __future__ import annotations

import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.quantized_matmul._packed_gemv import pack_int4_adjacent, pack_int4_split_k
from ejkernel.kernels._xla.quantized_matmul import packed_int4_gemv_xla, w4a4_gemv_xla
from jax import numpy as jnp


def _random_codes(rng, k, n):
    return rng.integers(-8, 8, size=(k, n)).astype(np.float32)


@pytest.mark.parametrize(("m", "k", "n"), [(1, 256, 512), (8, 512, 1024)])
def test_packed_int4_gemv_xla_matches_numpy_dequant(m, k, n):
    rng = np.random.default_rng(0)
    codes = _random_codes(rng, k, n)
    scale = (rng.uniform(0.5, 2.0, size=(1, n))).astype(np.float32)
    x = rng.normal(size=(m, k)).astype(np.float32)

    packed = pack_int4_split_k(jnp.asarray(codes, jnp.int8))
    got = np.asarray(packed_int4_gemv_xla(jnp.asarray(x, jnp.bfloat16), packed, jnp.asarray(scale)), np.float32)
    want = x @ (codes * scale)
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2 * np.abs(want).max())


@pytest.mark.parametrize(("m", "k", "n"), [(1, 256, 512), (8, 512, 1024)])
def test_w4a4_gemv_xla_matches_numpy_integer_dot(m, k, n):
    rng = np.random.default_rng(1)
    codes = _random_codes(rng, k, n)
    scale = (rng.uniform(0.5, 2.0, size=(1, n))).astype(np.float32)
    x_codes = rng.integers(-7, 8, size=(m, k)).astype(np.float32)

    packed = pack_int4_adjacent(jnp.asarray(codes, jnp.int8))
    got = np.asarray(
        w4a4_gemv_xla(jnp.asarray(x_codes, jnp.int4), packed, jnp.asarray(scale)),
        np.float32,
    )
    # Integer dot is exact; only the final bf16 cast rounds.
    want = (x_codes @ codes) * scale
    np.testing.assert_allclose(got, want, rtol=8e-3, atol=8e-3 * np.abs(want).max())


def test_w4a4_gemv_xla_rejects_float_activations():
    rng = np.random.default_rng(2)
    packed = pack_int4_adjacent(jnp.asarray(_random_codes(rng, 128, 256), jnp.int8))
    with pytest.raises(ValueError, match="int4"):
        w4a4_gemv_xla(jnp.ones((4, 128), jnp.bfloat16), packed, jnp.ones((1, 256)))


@pytest.mark.parametrize("in_dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
def test_packed_int4_gemv_xla_output_is_bf16_like_pallas(in_dtype):
    """Same op id, same output dtype: the Pallas kernel casts activations to
    bf16 and emits bf16 regardless of input dtype, so the XLA reference must
    not follow the input dtype."""
    rng = np.random.default_rng(3)
    packed = pack_int4_split_k(jnp.asarray(_random_codes(rng, 128, 256), jnp.int8))
    out = packed_int4_gemv_xla(jnp.ones((2, 128), in_dtype), packed, jnp.ones((1, 256)))
    assert out.dtype == jnp.bfloat16
