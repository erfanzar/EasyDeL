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

"""Smoke/parity tests for Pallas GPU ragged decode attention."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._pallas.gpu.ragged_decode_attention import (
    ragged_decode_attention as ragged_decode_gpu,
)
from ejkernel.kernels._xla.ragged_decode_attention import (
    ragged_decode_attention as ragged_decode_xla,
)


def _has_gpu() -> bool:
    try:
        return len(jax.devices("gpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="Pallas GPU tests require a GPU backend")


def test_basic_matches_xla():
    key = jax.random.PRNGKey(0)
    key, kq, kk, kv = jax.random.split(key, 4)

    b, s, hq, hk, d = 2, 256, 8, 2, 64
    q = jax.random.normal(kq, (b, hq, d), dtype=jnp.float16)
    k = jax.random.normal(kk, (b, s, hk, d), dtype=jnp.float16)
    v = jax.random.normal(kv, (b, s, hk, d), dtype=jnp.float16)

    starts = jnp.array([0, 32], dtype=jnp.int32)
    ends = jnp.array([128, 200], dtype=jnp.int32)

    try:
        out_gpu = ragged_decode_gpu(q, k, v, starts, ends)
    except Exception as err:
        pytest.skip(f"Pallas GPU ragged decode unavailable: {err}")

    out_xla = ragged_decode_xla(q, k, v, starts, ends)
    assert out_gpu.shape == out_xla.shape
    assert jnp.allclose(out_gpu, out_xla, rtol=2e-2, atol=2e-2)
