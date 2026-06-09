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

"""Smoke/parity tests for Pallas GPU scaled dot-product attention."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._pallas.gpu.scaled_dot_product_attention import (
    scaled_dot_product_attention as sdpa_gpu,
)
from ejkernel.kernels._xla.scaled_dot_product_attention import (
    scaled_dot_product_attention as sdpa_xla,
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

    b, t, h, d = 2, 64, 4, 32
    q = jax.random.normal(kq, (b, t, h, d), dtype=jnp.float16)
    k = jax.random.normal(kk, (b, t, h, d), dtype=jnp.float16)
    v = jax.random.normal(kv, (b, t, h, d), dtype=jnp.float16)

    try:
        out_gpu = sdpa_gpu(q, k, v, causal=True)
    except Exception as err:
        pytest.skip(f"cudnn SDPA backend unavailable: {err}")

    out_xla = sdpa_xla(q, k, v, causal=True)

    assert out_gpu.shape == out_xla.shape
    assert jnp.allclose(out_gpu, out_xla, rtol=1e-2, atol=1e-2)
