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

"""A8 slab grouped matmul writing the int32 paired layout, and the packed gather-sum reading it: bitwise on TPU."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.moe_row_gather import gather_sum_packed_rows, gather_sum_rows
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.kernels._xla.moe_row_gather import pack_rows_paired, unpack_rows_paired
from ejkernel.modules import grouped_matmul_channelwise, grouped_matmul_channelwise_packed_pair_width

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")
BF16, F32 = jnp.bfloat16, jnp.float32


def _bits(a):
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(a, BF16), jnp.uint16))


@pytest.mark.parametrize("width", [256, 2048, 4096])
def test_pack_unpack_roundtrip(width):
    rng = np.random.default_rng(width)
    y = jnp.asarray(rng.standard_normal((64, 4096)), F32).astype(BF16)
    back = unpack_rows_paired(pack_rows_paired(y, width), 4096, width)
    np.testing.assert_array_equal(_bits(back), _bits(y))


@pytest.mark.parametrize("skip_negative", [False, True])
@pytest.mark.parametrize("width", [2048, 4096])
def test_packed_gather_sum_equals_gather_sum(width, skip_negative):
    rng = np.random.default_rng(width + skip_negative)
    m, h, tokens, k = 4096, 4096, 512, 8
    y = np.array(jnp.asarray(rng.standard_normal((m, h)), F32).astype(BF16).astype(F32))  # writable copy
    idx = rng.integers(0, m, (tokens, k)).astype(np.int32)
    if skip_negative:
        y[m - 1] = np.nan  # only dead slots point here
        idx[rng.uniform(size=(tokens, k)) < 0.7] = -1
        idx[idx == m - 1] = 0
    y = jnp.asarray(y, F32).astype(BF16)
    w = jnp.asarray(rng.uniform(0.1, 1.0, (tokens, k)), F32)
    want = jax.jit(lambda a, i, ww: gather_sum_rows(a, i, ww, F32, skip_negative=skip_negative))(y, idx, w)
    got = jax.jit(lambda p, i, ww: gather_sum_packed_rows(p, i, ww, h, width, F32, skip_negative=skip_negative))(
        pack_rows_paired(y, width), idx, w
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert np.isfinite(np.asarray(got)).all()


@pytest.mark.parametrize("rows_per_expert", [40, 300, 512])
def test_slab_gmm_writes_the_packed_layout(rows_per_expert):
    e, k, n = 8, 512, 4096
    m = e * rows_per_expert // 32 * 32 + 64
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(rows_per_expert), 3)
    x = (jax.random.normal(k1, (m, k), F32) * 3).astype(BF16)
    codes = jax.random.randint(k2, (e, k, n), -127, 128, jnp.int32).astype(jnp.int8)
    scales = jax.random.uniform(k3, (e, 1, n), F32, 1e-3, 2e-3)
    sizes = np.random.RandomState(0).multinomial(m - 64, np.ones(e) / e).astype(np.int32)
    width = grouped_matmul_channelwise_packed_pair_width(m, codes.shape, codes.dtype, activation_bits=8)
    assert width is not None and n % width == 0
    kw = dict(activation_bits=8, platform="pallas", preferred_element_type=BF16, mask_padding=False)
    plain = jax.jit(lambda *a: grouped_matmul_channelwise(*a, **kw))(x, codes, scales, jnp.asarray(sizes))
    packed = jax.jit(lambda *a: grouped_matmul_channelwise(*a, **kw, packed_pair_width=width))(
        x, codes, scales, jnp.asarray(sizes)
    )
    assert packed.dtype == jnp.int32 and packed.shape == (m * n // 256, 128)
    hlo = (
        jax.jit(lambda *a: grouped_matmul_channelwise(*a, **kw, packed_pair_width=width))
        .lower(x, codes, scales, jnp.asarray(sizes))
        .compile()
        .as_text()
    )
    assert "moe_row_pack" not in hlo
    routed = int(sizes.sum())
    got = unpack_rows_paired(packed, n, width)
    np.testing.assert_array_equal(_bits(got)[:routed], _bits(plain)[:routed])
    # The XLA reference packs its own (masked) output: same routed bits.
    ref = grouped_matmul_channelwise(
        x, codes, scales, jnp.asarray(sizes), activation_bits=8, platform="xla", packed_pair_width=width
    )
    np.testing.assert_array_equal(_bits(unpack_rows_paired(ref, n, width))[:routed], _bits(plain)[:routed])


def test_registry_entries_share_one_signature():
    for name in ("moe_row_gather_sum_packed", "grouped_matmul_channelwise"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert kernel_registry.validate_signatures(name)
    assert kernel_registry.get("moe_row_gather_sum_packed", platform=Platform.XLA, backend=Backend.ANY) is not None
