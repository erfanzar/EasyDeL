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

"""Live-slot gather-sum (unrolled offset-list issue) vs an independent float32 numpy sum."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.moe_row_gather import gather_sum_packed_rows, gather_sum_rows, pack_rows
from ejkernel.kernels._pallas.tpu.moe_row_gather._pallas_impl import gather_sum_block_tokens

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")

BF16, F32 = jnp.bfloat16, jnp.float32
H, K = 2048, 8


def _reference(y, slots, w):
    """float32 slot-order sum; a dead slot (negative index) or a zero weight adds +0.0."""
    y = np.asarray(jnp.asarray(y, F32))
    out = None
    for j in range(slots.shape[1]):
        live = (slots[:, j] >= 0) & (w[:, j] != 0.0)
        rows = y[np.clip(slots[:, j], 0, y.shape[0] - 1)]
        term = np.where(live[:, None], rows * w[:, j : j + 1], np.float32(0.0)).astype(np.float32)
        out = term if out is None else (out + term).astype(np.float32)
    return out


def _case(seed, tokens, live_per_block):
    """Slots whose live count per grid step is ``live_per_block[i]`` (exercises unroll remainders and empty steps)."""
    rng = np.random.default_rng(seed)
    block = gather_sum_block_tokens(K)
    span = block * K
    slots = np.full((tokens * K,), -1, np.int32)
    nlive = 0
    for i, n in enumerate(live_per_block):
        pos = i * span + rng.permutation(span)[:n]
        slots[pos] = np.arange(nlive, nlive + n)
        nlive += n
    rows = max(-(-nlive // 256) * 256 + 256, tokens * K)
    slots[slots >= 0] = rng.permutation(nlive)
    yv = rng.standard_normal((rows, H)).astype(np.float32)
    yv[nlive:] = np.nan  # unwritten FFN tail: only dead slots point here
    w = rng.uniform(0.0, 1.0, (tokens, K)).astype(np.float32)
    w[rng.random((tokens, K)) < 0.05] = 0.0
    w[rng.random((tokens, K)) < 0.05] = -0.0
    slots = slots.reshape(tokens, K)
    w[slots < 0] = np.nan  # a dead slot's weight may be anything
    return jnp.asarray(yv, F32).astype(BF16), slots, w


@pytest.mark.parametrize(
    "live_per_block",
    [
        (1024, 1024, 1024, 1024),  # every slot live
        (1023, 517, 9, 0),  # unroll remainders and an empty step
        (1, 0, 255, 768),
    ],
)
def test_live_gather_sum_is_bitwise_to_float32_slot_order_sum(live_per_block):
    tokens = gather_sum_block_tokens(K) * len(live_per_block)
    y, slots, w = _case(0, tokens, live_per_block)
    out = jax.jit(lambda a, i, ww: gather_sum_rows(a, i, ww, F32, skip_negative=True))(y, jnp.asarray(slots), w)
    ref = _reference(y, slots, w)
    np.testing.assert_array_equal(np.asarray(out).view(np.int32), ref.view(np.int32))


def test_packed_live_gather_sum_matches_unpacked_bitwise():
    tokens = gather_sum_block_tokens(K) * 3
    y, slots, w = _case(1, tokens, (700, 1024, 3))
    fn = jax.jit(lambda p, i, ww: gather_sum_packed_rows(p, i, ww, H, H, F32, skip_negative=True))
    out = fn(pack_rows(y), jnp.asarray(slots), w)
    ref = _reference(y, slots, w)
    np.testing.assert_array_equal(np.asarray(out).view(np.int32), ref.view(np.int32))
