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
from ejkernel.ops import FwdParams


def _has_gpu() -> bool:
    try:
        return len(jax.devices("gpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="Pallas GPU tests require a GPU backend")


def test_basic_matches_xla():
    key = jax.random.PRNGKey(0)
    key, kq, kk, kv = jax.random.split(key, 4)

    b, s, hq, hk, d = 2, 263, 8, 2, 64
    q = jax.random.normal(kq, (b, hq, d), dtype=jnp.float16)
    k = jax.random.normal(kk, (b, s, hk, d), dtype=jnp.float16)
    v = jax.random.normal(kv, (b, s, hk, d), dtype=jnp.float16)

    starts = jnp.array([0, 32], dtype=jnp.int32)
    ends = jnp.array([128, 200], dtype=jnp.int32)

    out_xla = ragged_decode_xla(q, k, v, starts, ends)
    for params in (None, FwdParams(num_key_splits=4, blocksize_heads=16, kv_blocksize=16)):
        out_gpu = ragged_decode_gpu(q, k, v, starts, ends, fwd_params=params)
        assert out_gpu.shape == out_xla.shape
        assert out_gpu.dtype == q.dtype
        assert jnp.allclose(out_gpu, out_xla, rtol=2e-2, atol=2e-2)


def _dense_ragged_decode(q, k, v, starts, ends):
    """Independent float32 reference: softmax over keys in [start, end) per batch row."""
    q, k, v = (x.astype(jnp.float32) for x in (q, k, v))
    reps = q.shape[1] // k.shape[2]
    k = jnp.repeat(k, reps, axis=2)
    v = jnp.repeat(v, reps, axis=2)
    scale = 1.0 / jnp.sqrt(jnp.float32(q.shape[-1]))
    logits = jnp.einsum("bhd,bshd->bhs", q, k) * scale
    pos = jnp.arange(k.shape[1])[None, None, :]
    keep = (pos >= starts[:, None, None]) & (pos < ends[:, None, None])
    probs = jax.nn.softmax(jnp.where(keep, logits, -jnp.inf), axis=-1)
    return jnp.einsum("bhs,bshd->bhd", probs, v)


@pytest.mark.parametrize("seq_len", [263, 257, 40])
@pytest.mark.parametrize("num_key_splits,kv_blocksize", [(1, 16), (1, 64), (3, 32), (4, 16)])
def test_unaligned_length_masks_tail_tile_matches_dense(seq_len, num_key_splits, kv_blocksize):
    """A key length that is not a multiple of the tile (or split) must not leak tail/pad keys."""
    kq, kk, kv = jax.random.split(jax.random.PRNGKey(7), 3)
    b, hq, hk, d = 2, 8, 2, 64
    q = jax.random.normal(kq, (b, hq, d), dtype=jnp.float16)
    k = jax.random.normal(kk, (b, seq_len, hk, d), dtype=jnp.float16)
    v = jax.random.normal(kv, (b, seq_len, hk, d), dtype=jnp.float16)
    starts = jnp.array([0, 5], dtype=jnp.int32)
    ends = jnp.array([seq_len, seq_len - 3], dtype=jnp.int32)  # one row reads to the very last key

    params = FwdParams(num_key_splits=num_key_splits, blocksize_heads=16, kv_blocksize=kv_blocksize)
    out = ragged_decode_gpu(q, k, v, starts, ends, fwd_params=params)
    want = _dense_ragged_decode(q, k, v, starts, ends)
    assert out.shape == want.shape
    assert jnp.allclose(out.astype(jnp.float32), want, rtol=2e-2, atol=2e-2)


def test_split_reduction_keeps_fp32_partial_sums():
    q = jnp.zeros((1, 16, 64), dtype=jnp.float16)
    k = jnp.zeros((1, 512, 1, 64), dtype=jnp.float16)
    v = jnp.concatenate(
        [jnp.full((1, 256, 1, 64), 1000, dtype=jnp.float16), jnp.full((1, 256, 1, 64), -999, dtype=jnp.float16)],
        axis=1,
    )
    starts = jnp.array([0], dtype=jnp.int32)
    ends = jnp.array([512], dtype=jnp.int32)

    out = ragged_decode_gpu(q, k, v, starts, ends, fwd_params=FwdParams(num_key_splits=2, kv_blocksize=64))
    assert jnp.isfinite(out).all()
    assert jnp.allclose(out, jnp.full_like(out, 0.5), atol=0.02)


def test_unsupported_attention_options_are_rejected():
    q = jnp.ones((1, 1, 64), dtype=jnp.float16)
    kv = jnp.ones((1, 16, 1, 64), dtype=jnp.float16)
    starts = jnp.array([0], dtype=jnp.int32)
    ends = jnp.array([16], dtype=jnp.int32)
    for kwargs in ({"sliding_window": (4, 0)}, {"logits_soft_cap": 30.0}, {"softmax_aux": jnp.ones((1,))}):
        with pytest.raises(NotImplementedError):
            ragged_decode_gpu(q, kv, kv, starts, ends, **kwargs)
