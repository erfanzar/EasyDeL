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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ejkernel.kernels._triton.mean_pooling import mean_pooling as triton_mean_pooling

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def _block_mean_pooling_ref(x: jax.Array, chunk_size: int) -> jax.Array:
    z, seqlen, heads, dim = x.shape
    num_blocks = (seqlen + chunk_size - 1) // chunk_size
    pad_len = num_blocks * chunk_size - seqlen
    x_pad = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
    x_blk = x_pad.reshape(z, num_blocks, chunk_size, heads, dim)
    sums = jnp.sum(x_blk, axis=2)
    counts = jnp.minimum(chunk_size, seqlen - jnp.arange(num_blocks) * chunk_size).astype(sums.dtype)
    return sums / counts[None, :, None, None]


def test_mean_pooling_matches_reference():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 50, 4, 32), dtype=jnp.float16)
    chunk_size = 16

    out_triton = triton_mean_pooling(x, chunk_size=chunk_size)
    out_ref = _block_mean_pooling_ref(x, chunk_size)

    out_triton = jax.block_until_ready(out_triton)
    out_ref = jax.block_until_ready(out_ref)

    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_ref, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


def test_mean_pooling_fixed_3d_matches_reference():
    key = jax.random.PRNGKey(2)
    x = jax.random.normal(key, (3, 67, 48), dtype=jnp.float16)

    out_triton = triton_mean_pooling(x, chunk_size=16)
    out_ref = jnp.mean(x, axis=1)

    out_triton = jax.block_until_ready(out_triton)
    out_ref = jax.block_until_ready(out_ref)

    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_ref, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


def test_mean_pooling_varlen_2d_matches_reference():
    key = jax.random.PRNGKey(3)
    lengths = (5, 17, 9)
    cu_seqlens = jnp.array([0, 5, 22, 31], dtype=jnp.int32)
    x = jax.random.normal(key, (31, 40), dtype=jnp.float16)

    out_triton = triton_mean_pooling(x, chunk_size=8, cu_seqlens=cu_seqlens)
    out_ref = jnp.stack(
        [jnp.mean(x[int(cu_seqlens[i]) : int(cu_seqlens[i + 1])], axis=0) for i, _ in enumerate(lengths)]
    )

    out_triton = jax.block_until_ready(out_triton)
    out_ref = jax.block_until_ready(out_ref)

    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_ref, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


def test_mean_pooling_grad_matches_reference():
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (2, 50, 4, 32), dtype=jnp.float16)
    chunk_size = 16

    def loss_triton(inp):
        return jnp.sum(triton_mean_pooling(inp, chunk_size=chunk_size))

    def loss_ref(inp):
        return jnp.sum(_block_mean_pooling_ref(inp, chunk_size))

    grad_triton = jax.grad(loss_triton)(x)
    grad_ref = jax.grad(loss_ref)(x)

    grad_triton = jax.block_until_ready(grad_triton)
    grad_ref = jax.block_until_ready(grad_ref)

    np.testing.assert_allclose(
        np.asarray(grad_triton, dtype=np.float32),
        np.asarray(grad_ref, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


def test_mean_pooling_fixed_3d_grad_matches_reference():
    key = jax.random.PRNGKey(4)
    x = jax.random.normal(key, (2, 37, 24), dtype=jnp.float16)

    def loss_triton(inp):
        return jnp.sum(triton_mean_pooling(inp, chunk_size=8))

    def loss_ref(inp):
        return jnp.sum(jnp.mean(inp, axis=1))

    grad_triton = jax.grad(loss_triton)(x)
    grad_ref = jax.grad(loss_ref)(x)

    grad_triton = jax.block_until_ready(grad_triton)
    grad_ref = jax.block_until_ready(grad_ref)

    np.testing.assert_allclose(
        np.asarray(grad_triton, dtype=np.float32),
        np.asarray(grad_ref, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


def test_mean_pooling_varlen_2d_grad_matches_reference():
    key = jax.random.PRNGKey(5)
    cu_seqlens = jnp.array([0, 4, 10, 23], dtype=jnp.int32)
    x = jax.random.normal(key, (23, 20), dtype=jnp.float16)

    def loss_triton(inp):
        return jnp.sum(triton_mean_pooling(inp, chunk_size=8, cu_seqlens=cu_seqlens))

    def loss_ref(inp):
        return jnp.sum(
            jnp.stack(
                [jnp.mean(inp[int(cu_seqlens[i]) : int(cu_seqlens[i + 1])], axis=0) for i in range(len(cu_seqlens) - 1)]
            )
        )

    grad_triton = jax.grad(loss_triton)(x)
    grad_ref = jax.grad(loss_ref)(x)

    grad_triton = jax.block_until_ready(grad_triton)
    grad_ref = jax.block_until_ready(grad_ref)

    np.testing.assert_allclose(
        np.asarray(grad_triton, dtype=np.float32),
        np.asarray(grad_ref, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
