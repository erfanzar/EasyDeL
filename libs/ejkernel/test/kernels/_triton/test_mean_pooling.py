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
from ejkernel.kernels._xla.mean_pooling import mean_pooling as xla_mean_pooling

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


def _block_varlen_mean_pooling_ref(x: jax.Array, cu_seqlens: jax.Array, chunk_size: int) -> jax.Array:
    chunks = []
    for seq_idx in range(int(cu_seqlens.shape[0]) - 1):
        bos = int(cu_seqlens[seq_idx])
        eos = int(cu_seqlens[seq_idx + 1])
        for start in range(bos, eos, chunk_size):
            end = min(start + chunk_size, eos)
            chunks.append(jnp.mean(x[0, start:end], axis=0))
    return jnp.stack(chunks, axis=0)[None, ...]


def test_mean_pooling_fixed_3d_matches_xla_fwd_bwd_and_jit():
    kx, kw = jax.random.split(jax.random.PRNGKey(10), 2)
    x = jax.random.normal(kx, (3, 67, 48), dtype=jnp.float16)
    weights = jax.random.normal(kw, (3, 48), dtype=jnp.float32)

    out_tri = jax.block_until_ready(jax.jit(lambda inp: triton_mean_pooling(inp, block_size=32))(x))
    out_xla = jax.block_until_ready(jax.jit(lambda inp: xla_mean_pooling(inp))(x))
    np.testing.assert_allclose(
        np.asarray(out_tri, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )

    def _loss_tri(inp):
        return jnp.sum(triton_mean_pooling(inp, block_size=32).astype(jnp.float32) * weights)

    def _loss_xla(inp):
        return jnp.sum(xla_mean_pooling(inp).astype(jnp.float32) * weights)

    grad_tri = jax.block_until_ready(jax.jit(jax.grad(_loss_tri))(x))
    grad_xla = jax.block_until_ready(jax.jit(jax.grad(_loss_xla))(x))
    np.testing.assert_allclose(
        np.asarray(grad_tri, dtype=np.float32),
        np.asarray(grad_xla, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


def test_mean_pooling_varlen_2d_matches_xla_fwd_bwd_and_jit():
    kx, kw = jax.random.split(jax.random.PRNGKey(11), 2)
    # Include trailing padding tokens beyond cu_seqlens[-1] to verify zeroed padding grads.
    cu_seqlens = jnp.array([0, 5, 22, 31], dtype=jnp.int32)
    x = jax.random.normal(kx, (36, 40), dtype=jnp.float16)
    weights = jax.random.normal(kw, (3, 40), dtype=jnp.float32)

    out_tri = jax.block_until_ready(
        jax.jit(lambda inp, cu: triton_mean_pooling(inp, cu_seqlens=cu, block_size=16))(x, cu_seqlens)
    )
    out_xla = jax.block_until_ready(
        jax.jit(lambda inp, cu: xla_mean_pooling(inp, cu_seqlens=cu))(x, cu_seqlens)
    )
    np.testing.assert_allclose(
        np.asarray(out_tri, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )

    def _loss_tri(inp, cu):
        return jnp.sum(triton_mean_pooling(inp, cu_seqlens=cu, block_size=16).astype(jnp.float32) * weights)

    def _loss_xla(inp, cu):
        return jnp.sum(xla_mean_pooling(inp, cu_seqlens=cu).astype(jnp.float32) * weights)

    grad_tri = jax.block_until_ready(jax.jit(jax.grad(_loss_tri, argnums=0))(x, cu_seqlens))
    grad_xla = jax.block_until_ready(jax.jit(jax.grad(_loss_xla, argnums=0))(x, cu_seqlens))
    np.testing.assert_allclose(
        np.asarray(grad_tri, dtype=np.float32),
        np.asarray(grad_xla, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    np.testing.assert_array_equal(np.asarray(grad_tri[31:], dtype=np.float32), 0.0)


@pytest.mark.parametrize("chunk_size", [16, 12])
def test_mean_pooling_4d_varlen_fwd_and_bwd_matches_reference(chunk_size: int):
    kx, kw = jax.random.split(jax.random.PRNGKey(12 + chunk_size), 2)
    cu_seqlens = jnp.array([0, 5, 23, 35], dtype=jnp.int32)
    x = jax.random.normal(kx, (1, 38, 4, 24), dtype=jnp.float16)

    out_tri = jax.block_until_ready(triton_mean_pooling(x, chunk_size=chunk_size, cu_seqlens=cu_seqlens))
    out_ref = jax.block_until_ready(_block_varlen_mean_pooling_ref(x, cu_seqlens, chunk_size))
    np.testing.assert_allclose(
        np.asarray(out_tri, dtype=np.float32),
        np.asarray(out_ref, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )

    weights = jax.random.normal(kw, out_ref.shape, dtype=jnp.float32)

    def _loss_tri(inp):
        out = triton_mean_pooling(inp, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
        return jnp.sum(out.astype(jnp.float32) * weights)

    def _loss_ref(inp):
        return jnp.sum(_block_varlen_mean_pooling_ref(inp, cu_seqlens, chunk_size).astype(jnp.float32) * weights)

    grad_tri = jax.block_until_ready(jax.grad(_loss_tri)(x))
    grad_ref = jax.block_until_ready(jax.grad(_loss_ref)(x))
    np.testing.assert_allclose(
        np.asarray(grad_tri, dtype=np.float32),
        np.asarray(grad_ref, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    np.testing.assert_array_equal(np.asarray(grad_tri[:, 35:], dtype=np.float32), 0.0)


def test_mean_pooling_4d_jit_fwd_and_bwd():
    kx, kw = jax.random.split(jax.random.PRNGKey(30), 2)
    x = jax.random.normal(kx, (2, 35, 3, 24), dtype=jnp.float16)
    chunk_size = 12
    out_ref = jax.block_until_ready(_block_mean_pooling_ref(x, chunk_size))
    weights = jax.random.normal(kw, out_ref.shape, dtype=jnp.float32)

    out_tri = jax.block_until_ready(jax.jit(lambda inp: triton_mean_pooling(inp, chunk_size=chunk_size))(x))
    np.testing.assert_allclose(
        np.asarray(out_tri, dtype=np.float32),
        np.asarray(out_ref, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )

    def _loss_tri(inp):
        return jnp.sum(triton_mean_pooling(inp, chunk_size=chunk_size).astype(jnp.float32) * weights)

    def _loss_ref(inp):
        return jnp.sum(_block_mean_pooling_ref(inp, chunk_size).astype(jnp.float32) * weights)

    grad_tri = jax.block_until_ready(jax.jit(jax.grad(_loss_tri))(x))
    grad_ref = jax.block_until_ready(jax.jit(jax.grad(_loss_ref))(x))
    np.testing.assert_allclose(
        np.asarray(grad_tri, dtype=np.float32),
        np.asarray(grad_ref, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
