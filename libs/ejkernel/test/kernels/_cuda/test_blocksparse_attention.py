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

import importlib
import importlib.util
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ejkernel.kernels._cuda.blocksparse_attention import blocksparse_attention as cuda_blocksparse_attention
from ejkernel.kernels._xla.blocksparse_attention import blocksparse_attention as xla_blocksparse_attention
from ejkernel.ops import BwdParams, FwdParams

pytestmark = pytest.mark.skipif(
    jax.devices()[0].platform != "gpu" or shutil.which("nvcc") is None,
    reason="CUDA blocksparse tests require GPU backend and nvcc",
)

_HAS_TRITON = importlib.util.find_spec("triton") is not None
triton_blocksparse_attention = None
if _HAS_TRITON:
    try:
        _triton_blocksparse_module = importlib.import_module("ejkernel.kernels._triton.blocksparse_attention")
        triton_blocksparse_attention = _triton_blocksparse_module.blocksparse_attention
    except Exception:
        _HAS_TRITON = False


def _device_put_all(dev, *arrays):
    return [jax.device_put(arr, dev) for arr in arrays]


def test_blocksparse_attention_cuda_matches_xla():
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)

    batch, num_heads, q_len, head_dim = 1, 4, 64, 32
    num_kv_heads, kv_len = 2, 64

    q = jax.random.normal(kq, (batch, num_heads, q_len, head_dim), dtype=jnp.float16)
    k = jax.random.normal(kk, (batch, num_kv_heads, kv_len, head_dim), dtype=jnp.float16)
    v = jax.random.normal(kv, (batch, num_kv_heads, kv_len, head_dim), dtype=jnp.float16)

    fwd_params = FwdParams(q_blocksize=32, kv_blocksize=32, num_warps=4, num_stages=2)
    bwd_params = BwdParams(q_blocksize=32, kv_blocksize=32, num_warps=4, num_stages=2)

    dev = jax.devices("gpu")[0]
    q, k, v = _device_put_all(dev, q, k, v)

    out_cuda = cuda_blocksparse_attention(
        q,
        k,
        v,
        causal=True,
        sliding_window=(32, 32),
        softmax_scale=head_dim**-0.5,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
    )
    out_xla = xla_blocksparse_attention(
        q,
        k,
        v,
        causal=True,
        sliding_window=(32, 32),
        softmax_scale=head_dim**-0.5,
    )

    out_cuda = jax.block_until_ready(out_cuda)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(out_cuda, out_xla, rtol=1e-2, atol=1e-2)


def test_blocksparse_attention_cuda_softcap_sinks():
    key = jax.random.PRNGKey(42)
    kq, kk, kv = jax.random.split(key, 3)

    batch, num_heads, q_len, head_dim = 1, 4, 32, 32
    num_kv_heads, kv_len = 2, 32

    q = jax.random.normal(kq, (batch, num_heads, q_len, head_dim), dtype=jnp.float16)
    k = jax.random.normal(kk, (batch, num_kv_heads, kv_len, head_dim), dtype=jnp.float16)
    v = jax.random.normal(kv, (batch, num_kv_heads, kv_len, head_dim), dtype=jnp.float16)

    num_sinks = 4
    softmax_aux = jnp.linspace(-0.5, 0.5, num_sinks, dtype=jnp.float32).reshape(num_sinks)

    fwd_params = FwdParams(q_blocksize=16, kv_blocksize=16, num_warps=4, num_stages=2)
    bwd_params = BwdParams(q_blocksize=16, kv_blocksize=16, num_warps=4, num_stages=2)

    dev = jax.devices("gpu")[0]
    q, k, v, softmax_aux = _device_put_all(dev, q, k, v, softmax_aux)

    out_cuda = cuda_blocksparse_attention(
        q,
        k,
        v,
        causal=True,
        sliding_window=(16, 16),
        softmax_scale=head_dim**-0.5,
        softmax_aux=softmax_aux,
        logits_soft_cap=2.0,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
    )
    out_xla = xla_blocksparse_attention(
        q,
        k,
        v,
        causal=True,
        sliding_window=(16, 16),
        softmax_scale=head_dim**-0.5,
        softmax_aux=softmax_aux,
        logits_soft_cap=2.0,
    )

    out_cuda = jax.block_until_ready(out_cuda)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(out_cuda, out_xla, rtol=2e-2, atol=2e-2)


def test_blocksparse_attention_cuda_noncausal_matches_xla():
    key = jax.random.PRNGKey(7)
    kq, kk, kv = jax.random.split(key, 3)

    batch, num_heads, q_len, head_dim = 1, 2, 48, 32
    num_kv_heads, kv_len = 2, 48

    q = jax.random.normal(kq, (batch, num_heads, q_len, head_dim), dtype=jnp.float16)
    k = jax.random.normal(kk, (batch, num_kv_heads, kv_len, head_dim), dtype=jnp.float16)
    v = jax.random.normal(kv, (batch, num_kv_heads, kv_len, head_dim), dtype=jnp.float16)

    fwd_params = FwdParams(q_blocksize=16, kv_blocksize=16, num_warps=4, num_stages=2)
    bwd_params = BwdParams(q_blocksize=16, kv_blocksize=16, num_warps=4, num_stages=2)

    dev = jax.devices("gpu")[0]
    q, k, v = _device_put_all(dev, q, k, v)

    out_cuda = cuda_blocksparse_attention(
        q,
        k,
        v,
        causal=False,
        sliding_window=None,
        softmax_scale=head_dim**-0.5,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
    )
    out_xla = xla_blocksparse_attention(
        q,
        k,
        v,
        causal=False,
        sliding_window=None,
        softmax_scale=head_dim**-0.5,
    )

    out_cuda = jax.block_until_ready(out_cuda)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(out_cuda, out_xla, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton is required for CUDA-vs-Triton gradient parity")
def test_blocksparse_attention_cuda_grad_matches_triton_and_vjp():
    assert triton_blocksparse_attention is not None
    key = jax.random.PRNGKey(123)
    kq, kk, kv = jax.random.split(key, 3)

    batch, num_heads, q_len, head_dim = 1, 4, 64, 32
    num_kv_heads, kv_len = 2, 64

    q = jax.random.normal(kq, (batch, num_heads, q_len, head_dim), dtype=jnp.float16)
    k = jax.random.normal(kk, (batch, num_kv_heads, kv_len, head_dim), dtype=jnp.float16)
    v = jax.random.normal(kv, (batch, num_kv_heads, kv_len, head_dim), dtype=jnp.float16)

    fwd_params = FwdParams(q_blocksize=32, kv_blocksize=32, num_warps=4, num_stages=2)
    bwd_params = BwdParams(q_blocksize=32, kv_blocksize=32, num_warps=4, num_stages=2)

    dev = jax.devices("gpu")[0]
    q, k, v = _device_put_all(dev, q, k, v)

    def loss_cuda(q_, k_, v_):
        out = cuda_blocksparse_attention(
            q_,
            k_,
            v_,
            causal=True,
            sliding_window=(32, 32),
            softmax_scale=head_dim**-0.5,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
        )
        return jnp.mean(out)

    def loss_triton(q_, k_, v_):
        out = triton_blocksparse_attention(
            q_,
            k_,
            v_,
            causal=True,
            sliding_window=(32, 32),
            softmax_scale=head_dim**-0.5,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
        )
        return jnp.mean(out)

    grads_cuda = jax.jit(jax.grad(loss_cuda, argnums=(0, 1, 2)))(q, k, v)
    grads_triton = jax.jit(jax.grad(loss_triton, argnums=(0, 1, 2)))(q, k, v)
    grads_cuda = jax.tree_util.tree_map(jax.block_until_ready, grads_cuda)
    grads_triton = jax.tree_util.tree_map(jax.block_until_ready, grads_triton)

    for g_cuda, g_triton in zip(grads_cuda, grads_triton, strict=True):
        np.testing.assert_allclose(
            np.asarray(g_cuda, dtype=np.float32),
            np.asarray(g_triton, dtype=np.float32),
            rtol=3e-2,
            atol=3e-2,
        )

    def fwd_cuda(q_, k_, v_):
        return cuda_blocksparse_attention(
            q_,
            k_,
            v_,
            causal=True,
            sliding_window=(32, 32),
            softmax_scale=head_dim**-0.5,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
        )

    @jax.jit
    def vjp_eval(q_, k_, v_):
        out, vjp_fn = jax.vjp(fwd_cuda, q_, k_, v_)
        return vjp_fn(jnp.ones_like(out))

    vjp_grads = vjp_eval(q, k, v)
    for grad in vjp_grads:
        assert jnp.all(jnp.isfinite(grad))
