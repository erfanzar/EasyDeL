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


"""Numerical parity tests for ragged_page_attention_v2 (Triton implementation)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.errors import EjkernelRuntimeError
from ejkernel.kernels._triton.ragged_page_attention_v2 import ragged_page_attention_v2
from ejkernel.kernels._xla.ragged_page_attention_v2 import ragged_page_attention_v2 as xla_ragged_page_attention_v2

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def create_test_data(
    num_seqs=2,
    max_seq_len=128,
    num_q_heads=8,
    num_kv_heads=4,
    head_size=64,
    page_size=16,
    dtype=jnp.bfloat16,
    with_prefix=False,
    seed=42,
):
    """Create ragged paged attention test inputs, optionally with cached KV prefix."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 6)

    if with_prefix:
        prefix_lens = jax.random.randint(keys[0], (num_seqs,), 16, max(17, max_seq_len // 2))
        q_lens = jax.random.randint(keys[1], (num_seqs,), 4, max(5, max_seq_len // 2))
        context_lens = prefix_lens + q_lens
    else:
        q_lens = jax.random.randint(keys[0], (num_seqs,), 16, max_seq_len + 1)
        context_lens = q_lens

    total_query_tokens = int(jnp.sum(q_lens))
    queries = jax.random.normal(keys[2], (total_query_tokens, num_q_heads, head_size), dtype=jnp.float32).astype(dtype)

    max_pages = (int(jnp.max(context_lens)) + page_size - 1) // page_size
    num_pages = num_seqs * max_pages
    kv_pages = jax.random.normal(
        keys[3], (num_pages, page_size, 2 * num_kv_heads, head_size), dtype=jnp.float32
    ).astype(dtype)

    block_tables = jnp.arange(num_pages, dtype=jnp.int32).reshape(num_seqs, max_pages)
    query_start_loc = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(q_lens).astype(jnp.int32)])

    return {
        "queries": queries,
        "kv_pages": kv_pages,
        "context_lens": context_lens.astype(jnp.int32),
        "block_tables": block_tables,
        "query_start_loc": query_start_loc,
        "num_seqs": num_seqs,
        "q_lens": q_lens,
        "page_size": page_size,
    }


def _assert_matches_xla(
    data,
    *,
    optimized=False,
    sliding_window=None,
    logits_soft_cap=None,
    softmax_aux=None,
    softmax_scale=None,
    num_seqs=None,
    rtol=2e-2,
    atol=2e-2,
):
    if softmax_scale is None:
        softmax_scale = float(data["queries"].shape[-1]) ** -0.5
    active_num_seqs = data["num_seqs"] if num_seqs is None else num_seqs

    out_tri = ragged_page_attention_v2(
        queries=data["queries"],
        kv_pages=data["kv_pages"],
        context_lens=data["context_lens"],
        block_tables=data["block_tables"],
        query_start_loc=data["query_start_loc"],
        num_seqs=active_num_seqs,
        softmax_scale=softmax_scale,
        logits_soft_cap=logits_soft_cap,
        sliding_window=sliding_window,
        softmax_aux=softmax_aux,
        optimized=optimized,
    )
    out_xla = xla_ragged_page_attention_v2(
        queries=data["queries"],
        kv_pages=data["kv_pages"],
        context_lens=data["context_lens"],
        block_tables=data["block_tables"],
        query_start_loc=data["query_start_loc"],
        num_seqs=active_num_seqs,
        softmax_scale=softmax_scale,
        logits_soft_cap=logits_soft_cap,
        sliding_window=sliding_window,
        softmax_aux=softmax_aux,
        compute_dtype=jnp.float32,
        optimized=False,
    )

    out_tri = jax.block_until_ready(out_tri)
    out_xla = jax.block_until_ready(out_xla)
    assert out_tri.shape == data["queries"].shape
    assert out_tri.dtype == data["queries"].dtype
    assert jnp.all(jnp.isfinite(out_tri))
    np.testing.assert_allclose(
        np.asarray(out_tri, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_ragged_page_attention_v2_basic_and_prefix_matches_xla(optimized, dtype):
    """Both token-level and qblock-level kernels match XLA with cached KV prefix."""
    data = create_test_data(
        num_seqs=3,
        max_seq_len=96,
        num_q_heads=8,
        num_kv_heads=2,
        head_size=64,
        page_size=16,
        dtype=dtype,
        with_prefix=True,
        seed=11,
    )
    tol = 2e-4 if dtype == jnp.float32 else 2e-2
    _assert_matches_xla(data, optimized=optimized, rtol=tol, atol=tol)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    "sliding_window,logits_soft_cap",
    [
        (32, None),
        (None, 20.0),
        (48, 30.0),
    ],
)
def test_ragged_page_attention_v2_sliding_and_softcap_matches_xla(optimized, sliding_window, logits_soft_cap):
    """Verify sliding_window and logits_soft_cap numerical parity against XLA."""
    data = create_test_data(
        num_seqs=3,
        max_seq_len=128,
        num_q_heads=8,
        num_kv_heads=4,
        head_size=64,
        page_size=16,
        dtype=jnp.float16,
        with_prefix=True,
        seed=23,
    )
    _assert_matches_xla(
        data,
        optimized=optimized,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("optimized", [False, True])
def test_ragged_page_attention_v2_large_logits_softcap_no_nan(optimized):
    """Verify overflow-safe tanh prevents inf/inf NaN when |logits/cap| > 89."""
    data = create_test_data(
        num_seqs=2,
        max_seq_len=64,
        num_q_heads=4,
        num_kv_heads=2,
        head_size=64,
        page_size=16,
        dtype=jnp.float32,
        with_prefix=True,
        seed=31,
    )
    data["queries"] = data["queries"] * 100.0
    _assert_matches_xla(
        data,
        optimized=optimized,
        logits_soft_cap=10.0,
        rtol=2e-4,
        atol=2e-4,
    )


@pytest.mark.parametrize("optimized", [False, True])
def test_ragged_page_attention_v2_sinks_and_nonpow2_matches_xla(optimized):
    """Verify softmax_aux, non-pow2 head_dim/page_size, and active num_seqs subset."""
    data = create_test_data(
        num_seqs=4,
        max_seq_len=80,
        num_q_heads=8,
        num_kv_heads=2,
        head_size=48,
        page_size=24,
        dtype=jnp.float16,
        with_prefix=True,
        seed=47,
    )
    sinks = jax.random.normal(jax.random.PRNGKey(99), (8,), dtype=jnp.float32)
    _assert_matches_xla(
        data,
        optimized=optimized,
        sliding_window=40,
        logits_soft_cap=25.0,
        softmax_aux=sinks,
        num_seqs=jnp.array([2], dtype=jnp.int32),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("optimized", [False, True])
def test_ragged_page_attention_v2_jit_matches_xla(optimized):
    """Verify jax.jit tracing and numerical parity against XLA."""
    data = create_test_data(
        num_seqs=3,
        max_seq_len=96,
        num_q_heads=8,
        num_kv_heads=2,
        head_size=64,
        page_size=16,
        dtype=jnp.bfloat16,
        with_prefix=True,
        seed=59,
    )
    sinks = jax.random.normal(jax.random.PRNGKey(5), (8,), dtype=jnp.float32)
    num_seqs_arr = jnp.array([2], dtype=jnp.int32)
    scale = float(data["queries"].shape[-1]) ** -0.5

    @jax.jit
    def run_tri(q, kv, ctx, bt, qsl, nseq, aux):
        return ragged_page_attention_v2(
            queries=q,
            kv_pages=kv,
            context_lens=ctx,
            block_tables=bt,
            query_start_loc=qsl,
            num_seqs=nseq,
            softmax_scale=scale,
            logits_soft_cap=20.0,
            sliding_window=32,
            softmax_aux=aux,
            optimized=optimized,
        )

    out_tri = jax.block_until_ready(
        run_tri(
            data["queries"],
            data["kv_pages"],
            data["context_lens"],
            data["block_tables"],
            data["query_start_loc"],
            num_seqs_arr,
            sinks,
        )
    )
    out_xla = jax.block_until_ready(
        xla_ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=num_seqs_arr,
            softmax_scale=scale,
            logits_soft_cap=20.0,
            sliding_window=32,
            softmax_aux=sinks,
            compute_dtype=jnp.float32,
            optimized=False,
        )
    )
    np.testing.assert_allclose(
        np.asarray(out_tri, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


def test_ragged_page_attention_v2_rejects_unsupported_mask_value():
    """Passing non-default mask_value must fail explicitly instead of silently ignoring."""
    data = create_test_data(num_seqs=2, max_seq_len=64, seed=7)
    with pytest.raises(EjkernelRuntimeError):
        ragged_page_attention_v2(
            queries=data["queries"],
            kv_pages=data["kv_pages"],
            context_lens=data["context_lens"],
            block_tables=data["block_tables"],
            query_start_loc=data["query_start_loc"],
            num_seqs=data["num_seqs"],
            mask_value=-1e10,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
