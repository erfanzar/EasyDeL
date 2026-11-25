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
from ejkernel.kernels._triton.ragged_decode_attention import ragged_decode_attention as ragged_decode_triton
from ejkernel.kernels._xla.ragged_decode_attention import ragged_decode_attention as ragged_decode_xla
from ejkernel.ops import FwdParams

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def ragged_decode_ref(
    query,
    key,
    value,
    sequence_start,
    sequence_end,
    softmax_scale=1.0,
    sliding_window=None,
    logits_soft_cap=None,
    softmax_aux=None,
):
    """
    Dense reference for ragged decode. Matches Triton's streaming behavior:
    - Query at qpos=end-1
    - Valid tokens in [start,end) and within sliding window (if provided)
    - Sinks (aux logits) contribute to normalization mass but not values
    """
    B, HQ, D = query.shape
    S, HKV = key.shape[1], key.shape[2]
    assert HQ % HKV == 0
    G = HQ // HKV

    out = jnp.zeros((B, HQ, D), dtype=query.dtype)
    if softmax_aux is not None and getattr(softmax_aux, "ndim", None) not in (1, 2):
        raise ValueError(f"softmax_aux must be rank 1 or 2, got shape {softmax_aux.shape}")
    for b in range(B):
        s_b = int(sequence_start[b])
        e_b = int(sequence_end[b])
        if e_b <= s_b:
            continue
        qpos = e_b - 1

        left = sliding_window[0] if sliding_window is not None else None
        right = sliding_window[1] if sliding_window is not None else None

        for hq in range(HQ):
            kvh = hq // G
            q_vec = query[b, hq]

            t_idx = jnp.arange(S)
            mask = (t_idx >= s_b) & (t_idx < e_b)

            if left is not None:
                mask = mask & (t_idx >= (qpos - left))
            if right is not None:
                mask = mask & (t_idx <= (qpos + right))

            k_b = key[b, :, kvh, :]
            v_b = value[b, :, kvh, :]

            logits = jnp.einsum("d,sd->s", q_vec, k_b) * float(softmax_scale)
            if logits_soft_cap is not None and logits_soft_cap > 0:
                cap = float(logits_soft_cap)
                logits = cap * jnp.tanh(logits / cap)

            logits_seq = jnp.where(mask, logits, -jnp.inf)

            logits_sink = None
            if softmax_aux is not None:
                if softmax_aux.ndim == 1:
                    logits_sink = softmax_aux.astype(jnp.float32)
                elif softmax_aux.shape[0] == HKV:
                    logits_sink = softmax_aux[kvh].astype(jnp.float32)
                elif softmax_aux.shape[0] == HQ:
                    logits_sink = softmax_aux[hq].astype(jnp.float32)
                else:
                    raise ValueError(f"softmax_aux head dimension must be {HKV} or {HQ}, got {softmax_aux.shape[0]}")

            if logits_sink is None:
                m = jnp.max(logits_seq)
                p_seq = jnp.exp(logits_seq - m)
                denom = jnp.sum(p_seq)
            else:
                m = jnp.maximum(jnp.max(logits_seq), jnp.max(logits_sink))
                p_seq = jnp.exp(logits_seq - m)
                p_sink = jnp.exp(logits_sink - m)
                denom = jnp.sum(p_seq) + jnp.sum(p_sink)

            denom = jnp.where(denom == 0.0, 1.0, denom)
            w_seq = p_seq / denom
            o_vec = jnp.einsum("s,sd->d", w_seq, v_b)
            out = out.at[b, hq].set(o_vec.astype(query.dtype))
    return out


class TestRaggedDecodeAttentionTriton:
    def test_forward_shape(self):
        B, S, HQ, HKV, D = 2, 512, 8, 2, 64

        key = jax.random.PRNGKey(0)
        kq, kk, kv = jax.random.split(key, 3)

        q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
        k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
        v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)

        starts = jnp.array([0, 100], dtype=jnp.int32)
        ends = jnp.array([400, 480], dtype=jnp.int32)

        out = ragged_decode_triton(
            q,
            k,
            v,
            sequence_start=starts,
            sequence_end=ends,
            softmax_scale=1.0,
        )
        assert out.shape == (B, HQ, D)
        assert jnp.isfinite(out).all()

    @pytest.mark.parametrize(
        "sliding_window,logits_soft_cap",
        [
            (None, None),
            ((128, 0), None),
            ((256, 64), None),
            (None, 10.0),
            ((128, 64), 20.0),
        ],
    )
    def test_against_reference(self, sliding_window, logits_soft_cap):
        B, S, HQ, HKV, D = 2, 384, 16, 2, 64

        key = jax.random.PRNGKey(1)
        kq, kk, kv = jax.random.split(key, 3)

        q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
        k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
        v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)

        starts = jnp.array([0, 50], dtype=jnp.int32)
        ends = jnp.array([350, 320], dtype=jnp.int32)

        out_tri = ragged_decode_triton(
            q,
            k,
            v,
            sequence_start=starts,
            sequence_end=ends,
            softmax_scale=1.0,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=None,
        )

        out_ref = ragged_decode_ref(
            q,
            k,
            v,
            sequence_start=starts,
            sequence_end=ends,
            softmax_scale=1.0,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=None,
        )

        assert out_tri.shape == out_ref.shape
        assert jnp.allclose(out_tri, out_ref, rtol=1e-4, atol=1e-5)

    def test_with_sinks(self):
        B, S, HQ, HKV, D = 2, 256, 16, 4, 64

        key = jax.random.PRNGKey(2)
        kq, kk, kv, ka = jax.random.split(key, 4)

        q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
        k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
        v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)

        starts = jnp.array([10, 0], dtype=jnp.int32)
        ends = jnp.array([200, 220], dtype=jnp.int32)

        NS = 3
        aux_per_kv = jax.random.normal(ka, (HKV, NS), dtype=jnp.float32)
        aux_shared = jnp.linspace(-0.5, 0.5, NS, dtype=jnp.float32)

        out_tri = ragged_decode_triton(
            q,
            k,
            v,
            starts,
            ends,
            softmax_scale=0.75,
            sliding_window=(128, 16),
            logits_soft_cap=25.0,
            softmax_aux=aux_per_kv,
        )
        out_ref = ragged_decode_ref(
            q,
            k,
            v,
            starts,
            ends,
            softmax_scale=0.75,
            sliding_window=(128, 16),
            logits_soft_cap=25.0,
            softmax_aux=aux_per_kv,
        )
        assert jnp.allclose(out_tri, out_ref, rtol=2e-4, atol=2e-5)

        out_tri2 = ragged_decode_triton(
            q,
            k,
            v,
            starts,
            ends,
            softmax_scale=1.25,
            sliding_window=None,
            logits_soft_cap=None,
            softmax_aux=aux_shared,
        )
        out_ref2 = ragged_decode_ref(
            q,
            k,
            v,
            starts,
            ends,
            softmax_scale=1.25,
            sliding_window=None,
            logits_soft_cap=None,
            softmax_aux=aux_shared,
        )
        assert jnp.allclose(out_tri2, out_ref2, rtol=2e-4, atol=2e-5)

    def test_tail_block_and_gqa(self):
        B, S, HQ, HKV, D = 1, 410, 32, 4, 64

        key = jax.random.PRNGKey(3)
        kq, kk, kv = jax.random.split(key, 3)

        q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
        k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
        v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)

        starts = jnp.array([5], dtype=jnp.int32)
        ends = jnp.array([407], dtype=jnp.int32)

        out_tri = ragged_decode_triton(
            q,
            k,
            v,
            starts,
            ends,
            softmax_scale=1.0,
        )
        out_ref = ragged_decode_ref(
            q,
            k,
            v,
            starts,
            ends,
            softmax_scale=1.0,
        )
        assert jnp.allclose(out_tri, out_ref, rtol=2e-4, atol=2e-5)


def test_ragged_decode_attention_matches_xla_smoke():
    key = jax.random.PRNGKey(0)
    kq, kk, kv, ka = jax.random.split(key, 4)

    B, S, HQ, HKV, D = 2, 256, 16, 2, 64
    q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
    k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
    v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)
    starts = jnp.array([0, 32], dtype=jnp.int32)
    ends = jnp.array([200, 180], dtype=jnp.int32)
    sinks = jax.random.normal(ka, (4,), dtype=jnp.float32) * 0.1

    out_triton = ragged_decode_triton(
        q,
        k,
        v,
        sequence_start=starts,
        sequence_end=ends,
        softmax_scale=1.0,
        sliding_window=(128, 0),
        logits_soft_cap=20.0,
        softmax_aux=sinks,
    )
    out_xla = ragged_decode_xla(
        q,
        k,
        v,
        sequence_start=starts,
        sequence_end=ends,
        softmax_scale=1.0,
        sliding_window=(128, 0),
        logits_soft_cap=20.0,
        softmax_aux=sinks,
    )

    out_triton = jax.block_until_ready(out_triton)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(out_triton, out_xla, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("head_dim", [192, 256])
def test_ragged_decode_large_head_dim_matches_xla(head_dim):
    key = jax.random.PRNGKey(100 + head_dim)
    kq, kk, kv = jax.random.split(key, 3)

    B, S, HQ, HKV = 2, 320, 8, 2
    q = jax.random.normal(kq, (B, HQ, head_dim), dtype=jnp.float32)
    k = jax.random.normal(kk, (B, S, HKV, head_dim), dtype=jnp.float32)
    v = jax.random.normal(kv, (B, S, HKV, head_dim), dtype=jnp.float32)
    starts = jnp.array([0, 40], dtype=jnp.int32)
    ends = jnp.array([300, 290], dtype=jnp.int32)

    out_tri = ragged_decode_triton(
        q,
        k,
        v,
        sequence_start=starts,
        sequence_end=ends,
        softmax_scale=head_dim**-0.5,
        logits_soft_cap=20.0,
    )
    out_xla = ragged_decode_xla(
        q,
        k,
        v,
        sequence_start=starts,
        sequence_end=ends,
        softmax_scale=head_dim**-0.5,
        logits_soft_cap=20.0,
    )

    np.testing.assert_allclose(
        np.asarray(jax.block_until_ready(out_tri)),
        np.asarray(jax.block_until_ready(out_xla)),
        rtol=2e-4,
        atol=2e-4,
    )


@pytest.mark.parametrize("aux_kind", ["shared_1d", "per_kv_2d", "per_q_2d"])
def test_ragged_decode_multiblock_sinks_matches_xla(aux_kind):
    """Verify softmax_aux is folded once across multiple KV blocks (S > kv_blocksize)."""
    key = jax.random.PRNGKey(77)
    kq, kk, kv, ka = jax.random.split(key, 4)

    B, S, HQ, HKV, D = 2, 512, 8, 2, 64
    q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
    k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
    v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)
    starts = jnp.array([0, 64], dtype=jnp.int32)
    ends = jnp.array([480, 500], dtype=jnp.int32)

    if aux_kind == "shared_1d":
        sinks = jax.random.normal(ka, (3,), dtype=jnp.float32)
    elif aux_kind == "per_kv_2d":
        sinks = jax.random.normal(ka, (HKV, 3), dtype=jnp.float32)
    else:
        sinks = jax.random.normal(ka, (HQ, 3), dtype=jnp.float32)

    fwd_params = FwdParams(kv_blocksize=64)
    out_tri = ragged_decode_triton(
        q,
        k,
        v,
        sequence_start=starts,
        sequence_end=ends,
        softmax_scale=D**-0.5,
        fwd_params=fwd_params,
        sliding_window=(256, 0),
        logits_soft_cap=30.0,
        softmax_aux=sinks,
    )
    out_xla = ragged_decode_xla(
        q,
        k,
        v,
        sequence_start=starts,
        sequence_end=ends,
        softmax_scale=D**-0.5,
        fwd_params=fwd_params,
        sliding_window=(256, 0),
        logits_soft_cap=30.0,
        softmax_aux=sinks,
    )

    np.testing.assert_allclose(
        np.asarray(jax.block_until_ready(out_tri)),
        np.asarray(jax.block_until_ready(out_xla)),
        rtol=2e-4,
        atol=2e-4,
    )


def test_ragged_decode_jit_matches_xla():
    key = jax.random.PRNGKey(2026)
    kq, kk, kv, ka = jax.random.split(key, 4)

    B, S, HQ, HKV, D = 2, 384, 8, 2, 192
    q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.bfloat16)
    starts = jnp.array([16, 0], dtype=jnp.int32)
    ends = jnp.array([350, 0], dtype=jnp.int32)  # include empty sequence edge case
    sinks = jax.random.normal(ka, (HKV, 2), dtype=jnp.float32)

    @jax.jit
    def run_tri(q_, k_, v_, s_, e_, aux_):
        return ragged_decode_triton(
            q_,
            k_,
            v_,
            sequence_start=s_,
            sequence_end=e_,
            softmax_scale=D**-0.5,
            fwd_params=FwdParams(kv_blocksize=64),
            sliding_window=(128, 0),
            logits_soft_cap=25.0,
            softmax_aux=aux_,
        )

    out_tri = jax.block_until_ready(run_tri(q, k, v, starts, ends, sinks))
    out_xla = jax.block_until_ready(
        ragged_decode_xla(
            q,
            k,
            v,
            sequence_start=starts,
            sequence_end=ends,
            softmax_scale=D**-0.5,
            fwd_params=FwdParams(kv_blocksize=64),
            sliding_window=(128, 0),
            logits_soft_cap=25.0,
            softmax_aux=sinks,
        )
    )

    assert out_tri.dtype == q.dtype
    assert jnp.all(jnp.isfinite(out_tri))
    np.testing.assert_allclose(
        np.asarray(out_tri, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
