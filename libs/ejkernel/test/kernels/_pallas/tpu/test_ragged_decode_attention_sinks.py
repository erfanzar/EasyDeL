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

"""Attention-sink parity for the Pallas TPU ragged decode kernel.

The kernel output is compared with an independent dense float32 reference that
appends the raw sink logits to the softmax denominator (sinks are not scaled by
``softmax_scale``). Sink values differ per head, and a negative control checks
that per-head sinks are clearly distinguishable from a shared sink.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.ragged_decode_attention import ragged_decode_attention as ragged_decode_tpu
from ejkernel.ops.utils.datacarrier import FwdParams


def _has_tpu():
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")

B, S, HQ, HKV, D = 2, 512, 8, 2, 128
SCALE = D**-0.5


def _inputs():
    kq, kk, kv = jax.random.split(jax.random.PRNGKey(7), 3)
    q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
    k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
    v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)
    starts = jnp.array([0, 100], dtype=jnp.int32)
    ends = jnp.array([300, 480], dtype=jnp.int32)
    return q, k, v, starts, ends


def _per_head_rows(sinks):
    """Expand any accepted layout to ``(HQ, S_sinks)`` (independent of production code)."""
    sinks = jnp.asarray(sinks, jnp.float32)
    if sinks.ndim == 1:
        if sinks.shape[0] == HQ:
            return sinks[:, None]
        return jnp.broadcast_to(sinks[None, :], (HQ, sinks.shape[0]))
    if sinks.shape[0] == 1:
        return jnp.broadcast_to(sinks, (HQ, sinks.shape[1]))
    if sinks.shape[0] == HKV:
        return jnp.repeat(sinks, HQ // HKV, axis=0)
    assert sinks.shape[0] == HQ
    return sinks


def _dense_reference(q, k, v, starts, ends, sinks):
    """Dense f32 ragged decode attention with raw sink logits in the softmax denominator."""
    hp = jax.lax.Precision.HIGHEST
    kf = jnp.repeat(k, HQ // HKV, axis=2)
    vf = jnp.repeat(v, HQ // HKV, axis=2)
    logits = jnp.einsum("bhd,bshd->bhs", q * SCALE, kf, precision=hp)
    pos = jnp.arange(S)[None, None, :]
    valid = (pos >= starts[:, None, None]) & (pos < ends[:, None, None])
    logits = jnp.where(valid, logits, -1e30)
    if sinks is not None:
        aux = jnp.broadcast_to(_per_head_rows(sinks)[None], (B, HQ, _per_head_rows(sinks).shape[-1]))
        logits = jnp.concatenate([logits, aux], axis=-1)
    weights = jax.nn.softmax(logits, axis=-1)[..., :S]
    return jnp.einsum("bhs,bshd->bhd", weights, vf, precision=hp)


def _sink_values(shape):
    # The key softmax mass here has logsumexp ~6 (~300 visible keys), so sinks in [2, 9] take
    # ~1%..94% of the probability: the per-head effect is large next to any kernel error.
    n = int(np.prod(shape))
    return jnp.linspace(2.0, 9.0, n, dtype=jnp.float32).reshape(shape)


def _rel(a, b):
    """Max abs difference relative to the reference's max magnitude."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


PARITY_REL = 2e-2
EFFECT_REL = 0.2


_LAYOUTS = {
    "per_head_1d": (HQ,),
    "shared_1d": (3,),
    "shared_2d_singleton": (1, 3),
    "per_head_2d": (HQ, 3),
    "per_kv_2d": (HKV, 3),
}


class TestRaggedDecodeSinksTPU:
    @pytest.mark.parametrize("shape", list(_LAYOUTS.values()), ids=list(_LAYOUTS))
    def test_sink_layouts_match_dense_reference(self, shape):
        q, k, v, starts, ends = _inputs()
        sinks = _sink_values(shape)
        out = ragged_decode_tpu(q, k, v, starts, ends, softmax_scale=SCALE, softmax_aux=sinks)
        ref = _dense_reference(q, k, v, starts, ends, sinks)
        nosink = _dense_reference(q, k, v, starts, ends, None)
        err, effect = _rel(out, ref), _rel(nosink, ref)
        print(f"{shape}: kernel-vs-ref rel err {err:.3e}; sink effect rel {effect:.3e}")
        assert err < PARITY_REL, err
        # The comparison must resolve sinks: ignoring them would be far outside the tolerance.
        assert effect > EFFECT_REL, effect

    def test_sinks_change_output_vs_no_sinks(self):
        q, k, v, starts, ends = _inputs()
        sinks = _sink_values((HQ,))
        with_sink = ragged_decode_tpu(q, k, v, starts, ends, softmax_scale=SCALE, softmax_aux=sinks)
        without = ragged_decode_tpu(q, k, v, starts, ends, softmax_scale=SCALE)
        assert _rel(without, with_sink) > EFFECT_REL, "sinks were silently ignored"

    def test_negative_control_per_head_vs_shared(self):
        q, k, v, starts, ends = _inputs()
        per_head = _sink_values((HQ,))
        shared = jnp.broadcast_to(per_head[:1], (HQ,))
        out_per_head = ragged_decode_tpu(q, k, v, starts, ends, softmax_scale=SCALE, softmax_aux=per_head)
        out_shared = ragged_decode_tpu(q, k, v, starts, ends, softmax_scale=SCALE, softmax_aux=shared)
        ref_per_head = _dense_reference(q, k, v, starts, ends, per_head)
        ref_shared = _dense_reference(q, k, v, starts, ends, shared)
        assert _rel(ref_shared, ref_per_head) > EFFECT_REL
        assert _rel(out_shared, out_per_head) > EFFECT_REL
        assert _rel(out_per_head, ref_per_head) < PARITY_REL
        assert _rel(out_shared, ref_shared) < PARITY_REL

    def test_multi_block_and_padding(self):
        """Sink is folded once, after the last KV block, regardless of block size."""
        q, k, v, starts, ends = _inputs()
        sinks = _sink_values((HQ, 2))
        ref = _dense_reference(q, k, v, starts, ends, sinks)
        for block in (128, 256):
            out = ragged_decode_tpu(
                q,
                k,
                v,
                starts,
                ends,
                softmax_scale=SCALE,
                softmax_aux=sinks,
                fwd_params=FwdParams(q_blocksize=block, kv_blocksize=block),
            )
            assert _rel(out, ref) < PARITY_REL, (block, _rel(out, ref))

    def test_empty_row_with_disabled_sink_is_zero_not_nan(self):
        """An empty decode row (start == end) with a -inf sink carries no mass: output 0, never NaN (r10 S02-4)."""
        q, k, v, _, _ = _inputs()
        starts = jnp.array([0, 100], dtype=jnp.int32)
        ends = jnp.array([0, 480], dtype=jnp.int32)  # row 0 is empty
        sinks = jnp.full((HQ,), -jnp.inf, dtype=jnp.float32)
        out = np.asarray(ragged_decode_tpu(q, k, v, starts, ends, softmax_scale=SCALE, softmax_aux=sinks))
        assert np.isfinite(out).all(), "sink fold produced non-finite values"
        np.testing.assert_array_equal(out[0], np.zeros_like(out[0]))
        # The non-empty row must still match the dense reference (a -inf sink contributes nothing).
        ref = np.asarray(_dense_reference(q, k, v, starts, ends, None))
        assert _rel(out[1], ref[1]) < PARITY_REL

    def test_large_sink_dominates(self):
        """A sink far above every logit drives the output toward zero (m_final = sink)."""
        q, k, v, starts, ends = _inputs()
        sinks = jnp.full((HQ,), 60.0, dtype=jnp.float32)
        out = ragged_decode_tpu(q, k, v, starts, ends, softmax_scale=SCALE, softmax_aux=sinks)
        ref = _dense_reference(q, k, v, starts, ends, sinks)
        assert float(jnp.max(jnp.abs(ref))) < 1e-3
        np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=0, atol=1e-3)


def _dense_window_softcap_reference(q, k, v, starts, ends, window, soft_cap):
    """Dense f32 decode: query at ``end - 1``, keys in ``[q_pos - left, q_pos + right]``, tanh-capped logits."""
    hp = jax.lax.Precision.HIGHEST
    kf = jnp.repeat(k, HQ // HKV, axis=2)
    vf = jnp.repeat(v, HQ // HKV, axis=2)
    logits = jnp.einsum("bhd,bshd->bhs", q * SCALE, kf, precision=hp)
    if soft_cap is not None:
        logits = soft_cap * jnp.tanh(logits / soft_cap)
    pos = jnp.arange(S)[None, None, :]
    valid = (pos >= starts[:, None, None]) & (pos < ends[:, None, None])
    if window is not None:
        q_pos = (ends - 1)[:, None, None]
        valid &= (pos >= q_pos - window[0]) & (pos <= q_pos + window[1])
    logits = jnp.where(valid, logits, -1e30)
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("bhs,bshd->bhd", weights, vf, precision=hp)


@pytest.mark.parametrize(
    ("window", "soft_cap"),
    [((32, 0), None), (None, 0.5), ((100, 0), 0.5)],
    ids=["window", "softcap", "window+softcap"],
)
def test_sliding_window_and_soft_cap_are_applied(window, soft_cap):
    """The Pallas path used to accept but ignore ``sliding_window`` / ``logits_soft_cap`` (r10 S02-2)."""
    q, k, v, starts, ends = _inputs()
    out = ragged_decode_tpu(q, k, v, starts, ends, softmax_scale=SCALE, sliding_window=window, logits_soft_cap=soft_cap)
    ref = _dense_window_softcap_reference(q, k, v, starts, ends, window, soft_cap)
    plain = _dense_window_softcap_reference(q, k, v, starts, ends, None, None)
    err, effect = _rel(out, ref), _rel(plain, ref)
    print(f"window={window} soft_cap={soft_cap}: rel err {err:.3e}; option effect {effect:.3e}")
    assert effect > EFFECT_REL, effect  # negative control: ignoring the options would fail
    assert err < PARITY_REL, err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
