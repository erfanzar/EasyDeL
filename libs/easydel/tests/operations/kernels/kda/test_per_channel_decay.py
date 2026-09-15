# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Per-channel-decay KDA kernels (GLM-5-Next) — parity and dispatch tests.

The GLM-5-Next forget gate produces a per-channel log-decay of shape
``(batch, seq, num_heads, head_dim)`` instead of Kimi Linear's per-head
``(batch, seq, num_heads)``. These tests pin the three per-channel kernels
(chunked, recurrent, single-step decode) against each other and against an
independent numpy reference recurrence, plus the fused gate function.
"""

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.infra import EasyDeLBaseConfig
from easydel.operations._operation_impl import OperationMetadata
from easydel.operations.kernels.kda import (
    KernelDeltaAttnOp,
    _chunk_kda_fwd,
    _chunk_kda_per_channel_fwd,
    _recurrent_kda_fwd,
    _recurrent_kda_per_channel_fwd,
    _single_step_kda_per_channel_fwd_bthd,
    fused_kda_gate_per_channel,
    l2norm,
)

ATOL_CHUNKED = 5e-4
ATOL_TIGHT = 1e-5


def _reference_recurrence(query, key, value, beta, decay, use_qk_l2norm=True):
    """Independent numpy reference of the per-channel KDA recurrence.

    Transcribed from the HF ``recurrent_kimi_delta_attention`` loop (numpy,
    no easydel helpers beyond shape conventions) so the JAX kernels are not
    compared against themselves.
    """
    B, H, L, K = query.shape
    V = value.shape[-1]
    query = np.asarray(query, dtype=np.float64)
    key = np.asarray(key, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    decay = np.asarray(decay, dtype=np.float64)
    if use_qk_l2norm:
        query = query / np.sqrt((query * query).sum(-1, keepdims=True) + 1e-6)
        key = key / np.sqrt((key * key).sum(-1, keepdims=True) + 1e-6)
    query = query * (1.0 / np.sqrt(K))

    state = np.zeros((B, H, K, V))
    outputs = np.zeros((B, H, L, V))
    for t in range(L):
        g = np.exp(decay[:, :, t][..., None])  # (B, H, K, 1)
        b = beta[:, :, t][..., None]
        state = state * g
        kv_mem = (state * key[:, :, t][..., None]).sum(-2)
        delta = (value[:, :, t] - kv_mem) * b
        state = state + key[:, :, t][..., None] * delta[..., None, :]
        outputs[:, :, t] = (state * query[:, :, t][..., None]).sum(-2)
    return outputs.transpose(0, 1, 2, 3)


def _make_inputs(seed=0, batch=2, heads=3, seq=24, head_dim=8, d_state=6, dtype=jnp.float32):
    rng = jax.random.key(seed)
    query = jax.random.normal(rng, (batch, heads, seq, head_dim), dtype=jnp.float32) * 0.5
    key = jax.random.normal(jax.random.fold_in(rng, 1), (batch, heads, seq, head_dim), dtype=jnp.float32) * 0.5
    value = jax.random.normal(jax.random.fold_in(rng, 2), (batch, heads, seq, d_state), dtype=jnp.float32) * 0.5
    beta = jax.nn.sigmoid(jax.random.normal(jax.random.fold_in(rng, 3), (batch, heads, seq), dtype=jnp.float32))
    decay = (
        -5.0
        * jax.nn.sigmoid(jax.random.normal(jax.random.fold_in(rng, 4), (batch, heads, seq, head_dim), dtype=jnp.float32))
    ).astype(dtype)
    return query, key, value, beta, decay


def test_fused_gate_per_channel_matches_reference():
    """The fused gate equals the HF formula evaluated directly."""
    batch, seq, heads, head_dim = 2, 5, 3, 8
    qkv_dim = heads * head_dim
    rng = jax.random.key(11)
    gate = jax.random.normal(rng, (batch, seq, qkv_dim), dtype=jnp.float32) * 2.0
    a_log = jax.random.normal(jax.random.fold_in(rng, 1), (heads,), dtype=jnp.float32) * 0.3
    dt_bias = jax.random.normal(jax.random.fold_in(rng, 2), (qkv_dim,), dtype=jnp.float32) * 0.5
    lower_bound = -5.0

    out = fused_kda_gate_per_channel(gate, a_log, dt_bias, lower_bound=lower_bound)

    g = (np.asarray(gate) + np.asarray(dt_bias).reshape(1, 1, qkv_dim)).reshape(batch, seq, heads, head_dim)
    rate = np.exp(np.asarray(a_log)).reshape(1, 1, heads, 1)
    expected = lower_bound / (1.0 + np.exp(-rate * g))
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-5, rtol=1e-5)
    assert out.shape == (batch, seq, heads, head_dim)
    assert out.dtype == jnp.float32
    # Safe gate keeps the decay in [lower_bound, 0) — exp(g) contracts state.
    assert float(out.min()) >= lower_bound - 1e-6
    assert float(out.max()) <= 0.0


def test_fused_gate_softplus_variant():
    """Without a lower bound the gate falls back to the softplus form."""
    batch, seq, heads, head_dim = 2, 4, 2, 4
    qkv_dim = heads * head_dim
    rng = jax.random.key(12)
    gate = jax.random.normal(rng, (batch, seq, qkv_dim), dtype=jnp.float32)
    a_log = jax.random.normal(jax.random.fold_in(rng, 1), (heads,), dtype=jnp.float32) * 0.2
    dt_bias = jax.random.normal(jax.random.fold_in(rng, 2), (qkv_dim,), dtype=jnp.float32) * 0.3

    out = fused_kda_gate_per_channel(gate, a_log, dt_bias, lower_bound=None)

    g = (np.asarray(gate) + np.asarray(dt_bias).reshape(1, 1, qkv_dim)).reshape(batch, seq, heads, head_dim)
    rate = np.exp(np.asarray(a_log)).reshape(1, 1, heads, 1)
    softplus = np.where(g > 20.0, g, np.log1p(np.exp(g)))
    np.testing.assert_allclose(np.asarray(out), -rate * softplus, atol=1e-5, rtol=1e-5)
    assert float(out.max()) <= 0.0


def test_chunked_matches_recurrent_per_channel():
    """The chunked kernel agrees with the sequential recurrence."""
    query, key, value, beta, decay = _make_inputs(seq=48)
    out_rec, state_rec = _recurrent_kda_per_channel_fwd(query, key, value, beta, decay, use_qk_l2norm=True)
    out_chunk, state_chunk = _chunk_kda_per_channel_fwd(
        query, key, value, beta, decay, chunk_size=16, use_qk_l2norm=True
    )
    assert out_rec.shape == (2, 3, 48, 6)
    np.testing.assert_allclose(np.asarray(out_chunk), np.asarray(out_rec), atol=ATOL_CHUNKED, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(state_chunk), np.asarray(state_rec), atol=ATOL_CHUNKED, rtol=1e-3)


def test_recurrent_matches_numpy_reference():
    """The recurrent kernel matches an independent numpy recurrence."""
    query, key, value, beta, decay = _make_inputs(seq=16)
    out, state = _recurrent_kda_per_channel_fwd(query, key, value, beta, decay, use_qk_l2norm=True)
    expected = _reference_recurrence(query, key, value, beta, decay, use_qk_l2norm=True)
    np.testing.assert_allclose(np.asarray(out), expected, atol=ATOL_TIGHT, rtol=1e-4)
    assert state.shape == (2, 3, 8, 6)
    assert state.dtype == jnp.float32


def test_single_step_decode_matches_recurrent():
    """Feeding the batched sequence step-by-step reproduces the recurrent run."""
    query, key, value, beta, decay = _make_inputs(seq=10)

    out_full, state_full = _recurrent_kda_per_channel_fwd(query, key, value, beta, decay, use_qk_l2norm=True)

    state = jnp.zeros((2, 3, 8, 6), dtype=jnp.float32)
    for t in range(10):
        out_t, state = _single_step_kda_per_channel_fwd_bthd(
            query=query.transpose(0, 2, 1, 3)[:, t][..., None, :, :],
            key=key.transpose(0, 2, 1, 3)[:, t][..., None, :, :],
            value=value.transpose(0, 2, 1, 3)[:, t][..., None, :, :],
            beta=beta.transpose(0, 2, 1)[:, t][..., None, :],
            decay=decay.transpose(0, 2, 1, 3)[:, t][..., None, :, :],
            recurrent_state=state,
            use_qk_l2norm=True,
        )
        np.testing.assert_allclose(np.asarray(out_t[:, 0]), np.asarray(out_full[:, :, t]), atol=ATOL_TIGHT, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(state), np.asarray(state_full), atol=ATOL_TIGHT, rtol=1e-4)


def test_op_dispatch_per_channel_decay():
    """KernelDeltaAttnOp routes 4-dim decay through the per-channel kernels."""
    query, key, value, beta, decay = _make_inputs(seq=32)

    # BTHD layout at the op boundary: (B, S, H, K) and decay (B, S, H, K).
    query_bthd = query.transpose(0, 2, 1, 3)
    key_bthd = key.transpose(0, 2, 1, 3)
    value_bthd = value.transpose(0, 2, 1, 3)
    beta_bthd = beta.transpose(0, 2, 1)
    decay_bthd = decay.transpose(0, 2, 1, 3)

    metadata = OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        base_config=EasyDeLBaseConfig(),
    )
    op = KernelDeltaAttnOp(metadata)

    output = op(
        query=query_bthd,
        key=key_bthd,
        value=value_bthd,
        beta=beta_bthd,
        decay=decay_bthd,
        chunk_size=16,
    )
    assert output.attention_outputs.shape == (2, 32, 3, 6)
    assert output.recurrent_state.shape == (2, 3, 8, 6)

    expected, expected_state = _recurrent_kda_per_channel_fwd(query, key, value, beta, decay, use_qk_l2norm=True)
    np.testing.assert_allclose(
        np.asarray(output.attention_outputs).transpose(0, 2, 1, 3),
        np.asarray(expected),
        atol=ATOL_CHUNKED,
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(output.recurrent_state), np.asarray(expected_state), atol=ATOL_CHUNKED, rtol=1e-3
    )


def test_op_dispatch_decode_per_channel():
    """Decode mode (seq_len=1 + recurrent_state) uses the per-channel step."""
    query, key, value, beta, decay = _make_inputs(seq=1)
    rng = jax.random.key(21)
    state_in = jax.random.normal(rng, (2, 3, 8, 6), dtype=jnp.float32)

    metadata = OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        base_config=EasyDeLBaseConfig(),
    )
    op = KernelDeltaAttnOp(metadata)

    output = op(
        query=query.transpose(0, 2, 1, 3),  # (B, 1, H, K)
        key=key.transpose(0, 2, 1, 3),
        value=value.transpose(0, 2, 1, 3),
        beta=beta.transpose(0, 2, 1),
        decay=decay.transpose(0, 2, 1, 3),
        recurrent_state=state_in,
    )
    assert output.attention_outputs.shape == (2, 1, 3, 6)
    assert output.recurrent_state.shape == (2, 3, 8, 6)

    expected, expected_state = _single_step_kda_per_channel_fwd_bthd(
        query=query.transpose(0, 2, 1, 3),
        key=key.transpose(0, 2, 1, 3),
        value=value.transpose(0, 2, 1, 3),
        beta=beta.transpose(0, 2, 1),
        decay=decay.transpose(0, 2, 1, 3),
        recurrent_state=state_in,
        use_qk_l2norm=True,
    )
    np.testing.assert_allclose(np.asarray(output.attention_outputs), np.asarray(expected), atol=ATOL_TIGHT, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(output.recurrent_state), np.asarray(expected_state), atol=ATOL_TIGHT, rtol=1e-5
    )


def test_per_head_path_untouched():
    """3-dim decay still routes through the original per-head kernels."""
    rng = jax.random.key(31)
    batch, heads, seq, head_dim, d_state = 2, 3, 16, 8, 6
    query = jax.random.normal(rng, (batch, heads, seq, head_dim), dtype=jnp.float32) * 0.5
    key = jax.random.normal(jax.random.fold_in(rng, 1), (batch, heads, seq, head_dim), dtype=jnp.float32) * 0.5
    value = jax.random.normal(jax.random.fold_in(rng, 2), (batch, heads, seq, d_state), dtype=jnp.float32) * 0.5
    beta = jax.nn.sigmoid(jax.random.normal(jax.random.fold_in(rng, 3), (batch, heads, seq), dtype=jnp.float32))
    decay = jax.random.normal(jax.random.fold_in(rng, 4), (batch, heads, seq), dtype=jnp.float32) * 0.01

    out_rec, state_rec = _recurrent_kda_fwd(query, key, value, beta, decay, use_qk_l2norm=True)
    out_chunk, state_chunk = _chunk_kda_fwd(query, key, value, beta, decay, chunk_size=8, use_qk_l2norm=True)
    np.testing.assert_allclose(np.asarray(out_chunk), np.asarray(out_rec), atol=ATOL_CHUNKED, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(state_chunk), np.asarray(state_rec), atol=ATOL_CHUNKED, rtol=1e-3)


def test_l2norm_shared_by_paths():
    """Q/K L2-normalization applies before decay, matching the HF kernel."""
    x = jnp.array([[[3.0, 4.0]]])
    normalized = l2norm(x, axis=-1, eps=1e-6)
    np.testing.assert_allclose(np.asarray(normalized), [[[0.6, 0.8]]], atol=1e-6)


# ---------------------------------------------------------------------------
# Chunked-with-initial-state parity (resume-from-cache decode/prefill path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq", [5, 16, 17, 32, 33, 50])
def test_chunked_per_channel_matches_recurrent_with_initial_state(seq):
    """``_chunk_kda_per_channel_fwd`` honors a nonzero ``initial_state``.

    Sequence lengths not divisible by the chunk size (17, 33, 50 with chunk
    16) exercise the zero-pad tail: the resume-from-cache decode path feeds a
    carried-over state through the chunked kernel, so the chunked result must
    match the sequential recurrent ground truth from the same state.
    """
    query, key, value, beta, decay = _make_inputs(seq=seq)
    init = jax.random.normal(jax.random.key(101), (2, 3, 8, 6), dtype=jnp.float32) * 0.3

    out_rec, state_rec = _recurrent_kda_per_channel_fwd(
        query, key, value, beta, decay, initial_state=init, use_qk_l2norm=True
    )
    out_chunk, state_chunk = _chunk_kda_per_channel_fwd(
        query, key, value, beta, decay, chunk_size=16, initial_state=init, use_qk_l2norm=True
    )
    np.testing.assert_allclose(np.asarray(out_chunk), np.asarray(out_rec), atol=ATOL_CHUNKED, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(state_chunk), np.asarray(state_rec), atol=ATOL_CHUNKED, rtol=1e-3)


@pytest.mark.parametrize("seq", [5, 17, 33, 50])
def test_chunked_per_head_matches_recurrent_with_initial_state(seq):
    """The per-head chunked kernel honors a nonzero ``initial_state`` too."""
    rng = jax.random.key(103)
    query = jax.random.normal(rng, (2, 3, seq, 8), dtype=jnp.float32) * 0.5
    key = jax.random.normal(jax.random.fold_in(rng, 1), (2, 3, seq, 8), dtype=jnp.float32) * 0.5
    value = jax.random.normal(jax.random.fold_in(rng, 2), (2, 3, seq, 6), dtype=jnp.float32) * 0.5
    beta = jax.nn.sigmoid(jax.random.normal(jax.random.fold_in(rng, 3), (2, 3, seq), dtype=jnp.float32))
    decay = jax.random.normal(jax.random.fold_in(rng, 4), (2, 3, seq), dtype=jnp.float32) * 0.01
    init = jax.random.normal(jax.random.fold_in(rng, 5), (2, 3, 8, 6), dtype=jnp.float32) * 0.3

    out_rec, state_rec = _recurrent_kda_fwd(query, key, value, beta, decay, initial_state=init, use_qk_l2norm=True)
    out_chunk, state_chunk = _chunk_kda_fwd(
        query, key, value, beta, decay, chunk_size=16, initial_state=init, use_qk_l2norm=True
    )
    np.testing.assert_allclose(np.asarray(out_chunk), np.asarray(out_rec), atol=ATOL_CHUNKED, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(state_chunk), np.asarray(state_rec), atol=ATOL_CHUNKED, rtol=1e-3)


def test_initial_state_actually_advances_outputs():
    """Control: the chunked kernel does not silently ignore ``initial_state``.

    A vacuous parity (both kernels dropping the state) would still pass the
    tests above, so pin that a nonzero carried state shifts the outputs. The
    final state is checked with a weak decay — under the default strong decay
    (g in [-5, 0)) the carried state's influence underflows float32 within a
    few dozen tokens and both runs end bitwise-identical.
    """
    query, key, value, beta, decay = _make_inputs(seq=33)
    init = jax.random.normal(jax.random.key(107), (2, 3, 8, 6), dtype=jnp.float32) * 0.5

    out_chunk_init, _ = _chunk_kda_per_channel_fwd(
        query, key, value, beta, decay, chunk_size=16, initial_state=init, use_qk_l2norm=True
    )
    out_chunk_zero, _ = _chunk_kda_per_channel_fwd(
        query, key, value, beta, decay, chunk_size=16, initial_state=None, use_qk_l2norm=True
    )
    assert float(jnp.abs(out_chunk_init - out_chunk_zero).max()) > 1e-3

    out_rec_init, _ = _recurrent_kda_per_channel_fwd(
        query, key, value, beta, decay, initial_state=init, use_qk_l2norm=True
    )
    out_rec_zero, _ = _recurrent_kda_per_channel_fwd(query, key, value, beta, decay, use_qk_l2norm=True)
    assert float(jnp.abs(out_rec_init - out_rec_zero).max()) > 1e-3

    # Weak decay keeps the initial state's fingerprint in the final state.
    weak_decay = decay * 0.002
    _, state_chunk_init = _chunk_kda_per_channel_fwd(
        query, key, value, beta, weak_decay, chunk_size=16, initial_state=init, use_qk_l2norm=True
    )
    _, state_chunk_zero = _chunk_kda_per_channel_fwd(
        query, key, value, beta, weak_decay, chunk_size=16, initial_state=None, use_qk_l2norm=True
    )
    assert float(jnp.abs(state_chunk_init - state_chunk_zero).max()) > 1e-3
    _, state_rec_init = _recurrent_kda_per_channel_fwd(
        query, key, value, beta, weak_decay, initial_state=init, use_qk_l2norm=True
    )
    _, state_rec_zero = _recurrent_kda_per_channel_fwd(
        query, key, value, beta, weak_decay, use_qk_l2norm=True
    )
    assert float(jnp.abs(state_rec_init - state_rec_zero).max()) > 1e-3


# ---------------------------------------------------------------------------
# Explicit-reject contract: per-channel kernels require an explicit decay
# ---------------------------------------------------------------------------


def test_op_rejects_per_channel_decay_without_decay():
    """``per_channel_decay=True`` with ``decay=None`` must raise ValueError.

    The per-channel kernels have no zero-decay fallback (zero decay would
    mean "no memory decay", a different operation), so the op is contracted
    to reject the combination explicitly instead of silently zero-filling.
    """
    query, key, value, beta, _decay = _make_inputs(seq=8)
    metadata = OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        base_config=EasyDeLBaseConfig(),
    )
    op = KernelDeltaAttnOp(metadata)

    with pytest.raises(ValueError, match="per_channel_decay=True requires explicit"):
        op(
            query=query.transpose(0, 2, 1, 3),
            key=key.transpose(0, 2, 1, 3),
            value=value.transpose(0, 2, 1, 3),
            beta=beta.transpose(0, 2, 1),
            decay=None,
            per_channel_decay=True,
        )
    # The 4-dim inference path must not bypass the guard either: forcing the
    # flag with no decay is rejected before any shape-dependent work.
    with pytest.raises(ValueError, match="per_channel_decay=True requires explicit"):
        op.forward_native(
            query=query.transpose(0, 2, 1, 3),
            key=key.transpose(0, 2, 1, 3),
            value=value.transpose(0, 2, 1, 3),
            beta=beta.transpose(0, 2, 1),
            decay=None,
            per_channel_decay=True,
        )


def test_op_per_head_none_decay_zero_fill_kept():
    """Contrast: the per-head family still zero-fills ``decay=None``.

    Passing no decay must equal passing an explicit all-zeros decay there —
    the "no memory decay" semantics that made the per-channel reject necessary
    is exactly what the per-head None fallback implements.
    """
    query, key, value, beta, _decay = _make_inputs(seq=16)
    metadata = OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        base_config=EasyDeLBaseConfig(),
    )
    op = KernelDeltaAttnOp(metadata)
    bthd = lambda x: x.transpose(0, 2, 1, 3)  # noqa: E731

    out_none = op(
        query=bthd(query), key=bthd(key), value=bthd(value), beta=beta.transpose(0, 2, 1), decay=None
    )
    out_zero = op(
        query=bthd(query),
        key=bthd(key),
        value=bthd(value),
        beta=beta.transpose(0, 2, 1),
        decay=jnp.zeros((2, 16, 3), dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        np.asarray(out_none.attention_outputs), np.asarray(out_zero.attention_outputs), atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(out_none.recurrent_state), np.asarray(out_zero.recurrent_state), atol=0.0, rtol=0.0
    )


# ---------------------------------------------------------------------------
# Packed-segment reset (SFT sequence-packing training path)
# ---------------------------------------------------------------------------


def _segment_spans(seg_list: list[int]) -> list[tuple[int, int]]:
    """Maximal runs of equal non-negative ids as ``(start, end)`` spans."""
    spans = []
    start = 0
    for i in range(1, len(seg_list) + 1):
        if i == len(seg_list) or seg_list[i] != seg_list[i - 1]:
            if seg_list[start] >= 0:
                spans.append((start, i))
            start = i
    return spans


@pytest.mark.parametrize(
    "segments",
    [
        [0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 2, 2, 2, -1, -1],
    ],
)
def test_recurrent_per_channel_segment_reset_matches_per_segment_loops(segments):
    """Packed segments must reset the state at every document boundary.

    The packed-segment run must equal running the same recurrent kernel once
    per segment with a zeroed initial state — this is the SFT-packing training
    path: document n+1 must never see document n's recurrent memory.
    """
    seq = len(segments)
    query, key, value, beta, decay = _make_inputs(seq=seq, seed=7, batch=1)
    segment_ids = jnp.asarray([segments], dtype=jnp.int32)

    out_packed, state_packed = _recurrent_kda_per_channel_fwd(
        query, key, value, beta, decay, use_qk_l2norm=True, segment_ids=segment_ids
    )

    for start, end in _segment_spans(segments):
        out_seg, _ = _recurrent_kda_per_channel_fwd(
            query[:, :, start:end],
            key[:, :, start:end],
            value[:, :, start:end],
            beta[:, :, start:end],
            decay[:, :, start:end],
            initial_state=None,
            use_qk_l2norm=True,
        )
        np.testing.assert_allclose(
            np.asarray(out_packed[:, :, start:end]), np.asarray(out_seg), atol=ATOL_TIGHT, rtol=1e-5
        )

    # the final state is the last valid segment's end state; a row ending in
    # pads leaves the state zeroed (padded steps clear it by contract).
    if segments[-1] >= 0:
        last_start, last_end = _segment_spans(segments)[-1]
        _, state_last = _recurrent_kda_per_channel_fwd(
            query[:, :, last_start:last_end],
            key[:, :, last_start:last_end],
            value[:, :, last_start:last_end],
            beta[:, :, last_start:last_end],
            decay[:, :, last_start:last_end],
            initial_state=None,
            use_qk_l2norm=True,
        )
        np.testing.assert_allclose(np.asarray(state_packed), np.asarray(state_last), atol=ATOL_TIGHT, rtol=1e-5)
    else:
        assert float(jnp.abs(state_packed).max()) == 0.0


def test_recurrent_per_channel_segment_padding_zeroes_outputs_and_state():
    """Padded (``-1``) positions emit zero and leave zero state behind."""
    segments = [0, 0, -1, 1, 1, -1]
    query, key, value, beta, decay = _make_inputs(seq=len(segments), seed=13, batch=1)
    segment_ids = jnp.asarray([segments], dtype=jnp.int32)

    out_packed, state_packed = _recurrent_kda_per_channel_fwd(
        query, key, value, beta, decay, use_qk_l2norm=True, segment_ids=segment_ids
    )

    # padded positions produce exactly-zero outputs
    assert float(jnp.abs(out_packed[:, :, 2]).max()) == 0.0
    assert float(jnp.abs(out_packed[:, :, 5]).max()) == 0.0
    # a trailing pad zeroes the carried state entirely
    assert float(jnp.abs(state_packed).max()) == 0.0

    # valid spans still match isolated per-segment runs from zero state
    for start, end in _segment_spans(segments):
        out_seg, _ = _recurrent_kda_per_channel_fwd(
            query[:, :, start:end],
            key[:, :, start:end],
            value[:, :, start:end],
            beta[:, :, start:end],
            decay[:, :, start:end],
            use_qk_l2norm=True,
        )
        np.testing.assert_allclose(
            np.asarray(out_packed[:, :, start:end]), np.asarray(out_seg), atol=ATOL_TIGHT, rtol=1e-5
        )


def test_recurrent_per_head_segment_reset_matches_per_segment_loops():
    """The original per-head kernel honors the same packed-segment contract."""
    rng = jax.random.key(41)
    seq = 6
    query = jax.random.normal(rng, (1, 2, seq, 8), dtype=jnp.float32) * 0.5
    key = jax.random.normal(jax.random.fold_in(rng, 1), (1, 2, seq, 8), dtype=jnp.float32) * 0.5
    value = jax.random.normal(jax.random.fold_in(rng, 2), (1, 2, seq, 6), dtype=jnp.float32) * 0.5
    beta = jax.nn.sigmoid(jax.random.normal(jax.random.fold_in(rng, 3), (1, 2, seq), dtype=jnp.float32))
    decay = jax.random.normal(jax.random.fold_in(rng, 4), (1, 2, seq), dtype=jnp.float32) * 0.01
    segments = [0, 0, 0, 1, 1, 1]
    segment_ids = jnp.asarray([segments], dtype=jnp.int32)

    out_packed, state_packed = _recurrent_kda_fwd(query, key, value, beta, decay, use_qk_l2norm=True,
                                                  segment_ids=segment_ids)
    for start, end in _segment_spans(segments):
        out_seg, _ = _recurrent_kda_fwd(
            query[:, :, start:end],
            key[:, :, start:end],
            value[:, :, start:end],
            beta[:, :, start:end],
            decay[:, :, start:end],
            use_qk_l2norm=True,
        )
        np.testing.assert_allclose(
            np.asarray(out_packed[:, :, start:end]), np.asarray(out_seg), atol=ATOL_TIGHT, rtol=1e-5
        )
    _, state_last = _recurrent_kda_fwd(
        query[:, :, 3:], key[:, :, 3:], value[:, :, 3:], beta[:, :, 3:], decay[:, :, 3:], use_qk_l2norm=True
    )
    np.testing.assert_allclose(np.asarray(state_packed), np.asarray(state_last), atol=ATOL_TIGHT, rtol=1e-5)
