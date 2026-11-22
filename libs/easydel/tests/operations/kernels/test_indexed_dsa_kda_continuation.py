"""Indexed DSA and segmented KDA continuation regressions (accelerator only)."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.operations import OperationMetadata
from easydel.operations.kernels.glm_moe_dsa_indexer import GlmMoeDsaIndexerOp
from easydel.operations.kernels.kda import KernelDeltaAttnOp
from jax import numpy as jnp
from jax.sharding import Mesh


def metadata():
    return OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        _stored_mesh=Mesh(np.asarray(jax.devices()[:1]), ("tp",)),
    )


def test_indexed_dsa_continuation_matches_full_topk_with_unequal_offsets():
    op = GlmMoeDsaIndexerOp(SimpleNamespace())
    keys = np.array([[[2.0], [4.0], [1.0], [6.0], [3.0]], [[3.0], [1.0], [8.0], [2.0], [4.0]]], np.float32)
    storage = jnp.full((2, 8, 1), 999.0, jnp.float32)
    # Each row has its own physical write offsets and logical positions. Unused
    # slots contain larger scores than any real key, so missing validity is observable.
    slots = np.array([[1, 2, 3, 4, 5], [0, 1, 2, 3, 4]], np.int32)
    positions = np.full((2, 8), -1, np.int32)
    valid = np.zeros((2, 8), bool)
    for start, end in [(0, 3), (3, 5)]:
        for b in range(2):
            valid[b, slots[b, start:end]] = True
            positions[b, slots[b, start:end]] = np.arange(start, end)
        q_len = end - start
        out = op.forward_native(
            jnp.ones((2, q_len, 1, 1)),
            jnp.asarray(keys[:, start:end]),
            jnp.ones((2, q_len, 1)),
            jnp.broadcast_to(jnp.arange(start, end), (2, q_len)),
            qk_rope_head_dim=0,
            index_topk=2,
            softmax_scale=1.0,
            cached_keys=storage,
            use_cache=True,
            cache_positions=jnp.asarray(slots[:, start:end]),
            key_valid_mask=jnp.asarray(valid),
            query_positions=jnp.broadcast_to(jnp.arange(start, end), (2, q_len)),
            key_positions=jnp.asarray(positions),
            precision=jax.lax.Precision.HIGHEST,
        )
        full = op.forward_native(
            jnp.ones((2, end, 1, 1)),
            jnp.asarray(keys[:, :end]),
            jnp.ones((2, end, 1)),
            jnp.broadcast_to(jnp.arange(end), (2, end)),
            qk_rope_head_dim=0,
            index_topk=2,
            softmax_scale=1.0,
            query_positions=jnp.broadcast_to(jnp.arange(end), (2, end)),
            key_positions=jnp.broadcast_to(jnp.arange(end), (2, end)),
            precision=jax.lax.Precision.HIGHEST,
        )
        actual = np.asarray(out.topk_mask)
        for b in range(2):
            expected = np.zeros((q_len, 8), bool)
            expected[:, slots[b, :end]] = np.asarray(full.topk_mask)[b, start:end]
            np.testing.assert_array_equal(actual[b], expected)
            for q, logical in enumerate(range(start, end)):
                ranked = np.argsort(-keys[b, : logical + 1, 0], kind="stable")[:2]
                np.testing.assert_array_equal(np.flatnonzero(actual[b, q]), np.sort(slots[b, ranked]))
        storage = out.cached_keys
        assert storage.shape == (2, 8, 1)
        assert storage.dtype == jnp.float32
    for b in range(2):
        np.testing.assert_array_equal(np.asarray(storage)[b, slots[b]], keys[b])


def test_indexed_dsa_reset_padding_segments_and_reordered_positions():
    out = GlmMoeDsaIndexerOp(SimpleNamespace()).forward_native(
        jnp.ones((1, 3, 1, 1)),
        jnp.array([[[2.0], [9.0], [100.0]]]),
        jnp.ones((1, 3, 1)),
        jnp.array([[10, 11, 12]]),
        qk_rope_head_dim=0,
        index_topk=2,
        softmax_scale=1.0,
        cached_keys=jnp.full((1, 6, 1), 999.0),
        use_cache=True,
        reset_cache=True,
        cache_positions=jnp.array([[4, 1, -1]]),
        key_valid_mask=jnp.array([[False, True, False, False, True, False]]),
        query_positions=jnp.array([[10, 11, -1]]),
        key_positions=jnp.array([[-1, 11, -1, -1, 10, -1]]),
        segment_ids=jnp.array([[7, 8, -1]]),
        key_segment_ids=jnp.array([[-1, 8, -1, -1, 7, -1]]),
        precision=jax.lax.Precision.HIGHEST,
    )
    np.testing.assert_array_equal(out.cached_keys[0, :, 0], [0, 9, 0, 0, 2, 0])
    np.testing.assert_array_equal(out.topk_indices[0], [[4, -1], [1, -1], [-1, -1]])
    np.testing.assert_array_equal(
        out.topk_mask[0],
        [
            [False, False, False, False, True, False],
            [False, True, False, False, False, False],
            [False, False, False, False, False, False],
        ],
    )


def reference_kda(query, key, value, beta, decay, initial, segments, continuation):
    """Independent NumPy delta-rule recurrence, including the public reset policy."""
    state = initial.copy()
    outputs = np.zeros_like(value)
    previous = np.full(query.shape[0], -1)
    for t in range(query.shape[1]):
        for b in range(query.shape[0]):
            segment = segments[b, t]
            if segment < 0:
                if not continuation:
                    state[b] = 0
                    previous[b] = -1
                continue
            if segment != previous[b] and (not continuation or previous[b] >= 0):
                state[b] = 0
            previous[b] = segment
            q = query[b, t] / np.sqrt(np.sum(query[b, t] ** 2, axis=-1, keepdims=True) + 1e-6)
            k = key[b, t] / np.sqrt(np.sum(key[b, t] ** 2, axis=-1, keepdims=True) + 1e-6)
            g = decay[b, t]
            if g.ndim == 1:
                g = g[:, None]
            state[b] *= np.exp(g)[..., None]
            memory = np.sum(state[b] * k[..., None], axis=-2)
            delta = (value[b, t] - memory) * beta[b, t, :, None]
            state[b] += k[..., None] * delta[:, None, :]
            outputs[b, t] = np.sum(state[b] * q[..., None], axis=-2) / np.sqrt(query.shape[-1])
    return outputs, state


@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("continuation", [False, True])
def test_segmented_kda_continuation_matches_independent_reference(per_channel, continuation):
    rng = np.random.default_rng(741)
    q, k, v = (rng.normal(size=(2, 7, 1, 2)).astype(np.float32) for _ in range(3))
    beta = rng.uniform(0.1, 0.8, (2, 7, 1)).astype(np.float32)
    decay = rng.uniform(-0.2, -0.02, (2, 7, 1, 2) if per_channel else (2, 7, 1)).astype(np.float32)
    initial = rng.normal(size=(2, 1, 2, 2)).astype(np.float32)
    segments = np.array([[-1, 2, -1, 2, 9, 9, -1], [-1, -1, -1, -1, -1, -1, -1]], np.int32)
    expected_out, expected_state = reference_kda(q, k, v, beta, decay, initial, segments, continuation)
    out = KernelDeltaAttnOp(metadata()).forward_native(
        jnp.asarray(q),
        jnp.asarray(k),
        jnp.asarray(v),
        jnp.asarray(beta),
        jnp.asarray(decay),
        recurrent_state=jnp.asarray(initial),
        segment_ids=jnp.asarray(segments),
        chunk_size=2,
        continue_initial_state=continuation,
    )
    np.testing.assert_allclose(out.attention_outputs, expected_out, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(out.recurrent_state, expected_state, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("per_channel", [False, True])
def test_kda_multitoken_cached_segments_match_full_prefill(per_channel):
    rng = np.random.default_rng(581)
    q, k, v = (rng.normal(size=(1, 8, 1, 2)).astype(np.float32) for _ in range(3))
    beta = rng.uniform(0.1, 0.8, (1, 8, 1)).astype(np.float32)
    decay = rng.uniform(-0.2, -0.02, (1, 8, 1, 2) if per_channel else (1, 8, 1)).astype(np.float32)
    segments = np.zeros((1, 8), np.int32)
    op = KernelDeltaAttnOp(metadata())
    full = op.forward_native(
        *(jnp.asarray(x) for x in (q, k, v, beta, decay)), segment_ids=jnp.asarray(segments), chunk_size=2
    )
    state = None
    chunks = []
    for start, end in [(0, 3), (3, 6), (6, 8)]:
        out = op.forward_native(
            *(jnp.asarray(x[:, start:end]) for x in (q, k, v, beta, decay)),
            recurrent_state=state,
            segment_ids=jnp.asarray(segments[:, start:end]),
            chunk_size=2,
            continue_initial_state=True,
        )
        chunks.append(np.asarray(out.attention_outputs))
        state = out.recurrent_state
    np.testing.assert_allclose(np.concatenate(chunks, axis=1), full.attention_outputs, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(state, full.recurrent_state, atol=2e-6, rtol=2e-6)
