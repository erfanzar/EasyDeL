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

"""Physical-cache and packed-document contracts; accelerator numerics only."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.layers.indexer import IndexerConfig, SparseIndexer, TokenIndexer, TokenIndexerConfig
from jax import numpy as jnp


def _pool_indexer():
    indexer = SparseIndexer(
        IndexerConfig(
            kind="pool",
            hidden_size=4,
            q_input_dim=4,
            index_n_heads=1,
            index_head_dim=4,
            index_topk=3,
            kpool_size=3,
            head_reduction="uniform",
            query_source="hidden",
            rope_style="none",
            packed_state="key_gate_valid",
            select_tail=True,
        ),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    indexer.wq_b.weight.value = jnp.eye(4, dtype=jnp.float32)
    indexer.wk.weight.value = jnp.eye(4, dtype=jnp.float32)
    # Nonconstant per-slot/channel APE detects failure to restart slots at a
    # document boundary, even if token indices are later intersected correctly.
    indexer.index_kpool_compress_ape.value = jnp.asarray(_ape())
    return indexer


def _ape():
    return np.arange(12, dtype=np.float32).reshape(3, 4) * 0.17


def _packed_reference(hidden, valid):
    mean = hidden.mean(axis=-1, keepdims=True)
    centered = hidden - mean
    keys = centered / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + 1e-6)
    gates = np.broadcast_to(hidden.sum(axis=-1, keepdims=True), hidden.shape)
    return np.concatenate([keys, gates, valid[..., None]], axis=-1).astype(np.float32)


def _reference_mask(
    hidden_queries, packed, query_positions, key_positions, query_segments, key_segments, query_valid, pairwise=None
):
    """Independent NumPy learned-pool scoring and original-coordinate expansion."""
    batch, queries, _ = hidden_queries.shape
    mask = np.zeros((batch, queries, packed.shape[1]), dtype=bool)
    for b in range(batch):
        for q in range(queries):
            if not query_valid[b, q] or query_segments[b, q] < 0:
                continue
            visible = (
                (packed[b, :, -1] >= 0.5)
                & (key_segments[b] == query_segments[b, q])
                & (key_positions[b] >= 0)
                & (key_positions[b] <= query_positions[b, q])
            )
            if pairwise is not None:
                visible &= pairwise[b, q]
            visible_ids = np.flatnonzero(visible)
            groups = []
            for start in range(0, len(visible_ids) - 2, 3):
                members = visible_ids[start : start + 3]
                logits = packed[b, members, 4:8] + _ape()
                weights = np.exp(logits - logits.max(axis=0, keepdims=True))
                weights /= weights.sum(axis=0, keepdims=True)
                groups.append((members, (weights * packed[b, members, :4]).sum(axis=0)))
            if groups:
                scores = np.asarray([hidden_queries[b, q] @ summary / 2 for _, summary in groups])
                picked = int(np.argmax(scores))
                mask[b, q, groups[picked][0]] = True
            tail = len(visible_ids) % 3
            if tail:
                mask[b, q, visible_ids[-tail:]] = True
    return mask


def _assert_indices_and_mask(output, expected):
    np.testing.assert_array_equal(np.asarray(output.topk_mask), expected)
    actual_indices = np.asarray(output.topk_indices)
    for b in range(expected.shape[0]):
        for q in range(expected.shape[1]):
            picked = actual_indices[b, q]
            np.testing.assert_array_equal(np.sort(picked[picked >= 0]), np.flatnonzero(expected[b, q]))


def test_packed_pooling_restarts_unaligned_documents_before_ranking_and_ape():
    indexer = _pool_indexer()
    hidden = np.random.default_rng(403).normal(size=(4, 13, 4)).astype(np.float32)
    segments = np.broadcast_to([0] * 5 + [1] * 8, (4, 13)).copy()
    valid = np.ones((4, 13), dtype=bool)
    positions = np.broadcast_to(np.arange(13), (4, 13))
    packed = _packed_reference(hidden, valid)
    expected = _reference_mask(hidden, packed, positions, positions, segments, segments, valid)
    output = indexer.forward(jnp.asarray(hidden), segment_ids=jnp.asarray(segments))
    _assert_indices_and_mask(output, expected)

    separate = indexer.forward(jnp.asarray(hidden[:, 5:]), segment_ids=jnp.ones((4, 8), jnp.int32))
    np.testing.assert_array_equal(np.asarray(output.topk_mask)[:, 5:, 5:], np.asarray(separate.topk_mask))
    assert not np.asarray(output.topk_mask)[:, 5:, :5].any()
    # A foreign document must not spend this document's top-k budget, even
    # after a counterfactual changes all of its keys, gates and scores.
    changed = hidden.copy()
    changed[:, :5] = np.random.default_rng(404).normal(size=(4, 5, 4)) * 100
    counterfactual = indexer.forward(jnp.asarray(changed), segment_ids=jnp.asarray(segments))
    np.testing.assert_array_equal(np.asarray(counterfactual.topk_mask)[:, 5:], np.asarray(output.topk_mask)[:, 5:])


def test_jitted_fixed_capacity_pool_cache_matches_reference_at_unequal_row_offsets():
    indexer = _pool_indexer()
    capacity = 16
    state = jnp.zeros((4, capacity, 9), jnp.float32)
    expected_state = np.zeros((4, capacity, 9), np.float32)
    key_segments = np.full((4, capacity), -1, np.int32)
    starts = np.asarray([0, 2, 1, 3])
    key_positions = np.broadcast_to(np.arange(capacity) + 20, (4, capacity)).copy()
    rng = np.random.default_rng(405)

    @jax.jit
    def step(hidden, cache, slots, validity, q_positions, q_segments, kv_segments):
        return indexer.forward(
            hidden,
            cached_packed=cache,
            attention_mask=validity,
            cache_positions=slots,
            query_positions=q_positions,
            key_positions=jnp.asarray(key_positions),
            segment_ids=q_segments,
            key_segment_ids=kv_segments,
        )

    offset = 0
    for length in (4, 3, 2):
        hidden = rng.normal(size=(4, length, 4)).astype(np.float32)
        slots = starts[:, None] + offset + np.arange(length)
        query_valid = np.ones((4, length), dtype=bool)
        # Keep an internal hole and an invalid query, not only end padding.
        if offset == 0:
            query_valid[1, 1] = False
        segments = np.where(offset + np.arange(length)[None, :] < 5, 0, 1)
        segments = np.broadcast_to(segments, (4, length)).copy()
        current = _packed_reference(hidden, query_valid)
        for b in range(4):
            expected_state[b, slots[b, query_valid[b]]] = current[b, query_valid[b]]
            key_segments[b, slots[b, query_valid[b]]] = segments[b, query_valid[b]]
        output = step(
            jnp.asarray(hidden),
            state,
            jnp.asarray(slots),
            jnp.asarray(query_valid),
            jnp.asarray(slots + 20),
            jnp.asarray(segments),
            jnp.asarray(key_segments),
        )
        expected = _reference_mask(
            hidden, expected_state, slots + 20, key_positions, segments, key_segments, query_valid
        )
        _assert_indices_and_mask(output, expected)
        assert output.packed_state.shape == (4, capacity, 9)
        np.testing.assert_allclose(np.asarray(output.packed_state), expected_state, rtol=2e-5, atol=2e-5)
        state = output.packed_state
        offset += length


def test_two_stage_pool_api_consumes_cache_aligned_state_without_reprojecting_history():
    indexer = _pool_indexer()
    hidden = np.random.default_rng(406).normal(size=(4, 10, 4)).astype(np.float32)
    valid = np.ones((4, 10), dtype=bool)
    valid[:, 2] = False
    segments = np.broadcast_to([0] * 5 + [1] * 5, (4, 10)).copy()
    queries, state = indexer.project_state(jnp.asarray(hidden), attention_mask=jnp.asarray(valid))
    output = indexer.select_from_state(
        jnp.asarray(hidden[:, 6:]),
        queries[:, 6:],
        state,
        attention_mask=jnp.asarray(valid[:, 6:]),
        query_positions=jnp.broadcast_to(jnp.arange(6, 10), (4, 4)),
        segment_ids=jnp.asarray(segments[:, 6:]),
        key_segment_ids=jnp.asarray(segments),
    )
    packed = _packed_reference(hidden, valid)
    expected = _reference_mask(
        hidden[:, 6:],
        packed,
        np.broadcast_to(np.arange(6, 10), (4, 4)),
        np.broadcast_to(np.arange(10), (4, 10)),
        segments[:, 6:],
        segments,
        valid[:, 6:],
    )
    _assert_indices_and_mask(output, expected)
    np.testing.assert_allclose(np.asarray(output.packed_state), packed, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("layout", ["keys", "key_gate_valid"])
def test_fixed_capacity_token_strategy_excludes_unused_history_and_foreign_segments(layout):
    config = IndexerConfig(
        hidden_size=4,
        q_input_dim=4,
        index_n_heads=1,
        index_head_dim=4,
        index_topk=2,
        head_reduction="uniform",
        query_source="hidden",
        rope_style="none",
        packed_state=layout,
    )
    # key_gate_valid is a learned-pool representation, whose gate parameters
    # only exist for pool strategies; the token strategy's public keys layout
    # is the applicable representation for fixed-capacity token scoring.
    if layout == "key_gate_valid":
        config = config.with_changes(kind="pool", kpool_size=1, select_tail=True)
    indexer = SparseIndexer(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(1)
    )
    indexer.wq_b.weight.value = jnp.eye(4, dtype=jnp.float32)
    indexer.wk.weight.value = jnp.eye(4, dtype=jnp.float32)
    hidden = np.random.default_rng(407).normal(size=(4, 4, 4)).astype(np.float32)
    slots = np.broadcast_to(np.arange(2, 6), (4, 4))
    validity = np.zeros((4, 8), dtype=bool)
    validity[:, 2:6] = True
    segments = np.broadcast_to([0, 0, 1, 1], (4, 4))
    key_segments = np.broadcast_to([-1, -1, 0, 0, 1, 1, -1, -1], (4, 8))
    output = indexer.forward(
        jnp.asarray(hidden),
        cached_packed=jnp.zeros((4, 8, indexer.packed_state_dim), jnp.float32),
        cache_positions=jnp.asarray(slots),
        key_valid_mask=jnp.asarray(validity),
        segment_ids=jnp.asarray(segments),
        key_segment_ids=jnp.asarray(key_segments),
    )
    expected_picks = [[[2], [2, 3], [4], [4, 5]]] * 4
    for b, rows in enumerate(expected_picks):
        for q, expected in enumerate(rows):
            picked = np.asarray(output.topk_indices)[b, q]
            np.testing.assert_array_equal(np.sort(picked[picked >= 0]), expected)
    assert output.packed_state.shape == (4, 8, indexer.packed_state_dim)


class _IdentityNorm(spx.Module):
    """Identity reference layer isolates public token-cache semantics."""

    def forward(self, hidden_states):
        return hidden_states


def _token_indexer():
    indexer = TokenIndexer(
        TokenIndexerConfig(4, 4, 1, 4, 2, 0),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(2),
        base_config=EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 1)),
        norm_factory=_IdentityNorm,
    )
    indexer.wq_b.weight.value = jnp.eye(4, dtype=jnp.float32)
    indexer.wk.weight.value = jnp.eye(4, dtype=jnp.float32)
    indexer.kernels_proj.weight.value = jnp.ones((4, 1), jnp.float32)
    return indexer


def test_token_indexer_explicit_multitoken_continuation_retains_history_and_legacy_reset():
    indexer = _token_indexer()
    hidden = np.random.default_rng(408).normal(size=(4, 5, 4)).astype(np.float32)
    positions = jnp.broadcast_to(jnp.arange(5), (4, 5))
    prefill = indexer.forward(jnp.asarray(hidden[:, :2]), None, positions[:, :2], use_cache=True)
    continued = indexer.forward(
        jnp.asarray(hidden[:, 2:]),
        None,
        positions[:, 2:],
        cached_keys=prefill.cached_keys,
        use_cache=True,
        reset_cache=False,
    )
    np.testing.assert_allclose(np.asarray(continued.cached_keys), hidden, rtol=2e-5, atol=2e-5)
    scores = np.maximum(np.einsum("bqd,bkd->bqk", hidden[:, 2:], hidden) / 2, 0)
    scores *= hidden[:, 2:].sum(axis=-1, keepdims=True)
    for b in range(4):
        for q in range(3):
            expected = np.argsort(-scores[b, q, : q + 3], kind="stable")[:2]
            np.testing.assert_array_equal(np.asarray(continued.topk_indices)[b, q], expected)
    # The legacy API treats a multi-token call as a fresh prefill unless
    # an explicit continuation policy or indexed insertion slots are supplied.
    reset = indexer.forward(
        jnp.asarray(hidden[:, 2:]),
        None,
        positions[:, 2:],
        cached_keys=prefill.cached_keys,
        use_cache=True,
    )
    np.testing.assert_allclose(np.asarray(reset.cached_keys), hidden[:, 2:], rtol=2e-5, atol=2e-5)


def test_token_indexer_indexed_cache_preserves_multitoken_history_and_applies_visibility():
    indexer = _token_indexer()
    capacity = 12
    state = jnp.zeros((4, capacity, 4), jnp.float32)
    expected_state = np.zeros((4, capacity, 4), np.float32)
    validity = np.zeros((4, capacity), dtype=bool)
    key_segments = np.full((4, capacity), -1, np.int32)
    key_positions = np.broadcast_to(np.arange(capacity) + 20, (4, capacity))
    starts = np.asarray([0, 2, 1, 3])
    offset = 0
    rng = np.random.default_rng(409)
    for length in (4, 2):
        hidden = rng.normal(size=(4, length, 4)).astype(np.float32)
        slots = starts[:, None] + offset + np.arange(length)
        segments = np.broadcast_to(np.where(offset + np.arange(length) < 3, 0, 1), (4, length))
        for b in range(4):
            expected_state[b, slots[b]] = hidden[b]
            validity[b, slots[b]] = True
            key_segments[b, slots[b]] = segments[b]
        result = indexer.forward(
            jnp.asarray(hidden),
            None,
            jnp.asarray(slots),
            cached_keys=state,
            use_cache=True,
            cache_positions=jnp.asarray(slots),
            key_valid_mask=jnp.asarray(validity),
            query_positions=jnp.asarray(slots + 20),
            key_positions=jnp.asarray(key_positions),
            segment_ids=jnp.asarray(segments),
            key_segment_ids=jnp.asarray(key_segments),
        )
        assert result.cached_keys.shape == (4, capacity, 4)
        np.testing.assert_allclose(np.asarray(result.cached_keys), expected_state, rtol=2e-5, atol=2e-5)
        scores = np.maximum(np.einsum("bqd,bkd->bqk", hidden, expected_state) / 2, 0)
        scores *= hidden.sum(axis=-1, keepdims=True)
        expected_mask = np.zeros((4, length, capacity), dtype=bool)
        for b in range(4):
            for q in range(length):
                visible = validity[b] & (key_segments[b] == segments[b, q]) & (key_positions[b] <= slots[b, q] + 20)
                candidates = np.flatnonzero(visible)
                order = np.argsort(-scores[b, q, candidates], kind="stable")[:2]
                expected_mask[b, q, candidates[order]] = True
        np.testing.assert_array_equal(np.asarray(result.topk_mask), expected_mask)
        selection = indexer.selection_from_output(result)
        np.testing.assert_array_equal(np.asarray(selection.to_mask(capacity)), expected_mask)
        for b in range(4):
            for q in range(length):
                picked = np.asarray(selection.indices)[b, q]
                np.testing.assert_array_equal(np.sort(picked[picked >= 0]), np.flatnonzero(expected_mask[b, q]))
        state = result.cached_keys
        offset += length


@pytest.mark.parametrize("additive", [False, True])
@pytest.mark.parametrize("segments", [False, True])
def test_sliding_pairwise_visibility_regroups_pools_in_attention_time_coordinates(additive, segments):
    indexer = _pool_indexer()
    hidden = np.random.default_rng(410).normal(size=(4, 5, 4)).astype(np.float32)
    query_positions = np.broadcast_to([3, 4], (4, 2))
    key_positions = np.broadcast_to(np.arange(5), (4, 5))
    visible = (key_positions[:, None, :] <= query_positions[..., None]) & (
        key_positions[:, None, :] > query_positions[..., None] - 3
    )
    mask = np.where(visible, 0.0, -np.inf).astype(np.float32) if additive else visible
    projected, state = indexer.project_state(jnp.asarray(hidden))
    metadata = {}
    if segments:
        metadata = {"segment_ids": jnp.zeros((4, 2), jnp.int32), "key_segment_ids": jnp.zeros((4, 5), jnp.int32)}

    @jax.jit
    def select(packed, pairwise):
        return indexer.select_from_state(
            jnp.asarray(hidden[:, 3:]),
            projected[:, 3:],
            packed,
            pairwise_mask=pairwise,
            query_positions=jnp.asarray(query_positions),
            key_positions=jnp.asarray(key_positions),
            **metadata,
        )

    output = select(state, jnp.asarray(mask))
    _assert_indices_and_mask(output, visible)
    np.testing.assert_array_equal(np.asarray(output.packed_state), np.asarray(state))


@pytest.mark.cpu_ok
def test_pool_config_rejects_disabling_required_tail_selection():
    config = IndexerConfig(kind="pool", index_topk=3, kpool_size=3, packed_state="key_gate_valid", select_tail=True)
    # Pool selection requires a live tail so early queries cannot become
    # all-masked. Exercise the invalid option as a public validation contract,
    # not as a numerical pooling mode that cannot be constructed.
    with pytest.raises(ValueError, match="select_tail"):
        config.with_changes(select_tail=False)


@pytest.mark.parametrize("additive", [False, True])
def test_pairwise_holes_use_same_query_local_boundaries_for_pool_ranking_ape_and_tail(additive):
    indexer = _pool_indexer()
    hidden = np.random.default_rng(411).normal(size=(4, 11, 4)).astype(np.float32)
    projected, state = indexer.project_state(jnp.asarray(hidden))
    visible = np.zeros((4, 2, 11), dtype=bool)
    visible[:, 0, [1, 2, 4, 5, 7, 8, 9, 10]] = True  # Two complete pools and a two-token tail.
    visible[:, 1, [0, 3, 4, 6, 7, 8, 10]] = True  # Different boundaries and one-token tail.
    mask = np.where(visible, 0.0, -np.inf).astype(np.float32) if additive else visible
    positions = np.full((4, 2), 10, dtype=np.int32)
    key_positions = np.broadcast_to(np.arange(11), (4, 11))
    output = indexer.select_from_state(
        jnp.asarray(hidden[:, -2:]),
        projected[:, -2:],
        state,
        pairwise_mask=jnp.asarray(mask),
        query_positions=jnp.asarray(positions),
    )
    expected = _reference_mask(
        hidden[:, -2:],
        _packed_reference(hidden, np.ones((4, 11), bool)),
        positions,
        key_positions,
        np.zeros((4, 2), np.int32),
        np.zeros((4, 11), np.int32),
        np.ones((4, 2), bool),
        pairwise=visible,
    )
    _assert_indices_and_mask(output, expected)


@pytest.mark.parametrize("mask_kind", ["causal", "boolean", "additive", "additive_min"])
def test_legacy_token_adapter_excludes_masked_topk_fillers_without_indexed_metadata(mask_kind):
    indexer = _token_indexer()
    hidden = jnp.broadcast_to(jnp.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]), (4, 2, 4))
    positions = jnp.broadcast_to(jnp.arange(2), (4, 2))
    mask = None
    expected = np.broadcast_to([[True, False], [True, True]], (4, 2, 2)).copy()
    if mask_kind != "causal":
        expected[:, 1, 0] = False
        mask = jnp.asarray([[True, True], [False, True]])
        if mask_kind.startswith("additive"):
            blocked = jnp.finfo(jnp.float32).min if mask_kind == "additive_min" else -jnp.inf
            mask = jnp.where(mask, 0.0, blocked)
        mask = jnp.broadcast_to(mask, (4, 2, 2))
    output = indexer.forward(hidden, None, positions, attention_mask=mask)
    # The operation's documented legacy raw indices remain intact; only its
    # authoritative dense mask and the common adapter suppress filler lanes.
    np.testing.assert_array_equal(np.sort(np.asarray(output.topk_indices), axis=-1), np.broadcast_to([0, 1], (4, 2, 2)))
    selection = indexer.selection_from_output(output)
    np.testing.assert_array_equal(np.asarray(output.topk_mask), expected)
    np.testing.assert_array_equal(np.asarray(selection.to_mask(2)), expected)
    for b in range(4):
        for q in range(2):
            picked = np.asarray(selection.indices)[b, q]
            np.testing.assert_array_equal(np.sort(picked[picked >= 0]), np.flatnonzero(expected[b, q]))
