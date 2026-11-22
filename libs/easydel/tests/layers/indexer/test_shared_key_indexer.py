# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""SharedKeyIndexer scoring / top-k / candidate blocks and the selection helpers vs NumPy references."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.indexer import SharedKeyIndexer, SharedKeyIndexerConfig, select_candidate_blocks, selection_bias
from jax import numpy as jnp

HIDDEN = 8
Q_LORA = 6
HEADS = 4
DIM = 4
KEY_IN = 5
EPS = 1e-6
TOPK = 3
BLOCK = 2
TOPK_BLOCKS = 2
QUERY_SCALE = 2.0
MASK_MIN = float(np.finfo(np.float32).min)


def _rotate_queries(q):
    return q * QUERY_SCALE


def _build(*, owns_keys=True, is_candidate_source=False, uses_candidates=False, ratio=2, data=None):
    config = SharedKeyIndexerConfig(
        hidden_size=HIDDEN,
        q_lora_rank=Q_LORA,
        index_n_heads=HEADS,
        index_head_dim=DIM,
        index_topk=TOPK,
        ratio=ratio,
        key_input_dim=KEY_IN,
        owns_keys=owns_keys,
        is_candidate_source=is_candidate_source,
        uses_candidates=uses_candidates,
        candidate_topk_blocks=TOPK_BLOCKS,
        candidate_block_size=BLOCK,
        rms_norm_eps=EPS,
    )
    module = SharedKeyIndexer(
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    if data is not None:
        module.q_b_proj.weight.value = jnp.asarray(data["q_b"])
        module.weights_proj.weight.value = jnp.asarray(data["weights"])
        if owns_keys:
            module.wk.weight.value = jnp.asarray(data["wk"])
            module.k_norm.weight.value = jnp.asarray(data["k_norm"])
    return module


def _draw(seed, positions, entries):
    rng = np.random.default_rng(seed)
    batch, seq = positions.shape

    def normal(*shape):
        return rng.normal(size=shape).astype(np.float32)

    return {
        "q_b": normal(Q_LORA, HEADS * DIM),
        "weights": normal(HIDDEN, HEADS),
        "wk": normal(KEY_IN, DIM),
        "k_norm": rng.uniform(0.5, 1.5, size=(DIM,)).astype(np.float32),
        "hidden": normal(batch, seq, HIDDEN),
        "q_res": normal(batch, seq, Q_LORA),
        "keys": normal(batch, entries, DIM),
    }


def _reference_scores(data, positions, ratio):
    """float64 ``sum_h relu(q_h . k_t) * w_h`` with unreachable entries at ``-inf``."""
    batch, seq = positions.shape
    q = (data["q_res"].astype(np.float64) @ data["q_b"].astype(np.float64)).reshape(batch, seq, HEADS, DIM)
    q = q * QUERY_SCALE
    weights = (data["hidden"].astype(np.float64) @ data["weights"].astype(np.float64)) * (DIM**-0.5 * HEADS**-0.5)
    logits = np.einsum("bshd,btd->bsht", q, data["keys"].astype(np.float64))
    scores = (np.maximum(logits, 0.0) * weights[..., None]).sum(axis=2)
    lens = (positions + 1) // ratio
    reachable = np.arange(data["keys"].shape[1])[None, None, :] < lens[..., None]
    return np.where(reachable, scores, -np.inf), lens


def _block_scores(scores, lens, block_size):
    batch, seq, width = scores.shape
    pad = (-width) % block_size
    padded = np.pad(scores, ((0, 0), (0, 0), (0, pad)), constant_values=-np.inf)
    num_blocks = (width + pad) // block_size
    blocks = padded.reshape(batch, seq, num_blocks, block_size).max(axis=-1)
    last = (lens - 1) // block_size
    return np.where(np.arange(num_blocks)[None, None, :] == last[..., None], np.inf, blocks)


def _reference_blocks(scores, lens, topk_blocks, block_size):
    batch, seq, width = scores.shape
    blocks = _block_scores(scores, lens, block_size)
    keep = np.zeros(blocks.shape, bool)
    for b in range(batch):
        for s in range(seq):
            order = np.argsort(-blocks[b, s], kind="stable")[: min(topk_blocks, blocks.shape[-1])]
            keep[b, s, order] = blocks[b, s, order] > -np.inf
    return np.repeat(keep, block_size, axis=-1)[..., :width]


def _reference_block_ids(scores, lens, topk_blocks, block_size):
    """Published candidate block ids ``[B, S, min(topk_blocks, num_blocks)]``, ascending, ``-1``-padded."""
    batch, seq, width = scores.shape
    num_blocks = -(-width // block_size)
    keep = _reference_blocks(scores, lens, topk_blocks, block_size)
    keep = np.pad(keep, ((0, 0), (0, 0), (0, num_blocks * block_size - width)))
    keep = keep.reshape(batch, seq, num_blocks, block_size).any(axis=-1)
    kb = min(topk_blocks, num_blocks)
    out = np.full((batch, seq, kb), -1, np.int32)
    for b in range(batch):
        for s in range(seq):
            ids = np.flatnonzero(keep[b, s])
            out[b, s, : len(ids)] = ids
    return out


def _reference_topk(scores, lens, k):
    batch, seq, width = scores.shape
    k = min(k, width)
    out = np.empty((batch, seq, k), np.int32)
    for b in range(batch):
        for s in range(seq):
            idx = np.sort(np.argsort(-scores[b, s], kind="stable")[:k])
            out[b, s] = np.where(idx < lens[b, s], idx, -1)
    return out


def _boundary_gap(values, k):
    """Smallest gap between the k-th and (k+1)-th best non ``-inf`` value of any row (guards tie-breaking)."""
    k = min(k, values.shape[-1])
    gap = np.inf
    for row in values.reshape(-1, values.shape[-1]):
        live = np.sort(row[row > -np.inf])[::-1]
        if len(live) > k:
            gap = min(gap, live[k - 1] - live[k])
    return gap


def _find_case(positions, entries, *, ratio=2, restrict=False, check_blocks=False):
    """Deterministically pick a seed whose top-k (and block) boundaries are tie-free."""
    for seed in range(300):
        data = _draw(seed, positions, entries)
        scores, lens = _reference_scores(data, positions, ratio)
        if restrict:
            data["candidates"] = np.random.default_rng(10_000 + seed).random(scores.shape) > 0.4
            scores = np.where(data["candidates"], scores, -np.inf)
        gaps = [_boundary_gap(scores, TOPK)]
        if check_blocks:
            gaps.append(_boundary_gap(_block_scores(scores, lens, BLOCK), TOPK_BLOCKS))
        if min(gaps) > 1e-3:
            return data, scores, lens
    pytest.fail("no tie-free seed found")


def _positions(kind):
    if kind == "prefill":
        return np.tile(np.arange(9, dtype=np.int32), (2, 1))
    return np.array([[5], [3]], np.int32)  # one decode token per row at different positions


@pytest.mark.parametrize("kind", ["prefill", "decode"])
def test_topk_matches_numpy(kind):
    positions = _positions(kind)
    data, scores, lens = _find_case(positions, entries=6)
    module = _build(data=data)
    expected = _reference_topk(scores, lens, TOPK)
    args = (
        jnp.asarray(data["hidden"]),
        jnp.asarray(data["q_res"]),
        jnp.asarray(data["keys"]),
        None,
        jnp.asarray(positions),
    )
    run = jax.jit(lambda h, q, k, c, p: module(h, q, k, _rotate_queries, c, p))
    for topk, published in (module(args[0], args[1], args[2], _rotate_queries, None, args[4]), run(*args)):
        assert topk.dtype == jnp.int32
        assert topk.shape == expected.shape
        assert published is None
        np.testing.assert_array_equal(np.asarray(topk), expected)


def test_candidate_source_publishes_block_mask_without_restricting_itself():
    positions = _positions("prefill")
    data, scores, lens = _find_case(positions, entries=6, check_blocks=True)
    module = _build(data=data, is_candidate_source=True)
    topk, published = module(
        jnp.asarray(data["hidden"]),
        jnp.asarray(data["q_res"]),
        jnp.asarray(data["keys"]),
        _rotate_queries,
        None,
        jnp.asarray(positions),
    )
    np.testing.assert_array_equal(np.asarray(topk), _reference_topk(scores, lens, TOPK))
    assert published.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(published), _reference_block_ids(scores, lens, TOPK_BLOCKS, BLOCK))


@pytest.mark.parametrize("kind", ["prefill", "decode"])
def test_candidate_user_only_scores_inside_published_blocks(kind):
    """A candidate user ranks only entries inside the source's published blocks (prefill and decode rows)."""
    positions = _positions(kind)
    data, scores, lens = _find_case(positions, entries=6, check_blocks=True)
    args = (jnp.asarray(data["hidden"]), jnp.asarray(data["q_res"]), jnp.asarray(data["keys"]))
    _, published = _build(data=data, is_candidate_source=True)(*args, _rotate_queries, None, jnp.asarray(positions))
    mask = _reference_blocks(scores, lens, TOPK_BLOCKS, BLOCK)
    topk, out_published = _build(data=data, uses_candidates=True)(
        *args, _rotate_queries, published, jnp.asarray(positions)
    )
    assert out_published is None
    restricted = np.where(mask, scores, -np.inf)
    expected = _reference_topk(restricted, lens, TOPK)
    # Picks that only exist because the restricted row ran out of candidates are -inf ties in the reference; the
    # bounded selector reports them as -1 (documented).
    finite = np.take_along_axis(restricted, np.maximum(expected, 0), axis=-1) > -np.inf
    np.testing.assert_array_equal(np.asarray(topk), np.where(finite, expected, -1))


def test_selection_ignores_entries_beyond_reach():
    """Garbage (NaN / huge) keys past each query's reach never influence selection."""
    positions = _positions("decode")
    data, scores, lens = _find_case(positions, entries=6)
    keys = np.array(data["keys"])
    for b in range(keys.shape[0]):
        keys[b, int(lens[b].max()) :] = np.nan
    module = _build(data=data)
    topk, _ = module(
        jnp.asarray(data["hidden"]),
        jnp.asarray(data["q_res"]),
        jnp.asarray(keys),
        _rotate_queries,
        None,
        jnp.asarray(positions),
    )
    np.testing.assert_array_equal(np.asarray(topk), _reference_topk(scores, lens, TOPK))


@pytest.mark.parametrize("candidate_source", [False, True])
def test_no_entries_selects_nothing(candidate_source):
    positions = _positions("prefill")
    data = _draw(0, positions, 1)
    module = _build(data=data, is_candidate_source=candidate_source)
    hidden, q_res = jnp.asarray(data["hidden"]), jnp.asarray(data["q_res"])
    for index_k in (None, jnp.zeros((2, 0, DIM), jnp.float32)):
        topk, published = module(hidden, q_res, index_k, _rotate_queries, None, jnp.asarray(positions))
        assert topk.shape == (2, 9, 0) and topk.dtype == jnp.int32
        if candidate_source:
            assert published.shape == (2, 9, 0) and published.dtype == jnp.int32
        else:
            assert published is None


def test_keys_from_latent_matches_numpy():
    data = _draw(1, _positions("decode"), 3)
    module = _build(data=data)
    latent = np.random.default_rng(9).normal(size=(2, 5, KEY_IN)).astype(np.float32)
    projected = latent.astype(np.float64) @ data["wk"].astype(np.float64)
    normed = projected / np.sqrt(np.mean(projected**2, axis=-1, keepdims=True) + EPS) * data["k_norm"]
    got = module.keys_from_latent(jnp.asarray(latent), lambda keys: keys * 3.0)
    assert got.shape == (2, 5, DIM)
    np.testing.assert_allclose(np.asarray(got), normed * 3.0, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    ("owns_keys", "expected"),
    [
        (True, {"q_b_proj.weight", "weights_proj.weight", "wk.weight", "k_norm.weight"}),
        (False, {"q_b_proj.weight", "weights_proj.weight"}),
    ],
)
def test_parameter_leaves(owns_keys, expected):
    module = _build(owns_keys=owns_keys)
    _, state = spx.export(module)
    assert {path for collection, path, _ in state.items() if collection == "parameters"} == expected
    assert (module.wk is None) == (not owns_keys)
    assert (module.k_norm is None) == (not owns_keys)


def test_candidate_blocks_hand_checked_examples():
    inf = -np.inf
    scores = jnp.asarray([[[1.0, 5.0, 2.0, 2.5, 9.0, 3.0]]], jnp.float32)
    full = jnp.asarray([[6]], jnp.int32)
    # The block holding the newest reachable entry (4, 5) is always kept.
    np.testing.assert_array_equal(
        np.asarray(select_candidate_blocks(scores, full, 1, 2)), [[[False, False, False, False, True, True]]]
    )
    np.testing.assert_array_equal(
        np.asarray(select_candidate_blocks(scores, full, 2, 2)), [[[True, True, False, False, True, True]]]
    )
    masked = jnp.asarray([[[1.0, 5.0, 2.0, inf, inf, inf]]], jnp.float32)
    short = jnp.asarray([[3]], jnp.int32)
    expected = [[[True, True, True, True, False, False]]]  # block granularity: entry 3 rides along with entry 2
    for blocks in (2, 3):  # unreachable (-inf) blocks are dropped even when the budget allows them
        np.testing.assert_array_equal(np.asarray(select_candidate_blocks(masked, short, blocks, 2)), expected)


@pytest.mark.parametrize("width", [6, 7])
def test_candidate_blocks_match_numpy_including_empty_and_ragged_width(width):
    rng = np.random.default_rng(width)
    lens = rng.integers(0, width + 1, size=(2, 5)).astype(np.int32)
    lens[0, 0] = 0  # no reachable entry at all
    raw = rng.normal(size=(2, 5, width)).astype(np.float32)
    scores = np.where(np.arange(width)[None, None, :] < lens[..., None], raw, -np.inf).astype(np.float32)
    got = select_candidate_blocks(jnp.asarray(scores), jnp.asarray(lens), TOPK_BLOCKS, BLOCK)
    assert got.shape == scores.shape and got.dtype == jnp.bool_
    np.testing.assert_array_equal(
        np.asarray(got), _reference_blocks(scores.astype(np.float64), lens, TOPK_BLOCKS, BLOCK)
    )
    assert not np.asarray(got)[0, 0].any()


def test_selection_bias_opens_selected_entries_only():
    topk = np.array([[[0, 2, -1], [-1, -1, -1]]], np.int32)
    got = selection_bias(jnp.asarray(topk), 4)
    assert got.shape == (1, 1, 2, 4) and got.dtype == jnp.float32
    expected = np.full((1, 1, 2, 4), MASK_MIN, np.float32)
    expected[0, 0, 0, [0, 2]] = 0.0
    np.testing.assert_array_equal(np.asarray(got), expected)


def test_config_validation():
    with pytest.raises(ValueError):
        SharedKeyIndexerConfig(HIDDEN, Q_LORA, HEADS, DIM, TOPK, ratio=0, key_input_dim=KEY_IN)
    with pytest.raises(ValueError):
        SharedKeyIndexerConfig(HIDDEN, Q_LORA, HEADS, DIM, 0, ratio=2, key_input_dim=KEY_IN)
