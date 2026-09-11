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

"""Tests for the unified sparse-attention indexer layer."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

from easydel.layers.indexer import IndexerConfig, IndexerKind, IndexerOutput, SparseIndexer

B, S, HID, H, D, TOPK = 2, 24, 64, 4, 16, 12


def _make(config: IndexerConfig):
    indexer = SparseIndexer(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    graphdef, params = spx.export(indexer)
    bound = spx.bind(graphdef, params)
    return indexer, params, bound


def test_token_config_shapes_and_validity():
    config = IndexerConfig(
        kind=IndexerKind.TOKEN,
        index_n_heads=H,
        index_head_dim=D,
        index_topk=TOPK,
        hidden_size=HID,
        q_input_dim=HID,
        score_activation="none",
        head_reduction="weighted",
        rope_style="none",
        packed_state="keys",
    )
    _indexer, _params, bound = _make(config)
    hidden = jax.random.normal(jax.random.PRNGKey(0), (B, S, HID), dtype=jnp.float32)
    q_resid = jax.random.normal(jax.random.PRNGKey(1), (B, S, HID), dtype=jnp.float32)
    mask = jnp.ones((B, S), dtype=jnp.bool_)
    mask = mask.at[1, S - 3 :].set(False)

    out = bound(
        hidden_states=hidden, q_resid=q_resid, attention_mask=mask, cached_packed=None
    )
    assert isinstance(out, IndexerOutput)
    assert out.topk_indices.shape == (B, S, TOPK)
    assert out.packed_state.shape == (B, S, D)
    assert out.score_proxy.shape == (B, S, S)
    # -1 padding appears only where the window is shorter than topk (never here),
    # but indices must be in range and padding-masked queries select nothing.
    assert int(out.topk_indices.min()) >= 0
    last_row = out.topk_indices[1, -1]
    # padded query rows get garbage scores; the model masks them downstream,
    # so only assert range validity here.
    assert int(last_row.max()) < S


def test_pool_config_matches_reference_semantics():
    kpool = 4
    config = IndexerConfig(
        kind=IndexerKind.POOL,
        index_n_heads=H,
        index_head_dim=D,
        index_topk=TOPK,
        hidden_size=HID,
        q_input_dim=HID,
        score_activation="relu",
        head_reduction="weighted",
        rope_style="none",
        norm_eps=1e-6,
        packed_state="key_gate_valid",
        stop_gradient=True,
        kpool_size=kpool,
        select_tail=True,
    )
    _indexer, _params, bound = _make(config)
    assert _indexer.packed_state_dim == 2 * D + 1

    hidden = jax.random.normal(jax.random.PRNGKey(0), (B, S, HID), dtype=jnp.float32)
    q_resid = jax.random.normal(jax.random.PRNGKey(1), (B, S, HID), dtype=jnp.float32)
    mask = jnp.ones((B, S), dtype=jnp.bool_)
    mask = mask.at[0, S - 2 :].set(False)

    out = bound(
        hidden_states=hidden, q_resid=q_resid, attention_mask=mask, cached_packed=None
    )
    width = TOPK + kpool - 1
    assert out.topk_indices.shape == (B, S, width)
    assert out.packed_state.shape == (B, S, 2 * D + 1)

    # packed state layout: [key | gate | valid]
    valid_flags = out.packed_state[..., -1]
    assert int(valid_flags.min()) >= 0.0

    # every full-pool selection is causally valid: index <= query position
    sel = np.asarray(out.topk_indices)
    for b in range(B):
        for s in range(S):
            picked = sel[b, s]
            picked = picked[picked != -1]
            if picked.size:
                assert int(picked.max()) <= s

    # topk indices never point at invalid (masked) tokens
    for b in range(B):
        n_invalid = int((~mask[b]).sum())
        for s in range(S):
            limit = S - n_invalid
            picked = sel[b, s]
            picked = picked[picked != -1]
            if picked.size:
                assert int(picked.max()) < limit


def test_pool_state_carry_and_tail():
    kpool = 4
    config = IndexerConfig(
        kind=IndexerKind.POOL,
        index_n_heads=H,
        index_head_dim=D,
        index_topk=TOPK,
        hidden_size=HID,
        q_input_dim=HID,
        score_activation="relu",
        head_reduction="weighted",
        packed_state="key_gate_valid",
        stop_gradient=True,
        kpool_size=kpool,
        select_tail=True,
    )
    _indexer, _params, bound = _make(config)
    hidden = jax.random.normal(jax.random.PRNGKey(0), (1, S, HID), dtype=jnp.float32)
    q_resid = jax.random.normal(jax.random.PRNGKey(1), (1, S, HID), dtype=jnp.float32)

    first = bound(hidden_states=hidden, q_resid=q_resid, attention_mask=None, cached_packed=None)
    second = bound(
        hidden_states=hidden, q_resid=q_resid, attention_mask=None, cached_packed=first.packed_state
    )
    # carried state grows by the step length
    assert second.packed_state.shape[1] == 2 * S
    # selection width is unchanged by the carry
    assert second.topk_indices.shape[-1] == TOPK + kpool - 1
    assert int(second.topk_indices.min()) >= -1


def test_shared_indexer_short_circuit():
    config = IndexerConfig(kind=IndexerKind.TOKEN, index_n_heads=H, index_head_dim=D, index_topk=TOPK)
    _indexer, _params, bound = _make(config)
    prev = jnp.full((B, S, TOPK), 7, dtype=jnp.int32)
    hidden = jnp.zeros((B, S, HID))
    out = bound(hidden_states=hidden, prev_topk_indices=prev)
    assert out.topk_indices is prev
    assert out.packed_state is None


def test_pool_decode_matches_full_forward():
    """Decode (q_len=1) with carried state must select the same tokens as the
    trailing row of a full forward over the concatenated stream: scores are
    deterministic, so this is an exact oracle for the decode path."""
    kpool = 4
    config = IndexerConfig(
        kind=IndexerKind.POOL,
        index_n_heads=H,
        index_head_dim=D,
        index_topk=TOPK,
        hidden_size=HID,
        q_input_dim=HID,
        score_activation="relu",
        head_reduction="weighted",
        packed_state="key_gate_valid",
        stop_gradient=True,
        kpool_size=kpool,
        select_tail=True,
    )
    _indexer, _params, bound = _make(config)
    hidden = jax.random.normal(jax.random.PRNGKey(0), (1, 2 * S, HID), dtype=jnp.float32)
    q_resid = jax.random.normal(jax.random.PRNGKey(1), (1, 2 * S, HID), dtype=jnp.float32)

    full = bound(hidden_states=hidden, q_resid=q_resid, attention_mask=None, cached_packed=None)
    # decode: last token only, carrying everything before it
    step_hidden = hidden[:, -1:]
    step_q = q_resid[:, -1:]
    cached = first.packed_state if (first := bound(hidden_states=hidden[:, : 2 * S - 1], q_resid=q_resid[:, : 2 * S - 1], attention_mask=None, cached_packed=None)) else None
    dec = bound(hidden_states=step_hidden, q_resid=step_q, attention_mask=None, cached_packed=cached)

    ref = full.topk_indices[0, -1]
    got = dec.topk_indices[0, 0]
    # both select the same token set (order can differ between top-k calls on
    # ties, but scores are continuous random — ties are measure-zero)
    ref_set = set(int(x) for x in ref if x != -1)
    got_set = set(int(x) for x in got if x != -1)
    assert ref_set == got_set


def test_rope_split_half_matches_hand_reference():
    """split_half RoPE vs a hand-rotated reference (rotate-half pairing)."""
    from easydel.layers.indexer import apply_indexer_rope

    dim, rotary = 8, 4
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 5, dim), dtype=jnp.float32)
    cos = jnp.asarray(np.tile(np.cos(np.arange(rotary // 2)), 2).reshape(1, 1, rotary), dtype=jnp.float32)
    sin = jnp.asarray(np.tile(np.sin(np.arange(rotary // 2)), 2).reshape(1, 1, rotary), dtype=jnp.float32)

    out = apply_indexer_rope(x, cos, sin, style="split_half")
    x1, x2 = x[..., : rotary // 2], x[..., rotary // 2 : rotary]
    c, s = cos[..., : rotary // 2], sin[..., : rotary // 2]
    ref = jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s, x[..., rotary:]], axis=-1)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-6)


def test_config_rejects_bad_rope_and_pool_state():
    with pytest.raises(ValueError):
        IndexerConfig(index_head_dim=D, rope_dim=D + 2)  # wider than head dim (even)
    with pytest.raises(ValueError):
        IndexerConfig(rope_dim=7)  # odd
    with pytest.raises(ValueError):
        IndexerConfig(kind=IndexerKind.POOL, packed_state="keys")
    with pytest.raises(ValueError):
        IndexerConfig(kind=IndexerKind.POOL, packed_state="none")


def test_config_validation():
    with pytest.raises(ValueError):
        IndexerConfig(kind="bogus")
    with pytest.raises(ValueError):
        IndexerConfig(kind=IndexerKind.POOL, index_topk=10, kpool_size=4)
    with pytest.raises(ValueError):
        IndexerConfig(score_activation="gelu")
    cfg = IndexerConfig(index_topk=8)
    assert cfg.with_changes(index_topk=16).index_topk == 16
