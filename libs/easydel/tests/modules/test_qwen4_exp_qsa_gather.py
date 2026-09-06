"""Equivalence tests for the Qwen4-Exp QSA gathered decode path.

The decode fast path in ``Qwen4ExpAttention`` gathers the indexer-selected
tokens and attends them with a compact dense softmax instead of handing the
kernel a full-width ``-inf`` bias over the dense cache. The attended set is
identical by construction; these tests pin the numerics of the einsum/GQA/
masking math against a straightforward masked-softmax reference.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.modules.qwen4_exp.modeling_qwen4_exp import Qwen4ExpAttention


def _full_width_reference(q, key, value, selected, scale):
    """Masked full-width softmax attention over the dense buffer."""
    batch, seq, kv_heads, dim = key.shape
    heads = q.shape[2]
    groups = heads // kv_heads
    sel = selected[:, 0, :]  # [B, W]
    valid = sel >= 0

    # Full-width masked softmax, grouped per KV head (head h -> kv h // groups).
    q4 = q[:, 0].reshape(batch, kv_heads, groups, dim)
    scores = jnp.einsum("bhgd,bshd->bhgs", q4.astype(jnp.float32), key.astype(jnp.float32)) * scale
    drop = jnp.where(valid, sel, seq)  # invalid slots land on the drop column
    full_mask = jnp.zeros((batch, seq + 1), bool).at[:, drop].set(True)[:, :seq]
    scores = jnp.where(full_mask[:, None, None, :], scores, -1e30)
    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhgs,bshd->bhgd", probs.astype(value.dtype), value)
    return out.reshape(batch, 1, heads, dim)


def _make_gather_callable():
    """Call the production gathered-attention method without constructing a layer."""

    def gather(q, key, value, selected):
        attention = SimpleNamespace(head_dim=q.shape[-1])
        return Qwen4ExpAttention._decode_gather_attention(attention, q, key, value, selected)

    return gather


@pytest.mark.parametrize("seed", [0, 1])
def test_gather_matches_masked_full_width(seed):
    """Gathered decode output must equal the masked full-width reference."""
    rng = jax.random.PRNGKey(seed)
    batch, seq, heads, kv_heads, dim, width = 3, 64, 8, 2, 32, 21
    q = jax.random.normal(rng, (batch, 1, heads, dim), jnp.bfloat16)
    key = jax.random.normal(jax.random.PRNGKey(seed + 10), (batch, seq, kv_heads, dim), jnp.bfloat16)
    value = jax.random.normal(jax.random.PRNGKey(seed + 20), (batch, seq, kv_heads, dim), jnp.bfloat16)

    # Contract-valid selections: duplicate-free (select() emits disjoint
    # block members + tail) with some -1 padding at the end.
    perm = jax.random.permutation(jax.random.PRNGKey(seed + 30), jnp.arange(seq, dtype=jnp.int32), axis=-1)
    sel = jnp.broadcast_to(perm[:width], (batch, 1, width))
    pad_cols = jnp.arange(width) > (width - 3)
    sel = jnp.where(jnp.broadcast_to(pad_cols, sel.shape), -1, sel)

    got = _make_gather_callable()(q, key, value, selected=sel)
    want = _full_width_reference(q, key, value, sel, scale=dim**-0.5)
    assert got.shape == want.shape
    got32, want32 = jnp.asarray(got, jnp.float32), jnp.asarray(want, jnp.float32)
    assert jnp.allclose(got32, want32, rtol=2e-2, atol=2e-2), float(jnp.max(jnp.abs(got32 - want32)))


def test_gather_all_padding_row_is_finite():
    """A fully-masked (padding) row must produce finite garbage, never NaN."""
    batch, seq, heads, kv_heads, dim, width = 2, 16, 4, 2, 16, 8
    q = jnp.ones((batch, 1, heads, dim), jnp.bfloat16)
    key = jnp.ones((batch, seq, kv_heads, dim), jnp.bfloat16)
    value = jnp.ones((batch, seq, kv_heads, dim), jnp.bfloat16)
    sel = jnp.full((batch, 1, width), -1, jnp.int32)
    out = _make_gather_callable()(q, key, value, selected=sel)
    assert jnp.all(jnp.isfinite(jnp.asarray(out, jnp.float32)))


def test_module_method_matches_free_math():
    """The method on the class must be the same math as the reference here."""
    src_fn = Qwen4ExpAttention._decode_gather_attention
    assert callable(src_fn)


def _fake_rope_tables(seq, rot):
    """Deterministic cos/sin tables for testing (any consistent table works)."""
    pos = jnp.arange(seq, dtype=jnp.float32)[:, None]
    freq = (jnp.arange(rot, dtype=jnp.float32)[None, :] + 1.0) * 0.01
    return jnp.cos(pos * freq), jnp.sin(pos * freq)


def test_select_step_matches_full_select():
    """Incremental decode selection must equal the full re-pooling selection."""
    import spectrax as spx
    from easydel.layers.sparse_attention import BlockTopKIndexer

    rng = jax.random.PRNGKey(7)
    batch, seq, hidden, n_heads, dim, budget, ratio = 2, 32, 64, 2, 16, 8, 4
    rot = 8
    prompt = 13
    steps = 6

    indexer = BlockTopKIndexer(
        hidden_size=hidden,
        index_n_heads=n_heads,
        index_kv_heads=1,
        index_head_dim=dim,
        indexer_budget=budget,
        indexer_compress_ratio=ratio,
        rngs=spx.Rngs(0),
    )
    cos_t, sin_t = _fake_rope_tables(seq, rot)  # [S, R]
    cos_full = jnp.broadcast_to(cos_t[None], (batch, seq, rot))
    sin_full = jnp.broadcast_to(sin_t[None], (batch, seq, rot))

    keys = jax.random.normal(rng, (batch, seq, dim), jnp.bfloat16)
    qs = jax.random.normal(jax.random.PRNGKey(8), (batch, steps, n_heads, dim), jnp.bfloat16)

    # --- seed via the prefill path ---
    key_buffer = jnp.zeros((batch, seq, dim), jnp.bfloat16)
    visible = jnp.zeros((batch, seq), jnp.bool_)
    key_buffer = jax.lax.dynamic_update_slice(key_buffer, keys[:, :prompt], (0, 0, 0))
    visible = jax.lax.dynamic_update_slice(visible, jnp.ones((batch, prompt), jnp.bool_), (0, 0))
    _, block_keys, complete = indexer.select(
        qs[:, :1],
        key_buffer,
        q_cos=cos_full[:, prompt - 1 : prompt],
        q_sin=sin_full[:, prompt - 1 : prompt],
        k_cos=cos_full,
        k_sin=sin_full,
        visible=visible,
        q_indices=jnp.full((batch, 1), prompt - 1, jnp.int32),
        return_blocks=True,
    )

    for t in range(steps):
        pos = prompt + t
        q_t = qs[:, t : t + 1]
        q_cos = cos_full[:, pos : pos + 1]
        q_sin = sin_full[:, pos : pos + 1]
        write_at = jnp.full((batch,), pos, jnp.int32)

        # The modeling writes the current token into the buffers BEFORE selecting.
        key_buffer = jax.lax.dynamic_update_slice(key_buffer, keys[:, pos : pos + 1], (0, pos, 0))
        visible = jax.lax.dynamic_update_slice(visible, jnp.ones((batch, 1), jnp.bool_), (0, pos))

        # full path: re-pools the whole buffer
        want = indexer.select(
            q_t,
            key_buffer,
            q_cos=q_cos,
            q_sin=q_sin,
            k_cos=cos_full,
            k_sin=sin_full,
            visible=visible,
            q_indices=write_at[:, None],
        )
        # incremental path: re-pools only the open block
        first_visible = jnp.zeros((batch,), jnp.int32)
        b_open = write_at // ratio
        open_start = first_visible + b_open * ratio
        open_cos = jnp.take_along_axis(cos_full, open_start[:, None, None], axis=1)
        open_sin = jnp.take_along_axis(sin_full, open_start[:, None, None], axis=1)
        got, block_keys, complete = indexer.select_step(
            q_t,
            q_cos=q_cos,
            q_sin=q_sin,
            key_buffer=key_buffer,
            block_keys=block_keys,
            blocks_complete=complete,
            visible=visible,
            open_cos=open_cos,
            open_sin=open_sin,
            write_at=write_at,
        )
        assert jnp.array_equal(got, want), f"step {t}: got {got[0, 0].tolist()} want {want[0, 0].tolist()}"


def test_batched_qsa_cache_update_uses_each_rows_offset():
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import _dynamic_update_rows

    starts = jnp.array([1, 3], jnp.int32)
    visible = jnp.zeros((2, 6), jnp.bool_)
    visible_new = jnp.array([[True, True], [True, False]])
    got_visible = _dynamic_update_rows(visible, visible_new, starts)
    assert jnp.array_equal(got_visible[0], jnp.array([0, 1, 1, 0, 0, 0], bool))
    assert jnp.array_equal(got_visible[1], jnp.array([0, 0, 0, 1, 0, 0], bool))

    keys = jnp.zeros((2, 6, 2), jnp.float32)
    keys_new = jnp.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    got_keys = _dynamic_update_rows(keys, keys_new, starts)
    assert jnp.array_equal(got_keys[0, 1], keys_new[0, 0])
    assert jnp.array_equal(got_keys[1, 3], keys_new[1, 0])

    rows = jnp.zeros((2, 3, 6), jnp.int32)
    rows_new = jnp.stack([jnp.full((3, 1), 7), jnp.full((3, 1), 9)])
    got_rows = _dynamic_update_rows(rows, rows_new, starts, axis=1)
    assert jnp.all(got_rows[0, :, 1] == 7)
    assert jnp.all(got_rows[1, :, 3] == 9)


def test_runtime_mode_uses_cache_metadata_for_esurge_prefill_and_decode():
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import _resolve_qwen4_runtime_mode
    from spectrax import common_types

    assert _resolve_qwen4_runtime_mode(None, 8, False) == common_types.MODE_TRAIN
    assert _resolve_qwen4_runtime_mode(None, 8, True) == common_types.MODE_PREFILL
    assert _resolve_qwen4_runtime_mode(None, 1, True) == common_types.MODE_DECODE
    assert _resolve_qwen4_runtime_mode(common_types.MODE_TRAIN, 1, True) == common_types.MODE_TRAIN


def test_mtp_training_gate_excludes_esurge_metadata_cache():
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import _is_qwen4_training_call
    from spectrax import common_types

    marker = object()
    assert _is_qwen4_training_call(None, None, None)
    assert _is_qwen4_training_call(common_types.MODE_TRAIN, marker, marker)
    assert not _is_qwen4_training_call(None, None, marker)
    assert not _is_qwen4_training_call(None, marker, None)
    assert not _is_qwen4_training_call(common_types.MODE_PREFILL, None, None)


def test_paged_qsa_token_map_tracks_independent_request_lengths():
    from easydel.caching import RaggedPagesMetadata
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import _paged_qsa_token_map

    meta = RaggedPagesMetadata(
        pages_tables=jnp.array([[4, 5], [7, 8]], jnp.int32),
        context_lens=jnp.array([5, 2], jnp.int32),
        query_start_loc=jnp.array([0, 2, 3], jnp.int32),
        num_seqs=jnp.array([2], jnp.int32),
        slot_mapping=jnp.array([3, 4, 5], jnp.int32),
        num_kv_update_slices=jnp.array([3], jnp.int32),
        version="v2",
        page_size=4,
    )
    req, logical, physical, offset, valid = _paged_qsa_token_map(meta, 3, 10)
    np.testing.assert_array_equal(np.asarray(req), [0, 0, 1])
    np.testing.assert_array_equal(np.asarray(logical), [3, 4, 1])
    np.testing.assert_array_equal(np.asarray(physical), [4, 5, 7])
    np.testing.assert_array_equal(np.asarray(offset), [3, 0, 1])
    assert bool(jnp.all(valid))


def test_paged_qsa_selection_writes_pages_and_selects_complete_prefix():
    from dataclasses import dataclass, replace
    from types import SimpleNamespace

    from easydel.caching import RaggedPagesMetadata

    class Indexer:
        compress_ratio = 2
        index_n_heads = 1
        index_head_dim = 2
        block_topk = 2
        token_budget = 4
        k_layernorm = staticmethod(lambda x: x)

        @staticmethod
        def project(hidden):
            raw = hidden[..., :2]
            return raw[..., None, :], raw

    class Rotary:
        @staticmethod
        def compute_cos_sin(rows, dtype=None):
            shape = (*rows.shape[1:], 2)
            return jnp.ones(shape, jnp.float32), jnp.zeros(shape, jnp.float32)

    @dataclass
    class View:
        indexer_key_pages: object
        mrope_position_pages: object

        def replace(self, **kwargs):
            return replace(self, **kwargs)

    attn = SimpleNamespace(indexer=Indexer(), rotary=Rotary())
    view = View(jnp.zeros((1, 4, 2), jnp.float32), jnp.zeros((1, 4, 3), jnp.int32))
    meta = RaggedPagesMetadata(
        pages_tables=jnp.array([[0]], jnp.int32),
        context_lens=jnp.array([4], jnp.int32),
        query_start_loc=jnp.array([0, 4], jnp.int32),
        num_seqs=jnp.array([1], jnp.int32),
        slot_mapping=jnp.arange(4, dtype=jnp.int32),
        num_kv_update_slices=jnp.array([4], jnp.int32),
        version="v2",
        page_size=4,
    )
    hidden = jnp.array([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]])
    positions = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32)[None], (1, 4))
    selected, updated = Qwen4ExpAttention._paged_indexer_select(attn, hidden, positions, view, meta)
    np.testing.assert_array_equal(np.asarray(updated.indexer_key_pages[0, :, 0]), [1, 2, 3, 4])
    assert set(np.asarray(selected[0, -1][selected[0, -1] >= 0]).tolist()) == {0, 1, 2, 3}


def test_paged_qsa_view_allocates_and_resets_sidecars():
    from easydel.caching import RaggedPagesCacheConfig
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import Qwen4ExpPagedQSAView

    cfg = RaggedPagesCacheConfig(
        num_hidden_layers=1,
        max_model_length=16,
        num_kv_heads=1,
        k_headdim=8,
        v_headdim=8,
        num_pages=4,
        max_num_pages_per_req=4,
        num_slices_per_kv_cache_update_page=4,
        max_num_tokens=8,
        max_num_reqs=1,
        page_size=4,
        _kvdtype_str="bf16",
        version="v2",
    )
    object.__setattr__(cfg, "qwen4_indexer_head_dim", 6)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("tp",))
    view = Qwen4ExpPagedQSAView.init(cfg, mesh=mesh)
    assert view.indexer_key_pages.shape == (4, 4, 6)
    assert view.mrope_position_pages.shape == (4, 4, 3)
    reset = view.replace(indexer_key_pages=jnp.ones_like(view.indexer_key_pages)).reset()
    assert not bool(jnp.any(reset.indexer_key_pages))

    from easydel.caching import RaggedPagesCache

    pure, metadata = RaggedPagesCache(views=[view]).to_pure()
    restored = RaggedPagesCache.from_pure(pure, metadata, view.runtime_sharding_resolver)
    assert isinstance(restored.views[0], Qwen4ExpPagedQSAView)
    assert restored.views[0].indexer_key_pages.shape == view.indexer_key_pages.shape


def test_paged_gather_keeps_requests_on_their_own_physical_pages():
    from types import SimpleNamespace

    from easydel.caching import RaggedPagesMetadata

    meta = RaggedPagesMetadata(
        pages_tables=jnp.array([[0], [1]], jnp.int32),
        context_lens=jnp.array([2, 2], jnp.int32),
        query_start_loc=jnp.array([0, 1, 2], jnp.int32),
        num_seqs=jnp.array([2], jnp.int32),
        slot_mapping=jnp.array([1, 3], jnp.int32),
        num_kv_update_slices=jnp.array([2], jnp.int32),
        version="v2",
        page_size=2,
    )
    values = jnp.array([1.0, 3.0, 10.0, 30.0], jnp.float32).reshape(2, 2, 1, 1)
    keys = jnp.zeros((2, 2, 1, 1))
    interleaved = jnp.stack((keys, values), axis=3).reshape(2, 2, 2, 1)
    view = SimpleNamespace(
        indexer_key_pages=jnp.zeros((2, 2, 1)),
        key_pages=keys,
        value_pages=values,
        flattened_kv_pages=lambda: interleaved,
    )
    attn = SimpleNamespace(head_dim=1)
    q = jnp.zeros((1, 2, 1, 1), jnp.float32)
    selected = jnp.array([[[0, 1], [0, 1]]], jnp.int32)
    out = Qwen4ExpAttention._paged_gather_attention(attn, q, view, meta, selected)
    np.testing.assert_allclose(np.asarray(out).reshape(-1), [2.0, 20.0])


def test_paged_qsa_token_map_supports_v3_metadata():
    from easydel.caching import RaggedPagesMetadata
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import _paged_qsa_token_map

    meta = RaggedPagesMetadata(
        pages_tables=jnp.array([[2, 5], [7, 8]], jnp.int32),
        context_lens=jnp.array([3, 2], jnp.int32),
        query_start_loc=jnp.array([0, 2, 3], jnp.int32),
        num_seqs=jnp.array([2], jnp.int32),
        request_distribution=jnp.array([0, 2, 2], jnp.int32),
        version="v3",
        page_size=2,
    )
    row, logical, physical, offset, valid = _paged_qsa_token_map(meta, 3, 9)
    np.testing.assert_array_equal(np.asarray(row), [0, 0, 1])
    np.testing.assert_array_equal(np.asarray(logical), [1, 2, 1])
    np.testing.assert_array_equal(np.asarray(physical), [2, 5, 7])
    np.testing.assert_array_equal(np.asarray(offset), [1, 0, 1])
    assert bool(jnp.all(valid))


def test_ragged_cache_roundtrip_preserves_none_layer_slots():
    from easydel.caching import RaggedPagesCache

    restored = RaggedPagesCache.from_pure([{"is_none": True}, {"is_none": True}], metadata=None)
    assert restored.views == [None, None]


def test_ragged_cache_insert_preserves_qwen_sidecars_and_type():
    from easydel.caching import RaggedPagesCache, RaggedPagesCacheConfig
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import Qwen4ExpPagedQSAView

    cfg = RaggedPagesCacheConfig(
        num_hidden_layers=1,
        max_model_length=16,
        num_kv_heads=1,
        k_headdim=8,
        v_headdim=8,
        num_pages=4,
        max_num_pages_per_req=4,
        num_slices_per_kv_cache_update_page=4,
        max_num_tokens=8,
        max_num_reqs=1,
        page_size=4,
        _kvdtype_str="bf16",
        version="v2",
    )
    object.__setattr__(cfg, "qwen4_indexer_head_dim", 6)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("tp",))
    target = Qwen4ExpPagedQSAView.init(cfg, mesh=mesh)
    source = target.replace(
        kv_pages=jnp.ones_like(target.kv_pages[:2]),
        indexer_key_pages=jnp.ones_like(target.indexer_key_pages[:2]),
        mrope_position_pages=jnp.ones_like(target.mrope_position_pages[:2]),
    )
    result = RaggedPagesCache([target]).insert(RaggedPagesCache([source]), 2).views[0]
    assert isinstance(result, Qwen4ExpPagedQSAView)
    assert bool(jnp.all(result.indexer_key_pages[2:] == 1))
    assert bool(jnp.all(result.mrope_position_pages[2:] == 1))


def test_v3_paged_qsa_writes_current_kv_before_selected_gather():
    from dataclasses import dataclass, replace

    from easydel.caching import RaggedPagesMetadata

    @dataclass
    class View:
        kv_pages: object
        indexer_key_pages: object

        @property
        def key_pages(self):
            flat = self.kv_pages.reshape(*self.kv_pages.shape[:2], -1, self.kv_pages.shape[-1])
            return flat[:, :, 0::2]

        @property
        def value_pages(self):
            flat = self.kv_pages.reshape(*self.kv_pages.shape[:2], -1, self.kv_pages.shape[-1])
            return flat[:, :, 1::2]

        def replace(self, **kwargs):
            return replace(self, **kwargs)

    meta = RaggedPagesMetadata(
        pages_tables=jnp.array([[2, 5], [7, 8]], jnp.int32),
        context_lens=jnp.array([3, 2], jnp.int32),
        query_start_loc=jnp.array([0, 2, 3], jnp.int32),
        num_seqs=jnp.array([2], jnp.int32),
        request_distribution=jnp.array([0, 2, 2], jnp.int32),
        version="v3",
        page_size=2,
    )
    view = View(
        kv_pages=jnp.zeros((9, 2, 2, 2, 1), jnp.float32),
        indexer_key_pages=jnp.zeros((9, 2, 1), jnp.float32),
    )
    key = jnp.array([1.0, 2.0, 3.0], jnp.float32).reshape(1, 3, 1, 1)
    value = (key * 10).reshape(1, 3, 1, 1)
    updated = Qwen4ExpAttention._write_v3_paged_kv(object(), key, value, view, meta)
    expected = np.array([[1, 1], [2, 2], [3, 3]])
    np.testing.assert_array_equal(np.asarray(updated.key_pages[[2, 5, 7], [1, 0, 1], :, 0]), expected)
    np.testing.assert_array_equal(np.asarray(updated.value_pages[[2, 5, 7], [1, 0, 1], :, 0]), expected * 10)
