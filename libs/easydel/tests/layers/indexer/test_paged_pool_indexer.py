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

"""Paged k-pool selection vs the dense SparseIndexer on the full unpaged history.

The paged selector keeps its state in pages addressed by a (randomly permuted) page
table and is driven through a multi-step serving schedule: chunked prefill, idle
sequences, then decode steps crossing pool and page boundaries. Every step, each
scheduled token's selected token set must equal the dense indexer's selection for the
same position when run over the whole history at once.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.indexer import IndexerConfig, IndexerKind, SparseIndexer, paged_pool_select, paged_token_layout
from jax.sharding import NamedSharding, PartitionSpec

HID, H, D = 64, 4, 16
PAGE = 8
PAD_TOKENS = 2


def _indexer(kpool: int, topk: int, dim: int = D) -> SparseIndexer:
    config = IndexerConfig(
        kind=IndexerKind.POOL,
        index_n_heads=H,
        index_head_dim=dim,
        index_topk=topk,
        hidden_size=HID,
        q_input_dim=HID,
        score_activation="relu",
        head_reduction="weighted",
        query_source="q_lora",
        rope_style="none",
        packed_state="key_gate_valid",
        stop_gradient=True,
        kpool_size=kpool,
        select_tail=True,
    )
    indexer = SparseIndexer(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    # Non-trivial pooling weights (defaults are ape=0, gate=1).
    rng = np.random.default_rng(1)
    indexer.index_kpool_compress_ape.value = jnp.asarray(rng.standard_normal((kpool, dim)), jnp.float32)
    indexer.index_kpool_compress_gate.value = jnp.asarray(0.2 * rng.standard_normal((dim, HID)), jnp.float32)
    return indexer


def _schedule():
    """Per step, the number of new tokens for each of 3 live sequences (0 = idle)."""
    steps = [[13, 20, 5], [22, 0, 1], [1, 1, 1]]
    steps += [[1, 1, 1] for _ in range(9)]  # decode across pool (4) and page (8) boundaries
    return steps


def _kernel_mesh(num_devices):
    if num_devices is None:
        return None
    if len(jax.devices()) < num_devices:
        pytest.skip(f"needs {num_devices} devices")
    return jax.sharding.Mesh(np.asarray(jax.devices()[:num_devices]), ("x",))


def _head_constraint(mesh):
    """Pin ``[T, heads, ...]`` activations head-sharded over ``mesh`` (what a tensor-parallel projection emits)."""

    def constrain(x):
        spec = PartitionSpec(None, "x", *([None] * (x.ndim - 2)))
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, spec))

    return constrain


@pytest.mark.parametrize(
    ("kpool", "topk", "pool_chunk", "dim", "mesh_devices", "head_sharded", "aligned"),
    [
        (4, 12, 2, D, None, False, False),
        (4, 8, 128, D, None, False, False),
        (2, 10, 3, D, None, False, False),
        # dim 128 is tileable; a 1-device kernel mesh runs the fused Pallas selector directly
        # (no mesh on a multi-chip host would take the XLA fallback instead).
        (4, 12, 1024, 128, 1, False, False),
        # Production tp>1 dispatch: replicated shard_map over a multi-chip kernel mesh.
        (4, 12, 1024, 128, 4, False, False),
        (4, 8, 2, D, 4, False, False),
        # Same, with the query and head weights pinned head-sharded before the rows are split over the mesh
        # (``constrain_heads``, as the serving cache view does for tensor-parallel indexer projections).
        (4, 12, 1024, 128, 4, True, False),
        # Every step padded to a multiple of the mesh size: the fused top-k splits the rows of every step, and these
        # small steps update the pages replicated on every device.
        (4, 12, 1024, 128, 4, False, True),
    ],
)
def test_paged_pool_select_matches_dense_indexer(kpool, topk, pool_chunk, dim, mesh_devices, head_sharded, aligned):
    D = dim  # shadows the module default for this case
    kernel_mesh = _kernel_mesh(mesh_devices)
    constrain_heads = _head_constraint(kernel_mesh) if head_sharded else None
    indexer = _indexer(kpool, topk, dim)
    steps = _schedule()
    lengths = [sum(s[i] for s in steps) for i in range(3)]
    max_seqs, pages_per_seq = 4, -(-max(lengths) // PAGE)
    num_pages = max_seqs * pages_per_seq + 3
    rng = np.random.default_rng(0)
    block_tables = jnp.asarray(
        rng.permutation(num_pages)[: max_seqs * pages_per_seq].reshape(max_seqs, pages_per_seq), jnp.int32
    )
    streams_h = [jnp.asarray(rng.standard_normal((n, HID)), jnp.float32) for n in lengths]
    streams_q = [jnp.asarray(rng.standard_normal((n, HID)), jnp.float32) for n in lengths]

    with jax.default_matmul_precision("highest"):
        dense = [
            np.asarray(
                indexer(hidden_states=h[None], q_resid=q[None], attention_mask=None, cached_packed=None).topk_indices[0]
            )
            for h, q in zip(streams_h, streams_q, strict=True)
        ]

        token_pages = jnp.zeros((num_pages, PAGE, 2 * D), jnp.float32)
        pool_pages = jnp.zeros((num_pages, PAGE // kpool, D), jnp.float32)
        select = jax.jit(
            lambda *a: paged_pool_select(
                indexer, *a, pool_chunk=pool_chunk, kernel_mesh=kernel_mesh, constrain_heads=constrain_heads
            )
        )
        done = [0, 0, 0]
        for step in steps:
            hs, qs, owners = [], [], []
            for s, n in enumerate(step):
                hs.append(streams_h[s][done[s] : done[s] + n])
                qs.append(streams_q[s][done[s] : done[s] + n])
                owners += [(s, p) for p in range(done[s], done[s] + n)]
                done[s] += n
            pad = PAD_TOKENS + (-(sum(step) + PAD_TOKENS) % mesh_devices if aligned else 0)
            hs.append(jnp.asarray(rng.standard_normal((pad, HID)), jnp.float32))
            qs.append(jnp.asarray(rng.standard_normal((pad, HID)), jnp.float32))
            kv_lens = jnp.asarray([*done, 0], jnp.int32)
            qsl = jnp.asarray(np.concatenate([[0], np.cumsum([*step, 0])]), jnp.int32)
            selected, token_pages, pool_pages = select(
                jnp.concatenate(hs),
                jnp.concatenate(qs),
                token_pages,
                pool_pages,
                block_tables,
                kv_lens,
                qsl,
                jnp.int32(3),
            )
            selected = np.asarray(selected)
            assert selected.shape[1] == topk // kpool + 1
            for t, (s, p) in enumerate(owners):
                ids = selected[t][selected[t] >= 0]
                assert len(set(ids.tolist())) == len(ids), "chunk ids must be unique per row"
                got = {int(c) * kpool + o for c in ids for o in range(kpool) if int(c) * kpool + o <= p}
                ref = {int(i) for i in dense[s][p] if i >= 0}
                assert got == ref, f"seq {s} pos {p}: paged {sorted(got)} vs dense {sorted(ref)}"
            assert np.all(selected[len(owners) :] == -1), "padding tokens must select nothing"


@pytest.mark.parametrize("dim", [D, 128])
def test_rank_major_dp_paged_pool_select_matches_dense_indexer(dim):
    """Data-parallel serving: rank-major metadata, pages sharded over dp=2, one selector per rank.

    Rows 0-1 live on rank 0 and row 2 on rank 1; each row's pages come from its rank's page
    shard (global ids), tokens are laid out rank-major with rank-local ``query_start_loc``.
    Every selection must equal the dense indexer over the sequence's full history.
    """
    if len(jax.devices()) < 2:
        pytest.skip("needs 2 devices")
    kpool, topk, dp = 4, 12, 2
    D = dim  # shadows the module default for this case
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:dp]), ("dp",))
    indexer = _indexer(kpool, topk, dim)
    steps = _schedule()
    lengths = [sum(s[i] for s in steps) for i in range(3)]
    rows, rows_per_rank = 4, 2
    row_rank = [0, 0, 1]
    pages_per_seq = -(-max(lengths) // PAGE)
    half = rows_per_rank * pages_per_seq + 2
    num_pages = dp * half
    rng = np.random.default_rng(0)
    tables = np.zeros((rows, pages_per_seq), np.int32)
    for rank in range(dp):
        perm = rank * half + rng.permutation(half)
        tables[rank * rows_per_rank : (rank + 1) * rows_per_rank] = perm[: rows_per_rank * pages_per_seq].reshape(
            rows_per_rank, pages_per_seq
        )
    block_tables = jnp.asarray(tables)
    streams_h = [jnp.asarray(rng.standard_normal((n, HID)), jnp.float32) for n in lengths]
    streams_q = [jnp.asarray(rng.standard_normal((n, HID)), jnp.float32) for n in lengths]
    tokens_per_rank = 36  # >= the largest per-rank step (13 + 20)

    with jax.default_matmul_precision("highest"):
        dense = [
            np.asarray(
                indexer(hidden_states=h[None], q_resid=q[None], attention_mask=None, cached_packed=None).topk_indices[0]
            )
            for h, q in zip(streams_h, streams_q, strict=True)
        ]

        token_pages = jnp.zeros((num_pages, PAGE, 2 * D), jnp.float32)
        pool_pages = jnp.zeros((num_pages, PAGE // kpool, D), jnp.float32)
        select = jax.jit(lambda *a: paged_pool_select(indexer, *a, dp_sharding=(mesh, ("dp",))))
        done = [0, 0, 0]
        for step in steps:
            hs, qs, owners = [], [], []
            qsl = np.zeros((dp, rows_per_rank + 1), np.int32)
            num_live = np.zeros((dp,), np.int32)
            for rank in range(dp):
                used = 0
                for local_row in range(rows_per_rank):
                    s = rank * rows_per_rank + local_row
                    n = step[s] if s < 3 else 0
                    if s < 3:
                        hs.append(streams_h[s][done[s] : done[s] + n])
                        qs.append(streams_q[s][done[s] : done[s] + n])
                        owners += [(rank * tokens_per_rank + used + i, s, done[s] + i) for i in range(n)]
                        num_live[rank] = local_row + 1
                    used += n
                    qsl[rank, local_row + 1] = used
                pad = tokens_per_rank - used
                hs.append(jnp.asarray(rng.standard_normal((pad, HID)), jnp.float32))
                qs.append(jnp.asarray(rng.standard_normal((pad, HID)), jnp.float32))
            for s in range(3):
                done[s] += step[s]
            kv_lens = jnp.asarray([*done, 0], jnp.int32)
            selected, token_pages, pool_pages = select(
                jnp.concatenate(hs),
                jnp.concatenate(qs),
                token_pages,
                pool_pages,
                block_tables,
                kv_lens,
                jnp.asarray(qsl),
                jnp.asarray(num_live),
            )
            selected = np.asarray(selected)
            assert selected.shape == (dp * tokens_per_rank, topk // kpool + 1)
            live = set()
            for t, s, p in owners:
                live.add(t)
                ids = selected[t][selected[t] >= 0]
                assert len(set(ids.tolist())) == len(ids), "chunk ids must be unique per row"
                got = {int(c) * kpool + o for c in ids for o in range(kpool) if int(c) * kpool + o <= p}
                ref = {int(i) for i in dense[s][p] if i >= 0}
                assert got == ref, f"rank {row_rank[s]} seq {s} pos {p}: paged {sorted(got)} vs dense {sorted(ref)}"
            pads = [t for t in range(dp * tokens_per_rank) if t not in live]
            assert np.all(selected[pads] == -1), "padding tokens must select nothing"


def test_replicated_small_step_page_update_is_bitwise(monkeypatch):
    """On a multi-device kernel mesh, small steps update the pages replicated on every device; pages and selections
    equal the row-split update's bit for bit over a multi-step serving schedule."""
    from easydel.layers.indexer import _paged

    if jax.default_backend() != "tpu":
        pytest.skip("the fused top-k splits its rows over the kernel mesh only on TPU")
    kernel_mesh = _kernel_mesh(4)
    kpool, topk, dim = 4, 12, 128
    indexer = _indexer(kpool, topk, dim)
    steps = _schedule()
    lengths = [sum(s[i] for s in steps) for i in range(3)]
    max_seqs, pages_per_seq = 4, -(-max(lengths) // PAGE)
    num_pages = max_seqs * pages_per_seq + 3
    rng = np.random.default_rng(0)
    block_tables = jnp.asarray(
        rng.permutation(num_pages)[: max_seqs * pages_per_seq].reshape(max_seqs, pages_per_seq), jnp.int32
    )
    batches, done = [], [0, 0, 0]
    for step in steps:
        total = sum(step)
        pad = PAD_TOKENS + (-(total + PAD_TOKENS) % 4)
        assert total + pad <= _paged._REPLICATED_POOL_UPDATE_MAX_TOKENS
        for s in range(3):
            done[s] += step[s]
        batches.append(
            (
                jnp.asarray(rng.standard_normal((total + pad, HID)), jnp.float32),
                jnp.asarray(rng.standard_normal((total + pad, HID)), jnp.float32),
                jnp.asarray([*done, 0], jnp.int32),
                jnp.asarray(np.concatenate([[0], np.cumsum([*step, 0])]), jnp.int32),
            )
        )

    def run(max_tokens):
        monkeypatch.setattr(_paged, "_REPLICATED_POOL_UPDATE_MAX_TOKENS", max_tokens)
        select = jax.jit(lambda *a: paged_pool_select(indexer, *a, pool_chunk=1024, kernel_mesh=kernel_mesh))
        token_pages = jnp.zeros((num_pages, PAGE, 2 * dim), jnp.float32)
        pool_pages = jnp.zeros((num_pages, PAGE // kpool, dim), jnp.float32)
        selections = []
        with jax.default_matmul_precision("highest"):
            for hidden, resid, kv_lens, qsl in batches:
                selected, token_pages, pool_pages = select(
                    hidden, resid, token_pages, pool_pages, block_tables, kv_lens, qsl, jnp.int32(3)
                )
                selections.append(np.asarray(selected))
        return selections, np.asarray(token_pages), np.asarray(pool_pages)

    replicated = run(_paged._REPLICATED_POOL_UPDATE_MAX_TOKENS)
    row_split = run(0)
    for step, (a, b) in enumerate(zip(replicated[0], row_split[0], strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"selection of step {step}")
    np.testing.assert_array_equal(replicated[1], row_split[1], err_msg="token pages")
    np.testing.assert_array_equal(replicated[2], row_split[2], err_msg="pool pages")


@pytest.mark.parametrize("mesh_devices", [None, 4], ids=["no_mesh", "mesh4"])
def test_page_granular_update_is_bitwise(monkeypatch, mesh_devices):
    """Steps written one page at a time (``_pool_page_update_by_page``) leave the same token and pool pages, and select
    the same pools, as the row update, over the multi-step schedule (chunked prefill, idle rows, decode across pool and
    page boundaries, padding rows)."""
    from easydel.layers.indexer import _paged

    kernel_mesh = _kernel_mesh(mesh_devices)
    kpool, topk, dim = 4, 12, 128
    indexer = _indexer(kpool, topk, dim)
    steps = _schedule()
    lengths = [sum(s[i] for s in steps) for i in range(3)]
    max_seqs, pages_per_seq = 4, -(-max(lengths) // PAGE)
    num_pages = max_seqs * pages_per_seq + 3
    rng = np.random.default_rng(0)
    block_tables = jnp.asarray(
        rng.permutation(num_pages)[: max_seqs * pages_per_seq].reshape(max_seqs, pages_per_seq), jnp.int32
    )
    batches, done = [], [0, 0, 0]
    for step in steps:
        total = sum(step)
        pad = PAD_TOKENS + (-(total + PAD_TOKENS) % 4)
        for s in range(3):
            done[s] += step[s]
        batches.append(
            (
                jnp.asarray(rng.standard_normal((total + pad, HID)), jnp.float32),
                jnp.asarray(rng.standard_normal((total + pad, HID)), jnp.float32),
                jnp.asarray([*done, 0], jnp.int32),
                jnp.asarray(np.concatenate([[0], np.cumsum([*step, 0])]), jnp.int32),
            )
        )

    def run(min_tokens):
        monkeypatch.setattr(_paged, "_PAGE_GRANULAR_UPDATE_MIN_TOKENS", min_tokens)
        select = jax.jit(lambda *a: paged_pool_select(indexer, *a, pool_chunk=1024, kernel_mesh=kernel_mesh))
        token_pages = jnp.zeros((num_pages, PAGE, 2 * dim), jnp.float32)
        pool_pages = jnp.zeros((num_pages, PAGE // kpool, dim), jnp.float32)
        selections = []
        with jax.default_matmul_precision("highest"):
            for hidden, resid, kv_lens, qsl in batches:
                selected, token_pages, pool_pages = select(
                    hidden, resid, token_pages, pool_pages, block_tables, kv_lens, qsl, jnp.int32(3)
                )
                selections.append(np.asarray(selected))
        return selections, np.asarray(token_pages), np.asarray(pool_pages)

    by_page = run(0)
    by_row = run(1 << 30)
    assert np.any(by_row[2] != 0), "no pool summary was written"
    for step, (a, b) in enumerate(zip(by_page[0], by_row[0], strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"selection of step {step}")
    np.testing.assert_array_equal(by_page[1], by_row[1], err_msg="token pages")
    np.testing.assert_array_equal(by_page[2], by_row[2], err_msg="pool pages")


def test_paged_pool_select_rejects_unaligned_page_size():
    indexer = _indexer(4, 8)
    with pytest.raises(ValueError, match="multiple of kpool"):
        paged_pool_select(
            indexer,
            jnp.zeros((2, HID)),
            jnp.zeros((2, HID)),
            jnp.zeros((4, 6, 2 * D)),
            jnp.zeros((4, 1, D)),
            jnp.zeros((1, 4), jnp.int32),
            jnp.asarray([2], jnp.int32),
            jnp.asarray([0, 2], jnp.int32),
            jnp.int32(1),
        )


def _token_layout_reference(kv_lens, query_start_loc, num_seqs, total_tokens):
    """Row, absolute position and validity of every packed token, from a loop over the live rows' token ranges."""
    seq = np.zeros(total_tokens, np.int64)
    pos = np.zeros(total_tokens, np.int64)
    valid = np.zeros(total_tokens, bool)
    for row in range(min(max(num_seqs, 0), len(kv_lens))):
        start, end = query_start_loc[row], query_start_loc[row + 1]
        for token in range(start, min(end, total_tokens)):
            position = kv_lens[row] - (end - start) + (token - start)
            if 0 <= position < kv_lens[row]:
                seq[token], pos[token], valid[token] = row, position, True
    return seq, pos, valid


def _random_token_layout(seed, rows, live, max_new, pad):
    """A step of ``live`` rows with 0..max_new new tokens each (about a quarter idle) and ``rows - live`` empty rows."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, max_new + 1, size=live)
    counts[rng.random(live) < 0.25] = 0
    kv_lens = np.concatenate([counts + rng.integers(0, 4096, size=live), np.zeros(rows - live, np.int64)])
    query_start_loc = np.concatenate([[0], np.cumsum(counts), np.full(rows - live, counts.sum())])
    return kv_lens.tolist(), query_start_loc.tolist(), live, int(counts.sum()) + pad


@pytest.mark.parametrize(
    ("kv_lens", "query_start_loc", "num_seqs", "total_tokens"),
    [
        # Decode: one new token per live row, empty rows after them, padding tokens past the live rows.
        ([5, 9, 1, 130, 7, 0, 0, 0], [0, 1, 2, 3, 4, 5, 5, 5, 5], 5, 8),
        # Chunked prefill mixed with decode, with an idle row (no new tokens) between live rows.
        ([13, 4, 22, 40, 0, 0], [0, 13, 13, 14, 34, 34, 34], 4, 40),
        # A row past num_seqs that still holds tokens: its tokens are padding.
        ([3, 8, 4, 0], [0, 3, 5, 9, 9], 2, 10),
        # A slice whose first row starts past token 0: the leading tokens belong to other shards.
        ([6, 2, 50, 0, 0], [5, 7, 7, 10, 10, 10], 3, 12),
        # No live row.
        ([0, 0, 0], [0, 0, 0, 0], 0, 4),
        # 128 rows, 100 of them live with up to 40 new tokens each.
        _random_token_layout(3, rows=128, live=100, max_new=40, pad=7),
    ],
    ids=["decode", "mixed_idle_row", "row_past_num_seqs", "sliced_rows", "no_live_row", "random_128_rows"],
)
def test_paged_token_layout_matches_row_loop(kv_lens, query_start_loc, num_seqs, total_tokens):
    """Each live token gets its row and absolute position; every other token is invalid but still names a real row
    (its gathers stay in bounds and its writes are dropped)."""
    layout = jax.jit(paged_token_layout, static_argnums=3)
    seq, pos, valid = (
        np.asarray(x)
        for x in layout(
            jnp.asarray(kv_lens, jnp.int32), jnp.asarray(query_start_loc, jnp.int32), jnp.int32(num_seqs), total_tokens
        )
    )
    ref_seq, ref_pos, ref_valid = _token_layout_reference(kv_lens, query_start_loc, num_seqs, total_tokens)
    np.testing.assert_array_equal(valid, ref_valid)
    np.testing.assert_array_equal(seq[valid], ref_seq[valid])
    np.testing.assert_array_equal(pos[valid], ref_pos[valid])
    assert seq.min() >= 0 and seq.max() < len(kv_lens)
