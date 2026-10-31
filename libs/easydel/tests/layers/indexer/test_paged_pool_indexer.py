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
from easydel.layers.indexer import IndexerConfig, IndexerKind, SparseIndexer, paged_pool_select

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


@pytest.mark.parametrize(
    ("kpool", "topk", "pool_chunk", "dim", "mesh_devices"),
    [
        (4, 12, 2, D, None),
        (4, 8, 128, D, None),
        (2, 10, 3, D, None),
        # dim 128 is tileable; a 1-device kernel mesh runs the fused Pallas selector directly
        # (no mesh on a multi-chip host would take the XLA fallback instead).
        (4, 12, 1024, 128, 1),
        # Production tp>1 dispatch: replicated shard_map over a multi-chip kernel mesh.
        (4, 12, 1024, 128, 4),
        (4, 8, 2, D, 4),
    ],
)
def test_paged_pool_select_matches_dense_indexer(kpool, topk, pool_chunk, dim, mesh_devices):
    D = dim  # shadows the module default for this case
    kernel_mesh = _kernel_mesh(mesh_devices)
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
        select = jax.jit(lambda *a: paged_pool_select(indexer, *a, pool_chunk=pool_chunk, kernel_mesh=kernel_mesh))
        done = [0, 0, 0]
        for step in steps:
            hs, qs, owners = [], [], []
            for s, n in enumerate(step):
                hs.append(streams_h[s][done[s] : done[s] + n])
                qs.append(streams_q[s][done[s] : done[s] + n])
                owners += [(s, p) for p in range(done[s], done[s] + n)]
                done[s] += n
            hs.append(jnp.asarray(rng.standard_normal((PAD_TOKENS, HID)), jnp.float32))
            qs.append(jnp.asarray(rng.standard_normal((PAD_TOKENS, HID)), jnp.float32))
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
