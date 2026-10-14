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

"""DeepSeek-V4 block compressor step (batched packed prefill) equals the per-token decode step run column by column.

Covers the single-series (HCA) and two-series (CSA, Ca overlap carry) streams, starting from ragged prefixes (open
groups carried in from earlier steps) and applying two consecutive blocks with ragged valid prefixes (0, 1, partial,
full), so group closes inside a block, overlap carry across groups and across steps, and state refresh are all
exercised.
"""

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.layers import RMSNorm
from easydel.layers.compressed_window import advance_compressor_stream
from easydel.layers.rotary import apply_rope_interleaved_partial, labeled_rope_cos_sin

HEAD_DIM = 16
BLOCK = 7
PREFIX = np.array([0, 1, 3, 5, 2], dtype=np.int32)
BLOCK_VALID = (np.array([7, 1, 0, 3, 6], dtype=np.int32), np.array([4, 7, 2, 0, 7], dtype=np.int32))


def _config() -> ed.DeepseekV4Config:
    return ed.DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        q_lora_rank=32,
        num_experts_per_tok=2,
        n_routed_experts=8,
        n_shared_experts=1,
        max_position_embeddings=256,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
        ],
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
        hc_mult=2,
        hc_sinkhorn_iters=5,
        mlp_layer_types=["hash_moe", "moe", "moe", "moe"],
        sliding_window=8,
        o_groups=2,
        o_lora_rank=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=128,
        partial_rotary_factor=0.25,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )


@pytest.mark.parametrize(("rate", "two_series"), [(4, False), (2, True), (3, True)])
def test_block_compress_matches_per_token_decode(rate, two_series):
    config = _config()
    feat = 2 * HEAD_DIM if two_series else HEAD_DIM
    batch = PREFIX.shape[0]
    total = int(PREFIX.max()) + 2 * BLOCK
    n_slots = total // rate + 1
    rng = np.random.default_rng(rate)
    kv_all = jnp.asarray(rng.normal(size=(batch, total, feat)), jnp.float32)
    gate_all = jnp.asarray(rng.normal(size=(batch, total, feat)), jnp.float32)
    position_bias = jnp.asarray(rng.normal(size=(rate, feat)) * 0.5, jnp.float32)
    kv_norm = RMSNorm(dim=HEAD_DIM, eps=1e-6, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    rows = jnp.arange(batch)

    state0 = dict(
        buffer_kv=jnp.zeros((batch, rate, feat), jnp.float32),
        buffer_gate=jnp.zeros((batch, rate, feat), jnp.float32),
        entries=jnp.zeros((batch, n_slots, HEAD_DIM), jnp.float32),
        overlap_kv=jnp.zeros((batch, rate, HEAD_DIM), jnp.float32) if two_series else None,
        overlap_gate=jnp.full((batch, rate, HEAD_DIM), -jnp.inf, jnp.float32) if two_series else None,
    )

    def finalize(pooled, entry):
        normed = kv_norm(pooled)
        cos, sin = labeled_rope_cos_sin(config, "compress", (entry * rate).astype(jnp.int32), normed.dtype)
        return apply_rope_interleaved_partial(normed[:, None], cos, sin, unsqueeze_dim=1)[:, 0]

    common = dict(position_bias=position_bias, rate=rate, finalize=finalize)

    def decode(state, start, n_valid, steps):
        for col in range(steps):
            pos = jnp.asarray(start + col, jnp.int32)
            valid = jnp.asarray(col < n_valid)
            take = jnp.clip(pos, 0, total - 1)
            out = advance_compressor_stream(
                kv=kv_all[rows, take][:, None],
                gate=gate_all[rows, take][:, None],
                position=pos,
                valid=valid,
                **state,
                **common,
            )
            state = dict(zip(("entries", "buffer_kv", "buffer_gate", "overlap_kv", "overlap_gate"), out, strict=True))
            state = {k: state[k] for k in state0}
        return state

    def extend(state, start, n_valid):
        positions = jnp.asarray(start[:, None] + np.arange(BLOCK)[None, :], jnp.int32)
        take = jnp.clip(positions, 0, total - 1)
        out = advance_compressor_stream(
            kv=kv_all[rows[:, None], take],
            gate=gate_all[rows[:, None], take],
            positions=positions,
            valid=jnp.asarray(np.arange(BLOCK)[None, :] < n_valid[:, None]),
            **state,
            **common,
        )
        state = dict(zip(("entries", "buffer_kv", "buffer_gate", "overlap_kv", "overlap_gate"), out, strict=True))
        return {k: state[k] for k in state0}

    with jax.default_matmul_precision("highest"):
        # Ragged prefixes leave open groups (and, for two-series streams, overlaps) in the state.
        prefix_state = decode(state0, np.zeros(batch, np.int32), PREFIX, int(PREFIX.max()))
        ref, got = prefix_state, prefix_state
        start = PREFIX.copy()
        for n_valid in BLOCK_VALID:
            ref = decode(ref, start, n_valid, BLOCK)
            got = extend(got, start, n_valid)
            start = start + n_valid
            for name in state0:
                if state0[name] is None:
                    assert got[name] is None
                    continue
                np.testing.assert_allclose(
                    np.asarray(got[name]), np.asarray(ref[name]), rtol=1e-5, atol=1e-5, err_msg=f"{name} after {start}"
                )
    # Entries were actually emitted (the comparison is not vacuous).
    assert float(jnp.max(jnp.abs(got["entries"]))) > 0.0


def test_selective_indexer_block_extend_matches_per_token_decode():
    """``CompressedIndexer.cached_extend`` (block stream advance + reach-bounded shared selector) equals the per-token
    ``cached_forward`` decode loop: same selected entry set per valid cell and the same indexer stream state.

    ``index_topk=2`` over up to 9 reachable entries, so the selection really excludes entries (the V4 serving fixture
    keeps the indexer in its vacuous regime and never reaches this path).
    """
    from easydel.caching import CompressedWindowCache, CompressedWindowCacheConfig
    from easydel.layers.indexer import IndexerSelection
    from easydel.modules.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Indexer
    from spectrax import PartitionAxis

    config = ed.DeepseekV4Config(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=4,
        qk_rope_head_dim=2,
        q_lora_rank=3,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
        rms_norm_eps=3e-4,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    module = DeepseekV4Indexer(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(13)
    )
    batch = PREFIX.shape[0]
    total = int(PREFIX.max()) + 2 * BLOCK
    cache_config = CompressedWindowCacheConfig.create(
        num_hidden_layers=1,
        partition_axis=PartitionAxis(),
        batch_size=batch,
        max_length=24,
        sliding_window=4,
        head_dim=4,
        index_head_dim=4,
        csa_rate=2,
        hca_rate=4,
        layer_types=("compressed_sparse_attention",),
    )
    view0 = CompressedWindowCache.init_cache(config=cache_config, dtype=jnp.float32).views[0]
    n_slots = view0.num_entry_slots
    rng = np.random.default_rng(57)
    hidden = rng.normal(size=(batch, total + BLOCK, 8)).astype(np.float32)
    residual = rng.normal(size=(batch, total + BLOCK, 3)).astype(np.float32)
    rows = np.arange(batch)
    step = jax.jit(module.cached_forward)
    extend = jax.jit(module.cached_extend)

    def decode(view, start, n_valid, steps):
        masks = []
        for col in range(steps):
            pos = (start + col).astype(np.int32)
            view = view.replace(cache_position=jnp.asarray(pos))
            idx, view = step(
                jnp.asarray(hidden[rows, pos][:, None]),
                jnp.asarray(residual[rows, pos][:, None]),
                jnp.asarray(pos)[:, None],
                view,
                valid=jnp.asarray(col < n_valid),
            )
            masks.append(np.asarray(IndexerSelection(idx).to_mask(n_slots))[:, 0])
        return view, np.stack(masks, axis=1)

    fields = (
        "indexer_entries",
        "indexer_buffer_kv",
        "indexer_buffer_gate",
        "indexer_overlap_kv",
        "indexer_overlap_gate",
    )
    ref, _ = decode(view0, np.zeros(batch, np.int32), PREFIX, int(PREFIX.max()))
    got = ref
    start = PREFIX.copy()
    max_reach = 0
    for n_valid in BLOCK_VALID:
        positions = start[:, None] + np.arange(BLOCK)[None, :]
        valid = np.arange(BLOCK)[None, :] < n_valid[:, None]
        ref, ref_mask = decode(ref, start, n_valid, BLOCK)
        top_k, got = extend(
            jnp.asarray(hidden[rows[:, None], positions]),
            jnp.asarray(residual[rows[:, None], positions]),
            jnp.asarray(positions, jnp.int32),
            jnp.asarray(valid),
            got,
        )
        assert top_k is not None
        got_mask = np.asarray(IndexerSelection(top_k).to_mask(n_slots))
        np.testing.assert_array_equal(got_mask[valid], ref_mask[valid], err_msg=f"selection after {start}")
        for name in fields:
            np.testing.assert_allclose(
                np.asarray(getattr(got, name)), np.asarray(getattr(ref, name)), rtol=1e-5, atol=1e-5, err_msg=name
            )
        max_reach = max(max_reach, int(((positions + 1) // 2)[valid].max(initial=0)))
        start = start + n_valid
    # The selection truncated (reach beyond index_topk), so the comparison is not vacuous.
    assert max_reach > config.index_topk
