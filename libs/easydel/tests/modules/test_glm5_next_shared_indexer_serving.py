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

""" "Shared" DSA indexer layers own no indexer pages but still serve sparsely.

Only layers with a "full" indexer allocate paged indexer state. A shared layer reuses the
previous full layer's chunk ids, so its serving route must follow the cache config, not
its own (absent) buffers; otherwise it silently attends densely.
"""

from __future__ import annotations

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from easydel.caching import MLARaggedPagesCacheConfig, MLARaggedPagesCacheView, RaggedPagesMetadata
from spectrax import common_types

HID, KPOOL, TOPK, PAGE, SEQ, PAD_TOKENS = 32, 4, 8, 8, 24, 3


def _model():
    config = ed.Glm5NextTextConfig(
        vocab_size=128,
        hidden_size=HID,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        qk_nope_head_dim=8,
        v_head_dim=8,
        index_topk=TOPK,
        index_head_dim=8,
        index_n_heads=2,
        index_kpool=KPOOL,
        layer_types=["deepseek_sparse_attention"] * 2,
        indexer_types=["full", "shared"],
        mlp_layer_types=["dense", "dense"],
        hc_mult=2,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    config.attn_dtype = jnp.float32
    config.attn_mechanism = "multi_latent_ragged_page_attention_v2"
    with config.mesh:
        model = ed.Glm5NextForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
    return config, model


def test_shared_indexer_layer_without_pages_serves_the_reused_selection():
    config, model = _model()
    full_layer = model.model.layers[0].self_attn
    shared_layer = model.model.layers[1].self_attn

    max_seqs, pages_per_seq = 2, -(-SEQ // PAGE)
    cache_cfg = MLARaggedPagesCacheConfig.create(
        mesh=config.mesh,
        runtime_sharding_resolver=config.runtime_sharding_resolver,
        kvdtype=jnp.float32,
        num_hidden_layers=2,
        num_kv_heads=2,
        max_model_length=pages_per_seq * PAGE,
        kv_lora_rank=config.kv_lora_rank,
        qk_rope_head_dim=config.mla_cache_rope_width,
        page_size=PAGE,
        max_cache_tokens=(max_seqs * pages_per_seq + 3) * PAGE,
        indexer_key_dim=config.index_head_dim,
        indexer_gate_dim=config.index_head_dim,
        indexer_kpool=KPOOL,
        indexer_num_layers=1,
    )
    assert cache_cfg.has_indexer_pages

    rng = np.random.default_rng(0)
    tables = jnp.asarray(
        rng.permutation(cache_cfg.num_pages)[: max_seqs * pages_per_seq].reshape(max_seqs, pages_per_seq),
        jnp.int32,
    )
    hidden = jnp.asarray(rng.standard_normal((1, SEQ + PAD_TOKENS, HID)), jnp.float32)
    positions = jnp.zeros((1, hidden.shape[1]), jnp.int32)
    meta = RaggedPagesMetadata(
        pages_tables=tables,
        context_lens=jnp.asarray([SEQ, 0], jnp.int32),
        query_start_loc=jnp.asarray([0, SEQ, SEQ], jnp.int32),
        num_seqs=jnp.asarray([1], jnp.int32),
        page_size=PAGE,
        version="v3",
    )

    def view(layer_index, with_pages=True):
        return MLARaggedPagesCacheView.init(
            cache_cfg,
            layer_index=layer_index,
            mesh=config.mesh,
            runtime_sharding_resolver=config.runtime_sharding_resolver,
            with_indexer_pages=with_pages,
        )

    def run_shared(cache_view, previous):
        out, _ = shared_layer.forward_mla(
            hidden,
            None,
            positions,
            common_types.MODE_PREFILL,
            cache_view=cache_view,
            cache_metadata=meta,
            previous_topk_indices=previous,
            return_topk_indices=True,
        )
        return np.asarray(out.attention_output[0, :SEQ])

    with jax.default_matmul_precision("highest"), config.mesh:
        _, chunk_ids = full_layer.forward_mla(
            hidden,
            None,
            positions,
            common_types.MODE_PREFILL,
            cache_view=view(0),
            cache_metadata=meta,
            previous_topk_indices=None,
            return_topk_indices=True,
        )
        pageless = view(1, with_pages=False)
        assert pageless.indexer_token_pages is None and pageless.indexer_pool_pages is None

        got = run_shared(pageless, chunk_ids)
        want = run_shared(view(1), chunk_ids)
        # Every chunk selected: the dense-equivalent result of the same layer.
        num_chunks = -(-SEQ // KPOOL)
        all_chunks = jnp.broadcast_to(jnp.arange(num_chunks, dtype=jnp.int32), (chunk_ids.shape[0], num_chunks))
        dense = run_shared(view(1), all_chunks)

    def rel(a, b):
        return np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-30)

    covered = TOPK + KPOOL - 1
    err = rel(got, want).max()
    gap = rel(dense[covered:], want[covered:]).min()
    print("pageless-vs-paged shared rel err:", err, "sparse-vs-dense gap min:", gap)
    assert err < 1e-5, err
    # The selection must matter past its width, or a dense fallback would go unnoticed.
    assert gap > 100 * max(err, 1e-7), (gap, err)
