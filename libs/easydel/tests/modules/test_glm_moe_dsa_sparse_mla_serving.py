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

"""GLM-MoE-DSA serving path (MLA ragged pages + paged per-token indexer + sparse MLA op).

The serving layer runs a multi-step ragged schedule (chunked prefill, idle rows, decode)
over a permuted page table. Every scheduled token's attention output must equal the dense
top-k-masked reference forward of the same layer over the request's full history. Contexts
exceed ``index_topk`` so the per-token DSA selection is genuinely sparse.
"""

from __future__ import annotations

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from easydel.caching import MLARaggedPagesCacheConfig, MLARaggedPagesCacheView, RaggedPagesMetadata
from ejkernel.types import MaskInfo
from spectrax import common_types

HID, TOPK, PAGE = 32, 8, 8
PAD_TOKENS = 3


def _model(attn_mechanism: str | None, topk: int = TOPK):
    config = ed.GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=HID,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
        max_position_embeddings=128,
        rope_theta=10000.0,
        index_topk=topk,
        index_head_dim=16,
        index_n_heads=4,
        mlp_layer_types=["dense"],
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    config.attn_dtype = jnp.float32
    if attn_mechanism is not None:
        config.attn_mechanism = attn_mechanism
    with config.mesh:
        model = ed.GlmMoeDsaForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
    return config, model, model.model.layers[0].self_attn


def _dense_forward(layer, model, h):
    n = h.shape[0]
    out = layer.forward_mla(
        h[None],
        MaskInfo.dynamic_init(input_ids=jnp.ones((1, n), jnp.int32)),
        jnp.arange(n)[None, :],
        common_types.MODE_TRAIN,
        frequencies=model.model.frequencies,
    )
    return np.asarray(out.attention_output[0])


def _rel(a, b):
    return np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-30)


def test_ragged_sparse_token_dsa_matches_dense_topk_reference():
    ref_config, ref_model, ref_layer = _model(None)
    config, model, layer = _model("multi_latent_ragged_page_attention_v2")
    np.testing.assert_array_equal(
        np.asarray(ref_layer.indexer.wq_b.weight.value), np.asarray(layer.indexer.wq_b.weight.value)
    )

    steps = [[13, 9], [11, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    lengths = [sum(s[i] for s in steps) for i in range(2)]
    rng = np.random.default_rng(0)
    streams = [jnp.asarray(rng.standard_normal((n, HID)), jnp.float32) for n in lengths]

    with jax.default_matmul_precision("highest"), ref_config.mesh:
        reference = [_dense_forward(ref_layer, ref_model, h) for h in streams]

    # Negative control: a selection covering every token (dense attention) must differ from
    # the top-k reference past the selection width, or a silent dense fallback would pass.
    dense_config, dense_model, dense_layer = _model(None, topk=64)
    with jax.default_matmul_precision("highest"), dense_config.mesh:
        dense_out = _dense_forward(dense_layer, dense_model, streams[0])
    same_prefix = _rel(dense_out[:TOPK], reference[0][:TOPK])
    sparse_gap = _rel(dense_out[TOPK:], reference[0][TOPK:])
    print("dense-vs-topk prefix rel:", same_prefix.max(), "gap past width min/max:", sparse_gap.min(), sparse_gap.max())

    max_seqs, pages_per_seq = 3, -(-max(lengths) // PAGE)
    cache_cfg = MLARaggedPagesCacheConfig.create(
        mesh=config.mesh,
        runtime_sharding_resolver=config.runtime_sharding_resolver,
        kvdtype=jnp.float32,
        num_hidden_layers=1,
        num_kv_heads=2,
        max_model_length=pages_per_seq * PAGE,
        kv_lora_rank=config.kv_lora_rank,
        qk_rope_head_dim=config.qk_rope_head_dim,
        page_size=PAGE,
        max_cache_tokens=(max_seqs * pages_per_seq + 3) * PAGE,
        indexer_key_dim=config.index_head_dim,
        indexer_gate_dim=0,
        indexer_kpool=1,
    )
    num_pages = cache_cfg.num_pages
    assert num_pages >= max_seqs * pages_per_seq
    tables = jnp.asarray(
        rng.permutation(num_pages)[: max_seqs * pages_per_seq].reshape(max_seqs, pages_per_seq), jnp.int32
    )

    with jax.default_matmul_precision("highest"), config.mesh:
        view = MLARaggedPagesCacheView.init(
            cache_cfg, layer_index=0, mesh=config.mesh, runtime_sharding_resolver=config.runtime_sharding_resolver
        )
        assert view.indexer_token_pages is not None and view.indexer_pool_pages is None
        done = [0, 0]
        errors = []
        for step in steps:
            chunks, owners = [], []
            for s, n in enumerate(step):
                chunks.append(streams[s][done[s] : done[s] + n])
                owners += [(s, p) for p in range(done[s], done[s] + n)]
                done[s] += n
            chunks.append(jnp.asarray(rng.standard_normal((PAD_TOKENS, HID)), jnp.float32))
            packed = jnp.concatenate(chunks)[None]
            positions = [p for _, p in owners] + [0] * PAD_TOKENS  # RoPE needs true positions
            live = sum(1 for n in step if n > 0)
            assert [s for s, n in enumerate(step) if n > 0] == list(range(live)), "live rows first"
            meta = RaggedPagesMetadata(
                pages_tables=tables,
                context_lens=jnp.asarray([*done, 0], jnp.int32),
                query_start_loc=jnp.asarray(np.concatenate([[0], np.cumsum([*step, 0])]), jnp.int32),
                num_seqs=jnp.asarray([live], jnp.int32),
                page_size=PAGE,
                version="v3",
            )
            out = layer.forward_mla(
                packed,
                None,
                jnp.asarray(positions, jnp.int32)[None],
                common_types.MODE_PREFILL,
                cache_view=view,
                cache_metadata=meta,
                frequencies=model.model.frequencies,
            )
            view = out.cache_view
            got = np.asarray(out.attention_output[0])
            for t, (s, p) in enumerate(owners):
                errors.append((float(_rel(got[t], reference[s][p])), s, p))

    worst = max(errors)
    print("serving-vs-topk rel err max:", worst, "tokens checked:", len(errors))
    assert same_prefix.max() < 1e-4, same_prefix
    assert worst[0] < 1e-4, f"serving path diverges from top-k reference: {worst}"
    assert sparse_gap.min() > 100 * max(worst[0], 1e-7), (sparse_gap.min(), worst)
