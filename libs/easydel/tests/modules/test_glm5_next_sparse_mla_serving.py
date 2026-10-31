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

"""GLM-5 DSA serving path (MLA ragged pages + paged indexer + sparse MLA op) vs the reference.

The serving layer is driven through a multi-step ragged schedule (chunked prefill, idle
rows, decode) with a permuted page table. Every scheduled token's attention output must
equal the dense top-k-masked reference forward of the same layer over that request's full
history. Contexts exceed ``index_topk`` so the DSA selection is genuinely sparse.
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

HID, KPOOL, TOPK, PAGE = 32, 4, 8, 8
PAD_TOKENS = 3


def _model(attn_mechanism: str | None, topk: int = TOPK):
    config = ed.Glm5NextTextConfig(
        vocab_size=128,
        hidden_size=HID,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        qk_nope_head_dim=8,
        v_head_dim=8,
        index_topk=topk,
        index_head_dim=8,
        index_n_heads=2,
        index_kpool=KPOOL,
        layer_types=["deepseek_sparse_attention"],
        mlp_layer_types=["dense"],
        hc_mult=2,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    # Attention ops cast q/k/v to ``attn_dtype`` (default bf16); compare both paths in f32.
    config.attn_dtype = jnp.float32
    if attn_mechanism is not None:
        # glm5_next's performer reads ``attn_mechanism`` (as eSurge configures it).
        config.attn_mechanism = attn_mechanism
    with config.mesh:
        model = ed.Glm5NextForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
    layer = model.model.layers[0].self_attn
    # Non-trivial pooling weights (defaults are ape=0, gate=1), identical in both models.
    rng = np.random.default_rng(1)
    layer.indexer.index_kpool_compress_ape.value = jnp.asarray(rng.standard_normal((KPOOL, 8)), jnp.float32)
    layer.indexer.index_kpool_compress_gate.value = jnp.asarray(0.2 * rng.standard_normal((8, HID)), jnp.float32)
    return config, model, layer


def test_ragged_sparse_dsa_matches_dense_topk_reference():
    ref_config, ref_model, ref_layer = _model(None)
    config, _model_r, layer = _model("multi_latent_ragged_page_attention_v2")
    # Same seed -> same parameters; guard the premise of the comparison.
    np.testing.assert_array_equal(
        np.asarray(ref_layer.mla_kv_b_proj.weight.value), np.asarray(layer.mla_kv_b_proj.weight.value)
    )
    np.testing.assert_array_equal(
        np.asarray(ref_layer.indexer.wq_b.weight.value), np.asarray(layer.indexer.wq_b.weight.value)
    )

    steps = [[13, 9], [11, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    lengths = [sum(s[i] for s in steps) for i in range(2)]
    rng = np.random.default_rng(0)
    streams = [jnp.asarray(rng.standard_normal((n, HID)), jnp.float32) for n in lengths]

    with jax.default_matmul_precision("highest"), ref_config.mesh:
        reference = []
        for h in streams:
            n = h.shape[0]
            out = ref_layer.forward_mla(
                h[None],
                MaskInfo.dynamic_init(input_ids=jnp.ones((1, n), jnp.int32)),
                jnp.arange(n)[None, :],
                common_types.MODE_TRAIN,
                frequencies=ref_model.model.frequencies,
            )
            reference.append(np.asarray(out.attention_output[0]))

    # Negative control: the same weights with a selection covering every token (dense
    # attention) must differ from the top-k reference once the context exceeds the
    # selection width, otherwise this test could not detect a silent dense fallback.
    dense_config, dense_model, dense_layer = _model(None, topk=64)
    with jax.default_matmul_precision("highest"), dense_config.mesh:
        h = streams[0]
        n = h.shape[0]
        dense_out = np.asarray(
            dense_layer.forward_mla(
                h[None],
                MaskInfo.dynamic_init(input_ids=jnp.ones((1, n), jnp.int32)),
                jnp.arange(n)[None, :],
                common_types.MODE_TRAIN,
                frequencies=dense_model.model.frequencies,
            ).attention_output[0]
        )

    def rel(a, b):
        return np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-30)

    covered = TOPK + KPOOL - 1  # positions below this attend to everything either way
    same_prefix = rel(dense_out[:covered], reference[0][:covered])
    sparse_gap = rel(dense_out[covered:], reference[0][covered:])
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
        qk_rope_head_dim=config.mla_cache_rope_width,
        page_size=PAGE,
        max_cache_tokens=(max_seqs * pages_per_seq + 3) * PAGE,
        indexer_key_dim=config.index_head_dim,
        indexer_gate_dim=config.index_head_dim,
        indexer_kpool=KPOOL,
    )
    assert cache_cfg.indexer_dtype == jnp.float32
    num_pages = cache_cfg.num_pages
    assert num_pages >= max_seqs * pages_per_seq
    tables = jnp.asarray(
        rng.permutation(num_pages)[: max_seqs * pages_per_seq].reshape(max_seqs, pages_per_seq), jnp.int32
    )

    with jax.default_matmul_precision("highest"), config.mesh:
        view = MLARaggedPagesCacheView.init(
            cache_cfg, layer_index=0, mesh=config.mesh, runtime_sharding_resolver=config.runtime_sharding_resolver
        )
        assert view.indexer_token_pages is not None and view.indexer_pool_pages is not None
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
            live = sum(1 for n in step if n > 0)
            # Idle rows keep their context but schedule nothing; order live rows first.
            order = [s for s, n in enumerate(step) if n > 0] + [s for s, n in enumerate(step) if n == 0]
            assert order == sorted(order), "schedule keeps live rows first"
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
                jnp.zeros((1, packed.shape[1]), jnp.int32),
                common_types.MODE_PREFILL,
                cache_view=view,
                cache_metadata=meta,
            )
            view = out.cache_view
            got = np.asarray(out.attention_output[0])
            for t, (s, p) in enumerate(owners):
                errors.append((float(rel(got[t], reference[s][p])), s, p))

    worst = max(errors)
    print("serving-vs-topk rel err max:", worst, "tokens checked:", len(errors))
    assert same_prefix.max() < 1e-4, same_prefix
    assert worst[0] < 1e-4, f"serving path diverges from top-k reference: {worst}"
    # The comparison must resolve sparse vs dense: the smallest sparse/dense gap is far
    # above the serving error, so a silent dense fallback would fail the assertion above.
    assert sparse_gap.min() > 100 * max(worst[0], 1e-7), (sparse_gap.min(), worst)
