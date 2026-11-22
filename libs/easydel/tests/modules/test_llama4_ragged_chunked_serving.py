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

"""Llama4 chunked-attention layer served through ragged paged attention v3 vs its dense forward.

The chunked (RoPE) layer is driven through a ``RaggedPagesCacheView`` over a multi-step schedule
(prefill chunks, an idle row, decode) with a permuted page table. Every scheduled token must equal the
same layer's dense full-history forward under the chunked mask (``k <= q`` and ``k // C == q // C``).
Contexts exceed the chunk size so the mask is genuinely restrictive; a plain-causal dense forward is
the negative control.
"""

from __future__ import annotations

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from easydel.caching import RaggedPagesCacheConfig, RaggedPagesCacheView, RaggedPagesMetadata
from easydel.modules.llama4.modeling_llama4 import Llama4TextAttention
from ejkernel.types import MaskInfo
from spectrax import common_types

HID, HEADS, KV_HEADS, HEAD_DIM = 64, 4, 2, 128
CHUNK, PAGE, PAD_TOKENS = 8, 8, 3


def _layer(attn_mechanism: str):
    config = ed.Llama4TextConfig(
        vocab_size=128,
        hidden_size=HID,
        intermediate_size=128,
        intermediate_size_mlp=128,
        num_hidden_layers=4,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        use_qk_norm=False,
        attention_chunk_size=CHUNK,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    config.attn_dtype = jnp.float32  # attention ops cast to attn_dtype (default bf16)
    config.attn_mechanism = attn_mechanism
    with config.mesh:
        layer = Llama4TextAttention(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
            layer_idx=0,  # RoPE + chunked_attention
        )
    assert config.layer_types[0] == "chunked_attention"
    return config, layer


def _dense(config, layer, hidden, *, chunked: bool):
    n = hidden.shape[0]
    mask_info = MaskInfo.dynamic_init(input_ids=jnp.ones((1, n), jnp.int32))
    mask_info = mask_info.apply_chunked(CHUNK) if chunked else mask_info.apply_causal()
    with jax.default_matmul_precision("highest"), config.mesh:
        out = layer(hidden[None], mask_info, jnp.arange(n)[None, :], common_types.MODE_TRAIN)
    return np.asarray(out.attention_output[0])


def test_ragged_chunked_layer_matches_dense_chunked_reference():
    ref_config, ref_layer = _layer("vanilla")
    config, layer = _layer("ragged_page_attention_v3")
    assert layer.attention_performer.chunk_size == CHUNK

    steps = [[13, 9], [11, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    lengths = [sum(s[i] for s in steps) for i in range(2)]
    rng = np.random.default_rng(0)
    streams = [jnp.asarray(rng.standard_normal((n, HID)), jnp.float32) for n in lengths]

    chunked_ref = [_dense(ref_config, ref_layer, h, chunked=True) for h in streams]
    causal_ref = [_dense(ref_config, ref_layer, h, chunked=False) for h in streams]

    def rel(a, b):
        return np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-30)

    # Negative control: past the first chunk the chunked and plain-causal references must differ.
    gap = min(rel(causal_ref[s][CHUNK:], chunked_ref[s][CHUNK:]).min() for s in range(2))

    max_seqs, pages_per_seq = 3, -(-max(lengths) // PAGE)
    cache_cfg = RaggedPagesCacheConfig.create(
        mesh=config.mesh,
        runtime_sharding_resolver=config.runtime_sharding_resolver,
        kvdtype=jnp.float32,
        num_hidden_layers=1,
        num_kv_heads=KV_HEADS,
        max_model_length=pages_per_seq * PAGE,
        kv_head_dim_size=HEAD_DIM,
        page_size=PAGE,
        version="v3",
        max_cache_tokens=(max_seqs * pages_per_seq + 3) * PAGE,
    )
    num_pages = cache_cfg.num_pages
    assert num_pages >= max_seqs * pages_per_seq
    tables = jnp.asarray(
        rng.permutation(num_pages)[: max_seqs * pages_per_seq].reshape(max_seqs, pages_per_seq), jnp.int32
    )

    errors = []
    with jax.default_matmul_precision("highest"), config.mesh:
        view = RaggedPagesCacheView.init(
            cache_cfg, layer_index=0, mesh=config.mesh, runtime_sharding_resolver=config.runtime_sharding_resolver
        )
        done = [0, 0]
        for step in steps:
            chunks, owners, positions = [], [], []
            for s, n in enumerate(step):
                chunks.append(streams[s][done[s] : done[s] + n])
                owners += [(s, p) for p in range(done[s], done[s] + n)]
                positions += list(range(done[s], done[s] + n))
                done[s] += n
            chunks.append(jnp.asarray(rng.standard_normal((PAD_TOKENS, HID)), jnp.float32))
            positions += [0] * PAD_TOKENS
            packed = jnp.concatenate(chunks)[None]
            live = sum(1 for n in step if n > 0)
            order = [s for s, n in enumerate(step) if n > 0] + [s for s, n in enumerate(step) if n == 0]
            assert order == sorted(order), "schedule keeps live rows first"
            meta = RaggedPagesMetadata(
                pages_tables=tables,
                context_lens=jnp.asarray([*done, 0], jnp.int32),
                query_start_loc=jnp.asarray(np.concatenate([[0], np.cumsum([*step, 0])]), jnp.int32),
                num_seqs=jnp.asarray([live], jnp.int32),
                request_distribution=jnp.asarray([0, 0, live], jnp.int32),
                page_size=PAGE,
                version="v3",
            )
            out = layer(
                packed,
                None,
                jnp.asarray(positions, jnp.int32)[None, :],
                common_types.MODE_PREFILL,
                cache_view=view,
                cache_metadata=meta,
            )
            view = out.cache_view
            got = np.asarray(out.attention_output[0])
            for t, (s, p) in enumerate(owners):
                errors.append((float(rel(got[t], chunked_ref[s][p])), s, p))

    worst = max(errors)
    print("serving-vs-chunked rel err max:", worst, "tokens:", len(errors), "chunked-vs-causal gap min:", gap)
    assert worst[0] < 1e-4, f"ragged chunked serving diverges from dense chunked forward: {worst}"
    # The gap must dwarf the serving error, otherwise a silent plain-causal fallback would pass.
    assert gap > 100 * max(worst[0], 1e-7), (gap, worst)
