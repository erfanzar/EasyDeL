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

"""Falcon ALiBi attention served through ragged paged attention v3 vs its dense forward.

The ALiBi layer is driven through a ``RaggedPagesCacheView`` over a multi-step schedule (prefill
chunks, an idle row, decode) with a permuted page table. Every scheduled token must equal the same
layer's dense full-history forward with the HF Falcon bias (``bf16(slope_h) * key_position /
sqrt(head_dim)`` added to the logits). A dense forward without any bias is the negative control.
"""

from __future__ import annotations

import math

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from easydel.caching import RaggedPagesCacheConfig, RaggedPagesCacheView, RaggedPagesMetadata
from easydel.modules.falcon.modeling_falcon import FalconAttention, bloom_alibi_slopes
from ejkernel.types import MaskInfo
from spectrax import common_types

HID, HEADS = 512, 4
HEAD_DIM = HID // HEADS
PAGE, PAD_TOKENS = 8, 3


def _layer(attn_mechanism: str):
    config = ed.FalconConfig(
        vocab_size=128,
        hidden_size=HID,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        num_kv_heads=HEADS,
        ffn_hidden_size=128,
        alibi=True,
        multi_query=False,
        new_decoder_architecture=False,
        parallel_attn=False,
        bias=False,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    config.attn_dtype = jnp.float32
    config.attn_mechanism = attn_mechanism
    with config.mesh:
        layer = FalconAttention(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
            layer_idx=0,
        )
    return config, layer


def _dense(config, layer, hidden, *, with_alibi: bool):
    n = hidden.shape[0]
    mask_info = MaskInfo.dynamic_init(input_ids=jnp.ones((1, n), jnp.int32))
    alibi = None
    if with_alibi:
        # Independent of the kernel path: HF bias, slopes rounded to bf16 like ``built_bloom_alibi``.
        slopes = np.asarray(bloom_alibi_slopes(HEADS).astype(jnp.bfloat16).astype(jnp.float32)) / math.sqrt(HEAD_DIM)
        alibi = jnp.asarray(slopes[None, :, None, None] * np.arange(n, dtype=np.float32)[None, None, None, :])
    with jax.default_matmul_precision("highest"), config.mesh:
        out = layer(
            hidden[None],
            mask_info,
            jnp.arange(n)[None, :],
            common_types.MODE_TRAIN,
            alibi=alibi,
        )
    return np.asarray(out.attention_output[0])


def test_ragged_alibi_layer_matches_dense_reference():
    ref_config, ref_layer = _layer("vanilla")
    config, layer = _layer("ragged_page_attention_v3")

    steps = [[13, 9], [11, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    lengths = [sum(s[i] for s in steps) for i in range(2)]
    rng = np.random.default_rng(0)
    streams = [jnp.asarray(rng.standard_normal((n, HID)), jnp.float32) for n in lengths]

    alibi_ref = [_dense(ref_config, ref_layer, h, with_alibi=True) for h in streams]
    plain_ref = [_dense(ref_config, ref_layer, h, with_alibi=False) for h in streams]

    def rel(a, b):
        return np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-30)

    # Negative control: the bias must move the dense output well beyond the tolerance below.
    gap = min(rel(plain_ref[s][1:], alibi_ref[s][1:]).min() for s in range(2))

    max_seqs, pages_per_seq = 3, -(-max(lengths) // PAGE)
    cache_cfg = RaggedPagesCacheConfig.create(
        mesh=config.mesh,
        runtime_sharding_resolver=config.runtime_sharding_resolver,
        kvdtype=jnp.float32,
        num_hidden_layers=1,
        num_kv_heads=HEADS,
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
                errors.append((float(rel(got[t], alibi_ref[s][p])), s, p))

    worst = max(errors)
    print("serving-vs-alibi rel err max:", worst, "tokens:", len(errors), "alibi-vs-plain gap min:", gap)
    assert worst[0] < 1e-4, f"ragged ALiBi serving diverges from dense ALiBi forward: {worst}"
    assert gap > 100 * max(worst[0], 1e-7), (gap, worst)
