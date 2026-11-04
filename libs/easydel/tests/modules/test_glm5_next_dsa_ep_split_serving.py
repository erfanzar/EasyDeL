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

"""``dsa_serving_ep_split``: GLM-5 DSA serving with the sparse MLA attention split over the expert-parallel axis.

The flag must not change results. One layer per mesh runs a multi-step ragged schedule through the MLA ragged-pages
serving path (paged indexer + sparse MLA op) twice, with the flag off and on: chunked prefill whose token counts do not
divide the expert-parallel size, a mixed step (one row decodes while the other prefills), an idle row, padding tokens,
then decode. Every step's layer output and the final caches must match. On expert-parallel meshes the flag-on program
must really split (it indexes its expert-parallel shard); with ``ep == 1`` and with data-parallel KV pages it must keep
the unsplit program. ``None``, the automatic default, must build the flag-on program wherever the flag splits and the
unsplit program elsewhere.
"""

from __future__ import annotations

import collections
import functools

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.caching import MLARaggedPagesCacheConfig, MLARaggedPagesCacheView, RaggedPagesMetadata
from easydel.operations.kernels import multi_latent_ragged_page_attention as mlrpa
from spectrax import common_types

HID, HEADS, KPOOL, TOPK, PAGE = 64, 4, 4, 8, 8
PAD_TOKENS = 3
# Tokens per row and step. With the padding tokens a step packs 25, 11, 14, 5, 5, 5 tokens; contexts exceed
# index_topk, so the selection is sparse.
STEPS = [[13, 9], [1, 7], [11, 0], [1, 1], [1, 1], [1, 1]]


def _primitive_counts(fn, *args) -> collections.Counter:
    counts = collections.Counter()

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            counts.update([eqn.primitive.name])
            for param in eqn.params.values():
                for sub in param if isinstance(param, (tuple, list)) else (param,):
                    sub = getattr(sub, "jaxpr", sub)
                    if hasattr(sub, "eqns"):
                        walk(sub)

    walk(jax.make_jaxpr(fn)(*args).jaxpr)
    return counts


def _layer(dims):
    config = ed.Glm5NextTextConfig(
        vocab_size=128,
        hidden_size=HID,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_rope_head_dim=0,
        qk_nope_head_dim=16,
        v_head_dim=16,
        index_topk=TOPK,
        index_head_dim=16,
        index_n_heads=2,
        index_kpool=KPOOL,
        layer_types=["deepseek_sparse_attention"],
        mlp_layer_types=["dense"],
        hc_mult=2,
        sharding_axis_dims=dims,
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
    layer = model.model.layers[0].self_attn
    # Non-trivial pooling weights (defaults are ape=0, gate=1).
    rng = np.random.default_rng(1)
    layer.indexer.index_kpool_compress_ape.value = jnp.asarray(rng.standard_normal((KPOOL, 16)), jnp.float32)
    layer.indexer.index_kpool_compress_gate.value = jnp.asarray(0.2 * rng.standard_normal((16, HID)), jnp.float32)
    return config, layer


def _forward(layer, hidden, *, view, meta):
    """One eSurge-style serving call of the layer (eSurge runs every ragged step in decode mode)."""
    positions = jnp.zeros(hidden.shape[:2], jnp.int32)
    return layer.forward_mla(hidden, None, positions, common_types.MODE_DECODE, cache_view=view, cache_metadata=meta)


def _attention_output(layer, hidden, *, view, meta):
    return _forward(layer, hidden, view=view, meta=meta).attention_output


def _run(config, layer, ep_split: bool | None, trace_only: bool = False):
    """Run ``STEPS`` from an empty cache; returns per-step outputs, final cache arrays, first-step primitive counts.

    ``trace_only`` returns only the first step's primitive counts, without running a step.
    """
    layer.config.dsa_serving_ep_split = ep_split
    rng = np.random.default_rng(0)
    lengths = [sum(step[row] for step in STEPS) for row in range(2)]
    streams = [jnp.asarray(rng.standard_normal((n, HID)), jnp.float32) for n in lengths]
    max_seqs, pages_per_seq = 3, -(-max(lengths) // PAGE)
    cache_cfg = MLARaggedPagesCacheConfig.create(
        mesh=config.mesh,
        runtime_sharding_resolver=config.runtime_sharding_resolver,
        kvdtype=jnp.float32,
        num_hidden_layers=1,
        num_kv_heads=HEADS,
        max_model_length=pages_per_seq * PAGE,
        kv_lora_rank=config.kv_lora_rank,
        qk_rope_head_dim=config.mla_cache_rope_width,
        page_size=PAGE,
        max_cache_tokens=(max_seqs * pages_per_seq + 3) * PAGE,
        indexer_key_dim=config.index_head_dim,
        indexer_gate_dim=config.index_head_dim,
        indexer_kpool=KPOOL,
    )
    tables = jnp.asarray(
        rng.permutation(cache_cfg.num_pages)[: max_seqs * pages_per_seq].reshape(max_seqs, pages_per_seq), jnp.int32
    )
    outputs, counts = [], None
    with jax.default_matmul_precision("highest"), config.mesh:
        view = MLARaggedPagesCacheView.init(
            cache_cfg, layer_index=0, mesh=config.mesh, runtime_sharding_resolver=config.runtime_sharding_resolver
        )
        done = [0, 0]
        for index, step in enumerate(STEPS):
            chunks = []
            for row, n in enumerate(step):
                chunks.append(streams[row][done[row] : done[row] + n])
                done[row] += n
            chunks.append(jnp.asarray(rng.standard_normal((PAD_TOKENS, HID)), jnp.float32))
            packed = jnp.concatenate(chunks)[None]
            meta = RaggedPagesMetadata(
                pages_tables=tables,
                context_lens=jnp.asarray([*done, 0], jnp.int32),
                query_start_loc=jnp.asarray(np.concatenate([[0], np.cumsum([*step, 0])]), jnp.int32),
                num_seqs=jnp.asarray([sum(1 for n in step if n > 0)], jnp.int32),
                page_size=PAGE,
                version="v3",
            )
            if index == 0:
                # Trace on a shallow copy: the op stores its updated pages on the view object it is handed.
                traced = view.replace(kv_pages=view.kv_pages)
                counts = _primitive_counts(functools.partial(_attention_output, layer, view=traced, meta=meta), packed)
                if trace_only:
                    return counts
            out = _forward(layer, packed, view=view, meta=meta)
            view = out.cache_view
            outputs.append(np.asarray(out.attention_output[0]))
        cache = [np.asarray(view.kv_pages), np.asarray(view.indexer_token_pages), np.asarray(view.indexer_pool_pages)]
    return outputs, cache, counts


@pytest.mark.parametrize(
    ("dims", "compact_min_window"),
    [
        ((1, 1, 1, 4, 1, 1), None),
        ((1, 1, 1, 2, 2, 1), None),
        ((1, 1, 1, 1, 4, 1), None),
        ((1, 2, 1, 2, 1, 1), None),
        ((1, 1, 1, 4, 1, 1), 1),
        ((1, 1, 1, 2, 2, 1), 1),
    ],
    ids=["ep4tp1", "ep2tp2", "ep1tp4", "dp2ep2", "ep4tp1-compact", "ep2tp2-compact"],
)
def test_ep_split_serving_matches_unsplit(dims, compact_min_window, monkeypatch):
    if jax.device_count() < int(np.prod(dims)):
        pytest.skip(f"needs {int(np.prod(dims))} devices")
    if compact_min_window is not None:
        # These steps' per-shard windows are shorter than the production threshold, so by default they take the
        # single-call path (as decode does); lowering it runs every window as its own compact kernel call (the path
        # of production prefill steps).
        monkeypatch.setattr(mlrpa, "_EP_SPLIT_COMPACT_MIN_WINDOW", compact_min_window)
    config, layer = _layer(dims)
    # ``None``, the automatic rule, must build the split program exactly where ``True`` splits and the unsplit one
    # elsewhere, so the outputs and caches compared below are its own. Traced first, from the config as built
    # without the setting (``None`` with the merged base_config.py; one that still coerces the unset setting to
    # ``False`` gets ``None`` assigned).
    auto_counts = _run(config, layer, getattr(config, "dsa_serving_ep_split", None) or None, trace_only=True)
    off, off_cache, off_counts = _run(config, layer, False)
    on, on_cache, on_counts = _run(config, layer, True)
    ep, dp = dims[3], dims[1]
    if ep > 1 and dp == 1:
        # Guard against a vacuous comparison: the flag-on program indexes its expert-parallel shard.
        assert on_counts["axis_index"] > off_counts["axis_index"], (on_counts["axis_index"], off_counts["axis_index"])
        assert auto_counts == on_counts
    else:
        # ep == 1, or KV pages sharded over data parallelism: the flag must leave the program unchanged.
        assert on_counts == off_counts
        assert auto_counts == off_counts
    diffs = [float(np.max(np.abs(a - b))) for a, b in zip(on, off, strict=True)]
    bitwise = all(np.array_equal(a, b) for a, b in zip(on, off, strict=True))
    print(f"{dims}: per-step max|on - off| = {diffs}, bitwise={bitwise}")
    for step, (a, b) in enumerate(zip(on, off, strict=True)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5, err_msg=f"step {step}")
    for a, b in zip(on_cache, off_cache, strict=True):
        np.testing.assert_array_equal(a, b)
