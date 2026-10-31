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

"""``dsa_pool_training`` must equal the dense GLM-5-Next DSA path.

The pool path selects pools query-chunk by query-chunk (``SparseIndexer.select_pools_chunked``) and attends in the
weight-absorbed MLA form through ``dsa_pool_sparse_attention`` (``dsa_pool_kernel="bitmap"``) or
``dsa_gather_attention`` (``"gather"``), never building a ``[S, S]`` tensor. The reference is the dense top-k-mask
path (``dsa_sparse_training=False``). Contexts exceed ``index_topk`` so the selection is genuinely sparse; selections,
outputs and input gradients are compared for unpadded, right-padded, left-padded and hole-padded rows. Packed
documents (``MaskInfo.padding_only`` False) keep the dense path, and the bitmap kernel's tail-key limit is checked at
config time.
"""

from __future__ import annotations

import re

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.indexer import compact_valid_layout
from easydel.modules.glm5_next import modeling_glm5_next
from ejkernel.types import MaskInfo
from spectrax import common_types

HID, TOPK, KPOOL, SEQ, LATENT = 32, 8, 4, 96, 128


def _build(pool: bool, query_chunk: int = 32, kernel: str = "bitmap"):
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
        kv_lora_rank=LATENT,
        q_lora_rank=16,
        qk_rope_head_dim=0,
        qk_nope_head_dim=16,
        v_head_dim=16,
        index_topk=TOPK,
        index_head_dim=8,
        index_n_heads=2,
        index_kpool=KPOOL,
        layer_types=["deepseek_sparse_attention"],
        mlp_layer_types=["dense"],
        hc_mult=2,
        dsa_pool_training=pool,
        dsa_pool_query_chunk=query_chunk,
        dsa_pool_kernel=kernel,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    config.attn_dtype = jnp.float32
    config.dsa_sparse_training = False
    with config.mesh:
        model = ed.Glm5NextForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
    layer = model.model.layers[0].self_attn
    rng = np.random.default_rng(1)
    layer.indexer.index_kpool_compress_ape.value = jnp.asarray(rng.standard_normal((KPOOL, 8)), jnp.float32)
    layer.indexer.index_kpool_compress_gate.value = jnp.asarray(0.2 * rng.standard_normal((8, HID)), jnp.float32)
    return config, model, layer


PATTERNS = ["none", "right", "left", "holes"]


def _inputs(pattern: str):
    rng = np.random.default_rng(0)
    h = jnp.asarray(rng.standard_normal((2, SEQ, HID)), jnp.float32)
    valid = np.ones((2, SEQ), bool)
    if pattern == "right":
        valid[1, SEQ - 13 :] = False
    elif pattern == "left":
        valid[1, :13] = False
    elif pattern == "holes":
        valid[0, :5] = False
        valid[1] = np.random.default_rng(2).random(SEQ) < 0.7
    return h, valid


def _forward(layer, h, valid):
    batch, n, _ = h.shape
    mask_info = None
    if valid is not None:
        mask_info = MaskInfo.dynamic_init(
            input_ids=jnp.ones((batch, n), jnp.int32), attention_mask=jnp.asarray(valid, jnp.int32)
        )
    return layer.forward_mla(
        h, mask_info, jnp.broadcast_to(jnp.arange(n)[None, :], (batch, n)), common_types.MODE_TRAIN
    ).attention_output


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("query_chunk", [32, 40, SEQ])
def test_chunked_pool_selection_matches_dense_token_mask(pattern, query_chunk):
    config, _model, layer = _build(pool=True, query_chunk=query_chunk)
    h, valid = _inputs(pattern)
    with jax.default_matmul_precision("highest"), config.mesh:
        q_resid = layer.mla_q_a_layernorm(layer.mla_q_a_proj(h))
        dense = layer.indexer(hidden_states=h, q_resid=q_resid, attention_mask=jnp.asarray(valid))
        pool_mask, tails = layer.indexer.select_pools_chunked(
            h, q_resid, jnp.asarray(valid), query_chunk_size=query_chunk
        )
    want = np.asarray(dense.topk_mask)
    # The selection indexes each row's compact (valid-first) key grid; map it back to physical keys.
    compact = np.repeat(np.asarray(pool_mask), KPOOL, axis=-1)[:, :, :SEQ]
    tails = np.asarray(tails)
    for b in range(compact.shape[0]):
        for s in range(SEQ):
            for t in tails[b, s]:
                if t >= 0:
                    compact[b, s, t] = True
    order = np.asarray(compact_valid_layout(jnp.asarray(valid)).order)
    got = np.zeros_like(compact)
    for b in range(compact.shape[0]):
        got[b][:, order[b]] = compact[b]
    rows = valid[..., None]
    np.testing.assert_array_equal(got & rows, want & rows)
    assert not got[~valid].any(), "padded queries must select nothing"
    # Sparse at this context: later queries see fewer keys than are causally visible.
    assert got[0, -1].sum() < SEQ


@pytest.mark.parametrize("query_chunk", [32, 40])
def test_packed_pool_selection_is_the_bool_selection_bitpacked(query_chunk):
    """``packed=True`` packs inside each chunk: bit ``p % 32`` of word ``p // 32`` is pool ``p``, no other bit set."""
    config, _model, layer = _build(pool=True, query_chunk=query_chunk)
    h, valid = _inputs("right")
    with jax.default_matmul_precision("highest"), config.mesh:
        q_resid = layer.mla_q_a_layernorm(layer.mla_q_a_proj(h))
        pool_mask, tails = layer.indexer.select_pools_chunked(
            h, q_resid, jnp.asarray(valid), query_chunk_size=query_chunk
        )
        words, packed_tails = layer.indexer.select_pools_chunked(
            h, q_resid, jnp.asarray(valid), query_chunk_size=query_chunk, packed=True
        )
    pool_mask = np.asarray(pool_mask)
    num_pools = pool_mask.shape[-1]
    assert words.dtype == jnp.int32 and words.shape == (2, SEQ, -(-num_pools // 32))
    bits = np.asarray(words).view(np.uint32)
    unpacked = np.stack([(bits[..., p // 32] >> (p % 32)) & 1 for p in range(32 * bits.shape[-1])], axis=-1)
    np.testing.assert_array_equal(unpacked[..., :num_pools].astype(bool), pool_mask)
    assert not unpacked[..., num_pools:].any(), "bits past the last pool must be 0"
    np.testing.assert_array_equal(np.asarray(packed_tails), np.asarray(tails))


@pytest.mark.parametrize("query_chunk", [32, 40])
def test_index_pool_selection_is_the_bool_selection(query_chunk):
    """``as_indices=True`` returns exactly the selected pools of the bool mask as ids (``-1`` = unused slot)."""
    config, _model, layer = _build(pool=True, query_chunk=query_chunk)
    h, valid = _inputs("holes")
    with jax.default_matmul_precision("highest"), config.mesh:
        q_resid = layer.mla_q_a_layernorm(layer.mla_q_a_proj(h))
        pool_mask, tails = layer.indexer.select_pools_chunked(
            h, q_resid, jnp.asarray(valid), query_chunk_size=query_chunk
        )
        ids, id_tails = layer.indexer.select_pools_chunked(
            h, q_resid, jnp.asarray(valid), query_chunk_size=query_chunk, as_indices=True
        )
    pool_mask, ids = np.asarray(pool_mask), np.asarray(ids)
    assert ids.dtype == np.int32 and ids.shape == (2, SEQ, TOPK // KPOOL)
    rebuilt = np.zeros_like(pool_mask)
    for b in range(ids.shape[0]):
        for s in range(SEQ):
            picked = ids[b, s][ids[b, s] >= 0]
            assert len(set(picked.tolist())) == len(picked), "pool ids must be distinct"
            rebuilt[b, s, picked] = True
    np.testing.assert_array_equal(rebuilt, pool_mask)
    np.testing.assert_array_equal(np.asarray(id_tails), np.asarray(tails))
    with pytest.raises(ValueError, match="mutually exclusive"):
        layer.indexer.select_pools_chunked(h, q_resid, None, packed=True, as_indices=True)


def test_gather_kernel_requires_even_kpool():
    with pytest.raises(ValueError, match="even index_kpool"):
        ed.Glm5NextTextConfig(
            hidden_size=HID, num_attention_heads=2, num_key_value_heads=2, q_lora_rank=16, qk_rope_head_dim=0,
            index_topk=9, index_kpool=3, dsa_pool_training=True, dsa_pool_kernel="gather",
        )  # fmt: skip
    with pytest.raises(ValueError, match="dsa_pool_kernel"):
        ed.Glm5NextTextConfig(
            hidden_size=HID, num_attention_heads=2, num_key_value_heads=2, q_lora_rank=16, qk_rope_head_dim=0,
            index_topk=8, index_kpool=4, dsa_pool_kernel="dense",
        )  # fmt: skip


def test_bitmap_kernel_rejects_more_tail_keys_than_it_carries():
    """kpool=16 with the always-selected tail gives 15 tail keys; the bitmap kernel carries 8, gather has no limit."""
    common = dict(
        hidden_size=HID, num_attention_heads=2, num_key_value_heads=2, q_lora_rank=16, qk_rope_head_dim=0,
        index_topk=32, index_kpool=16, dsa_pool_training=True,
    )  # fmt: skip
    with pytest.raises(ValueError, match="tail keys"):
        ed.Glm5NextTextConfig(dsa_pool_kernel="bitmap", **common)
    assert ed.Glm5NextTextConfig(dsa_pool_kernel="gather", **common).dsa_pool_bitmap_tail_error() is not None
    assert ed.Glm5NextTextConfig(dsa_pool_kernel="bitmap", **{**common, "index_kpool": 8}).index_kpool == 8
    no_tail = ed.Glm5NextTextConfig(dsa_pool_kernel="bitmap", index_kpool_always_select_tail=False, **common)
    assert no_tail.dsa_pool_bitmap_tail_error() is None


def test_packed_documents_keep_the_dense_path(monkeypatch):
    """A packed-document mask (``padding_only`` False) is not taken by the pool path: same output as the dense layer."""
    calls = {"n": 0}
    real = modeling_glm5_next.Glm5NextDSAAttention._forward_dsa_pool_training

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return real(self, *args, **kwargs)

    monkeypatch.setattr(modeling_glm5_next.Glm5NextDSAAttention, "_forward_dsa_pool_training", counting)
    _ref_config, _ref_model, ref_layer = _build(pool=False)
    config, _model, layer = _build(pool=True)
    h, _ = _inputs("none")
    segments = np.zeros((2, SEQ), np.int32)
    segments[:, SEQ // 2 :] = 1
    mask_info = MaskInfo.from_segments(jnp.asarray(segments))
    assert not mask_info.padding_only
    positions = jnp.broadcast_to(jnp.arange(SEQ)[None, :], (2, SEQ))
    with jax.default_matmul_precision("highest"), config.mesh:
        out = layer.forward_mla(h, mask_info, positions, common_types.MODE_TRAIN).attention_output
        ref = ref_layer.forward_mla(h, mask_info, positions, common_types.MODE_TRAIN).attention_output
    assert calls["n"] == 0, "the pool path must not run on packed documents"
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kernel", ["bitmap", "gather"])
@pytest.mark.parametrize("pattern", PATTERNS)
def test_pool_training_matches_dense_forward_and_input_grad(pattern, kernel, monkeypatch):
    calls = {"n": 0, "gather": 0}
    real = modeling_glm5_next.Glm5NextDSAAttention._forward_dsa_pool_training
    real_gather = modeling_glm5_next.Glm5NextDSAAttention._forward_dsa_pool_gather

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return real(self, *args, **kwargs)

    def counting_gather(self, *args, **kwargs):
        calls["gather"] += 1
        return real_gather(self, *args, **kwargs)

    monkeypatch.setattr(modeling_glm5_next.Glm5NextDSAAttention, "_forward_dsa_pool_training", counting)
    monkeypatch.setattr(modeling_glm5_next.Glm5NextDSAAttention, "_forward_dsa_pool_gather", counting_gather)
    _ref_config, _ref_model, ref_layer = _build(pool=False)
    config, _model, layer = _build(pool=True, kernel=kernel)
    np.testing.assert_array_equal(
        np.asarray(ref_layer.mla_kv_b_proj.weight.value), np.asarray(layer.mla_kv_b_proj.weight.value)
    )
    h, valid = _inputs(pattern)
    rng = np.random.default_rng(3)
    ct = jnp.asarray(rng.standard_normal((2, SEQ, HID)), jnp.float32) * jnp.asarray(valid, jnp.float32)[..., None]

    with jax.default_matmul_precision("highest"), config.mesh:
        ref_out = _forward(ref_layer, h, valid)
        assert calls["n"] == 0
        out = _forward(layer, h, valid)
        assert calls["n"] == 1, "dsa_pool_training branch was not taken"
        assert calls["gather"] == (1 if kernel == "gather" else 0), "dsa_pool_kernel selected the wrong kernel"
        ref_grad = jax.grad(lambda x: jnp.sum(_forward(ref_layer, x, valid) * ct))(h)
        grad = jax.grad(lambda x: jnp.sum(_forward(layer, x, valid) * ct))(h)

    v = valid[..., None]
    assert np.isfinite(np.asarray(out)).all() and np.isfinite(np.asarray(grad)).all()
    assert _rel(np.asarray(out) * v, np.asarray(ref_out) * v) < 1e-3
    assert _rel(np.asarray(grad) * v, np.asarray(ref_grad) * v) < 1e-3


@pytest.mark.parametrize("kernel", ["bitmap", "gather"])
def test_pool_training_lowers_without_seq_by_seq_tensors(kernel):
    """The pool path's lowered fwd+bwd holds no tensor with two sequence-length dimensions."""
    seq = 2048
    config, _model, layer = _build(pool=True, query_chunk=512, kernel=kernel)
    h = jnp.zeros((1, seq, HID), jnp.float32)
    if jax.default_backend() != "tpu":
        pytest.skip("off TPU the dsa_sparse_attention XLA reference is dense by design")
    with config.mesh:
        # mask_info=None: the layer derives an unpadded causal mask itself; a caller-built dense
        # MaskInfo would put an [S, S] tensor in the graph that is not the layer's.
        text = jax.jit(jax.grad(lambda x: jnp.sum(_forward(layer, x, None)))).lower(h).as_text()
    shapes = set(re.findall(r"tensor<([0-9x]+)x(?:f32|bf16|i32|i1|i8|ui32)>", text))
    quadratic = [s for s in shapes if sum(1 for d in s.split("x") if int(d) >= seq) >= 2]
    assert not quadratic, f"[seq, seq]-sized tensors in the pool path: {sorted(quadratic)[:5]}"
    # The selection travels as packed words or pool ids: no bool tensor as large as the full [seq, pools] mask exists.
    masks = re.findall(r"tensor<([0-9x]+)xi1>", text)
    biggest = max((int(np.prod([int(d) for d in s.split("x")])) for s in masks), default=0)
    assert biggest < seq * (seq // KPOOL), f"a full-size [seq, pools] bool mask is still built ({biggest} elements)"
