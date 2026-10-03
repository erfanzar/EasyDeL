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

"""Sparse DSA training attention must equal the dense top-k-mask path.

With no KV cache, glm5_next and glm_moe_dsa route DSA layers through
``sparse_selected_training_attention`` (tile-skipping flash over the top-k mask).
The dense path (``config.dsa_sparse_training = False``) masks full attention with a
``[B, S, S]`` top-k mask and is the reference. Contexts exceed ``index_topk`` so the
selection is genuinely sparse; outputs and input gradients are compared.
"""

from __future__ import annotations

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.glm5_next import modeling_glm5_next
from easydel.modules.glm_moe_dsa import modeling_glm_moe_dsa
from ejkernel.types import MaskInfo
from spectrax import common_types

HID, TOPK, KPOOL, SEQ = 32, 8, 4, 40


def _glm5_next(sparse: bool):
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
        index_topk=TOPK,
        index_head_dim=8,
        index_n_heads=2,
        index_kpool=KPOOL,
        layer_types=["deepseek_sparse_attention"],
        mlp_layer_types=["dense"],
        hc_mult=2,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    config.attn_dtype = jnp.float32
    config.dsa_sparse_training = sparse
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


def _glm_moe_dsa(sparse: bool):
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
        index_topk=TOPK,
        index_head_dim=16,
        index_n_heads=4,
        mlp_layer_types=["dense"],
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    config.attn_dtype = jnp.float32
    config.dsa_sparse_training = sparse
    with config.mesh:
        model = ed.GlmMoeDsaForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
    return config, model, model.model.layers[0].self_attn


MODELS = {
    "glm5_next": (_glm5_next, modeling_glm5_next),
    "glm_moe_dsa": (_glm_moe_dsa, modeling_glm_moe_dsa),
}


def _forward(layer, model, h, attention_mask):
    batch, n, _ = h.shape
    mask_info = MaskInfo.dynamic_init(input_ids=jnp.ones((batch, n), jnp.int32), attention_mask=attention_mask)
    return layer.forward_mla(
        h,
        mask_info,
        jnp.broadcast_to(jnp.arange(n)[None, :], (batch, n)),
        common_types.MODE_TRAIN,
        frequencies=getattr(model.model, "frequencies", None),
    ).attention_output


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


@pytest.mark.parametrize("name", sorted(MODELS))
@pytest.mark.parametrize("padded", [False, True])
def test_sparse_training_matches_dense_topk_mask(name, padded, monkeypatch):
    build, module = MODELS[name]
    calls = {"n": 0}
    real = module.sparse_selected_training_attention

    def counting(**kw):
        calls["n"] += 1
        return real(**kw)

    monkeypatch.setattr(module, "sparse_selected_training_attention", counting)

    _ref_config, ref_model, ref_layer = build(sparse=False)
    config, model, layer = build(sparse=True)
    np.testing.assert_array_equal(
        np.asarray(ref_layer.mla_kv_b_proj.weight.value), np.asarray(layer.mla_kv_b_proj.weight.value)
    )

    rng = np.random.default_rng(0)
    h = jnp.asarray(rng.standard_normal((2, SEQ, HID)), jnp.float32)
    attention_mask = None
    valid = np.ones((2, SEQ), bool)
    if padded:
        valid[1, SEQ - 7 :] = False  # right padding on the second row
        attention_mask = jnp.asarray(valid, jnp.int32)
    ct = jnp.asarray(rng.standard_normal((2, SEQ, HID)), jnp.float32) * jnp.asarray(valid, jnp.float32)[..., None]

    with jax.default_matmul_precision("highest"), config.mesh:
        ref_out = _forward(ref_layer, ref_model, h, attention_mask)
        assert calls["n"] == 0
        out = _forward(layer, model, h, attention_mask)
        assert calls["n"] == 1, "sparse training branch was not taken"
        ref_grad = jax.grad(lambda x: jnp.sum(_forward(ref_layer, ref_model, x, attention_mask) * ct))(h)
        grad = jax.grad(lambda x: jnp.sum(_forward(layer, model, x, attention_mask) * ct))(h)

    v = valid[..., None]
    assert _rel(np.asarray(out) * v, np.asarray(ref_out) * v) < 1e-4
    assert _rel(np.asarray(grad) * v, np.asarray(ref_grad) * v) < 1e-4


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="auto-enable is TPU-only")
def test_auto_default_takes_sparse_path_at_threshold_and_matches_dense(monkeypatch):
    """``dsa_sparse_training`` unset: seq >= AUTO_SPARSE_MIN_SEQ picks the sparse op and equals dense."""
    from easydel.operations.kernels.dsa_sparse_attention import AUTO_SPARSE_MIN_SEQ

    calls = {"n": 0}
    real = modeling_glm5_next.sparse_selected_training_attention

    def counting(**kw):
        calls["n"] += 1
        return real(**kw)

    monkeypatch.setattr(modeling_glm5_next, "sparse_selected_training_attention", counting)
    _c, ref_model, ref_layer = _glm5_next(sparse=False)
    config, model, layer = _glm5_next(sparse=None)
    h = jnp.asarray(np.random.default_rng(0).standard_normal((1, AUTO_SPARSE_MIN_SEQ, HID)), jnp.float32)
    with jax.default_matmul_precision("highest"), config.mesh:
        ref_out = _forward(ref_layer, ref_model, h, None)
        assert calls["n"] == 0
        out = _forward(layer, model, h, None)
        assert calls["n"] == 1, "auto default did not select the sparse path at the threshold"
    assert _rel(out, ref_out) < 1e-3


@pytest.mark.parametrize("name", sorted(MODELS))
def test_selection_is_sparse_at_this_context(name):
    """Negative control: full attention differs from the top-k result past index_topk."""
    build, _ = MODELS[name]
    _, model, layer = build(sparse=True)
    h = jnp.asarray(np.random.default_rng(0).standard_normal((1, SEQ, HID)), jnp.float32)
    with jax.default_matmul_precision("highest"), model.config.mesh:
        sparse_out = np.asarray(_forward(layer, model, h, None))
        layer.config.dsa_sparse_training = False
        old_topk = layer.indexer.index_topk
        layer.indexer.index_topk = 4 * SEQ
        try:
            full_out = np.asarray(_forward(layer, model, h, None))
        finally:
            layer.indexer.index_topk = old_topk
            layer.config.dsa_sparse_training = True
    assert _rel(sparse_out[:, -8:], full_out[:, -8:]) > 1e-2
