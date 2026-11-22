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

"""GLM-4V-MoE router-logit collection and aux loss (computation test: run on TPU/GPU)."""

import easydel as ed
import numpy as np
import spectrax as spx
from easydel.modules.glm4v_moe.modeling_glm4v_moe import Glm4vMoeTextModel
from ejkernel.types import MaskInfo
from jax import numpy as jnp


def _tiny_config():
    cfg = ed.Glm4vMoeConfig()
    text = cfg.text_config
    text.vocab_size = 64
    text.hidden_size = 64
    text.intermediate_size = 128
    text.moe_intermediate_size = 32
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text.head_dim = 16
    text.num_hidden_layers = 3
    text.first_k_dense_replace = 1
    text.n_routed_experts = 4
    text.num_experts_per_tok = 2
    text.n_group = 1
    text.topk_group = 1
    text.norm_topk_prob = True
    # GLM-4V-MoE rotates half of each 16-channel head: four frequency pairs,
    # split across the temporal/height/width axes (not eight for full-head RoPE).
    text.rope_scaling = {"rope_type": "default", "mrope_section": [2, 1, 1]}
    text.scan_layers = False
    text.router_aux_loss_coef = 0.01
    text.attn_mechanism = "vanilla"
    text.moe_force_xla_gmm = True
    text.max_position_embeddings = 32
    vision = cfg.vision_config
    vision.depth = 1
    vision.hidden_size = 32
    vision.intermediate_size = 64
    vision.num_heads = 4
    vision.out_hidden_size = text.hidden_size
    vision.image_size = 8
    vision.patch_size = 2
    return cfg


def _reference_load_balancing(logits, top_k, mask=None):
    """Independent NumPy Switch-style load-balancing reference for one layer."""
    logits = np.asarray(logits).reshape(-1, logits.shape[-1])
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    selected = np.argsort(-probabilities, axis=-1)[:, :top_k]
    if mask is not None:
        valid = np.asarray(mask).reshape(-1).astype(bool)
        probabilities = probabilities[valid]
        selected = selected[valid]
    assignments = np.eye(logits.shape[-1])[selected]
    return logits.shape[-1] * np.sum(assignments.mean(axis=0) * probabilities.mean(axis=0))


def test_text_model_collects_router_logits_only_from_moe_layers():
    cfg = _tiny_config()
    model = Glm4vMoeTextModel(cfg.text_config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    ids = jnp.arange(8, dtype=jnp.int32)[None, :] % 64
    positions = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32)[None, None, :], (3, 1, 8))
    out = model(input_ids=ids, position_ids=positions, output_router_logits=True)
    # 3 layers, the first is dense: only the 2 sparse layers contribute router logits (no ``None`` entries).
    assert out.router_logits is not None and len(out.router_logits) == 2
    assert out.all_router_losses is not None and len(out.all_router_losses) == 2
    for layer_logits, layer_loss in zip(out.router_logits, out.all_router_losses, strict=True):
        assert layer_logits is not None
        assert np.asarray(layer_logits).shape[-1] == 4
        np.testing.assert_allclose(layer_loss, _reference_load_balancing(layer_logits, 2), rtol=1e-5)


def test_router_aux_loss_is_wired_from_text_config_coefficient():
    cfg = _tiny_config()
    model = ed.Glm4vMoeForConditionalGeneration(cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    ids = jnp.arange(8, dtype=jnp.int32)[None, :] % 64
    mask = jnp.array([[True, True, True, True, True, False, False, False]])
    with_logits = model(input_ids=ids, attention_mask=mask, output_router_logits=True)
    assert with_logits.aux_loss is not None
    expected = cfg.text_config.router_aux_loss_coef * sum(
        _reference_load_balancing(logits, 2, mask) for logits in with_logits.router_logits
    )
    np.testing.assert_allclose(with_logits.aux_loss, expected, rtol=1e-5)
    without_logits = model(input_ids=ids, output_router_logits=False)
    assert without_logits.aux_loss is None


def test_router_aux_loss_respects_mask_info_only_padding():
    cfg = _tiny_config()
    model = Glm4vMoeTextModel(cfg.text_config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    ids = jnp.arange(8, dtype=jnp.int32)[None, :]
    # Nonzero segment IDs are valid too; -1 alone denotes padding.
    segments = jnp.array([[2, 2, 2, 2, 2, -1, -1, -1]], dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32)[None, None, :], (3, 1, 8))
    out = model(
        input_ids=ids,
        mask_info=MaskInfo.from_segments(segments),
        position_ids=positions,
        output_router_logits=True,
    )
    assert len(out.router_logits) == len(out.all_router_losses) == 2
    for logits, loss in zip(out.router_logits, out.all_router_losses, strict=True):
        np.testing.assert_allclose(loss, _reference_load_balancing(logits, 2, segments >= 0), rtol=1e-5)


def test_cached_decode_router_aux_loss_uses_current_query_not_full_context():
    cfg = _tiny_config()
    model = Glm4vMoeTextModel(cfg.text_config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    cache = model.init_cache(batch_size=1, max_length=8)
    prefill = model(
        input_ids=jnp.array([[1, 2, 3]], dtype=jnp.int32),
        position_ids=jnp.broadcast_to(jnp.arange(3)[None, None, :], (3, 1, 3)),
        past_key_values=cache,
        mode="prefill",
    )
    # The cursor is 3, not the end of this fixed-capacity generation mask.
    full_mask = jnp.array([[True, True, True, True, False, False, False, False]])
    out = model(
        input_ids=jnp.array([[4]], dtype=jnp.int32),
        attention_mask=full_mask,
        position_ids=jnp.full((3, 1, 1), 3, dtype=jnp.int32),
        past_key_values=prefill.past_key_values,
        output_router_logits=True,
        mode="decode",
    )
    assert len(out.router_logits) == len(out.all_router_losses) == 2
    for logits, loss in zip(out.router_logits, out.all_router_losses, strict=True):
        assert np.asarray(logits).reshape(-1, 4).shape == (1, 4)
        expected = _reference_load_balancing(logits, 2, np.array([[True]]))
        assert expected > 0
        np.testing.assert_allclose(loss, expected, rtol=1e-5)
