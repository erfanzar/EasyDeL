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

"""Accelerator computation, numerical-reference parity, cache, and gradient tests for Inkling 2 / Inkling 2 Loop.

These tests execute JAX numerics and therefore run only on an accelerator
(TPU/GPU), enforced by ``libs/easydel/tests/conftest.py``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from easydel.layers.quantization.checkpoint._codecs import decode_scaled_elements
from easydel.layers.rotary import (
    vision_grid_pos_embed_interpolation_indices_and_weights,
    vision_grid_position_ids,
)
from easydel.modules.inkling2 import (
    Inkling2Config,
    Inkling2ForCausalLM,
    Inkling2ForConditionalGeneration,
    Inkling2LoopConfig,
    Inkling2LoopForCausalLM,
    Inkling2LoopForConditionalGeneration,
    Inkling2LoopTextConfig,
    Inkling2TextConfig,
    Inkling2VisionConfig,
    Inkling2VisionModel,
)
from easydel.modules.inkling2.modeling_inkling2 import Inkling2L2Clip
from easydel.operations import OperationMetadata, OperationRegistry
from easydel.utils.parameters_transformation import StateDictConverter
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from spectrax import common_types

# ---------------------------------------------------------------------------
# Independent pure-NumPy reference implementation of Inkling 2 & Inkling 2 Loop
# transcribed directly from upstream Hugging Face `modeling_inkling2.py` /
# `modular_inkling2_loop.py` (no production EasyDeL math reused).
# ---------------------------------------------------------------------------


def _np_sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _np_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x64 = x.astype(np.float64)
    m = np.max(x64, axis=axis, keepdims=True)
    e = np.exp(x64 - m)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def _np_rmsnorm(x: np.ndarray, weight: np.ndarray | None, eps: float = 1e-6) -> np.ndarray:
    xf = x.astype(np.float32)
    rms = xf / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)
    if weight is not None:
        rms = rms * weight.astype(np.float32)
    return rms.astype(np.float32)


def _np_layernorm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    xf = x.astype(np.float32)
    mean = np.mean(xf, axis=-1, keepdims=True)
    var = np.mean((xf - mean) ** 2, axis=-1, keepdims=True)
    return ((xf - mean) / np.sqrt(var + eps) * weight.astype(np.float32) + bias.astype(np.float32)).astype(np.float32)


def _np_gelu(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    # Exact erf GELU matching JAX/PyTorch default gelu
    from math import erf, sqrt

    vec_erf = np.vectorize(erf)
    return (0.5 * x64 * (1.0 + vec_erf(x64 / sqrt(2.0)))).astype(np.float32)


def _np_l2_clip(x: np.ndarray, threshold: float) -> np.ndarray:
    xf = x.astype(np.float32)
    norm = np.linalg.norm(xf, axis=-1, keepdims=True)
    scale = np.minimum(threshold / (norm + 1e-6), 1.0)
    return (xf * scale).astype(np.float32)


def _np_log_scale(x: np.ndarray, pos_ids: np.ndarray, n_floor: int, alpha: float) -> np.ndarray:
    ratio = np.maximum((pos_ids.astype(np.float32) + 1.0) / float(n_floor), 1.0)
    tau = 1.0 + alpha * np.log(ratio)
    return (x.astype(np.float32) * tau[..., None, None]).astype(np.float32)


def _np_static_short_conv(x: np.ndarray, conv_weight_k1c: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    xf = x.astype(np.float32)
    residual = xf
    if mask is not None:
        xf = xf * mask[:, :, None].astype(np.float32)
    k = conv_weight_k1c.shape[0]
    w_kc = conv_weight_k1c[:, 0, :].astype(np.float32)  # (K, C), index K-1 = current token
    padded = np.pad(xf, ((0, 0), (k - 1, 0), (0, 0)))
    seq_len = xf.shape[1]
    out = np.zeros_like(xf)
    for t in range(seq_len):
        window = padded[:, t : t + k, :]  # (B, K, C)
        out[:, t, :] = np.sum(window * w_kc[None, :, :], axis=1)
    return ((out + residual) / math.sqrt(k + 1)).astype(np.float32)


def _np_dynamic_short_conv(
    x: np.ndarray,
    dyn_in: np.ndarray,
    static_w_kc: np.ndarray,
    weight_proj_io: np.ndarray,
    num_heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    bsz, seq_len, hidden = x.shape
    k = static_w_kc.shape[0]
    head_dim = hidden // num_heads
    dyn_w = (dyn_in.astype(np.float32) @ weight_proj_io.astype(np.float32)).reshape(bsz, seq_len, num_heads, 1, k)
    st_w = static_w_kc.astype(np.float32).T.reshape(1, 1, num_heads, head_dim, k)
    weights = st_w + dyn_w

    xf = x.astype(np.float32)
    residual = xf
    if mask is not None:
        xf = xf * mask[:, :, None].astype(np.float32)
    padded = np.pad(xf, ((0, 0), (k - 1, 0), (0, 0)))
    windows = np.stack([padded[:, (k - 1 - lag) : (k - 1 - lag + seq_len), :] for lag in range(k)], axis=-1)
    windows = windows.reshape(bsz, seq_len, num_heads, head_dim, k)
    out = np.sum(windows * weights, axis=-1).reshape(bsz, seq_len, hidden)
    return ((out + residual) / math.sqrt(2 * k + 1)).astype(np.float32)


def _np_block_attn_res(block_states: np.ndarray, partial: np.ndarray, query: np.ndarray) -> list[np.ndarray]:
    stacked = np.concatenate([block_states, partial[..., None, :]], axis=-2).astype(np.float32)
    stacked = _np_rmsnorm(stacked, None, eps=1e-6)
    h = stacked.shape[-1]
    q = query.astype(np.float32).reshape(-1, h)
    scores = np.einsum("qh,bsnh->bsqn", q, stacked) * (h**-0.5)
    weights = _np_softmax(scores, axis=-1)
    out = np.einsum("bsqn,bsnh->bsqh", weights, stacked)
    return [out[:, :, i, :] for i in range(q.shape[0])]


def _np_message_gate(h: np.ndarray, msg: np.ndarray, gate_mod, eps: float) -> np.ndarray:
    hf, mf = h.astype(np.float32), msg.astype(np.float32)
    cat = np.concatenate([_np_rmsnorm(hf, None, eps), _np_rmsnorm(mf, None, eps)], axis=-1)
    bottleneck = cat @ np.asarray(gate_mod.down_proj.weight.value, dtype=np.float32)
    forget = np.tanh(bottleneck @ np.asarray(gate_mod.forget_gate_proj.weight.value, dtype=np.float32))
    gain = np.tanh(bottleneck @ np.asarray(gate_mod.message_gate_proj.weight.value, dtype=np.float32))
    return ((1.0 - forget) * hf + gain * mf).astype(np.float32)


def _np_moe_block(x: np.ndarray, moe_mod, cfg: Inkling2TextConfig) -> np.ndarray:
    bsz, seq_len, hidden = x.shape
    flat = x.reshape(-1, hidden).astype(np.float32)
    logits = flat @ np.asarray(moe_mod.gate.weight.value, dtype=np.float32)
    scores = _np_sigmoid(logits) if cfg.scoring_func == "sigmoid" else _np_softmax(logits, axis=-1)
    bias = np.asarray(moe_mod.gate.e_score_correction_bias.value, dtype=np.float32)
    sel_scores = scores + bias[None, :]
    topk_idx = np.argsort(-sel_scores, axis=-1, kind="stable")[:, : cfg.num_experts_per_tok]
    topk_w = np.take_along_axis(scores, topk_idx, axis=-1)
    topk_w = topk_w / (np.sum(topk_w, axis=-1, keepdims=True) + 1e-20) * cfg.route_scale

    w_gu = np.asarray(moe_mod.experts.gate_up_proj.weight.value, dtype=np.float32)  # [E, H, 2I] (tp=1 contiguous)
    w_down = np.asarray(moe_mod.experts.down_proj.weight.value, dtype=np.float32)  # [E, I, H]
    inter = cfg.moe_intermediate_size
    out = np.zeros_like(flat)
    for tok in range(flat.shape[0]):
        for slot in range(cfg.num_experts_per_tok):
            e = int(topk_idx[tok, slot])
            gu = flat[tok] @ w_gu[e]
            g, u = gu[:inter], gu[inter:]
            act = (
                np.tanh(g / cfg.mlp_gate_softcap)
                * cfg.mlp_gate_softcap
                * _np_sigmoid(g)
                * np.tanh(u / cfg.mlp_up_softcap)
                * cfg.mlp_up_softcap
            )
            out[tok] += (act @ w_down[e]) * topk_w[tok, slot]
    return out.reshape(bsz, seq_len, hidden)


def _np_attention(
    q_h: np.ndarray,
    k_h: np.ndarray,
    v_h: np.ndarray,
    attn_mod,
    pos_ids: np.ndarray,
    cfg: Inkling2TextConfig,
    *,
    is_full: bool,
) -> np.ndarray:
    bsz, seq_len, _ = q_h.shape
    nh, nkv, hd = attn_mod.num_heads, attn_mod.num_key_value_heads, attn_mod.head_dim

    q = (q_h @ np.asarray(attn_mod.q_proj.linear.weight.value, dtype=np.float32)).reshape(bsz, seq_len, nh, hd)
    q = _np_l2_clip(q * np.asarray(attn_mod.q_proj.norm.scale.value, dtype=np.float32), cfg.qk_l2_clip_threshold)
    if is_full:
        q = _np_log_scale(q, pos_ids, cfg.log_scaling_n_floor, cfg.log_scaling_alpha)

    k_lin = k_h @ np.asarray(attn_mod.k_proj.linear.weight.value, dtype=np.float32)
    v_lin = v_h @ np.asarray(attn_mod.v_proj.linear.weight.value, dtype=np.float32)
    if is_full:
        k_c = _np_dynamic_short_conv(
            k_lin,
            k_h,
            np.asarray(attn_mod.k_proj.sconv.static_weight.value, dtype=np.float32),
            np.asarray(attn_mod.k_proj.sconv.weight_proj.weight.value, dtype=np.float32),
            nkv,
        )
        v_c = _np_dynamic_short_conv(
            v_lin,
            v_h,
            np.asarray(attn_mod.v_proj.sconv.static_weight.value, dtype=np.float32),
            np.asarray(attn_mod.v_proj.sconv.weight_proj.weight.value, dtype=np.float32),
            nkv,
        )
    else:
        k_c = _np_static_short_conv(k_lin, np.asarray(attn_mod.k_proj.sconv.conv1d.weight.value, dtype=np.float32))
        v_c = _np_static_short_conv(v_lin, np.asarray(attn_mod.v_proj.sconv.conv1d.weight.value, dtype=np.float32))

    k = _np_l2_clip(k_c.reshape(bsz, seq_len, nkv, hd), cfg.qk_l2_clip_threshold)
    v = _np_rmsnorm(v_c.reshape(bsz, seq_len, nkv, hd), None, eps=cfg.rms_norm_eps)

    rep = nh // nkv
    k_rep = np.repeat(k, rep, axis=2)
    v_rep = np.repeat(v, rep, axis=2)

    # Relative position bias
    rpb = attn_mod.relative_position_bias
    rel = (q_h @ np.asarray(rpb.linear.weight.value, dtype=np.float32)).reshape(bsz, seq_len, nh, cfg.d_rel)
    rel = _np_l2_clip(rel * np.asarray(rpb.norm.scale.value, dtype=np.float32), cfg.rel_l2_clip_threshold)
    if is_full:
        rel = _np_log_scale(rel, pos_ids, cfg.log_scaling_n_floor, cfg.log_scaling_alpha)
    rel_logits = np.swapaxes(rel @ np.asarray(rpb.proj.value, dtype=np.float32), 1, 2)  # [B, H, Q, E]
    dist = np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :]
    gather_idx = np.broadcast_to(np.clip(dist, 0, rpb.rel_extent - 1)[None, None], (bsz, nh, seq_len, seq_len))
    pos_bias = np.take_along_axis(rel_logits, gather_idx, axis=-1)
    pos_bias = np.where((dist >= 0) & (dist < rpb.rel_extent), pos_bias, 0.0)

    logits = np.einsum("bqhd,bkhd->bhqk", q, k_rep) * (hd**-0.5) + pos_bias
    valid = dist >= 0
    if not is_full:
        valid = valid & (dist < cfg.sliding_window_size)
    logits = np.where(valid[None, None], logits, -np.inf)

    sinks = np.swapaxes(q_h @ np.asarray(attn_mod.sink_proj.weight.value, dtype=np.float32), 1, 2)[..., None]
    weights = _np_softmax(np.concatenate([logits, sinks], axis=-1), axis=-1)[..., :-1]
    ctx = np.einsum("bhqk,bkhd->bqhd", weights, v_rep).reshape(bsz, seq_len, nh * hd)
    gate = _np_sigmoid(q_h @ np.asarray(attn_mod.o_proj.gate_proj.weight.value, dtype=np.float32))
    return ((ctx * gate) @ np.asarray(attn_mod.o_proj.linear.weight.value, dtype=np.float32)).astype(np.float32)


def _np_decoder_layer(
    hidden: np.ndarray,
    block_states: np.ndarray,
    layer,
    pos_ids: np.ndarray,
    cfg: Inkling2TextConfig,
    messages: np.ndarray | None = None,
) -> np.ndarray:
    is_full = not cfg.is_sliding_layer(layer.layer_idx)
    block_residual = hidden
    if is_full:
        if layer.attention_input is None:
            q_h = k_h = v_h = hidden
        else:
            q_h, k_h, v_h = _np_block_attn_res(
                block_states, hidden, np.asarray(layer.attention_input.query.value, dtype=np.float32)
            )
        if messages is not None:
            q_h = _np_message_gate(q_h, messages[:, :, 0, :], layer.q_message_gate, cfg.rms_norm_eps)
            k_h = _np_message_gate(k_h, messages[:, :, 1, :], layer.k_message_gate, cfg.rms_norm_eps)
            v_h = _np_message_gate(v_h, messages[:, :, 2, :], layer.v_message_gate, cfg.rms_norm_eps)
        q_in = _np_rmsnorm(q_h, np.asarray(layer.q_input_layernorm.weight.value), cfg.rms_norm_eps)
        k_in = _np_rmsnorm(k_h, np.asarray(layer.k_input_layernorm.weight.value), cfg.rms_norm_eps)
        v_in = _np_rmsnorm(v_h, np.asarray(layer.v_input_layernorm.weight.value), cfg.rms_norm_eps)
    else:
        h_in = (
            hidden
            if layer.attention_input is None
            else _np_block_attn_res(
                block_states, hidden, np.asarray(layer.attention_input.query.value, dtype=np.float32)
            )[0]
        )
        if messages is None:
            q_in = k_in = v_in = _np_rmsnorm(h_in, np.asarray(layer.input_layernorm.weight.value), cfg.rms_norm_eps)
        else:
            q_in = _np_rmsnorm(
                _np_message_gate(h_in, messages[:, :, 0, :], layer.q_message_gate, cfg.rms_norm_eps),
                np.asarray(layer.input_layernorm.weight.value),
                cfg.rms_norm_eps,
            )
            k_in = _np_rmsnorm(
                _np_message_gate(h_in, messages[:, :, 1, :], layer.k_message_gate, cfg.rms_norm_eps),
                np.asarray(layer.input_layernorm.weight.value),
                cfg.rms_norm_eps,
            )
            v_in = _np_rmsnorm(
                _np_message_gate(h_in, messages[:, :, 2, :], layer.v_message_gate, cfg.rms_norm_eps),
                np.asarray(layer.input_layernorm.weight.value),
                cfg.rms_norm_eps,
            )

    attn_out = _np_attention(q_in, k_in, v_in, layer.self_attn, pos_ids, cfg, is_full=is_full)
    attn_out = _np_static_short_conv(
        attn_out, np.asarray(layer.post_attention_convolution.conv1d.weight.value, dtype=np.float32)
    )
    if layer.is_block_boundary:
        block_states = np.concatenate([block_states, block_residual[..., None, :]], axis=-2)
        residual = hidden = attn_out
    else:
        residual = hidden = block_residual + attn_out

    mlp_in = _np_block_attn_res(block_states, hidden, np.asarray(layer.mlp_input.query.value, dtype=np.float32))[0]
    if messages is not None:
        mlp_in = _np_message_gate(mlp_in, messages[:, :, 3, :], layer.mlp_message_gate, cfg.rms_norm_eps)
    mlp_normed = _np_rmsnorm(mlp_in, np.asarray(layer.post_attention_layernorm.weight.value), cfg.rms_norm_eps)
    mlp_out = _np_moe_block(mlp_normed, layer.mlp, cfg)
    mlp_out = _np_static_short_conv(
        mlp_out, np.asarray(layer.post_mlp_convolution.conv1d.weight.value, dtype=np.float32)
    )
    return (residual + mlp_out).astype(np.float32)


def _np_reference_text_forward(model: Inkling2ForCausalLM, input_ids: np.ndarray) -> np.ndarray:
    text_mod = model.model
    cfg = text_mod.config
    emb = np.asarray(text_mod.embed_tokens.weight.value, dtype=np.float32)[input_ids]
    hidden = _np_rmsnorm(emb, np.asarray(text_mod.embed_tokens.embed_norm.weight.value), cfg.rms_norm_eps)
    bsz, seq_len, hdim = hidden.shape
    pos_ids = np.broadcast_to(np.arange(seq_len, dtype=np.int32)[None, :], (bsz, seq_len))
    blocks = np.zeros((bsz, seq_len, 0, hdim), dtype=np.float32)

    if isinstance(cfg, Inkling2LoopTextConfig):
        ls, le = cfg.loop_layer_index_start, cfg.loop_layer_index_end
        for idx in range(ls):
            new_h = _np_decoder_layer(hidden, blocks, text_mod.layers[idx], pos_ids, cfg)
            if text_mod.layers[idx].is_block_boundary:
                blocks = np.concatenate([blocks, hidden[..., None, :]], axis=-2)
            hidden = new_h
        pre_h, pre_b = hidden, blocks
        n_pre = pre_b.shape[-2]
        for idx in range(ls, le):
            new_h = _np_decoder_layer(hidden, blocks, text_mod.layers[idx], pos_ids, cfg)
            if text_mod.layers[idx].is_block_boundary:
                blocks = np.concatenate([blocks, hidden[..., None, :]], axis=-2)
            hidden = new_h
        for _ in range(1, cfg.loop_num_passes):
            prev_b = blocks[..., n_pre + 1 :, :]
            prev_h = hidden
            hidden, blocks = pre_h, pre_b
            for l_i, idx in enumerate(range(ls, le)):
                q_msg = np.asarray(text_mod.message_input.query.value[l_i], dtype=np.float32)
                msgs = np.stack(_np_block_attn_res(prev_b, prev_h, q_msg), axis=-2)
                new_h = _np_decoder_layer(hidden, blocks, text_mod.layers[idx], pos_ids, cfg, messages=msgs)
                if text_mod.layers[idx].is_block_boundary:
                    blocks = np.concatenate([blocks, hidden[..., None, :]], axis=-2)
                hidden = new_h
        for idx in range(le, cfg.num_hidden_layers):
            new_h = _np_decoder_layer(hidden, blocks, text_mod.layers[idx], pos_ids, cfg)
            if text_mod.layers[idx].is_block_boundary:
                blocks = np.concatenate([blocks, hidden[..., None, :]], axis=-2)
            hidden = new_h
    else:
        for layer in text_mod.layers:
            new_h = _np_decoder_layer(hidden, blocks, layer, pos_ids, cfg)
            if layer.is_block_boundary:
                blocks = np.concatenate([blocks, hidden[..., None, :]], axis=-2)
            hidden = new_h

    hidden = _np_block_attn_res(blocks, hidden, np.asarray(text_mod.final_input.query.value, dtype=np.float32))[0]
    hidden = _np_rmsnorm(hidden, np.asarray(text_mod.norm.weight.value), cfg.rms_norm_eps)
    logits = hidden @ np.asarray(model.lm_head.weight.value, dtype=np.float32)
    return (np.tanh(logits / cfg.final_logit_softcapping) * cfg.final_logit_softcapping).astype(np.float32)


def _np_reference_vision_forward(
    vision_mod: Inkling2VisionModel, pixel_values: np.ndarray, image_grid_thw: np.ndarray
) -> np.ndarray:
    cfg = vision_mod.config
    merge, patch = cfg.spatial_merge_size, cfg.patch_size
    patch_grid = image_grid_thw * np.array([1, merge, merge], dtype=np.int64)
    ppi = patch_grid.prod(axis=-1)
    num_images = patch_grid.shape[0]
    num_patches = pixel_values.shape[0] * (merge**2)
    total_seq = num_patches + num_images
    patch_index = np.arange(num_patches) + np.repeat(np.arange(1, num_images + 1), ppi)

    patches = pixel_values.reshape(-1, merge, patch, merge, patch, cfg.num_channels).transpose(0, 1, 3, 2, 4, 5)
    p_emb = patches.reshape(-1, patch * patch * cfg.num_channels) @ np.asarray(
        vision_mod.embeddings.patch_embedding.weight.value, dtype=np.float32
    )
    interp_idx, interp_w = vision_grid_pos_embed_interpolation_indices_and_weights(
        patch_grid, num_grid_per_side=cfg.posemb_grid_size, spatial_merge_size=merge, align_corners=False
    )
    pos_table = np.asarray(vision_mod.embeddings.position_embedding.weight.value, dtype=np.float32)
    p_emb = p_emb + np.sum(pos_table[interp_idx] * interp_w[:, :, None], axis=1)

    cls_vec = np.asarray(vision_mod.embeddings.class_embedding.value, dtype=np.float32) + np.asarray(
        vision_mod.embeddings.cls_position_embedding.value, dtype=np.float32
    )
    x = np.broadcast_to(cls_vec[None, :], (total_seq, cfg.hidden_size)).copy()
    x[patch_index] = p_emb
    x = _np_layernorm(
        x,
        np.asarray(vision_mod.pre_layernorm.weight.value),
        np.asarray(vision_mod.pre_layernorm.bias.value),
        cfg.layer_norm_eps,
    )

    pos_2d = np.zeros((total_seq, 2), dtype=np.int32)
    pos_2d[patch_index] = vision_grid_position_ids(patch_grid, merge) + 1
    hd = cfg.hidden_size // cfg.num_attention_heads
    inv_freq = 1.0 / (cfg.rope_theta ** (np.arange(0, hd // 2, 2, dtype=np.float32) / (hd // 2)))
    freqs = pos_2d[..., None].astype(np.float32) * inv_freq
    cos = np.repeat(np.concatenate([np.cos(freqs[:, 1]), np.cos(freqs[:, 0])], axis=-1), 2, axis=-1)
    sin = np.repeat(np.concatenate([np.sin(freqs[:, 1]), np.sin(freqs[:, 0])], axis=-1), 2, axis=-1)

    def _rot_pair(t):
        tp_arr = t.reshape(*t.shape[:-1], -1, 2)
        return np.stack((-tp_arr[..., 1], tp_arr[..., 0]), axis=-1).reshape(t.shape)

    seg_ids = np.repeat(np.arange(num_images), ppi + 1)
    same_img = seg_ids[:, None] == seg_ids[None, :]

    for layer in vision_mod.layers:
        h = _np_layernorm(
            x,
            np.asarray(layer.layernorm_before.weight.value),
            np.asarray(layer.layernorm_before.bias.value),
            cfg.layer_norm_eps,
        )
        q = (
            h @ np.asarray(layer.self_attn.q_proj.weight.value, dtype=np.float32)
            + np.asarray(layer.self_attn.q_proj.bias.value, dtype=np.float32)
        ).reshape(total_seq, cfg.num_attention_heads, hd)
        k = (
            h @ np.asarray(layer.self_attn.k_proj.weight.value, dtype=np.float32)
            + np.asarray(layer.self_attn.k_proj.bias.value, dtype=np.float32)
        ).reshape(total_seq, cfg.num_attention_heads, hd)
        v = (
            h @ np.asarray(layer.self_attn.v_proj.weight.value, dtype=np.float32)
            + np.asarray(layer.self_attn.v_proj.bias.value, dtype=np.float32)
        ).reshape(total_seq, cfg.num_attention_heads, hd)
        q = q * cos[:, None, :] + _rot_pair(q) * sin[:, None, :]
        k = k * cos[:, None, :] + _rot_pair(k) * sin[:, None, :]
        scores = np.einsum("qhd,khd->hqk", q, k) * (hd**-0.5)
        scores = np.where(same_img[None], scores, -np.inf)
        w = _np_softmax(scores, axis=-1)
        attn_out = np.einsum("hqk,khd->qhd", w, v).reshape(total_seq, cfg.hidden_size)
        attn_out = attn_out @ np.asarray(layer.self_attn.out_proj.weight.value, dtype=np.float32) + np.asarray(
            layer.self_attn.out_proj.bias.value, dtype=np.float32
        )
        x = x + np.asarray(layer.lambda_1.value, dtype=np.float32) * attn_out

        h2 = _np_layernorm(
            x,
            np.asarray(layer.layernorm_after.weight.value),
            np.asarray(layer.layernorm_after.bias.value),
            cfg.layer_norm_eps,
        )
        mlp = _np_gelu(
            h2 @ np.asarray(layer.mlp.fc1.weight.value, dtype=np.float32)
            + np.asarray(layer.mlp.fc1.bias.value, dtype=np.float32)
        )
        mlp = mlp @ np.asarray(layer.mlp.fc2.weight.value, dtype=np.float32) + np.asarray(
            layer.mlp.fc2.bias.value, dtype=np.float32
        )
        x = x + np.asarray(layer.lambda_2.value, dtype=np.float32) * mlp

    patch_states = x[patch_index].reshape(-1, (merge**2) * cfg.hidden_size)
    m1 = _np_gelu(patch_states @ np.asarray(vision_mod.merger.linear_1.weight.value, dtype=np.float32))
    return (m1 @ np.asarray(vision_mod.merger.linear_2.weight.value, dtype=np.float32)).astype(np.float32)


def _perturb_zero_init_params(model: spx.Module, seed: int = 123) -> None:
    """Populate zero-initialized residual/conv/bias leaves so every mechanism actively participates."""
    rng = np.random.default_rng(seed)
    for path, var in spx.iter_variables(model):
        if not isinstance(var, spx.Parameter):
            continue
        name = ".".join(str(p) for p in path)
        if any(
            tok in name
            for tok in (
                "attention_input.query",
                "mlp_input.query",
                "final_input.query",
                "message_input.query",
                "static_weight",
                "e_score_correction_bias",
                ".scale",
            )
        ):
            val = rng.normal(loc=0.0, scale=0.25, size=var.value.shape)
            if ".scale" in name:
                val = 1.0 + val
            var.value = jnp.asarray(val, dtype=var.value.dtype)


def _tiny_text_config(**overrides) -> Inkling2TextConfig:
    defaults = dict(
        vocab_size=64,
        unpadded_vocab_size=50,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=8,
        sliding_window_size=3,
        d_rel=4,
        rel_extent=4,
        log_scaling_n_floor=2,
        log_scaling_alpha=0.15,
        max_position_embeddings=64,
        conv_kernel_size=3,
        moe_intermediate_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        route_scale=2.0,
        n_layers_per_attn_res_block=2,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        final_logit_softcapping=30.0,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    defaults.update(overrides)
    return Inkling2TextConfig(**defaults)


def _tiny_loop_text_config(**overrides) -> Inkling2LoopTextConfig:
    defaults = dict(
        vocab_size=64,
        unpadded_vocab_size=50,
        hidden_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=8,
        sliding_window_size=3,
        d_rel=4,
        rel_extent=4,
        log_scaling_n_floor=2,
        log_scaling_alpha=0.15,
        max_position_embeddings=64,
        conv_kernel_size=3,
        moe_intermediate_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        route_scale=2.0,
        n_layers_per_attn_res_block=2,
        loop_layer_index_start=2,
        loop_layer_index_end=4,
        loop_num_passes=3,
        loop_gate_rank=8,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        final_logit_softcapping=30.0,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    defaults.update(overrides)
    return Inkling2LoopTextConfig(**defaults)


def _tiny_vision_config(text_hidden_size: int = 32) -> Inkling2VisionConfig:
    return Inkling2VisionConfig(
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_channels=3,
        patch_size=4,
        spatial_merge_size=2,
        posemb_grid_size=4,
        projection_hidden_dim=32,
        projector_bias=False,
        text_hidden_size=text_hidden_size,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )


class TestVanillaAttnPerTokenSinks:
    """Direct numerical parity for 4-D per-token ``softmax_aux`` in ``VanillaAttn``."""

    def test_4d_per_token_sinks_parity(self):
        cfg = _tiny_text_config()
        rng = np.random.default_rng(7)
        bsz, q_len, kv_len, hq, hkv, hd = 2, 5, 5, 4, 2, 8
        q = rng.standard_normal((bsz, q_len, hq, hd)).astype(np.float32)
        k = rng.standard_normal((bsz, kv_len, hkv, hd)).astype(np.float32)
        v = rng.standard_normal((bsz, kv_len, hkv, hd)).astype(np.float32)
        bias = rng.standard_normal((bsz, hq, q_len, kv_len)).astype(np.float32) * 0.2
        sinks = rng.standard_normal((bsz, hq, q_len, 1)).astype(np.float32)

        op = OperationRegistry.create("vanilla", OperationMetadata.from_config(cfg))
        mask_info = MaskInfo.from_attention_mask(jnp.ones((bsz, q_len), dtype=jnp.bool_))
        out = op.forward_native(
            query=jnp.asarray(q),
            key=jnp.asarray(k),
            value=jnp.asarray(v),
            mask_info=mask_info,
            bias=jnp.asarray(bias),
            softmax_aux=jnp.asarray(sinks),
            softmax_scale=hd**-0.5,
            causal=True,
            sliding_window=3,
        )

        # Independent NumPy reference
        k_rep = np.repeat(k, hq // hkv, axis=2)
        v_rep = np.repeat(v, hq // hkv, axis=2)
        logits = np.einsum("bqhd,bkhd->bhqk", q, k_rep) * (hd**-0.5) + bias
        dist = np.arange(q_len)[:, None] - np.arange(kv_len)[None, :]
        logits = np.where(((dist >= 0) & (dist < 3))[None, None], logits, -np.inf)
        weights = _np_softmax(np.concatenate([logits, sinks], axis=-1), axis=-1)[..., :-1]
        ref = np.einsum("bhqk,bkhd->bqhd", weights, v_rep)

        np.testing.assert_allclose(np.asarray(out.attention_outputs), ref, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(out.attention_weights), weights, rtol=1e-4, atol=1e-4)


class TestInkling2NumericalParityAndCaching:
    """Numerical parity against independent references, KV+conv caching, packed isolation, and gradients."""

    def test_flat_causal_lm_reference_parity_and_cache_decode(self):
        cfg = _tiny_text_config()
        model = Inkling2ForCausalLM(cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(11))
        _perturb_zero_init_params(model, seed=11)

        # Reform-param schema must validate cleanly.
        StateDictConverter.validate_reform_param_schema(model._get_reform_param())

        input_ids = np.array([[1, 5, 9, 2, 14, 7, 3]], dtype=np.int32)
        full_out = model(input_ids=jnp.asarray(input_ids), output_router_logits=True)
        assert full_out.logits.shape == (1, 7, cfg.unpadded_vocab_size)
        assert full_out.logits.dtype == jnp.float32
        assert len(full_out.router_logits) == cfg.num_hidden_layers

        ref_logits = _np_reference_text_forward(model, input_ids)
        np.testing.assert_allclose(np.asarray(full_out.logits), ref_logits, rtol=1e-3, atol=1e-3)

        # Cached prefill (4 tokens > sliding_window_size=3) + 3 single-token decode steps
        cache = model.init_cache(batch_size=1, max_length=16)
        prefill_out = model(
            input_ids=jnp.asarray(input_ids[:, :4]),
            past_key_values=cache,
            mode=common_types.MODE_PREFILL,
        )
        np.testing.assert_allclose(
            np.asarray(prefill_out.logits), np.asarray(full_out.logits[:, :4]), rtol=1e-4, atol=1e-4
        )

        cache = prefill_out.past_key_values
        for step in range(4, 7):
            step_out = model(
                input_ids=jnp.asarray(input_ids[:, step : step + 1]),
                past_key_values=cache,
                mode=common_types.MODE_DECODE,
            )
            cache = step_out.past_key_values
            np.testing.assert_allclose(
                np.asarray(step_out.logits),
                np.asarray(full_out.logits[:, step : step + 1]),
                rtol=1e-4,
                atol=1e-4,
            )

    def test_loop_causal_lm_reference_parity_and_cache_decode(self):
        cfg = _tiny_loop_text_config()
        model = Inkling2LoopForCausalLM(cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(22))
        _perturb_zero_init_params(model, seed=22)
        StateDictConverter.validate_reform_param_schema(model._get_reform_param())

        input_ids = np.array([[2, 8, 4, 11, 6, 15]], dtype=np.int32)
        full_out = model(input_ids=jnp.asarray(input_ids), output_router_logits=True)
        assert full_out.logits.shape == (1, 6, cfg.unpadded_vocab_size)
        assert len(full_out.router_logits) == cfg.total_num_hidden_layers

        ref_logits = _np_reference_text_forward(model, input_ids)
        np.testing.assert_allclose(np.asarray(full_out.logits), ref_logits, rtol=1e-3, atol=1e-3)

        # Cached prefill + step-by-step decode across all 10 logical cache slots (6 physical + 2*2 extra passes)
        cache = model.init_cache(batch_size=1, max_length=16)
        assert len(cache) == cfg.total_num_hidden_layers
        prefill_out = model(
            input_ids=jnp.asarray(input_ids[:, :3]),
            past_key_values=cache,
            mode=common_types.MODE_PREFILL,
        )
        np.testing.assert_allclose(
            np.asarray(prefill_out.logits), np.asarray(full_out.logits[:, :3]), rtol=1e-4, atol=1e-4
        )

        cache = prefill_out.past_key_values
        for step in range(3, 6):
            step_out = model(
                input_ids=jnp.asarray(input_ids[:, step : step + 1]),
                past_key_values=cache,
                mode=common_types.MODE_DECODE,
            )
            cache = step_out.past_key_values
            np.testing.assert_allclose(
                np.asarray(step_out.logits),
                np.asarray(full_out.logits[:, step : step + 1]),
                rtol=1e-4,
                atol=1e-4,
            )

    def test_vision_and_vlm_conditional_generation_parity(self):
        vcfg = _tiny_vision_config(text_hidden_size=32)
        vision_mod = Inkling2VisionModel(vcfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(33))

        # Two ragged images: 2x2 LM tokens (4) + 1x2 LM tokens (2) = 6 image tokens
        image_grid_thw = np.array([[1, 2, 2], [1, 1, 2]], dtype=np.int64)
        rng = np.random.default_rng(33)
        pixel_values = rng.standard_normal(
            (6, vcfg.patch_size * vcfg.spatial_merge_size, vcfg.patch_size * vcfg.spatial_merge_size, 3)
        ).astype(np.float32)

        vout = vision_mod(pixel_values=jnp.asarray(pixel_values), image_grid_thw=image_grid_thw)
        ref_pool = _np_reference_vision_forward(vision_mod, pixel_values, image_grid_thw)
        assert vout.pooler_output.shape == (6, 32)
        np.testing.assert_allclose(np.asarray(vout.pooler_output), ref_pool, rtol=1e-4, atol=1e-4)

        # End-to-end VLM forward (both flat and looped)
        for cfg_cls, vlm_cls in (
            (Inkling2Config, Inkling2ForConditionalGeneration),
            (Inkling2LoopConfig, Inkling2LoopForConditionalGeneration),
        ):
            tcfg = _tiny_text_config() if cfg_cls is Inkling2Config else _tiny_loop_text_config()
            vlm_cfg = cfg_cls(text_config=tcfg, vision_config=_tiny_vision_config(tcfg.hidden_size), image_token_id=60)
            vlm = vlm_cls(vlm_cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(44))
            StateDictConverter.validate_reform_param_schema(vlm._get_reform_param())

            # Sequence with 6 image placeholder tokens (id=60)
            ids = jnp.array([[1, 60, 60, 60, 60, 4, 60, 60, 9]], dtype=jnp.int32)
            out = vlm(input_ids=ids, pixel_values=jnp.asarray(pixel_values), image_grid_thw=image_grid_thw)
            assert out.logits.shape == (1, 9, tcfg.unpadded_vocab_size)
            assert out.image_hidden_states.shape == (6, tcfg.hidden_size)
            assert bool(jnp.all(jnp.isfinite(out.logits)))

            # Verify process_encoder precomputed features + valid-length masking matches pixel_values exactly.
            img_feats = vlm.get_image_features(jnp.asarray(pixel_values), image_grid_thw)
            padded_feats = jnp.concatenate([img_feats, jnp.full((3, tcfg.hidden_size), 99.0, dtype=img_feats.dtype)])
            precomp_out = vlm(
                input_ids=ids,
                image_features=padded_feats,
                image_features_valid_length=jnp.asarray(img_feats.shape[0], dtype=jnp.int32),
            )
            np.testing.assert_allclose(np.asarray(precomp_out.logits), np.asarray(out.logits), rtol=1e-5, atol=1e-5)

            # Verify generation input preparation & single-step update (dropping multimodal keys after prefill).
            gen_kwargs = vlm.prepare_inputs_for_generation(
                input_ids=ids[:, :7],
                max_length=16,
                pad_token_id=0,
                pixel_values=jnp.asarray(pixel_values),
                image_grid_thw=image_grid_thw,
            )
            assert gen_kwargs["pixel_values"] is not None
            prefill_vlm = vlm(input_ids=ids[:, :7], **gen_kwargs)
            next_kwargs = vlm.update_inputs_for_generation(prefill_vlm, dict(gen_kwargs))
            assert "pixel_values" not in next_kwargs
            assert "image_grid_thw" not in next_kwargs
            step_vlm = vlm(input_ids=ids[:, 7:8], **next_kwargs)
            assert step_vlm.logits.shape == (1, 1, tcfg.unpadded_vocab_size)
            assert bool(jnp.all(jnp.isfinite(step_vlm.logits)))

    def test_lm_head_postprocess_and_ue8m0_codec_parity(self):
        cfg = _tiny_text_config()
        model = Inkling2ForCausalLM(cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(19))
        hidden = jnp.asarray(np.random.default_rng(19).standard_normal((1, 4, cfg.hidden_size)).astype(np.float32))
        direct = model.compute_lm_logits(hidden)
        via_fn = model.make_lm_head_fn()(hidden)
        post = model.get_lm_logit_postprocess()
        assert post is not None
        via_post = post(model.apply_lm_head(hidden))
        np.testing.assert_allclose(np.asarray(via_fn), np.asarray(direct), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(via_post), np.asarray(direct), rtol=1e-5, atol=1e-5)

        # Shared MXFP8 E8M0 scale decoding in easydel.layers.quantization.checkpoint._codecs
        rng = np.random.default_rng(29)
        w = rng.integers(-4, 5, size=(8, 64)).astype(np.float32)
        s = np.array([[127, 128], [126, 129]] * 4, dtype=np.uint8)
        decoded = decode_scaled_elements(
            jnp.asarray(w),
            jnp.asarray(s),
            transpose=True,
            dtype=jnp.float32,
            block_shape=(1, 32),
            scale_fmt="ue8m0",
        )
        expected = np.concatenate(
            [
                w[:, :32] * (2.0 ** (s[:, 0:1].astype(np.float32) - 127.0)),
                w[:, 32:] * (2.0 ** (s[:, 1:2].astype(np.float32) - 127.0)),
            ],
            axis=1,
        ).T
        np.testing.assert_allclose(np.asarray(decoded), expected, rtol=0, atol=0)

    def test_packed_segment_isolation(self):
        cfg = _tiny_text_config()
        model = Inkling2ForCausalLM(cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(55))
        _perturb_zero_init_params(model, seed=55)

        seg_a = jnp.array([[3, 7, 11]], dtype=jnp.int32)
        seg_b = jnp.array([[5, 9, 13, 2]], dtype=jnp.int32)
        out_a = model(input_ids=seg_a).logits
        out_b = model(input_ids=seg_b).logits

        packed_ids = jnp.concatenate([seg_a, seg_b], axis=1)
        seg_ids = jnp.array([[0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32)
        packed_pos = jnp.array([[0, 1, 2, 0, 1, 2, 3]], dtype=jnp.int32)
        mask_info = MaskInfo.from_segments(seg_ids)
        packed_out = model(input_ids=packed_ids, mask_info=mask_info, position_ids=packed_pos).logits

        np.testing.assert_allclose(np.asarray(packed_out[:, :3]), np.asarray(out_a), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.asarray(packed_out[:, 3:]), np.asarray(out_b), rtol=1e-4, atol=1e-4)

    def test_zero_input_l2_clip_and_full_backward_gradients_finite(self):
        clip = Inkling2L2Clip(threshold=4.0)
        zeros = jnp.zeros((2, 3, 8), dtype=jnp.float32)
        g_zero = jax.grad(lambda x: jnp.sum(clip(x)))(zeros)
        assert bool(jnp.all(jnp.isfinite(g_zero)))

        cfg = _tiny_loop_text_config()
        model = Inkling2LoopForCausalLM(cfg, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(66))
        # Verify fp32-pinned parameters stay float32 even under bfloat16 param_dtype
        for path, var in spx.iter_variables(model):
            if not isinstance(var, spx.Parameter):
                continue
            name = ".".join(str(p) for p in path)
            if any(
                pat in name
                for pat in (
                    "e_score_correction_bias",
                    "mlp.gate.weight",
                    "post_attention_convolution",
                    "post_mlp_convolution",
                    "sconv.conv1d",
                    "attention_input.query",
                    "mlp_input.query",
                    "final_input.query",
                    "message_input.query",
                    "forget_gate_proj",
                    "message_gate_proj",
                )
            ):
                assert var.value.dtype == jnp.float32, f"Expected fp32 for {name}, got {var.value.dtype}"

        gdef, params = spx.export(model)

        def loss_fn(p):
            m = spx.bind(gdef, p)
            logits = m(input_ids=jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)).logits
            return jnp.mean(logits**2)

        grads = jax.grad(loss_fn)(params)
        for path, leaf in grads.flatten().items():
            assert bool(jnp.all(jnp.isfinite(leaf))), f"Non-finite gradient at {path}"


class TestInkling2WeightConversion:
    """End-to-end ``StateDictConverter.huggingface_to_easydel`` tests on vendor & canonical Torch state dicts."""

    def test_loop_vlm_vendor_nvfp4_and_mxfp8_conversion(self):
        import torch

        tcfg = _tiny_loop_text_config(
            hidden_size=32,
            moe_intermediate_size=16,
            n_routed_experts=2,
            num_experts_per_tok=1,
            num_hidden_layers=6,
            n_layers_per_attn_res_block=2,
            loop_layer_index_start=2,
            loop_layer_index_end=4,
            loop_num_passes=2,
            loop_gate_rank=8,
            vocab_size=64,
            unpadded_vocab_size=50,
        )
        vcfg = _tiny_vision_config(text_hidden_size=32)
        vlm_cfg = Inkling2LoopConfig(text_config=tcfg, vision_config=vcfg, image_token_id=60)
        model = Inkling2LoopForConditionalGeneration(
            vlm_cfg, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(77)
        )
        expected_shapes = {k.split("/", 1)[-1]: tuple(v.shape) for k, v in model.parameters.flatten().items()}

        def _build_vendor_state_dict(mode: str) -> dict[str, torch.Tensor]:
            gen = torch.Generator().manual_seed(77)
            sd: dict[str, torch.Tensor] = {}
            sd["model.llm.embed.weight"] = torch.randn((64, 32), generator=gen, dtype=torch.bfloat16)
            sd["model.llm.embed_norm.weight"] = torch.ones((32,), dtype=torch.bfloat16)
            sd["model.llm.norm.weight"] = torch.ones((32,), dtype=torch.bfloat16)
            sd["model.llm.unembed.weight"] = torch.randn((64, 32), generator=gen, dtype=torch.bfloat16)
            sd["model.llm.attn_res.output_query"] = torch.zeros((32,), dtype=torch.float32)
            sd["model.llm.middle_cycle_cross_pass_queries"] = torch.randn((4, 32), generator=gen, dtype=torch.float32)
            sd["model.llm.middle_cycle_cross_pass_kv_queries"] = torch.randn((4, 32), generator=gen, dtype=torch.float32)

            for i in range(6):
                p = f"model.llm.layers.{i}"
                is_sliding = tcfg.is_sliding_layer(i)
                if i > 0:
                    q_shape = (32,) if is_sliding else (3, 32)
                    sd[f"model.llm.attn_res.attn_queries.{i - 1}"] = torch.zeros(q_shape, dtype=torch.float32)
                sd[f"model.llm.attn_res.mlp_queries.{i}"] = torch.zeros((32,), dtype=torch.float32)
                sd[f"{p}.attn.wq_du.weight"] = torch.randn((32, 32), generator=gen, dtype=torch.bfloat16)
                sd[f"{p}.attn.wk_dv.weight"] = torch.randn((16, 32), generator=gen, dtype=torch.bfloat16)
                sd[f"{p}.attn.wv_dv.weight"] = torch.randn((16, 32), generator=gen, dtype=torch.bfloat16)
                sd[f"{p}.attn.wo_ud.weight"] = torch.randn((32, 32), generator=gen, dtype=torch.bfloat16)
                sd[f"{p}.attn.gate_attn.weight"] = torch.randn((32, 32), generator=gen, dtype=torch.bfloat16)
                sd[f"{p}.attn.attn_sink_proj.weight"] = torch.randn((4, 32), generator=gen, dtype=torch.bfloat16)
                sd[f"{p}.attn.q_l2_clip.scale"] = torch.ones((4, 8), dtype=torch.bfloat16)
                sd[f"{p}.attn.rel_l2_clip.scale"] = torch.ones((4, 4), dtype=torch.bfloat16)
                sd[f"{p}.attn.wr_du.weight"] = torch.randn((16, 32), generator=gen, dtype=torch.bfloat16)
                rel_ext = tcfg.sliding_window_size if is_sliding else tcfg.rel_extent
                sd[f"{p}.attn.rel_logits_proj.proj"] = torch.randn((4, rel_ext), generator=gen, dtype=torch.bfloat16)
                sd[f"{p}.attn_sconv.weight"] = torch.randn((32, 1, 3), generator=gen, dtype=torch.float32)
                sd[f"{p}.mlp_sconv.weight"] = torch.randn((32, 1, 3), generator=gen, dtype=torch.float32)
                sd[f"{p}.mlp_norm.weight"] = torch.ones((32,), dtype=torch.bfloat16)
                if is_sliding:
                    sd[f"{p}.attn_norm.weight"] = torch.ones((32,), dtype=torch.bfloat16)
                    sd[f"{p}.attn.k_sconv.weight"] = torch.randn((16, 1, 3), generator=gen, dtype=torch.float32)
                    sd[f"{p}.attn.v_sconv.weight"] = torch.randn((16, 1, 3), generator=gen, dtype=torch.float32)
                else:
                    sd[f"{p}.q_input_norm.weight"] = torch.ones((32,), dtype=torch.bfloat16)
                    sd[f"{p}.k_input_norm.weight"] = torch.ones((32,), dtype=torch.bfloat16)
                    sd[f"{p}.v_input_norm.weight"] = torch.ones((32,), dtype=torch.bfloat16)
                    sd[f"{p}.attn.k_sconv.static_w"] = torch.randn((3, 16), generator=gen, dtype=torch.bfloat16)
                    sd[f"{p}.attn.k_sconv.weight_proj"] = torch.randn((6, 32), generator=gen, dtype=torch.bfloat16)
                    sd[f"{p}.attn.v_sconv.static_w"] = torch.randn((3, 16), generator=gen, dtype=torch.bfloat16)
                    sd[f"{p}.attn.v_sconv.weight_proj"] = torch.randn((6, 32), generator=gen, dtype=torch.bfloat16)
                if 2 <= i < 4:
                    for ctrl in ("attn", "k", "v", "mlp"):
                        sd[f"{p}.controllers.{ctrl}.down_weight"] = torch.randn(
                            (8, 64), generator=gen, dtype=torch.bfloat16
                        )
                        sd[f"{p}.controllers.{ctrl}.forget_weight"] = torch.randn(
                            (32, 8), generator=gen, dtype=torch.float32
                        )
                        sd[f"{p}.controllers.{ctrl}.gain_weight"] = torch.randn(
                            (32, 8), generator=gen, dtype=torch.float32
                        )

                sd[f"{p}.mlp.gate.weight"] = torch.randn((2, 32), generator=gen, dtype=torch.float32)
                sd[f"{p}.mlp.gate.bias"] = torch.zeros((2,), dtype=torch.float32)
                if mode == "nvfp4":
                    sd[f"{p}.mlp.experts.w13_weight"] = torch.full((2, 32, 16), 0x22, dtype=torch.uint8)
                    sd[f"{p}.mlp.experts.w13_weight_scale"] = torch.ones((2, 32, 2), dtype=torch.float32)
                    sd[f"{p}.mlp.experts.w13_weight_amax"] = torch.ones((2,), dtype=torch.float32)
                    sd[f"{p}.mlp.experts.w13_weight_outer_scale_denominator"] = torch.ones((2,), dtype=torch.float32)
                    sd[f"{p}.mlp.experts.w2_weight"] = torch.full((2, 32, 8), 0x22, dtype=torch.uint8)
                    sd[f"{p}.mlp.experts.w2_weight_scale"] = torch.ones((2, 32, 1), dtype=torch.float32)
                    sd[f"{p}.mlp.experts.w2_weight_amax"] = torch.ones((2,), dtype=torch.float32)
                    sd[f"{p}.mlp.experts.w2_weight_outer_scale_denominator"] = torch.ones((2,), dtype=torch.float32)
                else:
                    # MXFP8 (group size 32 along last axis)
                    sd[f"{p}.mlp.experts.w13_weight"] = torch.ones((2, 32, 32), dtype=torch.float32)
                    sd[f"{p}.mlp.experts.w13_weight_scale"] = torch.full((2, 32, 1), 127, dtype=torch.uint8)
                    # down_proj has in=16 < 32 in tiny config, so use raw BF16 w2_weight here
                    sd[f"{p}.mlp.experts.w2_weight"] = torch.ones((2, 32, 16), dtype=torch.bfloat16)

            # Vision weights
            sd["model.visual.backbone.embeddings.class_embedding"] = torch.zeros((24,), dtype=torch.bfloat16)
            sd["model.visual.backbone.embeddings.cls_position_embedding"] = torch.zeros((24,), dtype=torch.bfloat16)
            sd["model.visual.backbone.embeddings.patch_embedding.weight"] = torch.randn(
                (24, 48), generator=gen, dtype=torch.bfloat16
            )
            sd["model.visual.backbone.embeddings.position_embedding"] = torch.randn(
                (16, 24), generator=gen, dtype=torch.bfloat16
            )
            sd["model.visual.backbone.ln_pre.weight"] = torch.ones((24,), dtype=torch.bfloat16)
            sd["model.visual.backbone.ln_pre.bias"] = torch.zeros((24,), dtype=torch.bfloat16)
            sd["model.visual.proj_in.weight"] = torch.randn((32, 96), generator=gen, dtype=torch.bfloat16)
            sd["model.visual.proj_out.weight"] = torch.randn((32, 32), generator=gen, dtype=torch.bfloat16)
            for vi in range(2):
                vp = f"model.visual.backbone.layers.{vi}"
                sd[f"{vp}.ln_1.weight"] = torch.ones((24,), dtype=torch.bfloat16)
                sd[f"{vp}.ln_1.bias"] = torch.zeros((24,), dtype=torch.bfloat16)
                sd[f"{vp}.ln_2.weight"] = torch.ones((24,), dtype=torch.bfloat16)
                sd[f"{vp}.ln_2.bias"] = torch.zeros((24,), dtype=torch.bfloat16)
                sd[f"{vp}.ls_1.gamma"] = torch.full((24,), 0.1, dtype=torch.bfloat16)
                sd[f"{vp}.ls_2.gamma"] = torch.full((24,), 0.1, dtype=torch.bfloat16)
                sd[f"{vp}.mlp.c_fc.weight"] = torch.randn((48, 24), generator=gen, dtype=torch.bfloat16)
                sd[f"{vp}.mlp.c_fc.bias"] = torch.zeros((48,), dtype=torch.bfloat16)
                sd[f"{vp}.mlp.c_proj.weight"] = torch.randn((24, 48), generator=gen, dtype=torch.bfloat16)
                sd[f"{vp}.mlp.c_proj.bias"] = torch.zeros((24,), dtype=torch.bfloat16)
                for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    sd[f"{vp}.self_attn.{proj}.weight"] = torch.randn((24, 24), generator=gen, dtype=torch.bfloat16)
                    sd[f"{vp}.self_attn.{proj}.bias"] = torch.zeros((24,), dtype=torch.bfloat16)
            return sd

        for mode in ("nvfp4", "mxfp8"):
            raw_sd = _build_vendor_state_dict(mode)
            converted = model._build_transform_fn(shard_fns=None)(raw_sd, verbose=False)
            from easydel.utils.traversals import flatten_dict

            flat_conv = {".".join(str(p) for p in k): v for k, v in flatten_dict(converted).items()}
            assert set(flat_conv.keys()) == set(expected_shapes.keys())
            for k, exp_shape in expected_shapes.items():
                assert tuple(flat_conv[k].shape) == exp_shape, f"Shape mismatch for {k} ({mode})"
            # Verify static_weight was NOT transposed (remains [conv_kernel_size, kv_dim] = (3, 16))
            assert flat_conv["model.language_model.layers.1.self_attn.k_proj.sconv.static_weight"].shape == (3, 16)
            # Verify lm_head was unpadded to unpadded_vocab_size=50 and transposed to (32, 50)
            assert flat_conv["lm_head.weight"].shape == (32, 50)
            # Verify fp32-pinned tensors remained float32
            assert flat_conv["model.language_model.message_input.query"].dtype == jnp.float32
            assert flat_conv["model.language_model.layers.0.mlp.gate.weight"].dtype == jnp.float32
            assert (
                flat_conv["model.language_model.layers.0.post_attention_convolution.conv1d.weight"].dtype == jnp.float32
            )
