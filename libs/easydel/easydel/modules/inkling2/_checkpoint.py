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

"""Checkpoint key normalization and weight transformation helpers for Inkling 2 / Inkling 2 Loop.

Handles both canonical Hugging Face checkpoints and Thinking Machines' raw Hub
checkpoints (BF16, MXFP8 with E8M0 scales, and NVFP4 with E2M1 nibbles + E4M3
block scales + per-expert ``amax / denominator``).
"""

from __future__ import annotations

import re
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

from easydel.layers.layouts import (
    interleave_segments_last_axis,
    split_interleaved_segments_last_axis,
    tensor_parallel_size,
    torch_deinterleave_axis_segments_for_tp,
    torch_interleave_axis_segments_for_tp,
)
from easydel.layers.quantization.checkpoint._decode import (
    decode_e4m3_scale,
    decode_e8m0_scale,
    expand_scale,
    unpack_uint8_to_e2m1,
)
from easydel.utils.checkpoint_keys import KEEP, KeyRuleSet, Map, Rewrite

if tp.TYPE_CHECKING:
    from .inkling2_configuration import Inkling2TextConfig

FP32_PARAM_PATTERNS: tp.Final[tuple[str, ...]] = (
    "e_score_correction_bias",
    "mlp.gate.weight",
    "post_attention_convolution",
    "post_mlp_convolution",
    "sconv.conv1d",
    "attention_input",
    "mlp_input",
    "final_input",
)

LOOP_FP32_PARAM_PATTERNS: tp.Final[tuple[str, ...]] = (
    *FP32_PARAM_PATTERNS,
    "message_input",
    "message_gate_proj",
    "forget_gate_proj",
)

# E2M1 lookup values used only for pure-Torch / pure-NumPy host conversion paths
# (JAX inputs use `easydel.layers.quantization.checkpoint._decode.unpack_uint8_to_e2m1`).
_E2M1_VALUES: tp.Final[tuple[float, ...]] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

_ATTN_QUERY_INDEX_RE = re.compile(r"^(?P<prefix>(?:[A-Za-z0-9_]+\.)*)attn_res\.attn_queries\.(?P<idx>\d+)$")


class Inkling2KeyNormalizer:
    """Checkpoint key normalizer for Thinking Machines raw and canonical HF Inkling 2 checkpoints.

    Wraps an ordered :class:`KeyRuleSet` with the ``+1`` index shift required
    for ``attn_res.attn_queries.{i}`` (which belongs to
    ``layers.{i + 1}.attention_input.query`` because layer 0 has no
    ``attention_input``). Canonical Hugging Face / EasyDeL keys pass through
    untouched; unowned keys (such as ``mtp.*`` or vision keys on text-only
    models) normalize to ``None``.
    """

    def __init__(self, rules: KeyRuleSet, *, name: str = ""):
        self.rules = rules
        self.name = name or rules.name

    def normalize(self, key: str) -> str | None:
        """Normalize one raw checkpoint key to its EasyDeL name, or ``None`` if unowned."""
        mapped = self.rules(key)
        if mapped is None:
            return None
        match = _ATTN_QUERY_INDEX_RE.fullmatch(mapped)
        if match is not None:
            return f"{match.group('prefix')}layers.{int(match.group('idx')) + 1}.attention_input.query"
        return mapped

    __call__ = normalize

    def __repr__(self) -> str:
        return f"Inkling2KeyNormalizer({self.name!r})"


_VISION_REWRITES: tuple[Rewrite, ...] = (
    Rewrite(r"(?P<p>^|\.)vision_tower\.backbone\.embeddings\.", "{p}vision_tower.embeddings."),
    Rewrite(
        r"(?P<p>^|\.)vision_tower\.embeddings\.position_embedding$",
        "{p}vision_tower.embeddings.position_embedding.weight",
    ),
    Rewrite(r"(?P<p>^|\.)vision_tower\.backbone\.ln_pre\.", "{p}vision_tower.pre_layernorm."),
    Rewrite(r"(?P<p>^|\.)vision_tower\.backbone\.layers\.", "{p}vision_tower.layers."),
    Rewrite(r"(?P<p>^|\.)vision_tower\.proj_in\.", "{p}vision_tower.merger.linear_1."),
    Rewrite(r"(?P<p>^|\.)vision_tower\.proj_out\.", "{p}vision_tower.merger.linear_2."),
    Rewrite(r"(?P<layer>vision_tower\.layers\.\d+)\.ln_1\.", "{layer}.layernorm_before."),
    Rewrite(r"(?P<layer>vision_tower\.layers\.\d+)\.ln_2\.", "{layer}.layernorm_after."),
    Rewrite(r"(?P<layer>vision_tower\.layers\.\d+)\.mlp\.c_fc\.", "{layer}.mlp.fc1."),
    Rewrite(r"(?P<layer>vision_tower\.layers\.\d+)\.mlp\.c_proj\.", "{layer}.mlp.fc2."),
    Rewrite(r"(?P<layer>vision_tower\.layers\.\d+)\.ls_1\.gamma$", "{layer}.lambda_1"),
    Rewrite(r"(?P<layer>vision_tower\.layers\.\d+)\.ls_2\.gamma$", "{layer}.lambda_2"),
)

_TEXT_BODY_REWRITES: tuple[Rewrite, ...] = (
    Rewrite(r"(?P<p>^|\.)embed\.weight$", "{p}embed_tokens.weight"),
    Rewrite(r"(?<!embed_tokens)(?P<p>^|\.)embed_norm\.", "{p}embed_tokens.embed_norm."),
    Rewrite(r"\.mlp\.gate\.bias$", ".mlp.gate.e_score_correction_bias"),
    Rewrite(r"\.attn\.", ".self_attn."),
    Rewrite(r"\.self_attn\.wq_du\.", ".self_attn.q_proj.linear."),
    Rewrite(r"\.self_attn\.wk_dv\.", ".self_attn.k_proj.linear."),
    Rewrite(r"\.self_attn\.wv_dv\.", ".self_attn.v_proj.linear."),
    Rewrite(r"\.self_attn\.k_sconv\.", ".self_attn.k_proj.sconv."),
    Rewrite(r"\.self_attn\.v_sconv\.", ".self_attn.v_proj.sconv."),
    Rewrite(r"\.self_attn\.rel_logits_proj\.", ".self_attn.relative_position_bias."),
    Rewrite(r"\.self_attn\.wr_du\.", ".self_attn.relative_position_bias.linear."),
    Rewrite(r"\.self_attn\.q_l2_clip\.", ".self_attn.q_proj.norm."),
    Rewrite(r"\.self_attn\.rel_l2_clip\.", ".self_attn.relative_position_bias.norm."),
    Rewrite(r"\.self_attn\.wo_ud\.", ".self_attn.o_proj.linear."),
    Rewrite(r"\.self_attn\.gate_attn\.", ".self_attn.o_proj.gate_proj."),
    Rewrite(r"\.self_attn\.attn_sink_proj\.", ".self_attn.sink_proj."),
    Rewrite(r"\.attn_sconv\.", ".post_attention_convolution."),
    Rewrite(r"\.mlp_sconv\.", ".post_mlp_convolution."),
    Rewrite(r"(?P<conv>convolution|sconv)\.weight(?!_)", "{conv}.conv1d.weight"),
    Rewrite(r"(?P<conv>convolution|sconv)\.static_w$", "{conv}.static_weight"),
    Rewrite(r"(?P<conv>convolution|sconv)\.weight_proj$", "{conv}.weight_proj.weight"),
    Rewrite(r"\.mlp_norm\.", ".post_attention_layernorm."),
    Rewrite(r"\.attn_norm\.", ".input_layernorm."),
    Rewrite(r"\.(?P<kind>[qkv])_input_norm\.", ".{kind}_input_layernorm."),
    Rewrite(r"(?P<p>^|\.)attn_res\.mlp_queries\.(?P<idx>\d+)$", "{p}layers.{idx}.mlp_input.query"),
    Rewrite(r"(?P<p>^|\.)attn_res\.output_query$", "{p}final_input.query"),
)

_LOOP_BODY_REWRITES: tuple[Rewrite, ...] = (
    Rewrite(r"\.controllers\.attn\.", r".q_message_gate."),
    Rewrite(r"\.controllers\.k\.", r".k_message_gate."),
    Rewrite(r"\.controllers\.v\.", r".v_message_gate."),
    Rewrite(r"\.controllers\.mlp\.", r".mlp_message_gate."),
    Rewrite(r"_message_gate\.down_weight$", r"_message_gate.down_proj.weight"),
    Rewrite(r"_message_gate\.gain_weight$", r"_message_gate.message_gate_proj.weight"),
    Rewrite(r"_message_gate\.forget_weight$", r"_message_gate.forget_gate_proj.weight"),
)


def _build_inkling2_key_normalizer(
    *,
    is_vlm: bool,
    has_lm_head: bool,
    is_loop: bool,
    name: str,
) -> Inkling2KeyNormalizer:
    """Build a declarative :class:`Inkling2KeyNormalizer` for one model entry point."""
    rules: list[Map | Rewrite] = [
        Map(r"(?:model\.)?mtp\..*", None),
    ]
    if not is_vlm:
        rules.append(Map(r"(?:model\.)?(?:visual|vision_tower)\..*", None))

    if has_lm_head:
        rules.append(Map(r"model\.llm\.unembed\.weight", "lm_head.weight"))
    else:
        rules.append(Map(r"(?:model\.llm\.unembed|lm_head)\.weight", None))

    if is_vlm:
        if has_lm_head:
            rules.append(Rewrite(r"^model\.llm\.", "model.language_model.", count=1))
            rules.append(Rewrite(r"^model\.visual\.", "model.vision_tower.", count=1))
        else:
            rules.append(Rewrite(r"^model\.llm\.", "language_model.", count=1))
            rules.append(Rewrite(r"^model\.visual\.", "vision_tower.", count=1))
            rules.append(Rewrite(r"^model\.language_model\.", "language_model.", count=1))
            rules.append(Rewrite(r"^model\.vision_tower\.", "vision_tower.", count=1))
        rules.extend(_VISION_REWRITES)
    else:
        if has_lm_head:
            rules.append(Rewrite(r"^model\.llm\.", "model.", count=1))
            rules.append(Rewrite(r"^model\.language_model\.", "model.", count=1))
        else:
            rules.append(Rewrite(r"^model\.llm\.", "", count=1))
            rules.append(Rewrite(r"^model\.language_model\.", "", count=1))
            rules.append(Rewrite(r"^model\.", "", count=1))

    if is_loop:
        rules.extend(_LOOP_BODY_REWRITES)
    rules.extend(_TEXT_BODY_REWRITES)

    return Inkling2KeyNormalizer(KeyRuleSet(rules, otherwise=KEEP, name=name), name=name)


INKLING2_TEXT_KEY_RULES = _build_inkling2_key_normalizer(
    is_vlm=False, has_lm_head=False, is_loop=False, name="inkling2_text"
)
INKLING2_CAUSAL_LM_KEY_RULES = _build_inkling2_key_normalizer(
    is_vlm=False, has_lm_head=True, is_loop=False, name="inkling2_causal_lm"
)
INKLING2_VLM_MODEL_KEY_RULES = _build_inkling2_key_normalizer(
    is_vlm=True, has_lm_head=False, is_loop=False, name="inkling2_vlm_model"
)
INKLING2_VLM_KEY_RULES = _build_inkling2_key_normalizer(
    is_vlm=True, has_lm_head=True, is_loop=False, name="inkling2_vlm"
)
INKLING2_LOOP_TEXT_KEY_RULES = _build_inkling2_key_normalizer(
    is_vlm=False, has_lm_head=False, is_loop=True, name="inkling2_loop_text"
)
INKLING2_LOOP_CAUSAL_LM_KEY_RULES = _build_inkling2_key_normalizer(
    is_vlm=False, has_lm_head=True, is_loop=True, name="inkling2_loop_causal_lm"
)
INKLING2_LOOP_VLM_MODEL_KEY_RULES = _build_inkling2_key_normalizer(
    is_vlm=True, has_lm_head=False, is_loop=True, name="inkling2_loop_vlm_model"
)
INKLING2_LOOP_VLM_KEY_RULES = _build_inkling2_key_normalizer(
    is_vlm=True, has_lm_head=True, is_loop=True, name="inkling2_loop_vlm"
)

INKLING2_VISION_KEY_RULES = Inkling2KeyNormalizer(
    KeyRuleSet(
        [
            Rewrite(r"^model\.visual\.", "vision_tower.", count=1),
            Rewrite(r"^model\.vision_tower\.", "vision_tower.", count=1),
            Rewrite(r"^backbone\.", "vision_tower.backbone.", count=1),
            Rewrite(r"^proj_in\.", "vision_tower.proj_in.", count=1),
            Rewrite(r"^proj_out\.", "vision_tower.proj_out.", count=1),
            *_VISION_REWRITES,
            Rewrite(r"^vision_tower\.", "", count=1),
        ],
        otherwise=KEEP,
        name="inkling2_vision",
    ),
    name="inkling2_vision",
)


def _is_torch_tensor(x: tp.Any) -> bool:
    """Return whether ``x`` is a PyTorch tensor without importing torch eagerly."""
    return hasattr(x, "detach") and str(getattr(x, "dtype", "")).startswith("torch.")


def deinterleave_inkling2_gate_up(tensor: tp.Any) -> tp.Any:
    """Convert interleaved ``[E, 2 * I, H]`` (`[g0, u0, g1, u1, ...]`) to concatenated ``[gate | up]``."""
    num_experts, two_i, hidden = tensor.shape
    if _is_torch_tensor(tensor):
        return tensor.reshape(num_experts, two_i // 2, 2, hidden).transpose(1, 2).reshape(num_experts, two_i, hidden)
    if isinstance(tensor, jax.Array):
        return (
            tensor.reshape(num_experts, two_i // 2, 2, hidden).transpose(0, 2, 1, 3).reshape(num_experts, two_i, hidden)
        )
    arr = np.asarray(tensor)
    return arr.reshape(num_experts, two_i // 2, 2, hidden).transpose(0, 2, 1, 3).reshape(num_experts, two_i, hidden)


def interleave_inkling2_gate_up(tensor: tp.Any) -> tp.Any:
    """Inverse of :func:`deinterleave_inkling2_gate_up`: concatenated ``[gate | up]`` -> interleaved."""
    num_experts, two_i, hidden = tensor.shape
    if _is_torch_tensor(tensor):
        return tensor.reshape(num_experts, 2, two_i // 2, hidden).transpose(1, 2).reshape(num_experts, two_i, hidden)
    if isinstance(tensor, jax.Array):
        return (
            tensor.reshape(num_experts, 2, two_i // 2, hidden).transpose(0, 2, 1, 3).reshape(num_experts, two_i, hidden)
        )
    arr = np.asarray(tensor)
    return arr.reshape(num_experts, 2, two_i // 2, hidden).transpose(0, 2, 1, 3).reshape(num_experts, two_i, hidden)


def dequantize_inkling2_experts(
    weight: tp.Any,
    scale: tp.Any | None = None,
    amax: tp.Any | None = None,
    denominator: tp.Any | None = None,
    *,
    deinterleave: bool = False,
) -> tp.Any:
    """Dequantize Thinking Machines' NVFP4 or MXFP8 stacked expert weights.

    Matches upstream ``Inkling2LoopDequantizeExperts``:

    - **NVFP4** (``amax`` and ``denominator`` present): two E2M1 nibbles packed
      per ``uint8`` byte (even index in the low nibble), one float8 scale per
      16 elements along the last dimension, and per-expert factor
      ``amax / denominator``.
    - **MXFP8** (``scale`` present, ``amax`` is ``None``): float8 values with
      one E8M0 power-of-two scale ``2 ** (scale - 127)`` per 32 elements along
      the last dimension.
    - **Unquantized** (``scale is None``): passes ``weight`` through (applying
      ``deinterleave`` if requested).

    Uses EasyDeL's shared :mod:`easydel.layers.quantization.checkpoint._decode`
    bitcast primitives for JAX arrays, chunked PyTorch decoding for Hub CPU
    loading, and pure NumPy for host-only conversion tests.
    """
    if scale is None:
        return deinterleave_inkling2_gate_up(weight) if deinterleave else weight

    if isinstance(weight, jax.Array):
        if amax is None:
            decoded_scale = decode_e8m0_scale(jnp.asarray(scale), dtype=jnp.float32)
            out_jax = weight.astype(jnp.float32) * expand_scale(decoded_scale, weight.shape, block_shape=(1, 32))
        else:
            codes = unpack_uint8_to_e2m1(weight.astype(jnp.uint8), axis=-1).astype(jnp.float32)
            block_scale = decode_e4m3_scale(jnp.asarray(scale), dtype=jnp.float32)
            expert_factor = jnp.asarray(amax, dtype=jnp.float32) / jnp.asarray(denominator, dtype=jnp.float32)
            out_jax = codes * expand_scale(block_scale, codes.shape, block_shape=(1, 16)) * expert_factor[:, None, None]
        return deinterleave_inkling2_gate_up(out_jax) if deinterleave else out_jax

    if _is_torch_tensor(weight):
        import torch

        num_experts, rows = weight.shape[:2]
        columns = weight.shape[2] * 2 if amax is not None else weight.shape[2]
        out = torch.empty((num_experts, rows, columns), dtype=torch.bfloat16, device=weight.device)
        if amax is not None:
            e2m1 = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=weight.device)
            byte = torch.arange(256, device=weight.device)
            byte_to_vals = torch.stack((e2m1[byte & 0xF], e2m1[byte >> 4]), dim=-1)
            expert_factor = amax.float() / denominator.float()
        for start in range(0, num_experts, 8):
            sl = slice(start, start + 8)
            block_scale = scale[sl].float()
            if amax is None:
                vals = weight[sl].float().reshape(*block_scale.shape, 32)
                vals = vals * torch.exp2(block_scale - 127.0)[..., None]
            else:
                codes = weight[sl].to(torch.uint8).long()
                vals = byte_to_vals[codes].reshape(*block_scale.shape, 16) * block_scale[..., None]
                vals = vals * expert_factor[sl, None, None, None]
            out[sl] = vals.reshape(-1, rows, columns).to(torch.bfloat16)
        return deinterleave_inkling2_gate_up(out) if deinterleave else out

    w_np = np.asarray(weight)
    s_np = np.asarray(scale).astype(np.float32)
    num_experts, rows = w_np.shape[:2]
    if amax is None:
        columns = w_np.shape[2]
        vals = w_np.astype(np.float32).reshape(*s_np.shape, 32)
        vals = vals * np.exp2(s_np - 127.0)[..., None]
        out_np = vals.reshape(num_experts, rows, columns).astype(np.float32)
    else:
        columns = w_np.shape[2] * 2
        e2m1 = np.asarray(_E2M1_VALUES, dtype=np.float32)
        byte = np.arange(256, dtype=np.int32)
        byte_to_vals = np.stack((e2m1[byte & 0xF], e2m1[byte >> 4]), axis=-1)
        codes = w_np.astype(np.uint8)
        vals = byte_to_vals[codes].reshape(*s_np.shape, 16) * s_np[..., None]
        expert_factor = np.asarray(amax, dtype=np.float32) / np.asarray(denominator, dtype=np.float32)
        vals = vals * expert_factor[:, None, None, None]
        out_np = vals.reshape(num_experts, rows, columns).astype(np.float32)
    return deinterleave_inkling2_gate_up(out_np) if deinterleave else out_np


def merge_inkling2_loop_message_queries(queries: tp.Any, kv_queries: tp.Any) -> tp.Any:
    """Merge Thinking Machines' two ``(2 * L, H)`` message query tensors into ``(L, 4, H)``.

    ``queries`` stores ``[q, mlp]`` per loop layer; ``kv_queries`` stores
    ``[k, v]`` per loop layer. The merged tensor orders messages as
    ``[q, k, v, mlp]`` per loop layer.
    """
    if _is_torch_tensor(queries):
        import torch

        q_mlp = queries.unflatten(0, (-1, 2))
        k_v = kv_queries.unflatten(0, (-1, 2))
        return torch.cat([q_mlp[:, :1], k_v, q_mlp[:, 1:]], dim=1)
    q_np = np.asarray(queries)
    kv_np = np.asarray(kv_queries)
    num_layers = q_np.shape[0] // 2
    q_mlp = q_np.reshape(num_layers, 2, -1)
    k_v = kv_np.reshape(num_layers, 2, -1)
    return np.concatenate([q_mlp[:, :1], k_v, q_mlp[:, 1:]], axis=1)


def split_inkling2_loop_message_queries(merged: tp.Any) -> tuple[tp.Any, tp.Any]:
    """Inverse of :func:`merge_inkling2_loop_message_queries`: ``(L, 4, H)`` -> ``(queries, kv_queries)``."""
    if _is_torch_tensor(merged):
        queries = merged[:, [0, 3]].flatten(0, 1).contiguous()
        kv_queries = merged[:, 1:3].flatten(0, 1).contiguous()
        return queries, kv_queries
    arr = np.asarray(merged)
    num_layers, _, hidden = arr.shape
    queries = np.ascontiguousarray(arr[:, [0, 3], :].reshape(num_layers * 2, hidden))
    kv_queries = np.ascontiguousarray(arr[:, 1:3, :].reshape(num_layers * 2, hidden))
    return queries, kv_queries


def unpad_inkling2_lm_head(weight: tp.Any, unpadded_vocab_size: int) -> tp.Any:
    """Slice HF/vendor ``lm_head.weight`` ``[vocab, hidden]`` to ``unpadded_vocab_size`` and transpose to ``[H, V]``."""
    sliced = weight[:unpadded_vocab_size]
    if _is_torch_tensor(sliced):
        return sliced.transpose(0, 1).contiguous()
    return np.ascontiguousarray(np.asarray(sliced).T)


def pad_inkling2_lm_head(weight: tp.Any, vocab_size: int) -> tp.Any:
    """Transpose EasyDeL ``lm_head.weight`` ``[H, unpadded_V]`` to ``[unpadded_V, H]`` and zero-pad to ``vocab_size``."""
    if _is_torch_tensor(weight):
        import torch

        unpadded = weight.transpose(0, 1).contiguous()
        pad_rows = vocab_size - unpadded.shape[0]
        if pad_rows > 0:
            padding = unpadded.new_zeros(pad_rows, unpadded.shape[1])
            return torch.cat([unpadded, padding], dim=0)
        return unpadded
    unpadded = np.ascontiguousarray(np.asarray(weight).T)
    pad_rows = vocab_size - unpadded.shape[0]
    if pad_rows > 0:
        padding = np.zeros((pad_rows, unpadded.shape[1]), dtype=unpadded.dtype)
        return np.concatenate([unpadded, padding], axis=0)
    return unpadded


def tp_pack_expert_gate_up(gate_up: tp.Any, *, config: Inkling2TextConfig | None) -> tp.Any:
    """Transpose ``[E, 2*I, H]`` -> ``[E, H, 2*I]`` and TP-interleave the gate/up halves."""
    tp_size = 1 if getattr(config, "use_expert_tensor_mode", False) else tensor_parallel_size(config, arr=gate_up)
    if _is_torch_tensor(gate_up):
        import torch

        transposed = gate_up.transpose(-1, -2).contiguous()
        half = int(transposed.shape[2]) // 2
        return torch_interleave_axis_segments_for_tp(torch, transposed, (half, half), tp_size=tp_size, dim=2)
    transposed = np.ascontiguousarray(np.asarray(gate_up).transpose(0, 2, 1))
    half = int(transposed.shape[2]) // 2
    if tp_size <= 1:
        return transposed
    return np.asarray(interleave_segments_last_axis(jnp.asarray(transposed), (half, half), tp_size=tp_size))


def tp_unpack_expert_gate_up(gate_up: tp.Any, *, config: Inkling2TextConfig | None) -> tp.Any:
    """Inverse of :func:`tp_pack_expert_gate_up`: de-interleave TP halves and transpose to ``[E, 2*I, H]``."""
    tp_size = 1 if getattr(config, "use_expert_tensor_mode", False) else tensor_parallel_size(config, arr=gate_up)
    if _is_torch_tensor(gate_up):
        import torch

        half = int(gate_up.shape[2]) // 2
        deinterleaved = torch_deinterleave_axis_segments_for_tp(torch, gate_up, (half, half), tp_size=tp_size, dim=2)
        return deinterleaved.transpose(-1, -2).contiguous()
    arr = np.asarray(gate_up)
    half = int(arr.shape[2]) // 2
    if tp_size > 1:
        g, u = split_interleaved_segments_last_axis(jnp.asarray(arr), (half, half), tp_size=tp_size)
        arr = np.asarray(jnp.concatenate([g, u], axis=-1))
    return np.ascontiguousarray(arr.transpose(0, 2, 1))


def transpose_expert_3d(tensor: tp.Any) -> tp.Any:
    """Transpose a 3-D expert weight ``[E, A, B]`` -> ``[E, B, A]``."""
    if _is_torch_tensor(tensor):
        return tensor.transpose(-1, -2).contiguous()
    return np.ascontiguousarray(np.asarray(tensor).transpose(0, 2, 1))


__all__ = [
    "FP32_PARAM_PATTERNS",
    "INKLING2_CAUSAL_LM_KEY_RULES",
    "INKLING2_LOOP_CAUSAL_LM_KEY_RULES",
    "INKLING2_LOOP_TEXT_KEY_RULES",
    "INKLING2_LOOP_VLM_KEY_RULES",
    "INKLING2_LOOP_VLM_MODEL_KEY_RULES",
    "INKLING2_TEXT_KEY_RULES",
    "INKLING2_VISION_KEY_RULES",
    "INKLING2_VLM_KEY_RULES",
    "INKLING2_VLM_MODEL_KEY_RULES",
    "LOOP_FP32_PARAM_PATTERNS",
    "Inkling2KeyNormalizer",
    "deinterleave_inkling2_gate_up",
    "dequantize_inkling2_experts",
    "interleave_inkling2_gate_up",
    "merge_inkling2_loop_message_queries",
    "pad_inkling2_lm_head",
    "split_inkling2_loop_message_queries",
    "tp_pack_expert_gate_up",
    "tp_unpack_expert_gate_up",
    "transpose_expert_3d",
    "unpad_inkling2_lm_head",
]
