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

"""Spectrax implementation of DeepSeek-V4 (stateless forward).

DeepSeek-V4 combines several genuinely novel components:

- **Manifold-constrained hyper-connections (mHC)**: the residual stream is a
  stack of ``hc_mult`` parallel streams ``[B, S, hc, D]``; two
  :class:`DeepseekV4HyperConnection` modules per decoder layer collapse the
  streams into each sub-layer's input and re-expand the sub-layer output,
  mixing streams through a Sinkhorn-projected doubly-stochastic matrix.
  A final :class:`DeepseekV4HyperHead` collapses the streams before the final
  norm. All mHC math runs in float32.
- **Shared-KV MQA attention** with per-head learnable sinks (gpt-oss style),
  partial *interleaved* RoPE on the trailing ``qk_rope_head_dim`` channels
  (two rope bases: "main" for sliding layers, "compress" for CSA/HCA layers),
  an inverse rotation applied to the attention output (because ``V == K``
  carries RoPE), and a grouped block-diagonal low-rank output projection.
- **Compressed attention**: CSA/HCA layers run a compressor that emits one
  compressed KV entry per window of ``compress_rate`` source tokens; entries
  are concatenated after the token KVs with an additive per-query
  ``block_bias``. CSA additionally runs a Lightning Indexer that keeps only
  the ``index_topk`` best entries per query.
- **MoE on every layer**: ``sqrtsoftplus`` scoring, top-k selection on
  ``scores + e_score_correction_bias`` with weights gathered from the raw
  scores, clamped-SwiGLU experts and shared expert, and hash routing on the
  first ``mlp_layer_types == "hash_moe"`` layers via a frozen
  ``tid2eid[input_ids]`` table.

Two execution regimes are supported:

- **Stateless** forward (training / ``use_cache=False``): only complete
  compression windows are compressed; the remainder tokens are discarded,
  exactly like the HF reference in stateless mode.
- **Cached prefill + decode** via :class:`~easydel.caching.CompressedWindowCache`
  (see ``easydel/caching/compressed_window``): every layer keeps a
  ``sliding_window`` K==V ring; CSA/HCA layers additionally keep fixed-shape
  compressor stream state (open-window kv/gate buffers, preallocated
  compressed entries, CSA Ca-overlap slices). A window closes exactly when
  the stream position ``p`` satisfies ``(p + 1) % rate == 0``; the closing
  step emits one compressed entry under ``jnp.where`` selection, keeping all
  shapes static for ``jax.jit`` / ``lax.while_loop`` generation loops.
  Cache rows keep independent per-row stream counters. The vanilla
  ``generate()`` path advances all rows in lockstep and its batched prefill
  must start from an empty cache at position 0 without left padding.
- **eSurge packed serving**: the engine allocates the cache with one row per
  request slot (``batch = max_num_reqs``) and hands the model a flat
  ``[1, T]`` packed multi-request stream plus ragged metadata
  (``query_start_loc`` / ``num_seqs`` / ``recurrent_state_indices``). The
  model scatters the stream into a ``[num_slots, T]`` grid and each
  attention layer ``lax.scan``s the grid columns through the single-token
  decode step (:class:`DeepseekV4PackedMeta`), so prefill, chunked prefill,
  and decode all resume from per-slot positions. The XLA scan path is the
  correctness reference; a fused ragged kernel is a planned TPU follow-up.
"""

import math
import os
import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx
from ejkernel.modules import (  # pyright: ignore[reportMissingTypeStubs]
    compressed_window_attention,
    compressed_window_decode,
)
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array, Bool, Float, Int
from spectrax import apply_logical_sharding, common_types, nn

from easydel.caching import CompressedWindowCache, CompressedWindowCacheConfig, CompressedWindowCacheView
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers import (
    BaseMoeModule,
    ColumnParallelLinear,
    ColumnParallelMoELinear,
    Embed,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RMSNorm,
    RowParallelLinear,
    RowParallelMoELinear,
    compute_basic_inv_frequencies,
    compute_yarn_inv_frequencies,
    dense_gate_up_layout,
    gated_mlp_forward,
    split_fused_gate_up_projection,
)
from easydel.modules._base import BaseCausalLMModule

from .deepseek_v4_configuration import DeepseekV4Config

_MASK_MIN = float(jnp.finfo(jnp.float32).min)

# Platform for the fused compressed-window decode kernel. Default "xla" is the
# GSPMD-partitionable reference: it lowers correctly under any mesh (single or
# multi device) and is the numerically-exact contract. Set
# EASYDEL_COMPRESSED_WINDOW_DECODE_PLATFORM=pallas to route the decode attend
# through the Pallas TPU kernel; under tensor parallelism a Mosaic custom call
# cannot be auto-partitioned, so the pallas path is wrapped in shard_map (heads
# sharded on "tp", shared KV replicated). "auto" -> ejkernel platform=None.
_DECODE_KERNEL_PLATFORM = os.getenv("EASYDEL_COMPRESSED_WINDOW_DECODE_PLATFORM", "xla")
_DECODE_KERNEL_PLATFORM = None if _DECODE_KERNEL_PLATFORM == "auto" else _DECODE_KERNEL_PLATFORM
# Mesh axis the query/output heads are sharded over (EasyDeL's 6-axis
# convention: ("pp","dp","fsdp","ep","tp","sp")); the shared KV head is replicated.
_DECODE_HEAD_AXIS = "tp"

# Platform for the fused compressed-window TRAINING / PREFILL kernel (the
# full-sequence, differentiable sibling of the decode kernel). DEFAULT (env
# unset -> "auto"): the stateless / prefill forward routes through the fused
# ejkernel op, which resolves to the tiled-flash Pallas TPU kernel on TPU
# (``make_async_copy`` KV streaming + online softmax + sliding-window band-skip,
# a measured ~1.35-2.71x forward speedup that scales with sequence length) and
# to the differentiable XLA reference on CPU/GPU (identical f32 math). The
# forward band-skip is a parity-clean win; the backward is a dense-matmul
# recompute (fast, exact-gradient), so training is speed-neutral and inference
# prefill is faster. Set the env var to "xla" to force the dense reference (A/B),
# "pallas" to force Pallas, or "off"/"none"/"disabled" to fall back to the inline
# dense ``_attend`` (no fused op at all). Under tensor parallelism the Mosaic
# custom call cannot be auto-partitioned, so the Pallas path is wrapped in
# shard_map (heads on "tp", shared KV replicated); the XLA path is left to GSPMD.
_TRAIN_KERNEL_ENV = os.getenv("EASYDEL_COMPRESSED_WINDOW_ATTENTION_PLATFORM", "auto").lower()
_TRAIN_KERNEL_ENABLED = _TRAIN_KERNEL_ENV not in ("off", "none", "disabled")
_TRAIN_KERNEL_PLATFORM = None if _TRAIN_KERNEL_ENV in ("auto", "off", "none", "disabled") else _TRAIN_KERNEL_ENV


def _unweighted_rms_norm(x: Array, eps: float) -> Array:
    """Unweighted RMSNorm: ``x * rsqrt(mean(x^2) + eps)`` with an fp32 moment.

    Mirrors HF ``DeepseekV4UnweightedRMSNorm``: the second moment and rsqrt run
    in float32, the scale is cast back to ``x.dtype`` before multiplying.

    Args:
        x: Input array; normalization runs over the last axis.
        eps: Variance epsilon.

    Returns:
        Array of the same shape and dtype as ``x``.
    """
    scale = jax.lax.rsqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True) + eps)
    return x * scale.astype(x.dtype)


def _rope_inv_freq(config: DeepseekV4Config, label: str) -> tuple[Array, float]:
    """Compute half-size interleaved-RoPE inverse frequencies for a rope label.

    V4 pairs consecutive channels, so only ``rope_dim // 2`` unique theta
    entries exist — exactly the ``arange(0, dim, 2)`` layout the shared
    rotary compute fns produce, so this only maps V4's per-label rope params
    onto :func:`~easydel.layers.compute_basic_inv_frequencies` /
    :func:`~easydel.layers.compute_yarn_inv_frequencies`;
    :func:`_apply_interleaved_rope` duplicates cos/sin with a
    ``repeat_interleave(2)`` next to the rotation. Supports the ``default``
    rope and ``yarn`` (V4 forces ``attention_factor = 1.0`` for the compress
    rope at config time).

    Args:
        config: Model configuration.
        label: Rope-type label, ``"main"`` or ``"compress"``.

    Returns:
        Tuple of (float32 inverse frequencies of shape ``[rope_dim // 2]``,
        attention scaling factor).

    Raises:
        NotImplementedError: For unsupported ``rope_type`` values.
    """
    params = config.v4_rope_params(label)
    base = float(params["rope_theta"])
    partial_rotary_factor = float(params.get("partial_rotary_factor", 1.0))
    dim = int(config.head_dim * partial_rotary_factor)
    rope_type = params.get("rope_type", params.get("type", "default"))

    if rope_type in ("default", None):
        return compute_basic_inv_frequencies(base, dim), 1.0
    if rope_type != "yarn":
        raise NotImplementedError(f"DeepSeek-V4 rope_type {rope_type!r} is not supported (label={label!r}).")

    # transformers `_compute_yarn_parameters` field defaulting, specialised to
    # V4 (no mscale/mscale_all_dim keys; `factor` falls back to the max/original
    # position ratio).
    factor = params.get("factor")
    original_max = params.get("original_max_position_embeddings", config.max_position_embeddings)
    if factor is None:
        factor = config.max_position_embeddings / original_max
    attention_factor = params.get("attention_factor")
    if attention_factor is None:
        attention_factor = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0

    inv_freq = compute_yarn_inv_frequencies(
        base=base,
        rotary_dim=dim,
        beta_fast=params.get("beta_fast") or 32,
        beta_slow=params.get("beta_slow") or 1,
        max_position_embeddings=original_max,
        scaling_factor=factor,
        extrapolation_factor=1.0,
        truncate=params.get("truncate", True),
    )
    return inv_freq, float(attention_factor)


def _rope_cos_sin(
    config: DeepseekV4Config,
    label: str,
    position_ids: Array,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    """Compute half-size cos/sin tables for the given positions and rope label.

    Args:
        config: Model configuration.
        label: Rope-type label (``"main"`` or ``"compress"``).
        position_ids: Integer positions of shape ``[batch, seq]``.
        dtype: Output dtype (the reference casts cos/sin to the activation
            dtype; the rotation itself upcasts back to fp32).

    Returns:
        Tuple ``(cos, sin)`` of shape ``[batch, seq, rope_dim // 2]``.
    """
    inv_freq, attention_scaling = _rope_inv_freq(config, label)
    inv = jnp.asarray(inv_freq, dtype=jnp.float32)
    freqs = position_ids.astype(jnp.float32)[..., None] * inv[None, None, :]
    cos = jnp.cos(freqs) * attention_scaling
    sin = jnp.sin(freqs) * attention_scaling
    return cos.astype(dtype), sin.astype(dtype)


def _apply_interleaved_rope(x: Array, cos: Array, sin: Array, unsqueeze_dim: int = 1) -> Array:
    """Apply V4's interleaved RoPE to the *trailing* rope slice of ``x``.

    ``cos``/``sin`` come in half-sized (one entry per interleaved pair); they
    are expanded with ``repeat_interleave(2)`` and the last ``2 * cos.shape[-1]``
    channels of ``x`` are rotated in fp32 with the interleaved (GLM-style)
    ``rotate_half``. Leading "nope" channels pass through untouched. Pass
    ``-sin`` for the inverse rotation applied to the attention output.

    Args:
        x: Input of shape ``[..., seq, head_dim]`` with a broadcastable head
            axis at ``unsqueeze_dim``.
        cos: Half-size cosines ``[batch, seq, rope_dim // 2]``.
        sin: Half-size sines ``[batch, seq, rope_dim // 2]``.
        unsqueeze_dim: Axis at which to insert the head-broadcast dim into
            cos/sin (1 for ``[B, H, S, D]`` inputs, 2 for ``[B, S, H, D]``).

    Returns:
        Array of the same shape and dtype as ``x``.
    """
    cos = jnp.expand_dims(jnp.repeat(cos, 2, axis=-1), unsqueeze_dim).astype(jnp.float32)
    sin = jnp.expand_dims(jnp.repeat(sin, 2, axis=-1), unsqueeze_dim).astype(jnp.float32)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rope_f = rope.astype(jnp.float32)
    x1 = rope_f[..., 0::2]
    x2 = rope_f[..., 1::2]
    rotate_half = jnp.stack((-x2, x1), axis=-1).reshape(rope_f.shape)
    rotated = (rope_f * cos + rotate_half * sin).astype(x.dtype)
    return jnp.concatenate([nope, rotated], axis=-1)


def _entry_causal_bias(position_ids: Array, n_entries: int, rate: int) -> Array:
    """Build the additive causal bias over compressed entries for a query block.

    Query at position ``t`` may see entry ``w`` iff ``w < (t + 1) // rate``.

    Args:
        position_ids: Query positions ``[batch, seq]``.
        n_entries: Number of compressed-entry slots on the KV axis.
        rate: Compression rate.

    Returns:
        Additive fp32 bias ``[batch, 1, seq, n_entries]``.
    """
    entry_indices = jnp.arange(n_entries)
    causal_threshold = (position_ids + 1) // rate
    visible = entry_indices[None, None, :] < causal_threshold[..., None]
    return jnp.where(visible, 0.0, _MASK_MIN)[:, None, :, :].astype(jnp.float32)


def _decode_entry_visibility(position_ids: Array, cache_position: Array, n_entries: int, rate: int) -> Array:
    """Visibility of padded compressed-entry slots during a decode step.

    Entry ``w`` is visible iff it is causally reachable
    (``w < (position + 1) // rate``) AND it has actually been emitted
    (``w < (cache_position + 1) // rate``, counting the entry possibly
    emitted by the current token's window closing).

    Args:
        position_ids: Query positions ``[batch, 1]``.
        cache_position: Per-row stream positions ``[batch]`` of the current
            token.
        n_entries: Static padded entry capacity.
        rate: Compression rate.

    Returns:
        Boolean visibility ``[batch, 1, n_entries]``.
    """
    entry_indices = jnp.arange(n_entries)
    causal_threshold = (position_ids + 1) // rate  # [B, 1]
    emitted = (cache_position.astype(jnp.int32) + 1) // rate  # [B]
    visible = entry_indices[None, None, :] < causal_threshold[..., None]
    return visible & (entry_indices[None, None, :] < emitted[:, None, None])


def _compressor_decode_step(
    *,
    kv_t: Array,
    gate_t: Array,
    buffer_kv: Array,
    buffer_gate: Array,
    entries: Array,
    position_bias: Array,
    kv_norm: "DeepseekV4RMSNorm",
    cache_position: Array,
    rate: int,
    head_dim: int,
    config: "DeepseekV4Config",
    overlap_kv: Array | None = None,
    overlap_gate: Array | None = None,
    valid: Array | None = None,
) -> tuple[Array, Array, Array, Array | None, Array | None]:
    """Advance one compressor stream by a single token per row (fixed shapes).

    Each row's current token (stream position ``p = cache_position[b]``) is
    written to buffer slot ``p % rate``. When that fills the window
    (``(p + 1) % rate == 0``) one compressed entry is emitted for window
    ``w = p // rate`` — softmax-gated pooling over the window slots (plus the
    Ca overlap slots for two-series streams), weighted RMSNorm, and
    interleaved "compress"-RoPE at the deterministic position ``w * rate`` —
    and written to ``entries[b, w]``. All arithmetic runs unconditionally on
    the fixed-shape buffers; the emission is selected with ``jnp.where`` so
    the step stays jit/while_loop/scan safe. Rows advance independently.

    Args:
        kv_t: Raw kv projection of the current token ``[batch, 1, feat]``.
        gate_t: Raw gate projection ``[batch, 1, feat]`` (position bias NOT
            applied).
        buffer_kv: Open-window kv buffer ``[batch, rate, feat]``.
        buffer_gate: Open-window gate buffer ``[batch, rate, feat]``.
        entries: Padded emitted entries ``[batch, n_slots, dim]``.
        position_bias: Learned within-window bias ``[rate, feat]``.
        kv_norm: Weighted RMSNorm applied to the pooled entry.
        cache_position: Per-row int32 stream positions ``[batch]`` of the
            current token.
        rate: Compression rate (window width).
        head_dim: Output entry dimension (Ca/Cb series width for two-series
            streams; equals ``feat`` for single-series streams).
        config: Model config (for the "compress" rotary tables).
        overlap_kv: Two-series only — previous window's Ca kv slice
            ``[batch, rate, head_dim]``.
        overlap_gate: Two-series only — previous window's biased Ca gate
            slice in fp32 ``[batch, rate, head_dim]``.
        valid: Optional per-row bool mask ``[batch]``; rows with ``False``
            leave all state untouched (packed serving pads rows).

    Returns:
        Tuple ``(entries, buffer_kv, buffer_gate, overlap_kv, overlap_gate)``
        of updated state (overlaps are ``None`` for single-series streams).
    """
    batch = kv_t.shape[0]
    rows = jnp.arange(batch)
    p = cache_position.astype(jnp.int32)  # [B]
    fill = jnp.mod(p, rate)  # [B]
    new_kv_row = kv_t[:, 0].astype(buffer_kv.dtype)  # [B, feat]
    new_gate_row = gate_t[:, 0].astype(buffer_gate.dtype)
    if valid is not None:
        new_kv_row = jnp.where(valid[:, None], new_kv_row, buffer_kv[rows, fill])
        new_gate_row = jnp.where(valid[:, None], new_gate_row, buffer_gate[rows, fill])
    buffer_kv = buffer_kv.at[rows, fill].set(new_kv_row)
    buffer_gate = buffer_gate.at[rows, fill].set(new_gate_row)
    closed = fill == (rate - 1)  # [B]
    if valid is not None:
        closed = closed & valid
    window_index = p // rate  # [B]

    biased_gate = buffer_gate.astype(jnp.float32) + position_bias.astype(jnp.float32)  # [B, rate, feat]
    if overlap_kv is None:
        slot_kv = buffer_kv
        slot_gate = biased_gate
        new_overlap_kv = None
        new_overlap_gate = None
    else:
        slot_kv = jnp.concatenate([overlap_kv.astype(buffer_kv.dtype), buffer_kv[..., head_dim:]], axis=1)
        slot_gate = jnp.concatenate([overlap_gate.astype(jnp.float32), biased_gate[..., head_dim:]], axis=1)
        new_overlap_kv = jnp.where(closed[:, None, None], buffer_kv[..., :head_dim].astype(overlap_kv.dtype), overlap_kv)
        new_overlap_gate = jnp.where(closed[:, None, None], biased_gate[..., :head_dim], overlap_gate)

    weights = jax.nn.softmax(slot_gate, axis=1).astype(slot_kv.dtype)
    candidate = kv_norm(jnp.sum(slot_kv * weights, axis=1))  # [B, dim]
    rope_positions = (window_index * rate).astype(jnp.int32)[:, None]  # [B, 1]
    cos, sin = _rope_cos_sin(config, "compress", rope_positions, candidate.dtype)
    candidate = _apply_interleaved_rope(candidate[:, None, None, :], cos, sin, unsqueeze_dim=1)[:, 0, 0]

    n_slots = entries.shape[1]
    slot = jnp.clip(window_index, 0, n_slots - 1)  # [B]
    current = entries[rows, slot]  # [B, dim]
    selected = jnp.where(closed[:, None], candidate.astype(entries.dtype), current)
    entries = entries.at[rows, slot].set(selected)
    return entries, buffer_kv, buffer_gate, new_overlap_kv, new_overlap_gate


class DeepseekV4RMSNorm(RMSNorm):
    """Weighted RMSNorm used by DeepSeek-V4 (identical to the shared RMSNorm)."""

    ...


class DeepseekV4HyperConnection(spx.Module):
    """Manifold-Constrained Hyper-Connection (mHC) mixing module.

    Owns the learned ``fn`` / ``base`` / ``scale`` parameters that map the
    flattened ``hc_mult`` residual streams to three outputs (all fp32):

    - ``pre`` (``sigmoid + hc_eps``): stream-collapse weights producing the
      sub-layer input.
    - ``post`` (``2 * sigmoid``): placement weights of the sub-layer output
      back onto the streams.
    - ``comb``: an ``hc x hc`` stream mixer, softmax-initialised then
      projected onto doubly-stochastic matrices with ``hc_sinkhorn_iters``
      alternating column/row normalizations (applied *transposed* by the
      caller).
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the hyper-connection module.

        Args:
            config: Model configuration (reads ``hc_mult``,
                ``hc_sinkhorn_iters``, ``hc_eps``, ``hidden_size``,
                ``rms_norm_eps``, ``initializer_range``).
            dtype: Activation dtype (mHC math itself runs in fp32).
            param_dtype: Parameter storage dtype.
            rngs: Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = ArrayParam.bound(
            shape=(mix, self.hc_mult * config.hidden_size),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.initializer_range},
            key=rngs.param,
        )
        self.base = ArrayParam.bound(
            shape=(mix,),
            dtype=param_dtype,
            init_method="zeros",
            key=rngs.param,
        )
        self.scale = ArrayParam.bound(
            shape=(3,),
            dtype=param_dtype,
            init_method="ones",
            key=rngs.param,
        )

    def forward(self, hidden_streams: Float[Array, "batch seq hc hidden"]) -> tuple[Array, Array, Array]:
        """Compute ``(post, comb, collapsed)`` from the mHC mapping.

        Args:
            hidden_streams: Residual streams of shape ``[B, S, hc, D]``.

        Returns:
            Tuple of ``post`` (``[B, S, hc]``, fp32), ``comb``
            (``[B, S, hc, hc]``, fp32, Sinkhorn-projected), and ``collapsed``
            (``[B, S, D]``, input dtype) — the pre-weighted stream sum fed to
            the sub-layer.
        """
        hc = self.hc_mult
        eps = self.hc_eps
        batch, seq = hidden_streams.shape[:2]
        flat = hidden_streams.reshape(batch, seq, -1).astype(jnp.float32)
        flat = _unweighted_rms_norm(flat, self.rms_norm_eps)
        fn = self.fn.value.astype(jnp.float32)
        mix_logits = flat @ fn.T
        pre_w = mix_logits[..., :hc]
        post_w = mix_logits[..., hc : 2 * hc]
        comb_w = mix_logits[..., 2 * hc :]
        base = self.base.value.astype(jnp.float32)
        pre_b, post_b, comb_b = base[:hc], base[hc : 2 * hc], base[2 * hc :]
        scale = self.scale.value.astype(jnp.float32)

        pre = jax.nn.sigmoid(pre_w * scale[0] + pre_b) + eps
        post = 2.0 * jax.nn.sigmoid(post_w * scale[1] + post_b)
        comb_logits = comb_w.reshape(batch, seq, hc, hc) * scale[2] + comb_b.reshape(hc, hc)
        comb = jax.nn.softmax(comb_logits, axis=-1) + eps
        # Sinkhorn-Knopp: alternate column/row normalization onto the
        # doubly-stochastic manifold (fixed static iteration count).
        comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (jnp.sum(comb, axis=-1, keepdims=True) + eps)
            comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps)

        collapsed = jnp.sum(pre[..., None] * hidden_streams.astype(jnp.float32), axis=2)
        return post, comb, collapsed.astype(hidden_streams.dtype)


class DeepseekV4HyperHead(spx.Module):
    """Final hyper-connection stream collapse applied before the final norm."""

    def __init__(
        self,
        config: DeepseekV4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the hyper-head.

        Args:
            config: Model configuration.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            rngs: Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_fn = ArrayParam.bound(
            shape=(self.hc_mult, self.hc_mult * config.hidden_size),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.initializer_range},
            key=rngs.param,
        )
        self.hc_base = ArrayParam.bound(
            shape=(self.hc_mult,),
            dtype=param_dtype,
            init_method="zeros",
            key=rngs.param,
        )
        self.hc_scale = ArrayParam.bound(
            shape=(1,),
            dtype=param_dtype,
            init_method="ones",
            key=rngs.param,
        )

    def forward(self, x: Float[Array, "batch seq hc hidden"]) -> Float[Array, "batch seq hidden"]:
        """Collapse the ``hc_mult`` streams to a single hidden sequence.

        Args:
            x: Residual streams ``[B, S, hc, D]``.

        Returns:
            Collapsed hidden states ``[B, S, D]`` in the input dtype.
        """
        batch, seq = x.shape[:2]
        flat = _unweighted_rms_norm(x.reshape(batch, seq, -1).astype(jnp.float32), self.rms_norm_eps)
        mixes = flat @ self.hc_fn.value.astype(jnp.float32).T
        pre = jax.nn.sigmoid(mixes * self.hc_scale.value.astype(jnp.float32) + self.hc_base.value.astype(jnp.float32))
        pre = pre + self.hc_eps
        return jnp.sum(pre[..., None] * x.astype(jnp.float32), axis=2).astype(x.dtype)


class DeepseekV4GroupedLinear(spx.Module):
    """Block-diagonal grouped linear (``o_a_proj`` of the grouped output projection).

    The HF weight has shape ``[o_groups * o_lora_rank, in_per_group]`` and is
    consumed group-blockwise; after EasyDeL's standard 2D transpose on
    conversion the kernel is ``[in_per_group, o_groups * o_lora_rank]`` and
    group ``g`` owns columns ``[g * rank, (g + 1) * rank)``.
    """

    def __init__(
        self,
        in_features_per_group: int,
        out_features_per_group: int,
        n_groups: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        kernel_init: tp.Callable | None = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the grouped linear.

        Args:
            in_features_per_group: Per-group input width
                (``num_heads * head_dim / o_groups``).
            out_features_per_group: Per-group output width (``o_lora_rank``).
            n_groups: Number of head groups.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            kernel_init: Weight initializer; defaults to LeCun normal.
            rngs: Random number generators.
        """
        self.in_features_per_group = in_features_per_group
        self.out_features_per_group = out_features_per_group
        self.n_groups = n_groups
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        if kernel_init is None:
            kernel_init = jax.nn.initializers.lecun_normal()
        self.weight = spx.Parameter(
            kernel_init(rngs.param, (in_features_per_group, n_groups * out_features_per_group), param_dtype),
        )

    def forward(self, x: Float[Array, "batch seq groups in_per_group"]) -> Float[Array, "batch seq groups rank"]:
        """Apply the per-group projection.

        Args:
            x: Grouped input ``[B, S, o_groups, in_per_group]``.

        Returns:
            Grouped output ``[B, S, o_groups, o_lora_rank]``.
        """
        kernel = self.weight.value.astype(self.dtype if self.dtype is not None else x.dtype)
        kernel = kernel.reshape(self.in_features_per_group, self.n_groups, self.out_features_per_group)
        return jnp.einsum("bsgi,igr->bsgr", x.astype(kernel.dtype), kernel, precision=self.precision)


class DeepseekV4HCACompressor(spx.Module):
    """Heavily Compressed Attention compressor (stateless mode).

    Compresses every ``compress_rate`` (m' = 128) source tokens into a single
    compressed KV entry via a per-window softmax-gated sum
    (``gate + position_bias`` softmaxed over the window dim in fp32), followed
    by a weighted RMSNorm and interleaved RoPE ("compress" base) at the
    window-start absolute position. Only complete windows are compressed; the
    remainder is discarded (stateless semantics).
    """

    rope_layer_type = "compress"

    def __init__(
        self,
        config: DeepseekV4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the HCA compressor.

        Args:
            config: Model configuration.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.head_dim = config.head_dim
        linear_kwargs = dict(
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.kv_proj = ColumnParallelLinear(config.hidden_size, self.head_dim, **linear_kwargs)
        self.gate_proj = ColumnParallelLinear(config.hidden_size, self.head_dim, **linear_kwargs)
        self.position_bias = ArrayParam.bound(
            shape=(self.compress_rate, self.head_dim),
            dtype=param_dtype,
            init_method="zeros",
            key=rngs.param,
        )
        self.kv_norm = DeepseekV4RMSNorm(
            dim=self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        q_residual: Float[Array, "batch seq q_lora"],
        position_ids: Int[Array, "batch seq"],
    ) -> tuple[Array | None, Array | None]:
        """Produce compressed KV entries and the per-query block bias.

        Args:
            hidden_states: Attention-input hidden states ``[B, S, D_model]``.
            q_residual: Normed query LoRA residual (unused by HCA; kept for a
                uniform compressor interface).
            position_ids: Query positions ``[B, S]``.

        Returns:
            Tuple ``(compressed_kv, block_bias)`` where ``compressed_kv`` is
            ``[B, T, head_dim]`` (``T = S // compress_rate``) and
            ``block_bias`` is an additive fp32 mask ``[B, 1, S, T]``; both are
            ``None`` when no complete window fits.
        """
        del q_residual
        _batch, seq_len, _ = hidden_states.shape
        rate = self.compress_rate
        n_windows = seq_len // rate
        if n_windows == 0:
            return None, None
        usable = n_windows * rate
        kv = self.kv_proj(hidden_states)[:, :usable]
        gate = self.gate_proj(hidden_states)[:, :usable]
        compressed = self._compress_windows(kv, gate)
        block_bias = _entry_causal_bias(position_ids, n_windows, rate)
        return compressed, block_bias

    def _compress_windows(self, kv: Array, gate: Array) -> Array:
        """Compress complete windows of raw projections into post-RoPE entries.

        Args:
            kv: Raw kv projections ``[batch, n_windows * rate, head_dim]``.
            gate: Raw gate projections of the same shape.

        Returns:
            Compressed entries ``[batch, n_windows, head_dim]`` (post-RoPE at
            positions ``w * rate``).
        """
        batch = kv.shape[0]
        rate = self.compress_rate
        n_windows = kv.shape[1] // rate
        chunk_kv = kv.reshape(batch, n_windows, rate, -1)
        chunk_gate = gate.astype(jnp.float32).reshape(batch, n_windows, rate, -1)
        chunk_gate = chunk_gate + self.position_bias.value.astype(jnp.float32)
        weights = jax.nn.softmax(chunk_gate, axis=2).astype(chunk_kv.dtype)
        compressed = self.kv_norm(jnp.sum(chunk_kv * weights, axis=2))
        positions = jnp.broadcast_to((jnp.arange(n_windows) * rate)[None, :], (batch, n_windows))
        cos, sin = _rope_cos_sin(self.config, self.rope_layer_type, positions, compressed.dtype)
        return _apply_interleaved_rope(compressed[:, None], cos, sin, unsqueeze_dim=1)[:, 0]

    def cached_forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        q_residual: Float[Array, "batch seq q_lora"],
        position_ids: Int[Array, "batch seq"],
        cache_view: CompressedWindowCacheView,
        valid: Array | None = None,
    ) -> tuple[Array | None, Array | None, CompressedWindowCacheView]:
        """Run the HCA compressor with cache state (prefill or decode).

        Args:
            hidden_states: Attention-input hidden states ``[B, S, D_model]``.
            q_residual: Unused by HCA (uniform compressor interface).
            position_ids: Query positions ``[B, S]``.
            cache_view: This layer's cache view.
            valid: Optional per-row bool mask ``[B]`` for decode steps; rows
                with ``False`` leave state untouched.

        Returns:
            Tuple ``(compressed_kv, block_bias, updated_view)``:
            ``compressed_kv`` is ``[B, T, head_dim]`` on prefill (unpadded,
            ``T = S // rate``; ``None`` when no complete window fits) and the
            padded ``[B, n_slots, head_dim]`` entry array on decode.
        """
        del q_residual
        seq_len = hidden_states.shape[1]
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        if seq_len > 1:
            return self._cached_prefill(kv, gate, position_ids, cache_view)
        return self._cached_decode(kv, gate, position_ids, cache_view, valid=valid)

    def _cached_prefill(
        self,
        kv: Array,
        gate: Array,
        position_ids: Array,
        cache_view: CompressedWindowCacheView,
    ) -> tuple[Array | None, Array | None, CompressedWindowCacheView]:
        """Prefill-from-empty: stateless window compression plus state writes."""
        seq_len = kv.shape[1]
        rate = self.compress_rate
        n_windows = min(seq_len // rate, cache_view.num_entry_slots)
        usable = n_windows * rate
        compressed = None
        block_bias = None
        if n_windows > 0:
            compressed = self._compress_windows(kv[:, :usable], gate[:, :usable])
            block_bias = _entry_causal_bias(position_ids, n_windows, rate)
            cache_view = cache_view.replace(
                compressor_entries=cache_view.compressor_entries.at[:, :n_windows].set(
                    compressed.astype(cache_view.compressor_entries.dtype)
                ),
            )
        remainder = seq_len - usable
        if remainder > 0:
            cache_view = cache_view.replace(
                compressor_buffer_kv=cache_view.compressor_buffer_kv.at[:, :remainder].set(
                    kv[:, usable:].astype(cache_view.compressor_buffer_kv.dtype)
                ),
                compressor_buffer_gate=cache_view.compressor_buffer_gate.at[:, :remainder].set(
                    gate[:, usable:].astype(cache_view.compressor_buffer_gate.dtype)
                ),
            )
        return compressed, block_bias, cache_view

    def _cached_decode(
        self,
        kv: Array,
        gate: Array,
        position_ids: Array,
        cache_view: CompressedWindowCacheView,
        valid: Array | None = None,
    ) -> tuple[Array | None, Array | None, CompressedWindowCacheView]:
        """Single-token step: fill the window buffer, emit on window close."""
        rate = self.compress_rate
        n_slots = cache_view.num_entry_slots
        if n_slots == 0:
            return None, None, cache_view
        entries, buffer_kv, buffer_gate, _, _ = _compressor_decode_step(
            kv_t=kv,
            gate_t=gate,
            buffer_kv=cache_view.compressor_buffer_kv,
            buffer_gate=cache_view.compressor_buffer_gate,
            entries=cache_view.compressor_entries,
            position_bias=self.position_bias.value,
            kv_norm=self.kv_norm,
            cache_position=cache_view.cache_position,
            rate=rate,
            head_dim=self.head_dim,
            config=self.config,
            valid=valid,
        )
        cache_view = cache_view.replace(
            compressor_buffer_kv=buffer_kv,
            compressor_buffer_gate=buffer_gate,
            compressor_entries=entries,
        )
        visible = _decode_entry_visibility(position_ids, cache_view.cache_position, n_slots, rate)
        block_bias = jnp.where(visible, 0.0, _MASK_MIN)[:, None, :, :].astype(jnp.float32)
        return entries, block_bias, cache_view


class DeepseekV4IndexerScorer(spx.Module):
    """Lightning-indexer scoring head: ``sum_h w_{t,h} * relu(q_{t,h} . K^IComp_s)``."""

    def __init__(
        self,
        config: DeepseekV4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the scorer.

        Args:
            config: Model configuration.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators.
        """
        self.softmax_scale = config.index_head_dim**-0.5
        self.weights_scaling = config.index_n_heads**-0.5
        self.precision = precision
        self.weights_proj = ColumnParallelLinear(
            config.hidden_size,
            config.index_n_heads,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )

    def forward(self, q: Array, compressed_kv: Array, hidden_states: Array) -> Array:
        """Score queries against compressed indexer keys (fp32).

        Args:
            q: Indexer queries ``[B, S, H_i, D_i]``.
            compressed_kv: Compressed indexer keys ``[B, T, D_i]``.
            hidden_states: Hidden states ``[B, S, D_model]`` for the per-head
                weighting.

        Returns:
            Per-query index scores ``[B, S, T]`` (fp32).
        """
        scores = jnp.einsum(
            "bshd,btd->bsht",
            q.astype(jnp.float32),
            compressed_kv.astype(jnp.float32),
            precision=self.precision,
        )
        scores = jax.nn.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).astype(jnp.float32) * self.weights_scaling
        return jnp.sum(scores * weights[..., None], axis=2)


class DeepseekV4Indexer(spx.Module):
    """Lightning Indexer: picks the top-``index_topk`` compressed entries per query.

    Runs its own scaled-down two-series (Ca/Cb) compressor at
    ``index_head_dim`` over the same windows as the outer CSA compressor,
    scores queries (projected from the shared ``q_residual``) against the
    compressed keys, and returns the per-query top-k entry indices with
    ``-1`` marking invalid (causally unreachable) picks.
    """

    rope_layer_type = "compress"

    def __init__(
        self,
        config: DeepseekV4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the indexer.

        Args:
            config: Model configuration.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        linear_kwargs = dict(
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.kv_proj = ColumnParallelLinear(config.hidden_size, 2 * self.head_dim, **linear_kwargs)
        self.gate_proj = ColumnParallelLinear(config.hidden_size, 2 * self.head_dim, **linear_kwargs)
        self.position_bias = ArrayParam.bound(
            shape=(self.compress_rate, 2 * self.head_dim),
            dtype=param_dtype,
            init_method="zeros",
            key=rngs.param,
        )
        self.kv_norm = DeepseekV4RMSNorm(
            dim=self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.q_b_proj = ColumnParallelLinear(config.q_lora_rank, self.num_heads * self.head_dim, **linear_kwargs)
        self.scorer = DeepseekV4IndexerScorer(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        q_residual: Float[Array, "batch seq q_lora"],
        position_ids: Int[Array, "batch seq"],
    ) -> Array | None:
        """Compute per-query top-k compressed-entry indices.

        Args:
            hidden_states: Attention-input hidden states ``[B, S, D_model]``.
            q_residual: Normed query LoRA residual shared with core attention.
            position_ids: Query positions ``[B, S]``.

        Returns:
            Int array ``[B, S, k]`` of selected entry indices with ``-1`` for
            invalid picks, or ``None`` when no complete window fits.
        """
        _batch, seq_len, _ = hidden_states.shape
        rate = self.compress_rate
        n_windows = seq_len // rate
        if n_windows == 0:
            return None
        usable = n_windows * rate
        kv = self.kv_proj(hidden_states)[:, :usable]
        gate = self.gate_proj(hidden_states)[:, :usable]
        compressed, _, _ = self._compress_windows(kv, gate)
        index_scores = self._score_queries(hidden_states, q_residual, position_ids, compressed)
        return self._select_causal_top_k(index_scores, position_ids, n_windows)

    def _compress_windows(self, kv: Array, gate: Array) -> tuple[Array, Array, Array]:
        """Two-series window compression of raw projections into post-RoPE keys.

        Args:
            kv: Raw kv projections ``[batch, n_windows * rate, 2 * head_dim]``.
            gate: Raw gate projections of the same shape.

        Returns:
            Tuple of post-RoPE compressed keys ``[batch, n_windows, head_dim]``,
            the last window's Ca kv slice, and its biased fp32 Ca gate slice.
        """
        batch = kv.shape[0]
        rate = self.compress_rate
        n_windows = kv.shape[1] // rate
        compressed, ca_kv, ca_gate = _two_series_compress(
            kv,
            gate,
            self.position_bias.value,
            self.kv_norm,
            batch,
            n_windows,
            rate,
            self.head_dim,
        )
        positions = jnp.broadcast_to((jnp.arange(n_windows) * rate)[None, :], (batch, n_windows))
        cos, sin = _rope_cos_sin(self.config, self.rope_layer_type, positions, compressed.dtype)
        compressed = _apply_interleaved_rope(compressed[:, None], cos, sin, unsqueeze_dim=1)[:, 0]
        return compressed, ca_kv, ca_gate

    def _score_queries(
        self,
        hidden_states: Array,
        q_residual: Array,
        position_ids: Array,
        compressed: Array,
    ) -> Array:
        """Project + rotate indexer queries and score them against ``compressed``.

        Args:
            hidden_states: Hidden states ``[B, S, D_model]``.
            q_residual: Normed query LoRA residual ``[B, S, q_lora]``.
            position_ids: Query positions ``[B, S]``.
            compressed: Compressed indexer keys ``[B, T, index_head_dim]``.

        Returns:
            Index scores ``[B, S, T]`` (fp32).
        """
        batch, seq_len, _ = hidden_states.shape
        cos_q, sin_q = _rope_cos_sin(self.config, self.rope_layer_type, position_ids, hidden_states.dtype)
        q = self.q_b_proj(q_residual).reshape(batch, seq_len, self.num_heads, self.head_dim)
        q = _apply_interleaved_rope(q.transpose(0, 2, 1, 3), cos_q, sin_q, unsqueeze_dim=1).transpose(0, 2, 1, 3)
        return self.scorer(q, compressed, hidden_states)

    def _select_causal_top_k(self, index_scores: Array, position_ids: Array, compressed_len: int) -> Array:
        """Causally mask scores and keep the per-query top-k entry indices.

        Args:
            index_scores: Scores ``[B, S, T]`` (fp32).
            position_ids: Query positions ``[B, S]``.
            compressed_len: Number of scored entries ``T``.

        Returns:
            Int indices ``[B, S, k]`` with ``-1`` marking invalid picks.
        """
        rate = self.compress_rate
        top_k = min(self.index_topk, compressed_len)
        causal_threshold = (position_ids + 1) // rate  # [B, S]
        entry_indices = jnp.arange(compressed_len)
        future_mask = entry_indices[None, None, :] >= causal_threshold[..., None]
        index_scores = jnp.where(future_mask, -jnp.inf, index_scores)
        top_k_indices = jax.lax.top_k(index_scores, top_k)[1]
        invalid = top_k_indices >= causal_threshold[..., None]
        return jnp.where(invalid, -1, top_k_indices)

    def cached_forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        q_residual: Float[Array, "batch seq q_lora"],
        position_ids: Int[Array, "batch seq"],
        cache_view: CompressedWindowCacheView,
        valid: Array | None = None,
    ) -> tuple[Array | None, CompressedWindowCacheView]:
        """Run the indexer with cache state (prefill or decode).

        Args:
            hidden_states: Attention-input hidden states ``[B, S, D_model]``.
            q_residual: Normed query LoRA residual shared with core attention.
            position_ids: Query positions ``[B, S]``.
            cache_view: This layer's cache view (indexer stream fields).
            valid: Optional per-row bool mask ``[B]`` for decode steps; rows
                with ``False`` leave state untouched.

        Returns:
            Tuple ``(top_k_indices, updated_view)``. On prefill the indices
            cover the ``S // rate`` fresh entries (``None`` if no window
            fits); on decode they cover the padded entry axis. ``-1`` marks
            invalid picks in both cases.
        """
        seq_len = hidden_states.shape[1]
        if seq_len > 1:
            return self._cached_prefill(hidden_states, q_residual, position_ids, cache_view)
        return self._cached_decode(hidden_states, q_residual, position_ids, cache_view, valid=valid)

    def _cached_prefill(
        self,
        hidden_states: Array,
        q_residual: Array,
        position_ids: Array,
        cache_view: CompressedWindowCacheView,
    ) -> tuple[Array | None, CompressedWindowCacheView]:
        """Prefill-from-empty: stateless indexer math plus state writes."""
        seq_len = hidden_states.shape[1]
        rate = self.compress_rate
        n_windows = min(seq_len // rate, cache_view.num_entry_slots)
        usable = n_windows * rate
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        top_k_indices = None
        if n_windows > 0:
            compressed, ca_kv, ca_gate = self._compress_windows(kv[:, :usable], gate[:, :usable])
            cache_view = cache_view.replace(
                indexer_entries=cache_view.indexer_entries.at[:, :n_windows].set(
                    compressed.astype(cache_view.indexer_entries.dtype)
                ),
                indexer_overlap_kv=ca_kv.astype(cache_view.indexer_overlap_kv.dtype),
                indexer_overlap_gate=ca_gate.astype(jnp.float32),
            )
            index_scores = self._score_queries(hidden_states, q_residual, position_ids, compressed)
            top_k_indices = self._select_causal_top_k(index_scores, position_ids, n_windows)
        remainder = seq_len - usable
        if remainder > 0:
            cache_view = cache_view.replace(
                indexer_buffer_kv=cache_view.indexer_buffer_kv.at[:, :remainder].set(
                    kv[:, usable:].astype(cache_view.indexer_buffer_kv.dtype)
                ),
                indexer_buffer_gate=cache_view.indexer_buffer_gate.at[:, :remainder].set(
                    gate[:, usable:].astype(cache_view.indexer_buffer_gate.dtype)
                ),
            )
        return top_k_indices, cache_view

    def _cached_decode(
        self,
        hidden_states: Array,
        q_residual: Array,
        position_ids: Array,
        cache_view: CompressedWindowCacheView,
        valid: Array | None = None,
    ) -> tuple[Array | None, CompressedWindowCacheView]:
        """Single-token step over the padded indexer-entry axis."""
        rate = self.compress_rate
        n_slots = cache_view.num_entry_slots
        if n_slots == 0:
            return None, cache_view
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        entries, buffer_kv, buffer_gate, overlap_kv, overlap_gate = _compressor_decode_step(
            kv_t=kv,
            gate_t=gate,
            buffer_kv=cache_view.indexer_buffer_kv,
            buffer_gate=cache_view.indexer_buffer_gate,
            entries=cache_view.indexer_entries,
            position_bias=self.position_bias.value,
            kv_norm=self.kv_norm,
            cache_position=cache_view.cache_position,
            rate=rate,
            head_dim=self.head_dim,
            config=self.config,
            overlap_kv=cache_view.indexer_overlap_kv,
            overlap_gate=cache_view.indexer_overlap_gate,
            valid=valid,
        )
        cache_view = cache_view.replace(
            indexer_buffer_kv=buffer_kv,
            indexer_buffer_gate=buffer_gate,
            indexer_entries=entries,
            indexer_overlap_kv=overlap_kv,
            indexer_overlap_gate=overlap_gate,
        )
        index_scores = self._score_queries(hidden_states, q_residual, position_ids, entries)  # [B, 1, n_slots]
        visible = _decode_entry_visibility(position_ids, cache_view.cache_position, n_slots, rate)
        index_scores = jnp.where(visible, index_scores, -jnp.inf)
        top_k = min(self.index_topk, n_slots)
        top_k_indices = jax.lax.top_k(index_scores, top_k)[1]
        picked_visible = jnp.take_along_axis(visible, top_k_indices, axis=-1)
        return jnp.where(picked_visible, top_k_indices, -1), cache_view


def _two_series_compress(
    kv: Array,
    gate: Array,
    position_bias: Array,
    kv_norm: DeepseekV4RMSNorm,
    batch: int,
    n_windows: int,
    rate: int,
    head_dim: int,
) -> tuple[Array, Array, Array]:
    """Shared CSA two-series (Ca/Cb) window compression.

    Each source token contributes two independent series stored in one
    ``2 * head_dim`` projection: Ca (``[..., :head_dim]``) feeds the *next*
    window's entry, Cb (``[..., head_dim:]``) the *current* window's entry.
    Entry ``w`` is the fp32 softmax-gated combination over ``2 * rate`` slots
    of window ``w - 1``'s Ca and window ``w``'s Cb; window 0's Ca half stays
    zero-kv / ``-inf``-gate (prefill from an empty stream).

    Args:
        kv: Projected values ``[B, usable, 2 * head_dim]``.
        gate: Projected gates ``[B, usable, 2 * head_dim]``.
        position_bias: Learned bias ``[rate, 2 * head_dim]`` (added in fp32).
        kv_norm: Weighted RMSNorm applied to each compressed entry.
        batch: Batch size.
        n_windows: Number of complete windows.
        rate: Compression rate (window width).
        head_dim: Per-series feature width.

    Returns:
        Tuple of the compressed entries ``[B, n_windows, head_dim]``
        (pre-RoPE), the last window's Ca kv slice ``[B, rate, head_dim]``,
        and its biased Ca gate slice in fp32 ``[B, rate, head_dim]`` (the
        overlap state a cache must carry into the next window).
    """
    chunk_kv = kv.reshape(batch, n_windows, rate, -1)
    chunk_gate = gate.astype(jnp.float32).reshape(batch, n_windows, rate, -1) + position_bias.astype(jnp.float32)

    prev_ca_kv = jnp.concatenate(
        [jnp.zeros((batch, 1, rate, head_dim), dtype=chunk_kv.dtype), chunk_kv[:, :-1, :, :head_dim]],
        axis=1,
    )
    prev_ca_gate = jnp.concatenate(
        [jnp.full((batch, 1, rate, head_dim), -jnp.inf, dtype=jnp.float32), chunk_gate[:, :-1, :, :head_dim]],
        axis=1,
    )
    new_kv = jnp.concatenate([prev_ca_kv, chunk_kv[..., head_dim:]], axis=2)
    new_gate = jnp.concatenate([prev_ca_gate, chunk_gate[..., head_dim:]], axis=2)
    weights = jax.nn.softmax(new_gate, axis=2).astype(new_kv.dtype)
    compressed = kv_norm(jnp.sum(new_kv * weights, axis=2))
    return compressed, chunk_kv[:, -1, :, :head_dim], chunk_gate[:, -1, :, :head_dim]


class DeepseekV4CSACompressor(spx.Module):
    """Compressed Sparse Attention compressor (stateless mode).

    Runs the two-series (Ca/Cb) window compression at ``head_dim`` and a
    :class:`DeepseekV4Indexer` that gates per-query visibility of the
    compressed entries down to ``index_topk`` picks.
    """

    rope_layer_type = "compress"

    def __init__(
        self,
        config: DeepseekV4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the CSA compressor.

        Args:
            config: Model configuration.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.head_dim = config.head_dim
        linear_kwargs = dict(
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.kv_proj = ColumnParallelLinear(config.hidden_size, 2 * self.head_dim, **linear_kwargs)
        self.gate_proj = ColumnParallelLinear(config.hidden_size, 2 * self.head_dim, **linear_kwargs)
        self.position_bias = ArrayParam.bound(
            shape=(self.compress_rate, 2 * self.head_dim),
            dtype=param_dtype,
            init_method="zeros",
            key=rngs.param,
        )
        self.kv_norm = DeepseekV4RMSNorm(
            dim=self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.indexer = DeepseekV4Indexer(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        q_residual: Float[Array, "batch seq q_lora"],
        position_ids: Int[Array, "batch seq"],
    ) -> tuple[Array | None, Array | None]:
        """Produce compressed KV entries and the indexer-gated block bias.

        Args:
            hidden_states: Attention-input hidden states ``[B, S, D_model]``.
            q_residual: Normed query LoRA residual shared with core attention.
            position_ids: Query positions ``[B, S]``.

        Returns:
            Tuple ``(compressed_kv, block_bias)`` with ``compressed_kv`` of
            shape ``[B, T, head_dim]`` (``T = S // compress_rate``) and an
            additive fp32 ``block_bias`` ``[B, 1, S, T]`` opening only the
            indexer-selected, causally valid entries; both ``None`` when no
            complete window fits.
        """
        batch, seq_len, _ = hidden_states.shape
        rate = self.compress_rate
        n_windows = seq_len // rate
        if n_windows == 0:
            return None, None
        usable = n_windows * rate
        kv = self.kv_proj(hidden_states)[:, :usable]
        gate = self.gate_proj(hidden_states)[:, :usable]
        compressed, _, _ = self._compress_windows(kv, gate)
        top_k_indices = self.indexer(hidden_states, q_residual, position_ids)  # [B, S, k]
        block_bias = _indexer_opened_bias(top_k_indices, batch, seq_len, n_windows)
        return compressed, block_bias

    def _compress_windows(self, kv: Array, gate: Array) -> tuple[Array, Array, Array]:
        """Two-series window compression of raw projections into post-RoPE entries.

        Args:
            kv: Raw kv projections ``[batch, n_windows * rate, 2 * head_dim]``.
            gate: Raw gate projections of the same shape.

        Returns:
            Tuple of post-RoPE compressed entries ``[batch, n_windows,
            head_dim]``, the last window's Ca kv slice, and its biased fp32
            Ca gate slice.
        """
        batch = kv.shape[0]
        rate = self.compress_rate
        n_windows = kv.shape[1] // rate
        compressed, ca_kv, ca_gate = _two_series_compress(
            kv,
            gate,
            self.position_bias.value,
            self.kv_norm,
            batch,
            n_windows,
            rate,
            self.head_dim,
        )
        positions = jnp.broadcast_to((jnp.arange(n_windows) * rate)[None, :], (batch, n_windows))
        cos, sin = _rope_cos_sin(self.config, self.rope_layer_type, positions, compressed.dtype)
        compressed = _apply_interleaved_rope(compressed[:, None], cos, sin, unsqueeze_dim=1)[:, 0]
        return compressed, ca_kv, ca_gate

    def cached_forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        q_residual: Float[Array, "batch seq q_lora"],
        position_ids: Int[Array, "batch seq"],
        cache_view: CompressedWindowCacheView,
        valid: Array | None = None,
    ) -> tuple[Array | None, Array | None, CompressedWindowCacheView]:
        """Run the CSA compressor + indexer with cache state (prefill or decode).

        Args:
            hidden_states: Attention-input hidden states ``[B, S, D_model]``.
            q_residual: Normed query LoRA residual shared with core attention.
            position_ids: Query positions ``[B, S]``.
            cache_view: This layer's cache view.
            valid: Optional per-row bool mask ``[B]`` for decode steps; rows
                with ``False`` leave state untouched.

        Returns:
            Tuple ``(compressed_kv, block_bias, updated_view)``:
            ``compressed_kv`` is ``[B, T, head_dim]`` on prefill (unpadded;
            ``None`` when no complete window fits) and the padded
            ``[B, n_slots, head_dim]`` entry array on decode.
        """
        seq_len = hidden_states.shape[1]
        if seq_len > 1:
            return self._cached_prefill(hidden_states, q_residual, position_ids, cache_view)
        return self._cached_decode(hidden_states, q_residual, position_ids, cache_view, valid=valid)

    def _cached_prefill(
        self,
        hidden_states: Array,
        q_residual: Array,
        position_ids: Array,
        cache_view: CompressedWindowCacheView,
    ) -> tuple[Array | None, Array | None, CompressedWindowCacheView]:
        """Prefill-from-empty: stateless CSA math plus state writes."""
        batch, seq_len, _ = hidden_states.shape
        rate = self.compress_rate
        n_windows = min(seq_len // rate, cache_view.num_entry_slots)
        usable = n_windows * rate
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        compressed = None
        if n_windows > 0:
            compressed, ca_kv, ca_gate = self._compress_windows(kv[:, :usable], gate[:, :usable])
            cache_view = cache_view.replace(
                compressor_entries=cache_view.compressor_entries.at[:, :n_windows].set(
                    compressed.astype(cache_view.compressor_entries.dtype)
                ),
                compressor_overlap_kv=ca_kv.astype(cache_view.compressor_overlap_kv.dtype),
                compressor_overlap_gate=ca_gate.astype(jnp.float32),
            )
        remainder = seq_len - usable
        if remainder > 0:
            cache_view = cache_view.replace(
                compressor_buffer_kv=cache_view.compressor_buffer_kv.at[:, :remainder].set(
                    kv[:, usable:].astype(cache_view.compressor_buffer_kv.dtype)
                ),
                compressor_buffer_gate=cache_view.compressor_buffer_gate.at[:, :remainder].set(
                    gate[:, usable:].astype(cache_view.compressor_buffer_gate.dtype)
                ),
            )
        top_k_indices, cache_view = self.indexer.cached_forward(hidden_states, q_residual, position_ids, cache_view)
        block_bias = None
        if n_windows > 0:
            block_bias = _indexer_opened_bias(top_k_indices, batch, seq_len, n_windows)
        return compressed, block_bias, cache_view

    def _cached_decode(
        self,
        hidden_states: Array,
        q_residual: Array,
        position_ids: Array,
        cache_view: CompressedWindowCacheView,
        valid: Array | None = None,
    ) -> tuple[Array | None, Array | None, CompressedWindowCacheView]:
        """Single-token step: advance both compressor and indexer streams."""
        batch = hidden_states.shape[0]
        rate = self.compress_rate
        n_slots = cache_view.num_entry_slots
        if n_slots == 0:
            return None, None, cache_view
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        entries, buffer_kv, buffer_gate, overlap_kv, overlap_gate = _compressor_decode_step(
            kv_t=kv,
            gate_t=gate,
            buffer_kv=cache_view.compressor_buffer_kv,
            buffer_gate=cache_view.compressor_buffer_gate,
            entries=cache_view.compressor_entries,
            position_bias=self.position_bias.value,
            kv_norm=self.kv_norm,
            cache_position=cache_view.cache_position,
            rate=rate,
            head_dim=self.head_dim,
            config=self.config,
            overlap_kv=cache_view.compressor_overlap_kv,
            overlap_gate=cache_view.compressor_overlap_gate,
            valid=valid,
        )
        cache_view = cache_view.replace(
            compressor_buffer_kv=buffer_kv,
            compressor_buffer_gate=buffer_gate,
            compressor_entries=entries,
            compressor_overlap_kv=overlap_kv,
            compressor_overlap_gate=overlap_gate,
        )
        top_k_indices, cache_view = self.indexer.cached_forward(
            hidden_states, q_residual, position_ids, cache_view, valid=valid
        )
        block_bias = _indexer_opened_bias(top_k_indices, batch, 1, n_slots)
        return entries, block_bias, cache_view


def _indexer_opened_bias(top_k_indices: Array, batch: int, seq_len: int, compressed_len: int) -> Array:
    """Turn indexer picks into an additive bias over the compressed-entry axis.

    Args:
        top_k_indices: Selected entry indices ``[B, S, k]``; ``-1`` marks
            invalid picks (dropped).
        batch: Batch size.
        seq_len: Query length.
        compressed_len: Length of the compressed-entry axis being opened.

    Returns:
        Additive fp32 bias ``[B, 1, S, compressed_len]`` that is zero on the
        selected valid entries and ``_MASK_MIN`` elsewhere.
    """
    valid = top_k_indices >= 0
    safe_indices = jnp.where(valid, top_k_indices, compressed_len)
    opened = jnp.zeros((batch, seq_len, compressed_len + 1), dtype=bool)
    batch_idx = jnp.arange(batch)[:, None, None]
    seq_idx = jnp.arange(seq_len)[None, :, None]
    opened = opened.at[batch_idx, seq_idx, safe_indices].set(True)
    return jnp.where(opened[..., :compressed_len], 0.0, _MASK_MIN)[:, None, :, :].astype(jnp.float32)


_COMPRESSOR_CLASSES: dict[str, type | None] = {
    "sliding_attention": None,
    "compressed_sparse_attention": DeepseekV4CSACompressor,
    "heavily_compressed_attention": DeepseekV4HCACompressor,
}


class DeepseekV4PackedMeta(tp.NamedTuple):
    """Token-to-slot mapping for eSurge packed multi-request serving.

    The engine's flat ``[1, T]`` token stream concatenates each scheduled
    request's new tokens. This mapping scatters those tokens into a dense
    ``[num_slots, T]`` grid (one row per cache slot at its own stream
    position) that the attention layers scan column-by-column.

    Attributes:
        slot_of_token: ``[T]`` int32 cache-slot row per packed token;
            padded tokens carry ``num_slots`` (dropped on scatter).
        col_of_token: ``[T]`` int32 grid column per packed token (the
            token's index within its request's scheduled chunk).
        valid_grid: ``[num_slots, T]`` bool; ``True`` where a real token
            occupies the cell.
        positions_grid: ``[num_slots, T]`` int32 absolute stream positions
            (engine-provided; garbage at invalid cells).
    """

    slot_of_token: Array
    col_of_token: Array
    valid_grid: Array
    positions_grid: Array


def _build_packed_meta(
    query_start_loc: Array,
    num_seqs: Array,
    recurrent_state_indices: Array | None,
    position_ids: Array,
    num_slots: int,
) -> DeepseekV4PackedMeta:
    """Derive the packed token-to-slot grid mapping from ragged metadata.

    Args:
        query_start_loc: ``[R + 1]`` cumulative scheduled-token offsets per
            request row (padded with the total beyond the live rows).
        num_seqs: ``[1]`` int32 live request count.
        recurrent_state_indices: Optional physical cache-slot id per request
            row (``[R]`` or ``[1, R]``); ``None`` falls back to row identity.
        position_ids: Packed absolute positions ``[1, T]``.
        num_slots: Static number of cache slots (state rows).

    Returns:
        DeepseekV4PackedMeta: The scatter/gather mapping for this step.
    """
    qsl = query_start_loc.astype(jnp.int32).reshape(-1)
    num_rows = qsl.shape[0] - 1
    total_tokens = position_ids.shape[-1]
    token_idx = jnp.arange(total_tokens, dtype=jnp.int32)
    row_of_token = jnp.clip(jnp.searchsorted(qsl, token_idx, side="right") - 1, 0, num_rows - 1)
    live_rows = jnp.clip(num_seqs.reshape(-1)[0].astype(jnp.int32), 0, num_rows)
    total_real_tokens = jnp.take(qsl, live_rows)
    valid_token = token_idx < total_real_tokens
    if recurrent_state_indices is None:
        slots_by_row = jnp.arange(num_rows, dtype=jnp.int32)
    else:
        slots_by_row = recurrent_state_indices.astype(jnp.int32).reshape(-1)[:num_rows]
    slot_of_token = jnp.take(slots_by_row, row_of_token, mode="clip")
    slot_of_token = jnp.where(valid_token, slot_of_token, num_slots)
    col_of_token = token_idx - jnp.take(qsl, row_of_token)

    valid_grid = (
        jnp.zeros((num_slots, total_tokens), dtype=jnp.bool_)
        .at[slot_of_token, col_of_token]
        .set(valid_token, mode="drop")
    )
    positions_grid = (
        jnp.zeros((num_slots, total_tokens), dtype=jnp.int32)
        .at[slot_of_token, col_of_token]
        .set(position_ids.astype(jnp.int32).reshape(-1), mode="drop")
    )
    return DeepseekV4PackedMeta(
        slot_of_token=slot_of_token,
        col_of_token=col_of_token,
        valid_grid=valid_grid,
        positions_grid=positions_grid,
    )


class DeepseekV4Attention(spx.Module):
    """DeepSeek-V4 shared-KV MQA attention with sinks and compressed KV.

    Differences from classic attention (all mirrored from the HF reference):

    - Query path ``q_a_proj -> q_a_norm (weighted) -> q_b_proj -> q_b_norm``
      where ``q_b_norm`` is an *unweighted* per-head RMSNorm.
    - Single shared KV head: ``kv_proj`` projects to one ``head_dim`` vector
      per token; the same tensor is read as both key and value.
    - Partial interleaved RoPE on the trailing ``qk_rope_head_dim`` channels
      ("main" base on sliding layers, "compress" base on CSA/HCA layers).
    - Sliding-window causal masking of the token-KV part on ALL layers;
      CSA/HCA layers concatenate compressed entries after the token KVs with
      the compressor's additive ``block_bias``.
    - Per-head learnable sink logits appended as an extra softmax column
      whose probability mass is discarded.
    - Inverse rotation (``cos, -sin`` at the query positions) applied to the
      attention output's rope slice (``V == K`` carried RoPE).
    - Grouped block-diagonal output projection ``o_a_proj`` + ``o_b_proj``.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize V4 attention for one layer.

        Args:
            config: Model configuration.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators.
            layer_idx: Layer index (selects ``layer_types[layer_idx]``).
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.rope_layer_type = "main" if self.layer_type == "sliding_attention" else "compress"
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.scaling = self.head_dim**-0.5
        self.rms_norm_eps = config.rms_norm_eps

        linear_kwargs = dict(
            use_bias=config.attention_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.q_a_proj = ColumnParallelLinear(config.hidden_size, config.q_lora_rank, **linear_kwargs)
        self.q_a_norm = DeepseekV4RMSNorm(
            dim=config.q_lora_rank,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.q_b_proj = ColumnParallelLinear(config.q_lora_rank, self.num_heads * self.head_dim, **linear_kwargs)
        self.kv_proj = ColumnParallelLinear(config.hidden_size, self.head_dim, **linear_kwargs)
        self.kv_norm = DeepseekV4RMSNorm(
            dim=self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.o_b_proj = RowParallelLinear(config.o_groups * config.o_lora_rank, config.hidden_size, **linear_kwargs)
        self.sinks = ArrayParam.bound(
            shape=(self.num_heads,),
            dtype=param_dtype,
            init_method="zeros",
            key=rngs.param,
        )
        compressor_class = _COMPRESSOR_CLASSES[self.layer_type]
        if compressor_class is not None:
            self.compressor = compressor_class(
                config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.compressor = None

    def _attend(self, q: Array, kv: Array, attn_bias: Array) -> Array:
        """Sink-augmented shared-KV attention core.

        Args:
            q: Rotated queries ``[B, H, S, head_dim]``.
            kv: KV axis (token KVs and/or compressed entries)
                ``[B, T, head_dim]``.
            attn_bias: Additive fp32 bias ``[B, 1, S, T]`` (broadcast over
                heads).

        Returns:
            Attention output ``[B, H, S, head_dim]``.
        """
        batch, _, seq_len, _ = q.shape
        attn_logits = jnp.einsum("bhsd,btd->bhst", q, kv, precision=self.precision).astype(jnp.float32)
        attn_logits = attn_logits * self.scaling + attn_bias
        sinks = jnp.broadcast_to(
            self.sinks.value.astype(jnp.float32).reshape(1, self.num_heads, 1, 1),
            (batch, self.num_heads, seq_len, 1),
        )
        combined = jnp.concatenate([attn_logits, sinks], axis=-1)
        combined = combined - jnp.max(combined, axis=-1, keepdims=True)
        probs = jax.nn.softmax(combined, axis=-1)  # fp32 softmax (attn_softmax_dtype convention)
        scores = probs[..., :-1].astype(kv.dtype)  # drop the sink column
        return jnp.einsum("bhst,btd->bhsd", scores, kv, precision=self.precision)

    def _fused_attend(self, q: Array, kv: Array, attn_bias: Array) -> Array:
        """Sink-augmented shared-KV attention via the fused ejkernel op.

        Numerically identical to :meth:`_attend` (its XLA fallback is the exact
        reference math), but dispatches to the Pallas TPU kernel on device and
        collapses the per-column ``einsum -> sink softmax -> einsum`` sequence
        into a single fused custom call. Used on the decode path (single query
        per slot) where the KV axis is bounded and VMEM-resident.

        Args:
            q: Rotated queries ``[B, H, S, head_dim]`` (``S == 1`` on decode).
            kv: Shared KV axis ``[B, L, head_dim]`` (``K == V``): ring + entries.
            attn_bias: Additive fp32 bias ``[B, 1, S, L]`` (head-broadcast).

        Returns:
            Attention output ``[B, H, S, head_dim]``.
        """
        bias_bsl = attn_bias[:, 0]  # [B, S, L] (head axis is broadcast inside the kernel)
        sinks = self.sinks.value.astype(jnp.float32)  # [H]

        mesh = getattr(self.config, "mesh", None)
        jax_mesh = getattr(mesh, "jax_mesh", mesh)
        multi_device = jax_mesh is not None and getattr(jax_mesh, "size", 1) > 1
        # A Pallas (Mosaic) custom call cannot be auto-partitioned under a
        # multi-device mesh; wrap it in shard_map with the head-parallel specs.
        # The XLA reference is left to GSPMD (correct on any mesh).
        if _DECODE_KERNEL_PLATFORM != "xla" and multi_device and _DECODE_HEAD_AXIS in jax_mesh.axis_names:
            head = _DECODE_HEAD_AXIS
            return compressed_window_decode(
                q,
                kv,
                bias_bsl,
                sinks,
                softmax_scale=self.scaling,
                platform=_DECODE_KERNEL_PLATFORM,
                mesh=jax_mesh,
                in_specs=(
                    Ps(None, head, None, None),  # query: heads sharded
                    Ps(),  # kv: shared head replicated
                    Ps(),  # bias: replicated
                    Ps(head),  # sinks: per-head
                ),
                out_specs=Ps(None, head, None, None),
            )
        return compressed_window_decode(
            q,
            kv,
            bias_bsl,
            sinks,
            softmax_scale=self.scaling,
            platform=_DECODE_KERNEL_PLATFORM,
        )

    def _fused_attend_full(self, q: Array, kv: Array, attn_bias: Array) -> Array:
        """Sink-augmented shared-KV attention via the fused full-sequence op.

        The training / prefill sibling of :meth:`_fused_attend`: numerically
        identical to :meth:`_attend` (its XLA fallback is the exact reference
        math and is differentiable via XLA autodiff), but dispatches to the
        tiled-flash Pallas TPU kernel on device (``make_async_copy`` KV
        streaming + online softmax) when opted in. Used on the stateless
        forward and cached prefill, where the query length is ``> 1`` and the KV
        axis (token KVs + compressed entries) may be long.

        Gradients flow to ``q``, ``kv`` (both its key and value roles, since
        ``K == V``) and the per-head sinks; the additive ``bias`` is a
        stop-gradient structural / indexer-gated mask.

        Args:
            q: Rotated queries ``[B, H, S, head_dim]``.
            kv: Shared KV axis ``[B, T, head_dim]`` (``K == V``): token KVs
                concatenated with the compressed entries.
            attn_bias: Additive fp32 bias ``[B, 1, S, T]`` (head-broadcast).

        Returns:
            Attention output ``[B, H, S, head_dim]``.
        """
        bias_bsl = attn_bias[:, 0]  # [B, S, T] (head axis is broadcast inside the kernel)
        sinks = self.sinks.value.astype(jnp.float32)  # [H]
        # The token-KV region is the sequence (its length); anything beyond it on
        # the KV axis is the always-visible compressed entries. Passing the
        # sliding window + this split lets the Pallas kernel band-skip the
        # out-of-window token tiles (O(S*window)) and recompute its backward
        # through the matching windowed reference.
        seq_len = q.shape[2]
        window = self.sliding_window

        mesh = getattr(self.config, "mesh", None)
        jax_mesh = getattr(mesh, "jax_mesh", mesh)
        multi_device = jax_mesh is not None and getattr(jax_mesh, "size", 1) > 1
        # A Pallas (Mosaic) custom call cannot be auto-partitioned under a
        # multi-device mesh; wrap it in shard_map with the head-parallel specs.
        # Only when the op will actually resolve to Pallas (TPU + not forced xla);
        # the XLA path (CPU/GPU, or forced xla) is left to GSPMD on any mesh.
        want_pallas = _TRAIN_KERNEL_PLATFORM != "xla" and jax.default_backend() == "tpu"
        if want_pallas and multi_device and _DECODE_HEAD_AXIS in jax_mesh.axis_names:
            head = _DECODE_HEAD_AXIS
            return compressed_window_attention(
                q,
                kv,
                bias_bsl,
                sinks,
                softmax_scale=self.scaling,
                platform=_TRAIN_KERNEL_PLATFORM,
                window=window,
                token_kv_len=seq_len,
                mesh=jax_mesh,
                in_specs=(
                    Ps(None, head, None, None),  # query: heads sharded
                    Ps(),  # kv: shared head replicated
                    Ps(),  # bias: replicated
                    Ps(head),  # sinks: per-head
                ),
                out_specs=Ps(None, head, None, None),
            )
        return compressed_window_attention(
            q,
            kv,
            bias_bsl,
            sinks,
            softmax_scale=self.scaling,
            platform=_TRAIN_KERNEL_PLATFORM,
            window=window,
            token_kv_len=seq_len,
        )

    def _decode_attend(
        self,
        q: Array,
        kv: Array,
        hidden_states: Array,
        q_residual: Array,
        position_ids: Array,
        cache_view: CompressedWindowCacheView,
        valid: Array | None = None,
    ) -> tuple[Array, CompressedWindowCacheView]:
        """Single-token cached decode over the fixed-shape state (per-row).

        Every batch row advances its own stream by (at most) one token: the
        rotated token KV is written to its row's ring slot
        ``cache_position[b] % window``, the compressor streams advance, and
        the query attends over the ring plus the compressed entries.

        Args:
            q: Rotated queries ``[B, H, 1, head_dim]``.
            kv: Rotated token KV ``[B, 1, head_dim]``.
            hidden_states: Attention-input hidden states ``[B, 1, D_model]``
                (compressor projections are token-wise).
            q_residual: Normed query LoRA residual ``[B, 1, q_lora]``.
            position_ids: Query positions ``[B, 1]``.
            cache_view: This layer's cache state (per-row positions).
            valid: Optional per-row bool mask ``[B]``; rows with ``False``
                leave state untouched and produce ignorable outputs.

        Returns:
            Tuple of raw attention output ``[B, H, 1, head_dim]`` (before the
            inverse output rotation) and the updated cache view.
        """
        batch = q.shape[0]
        window = self.sliding_window
        stream_position = cache_view.cache_position.astype(jnp.int32)  # [B] pre-advance stream index
        rows = jnp.arange(batch)
        write_slot = jnp.mod(stream_position, window)  # [B]
        write_kv = kv[:, 0].astype(cache_view.window_kv.dtype)  # [B, head_dim]
        if valid is not None:
            write_kv = jnp.where(valid[:, None], write_kv, cache_view.window_kv[rows, write_slot])
        ring = cache_view.window_kv.at[rows, write_slot].set(write_kv)
        # Ring slot j holds absolute position p - ((p - j) mod W); slots
        # never written yet resolve to negative positions.
        slots = jnp.arange(window, dtype=jnp.int32)
        slot_positions = stream_position[:, None] - jnp.mod(stream_position[:, None] - slots[None, :], window)
        ring_bias = jnp.where(slot_positions >= 0, 0.0, _MASK_MIN).astype(jnp.float32)  # [B, W]
        attn_bias = ring_bias[:, None, None, :]  # [B, 1, 1, W]
        kv_axis = ring
        if self.compressor is not None:
            # The compressor reads cache_view.cache_position, so it must
            # run before the position advances below.
            compressed_kv, block_bias, cache_view = self.compressor.cached_forward(
                hidden_states, q_residual, position_ids, cache_view, valid=valid
            )
            if compressed_kv is not None:
                kv_axis = jnp.concatenate([ring, compressed_kv.astype(ring.dtype)], axis=1)
                attn_bias = jnp.concatenate([attn_bias, block_bias], axis=-1)
        advance = 1 if valid is None else valid.astype(jnp.int32)
        cache_view = cache_view.replace(window_kv=ring, cache_position=stream_position + advance)
        return self._fused_attend(q, kv_axis, attn_bias), cache_view

    def _packed_forward(
        self,
        hidden_states: Float[Array, "one tokens hidden"],
        position_embeddings: dict[str, tuple[Array, Array]],
        position_ids: Int[Array, "one tokens"],
        cache_view: CompressedWindowCacheView,
        packed_meta: "DeepseekV4PackedMeta",
    ) -> tuple[Array, CompressedWindowCacheView]:
        """Serve a packed multi-request token stream against per-slot state.

        eSurge hands the layer a flat ``[1, T]`` stream that concatenates
        each scheduled request's new tokens. The stream is scattered into a
        dense ``[num_slots, T]`` grid (one row per cache slot) and a
        ``lax.scan`` over the columns advances every slot by one token per
        column via :meth:`_decode_attend` — decode-only steps have a single
        column; prefill/chunked-prefill spans several. Padding cells carry
        ``valid=False`` and freeze their slot's state. Outputs are gathered
        back into the packed layout.

        Args:
            hidden_states: Packed hidden states ``[1, T, D_model]``.
            position_embeddings: Half-size rope tables at the packed
                positions ``[1, T, rope // 2]`` per rope type.
            position_ids: Packed absolute positions ``[1, T]``.
            cache_view: Slot-major cache state (``batch = num_slots``).
            packed_meta: Token-to-grid mapping (see
                :class:`DeepseekV4PackedMeta`).

        Returns:
            Tuple of the attention output ``[1, T, D_model]`` (garbage at
            padded token positions, which downstream never reads) and the
            updated cache view.
        """
        _, total_tokens, _ = hidden_states.shape
        num_slots = cache_view.cache_position.shape[0]
        cos, sin = position_embeddings[self.rope_layer_type]

        # Prefill-from-empty self-heal: a valid chunk whose first token sits
        # at absolute position 0 starts a brand-new stream, so its slot's
        # state is re-initialized before processing. This makes slot reuse
        # robust to any stale state (engine warmup runs, preemption
        # restarts) without relying on out-of-band clears.
        restart = packed_meta.valid_grid[:, 0] & (packed_meta.positions_grid[:, 0] == 0)
        cache_view = cache_view.reset_rows(restart)

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))  # [1, T, q_lora]
        q = self.q_b_proj(q_residual).reshape(1, total_tokens, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = _unweighted_rms_norm(q, self.rms_norm_eps)
        q = _apply_interleaved_rope(q, cos, sin, unsqueeze_dim=1)  # [1, H, T, head_dim]

        kv = self.kv_norm(self.kv_proj(hidden_states))[:, None, :, :]
        kv = _apply_interleaved_rope(kv, cos, sin, unsqueeze_dim=1)[:, 0]  # [1, T, head_dim]

        slot_tok = packed_meta.slot_of_token  # [T] (== num_slots for padding -> dropped)
        col_tok = packed_meta.col_of_token  # [T]

        def _to_grid(values: Array) -> Array:
            """Scatter per-token values ``[T, ...]`` into ``[num_slots, T, ...]``."""
            grid = jnp.zeros((num_slots, total_tokens, *values.shape[1:]), dtype=values.dtype)
            return grid.at[slot_tok, col_tok].set(values, mode="drop")

        hidden_grid = _to_grid(hidden_states[0])  # [S, T, D]
        qres_grid = _to_grid(q_residual[0])  # [S, T, q_lora]
        q_grid = _to_grid(q[0].transpose(1, 0, 2))  # [S, T, H, head_dim]
        kv_grid = _to_grid(kv[0])  # [S, T, head_dim]

        # eSurge shares ONE compiled packed path across prefill / chunked-prefill
        # / decode (no static decode signal). Only the grid columns that carry a
        # real token do any work: a pure decode step (each slot advances one
        # token) fills column 0 only, so scanning the full padded bucket width
        # ``T`` wasted ``T - 1`` columns of full compressor+attention compute per
        # step (~16x on the decode bucket). Iterate only the occupied columns via
        # a data-dependent ``while_loop`` (static shapes, one compile per bucket);
        # columns beyond the last real token stay zero and are never gathered.
        col_iota = jnp.arange(total_tokens, dtype=jnp.int32)
        column_has_token = jnp.any(packed_meta.valid_grid, axis=0)  # [T]
        num_used_cols = jnp.max(jnp.where(column_has_token, col_iota + 1, 0))  # 0..T

        attn_columns = jnp.zeros(
            (total_tokens, num_slots, self.num_heads, self.head_dim),
            dtype=cache_view.window_kv.dtype,
        )

        def _cond(state):
            """Continue while grid columns still carry unprocessed real tokens."""
            col, _view, _acc = state
            return col < num_used_cols

        def _body(state):
            """Advance every slot by one grid column (column ``col``)."""
            col, view, acc = state
            hidden_col = jax.lax.dynamic_index_in_dim(hidden_grid, col, axis=1, keepdims=False)  # [S, D]
            qres_col = jax.lax.dynamic_index_in_dim(qres_grid, col, axis=1, keepdims=False)  # [S, q_lora]
            q_col = jax.lax.dynamic_index_in_dim(q_grid, col, axis=1, keepdims=False)  # [S, H, head_dim]
            kv_col = jax.lax.dynamic_index_in_dim(kv_grid, col, axis=1, keepdims=False)  # [S, head_dim]
            pos_col = jax.lax.dynamic_index_in_dim(packed_meta.positions_grid, col, axis=1, keepdims=False)  # [S]
            valid_col = jax.lax.dynamic_index_in_dim(packed_meta.valid_grid, col, axis=1, keepdims=False)  # [S]
            attn_col, view = self._decode_attend(
                q=q_col[:, :, None, :],
                kv=kv_col[:, None, :],
                hidden_states=hidden_col[:, None, :],
                q_residual=qres_col[:, None, :],
                position_ids=pos_col[:, None],
                cache_view=view,
                valid=valid_col,
            )
            acc = jax.lax.dynamic_update_index_in_dim(acc, attn_col[:, :, 0, :].astype(acc.dtype), col, axis=0)
            return col + 1, view, acc

        _, cache_view, attn_columns = jax.lax.while_loop(
            _cond, _body, (jnp.int32(0), cache_view, attn_columns)
        )  # attn_columns: [T, S, H, head_dim]

        safe_slot = jnp.minimum(slot_tok, num_slots - 1)
        attn_packed = attn_columns[col_tok, safe_slot][None, ...]  # [1, T, H, head_dim]
        attn_output = _apply_interleaved_rope(attn_packed, cos, -sin, unsqueeze_dim=2)

        grouped = attn_output.reshape(1, total_tokens, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).reshape(1, total_tokens, -1)
        return checkpoint_name(self.o_b_proj(grouped), "attn_output"), cache_view

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        position_embeddings: dict[str, tuple[Array, Array]],
        position_ids: Int[Array, "batch seq"],
        token_bias: Float[Array, "batch 1 seq seq"] | None,
        cache_view: CompressedWindowCacheView | None = None,
        packed_meta: "DeepseekV4PackedMeta | None" = None,
    ) -> tuple[Array, CompressedWindowCacheView | None]:
        """Run V4 attention (stateless, cached prefill, decode, or packed).

        The regime is selected statically: no ``cache_view`` means the
        stateless forward; with a view and ``packed_meta`` the eSurge packed
        multi-request path runs; otherwise ``S > 1`` runs a prefill from an
        empty stream (stateless math plus state writes) and ``S == 1`` runs
        a single decode step over the fixed-shape cache state.

        Args:
            hidden_states: Normed, stream-collapsed hidden states
                ``[B, S, D_model]``.
            position_embeddings: ``{"main": (cos, sin), "compress": (cos, sin)}``
                half-size rope tables at the query positions.
            position_ids: Query positions ``[B, S]``.
            token_bias: Additive fp32 sliding-window causal (+ padding) mask
                over the token KVs ``[B, 1, S, S]``; ``None`` on cached
                decode steps (the ring mask is derived from the cache).
            cache_view: Optional per-layer cache state.
            packed_meta: Optional packed-serving token-to-slot mapping; when
                given (with ``cache_view``), the flat ``[1, T]`` stream is
                served against per-slot state.

        Returns:
            Tuple of the attention output ``[B, S, D_model]`` and the
            updated cache view (``None`` when running stateless).
        """
        if cache_view is not None and packed_meta is not None:
            return self._packed_forward(hidden_states, position_embeddings, position_ids, cache_view, packed_meta)

        batch, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = _unweighted_rms_norm(q, self.rms_norm_eps)  # HF q_b_norm (unweighted, per head)
        q = _apply_interleaved_rope(q, cos, sin, unsqueeze_dim=1)

        kv = self.kv_norm(self.kv_proj(hidden_states))[:, None, :, :]  # [B, 1, S, head_dim]
        kv = _apply_interleaved_rope(kv, cos, sin, unsqueeze_dim=1)[:, 0]  # [B, S, head_dim]

        if cache_view is None or seq_len > 1:
            # Stateless forward and cached prefill share the exact same
            # attention math; prefill additionally writes the cache state.
            if cache_view is not None:
                window = self.sliding_window
                ring = cache_view.window_kv
                if seq_len >= window:
                    slot_idx = jnp.arange(seq_len - window, seq_len) % window
                    ring = ring.at[:, slot_idx].set(kv[:, seq_len - window :].astype(ring.dtype))
                else:
                    ring = ring.at[:, :seq_len].set(kv.astype(ring.dtype))
                cache_view = cache_view.replace(
                    window_kv=ring,
                    cache_position=cache_view.cache_position + seq_len,
                )
            attn_bias = token_bias
            if self.compressor is not None:
                if cache_view is None:
                    compressed_kv, block_bias = self.compressor(hidden_states, q_residual, position_ids)
                else:
                    compressed_kv, block_bias, cache_view = self.compressor.cached_forward(
                        hidden_states, q_residual, position_ids, cache_view
                    )
                if compressed_kv is not None:
                    kv = jnp.concatenate([kv, compressed_kv.astype(kv.dtype)], axis=1)
                    attn_bias = jnp.concatenate([attn_bias, block_bias], axis=-1)
            # Default: inline dense ``_attend`` (graph unchanged). Opt in to the
            # fused differentiable full-sequence kernel via the env flag.
            if _TRAIN_KERNEL_ENABLED:
                attn_output = self._fused_attend_full(q, kv, attn_bias)
            else:
                attn_output = self._attend(q, kv, attn_bias)
        else:
            attn_output, cache_view = self._decode_attend(
                q=q,
                kv=kv,
                hidden_states=hidden_states,
                q_residual=q_residual,
                position_ids=position_ids,
                cache_view=cache_view,
            )

        # V == K carried RoPE: undo it on the output's rope slice with the
        # conjugate rotation at the query positions.
        attn_output = _apply_interleaved_rope(attn_output.transpose(0, 2, 1, 3), cos, -sin, unsqueeze_dim=2)

        grouped = attn_output.reshape(batch, seq_len, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).reshape(batch, seq_len, -1)
        return checkpoint_name(self.o_b_proj(grouped), "attn_output"), cache_view


class DeepseekV4MLP(spx.Module):
    """Clamped SwiGLU MLP used as the DeepSeek-V4 shared expert.

    ``down_proj(silu(clamp(gate, max=limit)) * clamp(up, -limit, limit))``
    with ``limit = config.swiglu_limit`` and width
    ``config.moe_intermediate_size``.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the shared-expert MLP.

        Args:
            config: Model configuration.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.limit = config.swiglu_limit
        # ejkernel's ``clamped_swiglu`` combine hard-codes silu, so declaring it
        # for a non-silu ``hidden_act`` would silently compute silu here while
        # the experts (which clamp around ``ACT2FN[hidden_act]``) keep the
        # configured activation. Declare only when the two agree; every other
        # activation keeps the explicit composition in ``forward``.
        if str(config.hidden_act).lower() in ("silu", "swish"):
            self.fused_act_name = "clamped_swiglu"
            self.fused_act_params = (float(config.swiglu_limit),)
        self.gate_up_proj = ColumnParallelLinear(
            self.hidden_size,
            (self.intermediate_size, self.intermediate_size),
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.mlp_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            layout=dense_gate_up_layout(self.intermediate_size),
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.mlp_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    @property
    def reform_param(self):
        """Checkpoint reform rules fusing HF ``gate_proj``/``up_proj`` into ``gate_up_proj``."""
        return self.gate_up_proj.build_reform_param("gate_up_proj", config=self.config)

    def forward(self, hidden_states: Float[Array, "batch seq hidden"]) -> Float[Array, "batch seq hidden"]:
        """Clamped-SwiGLU MLP.

        With a silu ``hidden_act`` this delegates to
        :func:`easydel.layers.gated_mlp_forward`, whose declared
        ``clamped_swiglu`` combine runs on both the fused and fallback paths.
        Any other activation keeps the explicit composition, which clamps
        around ``ACT2FN[hidden_act]`` exactly like the routed experts do.

        Args:
            hidden_states: Input ``[B, S, D_model]``.

        Returns:
            Output ``[B, S, D_model]``.
        """
        if getattr(self, "fused_act_name", None) is not None:
            return gated_mlp_forward(self, hidden_states)
        gate_up = checkpoint_name(self.gate_up_proj(hidden_states), "mlp_gate_up")
        gate, up = split_fused_gate_up_projection(gate_up, config=self.config)
        gate = jnp.clip(gate, max=self.limit)
        up = jnp.clip(up, min=-self.limit, max=self.limit)
        return checkpoint_name(self.down_proj(self.act_fn(gate) * up), "mlp_down")


class DeepseekV4Experts(spx.Module):
    """Stacked routed experts with the clamped SwiGLU activation.

    The HF checkpoint ships fused 3-D tensors ``experts.gate_up_proj``
    ``[E, 2M, H]`` (gate rows first, ``chunk(2)`` semantics) and
    ``experts.down_proj`` ``[E, H, M]`` in ``F.linear`` orientation; the
    ``reform_param`` rules split/transpose them into EasyDeL's
    ``[E, H, M]`` / ``[E, M, H]`` MoE-parallel layouts.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the stacked expert projections.

        Args:
            config: Model configuration.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.num_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.limit = config.swiglu_limit
        moe_kwargs = dict(
            rngs=rngs,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            partition_manager=config.runtime_sharding_resolver,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.gate_proj = ColumnParallelMoELinear(
            num_experts=self.num_experts,
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            **moe_kwargs,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=self.num_experts,
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            **moe_kwargs,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=self.num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            **moe_kwargs,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    @property
    def reform_param(self):
        """Split rules for the HF fused/stacked expert tensors."""
        intermediate_size = self.intermediate_size
        return {
            "gate_up_proj$": {
                "splits": [
                    {
                        "name": "gate_proj.weight",
                        "spliter": lambda x: x[:, :intermediate_size, :].permute(0, 2, 1),
                    },
                    {
                        "name": "up_proj.weight",
                        "spliter": lambda x: x[:, intermediate_size:, :].permute(0, 2, 1),
                    },
                ],
                "inverse_spliter": lambda torch, gate, up: torch.cat(
                    (gate.permute(0, 2, 1), up.permute(0, 2, 1)), dim=1
                ),
            },
            "down_proj$": {
                "splits": [
                    {"name": "down_proj.weight", "spliter": lambda x: x.permute(0, 2, 1)},
                ],
                "inverse_spliter": lambda x: x.permute(0, 2, 1),
            },
        }

    def forward(
        self,
        hidden_states: Array,
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Apply the clamped SwiGLU expert FFN over expert-grouped tokens.

        Args:
            hidden_states: Expert-grouped token representations.
            group_sizes: Per-expert group sizes.
            sorted_experts: Optional sorted expert indices.

        Returns:
            Expert outputs in the grouped layout.
        """
        gate = self.gate_proj(hidden_states, group_sizes, sorted_experts)
        up = self.up_proj(hidden_states, group_sizes, sorted_experts)
        gate = jnp.clip(gate, max=self.limit)
        up = jnp.clip(up, min=-self.limit, max=self.limit)
        return self.down_proj(self.act_fn(gate) * up, group_sizes, sorted_experts)


class DeepseekV4TopKRouter(spx.Module):
    """Learned V4 router: ``sqrtsoftplus`` scores + ``e_score_correction_bias`` selection.

    ``forward`` returns the raw per-expert *scores* (not logits); expert
    selection adds the persistent correction bias, while routing weights are
    gathered from the raw scores (see the select hook installed by
    :class:`DeepseekV4SparseMoeBlock`).
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the learned router.

        Args:
            config: Model configuration.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.scoring_func = config.scoring_func
        self.weight = ArrayParam.bound(
            shape=(self.hidden_dim, self.num_experts),
            dtype=param_dtype,
            init_method="normal",
            init_kwargs={"stddev": config.initializer_range},
            key=rngs.param,
        )
        self.e_score_correction_bias = ArrayParam.bound(
            shape=(self.num_experts,),
            dtype=param_dtype,
            init_method="zeros",
            key=rngs.param,
        )

    def _score(self, logits: Array) -> Array:
        """Apply the configured scoring activation (fp32)."""
        if self.scoring_func == "sqrtsoftplus":
            return jnp.sqrt(jax.nn.softplus(logits))
        if self.scoring_func == "softmax":
            return jax.nn.softmax(logits, axis=-1)
        if self.scoring_func == "sigmoid":
            return jax.nn.sigmoid(logits)
        raise NotImplementedError(f"Unsupported DeepSeek-V4 scoring_func: {self.scoring_func!r}")

    def forward(self, hidden_states: Array) -> Array:
        """Compute per-expert routing scores.

        Args:
            hidden_states: Flat token representations ``[tokens, D_model]``.

        Returns:
            Raw routing scores ``[tokens, n_routed_experts]`` (fp32).
        """
        logits = jnp.dot(
            hidden_states.astype(jnp.float32),
            self.weight.value.astype(jnp.float32),
            precision=self.precision,
        )
        return self._score(logits)


class DeepseekV4HashRouter(DeepseekV4TopKRouter):
    """Hash router used on ``mlp_layer_types == "hash_moe"`` layers.

    Expert *indices* come from the frozen ``tid2eid[input_ids]`` lookup table;
    the learned gate still provides the scores that weight the selected
    experts. ``tid2eid`` is stored in ``param_dtype`` (checkpoint values are
    small integer expert ids, exactly representable) and cast to int32 at
    lookup time.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the hash router.

        Args:
            config: Model configuration.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators.
        """
        super().__init__(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        self.tid2eid = ArrayParam.bound(
            shape=(config.vocab_size, self.top_k),
            dtype=param_dtype,
            init_method="zeros",
            key=rngs.param,
        )


class DeepseekV4SparseMoeBlock(BaseMoeModule):
    """DeepSeek-V4 MoE block: routed experts + clamped shared expert.

    Selection: top-k on ``scores + e_score_correction_bias`` (learned layers)
    or a frozen ``tid2eid[input_ids]`` lookup (hash layers). Weights: raw
    scores gathered at the selected indices, normalized
    (``/(sum + 1e-20)``), scaled by ``routed_scaling_factor``. Output:
    ``routed + shared_experts(residual)``.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the MoE block for one layer.

        Args:
            config: Model configuration.
            layer_idx: Layer index (selects ``mlp_layer_types[layer_idx]``).
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators.
        """
        super().__init__(
            config=config,
            n_routed_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            lbl_coef=None,
            rzl_coef=None,
            routing_strategy=MoeRoutingStrategy.TOP_K,
            load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
        )
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx
        self.is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"
        self.routed_scaling_factor = config.routed_scaling_factor
        # Pin an identity refine hook: the V4 select hook already normalizes and
        # applies `routed_scaling_factor`, and `moe_call`'s strategy defaults
        # would otherwise install a re-normalizing refine_weights_hook on
        # `self.moe_hooks` after the first call (dividing the scaling away on
        # every subsequent forward).
        self.moe_hooks = self.moe_hooks.replace(refine_weights_hook=lambda weights: weights)
        router_class = DeepseekV4HashRouter if self.is_hash else DeepseekV4TopKRouter
        self.gate = router_class(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.experts = DeepseekV4Experts(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.shared_experts = DeepseekV4MLP(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def _normalize_and_scale(self, weights: Array) -> Array:
        """Normalize gathered raw scores and apply ``routed_scaling_factor``."""
        weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-20)
        return weights * self.routed_scaling_factor

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        input_ids: Int[Array, "batch seq"] | None = None,
    ) -> tuple[Array, Array]:
        """Route tokens through experts and add the shared expert.

        Args:
            hidden_states: Post-norm hidden states ``[B, S, D_model]``.
            input_ids: Token ids ``[B, S]``; required on hash-MoE layers.

        Returns:
            Tuple of the MoE output ``[B, S, D_model]`` and the router scores
            (``[B * S, n_routed_experts]``).

        Raises:
            ValueError: If ``input_ids`` is missing on a hash-MoE layer.
        """
        num_experts = self.config.n_routed_experts
        gate_layer = self.gate
        if self.is_hash:
            if input_ids is None:
                raise ValueError(
                    "DeepSeek-V4 hash-MoE layers route via tid2eid[input_ids]; "
                    "`input_ids` must be provided (inputs_embeds-only forward is unsupported on hash layers)."
                )
            table = self.gate.tid2eid.value
            hash_indices = jnp.take(table, input_ids.reshape(-1).astype(jnp.int32), axis=0)

            # The select hook runs inside the fused-MoE shard_map on
            # token-sharded slices, so the frozen indices must ride along the
            # token axis. Append them as extra float columns on the gate
            # output (the expert axis is never sharded there); expert ids are
            # small integers, exactly representable in fp32.
            def _hash_gate(flat_hidden: Array) -> Array:
                scores = self.gate(flat_hidden)
                return jnp.concatenate([scores, hash_indices.astype(scores.dtype)], axis=-1)

            gate_layer = _hash_gate

            def _select_hash(augmented_scores: Array, pre_bias_logits: Array | None, k: int) -> tuple[Array, Array]:
                del pre_bias_logits, k
                scores = augmented_scores[:, :num_experts].astype(jnp.float32)
                indices = augmented_scores[:, num_experts:].astype(jnp.int32)
                weights = jnp.take_along_axis(scores, indices, axis=-1)
                return self._normalize_and_scale(weights), indices

            select_hook = _select_hash
        else:
            correction_bias = self.gate.e_score_correction_bias.value.astype(jnp.float32)

            def _select_topk(gate_scores: Array, pre_bias_logits: Array | None, k: int) -> tuple[Array, Array]:
                del pre_bias_logits
                scores = gate_scores.astype(jnp.float32)
                _, indices = jax.lax.top_k(scores + correction_bias, k)
                weights = jnp.take_along_axis(scores, indices, axis=-1)
                return self._normalize_and_scale(weights), indices

            select_hook = _select_topk

        hooks = self.moe_hooks.replace(
            normalize_gate_logits=lambda logits: logits,
            select_hook=select_hook,
        )
        limit = self.config.swiglu_limit

        def ffn_activation(gate: Array, up: Array) -> Array:
            """Clamped SwiGLU: ``silu(clamp(gate, max=limit)) * clamp(up, +-limit)``."""
            gate = jnp.clip(gate, max=limit)
            up = jnp.clip(up, min=-limit, max=limit)
            return self.experts.act_fn(gate) * up

        routed, router_scores = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=gate_layer,
            expert_layer=self.experts,
            wi_kernel=self.experts.gate_proj.kernel_view(),
            wu_kernel=self.experts.up_proj.kernel_view(),
            wd_kernel=self.experts.down_proj.kernel_view(),
            act_fn=self.experts.act_fn,
            ffn_activation=ffn_activation,
            hooks=hooks,
            layer_idx=self.layer_idx,
        )
        if self.is_hash:
            router_scores = router_scores[:, :num_experts]  # drop the carried hash-index columns
        out = routed + self.shared_experts(hidden_states)
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_scores, "moe_router_logits")


class DeepseekV4DecoderLayer(spx.Module):
    """DeepSeek-V4 decoder block with hyper-connection residual streams.

    The residual is a stack of ``hc_mult`` streams ``[B, S, hc, D]``. Each
    sub-layer site (attention, MoE) collapses the streams via its
    :class:`DeepseekV4HyperConnection` (``pre`` weights), runs the sub-layer on
    the normed collapsed input, then updates the streams as
    ``post[..., None] * out[..., None, :] + comb^T @ streams`` (``comb``
    applied transposed; both cast back to the stream dtype first).
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize one decoder layer.

        Args:
            config: Model configuration.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators.
            layer_idx: Layer index.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]

        self.self_attn = DeepseekV4Attention(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.mlp = DeepseekV4SparseMoeBlock(
            config,
            layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.input_layernorm = DeepseekV4RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = DeepseekV4RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.attn_hc = DeepseekV4HyperConnection(config, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.ffn_hc = DeepseekV4HyperConnection(config, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def forward(
        self,
        hidden_streams: Float[Array, "batch seq hc hidden"],
        position_embeddings: dict[str, tuple[Array, Array]],
        position_ids: Int[Array, "batch seq"],
        token_bias: Float[Array, "batch 1 seq seq"] | None,
        input_ids: Int[Array, "batch seq"] | None = None,
        output_router_logits: bool = False,
        cache_view: CompressedWindowCacheView | None = None,
        packed_meta: DeepseekV4PackedMeta | None = None,
    ) -> DecoderLayerOutput:
        """Run one V4 decoder block over the residual streams.

        Args:
            hidden_streams: Residual streams ``[B, S, hc, D]``.
            position_embeddings: Per-rope-type half-size cos/sin tables.
            position_ids: Query positions ``[B, S]``.
            token_bias: Additive sliding-window causal mask ``[B, 1, S, S]``;
                ``None`` on cached decode steps.
            input_ids: Token ids (threaded to hash-MoE routing).
            output_router_logits: Whether to surface router scores.
            cache_view: Optional per-layer compressed sliding-window cache
                state.
            packed_meta: Optional eSurge packed-serving token-to-slot
                mapping (threaded to attention).

        Returns:
            DecoderLayerOutput with the updated streams in ``hidden_states``
            and the updated ``cache_view``.
        """
        dtype = hidden_streams.dtype
        post, comb, collapsed = self.attn_hc(hidden_streams)
        attn_output, cache_view = self.self_attn(
            self.input_layernorm(collapsed),
            position_embeddings,
            position_ids,
            token_bias,
            cache_view=cache_view,
            packed_meta=packed_meta,
        )
        hidden_streams = post.astype(dtype)[..., None] * attn_output[..., None, :] + jnp.einsum(
            "bsji,bsjd->bsid", comb.astype(dtype), hidden_streams
        )

        post, comb, collapsed = self.ffn_hc(hidden_streams)
        mlp_output, router_logits = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        hidden_streams = post.astype(dtype)[..., None] * mlp_output[..., None, :] + jnp.einsum(
            "bsji,bsjd->bsid", comb.astype(dtype), hidden_streams
        )

        return DecoderLayerOutput(
            hidden_states=hidden_streams,
            attention_weight=None,
            cache_view=cache_view,
            router_logits=router_logits if output_router_logits else None,
        )


@register_module(TaskType.BASE_MODULE, config=DeepseekV4Config, model_type="deepseek_v4")
class DeepseekV4Model(EasyDeLBaseModule):
    """The base DeepSeek-V4 decoder.

    Embeddings are broadcast into ``hc_mult`` hyper-connection streams, the
    decoder layers update the streams, and the final
    :class:`DeepseekV4HyperHead` collapses them before the shared RMSNorm.

    Supports the stateless forward (training / ``use_cache=False``),
    cached prefill + decode via
    :class:`~easydel.caching.CompressedWindowCache`, and eSurge packed
    serving against per-slot cache rows (see the module docstring).
    """

    esurge_cache_family: tp.ClassVar[str] = "compressed_window"

    def __init__(
        self,
        config: DeepseekV4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the DeepSeek-V4 backbone.

        Args:
            config: Model configuration.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        with self.assign_layer_stage(0, total_layers=config.num_hidden_layers):
            self.embed_tokens = Embed(
                config.vocab_size,
                config.hidden_size,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        remat_layer_block = auto_remat(
            DeepseekV4DecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for layer_idx in range(config.num_hidden_layers):
            with self.assign_layer_stage(layer_idx, total_layers=config.num_hidden_layers):
                self.layers.append(
                    remat_layer_block(
                        config=config,
                        layer_idx=layer_idx,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )

        final_layer_idx = max(0, config.num_hidden_layers - 1)
        with self.assign_layer_stage(final_layer_idx, total_layers=config.num_hidden_layers):
            self.hc_head = DeepseekV4HyperHead(config, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
            self.norm = DeepseekV4RMSNorm(
                dim=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: tp.Any = None,
        cache_metadata: tp.Any = None,
    ) -> MoeModelOutput:
        """Forward pass through the V4 decoder stack (stateless or cached).

        Without ``past_key_values`` this is the stateless forward (training /
        ``use_cache=False``). With a :class:`CompressedWindowCache`, ``S > 1``
        runs a prefill from an empty stream and ``S == 1`` runs one decode
        step; cached streams advance in lockstep across the batch and prompts
        must not be left-padded (see the module docstring).

        Args:
            input_ids: Token ids ``[B, S]``. Required on hash-MoE layers.
            inputs_embeds: Optional precomputed embeddings ``[B, S, D]``.
            attention_mask: Padding mask ``[B, S]`` (1 = attend).
            mask_info: Optional precomputed mask info (positions/padding);
                ignored in cached mode (generation hands in a max-length
                padded MaskInfo that does not match the step shapes).
            position_ids: Positions ``[B, S]``; defaults to mask-derived
                (stateless/prefill) or the cache stream position (decode).
            output_attentions: Unsupported (weights are not materialized per
                layer); accepted for interface parity.
            output_hidden_states: Whether to collect per-layer streams.
            output_router_logits: Whether to collect per-layer router scores.
            mode: Runtime mode (accepted for interface parity).
            past_key_values: Optional :class:`CompressedWindowCache` with
                initialized views (from ``init_cache`` /
                ``init_operations_cache``).
            cache_metadata: Optional runtime metadata. When it carries
                ragged packed-serving fields (``query_start_loc`` and
                ``num_seqs``, i.e. eSurge's ``RaggedPagesMetadata``) with a
                flat ``[1, T]`` input, the packed multi-request path runs
                against per-slot cache rows; otherwise unused.

        Returns:
            MoeModelOutput with the collapsed final hidden state and, in
            cached mode, the updated cache in ``past_key_values``.

        Raises:
            TypeError: If ``past_key_values`` is not a
                :class:`CompressedWindowCache`.
            ValueError: If neither/both of ``input_ids``/``inputs_embeds``
                are given, or the cache views are uninitialized.
        """
        del mode
        cached = past_key_values is not None
        if cached:
            if not isinstance(past_key_values, CompressedWindowCache):
                raise TypeError(
                    "DeepSeek-V4 requires a CompressedWindowCache; got "
                    f"{type(past_key_values).__name__}. Build one via init_cache/init_operations_cache."
                )
            if len(past_key_values) != len(self.layers):
                raise ValueError(f"Cache has {len(past_key_values)} views for {len(self.layers)} layers.")
            if any(view is None for view in past_key_values.views):
                raise ValueError(
                    "CompressedWindowCache views are uninitialized; build the cache via "
                    "init_cache/init_operations_cache (init_empty placeholders cannot be decoded into)."
                )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_hidden_states = () if output_hidden_states else None
        all_router_logits = () if output_router_logits else None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
        batch, sequence_length, _ = inputs_embeds.shape

        packed_qsl = getattr(cache_metadata, "query_start_loc", None)
        packed_num_seqs = getattr(cache_metadata, "num_seqs", None)
        is_packed = cached and batch == 1 and packed_qsl is not None and packed_num_seqs is not None
        is_decode = cached and sequence_length == 1 and not is_packed

        packed_meta = None
        if is_packed:
            if position_ids is None:
                raise ValueError("Packed eSurge serving requires explicit position_ids for the token stream.")
            packed_meta = _build_packed_meta(
                query_start_loc=packed_qsl,
                num_seqs=packed_num_seqs,
                recurrent_state_indices=getattr(cache_metadata, "recurrent_state_indices", None),
                position_ids=position_ids,
                num_slots=int(past_key_values[0].cache_position.shape[0]),
            )
            token_bias = None  # attention derives all masks from per-slot cache state
        elif is_decode:
            if position_ids is None:
                stream_position = past_key_values[0].cache_position.astype(jnp.int32)  # [B]
                position_ids = stream_position[:, None]  # [B, 1]
            token_bias = None  # attention derives the ring mask from the cache state
        else:
            if cached:
                # Cached prefill assumes an unpadded, position-0 stream; the
                # (max-length padded) generation MaskInfo is ignored.
                if position_ids is None:
                    if attention_mask is not None:
                        am = attention_mask.astype(bool)
                        position_ids = jnp.where(am, am.astype(jnp.int32).cumsum(axis=-1) - 1, 0)
                    else:
                        position_ids = jnp.broadcast_to(
                            jnp.arange(sequence_length, dtype=jnp.int32)[None, :], (batch, sequence_length)
                        )
            else:
                mask_info = MaskInfo.dynamic_init(
                    mask_info=mask_info,
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
                if position_ids is None:
                    position_ids = mask_info.q_position_ids
            if attention_mask is not None:
                padding_mask = attention_mask.astype(bool)
            else:
                padding_mask = jnp.ones((batch, sequence_length), dtype=bool)

            # Sliding-window causal mask over the token KVs (applies on ALL layers).
            q_idx = jnp.arange(sequence_length)[:, None]
            k_idx = jnp.arange(sequence_length)[None, :]
            allowed = (k_idx <= q_idx) & (k_idx > q_idx - self.config.sliding_window)
            allowed = allowed[None, None, :, :] & padding_mask[:, None, None, :]
            token_bias = jnp.where(allowed, 0.0, _MASK_MIN).astype(jnp.float32)

        position_embeddings = {
            "main": _rope_cos_sin(self.config, "main", position_ids, inputs_embeds.dtype),
            "compress": _rope_cos_sin(self.config, "compress", position_ids, inputs_embeds.dtype),
        }

        hidden_streams = jnp.broadcast_to(
            inputs_embeds[:, :, None, :],
            (batch, sequence_length, self.config.hc_mult, self.config.hidden_size),
        ).astype(inputs_embeds.dtype)
        hidden_streams = apply_logical_sharding(
            hidden_streams,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        def _layer_loop(block, carry):
            """Run one decoder layer inside the (python-loop) layer scan."""
            hidden_streams, all_hidden_states, all_router_logits, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_streams,)
            cache_view = self._layer_cache_view_at(None, idx, enabled=cached, cache=past_key_values)
            with self._layer_stage_context(idx, layers=self.layers):
                layer_outputs = block(
                    hidden_streams=hidden_streams,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    token_bias=token_bias,
                    input_ids=input_ids,
                    output_router_logits=output_router_logits,
                    cache_view=cache_view,
                    packed_meta=packed_meta,
                )
            self._layer_cache_view_update(None, idx, layer_outputs.cache_view, enabled=cached, cache=past_key_values)
            hidden_streams = self._mark_layer_stage_boundary(layer_outputs.hidden_states, idx, layers=self.layers)
            if output_router_logits:
                all_router_logits += (layer_outputs.router_logits,)
            return hidden_streams, all_hidden_states, all_router_logits, idx + 1

        # NOTE: layers are structurally heterogeneous (per-layer compressor /
        # rope selection), so this always uses the python-loop path;
        # `scan_layers=True` is unsupported for this family.
        hidden_streams, all_hidden_states, all_router_logits, _ = self.layers.scan(
            _layer_loop,
            (hidden_streams, all_hidden_states, all_router_logits, 0),
            trace=True,
        )
        hidden_states = self.norm(self.hc_head(hidden_streams))

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=None,
            router_logits=all_router_logits,
            past_key_values=past_key_values if cached else None,
        )

    def create_compressed_window_cache_config(
        self,
        batch_size: int,
        max_length: int,
    ) -> CompressedWindowCacheConfig:
        """Build the cache configuration for this model's layer schedule.

        Args:
            batch_size: Number of lockstep sequences.
            max_length: Maximum supported stream length (prefill + decode).

        Returns:
            CompressedWindowCacheConfig: Validated static configuration.
        """
        config = self.config
        return CompressedWindowCacheConfig.create(
            num_hidden_layers=config.num_hidden_layers,
            partition_axis=config.partition_axis,
            batch_size=batch_size,
            max_length=max_length,
            sliding_window=config.sliding_window,
            head_dim=config.head_dim,
            index_head_dim=config.index_head_dim,
            csa_rate=config.compress_rates["compressed_sparse_attention"],
            hca_rate=config.compress_rates["heavily_compressed_attention"],
            layer_types=tuple(config.layer_types),
        )

    def init_operations_cache(
        self,
        batch_size: int,
        max_length: int,
        starts=None,
        dtype: jnp.dtype | None = None,
        **kwargs,
    ) -> CompressedWindowCache:
        """Allocate the compressed sliding-window decode cache.

        Overrides the generic mixin: DeepSeek-V4's compressor state is not
        representable by the operation-discovery cache families, so the
        model owns its cache construction. ``starts`` is accepted for
        interface parity but must be zero — cached streams start at
        position 0 and left-padded prompts are unsupported. Under eSurge
        ``batch_size`` is ``max_num_reqs`` and each row is an independent
        request slot at its own stream position.

        Args:
            batch_size: Number of cache rows (lockstep sequences for
                ``generate()``; request slots for eSurge).
            max_length: Maximum supported stream length.
            starts: Ignored (interface parity with the generation mixin).
            dtype: Cache storage dtype; defaults to the module compute dtype.
            **kwargs: Ignored engine-oriented arguments (page_size,
                hbm_utilization, quantizer, ...).

        Returns:
            CompressedWindowCache: Initialized cache for all layers.
        """
        del starts, kwargs
        return CompressedWindowCache.init_cache(
            config=self.create_compressed_window_cache_config(batch_size=batch_size, max_length=max_length),
            dtype=dtype if dtype is not None else self.dtype,
        )

    def init_cache(
        self,
        batch_size: int,
        max_length: int,
        starts=None,
        shardings=None,
        pad_token_id=None,
    ) -> CompressedWindowCache:
        """Allocate the compressed sliding-window decode cache.

        Args:
            batch_size: Number of lockstep sequences.
            max_length: Maximum supported stream length.
            starts: Ignored (streams start at position 0).
            shardings: Ignored (state is replicated).
            pad_token_id: Ignored.

        Returns:
            CompressedWindowCache: Initialized cache for all layers.
        """
        del shardings, pad_token_id
        return self.init_operations_cache(batch_size=batch_size, max_length=max_length, starts=starts)

    def get_encoder(self):
        """Decoder-only model: no encoder."""
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """Return the decoder (self)."""
        return self

    def get_lm_head(self):
        """Base model has no LM head."""
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """Return the token embedding layer."""
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=DeepseekV4Config, model_type="deepseek_v4")
class DeepseekV4ForCausalLM(BaseCausalLMModule[DeepseekV4Model, DeepseekV4Config]):  # type: ignore
    """DeepSeek-V4 with a causal language modeling head.

    Supports the stateless forward (training), cached prefill + decode
    generation via :class:`~easydel.caching.CompressedWindowCache`, and
    eSurge packed serving (see the module docstring).
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "deepseek_v4"
    _config_class = DeepseekV4Config
    esurge_cache_family: tp.ClassVar[str] = "compressed_window"

    def __init__(
        self,
        config: DeepseekV4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the causal LM head over the V4 backbone.

        Args:
            config: Model configuration.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators.
        """
        super().__init__(
            config=config,
            base_model_class=DeepseekV4Model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=None,
        )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: tp.Any = None,
        cache_metadata: tp.Any = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
        """Forward pass producing vocabulary logits.

        Args:
            input_ids: Token ids ``[B, S]``.
            inputs_embeds: Optional precomputed embeddings.
            attention_mask: Padding mask ``[B, S]``.
            mask_info: Optional precomputed mask info.
            position_ids: Optional positions ``[B, S]``.
            output_attentions: Interface parity (per-layer weights are not
                materialized).
            output_hidden_states: Whether to collect per-layer streams.
            output_router_logits: Whether to collect router scores.
            mode: Runtime mode (interface parity).
            past_key_values: Optional :class:`CompressedWindowCache` for
                cached prefill/decode.
            cache_metadata: Accepted for interface parity (unused).
            apply_lm_head: Whether to project to vocabulary logits.

        Returns:
            MoeCausalLMOutput with logits and optional per-layer outputs.
        """
        return self.forward_moe(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            apply_lm_head=apply_lm_head,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            aux_loss_fn=None,
        )

    def init_operations_cache(
        self,
        batch_size: int,
        max_length: int,
        starts=None,
        dtype: jnp.dtype | None = None,
        **kwargs,
    ) -> CompressedWindowCache:
        """Allocate the compressed sliding-window decode cache (delegates to the base model).

        Args:
            batch_size: Number of lockstep sequences.
            max_length: Maximum supported stream length.
            starts: Ignored (streams start at position 0; left padding is
                unsupported).
            dtype: Cache storage dtype; defaults to the module compute dtype.
            **kwargs: Ignored engine-oriented arguments.

        Returns:
            CompressedWindowCache: Initialized cache for all layers.
        """
        return self.model.init_operations_cache(
            batch_size=batch_size,
            max_length=max_length,
            starts=starts,
            dtype=dtype,
            **kwargs,
        )

    def init_cache(
        self,
        batch_size: int,
        max_length: int,
        starts=None,
        shardings=None,
        pad_token_id=None,
    ) -> CompressedWindowCache:
        """Allocate the compressed sliding-window decode cache (delegates to the base model).

        Args:
            batch_size: Number of lockstep sequences.
            max_length: Maximum supported stream length.
            starts: Ignored (streams start at position 0).
            shardings: Ignored (state is replicated).
            pad_token_id: Ignored.

        Returns:
            CompressedWindowCache: Initialized cache for all layers.
        """
        return self.model.init_cache(
            batch_size=batch_size,
            max_length=max_length,
            starts=starts,
            shardings=shardings,
            pad_token_id=pad_token_id,
        )


__all__ = [
    "DeepseekV4Attention",
    "DeepseekV4DecoderLayer",
    "DeepseekV4ForCausalLM",
    "DeepseekV4Model",
    "DeepseekV4SparseMoeBlock",
]
