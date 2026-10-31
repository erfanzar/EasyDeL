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
  decode step (:class:`PackedSlotMeta`), so prefill, chunked prefill,
  and decode all resume from per-slot positions. The XLA scan path is the
  correctness reference; a fused ragged kernel is a planned TPU follow-up.

With ``indexer_aux_loss_coef > 0``, stateless CSA layers train their indexers
against detached dense sink-augmented attention targets. Only indexer
parameters receive the auxiliary gradient; the LM-loss selection STE is
replaced by this warm-up objective. HCA and cached calls do not participate.
The default coefficient ``0.0`` retains the existing forward and STE.

Note:
    The fused compressed-window decode kernel stays on the ``"xla"`` platform by
    default. Measured on v5p-8 at cc32 with A16W4, ``"xla"`` is the faster
    default end to end even though the Pallas kernel itself is now the faster
    kernel: in isolation it runs at 1.09x median against XLA (up to 1.60x at
    kv_len 640, batch 128) after fixing its sublane layout and dropping a
    redundant KV pad, yet in the model it still loses, 889.7 tok/s against
    905.4. Two reasons, both structural rather than tuning:

    * a Mosaic custom call cannot be auto-partitioned, so the pallas path is
      wrapped in ``shard_map`` under a multi-device mesh, and its
      full-to-shard / shard-to-full conversions cost more than the kernel
      saves -- the XLA path pays nothing because GSPMD partitions it;
    * V4 mixes layer types, and the kernel only wins on the
      ``compressed_sparse`` shape (kv_len 640, 21 layers, 1.27x). The
      ``heavily_compressed`` shape (kv_len 144, 20 layers) is still 0.90x,
      roughly cancelling the gain.

    Attention is only ~3.8% of a decode step in any case, so the ceiling here is
    under a percent. Do not flip this default without re-measuring end to end; a
    faster kernel is not the same thing as a faster model.
"""

import functools
import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
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
from easydel.infra.sharding import inference_mode_forces_decode
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
    clamped_swiglu,
    dense_gate_up_layout,
    gated_mlp_forward,
    split_fused_gate_up_projection,
)
from easydel.layers.attention import CompressedWindowAttention
from easydel.layers.compressed_window import (
    DECODE_ATTEND_PLATFORM,
    MASK_MIN,
    TRAIN_ATTEND_PLATFORM,
    PackedSlotMeta,
    advance_compressor_stream,
    block_sink_attention,
    block_token_bias,
    build_packed_slot_meta,
    compress_windows,
    entry_visibility,
    gather_selected_entries,
    grid_to_tokens,
    occupied_columns,
    packed_block_size,
    raise_if_runtime_true,
    reject_packed_or_padded_batch,
    run_column_blocks,
    tokens_to_grid,
    trace_compressor_stream,
)
from easydel.layers.indexer import (
    CompressedIndexer,
    CompressedIndexerConfig,
    indexer_distillation_loss,
    is_vacuous_selection,
    selection_bias,
)
from easydel.layers.moe import moe_group_topk_select, normalize_topk_weights
from easydel.layers.residual import (
    HyperStreamSharding,
    ManifoldHyperConnection,
    ManifoldHyperConnectionConfig,
    ManifoldHyperHead,
    manifold_residual_write,
)
from easydel.layers.rotary import apply_rope_interleaved_partial, labeled_rope_cos_sin
from easydel.modules._base import BaseCausalLMModule
from easydel.utils.checkpoint_keys import KEEP, KeyRuleSet, Map, Nested, Rewrite

from .deepseek_v4_configuration import DeepseekV4Config

# Attend platforms (shared defaults in ``easydel.layers.compressed_window``); module globals so tests can pin a
# backend per family.
_DECODE_KERNEL_PLATFORM = DECODE_ATTEND_PLATFORM
_TRAIN_KERNEL_ENABLED = True
_TRAIN_KERNEL_PLATFORM = TRAIN_ATTEND_PLATFORM


def _entry_causal_bias(position_ids: Array, n_entries: int, rate: int) -> Array:
    """Additive fp32 causal bias ``[B, 1, S, n_entries]`` over compressed entries (see :func:`entry_visibility`)."""
    return jnp.where(entry_visibility(position_ids, n_entries, rate), 0.0, MASK_MIN)[:, None].astype(jnp.float32)


def _decode_entry_visibility(position_ids: Array, cache_position: Array, n_entries: int, rate: int) -> Array:
    """Causal and emitted-entry visibility ``[B, 1, n_entries]`` of padded entry slots during a decode step."""
    return entry_visibility(position_ids, n_entries, rate, emitted_position=cache_position)


def _entry_finalizer(config: DeepseekV4Config, kv_norm: RMSNorm, rate: int):
    """Entry norm + RoPE applied to pooled compressor groups (shared by every V4 compressor stream).

    Pooling itself is the registered ``gated_pool_compress`` operation; a pooled group ``g`` is normed with the
    stream's weighted RMSNorm and rotated with the ``"compress"`` rope at its first token, ``g * rate``.

    Returns:
        ``finalize(pooled [B, G, D], entry [B, G]) -> [B, G, D]``.
    """

    def finalize(pooled: Array, entry: Array) -> Array:
        compressed = kv_norm(pooled)
        cos, sin = labeled_rope_cos_sin(config, "compress", (entry * rate).astype(jnp.int32), compressed.dtype)
        return apply_rope_interleaved_partial(compressed[:, None], cos, sin, unsqueeze_dim=1)[:, 0]

    return finalize


def _compress_prefill(
    config: DeepseekV4Config,
    kv_norm: RMSNorm,
    position_bias: Array,
    kv: Array,
    gate: Array,
    rate: int,
    series_dim: int | None = None,
) -> tuple[Array, Array | None, Array | None]:
    """Compress the complete windows of a prefill into post-RoPE entries.

    Args:
        config: Model config (compress rope).
        kv_norm: The stream's entry norm.
        position_bias: Learned within-window bias ``[rate, F]``.
        kv: Raw kv projections ``[B, n_windows * rate, F]``.
        gate: Raw gate projections, same shape.
        rate: Compression rate.
        series_dim: Per-series width of a two-series (Ca/Cb) stream, else ``None``.

    Returns:
        ``(entries [B, n_windows, D], overlap_kv, overlap_gate)``; the overlaps (two-series only) are the last
        window's Ca kv and biased fp32 Ca gate ``[B, rate, D]`` a cache carries into the next window.
    """
    out = compress_windows(kv, gate, rate, position_bias=position_bias, series_dim=series_dim)
    batch, groups = out.pooled.shape[:2]
    entry = jnp.broadcast_to(jnp.arange(groups, dtype=jnp.int32)[None], (batch, groups))
    return _entry_finalizer(config, kv_norm, rate)(out.pooled, entry), out.overlap_kv, out.overlap_gate


class DeepseekV4HyperConnection(ManifoldHyperConnection):
    """V4 settings adapter for shared manifold-constrained hyper-connections."""

    def __init__(self, config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, precision=None, *, rngs):
        super().__init__(
            ManifoldHyperConnectionConfig.from_model_config(config),
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            mesh_source=config,
        )
        self.config = config


class DeepseekV4HyperHead(ManifoldHyperHead):
    """V4 settings adapter for the shared learned residual-stream head."""

    def __init__(self, config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, precision=None, *, rngs):
        super().__init__(
            ManifoldHyperConnectionConfig.from_model_config(config), dtype, param_dtype, precision, rngs=rngs
        )
        self.config = config


class DeepseekV4HCACompressor(spx.Module):
    """Heavily Compressed Attention compressor (stateless mode).

    Compresses every ``compress_rate`` (m' = 128) source tokens into a single
    compressed KV entry via a per-window softmax-gated sum
    (``gate + position_bias`` softmaxed over the window dim in fp32), followed
    by a weighted RMSNorm and interleaved RoPE ("compress" base) at the
    window-start absolute position. Only complete windows are compressed; the
    remainder is discarded (stateless semantics).
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
        self.kv_norm = RMSNorm(
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
        """Post-RoPE entries ``[B, n_windows, head_dim]`` of complete windows of raw projections."""
        return _compress_prefill(self.config, self.kv_norm, self.position_bias.value, kv, gate, self.compress_rate)[0]

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
        cache_view = trace_compressor_stream(
            cache_view,
            "compressor",
            kv=kv,
            gate=gate,
            positions=cache_view.cache_position[:, None],
            valid=None if valid is None else valid[:, None],
            rate=rate,
            head_dim=self.head_dim,
        )
        entries, buffer_kv, buffer_gate, _, _ = advance_compressor_stream(
            kv=kv,
            gate=gate,
            buffer_kv=cache_view.compressor_buffer_kv,
            buffer_gate=cache_view.compressor_buffer_gate,
            entries=cache_view.compressor_entries,
            position_bias=self.position_bias.value,
            position=cache_view.cache_position,
            rate=rate,
            valid=valid,
            finalize=_entry_finalizer(self.config, self.kv_norm, rate),
        )
        cache_view = cache_view.replace(
            compressor_buffer_kv=buffer_kv,
            compressor_buffer_gate=buffer_gate,
            compressor_entries=entries,
        )
        visible = _decode_entry_visibility(position_ids, cache_view.cache_position, n_slots, rate)
        block_bias = jnp.where(visible, 0.0, MASK_MIN)[:, None, :, :].astype(jnp.float32)
        return entries, block_bias, cache_view

    def cached_extend(
        self,
        hidden_states: Array,
        q_residual: Array,
        positions: Array,
        valid: Array,
        cache_view: CompressedWindowCacheView,
    ) -> tuple[None, CompressedWindowCacheView]:
        """Advance the HCA stream by a block of tokens per row (batched packed prefill).

        Returns ``(None, view)``: HCA attends every reachable entry (no selection).
        """
        del q_residual
        if cache_view.num_entry_slots == 0:
            return None, cache_view
        kv_blk = self.kv_proj(hidden_states)
        gate_blk = self.gate_proj(hidden_states)
        cache_view = trace_compressor_stream(
            cache_view,
            "compressor",
            kv=kv_blk,
            gate=gate_blk,
            positions=positions,
            valid=valid,
            rate=self.compress_rate,
            head_dim=self.head_dim,
        )
        entries, buffer_kv, buffer_gate, _, _ = advance_compressor_stream(
            kv=kv_blk,
            gate=gate_blk,
            buffer_kv=cache_view.compressor_buffer_kv,
            buffer_gate=cache_view.compressor_buffer_gate,
            entries=cache_view.compressor_entries,
            position_bias=self.position_bias.value,
            positions=positions,
            valid=valid,
            rate=self.compress_rate,
            finalize=_entry_finalizer(self.config, self.kv_norm, self.compress_rate),
        )
        return None, cache_view.replace(
            compressor_buffer_kv=buffer_kv, compressor_buffer_gate=buffer_gate, compressor_entries=entries
        )


class _DeepseekV4IndexerAdapter:
    """Ca/Cb compression and cached stream policy for the shared indexer."""

    def __init__(self, config: DeepseekV4Config):
        self.config = config

    def finalize_entries(self, owner: CompressedIndexer, pooled: Array, entry: Array) -> Array:
        """Weighted RMSNorm then compress-RoPE at each entry's first token (``entry * rate``)."""
        return _entry_finalizer(self.config, owner.kv_norm, owner.compress_rate)(pooled, entry)

    def rotate_queries(self, q: Array, position_ids: Array, dtype: jnp.dtype) -> Array:
        """Apply V4's partial interleaved compression RoPE to index queries."""
        cos, sin = labeled_rope_cos_sin(self.config, "compress", position_ids, dtype)
        return apply_rope_interleaved_partial(q.transpose(0, 2, 1, 3), cos, sin, unsqueeze_dim=1).transpose(0, 2, 1, 3)

    @staticmethod
    def entry_visibility(position_ids, cache_position, n_slots, rate):
        """Use emitted-entry counts as well as the query's causal boundary."""
        return _decode_entry_visibility(position_ids, cache_position, n_slots, rate)

    @staticmethod
    def selection_is_vacuous(index_topk, n_slots):
        """Retain the model's serving fast-path policy."""
        return is_vacuous_selection(index_topk, n_slots)


def _compressed_indexer_config(config: DeepseekV4Config) -> CompressedIndexerConfig:
    """Map model settings without introducing a nested parameter namespace."""
    return CompressedIndexerConfig(
        hidden_size=config.hidden_size,
        q_lora_rank=config.q_lora_rank,
        index_n_heads=config.index_n_heads,
        index_head_dim=config.index_head_dim,
        index_topk=config.index_topk,
        compress_rate=config.compress_rates["compressed_sparse_attention"],
        rms_norm_eps=config.rms_norm_eps,
        initializer_range=config.initializer_range,
        scoring_kernel=config.index_scoring_kernel,
    )


class DeepseekV4Indexer(CompressedIndexer):
    """V4's native Ca/Cb cache adapter over shared compressed-entry selection."""

    def __init__(self, config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, precision=None, *, rngs):
        super().__init__(
            _compressed_indexer_config(config),
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            compressor_adapter=_DeepseekV4IndexerAdapter(config),
        )
        # Row-parallel TPU top-k resolves its layout from the model mesh.
        self.mesh_source = config
        self.config = config


class DeepseekV4CSACompressor(spx.Module):
    """Compressed Sparse Attention compressor (stateless mode).

    Runs the two-series (Ca/Cb) window compression at ``head_dim`` and a
    :class:`DeepseekV4Indexer` that gates per-query visibility of the
    compressed entries down to ``index_topk`` picks.
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
        self.kv_norm = RMSNorm(
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
        return_scores: bool = False,
    ) -> tuple[Array | None, Array | None] | tuple[Array | None, Array | None, Array | None]:
        """Produce compressed KV entries and the indexer-gated block bias.

        Args:
            hidden_states: Attention-input hidden states ``[B, S, D_model]``.
            q_residual: Normed query LoRA residual shared with core attention.
            position_ids: Query positions ``[B, S]``.
            return_scores: Return detached-input indexer scores for warm-up
                instead of adding the LM-loss straight-through gradient.

        Returns:
            Tuple ``(compressed_kv, block_bias)`` with ``compressed_kv`` of
            shape ``[B, T, head_dim]`` (``T = S // compress_rate``) and an
            additive fp32 ``block_bias`` ``[B, 1, S, T]`` opening only the
            indexer-selected, causally valid entries; both ``None`` when no
            complete window fits. With ``return_scores``, a third element
            holds full-domain differentiable scores ``[B, S, T]`` (``-inf``
            outside the causal entry domain), or ``None`` without entries.

        Note:
            ``return_scores=True`` is the warm-up path: indexer inputs are
            detached and selection has no STE, so the indexer learns only
            through the auxiliary KL. Compressed attention entries remain
            attached to the main LM objective. The default path is unchanged.
        """
        seq_len = hidden_states.shape[1]
        rate = self.compress_rate
        n_windows = seq_len // rate
        if n_windows == 0:
            return (None, None, None) if return_scores else (None, None)
        usable = n_windows * rate
        kv = self.kv_proj(hidden_states)[:, :usable]
        gate = self.gate_proj(hidden_states)[:, :usable]
        compressed, _, _ = self._compress_windows(kv, gate)
        if return_scores:
            indices, scores = self.indexer(
                jax.lax.stop_gradient(hidden_states),
                jax.lax.stop_gradient(q_residual),
                position_ids,
                return_scores=True,
            )
            # CompressedIndexer returns raw scores; distillation requires the
            # full causal domain, NOT just the selected top-k subset.
            scores = jnp.where(entry_visibility(position_ids, n_windows, rate), scores, -jnp.inf)
            return compressed, selection_bias(indices, n_windows), scores
        selection = self.indexer(hidden_states, q_residual, position_ids, return_selection=True)
        block_bias = selection_bias(selection.indices, n_windows, score_proxy=selection.score_proxy, mask=selection.mask)
        return compressed, block_bias

    def _compress_windows(self, kv: Array, gate: Array) -> tuple[Array, Array, Array]:
        """Two-series post-RoPE entries plus the last window's Ca kv / biased fp32 Ca gate."""
        return _compress_prefill(
            self.config, self.kv_norm, self.position_bias.value, kv, gate, self.compress_rate, self.head_dim
        )

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
        seq_len = hidden_states.shape[1]
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
            block_bias = selection_bias(top_k_indices, n_windows)
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
        _batch = hidden_states.shape[0]
        rate = self.compress_rate
        n_slots = cache_view.num_entry_slots
        if n_slots == 0:
            return None, None, cache_view
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        cache_view = trace_compressor_stream(
            cache_view,
            "compressor",
            kv=kv,
            gate=gate,
            positions=cache_view.cache_position[:, None],
            valid=None if valid is None else valid[:, None],
            rate=rate,
            head_dim=self.head_dim,
            position_bias=self.position_bias.value,
        )
        entries, buffer_kv, buffer_gate, overlap_kv, overlap_gate = advance_compressor_stream(
            kv=kv,
            gate=gate,
            buffer_kv=cache_view.compressor_buffer_kv,
            buffer_gate=cache_view.compressor_buffer_gate,
            entries=cache_view.compressor_entries,
            position_bias=self.position_bias.value,
            position=cache_view.cache_position,
            rate=rate,
            overlap_kv=cache_view.compressor_overlap_kv,
            overlap_gate=cache_view.compressor_overlap_gate,
            valid=valid,
            finalize=_entry_finalizer(self.config, self.kv_norm, rate),
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
        # Gather the selected entries instead of handing the attention the full
        # padded entry axis with everything else masked out.
        #
        # The indexer picks `index_topk` entries (512); the entry axis is
        # `max_length / rate` (65536 at a 262144 context). Returning all of
        # them made the attention compute 65536 columns per head per layer per
        # step and throw away 99.2% of them via an additive -inf bias. Ablation
        # put that attention at ~68% of the decode step. The gather moves
        # ~512x512x2 bytes per layer instead.
        #
        # `-1` marks an invalid pick, exactly as `selection_bias` treats
        # it: index 0 is substituted so the gather stays in bounds and the bias
        # masks the column out, so the attended set is unchanged.
        picks = top_k_indices[:, 0]  # [B, K]
        k_sel = picks.shape[-1]

        def _by_gather(_):
            picked = picks >= 0
            g = jnp.take_along_axis(entries, jnp.where(picked, picks, 0)[:, :, None], axis=1)
            b = jnp.where(picked, 0.0, MASK_MIN)[:, None, None, :].astype(jnp.float32)
            return g, b

        def _by_prefix(_):
            # Every live entry is selected, so the picked set IS the live
            # prefix. Attention is permutation-invariant over the key axis, so
            # reading that prefix contiguously gives the same result as
            # gathering the same rows scattered -- at a fraction of the cost.
            # Computed over the full entry axis and sliced, rather than asked
            # for at k_sel width: the narrow form is strictly less arithmetic
            # but measured 15.8 -> 11.1 tok/s, XLA fusing the wide version
            # better. Identical values either way (visibility of entry i
            # depends only on i).
            vis = _decode_entry_visibility(position_ids, cache_view.cache_position, n_slots, rate)
            g = entries[:, :k_sel]
            b = jnp.where(vis[:, :, :k_sel], 0.0, MASK_MIN)[:, None, :, :].astype(jnp.float32)
            return g, b

        # `index_topk` picks out of `max_length / rate` entries. Until the
        # context outgrows `index_topk * rate` tokens there are fewer live
        # entries than picks, so the selection cannot exclude anything and the
        # scattered gather is buying nothing. Measured: the gather is ~50 ms of
        # a 92 ms decode step, three orders off the HBM roofline for the bytes
        # it moves, because 512 separate 1 KB rows is the worst case for the
        # memory system. The contiguous branch is the same rows in order.
        live = jnp.max((cache_view.cache_position.astype(jnp.int32) + 1) // rate)
        gathered, sel_bias = jax.lax.cond(live <= k_sel, _by_prefix, _by_gather, operand=None)
        return gathered, sel_bias, cache_view

    def cached_extend(
        self,
        hidden_states: Array,
        q_residual: Array,
        positions: Array,
        valid: Array,
        cache_view: CompressedWindowCacheView,
    ) -> tuple[Array | None, CompressedWindowCacheView]:
        """Advance the CSA compressor and indexer streams by a block of tokens per row (batched packed prefill).

        Returns:
            ``(top_k [B, blk, k] or None, view)``: ``None`` when the indexer selection is vacuous (every reachable
            entry is attended).
        """
        if cache_view.num_entry_slots == 0:
            return None, cache_view
        kv_blk = self.kv_proj(hidden_states)
        gate_blk = self.gate_proj(hidden_states)
        cache_view = trace_compressor_stream(
            cache_view,
            "compressor",
            kv=kv_blk,
            gate=gate_blk,
            positions=positions,
            valid=valid,
            rate=self.compress_rate,
            head_dim=self.head_dim,
            position_bias=self.position_bias.value,
        )
        entries, buffer_kv, buffer_gate, overlap_kv, overlap_gate = advance_compressor_stream(
            kv=kv_blk,
            gate=gate_blk,
            buffer_kv=cache_view.compressor_buffer_kv,
            buffer_gate=cache_view.compressor_buffer_gate,
            entries=cache_view.compressor_entries,
            position_bias=self.position_bias.value,
            positions=positions,
            valid=valid,
            rate=self.compress_rate,
            overlap_kv=cache_view.compressor_overlap_kv,
            overlap_gate=cache_view.compressor_overlap_gate,
            finalize=_entry_finalizer(self.config, self.kv_norm, self.compress_rate),
        )
        cache_view = cache_view.replace(
            compressor_buffer_kv=buffer_kv,
            compressor_buffer_gate=buffer_gate,
            compressor_entries=entries,
            compressor_overlap_kv=overlap_kv,
            compressor_overlap_gate=overlap_gate,
        )
        return self.indexer.cached_extend(
            hidden_states, q_residual, positions, valid, cache_view, kernel_mesh=getattr(self.config, "mesh", None)
        )


_COMPRESSOR_CLASSES: dict[str, type | None] = {
    "sliding_attention": None,
    "compressed_sparse_attention": DeepseekV4CSACompressor,
    "heavily_compressed_attention": DeepseekV4HCACompressor,
}


class DeepseekV4Attention(CompressedWindowAttention):
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
        layer_type = config.layer_types[layer_idx]
        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            rope_label="main" if layer_type == "sliding_attention" else "compress",
            q_head_norm_eps=config.rms_norm_eps,  # HF q_b_norm (unweighted, per head)
        )
        self.layer_type = layer_type
        compressor_class = _COMPRESSOR_CLASSES[layer_type]
        self.compressor = (
            None
            if compressor_class is None
            else compressor_class(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        )

    def _indexer_aux_active(self, cache_view: CompressedWindowCacheView | None) -> bool:
        """CSA warm-up applies only to stateless calls with a positive coefficient."""
        return (
            self.layer_type == "compressed_sparse_attention"
            and cache_view is None
            and self.config.indexer_aux_loss_coef > 0.0
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
        ring_bias = jnp.where(slot_positions >= 0, 0.0, MASK_MIN).astype(jnp.float32)  # [B, W]
        attn_bias = ring_bias[:, None, None, :]  # [B, 1, 1, W]
        kv_axis = ring
        if self.compressor is not None:
            # The compressor reads cache_view.cache_position, so it must
            # run before the position advances below.
            compressed_kv, block_bias, cache_view = self.compressor.cached_forward(
                hidden_states, q_residual, position_ids, cache_view, valid=valid
            )
            if compressed_kv is not None:
                # Rejoin rather than attending twice and merging. The split
                # existed to avoid concatenating a 65536-wide entry axis; the
                # selection now returns `index_topk` (512) rows, so the concat
                # is small and one fused call should beat two plus a softmax
                # merge.
                kv_axis = jnp.concatenate([ring, compressed_kv.astype(ring.dtype)], axis=1)
                attn_bias = jnp.concatenate([attn_bias, block_bias], axis=-1)
        advance = 1 if valid is None else valid.astype(jnp.int32)
        cache_view = cache_view.replace(window_kv=ring, cache_position=stream_position + advance)
        return self.fused_decode_attend(q, kv_axis, attn_bias[:, 0], platform=_DECODE_KERNEL_PLATFORM), cache_view

    def _extend_attend(
        self,
        q: Array,
        kv: Array,
        hidden_states: Array,
        q_residual: Array,
        positions: Array,
        valid: Array,
        cache_view: CompressedWindowCacheView,
    ) -> tuple[Array, CompressedWindowCacheView]:
        """Advance every row by a block of up to ``sliding_window`` tokens in one pass (batched packed prefill).

        Same visible set as :meth:`_decode_attend` applied column by column: each query sees the pre-block ring slots
        still inside its window, the block's own keys up to itself, and the compressed entries below its reach
        (all of them for HCA / a vacuous CSA selection, its indexer picks otherwise). Built on the shared
        ``layers.compressed_window`` block helpers.

        Args:
            q: Rotated queries ``[S, H, blk, D]``.
            kv: Rotated token KV ``[S, blk, D]``.
            hidden_states: ``[S, blk, D_model]``.
            q_residual: ``[S, blk, q_lora]``.
            positions: Stream positions ``[S, blk]`` (garbage at invalid cells).
            valid: ``[S, blk]`` bool, a prefix per row.
            cache_view: This layer's cache state.

        Returns:
            ``(attention [S, H, blk, D] before the inverse output rotation, view)``.
        """
        slots, block = positions.shape
        window = self.sliding_window
        positions = positions.astype(jnp.int32)
        n_valid = jnp.sum(valid, axis=1).astype(jnp.int32)
        rows = jnp.arange(slots)[:, None]
        ring = cache_view.window_kv
        token_bias = block_token_bias(positions, valid, window)  # [S, blk, W + blk]
        token_keys = jnp.concatenate([ring.astype(kv.dtype), kv], axis=1)
        write_slot = jnp.where(valid, positions % window, window)
        ring = ring.at[rows, write_slot].set(kv.astype(ring.dtype), mode="drop")
        gathered = gathered_bias = None
        if self.compressor is not None:
            top_k, cache_view = self.compressor.cached_extend(hidden_states, q_residual, positions, valid, cache_view)
            entries = cache_view.compressor_entries
            if entries is not None and top_k is None:
                # Every reachable entry: shared by the block's queries, masked per query by its reach.
                reach = (positions + 1) // self.compressor.compress_rate
                visible = jnp.arange(entries.shape[1], dtype=jnp.int32)[None, None, :] < reach[..., None]
                token_keys = jnp.concatenate([token_keys, entries.astype(kv.dtype)], axis=1)
                token_bias = jnp.concatenate(
                    [token_bias, jnp.where(visible, 0.0, MASK_MIN).astype(jnp.float32)], axis=-1
                )
            elif entries is not None and top_k.shape[-1] > 0:
                k = top_k.shape[-1]
                flat, flat_bias = gather_selected_entries(entries, top_k.reshape(slots, block * k))
                gathered = flat.reshape(slots, block, k, -1).astype(kv.dtype)
                gathered_bias = flat_bias.reshape(slots, block, k)
        out = block_sink_attention(
            q,
            token_keys,
            token_bias,
            self.sinks.value,
            scaling=self.scaling,
            gathered=gathered,
            gathered_bias=gathered_bias,
            # Same numerics as the per-column path this replaces: the registered compressed-window decode op runs
            # both contractions at HIGHEST (TPU default precision is single-pass bf16, ~1e-2 logit error).
            precision=jax.lax.Precision.HIGHEST,
        )
        new_position = jnp.where(n_valid > 0, positions[:, 0] + n_valid, cache_view.cache_position)
        return out, cache_view.replace(window_kv=ring, cache_position=new_position)

    def _packed_forward(
        self,
        hidden_states: Float[Array, "one tokens hidden"],
        position_embeddings: dict[str, tuple[Array, Array]],
        position_ids: Int[Array, "one tokens"],
        cache_view: CompressedWindowCacheView,
        packed_meta: "PackedSlotMeta",
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
                :class:`PackedSlotMeta`).

        Returns:
            Tuple of the attention output ``[1, T, D_model]`` (garbage at
            padded token positions, which downstream never reads) and the
            updated cache view.

        Note:
            A pure-decode bucket takes a fast path. There every request
            contributes exactly one token -- the invariant the executor derives
            the bucket's decode layout from, ``model_executor.py``'s
            ``decode_mode_specs(num_tokens == padded_num_reqs)`` -- so
            ``col_of_token`` is 0 everywhere and the ``[num_slots, T, ...]`` grid
            degenerates to ``[num_slots, ...]``. Building the full grid anyway is
            quadratic in the request count and dominates the step: profiling a
            cc128 decode measured 14126 device ops, 59.8% of them data movement,
            with ``broadcast`` alone at 13.8 ms (210 ops x 65.7 us -- the
            ``jnp.zeros`` for q_grid/attn_columns is 1.07 GB per layer at cc128)
            and the column ``while`` at 13.6 ms, against 2.3 ms (3.6%) for the
            actual expert matmul. Collapsing the token axis removes both: one
            ``_decode_attend`` on ``[num_slots, ...]`` state, no scratch grid, no
            loop.

        Note:
            The general path below covers prefill and chunked prefill, where a
            slot really can carry several tokens in one step; pure decode is
            handled by the fast path above, under the static
            :func:`inference_mode_forces_decode` signal, and never reaches it.
        """
        _, total_tokens, _ = hidden_states.shape
        num_slots = cache_view.cache_position.shape[0]
        cos, sin = position_embeddings[self.rope_label]

        # Prefill-from-empty self-heal: a valid chunk whose first token sits
        # at absolute position 0 starts a brand-new stream, so its slot's
        # state is re-initialized before processing. This makes slot reuse
        # robust to any stale state (engine warmup runs, preemption
        # restarts) without relying on out-of-band clears.
        restart = packed_meta.valid_grid[:, 0] & (packed_meta.positions_grid[:, 0] == 0)
        cache_view = cache_view.reset_rows(restart)

        q_residual, q = self.project_query(hidden_states, cos, sin)  # [1, T, q_lora], [1, H, T, head_dim]
        kv = self.project_kv(hidden_states, cos, sin)  # [1, T, head_dim]

        slot_tok = packed_meta.slot_of_token  # [T] (== num_slots for padding -> dropped)
        col_tok = packed_meta.col_of_token  # [T]
        safe_slot = jnp.minimum(slot_tok, num_slots - 1)

        def _decode_fast(cache_view):
            """One token per slot (pure decode): a single ``_decode_attend`` over ``[num_slots, ...]`` state."""

            def _to_slots(values: Array) -> Array:
                """Scatter per-token values ``[T, ...]`` into ``[num_slots, ...]``."""
                out = jnp.zeros((num_slots, *values.shape[1:]), dtype=values.dtype)
                return out.at[slot_tok].set(values, mode="drop")

            attn_col, cache_view = self._decode_attend(
                q=_to_slots(q[0].transpose(1, 0, 2))[:, :, None, :],  # [S, H, 1, head_dim]
                kv=_to_slots(kv[0])[:, None, :],  # [S, 1, head_dim]
                hidden_states=_to_slots(hidden_states[0])[:, None, :],  # [S, 1, D]
                q_residual=_to_slots(q_residual[0])[:, None, :],  # [S, 1, q_lora]
                position_ids=packed_meta.positions_grid[:, :1],  # [S, 1]
                cache_view=cache_view,
                valid=packed_meta.valid_grid[:, 0],  # [S]
            )
            attn_packed = attn_col[:, :, 0, :][safe_slot][None, ...]  # [1, T, H, head_dim]
            return self.project_output(attn_packed, cos, sin), cache_view

        def _grid_path(cache_view, block: int):
            """Grid path with a static ``block`` of columns per loop step (prefill, chunked prefill, verify)."""
            # Shared packed driver (``layers.compressed_window``, also used by DeepSeek-V4.1): prefill / chunked-prefill
            # buckets advance a block of columns per loop step through :meth:`_extend_attend`; buckets with at most one
            # token per slot keep the single-column :meth:`_decode_attend`. Only occupied columns are visited.
            width = -(-total_tokens // block) * block

            def _to_grid(values: Array) -> Array:
                """Scatter per-token values ``[T, ...]`` into ``[num_slots, width, ...]``."""
                return tokens_to_grid(values, slot_tok, col_tok, num_slots, width)

            def _pad_cols(grid: Array) -> Array:
                return grid if width == total_tokens else jnp.pad(grid, ((0, 0), (0, width - total_tokens)))

            hidden_grid = _to_grid(hidden_states[0])  # [S, W, D]
            qres_grid = _to_grid(q_residual[0])  # [S, W, q_lora]
            q_grid = _to_grid(q[0].transpose(1, 0, 2))  # [S, W, H, head_dim]
            kv_grid = _to_grid(kv[0])  # [S, W, head_dim]
            pos_grid = _pad_cols(packed_meta.positions_grid)
            valid_grid = _pad_cols(packed_meta.valid_grid)
            num_used = occupied_columns(valid_grid)
            attn_columns = jnp.zeros((width, num_slots, self.num_heads, self.head_dim), dtype=cache_view.window_kv.dtype)

            def _block_step(start: Array, carry):
                """Advance every slot by the ``block`` columns starting at ``start``."""
                view, acc = carry

                def cols(grid: Array) -> Array:
                    return jax.lax.dynamic_slice_in_dim(grid, start, block, axis=1)

                attn, view = self._extend_attend(
                    q=cols(q_grid).transpose(0, 2, 1, 3),
                    kv=cols(kv_grid),
                    hidden_states=cols(hidden_grid),
                    q_residual=cols(qres_grid),
                    positions=cols(pos_grid),
                    valid=cols(valid_grid),
                    cache_view=view,
                )
                acc = jax.lax.dynamic_update_slice_in_dim(
                    acc, attn.transpose(2, 0, 1, 3).astype(acc.dtype), start, axis=0
                )
                return view, acc

            def _column_step(col: Array, carry):
                """Advance every slot by one grid column (column ``col``)."""
                view, acc = carry
                hidden_col = jax.lax.dynamic_index_in_dim(hidden_grid, col, axis=1, keepdims=False)  # [S, D]
                qres_col = jax.lax.dynamic_index_in_dim(qres_grid, col, axis=1, keepdims=False)  # [S, q_lora]
                q_col = jax.lax.dynamic_index_in_dim(q_grid, col, axis=1, keepdims=False)  # [S, H, head_dim]
                kv_col = jax.lax.dynamic_index_in_dim(kv_grid, col, axis=1, keepdims=False)  # [S, head_dim]
                pos_col = jax.lax.dynamic_index_in_dim(pos_grid, col, axis=1, keepdims=False)  # [S]
                valid_col = jax.lax.dynamic_index_in_dim(valid_grid, col, axis=1, keepdims=False)  # [S]
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
                return view, acc

            cache_view, attn_columns = run_column_blocks(
                _block_step if block > 1 else _column_step,
                (cache_view, attn_columns),
                num_used=num_used,
                block=block,
            )  # attn_columns: [W, S, H, head_dim]

            attn_packed = grid_to_tokens(attn_columns, slot_tok, col_tok, num_slots)[None, ...]  # [1, T, H, head_dim]
            return self.project_output(attn_packed, cos, sin), cache_view

        def _general(cache_view):
            """Grid path: any number of tokens per slot, block width chosen at run time.

            The bucket's default block is wide (``min(64, T, window)``). A speculative verify step (``1 + k`` tokens on
            every slot) occupies few columns, so one bucket-wide block is mostly padding (measured on DeepSeek-V4-Flash:
            cc=8 verify forward 275.6 ms vs 124.6 ms with a slot-share block). The narrow block (one slot's share of the
            bucket) is used only when every occupied column fits in ONE narrow block -- the same single loop step,
            narrower -- so prefill steps keep the original wide-block program (a static narrow block made single-request
            prefill 196 -> 273 ms median).
            """
            wide = packed_block_size(total_tokens, num_slots, self.sliding_window)
            narrow = packed_block_size(total_tokens, num_slots, self.sliding_window, cap_by_slot_share=True)
            if narrow >= wide:
                return _grid_path(cache_view, wide)
            return jax.lax.cond(
                occupied_columns(packed_meta.valid_grid) <= narrow,
                lambda c: _grid_path(c, narrow),
                lambda c: _grid_path(c, wide),
                cache_view,
            )

        if not inference_mode_forces_decode():
            return _general(cache_view)
        # The decode-layout scope only says the bucket holds as many tokens as request slots. A speculative verify
        # window puts 1 + k tokens on ONE slot of such a bucket, so the one-token-per-slot fast path runs only when
        # at most one grid column is occupied (pure decode); otherwise every token goes through the grid path.
        return jax.lax.cond(occupied_columns(packed_meta.valid_grid) <= 1, _decode_fast, _general, cache_view)

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        position_embeddings: dict[str, tuple[Array, Array]],
        position_ids: Int[Array, "batch seq"],
        token_bias: Float[Array, "batch 1 seq seq"] | None,
        cache_view: CompressedWindowCacheView | None = None,
        packed_meta: "PackedSlotMeta | None" = None,
        return_indexer_aux_loss: bool = False,
    ) -> tuple[Array, CompressedWindowCacheView | None] | tuple[Array, CompressedWindowCacheView | None, Array | None]:
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
            return_indexer_aux_loss: Append the unscaled CSA warm-up loss to
                the returned tuple (``None`` when inactive or without entries).

        Returns:
            Tuple of the attention output ``[B, S, D_model]`` and the
            updated cache view (``None`` when running stateless), optionally
            followed by the indexer auxiliary loss.
        """
        if cache_view is not None and packed_meta is not None:
            output, cache_view = self._packed_forward(
                hidden_states, position_embeddings, position_ids, cache_view, packed_meta
            )
            return (output, cache_view, None) if return_indexer_aux_loss else (output, cache_view)

        indexer_aux_loss = None
        _, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings[self.rope_label]

        q_residual, q = self.project_query(hidden_states, cos, sin)
        kv = self.project_kv(hidden_states, cos, sin)  # [B, S, head_dim]

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
                index_scores = None
                if self._indexer_aux_active(cache_view):
                    compressed_kv, block_bias, index_scores = self.compressor(
                        hidden_states, q_residual, position_ids, return_scores=True
                    )
                elif cache_view is None:
                    compressed_kv, block_bias = self.compressor(hidden_states, q_residual, position_ids)
                else:
                    compressed_kv, block_bias, cache_view = self.compressor.cached_forward(
                        hidden_states, q_residual, position_ids, cache_view
                    )
                if index_scores is not None:
                    # The token-bias diagonal recovers the original per-key
                    # padding validity: every live token can attend itself.
                    token_valid = jnp.diagonal(token_bias[:, 0], axis1=-2, axis2=-1) == 0
                    indexer_aux_loss = indexer_distillation_loss(
                        q,
                        compressed_kv,
                        index_scores,
                        scaling=self.scaling,
                        window_kv=kv,
                        sinks=self.sinks.value,
                        token_valid=token_valid,
                        window=self.sliding_window,
                        precision=self.precision,
                    )
                if compressed_kv is not None:
                    kv = jnp.concatenate([kv, compressed_kv.astype(kv.dtype)], axis=1)
                    attn_bias = jnp.concatenate([attn_bias, block_bias], axis=-1)
            # Default: inline dense ``_attend`` (graph unchanged). Opt in to the
            # fused differentiable full-sequence kernel via the env flag.
            if _TRAIN_KERNEL_ENABLED:
                attn_output = self.fused_full_attend(q, kv, attn_bias[:, 0], platform=_TRAIN_KERNEL_PLATFORM)
            else:
                attn_output = self.sink_attend(q, kv, attn_bias)
        else:
            attn_output, cache_view = self._decode_attend(
                q=q,
                kv=kv,
                hidden_states=hidden_states,
                q_residual=q_residual,
                position_ids=position_ids,
                cache_view=cache_view,
            )

        # V == K carried RoPE: project_output undoes it at the query positions, then the grouped output projection.
        output = self.project_output(attn_output.transpose(0, 2, 1, 3), cos, sin)
        return (output, cache_view, indexer_aux_loss) if return_indexer_aux_loss else (output, cache_view)


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
        activated = clamped_swiglu(gate, up, limit=self.limit, act_fn=self.act_fn)
        return checkpoint_name(self.down_proj(activated), "mlp_down")


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
        activated = clamped_swiglu(gate, up, limit=self.limit, act_fn=self.act_fn)
        return self.down_proj(activated, group_sizes, sorted_experts)


class DeepseekV4TopKRouter(spx.Module):
    """Learned V4 router: ``sqrtsoftplus`` scores + ``e_score_correction_bias`` selection.

    ``forward`` returns the raw per-expert *scores* (not logits); expert
    selection adds the persistent correction bias, while routing weights are
    gathered from the raw scores (see the select hook installed by
    :class:`DeepseekV4SparseMoeBlock`).
    """

    #: Whether this router selects experts with ``e_score_correction_bias``.
    #: Subclasses that select some other way must clear it so the parameter is
    #: not declared: an unused parameter is still a REQUIRED one at load time,
    #: and the published checkpoint has no value for it.
    uses_score_correction: tp.ClassVar[bool] = True

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
        if self.uses_score_correction:
            self.e_score_correction_bias = ArrayParam.bound(
                shape=(self.num_experts,),
                dtype=jnp.float32,
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
    experts. ``tid2eid`` is stored in float32 (expert ids are exact up to
    2**24; bf16 would round odd ids above 256) and cast to int32 at lookup
    time.

    Selection never consults ``e_score_correction_bias`` here, and DeepSeek
    does not ship one for these layers, so the parameter is not declared.
    """

    uses_score_correction: tp.ClassVar[bool] = False

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
        # float32, not param_dtype: bf16 holds only even integers past 256, and
        # V4-Pro routes over 384 experts.
        self.tid2eid = ArrayParam.bound(
            shape=(config.vocab_size, self.top_k),
            dtype=jnp.float32,
            init_method="zeros",
            key=rngs.param,
        )

    @property
    def reform_param(self):
        """Load routing IDs in their declared fp32 storage, not the model dtype."""
        return {
            "tid2eid$": {
                "splits": [{"name": "tid2eid", "spliter": lambda tensor: tensor.float()}],
                "inverse_spliter": lambda tensor: tensor,
                # The generic converter casts every leaf to param_dtype. Keep
                # the fp32 table exact even when the other weights use bf16.
                "preserve_dtype": True,
            }
        }


class DeepseekV4SparseMoeBlock(BaseMoeModule):
    """DeepSeek-V4 MoE block: routed experts + clamped shared expert.

    Selection: top-k on ``scores + e_score_correction_bias`` (learned layers)
    or a frozen ``tid2eid[input_ids]`` lookup (hash layers). Weights: raw
    scores gathered at the selected indices, normalized
    (``/(sum + 1e-20)``) when ``norm_topk_prob``, scaled by
    ``routed_scaling_factor``. Output: ``routed + shared_experts(residual)``.
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

    def _select_hook(self, e_score_correction_bias: Array) -> functools.partial:
        """Learned-layer routing as a :func:`moe_group_topk_select` ``select_hook``.

        Selects the top-k of ``scores + e_score_correction_bias`` over the gate's already-scored output and weights
        them with the unbiased scores, renormalized when ``norm_topk_prob`` and scaled by ``routed_scaling_factor``.

        Args:
            e_score_correction_bias: Selection bias ``[n_routed_experts]`` or per token
                ``[tokens, n_routed_experts]``.

        Returns:
            functools.partial: A ``select_hook`` taking ``(gate_scores, pre_bias_logits, k)``.
        """
        return functools.partial(
            moe_group_topk_select,
            n_routed_experts=self.config.n_routed_experts,
            score_fn="none",
            e_score_correction_bias=e_score_correction_bias,
            n_group=1,
            norm_topk_prob=self.config.norm_topk_prob,
            routed_scaling_factor=self.routed_scaling_factor,
        )

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

            norm_topk_prob = self.config.norm_topk_prob

            def _select_hash(augmented_scores: Array, pre_bias_logits: Array | None, k: int) -> tuple[Array, Array]:
                del pre_bias_logits, k
                scores = augmented_scores[:, :num_experts].astype(jnp.float32)
                indices = augmented_scores[:, num_experts:].astype(jnp.int32)
                weights = jnp.take_along_axis(scores, indices, axis=-1)
                if norm_topk_prob:
                    weights = normalize_topk_weights(weights)
                return weights * self.routed_scaling_factor, indices

            select_hook = _select_hash
        else:
            select_hook = self._select_hook(self.gate.e_score_correction_bias.value)

        hooks = self.moe_hooks.replace(
            normalize_gate_logits=lambda logits: logits,
            select_hook=select_hook,
        )
        limit = self.config.swiglu_limit

        def ffn_activation(gate: Array, up: Array) -> Array:
            """Clamped SwiGLU: ``silu(clamp(gate, max=limit)) * clamp(up, +-limit)``."""
            return clamped_swiglu(gate, up, limit=limit, act_fn=self.experts.act_fn)

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


@auto_pytree
class _DeepseekV4DecoderLayerOutput(DecoderLayerOutput):
    """Layer output carrying an unscaled CSA indexer warm-up loss."""

    indexer_aux_loss: Array | None = None


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
        self.input_layernorm = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.attn_hc = DeepseekV4HyperConnection(
            config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
        )
        self.ffn_hc = DeepseekV4HyperConnection(
            config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
        )

    def forward(
        self,
        hidden_streams: Float[Array, "batch seq hc hidden"],
        position_embeddings: dict[str, tuple[Array, Array]],
        position_ids: Int[Array, "batch seq"],
        token_bias: Float[Array, "batch 1 seq seq"] | None,
        input_ids: Int[Array, "batch seq"] | None = None,
        output_router_logits: bool = False,
        cache_view: CompressedWindowCacheView | None = None,
        packed_meta: PackedSlotMeta | None = None,
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
            DecoderLayerOutput with the updated streams in ``hidden_states``,
            the updated ``cache_view`` and an optional unscaled
            ``indexer_aux_loss`` for stateless CSA warm-up.
        """
        post, comb, collapsed = self.attn_hc(hidden_streams)
        aux_active = self.self_attn._indexer_aux_active(cache_view)
        attn_result = self.self_attn(
            self.input_layernorm(collapsed),
            position_embeddings,
            position_ids,
            token_bias,
            cache_view=cache_view,
            packed_meta=packed_meta,
            return_indexer_aux_loss=aux_active,
        )
        indexer_aux_loss = None
        if aux_active:
            attn_output, cache_view, indexer_aux_loss = attn_result
        else:
            attn_output, cache_view = attn_result
        hidden_streams = manifold_residual_write(hidden_streams, attn_output, post, comb, precision=self.precision)

        post, comb, collapsed = self.ffn_hc(hidden_streams)
        mlp_output, router_logits = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        hidden_streams = manifold_residual_write(hidden_streams, mlp_output, post, comb, precision=self.precision)

        return _DeepseekV4DecoderLayerOutput(
            hidden_states=hidden_streams,
            attention_weight=None,
            cache_view=cache_view,
            router_logits=router_logits if output_router_logits else None,
            indexer_aux_loss=indexer_aux_loss,
        )


def _reject_nonempty_cached_prefill(past_key_values) -> None:
    """Reject multi-token continuation that would overwrite compressor state."""
    cache_position = past_key_values[0].cache_position
    raise_if_runtime_true(
        jnp.any(cache_position != 0),
        "DeepSeek-V4 cached multi-token prefill requires an empty cache; "
        "use one-token decode or reset the cache before prefill.",
    )


@auto_pytree
class DeepseekV4ModelOutput(MoeModelOutput):
    """:class:`MoeModelOutput` plus the DSpark main hidden states (only with ``output_dspark_hidden=True``).

    Attributes:
        main_hidden: ``[B, S, len(dspark_target_layer_ids) * D]`` concatenated target-layer output stream means.
    """

    main_hidden: Array | None = None


@auto_pytree
class DeepseekV4CausalLMOutput(MoeCausalLMOutput):
    """:class:`MoeCausalLMOutput` plus ``main_hidden`` (see :class:`DeepseekV4ModelOutput`)."""

    main_hidden: Array | None = None


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

    fp32_param_patterns: tp.ClassVar[tuple[str, ...]] = ("e_score_correction_bias",)

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
            self.hc_head = DeepseekV4HyperHead(
                config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
            )
            self.norm = RMSNorm(
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
        output_dspark_hidden: bool = False,
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
            output_dspark_hidden: Also return ``main_hidden`` ``[B, S, T * D]``: for every ``i`` in
                ``config.dspark_target_layer_ids`` (ascending) the stream mean of layer ``i``'s OUTPUT (reference
                ``Transformer.forward`` ``main_hiddens``; the DSpark draft model's input). The output is then a
                :class:`DeepseekV4ModelOutput`. Off by default (no extra work, unchanged program).

        Returns:
            MoeModelOutput with the collapsed final hidden state and, in
            cached mode, the updated cache in ``past_key_values``. During
            stateless CSA warm-up, ``indexer_aux_loss`` holds the unscaled
            per-layer KL sum; the task head applies the coefficient once.

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
        dspark_targets: frozenset[int] = frozenset()
        if output_dspark_hidden:
            dspark_targets = frozenset(int(i) for i in (getattr(self.config, "dspark_target_layer_ids", None) or ()))
            if not dspark_targets or max(dspark_targets) >= len(self.layers) or min(dspark_targets) < 0:
                raise ValueError(
                    "output_dspark_hidden requires `dspark_target_layer_ids` inside "
                    f"[0, {len(self.layers)}); got {sorted(dspark_targets)}."
                )

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
        if not is_packed and sequence_length > 1:
            reject_packed_or_padded_batch(mask_info, attention_mask, family="DeepSeek-V4")
            if not cached and packed_qsl is not None and packed_num_seqs is not None and batch == 1:
                raise ValueError(
                    "DeepSeek-V4 stateless packed-document calls are not supported: pass a cache or an "
                    "unpacked batch (CSA/HCA compression follows physical offsets)."
                )

        packed_meta = None
        if is_packed:
            if position_ids is None:
                raise ValueError("Packed eSurge serving requires explicit position_ids for the token stream.")
            packed_meta = build_packed_slot_meta(
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
                _reject_nonempty_cached_prefill(past_key_values)
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
            token_bias = jnp.where(allowed, 0.0, MASK_MIN).astype(jnp.float32)

        position_embeddings = {
            "main": labeled_rope_cos_sin(self.config, "main", position_ids, inputs_embeds.dtype),
            "compress": labeled_rope_cos_sin(self.config, "compress", position_ids, inputs_embeds.dtype),
        }

        hidden_streams = jnp.broadcast_to(
            inputs_embeds[:, :, None, :],
            (batch, sequence_length, self.config.hc_mult, self.config.hidden_size),
        ).astype(inputs_embeds.dtype)
        hidden_streams = apply_logical_sharding(
            hidden_streams,
            dynamic_axes=HyperStreamSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )

        def _layer_loop(block, carry):
            """Run one decoder layer inside the (python-loop) layer scan."""
            hidden_streams, all_hidden_states, all_router_logits, captures, indexer_aux_loss, idx = carry
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
            if idx in dspark_targets:
                # DSpark main hidden: the stream mean of the target layer's OUTPUT (V4.1 reads its input instead).
                captures += (hidden_streams.mean(axis=2),)
            layer_aux = layer_outputs.indexer_aux_loss
            if layer_aux is not None:
                indexer_aux_loss = layer_aux if indexer_aux_loss is None else indexer_aux_loss + layer_aux
            return hidden_streams, all_hidden_states, all_router_logits, captures, indexer_aux_loss, idx + 1

        # NOTE: layers are structurally heterogeneous (per-layer compressor /
        # rope selection), so this always uses the python-loop path;
        # `scan_layers=True` is unsupported for this family.
        hidden_streams, all_hidden_states, all_router_logits, captures, indexer_aux_loss, _ = self.layers.scan(
            _layer_loop,
            (hidden_streams, all_hidden_states, all_router_logits, () if output_dspark_hidden else None, None, 0),
            trace=True,
            **self._layer_scan_kwargs("text"),
        )
        hidden_states = self.norm(self.hc_head(hidden_streams))

        if output_dspark_hidden:
            return DeepseekV4ModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=None,
                router_logits=all_router_logits,
                past_key_values=past_key_values if cached else None,
                indexer_aux_loss=indexer_aux_loss,
                main_hidden=jnp.concatenate(captures, axis=-1),  # appended in ascending layer order
            )

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=None,
            router_logits=all_router_logits,
            past_key_values=past_key_values if cached else None,
            indexer_aux_loss=indexer_aux_loss,
        )

    def create_compressed_window_cache_config(
        self,
        batch_size: int,
        max_length: int,
        spec_trace_len: int = 0,
    ) -> CompressedWindowCacheConfig:
        """Build the cache configuration for this model's layer schedule.

        Args:
            batch_size: Number of lockstep sequences.
            max_length: Maximum supported stream length (prefill + decode).
            spec_trace_len: Projection trace length for replay-free speculative rollback (``0`` = no traces).

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
            spec_trace_len=spec_trace_len,
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
            **kwargs: Engine-oriented arguments. ``spec_trace_len`` (eSurge
                speculative decoding) sizes the replay-free rollback traces;
                the rest (page_size, hbm_utilization, quantizer, ...) are ignored.

        Returns:
            CompressedWindowCache: Initialized cache for all layers.
        """
        spec_trace_len = int(kwargs.pop("spec_trace_len", 0) or 0)
        del starts, kwargs
        return CompressedWindowCache.init_cache(
            config=self.create_compressed_window_cache_config(
                batch_size=batch_size, max_length=max_length, spec_trace_len=spec_trace_len
            ),
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


# ---------------------------------------------------------------- checkpoint keys
# DeepSeek publishes V4 (Flash and Pro) only in its own inference naming (``layers.N.attn.wq_a``,
# ``layers.N.ffn.experts.<i>.w1``); the transformers form is the *input* of its ``inference/convert.py`` and was never
# released. These rules are the inverse of that script's renames. Per-expert stacking and gate/up fusion stay in
# ``reform_param``.

#: Module-name renames applied to the trailing module (before ``.weight`` / ``.scale``, else the last segment).
DEEPSEEK_V4_LEAF_ALIASES: dict[str, str] = {
    # MLA attention projections
    "wq_a": "q_a_proj",
    "wq_b": "q_b_proj",
    "wkv": "kv_proj",
    "wo_a": "o_a_proj",
    "wo_b": "o_b_proj",
    "q_norm": "q_a_norm",
    # the compressor's own sub-projections
    "wgate": "gate_proj",
    "norm": "kv_norm",
    "ape": "position_bias",
    # MoE router
    "bias": "e_score_correction_bias",
    # per-expert SwiGLU triple (w1 = gate, w3 = up, w2 = down)
    "w1": "gate_proj",
    "w3": "up_proj",
    "w2": "down_proj",
}
_V4_ROOT_ALIASES = {
    "embed.weight": "model.embed_tokens.weight",
    "head.weight": "lm_head.weight",
    "norm.weight": "model.norm.weight",
    "hc_head_base": "model.hc_head.hc_base",
    "hc_head_fn": "model.hc_head.hc_fn",
    "hc_head_scale": "model.hc_head.hc_scale",
}
_V4_HC_ALIASES = {
    f"hc_{site}_{leaf}": f"{site}_hc.{leaf}" for site in ("attn", "ffn") for leaf in ("base", "fn", "scale")
}

DEEPSEEK_V4_KEY_RULES = KeyRuleSet(
    [
        Map(r"(?:model\.)?mtp\..*", None),  # the DSpark stack belongs to the draft model
        Map(r"(?:model|lm_head)\..*", KEEP),  # HF / already-normalized names
        Map(
            r"(?P<root>embed\.weight|head\.weight|norm\.weight|hc_head_(?:base|fn|scale))",
            "{root}",
            rename={"root": _V4_ROOT_ALIASES},
        ),
        Map(r"(?!layers\.\d+\.).*", None),
        Map(
            r"layers\.(?P<i>\d+)\.(?P<hc>hc_(?:attn|ffn)_(?:base|fn|scale))",
            "model.layers.{i}.{hc}",
            rename={"hc": _V4_HC_ALIASES},
        ),
        Map(r"layers\.(?P<i>\d+)\.attn\.attn_sink", "model.layers.{i}.self_attn.sinks"),
        # The lightning indexer nests the other way round: the checkpoint puts the compressor inside the indexer
        # (``attn.indexer.compressor.wkv``), EasyDeL the indexer inside the compressor, with scoring under ``scorer``.
        Rewrite(r"^(?P<p>layers\.\d+\.)attn\.indexer\.compressor\.", "{p}self_attn.compressor.indexer."),
        Rewrite(
            r"^(?P<p>layers\.\d+\.)attn\.indexer\.weights_proj", "{p}self_attn.compressor.indexer.scorer.weights_proj"
        ),
        Rewrite(r"^(?P<p>layers\.\d+\.)attn\.indexer\.", "{p}self_attn.compressor.indexer."),
        Rewrite(r"\.attn\.", ".self_attn."),
        Rewrite(r"\.ffn\.", ".mlp."),
        Rewrite(r"\.attn_norm\.", ".input_layernorm."),
        Rewrite(r"\.ffn_norm\.", ".post_attention_layernorm."),
        Map(
            r"layers\.(?P<i>\d+)\.(?P<path>(?:.*\.)?)(?P<leaf>[^.]*)\.(?P<suffix>weight|scale)",
            "model.layers.{i}.{path}{leaf}.{suffix}",
            rename={"leaf": DEEPSEEK_V4_LEAF_ALIASES},
        ),
        Map(
            r"layers\.(?P<i>\d+)\.(?P<path>(?:.*\.)?)(?P<leaf>[^.]*)",
            "model.layers.{i}.{path}{leaf}",
            rename={"leaf": DEEPSEEK_V4_LEAF_ALIASES},
        ),
    ],
    name="deepseek_v4",
)


def mtp_stage_block_rule(block_rules: KeyRuleSet) -> Nested:
    """``mtp.N.<block key>`` -> ``stages.N.<name>``: a DSpark stage block reuses the backbone's ``layers.N`` rules."""
    return Nested(
        r"mtp\.(?P<s>\d+)\.(?P<r>.+)",
        block_rules,
        inner_key="layers.{s}.{r}",
        inner_result=r"model\.layers\.{s}\.(?P<tail>.*)",
        target="stages.{s}.{tail}",
    )


#: Native DSpark keys (``mtp.N.*``) -> :class:`DeepseekV4DSpark` names. The tied ``mtp.N.embed`` / ``mtp.N.head``
#: (shared with the backbone) are unowned.
DEEPSEEK_V4_MTP_KEY_RULES = KeyRuleSet(
    [
        Map(r"stages\..*", KEEP),
        Map(r"mtp\.\d+\.(?:embed|head)\..*", None),
        Map(
            r"mtp\.(?P<s>\d+)\.(?P<r>main_proj\.(?:weight|scale)|main_norm\.weight|norm\.weight"
            r"|markov_head\.markov_w[12]\.weight|confidence_head\.proj\.weight)",
            "stages.{s}.{r}",
        ),
        Map(
            r"mtp\.(?P<s>\d+)\.(?P<hc>hc_head_(?:fn|base|scale))",
            "stages.{s}.{hc}",
            rename={"hc": {f"hc_head_{leaf}": f"hc_head.hc_{leaf}" for leaf in ("fn", "base", "scale")}},
        ),
        mtp_stage_block_rule(DEEPSEEK_V4_KEY_RULES),
    ],
    name="deepseek_v4_mtp",
)


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
    # DeepSeek publishes V4 only in its own inference naming (see DEEPSEEK_V4_KEY_RULES).
    checkpoint_key_rules: tp.ClassVar[KeyRuleSet] = DEEPSEEK_V4_KEY_RULES

    fp32_param_patterns: tp.ClassVar[tuple[str, ...]] = ("e_score_correction_bias",)

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
        output_dspark_hidden: bool = False,
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
            output_dspark_hidden: Also return the DSpark ``main_hidden`` (see :meth:`DeepseekV4Model.forward`); the
                output is then a :class:`DeepseekV4CausalLMOutput`.

        Returns:
            MoeCausalLMOutput with logits and optional per-layer outputs.
        """
        if output_dspark_hidden:
            if output_router_logits is None:
                output_router_logits = getattr(self.config, "output_router_logits", False)
            outputs = self.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                mask_info=mask_info,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_router_logits=output_router_logits,
                mode=mode,
                past_key_values=past_key_values,
                cache_metadata=cache_metadata,
                output_dspark_hidden=True,
            )
            logits = None
            if apply_lm_head:
                logits = self.compute_lm_logits(self.prepare_lm_head_inputs(outputs.last_hidden_state))
            return DeepseekV4CausalLMOutput(
                aux_loss=self.compute_aux_loss(outputs, attention_mask=attention_mask),
                indexer_aux_loss=outputs.indexer_aux_loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                last_hidden_state=outputs.last_hidden_state,
                attentions=None,
                router_logits=outputs.router_logits,
                past_key_values=outputs.past_key_values,
                main_hidden=outputs.main_hidden,
            )
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
    "DeepseekV4CausalLMOutput",
    "DeepseekV4DecoderLayer",
    "DeepseekV4ForCausalLM",
    "DeepseekV4Model",
    "DeepseekV4ModelOutput",
    "DeepseekV4SparseMoeBlock",
]
