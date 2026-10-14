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

"""DeepSeek-V4.1 text model (``model_type="deepseek_v41"``).

Semantics follow DeepSeek's reference ``inference/model.py`` (see ``.xerxes/projects/deepseek-v41/SPEC.md``).
Relative to DeepSeek-V4:

- Per-layer integer compression ratios. Only ``kv_source_layer_ids`` run a compressor; every later layer with the
  same ratio attends to that source's compressed KV. Compression is a plain softmax-gated pooling over
  non-overlapping groups (ratio 1 is a projection), with no position bias and no overlapping second series.
- Each query attends to its sliding window of token KV plus at most ``index_topk`` compressed entries chosen by
  the indexer of the most recent index source. Index keys are ``k_norm(wk(latent))`` of the KV source's pre-RoPE
  latent. Index sources after ``candidate_source_layer_id`` only score inside its published candidate blocks.
- No per-head query RMSNorm.
- Hyper-connection read gates are applied one sublayer late: attention collapses the streams with the previous
  layer's FFN ``pre``, the FFN with this layer's attention ``pre``; there is no learned ``hc_head``.
- Engram n-gram embeddings on ``engram_layer_ids``. The n-gram hashing runs on the host (``easydel.layers.engram``); its ids
  are passed in as ``engram_hash_ids``.
- MoE is V4's learned top-k block on every layer (no hash routing).

Cross-layer state (compressed KV, index keys, top-k selection, candidate blocks) is threaded through the layer loop
as an explicit carry rather than global slots.

Execution modes:

- stateless forward (training / scoring);
- cached prefill from an empty :class:`~easydel.caching.compressed_window.SharedCompressedWindowCache` (same math as
  the stateless forward, plus writes of the window ring, the sources' compressed entries / index keys and the open
  compression group) and single-token cached decode (per-row ring write, group close, indexer over the source's
  stored keys, attention over the ring plus the gathered selected entries). Rows are independent streams but
  prompts must not be left-padded.

Vision-language path (``DeepseekV41ForConditionalGeneration``): ViT + aligner features replace the span of image
tokens (``token_types >= 0``), image-span tokens route with ``gate.bias_vl`` instead of the correction bias and take no
part in the engram (reference ``Transformer.forward``). eSurge multimodal serving is NOT wired.

Optional features (config fields, defaults preserve the reference behaviour): ``output_dspark_hidden`` forward kwarg
(DSpark main-hidden hook), ``engram_table_memory_kind="pinned_host"`` (engram tables in host memory) and
``indexer_aux_loss_coef`` (indexer KL warm-up objective, stateless forward only).

Known gap: the reference's fp8/fp4 activation fake-quantization. In the stateless / prefill path the
compressed-entry selection is applied as a dense ``[B, 1, S, S + T]`` bias (exact, quadratic in sequence length); a
gather-based sparse kernel is the planned optimization.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx
from eformer.pytree import auto_pytree
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec as Ps
from jaxtyping import Array, Bool, Float, Int
from spectrax import apply_logical_sharding, common_types, nn

from easydel.caching.compressed_window import (
    SharedCompressedWindowCache,
    SharedCompressedWindowCacheConfig,
    SharedCompressedWindowCacheView,
    slot_partition_axes,
)
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import MoeCausalLMOutput, MoeModelOutput, VLMCausalLMOutput
from easydel.infra.utils import ArrayParam, auto_remat
from easydel.layers import Embed, RMSNorm, clamped_swiglu
from easydel.layers.attention import CompressedWindowAttention
from easydel.layers.compressed_window import (
    DECODE_ATTEND_PLATFORM,
    MASK_MIN,
    TRAIN_ATTEND_PLATFORM,
    GatedPoolCompressor,
    GatedPoolCompressorConfig,
    PackedSlotMeta,
    block_sink_attention,
    block_token_bias,
    build_packed_slot_meta,
    gather_selected_entries,
    grid_to_tokens,
    occupied_columns,
    packed_block_size,
    packed_row_of_token,
    raise_if_runtime_true,
    reject_packed_or_padded_batch,
    ring_prefill_write,
    ring_slot_positions,
    ring_visibility_bias,
    run_column_blocks,
    tokens_to_grid,
)
from easydel.layers.engram import ENGRAM_TABLE_KEY_RE, EngramModelMixin, GatedEngram
from easydel.layers.indexer import (
    IndexerScoreInputs,
    SharedKeyIndexer,
    SharedKeyIndexerConfig,
    select_candidate_blocks,
    selection_bias,
)
from easydel.layers.residual import (
    HyperStreamSharding,
    ManifoldHyperConnection,
    ManifoldHyperConnectionConfig,
    manifold_collapse,
    manifold_residual_write,
)
from easydel.layers.rotary import apply_rope_interleaved_partial, labeled_rope_cos_sin
from easydel.modules._base import BaseCausalLMModule, BaseVisionLanguageModule
from easydel.modules.deepseek_v4.modeling_deepseek_v4 import (
    DEEPSEEK_V4_KEY_RULES,
    DEEPSEEK_V4_LEAF_ALIASES,
    DeepseekV4SparseMoeBlock,
    mtp_stage_block_rule,
)
from easydel.utils.checkpoint_keys import KEEP, KeyRuleSet, Map

from .deepseek_v41_configuration import DeepseekV41Config
from .modeling_deepseek_v41_vision import (
    TOKEN_TYPE_IMAGE,
    TOKEN_TYPE_IMAGE_END,
    TOKEN_TYPE_IMAGE_NEW_LINE,
    TOKEN_TYPE_IMAGE_START,
    DeepseekV41Aligner,
    DeepseekV41VisionModel,
)

# Attend platforms (shared defaults in ``easydel.layers.compressed_window``); module globals so tests can pin a
# backend per family.
_DECODE_KERNEL_PLATFORM = DECODE_ATTEND_PLATFORM
_TRAIN_KERNEL_ENABLED = True
_TRAIN_KERNEL_PLATFORM = TRAIN_ATTEND_PLATFORM

_SharedState = dict[str, Array | None]
_SHARED_KEYS = ("compress_kv", "index_k", "topk", "candidates", "indexer_aux")


def _ced_decoder_tail(config: DeepseekV41Config) -> tuple[int, int] | None:
    """``(first_tail_layer, reach)`` of the window-only decoder tail, or ``None`` when there is none.

    The tail is every layer after the last KV source. Those layers read the shared compressed entries / index keys
    but store nothing beyond their sliding-window ring and stream position, so prompt tokens far from the end of a
    request only matter through the rings. With window ``W`` and ``k`` tail layers, the last token's output needs tail
    layer ``j`` (counted from the end, ``j = 0`` the last) at positions ``>= N - W - j * (W - 1)``, so the first tail
    layer's input is needed only for ``tokens_to_end = N - 1 - p < W + (k - 1) * (W - 1)``. The decode rings that the
    next steps read lie inside the same range, and the first tail layer's input is the output of the last KV source,
    which always runs: skipping the tail on steps whose tokens all lie outside ``reach`` is exact for every output a
    serving engine keeps (the last token's logits, later decode steps, DSpark main hidden of the last window).
    """
    num_layers = int(config.num_hidden_layers)
    sources = [int(s) for s in config.kv_source_layer_ids if int(s) < num_layers]
    start = max(sources) + 1 if sources else 0
    if start >= num_layers or any(config.owns_index_keys(i) for i in range(start, num_layers)):
        return None
    window = int(config.sliding_window)
    return start, window + (num_layers - start - 1) * (window - 1)


#: A CED step whose live tokens fit in ``bucket // _CED_COMPACT_DIVISOR`` (at least ``num_slots``) runs the decoder
#: tail on that many compacted tokens instead of the whole bucket.
_CED_COMPACT_DIVISOR = 8


class _CedCompactPlan(tp.NamedTuple):
    """Packed sub-stream of a step's live tokens (see :func:`_ced_compact_plan`).

    Attributes:
        token_index: ``[L]`` int32 packed index of each compacted token; ``T`` (out of range) past the live count.
        meta: Packed token-to-slot mapping of the compacted stream.
        col_offset: ``[num_slots]`` int32 grid column of each slot's first live token in the full step.
    """

    token_index: Array
    meta: PackedSlotMeta
    col_offset: Array


def _ced_compact_plan(
    live: Bool[Array, " tokens"],
    query_start_loc: Array,
    num_seqs: Array,
    recurrent_state_indices: Array | None,
    position_ids: Array,
    num_slots: int,
    length: int,
) -> _CedCompactPlan:
    """Compact a packed step to its live tokens, keeping each request's tokens contiguous and in order.

    A request's live tokens are a suffix of its scheduled chunk (``tokens_to_end`` falls with position), so the
    compacted stream is again a valid packed stream: row ``r`` keeps its last ``live_r`` tokens at their real positions,
    and every per-column grid of the full step maps to it by shifting slot ``s`` left by ``col_offset[s]``.

    Args:
        live: ``[T]`` bool, real tokens the decoder tail must compute.
        query_start_loc: ``[R + 1]`` cumulative scheduled-token offsets of the full step.
        num_seqs: ``[1]`` live request count.
        recurrent_state_indices: Cache slot per request row, or ``None`` for row identity.
        position_ids: ``[1, T]`` packed positions.
        num_slots: Static number of cache slots.
        length: Static compacted length ``L`` (callers ensure the live count fits).

    Returns:
        The compaction plan.
    """
    total = live.shape[0]
    qsl = query_start_loc.astype(jnp.int32).reshape(-1)
    num_rows = qsl.shape[0] - 1
    token_idx = jnp.arange(total, dtype=jnp.int32)
    row_of_token = packed_row_of_token(qsl, token_idx)
    live_per_row = jax.ops.segment_sum(live.astype(jnp.int32), row_of_token, num_segments=num_rows)
    new_qsl = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(live_per_row).astype(jnp.int32)])
    index = jnp.flatnonzero(live, size=length, fill_value=total).astype(jnp.int32)
    positions = jnp.take(position_ids.astype(jnp.int32).reshape(-1), index, mode="clip")[None]
    meta = build_packed_slot_meta(
        query_start_loc=new_qsl,
        num_seqs=num_seqs,
        recurrent_state_indices=recurrent_state_indices,
        position_ids=positions,
        num_slots=num_slots,
    )
    if recurrent_state_indices is None:
        slots_by_row = jnp.arange(num_rows, dtype=jnp.int32)
    else:
        slots_by_row = recurrent_state_indices.astype(jnp.int32).reshape(-1)[:num_rows]
    live_rows = jnp.clip(num_seqs.reshape(-1)[0].astype(jnp.int32), 0, num_rows)
    target = jnp.where(jnp.arange(num_rows, dtype=jnp.int32) < live_rows, slots_by_row, num_slots)
    dead_prefix = (qsl[1:] - qsl[:-1]) - live_per_row
    col_offset = jnp.zeros((num_slots,), jnp.int32).at[target].set(dead_prefix, mode="drop")
    return _CedCompactPlan(index, meta, col_offset)


def _ced_shift_grid(grid: Array, col_offset: Array) -> Array:
    """Shift a per-column ``[num_slots, width, ...]`` grid left by ``col_offset[slot]`` (same width, edge-clipped)."""
    width = grid.shape[1]
    cols = jnp.clip(jnp.arange(width, dtype=jnp.int32)[None, :] + col_offset[:, None], 0, width - 1)
    return jax.vmap(lambda g, c: jnp.take(g, c, axis=0))(grid, cols)


def _linear_kwargs(config: DeepseekV41Config, dtype, param_dtype, precision, rngs) -> dict[str, tp.Any]:
    """Common bias-free ColumnParallel/RowParallel constructor arguments."""
    return dict(
        use_bias=False,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        kernel_init=jax.nn.initializers.normal(config.initializer_range),
        rngs=rngs,
    )


class DeepseekV41Compressor(GatedPoolCompressor):
    """V4.1 binding of :class:`~easydel.layers.compressed_window.GatedPoolCompressor` (reference ``Compressor``)."""

    def __init__(
        self,
        config: DeepseekV41Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            GatedPoolCompressorConfig(
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                ratio=config.layer_compress_ratio(layer_idx),
                rms_norm_eps=config.rms_norm_eps,
                initializer_range=config.initializer_range,
            ),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )


class DeepseekV41Indexer(SharedKeyIndexer):
    """Chooses up to ``index_topk`` compressed entries per query (reference ``Indexer``).

    ``score[t] = sum_h relu(q_h . k_t) * w_h`` with ``w = weights_proj(x) * index_head_dim^-0.5 * n_heads^-0.5``;
    entries beyond the query's reach ``(position + 1) // ratio`` are excluded. Selection is a hard top-k (the selection
    carries no gradient, as in the reference's inference-mode module); with ``indexer_aux_loss_coef > 0`` the model
    trains the scores through :func:`indexer_distillation_loss`.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            SharedKeyIndexerConfig(
                hidden_size=config.hidden_size,
                q_lora_rank=config.q_lora_rank,
                index_n_heads=config.index_n_heads,
                index_head_dim=config.index_head_dim,
                index_topk=config.index_topk,
                ratio=config.layer_compress_ratio(layer_idx),
                key_input_dim=config.head_dim,
                owns_keys=config.owns_index_keys(layer_idx),
                is_candidate_source=layer_idx == config.candidate_source_layer_id,
                uses_candidates=config.uses_candidates(layer_idx),
                candidate_topk_blocks=config.candidate_topk_blocks,
                candidate_block_size=config.candidate_block_size,
                rms_norm_eps=config.rms_norm_eps,
                initializer_range=config.initializer_range,
            ),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        # Mesh source for the replicated-key Pallas selector dispatch (same pattern as the attention module).
        self.config = config

    def index_keys(self, latent: Array, cos: Array, sin: Array) -> Array:
        """Index keys ``[B, T, index_head_dim]`` from the KV source's pre-RoPE latent, roped at group positions."""
        return self.keys_from_latent(
            latent, lambda keys: apply_rope_interleaved_partial(keys[:, None], cos, sin, unsqueeze_dim=1)[:, 0]
        )

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        q_residual: Float[Array, "batch seq q_lora"],
        index_k: Array | None,
        cos: Array,
        sin: Array,
        candidates: Array | None,
        positions: Int[Array, "batch seq"],
        return_scores: bool = False,
        slot_axes: tp.Sequence[str] = (),
    ) -> tuple[Array, Array | None] | tuple[Array, Array | None, Array]:
        """Return ``(topk [B, S, k] int32 with -1 padding, published candidate block ids or None)``.

        Selection runs through the shared context-bounded selectors in :mod:`easydel.layers.indexer` and reads only
        entries below each query's ``(position + 1) // ratio``. With ``return_scores`` a third element carries the
        differentiable reachability-masked scores ``[B, S, T]``. ``slot_axes`` (from a slot-sharded cache) makes the
        selection slot-local.
        """
        return super().forward(
            hidden_states,
            q_residual,
            index_k,
            lambda q: apply_rope_interleaved_partial(q, cos, sin, unsqueeze_dim=2),
            candidates,
            positions,
            return_scores=return_scores,
            kernel_mesh=getattr(self.config, "mesh", None),
            slot_axes=slot_axes,
        )


def _window_token_bias(token_valid: Bool[Array, "batch seq"], window: int) -> Float[Array, "batch 1 seq seq"]:
    """Dense additive bias of the token region: key ``k`` visible to query ``q`` iff ``q - window < k <= q`` and valid.

    Only the dense fallback paths need this; the registered compressed-window op builds the same mask in-kernel from
    ``token_valid``.
    """
    seq = token_valid.shape[1]
    q_idx = jnp.arange(seq)[:, None]
    k_idx = jnp.arange(seq)[None, :]
    allowed = ((k_idx <= q_idx) & (k_idx > q_idx - window))[None, None] & token_valid.astype(bool)[:, None, None, :]
    return jnp.where(allowed, 0.0, MASK_MIN).astype(jnp.float32)


def indexer_distillation_loss(
    q: Float[Array, "batch heads seq dim"],
    window_kv: Float[Array, "batch seq dim"],
    entries: Float[Array, "batch entries dim"],
    sinks: Float[Array, "heads"],
    token_valid: Bool[Array, "batch seq"],
    scores: Float[Array, "batch seq entries"] | IndexerScoreInputs,
    *,
    window: int,
    scaling: float,
    precision: jax.lax.PrecisionLike = None,
    block_q: int = 64,
) -> Array:
    """Dense indexer warm-up objective ``KL(stop_grad(p) || softmax(scores))`` (DeepSeek DSA indexer training).

    ``p`` is the layer's dense attention distribution over the compressed entries: the sink-augmented softmax over
    ``[window tokens | compressed entries | sink]`` (all entries a query can reach, not only the top-k), summed over
    attention heads and L1-normalized over the entries. The indexer side is ``softmax`` of its scores over the same
    reachable entries. Everything on the attention side is under ``stop_gradient``, so only the indexer scores
    receive gradient. Queries that reach no entry contribute nothing; the loss is the mean KL over the queries that do.

    Args:
        q: Roped queries ``[B, H, S, D]``.
        window_kv: Roped token K/V (K == V) ``[B, S, D]``.
        entries: Roped compressed entries ``[B, T, D]``.
        sinks: Per-head attention sinks ``[H]``.
        token_valid: Per-key validity of the token region ``[B, S]``; key ``k`` is visible to query ``s`` iff
            ``s - window < k <= s`` and ``token_valid[k]``.
        scores: Indexer scores ``[B, S, T]`` with ``-inf`` at unreachable entries, or lazy
            :class:`~easydel.layers.indexer.IndexerScoreInputs` evaluated per query tile (no ``[B, S, T]`` score
            tensor in either AD direction).
        window: Sliding-window size.
        scaling: Attention logit scale.
        precision: Matmul precision.
        block_q: Queries per tile. Each tile only touches its ``block_q + window - 1`` token keys plus the entries,
            so the token part costs ``O(S * window)`` and no ``[B, H, S, S]`` tensor exists.

    Returns:
        Scalar fp32 loss.
    """
    q = jax.lax.stop_gradient(q)
    window_kv = jax.lax.stop_gradient(window_kv)
    entries = jax.lax.stop_gradient(entries).astype(window_kv.dtype)
    sinks = jax.lax.stop_gradient(sinks).astype(jnp.float32)
    batch, heads, seq, _ = q.shape
    num_entries = entries.shape[1]
    tile = max(1, min(block_q, seq))
    span = min(tile + window - 1, seq)
    tiles = -(-seq // tile)
    pad = tiles * tile - seq
    q_padded = jnp.pad(q, ((0, 0), (0, 0), (0, pad), (0, 0)))
    valid = token_valid.astype(bool)
    sink = sinks.reshape(1, heads, 1, 1)

    if isinstance(scores, IndexerScoreInputs):
        # Scores are evaluated per tile from the indexer's queries; padded rows get position -1 (reach nothing).
        iq = jnp.pad(scores.q, ((0, 0), (0, pad), (0, 0), (0, 0)))
        iw = jnp.pad(scores.weights, ((0, 0), (0, pad), (0, 0)))
        ipos = jnp.pad(scores.positions.astype(jnp.int32), ((0, 0), (0, pad)), constant_values=-1)

        def _tile_scores(index):
            start = index * tile
            return scores.rows(
                jax.lax.dynamic_slice_in_dim(iq, start, tile, axis=1),
                jax.lax.dynamic_slice_in_dim(iw, start, tile, axis=1),
                jax.lax.dynamic_slice_in_dim(ipos, start, tile, axis=1),
            )

    else:
        dense_padded = jnp.pad(scores, ((0, 0), (0, pad), (0, 0)), constant_values=-jnp.inf)

        def _tile_scores(index):
            return jax.lax.dynamic_slice_in_dim(dense_padded, index * tile, tile, axis=1)

    @jax.checkpoint
    def _tile_kl(index):
        """``(sum of per-query KL, number of queries reaching an entry)`` for queries in tile ``index``.

        The target is normalized per query row, so the KL is complete inside the tile: no ``[B, S, T]`` target or
        score tensor exists in either AD direction (the tile is rematerialized in the backward pass).
        """
        start = index * tile
        lo = jnp.clip(start - window + 1, 0, seq - span)
        q_t = jax.lax.dynamic_slice_in_dim(q_padded, start, tile, axis=2)
        k_t = jax.lax.dynamic_slice_in_dim(window_kv, lo, span, axis=1)
        key_ok = jax.lax.dynamic_slice_in_dim(valid, lo, span, axis=1)
        scores_t = _tile_scores(index)  # [B, tile, T], carries the indexer gradient
        reach_t = jax.lax.stop_gradient(scores_t) > -jnp.inf
        q_pos = start + jnp.arange(tile, dtype=jnp.int32)[:, None]
        k_pos = lo + jnp.arange(span, dtype=jnp.int32)[None, :]
        allowed = ((k_pos <= q_pos) & (k_pos > q_pos - window))[None] & key_ok[:, None, :]  # [B, tile, span]
        window_logits = jnp.einsum("bhsd,btd->bhst", q_t, k_t, precision=precision).astype(jnp.float32) * scaling
        window_logits = jnp.where(allowed[:, None], window_logits, MASK_MIN)
        entry_logits = jnp.einsum("bhsd,btd->bhst", q_t, entries, precision=precision).astype(jnp.float32)
        entry_logits = jnp.where(reach_t[:, None], entry_logits * scaling, MASK_MIN)
        sinks_t = jnp.broadcast_to(sink, (batch, heads, tile, 1))
        combined = jnp.concatenate([window_logits, entry_logits, sinks_t], axis=-1)
        probs = jax.nn.softmax(combined - jnp.max(combined, axis=-1, keepdims=True), axis=-1)
        target = jnp.sum(probs[..., span : span + num_entries], axis=1)  # [B, tile, T], summed over heads
        target = jax.lax.stop_gradient(target / jnp.maximum(jnp.sum(target, axis=-1, keepdims=True), 1e-30))
        log_pred = jax.nn.log_softmax(jnp.where(reach_t, scores_t, MASK_MIN).astype(jnp.float32), axis=-1)
        positive = target > 0
        kl = jnp.sum(jnp.where(positive, target * (jnp.log(jnp.where(positive, target, 1.0)) - log_pred), 0.0), axis=-1)
        row_valid = jnp.any(reach_t, axis=-1).astype(jnp.float32)
        return jnp.sum(kl * row_valid), jnp.sum(row_valid)

    kl_sums, counts = jax.lax.map(_tile_kl, jnp.arange(tiles, dtype=jnp.int32))
    return jnp.sum(kl_sums) / jnp.maximum(jnp.sum(counts), 1.0)


class DeepseekV41Attention(CompressedWindowAttention):
    """Shared-KV MQA with sinks over a sliding window plus indexer-selected compressed entries.

    Query ``q_b_proj(q_a_norm(q_a_proj(x)))`` (no per-head norm), single KV head ``kv_norm(kv_proj(x))`` used as
    both key and value, RoPE on the trailing ``qk_rope_head_dim`` channels (``compress`` rope on ratio > 0 layers,
    ``main`` otherwise), inverse RoPE on the output, grouped output projection ``o_a_proj`` + ``o_b_proj``.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            rope_label=config.rope_label(layer_idx),
        )
        self.ratio = config.layer_compress_ratio(layer_idx)
        layer_kwargs = dict(dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        self.compressor = (
            DeepseekV41Compressor(config, layer_idx, **layer_kwargs) if config.is_kv_source(layer_idx) else None
        )
        self.indexer = (
            DeepseekV41Indexer(config, layer_idx, **layer_kwargs) if config.is_index_source(layer_idx) else None
        )

    def _attend(self, q: Array, kv: Array, token_valid: Array, entry_topk: Array | None) -> Array:
        """Full-sequence attention over ``[own token KVs | compressed entries]``.

        Args:
            q: Rotated queries ``[B, H, S, D]``.
            kv: ``[B, S + T, D]``: the sequence's own token KVs, then the compressed entries (``T`` may be ``0``).
            token_valid: Per-key validity of the token region ``[B, S]``.
            entry_topk: Selected entry ids ``[B, S, k]`` (``-1`` padded), or ``None`` when there are no entries.
        """
        if _TRAIN_KERNEL_ENABLED:
            # The op builds the window / causal / padding mask from token_valid and the entry selection from
            # entry_topk, so neither a [B, S, S] nor a [B, S, T] bias (or its cotangent) is materialised.
            return self.fused_full_attend(
                q,
                kv,
                jnp.zeros((q.shape[0], q.shape[2], 0), jnp.float32),
                platform=_TRAIN_KERNEL_PLATFORM,
                token_valid=token_valid,
                entry_topk=entry_topk,
            )
        batch, _, seq, _ = q.shape
        num_entries = kv.shape[1] - seq
        token_bias = _window_token_bias(token_valid, self.sliding_window)
        if entry_topk is None:
            entry_bias = jnp.zeros((batch, 1, seq, num_entries), jnp.float32)
        else:
            entry_bias = selection_bias(entry_topk, num_entries)
        return self.sink_attend(q, kv, jnp.concatenate([token_bias, entry_bias], axis=-1))

    def _write_window_prefill(self, view: SharedCompressedWindowCacheView, kv: Array) -> SharedCompressedWindowCacheView:
        """Seed the ring with the last ``sliding_window`` post-RoPE token KVs of a prefill from position 0."""
        return view.replace(window_kv=ring_prefill_write(view.window_kv, kv, self.sliding_window))

    def _write_source_prefill(
        self,
        view: SharedCompressedWindowCacheView,
        hidden_states: Array,
        compress_kv: Array | None,
        index_k: Array | None,
    ) -> SharedCompressedWindowCacheView:
        """Store a prefill's complete groups (entries, index keys) and the open group's fp32 projections."""
        seq = hidden_states.shape[1]
        ratio = self.ratio
        groups = seq // ratio
        updates: dict[str, Array] = {}
        if compress_kv is not None:
            updates["compressed_kv"] = self._slot(
                view.compressed_kv.at[:, :groups].set(self._slot(compress_kv.astype(view.compressed_kv.dtype), view)),
                view,
            )
            if index_k is not None and view.index_k is not None:
                updates["index_k"] = self._slot(
                    view.index_k.at[:, :groups].set(self._slot(index_k.astype(view.index_k.dtype), view)), view
                )
        remainder = seq - groups * ratio
        if ratio > 1 and remainder:
            kv_tail, score_tail = self.compressor.project(hidden_states[:, groups * ratio :])
            updates["compressor_kv_state"] = view.compressor_kv_state.at[:, :remainder].set(kv_tail)
            updates["compressor_score_state"] = view.compressor_score_state.at[:, :remainder].set(score_tail)
        return view.replace(**updates)

    def _compress_step(
        self,
        hidden_states: Array,
        view: SharedCompressedWindowCacheView,
        position: Array,
        live: Array | None = None,
    ) -> SharedCompressedWindowCacheView:
        """Advance a KV source by one token per row; emit an entry (and index key) where the row's group closes.

        Rows with ``live == False`` (packed padding cells) leave every buffer untouched: their entry writes are
        dropped and only the small open-group state is selected back.
        """
        batch = hidden_states.shape[0]
        rows = jnp.arange(batch)
        dtype = hidden_states.dtype
        ratio = self.ratio
        if live is None:
            live = jnp.ones((batch,), dtype=bool)
        projections = self.compressor.project(hidden_states) if ratio > 1 else None
        latent, kv_state, score_state, closed, entry = self.compressor.decode_step(
            hidden_states, view.compressor_kv_state, view.compressor_score_state, position, projections=projections
        )
        closed = closed & live
        if ratio > 1:
            keep = live[:, None, None]
            kv_state = jnp.where(keep, kv_state, view.compressor_kv_state)
            score_state = jnp.where(keep, score_state, view.compressor_score_state)
            view = view.replace(
                compressor_kv_state=self._replicate(kv_state), compressor_score_state=self._replicate(score_state)
            )
            view = self._write_trace(view, projections, position[:, None], live[:, None])
        # A latent stands for the first token of its group: group g is roped at position g * ratio.
        cos_g, sin_g = labeled_rope_cos_sin(self.config, "compress", (entry * ratio)[:, None], dtype)
        target = jnp.where(closed, entry, view.compressed_kv.shape[1])
        compressed = apply_rope_interleaved_partial(latent[:, None], cos_g, sin_g, unsqueeze_dim=1)[:, 0, 0]
        # Pin the small update rows and the buffers to the cache's slot layout around the scatter: otherwise the tp
        # sharding of the projections leaks into the cache buffers and every step all-gathers them (measured:
        # [8, 262144, 32] -> [8, 262144, 128] gathers, 16% of decode). Same fix as
        # MLARaggedPagesCacheView.constrain_indexer_rows / constrain_indexer_pages.
        updates = {
            "compressed_kv": self._slot(
                view.compressed_kv.at[rows, target].set(
                    self._slot(compressed.astype(view.compressed_kv.dtype), view), mode="drop"
                ),
                view,
            )
        }
        if view.index_k is not None:
            keys = self._slot(self.indexer.index_keys(latent, cos_g, sin_g)[:, 0].astype(view.index_k.dtype), view)
            updates["index_k"] = self._slot(view.index_k.at[rows, target].set(keys, mode="drop"), view)
        return view.replace(**updates)

    def _write_trace(
        self,
        view: SharedCompressedWindowCacheView,
        projections: tuple[Array, Array],
        positions: Array,
        valid: Array,
    ) -> SharedCompressedWindowCacheView:
        """Record each valid cell's fp32 compressor projections at trace slot ``position % trace_len`` (opt-in).

        Feeds :func:`~easydel.caching.compressed_window.commit_speculative_window`; a no-op without a trace. Only a
        verify window (at most ``trace_len`` tokens per row) is ever read back, so wider prefill blocks may alias.
        """
        if view.compressor_trace_kv is None:
            return view
        kv_t, score_t = projections  # fp32 [S, blk, D]
        trace_len = view.compressor_trace_kv.shape[1]
        rows = jnp.arange(positions.shape[0])[:, None]
        slot = jnp.where(valid, positions.astype(jnp.int32) % trace_len, trace_len)
        return view.replace(
            compressor_trace_kv=self._replicate(
                view.compressor_trace_kv.at[rows, slot].set(kv_t.astype(jnp.float32), mode="drop")
            ),
            compressor_trace_score=self._replicate(
                view.compressor_trace_score.at[rows, slot].set(score_t.astype(jnp.float32), mode="drop")
            ),
        )

    def _replicate(self, x: Array | None) -> Array | None:
        """Constrain a cache buffer (or its update rows) to the replicated cache layout on the model mesh."""
        if x is None:
            return None
        return spx.with_sharding_constraint(x, Ps(), mesh=getattr(self.config, "mesh", None))

    def _slot(self, x: Array | None, view: SharedCompressedWindowCacheView | None) -> Array | None:
        """Constrain a slot-major array to the cache's slot layout (replicated when the cache is not slot-sharded)."""
        if x is None:
            return None
        meta = getattr(view, "metadata", None)
        spec = meta.slot_spec(x.ndim) if meta is not None else Ps()
        return spx.with_sharding_constraint(x, spec, mesh=getattr(self.config, "mesh", None))

    @staticmethod
    def _slot_axes(view: SharedCompressedWindowCacheView | None) -> tuple[str, ...]:
        """Mesh axes the cache splits slots over (``()`` = replicated cache)."""
        meta = getattr(view, "metadata", None)
        return tuple(getattr(meta, "slot_axes", ()) or ())

    def _indexer_aux_active(self, cache_view: SharedCompressedWindowCacheView | None) -> bool:
        """Whether this layer adds the indexer auxiliary loss (index source, stateless forward, coef > 0)."""
        return (
            self.indexer is not None
            and cache_view is None
            and self.ratio > 0
            and float(self.config.indexer_aux_loss_coef) > 0.0
        )

    def _full(
        self,
        hidden_states: Array,
        q: Array,
        kv: Array,
        q_residual: Array,
        cos: Array,
        sin: Array,
        position_ids: Array,
        token_valid: Array,
        shared: _SharedState,
        cache_view: SharedCompressedWindowCacheView | None,
    ) -> tuple[Array, _SharedState, SharedCompressedWindowCacheView | None]:
        """Stateless forward / cached prefill over the whole sequence (``token_valid``: per-key validity ``[B, S]``)."""
        seq = hidden_states.shape[1]
        aux_active = self._indexer_aux_active(cache_view)
        if cache_view is not None:
            cache_view = self._write_window_prefill(cache_view, kv)
        entry_topk = None
        if self.ratio > 0:
            shared = dict(shared)
            if self.compressor is not None:
                latent = self.compressor(hidden_states)
                compress_kv = index_k = None
                if latent is not None:
                    groups = latent.shape[1]
                    group_positions = position_ids[:, : groups * self.ratio : self.ratio]
                    cos_g, sin_g = labeled_rope_cos_sin(self.config, "compress", group_positions, hidden_states.dtype)
                    # The indexer reads the latent before RoPE, so its keys are built first.
                    if self.indexer is not None and self.indexer.owns_keys:
                        # With the auxiliary objective the latent is detached: only wk / k_norm learn from it.
                        key_latent = jax.lax.stop_gradient(latent) if aux_active else latent
                        index_k = self.indexer.index_keys(key_latent, cos_g, sin_g)
                    compress_kv = apply_rope_interleaved_partial(latent[:, None], cos_g, sin_g, unsqueeze_dim=1)[:, 0]
                shared["compress_kv"] = compress_kv
                shared["index_k"] = index_k
                if cache_view is not None:
                    cache_view = self._write_source_prefill(cache_view, hidden_states, compress_kv, index_k)
            index_scores = None
            if self.indexer is not None:
                if aux_active:
                    # The indexer reads detached activations: the auxiliary loss trains only its own parameters.
                    topk, published, index_scores = self.indexer(
                        jax.lax.stop_gradient(hidden_states),
                        jax.lax.stop_gradient(q_residual),
                        shared["index_k"],
                        cos,
                        sin,
                        shared["candidates"],
                        position_ids,
                        return_scores=True,
                    )
                else:
                    topk, published = self.indexer(
                        hidden_states, q_residual, shared["index_k"], cos, sin, shared["candidates"], position_ids
                    )
                shared["topk"] = topk
                if published is not None:
                    shared["candidates"] = published
            compress_kv = shared["compress_kv"]
            if compress_kv is not None and compress_kv.shape[1] > 0:
                if index_scores is not None:
                    aux = indexer_distillation_loss(
                        q,
                        kv,
                        compress_kv,
                        self.sinks.value,
                        token_valid,
                        index_scores,
                        window=self.sliding_window,
                        scaling=self.scaling,
                        precision=self.precision,
                    )
                    previous = shared.get("indexer_aux")
                    shared["indexer_aux"] = aux if previous is None else previous + aux
                kv = jnp.concatenate([kv, compress_kv.astype(kv.dtype)], axis=1)
                entry_topk = shared["topk"]
        # Token mask and entry selection are both built inside the op: no [B, S, S] or [B, S, T] bias here.
        attn_output = self._attend(q, kv, token_valid, entry_topk)
        if cache_view is not None:
            cache_view = cache_view.replace(cache_position=cache_view.cache_position + seq)
        return attn_output, shared, cache_view

    def _decode(
        self,
        hidden_states: Array,
        q: Array,
        kv: Array,
        q_residual: Array,
        cos: Array,
        sin: Array,
        position_ids: Array,
        shared: _SharedState,
        view: SharedCompressedWindowCacheView,
        valid: Array | None = None,
    ) -> tuple[Array, _SharedState, SharedCompressedWindowCacheView]:
        """One cached decode step per row: ring write, group close, selection, attention over ring + selected rows.

        Rows whose ``valid`` is ``False`` (packed padding cells) keep their cache state unchanged; their attention
        output is garbage and never read.
        """
        old_position = view.cache_position
        position = position_ids[:, 0].astype(jnp.int32)  # [B]
        live = jnp.ones(position.shape, dtype=bool) if valid is None else valid.astype(bool)
        window = self.sliding_window
        # Padding cells scatter to an out-of-range slot (dropped) instead of writing and being undone by a select over
        # every cache buffer afterwards (that select touched the multi-GB compressed buffers on every step).
        rows = jnp.arange(position.shape[0])
        slot = jnp.where(live, position % window, window)
        ring = view.window_kv.at[rows, slot].set(kv[:, 0].astype(view.window_kv.dtype), mode="drop")
        view = view.replace(window_kv=ring)
        # Slot s of the ring holds the latest stream position congruent to s; negative = not written yet.
        slot_positions = ring_slot_positions(position, window)
        keys = ring.astype(kv.dtype)
        bias = ring_visibility_bias(slot_positions)  # [B, W]
        if self.ratio > 0:
            shared = dict(shared)
            if self.compressor is not None:
                view = self._compress_step(hidden_states, view, position, live)
                shared["compress_kv"] = view.compressed_kv
                shared["index_k"] = view.index_k
            if self.indexer is not None:
                topk, published = self.indexer(
                    hidden_states,
                    q_residual,
                    shared["index_k"],
                    cos,
                    sin,
                    shared["candidates"],
                    position_ids,
                    slot_axes=self._slot_axes(view),
                )
                shared["topk"] = topk
                if published is not None:
                    shared["candidates"] = published
            entries = shared["compress_kv"]
            topk = shared["topk"][:, 0]  # [B, k]
            if entries is not None and topk.shape[-1] > 0:
                gathered, gathered_bias = gather_selected_entries(entries, topk)  # [B, k, D], [B, k]
                keys = jnp.concatenate([keys, gathered.astype(kv.dtype)], axis=1)
                bias = jnp.concatenate([bias, gathered_bias], axis=-1)
        attn_output = self.fused_decode_attend(q, keys, bias[:, None, :], platform=_DECODE_KERNEL_PLATFORM)
        view = view.replace(cache_position=jnp.where(live, position + 1, old_position))
        return attn_output, shared, view

    def _extend_compress(
        self,
        hidden_states: Array,
        view: SharedCompressedWindowCacheView,
        positions: Array,
        valid: Array,
    ) -> SharedCompressedWindowCacheView:
        """Advance a KV source by a block of tokens per slot: emit every group that closes inside the block.

        Equivalent to running :meth:`_compress_step` once per valid column. Each slot's valid cells are a prefix of
        the block at consecutive positions starting at ``positions[:, 0]``. A group whose first members were
        written by earlier steps takes them from the open-group state; the state is then refreshed with the latest
        projection per slot position, exactly as the per-token writes would leave it.
        """
        slots = positions.shape[0]
        ratio = self.ratio
        rows = jnp.arange(slots)[:, None]
        dtype = hidden_states.dtype
        # Block grouping / open-group state refresh is the shared compressor's ``extend_step``.
        projections = self.compressor.project(hidden_states) if ratio > 1 else None
        latent, kv_state, score_state, closed, entry = self.compressor.extend_step(
            hidden_states,
            view.compressor_kv_state,
            view.compressor_score_state,
            positions,
            valid,
            projections=projections,
        )
        if ratio > 1:
            view = view.replace(
                compressor_kv_state=self._replicate(kv_state), compressor_score_state=self._replicate(score_state)
            )
            view = self._write_trace(view, projections, positions, valid)
        cos_g, sin_g = labeled_rope_cos_sin(self.config, "compress", entry * ratio, dtype)
        target = jnp.where(closed, entry, view.compressed_kv.shape[1])
        compressed = apply_rope_interleaved_partial(latent[:, None], cos_g, sin_g, unsqueeze_dim=1)[:, 0]
        updates = {
            "compressed_kv": self._slot(
                view.compressed_kv.at[rows, target].set(
                    self._slot(compressed.astype(view.compressed_kv.dtype), view), mode="drop"
                ),
                view,
            )
        }
        if view.index_k is not None:
            keys = self._slot(self.indexer.index_keys(latent, cos_g, sin_g).astype(view.index_k.dtype), view)
            updates["index_k"] = self._slot(view.index_k.at[rows, target].set(keys, mode="drop"), view)
        return view.replace(**updates)

    def _extend(
        self,
        hidden_states: Array,
        q: Array,
        kv: Array,
        q_residual: Array,
        cos: Array,
        sin: Array,
        positions: Array,
        valid: Array,
        shared: _SharedState,
        view: SharedCompressedWindowCacheView,
    ) -> tuple[Array, _SharedState, SharedCompressedWindowCacheView]:
        """Advance every slot by a block of up to ``sliding_window`` tokens in one pass (batched packed prefill).

        Same result as :meth:`_decode` applied column by column: each query attends to the sink, the pre-block ring
        slots still inside its window, the block's own keys up to itself, and its reach-bounded selected entries.
        Entries emitted inside the block are written before selection; they are causally safe because selection
        never reads beyond a query's ``(position + 1) // ratio``.

        Args:
            hidden_states: ``[S, blk, hidden]``.
            q: Rotated queries ``[S, H, blk, D]``.
            kv: Rotated token K/V ``[S, blk, D]``.
            q_residual: ``[S, blk, q_lora]``.
            cos: Rope cosines ``[S, blk, rope // 2]``.
            sin: Rope sines ``[S, blk, rope // 2]``.
            positions: Stream positions ``[S, blk]`` (garbage at invalid cells).
            valid: ``[S, blk]`` bool; each slot's valid cells form a prefix of the block.
            shared: Cross-layer state (``topk`` / ``candidates`` sliced to the block's columns).
            view: This layer's cache view.

        Returns:
            ``(attention [S, H, blk, D], shared, view)``.
        """
        slots, block = positions.shape
        window = self.sliding_window
        rows = jnp.arange(slots)[:, None]
        positions = positions.astype(jnp.int32)
        n_valid = jnp.sum(valid, axis=1).astype(jnp.int32)
        pos0 = positions[:, 0]

        ring = view.window_kv
        token_bias = block_token_bias(positions, valid, window)  # [S, blk, W + blk]
        token_keys = jnp.concatenate([ring.astype(kv.dtype), kv], axis=1)  # [S, W + blk, D]
        write_slot = jnp.where(valid, positions % window, window)
        view = view.replace(window_kv=ring.at[rows, write_slot].set(kv.astype(ring.dtype), mode="drop"))

        gathered = gathered_bias = None
        if self.ratio > 0:
            shared = dict(shared)
            if self.compressor is not None:
                view = self._extend_compress(hidden_states, view, positions, valid)
                shared["compress_kv"] = view.compressed_kv
                shared["index_k"] = view.index_k
            if self.indexer is not None:
                topk, published = self.indexer(
                    hidden_states,
                    q_residual,
                    shared["index_k"],
                    cos,
                    sin,
                    shared["candidates"],
                    positions,
                    slot_axes=self._slot_axes(view),
                )
                shared["topk"] = topk
                if published is not None:
                    shared["candidates"] = published
            entries = shared["compress_kv"]
            topk = shared["topk"]  # [S, blk, k]
            k = topk.shape[-1]
            if entries is not None and k > 0:
                flat, flat_bias = gather_selected_entries(entries, topk.reshape(slots, block * k))
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
            precision=self.precision,
        )
        new_position = jnp.where(n_valid > 0, pos0 + n_valid, view.cache_position)
        return out, shared, view.replace(cache_position=new_position)

    def _packed(
        self,
        hidden_states: Array,
        q: Array,
        kv: Array,
        q_residual: Array,
        cos: Array,
        sin: Array,
        shared: _SharedState,
        view: SharedCompressedWindowCacheView,
        meta: PackedSlotMeta,
    ) -> tuple[Array, _SharedState, SharedCompressedWindowCacheView]:
        """eSurge packed serving through the shared packed driver (``layers.compressed_window``).

        The flat ``[1, T]`` stream is scattered into a ``[num_slots, T]`` grid (one row per cache slot at its own
        stream position); a ``while_loop`` over the occupied columns advances every slot by a block of columns per
        step (:meth:`_extend`), or one column (:meth:`_decode`) in pure-decode buckets, with padding cells frozen. Cross-layer state is per column: index sources record their selection (and published
        candidate blocks) as ``[num_slots, T, ...]`` grids that later layers read at the same column. Later layers
        read the sources' entry / key buffers in their end-of-step state; that is causal because an entry written
        at a later column sits beyond the reach ``(position + 1) // ratio`` of every earlier column.

        Returns:
            Attention output ``[1, H, T, D]`` (garbage at padding tokens), updated shared state and view.
        """
        total = hidden_states.shape[1]
        num_slots = view.cache_position.shape[0]
        # A valid first cell at position 0 starts a new stream: re-initialize that slot (robust to stale state).
        # Only the per-stream state needs resetting: entries, index keys and ring slots beyond a stream's position
        # are never read (bounded selectors stop at reach, ring slots are masked by slot position), so the multi-GB
        # buffers are left alone instead of being rewritten by a select on every step.
        view = view.reset_rows(meta.valid_grid[:, 0] & (meta.positions_grid[:, 0] == 0), stream_state_only=True)
        slot_tok, col_tok = meta.slot_of_token, meta.col_of_token
        # Shared packed driver: pure decode buckets (at most one token per slot) keep the single-column path (Pallas
        # selector, no wasted cells); prefill / mixed buckets advance blocks of columns at once.
        # Cap the block at one slot's share of the bucket: verify windows (1 + k tokens per slot) otherwise paid a
        # bucket-wide block of per-cell top-k gathers (~87% padding at k=5 over 8 slots).
        shared_in = shared

        def _grid(view: SharedCompressedWindowCacheView, block: int):
            """Advance every slot through the occupied grid columns, ``block`` columns per loop step."""
            width = -(-total // block) * block

            def to_grid(values: Array) -> Array:
                return tokens_to_grid(values, slot_tok, col_tok, num_slots, width)

            def pad_cols(grid: Array) -> Array:
                return grid if width == total else jnp.pad(grid, ((0, 0), (0, width - total)))

            hidden_grid = to_grid(hidden_states[0])
            qres_grid = to_grid(q_residual[0])
            q_grid = to_grid(q[0].transpose(1, 0, 2))  # [S, T, H, D]
            kv_grid = to_grid(kv[0])
            cos_grid = to_grid(cos[0])
            sin_grid = to_grid(sin[0])
            pos_grid = pad_cols(meta.positions_grid)
            valid_grid = pad_cols(meta.valid_grid)

            shared = dict(shared_in)
            if self.ratio > 0 and self.compressor is not None:
                shared["compress_kv"] = view.compressed_kv
                shared["index_k"] = view.index_k
            index_k = shared.get("index_k")
            entries = 0 if index_k is None else index_k.shape[1]
            has_indexer = self.ratio > 0 and self.indexer is not None
            publishes = has_indexer and self.indexer.is_candidate_source
            k = min(self.indexer.index_topk, entries) if has_indexer else 0
            topk_grid = jnp.full((num_slots, width, k), -1, dtype=jnp.int32)
            cand_width = self.indexer.candidate_width(entries) if publishes else 0
            cand_grid = jnp.full((num_slots, width, cand_width), -1, dtype=jnp.int32)

            num_used = occupied_columns(valid_grid)
            acc = jnp.zeros((width, num_slots, self.num_heads, self.head_dim), dtype=view.window_kv.dtype)

            def column(grid: Array, col: Array) -> Array:
                return jax.lax.dynamic_slice_in_dim(grid, col, 1, axis=1)

            def block_step(start: Array, carry):
                v, acc, tgrid, cgrid = carry

                def cols(grid: Array) -> Array:
                    return jax.lax.dynamic_slice_in_dim(grid, start, block, axis=1)

                blk_shared = dict(shared)
                if self.ratio > 0:
                    if self.indexer is None:
                        blk_shared["topk"] = cols(shared["topk"])
                    elif self.indexer.uses_candidates:
                        blk_shared["candidates"] = cols(shared["candidates"])
                attn, out_shared, v = self._extend(
                    cols(hidden_grid),
                    cols(q_grid).transpose(0, 2, 1, 3),
                    cols(kv_grid),
                    cols(qres_grid),
                    cols(cos_grid),
                    cols(sin_grid),
                    cols(pos_grid),
                    cols(valid_grid),
                    blk_shared,
                    v,
                )
                acc = jax.lax.dynamic_update_slice_in_dim(
                    acc, attn.transpose(2, 0, 1, 3).astype(acc.dtype), start, axis=0
                )
                if has_indexer:
                    tgrid = jax.lax.dynamic_update_slice_in_dim(tgrid, out_shared["topk"], start, axis=1)
                    if publishes:
                        cgrid = jax.lax.dynamic_update_slice_in_dim(cgrid, out_shared["candidates"], start, axis=1)
                return v, acc, tgrid, cgrid

            def column_step(col: Array, carry):
                v, acc, tgrid, cgrid = carry
                col_shared = dict(shared)
                if self.ratio > 0:
                    if self.indexer is None:
                        col_shared["topk"] = column(shared["topk"], col)
                    elif self.indexer.uses_candidates:
                        col_shared["candidates"] = column(shared["candidates"], col)
                q_col = jax.lax.dynamic_index_in_dim(q_grid, col, axis=1, keepdims=False)[:, :, None, :]
                attn, out_shared, v = self._decode(
                    column(hidden_grid, col),
                    q_col,
                    column(kv_grid, col),
                    column(qres_grid, col),
                    column(cos_grid, col),
                    column(sin_grid, col),
                    column(pos_grid, col),
                    col_shared,
                    v,
                    valid=jax.lax.dynamic_index_in_dim(valid_grid, col, axis=1, keepdims=False),
                )
                acc = jax.lax.dynamic_update_index_in_dim(acc, attn[:, :, 0, :].astype(acc.dtype), col, axis=0)
                if has_indexer:
                    tgrid = jax.lax.dynamic_update_slice_in_dim(tgrid, out_shared["topk"], col, axis=1)
                    if publishes:
                        cgrid = jax.lax.dynamic_update_slice_in_dim(cgrid, out_shared["candidates"], col, axis=1)
                return v, acc, tgrid, cgrid

            view, acc, topk_grid, cand_grid = run_column_blocks(
                block_step if block > 1 else column_step,
                (view, acc, topk_grid, cand_grid),
                num_used=num_used,
                block=block,
            )
            if self.ratio > 0:
                if self.compressor is not None:
                    shared["compress_kv"] = view.compressed_kv
                    shared["index_k"] = view.index_k
                if has_indexer:
                    shared["topk"] = topk_grid
                    if publishes:
                        shared["candidates"] = cand_grid
            attn = grid_to_tokens(acc, slot_tok, col_tok, num_slots)[None]  # [1, T, H, D]
            return attn.transpose(0, 2, 1, 3), shared, view

        block = packed_block_size(total, num_slots, self.sliding_window, cap_by_slot_share=True)
        if block == 1 and 1 < total <= self.sliding_window:
            # A bucket with no more tokens than slots is usually pure decode (one token per slot: the single-column
            # path). A speculative verify window puts 1 + k tokens on ONE slot of such a bucket, and walking those
            # columns one at a time ran `_decode` (with its per-layer gathers and all-to-alls) 1 + k times over every
            # slot. Choose at run time: the column path when at most one column is occupied, else one block of all
            # columns. Both branches use grid width `total`, so the selections they publish have the same shape.
            return jax.lax.cond(
                occupied_columns(meta.valid_grid) <= 1,
                lambda v: _grid(v, 1),
                lambda v: _grid(v, total),
                view,
            )
        return _grid(view, block)

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        position_embeddings: dict[str, tuple[Array, Array]],
        position_ids: Int[Array, "batch seq"],
        token_valid: Bool[Array, "batch seq"] | None,
        shared: _SharedState,
        cache_view: SharedCompressedWindowCacheView | None = None,
        packed_meta: PackedSlotMeta | None = None,
    ) -> tuple[Array, _SharedState, SharedCompressedWindowCacheView | None]:
        """Attention; returns the output, the updated cross-layer state and the updated cache view."""
        _, seq, _ = hidden_states.shape
        cos, sin = position_embeddings[self.rope_label]
        q_residual, q = self.project_query(hidden_states, cos, sin)
        # Serving: gather the single KV head (a few KB per token) once. Left column-sharded over tp, the RoPE cut at
        # 448/64 channels falls inside a 128-channel shard and XLA reshuffles it through two tp all-to-alls plus
        # relayout copies per layer (~0.73 ms of a V4.1 tp=4 decode step). Layout only; the math is unchanged.
        kv = self.project_kv(hidden_states, cos, sin, layout=self._replicate if cache_view is not None else None)

        if cache_view is not None and packed_meta is not None:
            attn_output, shared, cache_view = self._packed(
                hidden_states, q, kv, q_residual, cos, sin, shared, cache_view, packed_meta
            )
        elif cache_view is not None and seq == 1:
            attn_output, shared, cache_view = self._decode(
                hidden_states, q, kv, q_residual, cos, sin, position_ids, shared, cache_view
            )
        else:
            attn_output, shared, cache_view = self._full(
                hidden_states, q, kv, q_residual, cos, sin, position_ids, token_valid, shared, cache_view
            )
        if cache_view is not None:
            # One fixed cache layout for every step shape (eSurge compiles several buckets over the same buffers).
            cache_view = cache_view.constrain_layout(getattr(self.config, "mesh", None))
        # K == V carries RoPE: project_output undoes it at the query positions, then the grouped output projection.
        return self.project_output(attn_output.transpose(0, 2, 1, 3), cos, sin), shared, cache_view


class DeepseekV41SparseMoeBlock(DeepseekV4SparseMoeBlock):
    """DeepSeek-V4's learned top-k MoE block plus the V4.1 image-token router bias.

    With ``image_router_bias=True`` the router carries an extra fp32 ``gate.bias_vl`` ``[n_routed_experts]``; tokens
    flagged by ``image_mask`` select experts with ``scores + bias_vl`` instead of ``scores + e_score_correction_bias``
    (reference ``Gate.forward``). The routing weights still come from the unbiased scores. Without the flag the block
    declares no extra parameter and is exactly :class:`DeepseekV4SparseMoeBlock`.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        image_router_bias: bool = False,
    ):
        """Initialize the block.

        Args:
            config: Model configuration.
            layer_idx: Layer index.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators.
            image_router_bias: Declare ``gate.bias_vl`` (vision-language models only).
        """
        super().__init__(config, layer_idx, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        self.has_image_router_bias = bool(image_router_bias)
        if image_router_bias:
            self.gate.bias_vl = ArrayParam.bound(
                shape=(config.n_routed_experts,),
                dtype=jnp.float32,
                init_method="zeros",
                key=rngs.param,
                axis_names=(None,),  # replicated on the mesh (read by the select hook inside the MoE shard_map)
            )

    def forward(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        input_ids: Int[Array, "batch seq"] | None = None,
        image_mask: Bool[Array, "batch seq"] | None = None,
    ) -> tuple[Array, Array]:
        """Route tokens through experts and add the shared expert.

        Args:
            hidden_states: Post-norm hidden states ``[B, S, D_model]``.
            input_ids: Unused (V4.1 has no hash routing); accepted for interface parity.
            image_mask: Optional ``[B, S]`` bool, True for tokens inside an image span (select ``bias_vl``).

        Returns:
            Tuple of the MoE output ``[B, S, D_model]`` and the router scores ``[B * S, n_routed_experts]``.

        Raises:
            ValueError: If ``image_mask`` is given but the block was built without ``image_router_bias``.
        """
        if image_mask is None:
            return super().forward(hidden_states, input_ids)
        if not self.has_image_router_bias:
            raise ValueError("image_mask requires a block built with image_router_bias=True (vision enabled).")
        num_experts = self.config.n_routed_experts
        correction_bias = self.gate.e_score_correction_bias.value.astype(jnp.float32)
        image_bias = self.gate.bias_vl.value.astype(jnp.float32)
        flag = image_mask.reshape(-1)

        # The select hook runs inside the fused-MoE shard_map on token-sharded slices, so the per-token image flag rides
        # along the token axis as one extra float column on the gate output (the V4 hash router does the same).
        def _image_gate(flat_hidden: Array) -> Array:
            scores = self.gate(flat_hidden)
            return jnp.concatenate([scores, flag[:, None].astype(scores.dtype)], axis=-1)

        def _select_with_image_bias(augmented: Array, pre_bias_logits: Array | None, k: int) -> tuple[Array, Array]:
            del pre_bias_logits
            scores = augmented[:, :num_experts].astype(jnp.float32)
            is_image = augmented[:, num_experts:] > 0.5
            bias = jnp.where(is_image, image_bias[None, :], correction_bias[None, :])
            _, indices = jax.lax.top_k(scores + bias, k)
            weights = jnp.take_along_axis(scores, indices, axis=-1)
            return self._normalize_and_scale(weights), indices

        hooks = self.moe_hooks.replace(
            normalize_gate_logits=lambda logits: logits,
            select_hook=_select_with_image_bias,
        )
        limit = self.config.swiglu_limit

        def ffn_activation(gate: Array, up: Array) -> Array:
            """Clamped SwiGLU: ``silu(clamp(gate, max=limit)) * clamp(up, +-limit)``."""
            return clamped_swiglu(gate, up, limit=limit, act_fn=self.experts.act_fn)

        routed, router_scores = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=_image_gate,
            expert_layer=self.experts,
            wi_kernel=self.experts.gate_proj.kernel_view(),
            wu_kernel=self.experts.up_proj.kernel_view(),
            wd_kernel=self.experts.down_proj.kernel_view(),
            act_fn=self.experts.act_fn,
            ffn_activation=ffn_activation,
            hooks=hooks,
            layer_idx=self.layer_idx,
        )
        router_scores = router_scores[:, :num_experts]  # drop the carried image-flag column
        out = routed + self.shared_experts(hidden_states)
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_scores, "moe_router_logits")


class DeepseekV41DecoderLayer(spx.Module):
    """One V4.1 block: optional engram, attention and MoE between delayed-``pre`` hyper-connections."""

    def __init__(
        self,
        config: DeepseekV41Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        image_router_bias: bool = False,
    ):
        self.precision = precision
        layer_kwargs = dict(dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        norm_kwargs = dict(dim=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype)
        self.self_attn = DeepseekV41Attention(config, layer_idx, **layer_kwargs)
        self.mlp = DeepseekV41SparseMoeBlock(config, layer_idx, image_router_bias=image_router_bias, **layer_kwargs)
        self.input_layernorm = RMSNorm(**norm_kwargs, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(**norm_kwargs, rngs=rngs)
        self.attn_hc = ManifoldHyperConnection(
            ManifoldHyperConnectionConfig.from_model_config(config), **layer_kwargs, mesh_source=config
        )
        self.ffn_hc = ManifoldHyperConnection(
            ManifoldHyperConnectionConfig.from_model_config(config), **layer_kwargs, mesh_source=config
        )
        self.engram = (
            GatedEngram.from_config(config, layer_idx, **layer_kwargs) if layer_idx in config.engram_layer_ids else None
        )

    def forward(
        self,
        hidden_streams: Float[Array, "batch seq hc hidden"],
        pre_mix: Float[Array, "batch seq hc"],
        position_embeddings: dict[str, tuple[Array, Array]],
        position_ids: Int[Array, "batch seq"],
        token_valid: Bool[Array, "batch seq"] | None,
        shared: _SharedState,
        engram_hash_ids: Array | None = None,
        output_router_logits: bool = False,
        engram_rows: Array | None = None,
        engram_scales: Array | None = None,
        cache_view: SharedCompressedWindowCacheView | None = None,
        packed_meta: PackedSlotMeta | None = None,
        image_mask: Bool[Array, "batch seq"] | None = None,
        capture_main_hidden: bool = False,
    ) -> tuple[Array, Array, _SharedState, Array | None, SharedCompressedWindowCacheView | None, Array | None]:
        """Return ``(streams, ffn_pre, shared, router_logits, cache_view, main_hidden)``.

        ``ffn_pre`` is the next layer's mix. ``image_mask`` (``[B, S]`` bool, True inside image spans) selects the
        image routing bias and switches the engram off for those tokens (reference ``token_mask = ~image_mask``).
        With ``capture_main_hidden`` the last element is ``hidden_streams.mean(axis=2)`` ``[B, S, D]`` taken after this
        layer's engram and before its attention (the DSpark main-hidden hook); otherwise it is ``None``.
        """
        if self.engram is not None:
            idx = self.engram.layer_hash_index
            # Host-gathered fp8 rows (serving) skip the in-program table lookup; hash ids keep the lookup path.
            gathered = None if engram_rows is None else (engram_rows[:, :, idx], engram_scales[:, :, idx])
            hidden_streams = self.engram(
                hidden_streams,
                None if engram_hash_ids is None else engram_hash_ids[:, :, idx],
                None if image_mask is None else ~image_mask,
                gathered=gathered,
            )

        main_hidden = hidden_streams.mean(axis=2) if capture_main_hidden else None

        attn_pre, attn_post, attn_comb = self.attn_hc.coefficients(hidden_streams)
        attn_input = self.input_layernorm(manifold_collapse(pre_mix, hidden_streams))
        attn_output, shared, cache_view = self.self_attn(
            attn_input,
            position_embeddings,
            position_ids,
            token_valid,
            shared,
            cache_view=cache_view,
            packed_meta=packed_meta,
        )
        hidden_streams = manifold_residual_write(
            hidden_streams, attn_output, attn_post, attn_comb, precision=self.precision
        )

        ffn_pre, ffn_post, ffn_comb = self.ffn_hc.coefficients(hidden_streams)
        mlp_input = self.post_attention_layernorm(manifold_collapse(attn_pre, hidden_streams))
        mlp_output, router_logits = (
            self.mlp(mlp_input) if image_mask is None else self.mlp(mlp_input, image_mask=image_mask)
        )
        hidden_streams = manifold_residual_write(
            hidden_streams, mlp_output, ffn_post, ffn_comb, precision=self.precision
        )
        return hidden_streams, ffn_pre, shared, router_logits if output_router_logits else None, cache_view, main_hidden


@auto_pytree
class DeepseekV41ModelOutput(MoeModelOutput):
    """:class:`MoeModelOutput` plus the DSpark main hidden states and the indexer auxiliary loss.

    Attributes:
        main_hidden: ``[B, S, len(dspark_target_layer_ids) * D]`` concatenated target-layer hidden states, only when
            requested with ``output_dspark_hidden=True``.
        indexer_aux_loss: Unscaled sum over index-source layers of :func:`indexer_distillation_loss`, only when
            ``indexer_aux_loss_coef > 0`` on a stateless forward.
    """

    main_hidden: Array | None = None
    indexer_aux_loss: Array | None = None


@auto_pytree
class DeepseekV41CausalLMOutput(MoeCausalLMOutput):
    """:class:`MoeCausalLMOutput` plus ``main_hidden`` and the unscaled ``indexer_aux_loss`` (see the model output)."""

    main_hidden: Array | None = None
    indexer_aux_loss: Array | None = None


@auto_pytree
class DeepseekV41VLMOutput(VLMCausalLMOutput):
    """:class:`VLMCausalLMOutput` plus ``main_hidden`` and the unscaled ``indexer_aux_loss`` (see the model output)."""

    main_hidden: Array | None = None
    indexer_aux_loss: Array | None = None


def _total_aux_loss(config: DeepseekV41Config, base_aux: Array | None, indexer_aux: Array | None) -> Array | None:
    """Fold ``indexer_aux_loss_coef * indexer_aux`` into the router auxiliary loss (``None`` when both are absent)."""
    if indexer_aux is None:
        return base_aux
    scaled = float(config.indexer_aux_loss_coef) * indexer_aux
    return scaled if base_aux is None else base_aux + scaled


@register_module(TaskType.BASE_MODULE, config=DeepseekV41Config, model_type="deepseek_v41")
class DeepseekV41Model(EngramModelMixin, EasyDeLBaseModule):
    """DeepSeek-V4.1 decoder backbone (stateless forward, cached prefill/decode, eSurge packed serving)."""

    esurge_cache_family: tp.ClassVar[str] = "compressed_window"

    def __init__(
        self,
        config: DeepseekV41Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        image_router_bias: bool = False,
    ):
        super().__init__(config=config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        if float(config.gate_temp) != 1.0:
            raise NotImplementedError("DeepSeek-V4.1 router gate_temp != 1.0 is not supported by the shared V4 router.")
        num_layers = config.num_hidden_layers
        with self.assign_layer_stage(0, total_layers=num_layers):
            self.embed_tokens = Embed(
                config.vocab_size, config.hidden_size, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )
        remat_layer_block = auto_remat(
            DeepseekV41DecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.ModuleList([])
        for layer_idx in range(num_layers):
            with self.assign_layer_stage(layer_idx, total_layers=num_layers):
                self.layers.append(
                    remat_layer_block(
                        config=config,
                        layer_idx=layer_idx,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                        image_router_bias=image_router_bias,
                    )
                )
        with self.assign_layer_stage(max(0, num_layers - 1), total_layers=num_layers):
            self.norm = RMSNorm(
                dim=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs
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
        engram_hash_ids: Int[Array, "batch seq_len n_engram cols"] | None = None,
        image_mask: Bool[Array, "batch seq_len"] | None = None,
        output_dspark_hidden: bool = False,
        engram_rows: Array | None = None,
        engram_scales: Array | None = None,
        tokens_to_end: Int[Array, "batch seq_len"] | None = None,
    ) -> DeepseekV41ModelOutput:
        """Forward over the V4.1 stack (stateless, cached prefill, or one-token cached decode).

        Args:
            input_ids: Token ids ``[B, S]``.
            inputs_embeds: Optional embeddings ``[B, S, D]`` (exclusive with ``input_ids``).
            attention_mask: Padding mask ``[B, S]``; prompts must not be left-padded.
            mask_info: Optional precomputed mask info (packed segments are rejected; ignored when cached).
            position_ids: Positions ``[B, S]``; derived from the mask (or the cache position on decode) when omitted.
            output_attentions: Accepted for interface parity.
            output_hidden_states: Whether to collect the per-layer streams.
            output_router_logits: Whether to collect router scores.
            mode: Accepted for interface parity.
            past_key_values: Optional :class:`SharedCompressedWindowCache` (from ``init_cache``). ``S > 1`` prefills
                an empty cache; ``S == 1`` decodes one token per row.
            cache_metadata: Accepted for interface parity.
            engram_hash_ids: Host-computed n-gram hash ids ``[B, S, len(engram_layer_ids), cols]``; required when
                the config has engram layers.
            image_mask: Optional ``[B, S]`` bool, True for tokens inside image spans. Those tokens route with the
                ``bias_vl`` router bias and get no engram contribution. Requires a model built with
                ``image_router_bias=True`` (the vision-language backbone).
            output_dspark_hidden: Also return ``main_hidden`` ``[B, S, T * D]``: for every ``i`` in
                ``config.dspark_target_layer_ids`` (ascending), ``hidden_streams.mean(axis=2)`` taken after layer
                ``i``'s engram and before its attention (the reference's ``main_hiddens``). Works in the stateless,
                cached prefill / decode and packed paths. No extra computation when ``False``.
            engram_rows: Optional host-gathered fp8 row bytes ``uint8 [B, S, len(engram_layer_ids), cols, head_dim]``
                (:meth:`gather_engram_rows`); replaces ``engram_hash_ids`` so the program does no table lookup.
            engram_scales: Their e8m0 scale bytes ``uint8 [B, S, len(engram_layer_ids), cols, head_dim // 32]``.
            tokens_to_end: eSurge packed serving only: ``[1, T]`` distance of each token from its request's last
                known token (``esurge_requires_tokens_to_end`` side input). With ``config.ced_prefill_skip`` a step
                whose tokens all lie outside the decoder tail's reach skips that tail (see
                :func:`_ced_decoder_tail`); ignored elsewhere.

        Returns:
            DeepseekV41ModelOutput with the final normalized hidden state, when cached the updated cache, and the
            optional ``main_hidden`` / ``indexer_aux_loss`` fields.

        Raises:
            TypeError: If ``past_key_values`` is not a :class:`SharedCompressedWindowCache`.
            ValueError: On invalid inputs, an uninitialized cache, missing engram hash ids, or
                ``output_dspark_hidden`` without ``dspark_target_layer_ids``.
        """
        del mode, output_attentions
        config = self.config
        cached = past_key_values is not None
        views: list[SharedCompressedWindowCacheView] | None = None
        if cached:
            if not isinstance(past_key_values, SharedCompressedWindowCache):
                raise TypeError(
                    "DeepSeek-V4.1 requires a SharedCompressedWindowCache (build one via init_cache); got "
                    f"{type(past_key_values).__name__}."
                )
            views = list(past_key_values.views)
            if len(views) != len(self.layers) or any(view is None for view in views):
                raise ValueError("SharedCompressedWindowCache must hold one initialized view per decoder layer.")
        output_router_logits = output_router_logits if output_router_logits is not None else config.output_router_logits
        output_hidden_states = output_hidden_states if output_hidden_states is not None else config.output_hidden_states
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds.")
        if (engram_rows is None) != (engram_scales is None):
            raise ValueError("engram_rows and engram_scales must be given together.")
        if config.engram_layer_ids and engram_hash_ids is None and engram_rows is None:
            raise ValueError(
                "DeepSeek-V4.1 engram layers need host-computed `engram_hash_ids` "
                "(see easydel.layers.engram.EngramHasher)."
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
        batch, seq, _ = inputs_embeds.shape

        token_valid = None
        packed_meta = None
        packed_qsl = getattr(cache_metadata, "query_start_loc", None)
        packed_num_seqs = getattr(cache_metadata, "num_seqs", None)
        if cached and batch == 1 and packed_qsl is not None and packed_num_seqs is not None:
            # eSurge packed multi-request stream against per-slot cache rows.
            if position_ids is None:
                raise ValueError("Packed eSurge serving requires explicit position_ids for the token stream.")
            packed_meta = build_packed_slot_meta(
                query_start_loc=packed_qsl,
                num_seqs=packed_num_seqs,
                recurrent_state_indices=getattr(cache_metadata, "recurrent_state_indices", None),
                position_ids=position_ids,
                num_slots=int(views[0].cache_position.shape[0]),
            )
        elif cached and seq == 1:
            if position_ids is None:
                position_ids = views[0].cache_position.astype(jnp.int32)[:, None]
        else:
            if seq > 1:
                reject_packed_or_padded_batch(mask_info, attention_mask, family="DeepSeek-V4.1")
            if cached:
                raise_if_runtime_true(
                    jnp.any(views[0].cache_position != 0),
                    "DeepSeek-V4.1 cached multi-token prefill requires an empty cache; decode one token at a time.",
                )
                if position_ids is None:
                    position_ids = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32)[None, :], (batch, seq))
            else:
                mask_info = MaskInfo.dynamic_init(
                    mask_info=mask_info,
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
                if position_ids is None:
                    position_ids = mask_info.q_position_ids
            # Per-key validity only: the window / causal mask is built where it is consumed (in-kernel on TPU).
            token_valid = (
                attention_mask.astype(bool) if attention_mask is not None else jnp.ones((batch, seq), dtype=bool)
            )

        position_embeddings = {
            label: labeled_rope_cos_sin(config, label, position_ids, inputs_embeds.dtype)
            for label in ("main", "compress")
        }
        hidden_streams = jnp.broadcast_to(
            inputs_embeds[:, :, None, :], (batch, seq, config.hc_mult, config.hidden_size)
        ).astype(inputs_embeds.dtype)
        hidden_streams = apply_logical_sharding(
            hidden_streams, dynamic_axes=HyperStreamSharding, partition_manager=config.runtime_sharding_resolver
        )
        # Layer 0 collapses with a one-hot read of the first stream (reference `make_identity_pre_mix`).
        pre_mix = jnp.zeros((batch, seq, config.hc_mult), dtype=jnp.float32).at[..., 0].set(1.0)
        shared: _SharedState = dict.fromkeys(_SHARED_KEYS)
        dspark_targets = frozenset(config.dspark_target_layer_ids) if output_dspark_hidden else frozenset()
        if output_dspark_hidden and not dspark_targets:
            raise ValueError("output_dspark_hidden requires a non-empty `dspark_target_layer_ids`.")

        base_inputs = dict(
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            engram_hash_ids=engram_hash_ids,
            engram_rows=engram_rows,
            engram_scales=engram_scales,
            packed_meta=packed_meta,
            image_mask=image_mask,
        )

        def _call_layer(block, idx, streams, pre, state, cache_view, capture, inputs=None):
            inputs = base_inputs if inputs is None else inputs
            with self._layer_stage_context(idx, layers=self.layers):
                return block(
                    hidden_streams=streams,
                    pre_mix=pre,
                    token_valid=token_valid,
                    shared=state,
                    output_router_logits=output_router_logits,
                    cache_view=cache_view,
                    capture_main_hidden=capture,
                    **inputs,
                )

        # CED prefill skip: the window-only decoder tail runs only on tokens inside its receptive field (exact, see
        # `_ced_decoder_tail`). Branch 0 skips the tail (no live token), branch 1 runs it on the live tokens compacted
        # into `compact_len` slots, the last branch runs it on the whole step. Pure-decode buckets never get here.
        ced_start = None
        ced_branch = None
        compact_plan = None
        num_slots = int(packed_meta.valid_grid.shape[0]) if packed_meta is not None else 0
        if (
            packed_meta is not None
            and tokens_to_end is not None
            and bool(getattr(config, "ced_prefill_skip", False))
            and not output_hidden_states
            and not output_router_logits
            and self._pipeline_stage_count() <= 1
            and seq > num_slots
        ):
            tail = _ced_decoder_tail(config)
            if tail is not None:
                ced_start, reach = tail
                live = (jnp.asarray(tokens_to_end).reshape(-1) < reach) & (packed_meta.slot_of_token < num_slots)
                n_live = jnp.sum(live.astype(jnp.int32))
                compact_len = max(seq // _CED_COMPACT_DIVISOR, num_slots)
                if compact_len < seq:
                    compact_plan = _ced_compact_plan(
                        live,
                        packed_qsl,
                        packed_num_seqs,
                        getattr(cache_metadata, "recurrent_state_indices", None),
                        position_ids,
                        num_slots,
                        compact_len,
                    )
                    ced_branch = jnp.where(n_live == 0, 0, jnp.where(n_live <= compact_len, 1, 2))
                else:
                    ced_branch = jnp.where(n_live == 0, 0, 1)
                ced_branch = ced_branch.astype(jnp.int32)
        tail_targets = tuple(i for i in sorted(dspark_targets) if ced_start is not None and i >= ced_start)

        def _run_tail(streams, pre, tail_views, state, inputs=None):
            tail_views = list(tail_views)
            caps = ()
            for offset, layer_idx in enumerate(range(ced_start, len(self.layers))):
                capture = layer_idx in dspark_targets
                streams, pre, state, _router, tail_views[offset], main = _call_layer(
                    self.layers[layer_idx], layer_idx, streams, pre, state, tail_views[offset], capture, inputs
                )
                if capture:
                    caps += (main,)
            return streams, pre, tuple(tail_views), caps

        def _advance_tail_positions(tail_views):
            # Every slot's stream ends where the full step leaves it, as the packed attention would set it.
            valid = packed_meta.valid_grid
            end = jnp.max(jnp.where(valid, packed_meta.positions_grid + 1, 0), axis=1)
            any_valid = jnp.any(valid, axis=1)
            mesh = getattr(config, "mesh", None)
            return tuple(
                view.replace(
                    cache_position=jnp.where(any_valid, end, view.cache_position).astype(view.cache_position.dtype)
                ).constrain_layout(mesh)
                for view in tail_views
            )

        def _skip_tail(streams, pre, tail_views, state):
            # Only the stream positions advance (rings untouched).
            del state
            caps = tuple(jnp.zeros_like(streams[:, :, 0]) for _ in tail_targets)
            return streams, pre, _advance_tail_positions(tail_views), caps

        def _compact_tail(streams, pre, tail_views, state):
            # Live tokens only. Dead tokens keep their last-KV-source streams (never read, like the skip) and write no
            # ring entry; slots whose chunk is entirely dead still get their stream position advanced.
            plan = compact_plan

            def take(x):
                return None if x is None else jnp.take(x, plan.token_index, axis=1, mode="clip")

            inputs = dict(
                position_embeddings={k: (take(c), take(s)) for k, (c, s) in position_embeddings.items()},
                position_ids=take(position_ids),
                engram_hash_ids=take(engram_hash_ids),
                engram_rows=take(engram_rows),
                engram_scales=take(engram_scales),
                packed_meta=plan.meta,
                image_mask=take(image_mask),
            )
            c_state = dict(state)
            for key in ("topk", "candidates"):
                if c_state.get(key) is not None:
                    c_state[key] = _ced_shift_grid(c_state[key], plan.col_offset)
            c_streams = apply_logical_sharding(
                take(streams), dynamic_axes=HyperStreamSharding, partition_manager=config.runtime_sharding_resolver
            )
            c_streams, c_pre, tail_views, c_caps = _run_tail(c_streams, take(pre), tail_views, c_state, inputs)
            idx = plan.token_index
            streams = streams.at[:, idx].set(c_streams.astype(streams.dtype), mode="drop")
            streams = apply_logical_sharding(
                streams, dynamic_axes=HyperStreamSharding, partition_manager=config.runtime_sharding_resolver
            )
            pre = pre.at[:, idx].set(c_pre.astype(pre.dtype), mode="drop")
            caps = tuple(
                jnp.zeros_like(streams[:, :, 0]).at[:, idx].set(c.astype(streams.dtype), mode="drop") for c in c_caps
            )
            return streams, pre, _advance_tail_positions(tail_views), caps

        def _full_tail(streams, pre, tail_views, state):
            return _run_tail(streams, pre, tail_views, state)

        def _layer_loop(block, carry):
            streams, pre, state, all_hidden, all_router, captures, idx = carry
            if ced_start is not None and idx >= ced_start:
                if idx == ced_start:
                    branches = (
                        (_skip_tail, _full_tail)
                        if compact_plan is None
                        else (
                            _skip_tail,
                            _compact_tail,
                            _full_tail,
                        )
                    )
                    streams, pre, tail_views, tail_caps = jax.lax.switch(
                        ced_branch, branches, streams, pre, tuple(views[ced_start:]), state
                    )
                    views[ced_start:] = list(tail_views)
                    if captures is not None:
                        captures += tail_caps
                return streams, pre, state, all_hidden, all_router, captures, idx + 1
            if output_hidden_states:
                all_hidden += (streams,)
            capture = idx in dspark_targets
            cache_view = views[idx] if cached else None
            streams, pre, state, router, cache_view, main = _call_layer(
                block, idx, streams, pre, state, cache_view, capture
            )
            if cached:
                views[idx] = cache_view
            streams = self._mark_layer_stage_boundary(streams, idx, layers=self.layers)
            if output_router_logits:
                all_router += (router,)
            if capture:
                captures += (main,)
            return streams, pre, state, all_hidden, all_router, captures, idx + 1

        # Layers are heterogeneous (per-layer compressor/indexer/engram), so this is always the python loop; the
        # layer index is a python int, so the DSpark target ids are static.
        hidden_streams, pre_mix, final_shared, all_hidden, all_router, captures, _ = self.layers.scan(
            _layer_loop,
            (
                hidden_streams,
                pre_mix,
                shared,
                () if output_hidden_states else None,
                () if output_router_logits else None,
                () if output_dspark_hidden else None,
                0,
            ),
            trace=True,
        )
        hidden_states = self.norm(manifold_collapse(pre_mix, hidden_streams))
        main_hidden = None
        if output_dspark_hidden:
            main_hidden = jnp.concatenate(captures, axis=-1)  # appended in ascending layer order
        return DeepseekV41ModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden,
            attentions=None,
            router_logits=all_router,
            past_key_values=SharedCompressedWindowCache(views=views) if cached else None,
            main_hidden=main_hidden,
            indexer_aux_loss=final_shared.get("indexer_aux"),
        )

    def create_cache_config(
        self, batch_size: int, max_length: int, spec_trace_len: int = 0
    ) -> SharedCompressedWindowCacheConfig:
        """Cache layout for this model's ownership schedule (state only on KV sources / index-key owners).

        ``spec_trace_len > 0`` also allocates the projection traces used by replay-free speculative rollback.
        """
        config = self.config
        num_layers = config.num_hidden_layers
        return SharedCompressedWindowCacheConfig.create(
            num_hidden_layers=num_layers,
            partition_axis=config.partition_axis,
            batch_size=batch_size,
            max_length=max_length,
            sliding_window=config.sliding_window,
            head_dim=config.head_dim,
            index_head_dim=config.index_head_dim,
            compress_ratios=config.compress_ratios[:num_layers],
            kv_source_layer_ids=config.kv_source_layer_ids,
            index_key_layer_ids=[i for i in range(num_layers) if config.owns_index_keys(i)],
            slot_axes=slot_partition_axes(getattr(config, "mesh", None), batch_size),
            spec_trace_len=spec_trace_len,
        )

    def init_operations_cache(
        self,
        batch_size: int,
        max_length: int,
        starts=None,
        dtype: jnp.dtype | None = None,
        **kwargs,
    ) -> SharedCompressedWindowCache:
        """Allocate the shared compressed-window decode cache.

        ``spec_trace_len`` (eSurge, speculative decoding) sizes the replay-free rollback traces; ``starts`` and the
        other engine kwargs are ignored.
        """
        spec_trace_len = int(kwargs.pop("spec_trace_len", 0) or 0)
        del starts, kwargs
        return SharedCompressedWindowCache.init_cache(
            config=self.create_cache_config(batch_size=batch_size, max_length=max_length, spec_trace_len=spec_trace_len),
            dtype=dtype if dtype is not None else self.dtype,
            mesh=getattr(self.config, "mesh", None),
        )

    def init_cache(
        self,
        batch_size: int,
        max_length: int,
        starts=None,
        shardings=None,
        pad_token_id=None,
    ) -> SharedCompressedWindowCache:
        """Allocate the shared compressed-window decode cache (streams start at position 0)."""
        del shardings, pad_token_id
        return self.init_operations_cache(batch_size=batch_size, max_length=max_length, starts=starts)

    def get_decoder(self):
        """Return the decoder (this module)."""
        return self

    def get_lm_head(self):
        """Base model has no LM head."""
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """Return the token embedding layer."""
        return self.embed_tokens


# ---------------------------------------------------------------- checkpoint keys
# V4.1 ships in DeepSeek's inference naming like V4, so the V4 rules map attention, compressor, MoE, norm and
# hyper-connection keys. The indexer differs: it has no compressor of its own (keys come from the KV source's latent
# through ``wk``) and lives directly on the attention module. Vision / aligner / image delimiters and the image-token
# router bias belong to the VL model; the DSpark ``mtp.*`` stack to the draft model; engram tables are read by
# :mod:`easydel.layers.engram`'s own loader and never reach the converter.

DEEPSEEK_V41_KEY_RULES = KeyRuleSet(
    [
        Map(r"(?:mtp|model\.mtp|vision|aligner)\..*|image_start|image_end|image_newline", None),
        Map(r".*\.ffn\.gate\.bias_vl", None),
        Map(ENGRAM_TABLE_KEY_RE.pattern, None),
        Map(
            r"layers\.(?P<i>\d+)\.attn\.indexer\.(?P<leaf>[A-Za-z0-9_]+)\.(?P<suffix>weight|scale)",
            "model.layers.{i}.self_attn.indexer.{leaf}.{suffix}",
            rename={"leaf": DEEPSEEK_V4_LEAF_ALIASES},
        ),
    ],
    otherwise=DEEPSEEK_V4_KEY_RULES,
    name="deepseek_v41",
)

#: Vision-language model: everything the text rules map, plus vision / aligner / delimiters under ``model.`` and the
#: image-token router bias.
DEEPSEEK_V41_VL_KEY_RULES = KeyRuleSet(
    [
        Map(r"(?:model\.)?mtp\..*", None),
        Map(r"(?:vision|aligner)\..*|image_start|image_end|image_newline", "model.{0}"),
        Map(r"layers\.(?P<i>\d+)\.ffn\.gate\.bias_vl", "model.layers.{i}.mlp.gate.bias_vl"),
    ],
    otherwise=DEEPSEEK_V41_KEY_RULES,
    name="deepseek_v41_vl",
)

#: Native DSpark keys (``mtp.N.*``) -> :class:`DeepseekV41DSpark` names. The tied ``mtp.N.embed`` / ``mtp.N.head`` and
#: ``mtp.N.ffn.gate.bias_vl`` (never read by text drafts) are unowned.
DEEPSEEK_V41_MTP_KEY_RULES = KeyRuleSet(
    [
        Map(r"stages\..*", KEEP),
        Map(r"mtp\.\d+\.(?:ffn\.gate\.bias_vl|(?:embed|head)\..*)", None),
        Map(
            r"mtp\.(?P<s>\d+)\.(?P<r>main_proj\.(?:weight|scale)|main_norm\.weight|norm\.weight"
            r"|markov_head\.(?:embed|head)\.weight|confidence_head\.proj\.weight)",
            "stages.{s}.{r}",
        ),
        mtp_stage_block_rule(DEEPSEEK_V4_KEY_RULES),
    ],
    name="deepseek_v41_mtp",
)


@register_module(TaskType.CAUSAL_LM, config=DeepseekV41Config, model_type="deepseek_v41")
class DeepseekV41ForCausalLM(EngramModelMixin, BaseCausalLMModule[DeepseekV41Model, DeepseekV41Config]):  # type: ignore
    """DeepSeek-V4.1 with a causal language modeling head."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "deepseek_v41"
    _config_class = DeepseekV41Config
    checkpoint_key_rules: tp.ClassVar[KeyRuleSet] = DEEPSEEK_V41_KEY_RULES
    esurge_cache_family: tp.ClassVar[str] = "compressed_window"
    # Engram table management, eSurge side inputs and the trainer host-batch hook come from ``EngramModelMixin``.

    @property
    def esurge_requires_tokens_to_end(self) -> bool:
        """eSurge hook: request the ``tokens_to_end`` side input that drives the CED prefill skip."""
        return bool(getattr(self.config, "ced_prefill_skip", False))

    def __init__(
        self,
        config: DeepseekV41Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=DeepseekV41Model,
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
        engram_hash_ids: Int[Array, "batch seq_len n_engram cols"] | None = None,
        output_dspark_hidden: bool = False,
        engram_rows: Array | None = None,
        engram_scales: Array | None = None,
        tokens_to_end: Int[Array, "batch seq_len"] | None = None,
    ) -> DeepseekV41CausalLMOutput:
        """Forward producing vocabulary logits; arguments as :meth:`DeepseekV41Model.forward`.

        With ``config.indexer_aux_loss_coef > 0`` (stateless forward) ``aux_loss`` is ``coef`` times the summed
        indexer distillation loss (the unscaled value is ``indexer_aux_loss``); the trainer adds ``aux_loss`` to the
        language-model loss exactly as for MoE router losses. ``output_dspark_hidden`` fills ``main_hidden``.
        """
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
            engram_hash_ids=engram_hash_ids,
            output_dspark_hidden=output_dspark_hidden,
            engram_rows=engram_rows,
            engram_scales=engram_scales,
            tokens_to_end=tokens_to_end,
        )
        logits = None
        if apply_lm_head:
            logits = self.compute_lm_logits(self.prepare_lm_head_inputs(outputs.last_hidden_state))
        return DeepseekV41CausalLMOutput(
            aux_loss=_total_aux_loss(self.config, self.compute_router_aux_loss(outputs), outputs.indexer_aux_loss),
            logits=logits,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=None,
            router_logits=outputs.router_logits,
            past_key_values=outputs.past_key_values,
            main_hidden=outputs.main_hidden,
            indexer_aux_loss=outputs.indexer_aux_loss,
        )

    def init_operations_cache(
        self,
        batch_size: int,
        max_length: int,
        starts=None,
        dtype: jnp.dtype | None = None,
        **kwargs,
    ) -> SharedCompressedWindowCache:
        """Allocate the decode cache (delegates to the base model)."""
        return self.model.init_operations_cache(
            batch_size=batch_size, max_length=max_length, starts=starts, dtype=dtype, **kwargs
        )

    def init_cache(
        self,
        batch_size: int,
        max_length: int,
        starts=None,
        shardings=None,
        pad_token_id=None,
    ) -> SharedCompressedWindowCache:
        """Allocate the decode cache (delegates to the base model)."""
        return self.model.init_cache(
            batch_size=batch_size, max_length=max_length, starts=starts, shardings=shardings, pad_token_id=pad_token_id
        )


def _prepare_patches(pixel_values: Array, image_grid: tuple[int, int] | None) -> tuple[Array, tuple[int, int]]:
    """Normalize image patches to ``[N, P, D]`` and resolve the static patch grid.

    Accepted layouts (``N`` images sharing one grid, ``D = 3 * patch ** 2``): flat ``[N, P, D]`` / ``[N, P, 3, p, p]``
    (``image_grid`` required) and gridded ``[N, h, w, D]`` / ``[N, h, w, 3, p, p]`` (grid read from the shape).

    Raises:
        ValueError: On an unsupported rank, a missing grid, or a patch count that does not fill the grid.
    """
    if pixel_values.ndim in (4, 6):
        grid = (int(pixel_values.shape[1]), int(pixel_values.shape[2]))
        if image_grid is not None and (int(image_grid[0]), int(image_grid[1])) != grid:
            raise ValueError(f"image_grid {tuple(image_grid)} disagrees with pixel_values grid {grid}.")
        n_images = pixel_values.shape[0]
        return pixel_values.reshape(n_images, grid[0] * grid[1], -1), grid
    if pixel_values.ndim in (3, 5):
        if image_grid is None:
            raise ValueError("flat pixel_values [N, P, ...] need a static `image_grid=(n_vit_h, n_vit_w)`.")
        grid = (int(image_grid[0]), int(image_grid[1]))
        if pixel_values.shape[1] != grid[0] * grid[1]:
            raise ValueError(f"{pixel_values.shape[1]} patches do not fill the {grid[0]}x{grid[1]} grid.")
        return pixel_values.reshape(pixel_values.shape[0], grid[0] * grid[1], -1), grid
    raise ValueError(f"unsupported pixel_values rank {pixel_values.ndim}; expected 3, 4, 5 or 6.")


@register_module(TaskType.VISION_LM, config=DeepseekV41Config, model_type="deepseek_v41")
class DeepseekV41VLModel(DeepseekV41Model):
    """DeepSeek-V4.1 backbone with the vision tower, aligner and learned image-span delimiters.

    Adds ``vision`` (ViT), ``aligner``, ``image_start`` / ``image_end`` / ``image_newline`` (``[hidden]`` vectors) and
    a ``gate.bias_vl`` router bias on every MoE layer. Without images (``token_types=None``) it computes exactly the
    text backbone. eSurge multimodal serving is NOT wired: images are accepted by the stateless forward and by a
    cached prefill only.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        if not config.vision_enabled:
            raise ValueError("DeepseekV41VLModel needs `vision_config` with num_hidden_layers > 0.")
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            image_router_bias=True,
        )
        self.vision = DeepseekV41VisionModel(
            config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
        )
        self.aligner = DeepseekV41Aligner(config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        for name in ("image_start", "image_end", "image_newline"):
            setattr(
                self,
                name,
                ArrayParam.bound(
                    shape=(config.hidden_size,),
                    dtype=param_dtype,
                    init_method="normal",
                    init_kwargs={"stddev": config.initializer_range},
                    key=rngs.param,
                    # Replicated on the mesh: without axis metadata the parameter-init placement leaves a
                    # normal-init leaf committed to the key's single device, which clashes with mesh-sharded
                    # embeddings in ``merge_image_embeddings`` on any multi-device mesh.
                    axis_names=(None,),
                ),
            )

    def get_image_features(self, pixel_values: Array, image_grid: tuple[int, int] | None = None) -> Array:
        """Encode images to LLM-width tokens.

        Args:
            pixel_values: Patches of ``N`` images sharing one grid (see :func:`_prepare_patches`).
            image_grid: Static ``(n_vit_h, n_vit_w)`` for flat layouts.

        Returns:
            ``[N, n_llm_h * n_llm_w, hidden_size]`` aligner rows in reading order.
        """
        patches, grid = _prepare_patches(pixel_values, image_grid)
        return self.aligner(self.vision(patches, grid), grid)

    def merge_image_embeddings(self, embeds: Array, token_types: Array, features: Array) -> Array:
        """Overwrite image-span positions of ``embeds`` (reference ``merge_image_embeddings``).

        ``IMAGE_START`` / ``IMAGE_END`` / ``IMAGE_NEW_LINE`` positions take the learned delimiter vectors; ``IMAGE``
        positions take the aligner rows in reading order, consuming the rows of image 0, then image 1, ... over the
        row-major flattened ``(batch, position)`` order.

        Args:
            embeds: Token embeddings ``[B, S, D]``.
            token_types: ``[B, S]`` int (``-1`` text, ``0..3`` span positions).
            features: Aligner rows ``[N, n_llm, D]``.

        Returns:
            Merged embeddings ``[B, S, D]``.

        Raises:
            ValueError: If concrete token types hold a different number of IMAGE slots than aligner rows.
        """
        dtype = embeds.dtype
        rows = features.reshape(-1, features.shape[-1]).astype(dtype)
        is_image = token_types == TOKEN_TYPE_IMAGE
        if not isinstance(token_types, jax.core.Tracer):
            slots = int(jnp.sum(is_image))
            if slots != rows.shape[0]:
                raise ValueError(f"token_types hold {slots} IMAGE slots but the images provide {rows.shape[0]} rows.")
        flat = is_image.reshape(-1)
        gather = jnp.clip(jnp.cumsum(flat) - 1, 0, rows.shape[0] - 1)
        merged = jnp.where(flat[:, None], rows[gather], embeds.reshape(-1, embeds.shape[-1])).reshape(embeds.shape)
        for token_type, name in (
            (TOKEN_TYPE_IMAGE_START, "image_start"),
            (TOKEN_TYPE_IMAGE_END, "image_end"),
            (TOKEN_TYPE_IMAGE_NEW_LINE, "image_newline"),
        ):
            vector = getattr(self, name).value.astype(dtype)
            merged = jnp.where((token_types == token_type)[..., None], vector, merged)
        return merged

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
        engram_hash_ids: Int[Array, "batch seq_len n_engram cols"] | None = None,
        image_mask: Bool[Array, "batch seq_len"] | None = None,
        token_types: Int[Array, "batch seq_len"] | None = None,
        pixel_values: Array | None = None,
        image_grid: tuple[int, int] | None = None,
        output_dspark_hidden: bool = False,
        engram_rows: Array | None = None,
        engram_scales: Array | None = None,
        tokens_to_end: Int[Array, "batch seq_len"] | None = None,
    ) -> DeepseekV41ModelOutput:
        """Forward with optional images (stateless, cached prefill, or one-token decode with ``token_types=None``).

        Args:
            input_ids: Token ids ``[B, S]``; every image-span position carries ``config.image_token_id``.
            inputs_embeds: Optional embeddings instead of ``input_ids``.
            attention_mask: Padding mask (prompts must not be left-padded).
            mask_info: Optional precomputed mask info.
            position_ids: Optional positions.
            output_attentions: Interface parity.
            output_hidden_states: Whether to collect per-layer streams.
            output_router_logits: Whether to collect router scores.
            mode: Interface parity.
            past_key_values: Optional :class:`SharedCompressedWindowCache`. Images may only be given on the prefill
                (images in a decode step are meaningless: the whole span must be prefilled in one chunk).
            cache_metadata: Interface parity.
            engram_hash_ids: Host-computed hash ids; compute them with ``dead_mask = token_types >= 0``
                (``EngramHasher.hash(ids, dead_mask)``) so image tokens take no part in an n-gram.
            image_mask: Explicit ``[B, S]`` image-span mask; derived from ``token_types`` when omitted.
            token_types: ``[B, S]`` int token types (``-1`` text, ``0`` start, ``1`` image, ``2`` newline, ``3`` end).
            pixel_values: Patches of ``N`` images sharing one grid, consumed in order of appearance.
            image_grid: Static ``(n_vit_h, n_vit_w)`` for flat ``pixel_values`` layouts.
            output_dspark_hidden: Also return ``main_hidden`` (see :meth:`DeepseekV41Model.forward`).

        Returns:
            DeepseekV41ModelOutput.

        Raises:
            ValueError: If ``pixel_values`` is given without ``token_types``.
        """
        if token_types is not None:
            token_types = jnp.asarray(token_types)
            if image_mask is None:
                image_mask = token_types >= 0
        if pixel_values is not None:
            if token_types is None:
                raise ValueError("pixel_values need `token_types` to locate the image spans.")
            if (input_ids is None) == (inputs_embeds is None):
                raise ValueError("Specify exactly one of input_ids and inputs_embeds.")
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
            features = self.get_image_features(pixel_values, image_grid)
            inputs_embeds = self.merge_image_embeddings(inputs_embeds, token_types, features)
            input_ids = None
        return super().forward(
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
            engram_hash_ids=engram_hash_ids,
            image_mask=image_mask,
            output_dspark_hidden=output_dspark_hidden,
            engram_rows=engram_rows,
            engram_scales=engram_scales,
            tokens_to_end=tokens_to_end,
        )


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=DeepseekV41Config, model_type="deepseek_v41")
class DeepseekV41ForConditionalGeneration(  # type: ignore
    EngramModelMixin, BaseVisionLanguageModule[DeepseekV41VLModel, DeepseekV41Config]
):
    """DeepSeek-V4.1 vision-language model (image + text to text).

    Owns the text backbone plus ViT, aligner, image delimiters and ``lm_head``. Parameter names match the text model
    (``model.layers...``, ``lm_head``) with ``model.vision.*``, ``model.aligner.*``, ``model.image_{start,end,newline}``
    and ``model.layers.N.mlp.gate.bias_vl`` added. eSurge multimodal serving is out of scope.
    """

    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "deepseek_v41"
    _config_class = DeepseekV41Config
    _auto_register = False  # registered explicitly above
    checkpoint_key_rules: tp.ClassVar[KeyRuleSet] = DEEPSEEK_V41_VL_KEY_RULES

    @property
    def esurge_requires_tokens_to_end(self) -> bool:
        """eSurge hook: request the ``tokens_to_end`` side input that drives the CED prefill skip."""
        return bool(getattr(self.config, "ced_prefill_skip", False))

    _vision_tower_name = "vision"
    _projector_name = "aligner"
    loss_type = "ForCausalLM"
    esurge_cache_family: tp.ClassVar[str] = "compressed_window"
    # Token-type VLM prefill (image-span routing / engram dead mask): eSurge adds the ``token_types`` side input.
    esurge_requires_token_types: tp.ClassVar[bool] = True

    def esurge_image_expander_builder(self):
        """eSurge hook: the scheduler-side image placeholder expander."""
        from .processing_deepseek_v41 import DeepseekV41ImageExpander

        return DeepseekV41ImageExpander(self.config)

    def esurge_compute_prefill_embeddings(self, prompt_token_ids, features):
        """eSurge hook: encode a request's images and merge them into its prompt embeddings."""
        from .processing_deepseek_v41 import compute_prefill_embeddings

        return compute_prefill_embeddings(self, prompt_token_ids, features)

    def __init__(
        self,
        config: DeepseekV41Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=DeepseekV41VLModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            image_token_index=config.image_token_id,
            tie_word_embeddings=False,
            lm_head_bias=False,
        )

    def get_language_model(self):
        """Return the text backbone (the VL model itself holds the layers)."""
        return self.base_model

    def get_image_features(self, pixel_values: Array, image_grid: tuple[int, int] | None = None, **kwargs) -> Array:
        """Aligner rows ``[N, n_llm, hidden]`` of images sharing one grid."""
        del kwargs
        return self.base_model.get_image_features(pixel_values, image_grid)

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
        engram_hash_ids: Int[Array, "batch seq_len n_engram cols"] | None = None,
        token_types: Int[Array, "batch seq_len"] | None = None,
        pixel_values: Array | None = None,
        image_grid: tuple[int, int] | None = None,
        output_dspark_hidden: bool = False,
        engram_rows: Array | None = None,
        engram_scales: Array | None = None,
        tokens_to_end: Int[Array, "batch seq_len"] | None = None,
        **kwargs,
    ) -> DeepseekV41VLMOutput:
        """Forward producing vocabulary logits; arguments as :meth:`DeepseekV41VLModel.forward`.

        Extra ``**kwargs`` (trainer bookkeeping fields) are accepted for the VLM calling convention and ignored.
        """
        del kwargs
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
            engram_hash_ids=engram_hash_ids,
            token_types=token_types,
            pixel_values=pixel_values,
            image_grid=image_grid,
            output_dspark_hidden=output_dspark_hidden,
            engram_rows=engram_rows,
            engram_scales=engram_scales,
            tokens_to_end=tokens_to_end,
        )
        hidden_states = apply_logical_sharding(
            outputs.last_hidden_state,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        logits = self.compute_lm_logits(hidden_states) if apply_lm_head else None
        return DeepseekV41VLMOutput(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            last_hidden_state=hidden_states,
            attentions=None,
            router_logits=outputs.router_logits,
            aux_loss=_total_aux_loss(self.config, self.compute_router_aux_loss(outputs), outputs.indexer_aux_loss),
            main_hidden=outputs.main_hidden,
            indexer_aux_loss=outputs.indexer_aux_loss,
        )

    def init_operations_cache(
        self,
        batch_size: int,
        max_length: int,
        starts=None,
        dtype: jnp.dtype | None = None,
        **kwargs,
    ) -> SharedCompressedWindowCache:
        """Allocate the decode cache (delegates to the backbone)."""
        return self.model.init_operations_cache(
            batch_size=batch_size, max_length=max_length, starts=starts, dtype=dtype, **kwargs
        )

    def init_cache(
        self,
        batch_size: int,
        max_length: int,
        starts=None,
        shardings=None,
        pad_token_id=None,
    ) -> SharedCompressedWindowCache:
        """Allocate the decode cache (delegates to the backbone)."""
        return self.model.init_cache(
            batch_size=batch_size, max_length=max_length, starts=starts, shardings=shardings, pad_token_id=pad_token_id
        )


__all__ = [
    "DeepseekV41Attention",
    "DeepseekV41CausalLMOutput",
    "DeepseekV41Compressor",
    "DeepseekV41DecoderLayer",
    "DeepseekV41ForCausalLM",
    "DeepseekV41ForConditionalGeneration",
    "DeepseekV41Indexer",
    "DeepseekV41Model",
    "DeepseekV41ModelOutput",
    "DeepseekV41SparseMoeBlock",
    "DeepseekV41VLMOutput",
    "DeepseekV41VLModel",
    "indexer_distillation_loss",
    "select_candidate_blocks",
]
