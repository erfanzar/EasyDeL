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

"""Compressed sliding-window cache (DeepSeek-V4 style attention state).

This cache family serves architectures whose per-layer decode state is a
sliding-window K==V token cache optionally augmented with one or two
*windowed compressor* streams (DeepSeek-V4's Compressed Sparse Attention /
Heavily Compressed Attention and its Lightning Indexer):

- Every layer keeps a fixed-size ring of the last ``sliding_window`` token
  KV entries (single shared KV head, ``K == V``).
- Compressed layers additionally keep, per compressor stream:

  * a ``[batch, rate, feat]`` fill buffer of raw kv/gate projections for the
    tokens of the currently-open compression window,
  * a preallocated ``[batch, max_length // rate, dim]`` array of emitted
    compressed entries (window ``w`` writes index ``w``),
  * (two-series streams only) the previous closed window's Ca-slice
    ``overlap`` state of shape ``[batch, rate, dim]``.

All state is fixed-shape and JAX-transform-safe: the only dynamic quantity
is a per-row ``cache_position`` vector (``[batch]`` int32, tokens consumed so
far per stream), from which the buffer fill level (``pos % rate``), the
emitted-entry count (``pos // rate``) and the ring write slot
(``pos % sliding_window``) all derive. Rows are independent slots: eSurge
serves one request per row at its own position, while the vanilla
``generate()`` path advances all rows in lockstep (prompts must not be
left-padded there — batched prefill assumes every row starts at position 0).

Memory per layer (elements, batch ``B``, window ``W``, rate ``m``,
``T = max_length // m``, head dim ``D``, indexer dim ``Di``):

- sliding: ``B * W * D``
- HCA: ``+ 2 * B * m * D`` (buffers) ``+ B * T * D`` (entries)
- CSA: ``+ 2 * B * m * 2D`` (buffers) ``+ B * T * D`` (entries)
  ``+ 2 * B * m * D`` (overlap) and the same again at ``Di`` for the
  indexer stream (buffers ``2 * B * m * 2Di``, entries ``B * T * Di``,
  overlap ``2 * B * m * Di``).

The compression math itself (softmax-gated pooling, RMSNorm, rotary at the
deterministic window positions) lives in the model
(``easydel/modules/deepseek_v4``); this module only owns state layout and
allocation.
"""

from __future__ import annotations

import typing as tp

from eformer.pytree import auto_pytree, field
from jax import numpy as jnp
from jaxtyping import Array, Float, Int
from spectrax import PartitionAxis

from .._abstracts import BaseCache, BaseCacheConfig, BaseCacheView, BaseRunTimeMetadata

SLIDING_ATTENTION = "sliding_attention"
COMPRESSED_SPARSE_ATTENTION = "compressed_sparse_attention"
HEAVILY_COMPRESSED_ATTENTION = "heavily_compressed_attention"

_ALLOWED_LAYER_TYPES = (
    SLIDING_ATTENTION,
    COMPRESSED_SPARSE_ATTENTION,
    HEAVILY_COMPRESSED_ATTENTION,
)


@auto_pytree
class CompressedWindowCacheConfig(BaseCacheConfig):
    """Static configuration for the compressed sliding-window cache.

    Attributes:
        num_hidden_layers: Number of decoder layers.
        partition_axis: Axis-name configuration (kept for interface parity;
            state is currently replicated — see the module docstring).
        batch_size: Number of lockstep sequences.
        max_length: Maximum total stream length the cache must support
            (prefill + decode); bounds the compressed-entry allocations.
        sliding_window: Token-KV sliding window size (ring length).
        head_dim: Attention/compressor head dimension.
        index_head_dim: Lightning-indexer head dimension (CSA layers).
        csa_rate: Compression rate of ``compressed_sparse_attention`` layers.
        hca_rate: Compression rate of ``heavily_compressed_attention`` layers.
        layer_types: Per-layer attention regime, one of
            ``("sliding_attention", "compressed_sparse_attention",
            "heavily_compressed_attention")``.
    """

    num_hidden_layers: int = field(pytree_node=False)
    partition_axis: PartitionAxis = field(pytree_node=False)
    batch_size: int = field(pytree_node=False)
    max_length: int = field(pytree_node=False)
    sliding_window: int = field(pytree_node=False)
    head_dim: int = field(pytree_node=False)
    index_head_dim: int = field(pytree_node=False)
    csa_rate: int = field(pytree_node=False)
    hca_rate: int = field(pytree_node=False)
    layer_types: tuple[str, ...] = field(pytree_node=False)

    @classmethod
    def create(
        cls,
        num_hidden_layers: int,
        partition_axis: PartitionAxis,
        batch_size: int,
        max_length: int,
        sliding_window: int,
        head_dim: int,
        index_head_dim: int,
        csa_rate: int,
        hca_rate: int,
        layer_types: tp.Sequence[str],
    ) -> CompressedWindowCacheConfig:
        """Create and validate a compressed sliding-window cache config.

        Args:
            num_hidden_layers: Number of decoder layers.
            partition_axis: Axis-name configuration.
            batch_size: Number of lockstep sequences.
            max_length: Maximum supported stream length.
            sliding_window: Token-KV sliding window size.
            head_dim: Attention/compressor head dimension.
            index_head_dim: Lightning-indexer head dimension.
            csa_rate: CSA compression rate.
            hca_rate: HCA compression rate.
            layer_types: Per-layer attention regime.

        Returns:
            CompressedWindowCacheConfig: Validated configuration.

        Raises:
            ValueError: On non-positive dimensions, a layer-count mismatch,
                or an unknown layer type.
        """
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if sliding_window <= 0:
            raise ValueError("sliding_window must be positive")
        if head_dim <= 0 or index_head_dim <= 0:
            raise ValueError("head_dim and index_head_dim must be positive")
        if csa_rate <= 0 or hca_rate <= 0:
            raise ValueError("csa_rate and hca_rate must be positive")
        layer_types = tuple(layer_types)
        if len(layer_types) != num_hidden_layers:
            raise ValueError(f"len(layer_types) ({len(layer_types)}) must equal num_hidden_layers ({num_hidden_layers})")
        bad = [t for t in layer_types if t not in _ALLOWED_LAYER_TYPES]
        if bad:
            raise ValueError(f"layer_types entries must be one of {_ALLOWED_LAYER_TYPES}; got {bad}")
        return cls(
            num_hidden_layers=num_hidden_layers,
            partition_axis=partition_axis,
            batch_size=batch_size,
            max_length=max_length,
            sliding_window=sliding_window,
            head_dim=head_dim,
            index_head_dim=index_head_dim,
            csa_rate=csa_rate,
            hca_rate=hca_rate,
            layer_types=layer_types,
        )

    def rate_for(self, layer_type: str) -> int:
        """Return the compression rate for a compressed layer type.

        Args:
            layer_type: One of the compressed layer-type names.

        Returns:
            int: The window/compression rate.

        Raises:
            KeyError: If ``layer_type`` has no compressor.
        """
        if layer_type == COMPRESSED_SPARSE_ATTENTION:
            return self.csa_rate
        if layer_type == HEAVILY_COMPRESSED_ATTENTION:
            return self.hca_rate
        raise KeyError(f"layer type {layer_type!r} has no compression rate")


@auto_pytree
class CompressedWindowCacheView(BaseCacheView):
    """Per-layer state for compressed sliding-window attention.

    Field presence depends on the layer type (absent fields are ``None``):

    - all layers: ``window_kv`` ``[B, W, D]`` ring of post-RoPE token KV
      entries (token at stream position ``p`` occupies slot ``p % W``) and
      the per-row ``cache_position`` vector (tokens consumed so far).
    - CSA/HCA layers: ``compressor_*`` stream state. ``compressor_buffer_*``
      hold the raw kv/gate projections of the open window's tokens (token
      ``p`` at slot ``p % rate``; the gate is stored WITHOUT the learned
      position bias). ``compressor_entries`` holds emitted post-RoPE
      compressed entries at index ``w = p // rate``; ``pos // rate`` entries
      are live.
    - CSA layers only: ``compressor_overlap_*`` — the last closed window's
      Ca slice (kv in cache dtype; gate in fp32 WITH position bias applied,
      initialized to ``-inf`` so the slot has zero softmax weight before the
      first window closes) — plus the same four state groups again for the
      Lightning-indexer stream (``indexer_*``) at ``index_head_dim``.

    Attributes:
        window_kv: Token-KV ring ``[batch, sliding_window, head_dim]``.
        cache_position: Per-row int32 count of consumed stream tokens
            ``[batch]``.
        compressor_buffer_kv: Open-window kv projections
            ``[batch, rate, feat]`` (``feat = D`` for HCA, ``2D`` for CSA).
        compressor_buffer_gate: Open-window gate projections, same shape.
        compressor_entries: Emitted compressed KV entries
            ``[batch, max_length // rate, D]``.
        compressor_overlap_kv: Previous window's Ca kv slice
            ``[batch, rate, D]`` (CSA only).
        compressor_overlap_gate: Previous window's biased Ca gate slice in
            fp32 ``[batch, rate, D]`` (CSA only).
        indexer_buffer_kv: Indexer-stream analogue at ``2 * Di`` features.
        indexer_buffer_gate: Indexer-stream analogue at ``2 * Di`` features.
        indexer_entries: Indexer compressed keys
            ``[batch, max_length // rate, Di]``.
        indexer_overlap_kv: Indexer overlap kv ``[batch, rate, Di]``.
        indexer_overlap_gate: Indexer overlap gate (fp32)
            ``[batch, rate, Di]``.
        metadata: Shared static configuration.
        layer_index: Layer index this view belongs to.
    """

    window_kv: Float[Array, "batch window head_dim"] | None
    cache_position: Int[Array, "batch"]  # noqa: F821
    compressor_buffer_kv: Float[Array, "batch rate feat"] | None
    compressor_buffer_gate: Float[Array, "batch rate feat"] | None
    compressor_entries: Float[Array, "batch entries head_dim"] | None
    compressor_overlap_kv: Float[Array, "batch rate head_dim"] | None
    compressor_overlap_gate: Float[Array, "batch rate head_dim"] | None
    indexer_buffer_kv: Float[Array, "batch rate ifeat"] | None
    indexer_buffer_gate: Float[Array, "batch rate ifeat"] | None
    indexer_entries: Float[Array, "batch entries index_head_dim"] | None
    indexer_overlap_kv: Float[Array, "batch rate index_head_dim"] | None
    indexer_overlap_gate: Float[Array, "batch rate index_head_dim"] | None
    metadata: CompressedWindowCacheConfig = field(default=None)
    layer_index: int | None = field(pytree_node=False, default=None)

    @property
    def layer_type(self) -> str:
        """Return this layer's attention regime from the shared config."""
        return self.metadata.layer_types[self.layer_index]

    @property
    def rate(self) -> int:
        """Return this layer's compression rate (0 for sliding-only layers)."""
        if self.layer_type == SLIDING_ATTENTION:
            return 0
        return self.metadata.rate_for(self.layer_type)

    @property
    def num_entry_slots(self) -> int:
        """Return the static compressed-entry capacity (0 for sliding layers)."""
        rate = self.rate
        return 0 if rate == 0 else self.metadata.max_length // rate

    @classmethod
    def init(
        cls,
        config: CompressedWindowCacheConfig,
        layer_index: int | None = None,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        partition_specs=None,
    ) -> CompressedWindowCacheView:
        """Allocate zeroed state for one layer.

        Args:
            config: Shared static configuration.
            layer_index: Layer index (selects ``config.layer_types[i]``).
            dtype: Storage dtype for kv/entry state (gate overlap state is
                always fp32).
            partition_specs: Unused; state is replicated (see module doc).

        Returns:
            CompressedWindowCacheView: Freshly zeroed view.
        """
        del partition_specs
        batch = config.batch_size
        head_dim = config.head_dim
        layer_type = config.layer_types[layer_index if layer_index is not None else 0]

        window_kv = jnp.zeros((batch, config.sliding_window, head_dim), dtype=dtype)
        state: dict[str, Array | None] = {
            name: None
            for name in (
                "compressor_buffer_kv",
                "compressor_buffer_gate",
                "compressor_entries",
                "compressor_overlap_kv",
                "compressor_overlap_gate",
                "indexer_buffer_kv",
                "indexer_buffer_gate",
                "indexer_entries",
                "indexer_overlap_kv",
                "indexer_overlap_gate",
            )
        }
        if layer_type != SLIDING_ATTENTION:
            rate = config.rate_for(layer_type)
            n_entries = config.max_length // rate
            two_series = layer_type == COMPRESSED_SPARSE_ATTENTION
            feat = 2 * head_dim if two_series else head_dim
            state["compressor_buffer_kv"] = jnp.zeros((batch, rate, feat), dtype=dtype)
            state["compressor_buffer_gate"] = jnp.zeros((batch, rate, feat), dtype=dtype)
            state["compressor_entries"] = jnp.zeros((batch, n_entries, head_dim), dtype=dtype)
            if two_series:
                index_dim = config.index_head_dim
                state["compressor_overlap_kv"] = jnp.zeros((batch, rate, head_dim), dtype=dtype)
                state["compressor_overlap_gate"] = jnp.full((batch, rate, head_dim), -jnp.inf, dtype=jnp.float32)
                state["indexer_buffer_kv"] = jnp.zeros((batch, rate, 2 * index_dim), dtype=dtype)
                state["indexer_buffer_gate"] = jnp.zeros((batch, rate, 2 * index_dim), dtype=dtype)
                state["indexer_entries"] = jnp.zeros((batch, n_entries, index_dim), dtype=dtype)
                state["indexer_overlap_kv"] = jnp.zeros((batch, rate, index_dim), dtype=dtype)
                state["indexer_overlap_gate"] = jnp.full((batch, rate, index_dim), -jnp.inf, dtype=jnp.float32)
        return cls(
            window_kv=window_kv,
            cache_position=jnp.zeros((batch,), dtype=jnp.int32),
            metadata=config,
            layer_index=layer_index,
            **state,
        )

    def reset_slots(self, slot_indices: Array) -> CompressedWindowCacheView:
        """Reset the given batch rows (slots) to their freshly-initialized state.

        Used by serving engines that map one request per batch row: when a
        request finishes, its row is wiped so the next request assigned to
        the same slot starts from an empty stream. KV/buffer/entry state is
        zeroed, overlap gates return to ``-inf`` (zero pre-first-window
        softmax weight), and the row's ``cache_position`` returns to 0.

        Args:
            slot_indices: Int array of row indices to reset (out-of-range
                indices are ignored).

        Returns:
            CompressedWindowCacheView: View with the selected rows reset.
        """
        slot_arr = jnp.asarray(slot_indices, dtype=jnp.int32).reshape(-1)
        batch = self.cache_position.shape[0]
        reset_mask = jnp.zeros((batch,), dtype=jnp.bool_).at[slot_arr].set(True, mode="drop")
        return self.reset_rows(reset_mask)

    def reset_rows(self, reset_mask: Array) -> CompressedWindowCacheView:
        """Reset the rows selected by a boolean mask to empty-stream state.

        Traced-value form of :meth:`reset_slots` (safe inside ``jit``/``scan``):
        rows with ``True`` are re-initialized (zero state, ``-inf`` overlap
        gates, position 0), rows with ``False`` are untouched.

        Args:
            reset_mask: Bool array ``[batch]``; ``True`` selects rows to reset.

        Returns:
            CompressedWindowCacheView: View with the selected rows reset.
        """
        batch = self.cache_position.shape[0]
        keep = ~reset_mask.astype(jnp.bool_)

        def _zero_rows(value: Array | None) -> Array | None:
            if value is None:
                return None
            mask = keep.reshape((batch,) + (1,) * (value.ndim - 1))
            return jnp.where(mask, value, jnp.zeros((), dtype=value.dtype))

        def _neg_inf_rows(value: Array | None) -> Array | None:
            if value is None:
                return None
            mask = keep.reshape((batch,) + (1,) * (value.ndim - 1))
            return jnp.where(mask, value, jnp.full((), -jnp.inf, dtype=value.dtype))

        return self.replace(
            window_kv=_zero_rows(self.window_kv),
            cache_position=jnp.where(keep, self.cache_position, 0),
            compressor_buffer_kv=_zero_rows(self.compressor_buffer_kv),
            compressor_buffer_gate=_zero_rows(self.compressor_buffer_gate),
            compressor_entries=_zero_rows(self.compressor_entries),
            compressor_overlap_kv=_zero_rows(self.compressor_overlap_kv),
            compressor_overlap_gate=_neg_inf_rows(self.compressor_overlap_gate),
            indexer_buffer_kv=_zero_rows(self.indexer_buffer_kv),
            indexer_buffer_gate=_zero_rows(self.indexer_buffer_gate),
            indexer_entries=_zero_rows(self.indexer_entries),
            indexer_overlap_kv=_zero_rows(self.indexer_overlap_kv),
            indexer_overlap_gate=_neg_inf_rows(self.indexer_overlap_gate),
        )

    def concatenate_to_cache(self, *args, **kwargs) -> tp.Any:
        """Not used: DeepSeek-V4 attention updates the view fields directly.

        The compression math depends on model parameters (gate projections,
        position bias, RMSNorm, rotary), so updates happen in the model via
        functional ``.replace`` on this view rather than through a generic
        concatenate method.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "CompressedWindowCacheView is updated in-model via `.replace`; "
            "there is no generic concatenate_to_cache for compressed sliding-window state."
        )

    def __repr__(self) -> str:
        """Return a short representation with layer type and entry capacity."""
        window = self.window_kv.shape if self.window_kv is not None else None
        return (
            f"{self.__class__.__name__}(layer_index={self.layer_index}, "
            f"layer_type={self.metadata.layer_types[self.layer_index] if self.metadata else '?'}, "
            f"window_kv={window}, entry_slots={self.num_entry_slots if self.metadata else '?'})"
        )

    __str__ = __repr__


@auto_pytree
class CompressedWindowCache(BaseCache):
    """Multi-layer container of :class:`CompressedWindowCacheView` states.

    Attributes:
        views: One view per decoder layer (``None`` placeholders allowed for
            gradual population via ``init_empty``).
    """

    views: list[CompressedWindowCacheView | None]

    @classmethod
    def init_cache(
        cls,
        config: CompressedWindowCacheConfig,
        dtype: jnp.dtype | None = None,
    ) -> CompressedWindowCache:
        """Allocate views for all layers.

        Args:
            config: Shared static configuration.
            dtype: Storage dtype for kv/entry state; defaults to bfloat16.

        Returns:
            CompressedWindowCache: Cache with one initialized view per layer.
        """
        return cls(
            views=[
                CompressedWindowCacheView.init(
                    config=config,
                    layer_index=layer_index,
                    dtype=dtype if dtype is not None else jnp.bfloat16,
                )
                for layer_index in range(config.num_hidden_layers)
            ]
        )

    def reset_slots(self, slot_indices: Array) -> CompressedWindowCache:
        """Reset the given batch rows (slots) to empty-stream state in every layer.

        Args:
            slot_indices: Int array of row indices to reset.

        Returns:
            CompressedWindowCache: Cache with the selected rows reset in all
            initialized views (``None`` placeholders pass through).
        """
        return CompressedWindowCache(
            views=[view.reset_slots(slot_indices) if view is not None else None for view in self.views]
        )

    @classmethod
    def init_empty(cls, num_hidden_layers: int) -> CompressedWindowCache:
        """Create a container with ``None`` placeholders for each layer.

        Args:
            num_hidden_layers: Number of layer slots.

        Returns:
            CompressedWindowCache: Empty cache structure.
        """
        return cls(views=[None for _ in range(num_hidden_layers)])

    def __repr__(self) -> str:
        """Return a multi-line representation listing each layer view."""
        return f"{self.__class__.__name__}(\n  " + "\n  ".join(str(view) for view in self.views) + "\n)"

    __str__ = __repr__


class CompressedWindowMetadata(BaseRunTimeMetadata):
    """Runtime-metadata placeholder for compressed sliding-window caches.

    All step bookkeeping (fill level, entry count, ring slot) derives from
    the views' scalar ``cache_position``, so no extra runtime fields are
    currently needed; this class exists so future per-step metadata can be
    added without changing call sites.
    """

    ...


__all__ = (
    "CompressedWindowCache",
    "CompressedWindowCacheConfig",
    "CompressedWindowCacheView",
    "CompressedWindowMetadata",
)
