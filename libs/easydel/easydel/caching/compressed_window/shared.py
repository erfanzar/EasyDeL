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

"""Shared compressed sliding-window cache (DeepSeek-V4.1 style attention state).

DeepSeek-V4.1 compresses KV only on a few *source* layers; every later layer with the same compression ratio
attends to that source's compressed entries, and the indexer keys are likewise owned by the source. Decode state is
therefore stored only where it is produced:

- every layer: a ``[batch, sliding_window, head_dim]`` ring of post-RoPE token KV (token at stream position ``p`` in
  slot ``p % W``) and a per-row ``cache_position`` ``[batch]`` int32;
- KV-source layers: emitted post-RoPE compressed entries ``[batch, max_length // r, head_dim]`` (group ``g`` at index
  ``g``) and, for ratio ``r > 1``, the open group's fp32 kv / gate-score projections ``[batch, r, head_dim]`` (token
  ``p`` at slot ``p % r``);
- index-key owners (index sources that are also KV sources): indexer keys ``[batch, max_length // r, index_head_dim]``.

Consumer layers, top-k selections and candidate blocks hold no persistent state: they are recomputed every step from
the sources' state, which the model threads down the layer loop. All state is fixed-shape and transform-safe; rows
are independent slots at their own ``cache_position``. The pooling, normalization and rotary math live in the model
(``easydel/modules/deepseek_v41``); this module owns layout and allocation only.

Memory per layer (elements, batch ``B``, window ``W``, ratio ``r``, ``T = max_length // r``, head dim ``D``, indexer
dim ``Di``): ``B*W*D`` on every layer, ``+ B*T*D (+ 2*B*r*D if r > 1)`` on KV sources, ``+ B*T*Di`` on key owners.
"""

from __future__ import annotations

import functools
import typing as tp

import jax
import numpy
from eformer.pytree import auto_pytree, field
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array, Float, Int
from spectrax import PartitionAxis

from .._abstracts import BaseCache, BaseCacheConfig, BaseCacheView, BaseRunTimeMetadata


def slot_partition_axes(mesh: tp.Any, batch_size: int) -> tuple[str, ...]:
    """Mesh axes the cache slot (batch) axis is split over; ``()`` keeps the cache replicated.

    Every non-pipeline axis of size > 1 is taken, in mesh order, while the running product still divides
    ``batch_size``. Rows are independent streams, so a slot-sharded buffer keeps every cache write and top-k gather
    local (measured: ``probe_cache_slot_sharding.py``, no whole-buffer collective) and holds ``1/N`` of the rows per
    device. Pipeline (MPMD) meshes keep the replicated layout.

    Args:
        mesh: Model mesh (a JAX mesh or a SpectraX mesh exposing ``jax_mesh``), or ``None``.
        batch_size: Number of cache rows.

    Returns:
        Axis names for the slot dimension.
    """
    if mesh is None or getattr(mesh, "is_mpmd", False):
        return ()
    jax_mesh = getattr(mesh, "jax_mesh", mesh)
    shape = getattr(jax_mesh, "shape", None)
    names = getattr(jax_mesh, "axis_names", None)
    if not shape or not names:
        return ()
    if "pp" in names and int(shape["pp"]) > 1:
        return ()
    axes: list[str] = []
    product = 1
    for name in names:
        size = int(shape[name])
        if name == "pp" or size <= 1:
            continue
        if batch_size % (product * size) == 0:
            axes.append(str(name))
            product *= size
    return tuple(axes)


@auto_pytree
class SharedCompressedWindowCacheConfig(BaseCacheConfig):
    """Static configuration for the shared compressed sliding-window cache.

    Attributes:
        num_hidden_layers: Number of decoder layers with a view.
        partition_axis: Axis-name configuration (state is replicated; kept for interface parity).
        batch_size: Number of cache rows (lockstep sequences or request slots).
        max_length: Maximum stream length per row; bounds the compressed-entry allocations.
        sliding_window: Token-KV ring length.
        head_dim: Attention / compressor head dimension.
        index_head_dim: Indexer key dimension.
        compress_ratios: Per-layer compression ratio (0 = window only), one entry per layer.
        kv_source_layer_ids: Layers that compress their own KV and own compressed entries.
        index_key_layer_ids: Layers that own indexer keys (must be KV sources).
        slot_axes: Mesh axes the slot axis of the compressed entries and index keys is split over (``()`` =
            replicated); see :func:`slot_partition_axes`. Rings and open-group state stay replicated.
    """

    num_hidden_layers: int = field(pytree_node=False)
    partition_axis: PartitionAxis = field(pytree_node=False)
    batch_size: int = field(pytree_node=False)
    max_length: int = field(pytree_node=False)
    sliding_window: int = field(pytree_node=False)
    head_dim: int = field(pytree_node=False)
    index_head_dim: int = field(pytree_node=False)
    compress_ratios: tuple[int, ...] = field(pytree_node=False)
    kv_source_layer_ids: tuple[int, ...] = field(pytree_node=False)
    index_key_layer_ids: tuple[int, ...] = field(pytree_node=False)
    slot_axes: tuple[str, ...] = field(pytree_node=False, default=())

    @classmethod
    def create(
        cls,
        *,
        num_hidden_layers: int,
        partition_axis: PartitionAxis,
        batch_size: int,
        max_length: int,
        sliding_window: int,
        head_dim: int,
        index_head_dim: int,
        compress_ratios: tp.Sequence[int],
        kv_source_layer_ids: tp.Sequence[int],
        index_key_layer_ids: tp.Sequence[int],
        slot_axes: tp.Sequence[str] = (),
    ) -> SharedCompressedWindowCacheConfig:
        """Create and validate a shared compressed sliding-window cache config.

        Args:
            num_hidden_layers: Number of decoder layers with a view.
            partition_axis: Axis-name configuration.
            batch_size: Number of cache rows.
            max_length: Maximum stream length per row.
            sliding_window: Token-KV ring length.
            head_dim: Attention / compressor head dimension.
            index_head_dim: Indexer key dimension.
            compress_ratios: Per-layer compression ratios (at least ``num_hidden_layers`` entries).
            kv_source_layer_ids: Layers owning compressed entries.
            index_key_layer_ids: Layers owning indexer keys.
            slot_axes: Mesh axes for the slot axis of entries / index keys (``()`` = replicated).

        Returns:
            SharedCompressedWindowCacheConfig: Validated configuration.

        Raises:
            ValueError: On non-positive sizes, inconsistent ownership, or a source with ratio 0.
        """
        for name, value in (
            ("num_hidden_layers", num_hidden_layers),
            ("batch_size", batch_size),
            ("max_length", max_length),
            ("sliding_window", sliding_window),
            ("head_dim", head_dim),
            ("index_head_dim", index_head_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}")
        ratios = tuple(int(r) for r in compress_ratios)[:num_hidden_layers]
        if len(ratios) != num_hidden_layers or any(r < 0 for r in ratios):
            raise ValueError(f"compress_ratios must give a non-negative ratio for each of {num_hidden_layers} layers")
        sources = tuple(sorted(int(i) for i in kv_source_layer_ids))
        key_owners = tuple(sorted(int(i) for i in index_key_layer_ids))
        for src in sources:
            if not 0 <= src < num_hidden_layers or ratios[src] == 0:
                raise ValueError(f"KV source layer {src} is out of range or has compress ratio 0")
        if not set(key_owners) <= set(sources):
            raise ValueError(f"index key owners {key_owners} must be KV sources {sources}")
        return cls(
            num_hidden_layers=num_hidden_layers,
            partition_axis=partition_axis,
            batch_size=batch_size,
            max_length=max_length,
            sliding_window=sliding_window,
            head_dim=head_dim,
            index_head_dim=index_head_dim,
            compress_ratios=ratios,
            kv_source_layer_ids=sources,
            index_key_layer_ids=key_owners,
            slot_axes=tuple(str(a) for a in slot_axes),
        )

    def slot_spec(self, ndim: int) -> PartitionSpec:
        """PartitionSpec of an ``ndim``-rank slot-major buffer (or its per-slot update rows)."""
        if not self.slot_axes:
            return PartitionSpec()
        return PartitionSpec(tuple(self.slot_axes), *([None] * (ndim - 1)))

    def ratio(self, layer_index: int) -> int:
        """Compression ratio of ``layer_index`` (0 = window only)."""
        return int(self.compress_ratios[layer_index])

    def num_entry_slots(self, layer_index: int) -> int:
        """Compressed-entry capacity owned by ``layer_index`` (0 unless it is a KV source)."""
        if layer_index not in self.kv_source_layer_ids:
            return 0
        return self.max_length // self.ratio(layer_index)


@auto_pytree
class SharedCompressedWindowCacheView(BaseCacheView):
    """Per-layer decode state; fields a layer does not own are ``None``.

    Attributes:
        window_kv: Post-RoPE token-KV ring ``[batch, sliding_window, head_dim]``.
        cache_position: Per-row count of consumed stream tokens ``[batch]`` int32.
        compressor_kv_state: Open-group fp32 kv projections ``[batch, ratio, head_dim]`` (KV sources, ratio > 1).
        compressor_score_state: Open-group fp32 gate scores ``[batch, ratio, head_dim]`` (KV sources, ratio > 1).
        compressed_kv: Emitted post-RoPE compressed entries ``[batch, max_length // ratio, head_dim]`` (KV sources).
        index_k: Indexer keys ``[batch, max_length // ratio, index_head_dim]`` (index-key owners).
        metadata: Shared static configuration.
        layer_index: Layer this view belongs to.
    """

    window_kv: Float[Array, "batch window head_dim"] | None
    cache_position: Int[Array, "batch"]  # noqa: F821
    compressor_kv_state: Float[Array, "batch ratio head_dim"] | None
    compressor_score_state: Float[Array, "batch ratio head_dim"] | None
    compressed_kv: Float[Array, "batch entries head_dim"] | None
    index_k: Float[Array, "batch entries index_head_dim"] | None
    metadata: SharedCompressedWindowCacheConfig = field(default=None)
    layer_index: int | None = field(pytree_node=False, default=None)

    @property
    def ratio(self) -> int:
        """Compression ratio of this layer (0 = window only)."""
        return self.metadata.ratio(self.layer_index)

    @property
    def num_entry_slots(self) -> int:
        """Compressed-entry capacity owned by this layer."""
        return self.metadata.num_entry_slots(self.layer_index)

    @classmethod
    def init(
        cls,
        config: SharedCompressedWindowCacheConfig,
        layer_index: int | None = None,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        partition_specs=None,
        mesh: tp.Any = None,
    ) -> SharedCompressedWindowCacheView:
        """Allocate zeroed state for one layer.

        Args:
            config: Shared static configuration.
            layer_index: Layer index.
            dtype: Storage dtype of the ring, entries and index keys (open-group state is always fp32).
            partition_specs: Unused (the layout comes from ``config.slot_axes``).
            mesh: Mesh for the slot-sharded entry / index-key buffers; required when ``config.slot_axes`` is set,
                so the buffers are created sharded instead of materialised replicated first.

        Returns:
            SharedCompressedWindowCacheView: Freshly zeroed view.
        """
        del partition_specs
        layer_index = 0 if layer_index is None else layer_index
        batch, head_dim = config.batch_size, config.head_dim
        jax_mesh = getattr(mesh, "jax_mesh", mesh)
        if config.slot_axes and jax_mesh is None:
            raise ValueError("A slot-sharded SharedCompressedWindowCacheConfig needs the mesh at init.")

        # With a mesh, every buffer is allocated with an explicit NamedSharding. Executors take the compiled input /
        # output shardings of the cache from its leaves (``spx.extract_sharding_structure`` yields ``None`` for a leaf
        # without one), so an unsharded leaf would let every compiled step shape pick its own layout and a cache
        # produced by one bucket would be rejected by the next.
        place = jax_mesh is not None and not getattr(mesh, "is_mpmd", False)

        def _zeros(shape: tuple[int, ...], spec: PartitionSpec, buf_dtype) -> Array:
            if not place:
                return jnp.zeros(shape, dtype=buf_dtype)
            return jnp.zeros(shape, dtype=buf_dtype, device=NamedSharding(jax_mesh, spec))

        def _slot_zeros(shape: tuple[int, ...]) -> Array:
            return _zeros(shape, config.slot_spec(len(shape)), dtype)

        def _rep_zeros(shape: tuple[int, ...], buf_dtype=dtype) -> Array:
            return _zeros(shape, PartitionSpec(), buf_dtype)

        state: dict[str, Array | None] = dict.fromkeys(
            ("compressor_kv_state", "compressor_score_state", "compressed_kv", "index_k")
        )
        if layer_index in config.kv_source_layer_ids:
            ratio = config.ratio(layer_index)
            entries = config.max_length // ratio
            state["compressed_kv"] = _slot_zeros((batch, entries, head_dim))
            if ratio > 1:
                state["compressor_kv_state"] = _rep_zeros((batch, ratio, head_dim), jnp.float32)
                state["compressor_score_state"] = _rep_zeros((batch, ratio, head_dim), jnp.float32)
            if layer_index in config.index_key_layer_ids:
                state["index_k"] = _slot_zeros((batch, entries, config.index_head_dim))
        return cls(
            window_kv=_rep_zeros((batch, config.sliding_window, head_dim)),
            cache_position=_rep_zeros((batch,), jnp.int32),
            metadata=config,
            layer_index=layer_index,
            **state,
        )

    def constrain_layout(self, mesh: tp.Any) -> SharedCompressedWindowCacheView:
        """Pin every buffer to its allocation layout (call on the view a layer returns).

        Ring, positions and open-group state are replicated; compressed entries and index keys follow
        ``metadata.slot_spec``. Without this the sharding of the projections propagates into whichever buffers a step
        wrote, so different compiled step shapes return the cache in different layouts and a cache produced by one no
        longer matches another's compiled input shardings. No-op without a multi-device (non-MPMD) mesh.

        Args:
            mesh: Model mesh (JAX or SpectraX), or ``None``.

        Returns:
            The view with layout-constrained buffers.
        """
        if mesh is None or getattr(mesh, "is_mpmd", False):
            return self
        jax_mesh = getattr(mesh, "jax_mesh", mesh)
        if jax_mesh is None or getattr(jax_mesh, "size", 1) <= 1:
            return self
        meta = self.metadata

        def _pin(x: Array | None, spec: PartitionSpec) -> Array | None:
            if x is None:
                return None
            return jax.lax.with_sharding_constraint(x, NamedSharding(jax_mesh, spec))

        def _slot(x: Array | None) -> Array | None:
            if x is None:
                return None
            return _pin(x, meta.slot_spec(x.ndim) if meta is not None else PartitionSpec())

        rep = PartitionSpec()
        return self.replace(
            window_kv=_pin(self.window_kv, rep),
            cache_position=_pin(self.cache_position, rep),
            compressor_kv_state=_pin(self.compressor_kv_state, rep),
            compressor_score_state=_pin(self.compressor_score_state, rep),
            compressed_kv=_slot(self.compressed_kv),
            index_k=_slot(self.index_k),
        )

    def reset_rows(self, reset_mask: Array, *, stream_state_only: bool = False) -> SharedCompressedWindowCacheView:
        """Reset the rows selected by ``reset_mask`` (``[batch]`` bool) to the empty-stream state.

        Args:
            reset_mask: ``[batch]`` bool, rows to reset.
            stream_state_only: Reset only ``cache_position`` and the open-group compressor state, leaving the
                ring, compressed entries and index keys untouched. Valid for readers that never look past a stream's
                position (masked ring slots, reach-bounded entry selection); avoids a select over the full
                ``max_length``-sized buffers. ``False`` (default) zeroes every buffer row.

        Returns:
            The view with the selected rows reset.
        """
        batch = self.cache_position.shape[0]
        keep = ~reset_mask.astype(jnp.bool_)

        def _zero_rows(value: Array | None) -> Array | None:
            if value is None:
                return None
            mask = keep.reshape((batch,) + (1,) * (value.ndim - 1))
            return jnp.where(mask, value, jnp.zeros((), dtype=value.dtype))

        if stream_state_only:
            return self.replace(
                cache_position=jnp.where(keep, self.cache_position, 0),
                compressor_kv_state=_zero_rows(self.compressor_kv_state),
                compressor_score_state=_zero_rows(self.compressor_score_state),
            )
        return self.replace(
            window_kv=_zero_rows(self.window_kv),
            cache_position=jnp.where(keep, self.cache_position, 0),
            compressor_kv_state=_zero_rows(self.compressor_kv_state),
            compressor_score_state=_zero_rows(self.compressor_score_state),
            compressed_kv=_zero_rows(self.compressed_kv),
            index_k=_zero_rows(self.index_k),
        )

    def concatenate_to_cache(self, *args, **kwargs) -> tp.Any:
        """Not used: the model updates the view fields directly via ``.replace``.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "SharedCompressedWindowCacheView is updated in-model via `.replace`; there is no generic concatenate."
        )

    def __repr__(self) -> str:
        """Short representation with the layer's ratio and ownership."""
        owns = [n for n in ("compressed_kv", "index_k") if getattr(self, n) is not None]
        ratio = self.ratio if self.metadata is not None else "?"
        return f"{self.__class__.__name__}(layer_index={self.layer_index}, ratio={ratio}, owns={owns})"

    __str__ = __repr__


@auto_pytree
class SharedCompressedWindowCache(BaseCache):
    """Multi-layer container of :class:`SharedCompressedWindowCacheView` states.

    Attributes:
        views: One view per decoder layer (``None`` placeholders allowed via ``init_empty``).
    """

    views: list[SharedCompressedWindowCacheView | None]

    @classmethod
    def init_cache(
        cls,
        config: SharedCompressedWindowCacheConfig,
        dtype: jnp.dtype | None = None,
        mesh: tp.Any = None,
    ) -> SharedCompressedWindowCache:
        """Allocate views for all layers (``dtype`` defaults to bfloat16; ``mesh`` is needed when slot-sharded)."""
        return cls(
            views=[
                SharedCompressedWindowCacheView.init(
                    config=config,
                    layer_index=i,
                    dtype=dtype if dtype is not None else jnp.bfloat16,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def reset_slots(self, slot_indices: Array) -> SharedCompressedWindowCache:
        """Reset the given rows in every layer with one compiled call over a fixed ``[batch]`` mask."""
        views = [v for v in self.views if v is not None]
        if not views:
            return self
        batch = int(views[0].cache_position.shape[0])
        idx = numpy.asarray(slot_indices, dtype=numpy.int64).reshape(-1)
        mask = numpy.zeros((batch,), dtype=bool)
        mask[idx[(idx >= 0) & (idx < batch)]] = True
        if not mask.any():
            return self
        # Pin every output leaf to its input leaf's sharding: the [batch] mask meets slot-sharded entry buffers, and
        # without explicit out_shardings GSPMD propagates that slot layout into the replicated ring / position
        # buffers, which the compiled serving steps then reject (input-sharding mismatch).
        leaves, treedef = jax.tree_util.tree_flatten(self)
        shardings = tuple(getattr(leaf, "sharding", None) for leaf in leaves)
        if leaves and all(s is not None for s in shardings):
            return _reset_rows_pinned(treedef, shardings)(self, jnp.asarray(mask))
        return _reset_rows_compiled(self, jnp.asarray(mask))

    @classmethod
    def init_empty(cls, num_hidden_layers: int) -> SharedCompressedWindowCache:
        """Create a container with ``None`` placeholders for each layer."""
        return cls(views=[None for _ in range(num_hidden_layers)])

    def __repr__(self) -> str:
        """Multi-line representation listing each layer view."""
        return f"{self.__class__.__name__}(\n  " + "\n  ".join(str(v) for v in self.views) + "\n)"

    __str__ = __repr__


class SharedCompressedWindowMetadata(BaseRunTimeMetadata):
    """Runtime-metadata placeholder: all step bookkeeping derives from the views' ``cache_position``."""

    ...


def _reset_rows_impl(cache: SharedCompressedWindowCache, reset_mask: Array) -> SharedCompressedWindowCache:
    """Reset the masked rows in every layer (fixed ``[batch]`` mask shape)."""
    return SharedCompressedWindowCache(
        views=[view.reset_rows(reset_mask) if view is not None else None for view in cache.views]
    )


_reset_rows_compiled = jax.jit(_reset_rows_impl)


@functools.lru_cache(maxsize=16)
def _reset_rows_pinned(treedef: tp.Any, shardings: tuple[tp.Any, ...]) -> tp.Callable[..., SharedCompressedWindowCache]:
    """``_reset_rows_impl`` compiled with the cache's own per-leaf shardings as input and output shardings."""
    cache_shardings = jax.tree_util.tree_unflatten(treedef, list(shardings))
    return jax.jit(_reset_rows_impl, in_shardings=(cache_shardings, None), out_shardings=cache_shardings)


__all__ = (
    "SharedCompressedWindowCache",
    "SharedCompressedWindowCacheConfig",
    "SharedCompressedWindowCacheView",
    "SharedCompressedWindowMetadata",
    "slot_partition_axes",
)
