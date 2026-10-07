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

"""Gated n-gram (Engram) embedding written into hyper-connection streams.

The n-gram hashing runs on the host; this layer consumes its row ids. The ids fetch ``num_hash_cols`` table rows per
token, ``kv_proj`` maps them to one key per stream plus a shared value, and each stream's gate is
``sigmoid(signed_sqrt(normalized <h, q_weight * k_weight * key>))``.
"""

from __future__ import annotations

import functools
import itertools
import typing as tp
from dataclasses import dataclass

import jax
import numpy as np
import spectrax as spx
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from easydel.layers.embeddings._embeddings import Embed

__all__ = (
    "FP8_SCALE_BLOCK",
    "GatedEngram",
    "GatedEngramConfig",
    "engram_table_padded_rows",
)

_DEVICE = "device"
_PINNED_HOST = "pinned_host"
_MEMORY_KINDS = (_DEVICE, _PINNED_HOST)
_PARAM = "param"
_FP8 = "fp8"
_TABLE_DTYPES = (_PARAM, _FP8)

#: Columns sharing one e8m0 scale in the fp8 table layout (the reference ``fp8_block_size``).
FP8_SCALE_BLOCK = 32


def engram_table_padded_rows(rows: int, num_shards: int) -> int:
    """Rows of a row-sharded fp8 table after padding to a multiple of ``num_shards`` (reference ``convert.py``)."""
    return -(-int(rows) // int(num_shards)) * int(num_shards)


def _e8m0_to_f32(scale: Array) -> Array:
    """Exact ``2 ** (byte - 127)`` for e8m0 bytes via the float32 exponent field (byte 255 is NaN, 0 is 2**-127)."""
    byte = scale.astype(jnp.uint32)
    bits = jnp.where(byte == 0, jnp.uint32(0x00400000), byte << 23)
    bits = jnp.where(byte == 255, jnp.uint32(0x7FC00000), bits)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


#: Host-resident fp8 engram tables, ``key -> (weight uint8 [rows, head_dim], scale uint8 [rows, head_dim // 32])``.
#: Kept outside the module tree on purpose: a spectrax Variable would be passed into every jit as an argument (or
#: captured as a constant), and a plain attribute becomes GraphDef metadata. The module stores only its string key.
_HOST_TABLES: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_HOST_TABLE_IDS = itertools.count()


def _gather_host_rows(key: str, ids) -> tuple[np.ndarray, np.ndarray]:
    """NumPy row gather from the registered host table ``key`` (runs inside ``jax.pure_callback``)."""
    weight, scale = _HOST_TABLES[key]
    ids = np.asarray(ids)
    flat = ids.reshape(-1)
    return (
        weight[flat].reshape(*ids.shape, weight.shape[1]),
        scale[flat].reshape(*ids.shape, scale.shape[1]),
    )


def _pack_u32(rows: np.ndarray) -> np.ndarray:
    """View uint8 rows ``[..., n]`` as little-endian uint32 words ``[..., ceil(n / 4)]`` (zero-padded tail)."""
    pad = (-rows.shape[-1]) % 4
    if pad:
        rows = np.pad(rows, [(0, 0)] * (rows.ndim - 1) + [(0, pad)])
    return np.ascontiguousarray(rows).view(np.uint32)


def _unpack_u8(words: Array, width: int) -> Array:
    """Inverse of :func:`_pack_u32` on device: uint32 words ``[..., w]`` back to uint8 rows ``[..., width]``."""
    rows = jax.lax.bitcast_convert_type(words, jnp.uint8).reshape(*words.shape[:-1], words.shape[-1] * 4)
    return rows[..., :width]


def _gather_host_rows_packed(key: str, ids) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_gather_host_rows` with both outputs packed into uint32 words (the ``pure_callback`` body)."""
    weight, scale = _gather_host_rows(key, ids)
    return _pack_u32(weight), _pack_u32(scale)


def _dequantize_rows(rows: Array, scales: Array) -> Array:
    """float32 rows ``[..., head_dim]`` from fp8 e4m3 ``rows`` and e8m0 byte ``scales`` ``[..., head_dim // 32]``."""
    blocks = rows.astype(jnp.float32).reshape(*rows.shape[:-1], -1, FP8_SCALE_BLOCK)
    return (blocks * _e8m0_to_f32(scales)[..., None]).reshape(rows.shape)


def _take_rows(table: Array, ids: Array) -> Array:
    """Row gather ``table[ids]`` executed inside the host compute region.

    Uses ``lax.gather`` in ``PROMISE_IN_BOUNDS`` mode instead of ``jnp.take``: ``jnp.take``'s default ``fill`` mode adds
    a negative-index ``where`` and an out-of-bounds compare whose constants live in device memory, which a host compute
    region rejects (``memory_space of all inputs ... must be the same``). Engram hash ids are always valid table rows.
    """
    dnums = jax.lax.GatherDimensionNumbers(
        offset_dims=(ids.ndim,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,),
    )
    return jax.lax.gather(
        table,
        ids[..., None],
        dimension_numbers=dnums,
        slice_sizes=(1, table.shape[1]),
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


@dataclass(frozen=True)
class GatedEngramConfig:
    """Scalar settings of :class:`GatedEngram`, independent of model configs.

    Args:
        hidden_size: Hidden width of each stream.
        hc_mult: Number of hyper-connection streams.
        table_rows: Rows of the embedding table.
        num_hash_cols: Hash columns (table rows fetched) per token.
        head_dim: Width of one table row.
        eps: Epsilon of the gate normalizations.
        initializer_range: Standard deviation of the ``kv_proj`` initializer.
        table_memory_kind: ``"device"`` (default) or ``"pinned_host"``. With ``"pinned_host"`` the embedding table
            lives in pinned host memory as a non-trainable ``buffers`` leaf (same ``embed.weight`` path) and the
            lookup gathers only the requested rows on the host before moving them to device memory.
        table_dtype: ``"param"`` (default) stores the table as one ``embed.weight`` parameter in ``param_dtype``.
            ``"fp8"`` stores the released layout instead: ``embed.weight`` float8_e4m3fn ``[rows, head_dim]`` plus
            ``embed.scale`` uint8 (e8m0) ``[rows, head_dim // 32]``, both non-trainable ``buffers`` leaves, row-sharded
            over all mesh devices and dequantized per looked-up row. With ``table_memory_kind="pinned_host"`` the
            two leaves stay 1-row placeholders and the full table lives in ordinary host RAM (a process-level
            registry keyed by the module's ``_host_table_key``), gathered by row id in NumPy through
            ``jax.pure_callback`` so the per-call cost follows rows looked up, not table size. Fill it with
            :meth:`GatedEngram.set_table` or the checkpoint loader.
        layer_hash_index: Position of this layer among the model's engram layers, i.e. its index on the
            ``n_engram_layers`` axis of the host-computed hash ids / gathered rows.
    """

    hidden_size: int
    hc_mult: int
    table_rows: int
    num_hash_cols: int
    head_dim: int
    eps: float = 1e-6
    initializer_range: float = 0.02
    table_memory_kind: str = "device"
    table_dtype: str = "param"
    layer_hash_index: int = 0

    @classmethod
    def from_model_config(cls, config: tp.Any, layer_idx: int) -> GatedEngramConfig:
        """Settings of the engram on decoder layer ``layer_idx`` from a model config's ``engram_*`` fields.

        Reads ``engram_layer_ids``, ``engram_max_ngram_size``, ``engram_n_heads``, ``engram_vocab_size``,
        ``engram_head_dim``, ``hidden_size``, ``hc_mult``, ``rms_norm_eps``, ``initializer_range`` and optionally
        ``engram_num_embeddings``, ``engram_table_memory_kind`` and ``engram_table_dtype``.

        Args:
            config: Model config.
            layer_idx: Global decoder layer index; must be listed in ``config.engram_layer_ids``.

        Returns:
            The layer settings.

        Raises:
            ValueError: If ``layer_idx`` carries no engram, or ``engram_num_embeddings`` disagrees with the layout.
        """
        from ._hashing import engram_table_rows

        layer_ids = [int(i) for i in config.engram_layer_ids]
        if int(layer_idx) not in layer_ids:
            raise ValueError(f"layer {layer_idx} is not an engram layer (engram_layer_ids={layer_ids}).")
        index = layer_ids.index(int(layer_idx))
        rows = engram_table_rows(config)
        return cls(
            hidden_size=int(config.hidden_size),
            hc_mult=int(config.hc_mult),
            table_rows=rows[index],
            num_hash_cols=(int(config.engram_max_ngram_size) - 1) * int(config.engram_n_heads),
            head_dim=int(config.engram_head_dim),
            eps=float(config.rms_norm_eps),
            initializer_range=float(config.initializer_range),
            table_memory_kind=getattr(config, "engram_table_memory_kind", None) or _DEVICE,
            table_dtype=getattr(config, "engram_table_dtype", None) or _PARAM,
            layer_hash_index=index,
        )

    def __post_init__(self):
        if self.table_memory_kind not in _MEMORY_KINDS:
            raise ValueError(f"table_memory_kind must be one of {_MEMORY_KINDS}; got {self.table_memory_kind!r}.")
        if self.table_dtype not in _TABLE_DTYPES:
            raise ValueError(f"table_dtype must be one of {_TABLE_DTYPES}; got {self.table_dtype!r}.")
        if self.table_dtype == _FP8 and self.head_dim % FP8_SCALE_BLOCK:
            raise ValueError(f"fp8 tables need head_dim divisible by {FP8_SCALE_BLOCK}; got {self.head_dim}.")


class _Fp8EngramTable(spx.Module):
    """fp8 e4m3 rows plus one e8m0 byte scale per 32 columns, as ``weight`` / ``scale`` buffers."""

    def __init__(self, rows: int, head_dim: int, *, placeholder: bool):
        self.num_embeddings = rows
        self.features = head_dim
        n = 1 if placeholder else rows
        self.weight = spx.Buffer(jnp.zeros((n, head_dim), jnp.float8_e4m3fn))
        self.scale = spx.Buffer(jnp.ones((n, head_dim // FP8_SCALE_BLOCK), jnp.uint8))


class GatedEngram(spx.Module):
    """Adds a gated n-gram embedding value to every hyper-connection stream.

    Parameter leaves: ``embed.weight`` ``[table_rows, head_dim]``, ``kv_proj.weight``
    ``[num_hash_cols * head_dim, hidden * (hc_mult + 1)]``, ``q_weight`` and ``k_weight`` ``[hc_mult, hidden]``.

    With ``table_memory_kind="pinned_host"`` the table is placed in pinned host memory
    (:meth:`place_table_on_host`) and becomes a non-trainable ``buffers`` leaf: a dense gradient (and optimizer state)
    for a table of this size would defeat the offload, and no gradient flows to the table. ``kv_proj``, ``q_weight``
    and ``k_weight`` stay trainable. The lookup runs under ``jax.experimental.compute_on("device_host")`` and moves
    only the gathered rows to device memory, so it is jittable in training forwards and decode.
    """

    def __init__(
        self,
        config: GatedEngramConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the layer.

        Args:
            config: Scalar engram settings.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators (consumed in embed / kv_proj / q_weight / k_weight order).
        """
        # Deferred imports: ``easydel.infra.utils`` and ``easydel.layers.linears`` import this package's parent at
        # module level, so top-level imports here would be circular.
        from easydel.infra.utils import ArrayParam
        from easydel.layers.linears import ColumnParallelLinear

        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.eps
        self.layer_hash_index = int(config.layer_hash_index)
        self.num_hash_cols = config.num_hash_cols
        self.table_dtype = config.table_dtype
        self.table_rows = config.table_rows
        self.compute_dtype_name = jnp.dtype(dtype).name
        if config.table_dtype == _FP8:
            self.embed = _Fp8EngramTable(
                config.table_rows, config.head_dim, placeholder=config.table_memory_kind == _PINNED_HOST
            )
        else:
            self.embed = Embed(config.table_rows, config.head_dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.kv_proj = ColumnParallelLinear(
            self.num_hash_cols * config.head_dim,
            config.hidden_size * (config.hc_mult + 1),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        ones = dict(shape=(config.hc_mult, config.hidden_size), dtype=param_dtype, init_method="ones")
        self.q_weight = ArrayParam.bound(**ones, key=rngs.param)
        self.k_weight = ArrayParam.bound(**ones, key=rngs.param)
        self.table_memory_kind = _DEVICE
        if config.table_memory_kind == _PINNED_HOST:
            self.place_table_on_host()

    @classmethod
    def from_config(
        cls,
        config: tp.Any,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ) -> GatedEngram:
        """Build the engram of decoder layer ``layer_idx`` from a model config (see
        :meth:`GatedEngramConfig.from_model_config`)."""
        return cls(
            GatedEngramConfig.from_model_config(config, layer_idx),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    @property
    def reform_param(self) -> dict[str, dict[str, bool]]:
        """Checkpoint rules: ``q_weight`` / ``k_weight`` are plain ``[hc_mult, hidden]`` tensors, not linear kernels.

        The generic HF->EasyDeL converter transposes every 2-D tensor whose key contains ``"weight"``
        (``StateDictConverter.process_tensor``), which would flip these to ``[hidden, hc_mult]``. The released
        checkpoint stores them as ``[hc_mult, hidden]`` (reference ``model.py``), the same layout this module uses, so
        they are marked ``already_converted`` and loaded as-is.
        """
        return {"q_weight$": {"already_converted": True}, "k_weight$": {"already_converted": True}}

    def place_table_on_host(self) -> None:
        """Move the embedding table to pinned host memory and switch the lookup to the host gather path.

        Idempotent. The table becomes a non-trainable ``buffers`` leaf (same ``embed.weight`` path). Under abstract
        evaluation (a tracer value) only the mode is switched; place the loaded array by calling this method again
        after the weights are materialized.

        Raises:
            ValueError: If the backend exposes no ``pinned_host`` memory space.
        """
        if self.table_dtype == _FP8:
            self._place_fp8_table_on_host()
            return
        weight = self.embed.weight
        self.table_memory_kind = _PINNED_HOST
        weight.kind = "buffers"
        value = weight.value
        if isinstance(value, jax.core.Tracer) or not isinstance(value, jax.Array):
            return
        sharding = value.sharding
        if getattr(sharding, "memory_kind", None) == _PINNED_HOST:
            return
        device = next(iter(value.devices()))
        kinds = {memory.kind for memory in device.addressable_memories()}
        if _PINNED_HOST not in kinds:
            raise ValueError(
                f"backend {device.platform!r} has no 'pinned_host' memory space (available: {sorted(kinds)})."
            )
        weight.value = jax.device_put(value, sharding.with_memory_kind(_PINNED_HOST))

    def table_sharding(self, mesh: jax.sharding.Mesh | None = None, memory_kind: str | None = None):
        """Row sharding of the fp8 table: rows split jointly over every axis of ``mesh``.

        Args:
            mesh: Mesh to shard over. ``None`` builds a 1-D ``rows`` mesh over ``jax.devices()``; pass the model mesh
                if its device order differs (a jit mixing meshes with different device orders is rejected).
            memory_kind: Memory kind override; defaults to this module's ``table_memory_kind``.

        Returns:
            A :class:`jax.sharding.NamedSharding` with ``PartitionSpec(all_axes, None)``.
        """
        if mesh is None:
            mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("rows",))
        axes = tuple(mesh.axis_names)
        return jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(axes, None), memory_kind=memory_kind or self.table_memory_kind
        )

    def _place_fp8_table_on_host(self) -> None:
        """Switch an fp8 table to the host gather path; move a concrete full-size table into host RAM."""
        self.table_memory_kind = _PINNED_HOST
        if getattr(self, "_host_table_key", None) is None:
            self._host_table_key = f"gated-engram-{next(_HOST_TABLE_IDS)}"
        self.embed.weight.kind = "buffers"
        self.embed.scale.kind = "buffers"
        weight, scale = self.embed.weight.value, self.embed.scale.value
        if isinstance(weight, jax.core.Tracer) or not isinstance(weight, jax.Array) or weight.shape[0] < self.table_rows:
            return  # abstract or placeholder: fill later with set_table
        self._install_host_table(np.asarray(weight), np.asarray(scale))

    def _install_host_table(self, weight, scale) -> None:
        """Register NumPy ``weight`` (fp8 or uint8 bytes) / ``scale`` (uint8 e8m0) as this module's host table.

        The arrays are stored as given (no copy for contiguous NumPy inputs, so a read-only ``np.memmap`` over a
        local safetensors file works); the module keeps 1-row placeholder buffers.
        """
        features = self.embed.features
        w = np.asarray(weight)
        w = w if w.dtype == np.uint8 else w.view(np.uint8)
        s = np.asarray(scale)
        n_scale = features // FP8_SCALE_BLOCK
        if w.ndim != 2 or w.shape[0] < self.table_rows or w.shape[1] != features or s.shape != (w.shape[0], n_scale):
            raise ValueError(
                f"host table needs weight [>= {self.table_rows}, {features}] and matching scale [rows, {n_scale}]; "
                f"got {w.shape} and {s.shape}."
            )
        if s.dtype != np.uint8:
            raise ValueError(f"scale must be uint8 e8m0 bytes; got {s.dtype}.")
        if getattr(self, "_host_table_key", None) is None:
            self._host_table_key = f"gated-engram-{next(_HOST_TABLE_IDS)}"
        _HOST_TABLES[self._host_table_key] = (w, s)
        self.embed.weight.kind = "buffers"
        self.embed.scale.kind = "buffers"
        self.embed.weight.value = jnp.zeros((1, features), jnp.float8_e4m3fn)
        self.embed.scale.value = jnp.ones((1, n_scale), jnp.uint8)

    def release_host_table(self) -> None:
        """Drop this module's host table from the registry (modules rebound from its GraphDef share it)."""
        _HOST_TABLES.pop(getattr(self, "_host_table_key", None), None)

    def set_table(self, weight, scale, mesh: jax.sharding.Mesh | None = None) -> None:
        """Install the fp8 table: row-shard ``weight`` / ``scale`` into this module's memory kind.

        Args:
            weight: float8_e4m3fn ``[rows or padded_rows, head_dim]`` (NumPy or jax array).
            scale: uint8 e8m0 bytes ``[rows or padded_rows, head_dim // 32]``.
            mesh: Mesh to shard over (see :meth:`table_sharding`).

        Raises:
            ValueError: If the module is not in fp8 mode, or a dtype / shape disagrees with the config.
        """
        if self.table_dtype != _FP8:
            raise ValueError("set_table needs table_dtype='fp8'.")
        features = self.embed.features
        if jnp.dtype(weight.dtype) != jnp.dtype(jnp.float8_e4m3fn) or jnp.dtype(scale.dtype) != jnp.dtype(jnp.uint8):
            raise ValueError(f"expected float8_e4m3fn weight and uint8 scale; got {weight.dtype} and {scale.dtype}.")
        sharding = self.table_sharding(mesh)
        padded = engram_table_padded_rows(self.table_rows, len(sharding.mesh.devices.flat))
        for name, arr, cols in (
            ("weight", weight, features),
            ("scale", scale, features // FP8_SCALE_BLOCK),
        ):
            if tuple(arr.shape) not in ((self.table_rows, cols), (padded, cols)):
                raise ValueError(
                    f"{name} shape {tuple(arr.shape)} must be {(self.table_rows, cols)} or {(padded, cols)}."
                )
        if self.table_memory_kind == _PINNED_HOST:
            # Host RAM table, gathered by row id in NumPy: no shard padding needed.
            self._install_host_table(weight, scale)
            return
        if weight.shape[0] != padded:
            weight = np.concatenate([np.asarray(weight), np.zeros((padded - weight.shape[0], features), weight.dtype)])
            tail = np.ones((padded - scale.shape[0], scale.shape[1]), np.uint8)
            scale = np.concatenate([np.asarray(scale), tail])
        self.embed.weight.kind = "buffers"
        self.embed.scale.kind = "buffers"
        self.embed.weight.value = jax.device_put(weight, sharding)
        self.embed.scale.value = jax.device_put(scale, sharding)

    @staticmethod
    def _host_ids(table: Array, ids: Array) -> Array:
        """Place ``ids`` in host memory on the same devices as ``table``.

        A row-sharded pinned-host table spans every mesh device, so the ids must be replicated over that mesh in host
        memory; a single-device ``Space.Host`` placement would put them on one device only and ``compute_on`` rejects
        the mixed device sets. Unsharded or traced tables keep the plain ``Space.Host`` placement.
        """
        try:
            sharding = table.sharding
        except Exception:
            sharding = None
        if isinstance(sharding, jax.sharding.NamedSharding) and sharding.mesh.size > 1:
            target = jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec(), memory_kind=_PINNED_HOST)
            return jax.device_put(ids, target)
        return jax.device_put(ids, jax.memory.Space.Host)

    def _lookup_fp8(self, ids: Array) -> Array:
        """Dequantized rows ``[B, S, cols, head_dim]`` from the fp8 table (gather both arrays, then dequantize)."""
        weight, scale = self.embed.weight.value, self.embed.scale.value
        if self.table_memory_kind == _DEVICE:
            rows, scales = jnp.take(weight, ids, axis=0), jnp.take(scale, ids, axis=0)
        else:
            rows, scales = self._host_gather_rows(ids)
            rows = jax.lax.bitcast_convert_type(rows, jnp.float8_e4m3fn)
        return _dequantize_rows(rows, scales).astype(jnp.dtype(self.compute_dtype_name))

    def _host_gather_rows(self, ids: Array) -> tuple[Array, Array]:
        """Gathered uint8 ``(weight [..., head_dim], scale [..., head_dim // 32])`` rows from the host table.

        A ``jax.pure_callback`` NumPy gather: per-call cost follows the number of rows looked up, not the table size
        (measured on TPU: 8x512x8 rows in ~8 ms from a 25 GB table; the ``compute_on`` shard-local gather took 1.9 s on
        the 101 GB table). Only the gathered bytes are transferred to device. Works under jit with replicated or
        batch-sharded ids; the ids carry no tangent, so gradients never reach the callback.

        Raises:
            ValueError: If no host table is installed (call :meth:`set_table` or the checkpoint loader first).
        """
        key = getattr(self, "_host_table_key", None)
        if key not in _HOST_TABLES:
            raise ValueError("no host engram table installed; call set_table() or load_engram_tables() first.")
        weight, scale = _HOST_TABLES[key]
        weight_width, scale_width = int(weight.shape[1]), int(scale.shape[1])
        # The callback returns uint32-packed bytes, not uint8: on TPU, an 8-bit pure_callback result whose minor dim is
        # under 128 bytes comes back with scrambled rows in a multi-device program (measured: 1/4/32-byte rows wrong on
        # 4-device meshes with replicated or batch-sharded ids; 128-byte rows and uint32 words exact). The bytes are
        # identical; only the callback boundary dtype changes.
        shapes = (
            jax.ShapeDtypeStruct((*ids.shape, -(-weight_width // 4)), jnp.uint32),
            jax.ShapeDtypeStruct((*ids.shape, -(-scale_width // 4)), jnp.uint32),
        )
        weight_words, scale_words = jax.pure_callback(
            functools.partial(_gather_host_rows_packed, key), shapes, ids, vmap_method="sequential"
        )
        return _unpack_u8(weight_words, weight_width), _unpack_u8(scale_words, scale_width)

    def fp8_table_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """The installed fp8 table as NumPy, independent of its storage layout.

        Returns:
            ``(weight float8_e4m3fn [rows', head_dim], scale uint8 [rows', head_dim // 32])`` where ``rows'`` is the
            stored row count (padded to the shard count for device tables, unpadded for host tables).

        Raises:
            ValueError: If the module is not in fp8 mode.
        """
        if self.table_dtype != _FP8:
            raise ValueError("fp8_table_numpy needs table_dtype='fp8'.")
        if self.table_memory_kind == _PINNED_HOST:
            key = getattr(self, "_host_table_key", None)
            if key not in _HOST_TABLES:
                raise ValueError("no host engram table installed.")
            weight, scale = _HOST_TABLES[key]
        else:
            weight, scale = np.asarray(self.embed.weight.value), np.asarray(self.embed.scale.value)
        weight = np.ascontiguousarray(weight)
        weight = weight if weight.dtype != np.uint8 else weight.view(jnp.float8_e4m3fn)
        return weight, np.ascontiguousarray(scale)

    def _lookup(self, hash_ids: Int[Array, "batch seq cols"]) -> Array:
        """Table rows ``[B, S, cols, head_dim]`` for ``hash_ids``; host gather when the table is offloaded."""
        ids = hash_ids.astype(jnp.int32)
        if self.table_dtype == _FP8:
            return self._lookup_fp8(ids)
        if self.table_memory_kind == _DEVICE:
            return self.embed(ids)
        from jax.experimental.compute_on import compute_on

        gather = compute_on(_take_rows, compute_type="device_host", out_memory_spaces=jax.memory.Space.Host)
        # `gather` needs both operands in one memory space: move the (small) ids to the table's host memory, gather
        # there, and bring only the selected rows back to device memory.
        host_ids = self._host_ids(self.embed.weight.value, ids)
        rows = jax.device_put(gather(self.embed.weight.value, host_ids), jax.memory.Space.Device)
        return rows.astype(self.embed.dtype)

    @property
    def uses_host_table(self) -> bool:
        """Whether the fp8 table lives in host RAM (rows can be gathered on the host before the device step)."""
        return self.table_dtype == _FP8 and self.table_memory_kind == _PINNED_HOST

    def gather_host_rows(self, ids) -> tuple[np.ndarray, np.ndarray]:
        """Host-side NumPy gather of fp8 row bytes and e8m0 scales for ``ids`` (no JAX, no device work).

        Serving engines call this while preparing a step so the compiled program receives the rows as ordinary inputs
        (see :meth:`forward`'s ``gathered``) instead of stalling on a ``jax.pure_callback`` mid-program.

        Args:
            ids: Integer row ids of any shape.

        Returns:
            ``(weight uint8 [*ids.shape, head_dim], scale uint8 [*ids.shape, head_dim // 32])``.

        Raises:
            ValueError: If the table is not a host-resident fp8 table or none is installed.
        """
        if not self.uses_host_table:
            raise ValueError("gather_host_rows needs table_dtype='fp8' with table_memory_kind='pinned_host'.")
        key = getattr(self, "_host_table_key", None)
        if key not in _HOST_TABLES:
            raise ValueError("no host engram table installed; call set_table() or load_engram_tables() first.")
        return _gather_host_rows(key, ids)

    def forward(
        self,
        hidden_streams: Float[Array, "batch seq hc hidden"],
        hash_ids: Int[Array, "batch seq cols"] | None = None,
        token_mask: Bool[Array, "batch seq"] | None = None,
        gathered: tuple[Array, Array] | None = None,
    ) -> Array:
        """Add the gated engram value to every stream.

        Args:
            hidden_streams: Streams ``[B, S, hc_mult, hidden]``.
            hash_ids: Host-computed table row ids ``[B, S, num_hash_cols]`` (unused when ``gathered`` is given).
            token_mask: Optional ``[B, S]`` bool; tokens where it is ``False`` get a zero gate (unchanged streams).
            gathered: Optional pre-gathered fp8 rows ``(weight uint8 [B, S, cols, head_dim], scale uint8
                [B, S, cols, head_dim // 32])`` from :meth:`gather_host_rows`; only the dequantization runs on device.

        Returns:
            Updated streams, same shape and dtype as ``hidden_streams``.

        Raises:
            ValueError: If neither ``hash_ids`` nor ``gathered`` is given.
        """
        if gathered is not None:
            weight_bytes, scale_bytes = gathered
            batch, seq = weight_bytes.shape[:2]
            fp8 = jax.lax.bitcast_convert_type(weight_bytes.astype(jnp.uint8), jnp.float8_e4m3fn)
            rows = _dequantize_rows(fp8, scale_bytes).astype(jnp.dtype(self.compute_dtype_name)).reshape(batch, seq, -1)
        elif hash_ids is not None:
            batch, seq = hash_ids.shape[:2]
            rows = self._lookup(hash_ids).reshape(batch, seq, -1)
        else:
            raise ValueError("GatedEngram.forward needs hash_ids or gathered rows.")
        kv = self.kv_proj(rows)
        key = kv[..., : self.hc_mult * self.hidden_size].astype(jnp.float32)
        key = key.reshape(batch, seq, self.hc_mult, self.hidden_size)
        value = kv[..., self.hc_mult * self.hidden_size :].astype(jnp.float32)
        weight = self.q_weight.value.astype(jnp.float32) * self.k_weight.value.astype(jnp.float32)
        h = hidden_streams.astype(jnp.float32)
        rstd = jax.lax.rsqrt(jnp.mean(h * h, axis=-1) + self.eps) * jax.lax.rsqrt(
            jnp.mean(key * key, axis=-1) + self.eps
        )
        dot = jnp.sum(h * weight * key, axis=-1) * rstd * self.hidden_size**-0.5
        gate = jax.nn.sigmoid(jnp.copysign(jnp.sqrt(jnp.maximum(jnp.abs(dot), 1e-6)), dot))
        if token_mask is not None:
            gate = jnp.where(token_mask[..., None], gate, 0.0)
        return (h + gate[..., None] * value[..., None, :]).astype(hidden_streams.dtype)
