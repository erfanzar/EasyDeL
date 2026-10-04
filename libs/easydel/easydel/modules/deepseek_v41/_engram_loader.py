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

"""Row-chunked loader of the DeepSeek-V4.1 engram tables (fp8 ``embed.weight`` + e8m0 ``embed.scale``).

The two tables of the released checkpoint are ~100 GB each (shards 47 / 48) and must never pass through the fp32
quant decoder or a bf16 copy. This module reads them straight from the safetensors files. The header is parsed here
(``safe_open`` cannot hand back the one-byte ``F8_*`` dtypes through numpy); each device shard's row range is read in
bounded blocks into one uint8 buffer, placed into that device's memory (pinned host when the module offloads), and
the buffer is released before the next shard is read. Peak host RAM is the already-placed shards plus one shard buffer.

Row padding follows the reference ``convert.py``: tables are padded up to a multiple of the shard count with zero
weight bytes and scale byte ``1``.
"""

from __future__ import annotations

import json
import os
import struct
import typing as tp
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from easydel.layers.embeddings._engram import FP8_SCALE_BLOCK, engram_table_padded_rows

from ._native_checkpoint import engram_table_checkpoint_keys

__all__ = ["EngramTensorRef", "load_engram_tables", "read_engram_rows", "scan_engram_tensors"]

_ALLOWED_DTYPES: dict[str, tuple[str, ...]] = {"weight": ("F8_E4M3", "U8"), "scale": ("F8_E8M0", "U8")}
_PAD_VALUE = {"weight": 0, "scale": 1}
_DEFAULT_BLOCK_ROWS = 1 << 20


@dataclass(frozen=True)
class EngramTensorRef:
    """Location of one 2-D one-byte-per-element tensor inside a safetensors file.

    Attributes:
        path: File holding the tensor.
        name: Tensor key in that file.
        dtype: Safetensors dtype string (``F8_E4M3``, ``F8_E8M0`` or ``U8``).
        shape: ``(rows, cols)``.
        data_start: Absolute byte offset of the first element.
    """

    path: str
    name: str
    dtype: str
    shape: tuple[int, int]
    data_start: int

    @property
    def rows(self) -> int:
        """Number of rows."""
        return self.shape[0]

    @property
    def cols(self) -> int:
        """Number of columns (bytes per row)."""
        return self.shape[1]


def _read_header(path: str) -> tuple[dict[str, tp.Any], int]:
    """Parse a safetensors header; returns ``(header, absolute offset of the data region)``."""
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(size))
    return header, 8 + size


def scan_engram_tensors(paths: Iterable[str | os.PathLike]) -> dict[int, dict[str, EngramTensorRef]]:
    """Locate the engram table tensors across safetensors files without reading any tensor data.

    Args:
        paths: Safetensors files (e.g. the two last shards of the release, or all of them).

    Returns:
        ``{layer_id: {"weight": ref, "scale": ref}}``.

    Raises:
        ValueError: If a table tensor is not a 2-D one-byte tensor of the expected dtype, its byte range disagrees with
            its shape, or the same tensor appears in two files.
    """
    found: dict[int, dict[str, EngramTensorRef]] = {}
    for raw_path in paths:
        path = os.fspath(raw_path)
        header, base = _read_header(path)
        tables = engram_table_checkpoint_keys(k for k in header if k != "__metadata__")
        for layer, leaves in tables.items():
            for leaf, key in leaves.items():
                entry = header[key]
                dtype, shape = entry["dtype"], tuple(int(d) for d in entry["shape"])
                if dtype not in _ALLOWED_DTYPES[leaf] or len(shape) != 2:
                    raise ValueError(
                        f"{key} in {path}: expected 2-D {leaf} tensor of dtype {_ALLOWED_DTYPES[leaf]}; "
                        f"got {dtype} {shape}."
                    )
                begin, end = entry["data_offsets"]
                if end - begin != shape[0] * shape[1]:
                    raise ValueError(f"{key} in {path}: byte range {end - begin} disagrees with shape {shape}.")
                if leaf in found.get(layer, {}):
                    raise ValueError(f"{key} appears in more than one file ({found[layer][leaf].path} and {path}).")
                found.setdefault(layer, {})[leaf] = EngramTensorRef(path, key, dtype, shape, base + begin)  # type: ignore[arg-type]
    return found


def read_engram_rows(
    ref: EngramTensorRef,
    start: int,
    stop: int,
    *,
    padded_rows: int | None = None,
    pad_value: int = 0,
) -> np.ndarray:
    """Read rows ``[start, stop)`` of a table as raw uint8 bytes, padding rows past the end of the tensor.

    Args:
        ref: Tensor location from :func:`scan_engram_tensors`.
        start: First row.
        stop: One past the last row; may exceed ``ref.rows`` only up to ``padded_rows``.
        padded_rows: Padded row count of the sharded table (``None``: no padding allowed).
        pad_value: Byte used for rows ``>= ref.rows`` (0 for weights, 1 for e8m0 scales per ``convert.py``).

    Returns:
        uint8 array ``[stop - start, ref.cols]``.

    Raises:
        ValueError: On an invalid range or a short read.
    """
    limit = ref.rows if padded_rows is None else int(padded_rows)
    if not 0 <= start <= stop <= limit:
        raise ValueError(f"row range [{start}, {stop}) is outside [0, {limit}).")
    out = np.full((stop - start, ref.cols), pad_value, dtype=np.uint8)
    real_stop = min(stop, ref.rows)
    if real_stop > start:
        count = (real_stop - start) * ref.cols
        data = np.fromfile(ref.path, dtype=np.uint8, count=count, offset=ref.data_start + start * ref.cols)
        if data.size != count:
            raise ValueError(f"short read from {ref.path}: wanted {count} bytes, got {data.size}.")
        out[: real_stop - start] = data.reshape(real_stop - start, ref.cols)
    return out


def load_engram_tables(
    model: tp.Any,
    paths: Iterable[str | os.PathLike],
    mesh: tp.Any = None,
    *,
    block_rows: int = _DEFAULT_BLOCK_ROWS,
    mmap: bool = False,
) -> dict[int, int]:
    """Fill every fp8 engram table of ``model`` from safetensors files, shard by shard, into its memory kind.

    Each addressable device shard of ``GatedEngram.table_sharding(mesh)`` is assembled from bounded row blocks and
    placed with its own device_put (``pinned_host`` when the module offloads), so no full-table copy ever exists.

    Args:
        model: A model exposing ``layers`` (or ``model.layers``) whose entries have an ``engram`` (or ``None``).
            Engram layers absent from the model (truncated runs) are skipped.
        paths: Safetensors files that together hold the table tensors.
        mesh: Mesh passed to :meth:`GatedEngram.table_sharding`.
        block_rows: Rows read per block.
        mmap: Host tables only (``table_memory_kind="pinned_host"``). Map each table read-only straight from its
            safetensors file instead of copying it into process memory. With the shards staged on tmpfs
            (``/dev/shm``) each table exists once in RAM and only the pages a lookup touches are faulted in; the
            files must stay in place for the module's lifetime.

    Returns:
        ``{layer_id: unpadded_rows}`` of the loaded layers.

    Raises:
        ValueError: If a table is incomplete or its shape disagrees with the module.
    """
    import jax
    import ml_dtypes

    refs = scan_engram_tensors(paths)
    layers = getattr(model, "model", model).layers
    loaded: dict[int, int] = {}
    for layer_id in sorted(refs):
        if layer_id >= len(layers) or getattr(layers[layer_id], "engram", None) is None:
            continue
        tables = refs[layer_id]
        if set(tables) != {"weight", "scale"}:
            raise ValueError(f"layer {layer_id}: need both embed.weight and embed.scale; found {sorted(tables)}.")
        engram = layers[layer_id].engram
        rows, head_dim = engram.table_rows, engram.embed.features
        sharding = engram.table_sharding(mesh)
        padded = engram_table_padded_rows(rows, len(sharding.mesh.devices.flat))
        if sharding.memory_kind == "pinned_host":
            # Host tables live in ordinary host RAM as one unpadded uint8 array per leaf and are gathered by row id in
            # NumPy (see `GatedEngram._host_gather_rows`); read each in bounded blocks, then install via set_table.
            host: dict[str, np.ndarray] = {}
            for leaf, cols in (("weight", head_dim), ("scale", head_dim // FP8_SCALE_BLOCK)):
                ref = tables[leaf]
                if ref.shape != (rows, cols):
                    raise ValueError(f"layer {layer_id} {leaf}: file shape {ref.shape} != module shape {(rows, cols)}.")
                if mmap:
                    arr = np.memmap(ref.path, dtype=np.uint8, mode="r", offset=ref.data_start, shape=(rows, cols))
                else:
                    arr = np.empty((rows, cols), dtype=np.uint8)
                    for lo in range(0, rows, block_rows):
                        hi = min(lo + block_rows, rows)
                        arr[lo:hi] = read_engram_rows(ref, lo, hi, padded_rows=rows, pad_value=_PAD_VALUE[leaf])
                host[leaf] = arr
            engram.set_table(host["weight"].view(ml_dtypes.float8_e4m3fn), host["scale"], mesh)
            loaded[layer_id] = rows
            continue
        for leaf, cols in (("weight", head_dim), ("scale", head_dim // FP8_SCALE_BLOCK)):
            ref = tables[leaf]
            if ref.shape != (rows, cols):
                raise ValueError(f"layer {layer_id} {leaf}: file shape {ref.shape} != module shape {(rows, cols)}.")
            shape = (padded, cols)
            pieces = []
            for device, index in sharding.addressable_devices_indices_map(shape).items():
                row_slice = index[0]
                start = 0 if row_slice.start is None else row_slice.start
                stop = padded if row_slice.stop is None else row_slice.stop
                buf = np.empty((stop - start, cols), dtype=np.uint8)
                for lo in range(start, stop, block_rows):
                    hi = min(lo + block_rows, stop)
                    buf[lo - start : hi - start] = read_engram_rows(
                        ref, lo, hi, padded_rows=padded, pad_value=_PAD_VALUE[leaf]
                    )
                piece = buf.view(ml_dtypes.float8_e4m3fn) if leaf == "weight" else buf
                target = jax.sharding.SingleDeviceSharding(device, memory_kind=sharding.memory_kind)
                placed = jax.device_put(piece, target)
                jax.block_until_ready(placed)
                pieces.append(placed)
                del buf, piece
            variable = engram.embed.weight if leaf == "weight" else engram.embed.scale
            variable.kind = "buffers"
            variable.value = jax.make_array_from_single_device_arrays(shape, sharding, pieces)
        loaded[layer_id] = rows
    return loaded
