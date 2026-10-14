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

"""Model-level engram plumbing: :class:`EngramModelMixin`.

A model whose layers carry :class:`~easydel.layers.engram.GatedEngram` modules mixes this in and gets, as ordinary
methods:

* table management: :meth:`~EngramModelMixin.engram_modules`, :meth:`~EngramModelMixin.place_engram_tables_on_host`,
  :meth:`~EngramModelMixin.load_engram_tables`, :meth:`~EngramModelMixin.engram_hasher`;
* the eSurge side-input hooks (contract in :mod:`easydel.inference.esurge.multimodal.side_inputs`):
  :meth:`~EngramModelMixin.esurge_prepare_tokenizer`, :meth:`~EngramModelMixin.esurge_side_input_specs`,
  :meth:`~EngramModelMixin.esurge_side_input_provider`;
* the trainer hook :meth:`~EngramModelMixin.host_batch_transform`.

The model's forward takes one of:

* ``engram_hash_ids int32 [B, S, n_engram_layers, cols]`` (device tables, looked up in-program), or
* ``engram_rows uint8 [B, S, n_engram_layers, cols, head_dim]`` with ``engram_scales uint8 [..., head_dim // 32]``,
  gathered on the host from host-resident fp8 tables so the compiled program does no table lookup.

Axis ``n_engram_layers`` is indexed by each module's ``layer_hash_index``. The model config carries the ``engram_*``
fields described in :mod:`easydel.layers.engram._hashing`.
"""

from __future__ import annotations

import os
import typing as tp
import weakref
from collections.abc import Iterable

import numpy as np
import spectrax as spx

from . import _hashing
from ._hashing import EngramHasher, EngramWindowHasher
from ._layer import FP8_SCALE_BLOCK, GatedEngram

if tp.TYPE_CHECKING:
    from easydel.inference.esurge.multimodal.side_inputs import SideInputSpec, SideInputWindow

__all__ = ["EngramModelMixin", "EngramSideInputProvider", "find_engram_modules"]

# id(model) -> its engram modules, so nothing is stored on the module graph and the per-step eSurge spec check costs a
# dict lookup, not a tree walk. Keyed by id, never by the model: ``EasyDeLBaseModule.__hash__`` is a structural hash
# over array leaves (and fails on abstract ones). The entry is dropped by a finalizer when the model is collected, so
# a reused id never sees a stale entry; a rebound model (``spx.bind``) is a new object and gets a fresh walk.
_ENGRAM_MODULES: dict[int, tuple[GatedEngram, ...]] = {}


def find_engram_modules(model: tp.Any) -> tuple[GatedEngram, ...]:
    """Every :class:`GatedEngram` inside ``model``, ordered by ``layer_hash_index``.

    Args:
        model: A spectrax module tree.

    Returns:
        The engram modules (empty when the model has none).

    Raises:
        ValueError: If two modules share a ``layer_hash_index``.
    """
    key = id(model)
    cached = _ENGRAM_MODULES.get(key)
    if cached is not None:
        return cached
    found = tuple(
        sorted(spx.iter_modules(model, with_path=False, select=GatedEngram), key=lambda m: int(m.layer_hash_index))
    )
    indices = [int(m.layer_hash_index) for m in found]
    if len(set(indices)) != len(indices):
        raise ValueError(f"engram modules share a layer_hash_index: {indices}.")
    try:
        weakref.finalize(model, _ENGRAM_MODULES.pop, key, None)
    except TypeError:  # not weak-referenceable: no safe eviction, so do not cache
        return found
    _ENGRAM_MODULES[key] = found
    return found


def _engram_layer_count(config: tp.Any) -> int:
    return len(getattr(config, "engram_layer_ids", None) or ())


class EngramSideInputProvider:
    """Builds the engram side inputs of one eSurge step from each request's token history.

    Hash ids are rebuilt per window by :class:`~easydel.layers.engram.EngramWindowHasher`, so chunked prefill, decode,
    preemption and slot reuse need no per-request provider state. With host-resident tables the rows are gathered here
    and the device program receives ``engram_rows`` / ``engram_scales`` instead of ids.
    """

    def __init__(self, model: EngramModelMixin, token_map: np.ndarray | None) -> None:
        """Bind the provider to a model.

        Args:
            model: The served model.
            token_map: Compressed token map from :meth:`EngramModelMixin.esurge_prepare_tokenizer` (``None`` uses the
                identity map, which the config must allow).
        """
        self.specs = model.esurge_side_input_specs()
        self.hasher = EngramWindowHasher(model.engram_hasher(token_map))
        self.modules = model.engram_modules() if "engram_rows" in self.specs else ()

    def build(self, windows: tp.Sequence[SideInputWindow], num_tokens: int) -> dict[str, np.ndarray]:
        """Arrays ``[num_tokens, ...]`` per declared spec; window rows packed back to back, zero padding after."""
        hash_ids = np.zeros((int(num_tokens), self.hasher.n_layers, self.hasher.n_columns), dtype=np.int32)
        hashed = self.hasher.hash_windows([(w.token_ids, w.token_types, w.start, w.end) for w in windows])
        off = 0
        for rows in hashed:
            hash_ids[off : off + rows.shape[0]] = rows
            off += rows.shape[0]
        if not self.modules:
            return {"engram_hash_ids": hash_ids}
        out = {
            name: np.zeros((int(num_tokens), *self.specs[name].per_token_shape), dtype=self.specs[name].dtype)
            for name in ("engram_rows", "engram_scales")
        }
        # Gather only the live rows; padded rows keep zero bytes (never read downstream).
        for module in self.modules:
            idx = int(module.layer_hash_index)
            rows, scales = module.gather_host_rows(hash_ids[:off, idx])
            out["engram_rows"][:off, idx] = rows
            out["engram_scales"][:off, idx] = scales
        return out


class EngramModelMixin:
    """Engram table management plus the eSurge and trainer host-side hooks (see the module docstring).

    Stateless: everything is derived from the module tree and ``self.config``. Models without engram layers
    (``config.engram_layer_ids`` empty) declare no side inputs and leave batches untouched. The host class provides
    ``config``.
    """

    def engram_modules(self) -> tuple[GatedEngram, ...]:
        """This model's :class:`GatedEngram` modules, ordered by ``layer_hash_index``."""
        return find_engram_modules(self)

    def engram_uses_host_tables(self) -> bool:
        """Whether every engram layer keeps an fp8 table in host RAM (rows are then gathered on the host)."""
        modules = self.engram_modules()
        return bool(modules) and all(module.uses_host_table for module in modules)

    def place_engram_tables_on_host(self) -> int:
        """Move every engram table to host memory (call after loading real weights); idempotent.

        The tables become non-trainable ``buffers`` leaves and the layers switch to the host-gather lookup (see
        :meth:`GatedEngram.place_table_on_host`).

        Returns:
            Number of engram layers processed.
        """
        modules = self.engram_modules()
        for module in modules:
            module.place_table_on_host()
        return len(modules)

    def load_engram_tables(
        self, paths: Iterable[str | os.PathLike], mesh: tp.Any = None, **kwargs: tp.Any
    ) -> dict[int, int]:
        """Fill the fp8 engram tables from safetensors files (see :func:`easydel.layers.engram.load_engram_tables`).

        Args:
            paths: Safetensors files holding the ``layers.<id>.engram.embed.{weight,scale}`` tensors.
            mesh: Mesh to row-shard device tables over; defaults to ``self.config.mesh`` when available.
            **kwargs: ``block_rows`` / ``mmap``.

        Returns:
            ``{layer_id: rows}`` of the loaded layers.
        """
        from ._loader import load_engram_tables

        if mesh is None:
            mesh = getattr(getattr(self, "config", None), "mesh", None)
        return load_engram_tables(self, paths, mesh, **kwargs)

    def engram_hasher(self, token_map: np.ndarray | None = None) -> EngramHasher:
        """The n-gram hasher of this model's config (``token_map=None``: identity map, if the config allows it)."""
        return EngramHasher.from_config(self.config, token_map)

    def esurge_prepare_tokenizer(self, tokenizer: tp.Any) -> np.ndarray | None:
        """eSurge hook: build and validate the compressed token map the engram hasher needs.

        Args:
            tokenizer: Fast tokenizer, or ``None``.

        Returns:
            The compressed token map, or ``None`` when the identity map is valid (or the model has no engram).

        Raises:
            ValueError: If the config needs a compressed map but no fast tokenizer is available, or the tokenizer's
                compressed vocab disagrees with the config.
        """
        config = self.config
        if not _engram_layer_count(config):
            return None
        expected = int(config.engram_compressed_vocab_size)
        if tokenizer is None or getattr(tokenizer, "backend_tokenizer", None) is None:
            if expected != int(config.vocab_size):
                raise ValueError(
                    "This model hashes engram n-grams over the tokenizer's compressed vocabulary "
                    f"(engram_compressed_vocab_size={expected}), but no fast tokenizer is available to build it."
                )
            return None
        token_map = _hashing.build_engram_token_map(tokenizer)
        got = int(np.max(token_map)) + 1 if token_map.size else 0
        if got != expected:
            raise ValueError(
                f"tokenizer compressed vocab size {got} != config.engram_compressed_vocab_size {expected}; "
                "the tokenizer does not match this checkpoint."
            )
        return token_map

    def esurge_side_input_specs(self) -> dict[str, SideInputSpec]:
        """eSurge hook: ``engram_rows`` / ``engram_scales`` with host tables, ``engram_hash_ids`` otherwise."""
        from easydel.inference.esurge.multimodal.side_inputs import SideInputSpec

        config = self.config
        n_layers = _engram_layer_count(config)
        if not n_layers:
            return {}
        n_cols = (int(config.engram_max_ngram_size) - 1) * int(config.engram_n_heads)
        if self.engram_uses_host_tables():
            head_dim = int(config.engram_head_dim)
            return {
                "engram_rows": SideInputSpec((n_layers, n_cols, head_dim), np.uint8),
                "engram_scales": SideInputSpec((n_layers, n_cols, head_dim // FP8_SCALE_BLOCK), np.uint8),
            }
        return {"engram_hash_ids": SideInputSpec((n_layers, n_cols), np.int32)}

    def esurge_side_input_provider(self, tokenizer_state: np.ndarray | None) -> EngramSideInputProvider:
        """eSurge hook: the per-step engram side-input builder."""
        return EngramSideInputProvider(self, tokenizer_state)

    def host_batch_transform(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Trainer hook: swap ``engram_hash_ids [B, S, L, cols]`` for host-gathered rows when tables live on host.

        Args:
            batch: Purified host batch.

        Returns:
            The batch with ``engram_rows`` (uint8 ``[B, S, L, cols, head_dim]``) and ``engram_scales`` (uint8
            ``[B, S, L, cols, head_dim // 32]``) in place of ``engram_hash_ids`` when every engram table is
            host-resident; otherwise the batch unchanged.

        Raises:
            ValueError: If ``engram_hash_ids`` is not ``[B, S, L, cols]`` with room for every layer's index.
        """
        hash_ids = batch.get("engram_hash_ids")
        if hash_ids is None or "engram_rows" in batch or not self.engram_uses_host_tables():
            return batch
        ids = np.asarray(hash_ids)
        modules = self.engram_modules()
        need = 1 + max(int(module.layer_hash_index) for module in modules)
        if ids.ndim != 4 or ids.shape[2] < need:
            raise ValueError(f"engram_hash_ids must be [batch, seq, >= {need}, cols]; got shape {tuple(ids.shape)}.")
        rows = scales = None
        for module in modules:
            idx = int(module.layer_hash_index)
            r, s = module.gather_host_rows(ids[:, :, idx])
            if rows is None:
                rows = np.zeros((*ids.shape, r.shape[-1]), dtype=np.uint8)
                scales = np.zeros((*ids.shape, s.shape[-1]), dtype=np.uint8)
            rows[:, :, idx] = r
            scales[:, :, idx] = s
        out = {key: value for key, value in batch.items() if key != "engram_hash_ids"}
        out["engram_rows"] = rows
        out["engram_scales"] = scales
        return out
