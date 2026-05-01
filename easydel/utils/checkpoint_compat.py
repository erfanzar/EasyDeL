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

"""Backwards-compatibility adapters for pre-spectrax EasyDeL checkpoints.

The pre-spectrax converter wrote linear weights as ``...kernel``, embeddings
as ``...embedding`` and norms as ``...scale``, all wrapped under a ``params``
collection (or with no collection prefix at all when the bare module dict was
serialized).  Spectrax-era modules unify those leaves under ``...weight``
inside a ``parameters`` collection.  These adapters rewrite an older
flattened state in-place so it can be loaded into a current model without
re-saving the checkpoint.

Each helper is a pure function over the flat ``{path_tuple: leaf}`` state
dict; they make no JAX calls and have no model dependency.  Keeping them in
their own module makes adding a future format bump (v3, v4, ...) a matter
of registering another adapter rather than editing the bridge mixin.
"""

from __future__ import annotations

import typing as tp

from eformer.loggings import get_logger

logger = get_logger(__name__)

# Legacy parameter leaf names produced by the pre-spectrax converter.
# These all map back to the unified `.weight` name used by current modules;
# JAX-side shapes are identical between the two formats.
LEGACY_LEAF_RENAMES: dict[str, str] = {"kernel": "weight", "embedding": "weight", "scale": "weight"}

# Pre-spectrax checkpoints used `params` as the trainable collection;
# spectrax uses `parameters`. Older saves may also omit the collection
# wrapper entirely (the inner module dict was written directly).
LEGACY_COLLECTION_RENAMES: dict[str, str] = {"params": "parameters"}


def adapt_legacy_checkpoint_collections(
    flat_state: dict[tuple[tp.Any, ...], tp.Any], required_keys: set[tuple[tp.Any, ...]]
) -> dict[tuple[tp.Any, ...], tp.Any]:
    """Reconcile pre-spectrax collection prefixes with the current model layout.

    Old EasyDeL checkpoints either saved the trainable tree under ``params`` or
    skipped the collection wrapper entirely (the bare module dict was written
    out, e.g. ``model/...`` instead of ``parameters/model/...``). The current
    loader builds ``required_keys`` as ``(collection, *path)`` tuples, so any
    state key that does not start with a known collection name is dropped by
    the unexpected-keys filter. This helper rewrites such keys to align with
    the live model's collections.

    The decision is made per-key, not globally: a checkpoint may legitimately
    contain keys for several collections (e.g. ``rng`` plus the bare module
    tree), so we only rewrite the keys that don't already match.

    Args:
        flat_state: Flat dict mapping path tuples to leaf values from a
            legacy checkpoint.
        required_keys: Set of path tuples expected by the current model.
            The first element of each tuple is the collection name.

    Returns:
        A new dict with collection-prefixed keys aligned with ``required_keys``.
        Keys that already use a known collection are passed through unchanged.
    """
    if not flat_state or not required_keys:
        return flat_state

    known_collections = {k[0] for k in required_keys if isinstance(k, tuple) and k}
    if not known_collections:
        return flat_state

    rename_map = {old: new for old, new in LEGACY_COLLECTION_RENAMES.items() if new in known_collections}
    target_collection = "parameters" if "parameters" in known_collections else next(iter(known_collections))

    adapted: dict[tuple[tp.Any, ...], tp.Any] = {}
    rename_count = 0
    wrap_count = 0
    for key, value in flat_state.items():
        if not isinstance(key, tuple) or not key:
            adapted[key] = value
            continue
        first = key[0]
        if first in rename_map:
            adapted[(rename_map[first], *key[1:])] = value
            rename_count += 1
        elif first in known_collections:
            adapted[key] = value
        else:
            adapted[(target_collection, *key)] = value
            wrap_count += 1

    if wrap_count:
        logger.info(
            f"Legacy checkpoint missing collection prefix on {wrap_count} key(s); "
            f"wrapping under {target_collection!r} for backward compatibility."
        )
    if rename_count:
        renames_preview = ", ".join(f"{a!r}->{b!r}" for a, b in sorted(rename_map.items()))
        logger.info(f"Legacy checkpoint collection rename: {renames_preview} ({rename_count} key(s)).")
    return adapted


def rename_legacy_checkpoint_leaves(flat_state: dict[tuple[tp.Any, ...], tp.Any]) -> dict[tuple[tp.Any, ...], tp.Any]:
    """Rename pre-spectrax leaf suffixes (``.kernel``/``.embedding``/``.scale``) to ``.weight``.

    Old EasyDeL checkpoints stored linear weights as ``...kernel``, embeddings as
    ``...embedding`` and norms as ``...scale``. Current modules unify all of these
    under ``...weight`` with identical JAX-side shapes, so loading such a
    checkpoint into a freshly built model only requires a leaf-name rewrite.
    Quantized leaves (``quant_kernel``/``quant_scales``/``quant_biases``) are
    left untouched.

    Args:
        flat_state: Flat dict mapping path tuples to leaf values from a
            legacy checkpoint.

    Returns:
        A new dict with the legacy suffixes rewritten to ``"weight"``. When
        both new- and old-style leaves are present for the same path the
        old-style value wins and a warning is logged.
    """
    renamed: dict[tuple[tp.Any, ...], tp.Any] = {}
    legacy_count = 0
    collisions: list[tuple[tp.Any, ...]] = []
    for key, value in flat_state.items():
        if isinstance(key, tuple) and key:
            last = str(key[-1])
            new_last = LEGACY_LEAF_RENAMES.get(last)
            if new_last is not None:
                new_key = (*key[:-1], new_last)
                if new_key != key and (new_key in renamed or new_key in flat_state):
                    collisions.append(new_key)
                renamed[new_key] = value
                if new_key != key:
                    legacy_count += 1
                continue
        renamed[key] = value
    if legacy_count:
        logger.info(
            f"Renamed {legacy_count} legacy .kernel/.embedding/.scale checkpoint "
            "leaves to .weight for backward compatibility."
        )
    if collisions:
        preview = ", ".join("/".join(str(p) for p in k) for k in collisions[:5])
        suffix = f" (+{len(collisions) - 5} more)" if len(collisions) > 5 else ""
        logger.warning(
            f"Legacy checkpoint contained both new- and old-style leaves for: {preview}{suffix}. Old-style values won."
        )
    return renamed


__all__ = [
    "LEGACY_COLLECTION_RENAMES",
    "LEGACY_LEAF_RENAMES",
    "adapt_legacy_checkpoint_collections",
    "rename_legacy_checkpoint_leaves",
]
