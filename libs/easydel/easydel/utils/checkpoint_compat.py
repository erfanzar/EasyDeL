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


def legacy_checkpoint_key_aliases(key: str) -> tuple[str, ...]:
    """Return old EasyDeL checkpoint key aliases for a current dotted key.

    This is used when restoring pre-SpectraX optimizer TensorStore checkpoints
    into a current optimizer template. Those checkpoints may differ from current
    keys in three ways:

    1. The trainable collection may be ``params`` or omitted instead of
       ``parameters``.
    2. Parameter leaves may end in ``kernel``, ``embedding`` or ``scale``
       instead of the current unified ``weight`` suffix.
    3. Optimizer wrapper leaves were saved with a trailing ``.value`` suffix.

    Args:
        key: Current dotted template key, for example
            ``"tx.1.mu.parameters.lm_head.weight"`` or
            ``"1.mu.parameters.lm_head.weight"``.

    Returns:
        Ordered alternate checkpoint keys to try. The input key itself is not
        included.
    """
    parts = tuple(part for part in key.split(".") if part)
    if not parts:
        return ()

    collection_variants: set[tuple[str, ...]] = {parts}
    for idx, part in enumerate(parts):
        if part == "parameters":
            collection_variants.add(parts[:idx] + parts[idx + 1 :])
            collection_variants.add((*parts[:idx], "params", *parts[idx + 1 :]))
        elif part == "params":
            collection_variants.add((*parts[:idx], "parameters", *parts[idx + 1 :]))

    def with_value_suffix(candidate: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        """Return ``candidate`` and an optional ``+ ('value',)`` variant.

        Args:
            candidate: A candidate flat-path tuple.

        Returns:
            A one-tuple containing the candidate when it already ends in
            ``"value"``; otherwise a pair with the candidate followed by a
            copy that has ``"value"`` appended.
        """
        if candidate and candidate[-1] == "value":
            return (candidate,)
        return (candidate, (*candidate, "value"))

    aliases: list[tuple[str, ...]] = []
    for variant in collection_variants:
        aliases.extend(with_value_suffix(variant))
        for idx, part in enumerate(variant):
            if part == "weight":
                for legacy_leaf in LEGACY_LEAF_RENAMES:
                    aliases.extend(with_value_suffix((*variant[:idx], legacy_leaf, *variant[idx + 1 :])))

    ordered: dict[str, None] = {}
    original = ".".join(parts)
    for alias_parts in aliases:
        alias = ".".join(alias_parts)
        if alias and alias != original:
            ordered.setdefault(alias, None)
    return tuple(ordered)


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


def _leaf_shape(leaf: tp.Any) -> tuple[int, ...] | None:
    """Return a tuple shape for array-like leaves and state wrappers.

    Handles both raw arrays (with a ``.shape`` attribute) and Spectrax
    ``Parameter`` style wrappers that expose ``.value.shape``.

    Args:
        leaf: Array-like value or wrapper to inspect.

    Returns:
        A tuple of integer dimensions, or ``None`` when no shape information
        can be extracted from ``leaf``.
    """
    shape = getattr(leaf, "shape", None)
    if shape is None and hasattr(leaf, "value"):
        shape = getattr(leaf.value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _is_lm_head_weight_key(key: tuple[tp.Any, ...], lm_head_names: tuple[str, ...]) -> bool:
    """Check whether a flat state key names an lm-head weight leaf.

    Args:
        key: Flat path tuple ``(collection, ...module_path, leaf_name)``.
        lm_head_names: Module names that should be treated as lm-head modules.

    Returns:
        ``True`` when ``key`` is of the form
        ``("parameters", ..., <lm_head_name>, "weight")``.
    """
    return len(key) >= 3 and str(key[0]) == "parameters" and str(key[-1]) == "weight" and str(key[-2]) in lm_head_names


def _embedding_candidate_score(key: tuple[tp.Any, ...]) -> int:
    """Rank likely token embedding leaves over unrelated vision/projector embeddings.

    Awards positive points for module names typically used for the text token
    embedding table (``embed_tokens``, ``wte``, etc.) and deducts points for
    vision/patch embedding paths so that tied lm-head materialization picks
    the right source tensor.

    Args:
        key: Flat path tuple of the candidate parameter leaf.

    Returns:
        Heuristic score; higher means more likely to be a token embedding.
        Non-positive values are filtered out by the caller.
    """
    parts = tuple(str(part).lower() for part in key)
    joined = "/".join(parts)
    score = 0
    preferred_terms = (
        "embed_tokens",
        "token_embedding",
        "word_embeddings",
        "wte",
        "language_model/embed",
        "text_model/embed",
    )
    for term in preferred_terms:
        if term in joined:
            score += 8
    if "embed" in joined or "embedding" in joined:
        score += 2
    if "vision" in joined or "image" in joined or "patch" in joined:
        score -= 6
    return score


def materialize_tied_lm_head_from_embeddings(
    flat_state: dict[tuple[tp.Any, ...], tp.Any],
    required_leaves: dict[tuple[tp.Any, ...], tp.Any],
    *,
    tie_word_embeddings: bool,
    lm_head_names: tuple[str, ...] = ("lm_head",),
) -> dict[tuple[tp.Any, ...], tp.Any]:
    """Fill missing tied lm-head weights from the saved token embedding table.

    Hugging Face-style tied checkpoints commonly store only the input token
    embedding table when ``tie_word_embeddings=True``. EasyDeL still
    materializes an ``lm_head.weight`` parameter leaf so the runtime can use
    one model structure for tied and untied modules. For index-only legacy
    tensorstore checkpoints this means the live model expects
    ``parameters/.../lm_head/weight`` while the checkpoint only contains an
    embedding leaf such as ``parameters/model/language_model/embed_tokens``.

    This adapter keeps the load strict: it only creates missing lm-head leaves
    when the config explicitly says embeddings are tied and when a checkpoint
    embedding has exactly the required shape (or the required transposed
    shape). All other missing leaves are left alone for the materialization
    assertion to catch.

    Args:
        flat_state: Flat checkpoint state after legacy collection and leaf-name
            normalization.
        required_leaves: Flat live model leaves keyed by ``(collection, *path)``.
            The leaf shape is used to validate a tied-source candidate.
        tie_word_embeddings: Whether the model config declares tied input and
            output embeddings.
        lm_head_names: Module names that should be treated as lm-head modules.

    Returns:
        ``flat_state`` when no tied leaf is missing, otherwise a shallow copy
        with the missing lm-head weight leaf/leaves materialized.
    """
    if not tie_word_embeddings or not flat_state or not required_leaves:
        return flat_state

    missing_targets: list[tuple[tp.Any, ...]] = [
        key for key in required_leaves if _is_lm_head_weight_key(key, lm_head_names) and key not in flat_state
    ]
    if not missing_targets:
        return flat_state

    candidates: list[tuple[int, tuple[tp.Any, ...], tp.Any, tuple[int, ...]]] = []
    for key, value in flat_state.items():
        if not isinstance(key, tuple) or len(key) < 3 or str(key[0]) != "parameters" or str(key[-1]) != "weight":
            continue
        score = _embedding_candidate_score(key)
        if score <= 0:
            continue
        shape = _leaf_shape(value)
        if shape is not None:
            candidates.append((score, key, value, shape))

    if not candidates:
        return flat_state

    materialized = dict(flat_state)
    filled: list[tuple[tuple[tp.Any, ...], tuple[tp.Any, ...], bool]] = []
    for target_key in missing_targets:
        target_shape = _leaf_shape(required_leaves[target_key])
        if target_shape is None or len(target_shape) != 2:
            continue

        ranked_matches: list[tuple[int, tuple[tp.Any, ...], tp.Any, bool]] = []
        for score, source_key, source_value, source_shape in candidates:
            if source_shape == target_shape:
                ranked_matches.append((score, source_key, source_value, False))
            elif source_shape == tuple(reversed(target_shape)):
                ranked_matches.append((score, source_key, source_value, True))
        if not ranked_matches:
            continue

        _score, source_key, source_value, needs_transpose = max(ranked_matches, key=lambda item: item[0])
        materialized[target_key] = source_value.T if needs_transpose else source_value
        filled.append((target_key, source_key, needs_transpose))

    if filled:
        preview = ", ".join(
            f"{'/'.join(str(p) for p in target)} <- {'/'.join(str(p) for p in source)}{'.T' if transposed else ''}"
            for target, source, transposed in filled[:3]
        )
        suffix = f" (+{len(filled) - 3} more)" if len(filled) > 3 else ""
        logger.info(f"Materialized {len(filled)} tied lm_head weight leaf/leaves from embeddings: {preview}{suffix}.")
        return materialized

    return flat_state


__all__ = [
    "LEGACY_COLLECTION_RENAMES",
    "LEGACY_LEAF_RENAMES",
    "adapt_legacy_checkpoint_collections",
    "legacy_checkpoint_key_aliases",
    "materialize_tied_lm_head_from_embeddings",
    "rename_legacy_checkpoint_leaves",
]
