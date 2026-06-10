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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mesh-portable (canonical) on-disk layout for fused projections.

Fused projections (``qkv_proj``, ``gate_up_proj``, fused MoE experts) hold
their output channels **TP-interleaved** at runtime: for tensor-parallel size
``T`` and logical segments ``[A | B | C]``, the in-memory last axis is ordered
``[A_0 B_0 C_0 A_1 B_1 C_1 ...]`` so each TP rank owns a contiguous,
correctly-composed shard. That layout depends on the mesh — a checkpoint that
serializes it verbatim is only loadable at the exact tensor-parallel size it
was saved with. Loading it under any other ``tp`` silently mis-slices every
fused projection (names and shapes still match!), which scrambles Q/K/V and
gate/up and turns the model into a near-uniform sampler. This is exactly what
happened to the qwen3.6-27b checkpoints converted at ``tp=4`` and served at
``tp∈{1,2}``.

This module makes native checkpoints mesh-portable:

* :func:`canonicalize_fused_state` — de-interleave every fused parameter to
  the canonical contiguous ``[A | B | C]`` order before saving.
* :func:`runtimeize_fused_state` — re-interleave canonical parameters for the
  *current* runtime ``tp`` after loading.
* :func:`fused_layout_param_specs` — discover the fused parameters and their
  segment layouts by walking the module's linears.

Checkpoints carry a ``fused_param_layout.json`` marker; legacy checkpoints
without it are loaded verbatim (correct only when load-``tp`` equals
save-``tp``) and a warning is emitted when that cannot be verified.
"""

from __future__ import annotations

import json
import typing as tp

import jax
import numpy as np
from eformer.paths import ePath
from jax import numpy as jnp

from easydel.utils.helpers import get_logger
from easydel.utils.traversals import iter_module_search, tree_path_to_string

from ..linears._linear_moe import ParallelMoELinear
from ._runtime import tensor_parallel_size

if tp.TYPE_CHECKING:
    from easydel.infra.base_config import EasyDeLBaseConfig

logger = get_logger(__name__)

FUSED_LAYOUT_MARKER = "fused_param_layout.json"
CANONICAL = "canonical"


def _segment_sizes(layout: tp.Any) -> tuple[int, ...] | None:
    """Return the layout's logical segment sizes, or ``None`` when not fused."""
    sizes = getattr(layout, "segment_sizes", None)
    if not sizes:
        return None
    if not getattr(layout, "interleave_for_tp", True):
        return None
    return tuple(int(s) for s in sizes)


def fused_layout_param_specs(module: tp.Any) -> list[tuple[str, tuple[int, ...]]]:
    """Discover fused, TP-interleaved parameters on a module tree.

    Walks every submodule carrying a fused projection ``layout`` (dense
    ``FusedColumnLayout`` and fused MoE expert layouts alike) and returns the
    dotted module path together with the layout's logical segment sizes.

    Args:
        module: Root EasyDeL module (e.g. a ``ForCausalLM`` instance).

    Returns:
        List of ``(dotted_module_path, segment_sizes)`` tuples; the owning
        parameters are ``<path>.weight`` (and ``<path>.bias`` when present).
    """
    specs: list[tuple[str, tuple[int, ...]]] = []
    seen: set[int] = set()
    for path, sub in iter_module_search(module, object):
        if id(sub) in seen:
            continue
        seen.add(id(sub))
        dotted = ".".join(str(p) for p in path)

        # 1. dense fused projections: the linear carries a FusedColumnLayout
        layout = getattr(sub, "layout", None)
        if layout is not None:
            sizes = _segment_sizes(layout)
            if sizes is not None:
                specs.append((dotted, sizes))
                continue

        # 2. GDR packed projections. Only the MERGED-SPLIT variant interleaves
        #    in_proj_qkvz / in_proj_ba on the output axis (split via the
        #    TP-aware segment splitter); the default packed mode is per-key-head
        #    grouped — a pure reshape, mesh-independent — and must NOT be
        #    transformed.
        gdr = getattr(sub, "linear_attention_layout", None)
        if gdr is not None and getattr(sub, "uses_merged_split_proj", False):
            qkvz = getattr(gdr, "qkvz_segment_sizes", None)
            ba = getattr(gdr, "ba_segment_sizes", None)
            if qkvz and getattr(sub, "in_proj_qkvz", None) is not None:
                specs.append((f"{dotted}.in_proj_qkvz", tuple(int(x) for x in qkvz)))
            in_proj_ba = getattr(sub, "in_proj_ba", None)
            if in_proj_ba is not None:
                # packed [beta | alpha]: equal halves of 2 * num_v_heads
                if ba:
                    ba_sizes = tuple(int(x) for x in ba)
                else:
                    out = getattr(in_proj_ba, "out_features", None)
                    if out is None:
                        w = getattr(in_proj_ba, "weight", None)
                        shape = getattr(getattr(w, "value", w), "shape", None)
                        out = int(shape[-1]) if shape else 0
                    ba_sizes = (out // 2, out // 2) if out and out % 2 == 0 else None
                if ba_sizes:
                    specs.append((f"{dotted}.in_proj_ba", ba_sizes))

        # 3. fused MoE experts: ColumnParallelMoELinear named gate_up_proj holds
        #    [gate | up] (equal halves) TP-interleaved on the output axis
        if isinstance(sub, ParallelMoELinear) and dotted.endswith("gate_up_proj"):
            out = getattr(sub, "out_features", None)
            if out is None:
                w = getattr(sub, "weight", None)
                shape = getattr(getattr(w, "value", w), "shape", None)
                out = int(shape[-1]) if shape else None
            if out and out % 2 == 0:
                specs.append((dotted, (out // 2, out // 2)))
    return specs


def _transform_last_axis(x: tp.Any, segment_sizes: tuple[int, ...], tp_size: int, *, to_canonical: bool) -> tp.Any:
    """De-/re-interleave the last axis of a fused parameter.

    Args:
        x: Parameter value ``[..., sum(segment_sizes)]`` (weight or bias).
        segment_sizes: Logical segment widths.
        tp_size: Tensor-parallel size of the interleave.
        to_canonical: ``True`` = interleaved -> contiguous;
            ``False`` = contiguous -> interleaved.

    Returns:
        The transformed array (same type family as the input), or ``x``
        unchanged when the transform does not apply (tp<=1, shape mismatch,
        or indivisible segments).
    """
    total = sum(segment_sizes)
    if tp_size <= 1 or not hasattr(x, "shape") or x.ndim < 1 or int(x.shape[-1]) != total:
        return x
    if any(s % tp_size for s in segment_sizes):
        return x

    xp = np if isinstance(x, np.ndarray) else jnp
    locals_ = [s // tp_size for s in segment_sizes]
    if to_canonical:
        # [..., T, sum(local)] rank-major -> logical segments
        ranked = xp.reshape(x, (*x.shape[:-1], tp_size, total // tp_size))
        offsets = np.cumsum(locals_)[:-1].tolist()
        parts = xp.split(ranked, offsets, axis=-1)
        return xp.concatenate(
            [xp.reshape(p, (*x.shape[:-1], s)) for p, s in zip(parts, segment_sizes, strict=True)], axis=-1
        )
    # canonical -> rank-major interleave
    offsets = np.cumsum(segment_sizes)[:-1].tolist()
    segs = xp.split(x, offsets, axis=-1)
    per_rank = []
    for r in range(tp_size):
        for seg, loc in zip(segs, locals_, strict=True):
            per_rank.append(seg[..., r * loc : (r + 1) * loc])
    return xp.concatenate(per_rank, axis=-1)


def _key_to_string(key: tp.Any) -> str:
    return ".".join(str(part) for part in key) if isinstance(key, tuple) else str(key)


def _fused_leaf_markers(module_path: str) -> tuple[str, ...]:
    markers = []
    for suffix in (".weight", ".bias"):
        for prefix in (module_path, f"parameters.{module_path}"):
            marker = f"{prefix}{suffix}"
            markers.extend((marker, f"{marker}.value"))
    return tuple(markers)


def _matches_fused_leaf(key_str: str, module_path: str) -> bool:
    for marker in _fused_leaf_markers(module_path):
        if key_str.endswith(marker):
            return True
    return False


def _matches_fused_optimizer_leaf(key_str: str, module_path: str) -> bool:
    """Return whether an optimizer leaf path belongs to a fused parameter.

    Optimizer state is not standardized: Optax Adam stores slots as paths like
    ``...mu.parameters.<param-path>.weight``, while custom optimizers may store
    slots below the parameter path (``...<param-path>.weight.ema``) or inside a
    larger state namespace. Match the parameter path at path-component
    boundaries and let ``_transform_last_axis`` reject non-array or
    shape-incompatible leaves.
    """
    for marker in _fused_leaf_markers(module_path):
        if key_str == marker:
            return True
        if key_str.endswith(f".{marker}") or key_str.startswith(f"{marker}.") or f".{marker}." in key_str:
            return True
    return False


def _match_keys(flat_state: dict, module_path: str) -> list[tp.Any]:
    """Find state-dict keys belonging to a fused module's weight/bias."""
    hits = []
    for key in flat_state:
        if _matches_fused_leaf(_key_to_string(key), module_path):
            hits.append(key)
    return hits


def _apply(module, flat_state: dict, tp_size: int, *, to_canonical: bool, log_label: str) -> dict:
    if tp_size <= 1:
        return flat_state
    specs = fused_layout_param_specs(module)
    if not specs:
        return flat_state
    touched = 0
    for module_path, sizes in specs:
        for key in _match_keys(flat_state, module_path):
            value = flat_state[key]
            inner = value
            is_param = hasattr(value, "value") and not hasattr(value, "shape")
            if is_param:
                inner = value.value
            new_inner = _transform_last_axis(inner, sizes, tp_size, to_canonical=to_canonical)
            if new_inner is not inner:
                touched += 1
                if is_param:
                    value.value = new_inner
                else:
                    flat_state[key] = new_inner
    if touched:
        logger.info(f"{log_label} {touched} fused parameter leaf/leaves for tp={tp_size}.")
    return flat_state


def canonicalize_fused_state(module, flat_state: dict, config: EasyDeLBaseConfig | None = None) -> dict:
    """Convert fused params from runtime (TP-interleaved) to canonical order.

    Call before serializing a native checkpoint so the on-disk layout is
    independent of the save-time mesh.
    """
    tp_size = tensor_parallel_size(config if config is not None else module.config)
    return _apply(module, flat_state, tp_size, to_canonical=True, log_label="Canonicalized")


def runtimeize_fused_state(module, flat_state: dict, config: EasyDeLBaseConfig | None = None) -> dict:
    """Convert fused params from canonical order to the runtime TP interleave.

    Call after loading a canonical native checkpoint, before binding the
    state to the module.
    """
    tp_size = tensor_parallel_size(config if config is not None else getattr(module, "config", None))
    return _apply(module, flat_state, tp_size, to_canonical=False, log_label="Re-interleaved")


def _apply_pytree(module, tree: tp.Any, tp_size: int, *, to_canonical: bool, log_label: str) -> tp.Any:
    if tree is None or tp_size <= 1:
        return tree
    specs = fused_layout_param_specs(module)
    if not specs:
        return tree

    touched = 0

    def _maybe_transform(path, value):
        nonlocal touched
        key_str = tree_path_to_string(path, sep=".")
        for module_path, sizes in specs:
            if _matches_fused_optimizer_leaf(key_str, module_path):
                new_value = _transform_last_axis(value, sizes, tp_size, to_canonical=to_canonical)
                if new_value is not value:
                    touched += 1
                return new_value
        return value

    out = jax.tree_util.tree_map_with_path(_maybe_transform, tree)
    if touched:
        logger.info(f"{log_label} {touched} fused optimizer-state leaf/leaves for tp={tp_size}.")
    return out


def canonicalize_fused_optimizer_state(module, opt_state: tp.Any, config: EasyDeLBaseConfig | None = None) -> tp.Any:
    """Convert optimizer leaves for fused params to canonical order.

    Optax moment trees are usually keyed as ``...mu.parameters.<param-path>``
    and ``...nu.parameters.<param-path>``. Matching by parameter-path suffix
    keeps this independent of the optimizer chain prefix while reusing the
    exact same de-interleave transform used for parameters.
    """
    tp_size = tensor_parallel_size(config if config is not None else module.config)
    return _apply_pytree(module, opt_state, tp_size, to_canonical=True, log_label="Canonicalized")


def runtimeize_fused_optimizer_state(module, opt_state: tp.Any, config: EasyDeLBaseConfig | None = None) -> tp.Any:
    """Convert canonical optimizer leaves for fused params to runtime order."""
    tp_size = tensor_parallel_size(config if config is not None else getattr(module, "config", None))
    return _apply_pytree(module, opt_state, tp_size, to_canonical=False, log_label="Re-interleaved")


def write_fused_layout_marker(save_directory) -> None:
    """Record that this checkpoint stores fused params in canonical order."""
    try:
        (ePath(str(save_directory)) / FUSED_LAYOUT_MARKER).write_text(json.dumps({"fused_param_layout": CANONICAL}))
    except Exception as exc:  # pragma: no cover - marker is best-effort
        logger.warning(f"Could not write {FUSED_LAYOUT_MARKER}: {exc}")


def read_fused_layout_marker(load_directory) -> str | None:
    """Return the checkpoint's fused-layout tag (``None`` for legacy)."""
    try:
        path = ePath(str(load_directory)) / FUSED_LAYOUT_MARKER
        if path.exists():
            return json.loads(path.read_text()).get("fused_param_layout")
    except Exception:  # pragma: no cover
        return None
    return None
