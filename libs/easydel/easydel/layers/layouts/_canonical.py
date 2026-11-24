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

This module makes native checkpoints mesh-portable by *recording* the
interleave instead of normalizing it: weights are saved verbatim (the runtime
layout of the save mesh) and the save-time tp is stamped into ``config.json``
as ``fused_param_tp`` (recursively, sub-configs included) plus a sidecar
marker. Loading under a different mesh re-interleaves from the recorded tp to
the live tp and overwrites the field — the config always describes the layout
the weights actually carry. Canonical contiguous order is simply ``tp=1``.

* :func:`retp_fused_state` — re-interleave fused params saved at one tp for
  another (de-interleave from ``saved_tp``, interleave for ``target_tp``).
* :func:`canonicalize_fused_state` / :func:`runtimeize_fused_state` — the
  two legs of that transform against the live mesh.
* :func:`fused_layout_param_specs` — discover the fused parameters and their
  segment layouts by walking the module's linears (models expose this as the
  dynamic ``fused_params`` registry).

A practical consequence: a legacy verbatim checkpoint whose save-tp is known
can be made portable by stamping ``fused_param_tp`` into its ``config.json``
— no weight rewrite needed. Checkpoints with neither field nor marker are
assumed canonical (``tp=1``) and re-interleaved for the live mesh with a
warning; stamp the real save-tp when that assumption is wrong.
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

logger = get_logger("easydel-layout-canonical")

FUSED_LAYOUT_MARKER = "fused_param_layout.json"
FUSED_TP_FIELD = "fused_param_tp"
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
            if getattr(sub, "conv1d", None) is not None:
                specs.append((f"{dotted}.conv1d", (int(gdr.key_dim), int(gdr.key_dim), int(gdr.value_dim))))
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
        if (
            isinstance(sub, ParallelMoELinear)
            and dotted.endswith("gate_up_proj")
            and not getattr(sub, "use_expert_tensor_mode", False)
        ):
            # ExpertTensorMode shards experts over physical TP, leaving each
            # expert's gate/up channels contiguous at every physical TP size.
            out = getattr(sub, "out_features", None)
            if out is None:
                w = getattr(sub, "weight", None)
                shape = getattr(getattr(w, "value", w), "shape", None)
                out = int(shape[-1]) if shape else None
            if out and out % 2 == 0:
                specs.append((dotted, (out // 2, out // 2)))
    return specs


def _transform_last_axis(
    x: tp.Any,
    segment_sizes: tuple[int, ...],
    tp_size: int,
    *,
    to_canonical: bool,
    allow_packed_stride: bool = False,
    leaf_name: str | None = None,
) -> tp.Any:
    """De-/re-interleave the last axis of a fused parameter.

    Args:
        x: Parameter value ``[..., sum(segment_sizes)]`` (weight or bias), or a
            word-/group-packed quantized leaf ``[..., sum(segment_sizes) // stride]``
            when ``allow_packed_stride=True``.
        segment_sizes: Logical segment widths.
        tp_size: Tensor-parallel size of the interleave.
        to_canonical: ``True`` = interleaved -> contiguous;
            ``False`` = contiguous -> interleaved.
        allow_packed_stride: When ``True`` (quantized ``quant_*`` leaves), scale
            ``segment_sizes`` by ``stride = sum(segment_sizes) // x.shape[-1]``
            when every ``s // tp_size`` divides cleanly by ``stride``, and raise
            ``ValueError`` instead of silently no-oping when indivisible.
        leaf_name: Optional leaf path used in error messages.

    Returns:
        The transformed array (same type family as the input), or ``x``
        unchanged when the transform does not apply (tp<=1, non-array, or
        non-quantized shape mismatch / indivisible segments).
    """
    total = sum(segment_sizes)
    if tp_size <= 1 or x is None or not hasattr(x, "shape") or x.ndim < 1:
        return x
    last_dim = int(x.shape[-1])
    effective_segments = segment_sizes
    if last_dim != total:
        if not allow_packed_stride:
            return x
        if last_dim <= 0 or total % last_dim != 0:
            raise ValueError(
                f"Cannot re-interleave packed quantized fused leaf {leaf_name or '<unknown>'} "
                f"(shape={tuple(x.shape)}) for tp={tp_size}: last dim {last_dim} does not evenly "
                f"divide logical fused width {total} (segments={segment_sizes})."
            )
        stride = total // last_dim
        if any(s % (tp_size * stride) != 0 for s in segment_sizes):
            raise ValueError(
                f"Cannot re-interleave packed quantized fused leaf {leaf_name or '<unknown>'} "
                f"(shape={tuple(x.shape)}, stride={stride}) for tp={tp_size}: segment sizes "
                f"{segment_sizes} do not divide cleanly by tp*stride={tp_size * stride} "
                "(quantization words/groups straddle segment or TP boundaries; load from a dense "
                "checkpoint with apply_quantization=True or load at the saved TP size)."
            )
        effective_segments = tuple(s // stride for s in segment_sizes)
    elif any(s % tp_size for s in segment_sizes):
        if allow_packed_stride:
            raise ValueError(
                f"Cannot re-interleave quantized fused leaf {leaf_name or '<unknown>'} "
                f"(shape={tuple(x.shape)}) for tp={tp_size}: segment sizes {segment_sizes} "
                f"are not divisible by tp={tp_size}."
            )
        return x

    eff_total = sum(effective_segments)
    xp = np if isinstance(x, np.ndarray) else jnp
    locals_ = [s // tp_size for s in effective_segments]
    if to_canonical:
        # [..., T, sum(local)] rank-major -> logical segments
        ranked = xp.reshape(x, (*x.shape[:-1], tp_size, eff_total // tp_size))
        offsets = np.cumsum(locals_)[:-1].tolist()
        parts = xp.split(ranked, offsets, axis=-1)
        return xp.concatenate(
            [xp.reshape(p, (*x.shape[:-1], s)) for p, s in zip(parts, effective_segments, strict=True)], axis=-1
        )
    # canonical -> rank-major interleave
    offsets = np.cumsum(effective_segments)[:-1].tolist()
    segs = xp.split(x, offsets, axis=-1)
    per_rank = []
    for r in range(tp_size):
        for seg, loc in zip(segs, locals_, strict=True):
            per_rank.append(seg[..., r * loc : (r + 1) * loc])
    return xp.concatenate(per_rank, axis=-1)


def _reshard_like_source(new_value: tp.Any, source: tp.Any) -> tp.Any:
    """Re-apply the source leaf's sharding after a shape-preserving layout transform.

    ``_transform_last_axis`` rebuilds the fused last axis with eager ``jnp``
    ops on arrays that were already placed on their target ``NamedSharding``;
    GSPMD's inferred output sharding does not preserve the tensor-parallel
    axis of the fused output dim (the interleave slices cross shard
    boundaries), silently downgrading e.g. ``P(('fsdp','sp'), 'tp')`` to
    ``P(('fsdp','sp'), None)``. The transform preserves shape, so the source
    array's sharding is the correct placement for the result by construction.
    Non-JAX values (numpy leaves) and sharding-less sources pass through
    unchanged.
    """
    if not isinstance(new_value, jax.Array):
        return new_value
    src_sharding = getattr(source, "sharding", None)
    if src_sharding is None or getattr(new_value, "sharding", None) == src_sharding:
        return new_value
    return jax.device_put(new_value, src_sharding)


def _key_to_string(key: tp.Any) -> str:
    return ".".join(str(part) for part in key) if isinstance(key, tuple) else str(key)


#: Leaf suffixes that can hold a fused projection's data.
#:
#: ``.weight``/``.bias`` cover a dense linear. A *quantized* linear stores the
#: same fused axis in ``quant_*`` leaves instead and has no ``.weight`` at all,
#: so listing only the dense pair silently skipped every quantized fused
#: projection: ``fused_param_tp`` was still stamped to the live tp, telling the
#: runtime splitter the weight was interleaved while the stored codes were left
#: contiguous. That is invisible at tp=1 (the interleave is the identity) and
#: scrambles gate/up at tp>1 -- observed on DeepSeek-V4-Flash A16W4, where every
#: shared expert diverged by rel 1.3 at tp=2 while attention (no fused
#: projections) stayed clean.
#:
#: All of these carry the fused axis LAST, including ``quant_kernel_packed``:
#: ``pack_int4_adjacent`` packs adjacent *rows* (the contraction dim), leaving
#: the output axis last. ``_transform_last_axis`` re-checks
#: ``shape[-1] == sum(segment_sizes)`` and no-ops otherwise, so a leaf whose
#: last axis is not the fused one is skipped rather than corrupted.
_QUANTIZED_LEAF_SUFFIXES: tp.Final = (
    ".quant_kernel",
    ".quant_scales",
    ".quant_biases",
    ".quant_kernel_packed",
)

_FUSED_LEAF_SUFFIXES: tp.Final = (
    ".weight",
    ".bias",
    *_QUANTIZED_LEAF_SUFFIXES,
)


def _is_quantized_leaf(key_str: str) -> bool:
    for suffix in _QUANTIZED_LEAF_SUFFIXES:
        if key_str.endswith(suffix) or key_str.endswith(f"{suffix}.value"):
            return True
    return False


def _fused_leaf_markers(module_path: str) -> tuple[str, ...]:
    markers = []
    for suffix in _FUSED_LEAF_SUFFIXES:
        for prefix in (module_path, f"parameters.{module_path}"):
            marker = f"{prefix}{suffix}"
            markers.extend((marker, f"{marker}.value"))
    return tuple(markers)


def _matches_fused_leaf(key_str: str, module_path: str) -> bool:
    for marker in _fused_leaf_markers(module_path):
        if key_str == marker or key_str.endswith(f".{marker}"):
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


def _module_fused_specs(module) -> list[tuple[str, tuple[int, ...]]]:
    """Return the module's fused-param registry as ``(path, sizes)`` pairs.

    Prefers the model-level ``fused_params`` dict (the dynamic registry,
    overridable per model — the fused analogue of ``_get_reform_param``);
    falls back to walking the module tree directly.
    """
    registry = getattr(module, "fused_params", None)
    if registry:
        return [(path, tuple(int(s) for s in sizes)) for path, sizes in registry.items()]
    return fused_layout_param_specs(module)


def _split_origin_conv1d_paths(
    specs: list[tuple[str, tuple[int, ...]]],
    flat_state: dict,
) -> set[str]:
    """Return ``*.conv1d`` spec paths whose checkpoint state came from split projections.

    A native checkpoint saved with ``linear_attention_separate_proj=True`` stores
    ``in_proj_qkv`` (not ``in_proj_qkvz``) and stores ``conv1d.weight`` in
    contiguous ``[Q | K | V]`` order even when ``fused_param_tp > 1`` (stamped for
    full-attention ``qkv_proj`` / MLP ``gate_up_proj``). When such a checkpoint is
    loaded into a live ``uses_merged_split_proj=True`` model, those ``conv1d``
    leaves are already canonical (``saved_tp=1``) and must skip de-interleaving
    while still being interleaved to ``target_tp``.
    """
    split_paths: set[str] = set()
    for module_path, _ in specs:
        if not module_path.endswith(".conv1d"):
            continue
        parent = module_path[: -len(".conv1d")]
        if _match_keys(flat_state, f"{parent}.in_proj_qkv") and not _match_keys(flat_state, f"{parent}.in_proj_qkvz"):
            split_paths.add(module_path)
    return split_paths


def _split_origin_conv1d_paths_in_tree(
    specs: list[tuple[str, tuple[int, ...]]],
    tree: tp.Any,
) -> set[str]:
    """Optimizer-pytree counterpart of :func:`_split_origin_conv1d_paths`."""
    conv_parents = {
        module_path[: -len(".conv1d")]: module_path for module_path, _ in specs if module_path.endswith(".conv1d")
    }
    if not conv_parents:
        return set()
    seen_qkv: set[str] = set()
    seen_qkvz: set[str] = set()
    for path, _ in jax.tree_util.tree_leaves_with_path(tree):
        key_str = tree_path_to_string(path, sep=".")
        for parent in conv_parents:
            if _matches_fused_optimizer_leaf(key_str, f"{parent}.in_proj_qkv"):
                seen_qkv.add(parent)
            if _matches_fused_optimizer_leaf(key_str, f"{parent}.in_proj_qkvz"):
                seen_qkvz.add(parent)
    return {conv_parents[parent] for parent in (seen_qkv - seen_qkvz)}


def _apply(
    module,
    flat_state: dict,
    tp_size: int,
    *,
    to_canonical: bool,
    log_label: str,
    specs_override: list[tuple[str, tuple[int, ...]]] | None = None,
) -> dict:
    if tp_size <= 1:
        return flat_state
    specs = specs_override if specs_override is not None else _module_fused_specs(module)
    if not specs:
        return flat_state
    touched = 0
    for module_path, sizes in specs:
        for key in _match_keys(flat_state, module_path):
            key_str = _key_to_string(key)
            value = flat_state[key]
            inner = value
            is_param = hasattr(value, "value") and not hasattr(value, "shape")
            if is_param:
                inner = value.value
            new_inner = _transform_last_axis(
                inner,
                sizes,
                tp_size,
                to_canonical=to_canonical,
                allow_packed_stride=_is_quantized_leaf(key_str),
                leaf_name=key_str,
            )
            if new_inner is not inner:
                new_inner = _reshard_like_source(new_inner, inner)
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


def _apply_pytree(
    module,
    tree: tp.Any,
    tp_size: int,
    *,
    to_canonical: bool,
    log_label: str,
    specs_override: list[tuple[str, tuple[int, ...]]] | None = None,
) -> tp.Any:
    if tree is None or tp_size <= 1:
        return tree
    specs = specs_override if specs_override is not None else _module_fused_specs(module)
    if not specs:
        return tree

    touched = 0

    def _maybe_transform(path, value):
        nonlocal touched
        key_str = tree_path_to_string(path, sep=".")
        for module_path, sizes in specs:
            if _matches_fused_optimizer_leaf(key_str, module_path):
                new_value = _transform_last_axis(
                    value,
                    sizes,
                    tp_size,
                    to_canonical=to_canonical,
                    allow_packed_stride=_is_quantized_leaf(key_str),
                    leaf_name=key_str,
                )
                if new_value is not value:
                    new_value = _reshard_like_source(new_value, value)
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


def retp_fused_state(module, flat_state: dict, *, saved_tp: int, target_tp: int) -> dict:
    """Re-interleave fused params from one tensor-parallel size to another.

    The on-disk layout is the runtime interleave of the save-time mesh
    (``saved_tp``, recorded in the checkpoint's config). De-interleave from
    ``saved_tp`` to the canonical contiguous order, then interleave for the
    live mesh's ``target_tp``. ``canonical`` is simply ``tp=1`` — both legs
    no-op at 1.

    Split-origin linear-attention ``conv1d`` leaves (detected by the presence of
    ``in_proj_qkv`` without ``in_proj_qkvz``) are always contiguous on disk
    (``saved_tp=1``), so they skip de-interleaving and are still interleaved to
    ``target_tp`` even when ``saved_tp == target_tp > 1``.
    """
    saved_tp, target_tp = int(saved_tp), int(target_tp)
    specs = _module_fused_specs(module)
    if not specs:
        return flat_state
    split_conv_paths = _split_origin_conv1d_paths(specs, flat_state)
    if saved_tp == target_tp:
        if split_conv_paths and target_tp > 1:
            conv_specs = [spec for spec in specs if spec[0] in split_conv_paths]
            return _apply(
                module,
                flat_state,
                target_tp,
                to_canonical=False,
                log_label="Re-interleaved (split-origin conv1d)",
                specs_override=conv_specs,
            )
        return flat_state
    deinterleave_specs = [spec for spec in specs if spec[0] not in split_conv_paths]
    flat_state = _apply(
        module,
        flat_state,
        saved_tp,
        to_canonical=True,
        log_label=f"De-interleaved (tp={saved_tp})",
        specs_override=deinterleave_specs,
    )
    return _apply(
        module,
        flat_state,
        target_tp,
        to_canonical=False,
        log_label="Re-interleaved",
        specs_override=specs,
    )


def retp_fused_optimizer_state(module, opt_state: tp.Any, *, saved_tp: int, target_tp: int) -> tp.Any:
    """Re-interleave fused optimizer moments from ``saved_tp`` to ``target_tp``."""
    saved_tp, target_tp = int(saved_tp), int(target_tp)
    if opt_state is None:
        return opt_state
    specs = _module_fused_specs(module)
    if not specs:
        return opt_state
    split_conv_paths = _split_origin_conv1d_paths_in_tree(specs, opt_state)
    if saved_tp == target_tp:
        if split_conv_paths and target_tp > 1:
            conv_specs = [spec for spec in specs if spec[0] in split_conv_paths]
            return _apply_pytree(
                module,
                opt_state,
                target_tp,
                to_canonical=False,
                log_label="Re-interleaved (split-origin conv1d)",
                specs_override=conv_specs,
            )
        return opt_state
    deinterleave_specs = [spec for spec in specs if spec[0] not in split_conv_paths]
    opt_state = _apply_pytree(
        module,
        opt_state,
        saved_tp,
        to_canonical=True,
        log_label=f"De-interleaved (tp={saved_tp})",
        specs_override=deinterleave_specs,
    )
    return _apply_pytree(
        module,
        opt_state,
        target_tp,
        to_canonical=False,
        log_label="Re-interleaved",
        specs_override=specs,
    )


def _read_saved_linear_attention_flags(load_directory) -> tuple[bool | None, bool | None, bool]:
    """Read ``(linear_attention_merged_split_proj, linear_attention_separate_proj, found_config)`` from config.json."""
    if not load_directory:
        return None, None, False
    try:
        base = ePath(str(load_directory))
        config_path = base / "config.json"
        if not config_path.exists():
            parent_config = base.parent / "config.json"
            if parent_config.exists():
                config_path = parent_config
            else:
                return None, None, False
        payload = json.loads(config_path.read_text())
    except Exception:  # pragma: no cover
        return None, None, False
    if not isinstance(payload, dict):
        return None, None, False
    candidates = []
    text_cfg = payload.get("text_config")
    if isinstance(text_cfg, dict):
        candidates.append(text_cfg)
    candidates.append(payload)
    saved_merged: bool | None = None
    saved_separate: bool | None = None
    for cfg in candidates:
        if saved_merged is None and "linear_attention_merged_split_proj" in cfg:
            saved_merged = bool(cfg["linear_attention_merged_split_proj"])
        if saved_separate is None and "linear_attention_separate_proj" in cfg:
            saved_separate = bool(cfg["linear_attention_separate_proj"])
    return saved_merged, saved_separate, True


def _replace_leaf_parent(key: tp.Any, old_child: str, new_child: str) -> tp.Any:
    if isinstance(key, tuple):
        if len(key) >= 2 and str(key[-2]) == old_child:
            return (*key[:-2], new_child, key[-1])
        if len(key) >= 3 and str(key[-3]) == old_child and str(key[-1]) == "value":
            return (*key[:-3], new_child, key[-2], key[-1])
        return key
    key_str = str(key)
    if key_str.startswith(f"{old_child}."):
        return f"{new_child}.{key_str[len(old_child) + 1 :]}"
    return key_str.replace(f".{old_child}.", f".{new_child}.")


def reconcile_linear_attention_native_layout(
    load_directory,
    module,
    flat_state: dict,
    *,
    saved_tp: int | None,
) -> int:
    """Validate and reconcile cross-mode Qwen3-Next/3.5 linear-attention native checkpoints.

    Guards against silent weight corruption when ``linear_attention_merged_split_proj``
    differs between the saved checkpoint and the live model:

    - **merged_split_proj ↔ default packed mode** (identical leaf names and shapes
      ``in_proj_qkvz`` / ``in_proj_ba`` / ``conv1d``, incompatible channel layouts):
      always raises ``ValueError`` rather than silently scrambling projections.
    - **saved ``merged_split_proj=True`` → live ``uses_split_proj=True``** (explicit
      ``linear_attention_separate_proj=True`` override OR automatic ``can_merge_tp``
      fallback at higher TP): de-interleaves ``in_proj_qkvz``, ``in_proj_ba``, and
      ``conv1d`` from ``saved_tp`` and unfuses dense ``in_proj_qkvz``/``in_proj_ba``
      into ``in_proj_qkv``/``in_proj_z``/``in_proj_b``/``in_proj_a``.

    Returns:
        Number of linear-attention layers unfused from merged-split to split layout.
    """
    if not flat_state:
        return 0
    saved_merged_flag, _, found_config = _read_saved_linear_attention_flags(load_directory)
    if not found_config:
        return 0
    saved_merged = bool(saved_merged_flag)
    eff_saved_tp = 1 if saved_tp is None else int(saved_tp)

    unfused_layers = 0
    seen: set[int] = set()
    for path, sub in iter_module_search(module, object):
        if id(sub) in seen:
            continue
        seen.add(id(sub))
        gdr = getattr(sub, "linear_attention_layout", None)
        if gdr is None:
            continue
        dotted = ".".join(str(p) for p in path)
        live_merged = bool(getattr(sub, "uses_merged_split_proj", False))
        live_split = bool(getattr(sub, "uses_split_proj", False))

        qkvz_keys = _match_keys(flat_state, f"{dotted}.in_proj_qkvz")
        qkv_keys = _match_keys(flat_state, f"{dotted}.in_proj_qkv")
        if not qkvz_keys or qkv_keys:
            continue

        # Checkpoint carries in_proj_qkvz (not split in_proj_qkv).
        if saved_merged and not live_merged and not live_split:
            raise ValueError(
                f"Checkpoint {load_directory} was saved with linear_attention_merged_split_proj=True "
                f"([Q|K|V|Z] layout) and cannot be loaded into default per-key-head packed mode "
                f"(layer {dotted}). Pass linear_attention_merged_split_proj=True (or "
                f"linear_attention_separate_proj=True to unfuse into split projections)."
            )
        if not saved_merged and live_merged:
            raise ValueError(
                f"Checkpoint {load_directory} stores default per-key-head packed linear-attention weights "
                f"(linear_attention_merged_split_proj=False) and cannot be loaded with "
                f"linear_attention_merged_split_proj=True (layer {dotted})."
            )
        if not saved_merged and live_split:
            raise ValueError(
                f"Checkpoint {load_directory} stores default per-key-head packed linear-attention weights "
                f"and cannot be unfused into linear_attention_separate_proj=True (layer {dotted})."
            )
        if not (saved_merged and live_split):
            continue

        # Saved merged_split_proj=True -> live uses_split_proj=True: unfuse cleanly.
        ba_keys = _match_keys(flat_state, f"{dotted}.in_proj_ba")
        conv_keys = _match_keys(flat_state, f"{dotted}.conv1d")
        if any(_is_quantized_leaf(_key_to_string(k)) for k in (*qkvz_keys, *ba_keys)):
            raise ValueError(
                f"Cannot unfuse quantized linear_attention_merged_split_proj checkpoint into split "
                f"projections (layer {dotted}); load from a dense checkpoint or with a TP size where "
                f"linear_attention_merged_split_proj stays active."
            )

        qkvz_sizes = tuple(int(x) for x in gdr.qkvz_segment_sizes)
        conv_dim = int(gdr.key_dim) * 2 + int(gdr.value_dim)
        for key in list(qkvz_keys):
            raw = flat_state.pop(key)
            val = raw.value if hasattr(raw, "value") and not hasattr(raw, "shape") else raw
            canon = _transform_last_axis(val, qkvz_sizes, eff_saved_tp, to_canonical=True)
            xp = np if isinstance(canon, np.ndarray) else jnp
            qkv_part, z_part = xp.split(canon, [conv_dim], axis=-1)
            flat_state[_replace_leaf_parent(key, "in_proj_qkvz", "in_proj_qkv")] = qkv_part
            flat_state[_replace_leaf_parent(key, "in_proj_qkvz", "in_proj_z")] = z_part

        num_v_heads = int(getattr(sub, "num_v_heads", 0))
        for key in list(ba_keys):
            raw = flat_state.pop(key)
            val = raw.value if hasattr(raw, "value") and not hasattr(raw, "shape") else raw
            half = num_v_heads if num_v_heads > 0 else int(val.shape[-1]) // 2
            canon = _transform_last_axis(val, (half, half), eff_saved_tp, to_canonical=True)
            xp = np if isinstance(canon, np.ndarray) else jnp
            b_part, a_part = xp.split(canon, [half], axis=-1)
            flat_state[_replace_leaf_parent(key, "in_proj_ba", "in_proj_b")] = b_part
            flat_state[_replace_leaf_parent(key, "in_proj_ba", "in_proj_a")] = a_part

        conv_sizes = (int(gdr.key_dim), int(gdr.key_dim), int(gdr.value_dim))
        for key in list(conv_keys):
            raw = flat_state[key]
            is_param = hasattr(raw, "value") and not hasattr(raw, "shape")
            val = raw.value if is_param else raw
            canon = _transform_last_axis(val, conv_sizes, eff_saved_tp, to_canonical=True)
            if canon is not val:
                canon = _reshard_like_source(canon, val)
                if is_param:
                    raw.value = canon
                else:
                    flat_state[key] = canon

        unfused_layers += 1

    if unfused_layers:
        logger.info(
            "Unfused %d native merged-split linear-attention layer(s) (saved_tp=%d) into split projections.",
            unfused_layers,
            eff_saved_tp,
        )
    return unfused_layers


def validate_linear_attention_native_optimizer_layout(
    load_directory,
    module,
    opt_state: tp.Any,
) -> None:
    """Reject incompatible cross-mode linear-attention optimizer checkpoints.

    Mirrors :func:`reconcile_linear_attention_native_layout` for optimizer
    pytrees: when a checkpoint's optimizer state carries ``in_proj_qkvz``
    (without ``in_proj_qkv``) and the live model uses a different
    linear-attention projection layout — including ``can_merge_tp`` fallback
    to split projections at a higher TP size — fail loudly with ``ValueError``
    rather than silently retaining stale ``in_proj_qkvz`` / TP-interleaved
    ``conv1d`` optimizer moments.
    """
    if opt_state is None:
        return
    saved_merged_flag, _, found_config = _read_saved_linear_attention_flags(load_directory)
    if not found_config:
        return
    saved_merged = bool(saved_merged_flag)

    linear_subs: list[tuple[str, bool, bool]] = []
    seen: set[int] = set()
    for path, sub in iter_module_search(module, object):
        if id(sub) in seen:
            continue
        seen.add(id(sub))
        if getattr(sub, "linear_attention_layout", None) is None:
            continue
        dotted = ".".join(str(p) for p in path)
        linear_subs.append(
            (
                dotted,
                bool(getattr(sub, "uses_merged_split_proj", False)),
                bool(getattr(sub, "uses_split_proj", False)),
            )
        )
    if not linear_subs:
        return

    seen_qkvz: set[str] = set()
    seen_qkv: set[str] = set()
    for path, _ in jax.tree_util.tree_leaves_with_path(opt_state):
        key_str = tree_path_to_string(path, sep=".")
        for dotted, _, _ in linear_subs:
            if _matches_fused_optimizer_leaf(key_str, f"{dotted}.in_proj_qkvz"):
                seen_qkvz.add(dotted)
            if _matches_fused_optimizer_leaf(key_str, f"{dotted}.in_proj_qkv"):
                seen_qkv.add(dotted)

    for dotted, live_merged, live_split in linear_subs:
        if dotted not in seen_qkvz or dotted in seen_qkv:
            continue
        if saved_merged and not live_merged and not live_split:
            raise ValueError(
                f"Optimizer checkpoint {load_directory} was saved with linear_attention_merged_split_proj=True "
                f"and cannot be loaded into default per-key-head packed mode (layer {dotted})."
            )
        if not saved_merged and live_merged:
            raise ValueError(
                f"Optimizer checkpoint {load_directory} stores default per-key-head packed linear-attention moments "
                f"(linear_attention_merged_split_proj=False) and cannot be loaded with "
                f"linear_attention_merged_split_proj=True (layer {dotted})."
            )
        if not saved_merged and live_split:
            raise ValueError(
                f"Optimizer checkpoint {load_directory} stores default per-key-head packed linear-attention moments "
                f"and cannot be loaded into linear_attention_separate_proj=True (layer {dotted})."
            )
        if saved_merged and live_split:
            raise ValueError(
                f"Optimizer checkpoint {load_directory} was saved with linear_attention_merged_split_proj=True "
                f"(merged in_proj_qkvz/in_proj_ba moments) and cannot be restored into a split-projection model "
                f"(layer {dotted}, e.g. explicit linear_attention_separate_proj=True or can_merge_tp fallback at "
                f"higher TP). Re-initialize the optimizer or resume at a TP size where "
                f"linear_attention_merged_split_proj stays active."
            )


def write_fused_layout_marker(save_directory, tp_size: int = 1) -> None:
    """Record the tensor-parallel size whose interleave the checkpoint stores."""
    try:
        (ePath(str(save_directory)) / FUSED_LAYOUT_MARKER).write_text(json.dumps({FUSED_TP_FIELD: int(tp_size)}))
    except Exception as exc:  # pragma: no cover - marker is best-effort; config.json carries the same field
        logger.warning(f"Could not write {FUSED_LAYOUT_MARKER}: {exc}")


def _tp_from_payload(payload: dict) -> int | None:
    """Extract the fused tp from a marker/config JSON payload.

    The previous format tagged canonical (contiguous) checkpoints with
    ``fused_param_layout: "canonical"`` — canonical is exactly ``tp=1``.
    """
    tp_value = payload.get(FUSED_TP_FIELD)
    if tp_value is not None:
        return int(tp_value)
    if payload.get("fused_param_layout") == CANONICAL:
        return 1
    return None


def read_fused_checkpoint_tp(load_directory, config: EasyDeLBaseConfig | None = None) -> int | None:
    """Return the tp whose interleave the checkpoint's fused params carry.

    Resolution order: sidecar marker file, then the ``config.json`` in the
    same directory, then the in-memory ``config`` (last resort — it may
    describe a different checkpoint than the one being loaded). ``None``
    means a legacy checkpoint with no recorded layout: only loadable
    verbatim at the exact tp it was saved with.
    """
    try:
        marker_path = ePath(str(load_directory)) / FUSED_LAYOUT_MARKER
        if marker_path.exists():
            tp_value = _tp_from_payload(json.loads(marker_path.read_text()))
            if tp_value is not None:
                return tp_value
    except Exception:  # pragma: no cover
        pass
    try:
        config_path = ePath(str(load_directory)) / "config.json"
        if config_path.exists():
            tp_value = _tp_from_payload(json.loads(config_path.read_text()))
            if tp_value is not None:
                return tp_value
    except Exception:  # pragma: no cover
        pass
    if config is not None:
        tp_value = getattr(config, FUSED_TP_FIELD, None)
        if tp_value is not None:
            return int(tp_value)
        if getattr(config, "fused_param_layout", None) == CANONICAL:
            return 1
    return None


def _iter_sub_configs(value: tp.Any, config_cls: type) -> tp.Iterator[tp.Any]:
    """Yield config instances found in ``value`` (directly or in containers)."""
    if isinstance(value, config_cls):
        yield value
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_sub_configs(item, config_cls)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_sub_configs(item, config_cls)


def stamp_fused_tp_config(config: EasyDeLBaseConfig, tp_size: int) -> None:
    """Record the fused-interleave tp on a config, recursively.

    ``config.json`` is required for any load to succeed, so this field —
    unlike the sidecar marker — cannot be separated from a loadable
    checkpoint. Sub-configs (``text_config``, ``vision_config``, ...) are
    stamped too so a sub-config extracted and saved on its own stays
    self-describing. The retired ``fused_param_layout`` string tag is
    dropped wherever found.
    """
    from transformers import PretrainedConfig

    seen: set[int] = set()

    def _stamp(cfg: tp.Any) -> None:
        if id(cfg) in seen:
            return
        seen.add(id(cfg))
        cfg.__dict__.pop("fused_param_layout", None)
        setattr(cfg, FUSED_TP_FIELD, int(tp_size))
        for value in vars(cfg).values():
            for sub in _iter_sub_configs(value, PretrainedConfig):
                _stamp(sub)

    _stamp(config)


def strip_fused_tp_config(config: EasyDeLBaseConfig) -> None:
    """Remove the fused-layout bookkeeping fields from a config, recursively.

    Use on config copies handed to HuggingFace exports — the field describes
    EasyDeL's native checkpoint layout and must not leak into HF artifacts.
    """
    from transformers import PretrainedConfig

    seen: set[int] = set()

    def _strip(cfg: tp.Any) -> None:
        if id(cfg) in seen:
            return
        seen.add(id(cfg))
        cfg.__dict__.pop("fused_param_layout", None)
        cfg.__dict__.pop(FUSED_TP_FIELD, None)
        for value in vars(cfg).values():
            for sub in _iter_sub_configs(value, PretrainedConfig):
                _strip(sub)

    _strip(config)
