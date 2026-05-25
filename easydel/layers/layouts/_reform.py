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

"""Checkpoint reform rule builders for fused projections.

Generates the dict-of-dicts :type:`ReformParam` rules consumed by the
EasyDeL checkpoint loader / exporter, which let a single fused EasyDeL
tensor be loaded from (and re-exported to) several separate HF tensors
with a TP-interleaved layout.

Public helpers:
    :func:`interleaved_fusion_reform_param`: Generic multi-source rule for
        TP-interleaved packing on any axis.
    :func:`gate_up_fusion_reform_param`: Dense MLP ``gate/up -> gate_up``.
    :func:`qkv_fusion_reform_param`: Dense attention ``q/k/v -> qkv``.
    :func:`moe_gate_up_fusion_reform_param`: MoE expert ``gate/up`` from
        separate HF tensors via :class:`FusedExpertLayout`.
    :func:`moe_fused_gate_up_reform_param`: MoE expert ``gate_up`` from a
        pre-fused HF tensor.

The MoE helpers are thin wrappers around
:meth:`FusedExpertLayout.reform_param`; the dense helpers either delegate
to :func:`interleaved_fusion_reform_param` directly or, when explicit
segment sizes are needed, build a :class:`FusedColumnLayout` first.
"""

from __future__ import annotations

import typing as tp

from ._runtime import normalize_segment_sizes, tensor_parallel_size
from ._torch_packing import torch_deinterleave_segments_for_tp, torch_interleave_segments_for_tp
from ._types import EasyDeLBaseConfig


def interleaved_fusion_reform_param(
    target_name: str,
    source_names: tp.Sequence[str],
    *,
    config: EasyDeLBaseConfig | None = None,
    dim: int = 0,
    segment_sizes: tp.Sequence[int] | None = None,
    source_transforms: tp.Sequence[tp.Callable[[tp.Any], tp.Any] | None] | None = None,
    inverse_source_transforms: tp.Sequence[tp.Callable[[tp.Any], tp.Any] | None] | None = None,
    log_label: str | None = None,
) -> dict[str, dict[str, tp.Any]]:
    """Build a multi-source reform rule for TP-interleaved packing.

    Covers the common HF layout where separate per-purpose tensors
    (e.g. ``q_proj.weight``, ``k_proj.weight``, ``v_proj.weight``) need
    to be packed into a single EasyDeL column-parallel tensor whose
    output axis is rank-interleaved across the tensor-parallel mesh axis.

    Both forward (``fuser``) and inverse (``inverse_fuser``) closures are
    generated so that checkpoint export round-trips back to the source
    tensors. Optional pre-fusion and post-deinterleave transforms can be
    supplied for layouts that require additional reshaping (e.g.
    transposing a per-source tensor before fusion).

    Args:
        target_name: Destination tensor name in the EasyDeL state tree
            (e.g. ``"qkv_proj.weight"``); the returned dict key is this
            name followed by ``"$"`` so the loader treats it as a regex
            anchor.
        source_names: Sequence of HF source tensor names to fuse, in the
            concatenation order expected on the fused output axis.
        config: Owning model config used to resolve the active TP size.
        dim: Axis along which the fusion happens in the source tensors.
            Defaults to ``0`` (the first axis of HF ``nn.Linear`` weights).
        segment_sizes: Optional explicit per-segment sizes; when ``None``
            the inverse fuser falls back to even-chunk splitting (or
            torch ``chunk``) using the runtime tensor shape.
        source_transforms: Optional per-source pre-fusion callables. If
            provided, must align with ``source_names``.
        inverse_source_transforms: Optional per-segment post-deinterleave
            callables for export.
        log_label: Optional human-friendly label used in checkpoint
            reform logs.

    Returns:
        :type:`ReformParam`-style mapping ``{f"{target_name}$": {...}}``
        containing ``sources``, ``fuser``, ``inverse_fuser`` and
        ``log_label`` keys.

    Raises:
        ValueError: When ``segment_sizes``, ``source_transforms``, or
            ``inverse_source_transforms`` are provided but do not align
            with the length of ``source_names``.
    """
    source_names = tuple(source_names)
    segment_sizes = normalize_segment_sizes(segment_sizes) if segment_sizes is not None else None
    if segment_sizes is not None and len(segment_sizes) != len(source_names):
        raise ValueError("segment_sizes must have the same length as source_names")
    transforms = tuple(source_transforms or ())
    if transforms and len(transforms) != len(source_names):
        raise ValueError("source_transforms must have the same length as source_names")
    inverse_transforms = tuple(inverse_source_transforms or ())
    if inverse_transforms and len(inverse_transforms) != len(source_names):
        raise ValueError("inverse_source_transforms must have the same length as source_names")

    def _tp_size(arr: tp.Any | None = None) -> int:
        """Resolve the active TP size, optionally informed by a tensor's sharding.

        Args:
            arr: Optional tensor whose sharding can pin down the mesh.

        Returns:
            Active tensor-parallel size; ``1`` when no TP axis applies.
        """
        return tensor_parallel_size(config, arr=arr)

    def _fuser(torch: tp.Any, *tensors: tp.Any) -> tp.Any:
        """Apply pre-fusion transforms then TP-interleave the source tensors.

        Args:
            torch: Torch module reference.
            *tensors: Per-source tensors aligned with ``source_names``.

        Returns:
            TP-interleaved fused tensor matching ``target_name`` shape.
        """
        if transforms:
            tensors = tuple(
                transform(tensor) if transform is not None else tensor
                for tensor, transform in zip(tensors, transforms, strict=True)
            )
        return torch_interleave_segments_for_tp(
            torch, tensors, tp_size=_tp_size(tensors[0] if tensors else None), dim=dim
        )

    def _inverse_fuser(torch: tp.Any, tensor: tp.Any) -> tuple[tp.Any, ...]:
        """De-interleave and apply inverse transforms to recover source tensors.

        Three branches:

        * One source: pass through unchanged.
        * Explicit / inferable per-segment sizes: TP-aware de-interleave.
        * Otherwise: even-chunk fall-back via ``torch.chunk``.

        Args:
            torch: Torch module reference.
            tensor: Fused tensor to split.

        Returns:
            Tuple of per-source tensors aligned with ``source_names``.
        """
        if len(source_names) == 1:
            outputs = (tensor,)
        else:
            total = int(tensor.shape[dim])
            active_segment_sizes = segment_sizes
            if active_segment_sizes is None and total % len(source_names) == 0:
                active_segment_sizes = (total // len(source_names),) * len(source_names)
            if active_segment_sizes is None or sum(active_segment_sizes) != total:
                outputs = tuple(torch.chunk(tensor, len(source_names), dim=dim))
            else:
                outputs = torch_deinterleave_segments_for_tp(
                    torch,
                    tensor,
                    active_segment_sizes,
                    tp_size=_tp_size(tensor),
                    dim=dim,
                )
        if inverse_transforms:
            outputs = tuple(
                transform(part) if transform is not None else part
                for part, transform in zip(outputs, inverse_transforms, strict=True)
            )
        return outputs

    return {
        f"{target_name}$": {
            "sources": source_names,
            "fuser": _fuser,
            "inverse_fuser": _inverse_fuser,
            "log_label": log_label or f"{target_name} interleaved fusion groups",
        }
    }


def gate_up_fusion_reform_param(
    *,
    config: EasyDeLBaseConfig | None = None,
    include_bias: bool = False,
    target_prefix: str = "gate_up_proj",
    gate_prefix: str = "gate_proj",
    up_prefix: str = "up_proj",
) -> dict[str, dict[str, tp.Any]]:
    """Build reform rules to fuse dense MLP ``gate_proj`` / ``up_proj``.

    Convenience wrapper around :func:`interleaved_fusion_reform_param` that
    sources HF ``gate_proj`` / ``up_proj`` tensors into a single
    ``gate_up_proj``.

    Args:
        config: Owning model config used to resolve the TP size.
        include_bias: When ``True`` also generate bias-fusion rules.
        target_prefix: Destination prefix in the EasyDeL state tree.
        gate_prefix: HF source prefix for the gate half.
        up_prefix: HF source prefix for the up half.

    Returns:
        :type:`ReformParam` dict suitable for merging into a module's
        ``reform_param``.
    """
    weight_sources = (f"{gate_prefix}.weight", f"{up_prefix}.weight")
    bias_sources = (f"{gate_prefix}.bias", f"{up_prefix}.bias")
    reform_param = interleaved_fusion_reform_param(
        f"{target_prefix}.weight",
        weight_sources,
        config=config,
        segment_sizes=None,
        log_label="dense-MLP gate/up weight groups",
    )
    if include_bias:
        reform_param.update(
            interleaved_fusion_reform_param(
                f"{target_prefix}.bias",
                bias_sources,
                config=config,
                segment_sizes=None,
                log_label="dense-MLP gate/up bias groups",
            )
        )
    return reform_param


def moe_gate_up_fusion_reform_param(
    *,
    config: EasyDeLBaseConfig | None = None,
    include_bias: bool = False,
    target_prefix: str = "gate_up_proj",
    gate_prefix: str = "gate_proj",
    up_prefix: str = "up_proj",
) -> dict[str, dict[str, tp.Any]]:
    """Build reform rules to fuse MoE expert ``gate_proj`` / ``up_proj``.

    HF stacked MoE expert kernels store
    ``[experts, intermediate, hidden]``. EasyDeL's fast grouped-matmul
    consumes ``[experts, hidden, intermediate]`` and marks the rule
    ``already_converted=True`` so the generic 3-D tensor converter
    leaves the expert axis alone.

    Args:
        config: Owning model config used to resolve the TP size.
        include_bias: When ``True`` also generate bias-fusion rules.
        target_prefix: Destination prefix in the EasyDeL state tree.
        gate_prefix: HF source prefix for the gate half.
        up_prefix: HF source prefix for the up half.

    Returns:
        :type:`ReformParam` dict suitable for merging into a module's
        ``reform_param``; delegates to :class:`FusedExpertLayout`.
    """
    from ._moe import FusedExpertLayout

    return FusedExpertLayout(
        target_prefix=target_prefix,
        gate_prefix=gate_prefix,
        up_prefix=up_prefix,
    ).reform_param(config=config, include_bias=include_bias)


def moe_fused_gate_up_reform_param(
    *,
    config: EasyDeLBaseConfig | None = None,
    include_bias: bool = False,
    target_prefix: str = "gate_up_proj",
    source_prefix: str = "gate_up_proj",
    transpose_weight: bool = True,
) -> dict[str, dict[str, tp.Any]]:
    """Build reform rules to load a pre-fused HF ``gate_up_proj`` into EasyDeL layout.

    Used when the HF checkpoint already concatenates the gate and up
    halves into a single tensor (less common than the separate
    ``gate_proj`` / ``up_proj`` layout). Delegates to
    :class:`FusedExpertLayout` with ``source_is_fused=True``.

    Args:
        config: Owning model config used to resolve the TP size.
        include_bias: When ``True`` also generate bias-fusion rules.
        target_prefix: Destination prefix in the EasyDeL state tree.
        source_prefix: HF source prefix for the pre-fused tensor.
        transpose_weight: Whether to transpose the source weight from
            HF layout ``[experts, intermediate, hidden]`` to EasyDeL
            layout ``[experts, hidden, intermediate]``.

    Returns:
        :type:`ReformParam` dict ready to merge into a module's
        ``reform_param``.
    """
    from ._moe import FusedExpertLayout

    return FusedExpertLayout(
        target_prefix=target_prefix,
        source_prefix=source_prefix,
        source_is_fused=True,
        transpose_weight=transpose_weight,
    ).reform_param(config=config, include_bias=include_bias)


def moe_down_projection_reform_param(
    *,
    target_prefix: str = "down_proj",
    source_prefix: str = "down_proj",
    transpose_weight: bool = True,
    inverse_transpose_weight: bool | None = None,
) -> dict[str, dict[str, tp.Any]]:
    """Build reform rules for a stacked MoE down projection.

    Args:
        target_prefix: Destination EasyDeL down-projection prefix.
        source_prefix: Source checkpoint down-projection prefix.
        transpose_weight: Whether to transpose the source weight during
            load, matching HF ``[experts, out, in]`` to EasyDeL's runtime
            layout.
        inverse_transpose_weight: Whether to transpose back during export.
            Defaults to ``transpose_weight``.

    Returns:
        :type:`ReformParam` rule for the down-projection tensor.
    """
    if inverse_transpose_weight is None:
        inverse_transpose_weight = transpose_weight

    def _load_transform(tensor: tp.Any) -> tp.Any:
        return tensor.swapaxes(-1, -2) if transpose_weight else tensor

    def _export_transform(tensor: tp.Any) -> tp.Any:
        return tensor.swapaxes(-1, -2) if inverse_transpose_weight else tensor

    return {
        f"{target_prefix}$": {
            "splits": [{"name": f"{source_prefix}.weight", "spliter": _load_transform}],
            "inverse_spliter": _export_transform,
        }
    }


def hf_per_expert_swiglu_reform_param(
    *,
    num_experts: int,
    router_prefix: str = "router",
    router_target_prefix: str = "router",
    experts_prefix: str = "experts",
    fused_gate_up_prefix: str = "experts.gate_up_proj",
    fused_down_prefix: str = "experts.down_proj",
    gate_prefix: str = "w1",
    up_prefix: str = "w3",
    down_prefix: str = "w2",
) -> dict[str, dict[str, tp.Any]]:
    """Build HF-style per-expert SwiGLU reform rules.

    This covers models whose EasyDeL runtime keeps one module per expert
    with HF names such as ``experts.0.w1.weight`` / ``w2`` / ``w3``,
    while the checkpoint bridge may see stacked fused tensors such as
    ``experts.gate_up_proj`` and ``experts.down_proj``.

    Args:
        num_experts: Number of local experts to address.
        router_prefix: Source router prefix in the checkpoint.
        router_target_prefix: Runtime router prefix inside the module.
        experts_prefix: Runtime expert-list prefix.
        fused_gate_up_prefix: Source fused gate/up tensor prefix.
        fused_down_prefix: Source fused down tensor prefix.
        gate_prefix: Per-expert gate projection name.
        up_prefix: Per-expert up projection name.
        down_prefix: Per-expert down projection name.

    Returns:
        :type:`ReformParam` dict that expands the stacked checkpoint
        tensors into HF-style per-expert runtime names and can merge them
        back on export.
    """

    def _merge_expert_gate_up(torch: tp.Any, *tensors: tp.Any) -> tp.Any:
        experts = []
        for gate, up in zip(tensors[::2], tensors[1::2], strict=True):
            experts.append(torch.cat((gate.swapaxes(-1, -2), up.swapaxes(-1, -2)), dim=0))
        return torch.stack(experts, dim=0)

    def _merge_expert_down(torch: tp.Any, *tensors: tp.Any) -> tp.Any:
        return torch.stack(tuple(tensor.swapaxes(-1, -2) for tensor in tensors), dim=0)

    gate_up_splits: list[dict[str, tp.Any]] = []
    down_splits: list[dict[str, tp.Any]] = []
    for expert_idx in range(num_experts):
        gate_up_splits.append(
            {
                "name": f"{experts_prefix}.{expert_idx}.{gate_prefix}.weight",
                "spliter": lambda x, idx=expert_idx: x[idx, : x.shape[1] // 2, :].swapaxes(-1, -2),
            }
        )
        gate_up_splits.append(
            {
                "name": f"{experts_prefix}.{expert_idx}.{up_prefix}.weight",
                "spliter": lambda x, idx=expert_idx: x[idx, x.shape[1] // 2 :, :].swapaxes(-1, -2),
            }
        )
        down_splits.append(
            {
                "name": f"{experts_prefix}.{expert_idx}.{down_prefix}.weight",
                "spliter": lambda x, idx=expert_idx: x[idx].swapaxes(-1, -2),
            }
        )

    return {
        f"{router_prefix}.weight$": {
            "splits": [{"name": f"{router_target_prefix}.weight", "spliter": lambda x: x.swapaxes(-1, -2)}],
            "inverse_spliter": lambda x: x.swapaxes(-1, -2),
        },
        f"{fused_gate_up_prefix}$": {
            "splits": gate_up_splits,
            "inverse_spliter": _merge_expert_gate_up,
        },
        f"{fused_down_prefix}$": {
            "splits": down_splits,
            "inverse_spliter": _merge_expert_down,
        },
    }


def qkv_fusion_reform_param(
    *,
    config: EasyDeLBaseConfig | None = None,
    include_bias: bool = False,
    target_prefix: str = "qkv_proj",
    query_prefix: str = "q_proj",
    key_prefix: str = "k_proj",
    value_prefix: str = "v_proj",
    segment_sizes: tp.Sequence[int] | None = None,
) -> dict[str, dict[str, tp.Any]]:
    """Build reform rules to fuse dense attention ``q/k/v -> qkv``.

    When ``segment_sizes`` is provided (``(q_size, kv_size, v_size)``), a
    :class:`FusedColumnLayout` is built so the source tensor names follow
    the layout's segment ``source_prefix`` attribute; otherwise the
    sources are constructed directly from the explicit prefix arguments.
    Both branches end up calling :func:`interleaved_fusion_reform_param`
    with the resolved sources.

    Args:
        config: Owning model config used to resolve the TP size.
        include_bias: When ``True`` also generate bias-fusion rules.
        target_prefix: Destination prefix in the EasyDeL state tree.
        query_prefix: HF source prefix for the Q slice.
        key_prefix: HF source prefix for the K slice.
        value_prefix: HF source prefix for the V slice.
        segment_sizes: Optional ``(q_size, kv_size, v_size)`` widths; when
            ``None`` the inverse fuser relies on the runtime tensor shape
            to decide segment widths.

    Returns:
        :type:`ReformParam` dict suitable for merging into a module's
        ``reform_param``.
    """
    if segment_sizes is None:
        sources = (
            f"{query_prefix}.weight",
            f"{key_prefix}.weight",
            f"{value_prefix}.weight",
        )
        bias_sources = (
            f"{query_prefix}.bias",
            f"{key_prefix}.bias",
            f"{value_prefix}.bias",
        )
    else:
        from ._dense import FusedColumnLayout, FusedSegment

        q_size, kv_size, v_size = normalize_segment_sizes(segment_sizes)
        layout = FusedColumnLayout(
            segments=(
                FusedSegment("q", q_size, query_prefix),
                FusedSegment("k", kv_size, key_prefix),
                FusedSegment("v", v_size, value_prefix),
            ),
            log_label="dense-attention Q/K/V groups",
        )
        sources = tuple(segment.tensor_name("weight") for segment in layout.segments)
        bias_sources = tuple(segment.tensor_name("bias") for segment in layout.segments)

    reform_param = interleaved_fusion_reform_param(
        f"{target_prefix}.weight",
        sources,
        config=config,
        segment_sizes=segment_sizes,
        log_label="dense-attention Q/K/V weight groups",
    )
    if include_bias:
        reform_param.update(
            interleaved_fusion_reform_param(
                f"{target_prefix}.bias",
                bias_sources,
                config=config,
                segment_sizes=segment_sizes,
                log_label="dense-attention Q/K/V bias groups",
            )
        )
    return reform_param
