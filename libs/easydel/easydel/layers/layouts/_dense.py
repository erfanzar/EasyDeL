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

"""Declarative layouts for dense fused column-parallel projections.

Owns :class:`FusedColumnLayout` and its segment descriptor
:class:`FusedSegment`, plus the two canonical layout constructors
(:func:`dense_qkv_layout` and :func:`dense_gate_up_layout`) used by
:func:`easydel.layers.layouts.builders.build_fused_qkv_projection` and
:func:`build_fused_gate_up_projection`.

A layout owns two complementary contracts:

* Runtime activation splitting via :meth:`FusedColumnLayout.split` — knows
  how to undo TP-interleaved packing when tensor parallelism is active.
* Checkpoint reform rule generation via :meth:`FusedColumnLayout.reform_param`
  — generates rules that load separate HF q/k/v (or gate/up) tensors into
  one fused EasyDeL tensor with the correct TP-interleaved layout.

MoE expert layouts deliberately live in :mod:`easydel.layers.layouts.moe`
because their expert axis, transpose contract, and bias TP axis differ
from dense column projections.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._reform import interleaved_fusion_reform_param
from ._runtime import (
    split_contiguous_segments_last_axis,
    split_interleaved_segments_last_axis,
    tensor_parallel_size,
    with_tp_last_axis_sharding,
)
from ._types import Array, EasyDeLBaseConfig, ReformParam


@dataclass(frozen=True, slots=True)
class FusedSegment:
    """One logical sub-slice inside a fused column-parallel projection.

    Used to describe the constituent pieces of a fused tensor (e.g. the
    ``q``/``k``/``v`` parts of a QKV projection, or the ``gate``/``up``
    parts of a SwiGLU MLP) for the purpose of activation splitting and
    checkpoint reform rule generation.

    Attributes:
        name (str): Short human-readable identifier (``"q"``, ``"gate"``,
            ...). Must be unique inside its layout.
        size (int): Logical output width of this segment (the un-sharded
            number of features owned by it).
        source_prefix (str): HF-style attribute prefix of the original
            standalone tensor (e.g. ``"q_proj"``, ``"gate_proj"``). Used
            by :meth:`tensor_name` to build the source tensor name for
            checkpoint reform rules.
    """

    name: str
    size: int
    source_prefix: str

    def tensor_name(self, suffix: str) -> str:
        """Build the source-tensor name for a given parameter suffix.

        Args:
            suffix: Parameter suffix such as ``"weight"`` or ``"bias"``.

        Returns:
            String ``f"{source_prefix}.{suffix}"`` — the dotted name used
            in source checkpoints (HF format) for this segment.
        """
        return f"{self.source_prefix}.{suffix}"


@dataclass(frozen=True, slots=True)
class FusedColumnLayout:
    """Declarative layout for a dense fused column-parallel projection.

    Couples the segment layout of a fused tensor to both its runtime
    activation splitter and its checkpoint reform rule generator so that
    fusion stays a one-place change. Two canonical instances are produced
    by :func:`dense_qkv_layout` and :func:`dense_gate_up_layout`; ad-hoc
    layouts can be constructed directly for less common fusion patterns.

    The layout is intentionally *only* for dense projections; MoE expert
    layouts live in :class:`easydel.layers.layouts.moe.FusedExpertLayout`
    because the expert axis, transpose contract, and bias TP axis differ.

    Attributes:
        segments (tuple[FusedSegment, ...]): Per-segment descriptors in
            their concatenation order along the fused output axis.
        interleave_for_tp (bool): When ``True`` (the default) the layout
            assumes the fused activation/parameter is rank-interleaved on
            the TP mesh axis and uses the TP-aware splitter; set ``False``
            to use a plain contiguous split.
        torch_dim (int): Axis index along which fusion happens in source
            (torch / HF) tensors. ``0`` for ``nn.Linear`` weight matrices
            (output axis), nonzero for transposed / 3-D MoE weights.
        log_label (str | None): Optional human-friendly label used in
            checkpoint reform logs; falls back to a per-segment default.
    """

    segments: tuple[FusedSegment, ...]
    interleave_for_tp: bool = True
    torch_dim: int = 0
    log_label: str | None = None

    def __post_init__(self) -> None:
        """Validate segment uniqueness and positivity.

        Raises:
            ValueError: If ``segments`` is empty, if any two segments
                share the same ``name``, or if any segment size is not
                strictly positive.
        """
        if not self.segments:
            raise ValueError("FusedColumnLayout requires at least one segment")
        names = [segment.name for segment in self.segments]
        if len(set(names)) != len(names):
            raise ValueError(f"FusedColumnLayout segment names must be unique, got {names}")
        if any(int(segment.size) <= 0 for segment in self.segments):
            raise ValueError("FusedColumnLayout segment sizes must be positive")

    @property
    def segment_sizes(self) -> tuple[int, ...]:
        """Return the logical (un-sharded) output width of each segment."""
        return tuple(int(segment.size) for segment in self.segments)

    @property
    def out_features(self) -> int:
        """Return the total fused output width (sum of segment sizes)."""
        return sum(self.segment_sizes)

    def split(
        self,
        x: Array,
        *,
        config: EasyDeLBaseConfig | None = None,
        apply_sharding: bool = True,
    ) -> tuple[Array, ...]:
        """Split a runtime fused activation back into its logical segments.

        Two code paths:

        * ``interleave_for_tp=True`` (default): tries the TP-aware
          interleaved splitter. When TP is inactive or shapes prevent
          interleaving, falls through to the contiguous path; if TP is
          active but a segment size is not divisible by the TP size,
          raises :class:`ValueError` rather than silently producing
          wrong tensors.
        * ``interleave_for_tp=False``: always uses a contiguous last-axis
          split.

        Args:
            x: Activation of shape ``[..., out_features]``.
            config: Owning model config used to resolve the TP mesh axis;
                may be ``None`` in tests that pre-shard the tensor.
            apply_sharding: When ``True`` (default) re-applies the TP last
                axis sharding constraint to each output segment.

        Returns:
            Tuple of per-segment arrays in the same order as
            :attr:`segments`.

        Raises:
            ValueError: When TP is active but segment sizes are not
                divisible by the TP size.
        """
        if self.interleave_for_tp:
            parts = split_interleaved_segments_last_axis(
                x,
                self.segment_sizes,
                config=config,
                apply_sharding=apply_sharding,
            )
            if parts is not None:
                return parts
            tp_size = tensor_parallel_size(config, arr=x)
            if tp_size > 1 and x.ndim >= 1 and int(x.shape[-1]) == self.out_features:
                indivisible = tuple(size for size in self.segment_sizes if size % tp_size != 0)
                if indivisible:
                    raise ValueError(
                        "Fused TP-interleaved projection segments must be divisible by the active "
                        f"tensor-parallel size. Got segment_sizes={self.segment_sizes}, tp_size={tp_size}."
                    )

        outputs = split_contiguous_segments_last_axis(x, self.segment_sizes)
        if apply_sharding:
            return tuple(with_tp_last_axis_sharding(part, config) for part in outputs)
        return outputs

    def reform_param(
        self,
        target_prefix: str,
        *,
        config: EasyDeLBaseConfig | None = None,
        include_bias: bool = False,
        weight_log_label: str | None = None,
        bias_log_label: str | None = None,
    ) -> ReformParam:
        """Build checkpoint reform rules for this fused projection.

        Produces a :type:`ReformParam` mapping that the checkpoint
        loader uses to fuse the per-segment HF tensors
        (``{source_prefix}.weight``, optionally ``{source_prefix}.bias``)
        into the EasyDeL fused tensor ``{target_prefix}.weight`` with
        the correct TP-interleaved layout. Inverse rules are also wired
        in so :class:`PreTrainedModel`-style export round-trips back to
        separate tensors.

        Args:
            target_prefix: Destination prefix in the EasyDeL state tree
                (e.g. ``"qkv_proj"``, ``"gate_up_proj"``).
            config: Owning model config used to resolve the TP size.
            include_bias: When ``True`` also generate the bias-fusion
                rule. Default ``False`` because most attention/MLP fused
                projections are bias-free.
            weight_log_label: Optional override for the human-friendly
                weight log label. Falls back to :attr:`log_label` then
                to ``f"{target_prefix} fused weight groups"``.
            bias_log_label: Optional override for the bias log label.

        Returns:
            :type:`ReformParam` dict suitable for merging into a module's
            ``reform_param``.
        """
        reform_param = interleaved_fusion_reform_param(
            f"{target_prefix}.weight",
            tuple(segment.tensor_name("weight") for segment in self.segments),
            config=config,
            dim=self.torch_dim,
            segment_sizes=self.segment_sizes,
            log_label=weight_log_label or self.log_label or f"{target_prefix} fused weight groups",
        )
        if include_bias:
            reform_param.update(
                interleaved_fusion_reform_param(
                    f"{target_prefix}.bias",
                    tuple(segment.tensor_name("bias") for segment in self.segments),
                    config=config,
                    dim=self.torch_dim,
                    segment_sizes=self.segment_sizes,
                    log_label=bias_log_label or self.log_label or f"{target_prefix} fused bias groups",
                )
            )
        return reform_param


def dense_gate_up_layout(
    intermediate_size: int,
    *,
    gate_prefix: str = "gate_proj",
    up_prefix: str = "up_proj",
) -> FusedColumnLayout:
    """Construct the canonical dense MLP ``[gate | up]`` fused layout.

    Produces a two-segment layout with equal-sized ``gate`` and ``up``
    halves matching the SwiGLU pattern used by Llama, Mistral, Gemma,
    Qwen, etc.

    Args:
        intermediate_size: Width of each branch (gate and up are equal).
        gate_prefix: HF source-tensor prefix for the gate half.
        up_prefix: HF source-tensor prefix for the up half.

    Returns:
        :class:`FusedColumnLayout` with two equal-sized segments and
        ``"dense-MLP gate/up groups"`` as its default log label.
    """
    return FusedColumnLayout(
        segments=(
            FusedSegment("gate", int(intermediate_size), gate_prefix),
            FusedSegment("up", int(intermediate_size), up_prefix),
        ),
        log_label="dense-MLP gate/up groups",
    )


def dense_qkv_layout(
    q_size: int,
    kv_size: int,
    *,
    query_prefix: str = "q_proj",
    key_prefix: str = "k_proj",
    value_prefix: str = "v_proj",
) -> FusedColumnLayout:
    """Construct the canonical dense attention ``[Q | K | V]`` fused layout.

    Produces a three-segment layout with the canonical
    ``[q_size, kv_size, kv_size]`` shape. ``kv_size`` may equal ``q_size``
    for vanilla MHA or be smaller for GQA / MQA.

    Args:
        q_size: Width of the query slice (``num_heads * head_dim``).
        kv_size: Width of the key (and value) slice
            (``num_kv_heads * head_dim``).
        query_prefix: HF source-tensor prefix for the Q half.
        key_prefix: HF source-tensor prefix for the K half.
        value_prefix: HF source-tensor prefix for the V half.

    Returns:
        :class:`FusedColumnLayout` with three segments and
        ``"dense-attention Q/K/V groups"`` as its default log label.
    """
    return FusedColumnLayout(
        segments=(
            FusedSegment("q", int(q_size), query_prefix),
            FusedSegment("k", int(kv_size), key_prefix),
            FusedSegment("v", int(kv_size), value_prefix),
        ),
        log_label="dense-attention Q/K/V groups",
    )
