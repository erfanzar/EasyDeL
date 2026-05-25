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

"""Checkpoint layouts for MoE fused expert gate/up kernels.

Owns :class:`FusedExpertLayout`, the MoE counterpart to
:class:`FusedColumnLayout` from :mod:`easydel.layers.layouts.dense`. The
expert-axis arithmetic is different enough from dense projections that a
separate type is used:

* MoE expert weights have a leading ``[experts, ...]`` axis, so the
  TP axis lives on a non-leading axis of the parameter
  (:attr:`FusedExpertLayout.weight_tp_dim` and
  :attr:`FusedExpertLayout.bias_tp_dim`).
* HF checkpoints commonly use ``[experts, intermediate, hidden]`` while
  EasyDeL's grouped matmul consumes ``[experts, hidden, intermediate]`` —
  the layout transposes during reform when
  :attr:`FusedExpertLayout.transpose_weight` is ``True``.
* HF checkpoints come in two flavours: separate ``gate_proj`` /
  ``up_proj`` (``source_is_fused=False``, the default) and a single
  pre-fused ``gate_up_proj`` (``source_is_fused=True``). The reform
  rule branches accordingly.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from ._runtime import tensor_parallel_size
from ._torch_packing import (
    torch_deinterleave_axis_segments_for_tp,
    torch_deinterleave_segments_for_tp,
    torch_interleave_axis_segments_for_tp,
    torch_interleave_segments_for_tp,
)
from ._types import EasyDeLBaseConfig, ReformParam


@dataclass(frozen=True, slots=True)
class FusedExpertLayout:
    """Checkpoint layout for fast MoE expert gate/up kernels.

    MoE expert weights do not share the dense column-projection contract:
    checkpoints commonly store ``[experts, intermediate, hidden]`` while
    the EasyDeL grouped expert matmul consumes
    ``[experts, hidden, intermediate]``. TP packing therefore happens on
    the expert-local intermediate axis at load time, while runtime
    activations split the local gate/up halves contiguously inside the
    kernel.

    Attributes:
        target_prefix (str): Destination EasyDeL parameter prefix for the
            fused gate/up tensor (``"gate_up_proj"`` by default).
        gate_prefix (str): HF source prefix for the gate weight when
            ``source_is_fused=False``.
        up_prefix (str): HF source prefix for the up weight when
            ``source_is_fused=False``.
        source_prefix (str): HF source prefix when ``source_is_fused=True``
            and the checkpoint already stores a pre-fused
            ``gate_up_proj`` tensor.
        source_is_fused (bool): Whether the source checkpoint already
            stores gate/up fused into a single tensor. Defaults to
            ``False`` (HF stores them separately).
        transpose_weight (bool): Whether to transpose the last two axes
            of expert weights during reform (HF
            ``[experts, intermediate, hidden]`` -> EasyDeL
            ``[experts, hidden, intermediate]``). Defaults to ``True``.
        weight_tp_dim (int): Axis along which the weight is TP-interleaved.
            Default ``2`` corresponds to the intermediate axis after the
            transpose (``[experts, hidden, intermediate]``).
        bias_tp_dim (int): Axis along which the bias is TP-interleaved.
            Default ``1`` corresponds to the intermediate axis of a
            ``[experts, intermediate]`` bias tensor.
    """

    target_prefix: str = "gate_up_proj"
    gate_prefix: str = "gate_proj"
    up_prefix: str = "up_proj"
    source_prefix: str = "gate_up_proj"
    source_is_fused: bool = False
    transpose_weight: bool = True
    weight_tp_dim: int = 2
    bias_tp_dim: int = 1

    def reform_param(
        self,
        *,
        config: EasyDeLBaseConfig | None = None,
        include_bias: bool = False,
    ) -> ReformParam:
        """Build the checkpoint load/export rules for this expert layout.

        Branches between the ``source_is_fused`` and the standard
        (``gate``/``up`` separate) HF layouts. In both branches the rule
        is marked ``already_converted=True`` so the generic 3-D tensor
        converter does not double-handle the expert axis. Both forward
        (``fuser``) and reverse (``inverse_fuser``) closures are wired in
        so checkpoint export round-trips back to the source layout.

        Args:
            config: Owning model config used to resolve the TP size.
            include_bias: Whether to generate bias-fusion rules in
                addition to weight rules. Defaults to ``False`` because
                most MoE expert kernels are bias-free.

        Returns:
            :type:`ReformParam` dict ready to merge into a module's
            ``reform_param``.
        """

        def _tp_size(arr: tp.Any | None = None) -> int:
            """Return the active tensor-parallel size for the given tensor.

            Captures ``config`` from the enclosing scope and forwards to
            :func:`tensor_parallel_size`; the optional ``arr`` argument
            lets the resolver consult the tensor's own sharding when
            the global config does not pin down a mesh.

            Args:
                arr: Optional tensor whose sharding influences mesh
                    resolution.

            Returns:
                Active TP size (``1`` when no TP axis applies).
            """
            return tensor_parallel_size(config, arr=arr)

        if self.source_is_fused:

            def _weight_fuser(torch: tp.Any, gate_up: tp.Any) -> tp.Any:
                """Transpose and TP-interleave a pre-fused gate/up weight.

                Args:
                    torch: Torch module reference (used for torch ops).
                    gate_up: Pre-fused source weight tensor.

                Returns:
                    TP-interleaved fused weight tensor in EasyDeL layout.
                """
                if self.transpose_weight:
                    gate_up = gate_up.transpose(-1, -2).contiguous()
                half = int(gate_up.shape[self.weight_tp_dim]) // 2
                return torch_interleave_axis_segments_for_tp(
                    torch,
                    gate_up,
                    (half, half),
                    tp_size=_tp_size(gate_up),
                    dim=self.weight_tp_dim,
                )

            def _weight_inverse_fuser(torch: tp.Any, gate_up: tp.Any) -> tuple[tp.Any]:
                """De-interleave and transpose the fused weight back to HF layout.

                Args:
                    torch: Torch module reference.
                    gate_up: Fused weight tensor in EasyDeL layout.

                Returns:
                    Single-element tuple containing the original HF-style
                    pre-fused tensor.
                """
                half = int(gate_up.shape[self.weight_tp_dim]) // 2
                gate_up = torch_deinterleave_axis_segments_for_tp(
                    torch,
                    gate_up,
                    (half, half),
                    tp_size=_tp_size(gate_up),
                    dim=self.weight_tp_dim,
                )
                if self.transpose_weight:
                    gate_up = gate_up.transpose(-1, -2).contiguous()
                return (gate_up,)

            reform_param: ReformParam = {
                f"{self.target_prefix}.weight$": {
                    "sources": (self.source_prefix,),
                    "fuser": _weight_fuser,
                    "inverse_fuser": _weight_inverse_fuser,
                    "already_converted": True,
                    "log_label": "MoE fused gate/up weight groups",
                }
            }
            if include_bias:
                reform_param[f"{self.target_prefix}.bias$"] = {
                    "sources": (f"{self.source_prefix}_bias",),
                    "fuser": lambda torch, gate_up: torch_interleave_axis_segments_for_tp(
                        torch,
                        gate_up,
                        (
                            int(gate_up.shape[self.bias_tp_dim]) // 2,
                            int(gate_up.shape[self.bias_tp_dim]) // 2,
                        ),
                        tp_size=_tp_size(gate_up),
                        dim=self.bias_tp_dim,
                    ),
                    "inverse_fuser": lambda torch, gate_up: (
                        torch_deinterleave_axis_segments_for_tp(
                            torch,
                            gate_up,
                            (
                                int(gate_up.shape[self.bias_tp_dim]) // 2,
                                int(gate_up.shape[self.bias_tp_dim]) // 2,
                            ),
                            tp_size=_tp_size(gate_up),
                            dim=self.bias_tp_dim,
                        ),
                    ),
                    "already_converted": True,
                    "log_label": "MoE fused gate/up bias groups",
                }
            return reform_param

        def _weight_fuser(torch: tp.Any, gate: tp.Any, up: tp.Any) -> tp.Any:
            """Transpose and TP-interleave separate gate/up weights into one tensor.

            Args:
                torch: Torch module reference.
                gate: Per-expert gate weight tensor in HF layout.
                up: Per-expert up weight tensor in HF layout.

            Returns:
                Fused TP-interleaved gate/up weight in EasyDeL layout.
            """
            gate = gate.transpose(-1, -2).contiguous()
            up = up.transpose(-1, -2).contiguous()
            return torch_interleave_segments_for_tp(torch, (gate, up), tp_size=_tp_size(gate), dim=self.weight_tp_dim)

        def _weight_inverse_fuser(torch: tp.Any, gate_up: tp.Any) -> tuple[tp.Any, tp.Any]:
            """Split a fused gate/up tensor back into separate HF-layout tensors.

            Args:
                torch: Torch module reference.
                gate_up: Fused EasyDeL-layout gate/up weight tensor.

            Returns:
                Pair ``(gate, up)`` in the original HF layout
                ``[experts, intermediate, hidden]``.
            """
            half = int(gate_up.shape[self.weight_tp_dim]) // 2
            gate, up = torch_deinterleave_segments_for_tp(
                torch,
                gate_up,
                (half, half),
                tp_size=_tp_size(gate_up),
                dim=self.weight_tp_dim,
            )
            return gate.transpose(-1, -2).contiguous(), up.transpose(-1, -2).contiguous()

        reform_param = {
            f"{self.target_prefix}.weight$": {
                "sources": (f"{self.gate_prefix}.weight", f"{self.up_prefix}.weight"),
                "fuser": _weight_fuser,
                "inverse_fuser": _weight_inverse_fuser,
                "already_converted": True,
                "log_label": "MoE gate/up weight groups",
            }
        }
        if include_bias:
            reform_param[f"{self.target_prefix}.bias$"] = {
                "sources": (f"{self.gate_prefix}.bias", f"{self.up_prefix}.bias"),
                "fuser": lambda torch, gate, up: torch_interleave_segments_for_tp(
                    torch,
                    (gate, up),
                    tp_size=_tp_size(gate),
                    dim=self.bias_tp_dim,
                ),
                "inverse_fuser": lambda torch, gate_up: torch_deinterleave_segments_for_tp(
                    torch,
                    gate_up,
                    (
                        int(gate_up.shape[self.bias_tp_dim]) // 2,
                        int(gate_up.shape[self.bias_tp_dim]) // 2,
                    ),
                    tp_size=_tp_size(gate_up),
                    dim=self.bias_tp_dim,
                ),
                "already_converted": True,
                "log_label": "MoE gate/up bias groups",
            }
        return reform_param
