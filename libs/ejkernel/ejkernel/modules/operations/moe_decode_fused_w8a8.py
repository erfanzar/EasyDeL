# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


"""Decode MoE segment of one expert shard in one kernel (``moe_decode_fused_w8a8``).

For a few decode token rows, the routed W8A8 experts of this shard (gate/up, two-branch combine, down, top-k combine
weighting) plus the optional folded W8A8 shared expert, summed per token in float32 ``[T, H]``: what EasyDeL's
inference deferred MoE dispatch computes between the router's top-k and the MoE output reduction. On TPU one Pallas
kernel streams the active experts' int8 slabs through a VMEM ring; the XLA implementation (dense per-expert einsums)
is the reference and the fallback. APPROXIMATE against EasyDeL's XLA dispatch (summation order, in-kernel f32
lowering). Forward only (inference).
"""

from __future__ import annotations

from typing import Literal

from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel, Tuner
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import MoeDecodeFusedW8A8Config


class MoeDecodeFusedW8A8(Kernel[MoeDecodeFusedW8A8Config, Array]):
    """Fused decode MoE segment (Pallas TPU; XLA reference)."""

    version = "1"

    def __init__(self):
        """Initialize the operation under ``op_id="moe_decode_fused_w8a8"``."""
        super().__init__(op_id="moe_decode_fused_w8a8")

    def get_impl(self, cfg: MoeDecodeFusedW8A8Config):
        """Resolve the registry implementation for the config's platform/backend.

        Args:
            cfg: Operation config.

        Returns:
            The registered ``moe_decode_fused_w8a8`` callable.
        """
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(
        self,
        x,
        expert_ids,
        combine_weights,
        gate_codes,
        gate_scales,
        up_codes,
        up_scales,
        down_codes,
        down_scales,
        control,
        shared_gate_up_codes=None,
        shared_gate_up_scales=None,
        shared_down_codes=None,
        shared_down_scales=None,
        *,
        activation: str = "silu",
        act_params: tuple[float, ...] = (),
        shared_activation: str = "silu",
        shared_act_params: tuple[float, ...] = (),
        platform: Literal["pallas", "xla", "auto"] | None = None,
        cfg: MoeDecodeFusedW8A8Config,
        **_,
    ) -> Float[Array, "tokens hidden"]:
        """Execute the selected backend (see :func:`moe_decode_fused_w8a8`)."""
        if platform is not None:
            cfg = MoeDecodeFusedW8A8Config(platform=platform, backend=Backend.ANY if platform == "xla" else cfg.backend)
        impl = self.get_impl(cfg)
        return impl(
            x,
            expert_ids,
            combine_weights,
            gate_codes,
            gate_scales,
            up_codes,
            up_scales,
            down_codes,
            down_scales,
            control,
            shared_gate_up_codes,
            shared_gate_up_scales,
            shared_down_codes,
            shared_down_scales,
            activation=activation,
            act_params=tuple(float(v) for v in act_params),
            shared_activation=shared_activation,
            shared_act_params=tuple(float(v) for v in shared_act_params),
        )

    def heuristic_cfg(self, inv: Invocation) -> MoeDecodeFusedW8A8Config:
        """Return the default config (automatic platform; the ring depth comes from the VMEM capacity)."""
        return MoeDecodeFusedW8A8Config(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation):
        """Return the candidate configs (the heuristic only; nothing is autotuned)."""
        return [self.heuristic_cfg(inv)]


_moe_decode_fused_w8a8_executor: Executor = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=False, cache_miss_fallback="heuristics", validate_backward=False),
        tuner=Tuner(warmup=1, iters=1),
        persistent=PersistentCache("moe-decode-fused-w8a8", cfg_type=MoeDecodeFusedW8A8Config),
    )
)


def moe_decode_fused_w8a8_unsupported_reason(
    x,
    expert_ids,
    combine_weights,
    gate_codes,
    gate_scales,
    up_codes,
    up_scales,
    down_codes,
    down_scales,
    control,
    shared_gate_up_codes=None,
    shared_gate_up_scales=None,
    shared_down_codes=None,
    shared_down_scales=None,
) -> str | None:
    """Why the Pallas TPU kernel cannot take these operands (``None``: it can). Arguments as the op's."""
    from ejkernel.kernels._pallas.tpu.moe_decode_fused_w8a8._pallas_impl import unsupported_reason

    return unsupported_reason(
        x,
        expert_ids,
        combine_weights,
        gate_codes,
        gate_scales,
        up_codes,
        up_scales,
        down_codes,
        down_scales,
        control,
        shared_gate_up_codes,
        shared_gate_up_scales,
        shared_down_codes,
        shared_down_scales,
    )


def moe_decode_fused_w8a8(
    x: Float[Array, "tokens hidden"],
    expert_ids: Int[Array, "tokens k"],
    combine_weights: Float[Array, "tokens k"],
    gate_codes: Int[Array, "experts hidden inter"],
    gate_scales: Float[Array, "experts 1 inter"],
    up_codes: Int[Array, "experts hidden inter"],
    up_scales: Float[Array, "experts 1 inter"],
    down_codes: Int[Array, "experts inter hidden"],
    down_scales: Float[Array, "experts 1 hidden"],
    control: Int[Array, "3"],
    shared_gate_up_codes: Int[Array, "hidden two_sinter"] | None = None,
    shared_gate_up_scales: Float[Array, "1 two_sinter"] | None = None,
    shared_down_codes: Int[Array, "sinter hidden"] | None = None,
    shared_down_scales: Float[Array, "1 hidden"] | None = None,
    /,
    *,
    activation: str = "silu",
    act_params: tuple[float, ...] = (),
    shared_activation: str = "silu",
    shared_act_params: tuple[float, ...] = (),
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: MoeDecodeFusedW8A8Config | None = None,
) -> Float[Array, "tokens hidden"]:
    """Fused decode MoE segment of one expert shard, through the ejkernel executor (see the module docstring).

    Args:
        x: Token rows ``[T, H]`` (floating).
        expert_ids: Global expert id per (token, slot) ``[T, k]``.
        combine_weights: Combine weight per (token, slot) ``[T, k]``.
        gate_codes: This shard's int8 gate codes ``[E, H, M]``.
        gate_scales: Per-channel gate scales ``[E, 1, M]``.
        up_codes: int8 up codes ``[E, H, M]``.
        up_scales: Per-channel up scales ``[E, 1, M]``.
        down_codes: int8 down codes ``[E, M, H]``.
        down_scales: Per-channel down scales ``[E, 1, H]``.
        control: int32 ``[3]``: global id of local expert 0, shared-expert row start, shared-expert row end.
        shared_gate_up_codes: Optional shared expert int8 ``[H, 2I]`` (``[gate | up]``).
        shared_gate_up_scales: Its scales ``[1, 2I]``.
        shared_down_codes: Optional shared expert int8 ``[I, H]``.
        shared_down_scales: Its scales ``[1, H]``.
        activation: Routed ``resolve_mlp_combine`` name.
        act_params: Its static parameters.
        shared_activation: Shared-expert ``resolve_mlp_combine`` name.
        shared_act_params: Its static parameters.
        platform: Optional platform override (``"xla"`` / ``"pallas"``).
        cfg: Optional :class:`MoeDecodeFusedW8A8Config`.

    Returns:
        float32 ``[T, H]``.
    """
    return _moe_decode_fused_w8a8_executor(
        MoeDecodeFusedW8A8(),
        x,
        expert_ids,
        combine_weights,
        gate_codes,
        gate_scales,
        up_codes,
        up_scales,
        down_codes,
        down_scales,
        control,
        shared_gate_up_codes,
        shared_gate_up_scales,
        shared_down_codes,
        shared_down_scales,
        activation=activation,
        act_params=tuple(float(v) for v in act_params),
        shared_activation=shared_activation,
        shared_act_params=tuple(float(v) for v in shared_act_params),
        platform=platform,
        _cfg=cfg,
    )


__all__ = ("MoeDecodeFusedW8A8", "moe_decode_fused_w8a8", "moe_decode_fused_w8a8_unsupported_reason")
