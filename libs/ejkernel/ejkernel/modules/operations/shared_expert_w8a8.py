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


"""W8A8 gated MLP on a few decode rows (``shared_expert_w8a8``).

``down(combine(gate_up(rows)))`` with int8 weight codes, per-channel scales and per-row int8 activations: the gate/up
projection returns the rows' dtype, the two-branch ``combine`` (``resolve_mlp_combine``) runs in float32 and the down
projection returns float32 ``[r, H]``. This is the folded shared expert of EasyDeL's decode MoE dispatch. On TPU two
Pallas stages (gate/up + combine, down), each over a parallel grid of output-channel tiles, replace the two XLA int8
ragged dots and their pads; the XLA implementation (two one-group ``grouped_matmul_channelwise`` calls) is the
reference and the fallback. Forward only (inference).
"""

from __future__ import annotations

from typing import Literal

from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import AutotunePolicy, ConfigCache, ConfigSelectorChain, Executor, Invocation, Kernel, Tuner
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import SharedExpertW8A8Config


class SharedExpertW8A8(Kernel[SharedExpertW8A8Config, Array]):
    """W8A8 gated MLP on a few rows (Pallas TPU; XLA reference)."""

    version = "1"

    def __init__(self):
        """Initialize the operation under ``op_id="shared_expert_w8a8"``."""
        super().__init__(op_id="shared_expert_w8a8")

    def get_impl(self, cfg: SharedExpertW8A8Config):
        """Resolve the registry implementation for the config's platform/backend.

        Args:
            cfg: Operation config.

        Returns:
            The registered ``shared_expert_w8a8`` callable.
        """
        platform = detect_platform(self.op_id, cfg.platform)
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(
        self,
        rows: Float[Array, "rows hidden"],
        gate_up_codes: Int[Array, "hidden two_inter"],
        gate_up_scales: Float[Array, "1 two_inter"],
        down_codes: Int[Array, "inter hidden"],
        down_scales: Float[Array, "1 hidden"],
        *,
        activation: str = "silu",
        act_params: tuple[float, ...] = (),
        platform: Literal["pallas", "xla", "auto"] | None = None,
        cfg: SharedExpertW8A8Config,
        **_,
    ) -> Float[Array, "rows hidden"]:
        """Execute the selected backend (see :func:`shared_expert_w8a8`)."""
        if platform is not None:
            cfg = SharedExpertW8A8Config(platform=platform, backend=Backend.ANY if platform == "xla" else cfg.backend)
        impl = self.get_impl(cfg)
        return impl(
            rows,
            gate_up_codes,
            gate_up_scales,
            down_codes,
            down_scales,
            activation=activation,
            act_params=tuple(float(v) for v in act_params),
        )

    def heuristic_cfg(self, inv: Invocation) -> SharedExpertW8A8Config:
        """Return the default config (automatic platform; tiles come from the VMEM capacity)."""
        return SharedExpertW8A8Config(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation):
        """Return the candidate configs (the heuristic only; nothing is autotuned)."""
        return [self.heuristic_cfg(inv)]


_shared_expert_w8a8_executor: Executor = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=False, cache_miss_fallback="heuristics", validate_backward=False),
        tuner=Tuner(warmup=1, iters=1),
        persistent=PersistentCache("shared-expert-w8a8", cfg_type=SharedExpertW8A8Config),
    )
)


def shared_expert_w8a8_unsupported_reason(rows, gate_up_codes, gate_up_scales, down_codes, down_scales) -> str | None:
    """Why the Pallas TPU kernel cannot take these operands (``None``: it can)."""
    from ejkernel.kernels._pallas.tpu.shared_expert_w8a8._pallas_impl import unsupported_reason

    return unsupported_reason(rows, gate_up_codes, gate_up_scales, down_codes, down_scales)


def shared_expert_w8a8(
    rows: Float[Array, "rows hidden"],
    gate_up_codes: Int[Array, "hidden two_inter"],
    gate_up_scales: Float[Array, "1 two_inter"],
    down_codes: Int[Array, "inter hidden"],
    down_scales: Float[Array, "1 hidden"],
    /,
    *,
    activation: str = "silu",
    act_params: tuple[float, ...] = (),
    platform: Literal["pallas", "xla", "auto"] | None = None,
    cfg: SharedExpertW8A8Config | None = None,
) -> Float[Array, "rows hidden"]:
    """W8A8 gated MLP on a few rows, through the ejkernel executor (see the module docstring).

    Args:
        rows: Floating rows ``[r, H]``.
        gate_up_codes: int8 ``[H, 2I]`` (``[gate | up]``).
        gate_up_scales: Per-channel scales ``[1, 2I]``.
        down_codes: int8 ``[I, H]``.
        down_scales: Per-channel scales ``[1, H]``.
        activation: ``resolve_mlp_combine`` name (e.g. ``"silu"``, ``"clamped_swiglu"``).
        act_params: Its static parameters.
        platform: Optional platform override (``"xla"`` / ``"pallas"``).
        cfg: Optional :class:`SharedExpertW8A8Config`.

    Returns:
        float32 ``[r, H]``.
    """
    return _shared_expert_w8a8_executor(
        SharedExpertW8A8(),
        rows,
        gate_up_codes,
        gate_up_scales,
        down_codes,
        down_scales,
        activation=activation,
        act_params=tuple(float(v) for v in act_params),
        platform=platform,
        _cfg=cfg,
    )


__all__ = ("SharedExpertW8A8", "shared_expert_w8a8", "shared_expert_w8a8_unsupported_reason")
