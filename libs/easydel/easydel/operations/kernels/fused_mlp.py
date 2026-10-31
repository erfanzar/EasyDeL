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

"""Fused gated-MLP operation adapter.

Wires the eJKernel ``fused_mlp`` public operation (``down(act(gate(x)) *
up(x))``) into EasyDeL's operation registry. The adapter carries EasyDeL
runtime concerns — leading-dim flattening, runtime dtype, per-operation
config plumbing — and delegates every numerical decision to eJKernel:

* **dense** bf16 weights run the plain composition under JAX autodiff
  (bit-identical to the unfused path, measured);
* **channelwise int8/int4** codes run the fused-upcast / int-MXU composition
  with a measured-correct frozen-weight ``custom_vjp`` for training;
* **packed int4** weights (``pack_int4_adjacent`` layout) run the
  single-dispatch Pallas W4A4 decode kernel on TPUs whose MXU supports
  int4xint4, with a tile-exact XLA reference elsewhere.

Platform/backend selection happens inside eJKernel's executor (kernel
registry + capability probes), so every EasyDeL backend forward is a thin
delegate to :meth:`forward_native`.

This operation is cache-free and shape-preserving: it is not an attention
mechanism and returns a plain array rather than an ``AttentionOutput``.
"""

import jax
from ejkernel.modules import fused_mlp as ejkernel_fused_mlp
from jaxtyping import Array, Float

from easydel.utils.helpers import check_bool_flag

from .._operation_impl import OperationImpl, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
    RequirementsBuilder,
)


@OperationRegistry.register
class FusedMlpOp(OperationImpl):
    """Fused gated-MLP block ``down(act(gate(x)) * up(x))``.

    Registered under the name ``"fused_mlp"``. Supports dense, channelwise
    int8/int4, and bit-packed W4A4 weight formats, separate or fused
    (concat / TP-interleaved) gate_up layouts, and the shared activation
    table (silu, gelu, gelu_tanh, relu, sigmoid).

    Example:
        >>> from easydel.operations import OperationMetadata, OperationRegistry
        >>> metadata = OperationMetadata(runtime_dtype=jnp.bfloat16)
        >>> mlp_op = OperationRegistry.create("fused_mlp", metadata)
        >>> out = mlp_op(hidden_states, w_gate, w_up, w_down)
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Returns the registered name of this operation.

        Returns:
            The single name ``"fused_mlp"``.
        """
        return "fused_mlp"

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Returns requirements for FusedMlpOp.

        The MLP block is stateless and cache-free: it needs no attention
        metadata and is compatible with (but never reads) any cache type.

        Args:
            mode: Execution mode (unused for this operation).

        Returns:
            OperationRequirements: The declared requirements.
        """
        del mode
        return (
            RequirementsBuilder("fused_mlp")
            .require_metadata(MetadataField.NONE)
            .support_cache(CacheType.any())
            .requires_cache(False)
            .build()
        )

    @jax.named_scope("easydel-fused-mlp-native")
    def forward_native(
        self,
        hidden_states: Float[Array, "... hidden"],
        w_gate: Array | None = None,
        w_up: Array | None = None,
        w_down: Array | None = None,
        *,
        gate_up: Array | None = None,
        gate_up_layout: str = "concat",
        gate_up_segments: int = 1,
        gate_scale: Array | None = None,
        up_scale: Array | None = None,
        down_scale: Array | None = None,
        activation: str = "silu",
        quantize_activations: bool = False,
        packed_weights: tuple[Array, Array, Array] | None = None,
        **kwargs,
    ) -> Float[Array, "... hidden"]:
        """Compute the fused MLP block via the eJKernel public operation.

        Args:
            hidden_states: Activations ``[..., hidden]``; leading dims are
                flattened to a token axis for the kernel and restored on the
                output.
            w_gate: Gate projection ``[hidden, intermediate]`` — floating
                (dense) or integer codes (channelwise). Mutually exclusive
                with ``gate_up``.
            w_up: Up projection ``[hidden, intermediate]``.
            w_down: Down projection ``[intermediate, hidden]``.
            gate_up: Fused gate/up weight ``[hidden, 2*intermediate]``
                instead of separate arrays.
            gate_up_layout: ``"concat"`` or ``"interleaved"`` (EasyDeL's
                TP-portable fused layout).
            gate_up_segments: TP segment count for the interleaved layout.
            gate_scale: Per-output-channel scale for integer gate weights;
                with a fused ``gate_up`` scale, pass it here and leave
                ``up_scale`` ``None``.
            up_scale: Per-output-channel scale for integer up weights.
            down_scale: Per-output-channel scale for integer down weights.
            activation: Gate-branch activation name.
            quantize_activations: Integer formats: engage the int-MXU dot at
                prefill sizes and for the backward transposed products.
            packed_weights: ``(gate_packed, up_packed, down_packed)`` uint8
                arrays in ``pack_int4_adjacent`` layout, enabling the fused
                W4A4 decode kernel at decode token counts on capable TPUs.
            **kwargs: Forwarded to :func:`ejkernel.modules.fused_mlp`
                (e.g. ``prefill_threshold``, ``packed_tile_i``,
                ``decode_threshold``, ``platform`` overrides —
                ``platform="xla"`` disables the Pallas kernel and runs the
                vanilla-XLA composition). Pass ``mesh`` and ``in_specs``
                (``(x_spec, gate_up_spec, down_spec)``, describing the
                arrays' ACTUAL shardings) to enable the sharding-aware
                dense bf16 Pallas path: rows/tp layouts run the fused
                kernel (+psum), K-sharded fsdp weights and decode row
                counts fall back to the XLA composition so the op is never
                slower than GSPMD.

        Returns:
            ``[..., hidden]`` in the runtime dtype.

        Note:
            ``FORCE_NATIVE_RUNTIME=1`` forces ``platform="xla"`` here, so the
            flag keeps its repo-wide meaning (pure XLA reference path) even
            though every backend forward of this op delegates to
            ``forward_native``.
        """
        runtime_dtype = self.metadata.runtime_dtype
        x = hidden_states.astype(runtime_dtype)
        leading = x.shape[:-1]
        x2d = x.reshape(-1, x.shape[-1])

        cfg = self.metadata.get_operation_config("fused_mlp")
        if cfg is not None:
            kwargs.setdefault("prefill_threshold", cfg.prefill_threshold)
            kwargs.setdefault("packed_tile_i", cfg.tile_i)
            kwargs.setdefault("platform", cfg.platform)
        if check_bool_flag("FORCE_NATIVE_RUNTIME", False):
            kwargs["platform"] = "xla"

        out = ejkernel_fused_mlp(
            x2d,
            w_gate,
            w_up,
            w_down,
            gate_up=gate_up,
            gate_up_layout=gate_up_layout,
            gate_up_segments=gate_up_segments,
            gate_scale=gate_scale,
            up_scale=up_scale,
            down_scale=down_scale,
            activation=activation,
            quantize_activations=quantize_activations,
            packed_weights=packed_weights,
            **kwargs,
        )
        return out.reshape(*leading, out.shape[-1]).astype(runtime_dtype)

    def forward_tpu(self, *args, **kwargs) -> Float[Array, "... hidden"]:
        """TPU forward — eJKernel's executor picks Pallas/XLA internally."""
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> Float[Array, "... hidden"]:
        """GPU forward — delegates to the eJKernel composition."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> Float[Array, "... hidden"]:
        """CPU forward — delegates to the eJKernel composition."""
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> Float[Array, "... hidden"]:
        """CUDA forward — delegates to the eJKernel composition."""
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> Float[Array, "... hidden"]:
        """ROCm forward — delegates to the eJKernel composition."""
        return self.forward_native(*args, **kwargs)
