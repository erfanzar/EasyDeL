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

"""The seam between externally quantized checkpoints and EasyDeL's runtime.

Every pre-quantized checkpoint format (fp8, compressed-tensors, AWQ/GPTQ,
MXFP4, NVFP4, ...) is reduced to exactly one representation before it reaches
any layer: :class:`CanonicalQuantizedWeight`, described by a
:class:`QuantSpec`. That representation is *already* what
:class:`~easydel.layers.linears.ParallelLinearQuantized` stores
(``quant_kernel`` / ``quant_scales`` / ``quant_biases``), so landing on it
means a new checkpoint format costs one adapter and zero changes to layers,
forward paths, kernels, or the ~80 model families.

Three concepts, deliberately separated:

* :class:`SourceFormat` — what is *on disk*, parsed from the checkpoint's
  ``quantization_config``. Descriptive; never chosen by us.
* :class:`QuantSpec` — what we *run*. A policy decision. It is legal (and for
  block-scaled fp8, necessary) for the target to differ from the source; the
  adapter performs the dequantize/requantize that bridges them.
* :class:`ActivationPolicy` — whether activations are quantized at matmul
  time. This is a property of the *kernel invocation*, not of the weight,
  which is why it is metadata rather than an array.

The ``(mode, group_size, bits, needs_biases)`` table is **not** duplicated
here: :class:`QuantSpec` validates itself through
:func:`~easydel.layers.quantization.resolve_ejkernel_quant_params`, the single
source of truth already used by the self-quantization path.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field
from enum import StrEnum

from .._configs import QuantizationConfig, QuantizationType, resolve_ejkernel_quant_params

if tp.TYPE_CHECKING:
    from jax import Array


class ActivationQuantKind(StrEnum):
    """How activations are treated at matmul time.

    Attributes:
        NONE: Activations stay in the compute dtype; the weight is
            dequantized into the matmul (``W*A16``).
        DYNAMIC: Activations are quantized per token at run time, with scales
            computed on the fly from the activation itself (``W*A8``).
        STATIC: Activations are quantized with a calibrated per-tensor scale
            shipped in the checkpoint.
    """

    NONE = "none"
    DYNAMIC = "dynamic"
    STATIC = "static"


@dataclass(frozen=True, slots=True)
class ActivationPolicy:
    """Activation-quantization policy attached to a quantized weight.

    Static metadata only. The calibrated scale for
    :attr:`ActivationQuantKind.STATIC` is an array and therefore lives on
    :attr:`CanonicalQuantizedWeight.activation_scale`, not here, so this class
    stays hashable and usable as a JIT static argument.

    Attributes:
        kind: Which activation treatment applies.
        dtype: Requested quantized activation dtype (``"int8"`` or
            ``"fp8_e4m3"``). ``None`` means "let the kernel choose from
            hardware capability", which is the normal case — the choice
            depends on whether the target reports fp8/int8 matmul support.
    """

    kind: ActivationQuantKind = ActivationQuantKind.NONE
    dtype: tp.Literal["int8", "fp8_e4m3"] | None = None

    def __post_init__(self) -> None:
        """Reject dtype requests on policies that do not quantize activations.

        Raises:
            ValueError: If ``dtype`` is set while ``kind`` is
                :attr:`ActivationQuantKind.NONE`.
        """
        if self.kind is ActivationQuantKind.NONE and self.dtype is not None:
            raise ValueError(f"ActivationPolicy(kind=none) cannot carry dtype={self.dtype!r}.")

    @property
    def quantizes(self) -> bool:
        """Whether this policy quantizes activations at all."""
        return self.kind is not ActivationQuantKind.NONE

    @classmethod
    def none(cls) -> ActivationPolicy:
        """Build the weight-only (``W*A16``) policy."""
        return cls(kind=ActivationQuantKind.NONE)

    @classmethod
    def dynamic(cls, dtype: tp.Literal["int8", "fp8_e4m3"] | None = None) -> ActivationPolicy:
        """Build a dynamic per-token policy.

        Args:
            dtype: Quantized activation dtype, or ``None`` to defer the choice
                to the kernel's hardware probe.

        Returns:
            The corresponding :class:`ActivationPolicy`.
        """
        return cls(kind=ActivationQuantKind.DYNAMIC, dtype=dtype)

    @classmethod
    def static(cls, dtype: tp.Literal["int8", "fp8_e4m3"] | None = None) -> ActivationPolicy:
        """Build a static (calibrated per-tensor) policy.

        Args:
            dtype: Quantized activation dtype, or ``None`` to defer.

        Returns:
            The corresponding :class:`ActivationPolicy`.
        """
        return cls(kind=ActivationQuantKind.STATIC, dtype=dtype)


@dataclass(frozen=True, slots=True)
class QuantSpec:
    """The runtime quantization target — what the kernels will actually see.

    This is a *policy* decision, not a description of the checkpoint. A
    DeepSeek-style ``128x128`` block-scaled fp8 checkpoint has no matching
    ejkernel mode, so its target is something we can execute (per-channel
    affine int8, group-32 mxfp8, ...) and the adapter dequantizes/requantizes
    to get there. Keeping source and target distinct is what stops every
    format from growing its own kernel path.

    Attributes:
        dtype: Target quantization type. Must be one of the ejkernel-backed
            types; ``BINARY``/``TERNARY``/``TURBOQUANT`` have their own paths
            and are rejected here.
        group_size: Elements sharing one scale along the contracting axis.
        bits: Bit width of a quantized weight element.
        expert_dim: ``True`` when the weight carries a leading expert axis
            (MoE stacked experts: ``[E, K, N]`` with ``[E, blocks, 1, N]``
            scales) rather than a plain ``[K, N]``.
        activation: Activation-quantization policy for matmuls using this
            weight.
    """

    dtype: QuantizationType
    group_size: int
    bits: int
    expert_dim: bool = False
    activation: ActivationPolicy = field(default_factory=ActivationPolicy.none)

    def __post_init__(self) -> None:
        """Normalize ``dtype`` and validate the scheme combinatorics.

        Validation is delegated to
        :func:`resolve_ejkernel_quant_params` so the per-mode
        ``(group_size, bits)`` constraints live in exactly one place.

        Raises:
            ValueError: If the ``(dtype, group_size, bits)`` combination is
                not one the ejkernel quantizer accepts.
        """
        object.__setattr__(self, "dtype", QuantizationType(self.dtype))
        object.__setattr__(self, "group_size", int(self.group_size))
        object.__setattr__(self, "bits", int(self.bits))
        resolve_ejkernel_quant_params(self.to_quantization_config())

    @property
    def resolved(self) -> tuple[str, int, int, bool]:
        """Return the ejkernel ``(mode, group_size, bits, needs_biases)`` tuple."""
        return resolve_ejkernel_quant_params(self.to_quantization_config())

    @property
    def mode(self) -> str:
        """ejkernel mode string (``"affine"``, ``"nf4"``, ``"mxfp4"``, ...)."""
        return self.resolved[0]

    @property
    def needs_biases(self) -> bool:
        """Whether the scheme stores per-group zero-points alongside scales."""
        return self.resolved[3]

    def to_quantization_config(self) -> QuantizationConfig:
        """Project onto the config object the quantized layers already accept.

        Returns:
            A :class:`QuantizationConfig` carrying this spec's dtype, group
            size and bit width. Deliberately the same type the
            self-quantization path uses, so quantized layers need no new
            constructor argument to consume a checkpoint-derived spec.
        """
        return QuantizationConfig(dtype=self.dtype, group_size=self.group_size, bits=self.bits)


@dataclass(frozen=True, slots=True)
class SourceFormat:
    """What the checkpoint declares on disk, for one quantized tensor group.

    Purely descriptive. Adapters read this to know how to decode; the scheme
    resolver reads it to pick a :class:`QuantSpec` target.

    Attributes:
        quant_method: The checkpoint's ``quant_method`` (``"fp8"``,
            ``"compressed-tensors"``, ``"auto_awq"``, ...).
        weight_dtype: Serialized element type of the packed weight, as
            declared by the checkpoint (e.g. ``"float8_e4m3fn"``, ``"int4"``,
            ``"float4_e2m1fn"``). ``None`` when the format implies it.
        block_shape: Block dimensions for block-scaled schemes (e.g.
            ``(128, 128)`` for DeepSeek fp8). ``None`` for per-tensor,
            per-channel or group-scaled schemes.
        group_size: Group size for group-scaled schemes (AWQ/GPTQ/MX/NV).
            ``None`` when the scheme is not group-scaled.
        symmetric: Whether the weight quantization is symmetric (no
            zero-point).
        has_zero_point: Whether the checkpoint ships explicit zero-points.
        has_global_scale: Whether a second, per-tensor scale level exists
            (NVFP4's ``weight_scale_2``).
        activation: Activation scheme the checkpoint declares.
        raw: Format-specific leftovers from the parsed config, so adapters can
            reach fields this dataclass does not name without forcing a
            subclass per format. Excluded from hashing (a mapping is not
            hashable) but included in equality, so instances stay usable as
            dict keys and JIT static arguments.
    """

    quant_method: str
    weight_dtype: str | None = None
    block_shape: tuple[int, ...] | None = None
    group_size: int | None = None
    symmetric: bool = True
    has_zero_point: bool = False
    has_global_scale: bool = False
    activation: ActivationPolicy = field(default_factory=ActivationPolicy.none)
    raw: tp.Mapping[str, tp.Any] = field(default_factory=dict, hash=False)

    @property
    def is_block_scaled(self) -> bool:
        """Whether the source uses 2-D block scales rather than channel/group scales."""
        return self.block_shape is not None


@dataclass(frozen=True, slots=True)
class CanonicalQuantizedWeight:
    """One quantized weight in the single representation the runtime accepts.

    Produced by an adapter during checkpoint conversion and immediately split
    into the layer's ``quant_kernel`` / ``quant_scales`` / ``quant_biases``
    parameters. It exists only on the host during loading and never crosses a
    JIT boundary, so it is a plain dataclass rather than a pytree.

    Attributes:
        quant_kernel: Packed quantized weight in ejkernel layout.
        quant_scales: Per-group scales.
        quant_biases: Per-group zero-points/biases; ``None`` unless
            :attr:`QuantSpec.needs_biases`.
        spec: The target this weight was produced for.
        output_scale: Optional per-tensor scalar applied *after* the matmul.
            A per-tensor scalar commutes with the matmul
            (``x @ (w·s_block·s_global) == s_global·(x @ (w·s_block))``), so
            NVFP4's fp32 global scale is carried here exactly rather than
            being folded lossily into an e4m3 block code.
        activation_scale: Calibrated per-tensor activation scale; set only
            when :attr:`QuantSpec.activation` is
            :attr:`ActivationQuantKind.STATIC`.
    """

    quant_kernel: Array
    quant_scales: Array
    quant_biases: Array | None
    spec: QuantSpec
    output_scale: Array | None = None
    activation_scale: Array | None = None

    def __post_init__(self) -> None:
        """Check the payload against its own spec.

        Catches adapter bugs at conversion time, where the offending
        checkpoint key is still in scope, rather than as a shape error deep
        inside a kernel.

        Raises:
            ValueError: If biases are present/absent contrary to the spec, or
                if a static activation policy is missing its scale (or a
                non-static policy carries one).
        """
        if self.spec.needs_biases and self.quant_biases is None:
            raise ValueError(f"spec.dtype={self.spec.dtype} requires quant_biases, got None.")
        if not self.spec.needs_biases and self.quant_biases is not None:
            raise ValueError(f"spec.dtype={self.spec.dtype} does not use quant_biases, but one was provided.")

        is_static = self.spec.activation.kind is ActivationQuantKind.STATIC
        if is_static and self.activation_scale is None:
            raise ValueError("A static activation policy requires activation_scale.")
        if not is_static and self.activation_scale is not None:
            raise ValueError(
                f"activation_scale is only valid for a static activation policy, "
                f"got kind={self.spec.activation.kind}."
            )

    @classmethod
    def from_dense(
        cls,
        dense: Array,
        *,
        spec: QuantSpec,
        output_scale: Array | None = None,
        activation_scale: Array | None = None,
    ) -> CanonicalQuantizedWeight:
        """Pack a dense float weight into the canonical representation.

        This is the second half of every adapter and is deliberately shared:
        an adapter's only job is to *decode* its checkpoint into a dense
        array in EasyDeL's ``[in_features, out_features]`` orientation (with
        an optional leading expert axis). Packing, grouping and scale
        computation then run through ejkernel's ``prepack_quantized_weights``
        — the same call the self-quantization path uses — which guarantees the
        result is byte-identical in layout to a natively quantized layer.

        Args:
            dense: Dequantized weight, ``[in, out]`` or ``[experts, in, out]``.
            spec: Target quantization spec.
            output_scale: Optional per-tensor scalar applied after the matmul.
            activation_scale: Calibrated activation scale for static policies.

        Returns:
            The packed canonical weight.
        """
        from dataclasses import replace

        from easydel.layers.linears._linear_quantized import (
            _effective_ejkernel_group_size,
            prepack_quantized_weights_jit,
        )

        mode, group_size, bits, needs_biases = spec.resolved

        # Reuse the layer's own clamp so a checkpoint-derived weight is packed
        # with exactly the group size a natively quantized layer would choose,
        # then record that size on the spec. If the two disagreed, the layer
        # would dequantize with the wrong grouping and silently return noise.
        effective_group_size = _effective_ejkernel_group_size(mode, group_size, tuple(dense.shape))
        if effective_group_size != group_size:
            spec = replace(spec, group_size=effective_group_size)

        packed = prepack_quantized_weights_jit(
            dense,
            group_size=effective_group_size,
            bits=bits,
            mode=mode,
        )
        if needs_biases:
            quant_kernel, quant_scales, quant_biases = packed
        else:
            quant_kernel, quant_scales = packed
            quant_biases = None

        return cls(
            quant_kernel=quant_kernel,
            quant_scales=quant_scales,
            quant_biases=quant_biases,
            spec=spec,
            output_scale=output_scale,
            activation_scale=activation_scale,
        )

    def arrays(self) -> dict[str, Array]:
        """Return the non-``None`` array payload keyed by parameter name.

        The keys are the parameter names a quantized layer exposes, which is
        what the reform-rule generator turns into split targets.

        Returns:
            Mapping of parameter name to array, omitting absent optionals.
        """
        payload: dict[str, Array] = {
            "quant_kernel": self.quant_kernel,
            "quant_scales": self.quant_scales,
        }
        if self.quant_biases is not None:
            payload["quant_biases"] = self.quant_biases
        if self.output_scale is not None:
            payload["output_scale"] = self.output_scale
        if self.activation_scale is not None:
            payload["activation_scale"] = self.activation_scale
        return payload
