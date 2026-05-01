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

"""Quantization configuration types and settings for EasyDeL.

This module defines the configuration classes and enumerations used to control
weight quantization behavior in EasyDeL models. It provides a flexible system
for specifying quantization types, block sizes, and layer selection patterns.

The module supports multiple quantization formats including:
    - NF4 (4-bit NormalFloat): QLoRA-style quantization with normal distribution
    - AFFINE: Scale+bias quantization with configurable bit-width (ejkernel)
    - INT8: 8-bit integer quantization
    - MXFP4/MXFP8: Microscaling floating-point formats
    - NVFP8: NVIDIA's FP8 format (E4M3)
    - Binary/Ternary: Extreme quantization for efficiency

Example:
    >>> from easydel.layers.quantization import QuantizationConfig, QuantizationType
    >>>
    >>> # Configure NF4 quantization (4-bit)
    >>> config = QuantizationConfig(
    ...     dtype=QuantizationType.NF4,
    ...     group_size=64,
    ...     simulate=False
    ... )
    >>>
    >>> # Configure INT8 quantization
    >>> config = QuantizationConfig(dtype=QuantizationType.INT8)

See Also:
    - easydel.layers.components.quants._quants: Quantization implementations
    - easydel.layers.components.quants._straight_through: STE functions
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from easydel.utils.compiling_utils import hash_fn

DEFAULT_QUANTIZATION_PATTERN = r"^(?!.*(?:embedding|norm|lm_head)).*$"
"""Default regex pattern for selecting layers to quantize.

This pattern excludes common layers that should remain in full precision:
    - embedding: Input embeddings (require full precision for vocabulary)
    - norm: Normalization layers (sensitive to quantization)
    - lm_head: Language model head (output layer)

All other layers matching this pattern will be considered for quantization.
"""


class QuantizationType(enum.StrEnum):
    """Enumeration of supported quantization data types.

    This enum defines all quantization formats available in EasyDeL. Each format
    represents a different precision-memory tradeoff and may have different
    hardware support characteristics.

    Attributes:
        MXFP8: Microscaling FP8 format with shared exponent (E8M0 + E4M3 codes).
        MXFP4: Microscaling FP4 format (E2M1). Aggressive 4-bit float compression.
        NVFP8: NVIDIA FP8 format (E4M3). Optimized for NVIDIA hardware inference.
        NF4: 4-bit NormalFloat. QLoRA-style quantization with block-wise scaling.
            Best balance of quality and compression for LLM weights.
        AFFINE: Linear scale+bias quantization with configurable bit-width (2-8).
            This maps to ejkernel's affine mode and supports group_size + bits.
        INT8: 8-bit integer quantization. Widely supported, good inference speed.
            Alias for affine with bits=8 by default.
        TERNARY: 3-level quantization {-1, 0, 1}. Extreme compression with
            threshold-based discretization.
        BINARY: 2-level quantization {-1, 1}. Maximum compression using sign only.

    Example:
        >>> from easydel.layers.quantization import QuantizationType
        >>>
        >>> # Use NF4 for memory-efficient fine-tuning
        >>> quant_type = QuantizationType.NF4
        >>>
        >>> # Use INT8 for inference
        >>> quant_type = QuantizationType.INT8
        >>>
        >>> # Convert from string
        >>> quant_type = QuantizationType("nf4")

    Note:
        - NF4 and INT8 are the most commonly used formats for LLM deployment
        - Binary and Ternary provide extreme compression but with quality loss
        - MXFP formats are designed for hardware with microscaling support
    """

    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"
    NVFP8 = "nvfp8"
    NVFP4 = "nvfp4"
    NF4 = "nf4"
    AFFINE = "affine"
    INT8 = "int8"
    TERNARY = "ternary"
    BINARY = "binary"
    TURBOQUANT = "turboquant"


@dataclass
class QuantizationConfig:
    """Configuration for model weight quantization behavior.

    This dataclass controls how weights are quantized during training and inference.
    It provides fine-grained control over quantization type, precision, and which
    layers are affected through regex pattern matching.

    Attributes:
        dtype: The quantization type to use. Can be a QuantizationType enum value
            or its string representation (e.g., "nf4", "int8"). Defaults to NF4.
        runtime_dtype: Optional alternative dtype for runtime computation. If set,
            weights are stored in `dtype` but computed in `runtime_dtype`. Useful
            for mixed-precision inference. Defaults to None (use dtype).
        group_size: Group size for quantization schemes. Used for NF4 and
            ejkernel modes (affine, mxfp, nvfp). Larger groups improve throughput
            but may reduce accuracy. Defaults depend on dtype (nf4/affine=64,
            mxfp4/mxfp8=32, nvfp8=16).
        bits: Bit-width for ejkernel affine quantization (2-8). If not provided,
            defaults are chosen per mode (affine: 4, int8: 8).
        simulate: If True, uses straight-through estimation without actual bit
            packing. The quantization error is simulated but weights remain in
            original dtype. Useful for quantization-aware training (QAT) where
            gradients need to flow through. Defaults to False.
        jax_native: If True and the quantization type has a native JAX dtype
            (e.g., MXFP4/MXFP8/NVFP8), quantization uses `jnp.astype` instead
            of ejkernel. This applies even in simulation/QAT paths.
        pattern: Regex pattern for selecting which layers to quantize. Layers
            with names matching this pattern will be quantized. The default
            pattern excludes embedding, normalization, and output head layers.

    Example:
        >>> from easydel.layers.quantization import QuantizationConfig, QuantizationType
        >>>
        >>> # NF4 quantization with 64-element groups (recommended for LLMs)
        >>> config = QuantizationConfig(
        ...     dtype=QuantizationType.NF4,
        ...     group_size=64
        ... )
        >>>
        >>> # INT8 quantization for inference
        >>> config = QuantizationConfig(dtype=QuantizationType.INT8)
        >>>
        >>> # Affine quantization with explicit group_size and bits (ejkernel)
        >>> config = QuantizationConfig(
        ...     dtype=QuantizationType.AFFINE,
        ...     group_size=64,
        ...     bits=4
        ... )
        >>>
        >>> # Binary quantization (extreme compression)
        >>> config = QuantizationConfig(dtype=QuantizationType.BINARY)
        >>>
        >>> # Simulation mode for QAT (no actual bit packing)
        >>> config = QuantizationConfig(
        ...     dtype=QuantizationType.NF4,
        ...     simulate=True
        ... )
        >>>
        >>> # Custom layer pattern (only quantize attention layers)
        >>> config = QuantizationConfig(
        ...     dtype=QuantizationType.INT8,
        ...     pattern=r".*attention.*"
        ... )

    Note:
        The config is hashable and can be used as a dictionary key or in sets.
        String dtype values are automatically converted to QuantizationType
        in __post_init__.

    See Also:
        - EasyQuantizer: High-level API for applying quantization to models
        - quantize: Function to quantize individual arrays
        - straight_through: STE wrapper for training with quantization
    """

    dtype: QuantizationType | str = QuantizationType.NF4
    runtime_dtype: QuantizationType | str | None = None
    group_size: int | None = None
    bits: int | None = None
    simulate: bool = False
    jax_native: bool = False

    pattern: str = field(default=DEFAULT_QUANTIZATION_PATTERN)

    def __post_init__(self):
        """Post-initialization processing to normalize dtype values.

        Converts string dtype values to their corresponding QuantizationType
        enum values for consistent internal representation.
        """
        if isinstance(self.dtype, str):
            self.dtype = QuantizationType(self.dtype)
        if isinstance(self.runtime_dtype, str):
            self.runtime_dtype = QuantizationType(self.runtime_dtype)
        if self.group_size is not None:
            self.group_size = int(self.group_size)
        if self.bits is not None:
            self.bits = int(self.bits)
        self.jax_native = bool(self.jax_native)

    __hash__ = hash_fn


def resolve_ejkernel_quant_params(config: QuantizationConfig) -> tuple[str, int, int, bool]:
    """Map an EasyDeL :class:`QuantizationConfig` onto the ejkernel quantizer.

    The ejkernel quantization API expects four positional knobs: a mode string
    (``"affine" | "nf4" | "mxfp4" | "nvfp4" | "mxfp8" | "nvfp8"``), a
    ``group_size`` (number of contiguous weights that share one scale), a
    ``bits`` count, and a ``needs_biases`` flag indicating whether the scheme
    stores per-group zero-points in addition to scales. This function
    resolves the EasyDeL-side enum + optional overrides into that 4-tuple
    while validating the combinatorics each scheme imposes:

    * ``AFFINE`` / ``INT8`` — group_size must be a power-of-two in
      ``{16, 32, 64, 128, 256, 512, 1024}``, bits must be in ``[2, 8]``;
      stores per-group ``(scale, bias)`` so ``needs_biases=True``. INT8 is
      treated as affine with bits defaulting to 8.
    * ``NF4`` — fixed bits=4 (NormalFloat lookup), group_size in the same
      power-of-two set; no biases (zero-mean lookup table).
    * ``MXFP4`` / ``NVFP4`` — group_size pinned to 32 / 16, bits=4; no biases.
    * ``MXFP8`` / ``NVFP8`` — group_size pinned to 32 / 16, bits=8; no biases.

    Args:
        config: Resolved :class:`QuantizationConfig` whose ``dtype`` selects
            the scheme and whose optional ``group_size`` / ``bits`` override
            the per-scheme defaults.

    Returns:
        Tuple ``(mode, group_size, bits, needs_biases)`` ready to be unpacked
        into ejkernel's quantization entry points.

    Raises:
        ValueError: If the (mode, group_size, bits) combination violates the
            scheme's constraints, or if ``config.dtype`` is not one of the
            ejkernel-supported quantization types (e.g. ``BINARY``, ``TERNARY``,
            and ``TURBOQUANT`` are handled by their own paths).
    """
    dtype = config.dtype
    if isinstance(dtype, str):
        dtype = QuantizationType(dtype)

    if dtype in {QuantizationType.AFFINE, QuantizationType.INT8}:
        # Map INT8 to ejkernel affine quantization (8-bit by default).
        bits = 8 if dtype == QuantizationType.INT8 else 4
        if config.bits is not None:
            bits = int(config.bits)
        group_size = 64 if config.group_size is None else int(config.group_size)
        if group_size not in {16, 32, 64, 128, 256, 512, 1024}:
            if dtype == QuantizationType.INT8 and config.group_size is None:
                group_size = 64
            else:
                raise ValueError("affine mode supports group_size in {16, 32, 64, 128, 256, 512, 1024}.")
        if bits not in {2, 3, 4, 5, 6, 7, 8}:
            raise ValueError("affine mode supports bits in {2, 3, 4, 5, 6, 7, 8}.")
        return "affine", group_size, bits, True
    if dtype == QuantizationType.NF4:
        bits = 4 if config.bits is None else int(config.bits)
        if bits != 4:
            raise ValueError("nf4 requires bits=4.")
        group_size = 64 if config.group_size is None else int(config.group_size)
        if group_size not in {16, 32, 64, 128, 256, 512, 1024}:
            raise ValueError("nf4 mode supports group_size in {16, 32, 64, 128, 256, 512, 1024}.")
        return "nf4", group_size, 4, False
    if dtype == QuantizationType.MXFP4:
        group_size = 32 if config.group_size is None else int(config.group_size)
        bits = 4 if config.bits is None else int(config.bits)
        if group_size != 32 or bits != 4:
            raise ValueError("mxfp4 requires group_size=32 and bits=4.")
        return "mxfp4", 32, 4, False
    if dtype == QuantizationType.NVFP4:
        group_size = 16 if config.group_size is None else int(config.group_size)
        bits = 4 if config.bits is None else int(config.bits)
        if group_size != 16 or bits != 4:
            raise ValueError("nvfp4 requires group_size=16 and bits=4.")
        return "nvfp4", 16, 4, False
    if dtype == QuantizationType.MXFP8:
        group_size = 32 if config.group_size is None else int(config.group_size)
        bits = 8 if config.bits is None else int(config.bits)
        if group_size != 32 or bits != 8:
            raise ValueError("mxfp8 requires group_size=32 and bits=8.")
        return "mxfp8", 32, 8, False
    if dtype == QuantizationType.NVFP8:
        group_size = 16 if config.group_size is None else int(config.group_size)
        bits = 8 if config.bits is None else int(config.bits)
        if group_size != 16 or bits != 8:
            raise ValueError("nvfp8 requires group_size=16 and bits=8.")
        return "nvfp8", 16, 8, False

    raise ValueError(f"Unsupported quantization type for ejkernel: {dtype}")


def resolve_jax_native_dtype(dtype: QuantizationType | str | None):
    """Map an EasyDeL quantization type onto its native ``jax.numpy`` dtype.

    Used by the ``jax_native=True`` path in :class:`QuantizationConfig`: when
    the requested scheme is one that JAX/ml_dtypes can store directly
    (microscaling FP4/FP8 variants), quantization can be done with a simple
    ``jnp.astype`` rather than going through ejkernel's pack/unpack kernels.
    Schemes without a backing JAX dtype (NF4, AFFINE, INT8, BINARY, …) return
    ``None`` so the caller falls back to the regular path.

    The mapping currently handles:

    * ``MXFP4`` -> ``jnp.float4_e2m1fn``
    * ``MXFP8`` -> ``jnp.float8_e5m2``
    * ``NVFP8`` -> ``jnp.float8_e4m3``

    Args:
        dtype: Quantization type to resolve. May be a :class:`QuantizationType`,
            its string code, or ``None``.

    Returns:
        The matching ``jnp`` dtype, or ``None`` when the type has no native
        JAX representation, when the running JAX / ml_dtypes build doesn't
        expose the dtype, or when ``dtype`` itself is ``None``. The function
        is therefore safe to call defensively on any config.
    """
    if dtype is None:
        return None
    if isinstance(dtype, str):
        dtype = QuantizationType(dtype)
    dtype_name = {
        QuantizationType.MXFP4: "float4_e2m1fn",
        QuantizationType.MXFP8: "float8_e5m2",
        QuantizationType.NVFP8: "float8_e4m3",
    }.get(dtype)
    if dtype_name is None:
        return None
    try:
        import jax.numpy as jnp
    except Exception:
        return None
    return getattr(jnp, dtype_name, None)
