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

"""Runtime policy controls for quantization fast paths.

This module defines :class:`QuantRuntimeConfig`, a frozen dataclass that
controls implementation choices within the quantize/dequantize pipeline.
None of the flags change quantization semantics (the numerical values of
the output are identical across all flag combinations for the same mode and
metadata dtype); they only select among equivalent implementation strategies.

The module also provides :func:`resolve_runtime_config`, which returns a
backend-tuned default when no explicit config is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QuantRuntimeConfig:
    """Optional runtime policy for quantize/dequantize internals.

    All flags are purely implementation choices and do not alter the numerical
    semantics of quantization.  They are exposed so that advanced users and
    autotuning can select the fastest path for a particular backend and shape.

    Attributes:
        enable_u4_u8_fastpath: When ``True``, uses dedicated grouped
            pack/unpack kernels for bit-widths in ``{1, 2, 4, 8}`` and the
            generic cross-word path for affine bit-widths ``3, 5, 6, 7``.
            Default: ``True``.

        enable_threshold_codebook: When ``True``, uses binary-search
            (threshold/midpoint) codebook quantization instead of the
            streaming argmin fallback.  Default: ``True``.

        enable_parity_fallback: When ``True``, forces the slow streaming
            argmin codebook path regardless of ``enable_threshold_codebook``.
            Useful for numerical parity checks.  Default: ``False``.

        strict_shape_alignment: When ``True``, the fast grouped pack/unpack
            path raises ``ValueError`` if the last dimension is not a multiple
            of ``values_per_word`` rather than auto-padding.  Default:
            ``True``.

        prefer_compute_dtype: Floating-point dtype for dequantization
            arithmetic.  One of ``"bf16"``, ``"fp16"``, ``"fp32"``.
            Default: ``"fp32"``.

        affine_metadata_dtype: Storage dtype for affine ``scales`` and
            ``zeros`` arrays.  One of ``"input"`` (match the weight tensor
            input dtype), ``"bf16"``, ``"fp16"``, ``"fp32"``.
            Default: ``"input"``.

        dequant_output_dtype: Output dtype for the dequantized tensor.
            One of ``"compute"`` (same as ``prefer_compute_dtype``), ``"bf16"``,
            ``"fp16"``, ``"fp32"``.  Default: ``"fp32"``.

        dequant_unpack_policy: Unpacking strategy selector.  One of
            ``"auto"`` (use fast path when ``enable_u4_u8_fastpath=True``),
            ``"fast"`` (force fast path), ``"generic"`` (force generic path).
            Default: ``"auto"``.

        minifloat_decode_policy: Decode strategy for FP4/FP8 codebooks.
            One of ``"auto"`` (currently always table lookup), ``"table"``
            (constant-memory gather), ``"arith"`` (arithmetic decode).
            Default: ``"auto"``.
    """

    enable_u4_u8_fastpath: bool = True
    enable_threshold_codebook: bool = True
    enable_parity_fallback: bool = False
    strict_shape_alignment: bool = True
    prefer_compute_dtype: Literal["bf16", "fp16", "fp32"] = "fp32"
    affine_metadata_dtype: Literal["input", "bf16", "fp16", "fp32"] = "input"
    dequant_output_dtype: Literal["compute", "bf16", "fp16", "fp32"] = "fp32"
    dequant_unpack_policy: Literal["auto", "fast", "generic"] = "auto"
    minifloat_decode_policy: Literal["auto", "table", "arith"] = "auto"

    @classmethod
    def fastest_for_backend(
        cls,
        *,
        backend: str | None = None,
        keep_fp32_output: bool = False,
    ) -> "QuantRuntimeConfig":
        """Return an aggressive throughput profile tuned for the current backend.

        Selects reduced-precision compute and metadata dtypes to maximise
        dequantization throughput:

        - **TPU**: ``prefer_compute_dtype="bf16"``, ``affine_metadata_dtype="bf16"``.
        - **GPU / MPS / CPU**: ``prefer_compute_dtype="fp16"``,
          ``affine_metadata_dtype="fp16"``.

        All fast-path flags (``enable_u4_u8_fastpath``,
        ``enable_threshold_codebook``, ``strict_shape_alignment``) are enabled.

        Args:
            backend: JAX backend string (e.g., ``"tpu"``, ``"gpu"``,
                ``"cuda"``).  When ``None``, the current default backend from
                ``jax.default_backend()`` is used.
            keep_fp32_output: If ``True``, forces the output dtype to
                ``"fp32"`` regardless of the compute dtype.  Default
                ``False`` (output dtype matches compute dtype).

        Returns:
            A :class:`QuantRuntimeConfig` with backend-appropriate settings.
        """
        if backend is None:
            import jax

            backend = jax.default_backend()
        backend = str(backend).lower()

        if backend == "tpu":
            compute = "bf16"
            meta = "bf16"
        else:
            compute = "fp16"
            meta = "fp16"

        return cls(
            enable_u4_u8_fastpath=True,
            enable_threshold_codebook=True,
            enable_parity_fallback=False,
            strict_shape_alignment=True,
            prefer_compute_dtype=compute,
            affine_metadata_dtype=meta,
            dequant_output_dtype="fp32" if keep_fp32_output else "compute",
            dequant_unpack_policy="auto",
        )


def resolve_runtime_config(config: QuantRuntimeConfig | None) -> QuantRuntimeConfig:
    """Return *config* unchanged, or a backend-tuned fast config when ``None``.

    Args:
        config: Explicit runtime config, or ``None`` to auto-select.

    Returns:
        The provided *config*, or ``QuantRuntimeConfig.fastest_for_backend()``
        when *config* is ``None``.
    """
    if config is None:
        return QuantRuntimeConfig.fastest_for_backend()
    return config


__all__ = ("QuantRuntimeConfig", "resolve_runtime_config")
