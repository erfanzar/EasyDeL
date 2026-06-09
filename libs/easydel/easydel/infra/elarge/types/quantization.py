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

"""Quantization TypedDicts.

Defines configuration structures for model weight quantization and
KV cache compression, supporting various quantization formats
(NF4, INT8, FP8, ternary, binary) with configurable group sizes.
"""

from __future__ import annotations

import typing as tp
from typing import NotRequired, TypedDict

from easydel.infra.etils import EasyDeLPlatforms
from easydel.layers import QuantizationConfig


class EasyDeLQuantizationCfg(TypedDict, total=False):
    """Extended quantization configuration with layer selection patterns.

    Attributes:
        dtype: Quantization format for model weights (e.g., ``"nf4"``,
            ``"int8"``, ``"mxfp8"``).
        runtime_dtype: Quantization format used at inference time, allowing
            a different format than the stored weights.
        group_size: Number of elements per quantization group for
            block-wise quantization.
        bits: Number of bits for quantized representation.
        simulate: If ``True``, simulate quantization without actually
            quantizing (for debugging).
        jax_native: Use JAX-native quantization kernels instead of custom
            implementations.
        pattern: Regex pattern to select which layers to quantize
            (e.g., ``".*dense.*"``).
    """

    dtype: NotRequired[tp.Literal["nf4", "int8", "affine", "ternary", "binary", "mxfp8", "nvfp8", "mxfp4"]]
    runtime_dtype: NotRequired[tp.Literal["nf4", "int8", "affine", "ternary", "binary", "mxfp8", "nvfp8", "mxfp4"]]
    group_size: NotRequired[int]
    bits: NotRequired[int]
    simulate: NotRequired[bool]
    jax_native: NotRequired[bool]
    pattern: NotRequired[str]


class QuantizationCfg(TypedDict, total=False):
    """Quantization configuration for model compression and efficiency.

    Attributes:
        platform: Target platform for quantization kernel selection.
            ``None`` for auto-detection.
        kv_cache: Quantization config applied to the KV cache for memory
            savings during inference. ``None`` to disable KV cache
            quantization.
        model: Quantization config applied to model weights. ``None`` to
            disable weight quantization.
        apply_quantization: Master switch to enable/disable quantization.
        use_qmm_best_config: Use QMM (Quantized Matrix Multiply) best-known
            configuration for the target platform.
        qmm_platform_override: Override the detected platform for QMM
            kernel selection (e.g., ``"xla"``).
        qmm_tpu_path_override: Override the TPU QMM code path
            (e.g., ``"packed"``).
    """

    platform: NotRequired[EasyDeLPlatforms | None]
    kv_cache: NotRequired[QuantizationConfig | EasyDeLQuantizationCfg | None]
    model: NotRequired[QuantizationConfig | EasyDeLQuantizationCfg | None]
    apply_quantization: NotRequired[bool]
    use_qmm_best_config: NotRequired[bool]
    qmm_platform_override: NotRequired[str | None]
    qmm_tpu_path_override: NotRequired[str | None]
