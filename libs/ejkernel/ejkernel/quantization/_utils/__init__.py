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

"""Shared helper utilities for quantization.

This submodule provides low-level utilities used by the quantization
implementations:

- **bitpack**: Pack/unpack quantized values into uint32 arrays
- **fp_tables**: Lookup tables for minifloat formats (E2M1, E4M3, NF4)
- **grouping**: Group reshaping and codebook quantization utilities
"""

from .bitpack import _pack_bits, _unpack_bits
from .fp_tables import (
    _get_e2m1_max,
    _get_e2m1_table,
    _get_e2m1_threshold_map,
    _get_e4m3_max,
    _get_e4m3_q_threshold_map,
    _get_e4m3_table,
    _get_e4m3_table_q,
    _get_nf4_table,
    _get_nf4_threshold_map,
)
from .grouping import _quantize_to_codebook, _require_bits, _reshape_groups
from .qparams import (
    BackendQuantizationMode,
    GemvMode,
    KernelFamily,
    QuantizationAxis,
    QuantizationMode,
    RevSplitKMode,
    is_effective_4bit_mode,
    normalize_axis,
    normalize_gemv_mode,
    normalize_mode_and_bits,
    normalize_revsplitk_mode,
    normalize_revsplitk_parts,
    resolve_prepack_axis,
    resolve_qparams,
    resolve_runtime_axis_and_transpose,
    select_qmm_kernel_family,
    to_backend_mode,
)

__all__ = [
    "BackendQuantizationMode",
    "GemvMode",
    "KernelFamily",
    "QuantizationAxis",
    "QuantizationMode",
    "RevSplitKMode",
    "_get_e2m1_max",
    "_get_e2m1_table",
    "_get_e2m1_threshold_map",
    "_get_e4m3_max",
    "_get_e4m3_q_threshold_map",
    "_get_e4m3_table",
    "_get_e4m3_table_q",
    "_get_nf4_table",
    "_get_nf4_threshold_map",
    "_pack_bits",
    "_quantize_to_codebook",
    "_require_bits",
    "_reshape_groups",
    "_unpack_bits",
    "is_effective_4bit_mode",
    "normalize_axis",
    "normalize_gemv_mode",
    "normalize_mode_and_bits",
    "normalize_revsplitk_mode",
    "normalize_revsplitk_parts",
    "resolve_prepack_axis",
    "resolve_qparams",
    "resolve_runtime_axis_and_transpose",
    "select_qmm_kernel_family",
    "to_backend_mode",
]
