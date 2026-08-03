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

"""Support for loading externally quantized checkpoints.

EasyDeL's own quantization path takes full-precision weights and quantizes
them (:class:`~easydel.layers.quantization.EasyQuantizer`). This package
handles the other direction: checkpoints that arrive **already** quantized —
fp8, compressed-tensors, AWQ/GPTQ, MXFP4, NVFP4 — and must be decoded into a
form the existing kernels run.

The design is one seam and three thin layers above it:

* :class:`CanonicalQuantizedWeight` / :class:`QuantSpec` — the single runtime
  representation every format is reduced to. It is already what
  ``ParallelLinearQuantized`` stores, so nothing below the seam changes.
* :class:`CheckpointQuantAdapter` — one class per checkpoint format, owning
  every format-specific decision and nothing else.
* :class:`CheckpointQuantScheme` — path-level resolution of which modules are
  quantized and how, shared by all formats.
* :func:`checkpoint_quant_reform_param` — projects a resolved scheme onto the
  ``reform_param`` rules the existing converter already understands, so no new
  loading path exists.

Adding a format means adding an adapter. If it needs more than that, the seam
is wrong and should be fixed rather than bypassed.
"""

from ._adapter import (
    CHECKPOINT_QUANT_CATEGORY,
    CheckpointQuantAdapter,
    ConfigParse,
    IgnoreRules,
    available_quant_methods,
    get_adapter,
    register_adapter,
)
from ._canonical import (
    ActivationPolicy,
    ActivationQuantKind,
    CanonicalQuantizedWeight,
    QuantSpec,
    SourceFormat,
)
from ._codecs import (
    decode_awq_int4,
    decode_gptq_int4,
    decode_mxfp4,
    decode_nvfp4,
    decode_scaled_elements,
)
from ._decode import (
    decode_e4m3_scale,
    decode_e8m0_scale,
    expand_scale,
    largest_supported_group_size,
    unpack_uint8_to_e2m1,
    unpack_uint32_to_uint4,
)

# Importing the format adapters registers them; without this the registry is
# empty and every checkpoint reports "no adapter for quant_method".
from ._formats import (
    AwqAdapter,
    CompressedTensorsAdapter,
    GptqAdapter,
    Mxfp4Adapter,
    Nvfp4Adapter,
    ScaledElementsAdapter,
)
from ._reform import (
    ExtractField,
    FromCanonical,
    RebuildCanonical,
    ToCanonical,
    canonical_param_names,
    checkpoint_quant_reform_param,
)
from ._scheme import CheckpointQuantScheme, ResolvedScheme

__all__ = (
    "CHECKPOINT_QUANT_CATEGORY",
    "ActivationPolicy",
    "ActivationQuantKind",
    "AwqAdapter",
    "CanonicalQuantizedWeight",
    "CheckpointQuantAdapter",
    "CheckpointQuantScheme",
    "CompressedTensorsAdapter",
    "ConfigParse",
    "ExtractField",
    "FromCanonical",
    "GptqAdapter",
    "IgnoreRules",
    "Mxfp4Adapter",
    "Nvfp4Adapter",
    "QuantSpec",
    "RebuildCanonical",
    "ResolvedScheme",
    "ScaledElementsAdapter",
    "SourceFormat",
    "ToCanonical",
    "available_quant_methods",
    "canonical_param_names",
    "checkpoint_quant_reform_param",
    "decode_awq_int4",
    "decode_e4m3_scale",
    "decode_e8m0_scale",
    "decode_gptq_int4",
    "decode_mxfp4",
    "decode_nvfp4",
    "decode_scaled_elements",
    "expand_scale",
    "get_adapter",
    "largest_supported_group_size",
    "register_adapter",
    "unpack_uint8_to_e2m1",
    "unpack_uint32_to_uint4",
)
