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

"""Type aliases for the ELM configuration system.

Defines the fundamental type aliases used across all ELM configuration
structures for specifying data types, precision levels, partition rules,
dataset formats, and attention operation implementations.
"""

from __future__ import annotations

import typing as tp
from typing import Any

import jax
from jax import numpy as jnp

DTypeLike = (
    str
    | jnp.dtype
    | type
    | tp.Literal[
        "fp8",
        "bf16",
        "fp16",
        "fp32",
        "mxfp4",
        "mxfp8",
        "nvfp8",
    ]
)
"""Type alias for data type specifications.

Represents valid data type values that can be used for model computation and
parameter storage. Accepts JAX dtype objects, Python type objects, or string
literals for common precision formats.

Supported string literals:
    - "fp8": 8-bit floating point (FP8 E4M3 or E5M2).
    - "bf16": Brain floating point 16-bit.
    - "fp16": IEEE 754 half precision 16-bit.
    - "fp32": IEEE 754 single precision 32-bit.
    - "mxfp4": Microscaling 4-bit floating point.
    - "mxfp8": Microscaling 8-bit floating point.
    - "nvfp8": NVIDIA FP8 format.
"""

PrecisionLike = (
    str
    | jax.lax.Precision
    | None
    | tp.Literal[
        "HIGH",
        "DEFAULT",
        "HIGHEST",
        "highest",
        "float32",
        "high",
        "bfloat16_3x",
        "tensorfloat32",
        "default",
        "fastest",
    ]
)
"""Type alias for JAX precision level specifications."""

PartitionRules = tuple[tuple[str, Any], ...]
"""Type alias for model partitioning rules.

A sequence of (pattern, partition_spec) pairs used to determine how model
parameters are distributed across devices.
"""

DatasetTypeLike = tp.Literal[
    "json",
    "jsonl",
    "parquet",
    "csv",
    "arrow",
    "huggingface",
    "hf",
    "tsv",
    "txt",
    "text",
]
"""Type alias for supported dataset format types."""

OperationImplName = tp.Literal[
    "flash_attn2",
    "ring",
    "blocksparse",
    "ragged_page_attention_v2",
    "ragged_page_attention_v3",
    "multi_latent_ragged_page_attention_v1",
    "unified_attention",
    "paged_flash_attention",
    "sdpa",
    "cudnn",
    "cuda_flash_attn2",
    "vanilla",
    "autoregressive_decodeattn",
]
"""Type alias for registered attention operation implementation names."""
