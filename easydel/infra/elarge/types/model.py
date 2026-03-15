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

"""Model identification and loading TypedDicts.

Defines the configuration structures for specifying which model to load,
how to locate it, and how to configure dtype/precision during loading.
"""

from __future__ import annotations

import typing as tp
from typing import Any, Literal, NotRequired, Required, TypedDict

from easydel.infra.factory import TaskType

from .aliases import DTypeLike, PrecisionLike

if tp.TYPE_CHECKING:
    from ejkernel.modules.operations.configs import BaseOperationConfig  # pyright: ignore[reportMissingTypeStubs]


class OperationConfigsDict(TypedDict, total=False):
    """Configuration dictionary for ejkernel operation overrides.

    Maps operation implementation names to their corresponding configuration
    objects. When a configuration is provided for an operation, it overrides
    ejkernel's automatic tuning behavior. When ``None`` or not set, ejkernel
    will use its default autotune behavior.

    Attributes:
        flash_attn2: Config for Flash Attention 2 implementation.
        ring: Config for ring attention (sequence parallelism).
        blocksparse: Config for block-sparse attention.
        ragged_page_attention_v2: Config for ragged paged attention v2 (TPU).
        ragged_page_attention_v3: Config for ragged paged attention v3 (TPU).
        multi_latent_ragged_page_attention_v1: Config for MLA ragged paged attention.
        unified_attention: Config for unified attention dispatcher.
        paged_flash_attention: Config for paged flash attention.
        sdpa: Config for scaled dot-product attention (JAX default).
        vanilla: Config for naive reference attention implementation.
    """

    flash_attn2: NotRequired["BaseOperationConfig | None"]
    ring: NotRequired["BaseOperationConfig | None"]
    blocksparse: NotRequired["BaseOperationConfig | None"]
    ragged_page_attention_v2: NotRequired["BaseOperationConfig | None"]
    ragged_page_attention_v3: NotRequired["BaseOperationConfig | None"]
    multi_latent_ragged_page_attention_v1: NotRequired["BaseOperationConfig | None"]
    unified_attention: NotRequired["BaseOperationConfig | None"]
    paged_flash_attention: NotRequired["BaseOperationConfig | None"]
    sdpa: NotRequired["BaseOperationConfig | None"]
    vanilla: NotRequired["BaseOperationConfig | None"]


class ModelCfg(TypedDict, total=False):
    """Model configuration section for identifying and loading models.

    Specifies the model source (HuggingFace Hub ID or local path), optional
    custom tokenizer, task type for architecture selection, and additional
    loading arguments.

    Attributes:
        name_or_path: HuggingFace Hub model ID or local filesystem path
            (required).
        tokenizer: HuggingFace tokenizer name or path. Defaults to
            ``name_or_path`` when omitted.
        task: Model task type controlling which architecture head to use
            (e.g., ``"causal-language-model"``). ``"auto-bind"`` infers from
            the HuggingFace config.
        extra_kwargs: Additional keyword arguments forwarded to
            ``from_pretrained``.
    """

    name_or_path: Required[str]
    tokenizer: NotRequired[str]
    task: NotRequired[
        TaskType
        | str
        | Literal[
            "causal-language-model",
            "vision-language-model",
            "diffusion-language-model",
            "image-text-to-text",
            "base-module",
            "vision-module",
            "sequence-to-sequence",
            "speech-sequence-to-sequence",
            "zero-shot-image-classification",
            "sequence-classification",
            "audio-classification",
            "image-classification",
            "auto-bind",
        ]
    ]
    extra_kwargs: NotRequired[dict[str, Any]]


class LoaderCfg(TypedDict, total=False):
    """Model loading configuration for dtype, precision, and device settings.

    Attributes:
        device: Target device for model placement (e.g., ``"tpu"``).
        dtype: Data type for model parameters (e.g., ``"bf16"``, ``"fp32"``).
        param_dtype: Separate dtype for stored parameters when different from
            compute dtype.
        precision: JAX matmul precision level (``"DEFAULT"``, ``"HIGH"``,
            ``"HIGHEST"``).
        verbose: Enable verbose logging during model loading.
        from_torch: Force loading from PyTorch checkpoint. ``None`` for auto
            detection.
        trust_remote_code: Allow executing remote code from HuggingFace Hub
            model repositories.
    """

    device: NotRequired[Any]
    dtype: NotRequired[DTypeLike]
    param_dtype: NotRequired[DTypeLike]
    precision: NotRequired[PrecisionLike]
    verbose: NotRequired[bool]
    from_torch: NotRequired[bool | None]
    trust_remote_code: NotRequired[bool]
