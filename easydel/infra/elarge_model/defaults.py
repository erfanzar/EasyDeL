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

"""Default configuration values for ELM (EasyDeL Large Model).

This module defines the default configuration values that are applied to all
ELM configurations when values are not explicitly specified. These defaults
provide sensible starting points for model loading, sharding, quantization,
inference serving, and data mixture settings.

The configuration is organized into several sections:

    - **loader**: Settings for model loading including data types and verbosity.
    - **sharding**: Parallelism and sharding configuration for distributed training.
    - **quantization**: Quantization parameters for model compression.
    - **base_config**: Base model configuration overrides.
    - **esurge**: ESurge inference engine configuration.
    - **mixture**: Data mixture and loading configuration for training.

Example:
    Access default configuration values::

        from easydel.infra.elarge_model.defaults import DEFAULTS

        # Get default dtype for model loading
        default_dtype = DEFAULTS["loader"]["dtype"]  # Returns "bf16"

        # Get default axis dimensions for sharding
        axis_dims = DEFAULTS["sharding"]["axis_dims"]  # Returns (1, 1, 1, -1, 1)

Note:
    These defaults can be overridden by providing explicit values when
    creating an ELM configuration. The defaults are designed for common
    use cases on TPU/GPU hardware with tensor parallelism.

Attributes:
    DEFAULTS (ELMConfig): A dictionary containing all default configuration
        values for the ELM system. This includes settings for model loading,
        sharding strategies, quantization, base configuration, inference
        engine (ESurge), and data mixture handling.

See Also:
    - :mod:`easydel.infra.elarge_model.types`: Type definitions for ELMConfig.
    - :mod:`easydel.infra.elarge_model`: Main ELM module documentation.
"""

from __future__ import annotations

import pathlib

from .types import ELMConfig

DEFAULTS: ELMConfig = {
    # Loader configuration: Controls how models are loaded into memory.
    # - dtype: The computation dtype for model forward passes (bf16 = bfloat16).
    # - param_dtype: The dtype used to store model parameters (bf16 = bfloat16).
    # - verbose: Whether to print loading progress and diagnostics.
    "loader": {"dtype": "bf16", "param_dtype": "bf16", "verbose": True},
    # Sharding configuration: Defines how the model is distributed across devices.
    # - axis_dims: Tuple defining parallelism dimensions (dp, fsdp, ep, tp, sp).
    #   Values of 1 mean no parallelism; -1 means infer from available devices.
    #   Default (1, 1, 1, -1, 1) uses tensor parallelism across all devices.
    # - axis_names: Names for each parallelism axis:
    #   dp=data parallel, fsdp=fully sharded data parallel, ep=expert parallel,
    #   tp=tensor parallel, sp=sequence parallel.
    # - auto_shard_model: Automatically apply sharding rules to model parameters.
    # - use_ring_of_experts: Use ring communication for MoE expert parallelism.
    # - fsdp_is_ep_bound: Bind FSDP axis to expert parallel axis for MoE models.
    # - sp_is_ep_bound: Bind sequence parallel axis to expert parallel axis.
    "sharding": {
        "axis_dims": (1, 1, 1, -1, 1),
        "axis_names": ("dp", "fsdp", "ep", "tp", "sp"),
        "auto_shard_model": True,
        "use_ring_of_experts": False,
        "fsdp_is_ep_bound": True,
        "sp_is_ep_bound": True,
    },
    # Quantization configuration: Settings for model weight quantization.
    # - group_size: Group size for quantization (default 128 elements).
    # - apply_quantization: Whether to apply quantization to module weights.
    # - linear_pattern: Regex pattern matching linear layers to quantize.
    #   ".*" matches all linear layers by default.
    # - linear_group_size: Group size specifically for linear layer quantization.
    # - use_qmm_best_config: Prefer ejkernel tuned QMM block configs by default.
    # - qmm_platform_override: Optional explicit QMM platform override.
    # - qmm_tpu_path_override: Optional explicit TPU fused-path override.
    # - jax_native: If True and the dtype has a JAX-native format (e.g., MXFP4/MXFP8/NVFP8),
    #   quantization uses `astype` instead of ejkernel (applies even in QAT/simulate paths).
    "quantization": {
        "group_size": 128,
        "apply_quantization": False,
        "linear_pattern": ".*",
        "linear_group_size": 64,
        "use_qmm_best_config": False,
        "qmm_platform_override": None,
        "qmm_tpu_path_override": None,
    },
    # Base configuration: Overrides for the underlying model configuration.
    # - values: Dictionary of config values to set on the base model config.
    # - hardware_abstraction: Enable hardware-agnostic operations for portability.
    "base_config": {"values": {"hardware_abstraction": True}},
    # ESurge configuration: Settings for the ESurge inference serving engine.
    # - min_input_pad: Minimum padding for input sequences (for efficient batching).
    # - max_num_seqs: Maximum number of sequences to batch together.
    # - hbm_utilization: Target HBM (High Bandwidth Memory) utilization (0.0-1.0).
    #   0.80 means target 80% memory utilization for KV cache allocation.
    # - page_size: Page size for paged attention KV cache management.
    # - enable_prefix_caching: Cache common prefixes across requests.
    # - verbose: Print inference engine diagnostics and statistics.
    "esurge": {
        "min_input_pad": 16,
        "max_num_seqs": 32,
        "hbm_utilization": 0.80,
        "page_size": 128,
        "bind_graphstate_for_aot": False,
        "enable_prefix_caching": True,
        "extra_stops": None,
        "verbose": False,
    },
    # Mixture configuration: Settings for data loading and mixing during training.
    # - cache_dir: Directory for caching downloaded datasets.
    # - streaming: Use streaming mode for large datasets (reduces memory).
    # - text_target_field: Field name containing text data in the dataset.
    # - image_target_field: Field name containing image data (for multimodal).
    # - batch_size: Number of samples per training batch.
    # - shuffle_buffer_size: Buffer size for shuffling streaming datasets.
    # - seed: Random seed for reproducibility in shuffling/sampling.
    # - use_fast_loader: Use optimized data loading pipeline.
    # - num_workers: Number of worker processes for parallel data loading.
    # - prefetch_size: Number of batches to prefetch ahead.
    # - enable_caching: Cache processed examples for faster subsequent epochs.
    "mixture": {
        "cache_dir": f"{pathlib.Path.home()}/.cache/easydel",
        "streaming": True,
        "text_target_field": "text",
        "image_target_field": "image",
        "batch_size": 1,
        "shuffle_buffer_size": 1000,
        "seed": 42,
        "use_fast_loader": True,
        "num_workers": 4,
        "prefetch_size": 10,
        "enable_caching": True,
    },
}
