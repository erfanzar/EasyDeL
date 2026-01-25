# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Execution sub-components for eSurge runners.

This package provides modular executor components that handle specific aspects
of the eSurge inference pipeline. By separating concerns into distinct executors,
the codebase achieves better maintainability, testability, and flexibility.

Components:
    BatchMetadataPreparer: Handles CPU-first batch metadata preparation,
        performing fast NumPy computations on host and consolidating data
        for efficient device transfer.
    ModelStepExecutor: Manages compilation and execution of the model forward
        pass, including KV cache updates and hidden state/logits computation.
    SamplerExecutor: Manages compilation and execution of token sampling,
        applying temperature, top-k, top-p, and min-p filtering.

Architecture:
    The executors follow a compile-then-execute pattern:

    1. During startup, each executor pre-compiles functions for expected
       input configurations (token counts, batch sizes).
    2. At runtime, the appropriate compiled function is retrieved from
       an LRU cache based on the current input dimensions.
    3. The function is executed with the actual input data.

    This separation allows:
    - Independent optimization of each execution phase
    - Different compilation strategies (AOT vs JIT) per component
    - Easier debugging and profiling of individual phases
    - Future extension with alternative implementations

Example:
    >>> from easydel.inference.esurge.runners.executors import (
    ...     BatchMetadataPreparer,
    ...     ModelStepExecutor,
    ...     SamplerExecutor,
    ... )
    >>>
    >>> # Create executors with shared configuration
    >>> batch_prep = BatchMetadataPreparer(
    ...     metadata=cache_config,
    ...     empty_sharding=sharding,
    ...     max_num_tokens=4096,
    ...     max_num_reqs=32,
    ...     max_model_len=8192,
    ...     min_input_pad=8,
    ... )
    >>>
    >>> # Prepare batch metadata
    >>> batch_metadata = batch_prep.prepare_batch_metadata(...)
    >>>
    >>> # Execute model forward pass
    >>> model_outputs = model_executor.get_compiled(
    ...     num_tokens=256, padded_num_reqs=16
    ... )(graphstate, graphother, kv_pages, batch_metadata)
    >>>
    >>> # Sample tokens
    >>> rng_key, tokens, valid = sampler_executor.get_compiled(
    ...     num_tokens=256, padded_num_reqs=16
    ... )(batch_metadata, req_num_tokens, active_mask, logits, rng_key)
"""

from .batch_preparer import BatchMetadataPreparer
from .model_executor import ModelStepExecutor
from .sampler_executor import SamplerExecutor

__all__ = (
    "BatchMetadataPreparer",
    "ModelStepExecutor",
    "SamplerExecutor",
)
