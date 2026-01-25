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

"""Model runners for the eSurge inference engine.

This module provides the core execution components for running models
in the eSurge engine, including model runners and sequence buffering.
The eSurge engine is designed for high-performance inference with features
like paged attention, dynamic batching, and compilation caching.

Components:
    eSurgeRunner: Main model runner class that orchestrates model execution,
        handling state management, input preparation, forward pass execution,
        and token sampling.
    SequenceBuffer: Buffer for managing token sequences during generation,
        providing efficient storage and management of tokens, page tables,
        and generation metadata for batch processing.

Architecture:
    The module implements a fused execution model where compilation and
    execution are managed separately:

    1. Compilation: Functions are pre-compiled for multiple input configurations
       (token counts, batch sizes) to eliminate runtime compilation overhead.
    2. Execution: At runtime, the appropriate compiled function is selected
       based on the current input dimensions.

Performance Features:
    - Paged attention for memory-efficient KV cache management
    - Vectorized operations for batch processing
    - Pre-allocated buffers to minimize memory allocation
    - Compilation caching to avoid recompilation
    - Support for async scheduling to overlap sampling with forward passes

Example:
    >>> from easydel.inference.esurge.runners import eSurgeRunner
    >>>
    >>> # Initialize the runner with model and configuration
    >>> runner = eSurgeRunner(
    ...     model=model,
    ...     max_model_len=2048,
    ...     max_num_seqs=8,
    ...     hbm_utilization=0.9,
    ... )
    >>>
    >>> # Compile for expected configurations
    >>> runner.compile()
    >>>
    >>> # Execute model with scheduler output
    >>> output = runner.execute_model(scheduler_output)

See Also:
    - :mod:`easydel.inference.esurge.runners.execution_manager`: Manages compiled
      execution functions for different batch/token configurations.
    - :mod:`easydel.inference.esurge.runners.executors`: Sub-components for
      model execution, sampling, and batch preparation.
"""

from .model_runner import eSurgeRunner
from .sequence_buffer import SequenceBuffer

__all__ = ("SequenceBuffer", "eSurgeRunner")
