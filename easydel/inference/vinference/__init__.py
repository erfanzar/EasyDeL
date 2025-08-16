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

"""vInference: Streamlined inference engine for text generation.

vInference provides a simplified interface for text generation with
pre-compiled models, optimized for single and batch inference scenarios.
It focuses on ease of use while maintaining high performance.

Key Features:
    - Simple API for text generation
    - Pre-compilation for optimized inference
    - Support for streaming and non-streaming generation
    - Automatic mixed-precision and sharding
    - Built-in OpenAI API compatibility

Components:
    vInference: Main inference engine
    vInferenceConfig: Configuration for generation
    vInferencePreCompileConfig: Pre-compilation settings
    vInferenceApiServer: OpenAI-compatible API server
    PromptOutput: Output structure for generated text
    SampleState: Internal state for sampling

Example:
    >>> from easydel.inference.vinference import (
    ...     vInference,
    ...     vInferenceConfig
    ... )
    >>> # Create configuration
    >>> config = vInferenceConfig(
    ...     max_new_tokens=100,
    ...     temperature=0.7,
    ...     top_p=0.9
    ... )
    >>> # Initialize inference engine
    >>> engine = vInference(
    ...     model=model,
    ...     processor_class=tokenizer,
    ...     generation_config=config
    ... )
    >>> # Generate text
    >>> output = engine.generate(
    ...     "Once upon a time",
    ...     max_new_tokens=50
    ... )
    >>> print(output.text)

Note:
    vInference is optimized for scenarios where you need straightforward
    text generation without the complexity of continuous batching.
    For high-throughput scenarios, consider using vSurge instead.
"""

from .api_server import vInferenceApiServer
from .utilities import SampleState, vInferenceConfig, vInferencePreCompileConfig
from .vinference import PromptOutput, vInference

__all__ = (
    "PromptOutput",
    "SampleState",
    "vInference",
    "vInferenceApiServer",
    "vInferenceConfig",
    "vInferencePreCompileConfig",
)
