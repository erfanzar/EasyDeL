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

"""vSurge: High-performance batched inference engine.

vSurge is a production-ready inference engine optimized for high-throughput
text generation with continuous batching and efficient memory management.
It provides seamless integration with JAX/XLA for hardware acceleration.

Key Features:
    - Continuous batching for maximum throughput
    - Asynchronous request processing
    - OpenAI API compatibility
    - Smart tokenization with bytecode decoding
    - Efficient KV cache management
    - Support for streaming and non-streaming generation

Components:
    vDriver: Low-level driver for model execution
    vEngine: Core inference engine with scheduling
    vSurge: High-level interface for text generation
    vSurgeApiServer: OpenAI-compatible API server
    vSurgeRequest: Request configuration object

Example:
    >>> from easydel.inference.vsurge import vSurge, vDriver
    >>> # Initialize driver with model
    >>> driver = vDriver.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> # Create vSurge instance
    >>> surge = vSurge(driver)
    >>> # Start the engine
    >>> surge.start()
    >>> # Generate text
    >>> async def generate():
    ...     async for response in surge.generate(
    ...         "Hello, how are you?",
    ...         max_tokens=50
    ...     ):
    ...         print(response.text)

Note:
    vSurge is designed for production use with features like
    request prioritization, graceful shutdown, and comprehensive
    monitoring capabilities.
"""

from .core import vDriver, vEngine
from .server import vSurgeApiServer
from .vsurge import vSurge, vSurgeRequest

__all__ = "vDriver", "vEngine", "vSurge", "vSurgeApiServer", "vSurgeRequest"
