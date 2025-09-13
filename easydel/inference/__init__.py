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

"""EasyDeL Inference Module.

This module provides high-performance inference engines and APIs for running
large language models and speech recognition models efficiently on JAX/XLA.

Key Components:
    - vSurge: High-performance batched inference engine with continuous batching
    - vInference: Streamlined inference engine for single/batch generation
    - vWhisper: Speech recognition and transcription engine
    - OpenAI API compatibility layer for seamless integration

Example:
    Basic text generation with vInference:

    >>> from easydel.inference import vInference, vInferenceConfig
    >>> config = vInferenceConfig(
    ...     model_id="google/gemma-2b",
    ...     max_new_tokens=100
    ... )
    >>> engine = vInference(config=config)
    >>> response = engine.generate("Hello, how are you?")

    Running an OpenAI-compatible API server:

    >>> from easydel.inference import vSurgeApiServer, vSurge
    >>> surge = vSurge.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> server = vSurgeApiServer(surge)
    >>> server.run(host="0.0.0.0", port=8000)

Attributes:
    FunctionCallFormat: Enum defining function call formatting styles
    FunctionCallFormatter: Utility for formatting function calls in prompts
    InferenceApiRouter: FastAPI router for OpenAI API compatibility
    JitableSamplingParams: JAX-compatible sampling parameters
    SamplingParams: High-level sampling configuration
    vDriver: Low-level driver for vSurge engine
    vEngine: Core vSurge inference engine
    vInference: Streamlined inference interface
    vInferenceApiServer: API server for vInference
    vInferenceConfig: Configuration for vInference
    vInferencePreCompileConfig: Pre-compilation settings
    vSurge: High-performance batched inference engine
    vSurgeApiServer: API server for vSurge
    vSurgeRequest: Request object for vSurge
    vWhisperInference: Speech recognition engine
    vWhisperInferenceConfig: Configuration for vWhisper
"""

from .esurge import EngineRequest, EngineRequestStatus, eSurge, eSurgeApiServer, eSurgeRunner
from .oai_proxies import InferenceApiRouter
from .sampling_params import JitableSamplingParams, SamplingParams
from .tools import ToolParser, ToolParserManager
from .vinference import vInference, vInferenceApiServer, vInferenceConfig, vInferencePreCompileConfig
from .vsurge import vDriver, vEngine, vSurge, vSurgeApiServer, vSurgeRequest
from .vwhisper import vWhisperInference, vWhisperInferenceConfig

__all__ = (
    "EngineRequest",
    "EngineRequestStatus",
    "InferenceApiRouter",
    "JitableSamplingParams",
    "SamplingParams",
    "ToolParser",
    "ToolParserManager",
    "eSurge",
    "eSurgeApiServer",
    "eSurgeRunner",
    "vDriver",
    "vEngine",
    "vInference",
    "vInferenceApiServer",
    "vInferenceConfig",
    "vInferencePreCompileConfig",
    "vSurge",
    "vSurgeApiServer",
    "vSurgeRequest",
    "vWhisperInference",
    "vWhisperInferenceConfig",
)
