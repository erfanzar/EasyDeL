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
large language models and speech recognition workloads efficiently on JAX/XLA.

Key Components:
    - eSurge: Unified attention inference engine with paged KV caches
    - vWhisper: Speech recognition and transcription engine
    - OpenAI API compatibility layer for seamless integration

Example:
    Running an OpenAI-compatible API server::

        >>> from easydel.inference import eSurgeApiServer, eSurge
        >>> surge = eSurge(...)
        >>> server = eSurgeApiServer(surge)
        >>> server.run(host="0.0.0.0", port=8000)

Attributes:
    EngineRequest: Request object for inference engine operations.
    EngineRequestStatus: Enum representing the status of an engine request.
    InferenceApiRouter: FastAPI router for OpenAI API compatibility.
    JitableSamplingParams: JAX-compatible sampling parameters for JIT compilation.
    SamplingParams: High-level sampling configuration for text generation.
    ToolParser: Parser for extracting tool/function calls from model outputs.
    ToolParserManager: Manager for handling multiple tool parsers.
    eSurge: Main inference engine class for running language models.
    eSurgeApiServer: OpenAI-compatible API server for eSurge.
    eSurgeRunner: Low-level runner for eSurge inference operations.
    vWhisperInference: Speech recognition inference engine.
    vWhisperInferenceConfig: Configuration for vWhisper inference.
"""

from .esurge import EngineRequest, EngineRequestStatus, eSurge, eSurgeApiServer, eSurgeRunner
from .oai_proxies import InferenceApiRouter
from .sampling_params import JitableSamplingParams, SamplingParams
from .tools import ToolParser, ToolParserManager
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
    "vWhisperInference",
    "vWhisperInferenceConfig",
)
