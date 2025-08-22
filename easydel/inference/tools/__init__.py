# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.

"""EasyDeL Inference Tools Module.

This module provides the core infrastructure for parsing and extracting tool/function
calls from Large Language Model outputs. It includes:

Core Components:
    ToolParser: Abstract base class for all tool parsers
    ToolParserManager: Registry for managing parser implementations
    ToolCallingMixin: Mixin class for adding tool calling to inference servers

Available Parsers:
    Each parser is designed for specific model families and their unique
    tool calling formats. See parsers submodule for the complete list.

Usage:
    ```python
    from easydel.inference.tools import ToolParserManager, HermesToolParser

    # Get a parser by name
    parser_class = ToolParserManager.get_tool_parser("hermes")
    parser = parser_class(tokenizer)

    # Extract tool calls from model output
    result = parser.extract_tool_calls(model_output, request)
    ```
"""

# ideas are coming from vLLM.

from .abstract_tool import ToolParser, ToolParserManager
from .parsers import (
    DeepSeekV3ToolParser,
    Glm4MoeModelToolParser,
    Granite20bFCToolParser,
    GraniteToolParser,
    HermesToolParser,
    HunyuanA13BToolParser,
    Internlm2ToolParser,
    JambaToolParser,
    KimiK2ToolParser,
    Llama3JsonToolParser,
    Llama4PythonicToolParser,
    MinimaxToolParser,
    MistralToolParser,
    Phi4MiniJsonToolParser,
    PythonicToolParser,
    Qwen3CoderToolParser,
    SeedOssToolParser,
    Step3ToolParser,
)
from .tool_calling_mixin import ToolCallingMixin

__all__ = (
    "DeepSeekV3ToolParser",
    "Glm4MoeModelToolParser",
    "Granite20bFCToolParser",
    "GraniteToolParser",
    "HermesToolParser",
    "HermesToolParser",
    "HunyuanA13BToolParser",
    "Internlm2ToolParser",
    "JambaToolParser",
    "KimiK2ToolParser",
    "Llama3JsonToolParser",
    "Llama4PythonicToolParser",
    "MinimaxToolParser",
    "MistralToolParser",
    "Phi4MiniJsonToolParser",
    "PythonicToolParser",
    "Qwen3CoderToolParser",
    "SeedOssToolParser",
    "Step3ToolParser",
    "ToolCallingMixin",
    "ToolParser",
    "ToolParserManager",
)
