# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.

"""EasyDeL Inference Tools Module.

This module provides the core infrastructure for parsing and extracting tool/function
calls from Large Language Model outputs. It serves as the main entry point for all
tool calling functionality in EasyDeL inference servers.

The module includes:

Core Components:
    ToolParser: Abstract base class for all tool parsers. Provides the interface
        for extracting tool calls from both batch and streaming model outputs.
    ToolParserManager: Registry for managing parser implementations. Allows
        registration and retrieval of parser classes by name.
    ToolCallingMixin: Mixin class for adding tool calling capabilities to
        inference servers. Provides methods for parser initialization and
        tool call extraction.

Available Parsers:
    The module exports numerous model-specific parsers for different tool calling
    formats. Each parser is designed to handle the unique syntax and conventions
    used by specific model families:

    JSON-based parsers:
        - OpenAIToolParser: Standard OpenAI function calling format
        - Llama3JsonToolParser: Llama 3 JSON tool format
        - Phi4MiniJsonToolParser: Phi-4 mini JSON format

    XML-based parsers:
        - HermesToolParser: Hermes/Nous Research XML format
        - Qwen3XMLToolParser: Qwen 3 XML format

    Pythonic parsers:
        - PythonicToolParser: Generic Python function call format
        - Llama4PythonicToolParser: Llama 4 pythonic format
        - Olmo3PythonicToolParser: OLMo 3 pythonic format

    Model-specific parsers:
        - DeepSeekV3ToolParser, DeepSeekV31ToolParser, DeepSeekV32ToolParser
        - MistralToolParser
        - GraniteToolParser, Granite20bFCToolParser
        - JambaToolParser
        - Internlm2ToolParser
        - And many more...

Usage:
    Basic usage for extracting tool calls from model output:

    >>> from easydel.inference.tools import ToolParserManager, HermesToolParser
    >>>
    >>> # Get a parser by name
    >>> parser_class = ToolParserManager.get_tool_parser("hermes")
    >>> parser = parser_class(tokenizer)
    >>>
    >>> # Extract tool calls from model output
    >>> result = parser.extract_tool_calls(model_output, request)
    >>> if result.tools_called:
    ...     for tool_call in result.tool_calls:
    ...         print(f"Function: {tool_call.function.name}")
    ...         print(f"Arguments: {tool_call.function.arguments}")

    Using the mixin in an inference server:

    >>> from easydel.inference.tools import ToolCallingMixin
    >>>
    >>> class MyInferenceServer(ToolCallingMixin):
    ...     def __init__(self, tokenizer, tool_parser_name="hermes"):
    ...         self.tool_parsers = self.initialize_tool_parsers(
    ...             {"model": tokenizer},
    ...             tool_parser_name,
    ...             enable_function_calling=True
    ...         )

Note:
    Ideas and design patterns are inspired by vLLM's tool calling implementation.

See Also:
    - easydel.inference.tools.abstract_tool: Base classes for tool parsers
    - easydel.inference.tools.parsers: Model-specific parser implementations
    - easydel.inference.tools.tool_calling_mixin: Mixin for inference servers
    - easydel.inference.tools.utils: Utility functions for parsing
"""

# ideas are coming from vLLM.

from .abstract_tool import ToolParser, ToolParserManager
from .parsers import (
    DeepSeekV3ToolParser,
    DeepSeekV31ToolParser,
    DeepSeekV32ToolParser,
    Ernie45ToolParser,
    FunctionGemmaToolParser,
    GigaChat3ToolParser,
    Glm4MoeModelToolParser,
    Glm47MoeModelToolParser,
    Granite20bFCToolParser,
    GraniteToolParser,
    HermesToolParser,
    HunyuanA13BToolParser,
    Internlm2ToolParser,
    JambaToolParser,
    KimiK2ToolParser,
    Llama3JsonToolParser,
    Llama4PythonicToolParser,
    LongcatFlashToolParser,
    MinimaxM2ToolParser,
    MinimaxToolParser,
    MistralToolParser,
    Olmo3PythonicToolParser,
    OpenAIToolParser,
    Phi4MiniJsonToolParser,
    PythonicToolParser,
    Qwen3CoderToolParser,
    Qwen3XMLToolParser,
    SeedOssToolParser,
    Step3ToolParser,
    xLAMToolParser,
)
from .tool_calling_mixin import ToolCallingMixin

__all__ = (
    "DeepSeekV3ToolParser",
    "DeepSeekV31ToolParser",
    "DeepSeekV32ToolParser",
    "Ernie45ToolParser",
    "FunctionGemmaToolParser",
    "GigaChat3ToolParser",
    "Glm4MoeModelToolParser",
    "Glm47MoeModelToolParser",
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
    "LongcatFlashToolParser",
    "MinimaxM2ToolParser",
    "MinimaxToolParser",
    "MistralToolParser",
    "Olmo3PythonicToolParser",
    "OpenAIToolParser",
    "Phi4MiniJsonToolParser",
    "PythonicToolParser",
    "Qwen3CoderToolParser",
    "Qwen3XMLToolParser",
    "SeedOssToolParser",
    "Step3ToolParser",
    "ToolCallingMixin",
    "ToolParser",
    "ToolParserManager",
    "xLAMToolParser",
)
