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

"""
Tool Parsers Module for EasyDeL Inference.

This module provides a collection of specialized tool parsers for extracting and processing
function/tool calls from various Large Language Model (LLM) outputs. Each parser is designed
to handle the specific output format of different model families.

Available Parsers:
-----------------
- DeepSeekV3ToolParser: Handles DeepSeek V3 model tool call format
- Glm4MoeModelToolParser: Processes GLM-4 MoE model tool invocations
- Granite20bFCToolParser: Parser for Granite 20B function calling model
- GraniteToolParser: General Granite 3.0 model tool parser
- HermesToolParser: Processes Hermes-style tool calls
- HunyuanA13BToolParser: Handles Hunyuan A13B model tool outputs
- Internlm2ToolParser: Parser for InternLM2 action/plugin calls
- JambaToolParser: Processes Jamba model tool invocations
- KimiK2ToolParser: Handles Kimi K2 model tool call sections
- Llama4PythonicToolParser: Processes Llama 4 pythonic-style tool calls
- Llama3JsonToolParser: Handles Llama 3.x/4 JSON-formatted tool calls
- MinimaxToolParser: Parser for Minimax model tool invocations
- MistralToolParser: Processes Mistral model tool calls
- Phi4MiniJsonToolParser: Handles Phi-4-mini model functools format
- PythonicToolParser: General pythonic-style tool call parser
- Qwen3CoderToolParser: Processes Qwen3 Coder XML-style tool calls
- SeedOssToolParser: Handles Seed OSS model XML tool format
- Step3ToolParser: Processes Step3 model XML-like tool calls

All parsers inherit from the base ToolParser class and are registered with
the ToolParserManager for automatic selection based on model type.
"""

from .deepseekv3_tool_parser import DeepSeekV3ToolParser
from .glm4_moe_tool_parser import Glm4MoeModelToolParser
from .granite_20b_fc_tool_parser import Granite20bFCToolParser
from .granite_tool_parser import GraniteToolParser
from .hermes_tool_parser import HermesToolParser
from .hunyuan_a13b_tool_parser import HunyuanA13BToolParser
from .internlm2_tool_parser import Internlm2ToolParser
from .jamba_tool_parser import JambaToolParser
from .kimi_k2_tool_parser import KimiK2ToolParser
from .llama4_pythonic_tool_parser import Llama4PythonicToolParser
from .llama_tool_parser import Llama3JsonToolParser
from .minimax_tool_parser import MinimaxToolParser
from .mistral_tool_parser import MistralToolParser
from .phi4mini_tool_parser import Phi4MiniJsonToolParser
from .pythonic_tool_parser import PythonicToolParser
from .qwen3coder_tool_parser import Qwen3CoderToolParser
from .seed_oss_tool_parser import SeedOssToolParser
from .step3_tool_parser import Step3ToolParser

__all__ = [
    "DeepSeekV3ToolParser",
    "Glm4MoeModelToolParser",
    "Granite20bFCToolParser",
    "GraniteToolParser",
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
]
