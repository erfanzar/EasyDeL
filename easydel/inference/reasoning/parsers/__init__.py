# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model-specific reasoning parsers for EasyDeL inference."""

from .deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from .deepseek_v3_reasoning_parser import DeepSeekV3ReasoningParser
from .ernie45_reasoning_parser import Ernie45ReasoningParser
from .gptoss_reasoning_parser import GptOssReasoningParser
from .granite_reasoning_parser import GraniteReasoningParser
from .hunyuan_a13b_reasoning_parser import HunyuanA13BReasoningParser
from .identity_reasoning_parser import IdentityReasoningParser
from .minimax_m2_reasoning_parser import MiniMaxM2AppendThinkReasoningParser, MiniMaxM2ReasoningParser
from .mistral_reasoning_parser import MistralReasoningParser
from .olmo3_reasoning_parser import Olmo3ReasoningParser
from .qwen3_reasoning_parser import Qwen3ReasoningParser
from .seedoss_reasoning_parser import SeedOSSReasoningParser
from .step3_reasoning_parser import Step3ReasoningParser
from .step3p5_reasoning_parser import Step3p5ReasoningParser

__all__ = [
    "DeepSeekR1ReasoningParser",
    "DeepSeekV3ReasoningParser",
    "Ernie45ReasoningParser",
    "GptOssReasoningParser",
    "GraniteReasoningParser",
    "HunyuanA13BReasoningParser",
    "IdentityReasoningParser",
    "MiniMaxM2AppendThinkReasoningParser",
    "MiniMaxM2ReasoningParser",
    "MistralReasoningParser",
    "Olmo3ReasoningParser",
    "Qwen3ReasoningParser",
    "SeedOSSReasoningParser",
    "Step3ReasoningParser",
    "Step3p5ReasoningParser",
]
