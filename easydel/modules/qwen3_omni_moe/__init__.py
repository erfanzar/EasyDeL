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

"""Qwen3OmniMoe - Multimodal MoE model with audio, vision, and text.

This module implements the Qwen3OmniMoe architecture, which combines:

**Thinker** (Multimodal Understanding):
- Audio encoder for processing mel-spectrogram inputs
- Vision encoder for processing images and videos
- MoE text decoder for text generation

**Talker** (Speech Generation):
- Code predictor for acoustic token prediction
- MoE text model with shared experts for speech generation

**Code2Wav** (Vocoder):
- Transformer with LayerScale for codec-to-waveform conversion
- ConvNeXt upsampling blocks

Example:
    >>> from easydel.modules.qwen3_omni_moe import (
    ...     Qwen3OmniMoeConfig,
    ...     Qwen3OmniMoeForConditionalGeneration,
    ... )
    >>> config = Qwen3OmniMoeConfig()
    >>> model = Qwen3OmniMoeForConditionalGeneration(
    ...     config=config,
    ...     dtype=jnp.bfloat16,
    ... )
"""

from .modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeModel,
    Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration,
    Qwen3OmniMoeTalkerCodePredictorModel,
    Qwen3OmniMoeTalkerForConditionalGeneration,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerModel,
    Qwen3OmniMoeVisionEncoder,
)
from .qwen3_omni_moe_configuration import (
    Qwen3OmniMoeAudioConfig,
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeTalkerTextConfig,
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeVisionConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)

__all__ = (
    "Qwen3OmniMoeAudioConfig",
    "Qwen3OmniMoeAudioEncoder",
    "Qwen3OmniMoeAudioEncoderConfig",
    "Qwen3OmniMoeCode2Wav",
    "Qwen3OmniMoeCode2WavConfig",
    "Qwen3OmniMoeConfig",
    "Qwen3OmniMoeForConditionalGeneration",
    "Qwen3OmniMoeModel",
    "Qwen3OmniMoeTalkerCodePredictorConfig",
    "Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration",
    "Qwen3OmniMoeTalkerCodePredictorModel",
    "Qwen3OmniMoeTalkerConfig",
    "Qwen3OmniMoeTalkerForConditionalGeneration",
    "Qwen3OmniMoeTalkerTextConfig",
    "Qwen3OmniMoeTextConfig",
    "Qwen3OmniMoeThinkerConfig",
    "Qwen3OmniMoeThinkerForConditionalGeneration",
    "Qwen3OmniMoeThinkerModel",
    "Qwen3OmniMoeVisionConfig",
    "Qwen3OmniMoeVisionEncoder",
    "Qwen3OmniMoeVisionEncoderConfig",
)
