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

from __future__ import annotations

import typing as tp

from eformer.pytree import auto_pytree

from easydel.utils.compiling_utils import get_safe_hash_int

if tp.TYPE_CHECKING:
    from transformers import GenerationConfig
else:
    GenerationConfig = tp.Any


@auto_pytree
class vWhisperInferenceConfig:
    """
    Configuration class for Whisper inference.

    Args:
        batch_size (`int`, *optional*, defaults to 1):
            Batch size used for inference.
        max_length (`int`, *optional*):
            Maximum sequence length for generation.
        generation_config (`transformers.GenerationConfig`, *optional*):
            Generation configuration object.
        logits_processor (*optional*): Not used.
        return_timestamps (`bool`, *optional*):
            Whether to return timestamps with the transcribed text.
        task (`str`, *optional*):
            Task for the model (e.g., "transcribe", "translate").
        language (`str`, *optional*):
            Language of the input audio.
        is_multilingual (`bool`, *optional*):
            Whether the model is multilingual.
    """

    batch_size: int | None = 1
    max_length: int | None = None
    generation_config: GenerationConfig | None = None
    logits_processor = None
    return_timestamps = None
    task = None
    language = None
    is_multilingual = None

    def __hash__(self):
        return get_safe_hash_int("".join(str(k) + str(v) for k, v in self.__dict__.items()))
