# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

from .config import vWhisperInferenceConfig
from .core import vWhisperInference
from .generation import _compiled_generate, get_decoder_input_ids
from .utils import chunk_iter_with_batch, process_audio_input

# Import server-related functionality
try:
    from .cli import main as run_cli
    from .server import WhisperModel, create_whisper_app, run_server

    __all__ = (
        "WhisperModel",
        "_compiled_generate",
        "chunk_iter_with_batch",
        "create_whisper_app",
        "get_decoder_input_ids",
        "process_audio_input",
        "run_cli",
        "run_server",
        "vWhisperInference",
        "vWhisperInferenceConfig",
    )
except ImportError:
    # If FastAPI dependencies are not available, don't expose server functionality
    __all__ = (
        "_compiled_generate",
        "chunk_iter_with_batch",
        "get_decoder_input_ids",
        "process_audio_input",
        "vWhisperInference",
        "vWhisperInferenceConfig",
    )
