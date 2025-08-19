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

"""vWhisper: Speech recognition and transcription engine.

vWhisper provides high-performance speech-to-text transcription using
OpenAI's Whisper models, optimized for JAX/XLA acceleration.

Key Features:
    - Fast audio transcription and translation
    - Support for long-form audio with chunking
    - Timestamp generation for subtitles
    - Multiple language support
    - Streaming processing for real-time applications
    - REST API server for easy integration

Components:
    vWhisperInference: Main inference engine
    vWhisperInferenceConfig: Configuration settings
    WhisperModel: Model wrapper for API server
    create_whisper_app: FastAPI application factory
    run_server: Launch API server
    run_cli: Command-line interface

Example:
    >>> from easydel.inference.vwhisper import (
    ...     vWhisperInference,
    ...     vWhisperInferenceConfig
    ... )
    >>> from transformers import (
    ...     WhisperProcessor,
    ...     WhisperTokenizer
    ... )
    >>>
    >>> # Load model and processor
    >>> processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    >>> tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
    >>>
    >>> # Initialize inference engine
    >>> config = vWhisperInferenceConfig(
    ...     max_length=448,
    ...     batch_size=8
    ... )
    >>> engine = vWhisperInference(
    ...     model=whisper_model,
    ...     tokenizer=tokenizer,
    ...     processor=processor,
    ...     inference_config=config
    ... )
    >>>
    >>> # Transcribe audio
    >>> result = engine.transcribe(
    ...     "path/to/audio.mp3",
    ...     language="en",
    ...     return_timestamps=True
    ... )
    >>> print(result["text"])

CLI Usage:
    $ python -m easydel.inference.vwhisper.cli \
        --model-id openai/whisper-base \
        --host 0.0.0.0 \
        --port 8000

Note:
    vWhisper requires audio processing libraries like librosa
    and soundfile for full functionality.
"""

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
