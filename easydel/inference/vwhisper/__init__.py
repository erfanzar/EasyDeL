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

"""vWhisper: High-performance speech recognition and transcription engine.

This module provides the vWhisper inference engine, a JAX/XLA-optimized
implementation for speech-to-text transcription and translation using
OpenAI's Whisper models. vWhisper is designed for high-throughput
audio processing with support for long-form audio, timestamp generation,
and multiple languages.

Key Features:
    - Fast audio transcription with JAX/XLA acceleration
    - Audio-to-English translation capabilities
    - Support for long-form audio with automatic chunking
    - Timestamp generation for subtitle creation (SRT, VTT)
    - Multi-language support for transcription
    - Streaming processing for real-time applications
    - REST API server compatible with OpenAI's Whisper API
    - Command-line interface for easy usage

Components:
    vWhisperInference:
        Main inference engine for transcription and translation.
        Handles audio processing, chunking, and model inference.

    vWhisperInferenceConfig:
        Configuration class for controlling inference behavior
        including batch size, max length, and timestamp settings.

    WhisperModel:
        Singleton model wrapper used by the API server to avoid
        reloading the model on each request.

    create_whisper_app:
        Factory function to create a FastAPI application for
        serving Whisper transcription/translation endpoints.

    run_server:
        Utility function to launch the FastAPI server with
        configurable host, port, and model settings.

    run_cli:
        Command-line interface entry point for running the
        vWhisper server from the terminal.

Utility Functions:
    chunk_iter_with_batch:
        Generator for processing long audio into overlapping chunks.

    process_audio_input:
        Flexible audio input processor supporting files, URLs,
        bytes, and numpy arrays.

    get_decoder_input_ids:
        Helper to construct decoder input IDs for language/task tokens.

    _compiled_generate:
        JIT-compiled generation function for efficient inference.

Example:
    Basic transcription usage::

        >>> from easydel.inference.vwhisper import (
        ...     vWhisperInference,
        ...     vWhisperInferenceConfig
        ... )
        >>> from transformers import WhisperProcessor, WhisperTokenizer
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
        >>> # Transcribe audio file
        >>> result = engine.transcribe(
        ...     "path/to/audio.mp3",
        ...     language="en",
        ...     return_timestamps=True
        ... )
        >>> print(result["text"])

    Running the API server::

        >>> from easydel.inference.vwhisper import run_server
        >>> run_server(
        ...     model_name="openai/whisper-base",
        ...     host="0.0.0.0",
        ...     port=8000
        ... )

CLI Usage:
    Run the vWhisper server from command line::

        $ python -m easydel.inference.vwhisper.cli \
            --model-id openai/whisper-base \
            --host 0.0.0.0 \
            --port 8000 \
            --dtype bfloat16

Note:
    vWhisper requires additional audio processing libraries for full
    functionality:
        - librosa: For audio resampling
        - soundfile: For audio file I/O
        - ffmpeg: For audio format conversion (via transformers)

    The API server requires FastAPI and uvicorn to be installed.

See Also:
    - OpenAI Whisper: https://github.com/openai/whisper
    - Hugging Face Transformers Whisper: https://huggingface.co/docs/transformers/model_doc/whisper
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
