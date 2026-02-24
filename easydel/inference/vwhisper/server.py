# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""FastAPI server for vWhisper transcription and translation.

This module provides a REST API server for audio transcription and
translation using vWhisper, compatible with OpenAI's Whisper API format.
It includes endpoints for both transcription (speech-to-text in original
language) and translation (speech-to-English).

Classes:
    WhisperModel: Singleton model wrapper for efficient model reuse.
    ResponseFormat: Enum for supported response formats.
    TranscriptionResponse: Pydantic model for API responses.

Functions:
    create_whisper_app: Factory to create configured FastAPI application.
    run_server: Utility to launch the server with uvicorn.

Example:
    Running the server::

        >>> from easydel.inference.vwhisper.server import run_server
        >>> run_server(
        ...     model_name="openai/whisper-base",
        ...     host="0.0.0.0",
        ...     port=8000
        ... )

    Using the API with curl::

        $ curl -X POST http://localhost:8000/v1/audio/transcriptions \
            -H "Content-Type: multipart/form-data" \
            -F "file=@audio.mp3" \
            -F "language=en"

    Using the API with Python requests::

        >>> import requests
        >>> with open("audio.mp3", "rb") as f:
        ...     response = requests.post(
        ...         "http://localhost:8000/v1/audio/transcriptions",
        ...         files={"file": f},
        ...         data={"language": "en"}
        ...     )
        >>> print(response.json()["text"])

Note:
    This module requires FastAPI, uvicorn, and pydantic to be installed.
    These are optional dependencies of EasyDeL.
"""

import os
import tempfile
import typing as tp
from enum import StrEnum

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from jax import numpy as jnp
from pydantic import BaseModel, Field
from transformers import WhisperProcessor, WhisperTokenizer

import easydel as ed

from .core import vWhisperInference


class WhisperModel:
    """Singleton wrapper for the Whisper model.

    This class implements the singleton pattern to ensure only one instance
    of the Whisper model is loaded in memory, avoiding redundant loading
    when handling multiple API requests.

    The singleton pattern is particularly important for ML model serving
    because:
        - Model loading is expensive (time and memory)
        - Multiple requests should share the same model
        - Prevents OOM errors from multiple model instances

    Attributes:
        _instance (WhisperModel | None): Class-level singleton instance.
        model_name (str): HuggingFace model identifier.
        dtype: JAX data type for model weights.
        model: The loaded EasyDeL Whisper model.
        tokenizer (WhisperTokenizer): Tokenizer for text processing.
        processor (WhisperProcessor): Processor for audio features.
        inference (vWhisperInference): Configured inference engine.

    Example:
        Creating and using the singleton::

            >>> model = WhisperModel(
            ...     model_name="openai/whisper-base",
            ...     dtype=jnp.bfloat16
            ... )
            >>> # Second call returns the same instance
            >>> model2 = WhisperModel()
            >>> assert model is model2

        Using for inference::

            >>> result = model.inference(
            ...     audio_input="speech.mp3",
            ...     language="en"
            ... )
            >>> print(result["text"])

    Note:
        - The model is only initialized on the first call with model_name
        - Subsequent calls without model_name return the existing instance
        - Call with model_name=None after initialization returns existing instance
    """

    _instance = None

    def __new__(cls, model_name=None, dtype=jnp.bfloat16):
        """Create or return the singleton WhisperModel instance.

        Implements the singleton pattern to ensure only one model instance
        exists. The model is initialized on the first call when model_name
        is provided.

        Args:
            model_name (str | None, optional): HuggingFace model identifier
                (e.g., "openai/whisper-base"). Required for first instantiation,
                optional for subsequent calls. Defaults to None.
            dtype: JAX data type for model weights. Supports jnp.float32,
                jnp.float16, and jnp.bfloat16. Defaults to jnp.bfloat16.

        Returns:
            WhisperModel: The singleton instance. Returns None if called
                without model_name when no instance exists.

        Example:
            First call initializes the model::

                >>> model = WhisperModel("openai/whisper-base")

            Subsequent calls return the same instance::

                >>> same_model = WhisperModel()
                >>> assert model is same_model
        """
        if cls._instance is None and model_name is not None:
            cls._instance = super().__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.dtype = dtype
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize model components.

        Loads the Whisper model, tokenizer, and processor from HuggingFace,
        then creates the vWhisperInference engine. This method is called
        automatically during singleton creation.

        The initialization process:
            1. Loads the model using AutoEasyDeLModelForSpeechSeq2Seq
            2. Loads the WhisperTokenizer for text encoding/decoding
            3. Loads the WhisperProcessor for audio feature extraction
            4. Creates vWhisperInference with all components

        Raises:
            Exception: If model loading fails (e.g., invalid model name,
                network error, insufficient memory).

        Note:
            This method prints a loading message to stdout for user feedback.
            Model loading can take significant time depending on model size
            and network speed.
        """
        print(f"Loading model: {self.model_name}")
        self.model = ed.AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            param_dtype=self.dtype,
        )
        self.tokenizer = WhisperTokenizer.from_pretrained(self.model_name)
        self.processor = WhisperProcessor.from_pretrained(self.model_name)

        self.inference = vWhisperInference(
            model=self.model,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dtype=self.dtype,
        )


class ResponseFormat(StrEnum):
    """Supported response formats for transcription API.

    This enum defines the output formats supported by the transcription
    and translation endpoints, matching OpenAI's Whisper API specification.

    Attributes:
        json: JSON format with text and optional segments.
        text: Plain text format, just the transcription.
        srt: SubRip subtitle format for video subtitles.
        verbose_json: Detailed JSON with word-level timestamps.
        vtt: WebVTT subtitle format for web videos.

    Example:
        Using in API request::

            >>> response = requests.post(
            ...     "http://localhost:8000/v1/audio/transcriptions",
            ...     files={"file": audio_file},
            ...     data={"response_format": "json"}
            ... )

    Note:
        Currently, only "json" and "text" formats are fully implemented.
        Other formats return JSON as a fallback.
    """

    json = "json"
    text = "text"
    srt = "srt"
    verbose_json = "verbose_json"
    vtt = "vtt"


class TranscriptionResponse(BaseModel):
    """Pydantic model for transcription API responses.

    Defines the schema for successful transcription responses, providing
    type validation and automatic documentation generation for the API.

    Attributes:
        text (str): The transcribed text content. This field is required
            and contains the main transcription output.
        segments (list[dict[str, tp.Any]] | None): Optional list of
            transcript segments with timing information. Each segment
            contains text and timestamp data. Only present when
            timestamp_granularities is requested.

    Example:
        Response structure::

            {
                "text": "Hello, how are you?",
                "segments": [
                    {
                        "text": "Hello,",
                        "timestamp": [0.0, 0.5]
                    },
                    {
                        "text": "how are you?",
                        "timestamp": [0.5, 1.2]
                    }
                ]
            }
    """

    text: str = Field(..., description="The transcribed text")
    segments: list[dict[str, tp.Any]] | None = Field(None, description="Segments with timestamps")


def create_whisper_app(model_name: str = "openai/whisper-large-v3-turbo", dtype=jnp.bfloat16):
    """Create a FastAPI application for Whisper transcription.

    Factory function that creates and configures a FastAPI application
    with endpoints for audio transcription and translation. The app
    is compatible with OpenAI's Whisper API format.

    The created application includes:
        - CORS middleware for cross-origin requests
        - Root endpoint for health checks
        - /v1/audio/transcriptions for speech-to-text
        - /v1/audio/translations for speech-to-English translation

    Args:
        model_name (str, optional): HuggingFace model identifier for the
            Whisper model to load. Defaults to "openai/whisper-large-v3-turbo".
        dtype: JAX data type for model computations. Supports jnp.float32,
            jnp.float16, and jnp.bfloat16. Defaults to jnp.bfloat16.

    Returns:
        FastAPI: Configured FastAPI application instance ready to be
            served with uvicorn or another ASGI server.

    Example:
        Creating and running the app::

            >>> app = create_whisper_app("openai/whisper-base")
            >>> import uvicorn
            >>> uvicorn.run(app, host="0.0.0.0", port=8000)

        Using with custom configuration::

            >>> app = create_whisper_app(
            ...     model_name="openai/whisper-large-v3-turbo",
            ...     dtype=jnp.float16
            ... )

    Note:
        - Model loading happens during app creation, which may take time
        - The app uses a singleton WhisperModel to avoid reloading
        - CORS is configured to allow all origins for development
    """
    app = FastAPI(
        title="EasyDeL Whisper API",
        description="API for Whisper ASR model powered by EasyDeL",
        version="1.0.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize the model
    model_instance = WhisperModel(model_name=model_name, dtype=dtype)

    @app.get("/")
    def read_root():
        """Root endpoint for health check and API information.

        Returns:
            dict: A dictionary containing:
                - "message": Welcome message identifying the API
                - "model": Name of the loaded Whisper model
        """
        return {"message": "EasyDeL Whisper API", "model": model_instance.model_name}

    @app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
    async def create_transcription(
        file: UploadFile = File(...),
        model: str = Form(model_name),
        prompt: str | None = Form(None),
        response_format: ResponseFormat = Form(ResponseFormat.json),
        temperature: float = Form(0.0),
        language: str | None = Form(None),
        timestamp_granularities: tp.Optional[tp.List[str]] = Form(None),  # noqa
    ):
        """Transcribe audio to text using the Whisper model.

        This endpoint accepts audio files and returns the transcribed text.
        It is compatible with OpenAI's Whisper transcription API format.

        Args:
            file (UploadFile): The audio file to transcribe. Supports
                formats like mp3, wav, m4a, webm, etc.
            model (str): Model identifier (for API compatibility, not
                used as model is pre-loaded). Defaults to the loaded model.
            prompt (str | None): Optional text to guide the model's style.
                Currently not implemented.
            response_format (ResponseFormat): Output format. Options are
                "json", "text", "srt", "verbose_json", "vtt".
                Defaults to "json".
            temperature (float): Sampling temperature. Currently not
                implemented. Defaults to 0.0.
            language (str | None): Language code for the audio (e.g., "en",
                "fr"). If None, language may be auto-detected.
            timestamp_granularities (list[str] | None): Timestamp detail
                level. Include "word" for word-level timestamps.

        Returns:
            TranscriptionResponse: Dictionary containing:
                - "text": Transcribed text
                - "segments": List of segments (if timestamps requested)

        Raises:
            HTTPException: 500 error if transcription fails.

        Example:
            Using curl::

                $ curl -X POST http://localhost:8000/v1/audio/transcriptions \
                    -F "file=@speech.mp3" \
                    -F "language=en" \
                    -F "response_format=json"

            Response::

                {
                    "text": "Hello, this is a test transcription."
                }
        """
        try:
            # Create a temporary file to store the uploaded audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                # Read the uploaded file
                audio_content = await file.read()
                # Write to the temporary file
                temp_file.write(audio_content)
                temp_file_path = temp_file.name

            # Get timestamps based on granularities
            return_timestamps = False
            if timestamp_granularities and "word" in timestamp_granularities:
                return_timestamps = True

            # Process the audio with Whisper
            result = model_instance.inference(
                audio_input=temp_file_path,
                language=language,
                return_timestamps=return_timestamps,
            )

            # Format the response based on the requested format
            if response_format == ResponseFormat.text:
                return {"text": result["text"]}
            elif response_format == ResponseFormat.json:
                response = {"text": result["text"]}
                if "chunks" in result:
                    response["segments"] = result["chunks"]
                return response
            else:
                # For other formats, return JSON for now (can be extended)
                return {"text": result["text"]}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing audio: {e!s}") from e

        finally:
            # Clean up the temporary file
            if "temp_file_path" in locals():
                os.unlink(temp_file_path)

    @app.post("/v1/audio/translations")
    async def create_translation(
        file: UploadFile = File(...),
        model: str = Form(model_name),
        prompt: tp.Optional[str] = Form(None),  # noqa
        response_format: ResponseFormat = Form(ResponseFormat.json),
        temperature: float = Form(0.0),
        timestamp_granularities: tp.Optional[tp.List[str]] = Form(None),  # noqa
    ):
        """Translate audio to English text using the Whisper model.

        This endpoint accepts audio files in any supported language and
        returns the English translation. It is compatible with OpenAI's
        Whisper translation API format.

        Args:
            file (UploadFile): The audio file to translate. Supports
                formats like mp3, wav, m4a, webm, etc.
            model (str): Model identifier (for API compatibility, not
                used as model is pre-loaded). Defaults to the loaded model.
            prompt (str | None): Optional text to guide the model's style.
                Currently not implemented.
            response_format (ResponseFormat): Output format. Options are
                "json", "text", "srt", "verbose_json", "vtt".
                Defaults to "json".
            temperature (float): Sampling temperature. Currently not
                implemented. Defaults to 0.0.
            timestamp_granularities (list[str] | None): Timestamp detail
                level. Include "word" for word-level timestamps.

        Returns:
            dict: Dictionary containing:
                - "text": Translated English text
                - "segments": List of segments (if timestamps requested)

        Raises:
            HTTPException: 500 error if translation fails.

        Example:
            Translating French audio to English::

                $ curl -X POST http://localhost:8000/v1/audio/translations \
                    -F "file=@french_speech.mp3"

            Response::

                {
                    "text": "Hello, this is the English translation."
                }

        Note:
            Unlike transcription, translation always outputs English
            regardless of the source language.
        """
        try:
            # Create a temporary file to store the uploaded audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                # Read the uploaded file
                audio_content = await file.read()
                # Write to the temporary file
                temp_file.write(audio_content)
                temp_file_path = temp_file.name

            # Get timestamps based on granularities
            return_timestamps = False
            if timestamp_granularities and "word" in timestamp_granularities:
                return_timestamps = True

            # Process the audio with Whisper
            result = model_instance.inference(
                audio_input=temp_file_path,
                task="translate",
                return_timestamps=return_timestamps,
            )

            # Format the response based on the requested format
            if response_format == ResponseFormat.text:
                return {"text": result["text"]}
            elif response_format == ResponseFormat.json:
                response = {"text": result["text"]}
                if "chunks" in result:
                    response["segments"] = result["chunks"]
                return response
            else:
                # For other formats, return JSON for now (can be extended)
                return {"text": result["text"]}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")  # noqa

        finally:
            # Clean up the temporary file
            if "temp_file_path" in locals():
                os.unlink(temp_file_path)

    return app


def run_server(
    model_name: str = "openai/whisper-large-v3-turbo",
    host: str = "0.0.0.0",
    port: int = 8000,
    dtype=jnp.bfloat16,
):
    """Run the vWhisper FastAPI server.

    Convenience function to create and run the Whisper API server with
    uvicorn. This is the recommended way to start the server for
    production use.

    Args:
        model_name (str, optional): HuggingFace model identifier for the
            Whisper model. Common options include:
            - "openai/whisper-tiny" (fastest, lowest quality)
            - "openai/whisper-base"
            - "openai/whisper-small"
            - "openai/whisper-medium"
            - "openai/whisper-large-v3" (slowest, highest quality)
            - "openai/whisper-large-v3-turbo" (optimized large model)
            Defaults to "openai/whisper-large-v3-turbo".
        host (str, optional): Host address to bind the server. Use
            "0.0.0.0" to accept connections from any address, or
            "127.0.0.1" for localhost only. Defaults to "0.0.0.0".
        port (int, optional): Port number to listen on.
            Defaults to 8000.
        dtype: JAX data type for model computations. Options:
            - jnp.float32: Full precision (most compatible)
            - jnp.float16: Half precision (faster, less memory)
            - jnp.bfloat16: Brain float (good balance)
            Defaults to jnp.bfloat16.

    Returns:
        None: This function blocks until the server is stopped.

    Example:
        Basic usage::

            >>> from easydel.inference.vwhisper import run_server
            >>> run_server()  # Uses defaults

        Custom configuration::

            >>> run_server(
            ...     model_name="openai/whisper-small",
            ...     host="127.0.0.1",
            ...     port=9000,
            ...     dtype=jnp.float32
            ... )

    Note:
        - The server blocks until interrupted (Ctrl+C)
        - Model loading happens during startup
        - API documentation is available at http://host:port/docs
    """
    app = create_whisper_app(model_name=model_name, dtype=dtype)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
