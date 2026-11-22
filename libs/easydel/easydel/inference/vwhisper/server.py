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

import asyncio
import os
import tempfile
import threading
import typing as tp
from contextlib import asynccontextmanager
from enum import StrEnum

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from jax import numpy as jnp
from pydantic import BaseModel, Field
from transformers import WhisperProcessor, WhisperTokenizer

import easydel as ed

from .core import vWhisperInference

DEFAULT_MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MiB
MAX_UPLOAD_BYTES_ENV_VAR = "VWHISPER_MAX_UPLOAD_BYTES"
_UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def _resolve_max_upload_bytes(max_upload_bytes: int | None = None) -> int:
    """Resolve the upload byte limit from explicit config or environment."""
    if max_upload_bytes is None:
        raw_env = os.environ.get(MAX_UPLOAD_BYTES_ENV_VAR)
        if raw_env is None or not raw_env.strip():
            return DEFAULT_MAX_UPLOAD_BYTES
        try:
            max_upload_bytes = int(raw_env)
        except ValueError as err:
            raise ValueError(f"{MAX_UPLOAD_BYTES_ENV_VAR} must be a positive integer, got {raw_env!r}") from err
    if isinstance(max_upload_bytes, bool) or max_upload_bytes <= 0:
        raise ValueError(f"max_upload_bytes must be a positive integer, got {max_upload_bytes!r}")
    return max_upload_bytes


def _cleanup_temp_file(path: str | None) -> None:
    """Remove a temporary file if it still exists."""
    if path is not None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


@asynccontextmanager
async def _uploaded_temp_file(file: UploadFile, max_upload_bytes: int) -> tp.AsyncIterator[str]:
    """Stream an uploaded file to a temporary file with size enforcement and guaranteed cleanup."""
    suffix = os.path.splitext(os.path.basename(file.filename or ""))[1]
    temp_file = await asyncio.to_thread(tempfile.NamedTemporaryFile, delete=False, suffix=suffix)
    temp_file_path = temp_file.name
    try:
        total_bytes = 0
        try:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Uploaded file exceeds maximum allowed size of {max_upload_bytes} bytes.",
                    )
                await asyncio.to_thread(temp_file.write, chunk)
        finally:
            try:
                await asyncio.to_thread(temp_file.close)
            except BaseException:
                temp_file.close()
                raise
        yield temp_file_path
    finally:
        try:
            await asyncio.to_thread(_cleanup_temp_file, temp_file_path)
        except BaseException:
            _cleanup_temp_file(temp_file_path)
            raise


def _should_return_timestamps(
    response_format: "ResponseFormat",
    timestamp_granularities: list[str] | None,
) -> bool:
    """Determine whether Whisper generation must emit timestamps for the request."""
    if response_format in {ResponseFormat.srt, ResponseFormat.vtt, ResponseFormat.verbose_json}:
        return True
    if not timestamp_granularities:
        return False
    requested = {
        part.strip().lower()
        for item in timestamp_granularities
        for part in item.split(",")
        if part.strip()
    }
    return bool(requested & {"word", "segment"})


def _format_timestamp(seconds: float | None, *, decimal_marker: str) -> str:
    """Format a timestamp in seconds as HH:MM:SS{decimal_marker}mmm."""
    total_ms = round(max(0.0, float(seconds or 0.0)) * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_marker}{millis:03d}"


def _normalize_segments(result: dict[str, tp.Any]) -> list[dict[str, tp.Any]]:
    """Normalize vWhisper chunks into structured OpenAI-compatible segment dicts."""
    chunks = result.get("chunks")
    segments: list[dict[str, tp.Any]] = []
    if chunks:
        for idx, chunk in enumerate(chunks):
            ts = chunk.get("timestamp")
            if isinstance(ts, list | tuple) and len(ts) >= 2:
                start = 0.0 if ts[0] is None else float(ts[0])
                end = start if ts[1] is None else float(ts[1])
            else:
                start = float(chunk.get("start", 0.0) or 0.0)
                end = float(chunk.get("end", start) or start)
            segments.append(
                {
                    "id": idx,
                    "start": start,
                    "end": end,
                    "text": chunk.get("text", ""),
                    "timestamp": [start, end],
                }
            )
    elif result.get("text"):
        segments.append(
            {
                "id": 0,
                "start": 0.0,
                "end": 0.0,
                "text": result["text"],
                "timestamp": [0.0, 0.0],
            }
        )
    return segments


def _format_srt(segments: list[dict[str, tp.Any]]) -> str:
    """Render normalized segments in SubRip (SRT) subtitle format."""
    blocks = []
    for idx, seg in enumerate(segments, start=1):
        start_ts = _format_timestamp(seg["start"], decimal_marker=",")
        end_ts = _format_timestamp(seg["end"], decimal_marker=",")
        blocks.append(f"{idx}\n{start_ts} --> {end_ts}\n{seg['text'].strip()}\n")
    return "\n".join(blocks)


def _format_vtt(segments: list[dict[str, tp.Any]]) -> str:
    """Render normalized segments in WebVTT subtitle format."""
    lines = ["WEBVTT", ""]
    for seg in segments:
        start_ts = _format_timestamp(seg["start"], decimal_marker=".")
        end_ts = _format_timestamp(seg["end"], decimal_marker=".")
        lines.append(f"{start_ts} --> {end_ts}\n{seg['text'].strip()}\n")
    return "\n".join(lines)


def _format_whisper_response(
    result: dict[str, tp.Any],
    response_format: "ResponseFormat",
    *,
    task: str,
    language: str | None = None,
):
    """Format vWhisper inference output according to OpenAI Whisper response_format."""
    text = result.get("text", "")
    if response_format == ResponseFormat.text:
        return PlainTextResponse(content=text)

    if response_format == ResponseFormat.srt:
        return PlainTextResponse(content=_format_srt(_normalize_segments(result)))

    if response_format == ResponseFormat.vtt:
        return PlainTextResponse(content=_format_vtt(_normalize_segments(result)))

    if response_format == ResponseFormat.verbose_json:
        segments = _normalize_segments(result)
        duration = max((seg["end"] for seg in segments), default=0.0)
        resolved_language = language or result.get("language")
        if resolved_language is None:
            for chunk in result.get("chunks") or ():
                if chunk.get("language"):
                    resolved_language = chunk["language"]
                    break
        if resolved_language is None and task == "translate":
            resolved_language = "en"
        return {
            "task": task,
            "language": resolved_language,
            "duration": duration,
            "text": text,
            "segments": segments,
        }

    response: dict[str, tp.Any] = {"text": text}
    if "chunks" in result:
        response["segments"] = result["chunks"]
    return response


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

    _instance: tp.ClassVar["WhisperModel | None"] = None
    _lock: tp.ClassVar[threading.RLock] = threading.RLock()

    def __new__(cls, model_name=None, dtype=jnp.bfloat16):
        """Create or return the singleton WhisperModel instance.

        Implements a thread-safe singleton pattern to ensure only one model
        instance exists. The model is initialized on the first call when
        ``model_name`` is provided, and the singleton reference is only
        published after ``_initialize()`` succeeds.

        Args:
            model_name (str | None, optional): HuggingFace model identifier
                (e.g., "openai/whisper-base"). Required for first instantiation,
                optional for subsequent calls. Defaults to None.
            dtype: JAX data type for model weights. Supports jnp.float32,
                jnp.float16, and jnp.bfloat16. Defaults to jnp.bfloat16.

        Returns:
            WhisperModel | None: The singleton instance, or None if called
                without model_name when no instance exists.

        Raises:
            ValueError: If an instance is already initialized with a different
                ``model_name`` or ``dtype``.
        """
        with cls._lock:
            if cls._instance is not None:
                if model_name is not None and (
                    cls._instance.model_name != model_name or cls._instance.dtype != dtype
                ):
                    raise ValueError(
                        f"WhisperModel singleton is already initialized with "
                        f"model_name={cls._instance.model_name!r}, dtype={cls._instance.dtype!r}; "
                        f"cannot reinitialize with model_name={model_name!r}, dtype={dtype!r}. "
                        f"Call WhisperModel.reset() first to load a different model."
                    )
                return cls._instance

            if model_name is None:
                return None

            instance = super().__new__(cls)
            instance.model_name = model_name
            instance.dtype = dtype
            instance._initialize()
            cls._instance = instance
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the cached singleton instance."""
        with cls._lock:
            cls._instance = None

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
        self.model = ed.AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(  # pyright: ignore[reportPrivateLocalImportUsage]
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
        All formats ("json", "text", "srt", "verbose_json", and "vtt") are
        supported; "text", "srt", and "vtt" are returned as ``text/plain``.
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


def create_whisper_app(
    model_name: str = "openai/whisper-large-v3-turbo",
    dtype=jnp.bfloat16,
    max_upload_bytes: int | None = None,
):
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
        max_upload_bytes (int | None, optional): Maximum upload size in bytes.
            If None, reads ``VWHISPER_MAX_UPLOAD_BYTES`` or defaults to 100 MiB.

    Returns:
        FastAPI: Configured FastAPI application instance ready to be
            served with uvicorn or another ASGI server.
    """
    resolved_max_upload_bytes = _resolve_max_upload_bytes(max_upload_bytes)

    app = FastAPI(
        title="EasyDeL Whisper API",
        description="API for Whisper ASR model powered by EasyDeL",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    model_instance = WhisperModel(model_name=model_name, dtype=dtype)

    @app.get("/")
    def read_root():  # pyright: ignore[reportUnusedFunction]
        """Root endpoint for health check and API information.

        Returns:
            dict: A dictionary containing:
                - "message": Welcome message identifying the API
                - "model": Name of the loaded Whisper model
        """
        return {"message": "EasyDeL Whisper API", "model": model_instance.model_name}

    @app.post("/v1/audio/transcriptions", response_model=None)
    async def create_transcription(  # pyright: ignore[reportUnusedFunction]
        file: UploadFile = File(...),  # pyright: ignore[reportCallInDefaultInitializer]
        model: str = Form(model_name),  # pyright: ignore[reportCallInDefaultInitializer]
        prompt: str | None = Form(None),  # pyright: ignore[reportCallInDefaultInitializer]
        response_format: ResponseFormat = Form(ResponseFormat.json),  # pyright: ignore[reportCallInDefaultInitializer]
        temperature: float = Form(0.0),  # pyright: ignore[reportCallInDefaultInitializer]
        language: str | None = Form(None),  # pyright: ignore[reportCallInDefaultInitializer]
        timestamp_granularities: list[str] | None = Form(None),  # pyright: ignore[reportCallInDefaultInitializer]
        timestamp_granularities_bracket: list[str] | None = Form(  # pyright: ignore[reportCallInDefaultInitializer]
            None, alias="timestamp_granularities[]"
        ),
    ):
        """Transcribe audio to text using the Whisper model."""
        merged_granularities = [*(timestamp_granularities or ()), *(timestamp_granularities_bracket or ())] or None
        return_timestamps = _should_return_timestamps(response_format, merged_granularities)
        try:
            async with _uploaded_temp_file(file, resolved_max_upload_bytes) as temp_file_path:
                result = await asyncio.to_thread(
                    model_instance.inference,
                    audio_input=temp_file_path,
                    language=language,
                    return_timestamps=return_timestamps,
                )
            return _format_whisper_response(
                result,
                response_format,
                task="transcribe",
                language=language,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing audio: {e!s}") from e

    @app.post("/v1/audio/translations", response_model=None)
    async def create_translation(  # pyright: ignore[reportUnusedFunction]
        file: UploadFile = File(...),  # pyright: ignore[reportCallInDefaultInitializer]
        model: str = Form(model_name),  # pyright: ignore[reportCallInDefaultInitializer]
        prompt: str | None = Form(None),  # pyright: ignore[reportCallInDefaultInitializer]
        response_format: ResponseFormat = Form(ResponseFormat.json),  # pyright: ignore[reportCallInDefaultInitializer]
        temperature: float = Form(0.0),  # pyright: ignore[reportCallInDefaultInitializer]
        timestamp_granularities: list[str] | None = Form(None),  # pyright: ignore[reportCallInDefaultInitializer]
        timestamp_granularities_bracket: list[str] | None = Form(  # pyright: ignore[reportCallInDefaultInitializer]
            None, alias="timestamp_granularities[]"
        ),
    ):
        """Translate audio to English text using the Whisper model."""
        merged_granularities = [*(timestamp_granularities or ()), *(timestamp_granularities_bracket or ())] or None
        return_timestamps = _should_return_timestamps(response_format, merged_granularities)
        try:
            async with _uploaded_temp_file(file, resolved_max_upload_bytes) as temp_file_path:
                result = await asyncio.to_thread(
                    model_instance.inference,
                    audio_input=temp_file_path,
                    task="translate",
                    return_timestamps=return_timestamps,
                )
            return _format_whisper_response(
                result,
                response_format,
                task="translate",
                language="en",
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing audio: {e!s}") from e

    return app


def run_server(
    model_name: str = "openai/whisper-large-v3-turbo",
    host: str = "0.0.0.0",
    port: int = 8000,
    dtype=jnp.bfloat16,
    max_upload_bytes: int | None = None,
):
    """Run the vWhisper FastAPI server.

    Convenience function to create and run the Whisper API server with
    uvicorn. This is the recommended way to start the server for
    production use.

    Args:
        model_name (str, optional): HuggingFace model identifier for the
            Whisper model. Defaults to "openai/whisper-large-v3-turbo".
        host (str, optional): Host address to bind the server.
            Defaults to "0.0.0.0".
        port (int, optional): Port number to listen on.
            Defaults to 8000.
        dtype: JAX data type for model computations.
            Defaults to jnp.bfloat16.
        max_upload_bytes (int | None, optional): Maximum allowed upload size in
            bytes. If None, uses ``VWHISPER_MAX_UPLOAD_BYTES`` or 100 MiB.

    Returns:
        None: This function blocks until the server is stopped.
    """
    app = create_whisper_app(model_name=model_name, dtype=dtype, max_upload_bytes=max_upload_bytes)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
