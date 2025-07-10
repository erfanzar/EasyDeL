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

import os
import tempfile
import typing as tp
from enum import Enum

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from jax import numpy as jnp
from pydantic import BaseModel, Field
from transformers import WhisperProcessor, WhisperTokenizer

import easydel as ed

from .core import vWhisperInference


class WhisperModel:
    """Singleton wrapper for the Whisper model to avoid reloading."""

    _instance = None

    def __new__(cls, model_name=None, dtype=jnp.bfloat16):
        if cls._instance is None and model_name is not None:
            cls._instance = super().__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.dtype = dtype
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the model, tokenizer, and processor."""
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


class ResponseFormat(str, Enum):
    json = "json"
    text = "text"
    srt = "srt"
    verbose_json = "verbose_json"
    vtt = "vtt"


class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="The transcribed text")
    segments: list[dict[str, tp.Any]] | None = Field(None, description="Segments with timestamps")


def create_whisper_app(model_name: str = "openai/whisper-large-v3-turbo", dtype=jnp.bfloat16):
    """Create a FastAPI app for Whisper transcription."""
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
        """
        Transcribe audio to text using the Whisper model.

        This endpoint mimics OpenAI's Whisper API.
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
        """
        Translate audio to English text using the Whisper model.

        This endpoint mimics OpenAI's Whisper translation API.
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
    """
    Run the Whisper FastAPI server.

    Args:
        model_name: Name of the Whisper model to use (from HuggingFace)
        host: Host to bind the server
        port: Port to bind the server
        dtype: Data type for the model (default: bfloat16)
    """
    app = create_whisper_app(model_name=model_name, dtype=dtype)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
