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

import jax
import numpy as np
from flax import nnx as nn
from jax import numpy as jnp

from .config import vWhisperInferenceConfig
from .generation import _compiled_generate, get_decoder_input_ids
from .utils import chunk_iter_with_batch, process_audio_input

if tp.TYPE_CHECKING:
    from transformers import WhisperProcessor, WhisperTokenizer

    from easydel.modules.whisper import WhisperForConditionalGeneration


class vWhisperInference:
    """Speech-to-text inference engine using Whisper models.

    vWhisperInference provides a high-performance pipeline for transcribing
    and translating audio using OpenAI's Whisper models, optimized for JAX.
    It supports long-form audio processing with automatic chunking and
    can generate timestamps for subtitle creation.

    Features:
        - Audio transcription in multiple languages
        - Translation to English
        - Timestamp generation for subtitles
        - Long-form audio processing with chunking
        - Batch processing for efficiency
        - JAX/XLA acceleration

    Attributes:
        model: The Whisper model for conditional generation
        tokenizer: Tokenizer for text processing
        processor: Audio processor for feature extraction
        inference_config: Configuration settings
        dtype: Data type for computations
        graphdef: Model graph definition
        graphstate: Model state

    Args:
        model: Fine-tuned Whisper model for inference.
        tokenizer: Whisper tokenizer.
        processor: Whisper processor for audio processing.
        inference_config: Optional configuration settings.
        dtype: Data type for JAX computations (default: float32).

    Example:
        >>> engine = vWhisperInference(
        ...     model=whisper_model,
        ...     tokenizer=tokenizer,
        ...     processor=processor
        ... )
        >>> result = engine.transcribe(
        ...     "audio.mp3",
        ...     language="en"
        ... )
        >>> print(result["text"])
    """

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        tokenizer: WhisperTokenizer,
        processor: WhisperProcessor,
        inference_config: vWhisperInferenceConfig | None = None,
        dtype: jax.typing.DTypeLike = jnp.float32,
    ):
        """Initialize vWhisperInference engine.

        Args:
            model: Whisper model for conditional generation.
            tokenizer: Tokenizer for text processing.
            processor: Processor for audio feature extraction.
            inference_config: Configuration for inference behavior.
            dtype: JAX data type for computations.
        """
        if inference_config is None:
            inference_config = vWhisperInferenceConfig()
        self.dtype = dtype
        self.processor = processor
        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = tokenizer
        self.model = model
        graphdef, graphstate = nn.split(model)
        self.graphdef = graphdef
        self.graphstate = graphstate
        generation_config = inference_config.generation_config or self.model.generation_config
        inference_config.generation_config = generation_config
        self.generation_config = generation_config
        self.max_length = inference_config.max_length or self.generation_config.max_length
        self.inference_config = inference_config
        self.generate_function = _compiled_generate

    def _generate(
        self,
        input_features: jax.Array,
        language: str | None = None,
        task: str | None = None,
        return_timestamps: bool = False,
    ) -> jax.Array:
        """Generate text from audio features.

        Internal method for generating sequences from processed audio.

        Args:
            input_features: Processed audio features.
            language: Source language code.
            task: Task type ('transcribe' or 'translate').
            return_timestamps: Whether to generate timestamps.

        Returns:
            Generated token sequences.
        """
        forced_decoder_ids = dict(
            get_decoder_input_ids(
                model_config=self.model.config,
                generation_config=self.generation_config,
                language=language,
                task=task,
                return_timestamps=return_timestamps,
            )
        )
        output_sequences = self.generate_function(
            graphdef=self.graphdef,
            graphstate=self.graphstate,
            inference_config=self.inference_config,
            input_features=input_features,
            decoder_input_ids=forced_decoder_ids,
            return_timestamps=return_timestamps,
        ).sequences
        return output_sequences

    def _process_model_inputs(
        self,
        audio_input: str | bytes | np.ndarray | dict[str, np.ndarray | int],
        chunk_length_s: float = 30.0,
        stride_length_s: float | list[float] | None = None,
        batch_size: int | None = None,
    ):
        """Process audio input into model-ready features.

        Handles various audio input formats and performs chunking
        for long-form audio processing.

        Args:
            audio_input: Audio data (file path, bytes, or array).
            chunk_length_s: Length of audio chunks in seconds.
            stride_length_s: Overlap between chunks in seconds.
            batch_size: Number of chunks to process together.

        Yields:
            Processed audio features ready for model input.

        Raises:
            ValueError: If chunk length is less than stride length.
        """
        audio_array, stride = process_audio_input(
            audio_input=audio_input,
            feature_extractor=self.feature_extractor,
        )

        if chunk_length_s:
            if stride_length_s is None:
                stride_length_s = chunk_length_s / 6

            if isinstance(stride_length_s, int | float):
                stride_length_s = [stride_length_s, stride_length_s]

            chunk_length = round(chunk_length_s * self.feature_extractor.sampling_rate)
            stride_left = round(stride_length_s[0] * self.feature_extractor.sampling_rate)
            stride_right = round(stride_length_s[1] * self.feature_extractor.sampling_rate)

            if chunk_length < stride_left + stride_right:
                raise ValueError("Chunk length must be superior to stride length")

            yield from chunk_iter_with_batch(
                audio_array=audio_array,
                chunk_length=chunk_length,
                stride_left=stride_left,
                stride_right=stride_right,
                batch_size=batch_size,
                feature_extractor=self.feature_extractor,
            )
        else:
            processed = self.feature_extractor(
                audio_array,
                sampling_rate=self.feature_extractor.sampling_rate,
                return_tensors="np",
            )
            if stride is not None:
                processed["stride"] = stride
            yield processed

    def _process_model_outputs(
        self,
        model_outputs,
        return_timestamps: bool | None = None,
        return_language: str | None = None,
    ):
        """Process model outputs into final transcription.

        Converts raw model outputs into formatted text with optional
        timestamps and language information.

        Args:
            model_outputs: Raw outputs from the model.
            return_timestamps: Whether to include timestamps.
            return_language: Language code to include in output.
        """
        model_outputs = [
            dict(zip(output, t, strict=False)) for output in model_outputs for t in zip(*output.values(), strict=False)
        ]
        time_precision = self.feature_extractor.chunk_length / self.model.config.max_source_positions
        sampling_rate = self.feature_extractor.sampling_rate
        for output in model_outputs:
            if "stride" in output:
                chunk_length, stride_left, stride_right = output["stride"]
                output["stride"] = (
                    chunk_length / sampling_rate,
                    stride_left / sampling_rate,
                    stride_right / sampling_rate,
                )

        text, optional = self.tokenizer._decode_asr(
            model_outputs,
            return_timestamps=return_timestamps,
            return_language=return_language,
            time_precision=time_precision,
        )
        return {"text": text, **optional}

    def _single_batch_process(
        self,
        model_inputs: dict[str, tp.Any],
        batch_size: int,
        language: str | None = None,
        task: str | None = None,
        return_timestamps: bool = False,
    ):
        input_features = model_inputs.pop("input_features")
        input_batch_size = input_features.shape[0]
        if input_batch_size != batch_size:
            padding = np.zeros([batch_size - input_batch_size, *input_features.shape[1:]], input_features.dtype)
            input_features = np.concatenate([input_features, padding])
        output_tokens = self._generate(
            input_features=input_features,
            language=language,
            task=task,
            return_timestamps=return_timestamps,
        )[:input_batch_size]

        output = {"tokens": output_tokens[:, None, :]}
        stride = model_inputs.pop("stride", None)
        if stride is not None:
            output["stride"] = stride
        return output

    def generate(
        self,
        audio_input: str | bytes | np.ndarray | dict[str, np.ndarray | int],
        chunk_length_s: float = 30.0,
        stride_length_s: float | list[float] | None = None,
        batch_size: int | None = None,
        language: str | None = None,
        task: str | None = None,
        return_timestamps: bool | None = None,
    ):
        """
        Transcribe or translate audio input.

        Args:
            audio_input (`tp.Union[str, bytes, np.ndarray, tp.Dict[str, tp.Union[np.ndarray, int]]]`):
                Input audio. Can be a local file path, URL, bytes, numpy array, or a dictionary
                containing the array and sampling rate.
            chunk_length_s (`float`, *optional*, defaults to 30.0):
                Length of audio chunks in seconds.
            stride_length_s (`float` or `list[float]`, *optional*):
                Stride length for chunking audio, in seconds.  Defaults to `chunk_length_s / 6`.
            batch_size (`int`, *optional*):
                Batch size for processing. Defaults to the `batch_size` in `inference_config`.
            language (`str`, *optional*):
                Language of the input audio. Defaults to the `language` in `inference_config`.
            task (`str`, *optional*):
                Task to perform (e.g., "transcribe", "translate"). Defaults to the `task` in `inference_config`.
            return_timestamps (`bool`, *optional*):
                Whether to return timestamps with the transcription.
                    Defaults to the `return_timestamps` in `inference_config`.

        Returns:
            `dict`: A dictionary containing the transcribed text ("text") and optionally other information
            like timestamps or detected language.
        """
        batch_size = batch_size if batch_size is not None else self.inference_config.batch_size
        language = language if language is not None else self.inference_config.language
        task = task if task is not None else self.inference_config.task
        return_timestamps = (
            return_timestamps if return_timestamps is not None else self.inference_config.return_timestamps
        )

        dataloader = self._process_model_inputs(
            audio_input=audio_input,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
        )
        model_outputs = []
        for model_inputs in dataloader:
            model_outputs.append(
                self._single_batch_process(
                    model_inputs=model_inputs,
                    batch_size=batch_size,
                    language=language,
                    task=task,
                    return_timestamps=return_timestamps,
                )
            )
        return self._process_model_outputs(
            model_outputs,
            return_timestamps=return_timestamps,
        )

    __call__ = generate
