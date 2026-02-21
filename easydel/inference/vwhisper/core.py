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

"""Core vWhisper inference engine implementation.

This module provides the main vWhisperInference class for performing
speech-to-text transcription and translation using Whisper models
optimized for JAX/XLA execution.

Classes:
    vWhisperInference: Main inference engine for Whisper models.

Example:
    Basic usage::

        >>> from easydel.inference.vwhisper import (
        ...     vWhisperInference,
        ...     vWhisperInferenceConfig
        ... )
        >>> from transformers import WhisperProcessor, WhisperTokenizer
        >>> import easydel as ed
        >>>
        >>> # Load model and components
        >>> model = ed.AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
        ...     "openai/whisper-base"
        ... )
        >>> tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        >>>
        >>> # Create inference engine
        >>> engine = vWhisperInference(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     processor=processor
        ... )
        >>>
        >>> # Transcribe audio
        >>> result = engine.generate("audio.mp3", language="en")
        >>> print(result["text"])
"""

from __future__ import annotations

import typing as tp

import jax
import numpy as np
from flax import nnx as nn
from jax import numpy as jnp

from easydel.utils import Registry

from .config import vWhisperInferenceConfig
from .generation import _compiled_generate, get_decoder_input_ids
from .utils import chunk_iter_with_batch, process_audio_input

if tp.TYPE_CHECKING:
    from transformers import WhisperProcessor, WhisperTokenizer

    from easydel.modules.whisper import WhisperForConditionalGeneration


@Registry.register("serve", "vwhisper")
class vWhisperInference:
    """High-performance speech-to-text inference engine using Whisper models.

    vWhisperInference provides a complete pipeline for transcribing and
    translating audio using OpenAI's Whisper models, optimized for JAX/XLA
    execution. It handles audio preprocessing, chunking for long-form audio,
    batch processing, and output post-processing with optional timestamps.

    The engine supports multiple input formats including file paths, URLs,
    raw bytes, and numpy arrays. Long audio is automatically split into
    overlapping chunks that are processed in batches for efficiency.

    Features:
        - Multi-format audio input (files, URLs, bytes, numpy arrays)
        - Automatic long-form audio chunking with configurable overlap
        - Batch processing for improved throughput
        - Timestamp generation for subtitle creation
        - Multi-language transcription support
        - Audio-to-English translation
        - JIT-compiled generation for optimal performance

    Attributes:
        model (WhisperForConditionalGeneration): The Whisper model for
            conditional generation.
        tokenizer (WhisperTokenizer): Tokenizer for converting between
            text and token IDs.
        processor (WhisperProcessor): Processor containing the feature
            extractor for audio preprocessing.
        feature_extractor: The feature extractor component from the
            processor, used for converting audio to mel spectrograms.
        inference_config (vWhisperInferenceConfig): Configuration settings
            controlling batch size, max length, and other parameters.
        generation_config (GenerationConfig): Generation configuration
            from the model or inference config.
        dtype: JAX data type for model computations.
        graphdef: Flax NNX graph definition for the model.
        graphstate: Flax NNX graph state containing model parameters.
        max_length (int): Maximum sequence length for generation.
        generate_function: The JIT-compiled generation function.

    Example:
        Basic transcription::

            >>> engine = vWhisperInference(
            ...     model=whisper_model,
            ...     tokenizer=tokenizer,
            ...     processor=processor
            ... )
            >>> result = engine.generate("speech.mp3")
            >>> print(result["text"])

        Transcription with timestamps::

            >>> result = engine.generate(
            ...     "lecture.mp3",
            ...     language="en",
            ...     return_timestamps=True
            ... )
            >>> for chunk in result.get("chunks", []):
            ...     print(f"[{chunk['timestamp']}] {chunk['text']}")

        Translation to English::

            >>> result = engine.generate(
            ...     "french_speech.mp3",
            ...     task="translate"
            ... )
            >>> print(result["text"])  # English translation

        Processing audio from URL::

            >>> result = engine.generate(
            ...     "https://example.com/audio.wav",
            ...     batch_size=4
            ... )

        Using numpy array input::

            >>> import numpy as np
            >>> audio = np.random.randn(16000 * 30)  # 30 seconds at 16kHz
            >>> result = engine.generate(
            ...     {"array": audio, "sampling_rate": 16000}
            ... )

    Note:
        - The engine is registered with the EasyDeL Registry under
          ("serve", "vwhisper") for automatic discovery.
        - First call to generate() may be slow due to JIT compilation;
          subsequent calls will be faster.
        - For optimal performance, use batch_size > 1 when processing
          long audio files.
        - The engine expects single-channel (mono) audio input.

    See Also:
        vWhisperInferenceConfig: Configuration class for inference settings.
        WhisperModel: Singleton wrapper for API server usage.
    """

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        tokenizer: WhisperTokenizer,
        processor: WhisperProcessor,
        inference_config: vWhisperInferenceConfig | None = None,
        dtype: jax.typing.DTypeLike = jnp.float32,
    ):
        """Initialize the vWhisperInference engine.

        Sets up the inference pipeline by storing model components,
        extracting the graph definition and state for JIT compilation,
        and configuring generation parameters.

        Args:
            model (WhisperForConditionalGeneration): A loaded Whisper model
                for conditional generation. Should be an EasyDeL-compatible
                Whisper model instance.
            tokenizer (WhisperTokenizer): Hugging Face Whisper tokenizer
                for encoding/decoding text. Used for converting generated
                token IDs back to text.
            processor (WhisperProcessor): Hugging Face Whisper processor
                containing the feature extractor for audio preprocessing.
                Converts raw audio to mel spectrogram features.
            inference_config (vWhisperInferenceConfig | None, optional):
                Configuration object controlling inference behavior.
                If None, a default configuration is created.
                Defaults to None.
            dtype (jax.typing.DTypeLike, optional): JAX data type for
                model computations. Common choices are jnp.float32,
                jnp.float16, or jnp.bfloat16. Defaults to jnp.float32.

        Example:
            Basic initialization::

                >>> engine = vWhisperInference(
                ...     model=whisper_model,
                ...     tokenizer=tokenizer,
                ...     processor=processor
                ... )

            With custom configuration::

                >>> config = vWhisperInferenceConfig(
                ...     batch_size=8,
                ...     max_length=448,
                ...     language="en"
                ... )
                >>> engine = vWhisperInference(
                ...     model=whisper_model,
                ...     tokenizer=tokenizer,
                ...     processor=processor,
                ...     inference_config=config,
                ...     dtype=jnp.bfloat16
                ... )

        Note:
            - The model is split into graphdef and graphstate using
              Flax NNX's split function for JIT compilation compatibility.
            - If inference_config provides a generation_config, it takes
              precedence over the model's built-in generation_config.
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
        """Generate token sequences from processed audio features.

        Internal method that performs the core generation step, converting
        mel spectrogram features into token sequences using the Whisper
        model's decoder.

        Args:
            input_features (jax.Array): Processed audio features as mel
                spectrograms with shape (batch_size, num_mel_bins, seq_len).
                These should be pre-processed by the feature extractor.
            language (str | None, optional): Language code for the input
                audio (e.g., "en", "fr", "de"). Used to set the language
                token in the decoder input. If None, language detection
                may be performed. Defaults to None.
            task (str | None, optional): Task to perform:
                - "transcribe": Transcribe in the source language
                - "translate": Translate to English
                If None, defaults to "transcribe". Defaults to None.
            return_timestamps (bool, optional): Whether to generate
                timestamp tokens for word/segment timing. Defaults to False.

        Returns:
            jax.Array: Generated token sequences with shape
                (batch_size, max_length). Includes special tokens for
                language, task, and optionally timestamps.

        Note:
            This method calls the JIT-compiled _compiled_generate function
            for efficient execution. The forced_decoder_ids are constructed
            from language and task tokens.
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
        """Process and chunk audio input for model consumption.

        Handles various audio input formats and splits long audio into
        overlapping chunks suitable for batch processing. Each chunk
        is converted to mel spectrogram features.

        Args:
            audio_input (str | bytes | np.ndarray | dict): Audio input in
                one of the following formats:
                - str: Path to local audio file or URL
                - bytes: Raw audio data (any ffmpeg-supported format)
                - np.ndarray: Raw audio samples (1D array)
                - dict: Dictionary with "array" and "sampling_rate" keys
            chunk_length_s (float, optional): Length of each audio chunk
                in seconds. Whisper models are trained on 30-second
                segments, so this is the recommended value.
                Defaults to 30.0.
            stride_length_s (float | list[float] | None, optional):
                Overlap between adjacent chunks in seconds. Can be:
                - float: Same stride on both sides
                - list[float]: [left_stride, right_stride]
                - None: Defaults to chunk_length_s / 6
                Defaults to None.
            batch_size (int | None, optional): Number of chunks to yield
                per batch. If None, uses the value from inference_config.
                Defaults to None.

        Yields:
            dict: A dictionary containing:
                - "input_features": Mel spectrogram features for the batch
                - "stride": List of (chunk_len, left_stride, right_stride)
                  tuples for each chunk
                - Additional keys from the feature extractor

        Raises:
            ValueError: If chunk_length_s is less than the total stride
                length (stride_left + stride_right).

        Note:
            - For audio shorter than chunk_length_s, a single batch is
              yielded without chunking.
            - The stride ensures smooth transitions when merging
              transcriptions from adjacent chunks.
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
        """Convert raw model outputs to formatted transcription results.

        Post-processes the generated token sequences, applying the
        tokenizer's ASR decoding logic to produce final text output
        with optional timestamps.

        Args:
            model_outputs (list): List of dictionaries containing:
                - "tokens": Generated token sequences
                - "stride": Optional stride information for chunk merging
            return_timestamps (bool | None, optional): Whether to include
                timestamps in the output. Timestamps are formatted as
                (start_time, end_time) tuples. Defaults to None.
            return_language (str | None, optional): Language code to
                include in the output metadata. Defaults to None.

        Returns:
            dict: A dictionary containing:
                - "text" (str): The transcribed/translated text
                - "chunks" (list, optional): If return_timestamps is True,
                  a list of dictionaries with "text" and "timestamp" keys
                - Additional metadata from the tokenizer decoder

        Note:
            - The stride information is converted from samples to seconds
              for proper timestamp calculation.
            - The tokenizer's _decode_asr method handles merging overlapping
              chunks and aligning timestamps.
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
        """Process a single batch of audio chunks through the model.

        Handles padding for incomplete batches and runs generation on
        the input features.

        Args:
            model_inputs (dict[str, tp.Any]): Dictionary containing:
                - "input_features": Audio features to process
                - "stride": Optional stride information
            batch_size (int): Expected batch size. If the actual batch
                is smaller, zero-padding is applied.
            language (str | None, optional): Language code for transcription.
                Defaults to None.
            task (str | None, optional): Task to perform ("transcribe" or
                "translate"). Defaults to None.
            return_timestamps (bool, optional): Whether to generate
                timestamp tokens. Defaults to False.

        Returns:
            dict: A dictionary containing:
                - "tokens": Generated token sequences with shape
                  (actual_batch_size, 1, seq_len)
                - "stride": Stride information if provided in inputs

        Note:
            Padding is removed after generation to return only the
            valid outputs for the actual input batch size.
        """
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
        """Transcribe or translate audio input to text.

        Main entry point for audio transcription and translation. Handles
        the complete pipeline from audio input to formatted text output,
        including chunking, batch processing, and post-processing.

        Args:
            audio_input (str | bytes | np.ndarray | dict): Input audio in
                one of the following formats:
                - str: Local file path or URL to audio file
                - bytes: Raw audio data (any ffmpeg-supported format)
                - np.ndarray: Audio samples as 1D numpy array
                - dict: Dictionary with keys:
                    - "array": Audio samples (np.ndarray)
                    - "sampling_rate": Sample rate (int)
            chunk_length_s (float, optional): Length of audio chunks in
                seconds. Whisper is optimized for 30-second chunks.
                Defaults to 30.0.
            stride_length_s (float | list[float] | None, optional):
                Overlap between chunks in seconds for smooth merging.
                Can be a single value or [left, right] list.
                Defaults to chunk_length_s / 6 if None.
            batch_size (int | None, optional): Number of chunks to process
                simultaneously. Larger values increase throughput but
                use more memory. Defaults to inference_config.batch_size.
            language (str | None, optional): Language code of the audio
                (e.g., "en", "fr", "de", "ja"). If None, may be auto-detected
                by the model. Defaults to inference_config.language.
            task (str | None, optional): Task to perform:
                - "transcribe": Transcribe in source language
                - "translate": Translate to English
                Defaults to inference_config.task.
            return_timestamps (bool | None, optional): Whether to include
                word/segment timestamps in output. Useful for subtitles.
                Defaults to inference_config.return_timestamps.

        Returns:
            dict: A dictionary containing:
                - "text" (str): The transcribed or translated text
                - "chunks" (list, optional): If return_timestamps is True,
                  list of dicts with "text" and "timestamp" keys

        Example:
            Simple transcription::

                >>> result = engine.generate("audio.mp3")
                >>> print(result["text"])

            Transcription with language and timestamps::

                >>> result = engine.generate(
                ...     "speech.wav",
                ...     language="en",
                ...     return_timestamps=True
                ... )
                >>> print(result["text"])
                >>> for chunk in result.get("chunks", []):
                ...     start, end = chunk["timestamp"]
                ...     print(f"[{start:.2f}s - {end:.2f}s] {chunk['text']}")

            Translation with custom batch size::

                >>> result = engine.generate(
                ...     "german_speech.mp3",
                ...     task="translate",
                ...     batch_size=8
                ... )

        Note:
            - First call may be slow due to JIT compilation
            - For optimal performance with long audio, increase batch_size
            - The engine expects mono audio; stereo will cause errors
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
