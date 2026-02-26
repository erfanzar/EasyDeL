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

"""Utility functions for vWhisper audio processing.

This module provides utility functions for processing audio input and
chunking long-form audio for efficient batch processing with Whisper models.

Functions:
    chunk_iter_with_batch: Generate batched chunks from long audio.
    process_audio_input: Process various audio input formats.

Example:
    Processing audio with chunking::

        >>> from easydel.inference.vwhisper.utils import (
        ...     chunk_iter_with_batch,
        ...     process_audio_input
        ... )
        >>> import numpy as np
        >>>
        >>> # Process audio input
        >>> audio_array, stride = process_audio_input(
        ...     "audio.mp3",
        ...     feature_extractor
        ... )
        >>>
        >>> # Generate batched chunks
        >>> for batch in chunk_iter_with_batch(
        ...     audio_array,
        ...     chunk_length=480000,  # 30 seconds at 16kHz
        ...     stride_left=80000,
        ...     stride_right=80000,
        ...     batch_size=4,
        ...     feature_extractor=feature_extractor
        ... ):
        ...     process_batch(batch)
"""

import math

import numpy as np
import requests


def chunk_iter_with_batch(
    audio_array: np.ndarray,
    chunk_length: int,
    stride_left: int,
    stride_right: int,
    batch_size: int,
    feature_extractor,
):
    """Generate batched chunks from a long audio array with overlapping strides.

    This function divides a long audio array into overlapping chunks suitable
    for processing by Whisper models. The chunks are grouped into batches for
    efficient parallel processing. Overlapping strides ensure smooth transitions
    between chunks when reconstructing the final transcription.

    The chunking algorithm ensures that:
        - Each chunk has length `chunk_length` (except possibly the last)
        - Adjacent chunks overlap by `stride_left + stride_right` samples
        - The first chunk has no left stride (starts at the beginning)
        - The last chunk has no right stride (extends to the end)

    Args:
        audio_array (np.ndarray): Input audio array of shape (num_samples,).
            Must be a 1-dimensional array of audio samples.
        chunk_length (int): Length of each chunk in samples. For example,
            at 16kHz sampling rate, 480000 samples = 30 seconds.
        stride_left (int): Number of samples to overlap with the previous
            chunk on the left side.
        stride_right (int): Number of samples to overlap with the next
            chunk on the right side.
        batch_size (int): Number of chunks to include in each yielded batch.
            Larger batch sizes improve throughput but require more memory.
        feature_extractor: Hugging Face feature extractor (typically
            WhisperFeatureExtractor) used to convert raw audio to model
            input features.

    Yields:
        dict: A dictionary containing:
            - "stride": List of tuples (chunk_length, left_stride, right_stride)
              for each chunk in the batch. The strides indicate how much of
              each chunk overlaps with neighbors.
            - "input_features": Processed audio features from the feature
              extractor, ready for model input.
            - Additional keys from the feature extractor output.

    Example:
        >>> audio = np.random.randn(960000)  # 60 seconds at 16kHz
        >>> for batch in chunk_iter_with_batch(
        ...     audio_array=audio,
        ...     chunk_length=480000,  # 30 seconds
        ...     stride_left=80000,    # 5 seconds
        ...     stride_right=80000,   # 5 seconds
        ...     batch_size=2,
        ...     feature_extractor=feature_extractor
        ... ):
        ...     print(f"Batch has {len(batch['stride'])} chunks")
        ...     print(f"Strides: {batch['stride']}")

    Note:
        The stride values in the output are adjusted at chunk boundaries:
        - First chunk: left stride is 0
        - Last chunk: right stride is 0
        - Chunks at the end of audio: right stride may be 0 if the chunk
          extends beyond the audio length
    """
    inputs_len = audio_array.shape[0]
    step = chunk_length - stride_left - stride_right
    all_chunk_start_idx = np.arange(0, inputs_len, step)
    num_samples = len(all_chunk_start_idx)
    num_batches = math.ceil(num_samples / batch_size)
    batch_idx = np.array_split(np.arange(num_samples), num_batches)

    for idx in batch_idx:
        chunk_start_idx = all_chunk_start_idx[idx]
        chunk_end_idx = chunk_start_idx + chunk_length
        chunks = [
            audio_array[chunk_start:chunk_end]
            for chunk_start, chunk_end in zip(chunk_start_idx, chunk_end_idx, strict=False)
        ]
        processed = feature_extractor(
            chunks,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="np",
        )

        yield {
            "stride": [
                (chunk_l, _stride_l, _stride_r)
                for chunk_l, _stride_l, _stride_r in zip(
                    [chunk.shape[0] for chunk in chunks],
                    np.where(chunk_start_idx == 0, 0, stride_left),
                    np.where(
                        np.where(
                            stride_right > 0,
                            chunk_end_idx > inputs_len,
                            chunk_end_idx >= inputs_len,
                        ),
                        0,
                        stride_right,
                    ),
                    strict=False,
                )
            ],
            **processed,
        }


def process_audio_input(
    audio_input: str | bytes | np.ndarray | dict[str, np.ndarray | int],
    feature_extractor,
):
    """Process various audio input formats into a normalized numpy array.

    This function handles multiple audio input formats and converts them
    to a numpy array with the correct sampling rate for the Whisper model.
    It supports file paths (local and URLs), raw bytes, numpy arrays, and
    dictionaries containing audio data with metadata.

    The function performs the following operations:
        1. Fetches remote audio from URLs if provided
        2. Reads local audio files from disk
        3. Converts bytes to numpy arrays using ffmpeg
        4. Resamples audio to match the feature extractor's sampling rate
        5. Validates that the audio is single-channel

    Args:
        audio_input (str | bytes | np.ndarray | dict[str, np.ndarray | int]):
            The audio input in one of the following formats:

            - str: Path to a local audio file, or URL (http/https) to
              fetch audio from.
            - bytes: Raw audio data in any format supported by ffmpeg.
            - np.ndarray: Raw audio samples as a 1D numpy array.
            - dict: Dictionary with the following keys:
                - "array" (np.ndarray): Audio samples.
                - "sampling_rate" (int): Sampling rate of the audio.
                - "stride" (tuple, optional): Pre-computed stride info.

        feature_extractor: Hugging Face feature extractor (typically
            WhisperFeatureExtractor) that provides the target sampling
            rate via `feature_extractor.sampling_rate`.

    Returns:
        tuple[np.ndarray, tuple | None]: A tuple containing:
            - audio_array (np.ndarray): The processed audio as a 1D numpy
              array with the correct sampling rate.
            - stride (tuple | None): Stride information as
              (total_length, left_stride, right_stride) if provided in
              the input dictionary, otherwise None. The stride values
              are scaled according to any resampling performed.

    Raises:
        ValueError: If the input dictionary is missing required keys
            ("array" and "sampling_rate").
        ValueError: If the processed audio is not a numpy ndarray.
        ValueError: If the audio has more than one channel (not 1D).
        ValueError: If the stride is larger than the input audio length.
        ImportError: If librosa is required for resampling but not installed.

    Example:
        Processing a local file::

            >>> audio, stride = process_audio_input(
            ...     "speech.mp3",
            ...     feature_extractor
            ... )
            >>> print(f"Audio shape: {audio.shape}")
            >>> print(f"Stride: {stride}")

        Processing a URL::

            >>> audio, stride = process_audio_input(
            ...     "https://example.com/audio.wav",
            ...     feature_extractor
            ... )

        Processing a dictionary with resampling::

            >>> audio_dict = {
            ...     "array": np.random.randn(44100),  # 1 second at 44.1kHz
            ...     "sampling_rate": 44100
            ... }
            >>> audio, stride = process_audio_input(
            ...     audio_dict,
            ...     feature_extractor  # targets 16kHz
            ... )
            >>> print(f"Resampled shape: {audio.shape}")  # ~16000 samples

    Note:
        - For best performance, provide audio already at the target
          sampling rate to avoid resampling overhead.
        - The function uses ffmpeg (via transformers) for decoding
          audio bytes, so ffmpeg must be installed on the system.
        - Resampling requires librosa and soundfile to be installed.
    """
    stride = None

    if isinstance(audio_input, str):
        if audio_input.startswith("http://") or audio_input.startswith("https://"):
            audio_input = requests.get(audio_input).content
        else:
            with open(audio_input, "rb") as f:
                audio_input = f.read()

    if isinstance(audio_input, bytes):
        from transformers.pipelines.audio_utils import ffmpeg_read

        audio_input = ffmpeg_read(audio_input, feature_extractor.sampling_rate)

    ratio = 1
    if isinstance(audio_input, dict):
        stride = audio_input.get("stride", None)
        if not ("sampling_rate" in audio_input and "array" in audio_input):
            raise ValueError(
                "When passing a dictionary to FlaxWhisperPipline, the dict needs to contain an 'array' key "
                "containing the numpy array representing the audio, and a 'sampling_rate' key "
                "containing the sampling rate associated with the audio array."
            )

        in_sampling_rate = audio_input.get("sampling_rate")
        audio_input = audio_input.get("array", None)

        if in_sampling_rate != feature_extractor.sampling_rate:
            try:
                import librosa  # type:ignore
            except ImportError as err:
                raise ImportError(
                    "To support resampling audio files, please install 'librosa' and 'soundfile'."
                ) from err

            audio_input = librosa.resample(
                audio_input,
                orig_sr=in_sampling_rate,
                target_sr=feature_extractor.sampling_rate,
            )
            ratio = feature_extractor.sampling_rate / in_sampling_rate
        else:
            ratio = 1

    if not isinstance(audio_input, np.ndarray):
        raise ValueError(f"We expect a numpy ndarray as input, got `{type(audio_input)}`")
    if len(audio_input.shape) != 1:
        raise ValueError("We expect a single channel audio input for AutomaticSpeechRecognitionPipeline")

    if stride is not None:
        if stride[0] + stride[1] > audio_input.shape[0]:
            raise ValueError("Stride is too large for input")
        stride = (
            audio_input.shape[0],
            round(stride[0] * ratio),
            round(stride[1] * ratio),
        )

    return audio_input, stride
