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
    """
    Process an audio array into chunks with overlapping strides.

    Args:
        audio_array: Input audio array
        chunk_length: Length of each chunk in samples
        stride_left: Left stride in samples
        stride_right: Right stride in samples
        batch_size: Number of chunks to process at once
        feature_extractor: Feature extractor to process audio

    Yields:
        Batches of processed audio chunks
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
    """
    Process audio input into a numpy array with correct sampling rate.

    Args:
        audio_input: Input audio in various formats
        feature_extractor: Feature extractor with sampling rate info

    Returns:
        Tuple of (audio_array, stride)
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
