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

"""Whisper model implementation for EasyDeL.

This module provides the Whisper speech recognition model architecture for
automatic speech recognition (ASR), audio classification, and speech translation tasks.

Whisper is an encoder-decoder transformer architecture designed for robust speech recognition
across multiple languages and domains. It was developed by OpenAI and trained on 680,000 hours
of multilingual and multitask supervised data.

Key Architectural Features:
    - Encoder-Decoder Architecture: The encoder processes log-Mel spectrogram audio features,
      while the decoder generates text transcriptions autoregressively.
    - Convolutional Feature Extraction: The encoder uses two 1D convolutional layers to
      downsample audio features before transformer processing.
    - Sinusoidal Positional Embeddings: Fixed sinusoidal embeddings for encoder positions,
      learned embeddings for decoder positions.
    - Cross-Attention Mechanism: The decoder attends to encoder outputs via cross-attention
      to condition text generation on audio content.
    - Multi-Task Training: Supports transcription, translation, language identification,
      and voice activity detection through special token conditioning.

Model Variants:
    - WhisperForConditionalGeneration: Full encoder-decoder model for speech-to-text tasks
      including transcription and translation.
    - WhisperForAudioClassification: Encoder-only variant with pooling and classification
      head for audio tagging and classification tasks.

Usage Example:
    ```python
    from easydel import AutoEasyDeLModelForSpeechSeq2Seq, WhisperConfig
    import jax.numpy as jnp

    # Initialize model from pretrained weights
    model, params = AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-base",
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
    )

    # Prepare audio features (log-Mel spectrogram)
    # Shape: (batch_size, num_mel_bins=80, sequence_length=3000)
    input_features = jnp.ones((1, 80, 3000))

    # Generate transcription
    outputs = model.generate(
        input_features=input_features,
        params=params,
        max_length=448,
        language="en",
        task="transcribe",
    )
    ```

Configuration Example:
    ```python
    from easydel.modules.whisper import WhisperConfig

    config = WhisperConfig(
        vocab_size=51865,
        num_mel_bins=80,
        encoder_layers=6,
        encoder_attention_heads=8,
        decoder_layers=6,
        decoder_attention_heads=8,
        d_model=512,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        dropout=0.1,
    )
    ```

For more information on Whisper, see:
    - Paper: https://arxiv.org/abs/2212.04356
    - Blog: https://openai.com/blog/whisper/
"""

from .modeling_whisper import (
    WhisperForAudioClassification,
    WhisperForConditionalGeneration,
    WhisperTimeStampLogitsProcessor,
)
from .whisper_configuration import WhisperConfig

__all__ = (
    "WhisperConfig",
    "WhisperForAudioClassification",
    "WhisperForConditionalGeneration",
    "WhisperTimeStampLogitsProcessor",
)
