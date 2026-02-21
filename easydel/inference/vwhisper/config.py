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

"""Configuration classes for vWhisper inference.

This module provides the configuration class for controlling vWhisper
inference behavior, including batch size, generation settings, and
language/task options.

Classes:
    vWhisperInferenceConfig: Main configuration class for inference.

Example:
    Creating a basic configuration::

        >>> from easydel.inference.vwhisper import vWhisperInferenceConfig
        >>> config = vWhisperInferenceConfig(
        ...     batch_size=8,
        ...     max_length=448,
        ...     return_timestamps=True,
        ...     language="en"
        ... )

    Using with generation config::

        >>> from transformers import GenerationConfig
        >>> gen_config = GenerationConfig(
        ...     max_length=512,
        ...     do_sample=False
        ... )
        >>> config = vWhisperInferenceConfig(
        ...     generation_config=gen_config,
        ...     task="transcribe"
        ... )
"""

from __future__ import annotations

import typing as tp

from eformer.pytree import auto_pytree

from easydel.utils.compiling_utils import get_safe_hash_int

if tp.TYPE_CHECKING:
    from transformers import GenerationConfig
else:
    GenerationConfig = tp.Any


@auto_pytree
class vWhisperInferenceConfig:
    """Configuration class for vWhisper inference settings.

    This class encapsulates all configuration options for the vWhisper
    inference engine, controlling batch processing, generation behavior,
    and output formatting. It is decorated with @auto_pytree for JAX
    compatibility, allowing it to be used in JIT-compiled functions.

    The configuration supports both explicit parameter setting and
    inheritance from a GenerationConfig object. When parameters are
    not explicitly set, defaults from the model's generation config
    are used.

    Attributes:
        batch_size (int | None): Number of audio chunks to process
            simultaneously. Larger values improve throughput but
            require more memory. Defaults to 1.
        max_length (int | None): Maximum number of tokens to generate
            per chunk. If None, uses the model's default max_length.
        generation_config (GenerationConfig | None): Hugging Face
            GenerationConfig object for fine-grained control over
            text generation. If None, uses the model's default config.
        logits_processor: Reserved for custom logits processors.
            Currently not used but available for future extensions.
        return_timestamps (bool | None): Whether to include word or
            segment timestamps in the output. Useful for subtitle
            generation. If None, determined by the generation config.
        task (str | None): The task to perform. Options are:
            - "transcribe": Transcribe audio in the source language
            - "translate": Translate audio to English
            If None, defaults to "transcribe".
        language (str | None): The language of the input audio for
            transcription, or the source language for translation.
            Should be a language code (e.g., "en", "fr", "de") or
            language name. If None, language is auto-detected.
        is_multilingual (bool | None): Whether the model supports
            multiple languages. If None, determined from the model.

    Example:
        Basic configuration for English transcription::

            >>> config = vWhisperInferenceConfig(
            ...     batch_size=4,
            ...     max_length=448,
            ...     language="en",
            ...     task="transcribe",
            ...     return_timestamps=True
            ... )

        Configuration for translation with custom generation::

            >>> from transformers import GenerationConfig
            >>> gen_config = GenerationConfig(
            ...     max_length=512,
            ...     num_beams=5,
            ...     length_penalty=1.0
            ... )
            >>> config = vWhisperInferenceConfig(
            ...     batch_size=2,
            ...     generation_config=gen_config,
            ...     task="translate"
            ... )

        Using config with vWhisperInference::

            >>> inference = vWhisperInference(
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     processor=processor,
            ...     inference_config=config
            ... )

    Note:
        - The class is hashable, allowing it to be used as a key in
          dictionaries or with JAX's static_argnames.
        - Parameters set to None will inherit from the model's
          generation_config or use sensible defaults.
        - For long-form audio, batch_size refers to the number of
          30-second chunks processed together, not full audio files.

    See Also:
        vWhisperInference: The main inference class that uses this config.
        transformers.GenerationConfig: Hugging Face generation config.
    """

    batch_size: int | None = 1
    max_length: int | None = None
    generation_config: GenerationConfig | None = None
    logits_processor = None
    return_timestamps = None
    task = None
    language = None
    is_multilingual = None

    def __hash__(self):
        """Compute a hash value for the configuration.

        Creates a deterministic hash based on all configuration parameters,
        enabling use of the config as a dictionary key or with JAX's
        static_argnames for JIT compilation caching.

        Returns:
            int: A hash value derived from all configuration attributes.

        Note:
            The hash is computed by concatenating string representations
            of all key-value pairs and passing through get_safe_hash_int
            for a stable integer hash.
        """
        return get_safe_hash_int("".join(str(k) + str(v) for k, v in self.__dict__.items()))
