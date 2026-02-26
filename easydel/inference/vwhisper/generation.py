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

"""Generation utilities for vWhisper inference.

This module provides JIT-compiled generation functions and helper utilities
for constructing decoder input IDs for Whisper models. These functions
handle the low-level generation logic including language tokens, task
tokens, and timestamp control.

Functions:
    _compiled_generate: JIT-compiled function for efficient text generation.
    get_decoder_input_ids: Construct decoder input IDs for language/task tokens.

Example:
    Using get_decoder_input_ids::

        >>> from easydel.inference.vwhisper.generation import get_decoder_input_ids
        >>> decoder_ids = get_decoder_input_ids(
        ...     model_config=model.config,
        ...     generation_config=model.generation_config,
        ...     language="en",
        ...     task="transcribe",
        ...     return_timestamps=True
        ... )
        >>> print(decoder_ids)
        [(1, 50259), (2, 50358)]  # Language and task token positions

Note:
    These functions are primarily intended for internal use by the
    vWhisperInference class. Direct usage is possible but requires
    understanding of Whisper's token format.
"""

from flax import nnx as nn
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

from easydel.utils.compiling_utils import ejit


@ejit(static_argnames=["graphdef", "inference_config", "return_timestamps"])  # pyright: ignore[reportUntypedFunctionDecorator]
def _compiled_generate(  # pyright: ignore[reportUnusedFunction]
    graphdef,
    graphstate,
    inference_config,
    input_features,
    decoder_input_ids,
    return_timestamps,
):
    """JIT-compiled generation function for Whisper models.

    This function performs the core text generation step, optimized for
    JAX/XLA execution through JIT compilation. It merges the model graph
    and state, then calls the model's internal generation method.

    The function is decorated with @ejit (EasyDeL's enhanced JIT) with
    static arguments for the graph definition, inference config, and
    timestamp flag. This allows efficient caching and recompilation
    only when these static values change.

    Args:
        graphdef: Flax NNX graph definition from nn.split(model).
            Contains the model architecture without parameters.
        graphstate: Flax NNX graph state from nn.split(model).
            Contains the model parameters and state.
        inference_config (vWhisperInferenceConfig): Configuration object
            containing generation settings like max_length and
            generation_config.
        input_features (jax.Array): Processed audio features (mel
            spectrograms) with shape (batch_size, num_mel_bins, seq_len).
        decoder_input_ids (dict): Dictionary mapping position indices
            to token IDs for forced decoder tokens (language, task, etc.).
            Format: {position: token_id, ...}
        return_timestamps (bool): Whether to generate timestamp tokens.
            When True, the model produces special timestamp tokens that
            indicate word/segment boundaries.

    Returns:
        GenerationOutput: Object containing:
            - sequences: Generated token sequences (batch_size, seq_len)
            - Additional generation metadata depending on model config

    Example:
        Internal usage (called by vWhisperInference._generate)::

            >>> output = _compiled_generate(
            ...     graphdef=self.graphdef,
            ...     graphstate=self.graphstate,
            ...     inference_config=self.inference_config,
            ...     input_features=mel_features,
            ...     decoder_input_ids={1: 50259, 2: 50358},
            ...     return_timestamps=False
            ... )
            >>> tokens = output.sequences

    Note:
        - This function should not be called directly; use
          vWhisperInference.generate() instead.
        - The first call triggers JIT compilation, which may be slow.
        - Subsequent calls with the same static args are cached.
        - The model.mesh context is used for proper device placement.
    """
    model = nn.merge(graphdef, graphstate)
    with model.mesh:
        return model._force_generate(
            input_features=input_features,
            forced_decoder_ids=decoder_input_ids,
            return_timestamps=return_timestamps,
            generation_config=inference_config.generation_config,
        )


def get_decoder_input_ids(
    model_config,
    generation_config=None,
    task=None,
    language=None,
    return_timestamps=False,
):
    """Construct decoder input IDs for Whisper generation.

    Creates a list of (position, token_id) tuples that specify forced
    decoder tokens for language, task, and timestamp control. These
    tokens guide the Whisper model's generation behavior.

    For multilingual models, the decoder input sequence follows this format:
        - Position 1: Language token (e.g., <|en|>, <|fr|>)
        - Position 2: Task token (<|transcribe|> or <|translate|>)
        - Position 3 (optional): No-timestamps token if timestamps disabled

    Args:
        model_config: Whisper model configuration object containing
            model-specific settings like is_multilingual.
        generation_config: Generation configuration object containing
            language-to-ID mappings and task-to-ID mappings. If None,
            falls back to model_config. Defaults to None.
        task (str | None, optional): Task to perform:
            - "transcribe": Transcribe in source language
            - "translate": Translate to English
            - None: Defaults to "transcribe"
            Defaults to None.
        language (str | None, optional): Language of the audio. Accepts:
            - Language codes: "en", "fr", "de", "ja", etc.
            - Language names: "english", "french", "german"
            - Token format: "<|en|>", "<|french|>"
            If None, language detection may be used. Defaults to None.
        return_timestamps (bool, optional): Whether timestamp tokens
            should be generated. When False, the no_timestamps_token
            is added to suppress timestamp generation. Defaults to False.

    Returns:
        list[tuple[int, int]]: List of (position, token_id) tuples
            specifying forced decoder tokens. The list may contain:
            - Language token at position 1
            - Task token at position 2
            - No-timestamps token at position 3 (if return_timestamps=False)

    Raises:
        ValueError: If the language is not supported by the model.
            The error message includes a list of acceptable languages.

    Example:
        Basic usage for English transcription::

            >>> ids = get_decoder_input_ids(
            ...     model_config=model.config,
            ...     generation_config=model.generation_config,
            ...     language="en",
            ...     task="transcribe",
            ...     return_timestamps=False
            ... )
            >>> print(ids)
            [(1, 50259), (2, 50358), (3, 50362)]

        Translation with timestamps::

            >>> ids = get_decoder_input_ids(
            ...     model_config=model.config,
            ...     generation_config=model.generation_config,
            ...     language="fr",
            ...     task="translate",
            ...     return_timestamps=True
            ... )
            >>> print(ids)
            [(1, 50265), (2, 50357)]

        Using language name instead of code::

            >>> ids = get_decoder_input_ids(
            ...     model_config=model.config,
            ...     language="german",
            ...     task="transcribe"
            ... )

    Note:
        - For non-multilingual models, this function returns an empty list
          as language/task tokens are not needed.
        - The TO_LANGUAGE_CODE mapping from transformers is used to
          convert language names to codes.
        - Invalid language specifications raise ValueError with helpful
          suggestions for valid languages.
    """
    generation_config = generation_config or model_config
    is_multilingual = getattr(generation_config, "is_multilingual", None)
    decoder_input_ids = []
    if is_multilingual:
        if language is not None:
            language = language.lower()
            if language in generation_config.lang_to_id:
                language_token = language
            elif language in TO_LANGUAGE_CODE.values():
                language_token = f"<|{language}|>"
            elif language in TO_LANGUAGE_CODE:
                language_token = f"<|{TO_LANGUAGE_CODE[language]}|>"
            else:
                acceptable_languages = (
                    list(TO_LANGUAGE_CODE.values())
                    if len(language) == 2
                    else list(generation_config.lang_to_id)
                    if "<" in language or "|" in language or ">" in language
                    else list(TO_LANGUAGE_CODE)
                )
                raise ValueError(f"Unsupported language: {language}. Language should be one of: {acceptable_languages}.")

            decoder_input_ids.append((1, generation_config.lang_to_id[language_token]))

        if task is not None:
            decoder_input_ids.append((2, generation_config.task_to_id[task]))
        else:
            decoder_input_ids.append((2, generation_config.task_to_id["transcribe"]))

    if (
        not return_timestamps
        and decoder_input_ids
        and decoder_input_ids[-1][0] != generation_config.no_timestamps_token_id
    ):
        next_idx = (decoder_input_ids[-1][0] + 1) if decoder_input_ids else 1
        decoder_input_ids.append((next_idx, generation_config.no_timestamps_token_id))

    return decoder_input_ids
