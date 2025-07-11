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


from flax import nnx as nn
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

from easydel.utils.compiling_utils import ejit


@ejit(static_argnames=["graphdef", "inference_config", "return_timestamps"])
def _compiled_generate(
    graphdef,
    graphstate,
    inference_config,
    input_features,
    decoder_input_ids,
    return_timestamps,
):
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
    """Helper function to get decoder input IDs for Whisper."""
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
