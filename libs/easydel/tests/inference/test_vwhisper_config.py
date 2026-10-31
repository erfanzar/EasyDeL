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

"""vWhisper config, decoder-id and chunking regression tests (no array computation)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.vwhisper.config import vWhisperInferenceConfig
from easydel.inference.vwhisper.core import vWhisperInference
from easydel.inference.vwhisper.generation import get_decoder_input_ids

pytestmark = pytest.mark.cpu_ok


def _generation_config(is_multilingual: bool) -> SimpleNamespace:
    return SimpleNamespace(
        is_multilingual=is_multilingual,
        lang_to_id={"<|en|>": 11},
        task_to_id={"transcribe": 22, "translate": 23},
        no_timestamps_token_id=99,
    )


def test_config_accepts_documented_constructor_arguments():
    config = vWhisperInferenceConfig(
        batch_size=8, max_length=448, return_timestamps=True, language="en", task="translate"
    )
    assert (config.batch_size, config.max_length) == (8, 448)
    assert config.return_timestamps is True
    assert config.language == "en"
    assert config.task == "translate"
    assert vWhisperInferenceConfig().language is None


def test_config_hash_depends_on_language():
    assert hash(vWhisperInferenceConfig(language="en")) == hash(vWhisperInferenceConfig(language="en"))
    assert hash(vWhisperInferenceConfig(language="en")) != hash(vWhisperInferenceConfig(language="fr"))


def test_decoder_ids_force_language_task_and_no_timestamps():
    ids = get_decoder_input_ids(
        model_config=SimpleNamespace(),
        generation_config=_generation_config(True),
        language="en",
        task="transcribe",
        return_timestamps=False,
    )
    assert ids == [(1, 11), (2, 22), (3, 99)]


def test_decoder_ids_is_multilingual_override_wins_over_generation_config():
    ids = get_decoder_input_ids(
        model_config=SimpleNamespace(),
        generation_config=_generation_config(False),
        language="en",
        task="translate",
        return_timestamps=True,
        is_multilingual=True,
    )
    assert ids == [(1, 11), (2, 23)]


def test_decoder_ids_do_not_duplicate_existing_no_timestamps_token():
    config = _generation_config(True)
    config.task_to_id["transcribe"] = config.no_timestamps_token_id
    ids = get_decoder_input_ids(
        model_config=SimpleNamespace(), generation_config=config, language="en", task="transcribe"
    )
    assert ids == [(1, 11), (2, 99)]


def test_chunk_length_equal_to_total_stride_is_rejected_with_value_error():
    fake_self = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000))
    audio = np.zeros(32000, dtype=np.float32)
    with pytest.raises(ValueError, match="Chunk length"):
        next(vWhisperInference._process_model_inputs(fake_self, audio, chunk_length_s=1.0, stride_length_s=0.5))
