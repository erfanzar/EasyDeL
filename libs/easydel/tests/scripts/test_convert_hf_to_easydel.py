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

from types import SimpleNamespace

import pytest

from scripts.convert_hf_to_easydel import _infer_task_from_hf_config


@pytest.mark.parametrize(
    ("model_type", "architectures", "is_encoder_decoder", "expected"),
    [
        ("qwen3_5", ["Qwen3_5ForConditionalGeneration"], False, "image_text_to_text"),
        ("qwen3_5_moe", ["Qwen3_5MoeForConditionalGeneration"], False, "image_text_to_text"),
        ("gemma4", ["Gemma4ForConditionalGeneration"], False, "image_text_to_text"),
        ("glm_moe_dsa", ["GlmMoeDsaForCausalLM"], False, "causal_lm"),
        ("whisper", ["WhisperForConditionalGeneration"], True, "speech_seq2seq"),
        ("t5", ["T5ForConditionalGeneration"], True, "seq2seq"),
    ],
)
def test_infer_task_from_hf_config(model_type, architectures, is_encoder_decoder, expected):
    config = SimpleNamespace(
        model_type=model_type,
        architectures=architectures,
        is_encoder_decoder=is_encoder_decoder,
    )

    assert _infer_task_from_hf_config(config) == expected
