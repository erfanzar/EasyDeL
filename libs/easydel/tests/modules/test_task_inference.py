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
"""Task selection is pure configuration processing, with no model computation."""

import json
from types import SimpleNamespace

import pytest
from easydel.infra.factory import TaskType
from easydel.modules.auto import infer_task_from_config
from easydel.modules.auto.auto_configuration import infer_task_from_hf_config

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("as_object", [False, True])
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"model_type": "muse_glimmer"}, TaskType.IMAGE_TEXT_TO_TEXT),
        ({"model_type": "aya_vision"}, TaskType.IMAGE_TEXT_TO_TEXT),
        ({"model_type": "clip"}, TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION),
        ({"model_type": "siglip"}, TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION),
        ({"is_encoder_decoder": True}, TaskType.SEQUENCE_TO_SEQUENCE),
        ({"architectures": ["UnknownModel", "ModelForSequenceClassification"]}, TaskType.SEQUENCE_CLASSIFICATION),
        ({"architectures": ["ModelForCausalLM", "ModelForImageTextToText"]}, TaskType.IMAGE_TEXT_TO_TEXT),
        ({"architectures": ["ModelForCausalLM"], "model_type": "qwen3_vl"}, TaskType.IMAGE_TEXT_TO_TEXT),
        ({"model_type": "muse_glimmer", "is_encoder_decoder": True}, TaskType.IMAGE_TEXT_TO_TEXT),
        ({"model_type": "whisper", "is_encoder_decoder": True}, TaskType.SPEECH_SEQUENCE_TO_SEQUENCE),
        ({"model_type": "speech2text"}, TaskType.SPEECH_SEQUENCE_TO_SEQUENCE),
        ({"architectures": ["ModelForSpeechSeq2Seq"]}, TaskType.SPEECH_SEQUENCE_TO_SEQUENCE),
        ({"architectures": ["ModelForVision2Seq"]}, TaskType.IMAGE_TEXT_TO_TEXT),
        ({"architectures": ["T5ForConditionalGeneration"], "is_encoder_decoder": True}, TaskType.SEQUENCE_TO_SEQUENCE),
        ({"architectures": ["ModelForSeq2SeqLM"]}, TaskType.SEQUENCE_TO_SEQUENCE),
        ({"architectures": ["ModelForConditionalGeneration"]}, TaskType.IMAGE_TEXT_TO_TEXT),
        ({"architectures": ["ModelForCausalLM"]}, TaskType.CAUSAL_LM),
        ({"architectures": ["ModelForAudioClassification"]}, TaskType.AUDIO_CLASSIFICATION),
        ({"architectures": ["ModelForImageClassification"]}, TaskType.IMAGE_CLASSIFICATION),
        ({"architectures": ["ModelForZeroShotImageClassification"]}, TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION),
        ({"architectures": ["ModelForEmbedding"]}, TaskType.EMBEDDING),
        ({"architectures": ["ModelForSentenceEmbedding"]}, TaskType.EMBEDDING),
        ({"architectures": ["QwenOmniForConditionalGeneration"]}, TaskType.ANY_TO_ANY),
        ({"architectures": ["ModelForDiffusionLM"]}, TaskType.DIFFUSION_LM),
        ({"model_type": "custom_diffusion_model"}, TaskType.DIFFUSION_LM),
        ({"model_type": "custom_vision"}, TaskType.BASE_VISION),
        ({"model_type": "cohere", "architectures": None}, None),
        ({"architectures": ["UnrecognizedModel"]}, None),
        ({"architectures": "ModelForSequenceClassification"}, TaskType.SEQUENCE_CLASSIFICATION),
    ],
)
def test_loaded_configuration_task_inference(config, expected, as_object):
    loaded = SimpleNamespace(**config) if as_object else config
    assert infer_task_from_config(loaded) is expected


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"model_type": "muse_glimmer"}, TaskType.IMAGE_TEXT_TO_TEXT),
        ({"model_type": "clip"}, TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION),
        ({"is_encoder_decoder": True}, TaskType.SEQUENCE_TO_SEQUENCE),
        ({"architectures": ["UnknownModel", "ModelForSequenceClassification"]}, TaskType.SEQUENCE_CLASSIFICATION),
    ],
)
def test_local_path_uses_shared_configuration_semantics(tmp_path, config, expected):
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert infer_task_from_hf_config(str(tmp_path)) is expected


def test_ambiguous_conditional_generation_retains_both_callers_policies():
    config = {"architectures": ["CustomForConditionalGeneration"]}
    assert infer_task_from_config(config) is TaskType.IMAGE_TEXT_TO_TEXT
    assert infer_task_from_config(config, conditional_generation_task=TaskType.CAUSAL_LM) is TaskType.CAUSAL_LM
    config["is_encoder_decoder"] = True
    assert (
        infer_task_from_config(config, conditional_generation_task=TaskType.CAUSAL_LM) is TaskType.SEQUENCE_TO_SEQUENCE
    )


def test_hub_loading_preserves_authenticated_download_arguments(tmp_path, monkeypatch):
    import huggingface_hub

    config_path = tmp_path / "downloaded.json"
    config_path.write_text(json.dumps({"model_type": "muse_glimmer"}))
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(config_path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert infer_task_from_hf_config("example/model") is TaskType.IMAGE_TEXT_TO_TEXT
    assert calls == [{"repo_id": "example/model", "filename": "config.json", "repo_type": "model"}]


def test_failed_hub_download_retains_requests_fallback(monkeypatch):
    import huggingface_hub
    import requests

    def download(**kwargs):
        raise OSError("Hub temporarily unavailable")

    def get(url, *, timeout):
        assert url == "https://huggingface.co/example/model/raw/main/config.json"
        assert timeout == 10
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"model_type": "clip"})

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(requests, "get", get)
    assert infer_task_from_hf_config("example/model") is TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION
