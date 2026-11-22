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
"""Conversion CLI metadata regressions without model or array computation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from easydel.scripts import convert_hf_to_easydel as convert

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("failed_artifact", [None, "processor", "tokenizer", "image-processor", "feature-extractor"])
def test_remote_preprocessing_uses_staged_metadata_and_optional_failures_do_not_abort(
    tmp_path, monkeypatch, failed_artifact
):
    import easydel as ed
    import transformers
    from easydel.infra.mixins import bridge

    metadata = tmp_path / "metadata"
    metadata.mkdir()
    output = tmp_path / "converted"
    warnings = []
    monkeypatch.setattr(convert.logger, "warning", warnings.append)

    class _Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def save_pretrained(self, directory, **kwargs):
            (Path(directory) / "weights-converted").write_text("ok")

    for name in (
        "AutoEasyDeLModelForCausalLM",
        "AutoEasyDeLModelForSeq2SeqLM",
        "AutoEasyDeLModelForSpeechSeq2Seq",
        "AutoEasyDeLModelForImageTextToText",
        "AutoEasyDeLModelForZeroShotImageClassification",
        "AutoEasyDeLModelForSequenceClassification",
        "AutoEasyDeLModelForDiffusionLM",
        "AutoEasyDeLModel",
        "AutoEasyDeLVisionModel",
        "AutoEasyDeLAnyToAnyModel",
    ):
        monkeypatch.setitem(ed.__dict__, name, _Model)
    monkeypatch.setattr(bridge, "stage_remote_metadata", lambda *a, **k: (str(metadata), None))
    monkeypatch.setattr(transformers, "AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace()))

    def loader(artifact):
        class _Artifact:
            @classmethod
            def from_pretrained(cls, source, **kwargs):
                assert source == str(metadata)
                if artifact == failed_artifact:
                    raise ValueError(f"No {artifact} for this model")
                return cls()

            def save_pretrained(self, directory):
                (Path(directory) / f"{artifact}.json").write_text("{}")

        return _Artifact

    for name, artifact in (
        ("AutoProcessor", "processor"),
        ("AutoTokenizer", "tokenizer"),
        ("AutoImageProcessor", "image-processor"),
        ("AutoFeatureExtractor", "feature-extractor"),
    ):
        monkeypatch.setitem(transformers.__dict__, name, loader(artifact))

    convert.main(
        [
            "--source",
            "gs://bucket/model",
            "--out",
            str(output),
            "--task",
            "image_text_to_text",
            "--convert-mode",
            "from_pretrained",
            "--no-push-to-hub",
        ]
    )

    for artifact in ("processor", "tokenizer", "image-processor", "feature-extractor"):
        artifact_path = output / f"{artifact}.json"
        assert artifact_path.exists() is (artifact != failed_artifact)
        if artifact != failed_artifact:
            assert artifact_path.read_text() == "{}"
    assert (output / "weights-converted").exists()
    assert len(warnings) == (0 if failed_artifact is None else 1)
    if failed_artifact is not None:
        assert f"No {failed_artifact} for this model" in warnings[0]


@pytest.mark.parametrize(
    ("config", "registry_task", "cli_task"),
    [
        ({"model_type": "muse_glimmer"}, "IMAGE_TEXT_TO_TEXT", "image_text_to_text"),
        ({"is_encoder_decoder": True}, "SEQUENCE_TO_SEQUENCE", "seq2seq"),
        ({"model_type": "clip"}, "ZERO_SHOT_IMAGE_CLASSIFICATION", "zero_shot_image_classification"),
        ({"model_type": "siglip"}, "ZERO_SHOT_IMAGE_CLASSIFICATION", "zero_shot_image_classification"),
        (
            {"architectures": ["UnknownModel", "ModelForSequenceClassification"]},
            "SEQUENCE_CLASSIFICATION",
            "sequence_classification",
        ),
        ({"model_type": "whisper"}, "SPEECH_SEQUENCE_TO_SEQUENCE", "speech_seq2seq"),
        ({"model_type": "speech_to_text"}, "SPEECH_SEQUENCE_TO_SEQUENCE", "speech_seq2seq"),
        ({"model_type": "speech-to-text"}, "SPEECH_SEQUENCE_TO_SEQUENCE", "speech_seq2seq"),
        ({"architectures": ["ModelForSpeechSeq2Seq"]}, "SPEECH_SEQUENCE_TO_SEQUENCE", "speech_seq2seq"),
        (
            {"model_type": "muse_glimmer", "architectures": ["ModelForCausalLM"], "is_encoder_decoder": True},
            "IMAGE_TEXT_TO_TEXT",
            "image_text_to_text",
        ),
        ({"architectures": ["ModelForVision2Seq"]}, "IMAGE_TEXT_TO_TEXT", "image_text_to_text"),
        ({"architectures": ["ModelForSeq2SeqLM"]}, "SEQUENCE_TO_SEQUENCE", "seq2seq"),
        (
            {"architectures": ["ModelForZeroShotImageClassification"]},
            "ZERO_SHOT_IMAGE_CLASSIFICATION",
            "zero_shot_image_classification",
        ),
        ({"model_type": "model_diffusion"}, "DIFFUSION_LM", "diffusion_lm"),
        ({"architectures": ["ModelForDiffusionLM"]}, "DIFFUSION_LM", "diffusion_lm"),
        ({"architectures": ["ModelForCausalLM"]}, "CAUSAL_LM", "causal_lm"),
        ({"architectures": ["QwenOmniForConditionalGeneration"]}, "ANY_TO_ANY", "any_to_any"),
        ({"model_type": "vision_encoder"}, "BASE_VISION", "vision"),
    ],
)
def test_shared_and_converter_task_inference_preserve_supported_configurations(
    tmp_path, config, registry_task, cli_task
):
    from easydel.infra.factory import TaskType
    from easydel.modules.auto.auto_configuration import infer_task_from_config, infer_task_from_hf_config

    expected = getattr(TaskType, registry_task)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    assert infer_task_from_config(config) == expected
    assert infer_task_from_config(SimpleNamespace(**config)) == expected
    # Exercise public file-based inference without Hub requests or array computation.
    assert infer_task_from_hf_config(str(tmp_path)) == expected
    assert convert._infer_task_from_hf_config(SimpleNamespace(**config)) == cli_task
    assert convert._infer_task_from_hf_config(config) == cli_task


@pytest.mark.parametrize(
    ("config", "cli_task"),
    [
        ({"architectures": ["ModelForConditionalGeneration"]}, "causal_lm"),
        ({"architectures": ["ModelForConditionalGeneration"], "is_encoder_decoder": True}, "seq2seq"),
        ({"architectures": ["ModelForConditionalGeneration"], "model_type": "muse_glimmer"}, "image_text_to_text"),
        ({"architectures": ["WhisperForConditionalGeneration"]}, "speech_seq2seq"),
        ({}, "causal_lm"),
        ({"architectures": ["UnknownModel"]}, "causal_lm"),
    ],
)
def test_converter_retains_historical_conditional_generation_and_unknown_fallback(config, cli_task):
    from easydel.infra.factory import TaskType
    from easydel.modules.auto.auto_configuration import infer_task_from_config

    if config == {"architectures": ["ModelForConditionalGeneration"]}:
        # The library keeps its own historical image-to-text default; conversion
        # explicitly requests causal LM for otherwise ambiguous architectures.
        assert infer_task_from_config(config) == TaskType.IMAGE_TEXT_TO_TEXT
        assert infer_task_from_config(config, conditional_generation_task=TaskType.CAUSAL_LM) == TaskType.CAUSAL_LM
    assert convert._infer_task_from_hf_config(SimpleNamespace(**config)) == cli_task


def test_converter_family_table_is_a_compatible_shared_reexport():
    from easydel.modules.auto.auto_configuration import IMAGE_TEXT_TO_TEXT_MODEL_TYPES

    assert convert.IMAGE_TEXT_TO_TEXT_MODEL_TYPES is IMAGE_TEXT_TO_TEXT_MODEL_TYPES
    assert "muse_glimmer" in convert.IMAGE_TEXT_TO_TEXT_MODEL_TYPES
    assert "aya_vision" in convert.IMAGE_TEXT_TO_TEXT_MODEL_TYPES
