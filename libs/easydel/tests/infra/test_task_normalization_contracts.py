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

"""Array-free compatibility and task-dispatch regressions."""

import pytest
from easydel.infra import base_config
from easydel.infra.factory import TASK_ALIASES, Registry, TaskType, normalize_task
from easydel.infra.mixins import bridge
from easydel.modules._base import (
    BaseQuestionAnsweringModule,
    BaseTokenClassificationModule,
    create_task_model_class,
)

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("task", list(TaskType))
def test_shared_normalization_accepts_every_canonical_task(task):
    assert normalize_task(task) is task
    assert normalize_task(f"  {task.value.upper()}  ") is task
    assert normalize_task(task.value.replace("-", "_")) is task


@pytest.mark.parametrize(
    ("alias", "task"),
    [
        ("causal_lm", TaskType.CAUSAL_LM),
        ("lm", TaskType.CAUSAL_LM),
        ("seq2seq", TaskType.SEQUENCE_TO_SEQUENCE),
        ("sequence_to_sequence", TaskType.SEQUENCE_TO_SEQUENCE),
        ("speech_seq2seq", TaskType.SPEECH_SEQUENCE_TO_SEQUENCE),
        ("image_text_to_text", TaskType.IMAGE_TEXT_TO_TEXT),
        ("zero_shot_image_classification", TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION),
        ("diffusion_lm", TaskType.DIFFUSION_LM),
        ("sequence_classification", TaskType.SEQUENCE_CLASSIFICATION),
        ("embedding", TaskType.EMBEDDING),
        ("base", TaskType.BASE_MODULE),
    ],
)
def test_shared_normalization_preserves_legacy_aliases(alias, task):
    assert TASK_ALIASES[alias] is task
    assert normalize_task(f" {alias.upper().replace('_', '-')} ") is task


def test_shared_normalization_preserves_unknown_and_none_behavior():
    assert normalize_task(None) is None
    assert normalize_task("unknown_task") is None
    assert normalize_task("") is None


@pytest.mark.parametrize("consumer", ["elarge", "auto_configuration"])
def test_task_normalization_consumers_reexport_canonical_implementation(consumer):
    from easydel.infra.elarge import processing
    from easydel.modules.auto import auto_configuration

    module = processing if consumer == "elarge" else auto_configuration
    assert module.normalize_task is normalize_task
    assert module.TASK_ALIASES is TASK_ALIASES


@pytest.mark.parametrize("consumer", ["elarge", "auto_configuration"])
def test_task_normalization_consumers_accept_all_tasks_and_legacy_aliases(consumer):
    from easydel.infra.elarge import processing
    from easydel.modules.auto import auto_configuration

    module = processing if consumer == "elarge" else auto_configuration
    for task in TaskType:
        assert module.normalize_task(task) is task
        assert module.normalize_task(task.value) is task
        assert module.normalize_task(f" {task.value.upper().replace('-', '_')} ") is task
    for alias, task in TASK_ALIASES.items():
        assert module.normalize_task(alias) is task
        assert module.normalize_task(f" {alias.upper().replace('_', '-')} ") is task
    for value in (None, "", "unknown_task"):
        assert module.normalize_task(value) is None


@pytest.mark.parametrize(
    ("task", "base_class", "suffix"),
    [
        (TaskType.TOKEN_CLASSIFICATION, BaseTokenClassificationModule, "ForTokenClassification"),
        (TaskType.QUESTION_ANSWERING, BaseQuestionAnsweringModule, "ForQuestionAnswering"),
    ],
)
def test_token_and_qa_tasks_dispatch_and_register_without_arrays(task, base_class, suffix):
    class Config:
        pass

    class Backbone:
        pass

    model_type = f"task_wrapper_test_{task.value}"
    wrapper = create_task_model_class(task, "TaskWrapperTest", Backbone, Config, model_type)
    assert wrapper.__name__ == "TaskWrapperTest" + suffix
    assert issubclass(wrapper, base_class)
    assert wrapper._task_type is task
    assert wrapper._config_class is Config
    assert wrapper._model_type == model_type

    registry = Registry()
    registry.register_config(model_type)(Config)
    registered = registry.register_module(task, Config, model_type)(wrapper)
    registration = registry.get_module_registration(task.value, model_type)
    assert registration.module is wrapper is registered
    assert registration.config is registry.get_config(model_type) is Config
    assert wrapper._model_task is task


@pytest.mark.parametrize(
    ("name", "filename"),
    [
        ("CHAT_TEMPLATE_NAME", "chat_template.json"),
        ("FEATURE_EXTRACTOR_NAME", "preprocessor_config.json"),
        ("IMAGE_PROCESSOR_NAME", "preprocessor_config.json"),
        ("MODEL_CARD_NAME", "modelcard.json"),
        ("PROCESSOR_NAME", "processor_config.json"),
        ("TF2_WEIGHTS_INDEX_NAME", "tf_model.h5.index.json"),
        ("TF2_WEIGHTS_NAME", "tf_model.h5"),
        ("TF_WEIGHTS_NAME", "model.ckpt"),
        ("WEIGHTS_NAME", "pytorch_model.bin"),
        ("WEIGHTS_INDEX_NAME", "pytorch_model.bin.index.json"),
        ("SAFE_WEIGHTS_NAME", "model.safetensors"),
        ("SAFE_WEIGHTS_INDEX_NAME", "model.safetensors.index.json"),
        ("GENERATION_CONFIG_NAME", "generation_config.json"),
    ],
)
def test_checkpoint_filename_reexports_preserve_public_values(name, filename):
    assert getattr(bridge, name) == filename
    assert getattr(base_config, name) == filename
