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

"""Tests for ``easydel.infra.factory``.

The factory exposes a registry for configurations and modules. The tests use
*local* ``Registry`` instances (not the global ``registry`` singleton) so we
don't pollute global state. Tests that touch the global singleton are
read-only assertions.
"""

from __future__ import annotations

import pytest

from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.factory import (
    ConfigType,
    ModuleRegistration,
    Registry,
    TaskType,
)
from easydel.infra.factory import (
    register_config as global_register_config,
)
from easydel.infra.factory import (
    register_module as global_register_module,
)
from easydel.infra.factory import (
    registry as global_registry,
)


def test_config_type_is_str_enum_with_module_config_value():
    assert issubclass(ConfigType, str)
    assert ConfigType.MODULE_CONFIG == "module-config"
    assert ConfigType("module-config") is ConfigType.MODULE_CONFIG


def test_task_type_str_values_are_kebab_case_and_unique():
    """Each task value should be a unique kebab-case string."""
    seen: set[str] = set()
    for task in TaskType:
        assert isinstance(task.value, str)
        assert task.value == task.value.lower()
        assert " " not in task.value
        assert task.value not in seen, f"duplicate TaskType value: {task.value}"
        seen.add(task.value)


@pytest.mark.parametrize(
    "task_str,task_enum",
    [
        ("causal-language-model", TaskType.CAUSAL_LM),
        ("vision-language-model", TaskType.VISION_LM),
        ("base-module", TaskType.BASE_MODULE),
        ("sequence-to-sequence", TaskType.SEQUENCE_TO_SEQUENCE),
        ("sequence-classification", TaskType.SEQUENCE_CLASSIFICATION),
        ("audio-classification", TaskType.AUDIO_CLASSIFICATION),
        ("auto-bind", TaskType.AUTO_BIND),
        ("embedding", TaskType.EMBEDDING),
    ],
)
def test_task_type_string_literals_round_trip(task_str: str, task_enum: TaskType):
    """The string literals named in get_module_registration's type hint must lookup correctly."""
    assert TaskType(task_str) is task_enum


def test_new_registry_has_empty_pre_initialized_dictionaries():
    reg = Registry()

    assert ConfigType.MODULE_CONFIG in reg.config_registry
    assert reg.config_registry[ConfigType.MODULE_CONFIG] == {}

    assert set(reg.task_registry.keys()) == set(TaskType)
    for inner in reg.task_registry.values():
        assert inner == {}


def test_global_registry_is_a_registry_instance():
    assert isinstance(global_registry, Registry)

    assert getattr(global_register_config, "__self__", None) is global_registry
    assert getattr(global_register_module, "__self__", None) is global_registry


def _make_dummy_config_class(name: str = "DummyConfig") -> type[EasyDeLBaseConfig]:
    """Synthesize a minimal config class with a simple ``__init__``."""

    def __init__(self, hidden: int = 8, depth: int = 2):
        self.hidden = hidden
        self.depth = depth

    cls = type(name, (EasyDeLBaseConfig,), {"__init__": __init__})
    return cls


def test_register_config_stores_class_under_identifier():
    reg = Registry()
    DummyConfig = _make_dummy_config_class()
    reg.register_config("dummy")(DummyConfig)
    assert reg.get_config("dummy") is DummyConfig


def test_register_config_accepts_config_field_argument():
    reg = Registry()
    DummyConfig = _make_dummy_config_class()
    reg.register_config("dummy", config_field=ConfigType.MODULE_CONFIG)(DummyConfig)

    assert reg.config_registry[ConfigType.MODULE_CONFIG]["dummy"] is DummyConfig


def test_register_config_returns_the_class_unchanged():
    """The decorator must return the original class so ``@register_config`` works inline."""
    reg = Registry()
    DummyConfig = _make_dummy_config_class()
    decorated = reg.register_config("dummy")(DummyConfig)
    assert decorated is DummyConfig


def test_register_config_attaches_pretty_string_methods():
    """The decorator overrides ``__str__`` and ``__repr__`` for nicer logs."""
    reg = Registry()
    DummyConfig = _make_dummy_config_class()
    reg.register_config("dummy")(DummyConfig)

    instance = DummyConfig.__new__(DummyConfig)
    instance.hidden = 16
    instance.depth = 4

    rendered = str(instance)
    assert rendered.startswith("DummyConfig(")
    assert "hidden=16" in rendered
    assert "depth=4" in rendered
    assert repr(instance) == rendered


def test_get_config_raises_keyerror_for_unknown_type():
    reg = Registry()
    with pytest.raises(KeyError):
        reg.get_config("never_registered")


def _make_dummy_module_class(name: str = "DummyModule") -> type:
    return type(name, (), {})


def test_register_module_creates_module_registration_entry():
    reg = Registry()
    DummyConfig = _make_dummy_config_class()
    DummyModule = _make_dummy_module_class()
    reg.register_module(
        task_type=TaskType.CAUSAL_LM,
        config=DummyConfig,
        model_type="dummy",
        embedding_layer_names=["model.embed_tokens"],
        layernorm_names=["model.norm"],
    )(DummyModule)

    registration = reg.get_module_registration(TaskType.CAUSAL_LM, "dummy")
    assert isinstance(registration, ModuleRegistration)
    assert registration.module is DummyModule
    assert registration.config is DummyConfig
    assert registration.embedding_layer_names == ["model.embed_tokens"]
    assert registration.layernorm_names == ["model.norm"]


def test_register_module_stamps_class_with_task_and_model_type():
    """The decorator sets ``_model_task`` and ``_model_type`` on the class for downstream lookups."""
    reg = Registry()
    DummyConfig = _make_dummy_config_class()
    DummyModule = _make_dummy_module_class()
    reg.register_module(task_type=TaskType.CAUSAL_LM, config=DummyConfig, model_type="dummy")(DummyModule)
    assert DummyModule._model_task is TaskType.CAUSAL_LM
    assert DummyModule._model_type == "dummy"


def test_register_module_with_no_metadata_defaults_to_none():
    reg = Registry()
    DummyConfig = _make_dummy_config_class()
    DummyModule = _make_dummy_module_class()
    reg.register_module(task_type=TaskType.BASE_MODULE, config=DummyConfig, model_type="dummy")(DummyModule)
    reg_entry = reg.get_module_registration(TaskType.BASE_MODULE, "dummy")
    assert reg_entry.embedding_layer_names is None
    assert reg_entry.layernorm_names is None


def test_register_module_supports_multiple_task_types_for_same_model():
    """The same model_type can be registered under different tasks (e.g. llama causal-lm + seq-cls)."""
    reg = Registry()
    DummyConfig = _make_dummy_config_class()
    DummyCausalLM = _make_dummy_module_class("DummyCausalLM")
    DummyClassifier = _make_dummy_module_class("DummyClassifier")

    reg.register_module(task_type=TaskType.CAUSAL_LM, config=DummyConfig, model_type="dummy")(DummyCausalLM)
    reg.register_module(
        task_type=TaskType.SEQUENCE_CLASSIFICATION,
        config=DummyConfig,
        model_type="dummy",
    )(DummyClassifier)

    assert reg.get_module_registration(TaskType.CAUSAL_LM, "dummy").module is DummyCausalLM
    assert reg.get_module_registration(TaskType.SEQUENCE_CLASSIFICATION, "dummy").module is DummyClassifier


def test_register_module_returns_class_unchanged():
    reg = Registry()
    DummyConfig = _make_dummy_config_class()
    DummyModule = _make_dummy_module_class()
    decorated = reg.register_module(
        task_type=TaskType.BASE_MODULE,
        config=DummyConfig,
        model_type="dummy",
    )(DummyModule)
    assert decorated is DummyModule


def test_get_module_registration_accepts_string_task_type():
    """``get_module_registration`` accepts the kebab-case string aliases."""
    reg = Registry()
    DummyConfig = _make_dummy_config_class()
    DummyModule = _make_dummy_module_class()
    reg.register_module(task_type=TaskType.CAUSAL_LM, config=DummyConfig, model_type="dummy")(DummyModule)
    registration = reg.get_module_registration("causal-language-model", "dummy")
    assert registration.module is DummyModule


def test_get_module_registration_raises_for_unknown_task():
    reg = Registry()
    with pytest.raises(KeyError, match="task type"):
        reg.get_module_registration("non-existent-task", "dummy")


def test_get_module_registration_raises_for_unknown_model_type():
    reg = Registry()
    with pytest.raises(KeyError, match="model type"):
        reg.get_module_registration(TaskType.CAUSAL_LM, "no_such_model")


def test_task_registry_property_exposes_internal_state():
    reg = Registry()
    DummyConfig = _make_dummy_config_class()
    DummyModule = _make_dummy_module_class()
    reg.register_module(task_type=TaskType.CAUSAL_LM, config=DummyConfig, model_type="dummy")(DummyModule)
    snapshot = reg.task_registry
    assert "dummy" in snapshot[TaskType.CAUSAL_LM]
    assert snapshot[TaskType.CAUSAL_LM]["dummy"].module is DummyModule


def test_config_registry_property_exposes_internal_state():
    reg = Registry()
    DummyConfig = _make_dummy_config_class()
    reg.register_config("dummy")(DummyConfig)
    snapshot = reg.config_registry
    assert "dummy" in snapshot[ConfigType.MODULE_CONFIG]


def test_global_registry_has_canonical_models_registered():
    """Touching ``LlamaForCausalLM`` triggers its registration into the global registry."""
    import easydel as ed

    _ = ed.LlamaForCausalLM

    causal = global_registry.task_registry[TaskType.CAUSAL_LM]
    assert "llama" in causal
    reg_entry = causal["llama"]
    assert reg_entry.module is not None
    assert reg_entry.config is not None
    assert reg_entry.module.__name__ == "LlamaForCausalLM"


def test_global_registry_module_classes_carry_model_task_attribute():
    import easydel as ed

    cls = ed.LlamaForCausalLM
    assert getattr(cls, "_model_task", None) is TaskType.CAUSAL_LM
    assert getattr(cls, "_model_type", None) == "llama"


def test_module_registration_dataclass_round_trip():
    """``ModuleRegistration`` is a simple container; field access must work for all 4 attrs."""
    DummyConfig = _make_dummy_config_class()
    DummyModule = _make_dummy_module_class()
    reg_entry = ModuleRegistration(
        module=DummyModule,
        config=DummyConfig,
        embedding_layer_names=["a", "b"],
        layernorm_names=["c"],
    )
    assert reg_entry.module is DummyModule
    assert reg_entry.config is DummyConfig
    assert reg_entry.embedding_layer_names == ["a", "b"]
    assert reg_entry.layernorm_names == ["c"]


def test_module_registration_defaults_are_none():
    DummyConfig = _make_dummy_config_class()
    DummyModule = _make_dummy_module_class()
    reg_entry = ModuleRegistration(module=DummyModule, config=DummyConfig)
    assert reg_entry.embedding_layer_names is None
    assert reg_entry.layernorm_names is None
