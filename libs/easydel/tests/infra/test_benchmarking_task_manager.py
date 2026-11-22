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

"""A TaskManager cached for one (include_path, include_defaults) key must not leak into other configurations."""

import sys
import types
import typing as tp

import pytest
from easydel.infra.elarge import benchmarking


class _FakeTaskManager:
    built: tp.ClassVar[list["_FakeTaskManager"]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.all_tasks = {"foo"}
        _FakeTaskManager.built.append(self)


@pytest.fixture
def fake_lm_eval(monkeypatch):
    _FakeTaskManager.built = []
    tasks_module = types.ModuleType("lm_eval.tasks")
    tasks_module.TaskManager = _FakeTaskManager
    root = types.ModuleType("lm_eval")
    root.tasks = tasks_module
    monkeypatch.setitem(sys.modules, "lm_eval", root)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", tasks_module)
    monkeypatch.setattr(benchmarking, "_cached_task_manager", None)
    monkeypatch.setattr(benchmarking, "_cached_task_manager_key", None)
    return tasks_module


@pytest.mark.cpu_ok
def test_default_instruct_resolution_ignores_manager_cached_for_custom_include_path(fake_lm_eval):
    custom = _FakeTaskManager(include_path="/custom")
    custom.all_tasks = {"foo", "foo_instruct"}
    _FakeTaskManager.built.clear()
    benchmarking._cached_task_manager = custom
    benchmarking._cached_task_manager_key = (("/custom",), True)

    resolved = benchmarking.maybe_resolve_instruct_task_variants(["foo"], task_manager=None, apply_chat_template=True)

    assert resolved == ["foo"]
    assert len(_FakeTaskManager.built) == 1
    assert _FakeTaskManager.built[0].kwargs["include_path"] is None
    assert _FakeTaskManager.built[0].kwargs["include_defaults"] is True


@pytest.mark.cpu_ok
def test_default_instruct_resolution_reuses_manager_cached_for_default_key(fake_lm_eval):
    default = _FakeTaskManager()
    default.all_tasks = {"foo", "foo_instruct"}
    _FakeTaskManager.built.clear()
    benchmarking._cached_task_manager = default
    benchmarking._cached_task_manager_key = (None, True)

    resolved = benchmarking.maybe_resolve_instruct_task_variants(["foo"], task_manager=None, apply_chat_template=True)

    assert resolved == ["foo_instruct"]
    assert _FakeTaskManager.built == []


@pytest.mark.cpu_ok
@pytest.mark.parametrize("custom_config", [{"include_path": ["/custom"]}, {"include_defaults": False}])
def test_default_evaluation_does_not_inherit_custom_task_registry(fake_lm_eval, monkeypatch, custom_config):
    evaluator = types.ModuleType("lm_eval.evaluator")
    evaluator.simple_evaluate = lambda **kwargs: {"results": {}, "task_manager": kwargs["task_manager"]}
    monkeypatch.setattr(sys.modules["lm_eval"], "evaluator", evaluator, raising=False)
    monkeypatch.setitem(sys.modules, "lm_eval.evaluator", evaluator)
    adapters = types.ModuleType("easydel.inference.evaluations")

    class FakeAdapter:
        def __init__(self, **kwargs):
            pass

        def stop(self):
            pass

    adapters.eSurgeLMEvalAdapter = FakeAdapter
    monkeypatch.setitem(sys.modules, "easydel.inference.evaluations", adapters)
    args = {"surge": object(), "processor": object(), "tasks": ["foo"], "max_length": 16, "fallback_batch_size": 1}
    custom = benchmarking.run_lm_eval_with_esurge(**args, eval_config=custom_config)
    default = benchmarking.run_lm_eval_with_esurge(**args)
    assert custom["task_manager"] is not None
    assert default["task_manager"] is None


@pytest.mark.cpu_ok
def test_explicit_falsey_task_manager_is_honored(fake_lm_eval, monkeypatch):
    evaluator = types.ModuleType("lm_eval.evaluator")
    evaluator.simple_evaluate = lambda **kwargs: {"results": {}, "task_manager": kwargs["task_manager"]}
    monkeypatch.setattr(sys.modules["lm_eval"], "evaluator", evaluator, raising=False)
    monkeypatch.setitem(sys.modules, "lm_eval.evaluator", evaluator)
    adapters = types.ModuleType("easydel.inference.evaluations")
    adapters.eSurgeLMEvalAdapter = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, "easydel.inference.evaluations", adapters)

    class EmptyManager:
        all_tasks = ()

        def __bool__(self):
            return False

    manager = EmptyManager()
    result = benchmarking.run_lm_eval_with_esurge(
        surge=object(),
        processor=object(),
        tasks=["foo"],
        max_length=16,
        fallback_batch_size=1,
        eval_config={"task_manager": manager},
    )
    assert result["task_manager"] is manager
