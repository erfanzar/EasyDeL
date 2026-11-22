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

"""The PP microbatch skip diagnostics must not depend on an environment knob."""

import importlib
import inspect
from types import SimpleNamespace

import pytest


@pytest.mark.cpu_ok
def test_disabled_pipeline_falls_back_independently_of_old_debug_environment(monkeypatch, capsys):
    monkeypatch.setenv("ESURGE_PP_MB_DEBUG", "1")
    module = importlib.reload(importlib.import_module("easydel.inference.esurge.runners.executors.pipeline_microbatch"))
    manager = SimpleNamespace(_model_executor=SimpleNamespace(pipeline_plan=None, pipeline_runtime=None))
    executor = module.PipelineMicrobatchExecutor(manager)
    kwargs = dict.fromkeys(inspect.signature(executor.try_execute).parameters)
    assert executor.try_execute(**kwargs) is None
    assert capsys.readouterr().out == ""
