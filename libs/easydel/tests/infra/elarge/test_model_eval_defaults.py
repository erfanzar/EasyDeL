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

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from easydel.inference.sampling_params import SamplingParams
from easydel.infra.elarge.model import eLargeModel


def test_eval_defaults_to_raw_task_prompts(monkeypatch):
    """`eLargeModel.eval` should follow upstream lm-eval and keep chat templating opt-in."""
    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {},
        "eval": {},
    }
    elm._tokenizer = object()
    elm.build_tokenizer = lambda: None
    elm.build_esurge = lambda: object()

    calls: dict[str, object] = {}

    class _DummyEvalAdapter:
        def __init__(self, **kwargs):
            calls["adapter_kwargs"] = kwargs

        def stop(self):
            calls["stopped"] = True

    def _simple_evaluate(**kwargs):
        calls["simple_evaluate_kwargs"] = kwargs
        return {"results": {}}

    lm_eval_module = ModuleType("lm_eval")
    lm_eval_module.evaluator = ModuleType("lm_eval.evaluator")
    lm_eval_module.evaluator.simple_evaluate = _simple_evaluate

    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval_module)
    monkeypatch.setattr("easydel.inference.evaluations.eSurgeLMEvalAdapter", _DummyEvalAdapter)

    results = elm.eval(["gsm8k"])

    assert results == {"results": {}}
    assert calls["simple_evaluate_kwargs"]["apply_chat_template"] is False
    assert calls["stopped"] is True


def test_eval_can_force_hard_max_new_tokens(monkeypatch):
    """`hard_max_new_tokens` should be forwarded to the eval adapter."""
    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {},
        "eval": {"max_new_tokens": 123, "hard_max_new_tokens": True},
    }
    elm._tokenizer = object()
    elm.build_tokenizer = lambda: None
    elm.build_esurge = lambda: object()

    calls: dict[str, object] = {}

    class _DummyEvalAdapter:
        def __init__(self, **kwargs):
            calls["adapter_kwargs"] = kwargs

        def stop(self):
            calls["stopped"] = True

    def _simple_evaluate(**kwargs):
        calls["simple_evaluate_kwargs"] = kwargs
        return {"results": {}}

    lm_eval_module = ModuleType("lm_eval")
    lm_eval_module.evaluator = ModuleType("lm_eval.evaluator")
    lm_eval_module.evaluator.simple_evaluate = _simple_evaluate

    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval_module)
    monkeypatch.setattr("easydel.inference.evaluations.eSurgeLMEvalAdapter", _DummyEvalAdapter)

    elm.eval(["humaneval"])

    assert calls["adapter_kwargs"]["max_new_tokens"] == 123
    assert calls["adapter_kwargs"]["hard_max_new_tokens"] is True


def test_eval_forwards_sampling_params_to_eval_adapter(monkeypatch):
    """`sampling_params` should be forwarded to the eval adapter when configured."""
    elm = object.__new__(eLargeModel)
    sampling_params = SamplingParams(max_tokens=41, temperature=0.3, top_k=9)
    elm._config = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {},
        "eval": {"sampling_params": sampling_params},
    }
    elm._tokenizer = object()
    elm.build_tokenizer = lambda: None
    elm.build_esurge = lambda: object()

    calls: dict[str, object] = {}

    class _DummyEvalAdapter:
        def __init__(self, **kwargs):
            calls["adapter_kwargs"] = kwargs

        def stop(self):
            calls["stopped"] = True

    def _simple_evaluate(**kwargs):
        calls["simple_evaluate_kwargs"] = kwargs
        return {"results": {}}

    lm_eval_module = ModuleType("lm_eval")
    lm_eval_module.evaluator = ModuleType("lm_eval.evaluator")
    lm_eval_module.evaluator.simple_evaluate = _simple_evaluate

    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval_module)
    monkeypatch.setattr("easydel.inference.evaluations.eSurgeLMEvalAdapter", _DummyEvalAdapter)

    elm.eval(["gsm8k"])

    assert calls["adapter_kwargs"]["sampling_params"] is sampling_params


def test_eval_can_enable_thinking_for_chat_templating(monkeypatch):
    """`enable_thinking` should be forwarded to the eval adapter."""
    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {},
        "eval": {"enable_thinking": True},
    }
    elm._tokenizer = object()
    elm.build_tokenizer = lambda: None
    elm.build_esurge = lambda: object()

    calls: dict[str, object] = {}

    class _DummyEvalAdapter:
        def __init__(self, **kwargs):
            calls["adapter_kwargs"] = kwargs

        def stop(self):
            calls["stopped"] = True

    def _simple_evaluate(**kwargs):
        return {"results": {}}

    lm_eval_module = ModuleType("lm_eval")
    lm_eval_module.evaluator = ModuleType("lm_eval.evaluator")
    lm_eval_module.evaluator.simple_evaluate = _simple_evaluate

    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval_module)
    monkeypatch.setattr("easydel.inference.evaluations.eSurgeLMEvalAdapter", _DummyEvalAdapter)

    elm.eval(["humaneval"])

    assert calls["adapter_kwargs"]["enable_thinking"] is True


def test_eval_can_ignore_benchmark_eos_flags(monkeypatch):
    """`ignore_benchmark_eos_flags` should be forwarded to the eval adapter."""
    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {},
        "eval": {"ignore_benchmark_eos_flags": True},
    }
    elm._tokenizer = object()
    elm.build_tokenizer = lambda: None
    elm.build_esurge = lambda: object()

    calls: dict[str, object] = {}

    class _DummyEvalAdapter:
        def __init__(self, **kwargs):
            calls["adapter_kwargs"] = kwargs

        def stop(self):
            calls["stopped"] = True

    def _simple_evaluate(**kwargs):
        return {"results": {}}

    lm_eval_module = ModuleType("lm_eval")
    lm_eval_module.evaluator = ModuleType("lm_eval.evaluator")
    lm_eval_module.evaluator.simple_evaluate = _simple_evaluate

    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval_module)
    monkeypatch.setattr("easydel.inference.evaluations.eSurgeLMEvalAdapter", _DummyEvalAdapter)

    elm.eval(["humaneval"])

    assert calls["adapter_kwargs"]["ignore_benchmark_eos_flags"] is True


def test_eval_can_forward_chat_template_args_and_think_tokens(monkeypatch):
    """Advanced eval adapter kwargs should be forwarded through the shared runner."""
    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {},
        "eval": {
            "chat_template_args": {"foo": "bar"},
            "think_start_token": "<think>",
            "think_end_token": "</think>",
        },
    }
    elm._tokenizer = object()
    elm.build_tokenizer = lambda: None
    elm.build_esurge = lambda: object()

    calls: dict[str, object] = {}

    class _DummyEvalAdapter:
        def __init__(self, **kwargs):
            calls["adapter_kwargs"] = kwargs

        def stop(self):
            calls["stopped"] = True

    def _simple_evaluate(**kwargs):
        return {"results": {}}

    lm_eval_module = ModuleType("lm_eval")
    lm_eval_module.evaluator = ModuleType("lm_eval.evaluator")
    lm_eval_module.evaluator.simple_evaluate = _simple_evaluate

    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval_module)
    monkeypatch.setattr("easydel.inference.evaluations.eSurgeLMEvalAdapter", _DummyEvalAdapter)

    elm.eval(["humaneval"])

    assert calls["adapter_kwargs"]["chat_template_args"] == {"foo": "bar"}
    assert calls["adapter_kwargs"]["think_start_token"] == "<think>"
    assert calls["adapter_kwargs"]["think_end_token"] == "</think>"


def test_eval_chat_template_prefers_instruct_task_variant_when_available(monkeypatch):
    """Chat-mode code tasks should upgrade to *_instruct and skip extra chat wrapping when prefixed."""
    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {},
        "eval": {"apply_chat_template": True},
    }
    elm._tokenizer = object()
    elm.build_tokenizer = lambda: None
    elm.build_esurge = lambda: object()

    calls: dict[str, object] = {}

    class _DummyEvalAdapter:
        def __init__(self, **kwargs):
            calls["adapter_kwargs"] = kwargs

        def stop(self):
            calls["stopped"] = True

    def _simple_evaluate(**kwargs):
        calls["simple_evaluate_kwargs"] = kwargs
        return {"results": {}}

    class _DummyTaskManager:
        def __init__(self, **kwargs):
            calls["task_manager_kwargs"] = kwargs
            self.all_tasks = ["humaneval", "humaneval_instruct"]

        def _get_config(self, name):
            if name == "humaneval_instruct":
                return {"gen_prefix": "Here is the completed function:\n```python\n"}
            return {}

    lm_eval_module = ModuleType("lm_eval")
    lm_eval_module.evaluator = ModuleType("lm_eval.evaluator")
    lm_eval_module.evaluator.simple_evaluate = _simple_evaluate

    lm_eval_tasks_module = ModuleType("lm_eval.tasks")
    lm_eval_tasks_module.TaskManager = _DummyTaskManager

    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval_module)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", lm_eval_tasks_module)
    monkeypatch.setattr("easydel.inference.evaluations.eSurgeLMEvalAdapter", _DummyEvalAdapter)

    elm.eval(["humaneval"])

    assert calls["simple_evaluate_kwargs"]["tasks"] == ["humaneval_instruct"]
    assert calls["simple_evaluate_kwargs"]["apply_chat_template"] is False


def test_eval_applies_code_eval_scorer_overrides(monkeypatch):
    """Code-eval worker/timeout overrides should wrap simple_evaluate."""
    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {},
        "eval": {"code_eval_num_workers": 12, "code_eval_timeout": 1.5},
    }
    elm._tokenizer = object()
    elm.build_tokenizer = lambda: None
    elm.build_esurge = lambda: object()

    calls: dict[str, object] = {}

    class _DummyEvalAdapter:
        def __init__(self, **kwargs):
            calls["adapter_kwargs"] = kwargs

        def stop(self):
            calls["stopped"] = True

    def _simple_evaluate(**kwargs):
        calls["simple_evaluate_kwargs"] = kwargs
        return {"results": {}}

    class _DummyContext:
        def __enter__(self):
            calls["context_entered"] = True
            return None

        def __exit__(self, exc_type, exc, tb):
            calls["context_exited"] = True
            return False

    def _override_lm_eval_code_exec(*, num_workers=None, timeout=None):
        calls["override_kwargs"] = {
            "num_workers": num_workers,
            "timeout": timeout,
        }
        return _DummyContext()

    lm_eval_module = ModuleType("lm_eval")
    lm_eval_module.evaluator = ModuleType("lm_eval.evaluator")
    lm_eval_module.evaluator.simple_evaluate = _simple_evaluate

    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval_module)
    monkeypatch.setattr("easydel.inference.evaluations.eSurgeLMEvalAdapter", _DummyEvalAdapter)
    monkeypatch.setattr("easydel.infra.elarge.benchmarking.override_lm_eval_code_exec", _override_lm_eval_code_exec)

    elm.eval(["humaneval"])

    assert calls["override_kwargs"] == {"num_workers": 12, "timeout": 1.5}
    assert calls["context_entered"] is True
    assert calls["context_exited"] is True


def test_eval_defaults_code_eval_workers_to_capped_auto_value(monkeypatch):
    """Code tasks should cap auto-selected scorer workers on large hosts."""
    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {},
        "eval": {},
    }
    elm._tokenizer = object()
    elm.build_tokenizer = lambda: None
    elm.build_esurge = lambda: object()

    calls: dict[str, object] = {}

    class _DummyEvalAdapter:
        def __init__(self, **kwargs):
            calls["adapter_kwargs"] = kwargs

        def stop(self):
            calls["stopped"] = True

    def _simple_evaluate(**kwargs):
        return {"results": {}}

    class _DummyContext:
        def __enter__(self):
            calls["context_entered"] = True
            return None

        def __exit__(self, exc_type, exc, tb):
            calls["context_exited"] = True
            return False

    def _override_lm_eval_code_exec(*, num_workers=None, timeout=None):
        calls["override_kwargs"] = {
            "num_workers": num_workers,
            "timeout": timeout,
        }
        return _DummyContext()

    lm_eval_module = ModuleType("lm_eval")
    lm_eval_module.evaluator = ModuleType("lm_eval.evaluator")
    lm_eval_module.evaluator.simple_evaluate = _simple_evaluate

    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval_module)
    monkeypatch.setattr("easydel.inference.evaluations.eSurgeLMEvalAdapter", _DummyEvalAdapter)
    monkeypatch.setattr("easydel.infra.elarge.benchmarking.override_lm_eval_code_exec", _override_lm_eval_code_exec)
    monkeypatch.setattr("easydel.infra.elarge.benchmarking.os.cpu_count", lambda: 24)

    elm.eval(["humaneval"])

    assert calls["override_kwargs"] == {"num_workers": 16, "timeout": None}
    assert calls["context_entered"] is True
    assert calls["context_exited"] is True


def test_eval_routes_benchmark_configs_to_run_benchmarks(monkeypatch):
    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {},
        "eval": {"temperature": 0.2},
    }

    calls: dict[str, object] = {}

    def _run_benchmarks(*, benchmarks, output_path=None, **default_eval_overrides):
        calls["benchmarks"] = benchmarks
        calls["output_path"] = output_path
        calls["default_eval_overrides"] = default_eval_overrides
        return {"benchmarks": {"code": {"results": {}}}}

    elm.run_benchmarks = _run_benchmarks

    results = elm.eval(
        [
            {
                "name": "code",
                "tasks": ["humaneval"],
                "enable_thinking": True,
            }
        ],
        num_fewshot=3,
        output_path="bench.json",
        top_p=0.8,
    )

    assert results == {"benchmarks": {"code": {"results": {}}}}
    assert calls["output_path"] == "bench.json"
    assert calls["default_eval_overrides"]["num_fewshot"] == 3
    assert calls["default_eval_overrides"]["top_p"] == 0.8
    assert calls["benchmarks"][0]["name"] == "code"


def test_eval_rejects_task_alias_benchmark_configs():
    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {},
        "eval": {"temperature": 0.2},
    }

    with pytest.raises(TypeError, match="must use `tasks`"):
        elm.eval(
            [
                {
                    "name": "math",
                    "task": "gsm8k",
                    "num_fewshot": 5,
                }
            ],
            output_path="bench.json",
            top_p=0.8,
        )


def test_eval_keeps_structured_task_dicts_on_flat_eval_path(monkeypatch):
    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {},
        "eval": {},
    }
    elm._tokenizer = object()
    elm.build_tokenizer = lambda: None
    elm.build_esurge = lambda: object()

    calls: dict[str, object] = {}

    class _DummyEvalAdapter:
        def __init__(self, **kwargs):
            calls["adapter_kwargs"] = kwargs

        def stop(self):
            calls["stopped"] = True

    def _simple_evaluate(**kwargs):
        calls["simple_evaluate_kwargs"] = kwargs
        return {"results": {"mmlu": {"acc,none": 0.5}}}

    lm_eval_module = ModuleType("lm_eval")
    lm_eval_module.evaluator = ModuleType("lm_eval.evaluator")
    lm_eval_module.evaluator.simple_evaluate = _simple_evaluate

    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval_module)
    monkeypatch.setattr("easydel.inference.evaluations.eSurgeLMEvalAdapter", _DummyEvalAdapter)

    def _run_benchmarks(**_):
        raise AssertionError("structured task specs should stay on eval")

    elm.run_benchmarks = _run_benchmarks

    task_spec = {"task": "mmlu", "alias": "custom"}
    results = elm.eval([task_spec])

    assert results["results"]["mmlu"]["acc,none"] == 0.5
    assert calls["simple_evaluate_kwargs"]["tasks"] == [task_spec]
    assert calls["stopped"] is True


def test_run_benchmarks_reuses_engine_and_merges_default_eval_overrides(monkeypatch):
    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {"max_model_len": 8192, "max_num_seqs": 16},
        "eval": {"temperature": 0.2, "apply_chat_template": False},
    }
    elm._tokenizer = object()
    elm.build_tokenizer = lambda: None

    class _Engine:
        def __init__(self):
            self.terminate_calls = 0

        def terminate(self):
            self.terminate_calls += 1

    engine = _Engine()
    elm.build_esurge = lambda: engine

    run_calls: list[dict[str, object]] = []
    sampling_params = SamplingParams(max_tokens=24, temperature=0.5, top_k=7)

    def _fake_run_lm_eval_with_esurge(**kwargs):
        run_calls.append(kwargs)
        return {"results": {"humaneval": {"pass@1,create_test": 0.25}}}

    monkeypatch.setattr("easydel.infra.elarge.model.run_lm_eval_with_esurge", _fake_run_lm_eval_with_esurge)

    results = elm.run_benchmarks(
        [
            {
                "name": "code",
                "tasks": "humaneval",
                "enable_thinking": True,
                "sampling_params": sampling_params,
            }
        ],
        top_p=0.9,
    )

    assert results["benchmarks"]["code"]["results"]["humaneval"]["pass@1,create_test"] == 0.25
    assert len(run_calls) == 1
    assert run_calls[0]["surge"] is engine
    assert run_calls[0]["tasks"] == ["humaneval"]
    assert run_calls[0]["eval_config"]["temperature"] == 0.2
    assert run_calls[0]["eval_config"]["top_p"] == 0.9
    assert run_calls[0]["eval_config"]["enable_thinking"] is True
    assert run_calls[0]["eval_config"]["sampling_params"] is sampling_params
    assert run_calls[0]["stop_engine"] is False
    assert engine.terminate_calls == 1


def test_eval_benchmark_route_preserves_configured_num_fewshot_default(monkeypatch):
    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {"max_model_len": 8192, "max_num_seqs": 16},
        "eval": {"num_fewshot": 5, "temperature": 0.2},
    }
    elm._tokenizer = object()
    elm.build_tokenizer = lambda: None

    class _Engine:
        def __init__(self):
            self.terminate_calls = 0

        def terminate(self):
            self.terminate_calls += 1

    engine = _Engine()
    elm.build_esurge = lambda: engine

    run_calls: list[dict[str, object]] = []

    def _fake_run_lm_eval_with_esurge(**kwargs):
        run_calls.append(kwargs)
        return {"results": {}}

    monkeypatch.setattr("easydel.infra.elarge.model.run_lm_eval_with_esurge", _fake_run_lm_eval_with_esurge)

    elm.eval(
        [
            {
                "name": "code",
                "tasks": ["humaneval"],
            }
        ]
    )

    assert len(run_calls) == 1
    assert run_calls[0]["eval_config"]["num_fewshot"] == 5
    assert run_calls[0]["eval_config"]["temperature"] == 0.2
    assert engine.terminate_calls == 1


def test_eval_skips_code_eval_override_for_non_code_tasks_by_default(monkeypatch):
    """Non-code tasks should not enter the code-eval override context implicitly."""
    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "dummy-model"},
        "esurge": {},
        "eval": {},
    }
    elm._tokenizer = object()
    elm.build_tokenizer = lambda: None
    elm.build_esurge = lambda: object()

    calls: dict[str, object] = {}

    class _DummyEvalAdapter:
        def __init__(self, **kwargs):
            calls["adapter_kwargs"] = kwargs

        def stop(self):
            calls["stopped"] = True

    def _simple_evaluate(**kwargs):
        return {"results": {}}

    def _override_lm_eval_code_exec(*, num_workers=None, timeout=None):
        calls["override_kwargs"] = {
            "num_workers": num_workers,
            "timeout": timeout,
        }
        raise AssertionError("code_eval override should not be used for non-code tasks by default")

    lm_eval_module = ModuleType("lm_eval")
    lm_eval_module.evaluator = ModuleType("lm_eval.evaluator")
    lm_eval_module.evaluator.simple_evaluate = _simple_evaluate

    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval_module)
    monkeypatch.setattr("easydel.inference.evaluations.eSurgeLMEvalAdapter", _DummyEvalAdapter)
    monkeypatch.setattr("easydel.infra.elarge.benchmarking.override_lm_eval_code_exec", _override_lm_eval_code_exec)

    elm.eval(["gsm8k"])

    assert "override_kwargs" not in calls
