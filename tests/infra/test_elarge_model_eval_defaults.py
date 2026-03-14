from __future__ import annotations

import sys
from types import ModuleType

from easydel.infra.elarge_model.elarge_model import eLargeModel


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
    monkeypatch.setattr(
        "easydel.infra.elarge_model.elarge_model.override_lm_eval_code_exec", _override_lm_eval_code_exec
    )

    elm.eval(["humaneval"])

    assert calls["override_kwargs"] == {"num_workers": 12, "timeout": 1.5}
    assert calls["context_entered"] is True
    assert calls["context_exited"] is True


def test_eval_defaults_code_eval_workers_to_cpu_count(monkeypatch):
    """Code tasks should default scorer workers to the local CPU count."""
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
    monkeypatch.setattr(
        "easydel.infra.elarge_model.elarge_model.override_lm_eval_code_exec", _override_lm_eval_code_exec
    )
    monkeypatch.setattr("easydel.infra.elarge_model.elarge_model.os.cpu_count", lambda: 24)

    elm.eval(["humaneval"])

    assert calls["override_kwargs"] == {"num_workers": 24, "timeout": None}
    assert calls["context_entered"] is True
    assert calls["context_exited"] is True


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
    monkeypatch.setattr(
        "easydel.infra.elarge_model.elarge_model.override_lm_eval_code_exec", _override_lm_eval_code_exec
    )

    elm.eval(["gsm8k"])

    assert "override_kwargs" not in calls
