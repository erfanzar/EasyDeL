from __future__ import annotations

import base64
import pickle
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from easydel.infra.elarge.processing import make_serializable, override_lm_eval_code_exec


def _sample_callback() -> str:
    """Return a stable callable payload for serialization tests."""
    return "ok"


def test_make_serializable_handles_callable_and_array_like_values() -> None:
    """Serialize callables and array-likes into JSON-safe structures."""
    payload = {
        "callable": _sample_callback,
        "matrix": np.array([[1, 2], [3, 4]], dtype=np.int32),
    }

    assert make_serializable(payload) == {
        "callable": f"{__name__}._sample_callback",
        "matrix": [[1, 2], [3, 4]],
    }


def test_override_lm_eval_code_exec_runs_code_eval_in_subprocess(monkeypatch) -> None:
    """Override code-exec settings and run the scorer in an isolated subprocess."""

    class _DummyMetric:
        __module__ = "fake_code_eval.code_eval"

        def compute(self, *args, **kwargs):
            return {"ok": True}

    metric = _DummyMetric()

    humaneval_utils = ModuleType("lm_eval.tasks.humaneval.utils")
    humaneval_utils.compute_ = metric
    mbpp_utils = ModuleType("lm_eval.tasks.mbpp.utils")
    mbpp_utils.pass_at_k = metric

    monkeypatch.setitem(sys.modules, "lm_eval.tasks.humaneval.utils", humaneval_utils)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks.mbpp.utils", mbpp_utils)

    captured_payload = {}

    def _fake_run(cmd, *, input, capture_output, timeout, env, check):  # noqa
        del cmd, capture_output, timeout, env, check
        payload = pickle.loads(base64.b64decode(input))
        captured_payload.update(payload)
        encoded = base64.b64encode(pickle.dumps(({"pass@1": 1.0}, {0: []})))
        return SimpleNamespace(returncode=0, stdout=encoded, stderr=b"")

    monkeypatch.setattr("easydel.infra.elarge.processing.subprocess.run", _fake_run)

    with override_lm_eval_code_exec(num_workers=7, timeout=1.5):
        result = humaneval_utils.compute_.compute(
            references=["assert add(2, 3) == 5"],
            predictions=[["def add(a, b): return a + b"]],
            k=[1],
        )
        assert result == ({"pass@1": 1.0}, {0: []})
        assert captured_payload["references"] == ["assert add(2, 3) == 5"]
        assert captured_payload["predictions"] == [["def add(a, b): return a + b"]]
        assert captured_payload["k"] == [1]
        assert captured_payload["num_workers"] == 7
        assert captured_payload["timeout"] == 1.5

    assert humaneval_utils.compute_ is metric
    assert mbpp_utils.pass_at_k is metric
