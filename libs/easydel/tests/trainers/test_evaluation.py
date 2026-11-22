"""Evaluation must propagate failures instead of silently returning an empty result."""

from types import SimpleNamespace

import pytest
from easydel.trainers.trainer import trainer as trainer_module
from easydel.trainers.trainer.trainer import Trainer

pytestmark = pytest.mark.cpu_ok


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.setattr(trainer_module, "MetricsTracker", lambda: object())
    monkeypatch.setattr(trainer_module, "StepMetrics", lambda arguments: object())

    class Evaluator:
        eval = Trainer.eval
        arguments = SimpleNamespace(do_eval=True)
        finished = False
        error = None

        def start_evaluation_hook(self):
            pass

        def _run_evaluation(self, **kwargs):
            yield {"eval/loss": 1.0}
            if self.error is not None:
                raise self.error

        def finish(self):
            self.finished = True

    return Evaluator()


def test_evaluation_yields_metrics_and_exhausts_normally(evaluator):
    assert list(evaluator.eval(None)) == [{"eval/loss": 1.0}]


def test_runtime_failure_after_a_metric_is_propagated(evaluator):
    failure = RuntimeError("device execution failed")
    evaluator.error = failure
    evaluation = evaluator.eval(None)
    assert next(evaluation) == {"eval/loss": 1.0}
    with pytest.raises(RuntimeError) as error:
        next(evaluation)
    assert error.value is failure


def test_final_evaluation_failure_propagates_and_cleans_up(evaluator):
    failure = RuntimeError("evaluation compilation failed")
    evaluator.error = failure
    with pytest.raises(RuntimeError) as error:
        Trainer._finalize_training(evaluator, SimpleNamespace(state=None), None)
    assert error.value is failure
    assert evaluator.finished


def test_successful_finalization_returns_original_output(evaluator):
    output = SimpleNamespace(state=None)
    assert Trainer._finalize_training(evaluator, output, None) is output
    assert evaluator.finished
