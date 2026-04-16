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

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from easydel.infra.errors import EasyDeLPreemptionSignal
from easydel.trainers.trainer.trainer import Trainer


class _MeshCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _ProgressBarStub:
    def __init__(self):
        self.updates: list[int] = []
        self.closed = False

    def update(self, value: int):
        self.updates.append(value)

    def close(self):
        self.closed = True


class _StateStub:
    def __init__(self, step: int):
        self.step = step


class _MetricsTrackerStub:
    def update(self, *, loss, accuracy, step):
        return float(loss), float(accuracy)


class _StepMetricsStub:
    def start_step(self):
        return None

    def calculate(self, **kwargs):
        return {}


class _EvalStepMetricsStub:
    def start_step(self):
        return None

    def calculate(self, **kwargs):
        return {
            "eval/eval_step": kwargs["current_step"],
            "eval/mean_loss": kwargs["mean_loss"],
            "eval/mean_accuracy": kwargs["mean_accuracy"],
            "eval/attention_loss": jnp.asarray(kwargs["current_step"] * 10.0),
            "eval-mlperf/execution_time": jnp.asarray(kwargs["current_step"] * 0.5),
            "eval-mlperf/total_tokens": jnp.asarray(kwargs["current_step"] * 8.0),
            "eval-mlperf/total_flops": jnp.asarray(kwargs["current_step"] * 16.0),
        }


class _EvalMetricsTrackerStub:
    def __init__(self):
        self.calls: list[int] = []
        self.loss_sum = 0.0
        self.accuracy_sum = 0.0

    def update(self, loss, accuracy, step):
        self.calls.append(step)
        self.loss_sum += float(loss)
        self.accuracy_sum += float(accuracy)
        count = len(self.calls)
        return self.loss_sum / count, self.accuracy_sum / count


def _make_trainer(*, max_training_steps: int = 11, num_train_epochs: int = 2):
    trainer = object.__new__(Trainer)
    trainer.arguments = SimpleNamespace(
        log_all_workers=False,
        num_train_epochs=num_train_epochs,
        ids_to_pop_from_dataset=None,
        total_batch_size=1,
        gradient_accumulation_steps=1,
    )
    trainer.max_training_steps = max_training_steps
    trainer.dataloader_train = list(range(64))
    trainer._model = SimpleNamespace(mesh=_MeshCtx())
    progress_bar = _ProgressBarStub()
    trainer.create_progress_bar = lambda total, disabled=False, desc="": progress_bar
    trainer._prepare_training_output = lambda state, run_exception: state
    return trainer, progress_bar


def test_epoch_step_bounds_cover_remainder_steps():
    trainer, _ = _make_trainer(max_training_steps=11, num_train_epochs=2)

    assert trainer._get_epoch_step_bounds(0) == (0, 5)
    assert trainer._get_epoch_step_bounds(1) == (5, 11)


def test_run_training_loop_fast_forwards_iterator_when_resuming_mid_epoch(monkeypatch):
    monkeypatch.setattr("easydel.trainers.trainer.trainer.jax.process_index", lambda: 0)
    trainer, progress_bar = _make_trainer(max_training_steps=11, num_train_epochs=2)
    calls: list[dict[str, int]] = []

    def _train_epoch(
        *,
        state,
        train_dataset,
        train_iter,
        metrics_tracker,
        step_metrics,
        pbar,
        epoch,
        epoch_start_step=None,
        epoch_end_step=None,
    ):
        first_batch = next(train_iter)
        calls.append(
            {
                "epoch": epoch,
                "epoch_start_step": epoch_start_step,
                "epoch_end_step": epoch_end_step,
                "first_batch": first_batch,
            }
        )
        state.step = epoch_end_step
        return state, None, train_iter

    trainer._train_epoch = _train_epoch

    state = _StateStub(step=5)
    output, run_exception = Trainer._run_training_loop(
        trainer,
        state=state,
        metrics_tracker=None,
        step_metrics=None,
    )

    assert output.step == 11
    assert run_exception is None
    assert progress_bar.updates == [5]
    assert calls == [
        {
            "epoch": 1,
            "epoch_start_step": 5,
            "epoch_end_step": 11,
            "first_batch": 5,
        }
    ]
    assert progress_bar.closed is True


def test_train_epoch_uses_preemption_checkpoint_path_before_regular_checkpointing(monkeypatch):
    monkeypatch.setattr("easydel.trainers.trainer.trainer.jax.process_index", lambda: 0)
    trainer, _ = _make_trainer(max_training_steps=4, num_train_epochs=1)
    trainer.arguments.learning_rate = 1e-3
    trainer.arguments.max_length = 16
    trainer.data_collator = None
    trainer.scheduler = None
    trainer._backward_flops_per_token = 0.0
    trainer._extra_backward_flops_per_token = 0.0
    trainer.train_tracker = SimpleNamespace(trace_compilation=lambda: _MeshCtx())
    trainer.on_step_start = lambda state, step: state
    trainer.on_step_end = lambda state, metrics, step: (state, metrics)
    trainer.apply_training_hooks = lambda metrics: metrics
    logged_steps: list[int] = []
    trainer.log_metrics = lambda **kwargs: logged_steps.append(kwargs["step"])
    trainer.log_weight_distribution = lambda **kwargs: None
    trainer.log_watchers = lambda **kwargs: None
    trainer.maybe_generate = lambda **kwargs: None
    trainer.maybe_benchmark = lambda **kwargs: None
    trainer._should_save_tpu_preemption_checkpoint = lambda step: True
    trainer._should_run_evaluation = lambda current_step: False

    saved_steps: list[int] = []
    trainer._save_tpu_preemption_checkpoint = lambda state, step: saved_steps.append(step) or f"run-{step}"
    trainer._save_checkpoint_for_step = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("regular checkpoint save should not run")
    )

    def _execute_train_step(state, batch):
        del batch
        return (
            _StateStub(step=state.step + 1),
            SimpleNamespace(loss=0.0, accuracy=1.0, execution_time=0.0),
            None,
        )

    trainer._execute_train_step = _execute_train_step

    state, run_exception, _ = Trainer._train_epoch(
        trainer,
        state=_StateStub(step=0),
        train_dataset=[{"tokens": 1}],
        train_iter=iter([{"tokens": 1}]),
        metrics_tracker=_MetricsTrackerStub(),
        step_metrics=_StepMetricsStub(),
        pbar=None,
        epoch=0,
        epoch_start_step=0,
        epoch_end_step=4,
    )

    assert state.step == 1
    assert isinstance(run_exception, EasyDeLPreemptionSignal)
    assert saved_steps == [1]
    assert logged_steps == [1]


def test_eval_epoch_logs_batch_progress_locally_and_reports_summary_at_global_step():
    trainer, _ = _make_trainer(max_training_steps=8, num_train_epochs=1)
    trainer.arguments.max_length = 16
    trainer.arguments.eval_batch_size = 1
    trainer.max_evaluation_steps = 3
    trainer.data_collator = None
    trainer.evalu_tracker = SimpleNamespace(trace_compilation=lambda: _MeshCtx())
    trainer._forward_flops_per_token = 0.0
    trainer._extra_forward_flops_per_token = 0.0
    trainer._get_next_batch = lambda eval_iter, _: (next(eval_iter), eval_iter)
    eval_step_metrics = iter(
        [
            SimpleNamespace(loss=1.0, accuracy=0.1, execution_time=0.0),
            SimpleNamespace(loss=3.0, accuracy=0.2, execution_time=0.0),
            SimpleNamespace(loss=5.0, accuracy=0.3, execution_time=0.0),
        ]
    )
    trainer._execute_eval_step = lambda state, batch: next(eval_step_metrics)

    logged_calls: list[dict[str, object]] = []
    trainer.log_metrics = lambda **kwargs: logged_calls.append(kwargs)

    metrics = list(
        Trainer._eval_epoch(
            trainer,
            state=_StateStub(step=50),
            eval_dataset=[{"tokens": 1}] * 3,
            eval_iter=iter([{"tokens": 1}] * 3),
            metrics_tracker=_EvalMetricsTrackerStub(),
            step_metrics=_EvalStepMetricsStub(),
            pbar=None,
        )
    )

    assert len(metrics) == 3
    assert [call["step"] for call in logged_calls[:-1]] == [1, 2, 3]
    assert all(call["log_to_backends"] is False for call in logged_calls[:-1])
    assert logged_calls[-1]["step"] == 50
    assert logged_calls[-1]["update_progress"] is False
    assert logged_calls[-1]["force_report"] is True
    assert logged_calls[-1]["metrics"]["eval/loss"] == 3.0
    assert logged_calls[-1]["metrics"]["eval/accuracy"] == pytest.approx(0.2)
    assert logged_calls[-1]["metrics"]["eval/attention_loss"] == pytest.approx(20.0)
    assert logged_calls[-1]["metrics"]["eval-mlperf/total_tokens"] == pytest.approx(48.0)
    assert logged_calls[-1]["metrics"]["eval-mlperf/total_flops"] == pytest.approx(96.0)
    assert logged_calls[-1]["metrics"]["eval-mlperf/throughput"] == pytest.approx(16.0)
    assert logged_calls[-1]["metrics"]["eval-mlperf/tflops"] == pytest.approx(3.2e-11)


def test_eval_epoch_summary_matches_step_perplexity_behavior_for_large_losses():
    trainer, _ = _make_trainer(max_training_steps=8, num_train_epochs=1)
    trainer.arguments.max_length = 16
    trainer.arguments.eval_batch_size = 1
    trainer.max_evaluation_steps = 2
    trainer.data_collator = None
    trainer.evalu_tracker = SimpleNamespace(trace_compilation=lambda: _MeshCtx())
    trainer._forward_flops_per_token = 0.0
    trainer._extra_forward_flops_per_token = 0.0
    trainer._get_next_batch = lambda eval_iter, _: (next(eval_iter), eval_iter)
    eval_step_metrics = iter(
        [
            SimpleNamespace(loss=100.0, accuracy=0.1, execution_time=0.0),
            SimpleNamespace(loss=100.0, accuracy=0.2, execution_time=0.0),
        ]
    )
    trainer._execute_eval_step = lambda state, batch: next(eval_step_metrics)

    logged_calls: list[dict[str, object]] = []
    trainer.log_metrics = lambda **kwargs: logged_calls.append(kwargs)

    metrics = list(
        Trainer._eval_epoch(
            trainer,
            state=_StateStub(step=50),
            eval_dataset=[{"tokens": 1}] * 2,
            eval_iter=iter([{"tokens": 1}] * 2),
            metrics_tracker=_EvalMetricsTrackerStub(),
            step_metrics=_EvalStepMetricsStub(),
            pbar=None,
        )
    )

    assert len(metrics) == 2
    assert logged_calls[-1]["metrics"]["eval/loss"] == 100.0
    assert logged_calls[-1]["metrics"]["eval/perplexity"] == float(jnp.exp(100.0))
