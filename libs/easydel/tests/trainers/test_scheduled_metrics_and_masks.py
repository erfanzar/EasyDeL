"""Host-only scheduled-metric propagation and completion normalization regressions."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.trainers import training_utils
from easydel.trainers.binary_classifier_optimization_trainer._fn import RunningMoments
from easydel.trainers.binary_classifier_optimization_trainer.bco_trainer import BCOTrainer

pytestmark = pytest.mark.cpu_ok


@dataclasses.dataclass
class HostState:
    """Opaque Python state for exercising orchestration without JAX numerics."""

    tx: object
    step: int = 0
    graphstate: str = "parameters"
    opt_state: str = "optimizer"

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


def test_scheduled_auxiliary_metrics_reach_bco_threshold(monkeypatch):
    auxiliary = {
        "z_loss": 0.125,
        "accuracy": 0.75,
        "weight_sum": 4.0,
        "reward_moments/sum": 16.0,
        "reward_moments/sumsq": 84.0,
        "reward_moments/count": 4.0,
        "custom/diagnostic": 9.0,
    }

    # Stand in only for accelerator computation/shape setup. The scheduled step,
    # metrics assembly, optimizer state transition, and BCO host hook remain real.
    monkeypatch.setattr(
        training_utils._ScheduledValueAndGradCompiler,
        "get",
        lambda self, call: lambda tree, batch: (0.5, "gradients", auxiliary),
    )
    monkeypatch.setattr(training_utils, "make_assertions_and_get_sizes", lambda **kwargs: (4, 4, None))
    monkeypatch.setattr(training_utils.jax, "process_count", lambda: 1)
    monkeypatch.setattr(training_utils, "_SCHED_SECTION_LOG_BUDGET", [0])

    def apply_stage_local(**kwargs):
        return "updated parameters", "updated optimizer"

    state = HostState(tx=SimpleNamespace(apply_gradients_stage_local=apply_stage_local))

    def step_fn(state, batch, learning_rate_fn=None, loss_config=None):
        raise AssertionError("The ordinary step is not the scheduled implementation")

    adapter = training_utils.ScheduledLossAdapter(
        name="host_test", make_loss=lambda call: None, make_cache_key=lambda call: (), has_aux=True
    )
    scheduled_step = training_utils._compile_scheduled_training_step(
        step_fn=step_fn,
        mesh=object(),
        schedule=object(),
        batch_argnums=1,
        static_argnums=(),
        adapter=adapter,
    )
    new_state, metrics = scheduled_step(state, {"input_ids": "opaque batch"}, learning_rate_fn=lambda step: 0.01)
    assert new_state.step == 1
    assert new_state.graphstate == "updated parameters"
    assert new_state.opt_state == "updated optimizer"
    assert metrics.loss == 0.5
    assert metrics.z_loss == auxiliary["z_loss"]
    assert metrics.accuracy == auxiliary["accuracy"]
    assert metrics.weight_sum == auxiliary["weight_sum"]
    assert metrics.learning_rate == 0.01
    assert metrics.other_metrics == {key: value for key, value in auxiliary.items() if "/" in key}

    trainer = object.__new__(BCOTrainer)
    trainer.arguments = SimpleNamespace(gradient_accumulation_steps=1)
    trainer.running = RunningMoments()
    trainer.on_step_end(new_state, metrics, 1)
    assert trainer.running.as_dict() == pytest.approx({"mean": 4.0, "var": 5.0, "count": 4.0})
    assert auxiliary["reward_moments/count"] == 4.0


@pytest.mark.parametrize("aux", [None, {}, {"z_loss": 0.5}])
def test_stage_local_update_supports_absent_custom_metrics(aux):
    state = HostState(tx=SimpleNamespace(apply_gradients_stage_local=lambda **kwargs: ("new", "new opt")))
    new_state, metrics = training_utils._apply_stage_local_gradients(
        state=state, gradients=None, loss=1.0, loss_config=None, learning_rate_fn=None, aux=aux
    )
    assert new_state.step == 1
    assert metrics.other_metrics == {}
    assert metrics.z_loss == (0.5 if aux else None)


@pytest.mark.parametrize(
    ("assistant_only", "completion_only", "expected"),
    [(False, None, False), (True, None, True), (False, True, True), (True, False, False)],
)
def test_prompt_mask_option_precedence(assistant_only, completion_only, expected):
    arguments = SimpleNamespace(assistant_only_loss=assistant_only, completion_only_loss=completion_only)
    assert training_utils.resolve_mask_prompt(arguments) is expected
    assert training_utils.resolve_mask_prompt(SimpleNamespace()) is False


@pytest.mark.parametrize("fallback", [True, False])
def test_empty_completion_mask_fallback_preserves_trainer_conventions(fallback):
    inputs = np.array([[5, 6, 7, 8]], dtype=np.int64)
    completion = np.zeros((1, 4), dtype=np.int32)
    assistant = np.array([[0, 1, 1, 1]], dtype=np.int32)
    attention = np.array([[1, 1, 1, 0]], dtype=np.int8)
    batch = {
        "input_ids": inputs,
        "completion_mask": completion,
        "assistant_masks": assistant,
        "attention_mask": attention,
    }
    result = training_utils.normalize_completion_masks_and_labels(batch, assistant_mask_fallback_on_empty=fallback)
    assert result is batch
    assert "assistant_masks" not in result
    np.testing.assert_array_equal(result["completion_mask"], [[0, 1, 1, 0]] if fallback else [[0, 0, 0, 0]])
    np.testing.assert_array_equal(result["labels"], [[-100, 6, 7, -100]] if fallback else [[-100] * 4])
    assert result["completion_mask"].dtype == attention.dtype
    assert result["labels"].dtype == np.int32
    np.testing.assert_array_equal(inputs, [[5, 6, 7, 8]])
    np.testing.assert_array_equal(completion, [[0, 0, 0, 0]])
    np.testing.assert_array_equal(assistant, [[0, 1, 1, 1]])


def test_nonempty_explicit_completion_mask_precedes_assistant_mask():
    batch = {"input_ids": [[5, 6, 7]], "completion_mask": [[0, 0, 1]], "assistant_masks": [[1, 1, 0]]}
    result = training_utils.normalize_completion_masks_and_labels(batch)
    np.testing.assert_array_equal(result["completion_mask"], [[0, 0, 1]])
    np.testing.assert_array_equal(result["labels"], [[-100, -100, 7]])


def test_missing_completion_mask_uses_assistant_mask_and_preserves_existing_labels():
    labels = np.array([[-100, 9, 10]])
    batch = {"labels": labels, "assistant_masks": [[0, 1, 1]], "attention_mask": [[1, 1, 0]]}
    result = training_utils.normalize_completion_masks_and_labels(batch)
    assert result["labels"] is labels
    np.testing.assert_array_equal(result["completion_mask"], [[0, 1, 0]])
    np.testing.assert_array_equal(labels, [[-100, 9, 10]])


@pytest.mark.parametrize("with_attention", [False, True])
def test_mask_recovered_from_ignored_labels(with_attention):
    batch = {"labels": [[-100, 9, 10]]}
    if with_attention:
        batch["attention_mask"] = [[1, 1, 0]]
    result = training_utils.normalize_completion_masks_and_labels(batch)
    np.testing.assert_array_equal(result["completion_mask"], [[0, 1, 0 if with_attention else 1]])


def test_fully_supervised_batch_without_ignored_labels_stays_unmasked():
    batch = {"input_ids": [[1, 2]], "labels": [[1, 2]], "attention_mask": [[1, 0]]}
    result = training_utils.normalize_completion_masks_and_labels(batch)
    assert "completion_mask" not in result
    assert result["labels"] == [[1, 2]]
