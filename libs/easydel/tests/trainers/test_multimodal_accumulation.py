"""Named batch-axis handling through public accumulators and auxiliary forwards."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.infra.loss_utils import LossMetrics
from easydel.trainers import training_utils

pytestmark = pytest.mark.cpu_ok


def _batch(rows):
    ids = np.arange(rows * 8).reshape(rows, 8)
    return {
        "input_ids": ids,
        "labels": ids.copy(),
        "attention_mask": np.ones_like(ids),
        "position_ids": np.stack([ids + offset for offset in (0, 100, 200)]),
        "global_count": np.asarray(99),
    }


def _assert_aligned(batch):
    ids = batch["input_ids"]
    assert batch["position_ids"].shape == (3, ids.shape[0], ids.shape[1])
    for component, offset in enumerate((0, 100, 200)):
        np.testing.assert_array_equal(batch["position_ids"][component], ids + offset)
    assert batch["global_count"].shape == ()


@pytest.fixture
def host_runtime(monkeypatch):
    def scan(fn, carry, xs, length):
        for value in xs:
            carry, _ = fn(carry, value)
        return carry, None

    monkeypatch.setattr(training_utils, "jnp", np)
    monkeypatch.setattr(training_utils.jax, "eval_shape", lambda fn, *args: fn(*args))
    monkeypatch.setattr(training_utils.lax, "scan", scan)
    monkeypatch.setattr(
        training_utils.lax,
        "dynamic_slice_in_dim",
        lambda x, start, size, axis: np.take(x, range(start, start + size), axis=axis),
    )
    monkeypatch.setattr(
        training_utils.lax, "slice_in_dim", lambda x, start, end, axis: np.take(x, range(start, end), axis=axis)
    )


@pytest.mark.parametrize("rows,accumulation", [(3, 3), (4, 2)])
def test_minibatch_call_preserves_all_rotary_components_and_row_alignment(host_runtime, rows, accumulation):
    seen = []

    def grad_fn(tree, batch):
        _assert_aligned(batch)
        seen.append(batch["input_ids"].copy())
        value = np.asarray(batch["input_ids"].mean(), dtype=np.float32)
        return (value, LossMetrics(loss=value)), value

    batch = _batch(rows)
    gradient, metrics = training_utils.minibatch_call(
        SimpleNamespace(graphstate=None), batch, rows // accumulation, grad_fn
    )
    expected = batch["input_ids"].mean()
    assert float(gradient) == pytest.approx(expected)
    assert float(metrics.loss) == pytest.approx(expected)
    np.testing.assert_array_equal(np.concatenate(seen[1:]), batch["input_ids"])


@pytest.mark.parametrize("rows,microbatches", [(3, 3), (4, 2)])
def test_auxiliary_forward_preserves_multimodal_position_alignment(host_runtime, monkeypatch, rows, microbatches):
    monkeypatch.setattr(training_utils.spx, "jit", lambda fn, **kwargs: fn)
    monkeypatch.setattr(training_utils, "_SCHEDULED_AUXILIARY_CACHE", {})
    monkeypatch.setattr(training_utils, "_SCHED_SECTION_LOG_BUDGET", [0])
    monkeypatch.setattr(training_utils, "_scheduled_aux_pipeline_executor", lambda **kwargs: None)

    def forward(batch):
        _assert_aligned(batch)
        return batch["input_ids"]

    batch = _batch(rows)
    compiled = training_utils.cached_scheduled_auxiliary(forward, object(), microbatches=microbatches, batch_argnums=0)
    np.testing.assert_array_equal(compiled(batch), batch["input_ids"])
