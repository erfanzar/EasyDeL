"""Host orchestration regressions; all numerical primitives below are NumPy stand-ins."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.infra.loss_utils import LossMetrics
from easydel.trainers import training_utils
from easydel.trainers.trainer import _fn as base_fn
from easydel.trainers.trainer._fn import training_step

pytestmark = pytest.mark.cpu_ok


@dataclasses.dataclass
class HostState:
    tx: object
    graphstate: object = dataclasses.field(default_factory=lambda: np.asarray(1.0, dtype=np.float32))
    opt_state: str = "optimizer"
    step: int = 0

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


@pytest.mark.parametrize("configured_via_arguments", [False, True])
def test_spmd_schedule_config_keeps_ordinary_training_outputs(monkeypatch, configured_via_arguments):
    """A registered trainer step must not select a scheduled VJP on SPMD."""

    # The ordinary compiler stands in only for device compilation. This captures
    # observable outputs, not calls to a private selection predicate.
    def ordinary_jit(fn, **kwargs):
        return lambda *args, **kwargs: ("ordinary updated state", "ordinary metrics")

    monkeypatch.setattr(training_utils.spx, "jit", ordinary_jit)
    config = (
        {"arguments": SimpleNamespace(mpmd_scheduler=object())} if configured_via_arguments else {"schedule": object()}
    )
    compiled = training_utils.compile_trainer_step(training_step, mesh=SimpleNamespace(is_mpmd=False), **config)
    assert compiled("initial state", {"input_ids": "opaque batch"}) == ("ordinary updated state", "ordinary metrics")


@pytest.mark.parametrize("weighted,auxiliary", [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("rows", [3, 4])
def test_scheduled_pipeline_and_accumulation_preserve_global_objective(monkeypatch, weighted, auxiliary, empty, rows):
    """Two pipeline microbatches per accumulation call retain the full-batch mean."""
    monkeypatch.setattr(training_utils, "jnp", np)
    monkeypatch.setattr(base_fn, "jnp", np)
    monkeypatch.setattr(base_fn.jax.lax, "stop_gradient", lambda value: value)
    monkeypatch.setattr(
        training_utils.lax,
        "dynamic_slice_in_dim",
        lambda x, start, size, axis: np.take(x, range(start, start + size), axis=axis),
    )
    monkeypatch.setattr(training_utils.spx, "jit", lambda fn, **kwargs: fn)
    monkeypatch.setattr(training_utils.jax, "process_count", lambda: 1)
    monkeypatch.setattr(training_utils, "_SCHED_SECTION_LOG_BUDGET", [0])
    received_gradients = []

    def apply_gradients_stage_local(**kwargs):
        received_gradients.append(float(kwargs["grads"]))
        return kwargs["params"] - kwargs["grads"], "updated optimizer"

    def token_weight(tree, batch):
        return np.asarray(batch["valid"].sum(), dtype=np.float32)

    def loss_fn(tree, batch):
        assert batch["position_ids"].shape == (3, batch["input_ids"].shape[0], 3)
        np.testing.assert_array_equal(batch["position_ids"][0], batch["input_ids"])
        count = batch["valid"].sum()
        mean = np.asarray((batch["token_losses"] * batch["valid"]).sum() / max(count, 1), dtype=np.float32)
        loss = tree * mean
        metrics = LossMetrics(loss=loss, accuracy=mean, weight_sum=np.asarray(count, dtype=np.float32))
        if auxiliary:
            aux_loss = tree * batch["auxiliary"].mean()
            loss += aux_loss
            metrics = metrics.replace(loss=loss, other_metrics={"aux_loss": aux_loss})
            loss, metrics = base_fn._split_ce_aux_objective(
                loss, metrics, batch[base_fn._CE_TOKEN_SCALE_KEY], lambda part: part["valid"].sum(), batch
            )
        return loss, {"accuracy": metrics.accuracy, "weight_sum": metrics.weight_sum}

    def host_scheduled_vag(fn, *, argnums, has_aux, microbatch_weight_fn):
        # Simulate the established pipeline contract: per-invocation normalized
        # weights. Gradients are analytic for the linear scalar objective.
        def evaluate(tree, batch):
            # SpectraX sees batch-major arrays and splits every leaf on axis 0.
            np.testing.assert_array_equal(batch["position_ids"][:, 0], batch["input_ids"])
            assert batch["position_ids"].shape == (batch["input_ids"].shape[0], 3, 3)
            chunks = [
                {key: value[i : i + 1] for key, value in batch.items()} for i in range(batch["input_ids"].shape[0])
            ]
            weights = (
                np.asarray([microbatch_weight_fn(tree, chunk) for chunk in chunks])
                if microbatch_weight_fn
                else np.ones(len(chunks))
            )
            scales = weights / weights.sum() if weights.sum() else np.full(len(chunks), 1.0 / len(chunks))
            loss = np.asarray(0.0, dtype=np.float32)
            accuracy = np.asarray(0.0, dtype=np.float32)
            for chunk, scale in zip(chunks, scales, strict=True):
                value, aux = fn(tree, chunk)
                loss += value * scale
                accuracy += aux["accuracy"] * scale
            return (loss, {"accuracy": accuracy, "weight_sum": np.asarray(batch["valid"].sum(), dtype=np.float32)}), (
                loss / tree,
            )

        return evaluate

    monkeypatch.setattr(training_utils.spx, "sxvalue_and_grad", host_scheduled_vag)

    def custom_step(state, batch, gradient_accumulation_steps=1, loss_config=None, learning_rate_fn=None):
        raise AssertionError("Expected the registered scheduled implementation")

    adapter = training_utils.ScheduledLossAdapter(
        name="host_accumulation",
        make_loss=lambda call: loss_fn,
        make_cache_key=lambda call: (),
        has_aux=True,
        make_microbatch_weight=lambda call: token_weight if weighted else None,
    )
    training_utils.register_scheduled_loss_adapter(custom_step, adapter)
    state = HostState(tx=SimpleNamespace(apply_gradients_stage_local=apply_gradients_stage_local))
    batch = {
        "input_ids": np.arange(12, dtype=np.int32).reshape(4, 3),
        "valid": np.array([[1, 0], [0, 0], [1, 0], [1, 1]], dtype=np.float32),
        "token_losses": np.array([[2, 99], [99, 99], [8, 99], [8, 8]], dtype=np.float32),
    }
    batch = {key: value[:rows] for key, value in batch.items()}
    batch["position_ids"] = np.stack([batch["input_ids"]] * 3)
    if empty:
        batch["valid"][:] = 0
    if auxiliary:
        batch["auxiliary"] = np.arange(1, rows + 1, dtype=np.float32)
        batch[base_fn._CE_TOKEN_SCALE_KEY] = base_fn._ce_token_scale_leaf(lambda part: part["valid"].sum(), batch)
    compiled = training_utils.compile_trainer_step(custom_step, mesh=SimpleNamespace(is_mpmd=True), schedule=object())
    new_state, metrics = compiled(state, batch, gradient_accumulation_steps=(2 if rows == 4 else 3))
    # Uniform reduction intentionally keeps the historical per-row objective.
    token_weighted = weighted or auxiliary
    expected_ce = (
        0.0 if empty else ((6.5 if token_weighted else 4.5) if rows == 4 else (5.0 if token_weighted else 10.0 / 3))
    )
    expected = expected_ce + ((rows + 1) / 2 if auxiliary else 0.0)
    assert float(metrics.loss) == pytest.approx(expected)
    assert float(metrics.accuracy) == pytest.approx(expected_ce)
    assert float(metrics.weight_sum) == (0.0 if empty else (4.0 if rows == 4 else 2.0))
    assert received_gradients == pytest.approx([expected])
    assert float(new_state.graphstate) == pytest.approx(1.0 - expected)
    assert new_state.step == 1
