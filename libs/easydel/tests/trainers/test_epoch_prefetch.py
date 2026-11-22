"""Real host epoch loops must preserve mutable iterators and idle bucket queues."""

import contextlib
from types import SimpleNamespace

import pytest
from easydel.trainers.trainer import trainer as trainer_module
from easydel.trainers.trainer.trainer import Trainer

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("iterator_kind", ["list", "generator"])
@pytest.mark.parametrize("bucketed", [False, True])
def test_epoch_boundaries_replay_prefetched_rows_including_idle_buckets(monkeypatch, iterator_kind, bucketed):
    # Model computation is replaced by a host state transition; fetching,
    # collation, thread prefetching and both epoch loops remain production code.
    monkeypatch.setattr(trainer_module.jax, "device_get", lambda x: x)
    monkeypatch.setattr(
        trainer_module.jax.profiler, "StepTraceAnnotation", lambda *args, **kwargs: contextlib.nullcontext()
    )
    consumed = []

    def noop(*args, **kwargs):
        pass

    def execute(*, state, batch):
        consumed.append(batch)
        return SimpleNamespace(step=state.step + 1), SimpleNamespace(loss=0.0, accuracy=1.0), None

    def fetch(iterator, loader):
        return next(iterator), iterator

    def source(offset):
        values = list(range(offset, offset + 20))
        return iter(values) if iterator_kind == "list" else (value for value in values)

    host = SimpleNamespace(
        arguments=SimpleNamespace(
            dataloader_prefetch=True,
            dataloader_prefetch_workers=2,
            dataloader_prefetch_buffer_size=3,
            learning_rate=0.01,
            max_length=8,
            merge_lora_before_save=False,
        ),
        data_collator=lambda batch: batch,
        max_training_steps=4,
        _has_buckets=bucketed,
        _get_next_batch=fetch,
        _execute_train_step=execute,
        _runtime_batch_summary=lambda batch: batch,
        _apply_pose_to_batch=lambda batch, step: batch,
        _apply_randomized_yarn_to_batch=lambda batch, step: batch,
        _apply_process_encoder_to_batch=lambda batch, step: batch,
        on_step_start=lambda state, step: state,
        on_step_end=lambda state, metrics, step: (state, metrics),
        apply_training_hooks=lambda metrics, loss_value: metrics,
        _metric_scalar_to_float=float,
        _should_save_tpu_preemption_checkpoint=lambda step: False,
        _should_run_evaluation=lambda step: False,
        _profiler_should_block_until_ready=lambda: False,
        train_tracker=SimpleNamespace(trace_compilation=contextlib.nullcontext),
        scheduler=None,
        _backward_flops_per_token=0,
        _extra_backward_flops_per_token=0,
        training_batch_size=1,
    )
    for name in (
        "_runtime_trace",
        "_log_mesh_topology",
        "_maybe_start_profiler",
        "_maybe_stop_profiler",
        "log_metrics",
        "log_weight_distribution",
        "log_watchers",
        "maybe_generate",
        "maybe_benchmark",
        "_save_checkpoint_for_step",
    ):
        setattr(host, name, noop)
    if bucketed:
        host._bucket_loaders = {0: range(20), 1: range(100, 120)}
        host._bucket_iters = {0: source(0), 1: source(100)}
        host._bucket_collators = {0: lambda value: value, 1: lambda value: value}
        # Bucket 1 is warmed but never selected in epoch 0; select it next.
        host._bucket_rule = SimpleNamespace(select=lambda step: 0 if step < 2 else 1)
    step_metrics = SimpleNamespace(start_step=noop, calculate=lambda **kwargs: {})
    tracker = SimpleNamespace(update=lambda **kwargs: (0.0, 1.0))
    state = SimpleNamespace(step=0)
    iterator = source(0)
    # One step per epoch reproduces lookahead loss even when final-step refill
    # is disabled: initial warmup already consumed extra source rows.
    for epoch in range(4):
        state, error, iterator = Trainer._train_epoch(
            host, state, range(20), iterator, tracker, step_metrics, SimpleNamespace(), epoch, epoch, epoch + 1
        )
        assert error is None
    assert consumed == ([0, 1, 100, 101] if bucketed else [0, 1, 2, 3])
