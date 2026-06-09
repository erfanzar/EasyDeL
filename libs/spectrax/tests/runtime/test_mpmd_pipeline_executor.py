from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectrax.runtime.mpmd import MpmdPipelineExecutor
from spectrax.runtime.mpmd.pipeline_executor import _PreparedCall


class _FakeSubmesh:
    def __init__(self):
        self.devices = np.asarray(jax.devices()[:1])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _RecordingSubmesh(_FakeSubmesh):
    def __init__(self):
        super().__init__()
        self.active = 0
        self.enter_count = 0

    def __enter__(self):
        super().__enter__()
        self.active += 1
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.active -= 1
        return super().__exit__(exc_type, exc, tb)


def _fake_sxjit(stage0_delay: float = 0.0, stage1_delay: float = 0.0):
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    submesh = _FakeSubmesh()

    def stage0(x):
        if stage0_delay:
            time.sleep(stage0_delay)
        return (x + 1,)

    def stage1(x):
        if stage1_delay:
            time.sleep(stage1_delay)
        return (x * 2,)

    state = {
        "compiled": [
            (stage0, submesh, sharding, None, [("orig", 0)]),
            (stage1, submesh, sharding, None, [("stage", 0, 0)]),
        ],
        "placed": {},
        "dynamic": {0},
        "explicit_in_sh": {},
        "fn_outvar_map": [(1, 0)],
        "mpmd_mesh": None,
        "out_shardings": None,
        "result_treedef": None,
    }

    def fn(x):
        return (x + 1) * 2

    def prepare(*args, **kwargs):
        del kwargs
        assert len(args) == 1
        return state

    fn._mpmd_prepare = prepare
    return fn


def _mesh_asserting_sxjit():
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    submesh0 = _RecordingSubmesh()
    submesh1 = _RecordingSubmesh()

    def stage0(x):
        assert submesh0.active > 0
        return (x + 1,)

    def stage1(x):
        assert submesh1.active > 0
        return (x * 2,)

    state = {
        "compiled": [
            (stage0, submesh0, sharding, None, [("orig", 0)]),
            (stage1, submesh1, sharding, None, [("stage", 0, 0)]),
        ],
        "placed": {},
        "dynamic": {0},
        "explicit_in_sh": {},
        "fn_outvar_map": [(1, 0)],
        "mpmd_mesh": None,
        "out_shardings": None,
        "result_treedef": None,
    }

    def fn(x):
        return (x + 1) * 2

    def prepare(*args, **kwargs):
        del kwargs
        assert len(args) == 1
        return state

    fn._mpmd_prepare = prepare
    return fn, submesh0, submesh1


def _fake_stateful_sxjit(stage0_delay: float = 0.0, stage1_delay: float = 0.0):
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    submesh = _FakeSubmesh()

    def stage0(stage0_state, x):
        if stage0_delay:
            time.sleep(stage0_delay)
        next_state = stage0_state + x
        return (next_state, x + 1)

    def stage1(stage1_state, activation):
        if stage1_delay:
            time.sleep(stage1_delay)
        next_state = stage1_state + activation
        return (next_state, activation * 2)

    state = {
        "compiled": [
            (stage0, submesh, sharding, None, [("orig", 0), ("orig", 2)]),
            (stage1, submesh, sharding, None, [("orig", 1), ("stage", 0, 1)]),
        ],
        "placed": {},
        "dynamic": {0, 1, 2},
        "explicit_in_sh": {},
        "fn_outvar_map": [(0, 0), (1, 0), (1, 1)],
        "mpmd_mesh": None,
        "out_shardings": None,
        "result_treedef": None,
    }

    def fn(stage0_state, stage1_state, x):
        next_stage0 = stage0_state + x
        activation = x + 1
        next_stage1 = stage1_state + activation
        return next_stage0, next_stage1, activation * 2

    def prepare(*args, **kwargs):
        del kwargs
        assert len(args) == 3
        return state

    fn._mpmd_prepare = prepare
    return fn


def test_mpmd_pipeline_executor_dispatch_matches_plan():
    executor = MpmdPipelineExecutor()
    out = executor.dispatch(_fake_sxjit(), jnp.array([1, 2, 3], dtype=jnp.int32))

    np.testing.assert_array_equal(np.asarray(out), np.array([4, 6, 8], dtype=np.int32))
    assert executor.last_stats.stage_launches == 2
    assert executor.last_stats.microbatches == 1
    executor.shutdown()


def test_mpmd_pipeline_executor_dispatch_many_preserves_order_and_overlaps():
    executor = MpmdPipelineExecutor(use_workers=True)
    fn = _fake_sxjit(stage0_delay=0.03, stage1_delay=0.03)
    batches = [(jnp.array([i], dtype=jnp.int32),) for i in range(4)]

    t0 = time.time()
    outs = executor.dispatch_many(fn, batches)
    elapsed = time.time() - t0

    got = [int(np.asarray(out)[0]) for out in outs]
    assert got == [2, 4, 6, 8]
    assert executor.last_stats.stage_launches == 8
    assert executor.last_stats.microbatches == 4
    # Sequential stage-by-stage execution would sleep for roughly
    # 4 * 2 * 30ms = 240ms. The wavefront should overlap the stage workers.
    assert elapsed < 0.30
    executor.shutdown()


def test_mpmd_pipeline_executor_dispatch_many_carries_stage_local_state():
    executor = MpmdPipelineExecutor(use_workers=True)
    fn = _fake_stateful_sxjit(stage0_delay=0.05, stage1_delay=0.05)
    batches = [
        (jnp.array([0], dtype=jnp.int32), jnp.array([10], dtype=jnp.int32), jnp.array([i], dtype=jnp.int32))
        for i in range(4)
    ]

    t0 = time.time()
    outs = executor.dispatch_many(
        fn,
        batches,
        carry_input_output_map={
            0: {0: 0},
            1: {1: 0},
        },
    )
    elapsed = time.time() - t0

    got_stage0 = [int(np.asarray(out[0])[0]) for out in outs]
    got_stage1 = [int(np.asarray(out[1])[0]) for out in outs]
    got_values = [int(np.asarray(out[2])[0]) for out in outs]
    assert got_stage0 == [0, 1, 3, 6]
    assert got_stage1 == [11, 13, 16, 20]
    assert got_values == [2, 4, 6, 8]
    # Sequential execution would sleep for roughly 4 * 2 * 50ms = 400ms.
    # Carry dependencies serialize each stage's own state, but the two stages
    # should still overlap with each other.
    assert elapsed < 0.45
    executor.shutdown()


def test_mpmd_pipeline_executor_default_inline_wavefront_preserves_order():
    executor = MpmdPipelineExecutor()
    fn = _fake_sxjit()
    batches = [(jnp.array([i], dtype=jnp.int32),) for i in range(4)]

    outs = executor.dispatch_many(fn, batches)

    got = [int(np.asarray(out)[0]) for out in outs]
    assert got == [2, 4, 6, 8]
    assert executor.last_stats.stage_launches == 8
    assert executor.last_stats.microbatches == 4
    executor.shutdown()


def test_mpmd_pipeline_executor_enters_stage_submesh_for_every_launch():
    executor = MpmdPipelineExecutor()
    fn, submesh0, submesh1 = _mesh_asserting_sxjit()

    out = executor.dispatch(fn, jnp.array([1, 2, 3], dtype=jnp.int32))

    np.testing.assert_array_equal(np.asarray(out), np.array([4, 6, 8], dtype=np.int32))
    assert submesh0.enter_count == 1
    assert submesh1.enter_count == 1
    executor.shutdown()


def test_mpmd_pipeline_executor_workers_enter_stage_submesh_for_every_launch():
    executor = MpmdPipelineExecutor(use_workers=True)
    fn, submesh0, submesh1 = _mesh_asserting_sxjit()
    batches = [(jnp.array([i], dtype=jnp.int32),) for i in range(4)]

    outs = executor.dispatch_many(fn, batches)

    got = [int(np.asarray(out)[0]) for out in outs]
    assert got == [2, 4, 6, 8]
    assert submesh0.enter_count == 4
    assert submesh1.enter_count == 4
    executor.shutdown()


def test_mpmd_pipeline_executor_raises_for_missing_required_stage_output():
    executor = MpmdPipelineExecutor()
    call = _PreparedCall(
        state={
            "fn_outvar_map": [(1, 0)],
            "out_shardings": None,
            "result_treedef": None,
        },
        flat_args=[],
    )

    with pytest.raises(RuntimeError, match="missing output for required stage 1"):
        executor._assemble_result(call, [(jnp.array([1], dtype=jnp.int32),), None])
