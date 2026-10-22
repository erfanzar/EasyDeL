"""Documented executor convenience APIs remain usable without engine callers."""

from collections import OrderedDict
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.runners.executors.model_executor import ModelStepExecutor

pytestmark = pytest.mark.cpu_ok


def _executor(mpmd=False):
    executor = ModelStepExecutor.__new__(ModelStepExecutor)
    executor.mesh = SimpleNamespace(is_mpmd=mpmd)
    executor.use_aot_forward = False
    executor._pipeline_runtime = None
    executor._cache = OrderedDict()
    executor._backbone_cache = OrderedDict()
    executor._lm_head_cache = OrderedDict()
    return executor


def test_execute_many_keeps_input_order_on_spmd():
    executor = _executor()
    executor._cache[(2, 2, "model_step", "jit")] = lambda state, other, pages, metadata: (pages, metadata)
    assert executor.execute_many(
        num_tokens=2,
        padded_num_reqs=2,
        input_batches=[("state", "other", "pages1", "batch1"), ("state", "other", "pages2", "batch2")],
    ) == [("pages1", "batch1"), ("pages2", "batch2")]
    assert executor.execute_many(num_tokens=2, padded_num_reqs=2, input_batches=[]) == []


def test_legacy_has_requires_both_pipeline_halves():
    executor = _executor(mpmd=True)
    key = (8, 2, "model_step", "mpmd")
    assert not executor.has(key)
    executor._backbone_cache[(8, "backbone", "mpmd", False)] = object()
    assert not executor.has(key)
    executor._lm_head_cache[(2, "lm_head", "mpmd")] = object()
    assert executor.has(key)


def test_legacy_compile_preserves_spmd_return_value():
    executor = _executor()
    expected = object()
    executor.compile_model_step = lambda **kwargs: expected
    assert (
        executor.compile(
            num_tokens=2,
            padded_num_reqs=2,
            graphdef=None,
            graphstate=None,
            graphother=None,
            inputs=None,
        )
        is expected
    )


def test_legacy_compile_preserves_pipeline_return_value():
    executor = _executor(mpmd=True)
    expected = object()
    executor.compile_backbone = lambda **kwargs: expected
    executor.compile_lm_head = lambda **kwargs: None
    assert (
        executor.compile(
            num_tokens=2,
            padded_num_reqs=2,
            graphdef="definition",
            graphstate=None,
            graphother=None,
            inputs=None,
        )
        is expected
    )
    assert executor.graphdef == "definition"
