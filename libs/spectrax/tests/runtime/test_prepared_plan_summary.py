"""Host-only contracts for the public prepared-plan diagnostics API."""

from __future__ import annotations

import pytest
from spectrax.runtime.mpmd import MpmdPipelineExecutor
from spectrax.runtime.mpmd.pipeline_executor import _PrepareCacheEntry
from spectrax.runtime.mpmd.runtime import _InvarAssemblyPlan

pytestmark = pytest.mark.cpu_ok


@pytest.fixture
def executor():
    """Provide a prepared host-metadata fixture without arrays or devices."""
    executor = MpmdPipelineExecutor()
    executor._prepare_cache["bucket"] = _PrepareCacheEntry(
        state={
            "fn_outvar_map": [(0, 0), (1, 0), ("orig", 3), (1, 2)],
            "donate_argnums_per_stage": [[1, 3], []],
        },
        flat_args_template=[],
        runtime_static_cache={},
        runtime_static_flat_args={},
        arg_offsets=(0, 1, 3, 3, 6),
        arg_leaf_counts=(1, 2, 0, 3, 2),
        invar_plans=(
            _InvarAssemblyPlan(
                template=(None, None, None, "static"),
                dynamic_slots=((0, 1), (1, 3), (2, 6)),
                stage_slots=(),
                prev_slots=(),
            ),
            _InvarAssemblyPlan(
                template=(None,) * 6,
                dynamic_slots=((0, 4), (1, 5), (2, 7), (3, 3)),
                stage_slots=((4, 0, 0, None, 0),),
                prev_slots=((5, 1),),
            ),
        ),
    )
    yield executor
    executor.shutdown()


def test_summary_reports_argument_routing_and_output_ownership(executor):
    assert executor.get_prepared_plan_summary("bucket", kv_argnum=3, metadata_argnum=4) == {
        "arg_leaf_counts": (1, 2, 0, 3, 2),
        "num_stages": 2,
        "kv_leaves": 3,
        "metadata_leaves": 2,
        "dynamic_slots": (3, 4),
        "dynamic_kv_slots": (1, 3),
        "dynamic_metadata_slots": (1, 1),
        "stage_slots": (0, 1),
        "prev_slots": (0, 1),
        "total_invars": (4, 6),
        "donated_by_stage": ((1, 3), ()),
        "donation_counts": (2, 0),
        "kv_outputs_by_stage": {0: 1, 1: 1, "orig": 1},
    }


def test_summary_is_absent_before_prepare_and_after_clear(executor):
    assert executor.get_prepared_plan_summary("missing", kv_argnum=3, metadata_argnum=4) is None
    executor.clear_prepare_cache()
    assert executor.get_prepared_plan_summary("bucket", kv_argnum=3, metadata_argnum=4) is None


def test_summary_containers_are_detached_from_execution(executor):
    summary = executor.get_prepared_plan_summary("bucket", kv_argnum=3, metadata_argnum=4)
    summary["kv_outputs_by_stage"][0] = 999
    summary["donated_by_stage"] = ((999,),)
    summary["arg_leaf_counts"] = ()
    fresh = executor.get_prepared_plan_summary("bucket", kv_argnum=3, metadata_argnum=4)
    assert fresh["kv_outputs_by_stage"] == {0: 1, 1: 1, "orig": 1}
    assert fresh["donated_by_stage"] == ((1, 3), ())
    assert fresh["arg_leaf_counts"] == (1, 2, 0, 3, 2)


@pytest.mark.parametrize("argnum", [-1, 2, 5, 99])
def test_empty_or_unavailable_argument_spans_have_zero_counts(executor, argnum):
    summary = executor.get_prepared_plan_summary("bucket", kv_argnum=argnum, metadata_argnum=argnum)
    assert summary["kv_leaves"] == summary["metadata_leaves"] == 0
    assert summary["dynamic_kv_slots"] == summary["dynamic_metadata_slots"] == (0, 0)
    assert summary["kv_outputs_by_stage"] == {}


def test_legacy_plan_without_donation_metadata(executor):
    executor._prepare_cache["bucket"].state.pop("donate_argnums_per_stage")
    summary = executor.get_prepared_plan_summary("bucket", kv_argnum=3, metadata_argnum=4)
    assert summary["donated_by_stage"] == ()
    assert summary["donation_counts"] == ()
    assert summary["kv_outputs_by_stage"] == {0: 1, 1: 1, "orig": 1}
