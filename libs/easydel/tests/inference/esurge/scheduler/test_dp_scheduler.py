from __future__ import annotations

import threading
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from easydel.inference.esurge.core.interface import CacheGroupsConfig, CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.outputs import ModelRunnerOutput
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.scheduler.dp_scheduler import DPScheduler
from easydel.inference.esurge.scheduler.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams


def _kv_config(*, num_pages: int = 64, page_size: int = 4) -> CacheGroupsConfig:
    return CacheGroupsConfig(
        num_pages=num_pages,
        kv_cache_groups=[
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=page_size,
                    num_kv_heads=1,
                    head_size=4,
                    dtype=jnp.float32,
                    use_mla=False,
                ),
                layer_names=None,
            )
        ],
    )


def _make_dp_scheduler(*, async_scheduling: bool = False) -> DPScheduler:
    return DPScheduler(
        kv_cache_config=_kv_config(),
        dp_size=2,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        max_model_len=64,
        num_pages=64,
        page_size=4,
        enable_prefix_caching=False,
        async_scheduling=async_scheduling,
    )


def _request(req_id: str, *, prompt_len: int = 8) -> EngineRequest:
    return EngineRequest(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=8),
        eos_token_id=1,
    )


def _model_output(req_ids: list[str]) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[100 + i] for i in range(len(req_ids))],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        req_id_to_row_index={req_id: i for i, req_id in enumerate(req_ids)},
    )


def test_from_runner_uses_dp_scheduler_for_data_parallel_metadata() -> None:
    runner = SimpleNamespace(
        metadata=SimpleNamespace(
            num_pages=64,
            page_size=4,
            data_parallel_size=2,
        ),
        model=SimpleNamespace(config=SimpleNamespace()),
        kv_cache_groups=_kv_config().kv_cache_groups,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        max_model_len=64,
        max_num_seq_buckets=[4],
    )

    scheduler = Scheduler.from_runner(runner, max_num_batched_tokens=16, async_scheduling=False)
    try:
        assert isinstance(scheduler, DPScheduler)
        assert scheduler.use_worker_processes
    finally:
        scheduler.shutdown()


def test_dp_scheduler_routes_merges_offsets_and_splits_update() -> None:
    scheduler = _make_dp_scheduler()
    try:
        request0 = _request("r0")
        request1 = _request("r1")
        scheduler.add_request(request0)
        scheduler.add_request(request1)

        assert scheduler.req_id_to_dp_rank == {"r0": 0, "r1": 1}
        assert len(scheduler.schedulers[0].waiting) == 1
        assert len(scheduler.schedulers[1].waiting) == 1

        output = scheduler.schedule()

        assert output.req_id_to_dp_rank == {"r0": 0, "r1": 1}
        assert output.req_ids_per_rank == {0: ["r0"], 1: ["r1"]}
        assert output.total_num_scheduled_tokens == 16
        assert output.max_num_scheduled_tokens_per_dp_rank == 8

        page_ids_by_req = {req.req_id: req.page_ids[0] for req in output.scheduled_new_reqs}
        assert page_ids_by_req["r0"] == [1, 2]
        assert page_ids_by_req["r1"] == [33, 34]

        engine_outputs = scheduler.update_from_output(output, _model_output(list(output.num_scheduled_tokens)))

        assert set(engine_outputs) == {0}
        emitted = {
            out.request_id: out.new_token_ids
            for client_output in engine_outputs.values()
            for out in client_output.outputs
        }
        assert emitted == {"r0": [100], "r1": [101]}
        assert list(request0.output_token_ids) == [100]
        assert list(request1.output_token_ids) == [101]
        assert list(scheduler.schedulers[0].requests["r0"].output_token_ids) == [100]
        assert list(scheduler.schedulers[1].requests["r1"].output_token_ids) == [101]

        next_output = scheduler.schedule()

        assert next_output.scheduled_cached_reqs.req_ids == ["r0", "r1"]
        assert next_output.scheduled_cached_reqs.dp_ranks == [0, 1]
    finally:
        scheduler.shutdown()


def test_dp_scheduler_idle_schedule_does_not_desynchronize_next_update() -> None:
    scheduler = _make_dp_scheduler()
    try:
        idle_output = scheduler.schedule()
        assert idle_output.total_num_scheduled_tokens == 0
        assert not idle_output.scheduled_new_reqs
        assert idle_output.scheduled_cached_reqs.num_reqs == 0

        request0 = _request("r0")
        request1 = _request("r1")
        scheduler.add_request(request0)
        scheduler.add_request(request1)

        output = scheduler.schedule()
        engine_outputs = scheduler.update_from_output(output, _model_output(list(output.num_scheduled_tokens)))

        emitted = {
            out.request_id: out.new_token_ids
            for client_output in engine_outputs.values()
            for out in client_output.outputs
        }
        assert emitted == {"r0": [100], "r1": [101]}
        assert list(scheduler.schedulers[0].requests["r0"].output_token_ids) == [100]
        assert list(scheduler.schedulers[1].requests["r1"].output_token_ids) == [101]
    finally:
        scheduler.shutdown()


def test_dp_scheduler_worker_rpc_replies_stay_with_the_calling_thread() -> None:
    scheduler = _make_dp_scheduler()
    try:
        rank0 = scheduler.schedulers[0]
        original_get_result = rank0._get_result
        schedule_waiting = threading.Event()
        counts_received = threading.Event()
        results: dict[str, object] = {}
        errors: list[BaseException] = []

        def coordinated_get_result(command):
            if command.value == "schedule":
                schedule_waiting.set()
                counts_received.wait(timeout=0.5)
                return original_get_result(command)
            result = original_get_result(command)
            if command.value == "get_request_counts":
                counts_received.set()
            return result

        rank0._get_result = coordinated_get_result

        def run_schedule() -> None:
            try:
                results["schedule"] = scheduler.schedule()
            except BaseException as exc:
                errors.append(exc)

        def read_counts() -> None:
            try:
                results["counts"] = scheduler.get_request_counts()
            except BaseException as exc:
                errors.append(exc)

        schedule_thread = threading.Thread(target=run_schedule)
        counts_thread = threading.Thread(target=read_counts)
        schedule_thread.start()
        assert schedule_waiting.wait(timeout=5)
        counts_thread.start()
        schedule_thread.join(timeout=5)
        counts_thread.join(timeout=5)

        assert not schedule_thread.is_alive()
        assert not counts_thread.is_alive()
        assert not errors
        assert results["schedule"].total_num_scheduled_tokens == 0
        assert results["counts"] == (0, 0)
    finally:
        scheduler.shutdown()


def test_in_process_dp_scheduler_appends_each_sampled_token_once() -> None:
    """In-process ranks mutate the coordinator's own EngineRequest objects.

    Mirroring engine outputs on top of that appends every sampled token a
    second time, which then makes the next ``schedule()`` request two tokens
    per decode step instead of one.
    """
    scheduler = DPScheduler(
        kv_cache_config=_kv_config(),
        dp_size=2,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        max_model_len=64,
        num_pages=64,
        page_size=4,
        enable_prefix_caching=False,
        async_scheduling=False,
        use_worker_processes=False,
    )
    try:
        scheduler.add_request(_request("r0"))
        scheduler.add_request(_request("r1"))

        prefill = scheduler.schedule()
        scheduler.update_from_output(prefill, _model_output(list(prefill.num_scheduled_tokens)))
        assert list(scheduler.requests["r0"].output_token_ids) == [100]

        decode = scheduler.schedule()
        assert dict(decode.num_scheduled_tokens) == {"r0": 1, "r1": 1}
        scheduler.update_from_output(decode, _model_output(list(decode.num_scheduled_tokens)))

        assert list(scheduler.requests["r0"].output_token_ids) == [100, 100]
        assert list(scheduler.requests["r1"].output_token_ids) == [101, 101]
    finally:
        scheduler.shutdown()


def test_dp_scheduler_fanout_preserves_rank_order_and_update_pairing() -> None:
    """Concurrent send/serial receive must not reorder rank results."""
    scheduler = _make_dp_scheduler()
    try:
        scheduler.add_request(_request("r0"))
        scheduler.add_request(_request("r1"))

        output = scheduler.schedule()
        assert output.req_ids_per_rank == {0: ["r0"], 1: ["r1"]}
        assert [req.dp_rank for req in output.scheduled_new_reqs] == [0, 1]

        engine_outputs = scheduler.update_from_output(output, _model_output(list(output.num_scheduled_tokens)))
        emitted = {
            out.request_id: out.new_token_ids
            for client_output in engine_outputs.values()
            for out in client_output.outputs
        }
        assert emitted == {"r0": [100], "r1": [101]}
        assert list(scheduler.schedulers[0].requests["r0"].output_token_ids) == [100]
        assert list(scheduler.schedulers[1].requests["r1"].output_token_ids) == [101]
    finally:
        scheduler.shutdown()


def test_fanout_refuses_to_dispatch_once_closed() -> None:
    """A closed scheduler names itself rather than surfacing a torn pipe.

    The engine loop retries a generic exception; it cannot retry its way out of
    workers that no longer exist, so the distinction has to be in the type.
    """

    from easydel.inference.esurge.scheduler.dp_scheduler import (
        DPSchedulerClosed,
        _SchedulerCommand,
    )

    scheduler = _make_dp_scheduler()
    scheduler._mark_closed()

    with pytest.raises(DPSchedulerClosed):
        scheduler._fanout(_SchedulerCommand.SCHEDULE)


def test_shutdown_closes_before_stopping_the_workers() -> None:
    """Ordering, not just outcome: closed first, workers stopped second.

    Reversed, a caller already inside schedule() on the engine's loop thread
    races the workers' teardown and gets EOFError. On a two-slice run that cost
    five retries, which delayed rank 0 past jax.distributed's shutdown barrier
    and made rank 1 abort the job.
    """

    scheduler = _make_dp_scheduler()
    observed: list[bool] = []

    class _RecordingWorker:
        def shutdown(self) -> None:
            observed.append(scheduler._closed)

    scheduler.schedulers = [_RecordingWorker(), _RecordingWorker()]
    scheduler.shutdown()

    assert observed == [True, True]
    assert scheduler._closed is True
