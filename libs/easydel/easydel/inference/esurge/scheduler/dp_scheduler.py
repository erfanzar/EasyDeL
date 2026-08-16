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

"""Data-parallel scheduler coordinator for eSurge.

This rank-major TPU DP scheduler owns one normal scheduler per DP rank, routes
each request to one rank for its lifetime, fans out ``schedule()``, merges
scheduler outputs for the runner, then splits ``ModelRunnerOutput`` back to the
owning rank scheduler.
"""

from __future__ import annotations

import atexit
import dataclasses
import enum
import multiprocessing
import os
import signal
import threading
import time
import typing
from collections import defaultdict, deque
from collections.abc import Iterable
from multiprocessing.connection import Connection

import cloudpickle
from eformer.loggings import get_logger

from ..core.dp_sharding import pages_per_dp_shard
from ..core.interface import CacheGroupsConfig
from ..engine_types import EngineCoreOutputs, FinishReason
from ..outputs import LogprobsLists, ModelRunnerOutput
from ..request import EngineRequest, EngineRequestStatus
from .interface import SchedulerInterface
from .output import CachedRequestData, NewRequestData, SchedulerOutput
from .scheduler import Scheduler

if typing.TYPE_CHECKING:
    from ..runners.model_runner import eSurgeRunner

logger = get_logger("eSurgeDPScheduler")

_STATUS_BY_FINISH_REASON = {
    FinishReason.STOP: EngineRequestStatus.FINISHED_STOPPED,
    FinishReason.LENGTH: EngineRequestStatus.FINISHED_LENGTH_CAPPED,
    FinishReason.ABORT: EngineRequestStatus.FINISHED_ABORTED,
}


class _SchedulerCommand(enum.Enum):
    ADD_REQUEST = "add_request"
    SCHEDULE = "schedule"
    UPDATE_FROM_OUTPUT = "update_from_output"
    FINISH_REQUESTS = "finish_requests"
    GET_NUM_UNFINISHED_REQUESTS = "get_num_unfinished_requests"
    HAS_FINISHED_REQUESTS = "has_finished_requests"
    GET_REQUEST_COUNTS = "get_request_counts"
    RESET_PREFIX_CACHE = "reset_prefix_cache"
    GET_PENDING_PREFILL_TOKENS = "get_pending_prefill_tokens"
    PROBE_COMPUTED_TOKENS = "probe_computed_tokens"
    SNAPSHOT_REQUESTS = "snapshot_requests"
    SNAPSHOT_RUNNING = "snapshot_running"
    SNAPSHOT_WAITING = "snapshot_waiting"
    SHUTDOWN = "shutdown"


class DPSchedulerClosed(RuntimeError):
    """The DP scheduler's rank workers are gone; no further RPC is possible.

    Raised instead of letting a closed pipe surface as ``EOFError`` or
    ``BrokenPipeError``, which the engine loop cannot tell apart from a
    transient fault and therefore retries.
    """


class _SchedulerWorkerError(Exception):
    def __init__(self, rank: int, message: str) -> None:
        self.rank = rank
        self.message = message
        super().__init__(f"Scheduler worker {rank} error: {message}")

    def __reduce__(self):
        return (self.__class__, (self.rank, self.message))


def _pending_prefill_tokens(scheduler: Scheduler) -> int:
    total = 0
    for req in scheduler.running:
        total += max(0, int(req.num_prompt_tokens) - int(req.num_computed_tokens))
    for req in scheduler.waiting:
        total += int(req.num_prompt_tokens)
    return total


def _probe_computed_tokens(scheduler: Scheduler, request: EngineRequest) -> int:
    manager = scheduler.kv_cache_manager
    if not manager.enable_caching:
        return 0
    # Use the read-only probe so ranks that never own this request do not
    # accumulate a req_to_page_hashes entry (only the chosen rank frees it).
    return int(manager.probe_num_computed_tokens(request))


def _scheduler_worker_process(
    rank: int,
    input_conn: Connection,
    output_conn: Connection,
    scheduler_cls: type[Scheduler],
    scheduler_kwargs: dict[str, typing.Any],
) -> None:
    scheduler = scheduler_cls(**scheduler_kwargs)
    scheduler.data_parallel_size = 1
    logger.info("DPScheduler worker rank %d started (pid=%d).", rank, os.getpid())

    def _send_result(result: typing.Any) -> None:
        output_conn.send_bytes(cloudpickle.dumps(result))

    while True:
        try:
            command, data = cloudpickle.loads(input_conn.recv_bytes())
            match command:
                case _SchedulerCommand.ADD_REQUEST:
                    scheduler.add_request(data)
                    _send_result(None)
                case _SchedulerCommand.SCHEDULE:
                    _send_result(scheduler.schedule())
                case _SchedulerCommand.UPDATE_FROM_OUTPUT:
                    scheduler_output, model_runner_output = data
                    _send_result(scheduler.update_from_output(scheduler_output, model_runner_output))
                case _SchedulerCommand.FINISH_REQUESTS:
                    request_ids, finished_status = data
                    scheduler.finish_requests(request_ids, finished_status)
                    _send_result(None)
                case _SchedulerCommand.GET_NUM_UNFINISHED_REQUESTS:
                    _send_result(scheduler.get_num_unfinished_requests())
                case _SchedulerCommand.HAS_FINISHED_REQUESTS:
                    _send_result(scheduler.has_finished_requests())
                case _SchedulerCommand.GET_REQUEST_COUNTS:
                    _send_result(scheduler.get_request_counts())
                case _SchedulerCommand.RESET_PREFIX_CACHE:
                    _send_result(scheduler.reset_prefix_cache())
                case _SchedulerCommand.GET_PENDING_PREFILL_TOKENS:
                    _send_result(_pending_prefill_tokens(scheduler))
                case _SchedulerCommand.PROBE_COMPUTED_TOKENS:
                    _send_result(_probe_computed_tokens(scheduler, data))
                case _SchedulerCommand.SNAPSHOT_REQUESTS:
                    _send_result(dict(scheduler.requests))
                case _SchedulerCommand.SNAPSHOT_RUNNING:
                    _send_result(list(scheduler.running))
                case _SchedulerCommand.SNAPSHOT_WAITING:
                    _send_result(list(scheduler.waiting))
                case _SchedulerCommand.SHUTDOWN:
                    scheduler.shutdown()
                    _send_result(None)
                    os._exit(0)
                case _:
                    raise _SchedulerWorkerError(rank, f"Unknown command: {command!r}")
        except (KeyboardInterrupt, SystemExit):
            try:
                scheduler.shutdown()
            except Exception:
                pass
            os._exit(0)
        except Exception as exc:
            logger.error("DPScheduler worker rank %d failed.", rank, exc_info=True)
            _send_result(_SchedulerWorkerError(rank, str(exc)))


class _SchedulerWorkerClient:
    def __init__(
        self,
        *,
        rank: int,
        scheduler_cls: type[Scheduler],
        scheduler_kwargs: dict[str, typing.Any],
        start_method: str,
    ) -> None:
        self.rank = int(rank)
        self._rpc_lock = threading.Lock()
        ctx = multiprocessing.get_context(start_method)
        input_parent, input_child = ctx.Pipe()
        output_parent, output_child = ctx.Pipe()
        self._input_conn = input_parent
        self._output_conn = output_parent
        self._process = ctx.Process(
            target=_scheduler_worker_process,
            args=(self.rank, input_child, output_child, scheduler_cls, scheduler_kwargs),
            name=f"eSurgeDPSchedulerRank{self.rank}",
        )
        # A rank scheduler is pure host-side bookkeeping, but under the default
        # ``spawn`` start method the child re-imports easydel to unpickle the
        # target, and that import calls ``initialize_distributed()``, which
        # defaults to ON. A second incarnation connecting to the parent's
        # coordination service makes the parent's own distributed client abort
        # the whole process ("tried to connect with a different incarnation"),
        # killing the engine a few seconds after the workers come up. Every
        # other spawned EasyDeL worker already opts out the same way (see
        # ``workers/esurge/pipeline/worker_main.py``); this one must too, and it
        # has to happen in the parent because ``spawn`` snapshots ``os.environ``
        # at exec time -- before any code in the child runs.
        previous_distributed_init = os.environ.get("ENABLE_DISTRIBUTED_INIT")
        os.environ["ENABLE_DISTRIBUTED_INIT"] = "0"
        try:
            self._process.start()
        finally:
            if previous_distributed_init is None:
                os.environ.pop("ENABLE_DISTRIBUTED_INIT", None)
            else:
                os.environ["ENABLE_DISTRIBUTED_INIT"] = previous_distributed_init
        input_child.close()
        output_child.close()
        atexit.register(self._atexit_cleanup)

    def _atexit_cleanup(self) -> None:
        if self._process.is_alive():
            try:
                os.kill(self._process.pid, signal.SIGKILL)
            except OSError:
                pass
        self._process.join(timeout=1.0)

    def _send_command(self, command: _SchedulerCommand, data: typing.Any = None) -> None:
        self._input_conn.send_bytes(cloudpickle.dumps((command, data)))

    def _get_result(self, command: _SchedulerCommand) -> typing.Any:
        try:
            raw = self._output_conn.recv_bytes()
        except Exception as exc:
            if not self._process.is_alive():
                raise RuntimeError(
                    f"DPScheduler worker rank {self.rank} exited with code {self._process.exitcode} "
                    f"while handling {command.value!r}."
                ) from exc
            raise
        result = cloudpickle.loads(raw)
        if isinstance(result, _SchedulerWorkerError):
            raise result
        return result

    def _call(self, command: _SchedulerCommand, data: typing.Any = None) -> typing.Any:
        # ``multiprocessing.Connection`` does not preserve request/reply
        # ownership when multiple engine threads use the same client. Keep
        # each untagged RPC atomic so a queue-stats read cannot consume the
        # scheduler loop's result (or vice versa).
        with self._rpc_lock:
            self._send_command(command, data)
            return self._get_result(command)

    def begin_call(self, command: _SchedulerCommand, data: typing.Any = None) -> None:
        """Send an RPC and hold this client's lock until :meth:`end_call`.

        Splitting send from receive lets the coordinator dispatch every rank
        before blocking on any reply, so independent worker processes run
        their share concurrently instead of one-at-a-time. The lock is still
        held across the whole request/reply window, preserving the atomicity
        that :meth:`_call` provides against concurrent stats readers.

        Args:
            command: Worker command to dispatch.
            data: Command payload.
        """
        self._rpc_lock.acquire()
        try:
            self._send_command(command, data)
        except BaseException:
            self._rpc_lock.release()
            raise

    def end_call(self, command: _SchedulerCommand) -> typing.Any:
        """Receive the reply for a pending :meth:`begin_call` and unlock.

        Args:
            command: The command passed to the matching :meth:`begin_call`.

        Returns:
            The worker's deserialized result.
        """
        try:
            return self._get_result(command)
        finally:
            self._rpc_lock.release()

    def add_request(self, request: EngineRequest) -> None:
        self._call(_SchedulerCommand.ADD_REQUEST, request)

    def schedule(self) -> SchedulerOutput:
        return self._call(_SchedulerCommand.SCHEDULE)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        return self._call(_SchedulerCommand.UPDATE_FROM_OUTPUT, (scheduler_output, model_runner_output))

    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: EngineRequestStatus,
    ) -> None:
        self._call(_SchedulerCommand.FINISH_REQUESTS, (request_ids, finished_status))

    def get_num_unfinished_requests(self) -> int:
        return int(self._call(_SchedulerCommand.GET_NUM_UNFINISHED_REQUESTS))

    def has_finished_requests(self) -> bool:
        return bool(self._call(_SchedulerCommand.HAS_FINISHED_REQUESTS))

    def get_request_counts(self) -> tuple[int, int]:
        running, waiting = self._call(_SchedulerCommand.GET_REQUEST_COUNTS)
        return int(running), int(waiting)

    def reset_prefix_cache(self) -> bool:
        return bool(self._call(_SchedulerCommand.RESET_PREFIX_CACHE))

    def get_pending_prefill_tokens(self) -> int:
        return int(self._call(_SchedulerCommand.GET_PENDING_PREFILL_TOKENS))

    def probe_computed_tokens(self, request: EngineRequest) -> int:
        return int(self._call(_SchedulerCommand.PROBE_COMPUTED_TOKENS, request))

    @property
    def requests(self) -> dict[str, EngineRequest]:
        return self._call(_SchedulerCommand.SNAPSHOT_REQUESTS)

    @property
    def running(self) -> list[EngineRequest]:
        return self._call(_SchedulerCommand.SNAPSHOT_RUNNING)

    @property
    def waiting(self) -> list[EngineRequest]:
        return self._call(_SchedulerCommand.SNAPSHOT_WAITING)

    def shutdown(self) -> None:
        try:
            atexit.unregister(self._atexit_cleanup)
        except Exception:
            pass
        if self._process.is_alive():
            try:
                self._call(_SchedulerCommand.SHUTDOWN)
            except Exception:
                logger.warning("Failed to shut down DPScheduler worker rank %d cleanly.", self.rank, exc_info=True)
        self._process.join(timeout=5.0)
        if self._process.is_alive():
            try:
                os.kill(self._process.pid, signal.SIGKILL)
            except OSError:
                pass
            self._process.join(timeout=1.0)
        self._input_conn.close()
        self._output_conn.close()


class DPScheduler(SchedulerInterface):
    """Coordinator with one EasyDeL scheduler per DP rank.

    Unlike the legacy ``Scheduler`` DP hint path, this object gives each rank
    an independent waiting/running queue and KV page manager. Page IDs emitted
    by rank-local schedulers are translated back into EasyDeL's global KV page
    ID space before the runner sees them.
    """

    def __init__(
        self,
        *,
        kv_cache_config: CacheGroupsConfig,
        dp_size: int,
        max_num_seqs: int,
        max_num_batched_tokens: int | None,
        max_model_len: int,
        num_pages: int,
        page_size: int,
        enable_prefix_caching: bool = True,
        max_num_seq_buckets: list[int] | None = None,
        async_scheduling: bool = True,
        long_prefill_token_threshold: int | None = None,
        chunked_prefill_enabled: bool = False,
        token_safety_margin: int | None = None,
        policy: typing.Literal["priority", "fcfs"] = "fcfs",
        num_draft_tokens: int = 0,
        use_eagle: bool = False,
        include_finished_set: bool = False,
        use_worker_processes: bool = True,
        worker_start_method: str | None = None,
    ) -> None:
        self.dp_size = int(dp_size)
        if self.dp_size <= 1:
            raise ValueError(f"DPScheduler requires dp_size > 1, got {dp_size}.")

        pages_per_rank = pages_per_dp_shard(num_pages, self.dp_size)
        if pages_per_rank is None:
            raise ValueError(
                "DPScheduler requires the global KV page tensor to be divisible by DP size: "
                f"num_pages={num_pages}, dp_size={self.dp_size}."
            )

        self.global_num_pages = int(num_pages)
        self.pages_per_rank = int(pages_per_rank)
        self.page_size = int(page_size)
        self.max_num_running_reqs = int(max_num_seqs)
        self.max_num_scheduled_tokens = max_num_batched_tokens if max_num_batched_tokens is not None else max_model_len
        self.max_model_len = int(max_model_len)
        self.async_scheduling = bool(async_scheduling)
        self.data_parallel_size = self.dp_size
        self.use_worker_processes = bool(use_worker_processes)
        self.worker_start_method = worker_start_method or os.environ.get("EASYDEL_DP_SCHEDULER_START_METHOD", "spawn")
        self.req_id_to_dp_rank: dict[str, int] = {}
        self._requests: dict[str, EngineRequest] = {}
        self._cached_rank_outputs: deque[list[SchedulerOutput]] = deque()
        self._schedule_step = 0
        self._last_schedule_start = 0.0

        per_rank_config = CacheGroupsConfig(
            num_pages=self.pages_per_rank,
            kv_cache_groups=kv_cache_config.kv_cache_groups,
        )
        scheduler_cls: type[Scheduler]
        if async_scheduling:
            from .async_scheduler import AsyncScheduler

            scheduler_cls = AsyncScheduler
        else:
            scheduler_cls = Scheduler

        self.schedulers: list[Scheduler | _SchedulerWorkerClient] = []
        for rank in range(self.dp_size):
            scheduler_kwargs = dict(
                kv_cache_config=per_rank_config,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
                num_pages=self.pages_per_rank,
                page_size=page_size,
                enable_prefix_caching=enable_prefix_caching,
                max_num_seq_buckets=max_num_seq_buckets,
                async_scheduling=async_scheduling,
                long_prefill_token_threshold=long_prefill_token_threshold,
                chunked_prefill_enabled=chunked_prefill_enabled,
                token_safety_margin=token_safety_margin,
                policy=policy,
                num_draft_tokens=num_draft_tokens,
                use_eagle=use_eagle,
                include_finished_set=include_finished_set,
            )
            if self.use_worker_processes:
                scheduler = _SchedulerWorkerClient(
                    rank=rank,
                    scheduler_cls=scheduler_cls,
                    scheduler_kwargs=scheduler_kwargs,
                    start_method=self.worker_start_method,
                )
            else:
                scheduler = scheduler_cls(**scheduler_kwargs)
                scheduler.data_parallel_size = 1
            self.schedulers.append(scheduler)

        self._closed = False
        if self.use_worker_processes:
            # Registered after every rank client's own atexit hook, so LIFO runs
            # this one first. That ordering is the whole point: those hooks
            # SIGKILL the worker processes, and until this flag is set the
            # engine's background scheduler thread is still calling schedule().
            # Observed on a 2-slice v6e run -- the workers were killed at
            # interpreter exit while the loop was live, the loop took five
            # EOFError/BrokenPipeError retries to give up, rank 0 never reached
            # jax.distributed's shutdown barrier, and rank 1 aborted the whole
            # job five minutes later with "Shutdown barrier has failed".
            atexit.register(self._mark_closed)
            self.policy = policy
            self.max_num_seq_buckets = max_num_seq_buckets or [self.max_num_running_reqs]
        else:
            self.policy = typing.cast(Scheduler, self.schedulers[0]).policy
            self.max_num_seq_buckets = typing.cast(Scheduler, self.schedulers[0]).max_num_seq_buckets

        logger.info(
            "DPScheduler started %d rank-local scheduler%s. start_method=%s. "
            "Per-rank limits: max_seqs=%d, max_tokens=%s; global max_tokens across dp=%d.",
            self.dp_size,
            " worker processes" if self.use_worker_processes else "s",
            self.worker_start_method if self.use_worker_processes else "in-process",
            max_num_seqs,
            max_num_batched_tokens,
            int(self.max_num_scheduled_tokens) * self.dp_size,
        )

    @classmethod
    def from_runner(
        cls,
        runner: eSurgeRunner,
        max_num_batched_tokens: int | None = None,
        enable_prefix_caching: bool = True,
        async_scheduling: bool = True,
        long_prefill_token_threshold: int | None = None,
        num_draft_tokens: int = 0,
    ) -> DPScheduler:
        from ..core.interface import create_kv_cache_specs_from_config

        metadata = runner.metadata
        model_config = runner.model.config
        if max_num_batched_tokens is None:
            max_num_batched_tokens = runner.max_model_len

        kv_cache_groups = getattr(runner, "kv_cache_groups", None)
        if not kv_cache_groups:
            kv_cache_groups = create_kv_cache_specs_from_config(
                config=model_config,
                page_size=metadata.page_size,
                num_kv_heads=metadata.num_kv_heads,
                head_size=getattr(metadata, "k_headdim", None) or getattr(metadata, "head_dim", None),
                dtype=metadata.kvdtype,
                use_mla=False,
            )

        return cls(
            kv_cache_config=CacheGroupsConfig(num_pages=metadata.num_pages, kv_cache_groups=kv_cache_groups),
            dp_size=int(getattr(metadata, "data_parallel_size", 1) or 1),
            max_num_seqs=runner.max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=runner.max_model_len,
            num_pages=metadata.num_pages,
            page_size=metadata.page_size,
            enable_prefix_caching=enable_prefix_caching,
            max_num_seq_buckets=list(runner.max_num_seq_buckets) if runner.max_num_seq_buckets else None,
            async_scheduling=async_scheduling,
            long_prefill_token_threshold=long_prefill_token_threshold,
            num_draft_tokens=int(num_draft_tokens),
        )

    @property
    def requests(self) -> dict[str, EngineRequest]:
        return self._requests

    @property
    def running(self) -> list[EngineRequest]:
        if self.use_worker_processes:
            return [req for rank_running in self._fanout(_SchedulerCommand.SNAPSHOT_RUNNING) for req in rank_running]
        return [req for scheduler in self.schedulers for req in scheduler.running]

    @property
    def waiting(self) -> list[EngineRequest]:
        if self.use_worker_processes:
            return [req for rank_waiting in self._fanout(_SchedulerCommand.SNAPSHOT_WAITING) for req in rank_waiting]
        return [req for scheduler in self.schedulers for req in scheduler.waiting]

    def _fanout(
        self,
        command: _SchedulerCommand,
        payloads: list[typing.Any] | None = None,
        *,
        collect_errors: bool = False,
    ) -> list[typing.Any]:
        """Run one worker RPC per rank concurrently, results in rank order.

        Every rank's request is written before any reply is read, so the
        worker processes overlap instead of serializing behind the
        coordinator. Replies are still consumed in rank order, so the
        returned list is deterministic and each client's FIFO pairing is
        unchanged.

        Args:
            command: Worker command to dispatch to every rank.
            payloads: Per-rank payloads; ``None`` sends ``None`` to each rank.
            collect_errors: When True a failing rank yields ``None`` in the
                result list instead of aborting the whole fan-out.

        Returns:
            One result per rank, ordered by rank.

        Raises:
            Exception: The first rank error, unless ``collect_errors`` is set.
        """
        if self._closed:
            raise DPSchedulerClosed(f"DP scheduler is shut down; refusing to dispatch {command.value!r}")
        clients = typing.cast(list[_SchedulerWorkerClient], self.schedulers)
        started: list[_SchedulerWorkerClient] = []
        try:
            for rank, client in enumerate(clients):
                client.begin_call(command, None if payloads is None else payloads[rank])
                started.append(client)
        except BaseException:
            # Drain whatever was already dispatched; leaving replies queued
            # would desynchronize those clients' pipes for every later call.
            for client in started:
                try:
                    client.end_call(command)
                except Exception:
                    logger.debug("Discarding rank %d reply after a fan-out send failure.", client.rank, exc_info=True)
            raise

        results: list[typing.Any] = []
        first_error: BaseException | None = None
        for client in started:
            try:
                results.append(client.end_call(command))
            except BaseException as exc:
                results.append(None)
                if first_error is None:
                    first_error = exc
        if first_error is not None and not collect_errors:
            raise first_error
        return results

    def _global_page_id(self, rank: int, page_id: int) -> int:
        pid = int(page_id)
        if pid <= 0:
            return 0
        return int(rank) * self.pages_per_rank + pid

    def _offset_page_ids(self, rank: int, page_ids: tuple[list[int], ...]) -> tuple[list[int], ...]:
        return tuple([self._global_page_id(rank, pid) for pid in group_ids] for group_ids in page_ids)

    def _offset_new_request(self, rank: int, req: NewRequestData) -> NewRequestData:
        return dataclasses.replace(
            req,
            page_ids=self._offset_page_ids(rank, req.page_ids),
            dp_rank=rank,
        )

    def _find_best_rank_for_request(self, request: EngineRequest) -> int:
        if self.use_worker_processes:
            probes = self._fanout(
                _SchedulerCommand.PROBE_COMPUTED_TOKENS,
                [request] * self.dp_size,
                collect_errors=True,
            )
        else:
            probes = []
            for scheduler in self.schedulers:
                try:
                    probes.append(_probe_computed_tokens(typing.cast(Scheduler, scheduler), request))
                except Exception:
                    probes.append(None)

        best_cache_rank: int | None = None
        best_cache_tokens = 0
        for rank, cached_tokens in enumerate(probes):
            if cached_tokens is None:
                logger.debug("Prefix-cache rank probe failed for req %s rank %d.", request.request_id, rank)
                continue
            cached_tokens = int(cached_tokens)
            if cached_tokens > best_cache_tokens:
                best_cache_tokens = cached_tokens
                best_cache_rank = rank
        if best_cache_rank is not None:
            return best_cache_rank

        if self.use_worker_processes:
            pending = self._fanout(_SchedulerCommand.GET_PENDING_PREFILL_TOKENS)
        else:
            pending = [_pending_prefill_tokens(typing.cast(Scheduler, s)) for s in self.schedulers]
        return min(range(self.dp_size), key=lambda rank: (int(pending[rank]), rank))

    def add_request(self, request: EngineRequest) -> None:
        if request.request_id in self.req_id_to_dp_rank:
            raise ValueError(
                f"Request {request.request_id} already assigned to DP rank {self.req_id_to_dp_rank[request.request_id]}."
            )
        rank = self._find_best_rank_for_request(request)
        self.req_id_to_dp_rank[request.request_id] = rank
        self._requests[request.request_id] = request
        self.schedulers[rank].add_request(request)

    def schedule(self) -> SchedulerOutput:
        self._schedule_step += 1
        now = time.time()
        if self._last_schedule_start > 0:
            logger.debug("DPScheduler previous step e2e time: %.4f seconds", now - self._last_schedule_start)
        self._last_schedule_start = now

        if self.use_worker_processes:
            rank_outputs = self._fanout(_SchedulerCommand.SCHEDULE)
        else:
            rank_outputs = [scheduler.schedule() for scheduler in self.schedulers]
        combined_output = self._combine_scheduler_outputs(rank_outputs)
        # EngineLoop deliberately skips ``update_from_output`` for a completely
        # idle schedule. Caching those rank-local outputs would leave this
        # queue permanently ahead of the updates: after any idle period, the
        # next real model result would be applied to an old empty schedule and
        # its sampled tokens would be dropped.
        if (
            combined_output.total_num_scheduled_tokens > 0
            or combined_output.finished_req_ids
            or combined_output.preempted_req_ids
            or combined_output.scheduled_new_reqs
            or combined_output.scheduled_cached_reqs.num_reqs
        ):
            self._cached_rank_outputs.append(rank_outputs)
        return combined_output

    def _combine_scheduler_outputs(self, rank_outputs: list[SchedulerOutput]) -> SchedulerOutput:
        scheduled_new_reqs: list[NewRequestData] = []
        for rank, output in enumerate(rank_outputs):
            scheduled_new_reqs.extend(self._offset_new_request(rank, req) for req in output.scheduled_new_reqs)

        scheduled_cached_reqs = self._combine_cached_request_data(rank_outputs)
        num_scheduled_tokens: dict[str, int] = {}
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        finished_req_ids: set[str] = set()
        preempted_req_ids: set[str] = set()
        total_num_scheduled_tokens = 0
        max_tokens_per_rank = 0
        num_running_reqs = 0
        num_waiting_reqs = 0
        free_pages = 0
        token_budget_initial = 0
        token_budget_remaining = 0
        req_ids_per_rank: dict[int, list[str]] = {}

        for rank, output in enumerate(rank_outputs):
            num_scheduled_tokens.update(output.num_scheduled_tokens)
            scheduled_spec_decode_tokens.update(output.scheduled_spec_decode_tokens)
            finished_req_ids.update(output.finished_req_ids)
            preempted_req_ids.update(output.preempted_req_ids)
            total_num_scheduled_tokens += int(output.total_num_scheduled_tokens)
            max_tokens_per_rank = max(max_tokens_per_rank, int(output.total_num_scheduled_tokens))
            num_running_reqs += int(output.num_running_reqs)
            num_waiting_reqs += int(output.num_waiting_reqs)
            free_pages += int(output.free_pages or 0)
            token_budget_initial += int(output.token_budget_initial or 0)
            token_budget_remaining += int(output.token_budget_remaining or 0)
            req_ids_per_rank[rank] = list(output.num_scheduled_tokens)

        assigned_dp_rank = {
            req_id: self.req_id_to_dp_rank[req_id] for req_id in num_scheduled_tokens if req_id in self.req_id_to_dp_rank
        }
        for req_id in finished_req_ids:
            self.req_id_to_dp_rank.pop(req_id, None)

        suggested_bucket = None
        if rank_outputs:
            rank_buckets = [out.suggested_bucket for out in rank_outputs if out.suggested_bucket is not None]
            if rank_buckets:
                suggested_bucket = sum(int(bucket) for bucket in rank_buckets)

        return SchedulerOutput(
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=scheduled_cached_reqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            num_common_prefix_pages=rank_outputs[0].num_common_prefix_pages if rank_outputs else [],
            finished_req_ids=finished_req_ids,
            preempted_req_ids=preempted_req_ids,
            suggested_bucket=suggested_bucket,
            async_scheduling=all(output.async_scheduling for output in rank_outputs),
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=num_waiting_reqs,
            free_pages=free_pages,
            token_budget_initial=token_budget_initial,
            token_budget_remaining=token_budget_remaining,
            req_id_to_dp_rank=assigned_dp_rank,
            max_num_scheduled_tokens_per_dp_rank=max_tokens_per_rank,
            req_ids_per_rank=req_ids_per_rank,
        )

    def _combine_cached_request_data(self, rank_outputs: list[SchedulerOutput]) -> CachedRequestData:
        req_ids: list[str] = []
        resumed_from_preemption: list[bool] = []
        new_token_ids: list[list[int]] = []
        new_page_ids: list[tuple[list[int], ...]] = []
        num_computed_tokens: list[int] = []
        dp_ranks: list[int | None] = []

        for rank, output in enumerate(rank_outputs):
            cached = output.scheduled_cached_reqs
            req_ids.extend(cached.req_ids)
            resumed_from_preemption.extend(cached.resumed_from_preemption)
            new_token_ids.extend(cached.new_token_ids)
            new_page_ids.extend(self._offset_page_ids(rank, ids) for ids in cached.new_page_ids)
            num_computed_tokens.extend(cached.num_computed_tokens)
            dp_ranks.extend([rank] * len(cached.req_ids))

        return CachedRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_page_ids=new_page_ids,
            num_computed_tokens=num_computed_tokens,
            dp_ranks=dp_ranks,
        )

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        if self._cached_rank_outputs:
            rank_outputs = self._cached_rank_outputs.popleft()
        else:
            rank_outputs = [
                self._filter_scheduler_output_for_rank(scheduler_output, rank) for rank in range(self.dp_size)
            ]

        rank_model_outputs = self._split_model_output_by_rank(scheduler_output, model_runner_output)
        if self.use_worker_processes:
            per_rank_engine_outputs = self._fanout(
                _SchedulerCommand.UPDATE_FROM_OUTPUT,
                [(rank_outputs[rank], rank_model_outputs[rank]) for rank in range(self.dp_size)],
            )
        else:
            per_rank_engine_outputs = [
                scheduler.update_from_output(rank_outputs[rank], rank_model_outputs[rank])
                for rank, scheduler in enumerate(self.schedulers)
            ]

        combined: dict[int, EngineCoreOutputs] = {}
        finished_req_ids: set[str] = set()
        for rank_engine_outputs in per_rank_engine_outputs:
            for client_index, engine_outputs in rank_engine_outputs.items():
                if engine_outputs.finished_requests:
                    finished_req_ids.update(engine_outputs.finished_requests)
                existing = combined.get(client_index)
                if existing is None:
                    combined[client_index] = engine_outputs
                    continue
                existing.outputs.extend(engine_outputs.outputs)
                if engine_outputs.finished_requests:
                    if existing.finished_requests is None:
                        existing.finished_requests = set()
                    existing.finished_requests.update(engine_outputs.finished_requests)

        self._mirror_engine_outputs(combined)
        for req_id in scheduler_output.finished_req_ids | finished_req_ids:
            self.req_id_to_dp_rank.pop(req_id, None)
            if req_id in finished_req_ids:
                self._requests.pop(req_id, None)

        return combined

    def _mirror_engine_outputs(self, engine_outputs_by_client: dict[int, EngineCoreOutputs]) -> None:
        if not self.use_worker_processes:
            # In-process rank schedulers mutate the very ``EngineRequest``
            # objects stored in ``self._requests``: they have already
            # appended the sampled tokens and set the finished status /
            # stop reason. Mirroring here would append every token a second
            # time, which then makes the next ``schedule()`` ask the runner
            # for two tokens per request. Only worker-process ranks hold
            # distinct (cloudpickled) copies that need catching up.
            return
        for engine_outputs in engine_outputs_by_client.values():
            for output in engine_outputs.outputs:
                request = self._requests.get(output.request_id)
                if request is None:
                    continue
                if output.new_token_ids:
                    request.append_output_token_ids(output.new_token_ids)
                if output.finish_reason is not None:
                    request.status = _STATUS_BY_FINISH_REASON.get(
                        output.finish_reason,
                        EngineRequestStatus.FINISHED_STOPPED,
                    )
                    request.stop_reason = output.stop_reason
                    request.num_cached_tokens = output.num_cached_tokens

    def _filter_scheduler_output_for_rank(self, scheduler_output: SchedulerOutput, rank: int) -> SchedulerOutput:
        rank_req_ids = set(scheduler_output.req_ids_per_rank.get(rank, []))
        cached = scheduler_output.scheduled_cached_reqs
        cached_indices = [i for i, req_id in enumerate(cached.req_ids) if req_id in rank_req_ids]
        return SchedulerOutput(
            scheduled_new_reqs=[req for req in scheduler_output.scheduled_new_reqs if req.req_id in rank_req_ids],
            scheduled_cached_reqs=CachedRequestData(
                req_ids=[cached.req_ids[i] for i in cached_indices],
                resumed_from_preemption=[cached.resumed_from_preemption[i] for i in cached_indices],
                new_token_ids=[cached.new_token_ids[i] for i in cached_indices],
                new_page_ids=[cached.new_page_ids[i] for i in cached_indices],
                num_computed_tokens=[cached.num_computed_tokens[i] for i in cached_indices],
                dp_ranks=[rank] * len(cached_indices),
            ),
            num_scheduled_tokens={
                req_id: count
                for req_id, count in scheduler_output.num_scheduled_tokens.items()
                if req_id in rank_req_ids
            },
            total_num_scheduled_tokens=sum(
                count for req_id, count in scheduler_output.num_scheduled_tokens.items() if req_id in rank_req_ids
            ),
            scheduled_spec_decode_tokens={
                req_id: tokens
                for req_id, tokens in scheduler_output.scheduled_spec_decode_tokens.items()
                if req_id in rank_req_ids
            },
            num_common_prefix_pages=scheduler_output.num_common_prefix_pages,
            finished_req_ids={
                req_id for req_id in scheduler_output.finished_req_ids if self.req_id_to_dp_rank.get(req_id) == rank
            },
            preempted_req_ids={
                req_id for req_id in scheduler_output.preempted_req_ids if self.req_id_to_dp_rank.get(req_id) == rank
            },
            suggested_bucket=scheduler_output.suggested_bucket,
            async_scheduling=scheduler_output.async_scheduling,
            req_id_to_dp_rank={req_id: rank for req_id in rank_req_ids if req_id in scheduler_output.req_id_to_dp_rank},
        )

    def _split_model_output_by_rank(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> list[ModelRunnerOutput]:
        outputs: list[ModelRunnerOutput] = []
        for rank in range(self.dp_size):
            req_ids = scheduler_output.req_ids_per_rank.get(rank, [])
            global_indices = [
                model_runner_output.req_id_to_index[req_id]
                for req_id in req_ids
                if req_id in model_runner_output.req_id_to_index
            ]
            rank_req_ids = [req_id for req_id in req_ids if req_id in model_runner_output.req_id_to_index]
            rank_req_id_to_index = {req_id: idx for idx, req_id in enumerate(rank_req_ids)}

            outputs.append(
                ModelRunnerOutput(
                    req_ids=rank_req_ids,
                    req_id_to_index=rank_req_id_to_index,
                    sampled_token_ids=[model_runner_output.sampled_token_ids[i] for i in global_indices]
                    if model_runner_output.sampled_token_ids
                    else [],
                    spec_token_ids=[model_runner_output.spec_token_ids[i] for i in global_indices]
                    if model_runner_output.spec_token_ids is not None
                    else None,
                    logprobs=self._slice_logprobs(model_runner_output.logprobs, global_indices)
                    if model_runner_output.logprobs is not None
                    else None,
                    prompt_logprobs_dict={
                        req_id: model_runner_output.prompt_logprobs_dict[req_id]
                        for req_id in rank_req_ids
                        if req_id in model_runner_output.prompt_logprobs_dict
                    },
                    req_id_to_row_index={
                        req_id: model_runner_output.req_id_to_row_index[req_id]
                        for req_id in rank_req_ids
                        if model_runner_output.req_id_to_row_index is not None
                        and req_id in model_runner_output.req_id_to_row_index
                    }
                    if model_runner_output.req_id_to_row_index is not None
                    else None,
                    finished_sending={
                        req_id
                        for req_id in (model_runner_output.finished_sending or set())
                        if self.req_id_to_dp_rank.get(req_id) == rank
                    }
                    if model_runner_output.finished_sending is not None
                    else None,
                    finished_recving={
                        req_id
                        for req_id in (model_runner_output.finished_recving or set())
                        if self.req_id_to_dp_rank.get(req_id) == rank
                    }
                    if model_runner_output.finished_recving is not None
                    else None,
                    num_nans_in_logits={
                        req_id: model_runner_output.num_nans_in_logits[req_id]
                        for req_id in rank_req_ids
                        if model_runner_output.num_nans_in_logits is not None
                        and req_id in model_runner_output.num_nans_in_logits
                    }
                    if model_runner_output.num_nans_in_logits is not None
                    else None,
                    token_logprobs={
                        req_id: model_runner_output.token_logprobs[req_id]
                        for req_id in rank_req_ids
                        if model_runner_output.token_logprobs is not None
                        and req_id in model_runner_output.token_logprobs
                    }
                    if model_runner_output.token_logprobs is not None
                    else None,
                    num_accepted_spec_tokens={
                        req_id: model_runner_output.num_accepted_spec_tokens[req_id]
                        for req_id in rank_req_ids
                        if model_runner_output.num_accepted_spec_tokens is not None
                        and req_id in model_runner_output.num_accepted_spec_tokens
                    }
                    if model_runner_output.num_accepted_spec_tokens is not None
                    else None,
                    hidden_states={
                        req_id: model_runner_output.hidden_states[req_id]
                        for req_id in rank_req_ids
                        if model_runner_output.hidden_states is not None and req_id in model_runner_output.hidden_states
                    }
                    if model_runner_output.hidden_states is not None
                    else None,
                )
            )
        return outputs

    @staticmethod
    def _slice_logprobs(logprobs: LogprobsLists | None, indices: list[int]) -> LogprobsLists | None:
        if logprobs is None:
            return None
        return LogprobsLists(
            logprob_token_ids=[logprobs.logprob_token_ids[i] for i in indices],
            logprobs=[logprobs.logprobs[i] for i in indices],
            sampled_token_ranks=[logprobs.sampled_token_ranks[i] for i in indices],
        )

    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: EngineRequestStatus,
    ) -> None:
        if isinstance(request_ids, str):
            request_ids_iter = [request_ids]
        else:
            request_ids_iter = list(request_ids)

        per_rank: dict[int, list[str]] = defaultdict(list)
        for req_id in request_ids_iter:
            rank = self.req_id_to_dp_rank.get(req_id)
            if rank is not None:
                per_rank[rank].append(req_id)

        for rank, ids in per_rank.items():
            self.schedulers[rank].finish_requests(ids, finished_status)
        for req_id in request_ids_iter:
            self.req_id_to_dp_rank.pop(req_id, None)
            if (request := self._requests.pop(req_id, None)) is not None:
                request.status = finished_status

    def get_num_unfinished_requests(self) -> int:
        if self.use_worker_processes:
            return sum(int(count) for count in self._fanout(_SchedulerCommand.GET_NUM_UNFINISHED_REQUESTS))
        return sum(scheduler.get_num_unfinished_requests() for scheduler in self.schedulers)

    def has_finished_requests(self) -> bool:
        if self.use_worker_processes:
            return any(bool(flag) for flag in self._fanout(_SchedulerCommand.HAS_FINISHED_REQUESTS))
        return any(scheduler.has_finished_requests() for scheduler in self.schedulers)

    def get_request_counts(self) -> tuple[int, int]:
        if self.use_worker_processes:
            per_rank = self._fanout(_SchedulerCommand.GET_REQUEST_COUNTS)
        else:
            per_rank = [scheduler.get_request_counts() for scheduler in self.schedulers]
        running = 0
        waiting = 0
        for rank_running, rank_waiting in per_rank:
            running += int(rank_running)
            waiting += int(rank_waiting)
        return running, waiting

    def reset_prefix_cache(self) -> bool:
        if self.use_worker_processes:
            return all(bool(ok) for ok in self._fanout(_SchedulerCommand.RESET_PREFIX_CACHE))
        return all(scheduler.reset_prefix_cache() for scheduler in self.schedulers)

    def _mark_closed(self) -> None:
        """Refuse further fan-out. Safe to call from an atexit hook."""

        self._closed = True

    def shutdown(self) -> None:
        # Closed before the workers are told to stop, not after: a caller on
        # another thread that is mid-schedule() must get DPSchedulerClosed
        # rather than a torn pipe.
        self._mark_closed()
        if self.use_worker_processes:
            atexit.unregister(self._mark_closed)
        for scheduler in self.schedulers:
            scheduler.shutdown()
