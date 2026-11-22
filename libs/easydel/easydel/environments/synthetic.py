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

"""On-the-fly task synthesis, 2-phase verification gate, and difficulty-aware streaming."""

from __future__ import annotations

import collections
import collections.abc
import concurrent.futures
import dataclasses
import math
import queue
import threading
import time
import typing as tp

from easydel.data.core.protocols import ShardedDataSource, ShardInfo

from .auto import AutoEnvironment
from .harnesses import HarnessPreset
from .runtime import ContainerConfig, KiriRuntime, SandboxPool


@dataclasses.dataclass(frozen=True)
class TaskSanityReport:
    """Outcome of running :class:`TaskVerificationGate` on one candidate task."""

    accepted: bool
    reason: str
    baseline_reward: float | None = None
    reference_reward: float | None = None
    env_error: bool = False


class TaskVerificationGate:
    """Two-phase sanity gate ensuring synthesized tasks are neither trivial nor broken.

    1. **Triviality check**: Runs ``baseline_action`` (default empty/no-op) in an
       isolated environment; rejects if reward ``>= pass_threshold``.
    2. **Solvability check**: Runs the task's ``reference_solution`` (or
       ``reference_actions`` sequence for multi-turn tasks); rejects if reward
       ``< pass_threshold`` or if an infrastructure ``env_error`` occurs.
    """

    def __init__(
        self,
        *,
        environment: str | HarnessPreset | collections.abc.Mapping[str, tp.Any] = "auto",
        runtime: KiriRuntime | SandboxPool | None = None,
        backend: str | None = None,
        container_config: ContainerConfig | None = None,
        pass_threshold: float = 0.99,
        baseline_action: str | None = None,
    ):
        self._environment = environment
        self._runtime = runtime
        self._backend = backend
        self._container_config = container_config
        self._pass_threshold = float(pass_threshold)
        self._baseline_action = baseline_action

        self._lock = threading.Lock()
        self._stats = {
            "checked": 0,
            "accepted": 0,
            "rejected_trivial": 0,
            "rejected_unsolvable": 0,
            "rejected_missing_reference": 0,
            "rejected_infra_error": 0,
        }

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def _inc(self, key: str) -> None:
        with self._lock:
            self._stats[key] += 1

    def verify(self, task: collections.abc.Mapping[str, tp.Any]) -> TaskSanityReport:
        """Validate ``task`` against baseline failure and reference solution success."""
        self._inc("checked")
        ref_actions = task.get("reference_actions")
        ref_solution = task.get("reference_solution")
        if ref_actions is None and ref_solution is None:
            self._inc("rejected_missing_reference")
            return TaskSanityReport(accepted=False, reason="missing reference_solution or reference_actions")

        # Phase 1: Baseline / empty action must NOT pass.
        env_base = AutoEnvironment.from_task(
            task,
            environment=None if self._environment == "auto" else self._environment,
            runtime=self._runtime,
            backend=self._backend,
            container_config=self._container_config,
        )
        try:
            r0 = env_base.reset()
            if r0.info.get("env_error"):
                self._inc("rejected_infra_error")
                return TaskSanityReport(
                    accepted=False,
                    reason=str(r0.info.get("env_error_message", "reset env_error")),
                    env_error=True,
                )
            baseline_act = self._baseline_action
            if baseline_act is None:
                from .code_judge import CodeJudgeEnvironment

                baseline_act = "```python\npass\n```" if isinstance(env_base, CodeJudgeEnvironment) else ""
            base_step = env_base.step(baseline_act)
            if base_step.info.get("env_error"):
                self._inc("rejected_infra_error")
                return TaskSanityReport(
                    accepted=False,
                    reason=str(base_step.info.get("env_error_message", "baseline env_error")),
                    env_error=True,
                )
            base_reward = float(base_step.reward)
            if base_reward >= self._pass_threshold:
                self._inc("rejected_trivial")
                return TaskSanityReport(
                    accepted=False,
                    reason=f"trivial task: baseline scored {base_reward}",
                    baseline_reward=base_reward,
                )
        finally:
            env_base.close()

        # Phase 2: Reference solution / multi-turn action sequence MUST pass.
        if isinstance(ref_actions, str):
            actions: list[str] = [ref_actions]
        elif isinstance(ref_actions, (list, tuple)):
            actions = [str(a) for a in ref_actions]
        else:
            actions = [str(ref_solution)]
        env_ref = AutoEnvironment.from_task(
            task,
            environment=None if self._environment == "auto" else self._environment,
            runtime=self._runtime,
            backend=self._backend,
            container_config=self._container_config,
        )
        try:
            r1 = env_ref.reset()
            if r1.info.get("env_error"):
                self._inc("rejected_infra_error")
                return TaskSanityReport(
                    accepted=False,
                    reason=str(r1.info.get("env_error_message", "reset env_error")),
                    baseline_reward=base_reward,
                    env_error=True,
                )
            total_reward = 0.0
            for act in actions:
                step_res = env_ref.step(act)
                if step_res.info.get("env_error"):
                    self._inc("rejected_infra_error")
                    return TaskSanityReport(
                        accepted=False,
                        reason=str(step_res.info.get("env_error_message", "reference env_error")),
                        baseline_reward=base_reward,
                        env_error=True,
                    )
                total_reward += float(step_res.reward)
                if step_res.terminated or step_res.truncated:
                    break
            if total_reward < self._pass_threshold:
                self._inc("rejected_unsolvable")
                return TaskSanityReport(
                    accepted=False,
                    reason=f"reference solution scored {total_reward} < {self._pass_threshold}",
                    baseline_reward=base_reward,
                    reference_reward=total_reward,
                )
            self._inc("accepted")
            return TaskSanityReport(
                accepted=True,
                reason="ok",
                baseline_reward=base_reward,
                reference_reward=total_reward,
            )
        finally:
            env_ref.close()


class DifficultyTracker:
    """Thread-safe rolling ``pass@k`` tracker for online curriculum & filtering.

    Tracks per-task solve rates over the most recent ``window_size`` rollouts.
    Tasks whose solve rate reaches ``max_solve_rate`` (e.g. ``1.0``) are retired
    from active RL sampling, while tasks in ``(min_solve_rate, max_solve_rate)``
    deliver maximum group-relative advantage variance.
    """

    def __init__(
        self,
        *,
        window_size: int = 16,
        min_solve_rate: float = 0.0,
        max_solve_rate: float = 1.0,
        pass_threshold: float = 0.5,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        min_r = float(min_solve_rate)
        max_r = float(max_solve_rate)
        thresh = float(pass_threshold)
        if not (math.isfinite(min_r) and math.isfinite(max_r) and 0.0 <= min_r <= max_r <= 1.0):
            raise ValueError("Solve-rate bounds must satisfy 0 <= min_solve_rate <= max_solve_rate <= 1.")
        if not math.isfinite(thresh):
            raise ValueError(f"pass_threshold must be finite; got {pass_threshold}.")
        self._window_size = int(window_size)
        self._min_rate = min_r
        self._max_rate = max_r
        self._pass_threshold = thresh
        self._lock = threading.Lock()
        self._history: dict[str, collections.deque[float]] = {}

    def record(self, task_id: str, rewards: collections.abc.Sequence[float | None]) -> float:
        """Record rollout rewards for ``task_id`` and return its updated rolling pass rate.

        ``None`` and non-finite entries (unscorable samples) are ignored.
        """
        valid: list[float] = []
        for r in rewards:
            if r is None:
                continue
            val = float(r)
            if math.isfinite(val):
                valid.append(val)
        if not valid:
            return self.solve_rate(task_id) or 0.0
        with self._lock:
            buf = self._history.get(task_id)
            if buf is None:
                buf = collections.deque(maxlen=self._window_size)
                self._history[task_id] = buf
            for val in valid:
                buf.append(1.0 if val >= self._pass_threshold else 0.0)
            return sum(buf) / len(buf)

    def solve_rate(self, task_id: str) -> float | None:
        with self._lock:
            buf = self._history.get(task_id)
            if not buf:
                return None
            return sum(buf) / len(buf)

    def is_admissible(self, task_id: str, *, strict_band: bool = False) -> bool:
        """Return whether ``task_id`` should be sampled for RL.

        Unseen tasks are always admissible. Once observed, tasks with solve rate
        ``>= max_solve_rate`` (when ``max_solve_rate == 1.0``) are retired. When
        ``strict_band=True``, requires ``min_solve_rate < rate < max_solve_rate``.
        """
        rate = self.solve_rate(task_id)
        if rate is None:
            return True
        if strict_band:
            return self._min_rate < rate < self._max_rate
        return rate < self._max_rate

    def summary(self) -> dict[str, int | float]:
        with self._lock:
            rates = [sum(b) / len(b) for b in self._history.values() if b]
            total = len(rates)
            if total == 0:
                return {"tracked_tasks": 0, "mastered": 0, "unsolved": 0, "active_band": 0}
            mastered = sum(1 for r in rates if r >= self._max_rate)
            unsolved = sum(1 for r in rates if r <= self._min_rate)
            active = sum(1 for r in rates if self._min_rate < r < self._max_rate)
            return {
                "tracked_tasks": total,
                "mastered": mastered,
                "unsolved": unsolved,
                "active_band": active,
                "mean_solve_rate": sum(rates) / len(rates),
            }


class SyntheticTaskStream(ShardedDataSource[dict[str, tp.Any]]):
    """Streaming ``ShardedDataSource`` that synthesizes and sandbox-verifies tasks on the fly.

    Compatible with :func:`easydel.data.block_mixture_interleave`, Hugging Face
    trainer dataloaders, and direct iteration. Background worker threads call
    ``generator_fn(seed)``, optionally validate each task through
    :class:`TaskVerificationGate`, and enqueue verified rows into a bounded
    buffer so the TPU/GPU learner never stalls waiting on task generation.
    """

    def __init__(
        self,
        generator_fn: tp.Callable[[int], collections.abc.Mapping[str, tp.Any] | None],
        *,
        verification_gate: TaskVerificationGate | None = None,
        difficulty_tracker: DifficultyTracker | None = None,
        num_workers: int = 4,
        buffer_size: int = 64,
        max_tasks: int | None = None,
        base_seed: int = 0,
        strip_reference: bool = True,
        max_consecutive_generator_failures: int = 32,
        generator_backoff: float = 0.05,
        generator_backoff_max: float = 2.0,
    ):
        """Create the stream.

        Args:
            generator_fn: ``seed -> task`` callable (``None`` skips the seed).
            verification_gate: Optional two-phase sanity gate applied to each task.
            difficulty_tracker: Optional tracker; retired tasks are skipped.
            num_workers: Background producer threads.
            buffer_size: Maximum buffered verified tasks.
            max_tasks: Optional cap on tasks yielded per iteration.
            base_seed: First seed handed to ``generator_fn``.
            strip_reference: Remove ``reference_solution``/``reference_actions`` before yielding.
            max_consecutive_generator_failures: Consecutive ``generator_fn`` exceptions (across workers) after
                which the stream stops and re-raises the last exception from :meth:`next_task` / iteration. The
                same threshold applies separately to consecutive ``verification_gate.verify`` exceptions.
            generator_backoff: Initial sleep after a ``generator_fn`` exception; doubles per consecutive failure.
            generator_backoff_max: Cap on the back-off sleep.
        """
        if num_workers <= 0:
            raise ValueError("num_workers must be positive.")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        if max_consecutive_generator_failures <= 0:
            raise ValueError("max_consecutive_generator_failures must be positive.")
        self._max_generator_failures = int(max_consecutive_generator_failures)
        self._backoff_base = max(0.0, float(generator_backoff))
        self._backoff_max = max(self._backoff_base, float(generator_backoff_max))
        self._failure_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._consecutive_generator_failures = 0
        self._consecutive_gate_failures = 0
        self._consecutive_infra_failures = 0
        self._last_generator_error: BaseException | None = None
        self._generator_fn = generator_fn
        self._gate = verification_gate
        self._difficulty = difficulty_tracker
        self._num_workers = int(num_workers)
        self._buffer_size = int(buffer_size)
        self._max_tasks = max_tasks
        self._base_seed = int(base_seed)
        self._strip_reference = bool(strip_reference)

        self._queue: queue.Queue[dict[str, tp.Any]] = queue.Queue(maxsize=self._buffer_size)
        self._stop = threading.Event()
        self._seed_lock = threading.Lock()
        self._next_seed = self._base_seed
        self._emitted = 0
        self._fatal_error: BaseException | None = None
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._started = False

    @property
    def shard_names(self) -> collections.abc.Sequence[str]:
        return ("synthetic_stream_0",)

    def num_shards(self) -> int:
        return 1

    def get_shard_info(self, shard_name: str) -> ShardInfo | None:
        if shard_name != "synthetic_stream_0":
            return None
        return ShardInfo(shard_id=0, shard_name=shard_name, num_rows=self._max_tasks)

    def _claim_seed(self) -> int:
        with self._seed_lock:
            seed = self._next_seed
            self._next_seed += 1
            return seed

    def _ensure_started(self) -> threading.Event:
        with self._lifecycle_lock:
            if self._started:
                return self._stop
            # Allocate a fresh Event per start so workers from a previous closed executor stay permanently stopped.
            stop_event = threading.Event()
            self._stop = stop_event
            with self._failure_lock:
                self._consecutive_generator_failures = 0
                self._consecutive_gate_failures = 0
                self._consecutive_infra_failures = 0
                self._last_generator_error = None
                self._fatal_error = None
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._num_workers,
                thread_name_prefix="ed-synth-task",
            )
            for _ in range(self._num_workers):
                self._executor.submit(self._producer_loop, stop_event)
            self._started = True
            return stop_event

    def _record_generator_failure(self, exc: BaseException, stop_event: threading.Event) -> float:
        """Record a ``generator_fn`` failure and return the back-off delay before the next attempt.

        Failures are counted across all workers; after ``max_consecutive_generator_failures`` in a row the
        stream stops and keeps ``exc`` as its fatal error.
        """
        with self._failure_lock:
            if stop_event.is_set():
                return 0.0
            self._consecutive_generator_failures += 1
            self._last_generator_error = exc
            count = self._consecutive_generator_failures
            if count >= self._max_generator_failures:
                if self._fatal_error is None:
                    self._fatal_error = exc
                stop_event.set()
        return min(self._backoff_max, self._backoff_base * (2.0 ** (count - 1)))

    def _record_gate_failure(self, exc: BaseException, stop_event: threading.Event) -> float:
        """Record a ``verification_gate.verify`` exception and return the back-off delay before the next task.

        Uses the same consecutive-failure threshold as generator failures; any gate verdict (accepted or not)
        resets the count. Only once the threshold is reached does the stream stop with ``exc`` as its fatal error.
        """
        with self._failure_lock:
            if stop_event.is_set():
                return 0.0
            self._consecutive_gate_failures += 1
            self._last_generator_error = exc
            count = self._consecutive_gate_failures
            if count >= self._max_generator_failures:
                if self._fatal_error is None:
                    self._fatal_error = exc
                stop_event.set()
        return min(self._backoff_max, self._backoff_base * (2.0 ** (count - 1)))

    def _producer_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            seed = self._claim_seed()
            try:
                raw = self._generator_fn(seed)
            except Exception as exc:
                delay = self._record_generator_failure(exc, stop_event)
                if stop_event.wait(delay):
                    return
                continue
            if stop_event.is_set():
                return
            with self._failure_lock:
                self._consecutive_generator_failures = 0
            if raw is None:
                continue
            task = dict(raw)
            task_id = str(task.get("task_id", f"synth_{seed}"))
            task.setdefault("task_id", task_id)

            if self._difficulty is not None and not self._difficulty.is_admissible(task_id):
                continue

            if self._gate is not None:
                try:
                    report = self._gate.verify(task)
                except Exception as exc:
                    # Transient gate failures are retried like generator failures; only a run of
                    # ``max_consecutive_generator_failures`` stops the stream with ``exc`` as its fatal error.
                    delay = self._record_gate_failure(exc, stop_event)
                    if stop_event.wait(delay):
                        return
                    continue
                if stop_event.is_set():
                    return
                with self._failure_lock:
                    self._consecutive_gate_failures = 0
                if not report.accepted:
                    if report.env_error:
                        with self._failure_lock:
                            if stop_event.is_set():
                                return
                            self._consecutive_infra_failures += 1
                            if self._consecutive_infra_failures >= 8 and self._gate.stats().get("accepted", 0) == 0:
                                from .runtime import SandboxInfraError

                                if self._fatal_error is None:
                                    self._fatal_error = SandboxInfraError(report.reason)
                                stop_event.set()
                                return
                    else:
                        with self._failure_lock:
                            self._consecutive_infra_failures = 0
                    continue
                with self._failure_lock:
                    self._consecutive_infra_failures = 0

            if self._strip_reference:
                task.pop("reference_solution", None)
                task.pop("reference_actions", None)

            while not stop_event.is_set():
                try:
                    self._queue.put(task, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _get_fatal_error(self) -> BaseException | None:
        with self._failure_lock:
            return self._fatal_error

    def next_task(self, timeout: float = 30.0) -> dict[str, tp.Any]:
        """Return the next verified task dictionary, blocking up to ``timeout`` seconds."""
        stop_event = self._ensure_started()
        deadline = time.monotonic() + float(timeout)
        while True:
            try:
                return self._queue.get(timeout=min(0.1, max(0.0, deadline - time.monotonic())))
            except queue.Empty:
                pass
            fatal = self._get_fatal_error()
            if fatal is not None and stop_event.is_set() and self._queue.empty():
                raise fatal
            if time.monotonic() >= deadline:
                if fatal is not None:
                    raise fatal
                with self._failure_lock:
                    last = self._last_generator_error
                message = f"SyntheticTaskStream produced no verified task within {timeout}s."
                if last is not None:
                    raise TimeoutError(f"{message} Last generator error: {last!r}") from last
                raise TimeoutError(message)

    def open_shard(self, shard_name: str) -> collections.abc.Iterator[dict[str, tp.Any]]:
        if shard_name != "synthetic_stream_0":
            raise KeyError(f"Unknown shard {shard_name!r}")
        stop_event = self._ensure_started()
        count = 0
        while True:
            if self._max_tasks is not None and count >= self._max_tasks:
                break
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if stop_event.is_set():
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                else:
                    continue
            count += 1
            self._emitted += 1
            yield item
        fatal = self._get_fatal_error()
        if fatal is not None and (self._max_tasks is None or count < self._max_tasks):
            raise fatal

    def __iter__(self) -> collections.abc.Iterator[dict[str, tp.Any]]:
        return self.open_shard("synthetic_stream_0")

    def stats(self) -> dict[str, tp.Any]:
        out: dict[str, tp.Any] = {
            "emitted": self._emitted,
            "buffered": self._queue.qsize(),
        }
        if self._gate is not None:
            out["gate"] = self._gate.stats()
        if self._difficulty is not None:
            out["difficulty"] = self._difficulty.summary()
        return out

    def close(self) -> None:
        with self._lifecycle_lock:
            self._stop.set()
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
            self._started = False

    def __enter__(self) -> SyntheticTaskStream:
        self._ensure_started()
        return self

    def __exit__(self, exc_type: tp.Any, exc: tp.Any, tb: tp.Any) -> None:
        self.close()


__all__ = (
    "DifficultyTracker",
    "SyntheticTaskStream",
    "TaskSanityReport",
    "TaskVerificationGate",
)
