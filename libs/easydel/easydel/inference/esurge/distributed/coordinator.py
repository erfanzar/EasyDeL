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

"""Step coordination between the engine loop and the runner.

The engine loop never calls the runner directly; every step goes through a
:class:`StepCoordinator`. On a single host the :class:`LocalCoordinator` is
an identity pass-through with zero overhead. On a multi-host pod the
coordinator's job is to replicate the leader's runner-call stream — every
``execute_sync``/``execute_async``/``drain`` in the same order — to every
worker process, because the jitted step is a global-mesh collective that
deadlocks unless all processes enter it together.

Correctness rests on determinism: given an identical ordered stream of
runner invocations with identical ``SchedulerOutput`` payloads, every
host's runner state is bit-identical by construction (device outputs use
replicated sharding; host-side repairs are deterministic functions of
them). Acks, digests, and heartbeats exist for failure *detection*, never
for correctness.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    import threading

    from ..outputs import ModelRunnerOutput
    from ..scheduler.output import SchedulerOutput


class StepCoordinationError(RuntimeError):
    """Non-recoverable lockstep failure.

    Raised when a peer host NACKs a step, misses an ack deadline, stops
    heartbeating, or reports a sampled-token digest mismatch. A JAX
    collective cannot survive a lost participant, so the engine loop must
    abort in-flight requests and shut down — never retry the step.
    """


@dataclass
class StepHandle:
    """An in-flight asynchronously-dispatched step.

    Attributes:
        step_id: Monotonic step index assigned by the coordinator.
        runner_handle: The runner's async execution handle.
        scheduler_output: The step plan that produced this dispatch.
    """

    step_id: int
    runner_handle: typing.Any
    scheduler_output: SchedulerOutput


class StepCoordinator(typing.Protocol):
    """The engine loop's only gateway to step execution.

    Implementations must preserve the invocation order they receive: the
    sequence of ``execute_sync``/``execute_async``/``drain`` calls *is* the
    replicated program under multi-host operation.
    """

    is_leader: bool
    rank: int
    world_size: int
    supports_overlap: bool

    def start(self) -> None:
        """Establish the control plane (single host: no-op)."""
        ...

    def execute_sync(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Execute one step synchronously on every host."""
        ...

    def execute_async(self, scheduler_output: SchedulerOutput) -> StepHandle:
        """Dispatch one step asynchronously on every host."""
        ...

    def drain(self, handle: StepHandle) -> ModelRunnerOutput:
        """Materialize a previously dispatched step's output."""
        ...

    def check_health(self) -> None:
        """Raise :class:`StepCoordinationError` if lockstep is broken."""
        ...

    def run_worker_loop(self, stop_event: threading.Event) -> None:
        """Block replaying the leader's step stream (worker ranks only)."""
        ...

    def shutdown(self, reason: str = "") -> None:
        """Tear the control plane down (single host: no-op)."""
        ...


class LocalCoordinator:
    """Single-host coordinator: identity pass-through to the runner.

    Args:
        runner: The engine's :class:`eSurgeRunner`.
    """

    is_leader = True
    rank = 0
    world_size = 1
    supports_overlap = True

    def __init__(self, runner) -> None:
        self._runner = runner
        self._next_step_id = 0

    def start(self) -> None:
        """No control plane on a single host."""

    def execute_sync(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Run the step synchronously on the local runner."""
        self._next_step_id += 1
        return self._runner.execute_model(scheduler_output)

    def execute_async(self, scheduler_output: SchedulerOutput) -> StepHandle:
        """Dispatch the step asynchronously on the local runner."""
        self._next_step_id += 1
        return StepHandle(
            step_id=self._next_step_id,
            runner_handle=self._runner.execute_model_async(scheduler_output),
            scheduler_output=scheduler_output,
        )

    def drain(self, handle: StepHandle) -> ModelRunnerOutput:
        """Wait for the dispatched step's host-visible output."""
        return self._runner.wait_for_execution(handle.runner_handle)

    def check_health(self) -> None:
        """Single host: always healthy."""

    def run_worker_loop(self, stop_event) -> None:
        """Single host has no worker ranks."""
        raise StepCoordinationError("LocalCoordinator has no worker loop (world_size == 1).")

    def shutdown(self, reason: str = "") -> None:
        """No control plane to tear down."""


class DistributedControllerCoordinator:
    """Adapter running steps through the legacy ZMQ ``DistributedController``.

    Bridges the coordinator seam onto the pre-existing leader/worker control
    plane (blocking per-step dispatch + verify). Overlap execution is not
    supported by that plane, so ``supports_overlap`` is ``False`` whenever
    remote workers exist.

    Args:
        runner: The engine's :class:`eSurgeRunner`.
        controller: The engine's started :class:`DistributedController`.
    """

    def __init__(self, runner, controller) -> None:
        self._runner = runner
        self._controller = controller
        self._next_step_id = 0

    @property
    def is_leader(self) -> bool:
        """Whether this rank drives scheduling."""
        return bool(self._controller.is_leader)

    @property
    def rank(self) -> int:
        """This process's control-plane rank."""
        return int(self._controller.rank)

    @property
    def world_size(self) -> int:
        """Total control-plane ranks."""
        return int(self._controller.world_size)

    @property
    def supports_overlap(self) -> bool:
        """The blocking dispatch/verify plane cannot overlap steps."""
        return not self._controller.has_remote_workers

    def start(self) -> None:
        """Start the controller's control plane (handshake with workers)."""
        self._controller.start()

    def execute_sync(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Dispatch to workers, execute locally, then verify worker acks."""
        self._next_step_id += 1
        dispatch = None
        if self._controller.has_remote_workers:
            dispatch = self._controller.dispatch_step(scheduler_output)
        model_output = self._runner.execute_model(scheduler_output)
        if dispatch is not None:
            self._controller.verify_step(dispatch, model_output)
        return model_output

    def execute_async(self, scheduler_output: SchedulerOutput) -> StepHandle:
        """Async dispatch is only legal with no remote workers."""
        if self._controller.has_remote_workers:
            raise StepCoordinationError(
                "Distributed step synchronization failure: overlap_execution=True is not supported "
                "with remote distributed workers."
            )
        self._next_step_id += 1
        return StepHandle(
            step_id=self._next_step_id,
            runner_handle=self._runner.execute_model_async(scheduler_output),
            scheduler_output=scheduler_output,
        )

    def drain(self, handle: StepHandle) -> ModelRunnerOutput:
        """Wait for a locally-dispatched step (no remote workers involved)."""
        return self._runner.wait_for_execution(handle.runner_handle)

    def check_health(self) -> None:
        """Health is checked inline by the blocking verify path."""

    def run_worker_loop(self, stop_event) -> None:
        """The legacy worker control server runs its own thread; nothing to do."""

    def shutdown(self, reason: str = "") -> None:
        """Shut the controller's sockets/threads down."""
        self._controller.shutdown()
