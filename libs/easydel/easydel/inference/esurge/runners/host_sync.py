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

"""Host-payload synchronization policy for multi-process runtimes.

Two multi-host patterns feed the eSurge runner, with different host-payload
guarantees:

- **Replicated drivers** (the default; e.g. a trainer's rollout loop calling
  the engine identically on every host): per-host scheduler state can drift
  in small nondeterministic ways, so before every ``device_put`` with
  replicated sharding the host payload is force-broadcast from process 0
  (``multihost_utils.broadcast_one_to_all``). Those broadcasts are per-step
  global device collectives — correct, but not free.

- **The ZMQ step-replication plane** (``coordination="zmq"``): every host
  executes the leader's literal ``SchedulerOutput`` stream in order, so host
  payloads are identical by construction and the broadcasts are redundant
  work. Preparers and sampler runtimes accept a runner-owned
  ``host_payloads_replicated`` guarantee to skip them.

Explicit runner policies are isolated across sequential and concurrent engines.
The process-global :func:`mark_host_payloads_replicated` API remains available
for legacy external callers that omit an explicit policy; new engine wiring
must pass its own guarantee instead of relying on that compatibility flag.

NOTE: skipping the broadcasts under ``coordination="zmq"`` is validated on
CPU (2-process integration test); the throughput benefit on a real TPU pod
is expected but not yet measured on hardware.
"""

from __future__ import annotations

import jax

_HOST_PAYLOADS_REPLICATED = False


def mark_host_payloads_replicated(enabled: bool = True) -> None:
    """Declare that an external plane replicates this process's step stream.

    Called by the engine when the ZMQ step-coordination plane is active.

    Args:
        enabled: Whether host payloads are replicated by construction.
    """
    global _HOST_PAYLOADS_REPLICATED
    _HOST_PAYLOADS_REPLICATED = bool(enabled)


def host_payload_broadcast_needed(host_payloads_replicated: bool | None = None) -> bool:
    """Whether host payloads must be force-broadcast from process 0.

    Args:
        host_payloads_replicated: The owning runner's replication guarantee.
            Explicit ``False`` is isolated from the legacy process flag.
            ``None`` falls back to the process-global flag set by
            :func:`mark_host_payloads_replicated`; engine-owned preparers and
            samplers always pass their own policy.

    Returns:
        ``True`` on multi-process replicated drivers, otherwise ``False``.
    """
    replicated = _HOST_PAYLOADS_REPLICATED if host_payloads_replicated is None else host_payloads_replicated
    return jax.process_count() > 1 and not replicated
