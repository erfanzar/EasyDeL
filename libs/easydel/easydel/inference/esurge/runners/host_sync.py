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

import typing as tp

import jax
import numpy as np

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


def device_put_equality_check_redundant(host_payloads_replicated: bool | None = None) -> bool:
    """Whether ``jax.device_put``'s cross-process equality check is redundant for host payloads.

    Args:
        host_payloads_replicated: The owning runner's replication guarantee; ``None`` falls back to the
            process-global flag set by :func:`mark_host_payloads_replicated`.

    Returns:
        ``True`` only on multi-process runs whose step stream the ZMQ plane replicates, where every process
        builds identical host payloads by construction.
    """
    replicated = _HOST_PAYLOADS_REPLICATED if host_payloads_replicated is None else host_payloads_replicated
    return jax.process_count() > 1 and bool(replicated)


def _place_identical_host_value(value: tp.Any, sharding: tp.Any) -> tp.Any:
    """Place one host value that every process holds identically, from this process's local copy.

    Device arrays, Python scalars and targets that are not a ``jax.sharding.Sharding`` (``None``, a ``Device``)
    go through ``jax.device_put`` unchanged.
    """
    if isinstance(value, jax.Array) or not hasattr(value, "dtype") or not isinstance(sharding, jax.sharding.Sharding):
        return jax.device_put(value, sharding)
    host = np.asarray(value)
    dtype = jax.dtypes.canonicalize_dtype(host.dtype)
    if host.dtype != dtype:
        host = host.astype(dtype)
    return jax.make_array_from_callback(host.shape, sharding, lambda index: np.asarray(host[index]), dtype=dtype)


def put_host_payload(values: tp.Any, shardings: tp.Any, *, host_payloads_replicated: bool | None = None) -> tp.Any:
    """``jax.device_put`` for per-step host payloads, without the redundant cross-process equality check.

    On a multi-process run, ``jax.device_put`` of a host (NumPy) array onto a sharding that spans every process
    first verifies that all processes passed the same value (``multihost_utils.assert_equal``): an all-gather
    whose result is copied back to the host. That is a cross-host collective plus a blocking device round trip
    per array, queued behind any in-flight step, so per-step input placement serializes the host with the
    devices. Under the ZMQ step-replication plane every process replays the leader's scheduler output, so the
    payloads are identical by construction, and each process places its own addressable shards from its local
    copy instead (``jax.make_array_from_callback``: same sharding, same canonical dtype, no collective). Nothing
    here detects a divergence: the plane's digest check runs only every ``distributed_verify_digest_interval``
    steps and compares sampled tokens, which a divergent input feeding a cross-host collective can leave equal.
    In every other case this is exactly ``jax.device_put``.

    Args:
        values: One host value, or a tuple/list of them. ``jax.Array`` and Python-scalar entries, and entries
            whose target is not a ``jax.sharding.Sharding`` (``None``, a ``Device``), go through
            ``jax.device_put`` unchanged.
        shardings: One sharding (or ``None``) for all values, or one per value.
        host_payloads_replicated: The owning runner's replication guarantee (see
            :func:`device_put_equality_check_redundant`).

    Returns:
        The placed value(s): a tuple or list when ``values`` is one, otherwise a single array.
    """
    if not device_put_equality_check_redundant(host_payloads_replicated):
        return jax.device_put(values, shardings)
    if isinstance(values, tuple | list):
        if not isinstance(shardings, tuple | list):
            shardings = [shardings] * len(values)
        placed = [_place_identical_host_value(v, s) for v, s in zip(values, shardings, strict=True)]
        return tuple(placed) if isinstance(values, tuple) else placed
    return _place_identical_host_value(values, shardings)
