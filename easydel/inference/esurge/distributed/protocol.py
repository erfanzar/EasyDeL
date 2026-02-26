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

"""Wire protocol constants and hashing utilities for eSurge distributed control-plane.

This module defines the command vocabulary and serialization helpers shared by the
leader (:class:`~.leader_client.WorkerRpcClient`) and worker
(:class:`~.worker_server.WorkerControlServer`) sides of the distributed control-plane.

Constants:
    CMD_HELLO: Handshake command sent during initial connection.
    CMD_HEALTH: Health-check command for liveness probing.
    CMD_STEP: Command that triggers a single lockstep inference step on a worker.
    CMD_SHUTDOWN: Graceful shutdown command.
    STATUS_OK: Response status indicating success.
    STATUS_ERROR: Response status indicating failure.

Functions:
    make_config_fingerprint: Generates a deterministic SHA-256 digest of an engine
        configuration mapping so the leader can verify all workers run identical configs.
    compute_sampled_digest: Hashes request IDs together with their sampled token IDs
        to verify that every host produced the same sampling results after a step.
"""

from __future__ import annotations

import hashlib
import json
import typing as tp
from collections.abc import Mapping

CMD_HELLO = "hello"
CMD_HEALTH = "health"
CMD_STEP = "step"
CMD_SHUTDOWN = "shutdown"

STATUS_OK = "ok"
STATUS_ERROR = "error"


def _canonicalize(value: tp.Any) -> tp.Any:
    """Convert an arbitrary Python value into a stable, JSON-serializable structure.

    The function recursively walks *value* and normalises every leaf so that
    ``json.dumps`` with ``sort_keys=True`` always produces the same byte
    string for semantically equal inputs.  This is required by
    :func:`_hash_payload` to generate reproducible SHA-256 digests.

    Handled types:
        * ``None``, ``str``, ``bool``, ``int``, ``float`` — returned as-is.
        * ``bytes`` — decoded to UTF-8 (with ``errors="replace"``).
        * ``dict`` — keys are stringified, values recursed, items sorted by key.
        * ``list`` / ``tuple`` — elements recursed in order.
        * ``set`` — sorted by ``repr`` then recursed.
        * NumPy / JAX scalars — converted via ``.tolist()`` or ``.item()``.
        * Anything else — falls back to ``repr(value)``.

    Args:
        value: The value to canonicalize.

    Returns:
        A JSON-serializable representation of *value*.
    """

    if value is None or isinstance(value, (str, bool, int, float)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}

    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]

    if isinstance(value, set):
        return [_canonicalize(v) for v in sorted(value, key=repr)]

    # numpy / jax scalar compatibility without importing heavy deps.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _canonicalize(tolist())
        except Exception:
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _canonicalize(item())
        except Exception:
            pass

    return repr(value)


def _hash_payload(payload: tp.Any) -> str:
    """Return the SHA-256 hex digest of *payload* after canonicalization.

    The payload is first canonicalized via :func:`_canonicalize`, then
    serialized to a compact JSON byte string with sorted keys, and finally
    hashed with SHA-256.

    Args:
        payload: Arbitrary data to hash.

    Returns:
        A 64-character lowercase hexadecimal SHA-256 digest string.
    """
    encoded = json.dumps(
        _canonicalize(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_config_fingerprint(config: Mapping[str, tp.Any]) -> str:
    """Build a stable SHA-256 fingerprint of an engine configuration mapping.

    The leader sends its fingerprint during the handshake and compares it
    against each worker's fingerprint to guarantee that every host in the
    cluster is running an identical engine configuration.

    Args:
        config: Engine configuration as a string-keyed mapping (e.g. the
            serialized :class:`~easydel.inference.esurge.config.Config`).

    Returns:
        A 64-character hexadecimal SHA-256 digest of the canonicalized config.
    """

    return _hash_payload(dict(config))


def compute_sampled_digest(req_ids: list[str], sampled_token_ids: list[list[int]]) -> str:
    """Hash request IDs and sampled token IDs to verify lockstep sampling consistency.

    After every inference step the leader computes this digest from its own
    outputs and compares it to the digests reported by each worker.  A mismatch
    signals that the hosts have diverged (e.g. due to non-deterministic sampling
    or mismatched batches).

    Args:
        req_ids: Ordered list of request identifiers included in the step.
        sampled_token_ids: Parallel list of sampled token-ID sequences, one
            per request, in the same order as *req_ids*.

    Returns:
        A 64-character hexadecimal SHA-256 digest string.
    """

    req_ids_norm = [str(rid) for rid in req_ids]
    tokens_norm: list[list[int]] = []
    for row in sampled_token_ids:
        tokens_norm.append([int(tok) for tok in row])

    return _hash_payload({"req_ids": req_ids_norm, "sampled_token_ids": tokens_norm})
