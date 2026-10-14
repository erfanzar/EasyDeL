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

"""Shared platform and head-parallel ``shard_map`` plumbing for the compressed-window adapters.

The Mosaic custom call behind the TPU compressed-window kernels cannot be auto-partitioned by GSPMD on a multi-device
mesh, so the adapters wrap it in ``shard_map``. Queries/outputs are split over the resolved ``HEAD`` axis, every
batch-indexed operand over the resolved ``BATCH`` axis, and the single shared KV head is replicated over heads. The
XLA reference path is left to GSPMD (correct on any mesh).
"""

from __future__ import annotations

import typing as tp

import jax
from jax.sharding import PartitionSpec as Ps
from spectrax import common_types

_DISABLED_PLATFORMS: tp.Final = ("auto", "off", "none", "disabled")


def resolve_platform(platform: str | None, default: str) -> str | None:
    """Return the ejkernel platform override: the per-call value, else the adapter default; ``None`` means auto."""
    chosen = platform if platform is not None else default
    if chosen is None or str(chosen).lower() in _DISABLED_PLATFORMS:
        return None
    return str(chosen).lower()


def _axis_names(entry: tp.Any) -> set[str]:
    """Mesh-axis names referenced by one PartitionSpec entry."""
    if entry is None:
        return set()
    if isinstance(entry, tuple | list):
        return {str(name) for name in entry}
    return {str(entry)}


def head_parallel_shard_map(
    metadata: tp.Any,
    *,
    mode: str,
    platform: str | None,
    batch: int,
    num_heads: int,
    token_valid: bool = False,
    entry_topk: bool = False,
) -> dict[str, tp.Any]:
    """Return ``mesh`` / ``in_specs`` / ``out_specs`` kwargs for the ejkernel op, or ``{}`` when not needed.

    ``shard_map`` is only built when the op will resolve to the TPU Mosaic kernel (TPU backend and the platform is
    not forced to ``"xla"``) on a mesh with more than one device.

    Args:
        metadata: The adapter's :class:`~easydel.operations.OperationMetadata`.
        mode: Runtime mode used by the sharding resolver (``MODE_TRAIN`` or ``MODE_DECODE``).
        platform: Resolved platform override (``None`` = auto).
        batch: Global batch size (used to sanitize an indivisible batch spec).
        num_heads: Global query-head count (used to sanitize an indivisible head spec).
        token_valid: Append a spec for a per-key ``[batch, token_kv_len]`` validity operand.
        entry_topk: Append a spec for a ``[batch, q_len, k]`` selected-entry id operand (after ``token_valid``).

    Returns:
        Keyword arguments for ``(query, kv, bias, softmax_aux[, token_valid][, entry_topk])`` ejkernel calls.
    """
    if platform == "xla" or jax.default_backend() != "tpu":
        return {}
    mesh = metadata.mesh
    jax_mesh = getattr(mesh, "jax_mesh", mesh)
    if jax_mesh is None or getattr(jax_mesh, "size", 1) <= 1:
        return {}
    resolver = metadata.runtime_sharding_resolver.with_mesh(mesh)
    head = resolver.resolve(axes=[common_types.HEAD], mode=mode, shape=(num_heads,))[0]
    rows = resolver.resolve(axes=[common_types.BATCH], mode=mode, shape=(batch,))[0]
    if _axis_names(rows) & _axis_names(head):
        rows = None
    in_specs = (
        Ps(rows, head, None, None),  # query: batch + heads sharded
        Ps(rows, None, None),  # kv: shared head replicated over heads
        Ps(rows, None, None),  # bias: head-broadcast
        Ps(head),  # sinks: per head
    )
    if token_valid:
        in_specs = (*in_specs, Ps(rows, None))  # per-key token validity
    if entry_topk:
        in_specs = (*in_specs, Ps(rows, None, None))  # selected entry ids
    return dict(mesh=jax_mesh, in_specs=in_specs, out_specs=Ps(rows, head, None, None))
