"""Import-safe page-pool sizing shared by homogeneous and mixed caches."""

from __future__ import annotations

import threading
import time
import typing as tp
from collections.abc import Mapping
from dataclasses import dataclass


def cap_paged_cache_pages(
    num_pages: int,
    *,
    page_size: int,
    data_parallel_size: int = 1,
    max_cache_tokens: int | None = None,
) -> int:
    """Align a page pool to DP without exceeding an optional token ceiling.

    Args:
        num_pages: Available page count before alignment.
        page_size: Tokens in each page.
        data_parallel_size: Partitions of the page axis.
        max_cache_tokens: Optional positive ceiling on total tokens.

    Returns:
        Positive, DP-aligned page count within the ceiling.

    Raises:
        ValueError: If dimensions are invalid or no aligned page fits.
    """
    if page_size <= 0 or data_parallel_size <= 0:
        raise ValueError("page_size and data_parallel_size must be positive")
    pages = int(num_pages)
    if max_cache_tokens is not None and int(max_cache_tokens) > 0:
        pages = min(pages, int(max_cache_tokens) // int(page_size))
    pages = (pages // int(data_parallel_size)) * int(data_parallel_size)
    if pages <= 0:
        raise ValueError("No DP-aligned cache page fits the HBM budget and max_cache_tokens ceiling")
    return pages


@dataclass(frozen=True)
class PagedCacheBudget:
    """Shared pool sizing result, including the busiest physical stage charge."""

    num_pages: int
    page_bytes: int
    num_hidden_layers: int


def compute_paged_cache_budget(
    layer_page_bytes: Mapping[int, int],
    *,
    available_bytes: int | None,
    page_size: int,
    data_parallel_size: int = 1,
    stage_layer_groups: tp.Iterable[tp.Iterable[int]] | None = None,
    num_pages_override: int | None = None,
    max_cache_tokens: int | None = None,
) -> PagedCacheBudget:
    """Size a shared pool using the maximum physical-stage page footprint.

    Args:
        layer_page_bytes: Bytes occupied by one page in each layer, after
            resolving its actual storage layout and dtype.
        available_bytes: Effective HBM budget, required without an override.
        page_size: Tokens per page.
        data_parallel_size: Number of page-axis partitions.
        stage_layer_groups: Optional physical-stage layer ownership groups.
            Each iterable is consumed once, including generator groups.
        num_pages_override: Optional explicitly requested page count.
        max_cache_tokens: Optional positive total-token ceiling.

    Returns:
        Page count, maximum stage page footprint and busiest stage layer count.

    Raises:
        ValueError: If geometry is empty/invalid or no aligned pool fits.
    """
    if not layer_page_bytes or any(int(size) <= 0 for size in layer_page_bytes.values()):
        raise ValueError("Per-layer page footprints must be nonempty and positive")
    groups = (
        [list(layer_page_bytes)]
        if stage_layer_groups is None
        else [[int(idx) for idx in group if int(idx) in layer_page_bytes] for group in stage_layer_groups]
    )
    groups = [group for group in groups if group] or [list(layer_page_bytes)]
    page_bytes = max(sum(int(layer_page_bytes[idx]) for idx in group) for group in groups)
    if num_pages_override is None:
        if available_bytes is None:
            raise ValueError("available_bytes is required when num_pages_override is absent")
        num_pages_override = int(available_bytes) // page_bytes
    return PagedCacheBudget(
        num_pages=cap_paged_cache_pages(
            num_pages_override,
            page_size=page_size,
            data_parallel_size=data_parallel_size,
            max_cache_tokens=max_cache_tokens,
        ),
        page_bytes=page_bytes,
        num_hidden_layers=max(len(group) for group in groups),
    )


_AGREEMENT_LOCK = threading.Lock()
# (id(client), group) -> [client, {process_index: next round}]; holding the client keeps its id from being reused.
_AGREEMENT_ROUNDS: dict[tuple[int, str], list[tp.Any]] = {}


def mesh_process_indices(mesh: tp.Any) -> tuple[int, ...] | None:
    """Return the sorted JAX process indices that own devices of ``mesh``.

    Args:
        mesh: A ``jax.sharding.Mesh``, or a wrapper exposing one as ``jax_mesh`` or ``mesh``.

    Returns:
        The process indices, or ``None`` when no device array can be found on ``mesh``.
    """
    import numpy as np

    for candidate in (mesh, getattr(mesh, "jax_mesh", None), getattr(mesh, "mesh", None)):
        devices = getattr(candidate, "devices", None)
        if devices is None or callable(devices):
            continue
        try:
            flat = np.asarray(devices, dtype=object).reshape(-1)
            return tuple(sorted({int(device.process_index) for device in flat}))
        except (AttributeError, TypeError, ValueError):
            continue
    return None


def min_across_processes(
    value: int,
    *,
    what: str = "value",
    participants: tp.Iterable[int] | None = None,
    timeout_s: float = 1800.0,
) -> int:
    """Return the minimum of ``value`` over the participating JAX processes.

    Multi-controller runs size shared buffers such as paged KV pools from each host's own free HBM. After a real
    checkpoint load the hosts' free memory (and its largest free block) differs slightly, so independently derived
    page counts diverge, each host compiles a differently shaped step program, and the first collective step hangs
    or halts the TPU. Agreeing on the minimum keeps the cache geometry identical on every host.

    The values travel through the ``jax.distributed`` key-value store, so no device program runs. Every
    participant must call this the same number of times in the same order, which holds for SPMD setup code: the
    n-th call of each participant forms one round.

    Args:
        value: This process's value.
        what: Description used in the log line.
        participants: Process indices that share the buffer (for example :func:`mesh_process_indices` of its
            mesh); ``None`` means every process. A process outside the set, or a set of one, keeps ``value``.
        timeout_s: Seconds to wait for the other participants' values.

    Returns:
        The minimum over the participants (``value`` itself on a single process).

    Raises:
        RuntimeError: When a participant does not publish its value within ``timeout_s`` (chained to the key-value
            store's error): the processes did not call this the same number of times in the same order.
    """
    import jax

    count = int(jax.process_count())
    if count <= 1:
        return int(value)
    index = int(jax.process_index())
    peers = tuple(range(count)) if participants is None else tuple(sorted({int(p) for p in participants}))
    if len(peers) <= 1 or index not in peers:
        return int(value)
    from eformer.loggings import get_logger

    logger = get_logger(__name__)
    try:
        from jax._src.distributed import global_state
    except ImportError:  # private JAX module: without it no process can reach the key-value store
        global_state = None
    client = getattr(global_state, "client", None)
    if client is None:
        logger.warning(
            "%s: no jax.distributed client, so processes %s cannot agree; keeping this process's %d",
            what,
            list(peers),
            int(value),
        )
        return int(value)
    group = "all" if peers == tuple(range(count)) else "-".join(str(peer) for peer in peers)
    with _AGREEMENT_LOCK:
        rounds = _AGREEMENT_ROUNDS.setdefault((id(client), group), [client, {}])[1]
        round_id = rounds.get(index, 0)
        rounds[index] = round_id + 1
    prefix = f"easydel/min_across_processes/{group}/{round_id}"
    client.key_value_set(f"{prefix}/{index}", str(int(value)), allow_overwrite=True)
    deadline = time.monotonic() + float(timeout_s)
    values = []
    for peer in peers:
        try:
            raw = client.blocking_key_value_get(f"{prefix}/{peer}", max(1, int((deadline - time.monotonic()) * 1000)))
        except Exception as exc:
            raise RuntimeError(
                f"{what}: process {index} waited {float(timeout_s):.0f}s for process {peer} in agreement round "
                f"{round_id} of processes {list(peers)}. Every process that shares the mesh must build its cache "
                "configuration the same number of times and in the same order; one that skipped or repeated a "
                "build leaves the others waiting."
            ) from exc
        values.append(int(raw))
    agreed = min(values)
    logger.info("%s per process %s -> using %d", what, dict(zip(peers, values, strict=True)), agreed)
    return agreed
