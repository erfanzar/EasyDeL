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

"""Replay-free rollback of speculative verify windows on the compressed-window caches (DeepSeek-V4 / V4.1).

A speculative verify forward advances every per-stream field of a row (ring slots, ``cache_position``, open-group
compressor buffers, two-series overlaps) through all ``1 + k`` window tokens. When only a prefix is accepted,
:func:`commit_speculative_window` rebuilds those fields as if only the prefix had been consumed, from the pre-verify
snapshot plus the traces the verify forward recorded (the raw projection row of every consumed position and, for
two-series streams, the overlap of every window that closed), instead of restoring the snapshot and replaying the
prefix through the model. Compressed entries and index keys stay live: every reader masks them by reach, so an entry
emitted by a rejected position is invisible until the real token at that position rewrites it.
"""

from __future__ import annotations

import functools
import typing as tp

import jax
from jax import numpy as jnp
from jaxtyping import Array

from .cache import CompressedWindowCache, CompressedWindowCacheView
from .shared import SharedCompressedWindowCache, SharedCompressedWindowCacheView

#: Per view family: ``((open-group state field, per-position trace field), ...)`` and
#: ``((two-series overlap field, closed-window trace field), ...)``.
_StreamSpec = tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]

_SHARED_SPEC: _StreamSpec = (
    (
        ("compressor_kv_state", "compressor_trace_kv"),
        ("compressor_score_state", "compressor_trace_score"),
    ),
    (),
)
_V4_SPEC: _StreamSpec = (
    (
        ("compressor_buffer_kv", "compressor_trace_kv"),
        ("compressor_buffer_gate", "compressor_trace_gate"),
        ("indexer_buffer_kv", "indexer_trace_kv"),
        ("indexer_buffer_gate", "indexer_trace_gate"),
    ),
    (
        ("compressor_overlap_kv", "compressor_trace_overlap_kv"),
        ("compressor_overlap_gate", "compressor_trace_overlap_gate"),
        ("indexer_overlap_kv", "indexer_trace_overlap_kv"),
        ("indexer_overlap_gate", "indexer_trace_overlap_gate"),
    ),
)


def _view_spec(view: tp.Any) -> _StreamSpec | None:
    """Stream spec of a compressed-window view (``None`` for any other view type)."""
    if isinstance(view, SharedCompressedWindowCacheView):
        return _SHARED_SPEC
    if isinstance(view, CompressedWindowCacheView):
        return _V4_SPEC
    return None


def speculative_commit_limit(cache: tp.Any) -> int:
    """Longest per-row verify window :func:`commit_speculative_window` rolls back exactly.

    The bound is the ring width and every allocated trace length: within it no trace slot is written twice by one
    window and no ring slot written by a rejected position is also needed from the window.

    Returns:
        The limit, or ``0`` when the cache is not a compressed-window cache or some stream has no trace (the caller
        must replay instead).
    """
    if not isinstance(cache, CompressedWindowCache | SharedCompressedWindowCache):
        return 0
    limit: int | None = None
    for view in cache.views:
        if view is None:
            continue
        spec = _view_spec(view)
        if spec is None:
            return 0
        lengths = [] if view.window_kv is None else [int(view.window_kv.shape[1])]
        streams, overlaps = spec
        for state, trace in (*streams, *overlaps):
            if getattr(view, state) is None:
                continue
            traced = getattr(view, trace)
            if traced is None:
                return 0
            lengths.append(int(traced.shape[1]))
        for length in lengths:
            limit = length if limit is None else min(limit, length)
    return 0 if limit is None else limit


def _commit_view(
    live: dict[str, Array | None], pre: dict[str, Array | None], committed: Array, spec: _StreamSpec
) -> dict[str, Array]:
    """Per-stream fields of one view after keeping only the first ``committed`` tokens of each row's window.

    A row's window starts at its pre-window ``cache_position`` and spans ``count = live - pre`` positions; rows with
    ``committed >= count`` (or no window) keep ``live``. With ``frontier = start + committed``:

    - ring slots written by rejected positions ``[frontier, start + count)`` take their pre-window values;
    - open-group slot ``j`` takes the row of the latest position ``q < frontier`` with ``q = j (mod ratio)``: from
      the trace (slot ``q % L``) when ``q >= start``, else from the pre-window state;
    - a two-series overlap is the Ca half of the last window closed before ``frontier``: from the closed-window trace
      when that window closed inside the window, else the pre-window overlap.
    """
    streams, overlaps = spec
    start = pre["cache_position"].astype(jnp.int32)
    count = live["cache_position"].astype(jnp.int32) - start
    committed = jnp.clip(committed.astype(jnp.int32), 0, jnp.maximum(count, 0))
    rollback = committed < count
    frontier = start + committed
    rows = jnp.arange(start.shape[0])
    out = {"cache_position": jnp.where(rollback, frontier, live["cache_position"]).astype(live["cache_position"].dtype)}
    ring = live["window_kv"]
    if ring is not None:
        width = ring.shape[1]
        dist = (jnp.arange(width, dtype=jnp.int32)[None, :] - frontier[:, None]) % width
        restore = rollback[:, None] & (dist < (count - committed)[:, None])
        restore = restore.reshape(*restore.shape, *([1] * (ring.ndim - 2)))
        out["window_kv"] = jnp.where(restore, pre["window_kv"], ring)
    keep = rollback[:, None, None]
    for state, trace in streams:
        if live[state] is None:
            continue
        ratio = live[state].shape[1]
        length = live[trace].shape[1]
        member = (frontier // ratio * ratio)[:, None] + jnp.arange(ratio, dtype=jnp.int32)[None, :]
        member = jnp.where(member < frontier[:, None], member, member - ratio)  # latest position < frontier per slot
        from_trace = (member >= start[:, None])[..., None]
        traced = jnp.take_along_axis(live[trace], jnp.mod(member, length)[..., None], axis=1).astype(live[state].dtype)
        out[state] = jnp.where(keep, jnp.where(from_trace, traced, pre[state]), live[state])
    for state, trace in overlaps:
        if live[state] is None:
            continue
        rate = live[state].shape[1]
        windows = live[trace].shape[1]
        n_closed = frontier // rate  # windows closed by positions < frontier
        last = n_closed - 1
        from_trace = ((last >= 0) & (n_closed * rate - 1 >= start))[:, None, None]
        traced = live[trace][rows, jnp.mod(last, windows)].astype(live[state].dtype)
        out[state] = jnp.where(keep, jnp.where(from_trace, traced, pre[state]), live[state])
    return out


def _commit_impl(live_states, pre_states, committed, specs):
    """Apply :func:`_commit_view` to every view (``None`` placeholders pass through)."""
    return [
        None if spec is None else _commit_view(live, pre, committed, spec)
        for live, pre, spec in zip(live_states, pre_states, specs, strict=True)
    ]


_commit_compiled = jax.jit(_commit_impl, static_argnums=(3,))


@functools.lru_cache(maxsize=32)
def _commit_pinned(specs: tuple, flat_shardings: tuple, treedef: tp.Any) -> tp.Callable:
    """``_commit_impl`` compiled with each output pinned to the sharding of the live leaf it replaces."""
    del specs  # part of the cache key only (the output structure depends on it)
    out = jax.tree_util.tree_unflatten(treedef, list(flat_shardings))
    return jax.jit(_commit_impl, static_argnums=(3,), out_shardings=out)


def _output_names(view: tp.Any, spec: _StreamSpec) -> list[str]:
    """Keys of the dict :func:`_commit_view` returns for ``view`` (same presence rules)."""
    names = ["cache_position"]
    if view.window_kv is not None:
        names.append("window_kv")
    names.extend(state for state, _ in (*spec[0], *spec[1]) if getattr(view, state) is not None)
    return names


def _fields(view: tp.Any, spec: _StreamSpec, *, with_traces: bool) -> dict[str, Array | None]:
    """The arrays of ``view`` a commit reads (traces only from the live cache)."""
    names = ["window_kv", "cache_position"]
    for state, trace in (*spec[0], *spec[1]):
        names.append(state)
        if with_traces:
            names.append(trace)
    return {name: getattr(view, name) for name in names}


def commit_speculative_window(live: tp.Any, pre: tp.Any, committed: tp.Any) -> tp.Any:
    """Roll a verified speculative window back to its accepted prefix without re-running the model.

    Only the small per-stream fields are rebuilt, in one compiled call; compressed entries and index keys stay live.
    Every rolled-back row's window must fit :func:`speculative_commit_limit`.

    Args:
        live: :class:`CompressedWindowCache` or :class:`SharedCompressedWindowCache` after the verify forward, with
            traces allocated (``spec_trace_len > 0``) on every compressor stream.
        pre: The same cache before the verify forward (``window_kv``, ``cache_position`` and the open-group / overlap
            state are read).
        committed: ``[batch]`` tokens to keep per row, counted from the row's pre-window ``cache_position``. Rows
            whose window (``live - pre`` positions) is not longer than ``committed`` are left unchanged.

    Returns:
        The cache with every rolled-back row exactly as if only its committed tokens had been consumed.

    Raises:
        TypeError: If a view is not a compressed-window view.
        ValueError: If a compressor stream has no trace.
    """
    active, specs, live_states, pre_states, out_shardings = [], [], [], [], []
    for index, (lv, pv) in enumerate(zip(live.views, pre.views, strict=True)):
        if lv is None:
            continue
        spec = _view_spec(lv)
        if spec is None:
            raise TypeError(f"commit_speculative_window does not support {type(lv).__name__}")
        for state, trace in (*spec[0], *spec[1]):
            if getattr(lv, state) is not None and getattr(lv, trace) is None:
                raise ValueError(f"commit_speculative_window needs spec_trace_len > 0 ({trace} is missing).")
        active.append(index)
        specs.append(spec)
        live_states.append(_fields(lv, spec, with_traces=True))
        pre_states.append(_fields(pv, spec, with_traces=False))
        out_shardings.append({name: getattr(getattr(lv, name), "sharding", None) for name in _output_names(lv, spec)})
    specs_key = tuple(specs)
    flat, treedef = jax.tree_util.tree_flatten(out_shardings, is_leaf=lambda x: x is None)
    # Pin every output to the sharding of the live leaf it replaces: an unpinned jit may pick another layout for the
    # [batch] / ring outputs, and re-placing each mismatched leaf is one dispatch per leaf (hundreds on 43 layers).
    if flat and all(s is not None for s in flat):
        fn = _commit_pinned(specs_key, tuple(flat), treedef)
    else:
        fn = _commit_compiled
    new_states = fn(live_states, pre_states, jnp.asarray(committed, dtype=jnp.int32), specs_key)
    views = list(live.views)
    for index, new in zip(active, new_states, strict=True):
        lv = views[index]
        replaced = {}
        for name, value in new.items():
            sharding = getattr(getattr(lv, name), "sharding", None)
            replaced[name] = value if sharding is None or value.sharding == sharding else jax.device_put(value, sharding)
        views[index] = lv.replace(**replaced)
    return type(live)(views=views)


__all__ = ("commit_speculative_window", "speculative_commit_limit")
