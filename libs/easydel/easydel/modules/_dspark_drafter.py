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

"""Family-agnostic eSurge drafter for DSpark-style whole-block draft models (DeepSeek-V4 / V4.1).

A DSpark draft model reads the backbone's *main* hidden states (concatenated target-layer stream means, width
``len(dspark_target_layer_ids) * D``), keeps one sliding-window K/V ring per stage, and drafts a whole block greedily
in one pass. The drafter here owns the ring pool, the per-row position bookkeeping and the jitted programs; the draft
model supplies the architecture. The draft model must expose:

- ``config`` with ``hidden_size``, ``sliding_window``, ``head_dim`` and ``dspark_target_layer_ids``;
- ``block_size`` and ``num_stages``;
- ``_main_x(main_hidden)`` (stage-0 ``main_norm(main_proj(.))``) and ``stages[i].self_attn.main_kv(main_x, positions)``;
- ``draft(token_ids, main_hidden_step, state, embed, lm_head) -> (ids, logits, confidence, state)`` taking and
  returning a :class:`DSparkState`.
"""

from __future__ import annotations

import contextlib
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from jaxtyping import Array


class DSparkState(tp.NamedTuple):
    """Draft-model state: one sliding-window ring per stage and the next main position.

    Attributes:
        window_kv: Per stage ``[B, sliding_window, head_dim]`` post-RoPE main K/V ring (token ``p`` at slot ``p % W``).
        position: ``int32 [B]`` position of the main hidden state the next ``draft`` call gets.
    """

    window_kv: tuple[Array, ...]
    position: Array


def pad_pow2(count: int, minimum: int = 8) -> int:
    """Smallest power of two ``>= max(count, minimum)`` (bounds the number of compiled append programs)."""
    size = minimum
    while size < count:
        size *= 2
    return size


class DSparkDrafter:
    """eSurge drafter that runs a DSpark draft model over the backbone's main hidden states.

    Contract with the eSurge runner (the flags are read through ``getattr`` by the runner / strategy):

    - ``needs_main_hidden``: the model step returns ``concat([last_hidden, main_hidden], -1)`` rows (width
      ``(1 + T) * D``); the drafter reads the ``main_hidden`` part ``[D:]`` of every row.
    - ``whole_block_draft``: one :meth:`draft_window` call proposes the whole window (greedy), instead of the
      per-token loop of the generic strategy.
    - ``rolls_back_cache``: the target cache (compressed-window caches) is not paged, so the runner snapshots it before
      a speculative verify window and, when any draft is rejected, restores it and replays the accepted prefix. The
      DSpark rings are NOT rolled back: only committed positions are ever written into them.
    - ``supports_compressed_window_cache``: lifts the runner's compressed-window / drafter incompatibility guard.
    - ``supports_prefix_draft``: makes the runner return full hidden rows for prompt windows (the rings are seeded
      from every prompt position).

    Ring bookkeeping (host side, per drafter row ``r``): ``_through[r]`` is the highest main position whose K/V is in
    the rings of row ``r``. Window hidden rows reach the drafter through :meth:`observe_window` (committed real tokens,
    written eagerly) and :meth:`draft_window` (accepted draft positions, written lazily once the verify outcome is
    known, followed by the seed position itself). A row whose positions are not contiguous (missing hidden rows,
    chunked prefill) is marked broken and simply stops drafting: output stays correct, speculation just turns off.

    Only greedy requests are drafted for (the draft head is greedy).
    """

    supports_prefix_draft: bool = True
    supports_return_full_log_probs: bool = False
    requires_target_kv_cache: bool = False
    supports_compressed_window_cache: bool = True
    needs_main_hidden: bool = True
    whole_block_draft: bool = True
    rolls_back_cache: bool = True

    def __init__(
        self,
        *,
        target_model: tp.Any,
        dspark: tp.Any,
        num_draft_tokens: int | None = None,
        dtype: jnp.dtype | None = None,
    ):
        """Create the drafter.

        Args:
            target_model: The causal LM whose embedding and LM head the draft model shares.
            dspark: The DSpark draft model (built under ``config.mesh``; see the module docstring for its interface).
            num_draft_tokens: Drafted tokens per verify window; clamped to ``dspark.block_size``. ``None`` uses the
                full block.
            dtype: Ring storage dtype; defaults to the target model's ``dtype``.
        """
        block = int(dspark.block_size)
        requested = block if num_draft_tokens is None else int(num_draft_tokens)
        self.target_model = target_model
        self.dspark = dspark
        self.config = dspark.config
        self.block_size = block
        self.num_draft_tokens = max(1, min(requested, block))
        self.hidden_size = int(self.config.hidden_size)
        self.window = int(self.config.sliding_window)
        self.dtype = dtype if dtype is not None else getattr(target_model, "dtype", jnp.float32)
        self._capacity = 0
        self._rings: tuple[jax.Array, ...] | None = None
        self._owner: dict[int, str] = {}
        self._through: dict[int, int] = {}
        self._last_window: dict[int, tuple[jax.Array, int, int, int]] = {}
        self._broken: set[int] = set()
        self._programs: tuple[tp.Any, ...] | None = None

    def set_max_length(self, max_length: int) -> None:
        """Accept the runner's max model length (rings are window-sized, so nothing to size)."""
        del max_length

    def reset_request(self, request_id: str | None = None) -> None:
        """No-op: per-request state is reset when a row changes owner (see :meth:`observe_window`)."""
        del request_id

    def reset(self, batch_size: int) -> None:
        """Re-allocate zeroed rings for ``batch_size`` rows and forget every row owner."""
        self._alloc(max(1, int(batch_size)))

    def ensure_batch_size(self, batch_size: int) -> None:
        """Allocate the ring pool on first use (no-op when already large enough)."""
        if self._rings is None or self._capacity < int(batch_size):
            self._alloc(int(batch_size))

    def _alloc(self, rows: int) -> None:
        head_dim = int(self.config.head_dim)
        replicated = self._replicated_sharding()
        rings = tuple(jnp.zeros((rows, self.window, head_dim), self.dtype) for _ in range(int(self.dspark.num_stages)))
        self._rings = rings if replicated is None else tuple(jax.device_put(ring, replicated) for ring in rings)
        self._capacity = rows
        self._owner.clear()
        self._through.clear()
        self._last_window.clear()
        self._broken.clear()

    def _reset_row(self, row: int, req_id: str) -> None:
        """Hand ``row`` to ``req_id``: zero its rings and restart its position bookkeeping."""
        self._owner[row] = req_id
        self._through[row] = -1
        self._last_window.pop(row, None)
        self._broken.discard(row)
        self._rings = tuple(ring.at[row].set(0) for ring in self._rings)

    @contextlib.contextmanager
    def _context(self) -> tp.Iterator[None]:
        from easydel.utils import set_inference_mode

        with self.target_model.mesh, set_inference_mode():
            yield

    def _ensure_programs(self) -> None:
        """Export both models once and build the jitted append / draft programs."""
        if self._programs is not None:
            return
        dgraphdef, dstate = spx.export(self.dspark)
        tgraphdef, tstate = spx.export(self.target_model)
        hidden_size, window = self.hidden_size, self.window
        # Flatten both states once and hand the programs plain leaf tuples. Passing the spectrax ``State`` objects
        # themselves re-ran the keyed State flatten (path strings, key sorts) and the slow argument path on every
        # call: measured on DeepSeek-V4.1 cc=1, ~46 ms of host time per draft step against 2.6 ms of device time.
        dleaves, dtree = jax.tree_util.tree_flatten(dstate)
        tleaves, ttree = jax.tree_util.tree_flatten(tstate)

        def bind_draft(leaves):
            return spx.bind(dgraphdef, jax.tree_util.tree_unflatten(dtree, list(leaves)))

        def bind_target(leaves):
            return spx.bind(tgraphdef, jax.tree_util.tree_unflatten(ttree, list(leaves)))

        def gather_main(hidden, gather):
            # The only program keyed on the step's token bucket: everything after it sees ``[P, T * D]`` rows.
            return hidden[gather][:, hidden_size:]

        def append(dstate, rings, main_flat, positions, rows):
            dm = bind_draft(dstate)
            main_x = dm._main_x(main_flat[None])
            slots = positions % window
            out = []
            for stage, ring in zip(dm.stages, rings, strict=True):
                kv = stage.self_attn.main_kv(main_x, positions[None])[0]
                out.append(ring.at[rows, slots].set(kv.astype(ring.dtype), mode="drop"))
            return tuple(out)

        def step(dstate, tstate, rings, row, token, main_step, position):
            dm = bind_draft(dstate)
            tm = bind_target(tstate)
            state = DSparkState(tuple(ring[row][None] for ring in rings), position[None])
            ids, _logits, _confidence, new_state = dm.draft(
                token[None], main_step[None, None, :], state, tm.compute_embedding, tm.compute_lm_logits
            )
            new_rings = tuple(
                ring.at[row].set(win[0].astype(ring.dtype)) for ring, win in zip(rings, new_state.window_kv, strict=True)
            )
            return ids[0], new_rings

        def step_batch(dstate, tstate, rings, main, rows, tokens, positions):
            # Many rows in one draft pass. Padding rows carry ``rows == capacity``: they read row ``capacity - 1``
            # (clipped) and their ring writes are dropped.
            dm = bind_draft(dstate)
            tm = bind_target(tstate)
            capacity = rings[0].shape[0]
            safe_rows = jnp.minimum(rows, capacity - 1)
            state = DSparkState(tuple(ring[safe_rows] for ring in rings), positions)
            ids, _logits, _confidence, new_state = dm.draft(
                tokens, main[:, None, :], state, tm.compute_embedding, tm.compute_lm_logits
            )
            new_rings = tuple(
                ring.at[rows].set(win.astype(ring.dtype), mode="drop")
                for ring, win in zip(rings, new_state.window_kv, strict=True)
            )
            return ids, new_rings

        replicated = self._replicated_sharding()

        def pinned(fn):
            # Fixed (replicated) outputs keep the rings' sharding identical across programs, so every program has
            # one cache entry per shape instead of one per producer of its inputs.
            return jax.jit(fn) if replicated is None else jax.jit(fn, out_shardings=replicated)

        self._programs = (
            tuple(dleaves),
            tuple(tleaves),
            pinned(append),
            pinned(step),
            pinned(step_batch),
            pinned(gather_main),
        )

    def _replicated_sharding(self) -> jax.sharding.NamedSharding | None:
        """Replicated sharding on the target model's SPMD mesh, or ``None`` (no mesh / pipeline mesh)."""
        mesh = getattr(self.target_model, "mesh", None)
        if bool(getattr(mesh, "is_mpmd", False)):
            return None
        mesh = getattr(mesh, "jax_mesh", mesh)
        if not isinstance(mesh, jax.sharding.Mesh):
            return None
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def precompile(self, *, max_num_reqs: int, max_tokens: int, hidden_dtype: jnp.dtype | None = None) -> None:
        """Compile the ring-append and batched-draft programs for every padded row count serving can produce.

        Warm-up calls target the out-of-range row ``capacity`` only, so every ring write is dropped and the drafter
        state is unchanged. Only the per-token-bucket row gather stays lazy (it holds no draft-model work).

        Args:
            max_num_reqs: Request pool size (drafter rows).
            max_tokens: Largest per-step token bucket.
            hidden_dtype: Dtype of the model step's hidden rows; defaults to the ring dtype.
        """
        self.ensure_batch_size(max(1, int(max_num_reqs)))
        self._ensure_programs()
        dstate, tstate, append, _step, step_batch, _gather = self._programs
        capacity = self._capacity
        width = len(self.config.dspark_target_layer_ids) * self.hidden_size
        dtype = hidden_dtype if hidden_dtype is not None else self.dtype
        replicated = self._replicated_sharding()

        def main_rows(count: int) -> jax.Array:
            rows = jnp.zeros((count, width), dtype)
            return rows if replicated is None else jax.device_put(rows, replicated)

        # Appends write at most ``window`` positions per row per call (the step's committed tokens or a row's missing
        # accepted drafts), so they never exceed ``capacity * window``.
        append_limit = pad_pow2(min(capacity * self.window, max(int(max_tokens), capacity * (self.block_size + 1))))
        with self._context():
            size = pad_pow2(1)
            while size <= append_limit:
                positions = jnp.zeros((size,), jnp.int32)
                drop = jnp.full((size,), capacity, jnp.int32)
                self._rings = append(dstate, self._rings, main_rows(size), positions, drop)
                size *= 2
            size = pad_pow2(1)
            while size <= pad_pow2(capacity):
                zeros = jnp.zeros((size,), jnp.int32)
                drop = jnp.full((size,), capacity, jnp.int32)
                ids, self._rings = step_batch(dstate, tstate, self._rings, main_rows(size), drop, zeros, zeros)
                size *= 2
            jax.block_until_ready((ids, self._rings))

    def _write(self, hidden: jax.Array, gather: list[int], rows: list[int], positions: list[int]) -> None:
        """Write main K/V of ``hidden[gather]`` (positions ``positions``, rows ``rows``) into the rings."""
        count = len(gather)
        if count == 0:
            return
        self._ensure_programs()
        dstate, _tstate, append, _step, _step_batch, gather_main = self._programs
        pad = pad_pow2(count) - count
        gather_a = jnp.asarray(np.asarray(gather + [0] * pad, dtype=np.int32))
        positions_a = jnp.asarray(np.asarray(positions + [0] * pad, dtype=np.int32))
        rows_a = jnp.asarray(np.asarray(rows + [self._capacity] * pad, dtype=np.int32))  # out-of-range rows are dropped
        with self._context():
            main = gather_main(hidden, gather_a)
            self._rings = append(dstate, self._rings, main, positions_a, rows_a)

    def _catch_up(self, row: int, start: int) -> None:
        """Write positions accepted after the previous window (accepted drafts) up to ``start - 1`` from its hidden.

        A window whose verify accepted draft tokens committed positions beyond its real tokens; when no draft was
        proposed afterwards (:meth:`draft_window` writes them lazily) they are filled here from the remembered hidden
        rows of that window.
        """
        through = self._through.get(row, -1)
        previous = self._last_window.get(row)
        if through >= start - 1 or previous is None:
            return
        hidden, offset, pstart, pcount = previous
        if through + 1 < pstart:
            return  # gap before the remembered window: cannot be recovered, the caller marks the row broken
        positions = list(range(max(through + 1, start - self.window), min(start, pstart + pcount)))
        if positions:
            self._write(hidden, [offset + q - pstart for q in positions], [row] * len(positions), positions)
            if positions[-1] == start - 1:
                self._through[row] = start - 1

    def observe_window(self, hidden: jax.Array | None, entries: list[tuple[str, int, int, int, int, int]]) -> None:
        """Record the committed real tokens of one executed window.

        Args:
            hidden: Window hidden rows ``[N, (1 + T) * D]`` (last hidden | main hidden), or ``None``.
            entries: ``(req_id, row, offset, start, n_commit, count)`` per scheduled request: the request's rows start
                at ``hidden[offset]``, its first token is at position ``start``, ``n_commit`` leading tokens are
                committed (scheduled draft tokens excluded) and ``count`` tokens were scheduled in total.
                ``n_commit < 0`` means the hidden rows are unavailable.
        """
        if not entries:
            return
        self.ensure_batch_size(max(row for _rid, row, *_rest in entries) + 1)
        gather: list[int] = []
        rows: list[int] = []
        positions: list[int] = []
        for req_id, row, offset, start, n_commit, count in entries:
            req_id = str(req_id)
            if self._owner.get(row) != req_id or start == 0:
                self._reset_row(row, req_id)  # new request on this row, or a request restarted from position 0
            if row in self._broken:
                continue
            self._catch_up(row, start)
            if n_commit < 0 or hidden is None or self._through[row] != start - 1:
                self._broken.add(row)
                continue
            self._last_window[row] = (hidden, offset, start, int(count))
            if n_commit == 0:
                continue
            last = start + n_commit
            for q in range(max(start, last - self.window), last):  # only the last window of positions is kept
                gather.append(offset + q - start)
                rows.append(row)
                positions.append(q)
            self._through[row] = last - 1
        self._write(hidden, gather, rows, positions)

    def draft_window(
        self,
        *,
        req_id: str,
        row: int,
        seed_token: int,
        seed_position: int,
        hidden: jax.Array,
        offset: int,
        start: int,
        n: int,
        k: int,
    ) -> list[int]:
        """Greedily draft ``k`` tokens after ``seed_token`` (at position ``seed_position + 1``).

        Args:
            req_id: Request being drafted for.
            row: The request's stable drafter row.
            seed_token: Last accepted token.
            seed_position: Position of the main hidden state the draft reads (``seed_token``'s position minus one);
                must lie inside the window ``[start, start + n)`` of the last observed step.
            hidden: The last observed step's hidden rows ``[N, (1 + T) * D]``.
            offset: Row of ``hidden`` holding the request's first token of that step.
            start: Position of that first token.
            n: Number of tokens the request had in that step.
            k: Tokens to draft (``<= num_draft_tokens``).

        Returns:
            ``k`` drafted token ids, or ``[]`` when this row cannot draft (broken / out-of-window seed).
        """
        req_id = str(req_id)
        position = int(seed_position)
        local = position - int(start)
        if self._owner.get(row) != req_id or row in self._broken or not 0 <= local < int(n):
            return []
        through = self._through.get(row, -1)
        if through < start - 1:
            self._broken.add(row)
            return []
        missing = list(range(max(through + 1, int(start)), position))[-self.window :]
        if missing:
            self._write(hidden, [int(offset) + q - int(start) for q in missing], [row] * len(missing), missing)
        self._ensure_programs()
        dstate, tstate, _append, step, _step_batch, _gather = self._programs
        main_step = hidden[int(offset) + local][self.hidden_size :]
        with self._context():
            ids, self._rings = step(
                dstate,
                tstate,
                self._rings,
                jnp.asarray(row, dtype=jnp.int32),
                jnp.asarray(int(seed_token), dtype=jnp.int32),
                main_step,
                jnp.asarray(position, dtype=jnp.int32),
            )
        self._through[row] = position
        return [int(t) for t in np.asarray(ids)[1 : 1 + int(k)]]

    def draft_windows(
        self,
        hidden: jax.Array,
        entries: tp.Sequence[tuple[str, int, int, int, int, int, int, int]],
    ) -> list[list[int]]:
        """Batched :meth:`draft_window` over many requests of the same observed step: one draft pass, one host sync.

        Per row the bookkeeping is exactly :meth:`draft_window`'s (owner / broken / window checks, lazy write of the
        accepted positions, ``_through`` advanced to the seed position). Rows are independent, so writing every row's
        missing positions first and then drafting all rows together equals the sequential per-row calls.

        Args:
            hidden: The last observed step's hidden rows ``[N, (1 + T) * D]`` (shared by every entry).
            entries: ``(req_id, row, seed_token, seed_position, offset, start, n, k)`` per request, as the keyword
                arguments of :meth:`draft_window`; rows must be distinct.

        Returns:
            Drafted token ids per entry, in input order (``[]`` for rows that cannot draft).

        Raises:
            ValueError: If two entries share a drafter row.
        """
        results: list[list[int]] = [[] for _ in entries]
        live: list[tuple[int, int, int, int, int, int]] = []  # (entry index, row, token, hidden row, position, k)
        gather: list[int] = []
        rows: list[int] = []
        positions: list[int] = []
        seen: set[int] = set()
        for index, (req_id, row, seed_token, seed_position, offset, start, n, k) in enumerate(entries):
            row, start, offset = int(row), int(start), int(offset)
            if row in seen:
                raise ValueError(f"draft_windows: drafter row {row} appears twice.")
            seen.add(row)
            position = int(seed_position)
            local = position - start
            if self._owner.get(row) != str(req_id) or row in self._broken or not 0 <= local < int(n):
                continue
            if self._through.get(row, -1) < start - 1:
                self._broken.add(row)
                continue
            missing = list(range(max(self._through.get(row, -1) + 1, start), position))[-self.window :]
            gather.extend(offset + q - start for q in missing)
            rows.extend([row] * len(missing))
            positions.extend(missing)
            live.append((index, row, int(seed_token), offset + local, position, int(k)))
        if not live:
            return results
        self._write(hidden, gather, rows, positions)
        self._ensure_programs()
        dstate, tstate, _append, _step, step_batch, gather_main = self._programs
        pad = pad_pow2(len(live)) - len(live)
        rows_a = jnp.asarray(np.asarray([e[1] for e in live] + [self._capacity] * pad, dtype=np.int32))
        tokens_a = jnp.asarray(np.asarray([e[2] for e in live] + [0] * pad, dtype=np.int32))
        gather_a = jnp.asarray(np.asarray([e[3] for e in live] + [0] * pad, dtype=np.int32))
        positions_a = jnp.asarray(np.asarray([e[4] for e in live] + [0] * pad, dtype=np.int32))
        with self._context():
            main = gather_main(hidden, gather_a)
            ids, self._rings = step_batch(dstate, tstate, self._rings, main, rows_a, tokens_a, positions_a)
        ids_host = np.asarray(ids)
        for slot, (index, row, _token, _hidden_row, position, k) in enumerate(live):
            self._through[row] = position
            results[index] = [int(t) for t in ids_host[slot, 1 : 1 + k]]
        return results

    def draft(self, *args: tp.Any, **kwargs: tp.Any) -> tp.NoReturn:
        """Unsupported: this drafter is driven through :meth:`draft_window` (``whole_block_draft``)."""
        del args, kwargs
        raise NotImplementedError(f"{type(self).__name__} drafts whole windows via `draft_window`.")


__all__ = ["DSparkDrafter", "DSparkState", "pad_pow2"]
