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

"""Async scheduling payloads for overlapped TPU/JAX execution.

The overlap path keeps model execution on the scheduler thread, but returns
device-backed sampled-token arrays before the host blocks on copying them.
These payloads let the lifecycle loop prefetch the next scheduler step while
the device is still producing the current batch's sampled tokens.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from functools import partial

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp

from .states import CachedRequestState


@partial(jax.jit, static_argnums=1)
def _unpack_spec_out(spec_out: Array, num_draft_tokens: int) -> tuple[Array, Array, Array]:
    """Split a step's packed ``spec_out[R, 2k+2]`` into (last emitted token, accepted drafts, next drafts) per row.

    Columns ``[0, k+1)`` hold the emitted tokens (accepted drafts, then the bonus or corrected token), column ``k+1``
    the accepted drafts ``a`` (``-1``: the row verified nothing, so it emitted one token) and ``[k+2, 2k+2)`` the next
    drafts. ``-1`` drafts (drafter skipped for the row) become token ``0`` placeholders: verification of any proposed
    token is lossless (greedy compare, point-mass rejection), so a placeholder only costs its verify slot.
    """
    k = int(num_draft_tokens)
    spec_out = spec_out.astype(jnp.int32)
    accepted = jnp.maximum(spec_out[:, k + 1], 0)
    last = jnp.take_along_axis(spec_out[:, : k + 1], jnp.minimum(accepted, k)[:, None], axis=1)[:, 0]
    drafts = jnp.maximum(spec_out[:, k + 2 : 2 * k + 2], 0)
    return last, accepted, drafts


def spec_window_sources(
    window_req_ids: tp.Sequence[str | None],
    fed_by: tp.Mapping[str, tuple[int, int]],
    max_num_reqs: int,
) -> np.ndarray:
    """Host map from step ``N+1``'s window rows to step ``N``'s verify output rows.

    Built from plan data only (which request each window row holds, and where step ``N`` put it), so every host of
    a multi-host run computes the same map without reading step ``N``'s results.

    Args:
        window_req_ids: Step ``N+1``'s request per window row (``None`` for an empty row).
        fed_by: Per request verified at step ``N``: ``(output row, drafts verified)``.
        max_num_reqs: Width of the payload.

    Returns:
        int32 ``[2, max_num_reqs]``: row 0 the source output row (``-1``: none), row 1 the drafts verified (``d``).
    """
    sources = np.zeros((2, int(max_num_reqs)), dtype=np.int32)
    sources[0] = -1
    for row, rid in enumerate(window_req_ids[: int(max_num_reqs)]):
        hit = None if rid is None else fed_by.get(str(rid))
        if hit is not None:
            sources[0, row], sources[1, row] = int(hit[0]), int(hit[1])
    return sources


@dataclass(frozen=True)
class DeviceSpecWindowHandoff:
    """Step ``N``'s verify results on the device, and the host map that turns them into step ``N+1``'s windows.

    With drafts in flight, the host builds step ``N+1`` as if every draft step ``N`` verified was accepted: each
    speculative row is a window of ``1 + k`` placeholder tokens whose positions and sequence length run
    ``delta = d - a`` tokens ahead of the truth (``d`` drafts verified at step ``N``, ``a`` of them accepted). The
    device rewrites the window before the model step (``BatchMetadataPreparer.apply_spec_window_handoff``), so the
    host never reads step ``N``'s results to launch step ``N+1``.

    Attributes:
        last_token: int32 ``[P]``, per step-``N`` output row the token it emitted last (the bonus or corrected
            token): the first input of the row's next window.
        accepted: int32 ``[P]``, per step-``N`` output row the drafts it accepted (``a``).
        drafts: int32 ``[P, k]``, per step-``N`` output row the drafts it produced for its next window.
        sources: int32 ``[2, max_num_reqs]`` host payload in step ``N+1``'s window-row order. Row 0 is the step-``N``
            output row that feeds the window row (``-1``: none, the row's host inputs are already exact); row 1 is
            the drafts step ``N`` verified for it (``d``). The caller places it with the step's input sharding.
        rewrite_recurrent_base: Whether the step reads recurrent state from slot groups. The host's per-row slot
            base (``spec_recurrent_commit`` row 0) is then stale by step ``N - 1``'s rejected drafts, so the device
            sets it to ``corrected context before the step - a``: the row reads slot ``g + a``.
    """

    last_token: Array
    accepted: Array
    drafts: Array
    sources: tp.Any
    rewrite_recurrent_base: bool = False

    @classmethod
    def from_spec_out(
        cls,
        *,
        spec_out: Array,
        num_draft_tokens: int,
        sources: tp.Any,
        rewrite_recurrent_base: bool = False,
    ) -> DeviceSpecWindowHandoff:
        """Build the handoff from a step's packed speculative output, on the device.

        Args:
            spec_out: int32 ``[max_num_reqs, 2k+2]``, rows = the sequence-buffer rows at that step's dispatch (see
                :func:`_unpack_spec_out` for the columns).
            num_draft_tokens: ``k``, the runner's compiled draft count.
            sources: The ``[2, max_num_reqs]`` window-row map (see the class docstring).
            rewrite_recurrent_base: See the class docstring.

        Returns:
            The handoff; nothing is read back to the host.
        """
        last_token, accepted, drafts = _unpack_spec_out(spec_out, int(num_draft_tokens))
        return cls(
            last_token=last_token,
            accepted=accepted,
            drafts=drafts,
            sources=sources,
            rewrite_recurrent_base=rewrite_recurrent_base,
        )


@dataclass(frozen=True)
class SpecWindowCarry:
    """What step ``N`` leaves for step ``N+1`` when ``N+1`` is dispatched before ``N`` is drained.

    Attributes:
        spec_out: Step ``N``'s packed device output ``[max_num_reqs, 2k+2]`` (never donated, so it stays readable
            until step ``N+1``'s program has consumed it).
        num_draft_tokens: ``k``.
        rows: Per request in step ``N``, its sequence-buffer row at dispatch (the ``spec_out`` row).
    """

    spec_out: Array
    num_draft_tokens: int
    rows: tp.Mapping[str, int]

    def handoff_for(
        self,
        window_req_ids: tp.Sequence[str | None],
        inflight_drafts: tp.Mapping[str, int],
        max_num_reqs: int,
        *,
        rewrite_recurrent_base: bool = False,
    ) -> DeviceSpecWindowHandoff | None:
        """The handoff for one of step ``N+1``'s windows, or ``None`` when none of its rows is optimistic.

        Args:
            window_req_ids: Step ``N+1``'s request per window row.
            inflight_drafts: From step ``N+1``'s plan, per request scheduled while step ``N`` was still in flight,
                the drafts step ``N`` verified for it (``d >= 0``; its host-built start counts all ``d`` as
                accepted). Requests the core scheduled from committed state are absent and keep their host inputs.
            max_num_reqs: Width of the payload.
            rewrite_recurrent_base: See :class:`DeviceSpecWindowHandoff`.

        Raises:
            ValueError: When the plan names a request step ``N`` did not run.
        """
        missing = [rid for rid in inflight_drafts if rid not in self.rows]
        if missing:
            raise ValueError(f"Requests scheduled behind an in-flight step that it did not run: {missing[:4]}")
        fed_by = {str(rid): (int(self.rows[rid]), int(d)) for rid, d in inflight_drafts.items()}
        sources = spec_window_sources(window_req_ids, fed_by, max_num_reqs)
        if not bool((sources[0] >= 0).any()):
            return None
        return DeviceSpecWindowHandoff.from_spec_out(
            spec_out=self.spec_out,
            num_draft_tokens=self.num_draft_tokens,
            sources=sources,
            rewrite_recurrent_base=rewrite_recurrent_base,
        )


@dataclass(frozen=True)
class DeviceInputTokenHandoff:
    """Device-resident sampled tokens that should replace decode placeholders.

    The final stage samples token ``N`` on device, and the next decode step
    consumes that token directly from a device array instead of waiting for the
    host-side sequence buffer to be repaired first.

    The handoff is intentionally narrow:
        * ``input_positions`` index the flattened ``BatchMetadata.input_ids``
          buffer for the current window. The array is padded to a stable shape.
        * ``token_ids`` contains the sampled token ids gathered from the
          previous step's device output, one element per input position, padded
          the same way as ``input_positions``.
        * ``count`` marks how many entries from the current offset are live.
        * ``offset`` lets PP microbatches consume a contiguous slice of a
          window-level handoff without building per-microbatch scatter buffers
          on the launch path.

    CPU state is still repaired after the next device dispatch is queued, so
    streaming/accounting remains exact. The point of this object is only to
    remove the sampled-token host materialization from the critical path that
    launches the next PP decode step.
    """

    input_positions: Array
    token_ids: Array
    count: Array
    offset: Array | None = None


@dataclass
class AsyncWindowResult:
    """One scheduler-window's worth of in-flight sampled-token state.

    Built by :meth:`eSurgeRunner._execute_model_impl` whenever the runner
    needs to defer host materialization (overlap path or async scheduling).
    The ``sampled_token_ids`` and optional ``token_logprobs`` arrays have
    already had ``jax.copy_to_host_async`` called on them so the device
    transfer is in flight; the receiving thread realizes them with
    :func:`numpy.asarray` once the lifecycle loop reaches the
    finalize step.

    Attributes:
        req_ids: Request ids active in this window, ordered to match the
            row layout the runner used during execution. Aligns with
            ``row_positions`` element-wise.
        row_positions: Per-entry index into ``sampled_token_ids`` /
            ``token_logprobs`` for the corresponding ``req_ids[i]``. Values
            may be sparse (when a window contained dropped rows) so callers
            should index by these positions rather than enumerate.
        sampled_token_ids: Device array of sampled token ids with shape
            ``[num_reqs]``. Host-copy is already initiated; ``np.asarray``
            on the consumer side blocks until the transfer completes.
        valid_mask: Per-request flag — ``False`` rows correspond to slots
            whose sampled token must be discarded (partial prefill, padding).
            Same length as ``req_ids``.
        token_logprobs: Optional companion array of token logprobs (shape
            ``[num_reqs]``) for sampler-metrics callers, with host-copy
            already initiated. ``None`` when sampler metrics are disabled.
    """

    req_ids: list[str]
    row_positions: list[int]
    sampled_token_ids: Array
    valid_mask: list[bool]
    token_logprobs: Array | None = None


@dataclass
class AsyncPreResults:
    """Carry-across snapshot from the previous async-scheduling iteration.

    Created at the end of step ``N`` (when the runner could not yet write
    the sampled tokens back into the sequence buffer because the host copy
    was still in flight) and consumed at the start of step ``N+1`` by
    :meth:`eSurgeRunner._modify_prev_results`, which finalizes the sampled
    tokens, writes them into the buffer, and replaces the placeholders that
    :meth:`_update_placeholder` injected on step ``N``.

    Attributes:
        windows (list[AsyncWindowResult]): Per-window host-copy payloads
            captured from the previous step. Each contains the deferred
            sampled-token tensor plus metadata needed to map rows back to
            requests.
        request_seq_lens (list[tuple[int, int, CachedRequestState, int]]):
            Per-request rendezvous tuples
            ``(out_index, seq_row_idx, request_state, seq_len)`` so the
            finalizer can update the buffer at the correct row and append
            the freshly-decoded token to the request's output stream.
    """

    windows: list[AsyncWindowResult]
    request_seq_lens: list[tuple[int, int, CachedRequestState, int]]
