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

from dataclasses import dataclass

from jax import Array

from .states import CachedRequestState


@dataclass(frozen=True)
class DeviceInputTokenHandoff:
    """Device-resident sampled tokens that should replace decode placeholders.

    This is the eSurge equivalent of vLLM's PP sampled-token handoff: the final
    stage samples token ``N`` on device, and the next decode step consumes that
    token directly from a device array instead of waiting for the host-side
    sequence buffer to be repaired first.

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
