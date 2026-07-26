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

"""Pluggable speculative-decoding strategy interface for the eSurge runner.

:class:`~easydel.inference.esurge.runners.model_runner.eSurgeRunner` drives
speculative decoding through a strategy object exposed as ``runner.spec``:

- :class:`NullSpeculation` — inert strategy for plain (non-speculative)
  decode; every hook is a no-op and ``num_draft_tokens`` is ``0``.
- :class:`~easydel.inference.esurge.runners.spec.strategy.DrafterSpeculation`
  — the runner-native draft/verify/commit implementation backed by a
  drafter (:class:`~easydel.inference.speculative.DrafterProtocol`).

:class:`SpeculativeStrategy` documents the exact call surface
``eSurgeRunner._execute_model_impl`` (and ``reset_state``) uses. Fidelity to
the pre-extraction runner behavior takes precedence over interface elegance:
the members mirror the former ``eSurgeRunner._spec_*`` methods one-to-one.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import jax
    import numpy as np

    from ...scheduler import SchedulerOutput
    from ..states import CachedRequestState
    from .strategy import _SpecVerifyMetadata


class SpeculativeStrategy(typing.Protocol):
    """Call surface the eSurge runner drives speculative decoding through.

    Attributes:
        num_draft_tokens: Static draft tokens proposed per verify window
            (``0`` disables speculative decoding; the scheduler reads the
            runner mirror of this value). This is the compiled maximum.
        live_num_draft_tokens: The per-step dynamic draft budget resolved by
            :meth:`resolve_step_draft_budget`; equals ``num_draft_tokens``
            without a schedule. Draft paths read this value.
        uses_recurrent_candidates: Whether hybrid recurrent caches keep
            per-request speculative candidate rows for this configuration.
        num_drafts_generated: Total draft tokens proposed so far.
        num_drafts_accepted: Total draft tokens accepted by verification.
        num_verify_steps: Total verify windows processed.
        reject_backoff_steps: Drafter-call backoff after a full rejection
            (``0`` disables the stop-loss).
        debug_traces: Collected host-side verify traces (diagnostics).
        debug_max_traces: Trace budget (``EASYDEL_SPEC_DECODE_DEBUG_STEPS``).
        draft_full_log_probs_by_req: Per-request stacked draft
            log-probability tensors awaiting verification.
    """

    num_draft_tokens: int
    live_num_draft_tokens: int
    uses_recurrent_candidates: bool
    num_drafts_generated: int
    num_drafts_accepted: int
    num_verify_steps: int
    reject_backoff_steps: int
    debug_traces: list[dict[str, typing.Any]]
    debug_max_traces: int
    draft_full_log_probs_by_req: dict[str, jax.Array]

    def resolve_step_draft_budget(self, concurrency: int) -> int:
        """Resolve the per-step draft-token budget from the live concurrency."""
        ...

    def on_states_applied(self, scheduler_output: SchedulerOutput) -> None:
        """Materialize scheduler-provided draft tokens into the sequence buffer."""
        ...

    def on_reset(self) -> None:
        """Clear per-request speculative state (runner ``reset_state`` hook)."""
        ...

    def is_spec_request(self, req_state: CachedRequestState | None) -> bool:
        """Whether the request is eligible for speculative decoding."""
        ...

    def is_greedy_request(self, req_state: CachedRequestState | None) -> bool:
        """Whether speculative decoding should use the greedy path."""
        ...

    def cache_replay_required(self) -> bool:
        """Whether rejected drafts require an explicit KV replay."""
        ...

    def recurrent_candidate_count(self) -> int:
        """Per-request speculative recurrent candidate rows, if present."""
        ...

    def cache_has_recurrent_state(self) -> bool:
        """Whether the eSurge cache contains recurrent/SSM rows."""
        ...

    def rng_split(self, n: int = 1) -> list[jax.Array]:
        """Advance the manager-owned RNG by ``n`` and return ``n`` fresh keys."""
        ...

    def filtered_log_probs(
        self,
        logits_or_log_probs: jax.Array,
        req_state: CachedRequestState,
    ) -> jax.Array:
        """Apply the request's filtering+temperature and return log probabilities."""
        ...

    def sample_distribution(self, log_probs: jax.Array, req_state: CachedRequestState) -> int:
        """Sample one token id from ``log_probs`` using the request's mode."""
        ...

    def filter_and_sample_log_probs(
        self,
        logits_or_log_probs: jax.Array,
        req_state: CachedRequestState,
    ) -> tuple[jax.Array, jax.Array]:
        """Return sampled token IDs and filtered log-probs using one cached JIT."""
        ...

    def verify_sampled_window(
        self,
        *,
        target_logits: jax.Array,
        draft_log_probs: jax.Array,
        draft_tokens: list[int],
        req_state: CachedRequestState,
    ) -> tuple[int, int]:
        """Compiled rejection sampler for one non-greedy spec window."""
        ...

    def build_verify_metadata(
        self,
        *,
        request_id: str,
        row_pos: int,
        req_idx: int,
        start_pos: int,
        real_count: int,
        scheduled_draft_tokens: list[int],
        token_ids_window_cpu: np.ndarray,
        token_offset: int,
    ) -> _SpecVerifyMetadata:
        """Build target-logit/draft-token verification indices."""
        ...

    def record_verify_trace(
        self,
        *,
        meta: _SpecVerifyMetadata,
        logits_rows: jax.Array | np.ndarray | None,
        accepted: int,
        corrected_token: int,
        greedy: bool,
        source: str,
    ) -> None:
        """Record a small host trace for speculative acceptance diagnosis."""
        ...

    def project_hidden_rows(self, hidden_rows: jax.Array) -> jax.Array:
        """Project arbitrary-row hidden states through the LM head for verify."""
        ...

    def draft_next(
        self,
        *,
        req_id: str,
        seed_token: int,
        seed_position: int,
        seed_hidden: jax.Array,
        req_state: CachedRequestState | None,
        prefix_input_ids: jax.Array | None = None,
        prefix_hidden_states: jax.Array | None = None,
        prefix_position_ids: jax.Array | None = None,
    ) -> list[int]:
        """Generate up to ``num_draft_tokens`` draft tokens for one request."""
        ...

    def can_draft_in_verify(self) -> bool:
        """Whether the fused draft-in-verify program can serve this drafter."""
        ...

    def begin_draft_in_verify(
        self,
        entries: list[tuple[_SpecVerifyMetadata, int, int]],
        *,
        gathered_hidden: jax.Array,
        k_pad: int,
    ) -> dict | None:
        """Dispatch the fused verify+draft program without blocking on it."""
        ...

    def finish_draft_in_verify(self, handle: dict) -> dict[str, dict[str, typing.Any]]:
        """Read back a fused verify+draft dispatched by :meth:`begin_draft_in_verify`."""
        ...

    def record_acceptance(self, req_id: str, accepted: int, num_drafts: int) -> None:
        """Update the adaptive drafter stop-loss after a verify window."""
        ...

    def queue_or_commit_recurrent_candidate_state(
        self,
        *,
        req_id: str,
        row_pos: int,
        prefix_len: int,
        defer: bool,
    ) -> bool:
        """Make the live recurrent row match the accepted speculative prefix."""
        ...

    def commit_recurrent_candidate_state(self, row_pos: int, prefix_len: int) -> bool:
        """Copy a speculative recurrent candidate row into the live recurrent row."""
        ...

    def commit_cache_replay(self, **kwargs: typing.Any) -> None:
        """Replay the executor step over the committed speculative prefix."""
        ...

    def run_suffix_sample(self, row_pos: int, start_len: int, suffix_len: int = 1, **kwargs: typing.Any) -> tuple[int, jax.Array]:
        """Execute a one-row suffix forward pass; return sampled token + hidden row."""
        ...

    def replay_prefix_sample(self, row_pos: int, prefix_len: int, **kwargs: typing.Any) -> tuple[int, jax.Array]:
        """Replay the accepted speculative prefix in one fused forward pass."""
        ...

    def replay_prefix_sequential(self, row_pos: int, prefix_len: int, **kwargs: typing.Any) -> tuple[int, jax.Array]:
        """Replay an accepted speculative prefix one token at a time."""
        ...


class NullSpeculation:
    """Inert speculation strategy used when no drafter is attached.

    The queried surface (eligibility checks, counters, lifecycle hooks)
    behaves exactly like the pre-strategy runner with ``drafter=None``:
    everything is a no-op, every count is zero, and every eligibility check
    is ``False``. Draft/verify/commit operations — which the runner only
    reaches behind ``drafter is not None`` guards — fail loudly instead of
    silently corrupting decode state.
    """

    uses_recurrent_candidates: bool = False

    def __init__(self) -> None:
        """Initialize the inert strategy with zeroed counters and no state."""
        self.num_draft_tokens = 0
        self.live_num_draft_tokens = 0
        self.num_drafts_generated = 0
        self.num_drafts_accepted = 0
        self.num_verify_steps = 0
        self.reject_backoff_steps = 0
        self.debug_traces: list[dict[str, typing.Any]] = []
        self.debug_max_traces = 0
        self.draft_full_log_probs_by_req: dict[str, jax.Array] = {}

    def resolve_step_draft_budget(self, concurrency: int) -> int:
        """Always ``0``: no drafter, no draft budget at any concurrency."""
        del concurrency
        return 0

    def on_states_applied(self, scheduler_output: SchedulerOutput) -> None:
        """No-op: nothing to materialize without a drafter."""
        del scheduler_output

    def on_reset(self) -> None:
        """No-op: no per-request speculative state to clear."""

    def is_spec_request(self, req_state: CachedRequestState | None) -> bool:
        """Always ``False``: no request is spec-eligible without a drafter."""
        del req_state
        return False

    def is_greedy_request(self, req_state: CachedRequestState | None) -> bool:
        """Always ``False``: mirrors the drafter-gated greedy check."""
        del req_state
        return False

    def cache_replay_required(self) -> bool:
        """Always ``False``: no speculative KV to replay."""
        return False

    def recurrent_candidate_count(self) -> int:
        """Always ``0``: no speculative candidate rows are allocated."""
        return 0

    def cache_has_recurrent_state(self) -> bool:
        """Always ``False``: never queried on the plain decode path."""
        return False

    def record_acceptance(self, req_id: str, accepted: int, num_drafts: int) -> None:
        """No-op: there is no backoff state to update."""
        del req_id, accepted, num_drafts

    def draft_next(self, **kwargs: typing.Any) -> list[int]:
        """Return no drafts, matching the old ``drafter is None`` early exit."""
        del kwargs
        return []

    def can_draft_in_verify(self) -> bool:
        """Always ``False``: no drafter, no fused verify+draft program."""
        return False

    def begin_draft_in_verify(self, *args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        """Unreachable without a drafter; fails loudly."""
        del args, kwargs
        self._unsupported("begin_draft_in_verify")

    def finish_draft_in_verify(self, *args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        """Unreachable without a drafter; fails loudly."""
        del args, kwargs
        self._unsupported("finish_draft_in_verify")

    def _unsupported(self, operation: str) -> typing.NoReturn:
        """Raise for draft/verify/commit surfaces that require a drafter.

        Args:
            operation: Name of the strategy operation that was invoked.

        Raises:
            RuntimeError: Always; these paths must stay behind the runner's
                ``drafter is not None`` guards.
        """
        raise RuntimeError(f"NullSpeculation cannot perform {operation!r}: speculative decoding is disabled.")

    def rng_split(self, *args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        """Unreachable without a drafter; fails loudly."""
        del args, kwargs
        self._unsupported("rng_split")

    def filtered_log_probs(self, *args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        """Unreachable without a drafter; fails loudly."""
        del args, kwargs
        self._unsupported("filtered_log_probs")

    def sample_distribution(self, *args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        """Unreachable without a drafter; fails loudly."""
        del args, kwargs
        self._unsupported("sample_distribution")

    def filter_and_sample_log_probs(self, *args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        """Unreachable without a drafter; fails loudly."""
        del args, kwargs
        self._unsupported("filter_and_sample_log_probs")

    def verify_sampled_window(self, *args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        """Unreachable without a drafter; fails loudly."""
        del args, kwargs
        self._unsupported("verify_sampled_window")

    def build_verify_metadata(self, *args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        """Unreachable without a drafter; fails loudly."""
        del args, kwargs
        self._unsupported("build_verify_metadata")

    def record_verify_trace(self, *args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        """Unreachable without a drafter; fails loudly."""
        del args, kwargs
        self._unsupported("record_verify_trace")

    def project_hidden_rows(self, *args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        """Unreachable without a drafter; fails loudly."""
        del args, kwargs
        self._unsupported("project_hidden_rows")

    def queue_or_commit_recurrent_candidate_state(self, *args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        """Unreachable without a drafter; fails loudly."""
        del args, kwargs
        self._unsupported("queue_or_commit_recurrent_candidate_state")

    def commit_recurrent_candidate_state(self, *args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        """Unreachable without a drafter; fails loudly."""
        del args, kwargs
        self._unsupported("commit_recurrent_candidate_state")

    def commit_cache_replay(self, *args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        """Unreachable without a drafter; fails loudly."""
        del args, kwargs
        self._unsupported("commit_cache_replay")

    def run_suffix_sample(self, *args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        """Unreachable without a drafter; fails loudly."""
        del args, kwargs
        self._unsupported("run_suffix_sample")

    def replay_prefix_sample(self, *args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        """Unreachable without a drafter; fails loudly."""
        del args, kwargs
        self._unsupported("replay_prefix_sample")

    def replay_prefix_sequential(self, *args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        """Unreachable without a drafter; fails loudly."""
        del args, kwargs
        self._unsupported("replay_prefix_sequential")
