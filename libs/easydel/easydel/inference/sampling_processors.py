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

"""Logit processors shared by ordinary and speculative sampling."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

_MASKED_LOGIT = -1e10


@dataclasses.dataclass(frozen=True)
class SamplingConstraints:
    """One step's per-request logit bias, allow-lists and banned next tokens, as sparse ``(row, token)`` entries.

    Rows index the step's window rows. A row in ``restricted_rows`` may emit only the tokens its ``allowed`` entries
    name; ``banned`` entries are the next tokens a row's bad words forbid. A request names a handful of tokens, so a
    step carries those entries instead of dense ``[rows, vocab]`` arrays.
    """

    bias_rows: np.ndarray
    bias_tokens: np.ndarray
    bias_values: np.ndarray
    restricted_rows: np.ndarray
    allowed_rows: np.ndarray
    allowed_tokens: np.ndarray
    banned_rows: np.ndarray
    banned_tokens: np.ndarray

    @classmethod
    def from_entries(
        cls,
        bias: list[tuple[int, int, float]],
        restricted: list[int],
        allowed: list[tuple[int, int]],
        banned: list[tuple[int, int]],
    ) -> SamplingConstraints | None:
        """Build from ``(row, token, value)`` bias entries, restricted rows, and ``(row, token)`` allowed and banned
        entries; ``None`` when all four are empty."""
        if not (bias or restricted or allowed or banned):
            return None
        return cls(
            bias_rows=np.asarray([row for row, _, _ in bias], dtype=np.int32),
            bias_tokens=np.asarray([token for _, token, _ in bias], dtype=np.int32),
            bias_values=np.asarray([value for _, _, value in bias], dtype=np.float32),
            restricted_rows=np.asarray(restricted, dtype=np.int32),
            allowed_rows=np.asarray([row for row, _ in allowed], dtype=np.int32),
            allowed_tokens=np.asarray([token for _, token in allowed], dtype=np.int32),
            banned_rows=np.asarray([row for row, _ in banned], dtype=np.int32),
            banned_tokens=np.asarray([token for _, token in banned], dtype=np.int32),
        )


def _padded_size(n: int) -> int:
    """At least 8 and a power of two, so entry lists compile a bounded number of programs."""
    return max(8, 1 << max(0, int(n) - 1).bit_length())


def compact_sampling_constraints(
    gather_positions: np.ndarray,
    constraints: SamplingConstraints | None,
) -> tuple[np.ndarray, ...] | None:
    """Map a step's constraints onto the sampler's compact rows.

    Compact row ``i`` samples window row ``gather_positions[i]``, so every entry of window row ``w`` is repeated for
    each compact row that reads ``w``. Entry lists are padded to a power of two (at least 8) with an out-of-range row,
    which the device scatter drops.

    Returns:
        ``None`` when there is nothing to apply, else ``(bias_rows, bias_tokens, bias_values, restricted,
        allowed_rows, allowed_tokens, banned_rows, banned_tokens)``, ``restricted`` being a ``[num_compact_rows]`` bool
        mask, for :func:`apply_sparse_sampling_constraints`.
    """
    if constraints is None:
        return None
    gather = np.asarray(gather_positions, dtype=np.int64).reshape(-1)
    num_rows = int(gather.shape[0])

    def remap(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        entry, compact = np.nonzero(np.asarray(rows, dtype=np.int64).reshape(-1, 1) == gather.reshape(1, -1))
        return entry, compact.astype(np.int32)

    def pad(values: np.ndarray, fill, dtype, size: int) -> np.ndarray:
        out = np.full((size,), fill, dtype=dtype)
        out[: len(values)] = values
        return out

    bias_entry, bias_rows = remap(constraints.bias_rows)
    allowed_entry, allowed_rows = remap(constraints.allowed_rows)
    banned_entry, banned_rows = remap(constraints.banned_rows)
    nb, na, nn = _padded_size(len(bias_rows)), _padded_size(len(allowed_rows)), _padded_size(len(banned_rows))
    return (
        pad(bias_rows, num_rows, np.int32, nb),
        pad(constraints.bias_tokens[bias_entry], 0, np.int32, nb),
        pad(constraints.bias_values[bias_entry], 0.0, np.float32, nb),
        np.isin(gather, np.asarray(constraints.restricted_rows, dtype=np.int64)),
        pad(allowed_rows, num_rows, np.int32, na),
        pad(constraints.allowed_tokens[allowed_entry], 0, np.int32, na),
        pad(banned_rows, num_rows, np.int32, nn),
        pad(constraints.banned_tokens[banned_entry], 0, np.int32, nn),
    )


@jax.jit
def apply_sparse_sampling_constraints(
    logits: jax.Array,
    bias_rows: jax.Array,
    bias_tokens: jax.Array,
    bias_values: jax.Array,
    restricted: jax.Array,
    allowed_rows: jax.Array,
    allowed_tokens: jax.Array,
    banned_rows: jax.Array,
    banned_tokens: jax.Array,
) -> jax.Array:
    """Apply :func:`compact_sampling_constraints` output to ``[rows, vocab]`` logits.

    Adds the bias entries, masks every token a restricted row does not allow, then masks the banned entries, in the
    order of :func:`apply_sampling_constraints`. Entries whose row or token is out of range are dropped.
    """
    masked = jnp.asarray(_MASKED_LOGIT, dtype=logits.dtype)
    logits = logits.at[bias_rows, bias_tokens].add(bias_values.astype(logits.dtype), mode="drop")
    allowed = jnp.zeros(logits.shape, dtype=jnp.bool_).at[allowed_rows, allowed_tokens].set(True, mode="drop")
    logits = jnp.where(restricted[:, None] & ~allowed, masked, logits)
    return logits.at[banned_rows, banned_tokens].set(masked, mode="drop")


def apply_sampling_constraints(
    logits: jax.Array,
    logit_bias: jax.Array | None = None,
    allowed_token_ids_mask: jax.Array | None = None,
    bad_words_token_ids: jax.Array | None = None,
) -> jax.Array:
    """Apply per-request logit bias, allowed-token and bad-token masks given as dense arrays.

    ``allowed_token_ids_mask`` uses ``True`` for allowed tokens and ``bad_words_token_ids`` uses ``True`` for banned
    tokens. ``None`` leaves the corresponding constraint disabled. The masks use ``-1e10`` rather than NaN so greedy
    and sampled paths have finite logits for every row.
    """
    if logit_bias is not None:
        logits = logits + logit_bias.astype(logits.dtype)
    if allowed_token_ids_mask is not None:
        logits = jnp.where(allowed_token_ids_mask, logits, _MASKED_LOGIT)
    if bad_words_token_ids is not None:
        logits = jnp.where(bad_words_token_ids, _MASKED_LOGIT, logits)
    return logits
