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

"""Prefill-chunk alignment around indivisible image spans (pure Python, no JAX).

Some vision-language models (DeepSeek-V4.1) merge an image's embeddings and route its tokens as one unit, so a
prefill chunk must never end strictly inside an image span. The scheduler calls :func:`align_prefill_chunk` after it
has applied every other cap (chunk threshold, token budget) to the number of tokens it is about to schedule.
"""

from __future__ import annotations

import typing as tp


def check_image_spans_fit(spans: tp.Sequence[tuple[int, int]], max_tokens: int, request_id: str) -> None:
    """Reject requests that hold an image span longer than the per-step token budget.

    Args:
        spans: Half-open ``(start, stop)`` position ranges of the image spans.
        max_tokens: Per-step token budget (``max_num_batched_tokens``).
        request_id: Request id used in the error message.

    Raises:
        ValueError: If a span cannot be scheduled in one step.
    """
    for start, stop in spans:
        if stop - start > int(max_tokens):
            raise ValueError(
                f"Request {request_id} holds an image span of {stop - start} tokens (positions {start}..{stop}) "
                f"which exceeds max_num_batched_tokens={int(max_tokens)}; an image span must be prefilled in one "
                "step. Raise max_num_batched_tokens or send a smaller image."
            )


def align_prefill_chunk(
    spans: tp.Sequence[tuple[int, int]],
    num_computed_tokens: int,
    num_new_tokens: int,
    token_budget: int,
) -> int:
    """Adjust a prefill chunk so it does not end strictly inside an image span.

    If the chunk ``[num_computed, num_computed + num_new)`` would end inside a span, it is shrunk to end at the
    span's start; when it already begins at the span's start it is extended to cover the whole span (this may exceed a
    soft per-chunk cap such as ``long_prefill_token_threshold``).

    Args:
        spans: Sorted half-open ``(start, stop)`` position ranges of the image spans.
        num_computed_tokens: Tokens of the request already processed.
        num_new_tokens: Tokens about to be scheduled.
        token_budget: Tokens still available in this step.

    Returns:
        The aligned token count. ``0`` means the span does not fit the remaining budget of this step; the caller must
        skip the request for now (it fits in a fresh step because spans are validated against the full budget).
    """
    end = int(num_computed_tokens) + int(num_new_tokens)
    for start, stop in spans:
        if start < end < stop:
            if start > num_computed_tokens:
                return int(start - num_computed_tokens)
            needed = int(stop - num_computed_tokens)
            return needed if needed <= int(token_budget) else 0
    return int(num_new_tokens)


__all__ = ["align_prefill_chunk", "check_image_spans_fit"]
