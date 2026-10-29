"""Host-side positional dispatch boundaries for ragged attention v3."""

from __future__ import annotations

import numpy as np


def compute_request_distribution_prefix(
    scheduled: np.ndarray,
    num_computed_tokens: np.ndarray,
    num_requests: int,
) -> tuple[int, int, int]:
    """Return decode/prefill prefix ends, leaving any interleaved tail mixed.

    The kernel dispatches by row position, not by request-type counts. No
    token or page-table reordering is performed by this utility.

    Args:
        scheduled: Scheduled token counts in physical request-row order.
        num_computed_tokens: Already computed token counts in the same order.
        num_requests: Number of request rows the kernel should inspect.

    Returns:
        ``(decode_end, prefill_end, num_requests)`` positional boundaries.
    """
    if num_requests <= 0:
        return 0, 0, 0
    active = scheduled[:num_requests] > 0
    is_decode = (scheduled[:num_requests] == 1) & (num_computed_tokens[:num_requests] > 0) & active
    flags = np.zeros(num_requests, dtype=np.int8)
    flags[is_decode] = 1
    flags[active & ~is_decode] = 2
    non_decode = np.flatnonzero(flags != 1)
    decode_end = int(non_decode[0]) if non_decode.size else num_requests
    non_prefill = np.flatnonzero(flags[decode_end:] != 2)
    prefill_end = decode_end + (int(non_prefill[0]) if non_prefill.size else num_requests - decode_end)
    return decode_end, prefill_end, num_requests
