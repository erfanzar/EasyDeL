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

"""Shared page-axis, feature-flag, and request-distribution utilities.

The operation framework already owns the DP-page-axis implementation and the
single import-time read of ``EASYDEL_ENABLE_DP_LOCAL_PAGE_PATH``. Re-export
those objects rather than adding another flag read or axis implementation.
"""

import jax
from jax import numpy as jnp

from .._operation_impl import ENABLE_DP_LOCAL_PAGE_PATH, dp_page_axis

__all__ = ["ENABLE_DP_LOCAL_PAGE_PATH", "dp_page_axis", "request_distribution_bounds"]


def request_distribution_bounds(scheduled: jax.Array, context_lens: jax.Array) -> jax.Array:
    """Build the request distribution ``[decode_end, prefill_end, total]``.

    A request is active when it has scheduled tokens. Active requests with one
    scheduled token and a total context length greater than one are decode
    requests; all other active requests are prefills. Inactive rows do not
    contribute to any boundary.

    Args:
        scheduled: ``int32[num_requests]`` count of scheduled tokens per request.
        context_lens: ``int32[num_requests]`` total context length per request.

    Returns:
        An ``int32[3]`` array containing the decode boundary, prefill boundary,
        and total active request count.
    """
    scheduled = jnp.asarray(scheduled, dtype=jnp.int32)
    context_lens = jnp.asarray(context_lens, dtype=jnp.int32)

    has_tokens = scheduled > 0
    total = jnp.sum(has_tokens).astype(jnp.int32)
    is_decode = (scheduled == 1) & (context_lens > 1) & has_tokens
    decode = jnp.sum(is_decode).astype(jnp.int32)
    prefill_count = jnp.sum(has_tokens & (~is_decode)).astype(jnp.int32)
    return jnp.stack((decode, decode + prefill_count, total)).astype(jnp.int32)
