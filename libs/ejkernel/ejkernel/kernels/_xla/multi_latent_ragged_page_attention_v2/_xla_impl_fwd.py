# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""XLA fallback for multi-latent ragged paged attention v2."""

from __future__ import annotations

import jax

from ..multi_latent_ragged_page_attention._xla_impl_fwd import DEFAULT_MASK_VALUE
from ..multi_latent_ragged_page_attention._xla_impl_fwd import multi_latent_ragged_page_attention_impl as _v1_impl


def _normalize_case_block_size(
    value: tuple[int, int, int] | list[int] | int | None,
    field_name: str,
) -> int | None:
    """Collapse v2's ``(decode, prefill, mixed)`` block-size triple to a scalar.

    The XLA fallback uses a single generic kernel, so it takes the
    mixed-case tile (index 2) when a triple is provided.

    Args:
        value: Scalar, 3-tuple, or None block-size specification.
        field_name: Name of the field (for error messages).

    Returns:
        Scalar block size or None.

    Raises:
        ValueError: If a tuple with length != 3 is provided.
    """
    if value is None or isinstance(value, int):
        return value
    if len(value) != 3:
        raise ValueError(f"{field_name} must have exactly three entries.")
    return int(value[2])


def multi_latent_ragged_page_attention_v2_impl(
    queries_nope: jax.Array,
    queries_pe: jax.Array,
    keys_values: jax.Array,
    keys_pe: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    *,
    softmax_scale: float = 1.0,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Dispatch v2 calls through the existing v1 XLA MLA ragged fallback.

    Normalises per-case tuple block sizes to scalars (mixed-case tile)
    and forwards all arguments to the v1 ``_v1_impl``.

    Returns:
        Tuple of ``(attention_output, updated_kv_cache)``.
    """
    return _v1_impl(
        queries_nope=queries_nope,
        queries_pe=queries_pe,
        keys_values=keys_values,
        keys_pe=keys_pe,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        distribution=distribution,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_block=_normalize_case_block_size(
            num_kv_pages_per_block,
            "num_kv_pages_per_block",
        ),
        num_queries_per_block=_normalize_case_block_size(
            num_queries_per_block,
            "num_queries_per_block",
        ),
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
    )


__all__ = (
    "DEFAULT_MASK_VALUE",
    "multi_latent_ragged_page_attention_v2_impl",
)
