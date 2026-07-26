# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""v3 ragged-page KV storage layout tests.

The v3 page pool stores keys and values interleaved on a *combined head* axis,
which is the contract ``kv_cache_update`` writes against::

    new_kv_tokens: [total_tokens, num_combined_kv_heads, head_dim]
    num_combined_kv_heads == 2 * num_kv_heads

with the head dim padded up to :data:`HEAD_DIM_ALIGNMENT`. These tests pin that
allocation, keep the HBM page-size accounting in step with it, and record the
one geometry where the allocated layout does not match the kernel that ejkernel
dispatches for it (``head_dim == 64``).
"""

from __future__ import annotations

import pytest
from easydel.caching import RaggedPagesCacheConfig
from easydel.caching.ragged_page.cache import (
    HEAD_DIM_ALIGNMENT,
    align_to_multiple,
    get_dtype_packing,
    get_page_size_bytes,
)
from jax import numpy as jnp

GEOMETRIES = [(8, 64), (2, 64), (8, 128), (4, 128), (8, 256)]


def _config(*, num_kv_heads: int, head_dim: int, page_size: int = 64) -> RaggedPagesCacheConfig:
    """Build a minimal v3 ragged-page config."""
    return RaggedPagesCacheConfig(
        num_hidden_layers=1,
        max_model_length=256,
        num_kv_heads=num_kv_heads,
        k_headdim=head_dim,
        v_headdim=head_dim,
        num_pages=8,
        max_num_pages_per_req=4,
        page_size=page_size,
        _kvdtype_str="bf16",
        version="v3",
    )


@pytest.mark.parametrize(("num_kv_heads", "head_dim"), GEOMETRIES)
def test_v3_pages_match_the_cache_update_writer_contract(num_kv_heads, head_dim):
    """Allocation must equal combined-KV-heads x padded head dim, per the writer."""
    config = _config(num_kv_heads=num_kv_heads, head_dim=head_dim)
    shape, axes = config.get_shape_and_axes()
    num_pages, page_size, kv_groups, packing, storage_head_dim = shape
    assert len(axes) == len(shape)
    assert num_pages == 8
    assert packing == get_dtype_packing(jnp.bfloat16)

    expected_combined_heads = align_to_multiple(num_kv_heads * 2, packing)
    expected_head_dim = align_to_multiple(head_dim, HEAD_DIM_ALIGNMENT)
    assert kv_groups * packing == expected_combined_heads
    assert storage_head_dim == expected_head_dim
    assert page_size * kv_groups * packing * storage_head_dim == page_size * expected_combined_heads * expected_head_dim


@pytest.mark.parametrize(("num_kv_heads", "head_dim"), GEOMETRIES)
def test_page_size_bytes_agrees_with_the_allocated_shape(num_kv_heads, head_dim):
    """Pool sizing must not disagree with the real allocation in either direction.

    Over-reporting leaves HBM unused (fewer pages than fit); under-reporting
    oversubscribes the budget.
    """
    page_size = 64
    _, shaped_page_size, kv_groups, packing, storage_head_dim = _config(
        num_kv_heads=num_kv_heads, head_dim=head_dim, page_size=page_size
    ).get_shape_and_axes()[0]
    allocated_bytes = shaped_page_size * kv_groups * packing * storage_head_dim * jnp.finfo(jnp.bfloat16).bits // 8

    assert get_page_size_bytes(page_size, num_kv_heads, head_dim, jnp.bfloat16) == allocated_bytes


@pytest.mark.parametrize(("num_kv_heads", "head_dim"), [(8, 128), (4, 128), (8, 256)])
def test_padding_overhead_is_absent_when_the_head_dim_is_already_aligned(num_kv_heads, head_dim):
    """Aligned head dims store exactly the K/V bytes the geometry requires."""
    _, page_size, kv_groups, packing, storage_head_dim = _config(
        num_kv_heads=num_kv_heads, head_dim=head_dim
    ).get_shape_and_axes()[0]

    minimum = page_size * num_kv_heads * head_dim * 2
    assert page_size * kv_groups * packing * storage_head_dim == minimum


@pytest.mark.xfail(
    strict=True,
    reason=(
        "head_dim 64 allocates 2x the required KV bytes and the dispatched kernel rejects it. "
        "ejkernel dispatches ragged_page_attention_v3 on the query head dim, so head_dim 64 lands "
        "in _pallas_impl_fwd_h64, which reads pages as [pages, page_size, "
        "align_to(num_kv_heads, packing)/packing, packing, 2*head_dim] -- K|V packed in a 128-wide "
        "head-dim axis. EasyDeL instead allocates the combined-heads layout with head_dim padded "
        "64 -> 128, which both doubles the bytes and fails the kernel's head-count check. Fixing "
        "the allocation alone is not enough: kv_cache_update writes the combined-heads layout, so "
        "writer, reader and dispatch have to be reconciled together (verified on TPU: changing only "
        "the shape makes the kernel accept the pages and produce wrong tokens)."
    ),
)
@pytest.mark.parametrize(("num_kv_heads", "head_dim"), [(8, 64), (2, 64)])
def test_head_dim_64_pages_satisfy_the_dispatched_kernel(num_kv_heads, head_dim):
    """head_dim-64 pages should match what ejkernel's h64 kernel expects."""
    _, page_size, kv_groups, packing, storage_head_dim = _config(
        num_kv_heads=num_kv_heads, head_dim=head_dim
    ).get_shape_and_axes()[0]

    assert storage_head_dim == head_dim * 2
    assert kv_groups * packing == align_to_multiple(num_kv_heads, packing)
    assert page_size * kv_groups * packing * storage_head_dim == page_size * num_kv_heads * head_dim * 2
