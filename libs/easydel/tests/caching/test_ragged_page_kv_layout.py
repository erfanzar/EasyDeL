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

The page pool EasyDeL allocates has to be exactly the shape the dispatched
ejkernel kernel reads. ejkernel routes ``ragged_page_attention_v3`` on the query
head dim, and the two implementations disagree about where the key/value pair
lives:

* ``head_dim == 64`` -> ``_pallas_impl_fwd_h64``: K and V are concatenated into
  the 128-wide last axis, so the head axis carries ``num_kv_heads``.
* otherwise -> ``_pallas_impl_fwd``: the pair is interleaved on the head axis
  (``num_kv_heads * 2``) with the head dim padded to 128.

Rather than restating either formula, these tests compare against ejkernel's own
``get_kv_cache_shape`` helpers, so a layout change on either side fails loudly
instead of surfacing as a shape rejection (or, worse, silently wrong attention)
at serving time.
"""

from __future__ import annotations

import pytest
from easydel.caching import RaggedPagesCacheConfig
from easydel.caching.ragged_page.cache import get_page_size_bytes, kv_pair_shares_head_dim_axis
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._pallas_impl_fwd import (
    get_kv_cache_shape as ejkernel_shape_hd128,
)
from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._pallas_impl_fwd_h64 import (
    get_kv_cache_shape as ejkernel_shape_hd64,
)
from jax import numpy as jnp

GEOMETRIES = [(8, 64), (2, 64), (16, 64), (8, 128), (4, 128), (8, 256)]
NUM_PAGES = 8
PAGE_SIZE = 64


def _config(*, num_kv_heads: int, head_dim: int) -> RaggedPagesCacheConfig:
    """Build a minimal v3 ragged-page config."""
    return RaggedPagesCacheConfig(
        num_hidden_layers=1,
        max_model_length=256,
        num_kv_heads=num_kv_heads,
        k_headdim=head_dim,
        v_headdim=head_dim,
        num_pages=NUM_PAGES,
        max_num_pages_per_req=4,
        page_size=PAGE_SIZE,
        _kvdtype_str="bf16",
        version="v3",
    )


def _ejkernel_shape(num_kv_heads: int, head_dim: int) -> tuple[int, ...]:
    """Canonical page shape for the kernel ejkernel dispatches at this head dim."""
    helper = ejkernel_shape_hd64 if head_dim == 64 else ejkernel_shape_hd128
    return tuple(helper(NUM_PAGES, PAGE_SIZE, num_kv_heads, head_dim, jnp.bfloat16))


@pytest.mark.parametrize(("num_kv_heads", "head_dim"), GEOMETRIES)
def test_allocated_pages_match_the_dispatched_kernel_shape(num_kv_heads, head_dim):
    """The allocated pool must equal what the dispatched kernel expects to read."""
    allocated, axes = _config(num_kv_heads=num_kv_heads, head_dim=head_dim).get_shape_and_axes()

    assert tuple(allocated) == _ejkernel_shape(num_kv_heads, head_dim)
    assert len(axes) == len(allocated)


@pytest.mark.parametrize(("num_kv_heads", "head_dim"), GEOMETRIES)
def test_page_volume_is_the_minimum_for_the_geometry(num_kv_heads, head_dim):
    """Neither layout should cost more than the K/V bytes the geometry needs."""
    _, page_size, kv_groups, packing, storage_head_dim = _config(
        num_kv_heads=num_kv_heads, head_dim=head_dim
    ).get_shape_and_axes()[0]

    elements = page_size * kv_groups * packing * storage_head_dim
    minimum = page_size * num_kv_heads * head_dim * 2
    assert elements == minimum, f"{elements} elements per page for {num_kv_heads}x{head_dim}, minimum {minimum}"


@pytest.mark.parametrize(("num_kv_heads", "head_dim"), GEOMETRIES)
def test_page_size_bytes_agrees_with_the_allocated_shape(num_kv_heads, head_dim):
    """Pool sizing must match the real allocation, or HBM is mis-budgeted.

    Over-reporting leaves HBM unused (fewer pages than fit); under-reporting
    oversubscribes the budget.
    """
    _, page_size, kv_groups, packing, storage_head_dim = _config(
        num_kv_heads=num_kv_heads, head_dim=head_dim
    ).get_shape_and_axes()[0]
    allocated_bytes = page_size * kv_groups * packing * storage_head_dim * jnp.finfo(jnp.bfloat16).bits // 8

    assert get_page_size_bytes(page_size, num_kv_heads, head_dim, jnp.bfloat16) == allocated_bytes


def test_only_head_dim_64_concatenates_the_pair_in_the_head_dim_axis():
    """Guard the predicate that selects between the two layouts."""
    assert kv_pair_shares_head_dim_axis(64)
    assert not kv_pair_shares_head_dim_axis(128)
    assert not kv_pair_shares_head_dim_axis(256)
    assert not kv_pair_shares_head_dim_axis(96)
