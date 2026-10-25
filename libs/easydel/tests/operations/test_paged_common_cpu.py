"""Public page-axis decisions and registered cache layout contracts (abstract only)."""

from types import SimpleNamespace

import jax
import pytest
from easydel.axis import ATTN_DP
from easydel.caching import RaggedPagesMetadata
from easydel.operations import OperationRegistry
from easydel.operations.kernels import _paged_common as common
from easydel.operations.kernels import (
    multi_latent_ragged_page_attention,
    paged_flash_attention,
    ragged_page_attention,
    unified_attention,
)
from easydel.utils import flags
from jax import numpy as jnp
from spectrax import common_types as ct

from .test_paged_reference_contract import _cache, _metadata, _storage_shapes

pytestmark = pytest.mark.cpu_ok
ADAPTERS = [ragged_page_attention, multi_latent_ragged_page_attention, paged_flash_attention, unified_attention]


@pytest.mark.parametrize(
    "size, expected",
    [(None, ct.EMPTY), (0, ct.EMPTY), (-2, ct.EMPTY), (1, ct.EMPTY), (2, ATTN_DP), (8, ATTN_DP), ("2", ATTN_DP)],
)
@pytest.mark.parametrize("axis", [common.dp_page_axis] + [m.dp_page_axis for m in ADAPTERS])
def test_dp_axis_selection(size, expected, axis):
    metadata = SimpleNamespace() if size is None else SimpleNamespace(data_parallel_size=size)
    assert axis(SimpleNamespace(metadata=metadata)) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [(None, True), ("1", True), ("yes", True), ("easy", True), ("0", False), ("false", False), ("", False)],
)
def test_existing_flag_semantics(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(flags.EASYDEL_ENABLE_DP_LOCAL_PAGE_PATH, raising=False)
    else:
        monkeypatch.setenv(flags.EASYDEL_ENABLE_DP_LOCAL_PAGE_PATH, raw)
    assert flags.get_bool(flags.EASYDEL_ENABLE_DP_LOCAL_PAGE_PATH) == expected


@pytest.mark.parametrize("layout", ["v2", "v3", "unified"])
@pytest.mark.parametrize("head_dim", [4, 64])
def test_registered_cache_only_attention_preserves_storage_layout(layout, head_dim):
    """Equivalent helper wrappers are allowed; the public cache result is the contract."""
    name = "unified_attention" if layout == "unified" else "paged_attention_reference"
    operation = OperationRegistry.create(name, _metadata())
    storage_shapes = _storage_shapes(layout, head_dim)
    descriptors = [jax.ShapeDtypeStruct((4, 2, head_dim), jnp.float32)]
    descriptors += [jax.ShapeDtypeStruct(shape, jnp.float32) for shape in storage_shapes]

    def forward(query, *storage):
        cache = _cache(layout, head_dim, storage)
        runtime = RaggedPagesMetadata(
            jnp.arange(8).reshape(4, 2) % 4, jnp.full((4,), 2), jnp.arange(5), jnp.full((4,), 2), page_size=2
        )
        result = operation(
            query=query, cache_view=cache, cache_metadata=runtime, normalize_output=False, compute_dtype=jnp.float32
        )
        updated = result.cache_view
        arrays = (updated.key_cache, updated.value_cache) if layout == "unified" else (updated.kv_pages,)
        assert type(updated) is type(cache)
        return result.attention_outputs, arrays

    output, storage = jax.eval_shape(forward, *descriptors)
    assert (output.shape, output.dtype) == ((4, 2, head_dim), jnp.float32)
    assert [(array.shape, array.dtype) for array in storage] == [(shape, jnp.float32) for shape in storage_shapes]
