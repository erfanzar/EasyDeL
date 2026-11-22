"""Public compressed-cache metadata markers round-trip as empty pytrees (no array computation)."""

import jax
import pytest
from easydel.caching import CompressedWindowMetadata
from easydel.caching.compressed_window import SharedCompressedWindowMetadata

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("metadata_class", [CompressedWindowMetadata, SharedCompressedWindowMetadata])
def test_compressed_metadata_retains_typed_empty_pytree_contract(metadata_class):
    metadata = metadata_class()
    leaves, tree = jax.tree_util.tree_flatten(metadata)
    assert leaves == []
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    assert type(restored) is metadata_class
