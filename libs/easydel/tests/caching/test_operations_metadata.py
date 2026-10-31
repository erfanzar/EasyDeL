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

"""Regression tests: ``OperationsMetadata`` can be constructed and round-trips through ``unwrap_metadata``."""

import jax
import pytest
from easydel.caching import OperationsMetadata, RaggedPagesMetadata, RecurrentMetadata, unwrap_metadata
from easydel.caching.hybrid import HybridMetadata
from easydel.caching.transformer import TransformerMetadata

pytestmark = pytest.mark.cpu_ok


def test_wrap_and_unwrap_transformer_metadata():
    inner = TransformerMetadata(postpadded=True)
    wrapped = OperationsMetadata.wrap(inner)
    assert wrapped.cache_type == "transformer"
    assert wrapped.get_inner() is inner
    assert unwrap_metadata(wrapped, "transformer") is inner
    assert unwrap_metadata(wrapped) is inner


def test_wrap_hybrid_metadata_unwraps_to_transformer_view_metadata():
    wrapped = OperationsMetadata.wrap(HybridMetadata(postpadded=True))
    assert wrapped.cache_type == "hybrid"
    converted = unwrap_metadata(wrapped, "transformer")
    assert isinstance(converted, TransformerMetadata)
    assert converted.postpadded is True


def test_wrap_is_idempotent_and_none_safe():
    wrapped = OperationsMetadata.wrap(TransformerMetadata())
    assert OperationsMetadata.wrap(wrapped) is wrapped
    assert OperationsMetadata.wrap(None) is None
    assert unwrap_metadata(None, "transformer") is None


def test_wrap_rejects_unknown_metadata_types():
    with pytest.raises(TypeError):
        OperationsMetadata.wrap(object())


@pytest.mark.parametrize(
    "field, inner",
    [
        ("transformer", TransformerMetadata()),
        ("hybrid", HybridMetadata()),
        ("ragged", RaggedPagesMetadata(pages_tables=None, context_lens=None, query_start_loc=None, num_seqs=None)),
        ("recurrent", RecurrentMetadata()),
    ],
)
def test_documented_public_wrapper_constructor_dispatches_all_metadata_types(field, inner):
    wrapped = OperationsMetadata(**{field: inner})
    assert wrapped.cache_type == field
    assert wrapped.get_inner() is inner
    assert unwrap_metadata(wrapped, field) is inner
    assert OperationsMetadata.wrap(inner).get_inner() is inner


@pytest.mark.parametrize("metadata_class", [TransformerMetadata, HybridMetadata])
@pytest.mark.parametrize("wrapped", [False, True])
def test_postpadding_flag_is_static_while_positions_remain_pytree_leaves(metadata_class, wrapped):
    # Python positions exercise the public pytree contract without allocating or
    # executing arrays. A static flag must change the treedef (the JIT cache key),
    # not become a dynamic boolean leaf alongside indexes/starts.
    trees = []
    for postpadded in (False, True):
        metadata = metadata_class(postpadded=postpadded, starts=(2,), indexes=(6,))
        if wrapped:
            metadata = OperationsMetadata.wrap(metadata)
        leaves, tree = jax.tree_util.tree_flatten(metadata)
        assert leaves == [2, 6]
        restored = jax.tree_util.tree_unflatten(tree, leaves)
        dense_metadata = unwrap_metadata(restored, "transformer")
        assert dense_metadata.postpadded is postpadded
        assert dense_metadata.starts == (2,)
        assert dense_metadata.indexes == (6,)
        trees.append(tree)
    assert trees[0] != trees[1]
