"""eSurge request/KV data-parallel axis selection.

Only ``dp`` is a pure data axis in EasyDeL's ``("pp","dp","fsdp","ep","tp","sp")``
convention. Pointing request/KV-page data parallelism at a model-sharding axis of
size > 1 makes attention split requests over the same mesh axis the weights are
split on, so activations reshard between attention and the model body on every
layer. Measured on v5p-8 with Qwen/Qwen3.5-4B (1024-token prompts, 256-token
completions, concurrency 8, identical 1,024-page KV pool): a 4-way ``fsdp`` axis
costs 47.1 ms per decode step against 12.2 ms for a size-1 ``dp`` axis, i.e. 134
vs 340 output tok/s, and the entire gap is device forward time.
"""

from __future__ import annotations

import pytest
from easydel.inference.esurge.esurge_engine import (
    MODEL_SHARDING_MESH_AXES,
    PURE_DATA_MESH_AXIS,
    data_parallel_axis_shards_model,
)

# The SAO / post-training mesh: every device on the FSDP axis.
FSDP4_MESH = {"pp": 1, "dp": 1, "fsdp": 4, "ep": 1, "tp": 1, "sp": 1}
# True request-level DP: replicated weights, four data shards.
DP4_MESH = {"pp": 1, "dp": 4, "fsdp": 1, "ep": 1, "tp": 1, "sp": 1}


def test_pure_data_axis_is_never_reported_as_model_sharding() -> None:
    """``dp`` is safe at any size, including when it is the only populated axis."""
    assert data_parallel_axis_shards_model(PURE_DATA_MESH_AXIS, FSDP4_MESH) is False
    assert data_parallel_axis_shards_model(PURE_DATA_MESH_AXIS, DP4_MESH) is False


def test_fsdp_axis_on_the_sao_mesh_is_reported() -> None:
    """The measured 2.6x-slower configuration must be detectable at config time."""
    assert data_parallel_axis_shards_model("fsdp", FSDP4_MESH) is True


def test_size_one_model_axis_is_not_reported() -> None:
    """A size-1 axis splits no requests, so it costs nothing and must not warn."""
    assert data_parallel_axis_shards_model("fsdp", DP4_MESH) is False
    assert data_parallel_axis_shards_model("tp", FSDP4_MESH) is False


@pytest.mark.parametrize("axis", MODEL_SHARDING_MESH_AXES)
def test_every_model_sharding_axis_is_reported_when_populated(axis: str) -> None:
    """Every non-``dp`` axis carries model/sequence/expert/stage sharding."""
    assert data_parallel_axis_shards_model(axis, {**FSDP4_MESH, "fsdp": 1, axis: 4}) is True


def test_unknown_axis_is_not_reported() -> None:
    """An axis absent from the mesh is handled by the existing membership warning."""
    assert data_parallel_axis_shards_model("not_a_mesh_axis", FSDP4_MESH) is False


def test_pure_data_axis_is_not_also_listed_as_model_sharding() -> None:
    """The two axis groups must stay disjoint or the predicate contradicts itself."""
    assert PURE_DATA_MESH_AXIS not in MODEL_SHARDING_MESH_AXES
