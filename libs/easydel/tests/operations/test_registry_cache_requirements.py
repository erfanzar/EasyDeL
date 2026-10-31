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

"""Cache requirement overrides on registry-created operations are independent."""

from types import SimpleNamespace

import jax
import pytest
from easydel.infra import EasyDeLBaseConfig
from easydel.operations import OperationExecutor, OperationMetadata, OperationRegistry
from spectrax import common_types


@pytest.mark.cpu_ok
def test_registry_cache_override_does_not_leak_to_shared_metadata_or_existing_operations():
    metadata = SimpleNamespace(requires_cache=None)
    cached = OperationRegistry.create("ragged_page_attention_v2", metadata)
    cacheless = OperationRegistry.create("ragged_page_attention_v2", metadata, requires_cache=False)
    assert metadata.requires_cache is None
    assert cached.metadata is cacheless.metadata
    assert OperationExecutor.from_operations(prefill=cached).requires_cache is True
    assert OperationExecutor.from_operations(prefill=cacheless).requires_cache is False

    cacheful = OperationRegistry.create("ragged_page_attention_v2", metadata, requires_cache=True)
    assert OperationExecutor.from_operations(prefill=cached).requires_cache is True
    assert OperationExecutor.from_operations(prefill=cacheless).requires_cache is False
    assert OperationExecutor.from_operations(prefill=cacheful).requires_cache is True


def _require_tp2_devices():
    """Skip (lazily, so collection never initializes a backend) unless a tp=2 mesh fits."""
    count = len(jax.devices())
    if count < 2 or count % 2:
        pytest.skip(f"needs an even device count >= 2 for a tp=2 mesh, found {count}")


@pytest.mark.cpu_ok
def test_sink_logits_sharding_places_heads_on_first_axis():
    _require_tp2_devices()
    config = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, -1, 1, 2, 1))
    metadata = OperationMetadata(runtime_dtype="float32", base_config=config)
    sinks = SimpleNamespace(ndim=2, shape=(2, 1))
    specs = metadata.get_shardings(common_types.MODE_PREFILL, softmax_aux=sinks, num_q_heads=8, num_kv_heads=2)
    assert specs.query[2] is not None
    assert specs.softmax_aux[0] == specs.query[2]
    assert specs.softmax_aux[1] is None


@pytest.mark.cpu_ok
def test_shared_sink_logits_are_replicated_even_when_their_length_divides_tp():
    """Only the global head count distinguishes a per-head vector from shared sinks."""
    _require_tp2_devices()
    config = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, -1, 1, 2, 1))
    metadata = OperationMetadata(runtime_dtype="float32", base_config=config)
    mode = common_types.MODE_PREFILL

    def spec_for(shape):
        sinks = SimpleNamespace(ndim=len(shape), shape=shape)
        return metadata.get_shardings(mode, softmax_aux=sinks, num_q_heads=8, num_kv_heads=2).softmax_aux

    assert metadata.get_shardings(mode).query[2] is not None  # guards against vacuous None == None
    assert spec_for((8,))[0] == metadata.get_shardings(mode).query[2]
    assert spec_for((4,))[0] is None  # four sinks shared by eight query heads
    assert spec_for((2,))[0] is None  # two sinks, not two KV-head parameters
    assert spec_for((8, 3))[0] == metadata.get_shardings(mode).query[2]
    assert spec_for((2, 3))[0] == metadata.get_shardings(mode).query[2]
    assert spec_for((1, 3))[0] is None
    assert spec_for((4, 3))[0] is None  # not Hq/Hkv; never partition the sink axis
