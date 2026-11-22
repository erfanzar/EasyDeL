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

"""Array-free checkpoint metadata and shared public-export compatibility."""

from dataclasses import dataclass

import pytest
from easydel.layers import indexer
from easydel.layers.layouts import mixtral_moe_reform_param

pytestmark = pytest.mark.cpu_ok


@dataclass(frozen=True)
class _ShapeOnlyTensor:
    shape: tuple[int, ...]

    def swapaxes(self, axis1, axis2):
        shape = list(self.shape)
        shape[axis1], shape[axis2] = shape[axis2], shape[axis1]
        return type(self)(tuple(shape))


def test_mixtral_layout_preserves_all_checkpoint_paths_and_conversion_metadata():
    rules = mixtral_moe_reform_param()
    assert set(rules) == {
        "mlp.gate.weight$",
        "mlp.experts.down_proj$",
        "block_sparse_moe.experts.gate_up_proj.weight$",
    }
    router = rules["mlp.gate.weight$"]
    assert [split["name"] for split in router["splits"]] == ["block_sparse_moe.gate.weight"]
    down = rules["mlp.experts.down_proj$"]
    assert [split["name"] for split in down["splits"]] == ["block_sparse_moe.experts.w2.weight"]
    fused = rules["block_sparse_moe.experts.gate_up_proj.weight$"]
    assert fused["sources"] == ("mlp.experts.gate_up_proj",)
    assert fused["already_converted"] is True
    assert callable(fused["fuser"]) and callable(fused["inverse_fuser"])


@pytest.mark.parametrize(
    ("key", "source_shape", "runtime_shape"),
    [
        ("mlp.gate.weight$", (3, 8), (8, 3)),
        ("mlp.experts.down_proj$", (3, 8, 12), (3, 12, 8)),
    ],
)
def test_mixtral_router_and_down_rules_round_trip_shape_metadata(key, source_shape, runtime_shape):
    rule = mixtral_moe_reform_param()[key]
    source = _ShapeOnlyTensor(source_shape)
    runtime = rule["splits"][0]["spliter"](source)
    assert runtime.shape == runtime_shape
    assert rule["inverse_spliter"](runtime) == source


def test_shared_indexer_layer_norm_is_exported_without_renaming_its_class():
    assert "IndexerLayerNorm" in indexer.__all__
    assert indexer.IndexerLayerNorm.__name__ == "IndexerLayerNorm"
    assert indexer.IndexerLayerNorm.__module__ == "easydel.layers.indexer._indexer"


# Numeric top-k compatibility (including ties and nonfinite scores) is covered
# against independent index masks in the indexer selection tests, rather than
# asserting a particular private delegation path here.
