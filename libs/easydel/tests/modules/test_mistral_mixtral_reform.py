# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Down-projection reform rule shared by Mixtral and MiniMax (no array model computation)."""

import numpy as np
from easydel.layers import moe_down_projection_reform_param


def test_down_projection_rule_maps_mixtral_w2_and_roundtrips():
    rules = moe_down_projection_reform_param(
        target_prefix="mlp.experts.down_proj", source_prefix="block_sparse_moe.experts.w2"
    )
    rule = rules["mlp.experts.down_proj$"]
    assert [s["name"] for s in rule["splits"]] == ["block_sparse_moe.experts.w2.weight"]

    hf = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    loaded = rule["splits"][0]["spliter"](hf)
    assert loaded.shape == (2, 4, 3)
    np.testing.assert_array_equal(rule["inverse_spliter"](loaded), hf)
