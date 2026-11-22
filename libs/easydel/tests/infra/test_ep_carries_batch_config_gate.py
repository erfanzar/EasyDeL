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

"""The config-time ep-carries-batch gate agrees with the MoE module gate and the layout planner.

``sp_is_ep_bound`` folds sp into the expert axis, which only disqualifies ep-carried batch when the
sp axis is non-trivial. The config must extend ``PartitionAxis.batch_axis`` with ep exactly when the
module dispatch and the planner consider the path active.
"""

import jax
import pytest
import spectrax as spx
from easydel.layers.moe import estimate_moe_layout
from easydel.layers.moe._communication_utils import DP, resolve_eformer_axis
from jax import numpy as jnp

AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


def _block(dims):
    from easydel.modules.gpt_oss.gpt_oss_configuration import GptOssConfig
    from easydel.modules.gpt_oss.modeling_gpt_oss import GptOssMLP

    config = GptOssConfig(
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        num_local_experts=8,
        num_experts_per_tok=2,
        moe_method="fused_moe",
        moe_force_xla_gmm=True,
        sharding_axis_dims=dims,
        sharding_axis_names=AXIS_NAMES,
        scan_layers=False,
    )
    config.add_basic_configurations(
        use_ring_of_experts=False,
        fsdp_is_ep_bound=False,
        sp_is_ep_bound=True,
        moe_ep_carries_batch=True,
    )
    return GptOssMLP(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))


@pytest.mark.parametrize(
    ("dims", "expected_active"),
    [
        pytest.param((1, 1, 2, 2, 1, 1), True, id="sp1-bound-extends"),
        pytest.param((1, 1, 1, 2, 1, 2), False, id="sp2-bound-keeps"),
    ],
)
def test_config_batch_axis_agrees_with_module_and_planner(dims, expected_active):
    if jax.device_count() < 4:
        pytest.skip("needs >=4 devices")
    block = _block(dims)
    config_extended = "ep" in tuple(block.config.partition_axis.batch_axis)

    with block.config.mesh:
        expert_mesh = block._active_auto_expert_mesh(arr=None)
        module_axes = block._moe_batch_axis_names(expert_mesh, resolve_eformer_axis(DP, block.runtime_sharding_resolver))
    module_active = "ep" in module_axes

    sizes = {k: v for k, v in zip(AXIS_NAMES, dims, strict=True) if k != "pp"}
    kwargs = dict(
        tokens_per_device=64,
        num_experts_per_tok=2,
        hidden_size=32,
        num_experts=8,
        expert_params_bytes=1 << 20,
        dtype_bytes=4,
        fsdp_is_ep_bound=False,
        sp_is_ep_bound=True,
        use_ring_of_experts=False,
        chunk_size=None,
    )
    planner_active = (
        estimate_moe_layout(sizes, ep_carries_batch=True, **kwargs).local_tokens
        != estimate_moe_layout(sizes, ep_carries_batch=False, **kwargs).local_tokens
    )

    assert config_extended is expected_active, block.config.partition_axis.batch_axis
    assert module_active is expected_active
    assert planner_active is expected_active
    if expected_active:
        assert tuple(block.config.partition_axis.batch_axis) == ("fsdp", "dp", "ep")
