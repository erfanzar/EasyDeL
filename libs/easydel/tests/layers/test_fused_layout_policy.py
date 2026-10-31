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

"""Fused checkpoint policy regressions; numerical tests require an accelerator."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
import torch
from easydel.layers.layouts import (
    FusedColumnLayout,
    FusedSegment,
    dense_gate_up_layout,
    fused_layout_param_specs,
    retp_fused_optimizer_state,
    retp_fused_state,
    tensor_parallel_size,
)
from easydel.layers.linears import ColumnParallelLinear, ParallelMoELinear
from jax import numpy as jnp


class _MixedExpertAndDense(spx.Module):
    """Exercise discovered expert/dense parameters together, not a stub registry."""

    def __init__(self):
        self.gate_up_proj = ParallelMoELinear(
            4,
            4,
            8,
            direction="column",
            use_expert_tensor_mode=True,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=spx.Rngs(0),
        )
        self.shared_gate_up_proj = ColumnParallelLinear(
            4,
            8,
            use_bias=True,
            layout=dense_gate_up_layout(4),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=spx.Rngs(1),
        )


@pytest.mark.parametrize("target_tp", [2, 4])
def test_expert_tensor_mode_native_checkpoint_retp_preserves_expert_channels(target_tp):
    module = _MixedExpertAndDense()
    assert fused_layout_param_specs(module) == [("shared_gate_up_proj", (4, 4))]
    expert = jnp.arange(128, dtype=jnp.float32).reshape(4, 4, 8)
    dense = jnp.arange(32, dtype=jnp.float32).reshape(4, 8)
    expert_bias, dense_bias = jnp.arange(32, dtype=jnp.float32).reshape(4, 8), jnp.arange(8, dtype=jnp.float32)
    original = {
        "parameters.gate_up_proj.weight": expert,
        "parameters.gate_up_proj.bias": expert_bias,
        "parameters.shared_gate_up_proj.weight": dense,
        "parameters.shared_gate_up_proj.bias": dense_bias,
    }
    loaded = retp_fused_state(module, original.copy(), saved_tp=1, target_tp=target_tp)
    # Experts are still contiguous. Dense projections must still interleave:
    # excluding *all* gate_up_proj parameters would not satisfy this test.
    for suffix in ("weight", "bias"):
        key = f"parameters.gate_up_proj.{suffix}"
        np.testing.assert_array_equal(np.asarray(loaded[key]), np.asarray(original[key]))
    channels = np.concatenate(
        [
            np.concatenate(
                [
                    np.arange(rank * (4 // target_tp), (rank + 1) * (4 // target_tp)),
                    np.arange(4 + rank * (4 // target_tp), 4 + (rank + 1) * (4 // target_tp)),
                ]
            )
            for rank in range(target_tp)
        ]
    )
    np.testing.assert_array_equal(
        np.asarray(loaded["parameters.shared_gate_up_proj.weight"]), np.asarray(dense)[:, channels]
    )
    np.testing.assert_array_equal(np.asarray(loaded["parameters.shared_gate_up_proj.bias"]), channels)
    restored = retp_fused_state(module, loaded, saved_tp=target_tp, target_tp=1)
    for key, value in original.items():
        np.testing.assert_array_equal(np.asarray(restored[key]), np.asarray(value))
    moments = {f"mu.{key}": value for key, value in original.items()}
    moved_moments = retp_fused_optimizer_state(module, moments, saved_tp=1, target_tp=target_tp)
    np.testing.assert_array_equal(np.asarray(moved_moments["mu.parameters.gate_up_proj.weight"]), np.asarray(expert))
    moments_roundtrip = retp_fused_optimizer_state(module, moved_moments, saved_tp=target_tp, target_tp=1)
    for key, value in moments.items():
        np.testing.assert_array_equal(np.asarray(moments_roundtrip[key]), np.asarray(value))


@pytest.mark.parametrize("source_is_fused", [False, True])
@pytest.mark.parametrize("segments", [(4, 4), (4, 2, 2)])
def test_noninterleaved_dense_reform_and_runtime_split_agree_at_tp_two(source_is_fused, segments):
    config = SimpleNamespace(mesh=jax.sharding.AbstractMesh((2,), ("tp",)))
    assert tensor_parallel_size(config) == 2
    layout = FusedColumnLayout(
        tuple(FusedSegment(str(i), width, f"source_{i}") for i, width in enumerate(segments)),
        interleave_for_tp=False,
        source_is_fused=source_is_fused,
        target_prefix="projection",
    )
    rules = layout.reform_param(config=config, include_bias=True)
    for suffix in ("weight", "bias"):
        shape = (sum(segments), 4) if suffix == "weight" else (sum(segments),)
        hf = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
        sources = torch.split(hf, segments, dim=0)
        rule = rules[f"projection.{suffix}$"]
        expected_runtime = hf.T if suffix == "weight" else hf
        if source_is_fused:
            runtime = rule["splits"][0]["spliter"](hf)
            exported = rule["inverse_spliter"](torch, runtime)
            np.testing.assert_array_equal(exported.numpy(), hf.numpy())
        else:
            fused = rule["fuser"](torch, *sources)
            runtime = fused.T if suffix == "weight" else fused
            exported = rule["inverse_fuser"](torch, fused)
            for actual, expected in zip(exported, sources, strict=True):
                np.testing.assert_array_equal(actual.numpy(), expected.numpy())
            native_sources = [jnp.asarray((source.T if suffix == "weight" else source).numpy()) for source in sources]
            np.testing.assert_array_equal(np.asarray(rule["native_fuser"](*native_sources)), expected_runtime.numpy())
        np.testing.assert_array_equal(runtime.numpy(), expected_runtime.numpy())
        # Actual runtime split must recover each independently loaded source.
        parts = layout.split(jnp.asarray(runtime.numpy()), config=config, apply_sharding=False)
        for actual, expected in zip(parts, sources, strict=True):
            expected = expected.T if suffix == "weight" else expected
            np.testing.assert_array_equal(np.asarray(actual), expected.numpy())
