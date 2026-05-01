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

import re

import jax
import numpy as np
import pytest
import spectrax as spx
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from spectrax import nn
from spectrax.common_types import ColumnWise

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.sharding import (
    TensorLayout,
    coerce_runtime_sharding_resolver,
    metadata_for_layout,
    sharding_for_layout,
)
from easydel.infra.utils import ArrayParam


class _DummyConfig:
    mesh = Mesh(jax.devices()[:1], ("tp",))
    runtime_sharding_resolver = coerce_runtime_sharding_resolver(None, mesh=mesh)


class _DummyLayer(spx.Module):
    def __init__(self):
        super().__init__()
        self.weight = spx.Parameter(
            jnp.ones((4, 4), dtype=jnp.float32),
            sharding=sharding_for_layout(ColumnWise),
        )


class _ConfigBypassMixin:
    def __setattr__(self, name, value):
        if name in {"config", "dtype", "param_dtype", "precision", "rngs"}:
            object.__setattr__(self, name, value)
            return
        super().__setattr__(name, value)


class _DummyModel(_ConfigBypassMixin, EasyDeLBaseModule):
    def __init__(self, config: _DummyConfig, *, rngs: spx.Rngs):
        super().__init__(
            config=config,
            dtype=None,
            param_dtype=None,
            precision=None,
            rngs=rngs,
        )
        self.layers = nn.ModuleList([_DummyLayer() for _ in range(13)])


class _CompoundConfig:
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape((1, 1, 1)), ("fsdp", "sp", "tp"))
    runtime_sharding_resolver = coerce_runtime_sharding_resolver(None, mesh=mesh)


class _CompoundParameterLayer(spx.Module):
    def __init__(self):
        super().__init__()
        self.weight = spx.Parameter(
            jnp.ones((4, 4), dtype=jnp.float32),
            axis_names=(("fsdp", "sp"), "tp"),
        )


class _CompoundArrayParamLayer(spx.Module):
    def __init__(self):
        super().__init__()
        self.weight = ArrayParam.bound(
            shape=(4, 4),
            dtype=jnp.float32,
            init_method="zeros",
            key=None,
            axis_names=(("fsdp", "sp"), "tp"),
        )


class _AbstractParameterLayer(spx.Module):
    def __init__(self):
        super().__init__()
        self.weight = spx.Parameter(
            jax.ShapeDtypeStruct((4, 4), jnp.bfloat16),
            axis_names=(("fsdp", "sp"), "tp"),
        )


class _CompoundModel(_ConfigBypassMixin, EasyDeLBaseModule):
    layer_type: type[spx.Module]

    def __init__(self, config: _CompoundConfig, *, rngs: spx.Rngs, layer_type: type[spx.Module]):
        super().__init__(
            config=config,
            dtype=None,
            param_dtype=None,
            precision=None,
            rngs=rngs,
        )
        self.layer = layer_type()


def test_resolve_shardings_compacts_layer_indices():
    model = _DummyModel(config=_DummyConfig(), rngs=spx.Rngs(0))
    rules = model.resolve_shardings_regex()

    assert rules, "Expected sharding rules to be generated."
    assert rules[-1][0] == ".*"

    path = "layers/5/weight"
    matching = [(pat, spec) for pat, spec in rules if re.match(pat, path)]
    assert matching, "Expected a regex rule matching layers/5/weight."
    assert tuple(matching[0][1])[-1] == "tp"

    exact_rule = "^layers/5/weight$"
    assert all(pat != exact_rule for pat, _ in rules), "Expected compacted regex, not an exact rule."


def test_resolve_shardings_matches_optimizer_prefixed_paths():
    model = _DummyModel(config=_DummyConfig(), rngs=spx.Rngs(0))
    rules = model.resolve_shardings_regex()

    for path in ("mu/layers/5/weight", "0/mu/layers/5/weight"):
        matching = [(pat, spec) for pat, spec in rules if re.match(pat, path)]
        assert matching, "Expected a regex rule matching optimizer-prefixed parameter paths."
        assert tuple(matching[0][1])[-1] == "tp"


def test_metadata_for_layout_uses_spectrax_sharding_for_compound_axes():
    metadata = metadata_for_layout(TensorLayout.from_any(ColumnWise))

    sharding = metadata.get("sharding")
    assert sharding is not None
    assert tuple(sharding.axis_names) == TensorLayout.from_any(ColumnWise).axes
    assert "tensor_layout" not in metadata


def test_builtin_parameter_axis_names_shard_during_model_init():
    model = _CompoundModel(config=_CompoundConfig(), rngs=spx.Rngs(0), layer_type=_CompoundParameterLayer)

    sharding = model.layer.weight.value.sharding

    assert isinstance(sharding, NamedSharding)
    assert sharding.spec == PartitionSpec(("fsdp", "sp"), "tp")
    assert np.array_equal(sharding.mesh.devices, _CompoundConfig.mesh.devices)


def test_array_param_axis_names_shard_during_model_init():
    model = _CompoundModel(config=_CompoundConfig(), rngs=spx.Rngs(0), layer_type=_CompoundArrayParamLayer)

    sharding = model.layer.weight.value.sharding

    assert isinstance(sharding, NamedSharding)
    assert sharding.spec == PartitionSpec(("fsdp", "sp"), "tp")
    assert np.array_equal(sharding.mesh.devices, _CompoundConfig.mesh.devices)


def test_shard_model_preserves_abstract_parameter_leaves_with_sharding():
    model = _CompoundModel(config=_CompoundConfig(), rngs=spx.Rngs(0), layer_type=_AbstractParameterLayer)

    sharded = model.shard_model()
    value = sharded.layer.weight.value

    assert isinstance(value, jax.ShapeDtypeStruct)
    assert isinstance(value.sharding, NamedSharding)
    assert value.sharding.spec == PartitionSpec(("fsdp", "sp"), "tp")
    assert np.array_equal(value.sharding.mesh.devices, _CompoundConfig.mesh.devices)


def test_assert_parameters_materialized_rejects_abstract_trainables():
    model = _CompoundModel(config=_CompoundConfig(), rngs=spx.Rngs(0), layer_type=_AbstractParameterLayer)

    with pytest.raises(ValueError, match=r"parameters/layer\.weight"):
        model.assert_parameters_materialized(context="after checkpoint merge")
