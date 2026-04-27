import jax
import numpy as np
from jax.sharding import Mesh
from spectrax import common_types
from spectrax.sharding import current_axis_rules

from easydel.infra.sharding import CANONICAL_MESH_AXIS_NAMES, coerce_runtime_sharding_resolver


def _single_device_mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1, 1, 1))
    return Mesh(devices, CANONICAL_MESH_AXIS_NAMES)


def test_runtime_sharding_resolver_logical_axis_rules_skip_compound_semantics():
    resolver = coerce_runtime_sharding_resolver(None, mesh=_single_device_mesh())

    with resolver.logical_axis_rules() as active_rules:
        rules = dict(active_rules)

    assert rules["pp"] == "pp"
    assert rules["tp"] == "tp"
    assert rules[common_types.TENSOR_PARALLEL] == "tp"
    assert rules[common_types.SEQUENCE_PARALLEL] == "sp"
    assert common_types.BATCH not in rules
    assert dict(current_axis_rules()) == {}


def test_runtime_sharding_resolver_logical_axis_rules_accept_overrides():
    resolver = coerce_runtime_sharding_resolver(None, mesh=_single_device_mesh())

    with resolver.logical_axis_rules(
        overrides=[
            (common_types.BATCH, "dp"),
            ("tokens", "sp"),
        ]
    ) as active_rules:
        rules = dict(active_rules)

    assert rules[common_types.BATCH] == "dp"
    assert rules["tokens"] == "sp"
    assert dict(current_axis_rules()) == {}
