import jax
import numpy as np
import spectrax as spx
from jax.sharding import Mesh, PartitionSpec
from spectrax import common_types
from spectrax.sharding import current_axis_rules

from easydel.infra.etils import EasyDeLPlatforms
from easydel.infra.sharding import CANONICAL_MESH_AXIS_NAMES, AxisPolicy, coerce_runtime_sharding_resolver
from easydel.operations._operation_meta import OperationMetadata


def _single_device_mesh() -> Mesh:
    devices = np.asarray(jax.devices("cpu")[:1], dtype=object).reshape((1, 1, 1, 1, 1, 1))
    return Mesh(devices, CANONICAL_MESH_AXIS_NAMES)


def test_runtime_sharding_resolver_logical_axis_rules_preserve_compound_semantics():
    resolver = coerce_runtime_sharding_resolver(None, mesh=_single_device_mesh())

    with resolver.logical_axis_rules() as active_rules:
        rules = dict(active_rules)

    assert rules["pp"] == "pp"
    assert rules["tp"] == "tp"
    assert rules[common_types.TENSOR_PARALLEL] == "tp"
    assert rules[common_types.SEQUENCE_PARALLEL] == "sp"
    assert rules[common_types.BATCH] == ("fsdp", "dp")
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


def test_runtime_sharding_resolver_preserves_compound_batch_axis_in_specs():
    resolver = coerce_runtime_sharding_resolver(None, mesh=_single_device_mesh())

    spec = resolver.resolve(
        axes=[
            common_types.BATCH,
            common_types.QUERY_LENGTH,
            common_types.HEAD,
            common_types.HEAD_DIM,
        ],
        mode=common_types.MODE_TRAIN,
    )

    assert spec == PartitionSpec(("fsdp", "dp"), "sp", "tp", None)


def test_runtime_sharding_resolver_matches_partition_axis_for_all_builtin_semantics():
    resolver = coerce_runtime_sharding_resolver(None, mesh=_single_device_mesh())
    semantic_axes = [
        common_types.DATA_PARALLEL,
        common_types.FULLY_SHARDED_DATA_PARALLEL,
        common_types.TENSOR_PARALLEL,
        common_types.SEQUENCE_PARALLEL,
        common_types.EXPERT_PARALLEL,
        common_types.BATCH,
        common_types.LENGTH,
        common_types.QUERY_LENGTH,
        common_types.KV_LENGTH,
        common_types.HEAD,
        common_types.KV_HEAD,
        common_types.EMBED,
        common_types.MLP_INTERMEDIATE,
        common_types.VOCAB,
        common_types.EXPERT,
        common_types.EXPERT_GATE,
        common_types.HEAD_DIM,
        common_types.KV_HEAD_DIM,
        common_types.BIAS_HEAD_SEQ,
        common_types.BIAS_KV_SEQ,
    ]

    for mode in (common_types.MODE_TRAIN, common_types.MODE_DECODE):
        for axis in semantic_axes:
            expected = resolver.axis_policy.resolve_spec([axis], mode)
            actual = resolver.resolve(axes=[axis], mode=mode)
            assert actual == expected, f"{axis=} {mode=}"


def test_runtime_sharding_resolver_preserves_custom_non_batch_compound_axes():
    policy = AxisPolicy.from_partition_axis(
        {
            "hidden_state_axis": ("tp", "sp"),
            "head_axis": ("tp", "sp"),
        }
    )
    resolver = coerce_runtime_sharding_resolver(policy, mesh=_single_device_mesh())

    assert resolver.resolve(axes=[common_types.EMBED], mode=common_types.MODE_TRAIN) == PartitionSpec(("tp", "sp"))
    assert resolver.resolve(axes=[common_types.HEAD], mode=common_types.MODE_TRAIN) == PartitionSpec(("tp", "sp"))


def test_runtime_sharding_resolver_flattens_compound_entries_in_specs():
    resolver = coerce_runtime_sharding_resolver(None, mesh=_single_device_mesh())

    spec = resolver.resolve(
        axes=[
            (common_types.BATCH, common_types.SEQUENCE_PARALLEL),
            common_types.HEAD,
        ],
        mode=common_types.MODE_TRAIN,
    )

    assert spec == PartitionSpec(("fsdp", "dp", "sp"), "tp")


def test_runtime_sharding_resolver_variable_path_preserves_semantic_compounds():
    resolver = coerce_runtime_sharding_resolver(None, mesh=_single_device_mesh())
    var = spx.Variable(
        np.ones((1, 1), dtype=np.float32),
        metadata={"sharding": spx.Sharding(axis_names=(common_types.BATCH, common_types.HEAD))},
    )

    sharding = resolver.named_sharding_for_variable(var)

    assert sharding is not None
    assert sharding.spec == PartitionSpec(("fsdp", "dp"), "tp")


def test_runtime_sharding_resolver_variable_path_preserves_custom_non_batch_compounds():
    policy = AxisPolicy.from_partition_axis({"hidden_state_axis": ("tp", "sp")})
    resolver = coerce_runtime_sharding_resolver(policy, mesh=_single_device_mesh())
    var = spx.Variable(
        np.ones((1,), dtype=np.float32),
        metadata={"sharding": spx.Sharding(axis_names=(common_types.EMBED,))},
    )

    sharding = resolver.named_sharding_for_variable(var)

    assert sharding is not None
    assert sharding.spec == PartitionSpec(("tp", "sp"))


def test_operation_metadata_attention_shardings_preserve_compound_batch_axis():
    metadata = OperationMetadata(
        runtime_dtype=np.float32,
        platform=EasyDeLPlatforms.JAX,
        _stored_mesh=_single_device_mesh(),
    )

    shardings = metadata.get_shardings(common_types.MODE_TRAIN, layout="bthd")

    assert shardings.query3d == PartitionSpec(("fsdp", "dp"), "tp", None)
    assert shardings.query == PartitionSpec(("fsdp", "dp"), "sp", "tp", None)
    assert shardings.bias == PartitionSpec(("fsdp", "dp"), None, "sp", None)
    assert shardings.mask == PartitionSpec(("fsdp", "dp"), None, "sp", None)
    assert shardings.q_segment_ids == PartitionSpec(("fsdp", "dp"), "sp")


def test_operation_metadata_bhtd_shardings_preserve_compound_batch_axis():
    metadata = OperationMetadata(
        runtime_dtype=np.float32,
        platform=EasyDeLPlatforms.JAX,
        _stored_mesh=_single_device_mesh(),
    )

    shardings = metadata.get_shardings(common_types.MODE_TRAIN, layout="bhtd")

    assert shardings.query == PartitionSpec(("fsdp", "dp"), "tp", "sp", None)
    assert shardings.key == PartitionSpec(("fsdp", "dp"), "tp", "sp", None)
    assert shardings.value == PartitionSpec(("fsdp", "dp"), "tp", "sp", None)
    assert shardings.q_segment_ids == PartitionSpec(("fsdp", "dp"), "sp")
