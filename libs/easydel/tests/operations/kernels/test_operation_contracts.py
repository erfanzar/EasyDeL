"""Non-computational coverage of operation configuration and dispatch contracts."""

from types import SimpleNamespace

import jax
import pytest
from easydel.operations._base_operation import BaseOperation
from easydel.operations._operation_impl import axis_entry_shard_factor, dp_page_axis, run_mosaic_kernel
from easydel.operations.executor import OperationExecutor
from easydel.operations.kernels._compressed_window_common import configured_platform, resolve_platform
from easydel.operations.requirements import ExecutionMode, OperationRequirements
from ejkernel.modules.operations.configs import BaseOperationConfig
from jax.sharding import PartitionSpec as Ps

pytestmark = pytest.mark.cpu_ok


class ExampleOperation:
    """Array-free stand-in implementing the public executor protocol."""

    def __init__(self, name):
        self.name = name
        self.requirements = OperationRequirements.default()

    def get_impl_name(self):
        return self.name

    def get_requirements(self, mode):
        return self.requirements


@pytest.mark.parametrize("prefill,decode", [(None, None), ("prefill", None), (None, "decode"), ("prefill", "decode")])
def test_executor_public_factory_names_and_combined_requirements(prefill, decode):
    prefill_op = None if prefill is None else ExampleOperation((prefill, "alias"))
    decode_op = None if decode is None else ExampleOperation(decode)
    executor = OperationExecutor.from_flexible_attention(SimpleNamespace(impl=prefill_op, impl_decode=decode_op))
    assert executor.is_valid == (prefill is not None or decode is not None)
    assert executor.get_operation(ExecutionMode.PREFILL) is prefill_op
    assert executor.get_operation(ExecutionMode.DECODE) is (decode_op or prefill_op)
    assert executor.get_operation_name(ExecutionMode.PREFILL) == prefill
    assert executor.get_operation_name(ExecutionMode.DECODE) == (decode or prefill)
    requirements = executor.get_combined_requirements()
    assert requirements.metadata == OperationRequirements.default().metadata
    assert requirements.cache == OperationRequirements.default().cache


@pytest.mark.parametrize("config", [None, {"platform": "xla"}, BaseOperationConfig(platform="xla")])
def test_compressed_window_platform_configuration_and_call_override(config):
    metadata = SimpleNamespace(get_operation_config=lambda name: config)
    default = configured_platform(metadata, "compressed_window_attention")
    assert resolve_platform(None, default) == (None if config is None else "xla")
    assert resolve_platform("PALLAS", default) == "pallas"
    assert resolve_platform("auto", default) is None


@pytest.mark.parametrize("dp", [1, 2, 8])
def test_dp_page_axis_and_mesh_products(dp):
    from easydel.axis import ATTN_DP
    from spectrax import common_types

    view = SimpleNamespace(metadata=SimpleNamespace(data_parallel_size=dp))
    assert dp_page_axis(view) == (ATTN_DP if dp > 1 else common_types.EMPTY)
    mesh = SimpleNamespace(shape={"dp": dp, "tp": 2})
    assert axis_entry_shard_factor(mesh, ("dp", "tp")) == dp * 2
    assert axis_entry_shard_factor(None, "tp") == 1
    assert axis_entry_shard_factor(mesh, None) == 1


def test_backend_accessor_remains_available(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    assert BaseOperation.current_backend(None) == "cpu"


@pytest.mark.parametrize("platform", [None, "jax"])
def test_operation_metadata_resolves_optional_platform_without_base_config(platform):
    from easydel.operations import OperationMetadata

    # Metadata/resolver construction uses mesh geometry only, not device arrays.
    mesh = SimpleNamespace(shape={"tp": 1}, empty=False)
    kwargs = {} if platform is None else {"platform": platform}
    metadata = OperationMetadata(runtime_dtype="float32", _stored_mesh=mesh, **kwargs)
    assert metadata.platform == platform
    assert metadata.mesh is mesh
    assert metadata.get_operation_config("flash_attn2") is None


@pytest.mark.parametrize(
    "mesh_size,manual,auto_size,dim,layout,expect_kernel",
    [
        (1, False, 1, 4, True, True),
        (2, True, 1, 4, True, True),
        (2, True, 2, 4, True, False),
        (2, False, 1, 3, True, False),
        (2, False, 1, 4, False, False),
        (2, False, 1, 4, True, True),
    ],
)
def test_mosaic_dispatch_outputs_without_arrays(monkeypatch, mesh_size, manual, auto_size, dim, layout, expect_kernel):
    ambient = SimpleNamespace(
        empty=not manual,
        manual_axes={"tp"} if manual else set(),
        axis_names=("tp", "auto"),
        shape={"tp": 2, "auto": auto_size},
    )
    monkeypatch.setattr(jax.sharding, "get_abstract_mesh", lambda: ambient)
    # The mapped function is a Python stand-in: this test never constructs or computes JAX arrays.
    monkeypatch.setattr(jax, "shard_map", lambda kernel, **kwargs: kernel)
    mesh = SimpleNamespace(size=mesh_size, shape={"tp": mesh_size})
    kernel_result, fallback_result = object(), object()

    def specs():
        return ((Ps("tp"),), Ps("tp")) if layout else None

    result = run_mosaic_kernel(
        lambda arg: kernel_result,
        lambda reason: fallback_result,
        (SimpleNamespace(shape=(dim,)),),
        mesh=mesh,
        specs=specs,
    )
    assert result is (kernel_result if expect_kernel else fallback_result)
