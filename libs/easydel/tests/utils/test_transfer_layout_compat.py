"""Metadata-only regressions for transfer and checkpoint-layout compatibility."""

import importlib
from pathlib import Path

import jax
import pytest
import spectrax as spx
from easydel.layers.norms import LayerNorm
from easydel.utils.parameters_transformation import StateDictConverter, TensorConverter
from jax import numpy as jnp

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
def test_sharded_dlpack_transfer_stages_a_single_device_buffer(monkeypatch, platform):
    torch = pytest.importorskip("torch")
    from torch.utils import dlpack

    class Device:
        def __init__(self, platform):
            self.platform = platform

    class ShardedArray:
        is_fully_addressable = True

        def devices(self):
            return {Device(platform), Device(platform)}

        def __dlpack__(self, stream=None):
            raise BufferError("DLPack cannot represent a sharded array")

    class StagedArray:
        def __dlpack__(self, stream=None):
            return "gathered capsule"

    array = ShardedArray()
    host_value = object()
    host_device = Device("cpu")
    monkeypatch.setenv("EASY_SAFE_TRANSFER", "false")
    monkeypatch.setenv("EASYDEL_FORCE_TORCH_USE_CPU", "false")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: platform == "gpu")
    monkeypatch.setattr(dlpack, "from_dlpack", lambda value: value.__dlpack__())
    monkeypatch.setattr("easydel.utils.parameters_transformation.EASYDEL_PREFERRED_HOST_COPY", "cpu")
    monkeypatch.setattr("easydel.utils.parameters_transformation.EASYDEL_PREFERRED_HOST_COPY_INDEX", 0)

    def gather(value):
        assert value is array
        return host_value

    def devices(selected_platform):
        assert selected_platform == "cpu"
        return [host_device]

    def stage(value, device):
        assert value is host_value
        assert device is host_device
        return StagedArray()

    monkeypatch.setattr(jax, "device_get", gather)
    monkeypatch.setattr(jax, "devices", devices)
    monkeypatch.setattr(jax, "device_put", stage)
    assert TensorConverter.jax_to_pytorch(array) == "gathered capsule"


def test_compile_cache_paths_remain_explicitly_importable_and_lazy(monkeypatch):
    import ejkernel.callib
    from easydel.utils import compiling_utils, helpers

    requested = []
    cache_dir = Path("/tmp/easydel-compatibility-cache")
    compile_dir = Path("/tmp/ejkernel-compatibility-cache/env-test")

    def cache_path():
        requested.append("cache")
        return cache_dir

    def compile_path():
        requested.append("compile")
        return compile_dir

    monkeypatch.setattr(helpers, "get_cache_dir", cache_path)
    monkeypatch.setattr(ejkernel.callib, "get_effective_compile_dir", compile_path)
    importlib.reload(compiling_utils)
    assert requested == []

    from easydel.utils.compiling_utils import CACHE_DIR, COMPILE_FUNC_DIR

    assert CACHE_DIR == cache_dir
    assert COMPILE_FUNC_DIR == compile_dir
    assert requested == ["cache", "compile"]
    assert {"CACHE_DIR", "COMPILE_FUNC_DIR"} <= set(dir(compiling_utils))
    with pytest.raises(AttributeError, match="unknown_cache_attribute"):
        _ = compiling_utils.unknown_cache_attribute


class LayoutTensor:
    """String-labeled tensor carrying layout only, without array computations."""

    dtype = jnp.float32
    ndim = 2

    def __init__(self, rows):
        self.rows = tuple(tuple(row) for row in rows)
        self.shape = (len(self.rows), len(self.rows[0]))

    def permute(self, *axes):
        assert axes == (1, 0)
        return LayoutTensor(zip(*self.rows, strict=True))


class DeclaredNorm(LayerNorm):
    def __init__(self, tensor):
        # Initialize only parameter metadata, never normalization numerics.
        self.weight = spx.Parameter(tensor)


class PlainLeaf(spx.Module):
    def __init__(self, tensor):
        self.weight = spx.Parameter(tensor)


class LayoutModel(spx.Module):
    def __init__(self, converted, norm_type):
        self.norm = norm_type(converted["norm"]["weight"])
        self.projection = PlainLeaf(converted["projection"]["weight"])

    @property
    def parameters(self):
        return spx.export(self)[1]


@pytest.mark.parametrize("shape", [(2, 3), (2, 2)])
@pytest.mark.parametrize("explicit_declaration", [False, True])
def test_declared_norm_import_export_preserves_layout_including_square_weights(monkeypatch, shape, explicit_declaration):
    source = LayoutTensor([[f"r{row}c{col}" for col in range(shape[1])] for row in range(shape[0])])
    monkeypatch.setattr(jax, "local_device_count", lambda: 0)
    monkeypatch.setattr(jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(TensorConverter, "convert_pytorch_to_jnp", lambda value, dtype: value)
    monkeypatch.setattr(TensorConverter, "jax_to_pytorch", lambda value: value)
    converted = StateDictConverter.huggingface_to_easydel(
        {"norm.weight": source, "projection.weight": source},
        layernorm_names=["norm"],
        dtype=jnp.float32,
        verbose=False,
    )
    assert converted["norm"]["weight"].rows == source.rows
    assert converted["projection"]["weight"].rows == tuple(zip(*source.rows, strict=True))
    model = LayoutModel(converted, PlainLeaf if explicit_declaration else DeclaredNorm)
    options = {"layernorm_names": ["norm"]} if explicit_declaration else {}
    exported = StateDictConverter.easydel_to_torch(model, dtype=jnp.float32, **options)
    assert set(exported) == {"norm.weight", "projection.weight"}
    assert exported["norm.weight"].shape == shape
    assert exported["norm.weight"].rows == source.rows
    assert exported["projection.weight"].rows == source.rows
