"""Tiny checkpoint-conversion numerics; accelerator-only."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.utils.parameters_transformation import StateDictConverter, TensorConverter
from jax import numpy as jnp


class NamedParameters(spx.Module):
    def __init__(self):
        self.block = ParameterGroup()

    @property
    def parameters(self):
        return spx.export(self)[1]


class ParameterGroup(spx.Module):
    def __init__(self):
        self.scale = spx.Parameter(jnp.array([1.0, 2.0]))
        self.embedding_proj = ParameterLeaf()
        self.kernel_proj = ParameterLeaf()
        self.scale_proj = ParameterLeaf()


class ParameterLeaf(spx.Module):
    def __init__(self):
        self.weight = spx.Parameter(jnp.array([3.0, 4.0]))


def test_export_preserves_declared_scale_and_module_names():
    pytest.importorskip("torch")
    result = StateDictConverter.easydel_to_torch(NamedParameters(), dtype=jnp.float32)
    assert set(result) == {
        "block.scale",
        "block.embedding_proj.weight",
        "block.kernel_proj.weight",
        "block.scale_proj.weight",
    }
    np.testing.assert_array_equal(result["block.scale"].numpy(), [1.0, 2.0])
    for key in ("block.embedding_proj.weight", "block.kernel_proj.weight", "block.scale_proj.weight"):
        np.testing.assert_array_equal(result[key].numpy(), [3.0, 4.0])


def test_declared_multidimensional_layernorm_is_not_dense_transposed():
    torch = pytest.importorskip("torch")
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    result = StateDictConverter.huggingface_to_easydel(
        {"norm.weight": source, "projection.weight": source},
        layernorm_names=["norm"],
        dtype=jnp.float32,
        verbose=False,
    )
    np.testing.assert_array_equal(np.asarray(result["norm"]["weight"]), source.numpy())
    np.testing.assert_array_equal(np.asarray(result["projection"]["weight"]), source.T.numpy())


def test_conversion_honors_requested_dtype_for_all_parameters():
    torch = pytest.importorskip("torch")
    result = StateDictConverter.huggingface_to_easydel(
        {"bias": torch.tensor([1.0, 2.0]), "indices": torch.tensor([1, 2], dtype=torch.int32)},
        dtype=jnp.bfloat16,
        verbose=False,
    )
    assert result["bias"].dtype == jnp.bfloat16
    assert result["indices"].dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(result["indices"]), [1, 2])
    # Packed/checkpoint leaves have a separate explicit dtype-preserving API.
    packed = TensorConverter.to_jax_preserving_dtype(torch.tensor([257, 383], dtype=torch.int32))
    assert packed.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(packed), [257, 383])


def test_dlpack_transfer_preserves_values_and_device(monkeypatch):
    torch = pytest.importorskip("torch")
    if jax.default_backend() == "tpu":
        pytest.skip("Zero-copy DLPack sharing targets compatible CPU/GPU runtimes")
    if not torch.cuda.is_available():
        pytest.skip("The GPU computation test needs a CUDA-enabled torch installation")
    monkeypatch.setenv("EASY_SAFE_TRANSFER", "false")
    array = jnp.arange(6, dtype=jnp.float32)
    converted = TensorConverter.jax_to_pytorch(array)
    assert converted.device.type == "cuda"
    np.testing.assert_array_equal(converted.cpu().numpy(), np.arange(6, dtype=np.float32))
