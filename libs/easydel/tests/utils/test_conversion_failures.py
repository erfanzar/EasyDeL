"""Conversion failure propagation and metadata-only dispatch regressions."""

import jax
import pytest
from easydel.utils.parameters_transformation import DtypeHandler, StateDictConverter, TensorConverter

pytestmark = pytest.mark.cpu_ok


class InvalidTensor:
    shape = (2, 3)

    def permute(self, *axes):
        raise ValueError("invalid checkpoint tensor")


def test_conversion_failure_preserves_original_exception_and_parameter_context(monkeypatch):
    monkeypatch.setattr(jax, "local_device_count", lambda: 0)
    with pytest.raises(ValueError, match="invalid checkpoint tensor") as error:
        StateDictConverter.huggingface_to_easydel({"projection.weight": InvalidTensor()}, verbose=False)
    assert any("projection.weight" in note for note in error.value.__notes__)


@pytest.mark.parametrize("stage", ["callback", "sharding"])
def test_user_conversion_hooks_cannot_silently_drop_parameters(monkeypatch, stage):
    value = object()
    monkeypatch.setattr(jax, "local_device_count", lambda: 0)
    monkeypatch.setattr(TensorConverter, "convert_pytorch_to_jnp", lambda tensor, dtype: value)

    def broken(*args):
        raise RuntimeError("user conversion failed")

    options = {"callback": broken} if stage == "callback" else {"shard_fns": {("bias",): broken}}
    with pytest.raises(RuntimeError, match="user conversion failed") as error:
        StateDictConverter.huggingface_to_easydel({"bias": object()}, verbose=False, **options)
    assert any("bias" in note for note in error.value.__notes__)


def test_cpu_dlpack_transfer_does_not_require_cuda_or_copy_via_host(monkeypatch):
    torch = pytest.importorskip("torch")
    from torch.utils import dlpack

    class DLPackArray:
        def devices(self):
            return {SimpleNamespaceHashable()}

        def __dlpack__(self, stream=None):
            return "capsule"

    class SimpleNamespaceHashable:
        platform = "cpu"

    array = DLPackArray()
    monkeypatch.setenv("EASY_SAFE_TRANSFER", "false")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dlpack, "from_dlpack", lambda x: x.__dlpack__())

    def forbidden(*args):
        pytest.fail("Compatible CPU DLPack arrays must not round-trip through host copying")

    monkeypatch.setattr(jax, "device_get", forbidden)
    assert TensorConverter.jax_to_pytorch(array) == "capsule"


def test_explicit_conversion_dtype_applies_to_integer_checkpoint_parameters(monkeypatch):
    import numpy as np

    class IntegerCheckpointTensor:
        dtype = "torch.int32"

        def cpu(self):
            return self

        def detach(self):
            return self

        def numpy(self):
            return np.array([257, 383], dtype=np.int32)

    # Exercise the actual conversion policy with a host-only array boundary.
    monkeypatch.setattr(jax.numpy, "array", np.asarray)
    converted = TensorConverter.convert_pytorch_to_jnp(IntegerCheckpointTensor(), np.float32)
    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, [257, 383])


@pytest.mark.parametrize("dtype", [jax.numpy.float32, jax.numpy.bfloat16, jax.numpy.float16])
@pytest.mark.parametrize("conversion", ["eager", "streaming_leaf"])
def test_hash_router_checkpoint_ids_keep_fp32_storage(monkeypatch, dtype, conversion):
    import numpy as np
    from easydel.modules.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HashRouter

    torch = pytest.importorskip("torch")
    ids = torch.tensor([[257, 383], [2049, 1]], dtype=torch.int32)
    prefix = "model.layers.0.mlp.gate"
    rules = {
        f"{prefix}.{key}": {
            **rule,
            "splits": [{**split, "name": f"{prefix}.{split['name']}"} for split in rule["splits"]],
        }
        for key, rule in DeepseekV4HashRouter.reform_param.fget(None).items()
    }
    checkpoint = {f"{prefix}.tid2eid": ids, f"{prefix}.weight": torch.ones((2, 3))}

    # Replace only device-array boundaries: rule selection, dtype policy, and
    # the eager/streaming shared converter all run on real host tensors.
    monkeypatch.setattr(jax, "local_device_count", lambda: 0)
    monkeypatch.setattr(jax.numpy, "array", np.asarray)
    monkeypatch.setattr(TensorConverter, "pytorch_to_jax", lambda tensor: tensor.cpu().detach().numpy())
    if conversion == "eager":
        converted = StateDictConverter.huggingface_to_easydel(
            checkpoint, dtype=dtype, reform_param=rules, verbose=False
        )["model"]["layers"][0]["mlp"]["gate"]
    else:
        config = {"dtype": dtype, "reform_param": rules}
        converted = {
            path[-1]: array
            for key, tensor in checkpoint.items()
            for path, array in StateDictConverter.process_tensor(key, tensor, config)
        }
    assert converted["tid2eid"].dtype == np.float32
    np.testing.assert_array_equal(converted["tid2eid"], ids.numpy())
    assert converted["weight"].dtype == np.dtype(dtype)
    np.testing.assert_array_equal(converted["weight"], np.ones((3, 2)))


def test_float_dtype_helper_preserves_nonfloating_values():
    class MetadataTensor:
        def __init__(self, dtype):
            self.dtype = dtype

        def astype(self, dtype):
            return ("cast", dtype)

    integer = MetadataTensor("int32")
    assert DtypeHandler.float_tensor_to_dtype(integer, "bf16") is integer
    assert DtypeHandler.float_tensor_to_dtype(MetadataTensor("float32"), "fp16") == ("cast", jax.numpy.float16)
