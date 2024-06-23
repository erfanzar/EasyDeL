from fjformer.checkpoint import float_tensor_to_dtype, CheckpointManager
from flax.traverse_util import flatten_dict
import jax
from jax import dlpack
import os


def match_keywords(string, positives, negatives):
    for positive in positives:
        if positive not in string:
            return False
    for negative in negatives:
        if negative in string:
            return False
    return True


def load_and_convert_checkpoint(path):
    import torch

    _, flax_params = CheckpointManager.load_state_checkpoint(path)
    flax_params = flatten_dict(flax_params["params"], sep=".")
    torch_params = {}
    for key, tensor in flax_params.items():
        if match_keywords(key, ["kernel"], ["norm", "ln_f"]):
            tensor = tensor.T
        tensor = float_tensor_to_dtype(tensor, "fp16")
        torch_params[key] = torch.from_numpy(tensor)
    return torch_params


def get_pytorch_model_and_config_by_type(): ...


def jax2pt(x: jax.Array):
    from torch.utils import dlpack as dlpack_pt
    from torch import cuda

    _jax_device = list(x.devices())[0].platform
    cpu_force = not cuda.is_available()
    if (
        _jax_device in ["cpu", "gpu"]
        and not cpu_force
        and not bool(os.environ.get("EASYDEL_FORCE_TORCH_USE_CPU", "false"))
    ):
        dl_pack_jax = dlpack.to_dlpack(
            x,
            stream=True if (_jax_device == "gpu" and not cpu_force) else None,
            src_device=list(x.devices())[0],
        )
    else:
        device = os.environ.get("EASYDEL_PERFRED_HOST_COPY", "cpu")
        if device.lower() == "none":
            device = None  # Auto JAX Select
        perfred_host = jax.devices(device)[
            int(os.environ.get("EASYDEL_PERFRED_HOST_COPY_IDEX", "0"))
        ]
        x = jax.device_get(x)  # make sure it's local
        x = jax.device_put(x, perfred_host)
        dl_pack_jax = dlpack.to_dlpack(
            x,
            stream=None,
        )
    return dlpack_pt.from_dlpack(dl_pack_jax)


def pt2jax(x):
    from torch.utils import dlpack as dlpack_pt

    org_shape = x.shape
    if org_shape[0] == 1 and len(org_shape) > 1:
        x = x.view(org_shape[1:])  # Prevent Major-to-Major BUG.

    device = os.environ.get("EASYDEL_PERFRED_HOST_PUT", "none")
    device = None if device.lower() == "none" else device  # Auto JAX Select
    return dlpack.from_dlpack(
        dlpack_pt.to_dlpack(x),
        jax.devices(device)[int(os.environ.get("EASYDEL_PERFRED_HOST_PUT_IDEX", "0"))],
    ).reshape(org_shape)
