import jax

from flax.traverse_util import flatten_dict
from flax.serialization import from_bytes, to_bytes, to_state_dict
import msgpack
import os


def get_float_dtype_by_name(dtype):
    return {
        'bf16': jax.numpy.bfloat16,
        'bfloat16': jax.numpy.bfloat16,
        'fp16': jax.numpy.float16,
        'float16': jax.numpy.float16,
        'fp32': jax.numpy.float32,
        'float32': jax.numpy.float32,
        'fp64': jax.numpy.float64,
        'float64': jax.numpy.float64,
    }[dtype]


def float_tensor_to_dtype(tensor, dtype):
    if dtype is None or dtype == '':
        return tensor
    if isinstance(dtype, str):
        dtype = get_float_dtype_by_name(dtype)
    float_dtypes = (jax.numpy.bfloat16, jax.numpy.float16, jax.numpy.float32, jax.numpy.float64)
    if getattr(tensor, 'dtype', None) in float_dtypes:
        tensor = tensor.astype(dtype)
    return tensor


def match_keywords(string, ts, ns):
    for t in ts:
        if t not in string:
            return False
    for n in ns:
        if n in string:
            return False
    return True


def huggingface_to_easydel(state_dict, embedding_layer_name: str, device, dtype: jax.numpy.dtype = jax.numpy.float16):
    _l = len('.weight')
    with jax.default_device(device):
        flax_dict = {}
        for key, tensor in state_dict.items():
            if embedding_layer_name in key:
                tensor = tensor.transpose(0, 1)
                key = key[:-_l] + '.embedding'
            elif match_keywords(key, ['kernel'], ['none']):
                if len(tensor.shape) == 2:
                    tensor = tensor.transpose(0, 1)
                if key.endswith('.weight'):
                    key = key[:-_l] + '.kernel'
            key_tuple = key.split('.')
            key_names = ()
            tensor = tensor.detach().cpu().numpy()
            for k in key_tuple:
                key_names += k,
            flax_dict[key_names] = tensor.astype(dtype)
        return flax_dict


def read_ckpt(path: [str, os.PathLike], shard_fns=None, add_extra_past_fix: list = None):
    tensors = {}
    with open(path, 'rb') as stream:
        unpacker = msgpack.Unpacker(stream, read_size=83886080, max_buffer_size=0)
        for key, value in unpacker:
            if add_extra_past_fix is not None:
                key = add_extra_past_fix + key
            key = tuple(key)
            tensor = from_bytes(None, value)
            if shard_fns is not None:
                tensor = shard_fns[key](tensor)
            tensors[key] = tensor
    return tensors


def save_ckpt(train_state, path, gather_fns=None, float_dtype=None):
    train_state = to_state_dict(train_state)
    packer = msgpack.Packer()
    flatten_train_state = flatten_dict(train_state)
    if gather_fns is not None:
        gather_fns = flatten_dict(to_state_dict(gather_fns))

    with open(path, "wb") as stream:
        for key, value in flatten_train_state.items():
            if gather_fns is not None:
                value = gather_fns[key](value)
            value = float_tensor_to_dtype(value, float_dtype)
            stream.write(packer.pack((key, to_bytes(value))))
