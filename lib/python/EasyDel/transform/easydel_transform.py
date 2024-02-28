import gc

import flax.traverse_util
import jax
import numpy
from flax import traverse_util

from flax.traverse_util import flatten_dict
from flax.serialization import from_bytes, to_bytes, to_state_dict
import msgpack
import os
from fjformer import get_dtype
from jax import numpy as jnp
from typing import List, Optional, Mapping, Callable
from tqdm import tqdm


def float_tensor_to_dtype(tensor, dtype):
    """
    The float_tensor_to_dtype function is used to convert a tensor's dtype to the specified dtype.

    :param tensor: Convert the tensor to a float dtype
    :param dtype: Convert the tensor to a specific dtype
    :return: A tensor with the specified dtype
    
    """
    if dtype is None or dtype == "":
        return tensor
    if isinstance(dtype, str):
        dtype = get_dtype(dtype)
    float_dtypes = (jax.numpy.bfloat16, jax.numpy.float16, jax.numpy.float32, jax.numpy.float64)
    if getattr(tensor, "dtype", None) in float_dtypes:
        tensor = tensor.astype(dtype)
    return tensor


def match_keywords(string, ts, ns):
    """
    The match_keywords function takes a string, and two lists of strings.
    The first list is the &quot;must-have&quot; keywords, and the second list is the &quot;not-allowed&quot; keywords.
    It returns True if all the must-have keywords are in string, but none of not allowed are in it.

    :param string: Pass in the text that is being searched
    :param ts: Specify the required keywords and ns is used to specify the non-required keywords
    :param ns: Specify a list of negative keywords
    :return: True if all the keywords in ts are present and none of the
    
    """
    for t in ts:
        if t not in string:
            return False
    for n in ns:
        if n in string:
            return False
    return True


# def huggingface_to_easydel(
#         state_dict,
#         *,
#         device,
#         embedding_layer_names: Optional[List[str]] = None,
#         layer_norm_names: Optional[List[str]] = None,
#         shard_fns: Optional[Mapping[tuple, Callable]] = None,
#         dtype: jax.numpy.dtype = jax.numpy.float16,
#         **kwargs
# ):
#     """
#     The huggingface_to_easydel function takes a huggingface model's state_dict and converts it to an easydel
#     model's flax_dict. The function is designed to be used in conjunction with the load_huggingface function, which
#     loads a huggingface model from disk. The embedding layer name must be specified as well as the device on which
#     the conversion will take place.
#
#     :param state_dict: Load the weights from a huggingface model
#     :param embedding_layer_names: List[str]: Identify the embedding layer in the huggingface model
#     :param device: Determine which device the model will be loaded on
#     :param layer_norm_names: Replaces weight or kernel with (scale)
#     :param shard_fns: Optional[Mapping[tuple, Callable]]: Sharding Function to be used to shard model
#     :param dtype: jax.numpy.dtype: Specify the data type of the tensors
#     :return: A dictionary of the weights and biases in a format that can be used by flax (it's an UnFlattenDict)
#
#     """
#     embedding_layer_names = embedding_layer_names or []
#     layer_norm_names = layer_norm_names or []
#     if isinstance(embedding_layer_names, str):
#         embedding_layer_names = [embedding_layer_names]
#     _l = len(".weight")
#     _b = len(".bias")
#     with jax.default_device(device):
#         flax_dict = {}
#         pbar = tqdm(total=len(list(state_dict.keys())))
#         pbar.set_description("Converting Model")
#         for key, tensor in state_dict.items():
#             do_rc = True
#             for embedding_layer_name in embedding_layer_names:
#                 if embedding_layer_name in key:
#                     key = key[:-_l] + ".embedding"
#                     do_rc = False
#             for layer_norm in layer_norm_names:
#                 if layer_norm in key:
#                     if key.endswith(".weight"):
#                         key = key[:-_l] + ".scale"
#                         do_rc = False
#             if match_keywords(key, ["weight"], ["none"]) and do_rc:
#                 if len(tensor.shape) == 2:
#                     tensor = tensor.transpose(0, 1)
#                 if key.endswith(".weight"):
#                     key = key[:-_l] + ".kernel"
#             key_tuple = key.split(".")
#             key_names = ()
#             numpy_tensor: numpy.array = tensor.detach().cpu().numpy()
#             tensor = jax.numpy.array(numpy_tensor)
#             del numpy_tensor
#             gc.collect()
#             for k in key_tuple:
#                 key_names += k,
#             if shard_fns is not None:
#                 tensor = shard_fns[key_names](tensor)
#             flax_dict[key_names] = tensor
#             pbar.update(1)
#         return flax.traverse_util.unflatten_dict(flax_dict)

def huggingface_to_easydel(
        state_dict,
        *,
        device,
        embedding_layer_names: Optional[List[str]] = None,
        layer_norm_names: Optional[List[str]] = None,
        shard_fns: Optional[Mapping[tuple, Callable]] = None,
        dtype: jax.numpy.dtype = jax.numpy.float16,
        **kwargs
):
    """
    The huggingface_to_easydel function takes a huggingface model's state_dict and converts it to an easydel
    model's flax_dict. The function is designed to be used in conjunction with the load_huggingface function, which
    loads a huggingface model from disk. The embedding layer name must be specified as well as the device on which
    the conversion will take place.

    :param state_dict: Load the weights from a huggingface model
    :param embedding_layer_names: List[str]: Identify the embedding layer in the huggingface model
    :param device: Determine which device the model will be loaded on
    :param layer_norm_names: Replaces weight or kernel with (scale)
    :param shard_fns: Optional[Mapping[tuple, Callable]]: Sharding Function to be used to shard model
    :param dtype: jax.numpy.dtype: Specify the data type of the tensors
    :return: A dictionary of the weights and biases in a format that can be used by flax (it's an UnFlattenDict)

    """
    embedding_layer_names = set(embedding_layer_names or [])
    layer_norm_names = set(layer_norm_names or [])
    _l = len(".weight")
    _b = len(".bias")

    with jax.default_device(device):
        flax_dict = {}
        pbar = tqdm(total=len(state_dict))
        pbar.set_description("Converting Model")

        for key, tensor in state_dict.items():
            # Determine if renaming is necessary
            new_key = key
            if any(layer_name in key for layer_name in embedding_layer_names):
                new_key = key[:-_l] + ".embedding"
            elif any(layer_norm in key for layer_norm in layer_norm_names):
                new_key = key.replace(".weight", ".scale")
            elif "weight" in key:
                if len(tensor.shape) == 2:
                    tensor = tensor.transpose(0, 1)
                new_key = key.replace(".weight", ".kernel")

            key_tuple = tuple(new_key.split("."))
            # Convert tensor to jax.numpy.array without detaching and moving to CPU
            tensor = jnp.array(tensor.cpu().detach().numpy(), dtype=dtype)

            # Apply sharding functions if provided
            if shard_fns and key_tuple in shard_fns:
                tensor = shard_fns[key_tuple](tensor)

            flax_dict[key_tuple] = tensor

            # Update progress bar less frequently to reduce overhead
            pbar.update(1)
        pbar.close()

        gc.collect()
        return traverse_util.unflatten_dict(flax_dict)


def read_ckpt(path: [str, os.PathLike], shard_fns=None, add_extra_past_fix: list = None):
    """
    The read_ckpt function reads a checkpoint file and returns the tensors in it.

    :param path: [str, os.PathLike]: Specify the path to the checkpoint file
    :param shard_fns: Shard the tensors
    :param add_extra_past_fix: list: Add an extra past to the key
    :return: A dictionary of tensors
    
    """
    tensors = {}
    with open(path, "rb") as stream:
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
    """
    The save_ckpt function saves the state of a training run to disk.

    :param train_state: Store the current state of the training process
    :param path: Specify the location of the checkpoint file
    :param gather_fns: Specify a function that will be used to convert the tensor to bytes
    :param float_dtype: Convert the tensor to a specific dtype
    :return: Nothing
    
    """

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


def easystate_to_torch(
        state,
        dtype=jnp.float16,
        transpose_needed=None,
        transpose_not_needed=None,
        select_params_field: bool = True,

):
    import torch

    if transpose_needed is None:
        transpose_needed = ["kernel"]
    if transpose_not_needed is None:
        transpose_not_needed = ["none"]

    def match_keywords(string, do_transpose, dont_transpose):
        for dtr in do_transpose:
            if dtr not in string:
                return False
        for ntr in dont_transpose:
            if ntr in string:
                return False
        return True

    model_parameters = flatten_dict(
        state.params["params"],
        sep="."
    ) if select_params_field else flatten_dict(
        state.params,
        sep="."
    )
    torch_state_dict = {}
    pbar = tqdm(
        model_parameters.items(),
        desc="Converting EasyDelState to torch state_dict"
    )
    for key, tensor in pbar:
        if match_keywords(key, transpose_needed, transpose_not_needed):
            tensor = tensor.T
        tensor = tensor.astype(get_dtype(dtype))
        torch_state_dict[
            key.replace(".kernel", ".weight").replace(".embedding", ".weight").replace(".scale", ".weight")
        ] = torch.from_numpy(tensor)
    return torch_state_dict


def easystate_to_huggingface_model(
        state,
        config,
        base_huggingface_module,
        base_huggingface_module_kwarguments=None,
        dtype=jnp.float16,
        transpose_needed=None,
        transpose_not_needed=None,
        select_params_field: bool = True
):
    if base_huggingface_module_kwarguments is None:
        base_huggingface_module_kwarguments = {}
    state_dict = easystate_to_torch(
        state=state,
        dtype=dtype,
        transpose_needed=transpose_needed,
        transpose_not_needed=transpose_not_needed,
        select_params_field=select_params_field
    )
    model = base_huggingface_module(
        config=config,
        **base_huggingface_module_kwarguments
    )
    model.load_state_dict(state_dict)
    return model
