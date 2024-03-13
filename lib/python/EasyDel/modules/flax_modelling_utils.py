import functools

import fjformer
from einops import rearrange
from fjformer.bits import config as q_config, q_flax
from flax.linen import combine_masks
from jax import lax, numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.interpreters import pxla
import jax
from flax import linen as nn
from functools import partial
import chex
from typing import Sequence, Optional, Literal
from jax.experimental.mesh_utils import create_device_mesh
from .easydel_modelling_utils import EasyMethod
from jax.sharding import PartitionSpec

from ..etils.errors import EasyDelBlockWiseFFNError

ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
    "leaky_relu": partial(nn.leaky_relu, negative_slope=0.01),
    "glu": nn.glu,
    "elu": nn.elu,
    "softmax": nn.softmax
}


def canonicalize_dtype(
        *args, dtype: Optional[chex.ArrayDType] = None, inexact: bool = True
) -> chex.ArrayDType:
    """Canonicalize an optional dtype to the definitive dtype.

    If the ``dtype`` is None this function will infer the dtype. If it is not
    None it will be returned unmodified or an exceptions is raised if the dtype
    is invalid.
    from the input arguments using ``jnp.result_type``.

    Args:
      *args: JAX array compatible values. None values
        are ignored.
      dtype: Optional dtype override. If specified the arguments are cast to
        the specified dtype instead and dtype inference is disabled.
      inexact: When True, the output dtype must be a subdtype
      of `jnp.inexact`. Inexact dtypes are real or complex floating points. This
      is useful when you want to apply operations that don't work directly on
      integers like taking a mean for example.
    Returns:
      The dtype that *args should be cast to.
    """
    if dtype is None:
        args_filtered = [jax.numpy.asarray(x) for x in args if x is not None]
        dtype = jax.numpy.result_type(*args_filtered)
        if inexact and not jax.numpy.issubdtype(dtype, jax.numpy.inexact):
            dtype = jax.numpy.promote_types(jax.numpy.float32, dtype)
    if inexact and not jax.numpy.issubdtype(dtype, jax.numpy.inexact):
        raise ValueError(f'Dtype must be inexact: {dtype}')
    return dtype


def get_names_from_partition_spec(partition_specs):
    """
    The get_names_from_partition_spec function takes a partition_specs argument, which is either a dictionary or list.
    If it's a dictionary, the function converts it to a list of values. Then for each item in the partition_specs list:
        If the item is None, continue (do nothing) and move on to next iteration of loop.
        If the item is an instance of str (i.e., if it's just one string), add that string to names set and move
        on to next iteration of loop.
        Otherwise, (if not None or str), call get_names_from_partition_spec recurs

    :param partition_specs: Define the partitioning of a table
    :return: A list of the names of all partitions

    """
    names = set()
    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_partition_spec(item))

    return list(names)


def names_in_mesh(*names):
    """
    The names_in_mesh function is a decorator that can be used to check whether
    the names of the axes passed into a function are valid.  It will raise an
    exception if any of the axis names are not in the physical mesh.  For example,
    if you have a function that takes two axes as arguments, and you want to make sure they're both in your mesh:

    :param names: Collect all the names passed to the function into a tuple
    :return: A boolean indicating whether all the given

    """
    return set(names) <= set(pxla.thread_resources.env.physical_mesh.axis_names)


with_sharding_constraint = fjformer.with_sharding_constraint


def get_gradient_checkpoint_policy(name):
    """
    The get_gradient_checkpoint_policy function is a helper function that returns the gradient checkpoint policy
        specified by the name parameter.

    :param name: Select the checkpoint policy from the dictionary
    :return: A function that is used in the jax

    """
    gradients = dict(
        everything_saveable=jax.checkpoint_policies.everything_saveable,
        nothing_saveable=jax.checkpoint_policies.nothing_saveable,
        dots_saveable=jax.checkpoint_policies.dots_saveable,
        checkpoint_dots=jax.checkpoint_policies.checkpoint_dots,
        dots_with_no_batch_dims_saveable=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
        checkpoint_dots_with_no_batch_dims=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
        save_anything_except_these_names=jax.checkpoint_policies.save_anything_except_these_names,
        save_any_names_but_these=jax.checkpoint_policies.save_any_names_but_these,
        save_only_these_names=jax.checkpoint_policies.save_only_these_names,
        save_from_both_policies=jax.checkpoint_policies.save_from_both_policies
    )
    return gradients[name]


def repeat_kv_bnsh(x: chex.Array, n_rep: int) -> chex.Array:
    """
    The repeat_kv_bnsh function is used to repeat the key and value vectors for each head in a multi-head attention
    module. This function takes as input an array of shape (batch_size, n_heads, sequence_length, head_dim) and returns
    an array of shape (batch_size, n_heads * nrep, sequence length, head dim). The reason this is necessary is because the
    attention module expects keys/values/queries to be repeated across heads but not across batches. However we want our
    keys/values/queries to be repeated both across heads AND batches so that we can use them

    :param x: chex.Array: Pass in the input to the function
    :param n_rep: int: Repeat the key and value heads
    :return: A new array with the same shape as x, except for the second dimension which is n_kv_heads * n_rep

    """
    bs, n_kv_heads, s, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, jax.numpy.newaxis, :, :]
    x = jax.numpy.repeat(x, n_rep, axis=2)

    return x.reshape(bs, n_kv_heads * n_rep, s, head_dim)


def repeat_kv_bsnh(x: chex.Array, n_rep: int) -> chex.Array:
    """
    The repeat_kv_bsnh function is used to repeat the key and value vectors for each head.

    :param x: chex.Array: Specify the input array
    :param n_rep: int: Repeat the key-value attention heads n_rep times
    :return: A new array with the same batch size, sequence length, and head dimension as the input array

    """
    bs, s, n_kv_heads, head_dim = x.shape
    x = x.transpose(0, 2, 1, 3)
    if n_rep == 1:
        return x
    x = x[:, :, jax.numpy.newaxis, :, :]
    x = jax.numpy.repeat(x, n_rep, axis=2)

    x = x.transpose(0, 2, 1, 3)

    return x.reshape(bs, s, n_kv_heads * n_rep, head_dim)


def precompute_freq_cis(
        dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0, rope_type: str | None = None,
        t_dtype: jnp.dtype = jnp.int32
):
    if rope_type == "none":
        rope_type = None
    assert rope_type in [
        "linear",
        "dynamic",
        None
    ], "wrong rope type has been given"
    if t_dtype == jnp.int64:
        jax.config.update("jax_enable_x64", True)
    t = jax.numpy.arange(max_position_embeddings, dtype=t_dtype)

    if rope_type == "linear":
        t = t / scaling_factor

    if rope_type == "dynamic":
        base = base * (
                scaling_factor - (scaling_factor - 1)
        ) ** (dim / (dim - 2))

    inv_freq = 1.0 / (
            base ** (jax.numpy.arange(0, dim, 2, dtype=jax.numpy.float32) / dim)
    )
    freq = jax.numpy.einsum(
        "i , j -> i j", t, inv_freq
    ).astype("float32")

    embed = jax.numpy.concatenate((freq, freq), axis=-1)
    return jax.numpy.sin(embed)[:, :], jax.numpy.cos(embed)[:, :]


def rotate_half(x):
    """
    The rotate_half function takes a complex-valued array and rotates the
    phase of its second half by 180 degrees. This is equivalent to multiplying
    the second half by -i, or equivalently rotating it 90 degrees counterclockwise.


    :param x: Specify the input array
    :return: A new array that is the same as the input

    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jax.numpy.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(tensor, sin_, cos_):
    """
    The apply_rotary_pos_emb function applies a rotary positional embedding to the input tensor.
    b,h,s,d or pytorch style

    :param tensor: Store the tensor that is passed into the function
    :param sin_: Rotate the tensor by pi/2
    :param cos_: Apply the cosine function to the tensor
    :return: A tensor with the same shape as the input tensor

    """
    b, h, s, d = tensor.shape
    return (tensor * cos_[:, :, :s, :]) + (rotate_half(tensor) * sin_[:, :, :s, :])


def get_ranks_and_size(mesh):
    """
    The get_ranks_and_size function is used to determine the number of MPI processes
    (``mp_node_size``) and the number of devices per process (``dp_node_size``).
    The ``mesh.shape[mp]`` determines how many MPI processes are needed,
    and then we divide that by the local device count to get ``mp_node_size = max( 1, mp / jax.local )`.
    This means that if there are more than enough devices for all MPI ranks on a node, each rank will only use one device; otherwise it will use

    :param mesh: Get the shape of the mesh
    :return: A dictionary with the following keys:

    """
    out = dict(mesh=mesh)
    total_process_size = mesh.shape["tp"] * mesh.shape["sp"]
    mp_node_size = max(1, total_process_size // jax.local_device_count())
    dp_node_size = jax.process_count() // mp_node_size
    out.update(mp_node_size=mp_node_size,
               dp_node_size=dp_node_size)

    dp_node_rank = jax.process_index() // mp_node_size
    mp_node_rank = jax.process_index() % mp_node_size
    out.update(dp_node_rank=dp_node_rank,
               mp_node_rank=mp_node_rank)
    return out


def create_mesh(
        axis_dims: Sequence[int] = (1, -1, 1, 1), axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"), backend=""
):
    """
    The create_mesh function creates a mesh object that can be used to shard arrays.

    :param axis_dims: Sequence[int]: Specify the dimensions of the mesh
    :param axis_names: Sequence[str]: Name the axes of the mesh
    :param backend: Specify the backend to use
    :return: A mesh object

    """
    array_devices = jax.numpy.ones(
        (len(jax.devices() if backend == "" else jax.devices(backend)), 1))
    resh = array_devices.reshape(axis_dims).shape

    return jax.sharding.Mesh(
        create_device_mesh(resh), axis_names
    )


def add_start_docstrings(*docstr):
    """
    The add_start_docstrings function is a decorator that adds the docstrings to the beginning of a function.
    The add_start_docstrings function takes in an arbitrary number of strings and returns a decorator.
    The returned decorator takes in one argument, fn, which is assumed to be a function. The docstring for fn is set equal to
    the concatenation of all the strings passed into add_start_docstrings plus (if it exists) the original docstring for fn.

    :param docstr: Pass in a variable number of arguments to the function
    :return: A decorator that adds the docstrings to the function

    """

    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + \
                     (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def get_dot_general_by_bits(
        bits: Optional[int] = None,
        mode: Literal["train", "serve", "convert"] = EasyMethod.TRAIN
) -> dict:
    """
    The get_general_dot function is a helper function that returns a q_flax.QDotGeneral object
    with the specified number of bits for forward and backward passes. If no bits are specified,
    the function returns None.

    :param bits: Optional[int]: Specify the number of bits for quantization
    :param mode: EasyMethod: Specify the use of model to init the QDot Method for (e.q TRAIN,SERVE,...)
    :return: A dict that contain dot_general_cls
    """
    if mode == EasyMethod.TRAIN:
        rhs_quant_mode = q_flax.QuantMode.TRAIN
    elif mode == EasyMethod.EVAL or mode == EasyMethod.SERVE:
        rhs_quant_mode = q_flax.QuantMode.SERVE
    elif mode == EasyMethod.CONVERT:
        rhs_quant_mode = q_flax.QuantMode.CONVERT
    else:
        raise ValueError("Unknown Quant Method for EasyMethod")
    if bits is not None:
        return {
            "dot_general_cls": functools.partial(
                q_flax.QDotGeneral,
                q_config.fully_quantized(
                    fwd_bits=bits,
                    bwd_bits=bits
                ),
                rhs_quant_mode=rhs_quant_mode
            )
        }
    return {}  # empty just in case of not getting any error


class BaseJAXAttentionModule(nn.Module):
    config: "EasyDelPretrainedConfig"

    @nn.compact
    def _concatenate_to_cache(self, key, value, query_states, attention_mask):
        """
        The _concatenate_to_cache function is used to concatenate the key and value vectors
        of a query_states with those of previous queries. This allows for the attention mechanism to
        look at all previous queries when computing its output. The function takes in three
        arguments: key, value, and query_states. It also uses two variables that are stored in the cache:
        cached_key and cached_value.

        :param self: Access the variables stored in the cache
        :param key: Store the keys of the encoder-decoder attention
        :param value: Initialize the cached_value variable
        :param query_states: Determine the number of cache vectors to update
        :param attention_mask: Mask out the padded vectors in the cache
        :return: The key, value and attention_mask
        """
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            cur_index = cache_index.value
            if query_states.shape[1] == 1 and self.config.use_sharded_kv_caching:
                mesh = self.config.jax_mesh()

                def fn(
                        _cached_key,
                        _cached_value,
                        _key,
                        _value,
                        _cur_index
                ):
                    assert _key.shape[1] == 1 and _value.shape[1] == 1, (_key.shape, _value.shape)
                    sp_size = max_length // mesh.shape["sp"]
                    axis_index = jax.lax.axis_index("sp")
                    _cur_index = _cur_index - axis_index * sp_size
                    _key, _value = jax.lax.cond(
                        jnp.logical_and(cur_index >= 0, _cur_index < sp_size),
                        lambda: (
                            _cached_key.at[:, cur_index].set(_key[:, -1]),
                            _cached_value.at[:, cur_index].set(_value[:, -1]),
                        ),
                        lambda: (_cached_key, _cached_value),
                    )
                    return _key, _value

                fn = shard_map(
                    fn, mesh=mesh,
                    in_specs=(
                        PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                        PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                        PartitionSpec(("dp", "fsdp"), None, "tp", None),
                        PartitionSpec(("dp", "fsdp"), None, "tp", None),
                        PartitionSpec()
                    ),
                    out_specs=(
                        PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                        PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
                    ),
                    check_rep=False
                )
                key, value = fn(cached_key.value, cached_value.value, key, value, cur_index)
            else:
                *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
                cur_index = cache_index.value
                indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                key = lax.dynamic_update_slice(cached_key.value, key, indices)
                value = lax.dynamic_update_slice(cached_value.value, value, indices)
                num_updated_cache_vectors = query_states.shape[1]
                pad_mask = jnp.broadcast_to(
                    jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                    tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
                )
                attention_mask = combine_masks(pad_mask, attention_mask)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query_states.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
        return key, value, attention_mask


def block_wise_ffn(remat_ffn, inputs, chunk_size: int, deterministic: bool):
    generating = inputs.shape[1] == 1
    try:
        if generating:
            return remat_ffn(inputs, deterministic)
        else:
            inputs = rearrange(inputs, 'b (c n) d -> b c n d', c=chunk_size)

            def scan_ffn(remat_ffn_, carry, hidden_states):
                outputs = remat_ffn_(hidden_states, deterministic)
                return carry, outputs

            scan_axis = inputs.ndim - 2
            _, output = nn.scan(
                scan_ffn,
                variable_broadcast="params",
                split_rngs={"params": False, "dropout": True},
                in_axes=scan_axis,
                out_axes=scan_axis,
            )(remat_ffn, None, inputs)
            output = rearrange(output, 'b c n d -> b (c n) d')
            return output
    except Exception as e:
        raise EasyDelBlockWiseFFNError(
            "You Are using BlockWise FFN from near-infinite-context length paper and you might be passing "
            "input arguments in wrong way in case that you don't want to use this just pass `use_scan_mlp=False` in "
            "model config or in config_kwargs in AutoEasyDeLModelForCausalLM or change `scan_mlp_chunk_size` "
            f"in configs for more information read Docs.\nOriginal Error\n{e}"
        )


def read_depth(
        params: dict,
        path: str | None = None,
        state: dict | None = None
):
    if state is None:
        state = {}
    for key, value in params.items():
        if isinstance(value, dict):
            accureated_path = path + "/" + key if path is not None else key
            state = read_depth(
                params[key],
                path=key if path is None else accureated_path,
                state=state
            )
        else:
            value_string = type(value).__name__ + f"(shape={value.shape})"
            state[path] = value_string
    return state


def get_maximum_depths(dictionary: dict):
    maximums = {}
    minimums = {}
    for k, v in dictionary.items():
        splits = k.split("/")
        for index, split in enumerate(splits):
            try:
                split = int(split)
                if str(index) in maximums.keys():
                    current = maximums[str(index)]
                    if current < split:
                        maximums[str(index)] = split
                else:
                    maximums[str(index)] = split
                if str(index) in minimums.keys():
                    split = int(split)
                    if str(index) in minimums.keys():
                        current = minimums[str(index)]
                        if current > split:
                            minimums[str(index)] = split
                else:
                    minimums[str(index)] = split
            except ValueError:
                ...
    return maximums, minimums
