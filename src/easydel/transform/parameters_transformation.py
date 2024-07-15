import gc
import re
from typing import Callable, List, Mapping, Optional

import fjformer
import jax
import transformers
from fjformer.checkpoint import get_dtype
from flax import traverse_util
from flax.traverse_util import flatten_dict
from jax import numpy as jnp
from tqdm.autonotebook import tqdm

from easydel.etils.etils import get_logger
from easydel.transform.utils import jax2pt, pt2jax

logger = get_logger(__name__)


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
    float_dtypes = (
        jax.numpy.bfloat16,
        jax.numpy.float16,
        jax.numpy.float32,
        jax.numpy.float64,
    )
    if getattr(tensor, "dtype", None) in float_dtypes:
        tensor = tensor.astype(dtype)
    return tensor


def match_keywords(string, ts, ns):
    """The match_keywords function takes a string, and two lists of strings.
    The first list is the &quot;must-have&quot; keywords, and the second list is the &quot;not-allowed&quot; keywords.
    It returns True if all the must-have keywords are in string, but none of not allowed are in it.

    Args:
        string: Pass in the text that is being searched
        ts: Specify the required keywords and ns is used to specify the
            non-required keywords
        ns: Specify a list of negative keywords

    Returns:
        True if all the keywords in ts are present and none of the
    """
    for t in ts:
        if t not in string:
            return False
    for n in ns:
        if n in string:
            return False
    return True


class _DummyContextManager:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def torch_dict_to_easydel_params(
    state_dict,
    *,
    device: Optional[jax.Device] = None,
    embedding_layer_names: Optional[List[str]] = None,
    layer_norm_names: Optional[List[str]] = None,
    shard_fns: Optional[Mapping[tuple, Callable]] = None,
    convert_to_8bit: bool = False,
    params_pattern_selection: Optional[re.Pattern] = None,
    dtype: jax.numpy.dtype = jax.numpy.float16,
    rnn_based_or_rwkv: bool = False,
    verbose: bool = True,
    remove_state_dict: bool = False,
    lm_head_name: Optional[str] = None,
    uses_tie_word_embedding: bool = False,
    **kwargs,
):
    """The torch_dict_to_easydel_params function takes a torch model's state_dict and converts it to an easydel
    model's params. The function is designed to be used in conjunction with the load_huggingface function, which
    loads a huggingface model from disk. The embedding layer name must be specified as well as the device on which
    the conversion will take place.

    Args:
        state_dict: Load the weights from a huggingface model
        embedding_layer_names: List[str]: Identify the embedding layer in the huggingface model
        device: Determine which device the model will be loaded on
        layer_norm_names: Replaces weight or kernel with (scale)
        shard_fns: Optional[Mapping[tuple, Callable]]: Sharding Function to be used to shard model
        convert_to_8bit: bool: whenever to convert the into 8bit format
        params_pattern_selection: Optional[re.Pattern]: patter to use to find the parameters of the model which will
        dtype: jax.numpy.dtype: Specify the data type of the tensors
        rnn_based_or_rwkv: bool: rnn_based_or_rwkv is a conditioner  which decide whenever it finds a value in tree
        verbose: bool: whenever to log sharding or converting process
        remove_state_dict: bool : whether to remove state dict during  the transforming process
    be converted to 8bit format.
    that start with time_mix_ it will automatically reshape that for easydel use case

    Returns:
        A dictionary of the weights and biases in a format that can be
        used by flax (it's an UnFlattenDict)
    """
    try:
        import torch

        if torch.cuda.is_available():

            def _clear():
                gc.collect()
                torch.cuda.empty_cache()

        else:

            def _clear():
                gc.collect()

    except ModuleNotFoundError:

        class torch:
            bfloat16 = None

        def _clear():
            gc.collect()

    embedding_layer_names = set(embedding_layer_names or [])
    layer_norm_names = set(layer_norm_names or [])
    _l = len(".weight")
    _b = len(".bias")

    if convert_to_8bit:
        assert params_pattern_selection is not None, (
            "in case of converting parameters to 8bit you should pass "
            "`params_pattern_selection` too, to tell the quantizer which parameters should be quantized."
        )
    if device is None:
        device = jax.devices()[0]
    ctx_m = jax.default_device(device) if shard_fns is None else _DummyContextManager()
    with ctx_m:
        flax_dict = {}
        pbar = tqdm(total=len(state_dict), disable=not verbose)
        pbar.set_description("Converting Model")
        missed_shardings = 0
        for key in list(state_dict.keys()):
            tensor = state_dict.pop(key)
            new_key = key
            if any(layer_name in key for layer_name in embedding_layer_names):
                new_key = key[:-_l] + ".embedding"
            elif rnn_based_or_rwkv and ("time_mix_" in key or "time_" in key):
                tensor = tensor.reshape(-1)
            elif any(layer_norm in key for layer_norm in layer_norm_names):
                new_key = key.replace(".weight", ".scale")
            elif "weight" in key:
                if len(tensor.shape) == 2:
                    tensor = tensor.transpose(0, 1)
                new_key = key.replace(".weight", ".kernel")

            key_tuple = tuple(new_key.split("."))
            if uses_tie_word_embedding is not None and lm_head_name is not None:
                if uses_tie_word_embedding:
                    if key_tuple[0] == lm_head_name:
                        continue

            # Convert tensor to jax.numpy.array and delete the tensor to free memory
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            array = jax.lax.convert_element_type(
                pt2jax(tensor), dtype
            ).block_until_ready()
            if remove_state_dict:
                del tensor
                _clear()

            # Apply sharding functions if
            if shard_fns is not None:
                if key_tuple in shard_fns:
                    array = shard_fns[key_tuple](array)
                else:
                    missed_shardings += 1
            if convert_to_8bit and params_pattern_selection.search("/".join(key_tuple)):
                array = fjformer.linen.linen.Int8Params(*fjformer.linen.quantize(array))
            flax_dict[key_tuple] = array
            pbar.set_postfix(missed_shardings=missed_shardings)
            pbar.update(1)

        pbar.close()
        _clear()
        return traverse_util.unflatten_dict(flax_dict)


def easystate_to_torch(
    state,
    dtype=jnp.float16,
    transpose_needed=None,
    transpose_not_needed=None,
    select_params_field: bool = True,
    rnn_based_or_rwkv: bool = False,
):
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

    model_parameters = (
        flatten_dict(state.params["params"], sep=".")
        if select_params_field
        else flatten_dict(state.params, sep=".")
    )
    torch_state_dict = {}
    pbar = tqdm(
        model_parameters.items(), desc="Converting EasyDeLState to torch state_dict"
    )
    for key, tensor in pbar:
        if match_keywords(key, transpose_needed, transpose_not_needed):
            tensor = tensor.T
        elif rnn_based_or_rwkv and ("time_mix_" in key or "time_" in key):
            tensor = tensor.reshape(1, 1, -1)
        if tensor.dtype != get_dtype(dtype):
            tensor = tensor.astype(
                get_dtype(dtype)
            )  # ignores double allocation on GPUs
        key = (
            key.replace(".kernel", ".weight")
            .replace(".embedding", ".weight")
            .replace(".scale", ".weight")
        )

        torch_state_dict[key] = jax2pt(tensor)

    return torch_state_dict


def easystate_to_huggingface_model(
    state,
    config,
    base_huggingface_module: transformers.PreTrainedModel,
    base_huggingface_module_kwarguments=None,
    dtype=jnp.float16,
    transpose_needed=None,
    transpose_not_needed=None,
    select_params_field: bool = True,
    rnn_based_or_rwkv: bool = False,
    auto_correct: bool = True,
    use_meta_torch: bool = True,
):
    if not rnn_based_or_rwkv and auto_correct:
        if isinstance(
            base_huggingface_module, transformers.RwkvForCausalLM
        ) or isinstance(base_huggingface_module, transformers.RwkvModel):
            logger.warning(
                "Rnn Based Model detected 'setting `rnn_based_or_rwkv = True`' for correct weight handling"
            )
            rnn_based_or_rwkv = True
    if base_huggingface_module_kwarguments is None:
        base_huggingface_module_kwarguments = {}

    if auto_correct:
        for k, v in state.unsafe_dict(config.__dict__).items():
            if not hasattr(config, k):
                setattr(config, k, v)

    state_dict = easystate_to_torch(
        state=state,
        dtype=dtype,
        transpose_needed=transpose_needed,
        transpose_not_needed=transpose_not_needed,
        select_params_field=select_params_field,
        rnn_based_or_rwkv=rnn_based_or_rwkv,
    )
    import torch

    if use_meta_torch:
        with torch.device("meta"):
            model = base_huggingface_module(
                config=config,
                **base_huggingface_module_kwarguments,
            )
        model.load_state_dict(state_dict, assign=True, strict=True)
    else:
        model = base_huggingface_module(
            config=config,
            **base_huggingface_module_kwarguments,
        )
        model.load_state_dict(state_dict, assign=True, strict=True)
    return model
