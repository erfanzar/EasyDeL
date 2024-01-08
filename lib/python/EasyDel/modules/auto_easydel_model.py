import functools
import gc
import typing

import jax.numpy

from flax.traverse_util import unflatten_dict
from transformers import AutoConfig, AutoModelForCausalLM

from ..transform.easydel_transform import huggingface_to_easydel
from .easydel_modelling_utils import EasyDelFlaxPretrainedModel
from ..etils.errors import EasyDelRuntimeError


def get_modules_by_type(model_type: str):
    """
    The get_modules_by_type function is a helper function that returns the following:
        1. The config class for the model type specified (e.g., LlamaConfig, FalconConfig)
        2. The Flax Model class for the model type specified (e.g., FlaxLlamaForCausalLM, FlaxFalconForCausalLM)
        3. A function to convert a HuggingFace pretrained checkpoint into an EasyDel checkpoint

    :param model_type: str: Determine which model to use
    :return: A tuple of three elements (BaseConfig,BaseModel,Func To Transform Model from Torch to EasyDeL)
    
    """
    if model_type == "llama":
        from .llama import LlamaConfig as _LlamaConfig
        from .llama import FlaxLlamaForCausalLM as _FlaxLlamaForCausalLM
        from ..transform import llama_convert_hf_to_flax as _llama_convert_hf_to_flax
        return (
            _LlamaConfig,
            _FlaxLlamaForCausalLM,
            _llama_convert_hf_to_flax
        )
    elif model_type == "falcon":
        from .falcon import FlaxFalconForCausalLM as _FlaxFalconForCausalLM
        from .falcon import FalconConfig as _FalconConfig
        from ..transform import falcon_convert_hf_to_flax as _falcon_convert_pt_to_flax

        return (
            _FalconConfig,
            _FlaxFalconForCausalLM,
            _falcon_convert_pt_to_flax
        )
    elif model_type == "mpt":
        from .mosaic_mpt import FlaxMptForCausalLM as _FlaxMptForCausalLM
        from .mosaic_mpt import MptConfig as _MptConfig
        return (
            _MptConfig,
            _FlaxMptForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_names="wte")
        )

    elif model_type == "mistral":
        from .mistral import FlaxMistralForCausalLM as _FlaxMistralForCausalLM
        from .mistral import MistralConfig as _MistralConfig
        from ..transform import mistral_convert_hf_to_flax as _mistral_convert_hf_to_flax
        return (
            _MistralConfig,
            _FlaxMistralForCausalLM,
            _mistral_convert_hf_to_flax
        )
    elif model_type == "gptj":
        from .gpt_j import FlaxGPTJForCausalLM as _FlaxGPTJForCausalLM
        from .gpt_j import GPTJConfig as _GPTJConfig
        return (
            _GPTJConfig,
            _FlaxGPTJForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_names="wte")
        )

    elif model_type == "gpt_neox":
        from .gpt_neo_x import FlaxGPTNeoXForCausalLM as _FlaxGPTNeoXForCausalLM
        from .gpt_neo_x import GPTNeoXConfig as _GPTNeoXConfig

        return (
            _GPTNeoXConfig,
            _FlaxGPTNeoXForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_names="wte")
        )
    elif model_type == "palm":
        from .palm import FlaxPalmForCausalLM as _FlaxPalmForCausalLM
        from .palm import PalmConfig as _PalmConfig
        return (
            _PalmConfig,
            _FlaxPalmForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_names="wte")
        )
    elif model_type == "lt":
        from .lucid_transformer import FlaxLTForCausalLM as _FlaxLTForCausalLM
        from .lucid_transformer import FlaxLTConfig as _FlaxLTConfig

        return (
            _FlaxLTConfig,
            _FlaxLTForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_names="wte")
        )
    elif model_type == "gpt2":
        from .gpt2 import FlaxGPT2LMHeadModel as _FlaxGPT2LMHeadModel
        from .gpt2 import GPT2Config as _GPT2Config

        return (
            _GPT2Config,
            _FlaxGPT2LMHeadModel,
            functools.partial(huggingface_to_easydel, embedding_layer_names=["wte", "wpe"])
        )

    elif model_type == "mixtral":
        from .mixtral import FlaxMixtralForCausalLM as _FlaxMixtralForCausalLM
        from .mixtral import MixtralConfig as _MixtralConfig
        return (
            _MixtralConfig,
            _FlaxMixtralForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_names=["embed_tokens"])
        )
    else:
        raise EasyDelRuntimeError(f'Model Type ({model_type}) is not supported or is not found')


def is_flatten(pytree: dict):
    """
    The is_flatten function checks if the pytree is flattened.
        If it is, then the first key in the dictionary will be a tuple of (mpl, mpl_id).
        Otherwise, it will be an integer representing mpl_id.

    :param pytree: dict: Pass the pytree to the function
    :return: True if the pytree is a flattened tree, and false otherwise
    
    """
    mpl = [k for k in pytree.keys()][0]
    return True if isinstance(mpl, tuple) else False


class AutoEasyDelModelForCausalLM:
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            device=jax.devices('cpu')[0],
            dtype: jax.numpy.dtype = jax.numpy.float32,
            param_dtype: jax.numpy.dtype = jax.numpy.float32,
            precision: jax.lax.Precision = jax.lax.Precision("fastest"),
            sharding_axis_dims: typing.Sequence[int] = (1, -1, 1, 1),
            sharding_axis_names: typing.Sequence[str] = ("dp", "fsdp", "tp", "sp"),
            q_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            k_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            v_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            b_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), None, None, None),
            a_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            use_shard_map: bool = False,
            input_shape: typing.Sequence[int] = (1, 1),
            backend: typing.Optional[str] = None,
            **kwargs
    ) -> typing.Tuple[EasyDelFlaxPretrainedModel, dict]:
        """
        The from_pretrained function is a helper function that allows you to instantiate a model from the pretrained
        model repository. It takes as input the name of the model (e.g., 'bert-base-uncased') and returns an instance of
        the class corresponding to your model, with all weights loaded from disk.

        :param cls: Create an instance of the class that called this function
        :param pretrained_model_name_or_path: str: Identify the model in the huggingface model hub
        :param device: Specify the device on which to run the model
        :param dtype: jax.numpy.dtype: Specify the data type of the model
        :param param_dtype: jax.numpy.dtype: Specify the dtype of the parameters
        :param precision: jax.lax.Precision: Control the precision of the model
        :param sharding_axis_dims: typing.Sequence[int]: Specify the dimension of each axis in the sharded model
        :param sharding_axis_names: typing.Sequence[str]: Specify the order of sharding
        :param q_ps: jax.sharding.PartitionSpec: Specify the partitioning of the query tensor
        :param k_ps: jax.sharding.PartitionSpec: Partition the key matrix
        :param v_ps: jax.sharding.PartitionSpec: Specify the partitioning of the value tensor
        :param b_ps: jax.sharding.PartitionSpec: Specify the Attention Bias partition spec
        :param a_ps: jax.sharding.PartitionSpec: Specify the partitioning of the attention weights
        :param use_shard_map: bool: whenever to use shard_map for attention
        :param input_shape: typing.Sequence[int]: Specify the shape of the input to the model
        :param backend: typing.Optional[str]: backend to use for model
        :param kwargs: Pass additional arguments to the model and config classes
        :return: A model and parameters
        
        """

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model_type = config.model_type

        cfg, module, trf = get_modules_by_type(model_type)

        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        cfg = cfg.from_pretrained(pretrained_model_name_or_path)
        if hasattr(cfg, 'add_jax_args'):
            cfg.add_jax_args()
        cfg.add_partitions(
            axis_dims=sharding_axis_dims,
            axis_names=sharding_axis_names,
            q_ps=q_ps,
            k_ps=k_ps,
            v_ps=v_ps,
            b_ps=b_ps,
            a_ps=a_ps,
            backend=backend,
            use_shard_map=use_shard_map,
        )
        ed_model = module(
            config=cfg,
            _do_init=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            input_shape=input_shape
        )

        params = trf(model.state_dict(), config=config, device=device)
        del model,
        gc.collect()

        if is_flatten(params):
            params = unflatten_dict(params)

        return ed_model, params


class AutoEasyDelConfig:
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            sharding_axis_dims: typing.Sequence[int] = (1, -1, 1, 1),
            sharding_axis_names: typing.Sequence[str] = ("dp", "fsdp", "tp", "sp"),
            q_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            k_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            v_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            b_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), None, None, None),
            a_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            use_shard_map: bool = False,
            backend: typing.Optional[str] = None,
            **kwargs
    ) -> typing.Tuple[EasyDelFlaxPretrainedModel, dict]:
        """
        The from_pretrained function is a helper function that allows you to instantiate a model from the pretrained
        model repository. It takes as input the name of the model (e.g., 'bert-base-uncased') and returns an instance of
        the class corresponding to your model, with all weights loaded from disk.

        :param cls: Create an instance of the class that called this function
        :param pretrained_model_name_or_path: str: Identify the model in the huggingface model hub
        :param sharding_axis_dims: typing.Sequence[int]: Specify the dimension of each axis in the sharded model
        :param sharding_axis_names: typing.Sequence[str]: Specify the order of sharding
        :param q_ps: jax.sharding.PartitionSpec: Specify the partitioning of the query tensor
        :param k_ps: jax.sharding.PartitionSpec: Partition the key matrix
        :param v_ps: jax.sharding.PartitionSpec: Specify the partitioning of the value tensor
        :param b_ps: jax.sharding.PartitionSpec: Specify the Attention Bias partition spec
        :param a_ps: jax.sharding.PartitionSpec: Specify the partitioning of the attention weights
        :param use_shard_map: bool: whenever to use shard_map for attention
        :param backend: typing.Optional[str]: backend to use for model
        :param kwargs: Pass additional arguments to the model and config classes
        :return: A Model Config

        """

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model_type = config.model_type

        cfg, module, trf = get_modules_by_type(model_type)
        cfg = cfg.from_pretrained(pretrained_model_name_or_path)
        if hasattr(cfg, 'add_jax_args'):
            cfg.add_jax_args()
        cfg.add_partitions(
            axis_dims=sharding_axis_dims,
            axis_names=sharding_axis_names,
            q_ps=q_ps,
            k_ps=k_ps,
            v_ps=v_ps,
            b_ps=b_ps,
            a_ps=a_ps,
            backend=backend,
            use_shard_map=use_shard_map,
        )

        return cfg
