import functools
import gc
import warnings
from typing import Sequence, Optional, Tuple, Mapping, Callable, Type, Any

import flax.traverse_util
import jax.numpy
from fjformer import match_partition_rules, make_shard_and_gather_fns

from flax.traverse_util import unflatten_dict
from transformers import AutoConfig, AutoModelForCausalLM

from ..etils.etils import get_logger
from ..transform.easydel_transform import huggingface_to_easydel
from .easydel_modelling_utils import EasyDelFlaxPretrainedModel, EasyDelPretrainedConfig
from ..etils.errors import EasyDelRuntimeError
from jax.sharding import PartitionSpec

logger = get_logger(name=__name__)


def get_modules_by_type(model_type: str) -> Tuple[
    Type[EasyDelPretrainedConfig],
    Type[EasyDelFlaxPretrainedModel],
    Callable
]:
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
        return (
            _LlamaConfig,
            _FlaxLlamaForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_names=["embed_tokens"])
        )
    elif model_type == "gemma":

        from .gemma import GemmaConfig as _GemmaConfig
        from .gemma import FlaxGemmaForCausalLM as _FlaxGemmaForCausalLM
        return (
            _GemmaConfig,
            _FlaxGemmaForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_names=["embed_tokens"])
        )
    elif model_type == "falcon":
        from .falcon import FlaxFalconForCausalLM as _FlaxFalconForCausalLM
        from .falcon import FalconConfig as _FalconConfig
        return (
            _FalconConfig,
            _FlaxFalconForCausalLM,
            functools.partial(
                huggingface_to_easydel,
                embedding_layer_names=["word_embeddings"],
                layer_norm_names=[
                    "input_layernorm",
                    "ln_f",
                    "ln_attn",
                    "ln_mlp",
                    "post_attention_layernorm"
                ]
            )
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
        return (
            _MistralConfig,
            _FlaxMistralForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_names=["embed_tokens"])
        )
    elif model_type == "gptj":
        from .gpt_j import FlaxGPTJForCausalLM as _FlaxGPTJForCausalLM
        from .gpt_j import GPTJConfig as _GPTJConfig
        return (
            _GPTJConfig,
            _FlaxGPTJForCausalLM,
            functools.partial(
                huggingface_to_easydel,
                embedding_layer_names="wte",
                layer_norm_names=[
                    "ln_1", "ln_2", "ln_f"
                ]
            )
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
            functools.partial(
                huggingface_to_easydel,
                embedding_layer_names=["wte", "wpe"],
                layer_norm_names=[
                    "ln_1", "ln_2", "ln_f"
                ]
            )
        )

    elif model_type == "mixtral":
        from .mixtral import FlaxMixtralForCausalLM as _FlaxMixtralForCausalLM
        from .mixtral import MixtralConfig as _MixtralConfig
        return (
            _MixtralConfig,
            _FlaxMixtralForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_names=["embed_tokens"])
        )
    elif model_type == "phi":
        from .phi import FlaxPhiForCausalLM as _FlaxPhiForCausalLM
        from .phi import PhiConfig as _PhiConfig
        return (
            _PhiConfig,
            _FlaxPhiForCausalLM,
            functools.partial(
                huggingface_to_easydel,
                embedding_layer_names=["embed_tokens"],
                layer_norm_names=[
                    "input_layernorm",
                    "final_layernorm",
                    "q_layernorm",
                    "k_layernorm"
                ])
        )
    if model_type == "qwen":
        from .qwen1 import Qwen1Config as _Qwen1Config
        from .qwen1 import FlaxQwenForCausalLM as _FlaxQwen1ForCausalLM
        return (
            _Qwen1Config,
            _FlaxQwen1ForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_names=["wte"])
        )

    if model_type == "qwen2":
        from .qwen2 import Qwen2Config as _Qwen2Config
        from .qwen2 import FlaxQwen2ForCausalLM as _FlaxQwen2ForCausalLM
        return (
            _Qwen2Config,
            _FlaxQwen2ForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_names=["embed_tokens"])
        )
    if model_type == "stablelm":
        from .stablelm import StableLmConfig as _StableLmConfig
        from .stablelm import FlaxStableLmForCausalLM as _FlaxStableLmForCausalLM

        return (
            _StableLmConfig,
            _FlaxStableLmForCausalLM,
            functools.partial(
                huggingface_to_easydel,
                embedding_layer_names=["embed_tokens"],
                layer_norm_names=["input_layernorm", "post_attention_layernorm", "norm"]
            )
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
            precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
            sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
            sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
            query_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            key_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            value_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            bias_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), None, None, None),
            attention_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            use_shard_map: bool = False,
            input_shape: Sequence[int] = (1, 1),
            shard_fns: Optional[Mapping[tuple, Callable] | dict] = None,
            backend: Optional[str] = None,
            config_kwargs: Optional[Mapping[str, Any]] = None,
            **kwargs
    ) -> Tuple[EasyDelFlaxPretrainedModel, dict]:
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
        :param query_partition_spec: PartitionSpec: Specify the partitioning of the query tensor
        :param key_partition_spec: PartitionSpec: Partition the key matrix
        :param value_partition_spec: PartitionSpec: Specify the partitioning of the value tensor
        :param bias_partition_spec: PartitionSpec: Specify the Attention Bias partition spec
        :param attention_partition_spec: PartitionSpec: Specify the partitioning of the attention weights
        :param use_shard_map: bool: whenever to use shard_map for attention
        :param input_shape: typing.Sequence[int]: Specify the shape of the input to the model
        :param shard_fns: Optional[Mapping[tuple, Callable]]: Sharding Function to be used to shard model
        :param backend: typing.Optional[str]: backend to use for model
        :param config_kwargs: Optional[Mapping[str, Any]]: Config kwargs to be added to config before creating module
        :param kwargs: Pass additional arguments to the model and config classes
        :return: A model and parameters

        """

        logger.debug(f"Downloading model config from {pretrained_model_name_or_path}")
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model_type = config.model_type

        cfg, module, trf = get_modules_by_type(model_type)

        logger.debug(f"Downloading model weights from {pretrained_model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        cfg = cfg.from_pretrained(pretrained_model_name_or_path)
        state_dict = model.state_dict()
        logger.debug(f"adding model basic EasyDeL configurations.")
        if hasattr(cfg, 'add_jax_args'):
            cfg.add_jax_args()
        cfg.add_basic_configurations(
            axis_dims=sharding_axis_dims,
            axis_names=sharding_axis_names,
            query_partition_spec=query_partition_spec,
            key_partition_spec=key_partition_spec,
            value_partition_spec=value_partition_spec,
            bias_partition_spec=bias_partition_spec,
            attention_partition_spec=attention_partition_spec,
            backend=backend,
            use_shard_map=use_shard_map,
        )
        if config_kwargs is not None:
            for k, v in config_kwargs.items():
                setattr(cfg, k, v)

        logger.debug("creating easydel model")
        ed_model = module(
            config=cfg,
            _do_init=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            input_shape=input_shape
        )

        needs = [
            s.replace(".kernel", ".weight").replace(".scale", ".weight").replace(".embedding", ".weight") for s in
            list(flax.traverse_util.flatten_dict(ed_model.params_shape_tree, sep=".").keys())
        ]
        for k in list(state_dict.keys()):
            if k not in needs:
                logger.debug(f"removing {k} from weights as it was not needed by flax model")
                del state_dict[k]
        if shard_fns is not None:
            logger.debug("sharding model parameters based on the given shard_fns.")
            if not is_flatten(shard_fns):
                shard_fns = flax.traverse_util.flatten_dict(shard_fns)
        with cfg.jax_mesh():
            logger.debug("converting huggingface-model to easydel-model.")
            params = trf(state_dict, config=config, device=device, shard_fns=shard_fns)
        logger.debug("deleting huggingface-model")

        del state_dict
        del model
        gc.collect()

        if is_flatten(params):
            logger.info("converted parameters are flatten making them unflatten ")
            params = unflatten_dict(params)

        return ed_model, params


class AutoEasyDelConfig:
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
            sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
            query_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp",
                                                                None),
            key_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp",
                                                              None),
            value_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp",
                                                                None),
            bias_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), None, None,
                                                               None),
            attention_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp",
                                                                    "tp", None),
            use_shard_map: bool = False,
            backend: Optional[str] = None,
            **kwargs
    ) -> EasyDelPretrainedConfig:
        """
        The from_pretrained function is a helper function that allows you to instantiate a model from the pretrained
        model repository. It takes as input the name of the model (e.g., 'bert-base-uncased') and returns an instance of
        the class corresponding to your model, with all weights loaded from disk.

        :param cls: Create an instance of the class that called this function
        :param pretrained_model_name_or_path: str: Identify the model in the huggingface model hub
        :param sharding_axis_dims: Sequence[int]: Specify the dimension of each axis in the sharded model
        :param sharding_axis_names: Sequence[str]: Specify the order of sharding
        :param query_partition_spec: PartitionSpec: Specify the partitioning of the query tensor
        :param key_partition_spec: PartitionSpec: Partition the key matrix
        :param value_partition_spec: PartitionSpec: Specify the partitioning of the value tensor
        :param bias_partition_spec: PartitionSpec: Specify the Attention Bias partition spec
        :param attention_partition_spec: PartitionSpec: Specify the partitioning of the attention weights
        :param use_shard_map: bool: whenever to use shard_map for attention
        :param backend: Optional[str]: backend to use for model
        :param kwargs: Pass additional arguments to the model and config classes
        :return: A Model Config

        """

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model_type = config.model_type

        cfg, module, trf = get_modules_by_type(model_type)
        cfg = cfg.from_pretrained(pretrained_model_name_or_path)
        if hasattr(cfg, 'add_jax_args'):
            cfg.add_jax_args()
        cfg.add_basic_configurations(
            axis_dims=sharding_axis_dims,
            axis_names=sharding_axis_names,
            query_partition_spec=query_partition_spec,
            key_partition_spec=key_partition_spec,
            value_partition_spec=value_partition_spec,
            bias_partition_spec=bias_partition_spec,
            attention_partition_spec=attention_partition_spec,
            backend=backend,
            use_shard_map=use_shard_map,
        )

        return cfg


class AutoShardAndGatherFunctions:
    @classmethod
    def from_config(
            cls,
            config: EasyDelPretrainedConfig,
            partition_rules: Optional[Tuple[Tuple[str, PartitionSpec]]] = None,
            flatten: bool = True
    ):
        if partition_rules is None:
            warnings.warn("Using config partition rules from `get_partition_rules(fully_sharded_data_parallel=True)`")
            partition_rules = config.get_partition_rules(True)
        _, module, _ = get_modules_by_type(config.model_type)
        model = module(
            config=config,
            _do_init=False
        )

        partition_specs = match_partition_rules(
            partition_rules,
            model.params_shape_tree
        )
        shard_fns, gather_fns = make_shard_and_gather_fns(
            partition_specs=partition_specs,
            dtype_specs=jax.numpy.float16
        )
        if flatten and not is_flatten(shard_fns):
            gather_fns = flax.traverse_util.flatten_dict(gather_fns)
            shard_fns = flax.traverse_util.flatten_dict(shard_fns)
        elif not flatten and is_flatten(shard_fns):
            gather_fns = flax.traverse_util.unflatten_dict(gather_fns)
            shard_fns = flax.traverse_util.unflatten_dict(shard_fns)

        return shard_fns, gather_fns

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            partition_rules: Optional[Tuple[Tuple[str, PartitionSpec]]] = None,
            flatten: bool = True
    ):
        config = AutoEasyDelConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(
            config=config,
            partition_rules=partition_rules,
            flatten=flatten
        )
