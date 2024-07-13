import functools
import gc
import re
import warnings
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Type

# import fjformer.linen.linen
import flax.traverse_util
import jax.numpy
from fjformer import make_shard_and_gather_fns, match_partition_rules
from flax.traverse_util import unflatten_dict
from jax.sharding import PartitionSpec
from transformers import AutoConfig, AutoModelForCausalLM

from easydel.etils.errors import EasyDeLRuntimeError
from easydel.etils.etils import get_logger
from easydel.etils.partition_module import PartitionAxis
from easydel.models.modelling_utils import (
    BaseNNXModule,
    EDPretrainedConfig,
)
from easydel.transform.transform import torch_dict_to_flatten_dict
from easydel.utils.traversal import nnx_init, attech_tree_to_nnx_model

logger = get_logger(name=__name__)


def get_models_by_type(
    model_type: str,
) -> Tuple[Type[EDPretrainedConfig], Type[BaseNNXModule] | Any, partial | Any]:
    """
    The get_models_by_type function is a helper function that returns the following:
        1. The config class for the model type specified (e.g., LlamaConfig, FalconConfig)
        2. The  Model class for the model type specified (e.g., LlamaForCausalLM, FalconForCausalLM)
        3. A function to convert a HuggingFace pretrained checkpoint into an easydel checkpoint

    :param model_type: str: Determine which model to use
    :return: A tuple of three elements (BaseConfig,BaseModel,Func To Transform Model from Torch to EasyDeL)

    """
    if model_type == "llama":
        from easydel.models.llama import LlamaForCausalLM as _LlamaForCausalLM
        from easydel.models.llama import LlamaConfig as _LlamaConfig

        return (
            _LlamaConfig,
            _LlamaForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["embed_tokens"],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "gemma":
        from easydel.models.gemma import GemmaForCausalLM as _GemmaForCausalLM
        from easydel.models.gemma import GemmaConfig as _GemmaConfig

        return (
            _GemmaConfig,
            _GemmaForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["embed_tokens"],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "gemma2":
        from easydel.models.gemma2 import (
            Gemma2ForCausalLM as _Gemma2ForCausalLM,
        )
        from easydel.models.gemma2 import Gemma2Config as _Gemma2Config

        return (
            _Gemma2Config,
            _Gemma2ForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["embed_tokens"],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "falcon":
        from easydel.models.falcon import FalconConfig as _FalconConfig
        from easydel.models.falcon import (
            FalconForCausalLM as _FalconForCausalLM,
        )

        return (
            _FalconConfig,
            _FalconForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["word_embeddings"],
                layer_norm_names=[
                    "input_layernorm",
                    "ln_f",
                    "ln_attn",
                    "ln_mlp",
                    "post_attention_layernorm",
                ],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "mpt":
        from easydel.models.mosaic_mpt import MptForCausalLM as _MptForCausalLM
        from easydel.models.mosaic_mpt import MptConfig as _MptConfig

        return (
            _MptConfig,
            _MptForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["wte"],
                rnn_based_or_rwkv=False,
                layer_norm_names=["norm_1", "norm_2", "norm_f"],
                lm_head_name="lm_head",
            ),
        )

    elif model_type == "mistral":
        from easydel.models.mistral import (
            MistralForCausalLM as _MistralForCausalLM,
        )
        from easydel.models.mistral import MistralConfig as _MistralConfig

        return (
            _MistralConfig,
            _MistralForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["embed_tokens"],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "gptj":
        from easydel.models.gpt_j import GPTJForCausalLM as _GPTJForCausalLM
        from easydel.models.gpt_j import GPTJConfig as _GPTJConfig

        return (
            _GPTJConfig,
            _GPTJForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names="wte",
                layer_norm_names=[
                    "ln_1",
                    "ln_2",
                    "ln_f",
                ],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )

    elif model_type == "gpt_neox":
        from easydel.models.gpt_neo_x import (
            GPTNeoXForCausalLM as _GPTNeoXForCausalLM,
        )
        from easydel.models.gpt_neo_x import GPTNeoXConfig as _GPTNeoXConfig

        return (
            _GPTNeoXConfig,
            _GPTNeoXForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names="wte",
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )

    elif model_type == "gpt2":
        from easydel.models.gpt2 import GPT2LMHeadModel as _GPT2LMHeadModel
        from easydel.models.gpt2 import GPT2Config as _GPT2Config

        return (
            _GPT2Config,
            _GPT2LMHeadModel,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["wte", "wpe"],
                layer_norm_names=["ln_1", "ln_2", "ln_f"],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "mixtral":
        from easydel.models.mixtral import (
            MixtralForCausalLM as _MixtralForCausalLM,
        )
        from easydel.models.mixtral import MixtralConfig as _MixtralConfig

        return (
            _MixtralConfig,
            _MixtralForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["embed_tokens"],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "phi":
        from easydel.models.phi import PhiForCausalLM as _PhiForCausalLM
        from easydel.models.phi import PhiConfig as _PhiConfig

        return (
            _PhiConfig,
            _PhiForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["embed_tokens"],
                layer_norm_names=[
                    "input_layernorm",
                    "final_layernorm",
                    "q_layernorm",
                    "k_layernorm",
                ],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "qwen":
        from easydel.models.qwen1 import Qwen1ForCausalLM as _Qwen1ForCausalLM
        from easydel.models.qwen1 import Qwen1Config as _Qwen1Config

        return (
            _Qwen1Config,
            _Qwen1ForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["wte"],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )

    elif model_type == "qwen2":
        from easydel.models.qwen2 import Qwen2ForCausalLM as _Qwen2ForCausalLM
        from easydel.models.qwen2 import Qwen2Config as _Qwen2Config

        return (
            _Qwen2Config,
            _Qwen2ForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["embed_tokens"],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "stablelm":
        from easydel.models.stablelm import (
            StableLmForCausalLM as _StableLmForCausalLM,
        )
        from easydel.models.stablelm import StableLmConfig as _StableLmConfig

        return (
            _StableLmConfig,
            _StableLmForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["embed_tokens"],
                layer_norm_names=[
                    "input_layernorm",
                    "post_attention_layernorm",
                    "norm",
                    "norms",
                ],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "rwkv":
        from easydel.models.rwkv import RwkvForCausalLM as _RwkvForCausalLM
        from easydel.models.rwkv import RwkvConfig as _RwkvConfig

        return (
            _RwkvConfig,
            _RwkvForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["embeddings"],
                layer_norm_names=["ln_out", "ln2", "ln1", "pre_ln"],
                rnn_based_or_rwkv=True,
                lm_head_name="head",
            ),
        )
    elif model_type == "mamba":
        from easydel.models.mamba import MambaForCausalLM as _MambaForCausalLM
        from easydel.models.mamba import MambaConfig as _MambaConfig

        return (
            _MambaConfig,
            _MambaForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["embeddings"],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "grok-1":
        from easydel.models.grok_1 import Grok1ForCausalLM as _Grok1ForCausalLM
        from easydel.models.grok_1 import Grok1Config as _Grok1Config

        return (
            _Grok1Config,
            _Grok1ForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["embed_tokens"],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "qwen2_moe":
        from easydel.models.qwen2_moe import (
            Qwen2MoeForCausalLM as _Qwen2MoeForCausalLM,
        )
        from easydel.models.qwen2_moe import Qwen2MoeConfig as _Qwen2MoeConfig

        return (
            _Qwen2MoeConfig,
            _Qwen2MoeForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["embed_tokens"],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "cohere":
        from easydel.models.cohere import CohereConfig as _CohereConfig
        from easydel.models.cohere import (
            CohereForCausalLM as _CohereForCausalLM,
        )

        return (
            _CohereConfig,
            _CohereForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["embed_tokens"],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "dbrx":
        from easydel.models.dbrx import DbrxConfig as _DbrxConfig
        from easydel.models.dbrx import DbrxForCausalLM as _DbrxForCausalLM

        return (
            _DbrxConfig,
            _DbrxForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["wte"],
                rnn_based_or_rwkv=False,
                layer_norm_names=["norm_1", "norm_2", "norm_f"],
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "phi3":
        from easydel.models.phi3 import Phi3ForCausalLM as _Phi3ForCausalLM
        from easydel.models.phi3 import Phi3Config as _Phi3Config

        return (
            _Phi3Config,
            _Phi3ForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["embed_tokens"],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )

    elif model_type == "arctic":
        from easydel.models.arctic import ArcticConfig as _ArcticConfig
        from easydel.models.arctic import (
            ArcticForCausalLM as _ArcticForCausalLM,
        )

        return (
            _ArcticConfig,
            _ArcticForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["embed_tokens"],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "openelm":
        from easydel.models.openelm import (
            OpenELMForCausalLM as _OpenELMForCausalLM,
        )
        from easydel.models.openelm import OpenELMConfig as _OpenELMConfig

        return (
            _OpenELMConfig,
            _OpenELMForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["token_embeddings"],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "deepseek_v2":
        from easydel.models.deepseek_v2 import DeepseekV2Config as _DeepseekV2Config
        from easydel.models.deepseek_v2 import (
            DeepseekV2ForCausalLM as _DeepseekV2ForCausalLM,
        )

        return (
            _DeepseekV2Config,
            _DeepseekV2ForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["embed_tokens"],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    elif model_type == "olmo":
        from easydel.models.olmo import OlmoForCausalLM as _OlmoForCausalLM
        from easydel.models.olmo import OlmoConfig as _OlmoConfig

        return (
            _OlmoConfig,
            _OlmoForCausalLM,
            functools.partial(
                torch_dict_to_flatten_dict,
                embedding_layer_names=["embed_tokens"],
                rnn_based_or_rwkv=False,
                lm_head_name="lm_head",
            ),
        )
    raise EasyDeLRuntimeError(
        f"Model Type ({model_type}) is not supported or is not found"
    )


def is_flatten(pytree: dict):
    """The is_flatten function checks if the pytree is flattened.
        If it is, then the first key in the dictionary will be a tuple of (mpl, mpl_id).
        Otherwise, it will be an integer representing mpl_id.

    Args:
        pytree: dict: Pass the pytree to the function

    Returns:
        True if the pytree is a flattened tree, and false otherwise
    """
    mpl = [k for k in pytree.keys()][0]
    return True if isinstance(mpl, tuple) else False


class AutoEasyDeLModelForCausalLM:
    """This class provides a convenient way to load and shard pretrained causal language models from the Hugging Face Hub
    and convert them into EasyDeL compatible models. It utilizes the EasyDeL library for distributed training and inference
    with JAX.

    This class inherits from the `BaseNNXModule` class, providing functionalities for model loading,
    parameter sharding, and interaction with the EasyDeL framework.

    Attributes:
        None

    Examples:
        ```python
        import jax
        from easydel import AutoEasyDeLModelForCausalLM

        # Load a GPT-2 model on a single CPU
        model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
            "gpt2",
            device=jax.devices("cpu")[0]
        )

        # Load a GPT-2 model sharded across 8 GPUs with data parallelism (DP) and fully sharded data parallelism (FSDP)
        model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
            "gpt2",
            sharding_axis_dims=(1, 8, 1, 1),
            sharding_axis_names=("dp", "fsdp", "tp", "sp"),
            device=jax.devices("cpu")[0], # offload to CPU [OPTIONAL]
            from_torch=True,
        )
        ```
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device=jax.devices("cpu")[0],
        dtype: jax.numpy.dtype = jax.numpy.float32,
        param_dtype: jax.numpy.dtype = jax.numpy.float32,
        precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
        sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
        sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
        partition_axis: PartitionAxis = PartitionAxis(),
        shard_attention_computation: bool = True,
        shard_fns: Optional[Mapping[tuple, Callable] | dict] = None,
        backend: Optional[str] = None,
        config_kwargs: Optional[Mapping[str, Any]] = None,
        auto_shard_params: bool = False,
        partition_rules: Optional[Tuple[Tuple[str, PartitionSpec], ...]] = None,
        load_in_8bit: bool = False,
        bit_targeted_params: Optional[List[str]] = None,
        verbose_params: bool = False,
        safe: bool = True,
        from_torch: bool = True,
        **kwargs,
    ) -> Tuple[BaseNNXModule, dict]:
        """Loads and shards a pretrained causal language model from the Hugging Face Hub and converts it into an
        EasyDeL compatible model.

        Args:
            pretrained_model_name_or_path (str): Path or name of the pretrained model in the Hugging Face Hub.
            device (jax.Array, optional): Device to load the model on. Defaults to the first CPU.
            dtype (jax.numpy.dtype, optional): Data type of the model. Defaults to jax.numpy.float32.
            param_dtype (jax.numpy.dtype, optional): Data type of the model parameters. Defaults to jax.numpy.float32.
            precision (jax.lax.Precision, optional): Precision for computations. Defaults to jax.lax.Precision("fastest").
            sharding_axis_dims (Sequence[int], optional): Dimensions of each sharding axis. Defaults to (1, -1, 1, 1).
            sharding_axis_names (Sequence[str], optional): Names of the sharding axes. Defaults to ("dp", "fsdp", "tp", "sp").
            partition_axis (PartitionAxis) : PartitionAxis is new module used for partitioning arrays in easydel.
            shard_attention_computation (bool, optional): Whether to shard attention computation. Defaults to True.
            shard_fns (Optional[Mapping[tuple, Callable] | dict], optional): Sharding functions to use for the model. If None,
                auto-sharding is used if auto_shard_params is True. Defaults to None.
            backend (Optional[str], optional): Backend to use for the model. Defaults to None.
            config_kwargs (Optional[Mapping[str, Any]], optional): Configuration keyword arguments to pass to the model config.
                Defaults to None.
            auto_shard_params (bool, optional): Whether to automatically shard the model parameters. Defaults to False.
            partition_rules (Optional[Tuple[Tuple[str, PartitionSpec]]], optional): Custom partition rules for parameter
                sharding. If not None, shard_fns should also be provided. Defaults to None.
            load_in_8bit (bool, optional): Whether to load the model parameters in 8-bit precision. Defaults to False.
            bit_targeted_params (Optional[List[str]], optional): List of parameter names to convert to 8-bit precision. If
                None and load_in_8bit is True, all kernels and embeddings are converted to 8-bit. Defaults to None.
            verbose_params (bool): whenever to log number of parameters in converting state.
            safe (bool): whenever to use safetensors to load engine or parameters (requires engine or parameters to be saved with safe=True while saving them)
            from_torch (bool): whenever to load the model from transformers-pytorch.
            **kwargs: Additional keyword arguments to pass to the model and config classes.

        Returns:
            Tuple[BaseNNXModule, dict]: A tuple containing the EasyDeL model and the loaded and sharded
                model parameters.
        """
        if from_torch:
            return cls._from_torch(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                param_dtype=param_dtype,
                dtype=dtype,
                shard_fns=shard_fns,
                auto_shard_params=auto_shard_params,
                precision=precision,
                backend=backend,
                verbose_params=verbose_params,
                partition_axis=partition_axis,
                load_in_8bit=load_in_8bit,
                partition_rules=partition_rules,
                bit_targeted_params=bit_targeted_params,
                sharding_axis_names=sharding_axis_names,
                sharding_axis_dims=sharding_axis_dims,
                config_kwargs=config_kwargs,
                device=device,
                shard_attention_computation=shard_attention_computation,
                **kwargs,
            )
        with jax.default_device(device):
            return cls._from_easydel_params(
                auto_shard_params=auto_shard_params,
                partition_axis=partition_axis,
                sharding_axis_dims=sharding_axis_dims,
                sharding_axis_names=sharding_axis_names,
                shard_fns=shard_fns,
                param_dtype=param_dtype,
                config_kwargs=config_kwargs,
                partition_rules=partition_rules,
                precision=precision,
                dtype=dtype,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                safe=safe,
            )

    @staticmethod
    def _from_torch(
        pretrained_model_name_or_path,
        device,
        dtype: jax.numpy.dtype,
        param_dtype: jax.numpy.dtype,
        precision: Optional[jax.lax.Precision],
        sharding_axis_dims: Sequence[int],
        sharding_axis_names: Sequence[str],
        partition_axis: PartitionAxis,
        shard_attention_computation: bool,
        shard_fns: Optional[Mapping[tuple, Callable] | dict],
        backend: Optional[str],
        config_kwargs: Optional[Mapping[str, Any]],
        auto_shard_params: bool,
        partition_rules: Optional[Tuple[Tuple[str, PartitionSpec], ...]],
        load_in_8bit: bool,
        bit_targeted_params: Optional[List[str]],
        verbose_params: bool,
        **kwargs,
    ):
        try:
            import torch

            if torch.cuda.is_available():

                def _clear():
                    gc.collect()
                    torch.cuda.empty_cache()

            else:

                class torch:
                    bfloat16 = None

                def _clear():
                    gc.collect()

        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "in order to load model from torch you should install torch first "
                "run `pip install torch`"
            )

        logger.debug(f"Downloading model config from {pretrained_model_name_or_path}")
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model_type: str = config.model_type

        cfg, module, trf = get_models_by_type(model_type)

        logger.debug(f"Downloading model weights from {pretrained_model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        if verbose_params:
            print(
                f"PyTorch - HF Model contains {sum(p.numel() for p in model.parameters()) / 1e9} Billion Parameters"
            )
        cfg = cfg.from_pretrained(pretrained_model_name_or_path)
        state_dict = model.state_dict()

        # Clear and collect memory after deleting the model
        del model
        _clear()

        logger.debug("adding model basic EasyDeL configurations.")
        if hasattr(cfg, "add_jax_args"):
            cfg.add_jax_args()
        cfg.add_basic_configurations(
            axis_dims=sharding_axis_dims,
            axis_names=sharding_axis_names,
            partition_axis=partition_axis,
            backend=backend,
            shard_attention_computation=shard_attention_computation,
        )
        if config_kwargs is not None:
            for k, v in config_kwargs.items():
                setattr(cfg, k, v)
        logger.debug("creating easydel model")
        ed_model = nnx_init(
            module,
            config=cfg,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )

        if shard_fns is not None:
            if auto_shard_params:
                warnings.warn(
                    "`auto_shard_params` will be ignored since you are passing custom sharding functions"
                )
            logger.debug("sharding model parameters based on the given shard_fns.")
            if not is_flatten(shard_fns):
                shard_fns = flax.traverse_util.flatten_dict(shard_fns)
        elif auto_shard_params:
            shard_fns, _ = AutoShardAndGatherFunctions.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                partition_rules=partition_rules,
                sharding_axis_dims=sharding_axis_dims,
                sharding_axis_names=sharding_axis_names,
                partition_axis=partition_axis,
                shard_attention_computation=shard_attention_computation,
                backend=backend,
                config_kwargs=config_kwargs,
            )
        logger.debug("converting huggingface-model to easydel-model.")
        params_pattern_selection = None
        if load_in_8bit:
            if bit_targeted_params is None:
                warnings.warn(
                    "since `bit_targeted_params` is set to None, auto loader will convert all of"
                    " kernels(weights) and embeddings to 8bit by default"
                )
                bit_targeted_params = ["kernel", "embedding"]

                params_pattern_selection = re.compile(
                    "({})".format("|".join(bit_targeted_params))
                )
        uses_tie_word_embedding = getattr(config, "tie_word_embeddings", False)
        params = trf(
            state_dict,
            config=config,
            device=device,
            shard_fns=shard_fns,
            convert_to_8bit=load_in_8bit,
            params_pattern_selection=params_pattern_selection,
            remove_state_dict=True,
            uses_tie_word_embedding=uses_tie_word_embedding,
            dtype=param_dtype,
        )

        # Clear and collect memory after converting the model
        del state_dict
        _clear()

        if is_flatten(params):
            logger.info("converted parameters are flatten making them unflatten ")
            params = unflatten_dict(params)

        if verbose_params:
            print(
                f"JAX - EasyDeL Model contains "
                f"{sum(n.size for n in jax.tree_util.tree_flatten(flax.core.unfreeze(params))[0]) / 1e9}"
                f" Billion Parameters"
            )
        ed_model = attech_tree_to_nnx_model(model=ed_model, tree=params)
        return ed_model

    @staticmethod
    def _from_easydel_params(
        pretrained_model_name_or_path,
        dtype: jax.numpy.dtype,
        param_dtype: jax.numpy.dtype,
        precision: Optional[jax.lax.Precision],
        sharding_axis_dims: Sequence[int],
        sharding_axis_names: Sequence[str],
        partition_axis: PartitionAxis,
        shard_fns: Optional[Mapping[tuple, Callable] | dict],
        config_kwargs: Optional[Mapping[str, Any]],
        auto_shard_params: bool,
        partition_rules: Optional[Tuple[Tuple[str, PartitionSpec], ...]],
        safe: bool,
        # load_in_8bit: bool,
        # bit_targeted_params: Optional[List[str]],
    ):
        from easydel.models.modelling_utils import BaseNNXModule

        return BaseNNXModule.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            dtype=dtype,
            precision=precision,
            param_dtype=param_dtype,
            partition_axis=partition_axis,
            auto_shard_params=auto_shard_params,
            shard_fns=shard_fns,
            sharding_axis_dims=sharding_axis_dims,
            sharding_axis_names=sharding_axis_names,
            config_kwargs=config_kwargs,
            partition_rules=partition_rules,
            safe=safe,
        )


class AutoEasyDeLConfig:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
        sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
        partition_axis: PartitionAxis = PartitionAxis(),
        shard_attention_computation: bool = True,
        backend: Optional[str] = None,
        from_torch: bool = False,
        **kwargs,
    ) -> EDPretrainedConfig:
        """The from_pretrained function is a helper function that allows you to instantiate a model from the pretrained
        model repository. It takes as input the name of the model (e.g., 'bert-base-uncased') and returns an instance of
        the class corresponding to your model, with all weights loaded from disk.

        Args:
            cls: Create an instance of the class that called this function
            pretrained_model_name_or_path: str: Identify the model in the huggingface model hub
            sharding_axis_dims: Sequence[int]: Specify the dimension of each axis in the sharded model
            sharding_axis_names: Sequence[str]: Specify the order of sharding
            partition_axis (PartitionAxis) : PartitionAxis is new module used for partitioning arrays in easydel.
            shard_attention_computation: bool: whenever to use shard_map for attention
            backend: Optional[str]: backend to use for model
            from_torch: should config be loaded from torch models or not.
            **kwargs: Pass additional arguments to the model and config classes
        generation process

        Returns:
            A Model Config
        """
        cls_main = AutoConfig if from_torch else EDPretrainedConfig
        config = cls_main.from_pretrained(pretrained_model_name_or_path)
        model_type: str = config.model_type

        cfg, module, trf = get_models_by_type(model_type)
        cfg = cfg.from_pretrained(pretrained_model_name_or_path)
        if hasattr(cfg, "add_jax_args"):
            cfg.add_jax_args()
        cfg.add_basic_configurations(
            axis_dims=sharding_axis_dims,
            axis_names=sharding_axis_names,
            partition_axis=partition_axis,
            backend=backend,
            shard_attention_computation=shard_attention_computation,
        )

        return cfg


class AutoShardAndGatherFunctions:
    """
    A class to automatically generate shard and gather functions for a given model configuration.

    This class provides two methods to generate shard and gather functions:

    - `from_config`: Generates functions based on a provided `EDPretrainedConfig` object.
    - `from_pretrained`: Generates functions based on a pretrained model name or path.

    Attributes:
        None

    Methods:
        from_config: Generates shard and gather functions based on a provided `EDPretrainedConfig` object.
        from_pretrained: Generates functions based on a pretrained model name or path.
    """

    @classmethod
    def from_config(
        cls,
        config: EDPretrainedConfig,
        partition_rules: Optional[Tuple[Tuple[str, PartitionSpec]]] = None,
        flatten: bool = True,
        depth_target: Optional[List[str]] = None,
    ):
        """
        Generates shard and gather functions based on a provided `EDPretrainedConfig` object.

        Args:
            config: An `EDPretrainedConfig` object containing the model configuration.
            partition_rules: A tuple of tuples containing partition rule names and `PartitionSpec` objects.
                If None, uses the default partition rules from the `config`.
            flatten: Whether to flatten the shard and gather functions. Defaults to True.
            depth_target: Pad the sharding to depth, for example make {params:tensor} with depth_target = ["row"] to {row:{params:tensor}}. Defaults to None.

        Returns:
            A tuple containing the shard and gather functions.
        """
        if partition_rules is None:
            warnings.warn(
                "Using config partition rules from `get_partition_rules(fully_sharded_data_parallel=True)`"
            )
            partition_rules = config.get_partition_rules(True)
        _, module, _ = get_models_by_type(config.model_type)
        model = module(config=config)

        partition_specs = match_partition_rules(
            partition_rules, model.params_shape_tree
        )
        shard_fns, gather_fns = make_shard_and_gather_fns(
            partition_specs=partition_specs,
            mesh=config.mesh,
        )
        if depth_target is not None:
            for dp in depth_target[::-1]:
                gather_fns = {dp: gather_fns}
                shard_fns = {dp: shard_fns}
        if flatten and not is_flatten(shard_fns):
            gather_fns = flax.traverse_util.flatten_dict(gather_fns)
            shard_fns = flax.traverse_util.flatten_dict(shard_fns)
        elif not flatten and is_flatten(shard_fns):
            gather_fns = flax.traverse_util.unflatten_dict(gather_fns)
            shard_fns = flax.traverse_util.unflatten_dict(shard_fns)

        return shard_fns, gather_fns

    @staticmethod
    def from_params(params, partition_rules, mesh):
        partition_specs = match_partition_rules(partition_rules, params)
        return make_shard_and_gather_fns(
            partition_specs=partition_specs,
            mesh=mesh,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
        sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
        partition_axis: PartitionAxis = PartitionAxis(),
        shard_attention_computation: bool = True,
        backend: Optional[str] = None,
        partition_rules: Optional[Tuple[Tuple[str, PartitionSpec]]] = None,
        flatten: bool = True,
        config_kwargs: Optional[Mapping[str, Any]] = None,
        depth_target: Optional[List[str]] = None,
        from_torch: bool = False,
    ) -> Tuple[Mapping[str, Callable], Mapping[str, Callable]]:
        """
        Generates shard and gather functions based on a pretrained model name or path.

        Args:
            pretrained_model_name_or_path: The name or path of the pretrained model.
            sharding_axis_dims: The dimensions of the sharding axes. Defaults to (1, -1, 1, 1).
            sharding_axis_names: The names of the sharding axes. Defaults to ("dp", "fsdp", "tp", "sp").
            partition_axis (PartitionAxis) : PartitionAxis is new module used for partitioning arrays in easydel.
            shard_attention_computation: Whether to shard the attention computation. Defaults to True.
            backend: The backend to use for sharding. Defaults to None.
            partition_rules: A tuple of tuples containing partition rule names and `PartitionSpec` objects.
                If None, uses the default partition rules from the `config`.
            flatten: Whether to flatten the shard and gather functions. Defaults to True.
            config_kwargs: Additional keyword arguments to pass to the `AutoEasyDeLConfig` constructor. Defaults to None.
            depth_target: Pad the sharding to depth, for example make {params:tensor} with depth_target = ["row"] to {row:{params:tensor}}. Defaults to None.
            from_torch: should config be loaded from torch models or not.
        Returns:
            A tuple containing the shard and gather functions.
        """
        config = AutoEasyDeLConfig.from_pretrained(
            pretrained_model_name_or_path,
            sharding_axis_dims=sharding_axis_dims,
            sharding_axis_names=sharding_axis_names,
            partition_axis=partition_axis,
            shard_attention_computation=shard_attention_computation,
            backend=backend,
            from_torch=from_torch,
        )
        if config_kwargs is not None:
            for k, v in config_kwargs.items():
                setattr(config, k, v)
        return cls.from_config(
            config=config,
            partition_rules=partition_rules,
            flatten=flatten,
            depth_target=depth_target,
        )
