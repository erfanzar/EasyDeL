import functools
import gc
import typing

import jax.numpy
import torch.cuda
from flax.traverse_util import unflatten_dict
from transformers import AutoConfig, PretrainedConfig, AutoModelForCausalLM

from ..transform.easydel_transform import huggingface_to_easydel

from transformers import FlaxPreTrainedModel


class EasyDelRunTimeError(Exception):
    ...


def get_modules_by_type(model_type: str):
    if model_type == "llama":
        from . import LlamaConfig as _LlamaConfig
        from . import FlaxLlamaForCausalLM as _FlaxLlamaForCausalLM
        from ..transform import llama_convert_hf_to_flax as _llama_convert_hf_to_flax
        return (
            _LlamaConfig,
            _FlaxLlamaForCausalLM,
            _llama_convert_hf_to_flax
        )
    elif model_type == "falcon":
        from . import FlaxFalconForCausalLM as _FlaxFalconForCausalLM
        from . import FalconConfig as _FalconConfig
        from ..transform import falcon_convert_hf_to_flax as _falcon_convert_pt_to_flax

        return (
            _FalconConfig,
            _FlaxFalconForCausalLM,
            _falcon_convert_pt_to_flax
        )
    elif model_type == "mpt":
        from . import FlaxMptForCausalLM as _FlaxMptForCausalLM
        from . import MptConfig as _MptConfig
        return (
            _MptConfig,
            _FlaxMptForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_name="wte")
        )

    elif model_type == "mistral":
        from . import FlaxMistralForCausalLM as _FlaxMistralForCausalLM
        from . import MistralConfig as _MistralConfig
        from ..transform import mistral_convert_hf_to_flax as _mistral_convert_hf_to_flax
        return (
            _MistralConfig,
            _FlaxMistralForCausalLM,
            _mistral_convert_hf_to_flax
        )
    elif model_type == "gptj":
        from . import FlaxGPTJForCausalLM as _FlaxGPTJForCausalLM
        from . import GPTJConfig as _GPTJConfig
        return (
            _GPTJConfig,
            _FlaxGPTJForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_name="wte")
        )

    elif model_type == "gpt_neox":
        from . import FlaxGPTNeoXForCausalLM as _FlaxGPTNeoXForCausalLM
        from . import GPTNeoXConfig as _GPTNeoXConfig

        return (
            _GPTNeoXConfig,
            _FlaxGPTNeoXForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_name="wte")
        )
    elif model_type == "palm":
        from . import FlaxPalmForCausalLM as _FlaxPalmForCausalLM
        from . import PalmConfig as _PalmConfig
        return (
            _PalmConfig,
            _FlaxPalmForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_name="wte")
        )
    elif model_type == "lt":
        from . import FlaxLTForCausalLM as _FlaxLTForCausalLM
        from . import FlaxLTConfig as _FlaxLTConfig

        return (
            _FlaxLTConfig,
            _FlaxLTForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_name="wte")
        )

    else:
        raise EasyDelRunTimeError(f'Model Type ({model_type}) is not supported or is not found')


def is_flatten(pytree: dict):
    mpl = [k for k in pytree.keys()][0]
    return True if isinstance(mpl, tuple) else False


class AutoEasyDelModelForCausalLM:
    @classmethod
    def from_pretrained(
            cls,
            repo_id: str,
            device,
            dtype: jax.numpy.dtype = jax.numpy.float32,
            param_dtype: jax.numpy.dtype = jax.numpy.float32,
            precision: jax.lax.Precision = jax.lax.Precision('fastest'),
            # =jax.devices('cpu')[0]
            **kwargs
    ) -> typing.Union[FlaxPreTrainedModel, dict]:
        """
        returns Model and Parameters for the Model
        """
        config = AutoConfig.from_pretrained(repo_id)
        model_type = config.model_type

        cfg, module, trf = get_modules_by_type(model_type)

        model = AutoModelForCausalLM.from_pretrained(repo_id, **kwargs)
        cfg = cfg.from_pretrained(repo_id)
        if hasattr(cfg, 'add_jax_args'):
            cfg.add_jax_args()
        ed_model = module(
            config=cfg,
            _do_init=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision
        )

        params = trf(model.state_dict(), config=config, device=device)
        del model,
        gc.collect()
        torch.cuda.empty_cache()

        if is_flatten(params):
            params = unflatten_dict(params)

        return ed_model, params
