import functools
import gc
import typing

import jax.numpy

from flax.traverse_util import unflatten_dict
from transformers import AutoConfig, PretrainedConfig, AutoModelForCausalLM

from EasyDel.transform.easydel_transform import huggingface_to_easydel

from transformers import FlaxPreTrainedModel


class EasyDelRunTimeError(Exception):
    ...


def get_modules_by_type(model_type: str):
    if model_type == "llama":
        from EasyDel.modules.llama import LlamaConfig as _LlamaConfig
        from EasyDel.modules.llama import FlaxLlamaForCausalLM as _FlaxLlamaForCausalLM
        from EasyDel.transform import llama_convert_hf_to_flax as _llama_convert_hf_to_flax
        return (
            _LlamaConfig,
            _FlaxLlamaForCausalLM,
            _llama_convert_hf_to_flax
        )
    elif model_type == "falcon":
        from EasyDel.modules.falcon import FlaxFalconForCausalLM as _FlaxFalconForCausalLM
        from EasyDel.modules.falcon import FalconConfig as _FalconConfig
        from EasyDel.transform import falcon_convert_hf_to_flax as _falcon_convert_pt_to_flax

        return (
            _FalconConfig,
            _FlaxFalconForCausalLM,
            _falcon_convert_pt_to_flax
        )
    elif model_type == "mpt":
        from EasyDel.modules.mosaic_mpt import FlaxMptForCausalLM as _FlaxMptForCausalLM
        from EasyDel.modules.mosaic_mpt import MptConfig as _MptConfig
        return (
            _MptConfig,
            _FlaxMptForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_name="wte")
        )

    elif model_type == "mistral":
        from EasyDel.modules.mistral import FlaxMistralForCausalLM as _FlaxMistralForCausalLM
        from EasyDel.modules.mistral import MistralConfig as _MistralConfig
        from EasyDel.transform import mistral_convert_hf_to_flax as _mistral_convert_hf_to_flax
        return (
            _MistralConfig,
            _FlaxMistralForCausalLM,
            _mistral_convert_hf_to_flax
        )
    elif model_type == "gptj":
        from EasyDel.modules.gpt_j import FlaxGPTJForCausalLM as _FlaxGPTJForCausalLM
        from EasyDel.modules.gpt_j import GPTJConfig as _GPTJConfig
        return (
            _GPTJConfig,
            _FlaxGPTJForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_name="wte")
        )

    elif model_type == "gpt_neox":
        from EasyDel.modules.gpt_neo_x import FlaxGPTNeoXForCausalLM as _FlaxGPTNeoXForCausalLM
        from EasyDel.modules.gpt_neo_x import GPTNeoXConfig as _GPTNeoXConfig

        return (
            _GPTNeoXConfig,
            _FlaxGPTNeoXForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_name="wte")
        )
    elif model_type == "palm":
        from EasyDel.modules.palm import FlaxPalmForCausalLM as _FlaxPalmForCausalLM
        from EasyDel.modules.palm import PalmConfig as _PalmConfig
        return (
            _PalmConfig,
            _FlaxPalmForCausalLM,
            functools.partial(huggingface_to_easydel, embedding_layer_name="wte")
        )
    elif model_type == "lt":
        from EasyDel.modules.lucid_transformer import FlaxLTForCausalLM as _FlaxLTForCausalLM
        from EasyDel.modules.lucid_transformer import FlaxLTConfig as _FlaxLTConfig

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

        if is_flatten(params):
            params = unflatten_dict(params)

        return ed_model, params
