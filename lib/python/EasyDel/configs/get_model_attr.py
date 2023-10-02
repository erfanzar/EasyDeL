import logging

from EasyDel import FlaxLlamaForCausalLM, FlaxMptForCausalLM, FlaxGPTJForCausalLM, FlaxGPTNeoXForCausalLM, \
    FlaxT5ForConditionalGeneration, FlaxOPTForCausalLM, FlaxFalconForCausalLM

from EasyDel.configs.configs import llama_2_configs, gptj_configs, mpt_configs, opt_configs, falcon_configs, \
    llama_configs

logger = logging.getLogger()
AVAILABLE_MODELS = [
    'llama', 'llama_2', 'falcon', 'opt', 'gpt_j', 'gpt_neox', 'mpt', 't5'
]

AVAILABLE_CONFIGS = [
    'llama', 'llama_2', 'falcon', 'opt', 'gpt_j', 'mpt', 't5'
]


def get_model_and_config(model, model_type):
    assert model in AVAILABLE_MODELS, f'{model} is not supported, available models are {AVAILABLE_MODELS}'
    if model not in AVAILABLE_CONFIGS:
        logger.warning(f'config for {model} is not available you will receive config as None')
    try:
        if model == 'llama' or model == 'llama_2':
            model_class = FlaxLlamaForCausalLM
            if model == 'llama':
                config_dict = llama_configs[model_type]
            elif model == 'llama_2':
                config_dict = llama_2_configs[model_type]
            else:
                raise ValueError('report BUG! :)')
        elif model == 'falcon':
            model_class = FlaxFalconForCausalLM
            config_dict = falcon_configs[model_type]
        elif model == 'opt':
            model_class = FlaxOPTForCausalLM
            config_dict = opt_configs[model_type]
        elif model == 'gpt_j':
            model_class = FlaxGPTJForCausalLM
            config_dict = gptj_configs[model_type]
        elif model == 't5':
            model_class = FlaxT5ForConditionalGeneration
            config_dict = None
        elif model == 'mpt':
            model_class = FlaxMptForCausalLM
            config_dict = mpt_configs[model_type]
        elif model == 'gpt_neox':
            model_class = FlaxGPTNeoXForCausalLM
            config_dict = None
        else:
            raise ValueError('bug?')

    except KeyError:
        raise ValueError(f'Cannot find config for {model}-{model_type}')
    return model_class, config_dict
