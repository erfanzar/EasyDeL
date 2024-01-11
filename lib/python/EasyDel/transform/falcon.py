import jax
from fjformer import load_and_convert_checkpoint_to_torch
from jax import numpy as jnp
from tqdm import tqdm
from transformers import FalconForCausalLM
from ..modules.falcon import FalconConfig
import torch
from typing import Dict
import gc


def match_keywords(string, ts, ns):
    for t in ts:
        if t not in string:
            return False
    for n in ns:
        if n in string:
            return False
    return True


def falcon_from_pretrained(model_id, device):
    """
    return: Weight or Params for EasyDel Model , Config
    """
    # Requested By vwxyzjn at https://github.com/erfanzar/EasyDeL/issues/15#issue-1881044170
    config = FalconConfig.from_pretrained(model_id)
    model = FalconForCausalLM.from_pretrained(model_id)
    easydel_wights = falcon_convert_pt_to_flax_7b(
        state_dict=model.state_dict(),
        config=config,
        device=device
    )
    del model
    gc.collect()
    config.add_jax_args()
    return easydel_wights, config


def falcon_convert_pt_to_flax_7b(
        state_dict,
        config: FalconConfig,
        device,
        is_pb: bool = False
):
    with jax.default_device(device):
        state_dict_flax = {('transformer', 'wte', 'embedding'): state_dict[
            'transformer.word_embeddings.weight'].cpu().detach().numpy()}
        pbar = tqdm(iterable=range(config.num_hidden_layers))
        for i in pbar:
            pbar.set_description('Converting Layers')
            state_dict_flax[('transformer', 'h', f'{i}', 'input_layernorm', 'scale')] = state_dict[
                f'transformer.h.{i}.input_layernorm.weight'].cpu().detach().numpy()
            state_dict_flax[('transformer', 'h', f'{i}', 'input_layernorm', 'bias')] = state_dict[
                f'transformer.h.{i}.input_layernorm.bias'].cpu().detach().numpy()
            state_dict_flax[('transformer', 'h', f'{i}', 'mlp', 'down', 'kernel')] = jnp.transpose(
                state_dict[f'transformer.h.{i}.mlp.dense_4h_to_h.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer', 'h', f'{i}', 'mlp', 'up', 'kernel')] = jnp.transpose(
                state_dict[f'transformer.h.{i}.mlp.dense_h_to_4h.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[
                ('transformer', 'h', f'{i}', 'self_attention', 'w_qkv', 'kernel')] = jnp.transpose(
                state_dict[f'transformer.h.{i}.self_attention.query_key_value.weight'].cpu().detach().numpy(),
                (1, 0))

            state_dict_flax[('transformer', 'h', f'{i}', 'self_attention', 'wo', 'kernel')] = jnp.transpose(
                state_dict[f'transformer.h.{i}.self_attention.dense.weight'].cpu().detach().numpy(), (1, 0))
            try:
                state_dict_flax[('transformer', 'h', f'{i}', 'post_attention_layernorm', 'scale')] = state_dict[
                    f'transformer.h.{i}.post_attention_layernorm.weight'].cpu().detach().numpy()
            except KeyError:
                if is_pb:
                    raise KeyError(
                        'tried to access some of model weight but they were unavailable please open a bug or '
                        'check model config'
                    )
            if config.bias:
                state_dict_flax[
                    ('transformer', 'h', f'{i}', 'self_attention', 'w_qkv', 'bias')] = state_dict[
                    f'transformer.h.{i}.self_attention.query_key_value.bias'].cpu().detach().numpy()
                state_dict_flax[('transformer', 'h', f'{i}', 'self_attention', 'wo', 'bias')] = state_dict[
                    f'transformer.h.{i}.self_attention.dense.bias'].cpu().detach().numpy()
                state_dict_flax[('transformer', 'h', f'{i}', 'mlp', 'down', 'bias')] = state_dict[
                    f'transformer.h.{i}.mlp.dense_4h_to_h.bias'].cpu().detach().numpy()
                state_dict_flax[('transformer', 'h', f'{i}', 'mlp', 'up', 'bias')] = state_dict[
                    f'transformer.h.{i}.mlp.dense_h_to_4h.bias'].cpu().detach().numpy()
                try:
                    state_dict_flax[('transformer', 'h', f'{i}', 'post_attention_layernorm', 'bias')] = state_dict[
                        f'transformer.h.{i}.post_attention_layernorm.bias'].cpu().detach().numpy()
                except KeyError:
                    if is_pb:
                        raise KeyError(
                            'tried to access some of model weight but they were unavailable please open a bug or '
                            'check model config'
                        )
        state_dict_flax[('transformer', 'ln_f', 'scale')] = state_dict[
            f'transformer.ln_f.weight'].cpu().detach().numpy()
        state_dict_flax[('transformer', 'ln_f', 'bias')] = state_dict[
            f'transformer.ln_f.bias'].cpu().detach().numpy()
        state_dict_flax[('lm_head', 'kernel')] = jnp.transpose(
            state_dict[f'lm_head.weight'].cpu().detach().numpy(), (1, 0))
    return state_dict_flax


def falcon_convert_flax_to_pt_7b(state_dict_flax, num_hidden_layers: int, device="cpu", bias=False):
    import torch

    state_dict = {'transformer.word_embeddings.weight': torch.from_numpy(
        state_dict_flax[('transformer', 'wte', 'embedding')]).to(device)}

    pbar = tqdm(iterable=range(num_hidden_layers))
    for i in pbar:
        pbar.set_description('Converting Layers')
        state_dict[f'transformer.h.{i}.input_layernorm.weight'] = torch.from_numpy(
            state_dict_flax[('transformer', 'h', f'{i}', 'input_layernorm', 'scale')]).to(device)
        state_dict[f'transformer.h.{i}.input_layernorm.bias'] = torch.from_numpy(
            state_dict_flax[('transformer', 'h', f'{i}', 'input_layernorm', 'bias')]).to(device)

        state_dict[f'transformer.h.{i}.mlp.dense_4h_to_h.weight'] = torch.from_numpy(
            jnp.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'mlp', 'down', 'kernel')], (1, 0))).to(device)
        state_dict[f'transformer.h.{i}.mlp.dense_h_to_4h.weight'] = torch.from_numpy(
            jnp.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'mlp', 'up', 'kernel')], (1, 0))).to(device)

        state_dict[f'transformer.h.{i}.self_attention.query_key_value.weight'] = torch.from_numpy(
            jnp.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'self_attention', 'w_qkv', 'kernel')],
                          (1, 0))).to(device)
        state_dict[f'transformer.h.{i}.self_attention.dense.weight'] = torch.from_numpy(
            jnp.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'self_attention', 'wo', 'kernel')], (1, 0))).to(
            device)

        if bias:
            state_dict[f'transformer.h.{i}.self_attention.query_key_value.bias'] = torch.from_numpy(
                state_dict_flax[('transformer', 'h', f'{i}', 'self_attention', 'w_qkv', 'bias')]).to(device)
            state_dict[f'transformer.h.{i}.self_attention.dense.bias'] = torch.from_numpy(
                state_dict_flax[('transformer', 'h', f'{i}', 'self_attention', 'wo', 'bias')]).to(device)
            state_dict[f'transformer.h.{i}.mlp.dense_4h_to_h.bias'] = torch.from_numpy(
                state_dict_flax[('transformer', 'h', f'{i}', 'mlp', 'down', 'bias')]).to(device)
            state_dict[f'transformer.h.{i}.mlp.dense_h_to_4h.bias'] = torch.from_numpy(
                state_dict_flax[('transformer', 'h', f'{i}', 'mlp', 'up', 'bias')]).to(device)

    state_dict['transformer.ln_f.weight'] = torch.from_numpy(
        state_dict_flax[('transformer', 'ln_f', 'scale')]).to(device)
    state_dict['transformer.ln_f.bias'] = torch.from_numpy(
        state_dict_flax[('transformer', 'ln_f', 'bias')]).to(device)
    state_dict['lm_head.weight'] = torch.from_numpy(
        jnp.transpose(state_dict_flax[('lm_head', 'kernel')], (1, 0))).to(device)

    return state_dict




def falcon_convert_hf_to_flax(state_dict: Dict[str, torch.Tensor], config: FalconConfig, device):
    lw = len('.weight')
    with jax.default_device(device):
        flax_dict = {}
        for key, tensor in state_dict.items():
            if match_keywords(key, ['kernel'], ['none']):
                if len(tensor.shape) == 2:
                    tensor = tensor.transpose(0, 1)
            if key.endswith('.weight'):
                key = key[:-lw] + '.kernel'
            key_tuple = key.split('.')
            key_names = ()
            tensor = tensor.detach().cpu().numpy()
            for k in key_tuple:
                key_names += k,
            flax_dict[key_names] = tensor
    return flax_dict
