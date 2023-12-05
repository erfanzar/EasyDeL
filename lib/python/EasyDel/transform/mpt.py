import gc

from ..modules.mosaic_mpt import MptConfig
from jax import numpy as jnp
import jax
import torch
import numpy as np
from transformers import AutoModelForCausalLM


def mpt_convert_flax_to_pt_7b(state_dict_flax, n_layers: int, device=torch.device('cpu'), use_lm_head=False):
    # CONVERTER MPT-7B
    state_dict = {'transformer.wte.weight': torch.from_numpy(state_dict_flax[('transformer', 'wte', 'embedding')]).to(
        device)}

    for i in range(n_layers):
        state_dict[f'transformer.blocks.{i}.norm_1.weight'] = torch.from_numpy(state_dict_flax[
                                                                                   ('transformer', 'h', f'{i}',
                                                                                    'norm_1',
                                                                                    'scale')]).to(device)
        state_dict[f'transformer.blocks.{i}.norm_2.weight'] = torch.from_numpy(state_dict_flax[
                                                                                   ('transformer', 'h', f'{i}',
                                                                                    'norm_2',
                                                                                    'scale')]).to(device)
        state_dict[f'transformer.blocks.{i}.ffn.down_proj.weight'] = torch.from_numpy(
            np.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'ffn', 'down', 'kernel')], (1, 0))).to(device)
        state_dict[f'transformer.blocks.{i}.ffn.up_proj.weight'] = torch.from_numpy(
            np.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'ffn', 'up', 'kernel')], (1, 0))).to(device)
        state_dict[f'transformer.blocks.{i}.attn.Wqkv.weight'] = torch.from_numpy(
            np.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'attn', 'w_qkv', 'kernel')], (1, 0))).to(device)
        state_dict[f'transformer.blocks.{i}.attn.out_proj.weight'] = torch.from_numpy(
            np.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'attn', 'wo', 'kernel')], (1, 0))).to(device)

    state_dict['transformer.norm_f.weight'] = torch.from_numpy(state_dict_flax[
                                                                   ('transformer', 'norm_f', 'scale')]).to(device)

    if use_lm_head:
        state_dict['lm_head.weight'] = torch.from_numpy(np.transpose(
            state_dict_flax[('lm_head', 'kernel')], (1, 0))).to(device)

    return state_dict


def mpt_convert_pt_to_flax_7b(state_dict, n_layers: int, device, use_lm_head=False):
    # CONVERTER MPT-7B
    with jax.default_device(device):
        state_dict_flax = {('transformer', 'wte', 'embedding'): state_dict[
            'transformer.wte.weight'].cpu().detach().numpy()}
        for i in range(n_layers):
            state_dict_flax[('transformer', 'h', f'{i}', 'norm_1', 'scale')] = state_dict[
                f'transformer.blocks.{i}.norm_1.weight'].cpu().detach().numpy()
            state_dict_flax[('transformer', 'h', f'{i}', 'norm_2', 'scale')] = state_dict[
                f'transformer.blocks.{i}.norm_2.weight'].cpu().detach().numpy()
            state_dict_flax[('transformer', 'h', f'{i}', 'ffn', 'down', 'kernel')] = jnp.transpose(
                state_dict[f'transformer.blocks.{i}.ffn.down_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer', 'h', f'{i}', 'ffn', 'up', 'kernel')] = jnp.transpose(
                state_dict[f'transformer.blocks.{i}.ffn.up_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer', 'h', f'{i}', 'attn', 'w_qkv', 'kernel')] = jnp.transpose(
                state_dict[f'transformer.blocks.{i}.attn.Wqkv.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer', 'h', f'{i}', 'attn', 'wo', 'kernel')] = jnp.transpose(
                state_dict[f'transformer.blocks.{i}.attn.out_proj.weight'].cpu().detach().numpy(), (1, 0))
        state_dict_flax[('transformer', 'norm_f', 'scale')] = state_dict[
            f'transformer.norm_f.weight'].cpu().detach().numpy()
        if use_lm_head:
            state_dict_flax[('lm_head', 'kernel')] = jnp.transpose(
                state_dict[f'lm_head.weight'].cpu().detach().numpy(), (1, 0))
    return state_dict_flax


def mpt_convert_pt_to_flax_1b(state_dict, n_layers: int, device, use_lm_head=False, ):
    # CONVERTER MPT-1B
    with jax.default_device(device):
        state_dict_flax = {(('transformer', 'wte', 'embedding')): state_dict[
            'transformer.wte.weight'].cpu().detach().numpy()}
        for i in range(n_layers):
            state_dict_flax[('transformer', 'h', f'{i}', 'norm_1', 'scale')] = state_dict[
                f'transformer.blocks.{i}.ln_1.weight'].cpu().detach().numpy()
            state_dict_flax[('transformer', 'h', f'{i}', 'norm_2', 'scale')] = state_dict[
                f'transformer.blocks.{i}.ln_2.weight'].cpu().detach().numpy()
            state_dict_flax[('transformer', 'h', f'{i}', 'ffn', 'down', 'kernel')] = jnp.transpose(
                state_dict[f'transformer.blocks.{i}.mlp.mlp_down.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer', 'h', f'{i}', 'ffn', 'up', 'kernel')] = jnp.transpose(
                state_dict[f'transformer.blocks.{i}.mlp.mlp_up.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer', 'h', f'{i}', 'attn', 'w_qkv', 'kernel')] = jnp.transpose(
                state_dict[f'transformer.blocks.{i}.attn.Wqkv.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer', 'h', f'{i}', 'attn', 'wo', 'kernel')] = jnp.transpose(
                state_dict[f'transformer.blocks.{i}.attn.out_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer', 'h', f'{i}', 'attn', 'q_ln', 'scale')] = state_dict[
                f'transformer.blocks.{i}.attn.q_ln.weight'].cpu().detach().numpy()
            state_dict_flax[('transformer', 'h', f'{i}', 'attn', 'k_ln', 'scale')] = state_dict[
                f'transformer.blocks.{i}.attn.k_ln.weight'].cpu().detach().numpy()
        state_dict_flax[('transformer', 'norm_f', 'scale')] = state_dict[
            f'transformer.ln_f.weight'].cpu().detach().numpy()
        if use_lm_head:
            state_dict_flax[('lm_head', 'kernel')] = jnp.transpose(
                state_dict[f'lm_head.weight'].cpu().detach().numpy(),
                (1, 0))
    return state_dict_flax


def mpt_convert_flax_to_pt_1b(state_dict_flax, n_layers: int, device=torch.device('cpu'), use_lm_head=False):
    # CONVERTER MPT-1B
    state_dict = {'transformer.wte.weight': torch.from_numpy(state_dict_flax[
                                                                 ('transformer', 'wte', 'embedding')]).to(device)}

    for i in range(n_layers):
        state_dict[f'transformer.blocks.{i}.ln_1.weight'] = torch.from_numpy(state_dict_flax[
                                                                                 ('transformer', 'h', f'{i}',
                                                                                  'norm_1',
                                                                                  'scale')]).to(device)
        state_dict[f'transformer.blocks.{i}.ln_2.weight'] = torch.from_numpy(state_dict_flax[
                                                                                 ('transformer', 'h', f'{i}',
                                                                                  'norm_2',
                                                                                  'scale')]).to(device)
        state_dict[f'transformer.blocks.{i}.mlp.mlp_down.weight'] = torch.from_numpy(
            np.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'ffn', 'down', 'kernel')], (1, 0))).to(device)
        state_dict[f'transformer.blocks.{i}.mlp.mlp_up.weight'] = torch.from_numpy(
            np.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'ffn', 'up', 'kernel')], (1, 0))).to(device)
        state_dict[f'transformer.blocks.{i}.attn.Wqkv.weight'] = torch.from_numpy(
            np.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'attn', 'w_qkv', 'kernel')], (1, 0))).to(device)
        state_dict[f'transformer.blocks.{i}.attn.out_proj.weight'] = torch.from_numpy(
            np.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'attn', 'wo', 'kernel')], (1, 0))).to(device)
        state_dict[f'transformer.blocks.{i}.attn.q_ln.weight'] = torch.from_numpy(state_dict_flax[
                                                                                      (
                                                                                          'transformer', 'h', f'{i}',
                                                                                          'attn',
                                                                                          'q_ln', 'scale')]).to(
            device)
        state_dict[f'transformer.blocks.{i}.attn.k_ln.weight'] = torch.from_numpy(state_dict_flax[
                                                                                      (
                                                                                          'transformer', 'h', f'{i}',
                                                                                          'attn',
                                                                                          'k_ln', 'scale')]).to(
            device)

    state_dict['transformer.ln_f.weight'] = torch.from_numpy(state_dict_flax[
                                                                 ('transformer', 'norm_f', 'scale')]).to(device)

    if use_lm_head:
        state_dict['lm_head.weight'] = torch.from_numpy(np.transpose(
            state_dict_flax[('lm_head', 'kernel')], (1, 0))).to(device)

    return state_dict


def mpt_from_pretrained(model_id, device, **kwargs):
    """
    return: Weight or Params for EasyDel Model , Config
    """
    config = MptConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, **kwargs)

    easydel_wights = mpt_convert_pt_to_flax_7b(
        state_dict=model.state_dict(),
        n_layers=config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else config.n_layers,
        device=device
    )
    config.add_jax_args()

    del model
    gc.collect()
    return easydel_wights, config
