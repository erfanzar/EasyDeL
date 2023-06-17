from jax import numpy as jnp
import jax
import torch
import numpy as np


def convert_flax_to_pt_7b(state_dict_flax, n_layers: int, device=torch.device('cpu'), use_lm_head=False):
    # CONVERTER MPT-7B
    state_dict_pt = {}
    state_dict_pt['transformer.wte.weight'] = torch.from_numpy(state_dict_flax[('transformer', 'wte', 'embedding')]).to(
        device)

    for i in range(n_layers):
        state_dict_pt[f'transformer.blocks.{i}.norm_1.weight'] = torch.from_numpy(state_dict_flax[
                                                                                  ('transformer', 'h', f'{i}', 'norm_1',
                                                                                   'scale')]).to(device)
        state_dict_pt[f'transformer.blocks.{i}.norm_2.weight'] = torch.from_numpy(state_dict_flax[
                                                                                  ('transformer', 'h', f'{i}', 'norm_2',
                                                                                   'scale')]).to(device)
        state_dict_pt[f'transformer.blocks.{i}.ffn.down_proj.weight'] = torch.from_numpy(
            np.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'ffn', 'down', 'kernel')], (1, 0))).to(device)
        state_dict_pt[f'transformer.blocks.{i}.ffn.up_proj.weight'] = torch.from_numpy(
            np.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'ffn', 'up', 'kernel')], (1, 0))).to(device)
        state_dict_pt[f'transformer.blocks.{i}.attn.Wqkv.weight'] = torch.from_numpy(
            np.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'attn', 'w_qkv', 'kernel')], (1, 0))).to(device)
        state_dict_pt[f'transformer.blocks.{i}.attn.out_proj.weight'] = torch.from_numpy(
            np.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'attn', 'wo', 'kernel')], (1, 0))).to(device)

    state_dict_pt['transformer.norm_f.weight'] = torch.from_numpy(state_dict_flax[
                                                                  ('transformer', 'norm_f', 'scale')]).to(device)

    if use_lm_head:
        state_dict_pt['lm_head.weight'] = torch.from_numpy(np.transpose(
            state_dict_flax[('lm_head', 'kernel')], (1, 0))).to(device)

    return state_dict_pt


def convert_pt_to_flax_7b(state_dict_pt, n_layers: int, device=jax.devices('cpu')[0], use_lm_head=False):
    # CONVERTER MPT-7B
    with jax.default_device(device):
        state_dict_flax = {}
        state_dict_flax[('transformer'), ('wte'), ('embedding')] = state_dict_pt[
            'transformer.wte.weight'].cpu().detach().numpy()
        for i in range(n_layers):
            state_dict_flax[('transformer'), ('h'), (f'{i}'), ('norm_1'), ('scale')] = state_dict_pt[
                f'transformer.blocks.{i}.norm_1.weight'].cpu().detach().numpy()
            state_dict_flax[('transformer'), ('h'), (f'{i}'), ('norm_2'), ('scale')] = state_dict_pt[
                f'transformer.blocks.{i}.norm_2.weight'].cpu().detach().numpy()
            state_dict_flax[('transformer'), ('h'), (f'{i}'), ('ffn'), ('down'), ('kernel')] = jnp.transpose(
                state_dict_pt[f'transformer.blocks.{i}.ffn.down_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer'), ('h'), (f'{i}'), ('ffn'), ('up'), ('kernel')] = jnp.transpose(
                state_dict_pt[f'transformer.blocks.{i}.ffn.up_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer'), ('h'), (f'{i}'), ('attn'), ('w_qkv'), ('kernel')] = jnp.transpose(
                state_dict_pt[f'transformer.blocks.{i}.attn.Wqkv.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer'), ('h'), (f'{i}'), ('attn'), ('wo'), ('kernel')] = jnp.transpose(
                state_dict_pt[f'transformer.blocks.{i}.attn.out_proj.weight'].cpu().detach().numpy(), (1, 0))
        state_dict_flax[('transformer'), ('norm_f'), ('scale')] = state_dict_pt[
            f'transformer.norm_f.weight'].cpu().detach().numpy()
        if use_lm_head:
            state_dict_flax[('lm_head'), ('kernel')] = jnp.transpose(
                state_dict_pt[f'lm_head.weight'].cpu().detach().numpy(), (1, 0))
    return state_dict_flax


def convert_pt_to_flax_1b(state_dict_pt, n_layers: int, device=jax.devices('cpu')[0], use_lm_head=False, ):
    # CONVERTER MPT-1B
    with jax.default_device(device):
        state_dict_flax = {}
        state_dict_flax[('transformer'), ('wte'), ('embedding')] = state_dict_pt[
            'transformer.wte.weight'].cpu().detach().numpy()
        for i in range(n_layers):
            state_dict_flax[('transformer'), ('h'), (f'{i}'), ('norm_1'), ('scale')] = state_dict_pt[
                f'transformer.blocks.{i}.ln_1.weight'].cpu().detach().numpy()
            state_dict_flax[('transformer'), ('h'), (f'{i}'), ('norm_2'), ('scale')] = state_dict_pt[
                f'transformer.blocks.{i}.ln_2.weight'].cpu().detach().numpy()
            state_dict_flax[('transformer'), ('h'), (f'{i}'), ('ffn'), ('down'), ('kernel')] = jnp.transpose(
                state_dict_pt[f'transformer.blocks.{i}.mlp.mlp_down.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer'), ('h'), (f'{i}'), ('ffn'), ('up'), ('kernel')] = jnp.transpose(
                state_dict_pt[f'transformer.blocks.{i}.mlp.mlp_up.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer'), ('h'), (f'{i}'), ('attn'), ('w_qkv'), ('kernel')] = jnp.transpose(
                state_dict_pt[f'transformer.blocks.{i}.attn.Wqkv.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer'), ('h'), (f'{i}'), ('attn'), ('wo'), ('kernel')] = jnp.transpose(
                state_dict_pt[f'transformer.blocks.{i}.attn.out_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer'), ('h'), (f'{i}'), ('attn'), ('q_ln'), ('scale')] = state_dict_pt[
                f'transformer.blocks.{i}.attn.q_ln.weight'].cpu().detach().numpy()
            state_dict_flax[('transformer'), ('h'), (f'{i}'), ('attn'), ('k_ln'), ('scale')] = state_dict_pt[
                f'transformer.blocks.{i}.attn.k_ln.weight'].cpu().detach().numpy()
        state_dict_flax[('transformer'), ('norm_f'), ('scale')] = state_dict_pt[
            f'transformer.ln_f.weight'].cpu().detach().numpy()
        if use_lm_head:
            state_dict_flax[('lm_head'), ('kernel')] = jnp.transpose(
                state_dict_pt[f'lm_head.weight'].cpu().detach().numpy(),
                (1, 0))
    return state_dict_flax


def convert_flax_to_pt_1b(state_dict_flax, n_layers: int, device=torch.device('cpu'), use_lm_head=False):
    # CONVERTER MPT-1B
    state_dict_pt = {}
    state_dict_pt['transformer.wte.weight'] = torch.from_numpy(state_dict_flax[
                                                               ('transformer', 'wte', 'embedding')]).to(device)

    for i in range(n_layers):
        state_dict_pt[f'transformer.blocks.{i}.ln_1.weight'] = torch.from_numpy(state_dict_flax[
                                                                                ('transformer', 'h', f'{i}', 'norm_1',
                                                                                 'scale')]).to(device)
        state_dict_pt[f'transformer.blocks.{i}.ln_2.weight'] = torch.from_numpy(state_dict_flax[
                                                                                ('transformer', 'h', f'{i}', 'norm_2',
                                                                                 'scale')]).to(device)
        state_dict_pt[f'transformer.blocks.{i}.mlp.mlp_down.weight'] = torch.from_numpy(
            np.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'ffn', 'down', 'kernel')], (1, 0))).to(device)
        state_dict_pt[f'transformer.blocks.{i}.mlp.mlp_up.weight'] = torch.from_numpy(
            np.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'ffn', 'up', 'kernel')], (1, 0))).to(device)
        state_dict_pt[f'transformer.blocks.{i}.attn.Wqkv.weight'] = torch.from_numpy(
            np.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'attn', 'w_qkv', 'kernel')], (1, 0))).to(device)
        state_dict_pt[f'transformer.blocks.{i}.attn.out_proj.weight'] = torch.from_numpy(
            np.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'attn', 'wo', 'kernel')], (1, 0))).to(device)
        state_dict_pt[f'transformer.blocks.{i}.attn.q_ln.weight'] = torch.from_numpy(state_dict_flax[
                                                                                     (
                                                                                         'transformer', 'h', f'{i}',
                                                                                         'attn',
                                                                                         'q_ln', 'scale')]).to(device)
        state_dict_pt[f'transformer.blocks.{i}.attn.k_ln.weight'] = torch.from_numpy(state_dict_flax[
                                                                                     (
                                                                                         'transformer', 'h', f'{i}',
                                                                                         'attn',
                                                                                         'k_ln', 'scale')]).to(device)

    state_dict_pt['transformer.ln_f.weight'] = torch.from_numpy(state_dict_flax[
                                                                ('transformer', 'norm_f', 'scale')]).to(device)

    if use_lm_head:
        state_dict_pt['lm_head.weight'] = torch.from_numpy(np.transpose(
            state_dict_flax[('lm_head', 'kernel')], (1, 0))).to(device)

    return state_dict_pt
