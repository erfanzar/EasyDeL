import jax
from jax import numpy as jnp
import torch


# CONVERTER Falcon-7B
def convert_pt_to_flax_7b(state_dict_pt, n_layers: int, device=jax.devices('cpu')[0], bias=False):
    with jax.default_device(device):
        state_dict_flax = {}
        state_dict_flax[('transformer', 'wte', 'embedding')] = state_dict_pt[
            'transformer.word_embeddings.weight'].cpu().detach().numpy()
        for i in range(n_layers):
            state_dict_flax[('transformer', 'h', f'{i}', 'input_layernorm', 'scale')] = state_dict_pt[
                f'transformer.h.{i}.input_layernorm.weight'].cpu().detach().numpy()
            state_dict_flax[('transformer', 'h', f'{i}', 'input_layernorm', 'bias')] = state_dict_pt[
                f'transformer.h.{i}.input_layernorm.bias'].cpu().detach().numpy()
            state_dict_flax[('transformer', 'h', f'{i}', 'mlp', 'down', 'kernel')] = jnp.transpose(
                state_dict_pt[f'transformer.h.{i}.mlp.dense_4h_to_h.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer', 'h', f'{i}', 'mlp', 'up', 'kernel')] = jnp.transpose(
                state_dict_pt[f'transformer.h.{i}.mlp.dense_h_to_4h.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[
                ('transformer', 'h', f'{i}', 'self_attention', 'w_qkv', 'kernel')] = jnp.transpose(
                state_dict_pt[f'transformer.h.{i}.self_attention.query_key_value.weight'].cpu().detach().numpy(),
                (1, 0))

            state_dict_flax[('transformer', 'h', f'{i}', 'self_attention', 'wo', 'kernel')] = jnp.transpose(
                state_dict_pt[f'transformer.h.{i}.self_attention.dense.weight'].cpu().detach().numpy(), (1, 0))

            state_dict_flax[('transformer', 'h', f'{i}', 'post_attention_layernorm', 'scale')] = state_dict_pt[
                f'transformer.h.{i}.post_attention_layernorm.weight'].cpu().detach().numpy()

            if bias:
                state_dict_flax[
                    ('transformer', 'h', f'{i}', 'self_attention', 'w_qkv', 'bias')] = state_dict_pt[
                    f'transformer.h.{i}.self_attention.query_key_value.bias'].cpu().detach().numpy()
                state_dict_flax[('transformer', 'h', f'{i}', 'self_attention', 'wo', 'bias')] = state_dict_pt[
                    f'transformer.h.{i}.self_attention.dense.bias'].cpu().detach().numpy()
                state_dict_flax[('transformer', 'h', f'{i}', 'mlp', 'down', 'bias')] = state_dict_pt[
                    f'transformer.h.{i}.mlp.dense_4h_to_h.bias'].cpu().detach().numpy()
                state_dict_flax[('transformer', 'h', f'{i}', 'mlp', 'up', 'bias')] = state_dict_pt[
                    f'transformer.h.{i}.mlp.dense_h_to_4h.bias'].cpu().detach().numpy()
                state_dict_flax[('transformer', 'h', f'{i}', 'post_attention_layernorm', 'bias')] = state_dict_pt[
                    f'transformer.h.{i}.post_attention_layernorm.bias'].cpu().detach().numpy()

        state_dict_flax[('transformer', 'ln_f', 'scale')] = state_dict_pt[
            f'transformer.ln_f.weight'].cpu().detach().numpy()
        state_dict_flax[('transformer', 'ln_f', 'bias')] = state_dict_pt[
            f'transformer.ln_f.bias'].cpu().detach().numpy()
        state_dict_flax[('lm_head', 'kernel')] = jnp.transpose(
            state_dict_pt[f'lm_head.weight'].cpu().detach().numpy(), (1, 0))
    return state_dict_flax


def convert_flax_to_pt_7b(state_dict_flax, n_layers: int, device="cpu", bias=False):
    import torch

    state_dict_pt = {}
    state_dict_pt['transformer.word_embeddings.weight'] = torch.from_numpy(
        state_dict_flax[('transformer', 'wte', 'embedding')]).to(device)

    for i in range(n_layers):
        state_dict_pt[f'transformer.h.{i}.input_layernorm.weight'] = torch.from_numpy(
            state_dict_flax[('transformer', 'h', f'{i}', 'input_layernorm', 'scale')]).to(device)
        state_dict_pt[f'transformer.h.{i}.input_layernorm.bias'] = torch.from_numpy(
            state_dict_flax[('transformer', 'h', f'{i}', 'input_layernorm', 'bias')]).to(device)

        state_dict_pt[f'transformer.h.{i}.mlp.dense_4h_to_h.weight'] = torch.from_numpy(
            jnp.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'mlp', 'down', 'kernel')], (1, 0))).to(device)
        state_dict_pt[f'transformer.h.{i}.mlp.dense_h_to_4h.weight'] = torch.from_numpy(
            jnp.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'mlp', 'up', 'kernel')], (1, 0))).to(device)

        state_dict_pt[f'transformer.h.{i}.self_attention.query_key_value.weight'] = torch.from_numpy(
            jnp.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'self_attention', 'w_qkv', 'kernel')],
                          (1, 0))).to(device)
        state_dict_pt[f'transformer.h.{i}.self_attention.dense.weight'] = torch.from_numpy(
            jnp.transpose(state_dict_flax[('transformer', 'h', f'{i}', 'self_attention', 'wo', 'kernel')], (1, 0))).to(
            device)

        if bias:
            state_dict_pt[f'transformer.h.{i}.self_attention.query_key_value.bias'] = torch.from_numpy(
                state_dict_flax[('transformer', 'h', f'{i}', 'self_attention', 'w_qkv', 'bias')]).to(device)
            state_dict_pt[f'transformer.h.{i}.self_attention.dense.bias'] = torch.from_numpy(
                state_dict_flax[('transformer', 'h', f'{i}', 'self_attention', 'wo', 'bias')]).to(device)
            state_dict_pt[f'transformer.h.{i}.mlp.dense_4h_to_h.bias'] = torch.from_numpy(
                state_dict_flax[('transformer', 'h', f'{i}', 'mlp', 'down', 'bias')]).to(device)
            state_dict_pt[f'transformer.h.{i}.mlp.dense_h_to_4h.bias'] = torch.from_numpy(
                state_dict_flax[('transformer', 'h', f'{i}', 'mlp', 'up', 'bias')]).to(device)

    state_dict_pt['transformer.ln_f.weight'] = torch.from_numpy(
        state_dict_flax[('transformer', 'ln_f', 'scale')]).to(device)
    state_dict_pt['transformer.ln_f.bias'] = torch.from_numpy(
        state_dict_flax[('transformer', 'ln_f', 'bias')]).to(device)
    state_dict_pt['lm_head.weight'] = torch.from_numpy(
        jnp.transpose(state_dict_flax[('lm_head', 'kernel')], (1, 0))).to(device)

    return state_dict_pt
