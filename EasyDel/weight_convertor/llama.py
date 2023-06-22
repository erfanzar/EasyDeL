from jax import numpy as jnp
import jax
import torch


def convert_pt_to_flax(state_dict_pt, n_layers: int, device=jax.devices('cpu')[0]):
    with jax.default_device(device):
        state_dict_flax = {}
        state_dict_flax[('model', 'wte', 'embedding')] = state_dict_pt[
            'model.embed_tokens.weight'].cpu().detach().numpy()
        for i in range(n_layers):
            state_dict_flax[('model', 'h', f'{i}', 'attention_norm', 'kernel')] = state_dict_pt[
                f'model.layers.{i}.input_layernorm.weight'].cpu().detach().numpy()
            state_dict_flax[('model', 'h', f'{i}', 'ffn_norm', 'kernel')] = state_dict_pt[
                f'model.layers.{i}.post_attention_layernorm.weight'].cpu().detach().numpy()
            state_dict_flax[('model', 'h', f'{i}', 'feed_forward', 'down_proj', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.mlp.down_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('model', 'h', f'{i}', 'feed_forward', 'gate_proj', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.mlp.gate_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('model', 'h', f'{i}', 'feed_forward', 'up_proj', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.mlp.up_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('model', 'h', f'{i}', 'attention', 'k_proj', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.self_attn.k_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('model', 'h', f'{i}', 'attention', 'v_proj', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.self_attn.v_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('model', 'h', f'{i}', 'attention', 'q_proj', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.self_attn.q_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('model', 'h', f'{i}', 'attention', 'o_proj', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.self_attn.o_proj.weight'].cpu().detach().numpy(), (1, 0))

        state_dict_flax[('model', 'ln_f', 'kernel')] = state_dict_pt[f'model.norm.weight'].cpu().detach().numpy()
        state_dict_flax[('lm_head', 'kernel')] = jnp.transpose(
            state_dict_pt[f'lm_head.weight'].cpu().detach().numpy(),
            (1, 0))
    return state_dict_flax


def convert_flax_to_pt(flax_params, n_layers, dim, num_attention_heads, dtype=jnp.float16):
    def match_keywords(string, ts, ns):
        for t in ts:
            if t not in string:
                return False
        for n in ns:
            if n in string:
                return False
        return True

    torch_params = {}
    for key, tensor in flax_params.items():
        if match_keywords(key, ['kernel'], ['none']):
            tensor = tensor.T
        torch_params[key] = torch.from_numpy(tensor.astype(dtype=dtype))

    def permute(w):
        return w.view(num_attention_heads, dim // num_attention_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)

    state_dict = {}
    inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, dim // num_attention_heads, 2).float() / (dim // num_attention_heads)))
    for layer_i in range(n_layers):
        state_dict.update({
            f"model.layers.{layer_i}.self_attn.q_proj.weight": permute(
                torch_params[f"transformer.h.{layer_i}.attention.wq.kernel"]
            ),
            f"model.layers.{layer_i}.self_attn.k_proj.weight": permute(
                torch_params[f"transformer.h.{layer_i}.attention.wk.kernel"]
            ),
            f"model.layers.{layer_i}.self_attn.v_proj.weight": torch_params[
                f"transformer.h.{layer_i}.attention.wv.kernel"],
            f"model.layers.{layer_i}.self_attn.o_proj.weight": torch_params[
                f"transformer.h.{layer_i}.attention.wo.kernel"],

            f"model.layers.{layer_i}.mlp.gate_proj.weight": torch_params[
                f"transformer.h.{layer_i}.feed_forward.w1.kernel"],
            f"model.layers.{layer_i}.mlp.down_proj.weight": torch_params[
                f"transformer.h.{layer_i}.feed_forward.w2.kernel"],
            f"model.layers.{layer_i}.mlp.up_proj.weight": torch_params[
                f"transformer.h.{layer_i}.feed_forward.w3.kernel"],

            f"model.layers.{layer_i}.input_layernorm.weight": torch_params[
                f"transformer.h.{layer_i}.attention_norm.kernel"],
            f"model.layers.{layer_i}.post_attention_layernorm.weight": torch_params[
                f"transformer.h.{layer_i}.ffn_norm.kernel"],
            f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq": inv_freq

        })

    state_dict.update({
        "model.embed_tokens.weight": torch_params["transformer.wte.embedding"],
        "model.norm.weight": torch_params["transformer.ln_f.kernel"],
        "lm_head.weight": torch_params["lm_head.kernel"],
    })
    return state_dict
