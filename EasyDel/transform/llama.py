from pathlib import Path

from fjutils import StreamingCheckpointer
from jax import numpy as jnp
import jax
import torch
from transformers import LlamaForCausalLM
from EasyDel.modules.llama import LlamaConfig


def llama_convert_hf_to_flax_load(checkpoints_dir, num_hidden_layers=32, num_attention_heads=32, hidden_size=4096,
                            device=jax.devices('cpu')[0]):
    # Edited From EasyLM
    ckpt_paths = sorted(Path(checkpoints_dir).glob("*.bin"))
    state_dict = {}
    with jax.default_device(device):
        for i, ckpt_path in enumerate(ckpt_paths):
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            for k, v in checkpoint.items():
                if k.startswith("model."):
                    k = k[6:]
                state_dict[k] = v

        def inverse_permute(w):

            reshaped_w = w.reshape(num_attention_heads, 2, hidden_size // num_attention_heads // 2, hidden_size)
            transposed_w = reshaped_w.transpose(0, 2, 1, 3)
            inverted_w = transposed_w.reshape(hidden_size, hidden_size)
            return inverted_w

        jax_weights = {
            "transformer": {
                "wte": {"embedding": state_dict["embed_tokens.weight"].numpy()},
                "ln_f": {"kernel": state_dict["norm.weight"].numpy()},
                "h": {
                    "%d"
                    % (layer): {
                        "attention": {
                            "wq": {
                                "kernel": inverse_permute(

                                    state_dict[f"layers.{layer}.self_attn.q_proj.weight"].numpy(),
                                ).transpose()
                            },
                            "wk": {
                                "kernel": inverse_permute(

                                    state_dict[f"layers.{layer}.self_attn.k_proj.weight"].numpy(),
                                ).transpose()
                            },
                            "wv": {
                                "kernel": state_dict[f"layers.{layer}.self_attn.v_proj.weight"]
                                .numpy()
                                .transpose()
                            },
                            "wo": {
                                "kernel": state_dict[f"layers.{layer}.self_attn.o_proj.weight"]
                                .numpy()
                                .transpose()
                            },
                        },
                        "feed_forward": {
                            "w1": {
                                "kernel": state_dict[f"layers.{layer}.mlp.gate_proj.weight"]
                                .numpy()
                                .transpose()
                            },
                            "w2": {
                                "kernel": state_dict[f"layers.{layer}.mlp.down_proj.weight"]
                                .numpy()
                                .transpose()
                            },
                            "w3": {
                                "kernel": state_dict[f"layers.{layer}.mlp.up_proj.weight"]
                                .numpy()
                                .transpose()
                            },
                        },
                        "attention_norm": {
                            "kernel": state_dict[f"layers.{layer}.input_layernorm.weight"].numpy()
                        },
                        "ffn_norm": {
                            "kernel": state_dict[
                                f"layers.{layer}.post_attention_layernorm.weight"
                            ].numpy()
                        },
                    }
                    for layer in range(num_hidden_layers)
                },
            },
            "lm_head": {"kernel": state_dict["lm_head.weight"].numpy().transpose()},
        }

        return jax_weights


def llama_convert_hf_to_flax(state_dict, num_hidden_layers=32, num_attention_heads=32, hidden_size=4096,
                       device=jax.devices('cpu')[0]):
    with jax.default_device(device):
        def inverse_permute(w):
            reshaped_w = w.reshape(num_attention_heads, 2, hidden_size // num_attention_heads // 2, hidden_size)
            transposed_w = reshaped_w.transpose(0, 2, 1, 3)
            inverted_w = transposed_w.reshape(hidden_size, hidden_size)
            return inverted_w

        jax_weights = {
            "transformer": {
                "wte": {"embedding": state_dict["model.embed_tokens.weight"].numpy()},
                "ln_f": {"kernel": state_dict["model.norm.weight"].numpy()},
                "h": {
                    "%d"
                    % (layer): {
                        "attention": {
                            "wq": {
                                "kernel": inverse_permute(

                                    state_dict[f"model.layers.{layer}.self_attn.q_proj.weight"].numpy(),
                                ).transpose()
                            },
                            "wk": {
                                "kernel": inverse_permute(

                                    state_dict[f"model.layers.{layer}.self_attn.k_proj.weight"].numpy(),
                                ).transpose()
                            },
                            "wv": {
                                "kernel": state_dict[f"model.layers.{layer}.self_attn.v_proj.weight"]
                                .numpy()
                                .transpose()
                            },
                            "wo": {
                                "kernel": state_dict[f"model.layers.{layer}.self_attn.o_proj.weight"]
                                .numpy()
                                .transpose()
                            },
                        },
                        "feed_forward": {
                            "w1": {
                                "kernel": state_dict[f"model.layers.{layer}.mlp.gate_proj.weight"]
                                .numpy()
                                .transpose()
                            },
                            "w2": {
                                "kernel": state_dict[f"model.layers.{layer}.mlp.down_proj.weight"]
                                .numpy()
                                .transpose()
                            },
                            "w3": {
                                "kernel": state_dict[f"model.layers.{layer}.mlp.up_proj.weight"]
                                .numpy()
                                .transpose()
                            },
                        },
                        "attention_norm": {
                            "kernel": state_dict[f"model.layers.{layer}.input_layernorm.weight"].numpy()
                        },
                        "ffn_norm": {
                            "kernel": state_dict[
                                f"model.layers.{layer}.post_attention_layernorm.weight"
                            ].numpy()
                        },
                    }
                    for layer in range(num_hidden_layers)
                },
            },
            "lm_head": {"kernel": state_dict["lm_head.weight"].numpy().transpose()},
        }

        return jax_weights


def llama_convert_pt_to_flax(state_dict_pt, n_layers: int, device=jax.devices('cpu')[0]):
    with jax.default_device(device):
        state_dict_flax = {}
        state_dict_flax[('transformer', 'wte', 'embedding')] = state_dict_pt[
            'model.embed_tokens.weight'].cpu().detach().numpy()
        for i in range(n_layers):
            state_dict_flax[('transformer', 'h', f'{i}', 'attention_norm', 'kernel')] = state_dict_pt[
                f'model.layers.{i}.input_layernorm.weight'].cpu().detach().numpy()
            state_dict_flax[('transformer', 'h', f'{i}', 'ffn_norm', 'kernel')] = state_dict_pt[
                f'model.layers.{i}.post_attention_layernorm.weight'].cpu().detach().numpy()
            state_dict_flax[('transformer', 'h', f'{i}', 'feed_forward', 'w2', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.mlp.down_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer', 'h', f'{i}', 'feed_forward', 'w1', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.mlp.gate_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer', 'h', f'{i}', 'feed_forward', 'w3', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.mlp.up_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer', 'h', f'{i}', 'attention', 'wk', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.self_attn.k_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer', 'h', f'{i}', 'attention', 'wv', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.self_attn.v_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer', 'h', f'{i}', 'attention', 'wq', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.self_attn.q_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('transformer', 'h', f'{i}', 'attention', 'wo', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.self_attn.o_proj.weight'].cpu().detach().numpy(), (1, 0))

        state_dict_flax[('transformer', 'ln_f', 'kernel')] = state_dict_pt[f'model.norm.weight'].cpu().detach().numpy()
        state_dict_flax[('lm_head', 'kernel')] = jnp.transpose(
            state_dict_pt[f'lm_head.weight'].cpu().detach().numpy(),
            (1, 0))
    return state_dict_flax


def llama_convert_flax_to_pt(flax_params, n_layers, dim, num_attention_heads, dtype=jnp.float16):
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


def llama_from_pretrained(model_id, device=jax.devices('cpu')[0]):
    """
    return: Weight or Params for EasyDel Model , Config
    """
    config = LlamaConfig.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id)
    easydel_wights = llama_convert_hf_to_flax(
        state_dict=model.state_dict(),
        num_hidden_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        device=device
    )
    config.add_jax_args()
    return easydel_wights, config
