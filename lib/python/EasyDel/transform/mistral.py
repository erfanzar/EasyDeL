import gc
from pathlib import Path

from fjformer import load_and_convert_checkpoint_to_torch
from jax import numpy as jnp
import jax
import torch
from transformers import MistralForCausalLM
from ..modules.mistral import MistralConfig


def inverse_permute(tensor, head, dim_in, dim_out):
    return tensor.reshape(head, 2, dim_in // head // 2, dim_out).transpose(0, 2, 1, 3).reshape(
        dim_out, dim_in)


def permute(tensor, head, dim_in, dim_out):
    return tensor.view(head, dim_out // head // 2, 2, dim_in).transpose(1, 2).reshape(dim_out, dim_in)


def match_keywords(string, ts, ns):
    for t in ts:
        if t not in string:
            return False
    for n in ns:
        if n in string:
            return False
    return True


def mistral_convert_hf_to_flax_load(checkpoints_dir, config: MistralConfig,
                                    device):
    kv_dim = config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
    ckpt_paths = sorted(Path(checkpoints_dir).glob("*.bin"))
    state_dict = {}
    with jax.default_device(device):
        for i, checkpoint_path in enumerate(ckpt_paths):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            for k, v in checkpoint.items():
                if k.startswith("model."):
                    k = k[6:]
                state_dict[k] = v

        jax_weights = {
            "model": {
                "embed_tokens": {"embedding": state_dict["embed_tokens.weight"].cpu().numpy()},
                "norm": {"kernel": state_dict["norm.weight"].cpu().numpy()},
                "layers": {
                    "%d"
                    % (layer): {
                        "self_attn": {
                            "q_proj": {
                                "kernel": inverse_permute(

                                    state_dict[f"layers.{layer}.self_attn.q_proj.weight"].cpu().numpy(),
                                    config.num_attention_heads,
                                    config.hidden_size, config.hidden_size
                                ).transpose()
                            },
                            "k_proj": {
                                "kernel": inverse_permute(
                                    state_dict[f"layers.{layer}.self_attn.k_proj.weight"].cpu().numpy(),
                                    config.num_key_value_heads,
                                    config.hidden_size, kv_dim
                                ).transpose()
                            },
                            "v_proj": {
                                "kernel": state_dict[f"layers.{layer}.self_attn.v_proj.weight"]
                                .cpu().numpy()
                                .transpose()
                            },
                            "o_proj": {
                                "kernel": state_dict[f"layers.{layer}.self_attn.o_proj.weight"]
                                .cpu().numpy()
                                .transpose()
                            },
                        },
                        "mlp": {
                            "gate_proj": {
                                "kernel": state_dict[f"layers.{layer}.mlp.gate_proj.weight"]
                                .cpu().numpy()
                                .transpose()
                            },
                            "down_proj": {
                                "kernel": state_dict[f"layers.{layer}.mlp.down_proj.weight"]
                                .cpu().numpy()
                                .transpose()
                            },
                            "up_proj": {
                                "kernel": state_dict[f"layers.{layer}.mlp.up_proj.weight"]
                                .cpu().numpy()
                                .transpose()
                            },
                        },
                        "input_layernorm": {
                            "kernel": state_dict[f"layers.{layer}.input_layernorm.weight"].cpu().numpy()
                        },
                        "post_attention_layernorm": {
                            "kernel": state_dict[
                                f"layers.{layer}.post_attention_layernorm.weight"
                            ].cpu().numpy()
                        },
                    }
                    for layer in range(config.num_hidden_layers)
                },
            },
            "lm_head": {"kernel": state_dict["lm_head.weight"].cpu().numpy().transpose()},
        }

        return jax_weights


def mistral_convert_hf_to_flax(state_dict, config: MistralConfig,
                               device):
    with jax.default_device(device):
        jax_weights = {
            "model": {
                "embed_tokens": {"embedding": state_dict["model.embed_tokens.weight"].cpu().numpy()},
                "norm": {"kernel": state_dict["model.norm.weight"].cpu().numpy()},
                "layers": {
                    f"{layer}": {
                        "self_attn": {
                            "q_proj": {
                                "kernel": state_dict[
                                    f"model.layers.{layer}.self_attn.q_proj.weight"].cpu().numpy().transpose()
                            },
                            "k_proj": {
                                "kernel": state_dict[
                                    f"model.layers.{layer}.self_attn.k_proj.weight"].cpu().numpy().transpose()
                            },
                            "v_proj": {
                                "kernel": state_dict[
                                    f"model.layers.{layer}.self_attn.v_proj.weight"].cpu().numpy().transpose()
                            },
                            "o_proj": {
                                "kernel": state_dict[
                                    f"model.layers.{layer}.self_attn.o_proj.weight"].cpu().numpy().transpose()
                            },
                        },
                        "mlp": {
                            "gate_proj": {
                                "kernel": state_dict[f"model.layers.{layer}.mlp.gate_proj.weight"]
                                .cpu().numpy()
                                .transpose()
                            },
                            "down_proj": {
                                "kernel": state_dict[f"model.layers.{layer}.mlp.down_proj.weight"]
                                .cpu().numpy()
                                .transpose()
                            },
                            "up_proj": {
                                "kernel": state_dict[f"model.layers.{layer}.mlp.up_proj.weight"]
                                .cpu().numpy()
                                .transpose()
                            },
                        },
                        "input_layernorm": {
                            "kernel": state_dict[f"model.layers.{layer}.input_layernorm.weight"].cpu().numpy()
                        },
                        "post_attention_layernorm": {
                            "kernel": state_dict[
                                f"model.layers.{layer}.post_attention_layernorm.weight"
                            ].cpu().numpy()
                        },
                    }
                    for layer in range(config.num_hidden_layers)
                },
            },
            "lm_head": {"kernel": state_dict["lm_head.weight"].cpu().numpy().transpose()},
        }

        return jax_weights


def mistral_convert_pt_to_flax(state_dict_pt, config: MistralConfig, device):
    with jax.default_device(device):
        state_dict_flax = {('model', 'embed_tokens', 'embedding'): state_dict_pt[
            'model.embed_tokens.weight'].cpu().detach().numpy()}
        for i in range(config.num_hidden_layers):
            state_dict_flax[('model', 'layers', f'{i}', 'input_layernorm', 'kernel')] = state_dict_pt[
                f'model.layers.{i}.input_layernorm.weight'].cpu().detach().numpy()
            state_dict_flax[('model', 'layers', f'{i}', 'post_attention_layernorm', 'kernel')] = state_dict_pt[
                f'model.layers.{i}.post_attention_layernorm.weight'].cpu().detach().numpy()
            state_dict_flax[('model', 'layers', f'{i}', 'mlp', 'down_proj', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.mlp.down_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('model', 'layers', f'{i}', 'mlp', 'gate_proj', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.mlp.gate_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('model', 'layers', f'{i}', 'mlp', 'up_proj', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.mlp.up_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('model', 'layers', f'{i}', 'self_attn', 'k_proj', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.self_attn.k_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('model', 'layers', f'{i}', 'self_attn', 'v_proj', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.self_attn.v_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('model', 'layers', f'{i}', 'self_attn', 'q_proj', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.self_attn.q_proj.weight'].cpu().detach().numpy(), (1, 0))
            state_dict_flax[('model', 'layers', f'{i}', 'self_attn', 'o_proj', 'kernel')] = jnp.transpose(
                state_dict_pt[f'model.layers.{i}.self_attn.o_proj.weight'].cpu().detach().numpy(), (1, 0))

        state_dict_flax[('model', 'norm', 'kernel')] = state_dict_pt[f'model.norm.weight'].cpu().detach().numpy()
        state_dict_flax[('lm_head', 'kernel')] = jnp.transpose(
            state_dict_pt[f'lm_head.weight'].cpu().detach().numpy(),
            (1, 0))
    return state_dict_flax


def mistral_convert_flax_to_pt(flax_params, config: MistralConfig, dtype=jnp.float16):
    torch_params = {}
    for key, tensor in flax_params.items():
        if match_keywords(key, ['kernel'], ['none']):
            tensor = tensor.T
        torch_params[key] = torch.from_numpy(tensor.astype(dtype=dtype))

    state_dict = {}
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, config.hidden_size // config.num_attention_heads, 2).float() / (
            config.hidden_size // config.num_attention_heads)))
    for layer_i in range(config.num_hidden_layers):
        state_dict.update({
            f"model.layers.{layer_i}.self_attn.q_proj.weight": torch_params[
                f"model.layers.{layer_i}.self_attn.q_proj.kernel"],
            f"model.layers.{layer_i}.self_attn.k_proj.weight": torch_params[
                f"model.layers.{layer_i}.self_attn.k_proj.kernel"],
            f"model.layers.{layer_i}.self_attn.v_proj.weight": torch_params[
                f"model.layers.{layer_i}.self_attn.v_proj.kernel"],
            f"model.layers.{layer_i}.self_attn.o_proj.weight": torch_params[
                f"model.layers.{layer_i}.self_attn.o_proj.kernel"],

            f"model.layers.{layer_i}.mlp.gate_proj.weight": torch_params[
                f"model.layers.{layer_i}.mlp.gate_proj.kernel"],
            f"model.layers.{layer_i}.mlp.down_proj.weight": torch_params[
                f"model.layers.{layer_i}.mlp.down_proj.kernel"],
            f"model.layers.{layer_i}.mlp.up_proj.weight": torch_params[
                f"model.layers.{layer_i}.mlp.up_proj.kernel"],

            f"model.layers.{layer_i}.input_layernorm.weight": torch_params[
                f"model.layers.{layer_i}.input_layernorm.kernel"],
            f"model.layers.{layer_i}.post_attention_layernorm.weight": torch_params[
                f"model.layers.{layer_i}.post_attention_layernorm.kernel"],
            f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq": inv_freq

        })

    state_dict.update({
        "model.embed_tokens.weight": torch_params["model.embed_tokens.embedding"],
        "model.norm.weight": torch_params["model.norm.kernel"],
        "lm_head.weight": torch_params["lm_head.kernel"],
    })
    return state_dict


def mistral_easydel_to_hf(path, config: MistralConfig):
    """
    Takes path to easydel saved ckpt and return the model in pytorch (Transformers Huggingface)
    """
    torch_params = load_and_convert_checkpoint_to_torch(path)
    edited_params = {}
    for k, v in torch_params.items():
        edited_params[k.replace('.kernel', '.weight').replace('.embedding', '.weight')] = v
    model = MistralForCausalLM(config=config)
    model.load_state_dict(edited_params)
    return model


def mistral_from_pretrained(model_id, device):
    """
    return: Weight or Params for EasyDel Model , Config
    """
    config = MistralConfig.from_pretrained(model_id)
    model = MistralForCausalLM.from_pretrained(model_id)
    easydel_wights = mistral_convert_hf_to_flax(
        state_dict=model.state_dict(),
        config=config,
        device=device
    )
    config.add_jax_args()

    del model
    gc.collect()
    return easydel_wights, config
