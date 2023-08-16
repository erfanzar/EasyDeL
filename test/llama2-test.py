from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
print(str(Path(__file__).resolve().parent.parent))
import torch

import jax
from jax import numpy as jnp
from EasyDel.modules.llama.modelling_llama_flax import pre_compute_llama_freqs_cis, rotate_half_llama, \
    apply_rotary_pos_emb_llama, LlamaConfig, create_freqs_cis_from_config, FlaxLlamaAttention, apply_rotary_emb_v2, \
    apply_rotary_emb
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding, LlamaMLP, \
    LlamaAttention, LlamaForCausalLM, apply_rotary_pos_emb as apply_rotary_pos_emb_torch
import numpy as np


def get_apply_fn(config):
    if config.rotary_type == 'complex':
        fn = apply_rotary_emb
    elif config.rotary_type == 'open':
        fn = apply_rotary_pos_emb
    elif config.rotary_type == 'lm2':
        fn = apply_rotary_emb_v2
    elif config.rotary_type == 'normal':
        fn = apply_rotary_pos_emb_llama
    else:
        raise RuntimeError
    return fn


def pt2np(array):
    return array.detach().cpu().numpy()


def np2jax(array):
    return jnp.asarray(array)


def pt2jax(array):
    return np2jax(pt2np(array))


def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device = 'cpu', past_key_values_length: int = 0
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len=None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def transfer_weight(w):
    return jnp.asarray(w.transpose(0, 1).detach().numpy()).astype('float32')


def test_rope(config: LlamaConfig):
    dim = config.hidden_size // config.num_attention_heads
    rope_torch = LlamaRotaryEmbedding(
        dim, config.max_position_embeddings
    )
    cos_torch, sin_torch = rope_torch.cos_cached, rope_torch.sin_cached
    freq_cis = create_freqs_cis_from_config(
        config
    )
    cos_jax, sin_jax = freq_cis
    assert np.allclose(
        cos_jax, pt2jax(cos_torch).squeeze()
    ), 'cos rope assertion Failed !'
    assert np.allclose(
        sin_jax, pt2jax(sin_torch).squeeze()
    ), 'sin rope assertion Failed !'
    return True


def test_apply_rotary(config: LlamaConfig):
    q = torch.randn(
        (1, config.max_position_embeddings, config.num_attention_heads,
         config.hidden_size // config.num_attention_heads))
    jax_q = pt2jax(q)
    q = q.transpose(1, 2)

    k = torch.randn(
        (1, config.max_position_embeddings, config.num_attention_heads,
         config.hidden_size // config.num_attention_heads))
    jax_k = pt2jax(k)
    k = k.transpose(1, 2)

    position_ids = torch.arange(config.max_position_embeddings).reshape(1, -1)
    position_ids_jax = pt2jax(position_ids)

    dim = config.hidden_size // config.num_attention_heads
    rope_torch = LlamaRotaryEmbedding(
        dim, config.max_position_embeddings
    )
    cos_torch, sin_torch = rope_torch.cos_cached, rope_torch.sin_cached
    freq_cis = create_freqs_cis_from_config(config)

    # Applying

    fn = get_apply_fn(config)
    if config.rotary_type == 'normal':
        cos_jax, sin_jax = freq_cis
        jax_q, jax_k = fn(jax_q, jax_k, sin=sin_jax[:, :, :config.max_position_embeddings, :],
                          cos=cos_jax[:, :, :config.max_position_embeddings, :],
                          position_ids=position_ids_jax, index=1 if config.do_torch_attn else 2)
    if config.rotary_type == 'complex':
        freq_cis = jnp.take(freq_cis, position_ids_jax, axis=0)
        jax_q, jax_k = fn(jax_q, jax_k, freqs_cis=freq_cis)
    q, k = apply_rotary_pos_emb_torch(q, k, cos_torch, sin_torch, position_ids)
    assert np.allclose(jax_q, pt2jax(q.transpose(1, 2))), 'Assertion for Q Failed in Applying Rope'
    assert np.allclose(jax_k, pt2jax(k.transpose(1, 2))), 'Assertion for K Failed in Applying Rope'
    return True


def test_attention(config: LlamaConfig):
    hidden_state = torch.randn(1, config.max_position_embeddings, config.hidden_size, dtype=torch.float32)
    position_ids = torch.arange(config.max_position_embeddings).reshape(1, -1)
    attention_mask = torch.ones((1, config.max_position_embeddings))
    causal_mask = _make_causal_mask(attention_mask.shape, torch.float32)

    jax_hidden_state = pt2jax(hidden_state)
    jax_position_ids = pt2jax(position_ids)
    jax_attention_mask = pt2jax(attention_mask)

    expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=config.max_position_embeddings)
    combined_attention_mask = (expanded_attn_mask + causal_mask)
    config.pretraining_tp = 1
    torch_attn = LlamaAttention(
        config=config
    )
    flax_attn = FlaxLlamaAttention(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision('highest')
    )

    freq_cis_jax = create_freqs_cis_from_config(
        config
    )

    flax_params = {
        'params': {
            'k_proj': {'kernel': transfer_weight(torch_attn.k_proj.weight)},
            'o_proj': {'kernel': transfer_weight(torch_attn.o_proj.weight)},
            'q_proj': {'kernel': transfer_weight(torch_attn.q_proj.weight)},
            'v_proj': {'kernel': transfer_weight(torch_attn.v_proj.weight)}
        }
    }

    pred_torch = torch_attn.forward(
        hidden_states=hidden_state,
        attention_mask=combined_attention_mask,
        position_ids=position_ids
    )[0]
    pred_jax = flax_attn.apply(
        flax_params,
        attention_mask=jax_attention_mask,
        hidden_states=jax_hidden_state,
        freqs_cis=freq_cis_jax,
        position_ids=jax_position_ids
    )[0]
    assert np.allclose(
        pred_jax, pt2jax(pred_torch)
    ), 'Jax and Torch Attn predictions are not the same Failed !'
    return True


if __name__ == "__main__":
    torch.manual_seed(42)
    config_ = LlamaConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_attention_heads=8,
        num_hidden_layers=2,
        rotary_type='lm2',
        from_pt=True,
        do_torch_attn=False,
        use_torch_to_init_rope_normal=True
    )
    test_rope(config_)
    print('Rope Test Passed Successfully')
    test_apply_rotary(config_)
    print('Applying Rope Test Passed Successfully')
    test_attention(config_)
    print('Attention Test Passed Successfully')
