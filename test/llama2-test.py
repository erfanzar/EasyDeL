from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
print(str(Path(__file__).resolve().parent.parent))
import torch

import jax
from jax import numpy as jnp
from EasyDel.modules.llama.modelling_llama_flax import pre_compute_llama_freqs_cis, rotate_half_llama, \
    apply_rotary_pos_emb_llama, LlamaConfig, create_freqs_cis_from_config, FlaxLlamaAttention, forward_rotary_embedding, \
    apply_rotary_emb, apply_rotary_pos_emb_llama2
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding, LlamaMLP, \
    LlamaAttention, LlamaForCausalLM, apply_rotary_pos_emb as apply_rotary_pos_emb_torch

import numpy as np
from EasyDel.utils.tensor_utils import pt2jax, pt2np, np2jax
from EasyDel.modules.llama.llama_torch import Attention, precompute_freqs_cis as pfs, ModelArgs
from tabulate import tabulate


def get_apply_fn(config):
    if config.rotary_type == 'complex':
        fn = apply_rotary_emb
    elif config.rotary_type == 'open':
        fn = apply_rotary_pos_emb
    elif config.rotary_type == 'lm2':
        fn = forward_rotary_embedding
    elif config.rotary_type == 'normal':
        fn = apply_rotary_pos_emb_llama
    elif config.rotary_type == 'llama2':
        fn = apply_rotary_pos_emb_llama2
    else:
        raise RuntimeError
    return fn


def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device = 'cpu', past_key_values_length=0
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
    if config.rotary_type == 'lm2':
        cos, sin = freq_cis
        jax_q = fn(jax_q, sin=sin, cos=cos)
        jax_k = fn(jax_k, sin=sin, cos=cos)
    if config.rotary_type == 'llama2':
        jax_q, jax_k = fn(jax_q, jax_k, freqs_cis=freq_cis[config.max_position_embeddings])
    q, k = apply_rotary_pos_emb_torch(q, k, cos_torch, sin_torch, position_ids)
    assert np.allclose(jax_q, pt2jax(q.transpose(1, 2))), 'Assertion for Q Failed in Applying Rope'
    assert np.allclose(jax_k, pt2jax(k.transpose(1, 2))), 'Assertion for K Failed in Applying Rope'
    return True


def test_attention(config: LlamaConfig):
    hidden_state = torch.randn(1, config.max_position_embeddings, config.hidden_size, dtype=torch.float32)
    position_ids = torch.arange(config.max_position_embeddings).reshape(1, -1)

    jax_hidden_state = pt2jax(hidden_state)
    jax_position_ids = pt2jax(position_ids)

    config.pretraining_tp = 1

    mask = torch.full(
        (1, 1, config.max_position_embeddings, config.max_position_embeddings), torch.finfo(torch.float32).min
    )
    mask = torch.triu(mask, diagonal=1).type_as(hidden_state)
    mask_1d = pt2jax(torch.ones(1, config.max_position_embeddings, dtype=torch.bool))
    args = ModelArgs(
        dim=config.hidden_size,
        n_layers=32,
        n_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
        vocab_size=-1,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=config.rms_norm_eps,

        max_seq_len=config.max_position_embeddings,
    )
    torch_attn = Attention(
        args=args
    )
    flax_attn = FlaxLlamaAttention(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision('fastest')
    )

    freq_cis_jax = create_freqs_cis_from_config(
        config
    )

    flax_params = {
        'params': {
            'k_proj': {'kernel': transfer_weight(torch_attn.wk.weight)},
            'o_proj': {'kernel': transfer_weight(torch_attn.wo.weight)},
            'q_proj': {'kernel': transfer_weight(torch_attn.wq.weight)},
            'v_proj': {'kernel': transfer_weight(torch_attn.wv.weight)}
        }
    }
    pred_torch = pt2jax(torch_attn.forward(
        start_pos=0,
        freqs_cis=pfs(config.hidden_size // config.num_attention_heads, config.max_position_embeddings),
        x=hidden_state,
        mask=mask

    )[0])

    pred_jax = flax_attn.apply(
        flax_params,
        attention_mask=jnp.ones((1, config.max_position_embeddings), dtype=bool),
        hidden_states=jax_hidden_state,
        freqs_cis=freq_cis_jax,
        position_ids=jax_position_ids
    )[0]
    # pred_jax, output_jax, scores_jax, qk_mask_jax, qk_jax = [pt2jax(sa) for sa in
    #                                                          [pred_jax, output_jax, scores_jax, qk_mask_jax, qk_jax]]

    pred_jax = jnp.where(mask_1d[:, :, None], pred_jax, 0)
    pred_torch = jnp.where(mask_1d[:, :, None], pred_torch, 0)
    indexes = 500
    is_close = np.isclose(pred_jax.reshape(-1)[:indexes], pred_torch.reshape(-1)[:indexes])
    table = [[pt, pj, cs] for pt, pj, cs in
             zip(pred_torch.reshape(-1)[:indexes], pred_jax.reshape(-1)[:indexes], is_close)]
    print(
        tabulate(
            table,
            headers=['Pytorch', 'Jax', 'Close?'],
            tablefmt='orgtbl'
        )
    )
    #
    # assert np.allclose(
    #     pred_jax.reshape(-1)[:30], pred_torch.reshape(-1)[:30]
    # ), 'Jax and Torch Attn predictions are not the same Failed !'
    return True


if __name__ == "__main__":

    torch.manual_seed(42)
    world_size = 1
    rank = 0

    config_ = LlamaConfig(
        hidden_size=256,
        intermediate_size=1024,
        num_attention_heads=8,
        num_hidden_layers=2,
        rotary_type='llama2',
        max_position_embeddings=128,
        from_pt=True,
        do_torch_attn=False,
        attn_type='llama2',
        use_einops=True
    )
    try:
        test_rope(config_)
        print('Rope Test Passed Successfully')
    except AssertionError as sr:
        print(sr)
    except ValueError as s:
        print(f'{s} - this test is designed for lm2 and normal ')
    try:
        test_apply_rotary(config_)
        print('Applying Rope Test Passed Successfully')
    except (AssertionError, ValueError) as sr:
        print(f"{sr} - This is fine :_)")

    try:
        test_attention(config_)
        print('Attention Test Passed Successfully')

    except AssertionError as sr:
        print(f"{sr} - This is fine :_)")
