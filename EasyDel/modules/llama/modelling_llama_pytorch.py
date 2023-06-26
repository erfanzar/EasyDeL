import math
import os
from dataclasses import dataclass
from typing import Optional, Union, Any

import torch
import torch.utils.checkpoint

from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import logging


def make2d(tensor):
    return tensor.view(-1, tensor.size(-1))


logger = logging.get_logger(__name__)

try:
    if os.environ['USE_JIT'] == '1':
        Module = torch.jit.ScriptModule
        function = torch.jit.script_method
    else:
        Module = nn.Module


        def function(func):
            return func
except KeyError:
    Module = nn.Module


    def function(func):
        return func


class LlamaConfig(PretrainedConfig):
    def __init__(self,
                 initializer_range: float = 0.02,
                 hidden_size: int = 768,
                 bos_token_id=2,
                 eos_token_id=1,
                 pad_token_id=0,
                 intermediate_size: int = 2048,
                 num_hidden_layers: int = 4,
                 rms_norm_eps: int = 1e-6,
                 vocab_size: int = 32000,
                 num_attention_heads: int = 8,
                 use_cache: bool = True,
                 weight_decay: float = 0.02,
                 max_sequence_length: int = 768,

                 ):
        super().__init__(eos_token_id=eos_token_id, bos_token_id=bos_token_id, pad_token_id=pad_token_id)
        self.max_sequence_length = max_sequence_length
        self.weight_decay = weight_decay
        self.use_cache = use_cache
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range


class LlamaRMSNorm(Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        dt = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = hidden_states * torch.rsqrt(
            hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)
        hidden_states = hidden_states.to(dt)
        return self.weight * hidden_states


class LlamaRotaryEmbedding(Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.inv_freq = inv_freq

        self.max_seq_length_cached = max_position_embeddings
        t = torch.arange(self.max_seq_length_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    @function
    def forward(self, x, seq_length: int):
        if seq_length > self.max_seq_length_cached:
            self.max_seq_length_cached = seq_length
            t = torch.arange(self.max_seq_length_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return (
            self.cos_cached[:, :, :seq_length, ...].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:, :, :seq_length, ...].to(dtype=x.dtype, device=x.device),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset: q.shape[-2] + offset, :]
    sin = sin[..., offset: q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,

    ):
        # same as what used on Llama
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = nn.functional.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Conv1D(Module):
    def __init__(self, in_c, out_c, bias=False):
        super().__init__()
        self.out_c = out_c
        self.weight = nn.Parameter(torch.empty(in_c, out_c))
        self.bias = nn.Parameter(torch.ones(out_c)) if bias else None
        torch.nn.init.normal_(self.weight, std=0.02)

    @function
    def forward(self, x):
        out_size = x.size()[:-1] + (self.out_c,)
        x = x.view(-1, x.size(-1))

        out = torch.addmm(self.bias, x, self.weight) if self.bias is not None else torch.matmul(x, self.weight)
        return out.view(out_size)


class LlamaAttention(Module):

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert (self.head_dim * num_heads) == self.hidden_size

        # self.q_proj = Conv1D(
        #     hidden_size,
        #     num_heads * self.head_dim,
        #     bias=False,
        # )
        # self.k_proj = Conv1D(
        #     hidden_size,
        #     num_heads * self.head_dim,
        #     bias=False,
        # )
        # self.v_proj = Conv1D(
        #     hidden_size,
        #     num_heads * self.head_dim,
        #     bias=False,
        # )
        # self.o_proj = Conv1D(
        #     num_heads * self.head_dim,
        #     hidden_size,
        #     bias=False,
        # )

        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

    def _shape(self, tensor: torch.Tensor, seq_length: int, bsz: int):
        return tensor.view(bsz, seq_length, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @function
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, q_len, _ = hidden_states.size()
        # logger.info(f'In attention : {hidden_states.dtype}')
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_length = key_states.shape[-2]
        offset = 0
        # if past_key_value is not None:
        #     offset = past_key_value[0].shape[-2]
        #     kv_seq_length += offset

        cos, sin = self.rotary_emb(value_states, seq_length=kv_seq_length)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, offset=offset)

        # if past_key_value is not None:
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        assert attn_weights.size() == (bsz, self.num_heads, q_len, kv_seq_length)

        if attention_mask is not None:
            assert attention_mask.size() == (bsz, 1, q_len, kv_seq_length)
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        # if not output_attentions:
        #     attn_weights = torch.tensor([None])

        # return attn_output, attn_weights, past_key_value
        # Usable With JIT
        return attn_output


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int = 0):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len != 0 else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask
    inverted_mask_cast = inverted_mask.to(dtype)

    msk = inverted_mask_cast.to(torch.bool)
    return inverted_mask_cast.masked_fill(msk, -65504)


def _make_causal_mask(input_ids: torch.Tensor, dtype: torch.dtype):
    bsz, tgt_len = input_ids.shape

    mask = torch.full((tgt_len, tgt_len), torch.tensor(-65504))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


class LlamaBlock(Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size, )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @function
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor = None,

    ):
        residual = hidden_states
        # logger.info(f'hidden_states _ NORM : {hidden_states.dtype}')
        hidden_states = self.input_layernorm(hidden_states)
        # logger.info(f'hidden_states _ NORM : {hidden_states.dtype}')
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states

        # if output_attentions:
        #     ...
        # TODO: Write down the use with JIT
        # if use_cache:
        #     ...
        # TODO: Write down the use with JIT

        return outputs


class LlamaModel(PreTrainedModel):

    def __init__(self, config: LlamaConfig):
        super(LlamaModel, self).__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.config = config
        self.apply(self._init_weights)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @staticmethod
    def _set_gradient_checkpointing(module, value=False):
        if isinstance(module, LlamaBlock):
            module.gradient_checkpointing = value

    @staticmethod
    def _prepare_attention_mask(attention_mask, input_ids, inputs_embeds):

        input_shape = input_ids.shape
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_ids, inputs_embeds.dtype
            )
        else:
            combined_attention_mask = torch.tensor([1])
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                combined_attention_mask)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @function
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask)
        attention_mask = make2d(attention_mask)
        input_ids = make2d(input_ids)
        batch_size, seq_lengthgth = input_ids.shape
        seq_lengthgth_with_past = seq_lengthgth
        # logger.info(f'input_ids : {input_ids.dtype}')
        hidden_states = self.embed_tokens(input_ids)
        # logger.info(f'hidden_states : {hidden_states.dtype}')
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_lengthgth_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = make2d(attention_mask)
        attention_mask = self._prepare_attention_mask(
            attention_mask, input_ids, hidden_states
        )

        attention_mask = attention_mask.to(hidden_states)

        for idx, block in enumerate(self.layers):
            # logger.info(f'hidden_states {idx}: {hidden_states.dtype}')
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,

            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class LlamaForCausalLM(PreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @function
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            **kwargs
    ):
        outputs = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs
        logits = self.lm_head(hidden_states)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                                     shift_labels.view(-1))
        else:
            loss = None

        return loss, logits

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
