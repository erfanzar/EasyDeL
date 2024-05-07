import functools
import math
import typing

import fjformer
import flax.core
from jax import numpy as jnp, Array, lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec
import jax
from flax import linen as nn
from flax.traverse_util import unflatten_dict, flatten_dict
from flax.core import freeze, unfreeze
from typing import Union, Optional, Tuple
from flax.struct import dataclass
from ..attention_module import AttentionModule
from ..easydel_modelling_utils import EasyDeLFlaxPretrainedModel
from fjformer.linen import Linear
from flax.linen import partitioning as nn_partitioning, combine_masks
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput

from ..flax_modelling_utils import (
    ACT2FN,
    with_sharding_constraint,
    get_gradient_checkpoint_policy,
    repeat_kv_bnsh,
    apply_rotary_pos_emb,
    precompute_freq_cis,
    get_dot_general_by_bits,
    BaseJAXAttentionModule,
    block_wise_ffn
)
import chex
from .jetmoe_configuration import JetMoEConfig
from transformers.modeling_flax_outputs import FlaxCausalLMOutput, FlaxSequenceClassifierOutput, FlaxBaseModelOutput

re_mat = nn_partitioning.remat


@dataclass
class JetMoEBaseModelOutputWithPast(FlaxBaseModelOutput):
    last_hidden_state: chex.Array = None
    hidden_states: Optional[Tuple[chex.Array]] = None
    attentions: Optional[Tuple[chex.Array]] = None
    aux_loss: Optional[chex.Array] = None


@dataclass
class JetMoECausalLMOutputWithPast(FlaxCausalLMOutput):
    loss: Optional[chex.Array] = None
    logits: chex.Array = None
    hidden_states: Optional[Tuple[chex.Array]] = None
    attentions: Optional[Tuple[chex.Array]] = None
    aux_loss: Optional[chex.Array] = None


@dataclass
class JetMoESequenceClassifierOutputWithPast(FlaxSequenceClassifierOutput):
    loss: Optional[chex.Array] = None
    logits: chex.Array = None
    hidden_states: Optional[Tuple[chex.Array]] = None
    attentions: Optional[Tuple[chex.Array]] = None
    aux_loss: Optional[chex.Array] = None


def _make_sliding_window_causal_mask(
        input_ids_shape,
        dtype: jnp.dtype,
        past_key_values_length: int = 0,
        sliding_window: int = 4096,
):
    """
    Make causal mask used for sliding window attention
    """
    bsz, tgt_len = input_ids_shape

    tensor = jnp.full(
        (tgt_len, tgt_len),
        fill_value=1,
    )
    mask = jnp.tril(tensor, 0)
    mask = jnp.triu(mask, -sliding_window)
    mask = jnp.log(mask).astype(dtype)

    if past_key_values_length > 0:
        mask = jnp.concatenate(
            [jnp.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1)
    return mask[None, None, :, :].repeat(bsz, 0)


def compute_gating(k: int, num_experts: int, top_k_gates: jnp.ndarray, top_k_indices: jnp.ndarray) -> Tuple[
    chex.Array, chex.Array, chex.Array, chex.Array
]:
    """
    Compute gating values for the mixture of experts based on probabilities and top-k indices.
    """
    zeros = jnp.zeros([top_k_gates.shape[0], num_experts], dtype=top_k_gates.dtype)
    gates = zeros.at[jnp.arange(zeros.shape[0])[:, None], top_k_indices].set(1)
    expert_size = gates.astype(jnp.int32).sum(axis=0)
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    index_sorted_experts = jnp.argsort(top_k_experts)
    batch_index = lax.div(index_sorted_experts, k)
    batch_gates = top_k_gates[index_sorted_experts]
    return batch_gates, batch_index, expert_size, index_sorted_experts


class FlaxParallelExperts(nn.Module):
    num_experts: int
    input_size: int
    output_size: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, lax.Precision]] = None

    def setup(self) -> None:
        self.weight = self.param(
            "kernel", nn.initializers.uniform(
                1.0 / self.output_size
            ),
            (self.num_experts, self.input_size, self.output_size),
            self.param_dtype
        )

    def __call__(self, inputs: chex.Array, expert_size: int) -> chex.Array:
        input_list = jnp.split(inputs, expert_size, axis=0)
        output_list = []
        for expert_idx in range(self.num_experts):
            output_list.append(
                jax.lax.batch_matmul(
                    self.weight,
                    jnp.astype(input_list[expert_idx], self.dtype),
                    precision=self.precision
                )
            )
        return jnp.concatenate(output_list, axis=0)


class TopKGating(nn.Module):
    input_size: int
    num_experts: int
    top_k: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, lax.Precision]] = None

    def setup(self):
        self.layer = Linear(
            self.num_experts,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def compute_aux_loss(self, probs, logits, gates):
        count = logits.shape[0]
        probs = probs.sum(axis=0)
        freq = (gates > 0).astype(jnp.float32).sum(axis=0)
        lsesq = (jnp.log(jnp.exp(logits).sum(axis=-1)) ** 2).sum()

        probs_normalized = probs / jnp.sum(probs)
        freq_normalized = freq / jnp.sum(freq)
        switchloss = self.num_experts * (probs_normalized * freq_normalized).sum()
        zloss = lsesq / count
        loss = switchloss + 0.1 * zloss
        return loss

    def __call__(self, x, deterministic=False):
        logits = self.layer(x).astype(jnp.float32)
        top_k_logits, top_k_indices = jax.lax.top_k(logits, k=self.top_k)
        top_k_gates = nn.softmax(top_k_logits, axis=1).astype(x.dtype)

        if not deterministic:
            probs = nn.softmax(logits, axis=1)
            zeros = jnp.zeros_like(probs)
            gates = zeros.at[jnp.arange(zeros.shape[0])[:, None], top_k_indices].set(top_k_gates)
            loss = self.compute_aux_loss(probs, logits, gates)
        else:
            loss = 0.0

        return top_k_indices, top_k_gates, loss


class FlaxMoE(nn.Module):
    input_size: int
    hidden_size: int
    num_experts: int
    top_k: int
    bias: bool = True
    activation: str = None
    glu: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[str, lax.Precision]] = None

    def setup(self) -> None:
        if self.bias:
            self.bias_kernel = self.param(
                "bias",
                nn.initializers.zeros,
                (self.input_size,)
            )
        else:
            self.bias_kernel = None

        self.input_linear = FlaxParallelExperts(
            self.num_experts,
            self.input_size,
            self.hidden_size * 2 if self.glu else self.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.output_linear = FlaxParallelExperts(
            self.num_experts,
            self.hidden_size,
            self.input_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

        self.top_k = min(self.top_k, self.num_experts)
        self.activation_fn = ACT2FN[self.activation]

        self.router = TopKGating(
            input_size=self.input_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def compute_gate(self, x):
        top_k_indices, self.top_k_gates, loss = self.router(x)

        self.batch_gates, self.batch_index, expert_size, self.index_sorted_experts = compute_gating(
            self.top_k, self.num_experts, self.top_k_gates, top_k_indices
        )
        self.expert_size = expert_size.tolist()

        return loss

    def batch_forward(self, x):
        bsz, length, emb_size = x.shape
        x = x.reshape(-1, emb_size)
        loss = self.compute_gate(x)

        expert_inputs = x[self.batch_index]
        h = self.input_linear(expert_inputs, self.expert_size)
        if self.glu:
            h, g = h.chunk(2, dim=-1)
            h = self.activation_fn(h) * g
        else:
            h = self.activation_fn(h)
        expert_outputs = self.output_linear(h, self.expert_size)

        expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = jnp.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype)
        y = zeros.at[0, self.batch_index].add(expert_outputs)
        y = y.view((bsz, length, self.input_size))
        if self.bias_kernel is not None:
            bias = fjformer.linen.linen.control_quantization(self.bias_kernel, self.dtype)
            y = y + bias
        return y, loss

    def single_forward(self, x):
        bsz, length, emb_size = x.shape

        x = x.reshape(1, self.input_size)
        top_k_indices, top_k_gates, loss = self.router(x)

        y_list = []
        for i in range(self.top_k):
            expert_idx = top_k_indices[0, i]

            h = jax.lax.batch_matmul(
                self.input_linear.weight[expert_idx].astype(self.dtype),
                x.astype(self.dtype),
                precision=self.precision
            )
            if self.glu:
                h, g = jnp.split(h, 2, axis=-1)
                h = self.activation_fn(h) * g
            else:
                h = self.activation_fn(h)
            y = jax.lax.batch_matmul(
                self.output_linear.weight[expert_idx].astype(self.dtype),
                h.astype(self.dtype),
            ) * top_k_gates[0, i]

            y_list.append(y)

        y = jnp.sum(jnp.concatenate(y_list))
        y = y.reshape(bsz, length, self.input_size)
        if self.bias_kernel is not None:
            y = y + self.bias_kernel
        return y, loss

    def __call__(self, x, deterministic: bool = False):
        bsz, length, emb_size = x.shape
        if bsz * length == 1:
            return self.single_forward(x)
        else:
            return self.batch_forward(x)

    def single_map(self, x, deterministic: bool = False):
        bsz, length, emb_size = x.shape

        x = x.reshape(1, self.input_size)
        self.top_k_indices, self.top_k_gates, loss = self.router(x, deterministic=False)  # type: ignore

        y_list = []
        for i in range(self.top_k):
            expert_idx = self.top_k_indices[0, i]
            y = jax.lax.batch_matmul(
                self.input_linear.weight[expert_idx],
                x,
                precision=self.precision
            )
            y_list.append(y)
        y = jnp.concatenate(y_list, axis=0)
        y = y.reshape(bsz, length, self.top_k, -1)
        return y, loss

    def batch_map(self, x, deterministic: bool = False):

        bsz, length, emb_size = x.shape
        x = x.reshape(-1, emb_size)
        loss = self.compute_gate(x)

        expert_inputs = x[self.batch_index]
        expert_outputs = self.input_linear(expert_inputs, self.expert_size)

        zeros = jnp.zeros(
            (bsz * length * self.top_k, self.hidden_size), dtype=expert_outputs.dtype
        )
        y = zeros.at[0, self.index_sorted_experts].add(expert_outputs)
        y = y.reshape(bsz, length, self.top_k, -1)
        return y, loss

    def map(self, x, deterministic: bool = False):
        bsz, length, emb_size = x.shape
        if bsz * length == 1:
            return self.single_map(x, deterministic=deterministic)
        else:
            return self.batch_map(x, deterministic=deterministic)

    def single_reduce(self, x, deterministic: bool = False):
        bsz, length, k, emb_size = x.shape

        x = x.reshape(k, emb_size)

        y_list = []
        for i in range(self.top_k):
            expert_idx = self.top_k_indices[0, i]
            y = jax.lax.batch_matmul(
                self.output_linear.weight[expert_idx],
                x[i],
                precision=self.precision
            ) * self.top_k_gates[0, i]
            y_list.append(y)
        y = jnp.sum(
            jnp.concatenate(y_list)
        )
        y = y.reshape(bsz, length, self.input_size)
        if self.bias_kernel is not None:
            y = y + self.bias_kernel
        return y

    def batch_reduce(self, x, deterministic: bool = False):
        bsz, length, k, emb_size = x.shape
        x = x.reshape(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_linear(expert_inputs, self.expert_size)

        expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = jnp.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype)
        y = zeros.at[0, self.batch_index].add(expert_outputs)
        y = y.reshape(bsz, length, self.input_size)
        if self.bias_kernel is not None:
            y = y + self.bias_kernel
        return y

    def reduce(self, x, deterministic: bool = False):
        bsz, length, k, emb_size = x.shape
        if bsz * length == 1:
            return self.single_reduce(x, deterministic=deterministic)
        else:
            return self.batch_reduce(x, deterministic=deterministic)


class JetMoERMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16

    def setup(self) -> None:
        self.weight = self.param(
            'kernel',
            nn.initializers.ones,
            (self.dim,),
            self.param_dtype,
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)
        weight = jnp.asarray(self.weight, self.dtype)
        return output * weight


class FlaxJetMoERotaryEmbedding(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def __call__(self, key, query, freq_cis, position_ids):
        sin, cos = freq_cis

        sin = sin[position_ids][:, None, :, :]
        cos = cos[position_ids][:, None, :, :]

        key = apply_rotary_pos_emb(key, sin, cos)
        query = apply_rotary_pos_emb(query, sin, cos)

        return query.astype(self.dtype), key.astype(self.dtype)


class FlaxJetMoEMLP(nn.Module):
    config: JetMoEConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[Union[str, jax.lax.Precision]] = jax.lax.Precision("fastest")

    def setup(self) -> None:
        dense = functools.partial(
            Linear,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=nn.initializers.normal(),
            **get_dot_general_by_bits(self.config.bits, self.config.easy_method)
        )
        self.gate_proj = dense(self.config.intermediate_size)
        self.up_proj = dense(self.config.intermediate_size)
        self.down_proj = dense(self.config.hidden_size)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(
            self,
            x: chex.Array,
            e: bool = False  # Ignored
    ):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
