# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma4 Assistant (MTP drafter) modeling for EasyDeL.

Implements the drafter architecture inferred from the official
``google/gemma-4-*-it-assistant`` safetensors checkpoint:

Top-level modules
-----------------
- ``model.embed_tokens``: drafter's own (vocab, hidden) embedding,
  tied to the LM head.
- ``model.layers.0..3``: 4 Gemma4-style decoder layers (sliding /
  sliding / sliding / full attention). Each layer has **only**
  ``q_proj`` + ``o_proj`` plus dual norms — there are no ``k_proj``
  / ``v_proj`` tensors. K and V come from the **target** model's KV
  cache at runtime, supplied via the speculative-decode controller.
- ``model.norm``: final RMSNorm.
- ``pre_projection``: ``Linear(2 * backbone_hidden_size -> hidden)``.
  Fuses concat(target_token_embeds, target_hidden_states).
- ``post_projection``: ``Linear(hidden -> backbone_hidden_size)``.
  Feeds the drafter's draft hidden back to backbone space for the
  per-step feedback buffer that the next draft step consumes.
- ``masked_embedding.centroids``: ``(num_centroids, hidden)`` matrix
  of cluster prototypes for the sparse output head.
- ``masked_embedding.token_ordering``: ``(vocab_size,)`` int64
  permutation that groups vocabulary into ``num_centroids`` contiguous
  clusters of ``vocab_size // num_centroids`` tokens each.

Forward semantics (training / standalone inference)
---------------------------------------------------
The drafter is normally driven by an HF ``assistant_model=`` hook on
the target's ``.generate(...)``. For training and verification it can
also be run standalone given target-provided KV tensors.

This file implements:

1. :class:`Gemma4AssistantCentroidHead` — full implementation of the
   two-stage sparse softmax over the vocabulary. Returns either
   ``(top_logits, top_token_ids)`` for cheap argmax/sampling or a
   dense ``(B, S, V)`` tensor with ``-inf`` outside the selected
   tokens.
2. :class:`Gemma4AssistantDecoderLayer` — single drafter block with
   Q-only attention. Forward signature accepts ``key_states`` and
   ``value_states`` directly from the caller (cross-model KV
   sharing). Falls back to ``key_states = value_states = query_states``
   if not provided — useful for shape-level sanity tests but
   semantically wrong for speculative decoding.
3. :class:`Gemma4AssistantModel` — wires embed → projection → 4
   layers → final norm → post_projection.
4. :class:`Gemma4AssistantForCausalLM` — adds tied LM head and the
   centroid output path.

Runtime contract for speculative decoding (TODO — not implemented in
this file): the caller must call :meth:`forward_with_target` (added
in a follow-up) and pass per-layer ``(k, v)`` tensors gathered from
the target model's KV cache. See ``docs/spec_decode_gemma4.md`` for
the proposer protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, partial

import jax
import jax.numpy as jnp
import spectrax as spx
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Float, Int
from spectrax import apply_logical_sharding, common_types, nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import BaseModelOutput
from easydel.infra.utils import ACT2FN
from easydel.layers import (
    ColumnParallelLinear,
    Embed,
    RowParallelLinear,
    dense_gate_up_layout,
    get_frequencies,
    split_fused_gate_up_projection,
)
from easydel.layers.rotary._compute_fns import apply_basic_rope
from easydel.modules.gemma4.modeling_gemma4 import Gemma4RMSNorm

from .configuration_gemma4_assistant import (
    Gemma4AssistantConfig,
    Gemma4AssistantTextConfig,
)


@dataclass(frozen=True)
class Gemma4AssistantOutput:
    """Output of :class:`Gemma4AssistantForCausalLM`.

    Attributes:
        last_hidden_state: ``(B, S, hidden)`` final drafter hidden
            states (post-final-RMSNorm; pre-``post_projection``).
        backbone_hidden_state: ``(B, S, backbone_hidden_size)`` —
            ``post_projection(last_hidden_state)``. Fed back to the
            next draft step's ``pre_projection``.
        logits: Either dense ``(B, S, vocab_size)`` with ``-inf``
            outside selected centroid tokens, or ``None`` when only
            sparse outputs were requested.
        top_logits: ``(B, S, top_k * tokens_per_centroid)`` sparse
            logits over the centroid-selected candidate tokens.
        top_token_ids: ``(B, S, top_k * tokens_per_centroid)`` vocab
            IDs that align with ``top_logits``.
    """

    last_hidden_state: Float[Array, "batch seq_len hidden"]
    backbone_hidden_state: Float[Array, "batch seq_len backbone_hidden"]
    logits: Float[Array, "batch seq_len vocab"] | None = None
    top_logits: Float[Array, "batch seq_len candidates"] | None = None
    top_token_ids: Int[Array, "batch seq_len candidates"] | None = None




class Gemma4AssistantCentroidHead(spx.Module):
    """Two-stage centroid-clustered output head.

    .. note::
       **TPU performance caveat:** the algorithmic FLOP savings vs full
       softmax (~102x at ``V=262144``, ``num_centroids=2048``,
       ``top_k=32``) do NOT fully translate to TPU wallclock at small
       to moderate batch/seq scales. The dense
       ``hidden @ embed.T`` is a single fast MXU matmul, while the
       centroid path requires a gather over the 262K-row embedding
       table that is slow on TPU. Measured speedups on v5p:
       ``B=1 S=32 → 0.57x``, ``B=4 S=256 → 0.12x``. The win is real on
       GPU (where vLLM ships a fused gather kernel) and at much larger
       ``V``. The implementation here matches the HF checkpoint
       semantics; pick the dense fallback (``use_ordered_embeddings=False``)
       on TPU until a fused TPU kernel exists.

    Encoded by the ``masked_embedding.*`` tensors:

    - ``centroids.weight``: ``(num_centroids, hidden)`` prototypes.
      First stage computes ``centroid_logits = hidden @ centroids.T``,
      then picks the top-``centroid_intermediate_top_k`` centroids
      per position.
    - ``token_ordering``: ``(vocab_size,)`` int64 permutation that
      reorders the vocabulary so each consecutive
      ``tokens_per_centroid = vocab_size // num_centroids`` block
      belongs to one centroid. Tokens in selected centroids are
      gathered into the candidate set.

    Second stage scores the candidate tokens against the drafter's
    tied LM-head matrix (passed in as ``embed_weight``):

    .. code-block:: text

        candidate_ids   = token_ordering[top_centroids]            # (B, S, top_k*tokens_per_c)
        candidate_embs  = embed_weight[candidate_ids]              # (B, S, K, H)
        top_logits      = einsum("bsh, bskh -> bsk",
                                 hidden, candidate_embs)

    Returning either a dense ``(B, S, V)`` tensor with ``-inf``
    outside the candidate set or the (sparse logits, candidate IDs)
    pair lets callers pick between exact sampling and cheap argmax.
    """

    def __init__(
        self,
        config: Gemma4AssistantConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Build the centroid head.

        Args:
            config: Drafter config; reads ``num_centroids``,
                ``centroid_intermediate_top_k``, and
                ``text_config.vocab_size`` / ``text_config.hidden_size``.
            dtype: Compute dtype.
            param_dtype: Param storage dtype.
            precision: Matmul precision.
            rngs: PRNG container.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.num_centroids = int(config.num_centroids)
        self.top_k = int(config.centroid_intermediate_top_k)
        vocab = int(config.text_config.vocab_size)
        if self.num_centroids == 0:
            raise ValueError("Gemma4AssistantCentroidHead requires num_centroids > 0.")
        if vocab % self.num_centroids != 0:
            raise ValueError(f"vocab_size ({vocab}) must be divisible by num_centroids ({self.num_centroids}).")
        self.tokens_per_centroid = vocab // self.num_centroids

        self.centroids = ColumnParallelLinear(
            config.text_config.hidden_size,
            self.num_centroids,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(0.02),
            rngs=rngs,
        )
        self.token_ordering = spx.Parameter(jnp.arange(vocab, dtype=jnp.int32))

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden"],
        embed_weight: Float[Array, "vocab hidden"],
        return_dense_logits: bool = False,
    ) -> tuple[
        Float[Array, "batch seq_len candidates"],
        Int[Array, "batch seq_len candidates"],
        Float[Array, "batch seq_len vocab"] | None,
    ]:
        """Run the centroid head.

        Args:
            hidden_states: Drafter hidden ``(B, S, H)``.
            embed_weight: Drafter token-embedding matrix
                ``(vocab, hidden)``. Used as the tied LM head weight.
                Pass ``model.embed_tokens.weight``.
            return_dense_logits: Whether to also materialize a dense
                ``(B, S, V)`` logits tensor with ``-inf`` outside the
                selected candidates. Sampling utilities that need a
                full-vocab tensor should set ``True``; argmax /
                top-token paths can leave it ``False`` and use
                ``top_logits`` + ``top_token_ids`` directly.

        Returns:
            ``(top_logits, top_token_ids, dense_logits | None)`` —
            ``top_logits`` and ``top_token_ids`` are
            ``(B, S, top_k * tokens_per_centroid)``; ``dense_logits``
            is ``(B, S, vocab_size)`` or ``None``.
        """
        centroid_logits = self.centroids(hidden_states).astype(jnp.float32)  # (B, S, C)
        _, top_centroids = jax.lax.top_k(centroid_logits, self.top_k)  # (B, S, top_k)

        clusters = self.token_ordering.value.reshape(self.num_centroids, self.tokens_per_centroid)
        candidate_ids = clusters[top_centroids]  # (B, S, top_k, tokens_per_centroid)
        b, s = top_centroids.shape[:2]
        K = self.top_k * self.tokens_per_centroid
        candidate_ids = candidate_ids.reshape(b, s, K).astype(jnp.int32)

        candidate_embs = embed_weight[candidate_ids]  # (B, S, K, H)
        top_logits = jnp.einsum(
            "bsh,bskh->bsk",
            hidden_states.astype(jnp.float32),
            candidate_embs.astype(jnp.float32),
        )

        dense_logits = None
        if return_dense_logits:
            vocab = int(self.config.text_config.vocab_size)
            neg_inf = jnp.full((b, s, vocab), -jnp.inf, dtype=jnp.float32)
            dense_logits = neg_inf.at[
                jnp.arange(b)[:, None, None],
                jnp.arange(s)[None, :, None],
                candidate_ids,
            ].set(top_logits)
        return top_logits, candidate_ids, dense_logits




class Gemma4AssistantMLP(spx.Module):
    """Gemma4-style GeGLU MLP (gate * gelu_pytorch_tanh + up → down)."""

    def __init__(
        self,
        config: Gemma4AssistantTextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Build the GeGLU MLP.

        Args:
            config: Drafter text config; reads ``hidden_size`` and
                ``intermediate_size``.
            dtype: Compute dtype.
            param_dtype: Param dtype.
            precision: Matmul precision.
            rngs: PRNG.
        """
        self.gate_up_proj = ColumnParallelLinear(
            config.hidden_size,
            (config.intermediate_size, config.intermediate_size),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(0.02),
            layout=dense_gate_up_layout(config.intermediate_size),
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(0.02),
        )
        self.act_fn = ACT2FN.get("gelu_pytorch_tanh", ACT2FN["gelu"])

    @property
    def reform_param(self):
        """Checkpoint-reform rules for the fused ``gate_up_proj`` parameter.

        Returns:
            dict: Split/merge transformations built from the fused gate-up
            projection layout, used by the checkpoint loader to map between
            the HF-stored fused tensor and EasyDeL's split representation.
        """
        return self.gate_up_proj.build_reform_param("gate_up_proj", config=self.config)

    def forward(self, x: Float[Array, "batch seq_len hidden"]) -> Float[Array, "batch seq_len hidden"]:
        """SwiGLU forward (despite the name; Gemma4 uses gelu)."""
        gate_up = self.gate_up_proj(x)
        gate, up = split_fused_gate_up_projection(gate_up, config=self.config)
        return self.down_proj(self.act_fn(gate) * up)


class Gemma4AssistantQOnlyAttention(spx.Module):
    """Q-only attention for the drafter.

    Maintains only ``q_proj``, ``q_norm``, and ``o_proj`` weights.
    K and V tensors are supplied by the caller (the speculative-decode
    controller, which extracts them from the TARGET model's KV cache
    at the layer assigned to this drafter layer).

    The HF assistant checkpoint stores per-layer-type dimensions:

    - sliding-attention layers: ``q_proj`` outputs
      ``num_attention_heads * head_dim`` (e.g. 4 * 256 = 1024 for E4B).
    - full-attention layer: ``q_proj`` outputs
      ``num_attention_heads * global_head_dim`` (e.g. 4 * 512 = 2048).

    Target-side RoPE is already baked into K. This layer applies the
    matching Gemma4 RoPE to Q before cross-attending to the target K/V.
    """

    def __init__(
        self,
        config: Gemma4AssistantTextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Build the Q-only attention block.

        Picks ``head_dim`` based on ``config.layer_types[layer_idx]``: the
        full-attention layer uses ``global_head_dim``, sliding layers use
        ``head_dim``. Allocates ``q_proj`` (column-parallel), ``o_proj``
        (row-parallel), and the per-head ``q_norm`` (Gemma4 RMSNorm).

        Args:
            config (Gemma4AssistantTextConfig): Drafter text config.
            layer_idx (int): Zero-based layer index in the drafter stack.
            dtype (jnp.dtype, optional): Compute dtype. Defaults to ``jnp.bfloat16``.
            param_dtype (jnp.dtype, optional): Parameter dtype. Defaults to ``jnp.bfloat16``.
            precision (jax.lax.PrecisionLike, optional): Matmul precision.
            rngs (spx.Rngs): PRNG container.
        """
        self.config = config
        self.layer_idx = layer_idx
        layer_type = config.layer_types[layer_idx]
        head_dim = config.global_head_dim if layer_type == "full_attention" else config.head_dim
        num_heads = config.num_attention_heads
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.layer_type = layer_type

        q_out = num_heads * head_dim
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            q_out,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(0.02),
        )
        self.o_proj = RowParallelLinear(
            q_out,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(0.02),
        )
        self.q_norm = Gemma4RMSNorm(dim=head_dim, epsilon=config.rms_norm_eps, param_dtype=param_dtype)

    @cached_property
    def rope_frequencies(self) -> Array:
        """Precompute RoPE frequencies for this layer's attention type.

        Picks the ``full_attention`` vs. ``sliding_attention`` entry in
        ``config.rope_parameters`` based on ``self.layer_type`` and supports
        the Gemma4 ``proportional`` rope variant (cos/sin halves with optional
        no-positional tail) as well as the standard partial-RoPE path.

        Returns:
            Array: Frequency tensor used by :func:`apply_basic_rope` for the
            query rotation, sized to ``max_position`` and ``head_dim``.
        """
        max_position = int(
            getattr(self.config, "granted_freq_max_position_embedding", None) or self.config.max_position_embeddings
        )
        if self.layer_type == "full_attention":
            global_params = self.config.rope_parameters.get("full_attention", {})
            base = global_params.get("rope_theta", 1_000_000.0)
            partial_rotary = global_params.get("partial_rotary_factor", 1.0)
            rope_type = global_params.get("rope_type", "default")
            head_dim = self.head_dim

            if rope_type == "proportional":
                rope_angles = int(partial_rotary * head_dim // 2)
                inv_freq_rotated = 1.0 / (base ** (jnp.arange(0, 2 * rope_angles, 2, dtype=jnp.float32) / head_dim))
                nope_angles = head_dim // 2 - rope_angles
                inv_freq = (
                    jnp.concatenate((inv_freq_rotated, jnp.zeros((nope_angles,), dtype=jnp.float32)), axis=0)
                    if nope_angles > 0
                    else inv_freq_rotated
                )
                positions = jnp.arange(max_position, dtype=jnp.float32)[:, None]
                phase = positions * inv_freq[None, :]
                return jnp.concatenate((jnp.cos(phase), jnp.sin(phase)), axis=-1)
            if partial_rotary < 1.0:
                rotated_dim = int(head_dim * partial_rotary)
                rotated_frequencies = get_frequencies(
                    head_size=rotated_dim,
                    rotary_dim=rotated_dim,
                    max_position=max_position,
                    base=base,
                    rope_scaling=None,
                    partial_rotary_factor=1.0,
                )
                rotated_cos, rotated_sin = jnp.split(rotated_frequencies, 2, axis=-1)
                pass_dim = head_dim // 2 - rotated_cos.shape[-1]
                return jnp.concatenate(
                    (
                        jnp.concatenate(
                            (rotated_cos, jnp.ones((rotated_cos.shape[0], pass_dim), dtype=rotated_cos.dtype)),
                            axis=-1,
                        ),
                        jnp.concatenate(
                            (rotated_sin, jnp.zeros((rotated_sin.shape[0], pass_dim), dtype=rotated_sin.dtype)),
                            axis=-1,
                        ),
                    ),
                    axis=-1,
                )
            return get_frequencies(
                head_size=head_dim,
                rotary_dim=head_dim,
                max_position=max_position,
                base=base,
                rope_scaling=None,
            )

        local_params = self.config.rope_parameters.get("sliding_attention", {})
        base = local_params.get("rope_theta", 10_000.0)
        return get_frequencies(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=base,
            rope_scaling=None,
        )

    def _apply_query_rope(
        self,
        query_states: Float[Array, "batch seq_len num_heads head_dim"],
        position_ids: Int[Array, "batch seq_len"],
    ) -> Float[Array, "batch seq_len num_heads head_dim"]:
        """Apply NeoX-style RoPE to the drafter query states only.

        Args:
            query_states (Array): Query tensor of shape
                ``(batch, seq_len, num_heads, head_dim)``.
            position_ids (Array): Query position indices ``(batch, seq_len)``.

        Returns:
            Array: Rotated query tensor with the same shape and dtype as
            ``query_states``. The key argument to ``apply_basic_rope`` is
            ignored because target-side K is supplied pre-rotated.
        """
        query_states, _ = apply_basic_rope(
            query=query_states,
            key=query_states,
            positions=position_ids.astype(jnp.int32),
            frequencies=self.rope_frequencies,
            rotary_dim=self.head_dim,
            is_neox_style=True,
            dtype=query_states.dtype,
        )
        return query_states

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden"],
        key_states: Float[Array, "batch kv_len num_kv head_dim"] | None = None,
        value_states: Float[Array, "batch kv_len num_kv head_dim"] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Float[Array, "batch 1 q_len kv_len"] | None = None,
    ) -> Float[Array, "batch seq_len hidden"]:
        """Apply Q-only attention.

        Args:
            hidden_states: ``(B, S, H)`` drafter hidden states.
            key_states: ``(B, KV, num_kv_heads, head_dim)`` key tensor
                from the target model's KV cache for this layer.
                Required for semantically correct speculative-decode
                behavior. When ``None``, we fall back to ``K = Q``
                (purely a shape sanity check; NOT correct).
            value_states: ``(B, KV, num_kv_heads, head_dim)`` value
                tensor from target. Same notes as ``key_states``.
            position_ids: Position IDs for RoPE on Q.
            attention_mask: Optional float-additive attention mask.

        Returns:
            ``(B, S, H)`` attention output (post ``o_proj``).
        """
        b, s, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(b, s, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        if position_ids is None:
            position_ids = jnp.arange(s, dtype=jnp.int32)[None, :]
        q = self._apply_query_rope(q, position_ids)

        if key_states is None or value_states is None:
            key_states = q
            value_states = q

        num_kv = key_states.shape[-2]
        if num_kv != self.num_heads:
            repeats = self.num_heads // num_kv
            key_states = jnp.repeat(key_states, repeats, axis=-2)
            value_states = jnp.repeat(value_states, repeats, axis=-2)

        scale = 1.0 / jnp.sqrt(jnp.float32(self.head_dim))
        q_bhsd = jnp.transpose(q, (0, 2, 1, 3)).astype(jnp.float32)
        k_bhsd = jnp.transpose(key_states, (0, 2, 1, 3)).astype(jnp.float32)
        v_bhsd = jnp.transpose(value_states, (0, 2, 1, 3)).astype(jnp.float32)
        scores = jnp.einsum("bhsd,bhtd->bhst", q_bhsd, k_bhsd) * scale
        if attention_mask is not None:
            scores = scores + attention_mask.astype(jnp.float32)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhst,bhtd->bhsd", attn, v_bhsd).astype(hidden_states.dtype)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out)


class Gemma4AssistantDecoderLayer(spx.Module):
    """One drafter decoder block.

    Layout (matches HF ``model.layers.{i}.*``):

    - ``input_layernorm`` (RMSNorm)
    - ``self_attn`` (Q-only attention)
    - ``post_attention_layernorm`` (RMSNorm)
    - ``pre_feedforward_layernorm`` (RMSNorm)
    - ``mlp`` (GeGLU)
    - ``post_feedforward_layernorm`` (RMSNorm)
    - ``layer_scalar`` (scalar learned residual gate)
    """

    def __init__(
        self,
        config: Gemma4AssistantTextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Build the decoder block."""
        self.config = config
        self.layer_idx = layer_idx
        norm = partial(Gemma4RMSNorm, dim=config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=param_dtype)
        self.input_layernorm = norm()
        self.post_attention_layernorm = norm()
        self.pre_feedforward_layernorm = norm()
        self.post_feedforward_layernorm = norm()
        self.self_attn = Gemma4AssistantQOnlyAttention(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = Gemma4AssistantMLP(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.layer_scalar = spx.Parameter(jnp.ones((1,), dtype=param_dtype))

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden"],
        key_states: Float[Array, "batch kv_len num_kv head_dim"] | None = None,
        value_states: Float[Array, "batch kv_len num_kv head_dim"] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Float[Array, "batch 1 q_len kv_len"] | None = None,
    ) -> Float[Array, "batch seq_len hidden"]:
        """One block forward.

        Args:
            hidden_states: Input ``(B, S, H)``.
            key_states / value_states: Target-derived K/V (see
                :class:`Gemma4AssistantQOnlyAttention.forward`).
            position_ids: Position IDs for Q-side RoPE.
            attention_mask: Optional float-additive mask.

        Returns:
            ``(B, S, H)`` output.
        """
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, key_states, value_states, position_ids, attention_mask)
        h = self.post_attention_layernorm(h)
        hidden_states = residual + h

        residual = hidden_states
        h = self.pre_feedforward_layernorm(hidden_states)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        hidden_states = (residual + h) * self.layer_scalar.value
        return hidden_states




@register_module(TaskType.BASE_MODULE, config=Gemma4AssistantConfig, model_type="gemma4_assistant")
class Gemma4AssistantModel(EasyDeLBaseModule):
    """Drafter base model: embed → projection → layers → final norm.

    Does NOT include the LM head or the centroid head; those live on
    :class:`Gemma4AssistantForCausalLM`.
    """

    def __init__(
        self,
        config: Gemma4AssistantConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Build the drafter base model.

        Args:
            config: Drafter top-level config.
            dtype: Compute dtype.
            param_dtype: Param dtype.
            precision: Matmul precision.
            rngs: PRNG container.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        text_cfg: Gemma4AssistantTextConfig = config.text_config
        self.embed_tokens = Embed(
            text_cfg.vocab_size,
            text_cfg.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=getattr(text_cfg, "initializer_range", 0.02)),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = nn.ModuleList(
            [
                Gemma4AssistantDecoderLayer(
                    config=text_cfg,
                    layer_idx=i,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for i in range(text_cfg.num_hidden_layers)
            ]
        )
        self.norm = Gemma4RMSNorm(
            dim=text_cfg.hidden_size,
            epsilon=text_cfg.rms_norm_eps,
            param_dtype=param_dtype,
        )

    def get_embedding(self) -> Embed:
        """Return the drafter's token embedding (doubles as LM head)."""
        return self.embed_tokens

    def forward(
        self,
        backbone_hidden_states: Float[Array, "batch seq_len backbone_hidden"],
        target_token_embeds: Float[Array, "batch seq_len backbone_hidden"],
        pre_projection: ColumnParallelLinear,
        target_key_value_pairs: list[tuple[Array, Array] | None] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Float[Array, "batch 1 q_len kv_len"] | None = None,
    ) -> BaseModelOutput:
        """Run the drafter trunk.

        Args:
            backbone_hidden_states: ``(B, S, backbone_hidden)`` final
                hidden states from the TARGET model at the current
                step.
            target_token_embeds: ``(B, S, backbone_hidden)`` target's
                embedding of the most recent input tokens. The
                drafter's ``pre_projection`` fuses these with
                ``backbone_hidden_states`` via concat.
            pre_projection: The drafter's
                ``Linear(2 * backbone_hidden -> hidden)`` projection
                module (lives on the wrapping ForCausalLM; passed in
                so this trunk class doesn't own backbone-dependent
                params).
            target_key_value_pairs: Per-drafter-layer ``(K, V)``
                tensors gathered from the target model's KV cache.
                ``None`` (or any per-layer ``None``) falls back to
                self-K/V — only correct shape, not correct semantics.
                **TODO: full speculative-decode controller; see
                module docstring.**
            position_ids: Position IDs for Q-side RoPE.
            attention_mask: Optional float-additive attention mask.

        Returns:
            :class:`BaseModelOutput` with ``last_hidden_state``
            ``(B, S, hidden)`` (drafter-dim, pre ``post_projection``).
        """
        combined = jnp.concatenate([target_token_embeds, backbone_hidden_states], axis=-1)
        h = pre_projection(combined)
        h = apply_logical_sharding(
            h,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.runtime_sharding_resolver,
        )
        for i, layer in enumerate(self.layers):
            kv = target_key_value_pairs[i] if target_key_value_pairs else None
            k_in, v_in = kv if kv is not None else (None, None)
            h = layer(h, k_in, v_in, position_ids, attention_mask)
            h = checkpoint_name(h, f"gemma4_assistant_layer_{i}")
        h = self.norm(h)
        return BaseModelOutput(last_hidden_state=h)


@register_module(TaskType.CAUSAL_LM, config=Gemma4AssistantConfig, model_type="gemma4_assistant")
class Gemma4AssistantForCausalLM(EasyDeLBaseModule):
    """Standalone Gemma4 Assistant (drafter) with centroid LM head.

    Owns the backbone-dimension projections (``pre_projection``,
    ``post_projection``) and the centroid output head
    (``masked_embedding``). The drafter trunk lives under
    ``self.model``.

    Speculative-decoding driver should:

    1. Run the target model on the prompt, capturing per-layer K/V
       at the layers the drafter will read from.
    2. Compute ``target_token_embeds`` via
       ``target.get_input_embeddings()(input_ids) * sqrt(backbone_hidden_size)``.
    3. Call :meth:`forward` with
       ``target_key_value_pairs=[...]``.
    4. Use returned ``logits`` (or ``top_logits``/``top_token_ids``)
       to sample / argmax draft tokens.
    5. Feed ``backbone_hidden_state`` back into the next draft step's
       ``backbone_hidden_states`` argument.

    The cross-model KV-sharing controller is NOT implemented in this
    file — it requires coordinated forward passes between the target
    and the drafter. The :meth:`forward` API is ready to accept the
    K/V tensors; see the follow-up esurge-integration PR for the
    controller logic.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "gemma4_assistant"
    _config_class = Gemma4AssistantConfig
    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: Gemma4AssistantConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Build the standalone drafter model.

        Args:
            config: Drafter top-level config.
            dtype: Compute dtype.
            param_dtype: Param dtype.
            precision: Matmul precision.
            rngs: PRNG container.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.config = config
        self.model = Gemma4AssistantModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        text_cfg = config.text_config
        self.pre_projection = ColumnParallelLinear(
            2 * config.backbone_hidden_size,
            text_cfg.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(0.02),
        )
        self.post_projection = ColumnParallelLinear(
            text_cfg.hidden_size,
            config.backbone_hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(0.02),
        )
        if config.use_ordered_embeddings and int(config.num_centroids) > 0:
            self.masked_embedding = Gemma4AssistantCentroidHead(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.masked_embedding = None

    def get_input_embeddings(self) -> Embed:
        """Return the tied embedding/LM-head matrix."""
        return self.model.get_embedding()

    def forward(
        self,
        backbone_hidden_states: Float[Array, "batch seq_len backbone_hidden"],
        target_token_embeds: Float[Array, "batch seq_len backbone_hidden"],
        target_key_value_pairs: list[tuple[Array, Array] | None] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Float[Array, "batch 1 q_len kv_len"] | None = None,
        return_dense_logits: bool = False,
    ) -> Gemma4AssistantOutput:
        """Run the drafter end-to-end.

        Args:
            backbone_hidden_states: ``(B, S, backbone_hidden)`` final
                hidden states from the target model.
            target_token_embeds: ``(B, S, backbone_hidden)``
                target-scaled token embeddings.
            target_key_value_pairs: Per-layer K/V from the target's
                KV cache. ``None`` → self-K/V fallback (NOT
                semantically correct; for shape sanity only).
            position_ids: Position IDs for Q-side RoPE.
            attention_mask: Optional float-additive mask.
            return_dense_logits: Whether the centroid head should
                also produce a dense ``(B, S, V)`` logits tensor with
                ``-inf`` outside selected centroids.

        Returns:
            :class:`Gemma4AssistantOutput` with drafter hidden,
            backbone-projected hidden, and centroid-head outputs.
        """
        outputs = self.model(
            backbone_hidden_states=backbone_hidden_states,
            target_token_embeds=target_token_embeds,
            pre_projection=self.pre_projection,
            target_key_value_pairs=target_key_value_pairs,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        h = outputs.last_hidden_state
        next_backbone = self.post_projection(h)

        if self.masked_embedding is not None:
            embed_w = self.model.embed_tokens.weight.value
            top_logits, top_ids, dense = self.masked_embedding(
                h,
                embed_w,
                return_dense_logits=return_dense_logits,
            )
        else:
            embed_w = self.model.embed_tokens.weight.value
            dense = jnp.einsum("bsh,vh->bsv", h.astype(jnp.float32), embed_w.astype(jnp.float32))
            top_logits = None
            top_ids = None
        return Gemma4AssistantOutput(
            last_hidden_state=h,
            backbone_hidden_state=next_backbone,
            logits=dense,
            top_logits=top_logits,
            top_token_ids=top_ids,
        )


__all__ = [
    "Gemma4AssistantCentroidHead",
    "Gemma4AssistantForCausalLM",
    "Gemma4AssistantModel",
    "Gemma4AssistantOutput",
]
