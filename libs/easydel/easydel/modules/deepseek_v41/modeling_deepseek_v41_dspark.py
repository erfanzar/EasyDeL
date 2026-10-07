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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V4.1 DSpark multi-token-prediction draft model (reference ``DSparkBlock`` / ``Transformer.forward_spec``).

The draft model is ``num_nextn_predict_layers`` stages (checkpoint prefix ``mtp.N``). Every stage is a backbone-style
block (delayed-``pre`` hyper-connections, window-only attention, learned top-k MoE with the ``dspark_*`` expert counts)
whose attention differs from the backbone in what it reads:

- the *main* hidden states of the backbone's ``dspark_target_layer_ids`` (hook below) are projected by stage 0
  (``main_proj`` then ``main_norm``) into ``main_x``; every stage turns ``main_x`` into window K/V with its own
  ``kv_proj`` / ``kv_norm`` and writes them into its sliding-window ring (token ``p`` at slot ``p % window``);
- a draft block of ``dspark_block_size`` tokens (the last accepted token followed by noise tokens) is the query side:
  its queries attend to the whole valid ring plus the block's own K/V, bidirectionally, at positions
  ``p + 1 .. p + block`` where ``p`` is the position of the main hidden state;
- after the last stage, ``norm`` and the (tied, caller supplied) LM head produce block logits, a rank-``r`` Markov head
  adds a bias that depends on the previously drafted token, tokens are drafted sequentially (greedy) and a fp32
  confidence head scores each position.

The embedding and LM head are shared with the backbone: :meth:`DeepseekV41DSpark.draft` takes them as callables.

Backbone hook required (NOT implemented here, see ``modeling_deepseek_v41.py``): while looping over the decoder
layers, immediately before layer ``i`` runs and AFTER that layer's engram has been applied to ``hidden_streams``,
for every ``i`` in ``config.dspark_target_layer_ids`` append ``hidden_streams.mean(axis=2)`` (``[B, S, D]``, mean over
the ``hc_mult`` streams, i.e. the attention input of the target layer, not its output); after the loop concatenate the
captured tensors on the last axis in ascending layer order -> ``main_hidden [B, S, len(targets) * D]``.

Deviations from the reference: no fp8/fp4 activation fake-quantization of the window K/V; only greedy drafting
(temperature 0); markov / confidence head parameters are always fp32; ``mtp.N.ffn.gate.bias_vl`` is not owned
(drafts are text, the reference never selects it for them); all rows of a batch advance by one position per step
(per-row positions are supported, left padding is not).
"""

from __future__ import annotations

import copy
import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx
from jaxtyping import Array, Float, Int
from spectrax import nn

from easydel.infra.spec_decode import register_drafter
from easydel.layers import ColumnParallelLinear, ParallelLinear
from easydel.layers.embeddings._embeddings import default_embed_init
from easydel.layers.residual import manifold_collapse, manifold_residual_write
from easydel.modules._drafting import AcceptRatePredictor, VanillaMarkovHead, WindowRingDraftAttentionMixin
from easydel.modules._dspark_drafter import DSparkDrafter, DSparkState
from easydel.modules.deepseek_v4.modeling_deepseek_v4 import (
    _MASK_MIN,
    DeepseekV4HyperConnection,
    DeepseekV4RMSNorm,
    _apply_interleaved_rope,
    _rope_cos_sin,
)

from .deepseek_v41_configuration import DeepseekV41Config
from .modeling_deepseek_v41 import DeepseekV41Attention, DeepseekV41SparseMoeBlock

EmbedFn = tp.Callable[[Array], Array]
LmHeadFn = tp.Callable[[Array], Array]


#: The shared DSpark draft state (one window ring per stage + next main position), kept under its V4.1 name.
DeepseekV41DSparkState = DSparkState


def _draft_config(config: DeepseekV41Config) -> DeepseekV41Config:
    """Config whose MoE uses the DSpark expert counts (reference ``ModelArgs.get_moe_config`` for MTP layers)."""
    experts = int(config.dspark_n_routed_experts or config.n_routed_experts)
    top_k = int(config.dspark_num_experts_per_tok or config.num_experts_per_tok)
    if experts == config.n_routed_experts and top_k == config.num_experts_per_tok:
        return config
    draft = copy.deepcopy(config)
    draft.n_routed_experts = experts
    draft.num_experts_per_tok = top_k
    return draft


class DeepseekV41DSparkAttention(WindowRingDraftAttentionMixin, DeepseekV41Attention):
    """V4.1 window-ring attention using shared draft attention mechanics."""

    draft_rope_cos_sin = staticmethod(_rope_cos_sin)
    draft_apply_rope = staticmethod(_apply_interleaved_rope)
    draft_mask_min = _MASK_MIN


class DeepseekV41MarkovHead(VanillaMarkovHead):
    """V4.1 checkpoint-compatible fp32 Markov head backed by the shared implementation."""

    def __init__(self, config: DeepseekV41Config, precision: jax.lax.PrecisionLike = None, *, rngs: spx.Rngs):
        super().__init__(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=precision,
            rngs=rngs,
            embedding_name="embed",
            projection_name="head",
            fp32=True,
            embedding_init=default_embed_init,
        )


class DeepseekV41ConfidenceHead(AcceptRatePredictor):
    """V4.1 checkpoint-compatible confidence head backed by the shared predictor."""

    def __init__(self, config: DeepseekV41Config, precision: jax.lax.PrecisionLike = None, *, rngs: spx.Rngs):
        super().__init__(
            config.hidden_size + int(config.dspark_markov_rank),
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=precision,
            rngs=rngs,
            use_bias=False,
            fp32=True,
            projection_cls=ParallelLinear,
        )

    def forward(self, hidden: Array, markov_embed: Array) -> Array:
        """Score concatenated hidden and Markov features, retaining the V4.1 call signature."""
        features = jnp.concatenate([hidden.astype(jnp.float32), markov_embed.astype(jnp.float32)], axis=-1)
        # The shared predictor already squeezes the scalar output axis.
        return super().forward(features)


class DeepseekV41DSparkStage(spx.Module):
    """One DSpark stage (reference ``DSparkBlock``): attention + MoE between delayed-``pre`` hyper-connections.

    Stage 0 also owns ``main_proj`` / ``main_norm``; the last stage owns ``norm``, ``markov_head`` and
    ``confidence_head``.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        layer_idx: int,
        stage_idx: int,
        num_stages: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        self.precision = precision
        self.stage_idx = stage_idx
        layer_kwargs = dict(dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
        norm_kwargs = dict(dim=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype)
        self.self_attn = DeepseekV41DSparkAttention(config, layer_idx, **layer_kwargs)
        self.mlp = DeepseekV41SparseMoeBlock(config, layer_idx, **layer_kwargs)
        self.input_layernorm = DeepseekV4RMSNorm(**norm_kwargs, rngs=rngs)
        self.post_attention_layernorm = DeepseekV4RMSNorm(**norm_kwargs, rngs=rngs)
        self.attn_hc = DeepseekV4HyperConnection(config, **layer_kwargs)
        self.ffn_hc = DeepseekV4HyperConnection(config, **layer_kwargs)
        if stage_idx == 0:
            self.main_proj = ColumnParallelLinear(
                len(config.dspark_target_layer_ids) * config.hidden_size,
                config.hidden_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
                rngs=rngs,
            )
            self.main_norm = DeepseekV4RMSNorm(**norm_kwargs, rngs=rngs)
        if stage_idx == num_stages - 1:
            self.norm = DeepseekV4RMSNorm(**norm_kwargs, rngs=rngs)
            self.markov_head = DeepseekV41MarkovHead(config, precision, rngs=rngs)
            self.confidence_head = DeepseekV41ConfidenceHead(config, precision, rngs=rngs)

    def forward(
        self,
        hidden_streams: Float[Array, "batch block hc hidden"],
        pre_mix: Float[Array, "batch block hc"],
        main_x: Float[Array, "batch 1 hidden"],
        ring: Array,
        position: Int[Array, "batch"],
    ) -> tuple[Array, Array, Array]:
        """Return ``(streams, ffn_pre, ring)``; ``ffn_pre`` is the next stage's (or the head's) read mix."""
        attn_pre, attn_post, attn_comb = self.attn_hc.coefficients(hidden_streams)
        attn_input = self.input_layernorm(manifold_collapse(pre_mix, hidden_streams))
        attn_output, ring = self.self_attn.draft_attend(attn_input, main_x, ring, position)
        hidden_streams = manifold_residual_write(
            hidden_streams, attn_output, attn_post, attn_comb, precision=self.precision
        )
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc.coefficients(hidden_streams)
        mlp_input = self.post_attention_layernorm(manifold_collapse(attn_pre, hidden_streams))
        mlp_output, _ = self.mlp(mlp_input)
        hidden_streams = manifold_residual_write(
            hidden_streams, mlp_output, ffn_post, ffn_comb, precision=self.precision
        )
        return hidden_streams, ffn_pre, ring


class DeepseekV41DSpark(spx.Module):
    """DeepSeek-V4.1 DSpark draft model (all ``num_nextn_predict_layers`` stages, greedy drafting).

    Build under ``with config.mesh:``. Usage: ``state = model.prefill(main_hidden)`` once per prompt, then for every
    step ``draft_ids, logits, confidence, state = model.draft(token, main_hidden_step, state, embed, lm_head)``.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the draft stages.

        Args:
            config: V4.1 config with ``dspark_block_size > 0``, ``dspark_target_layer_ids`` and MTP stages.
            dtype: Activation dtype.
            param_dtype: Backbone-style parameter storage dtype (markov / confidence heads are always fp32).
            precision: Matmul precision.
            rngs: Random number generators.

        Raises:
            ValueError: If DSpark is not configured.
            NotImplementedError: If ``gate_temp != 1``.
        """
        if int(config.dspark_block_size) <= 0 or not config.dspark_target_layer_ids:
            raise ValueError("DSpark needs dspark_block_size > 0 and non-empty dspark_target_layer_ids.")
        if config.num_nextn_predict_layers <= 0:
            raise ValueError("DSpark needs num_nextn_predict_layers > 0.")
        if float(config.gate_temp) != 1.0:
            raise NotImplementedError("DeepSeek-V4.1 router gate_temp != 1.0 is not supported by the shared V4 router.")
        self.config = config
        self.precision = precision
        self.block_size = int(config.dspark_block_size)
        self.noise_token_id = int(config.dspark_noise_token_id)
        self.hc_mult = int(config.hc_mult)
        self.num_stages = int(config.num_nextn_predict_layers)
        draft_config = _draft_config(config)
        self.stages = nn.ModuleList([])
        for stage_idx in range(self.num_stages):
            self.stages.append(
                DeepseekV41DSparkStage(
                    draft_config,
                    config.num_hidden_layers + stage_idx,
                    stage_idx,
                    self.num_stages,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
            )

    def _main_x(self, main_hidden: Array) -> Array:
        first = self.stages[0]
        return first.main_norm(first.main_proj(main_hidden))

    def prefill(self, main_hidden: Float[Array, "batch seq targets_hidden"]) -> DeepseekV41DSparkState:
        """Seed every stage's window ring from the prompt's main hidden states (reference ``start_pos == 0``).

        Args:
            main_hidden: Concatenated target-layer hidden states ``[B, S, len(targets) * D]`` of positions ``0..S-1``.

        Returns:
            State whose ``position`` is ``S`` (the position of the first main hidden state :meth:`draft` receives).
        """
        main_x = self._main_x(main_hidden)
        rings = tuple(stage.self_attn.seed_ring(main_x) for stage in self.stages)
        position = jnp.full((main_hidden.shape[0],), main_hidden.shape[1], dtype=jnp.int32)
        return DeepseekV41DSparkState(rings, position)

    def draft(
        self,
        token_ids: Int[Array, "batch"],
        main_hidden_step: Float[Array, "batch 1 targets_hidden"],
        state: DeepseekV41DSparkState,
        embed: EmbedFn,
        lm_head: LmHeadFn,
    ) -> tuple[Array, Array, Array, DeepseekV41DSparkState]:
        """Draft one block greedily (reference ``forward_spec`` at ``start_pos > 0``); jittable.

        Args:
            token_ids: The last accepted token ``[B]`` (position ``state.position + 1``).
            main_hidden_step: Main hidden state of position ``state.position`` ``[B, 1, len(targets) * D]``.
            state: State from :meth:`prefill` or a previous :meth:`draft`.
            embed: Backbone embedding lookup ``ids [..] -> [.., D]`` (shared with the backbone).
            lm_head: Backbone LM head ``hidden [.., D] -> logits [.., V]`` (shared with the backbone).

        Returns:
            ``(draft_ids [B, block + 1] int32, logits [B, block, V] fp32, confidence [B, block] fp32, new_state)``;
            ``draft_ids[:, 0]`` is ``token_ids``, ``draft_ids[:, i + 1]`` the greedy token of ``logits[:, i]`` (which
            already includes the markov bias of ``draft_ids[:, i]``). ``new_state.position`` is ``position + 1``.
        """
        stages = list(self.stages)
        batch = token_ids.shape[0]
        block = self.block_size
        token_ids = token_ids.astype(jnp.int32)
        position = state.position.astype(jnp.int32)

        main_x = self._main_x(main_hidden_step)
        block_ids = jnp.full((batch, block), self.noise_token_id, dtype=jnp.int32).at[:, 0].set(token_ids)
        embedded = embed(block_ids)
        streams = jnp.broadcast_to(embedded[:, :, None, :], (batch, block, self.hc_mult, embedded.shape[-1]))
        pre_mix = jnp.zeros((batch, block, self.hc_mult), jnp.float32).at[..., 0].set(1.0)
        rings = []
        for stage, ring in zip(stages, state.window_kv, strict=True):
            streams, pre_mix, ring = stage(streams, pre_mix, main_x, ring, position)
            rings.append(ring)

        last = stages[-1]
        hidden = manifold_collapse(pre_mix, streams)
        logits = lm_head(last.norm(hidden)).astype(jnp.float32)
        markov = last.markov_head
        rank = int(self.config.dspark_markov_rank)

        def step(i, carry):
            out_ids, out_logits, embeds = carry
            bias, markov_embed = markov(out_ids[:, i])
            current = out_logits[:, i] + bias
            out_logits = out_logits.at[:, i].set(current)
            out_ids = out_ids.at[:, i + 1].set(jnp.argmax(current, axis=-1).astype(jnp.int32))
            return out_ids, out_logits, embeds.at[:, i].set(markov_embed)

        init = (
            jnp.zeros((batch, block + 1), jnp.int32).at[:, 0].set(token_ids),
            logits,
            jnp.zeros((batch, block, rank), jnp.float32),
        )
        out_ids, logits, markov_embeds = jax.lax.fori_loop(0, block, step, init)
        confidence = last.confidence_head(hidden, markov_embeds)
        return out_ids, logits, confidence, DeepseekV41DSparkState(tuple(rings), position + 1)


class DeepseekV41DSparkDrafter(DSparkDrafter):
    """eSurge drafter for the V4.1 DSpark draft model (the shared :class:`~easydel.modules._dspark_drafter.DSparkDrafter`)."""


@register_drafter("deepseek_v41_dspark", "v41_dspark")
def _build_deepseek_v41_dspark_drafter(
    target_model: tp.Any,
    *,
    num_draft_tokens: int = 1,
    assistant_model: tp.Any | None = None,
    target_embed_module: tp.Any | None = None,
    layer_mapping: list[int] | None = None,
    target_config: tp.Any | None = None,
    drafter_model: tp.Any | None = None,
    **kwargs: tp.Any,
) -> DeepseekV41DSparkDrafter:
    """Registry builder: ``model.drafter(method="deepseek_v41_dspark", drafter_model=<DeepseekV41DSpark>)``.

    The DSpark module may also arrive in the ``assistant_model`` slot (``eSurgeDrafterConfig.assistant_model``).

    Raises:
        ValueError: If no DSpark module is supplied.
    """
    del target_embed_module, layer_mapping, target_config
    dspark = drafter_model if drafter_model is not None else assistant_model
    if dspark is None:
        raise ValueError("method='deepseek_v41_dspark' requires drafter_model=<DeepseekV41DSpark>.")
    return DeepseekV41DSparkDrafter(
        target_model=target_model, dspark=dspark, num_draft_tokens=num_draft_tokens, **kwargs
    )


__all__ = [
    "DeepseekV41ConfidenceHead",
    "DeepseekV41DSpark",
    "DeepseekV41DSparkAttention",
    "DeepseekV41DSparkDrafter",
    "DeepseekV41DSparkStage",
    "DeepseekV41DSparkState",
    "DeepseekV41MarkovHead",
]
