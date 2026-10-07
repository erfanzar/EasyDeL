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

"""DeepSeek-V4 DSpark draft model (``DeepSeek-V4-Flash-0731`` ``mtp.N.*``; reference ``inference/model.py``).

The released V4 checkpoint ships a DSpark speculative-decoding module (``DSparkBlock`` x ``n_mtp_layers``, block size
``dspark_block_size``) even though its ``config.json`` keeps ``num_nextn_predict_layers: 1``; the stage count is the
number of ``mtp.N`` prefixes in the checkpoint (``n_mtp_layers`` in ``inference/config.json``).

Differences from the V4.1 DSpark (:mod:`easydel.modules.deepseek_v41.modeling_deepseek_v41_dspark`):

- stages are plain V4 blocks: every sublayer reads the streams through its own hyper-connection ``pre`` mix (no delayed
  ``pre`` carried between stages); the MoE is the backbone's (learned top-k, the backbone expert counts);
- the draft query gets the unweighted per-head RMS norm of V4 attention after ``wq_b`` (``draft_q_head_norm_eps``);
- the last stage collapses the streams with its own learned head (``hc_head_{fn,base,scale}``) instead of a ``pre``
  mix, and its Markov head stores ``markov_w1`` (embedding) / ``markov_w2`` (projection);
- the backbone's main hidden of target layer ``i`` is the stream mean of that layer's OUTPUT (V4.1: its input); see
  ``DeepseekV4Model.forward(output_dspark_hidden=True)``.

MTP stages have compress ratio 0, so they use the plain (``"main"``) rope without YaRN. As in the V4.1 port, the fp8
activation fake-quantization of the window K/V is not emulated and drafting is greedy only.
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
from easydel.layers.residual import manifold_residual_write
from easydel.modules._drafting import AcceptRatePredictor, VanillaMarkovHead, WindowRingDraftAttentionMixin
from easydel.modules._dspark_drafter import DSparkDrafter, DSparkState

from .deepseek_v4_configuration import DeepseekV4Config
from .modeling_deepseek_v4 import (
    _MASK_MIN,
    DeepseekV4Attention,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4RMSNorm,
    DeepseekV4SparseMoeBlock,
    _apply_interleaved_rope,
    _rope_cos_sin,
)

EmbedFn = tp.Callable[[Array], Array]
LmHeadFn = tp.Callable[[Array], Array]


def dspark_num_stages(config: DeepseekV4Config) -> int:
    """Number of DSpark stages: ``dspark_num_stages`` / ``n_mtp_layers`` when set, else ``num_nextn_predict_layers``."""
    for name in ("dspark_num_stages", "n_mtp_layers"):
        value = getattr(config, name, None)
        if value:
            return int(value)
    return int(config.num_nextn_predict_layers)


def _draft_config(config: DeepseekV4Config, num_stages: int) -> DeepseekV4Config:
    """Config whose layer schedules cover the stage ids ``num_hidden_layers + s`` (window-only attention, learned MoE)."""
    total = int(config.num_hidden_layers) + int(num_stages)
    draft = copy.deepcopy(config)
    layer_types = list(config.layer_types)[: config.num_hidden_layers]
    mlp_layer_types = list(config.mlp_layer_types)[: config.num_hidden_layers]
    draft.layer_types = layer_types + ["sliding_attention"] * (total - len(layer_types))
    draft.mlp_layer_types = mlp_layer_types + ["moe"] * (total - len(mlp_layer_types))
    return draft


class DeepseekV4DSparkAttention(WindowRingDraftAttentionMixin, DeepseekV4Attention):
    """V4 window-ring draft attention: shared draft mechanics plus V4's unweighted per-head query norm."""

    draft_rope_cos_sin = staticmethod(_rope_cos_sin)
    draft_apply_rope = staticmethod(_apply_interleaved_rope)
    draft_mask_min = _MASK_MIN
    rope_label = "main"

    def __init__(self, config: DeepseekV4Config, layer_idx: int, **kwargs: tp.Any):
        super().__init__(config, layer_idx=layer_idx, **kwargs)
        if self.layer_type != "sliding_attention":
            raise ValueError(f"DSpark stage layer {layer_idx} must be window-only, got {self.layer_type!r}.")
        self.draft_q_head_norm_eps = float(config.rms_norm_eps)


class DeepseekV4MarkovHead(VanillaMarkovHead):
    """V4 checkpoint-compatible fp32 Markov head (``markov_w1`` embedding, ``markov_w2`` projection)."""

    def __init__(self, config: DeepseekV4Config, precision: jax.lax.PrecisionLike = None, *, rngs: spx.Rngs):
        super().__init__(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=precision,
            rngs=rngs,
            embedding_name="markov_w1",
            projection_name="markov_w2",
            fp32=True,
            embedding_init=default_embed_init,
        )


class DeepseekV4ConfidenceHead(AcceptRatePredictor):
    """V4 confidence head: fp32 ``proj`` over ``concat([hidden, markov_embed])``."""

    def __init__(self, config: DeepseekV4Config, precision: jax.lax.PrecisionLike = None, *, rngs: spx.Rngs):
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
        """Score concatenated hidden and Markov features."""
        features = jnp.concatenate([hidden.astype(jnp.float32), markov_embed.astype(jnp.float32)], axis=-1)
        return super().forward(features)


class DeepseekV4DSparkStage(spx.Module):
    """One DSpark stage (reference ``DSparkBlock``): a V4 block whose attention drafts against a main-K/V ring.

    Stage 0 also owns ``main_proj`` / ``main_norm``; the last stage owns ``hc_head``, ``norm``, ``markov_head`` and
    ``confidence_head``.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
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
        self.self_attn = DeepseekV4DSparkAttention(config, layer_idx, **layer_kwargs)
        self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx, **layer_kwargs)
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
            self.hc_head = DeepseekV4HyperHead(config, **layer_kwargs)
            self.norm = DeepseekV4RMSNorm(**norm_kwargs, rngs=rngs)
            self.markov_head = DeepseekV4MarkovHead(config, precision, rngs=rngs)
            self.confidence_head = DeepseekV4ConfidenceHead(config, precision, rngs=rngs)

    def forward(
        self,
        hidden_streams: Float[Array, "batch block hc hidden"],
        main_x: Float[Array, "batch 1 hidden"],
        ring: Array,
        position: Int[Array, "batch"],
    ) -> tuple[Array, Array]:
        """Return ``(streams, ring)`` (reference ``Block.forward`` at ``start_pos > 0``)."""
        post, comb, collapsed = self.attn_hc(hidden_streams)
        attn_output, ring = self.self_attn.draft_attend(self.input_layernorm(collapsed), main_x, ring, position)
        hidden_streams = manifold_residual_write(hidden_streams, attn_output, post, comb, precision=self.precision)
        post, comb, collapsed = self.ffn_hc(hidden_streams)
        mlp_output, _ = self.mlp(self.post_attention_layernorm(collapsed))
        hidden_streams = manifold_residual_write(hidden_streams, mlp_output, post, comb, precision=self.precision)
        return hidden_streams, ring


class DeepseekV4DSpark(spx.Module):
    """DeepSeek-V4 DSpark draft model (all stages, greedy drafting).

    Build under ``with config.mesh:``. Usage: ``state = model.prefill(main_hidden)`` once per prompt, then for every
    step ``draft_ids, logits, confidence, state = model.draft(token, main_hidden_step, state, embed, lm_head)``. The
    interface equals :class:`~easydel.modules.deepseek_v41.DeepseekV41DSpark`, so the shared
    :class:`~easydel.modules._dspark_drafter.DSparkDrafter` serves both.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        num_stages: int | None = None,
    ):
        """Initialize the draft stages.

        Args:
            config: Backbone V4 config with ``dspark_block_size > 0``, ``dspark_target_layer_ids``,
                ``dspark_noise_token_id`` and ``dspark_markov_rank`` (kept as plain attributes).
            dtype: Activation dtype.
            param_dtype: Backbone-style parameter storage dtype (markov / confidence heads are always fp32).
            precision: Matmul precision.
            rngs: Random number generators.
            num_stages: Number of ``mtp.N`` stages; defaults to :func:`dspark_num_stages`.

        Raises:
            ValueError: If DSpark is not configured.
        """
        if int(getattr(config, "dspark_block_size", 0) or 0) <= 0 or not getattr(
            config, "dspark_target_layer_ids", None
        ):
            raise ValueError("DSpark needs dspark_block_size > 0 and non-empty dspark_target_layer_ids.")
        stages = int(num_stages) if num_stages is not None else dspark_num_stages(config)
        if stages <= 0:
            raise ValueError("DSpark needs at least one stage.")
        self.config = config
        self.precision = precision
        self.block_size = int(config.dspark_block_size)
        self.noise_token_id = int(config.dspark_noise_token_id)
        self.hc_mult = int(config.hc_mult)
        self.num_stages = stages
        draft_config = _draft_config(config, stages)
        self.stages = nn.ModuleList([])
        for stage_idx in range(stages):
            self.stages.append(
                DeepseekV4DSparkStage(
                    draft_config,
                    config.num_hidden_layers + stage_idx,
                    stage_idx,
                    stages,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
            )

    def _main_x(self, main_hidden: Array) -> Array:
        first = self.stages[0]
        return first.main_norm(first.main_proj(main_hidden))

    def prefill(self, main_hidden: Float[Array, "batch seq targets_hidden"]) -> DSparkState:
        """Seed every stage's window ring from the prompt's main hidden states (reference ``start_pos == 0``).

        Returns:
            State whose ``position`` is ``S`` (the position of the first main hidden state :meth:`draft` receives).
        """
        main_x = self._main_x(main_hidden)
        rings = tuple(stage.self_attn.seed_ring(main_x) for stage in self.stages)
        position = jnp.full((main_hidden.shape[0],), main_hidden.shape[1], dtype=jnp.int32)
        return DSparkState(rings, position)

    def draft(
        self,
        token_ids: Int[Array, "batch"],
        main_hidden_step: Float[Array, "batch 1 targets_hidden"],
        state: DSparkState,
        embed: EmbedFn,
        lm_head: LmHeadFn,
    ) -> tuple[Array, Array, Array, DSparkState]:
        """Draft one block greedily (reference ``forward_spec`` at ``start_pos > 0``); jittable.

        Args:
            token_ids: The last accepted token ``[B]`` (position ``state.position + 1``).
            main_hidden_step: Main hidden state of position ``state.position`` ``[B, 1, len(targets) * D]``.
            state: State from :meth:`prefill` or a previous :meth:`draft`.
            embed: Backbone embedding lookup (shared with the backbone).
            lm_head: Backbone LM head ``hidden -> logits`` (shared with the backbone).

        Returns:
            ``(draft_ids [B, block + 1] int32, logits [B, block, V] fp32, confidence [B, block] fp32, new_state)``.
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
        rings = []
        for stage, ring in zip(stages, state.window_kv, strict=True):
            streams, ring = stage(streams, main_x, ring, position)
            rings.append(ring)

        last = stages[-1]
        hidden = last.hc_head(streams)
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
        return out_ids, logits, confidence, DSparkState(tuple(rings), position + 1)


class DeepseekV4DSparkDrafter(DSparkDrafter):
    """eSurge drafter for the V4 DSpark draft model (the shared DSpark drafter)."""


@register_drafter("deepseek_v4_dspark", "v4_dspark")
def _build_deepseek_v4_dspark_drafter(
    target_model: tp.Any,
    *,
    num_draft_tokens: int = 1,
    assistant_model: tp.Any | None = None,
    target_embed_module: tp.Any | None = None,
    layer_mapping: list[int] | None = None,
    target_config: tp.Any | None = None,
    drafter_model: tp.Any | None = None,
    **kwargs: tp.Any,
) -> DeepseekV4DSparkDrafter:
    """Registry builder: ``model.drafter(method="deepseek_v4_dspark", drafter_model=<DeepseekV4DSpark>)``.

    Raises:
        ValueError: If no DSpark module is supplied.
    """
    del target_embed_module, layer_mapping, target_config
    dspark = drafter_model if drafter_model is not None else assistant_model
    if dspark is None:
        raise ValueError("method='deepseek_v4_dspark' requires drafter_model=<DeepseekV4DSpark>.")
    return DeepseekV4DSparkDrafter(target_model=target_model, dspark=dspark, num_draft_tokens=num_draft_tokens, **kwargs)


__all__ = [
    "DeepseekV4ConfidenceHead",
    "DeepseekV4DSpark",
    "DeepseekV4DSparkAttention",
    "DeepseekV4DSparkDrafter",
    "DeepseekV4DSparkStage",
    "DeepseekV4MarkovHead",
    "dspark_num_stages",
]
