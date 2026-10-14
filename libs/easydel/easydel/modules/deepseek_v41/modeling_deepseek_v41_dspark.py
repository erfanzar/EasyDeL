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

The draft model is ``num_nextn_predict_layers`` stages (checkpoint prefix ``mtp.N``) on the shared
:class:`~easydel.modules.dspark.DSparkStageStack` orchestration. Every stage is a backbone-style block
(delayed-``pre`` hyper-connections, window-only attention, learned top-k MoE with the ``dspark_*`` expert counts)
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

What is V4.1-specific is the delayed ``pre`` mix: each stage's FFN read mix is carried into the next stage's attention
and the final collapse uses the last stage's.

Backbone hook required (see ``modeling_deepseek_v41.py``): while looping over the decoder layers, immediately before
layer ``i`` runs and AFTER that layer's engram has been applied to ``hidden_streams``, for every ``i`` in
``config.dspark_target_layer_ids`` append ``hidden_streams.mean(axis=2)`` (the attention input of the target layer, not
its output); after the loop concatenate the captured tensors on the last axis in ascending layer order.

Deviations from the reference: no fp8/fp4 activation fake-quantization of the window K/V; only greedy drafting
(temperature 0); markov / confidence head parameters are always fp32; ``mtp.N.ffn.gate.bias_vl`` is not owned
(drafts are text, the reference never selects it for them); all rows of a batch advance by one position per step
(per-row positions are supported, left padding is not).
"""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import spectrax as spx
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Float, Int

from easydel.layers import RMSNorm
from easydel.layers.attention import WindowRingDraftAttentionMixin
from easydel.layers.residual import (
    ManifoldHyperConnection,
    ManifoldHyperConnectionConfig,
    manifold_collapse,
    manifold_residual_write,
)
from easydel.modules.dspark import (
    DSparkConfidenceHead,
    DSparkDrafter,
    DSparkMarkovHead,
    DSparkStageStack,
    DSparkState,
    dspark_main_projection,
    register_dspark_drafter,
)

from .deepseek_v41_configuration import DeepseekV41Config
from .modeling_deepseek_v41 import DEEPSEEK_V41_MTP_KEY_RULES, DeepseekV41Attention, DeepseekV41SparseMoeBlock

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
        self.input_layernorm = RMSNorm(**norm_kwargs, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(**norm_kwargs, rngs=rngs)
        hc_config = ManifoldHyperConnectionConfig.from_model_config(config)
        self.attn_hc = ManifoldHyperConnection(hc_config, **layer_kwargs, mesh_source=config)
        self.ffn_hc = ManifoldHyperConnection(hc_config, **layer_kwargs, mesh_source=config)
        if stage_idx == 0:
            self.main_proj, self.main_norm = dspark_main_projection(config, **layer_kwargs)
        if stage_idx == num_stages - 1:
            self.norm = RMSNorm(**norm_kwargs, rngs=rngs)
            self.markov_head = DSparkMarkovHead(
                config, precision, rngs=rngs, embedding_name="embed", projection_name="head"
            )
            self.confidence_head = DSparkConfidenceHead(config, precision, rngs=rngs)

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
        attn_output = checkpoint_name(attn_output, "attn_output")
        hidden_streams = manifold_residual_write(
            hidden_streams, attn_output, attn_post, attn_comb, precision=self.precision
        )
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc.coefficients(hidden_streams)
        mlp_input = self.post_attention_layernorm(manifold_collapse(attn_pre, hidden_streams))
        mlp_output, _ = self.mlp(mlp_input)
        mlp_output = checkpoint_name(mlp_output, "mlp_output")
        hidden_streams = manifold_residual_write(
            hidden_streams, mlp_output, ffn_post, ffn_comb, precision=self.precision
        )
        return hidden_streams, ffn_pre, ring


class DeepseekV41DSpark(DSparkStageStack):
    """DeepSeek-V4.1 DSpark draft model (all ``num_nextn_predict_layers`` stages, greedy drafting).

    Load released weights with ``DeepseekV41DSpark.from_mtp_shards(config, shards, quantization_config=...)``.
    """

    checkpoint_key_rules = DEEPSEEK_V41_MTP_KEY_RULES

    @staticmethod
    def validate_config(config: DeepseekV41Config) -> None:
        """Reject a router temperature the shared V4 router does not implement."""
        if float(config.gate_temp) != 1.0:
            raise NotImplementedError("DeepSeek-V4.1 router gate_temp != 1.0 is not supported by the shared V4 router.")

    @staticmethod
    def draft_config(config: DeepseekV41Config, num_stages: int) -> DeepseekV41Config:
        """Config with the DSpark expert counts."""
        del num_stages
        return _draft_config(config)

    def build_stage(self, config, layer_idx, stage_idx, num_stages, *, dtype, param_dtype, precision, rngs):
        """A delayed-``pre`` V4.1 block with window-ring draft attention."""
        return DeepseekV41DSparkStage(
            config,
            layer_idx,
            stage_idx,
            num_stages,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def init_carry(self, batch: int, block: int) -> Array:
        """Stage 0 reads stream 0 only (one-hot delayed ``pre`` mix)."""
        return jnp.zeros((batch, block, self.hc_mult), jnp.float32).at[..., 0].set(1.0)

    def collapse(self, streams: Array, carry: Array) -> Array:
        """Collapse with the last stage's FFN read mix."""
        return manifold_collapse(carry, streams)


@register_dspark_drafter("deepseek_v41_dspark", "v41_dspark")
class DeepseekV41DSparkDrafter(DSparkDrafter):
    """eSurge drafter for the V4.1 DSpark draft model (the shared :class:`~easydel.modules.dspark.DSparkDrafter`).

    ``model.drafter(method="deepseek_v41_dspark", drafter_model=<DeepseekV41DSpark>)``; the draft model may also arrive
    in the ``assistant_model`` slot (``eSurgeDrafterConfig.assistant_model``).
    """


__all__ = [
    "DeepseekV41DSpark",
    "DeepseekV41DSparkAttention",
    "DeepseekV41DSparkDrafter",
    "DeepseekV41DSparkStage",
    "DeepseekV41DSparkState",
]
