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

The orchestration (stage-0 main projection, ring seeding, block drafting, Markov loop, confidence head, loading) is the
shared :class:`~easydel.modules.dspark.DSparkStageStack`. What is V4-specific:

- stages are plain V4 blocks: every sublayer reads the streams through its own hyper-connection ``pre`` mix (no delayed
  ``pre`` carried between stages); the MoE is the backbone's (learned top-k, the backbone expert counts);
- the draft query gets the unweighted per-head RMS norm of V4 attention after ``wq_b`` (``q_head_norm_eps``);
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
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Float, Int

from easydel.layers import RMSNorm
from easydel.layers.attention import WindowRingDraftAttentionMixin
from easydel.layers.residual import manifold_residual_write
from easydel.modules.dspark import (
    DSparkConfidenceHead,
    DSparkDrafter,
    DSparkMarkovHead,
    DSparkStageStack,
    dspark_main_projection,
    register_dspark_drafter,
)

from .deepseek_v4_configuration import DeepseekV4Config
from .modeling_deepseek_v4 import (
    DEEPSEEK_V4_MTP_KEY_RULES,
    DeepseekV4Attention,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4SparseMoeBlock,
)


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

    def __init__(self, config: DeepseekV4Config, layer_idx: int, **kwargs: tp.Any):
        super().__init__(config, layer_idx=layer_idx, **kwargs)
        if self.layer_type != "sliding_attention":
            raise ValueError(f"DSpark stage layer {layer_idx} must be window-only, got {self.layer_type!r}.")


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
        self.input_layernorm = RMSNorm(**norm_kwargs, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(**norm_kwargs, rngs=rngs)
        self.attn_hc = DeepseekV4HyperConnection(config, **layer_kwargs)
        self.ffn_hc = DeepseekV4HyperConnection(config, **layer_kwargs)
        if stage_idx == 0:
            self.main_proj, self.main_norm = dspark_main_projection(config, **layer_kwargs)
        if stage_idx == num_stages - 1:
            self.hc_head = DeepseekV4HyperHead(config, **layer_kwargs)
            self.norm = RMSNorm(**norm_kwargs, rngs=rngs)
            self.markov_head = DSparkMarkovHead(
                config, precision, rngs=rngs, embedding_name="markov_w1", projection_name="markov_w2"
            )
            self.confidence_head = DSparkConfidenceHead(config, precision, rngs=rngs)

    def forward(
        self,
        hidden_streams: Float[Array, "batch block hc hidden"],
        carry: None,
        main_x: Float[Array, "batch 1 hidden"],
        ring: Array,
        position: Int[Array, "batch"],
    ) -> tuple[Array, None, Array]:
        """Return ``(streams, None, ring)`` (reference ``Block.forward`` at ``start_pos > 0``; V4 carries nothing)."""
        del carry
        post, comb, collapsed = self.attn_hc(hidden_streams)
        attn_output, ring = self.self_attn.draft_attend(self.input_layernorm(collapsed), main_x, ring, position)
        attn_output = checkpoint_name(attn_output, "attn_output")
        hidden_streams = manifold_residual_write(hidden_streams, attn_output, post, comb, precision=self.precision)
        post, comb, collapsed = self.ffn_hc(hidden_streams)
        mlp_output, _ = self.mlp(self.post_attention_layernorm(collapsed))
        mlp_output = checkpoint_name(mlp_output, "mlp_output")
        hidden_streams = manifold_residual_write(hidden_streams, mlp_output, post, comb, precision=self.precision)
        return hidden_streams, None, ring


class DeepseekV4DSpark(DSparkStageStack):
    """DeepSeek-V4 DSpark draft model (all stages, greedy drafting) on the shared stage-stack orchestration.

    Load released weights with ``DeepseekV4DSpark.from_mtp_shards(config, shards, num_stages=3,
    quantization_config=...)`` (routed experts default to fp4 storage, as ``inference/config.json`` states).
    """

    checkpoint_key_rules = DEEPSEEK_V4_MTP_KEY_RULES
    default_expert_dtype = "fp4"

    @staticmethod
    def resolve_num_stages(config: DeepseekV4Config, num_stages: int | None) -> int:
        """``num_stages`` when given, else :func:`dspark_num_stages`."""
        return int(num_stages) if num_stages is not None else dspark_num_stages(config)

    @staticmethod
    def draft_config(config: DeepseekV4Config, num_stages: int) -> DeepseekV4Config:
        """Layer schedules extended over the stage ids (window-only attention, learned MoE)."""
        return _draft_config(config, num_stages)

    def build_stage(self, config, layer_idx, stage_idx, num_stages, *, dtype, param_dtype, precision, rngs):
        """A plain V4 block with window-ring draft attention."""
        return DeepseekV4DSparkStage(
            config,
            layer_idx,
            stage_idx,
            num_stages,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def collapse(self, streams: Array, carry: None) -> Array:
        """The last stage's learned stream-collapse head."""
        del carry
        return self.stages[-1].hc_head(streams)


@register_dspark_drafter("deepseek_v4_dspark", "v4_dspark")
class DeepseekV4DSparkDrafter(DSparkDrafter):
    """eSurge drafter for the V4 DSpark draft model (the shared DSpark drafter).

    ``model.drafter(method="deepseek_v4_dspark", drafter_model=<DeepseekV4DSpark>)``.
    """


__all__ = [
    "DeepseekV4DSpark",
    "DeepseekV4DSparkAttention",
    "DeepseekV4DSparkDrafter",
    "DeepseekV4DSparkStage",
    "dspark_num_stages",
]
