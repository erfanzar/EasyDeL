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

"""Backbone-block DSpark draft models: one shared orchestration, family-specific stage blocks.

Released DSpark checkpoints (DeepSeek-V4-Flash-0731, DeepSeek-V4.1-Flash) ship ``mtp.N.*`` stages that are full
backbone blocks whose attention drafts against a sliding-window ring of *main* K/V. Everything around the blocks is
identical across families and lives here (:class:`DSparkStageStack`):

- stage 0 projects the backbone's concatenated target-layer hidden states to ``main_x`` (``main_proj`` / ``main_norm``,
  :func:`dspark_main_projection`);
- :meth:`DSparkStageStack.prefill` seeds every stage's ring; :meth:`DSparkStageStack.draft` runs a block (last accepted
  token + noise tokens) through the stages, collapses the streams, applies the shared LM head, then drafts greedily
  with the low-rank Markov bias and scores each position with the confidence head (:class:`DSparkMarkovHead`,
  :class:`DSparkConfidenceHead`);
- :meth:`DSparkStageStack.from_mtp_shards` loads native ``mtp.*`` shards through the class's declared
  ``checkpoint_key_rules`` (see :mod:`easydel.utils.checkpoint_keys`).

A family subclass supplies only what differs:

- :meth:`~DSparkStageStack.build_stage` -- the stage block, called as
  ``stage(streams, carry, main_x, ring, position) -> (streams, carry, ring)``;
- :meth:`~DSparkStageStack.init_carry` / :meth:`~DSparkStageStack.collapse` -- what a stage hands the next one (e.g.
  V4.1's delayed hyper-connection ``pre`` mix) and how the final streams become one hidden state;
- :meth:`~DSparkStageStack.resolve_num_stages`, :meth:`~DSparkStageStack.draft_config`,
  :meth:`~DSparkStageStack.validate_config`.

The serving runtime (window-ring pool, per-row positions, jitted programs) is the shared
:class:`~easydel.modules.dspark.DSparkDrafter`.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx
from jaxtyping import Array, Float, Int
from spectrax import nn

from easydel.layers import ColumnParallelLinear, ParallelLinear, RMSNorm
from easydel.layers.embeddings._embeddings import default_embed_init
from easydel.modules._drafting import AcceptRatePredictor, VanillaMarkovHead

from ._drafter import DSparkState

if tp.TYPE_CHECKING:
    from easydel.utils.checkpoint_keys import KeyRuleSet

    from ._loader import StageShards

__all__ = (
    "DSparkConfidenceHead",
    "DSparkMarkovHead",
    "DSparkStageStack",
    "dspark_main_projection",
)

EmbedFn = tp.Callable[[Array], Array]
LmHeadFn = tp.Callable[[Array], Array]


def dspark_main_projection(
    config: tp.Any,
    *,
    dtype: jnp.dtype,
    param_dtype: jnp.dtype,
    precision: jax.lax.PrecisionLike,
    rngs: spx.Rngs,
) -> tuple[ColumnParallelLinear, RMSNorm]:
    """Stage-0 ``(main_proj, main_norm)``: ``len(dspark_target_layer_ids) * D -> D``, then RMSNorm.

    Built in this order (``main_proj`` first), so assigning the tuple keeps the checkpoint's RNG order.
    """
    main_proj = ColumnParallelLinear(
        len(config.dspark_target_layer_ids) * config.hidden_size,
        config.hidden_size,
        use_bias=False,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        kernel_init=jax.nn.initializers.normal(config.initializer_range),
        rngs=rngs,
    )
    main_norm = RMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
    return main_proj, main_norm


class DSparkMarkovHead(VanillaMarkovHead):
    """fp32 low-rank Markov bias over the previously drafted token, with checkpoint-specific leaf names.

    DeepSeek-V4.1 stores ``embed`` / ``head``; DeepSeek-V4 stores ``markov_w1`` / ``markov_w2``.
    """

    def __init__(
        self,
        config: tp.Any,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        embedding_name: str = "embed",
        projection_name: str = "head",
    ):
        super().__init__(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=precision,
            rngs=rngs,
            embedding_name=embedding_name,
            projection_name=projection_name,
            fp32=True,
            embedding_init=default_embed_init,
        )


class DSparkConfidenceHead(AcceptRatePredictor):
    """fp32 per-position acceptance logit over ``concat([hidden, markov_embed])`` (leaf ``proj``)."""

    def __init__(self, config: tp.Any, precision: jax.lax.PrecisionLike = None, *, rngs: spx.Rngs):
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
        """Score concatenated hidden and Markov features ``[B, block]``."""
        features = jnp.concatenate([hidden.astype(jnp.float32), markov_embed.astype(jnp.float32)], axis=-1)
        return super().forward(features)


class DSparkStageStack(spx.Module):
    """Backbone-block DSpark draft model (all stages, greedy drafting); family subclasses supply the stage block.

    Build under ``with config.mesh:``. Usage: ``state = model.prefill(main_hidden)`` once per prompt, then for every
    step ``draft_ids, logits, confidence, state = model.draft(token, main_hidden_step, state, embed, lm_head)``.
    Stage 0 must own ``main_proj`` / ``main_norm`` (:func:`dspark_main_projection`) and the last stage ``norm``,
    ``markov_head`` and ``confidence_head``.

    Attributes:
        checkpoint_key_rules: Rule set mapping native ``mtp.N.*`` keys to ``stages.N.*`` (used by
            :meth:`from_mtp_shards`).
        default_expert_dtype: Routed-expert storage format folded into a checkpoint ``quantization_config`` that does
            not state it (the released V4 ``config.json`` keeps ``expert_dtype: fp4`` outside it).
    """

    checkpoint_key_rules: tp.ClassVar[KeyRuleSet | None] = None
    default_expert_dtype: tp.ClassVar[str | None] = None

    def __init__(
        self,
        config: tp.Any,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        num_stages: int | None = None,
    ):
        """Initialize the draft stages.

        Args:
            config: Backbone config with ``dspark_block_size > 0``, ``dspark_target_layer_ids``,
                ``dspark_noise_token_id`` and ``dspark_markov_rank``.
            dtype: Activation dtype.
            param_dtype: Backbone-style parameter storage dtype (Markov / confidence heads are always fp32).
            precision: Matmul precision.
            rngs: Random number generators.
            num_stages: Number of ``mtp.N`` stages; ``None`` uses :meth:`resolve_num_stages`'s default.

        Raises:
            ValueError: If DSpark is not configured or resolves to no stage.
        """
        if int(getattr(config, "dspark_block_size", 0) or 0) <= 0 or not getattr(
            config, "dspark_target_layer_ids", None
        ):
            raise ValueError("DSpark needs dspark_block_size > 0 and non-empty dspark_target_layer_ids.")
        stages = int(self.resolve_num_stages(config, num_stages))
        if stages <= 0:
            raise ValueError("DSpark needs at least one stage.")
        self.validate_config(config)
        self.config = config
        self.precision = precision
        self.block_size = int(config.dspark_block_size)
        self.noise_token_id = int(config.dspark_noise_token_id)
        self.hc_mult = int(config.hc_mult)
        self.num_stages = stages
        draft_config = self.draft_config(config, stages)
        self.stages = nn.ModuleList([])
        for stage_idx in range(stages):
            self.stages.append(
                self.build_stage(
                    draft_config,
                    int(config.num_hidden_layers) + stage_idx,
                    stage_idx,
                    stages,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
            )

    # ------------------------------------------------------------------ family hooks
    @staticmethod
    def resolve_num_stages(config: tp.Any, num_stages: int | None) -> int:
        """Stage count: ``num_stages`` when given, else ``config.num_nextn_predict_layers``."""
        return int(num_stages) if num_stages is not None else int(config.num_nextn_predict_layers)

    @staticmethod
    def validate_config(config: tp.Any) -> None:
        """Reject configurations the family's draft blocks cannot represent (default: accept)."""
        del config

    @staticmethod
    def draft_config(config: tp.Any, num_stages: int) -> tp.Any:
        """Config the stage blocks are built from (stage ids are ``num_hidden_layers + s``); default unchanged."""
        del num_stages
        return config

    def build_stage(
        self,
        config: tp.Any,
        layer_idx: int,
        stage_idx: int,
        num_stages: int,
        *,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: spx.Rngs,
    ) -> spx.Module:
        """Build stage ``stage_idx`` (called as ``stage(streams, carry, main_x, ring, position)``)."""
        raise NotImplementedError

    def init_carry(self, batch: int, block: int) -> tp.Any:
        """Value handed to stage 0 alongside the streams (default ``None``)."""
        del batch, block
        return None

    def collapse(self, streams: Array, carry: tp.Any) -> Array:
        """Collapse the final ``[B, block, hc, D]`` streams to ``[B, block, D]``."""
        raise NotImplementedError

    # ------------------------------------------------------------------ shared orchestration
    def _main_x(self, main_hidden: Array) -> Array:
        first = self.stages[0]
        return first.main_norm(first.main_proj(main_hidden))

    def prefill(self, main_hidden: Float[Array, "batch seq targets_hidden"]) -> DSparkState:
        """Seed every stage's window ring from the prompt's main hidden states (reference ``start_pos == 0``).

        Args:
            main_hidden: Concatenated target-layer hidden states ``[B, S, len(targets) * D]`` of positions ``0..S-1``.

        Returns:
            State whose ``position`` is ``S`` (the position of the first main hidden state :meth:`draft` receives).
        """
        main_x = self._main_x(main_hidden)
        rings = tuple(stage.self_attn.seed_ring(main_x) for stage in self.stages)
        position = jnp.full((main_hidden.shape[0],), main_hidden.shape[1], dtype=jnp.int32)
        return DSparkState(rings, position)

    def draft(
        self,
        token_ids: Int[Array, "batch"],  # noqa: F821
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
            embed: Backbone embedding lookup ``ids [..] -> [.., D]`` (shared with the backbone).
            lm_head: Backbone LM head ``hidden [.., D] -> logits [.., V]`` (shared with the backbone).

        Returns:
            ``(draft_ids [B, block + 1] int32, logits [B, block, V] fp32, confidence [B, block] fp32, new_state)``;
            ``draft_ids[:, 0]`` is ``token_ids``, ``draft_ids[:, i + 1]`` the greedy token of ``logits[:, i]`` (which
            already includes the Markov bias of ``draft_ids[:, i]``). ``new_state.position`` is ``position + 1``.
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
        carry = self.init_carry(batch, block)
        rings = []
        for stage, ring in zip(stages, state.window_kv, strict=True):
            streams, carry, ring = stage(streams, carry, main_x, ring, position)
            rings.append(ring)

        last = stages[-1]
        hidden = self.collapse(streams, carry)
        logits = lm_head(last.norm(hidden)).astype(jnp.float32)
        markov = last.markov_head
        rank = int(self.config.dspark_markov_rank)

        def step(i, loop_carry):
            out_ids, out_logits, embeds = loop_carry
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

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_mtp_shards(
        cls,
        config: tp.Any,
        shard_paths: tp.Sequence[str] | None,
        *,
        quantization_config: tp.Mapping[str, tp.Any] | None = None,
        expert_dtype: str | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        stage_shards: StageShards | None = None,
        num_stages: int | None = None,
    ) -> DSparkStageStack:
        """Build the draft model on ``config.mesh`` holding the weights of native ``mtp.*`` shards.

        Keys go through the class's ``checkpoint_key_rules``; pre-quantized weights through the bridge's quant
        decoder; layout through ``reform_param``; placement from the abstract module's sharding metadata (see
        :func:`easydel.modules.dspark.load_dspark_model`).

        Args:
            config: The backbone config carrying the ``dspark_*`` fields (its mesh is used for placement).
            shard_paths: Safetensors files containing the ``mtp.*`` tensors; ``None`` when ``stage_shards`` is given.
            quantization_config: The checkpoint's ``quantization_config``; ``None`` for an unquantized checkpoint.
            expert_dtype: Routed-expert storage format folded into ``quantization_config`` when absent there;
                defaults to :attr:`default_expert_dtype`.
            dtype: Activation dtype of the draft model.
            param_dtype: Parameter storage dtype (fp8 / fp4 values decode exactly into bf16).
            precision: Matmul precision.
            stage_shards: Optional per-stage shard provider used instead of ``shard_paths``.
            num_stages: Stage count (see :meth:`resolve_num_stages`).

        Returns:
            The draft model with every parameter loaded and placed on its sharding.

        Raises:
            TypeError: If the class declares no ``checkpoint_key_rules``.
            ValueError: If the shards miss a stage or a parameter, carry an unexpected one, or a shape disagrees.
        """
        from ._loader import load_dspark_model

        if cls.checkpoint_key_rules is None:
            raise TypeError(f"{cls.__name__} declares no checkpoint_key_rules.")
        qcfg = None
        if quantization_config is not None:
            qcfg = dict(quantization_config)
            fmt = expert_dtype if expert_dtype is not None else cls.default_expert_dtype
            if fmt is not None:
                qcfg.setdefault("expert_dtype", fmt)
        return load_dspark_model(
            config,
            shard_paths,
            build=lambda: cls(
                config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=spx.Rngs(0),
                num_stages=num_stages,
            ),
            key_normalizer=cls.checkpoint_key_rules,
            quantization_config=qcfg,
            param_dtype=param_dtype,
            stage_shards=stage_shards,
        )
