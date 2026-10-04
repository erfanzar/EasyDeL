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

import contextlib
import copy
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from jaxtyping import Array, Float, Int
from spectrax import nn

from easydel.infra.spec_decode import register_drafter
from easydel.layers import ColumnParallelLinear, ParallelLinear
from easydel.layers.embeddings._embeddings import default_embed_init
from easydel.layers.residual import manifold_collapse, manifold_residual_write
from easydel.modules._drafting import AcceptRatePredictor, VanillaMarkovHead, WindowRingDraftAttentionMixin
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


class DeepseekV41DSparkState(tp.NamedTuple):
    """Draft-model state: one sliding-window ring per stage and the next main position.

    Attributes:
        window_kv: Per stage ``[B, sliding_window, head_dim]`` post-RoPE main K/V ring (token ``p`` at slot ``p % W``).
        position: ``int32 [B]`` position of the main hidden state the next :meth:`DeepseekV41DSpark.draft` call gets.
    """

    window_kv: tuple[Array, ...]
    position: Array


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


def _pad_pow2(count: int, minimum: int = 8) -> int:
    """Smallest power of two ``>= max(count, minimum)`` (bounds the number of compiled append programs)."""
    size = minimum
    while size < count:
        size *= 2
    return size


class DeepseekV41DSparkDrafter:
    """eSurge drafter that runs the V4.1 DSpark draft model over the backbone's main hidden states.

    Contract with the eSurge runner (the flags are read through ``getattr`` by the runner / strategy):

    - ``needs_main_hidden``: the model step returns ``concat([last_hidden, main_hidden], -1)`` rows (width
      ``(1 + T) * D``); the drafter reads the ``main_hidden`` part ``[D:]`` of every row.
    - ``whole_block_draft``: one :meth:`draft_window` call proposes the whole window (greedy), instead of the
      per-token loop of the generic strategy.
    - ``rolls_back_cache``: the target cache (``SharedCompressedWindowCache``) is not paged, so the runner snapshots
      it before a speculative verify window and, when any draft is rejected, restores it and replays the accepted
      prefix. The DSpark rings are NOT rolled back: only committed positions are ever written into them.
    - ``supports_compressed_window_cache``: lifts the runner's compressed-window / drafter incompatibility guard.
    - ``supports_prefix_draft``: makes the runner return full hidden rows for prompt windows (the rings are seeded
      from every prompt position).

    Ring bookkeeping (host side, per drafter row ``r``): ``_through[r]`` is the highest main position whose K/V is in
    the rings of row ``r``. Window hidden rows reach the drafter through :meth:`observe_window` (committed real tokens,
    written eagerly) and :meth:`draft_window` (accepted draft positions, written lazily once the verify outcome is
    known, followed by the seed position itself). A row whose positions are not contiguous (missing hidden rows,
    chunked prefill) is marked broken and simply stops drafting: output stays correct, speculation just turns off.

    Only greedy requests are drafted for (the draft head is greedy).
    """

    supports_prefix_draft: bool = True
    supports_return_full_log_probs: bool = False
    requires_target_kv_cache: bool = False
    supports_compressed_window_cache: bool = True
    needs_main_hidden: bool = True
    whole_block_draft: bool = True
    rolls_back_cache: bool = True

    def __init__(
        self,
        *,
        target_model: tp.Any,
        dspark: DeepseekV41DSpark,
        num_draft_tokens: int | None = None,
        dtype: jnp.dtype | None = None,
    ):
        """Create the drafter.

        Args:
            target_model: The V4.1 causal LM whose embedding and LM head the draft model shares.
            dspark: The DSpark draft model (built under ``config.mesh``).
            num_draft_tokens: Drafted tokens per verify window; clamped to ``dspark.block_size``. ``None`` uses the
                full block.
            dtype: Ring storage dtype; defaults to the target model's ``dtype``.
        """
        block = int(dspark.block_size)
        requested = block if num_draft_tokens is None else int(num_draft_tokens)
        self.target_model = target_model
        self.dspark = dspark
        self.config = dspark.config
        self.block_size = block
        self.num_draft_tokens = max(1, min(requested, block))
        self.hidden_size = int(self.config.hidden_size)
        self.window = int(self.config.sliding_window)
        self.dtype = dtype if dtype is not None else getattr(target_model, "dtype", jnp.float32)
        self._capacity = 0
        self._rings: tuple[jax.Array, ...] | None = None
        self._owner: dict[int, str] = {}
        self._through: dict[int, int] = {}
        self._last_window: dict[int, tuple[jax.Array, int, int, int]] = {}
        self._broken: set[int] = set()
        self._programs: tuple[tp.Any, ...] | None = None

    def set_max_length(self, max_length: int) -> None:
        """Accept the runner's max model length (rings are window-sized, so nothing to size)."""
        del max_length

    def reset_request(self, request_id: str | None = None) -> None:
        """No-op: per-request state is reset when a row changes owner (see :meth:`observe_window`)."""
        del request_id

    def reset(self, batch_size: int) -> None:
        """Re-allocate zeroed rings for ``batch_size`` rows and forget every row owner."""
        self._alloc(max(1, int(batch_size)))

    def ensure_batch_size(self, batch_size: int) -> None:
        """Allocate the ring pool on first use (no-op when already large enough)."""
        if self._rings is None or self._capacity < int(batch_size):
            self._alloc(int(batch_size))

    def _alloc(self, rows: int) -> None:
        head_dim = int(self.config.head_dim)
        self._rings = tuple(
            jnp.zeros((rows, self.window, head_dim), self.dtype) for _ in range(int(self.dspark.num_stages))
        )
        self._capacity = rows
        self._owner.clear()
        self._through.clear()
        self._last_window.clear()
        self._broken.clear()

    def _reset_row(self, row: int, req_id: str) -> None:
        """Hand ``row`` to ``req_id``: zero its rings and restart its position bookkeeping."""
        self._owner[row] = req_id
        self._through[row] = -1
        self._last_window.pop(row, None)
        self._broken.discard(row)
        self._rings = tuple(ring.at[row].set(0) for ring in self._rings)

    @contextlib.contextmanager
    def _context(self) -> tp.Iterator[None]:
        from easydel.utils import set_inference_mode

        with self.target_model.mesh, set_inference_mode():
            yield

    def _ensure_programs(self) -> None:
        """Export both models once and build the jitted append / draft programs."""
        if self._programs is not None:
            return
        dgraphdef, dstate = spx.export(self.dspark)
        tgraphdef, tstate = spx.export(self.target_model)
        hidden_size, window = self.hidden_size, self.window

        def append(dstate, rings, hidden, gather, positions, rows):
            dm = spx.bind(dgraphdef, dstate)
            main_flat = hidden[gather][:, hidden_size:]
            main_x = dm._main_x(main_flat[None])
            slots = positions % window
            out = []
            for stage, ring in zip(dm.stages, rings, strict=True):
                kv = stage.self_attn.main_kv(main_x, positions[None])[0]
                out.append(ring.at[rows, slots].set(kv.astype(ring.dtype), mode="drop"))
            return tuple(out)

        def step(dstate, tstate, rings, row, token, main_step, position):
            dm = spx.bind(dgraphdef, dstate)
            tm = spx.bind(tgraphdef, tstate)
            state = DeepseekV41DSparkState(tuple(ring[row][None] for ring in rings), position[None])
            ids, _logits, _confidence, new_state = dm.draft(
                token[None], main_step[None, None, :], state, tm.compute_embedding, tm.compute_lm_logits
            )
            new_rings = tuple(
                ring.at[row].set(win[0].astype(ring.dtype)) for ring, win in zip(rings, new_state.window_kv, strict=True)
            )
            return ids[0], new_rings

        self._programs = (dstate, tstate, jax.jit(append), jax.jit(step))

    def _write(self, hidden: jax.Array, gather: list[int], rows: list[int], positions: list[int]) -> None:
        """Write main K/V of ``hidden[gather]`` (positions ``positions``, rows ``rows``) into the rings."""
        count = len(gather)
        if count == 0:
            return
        self._ensure_programs()
        dstate, _tstate, append, _step = self._programs
        pad = _pad_pow2(count) - count
        gather_a = jnp.asarray(np.asarray(gather + [0] * pad, dtype=np.int32))
        positions_a = jnp.asarray(np.asarray(positions + [0] * pad, dtype=np.int32))
        rows_a = jnp.asarray(np.asarray(rows + [self._capacity] * pad, dtype=np.int32))  # out-of-range rows are dropped
        with self._context():
            self._rings = append(dstate, self._rings, hidden, gather_a, positions_a, rows_a)

    def _catch_up(self, row: int, start: int) -> None:
        """Write positions accepted after the previous window (accepted drafts) up to ``start - 1`` from its hidden.

        A window whose verify accepted draft tokens committed positions beyond its real tokens; when no draft was
        proposed afterwards (:meth:`draft_window` writes them lazily) they are filled here from the remembered hidden
        rows of that window.
        """
        through = self._through.get(row, -1)
        previous = self._last_window.get(row)
        if through >= start - 1 or previous is None:
            return
        hidden, offset, pstart, pcount = previous
        if through + 1 < pstart:
            return  # gap before the remembered window: cannot be recovered, the caller marks the row broken
        positions = list(range(max(through + 1, start - self.window), min(start, pstart + pcount)))
        if positions:
            self._write(hidden, [offset + q - pstart for q in positions], [row] * len(positions), positions)
            if positions[-1] == start - 1:
                self._through[row] = start - 1

    def observe_window(self, hidden: jax.Array | None, entries: list[tuple[str, int, int, int, int, int]]) -> None:
        """Record the committed real tokens of one executed window.

        Args:
            hidden: Window hidden rows ``[N, (1 + T) * D]`` (last hidden | main hidden), or ``None``.
            entries: ``(req_id, row, offset, start, n_commit, count)`` per scheduled request: the request's rows start
                at ``hidden[offset]``, its first token is at position ``start``, ``n_commit`` leading tokens are
                committed (scheduled draft tokens excluded) and ``count`` tokens were scheduled in total.
                ``n_commit < 0`` means the hidden rows are unavailable.
        """
        if not entries:
            return
        self.ensure_batch_size(max(row for _rid, row, *_rest in entries) + 1)
        gather: list[int] = []
        rows: list[int] = []
        positions: list[int] = []
        for req_id, row, offset, start, n_commit, count in entries:
            req_id = str(req_id)
            if self._owner.get(row) != req_id or start == 0:
                self._reset_row(row, req_id)  # new request on this row, or a request restarted from position 0
            if row in self._broken:
                continue
            self._catch_up(row, start)
            if n_commit < 0 or hidden is None or self._through[row] != start - 1:
                self._broken.add(row)
                continue
            self._last_window[row] = (hidden, offset, start, int(count))
            if n_commit == 0:
                continue
            last = start + n_commit
            for q in range(max(start, last - self.window), last):  # only the last window of positions is kept
                gather.append(offset + q - start)
                rows.append(row)
                positions.append(q)
            self._through[row] = last - 1
        self._write(hidden, gather, rows, positions)

    def draft_window(
        self,
        *,
        req_id: str,
        row: int,
        seed_token: int,
        seed_position: int,
        hidden: jax.Array,
        offset: int,
        start: int,
        n: int,
        k: int,
    ) -> list[int]:
        """Greedily draft ``k`` tokens after ``seed_token`` (at position ``seed_position + 1``).

        Args:
            req_id: Request being drafted for.
            row: The request's stable drafter row.
            seed_token: Last accepted token.
            seed_position: Position of the main hidden state the draft reads (``seed_token``'s position minus one);
                must lie inside the window ``[start, start + n)`` of the last observed step.
            hidden: The last observed step's hidden rows ``[N, (1 + T) * D]``.
            offset: Row of ``hidden`` holding the request's first token of that step.
            start: Position of that first token.
            n: Number of tokens the request had in that step.
            k: Tokens to draft (``<= num_draft_tokens``).

        Returns:
            ``k`` drafted token ids, or ``[]`` when this row cannot draft (broken / out-of-window seed).
        """
        req_id = str(req_id)
        position = int(seed_position)
        local = position - int(start)
        if self._owner.get(row) != req_id or row in self._broken or not 0 <= local < int(n):
            return []
        through = self._through.get(row, -1)
        if through < start - 1:
            self._broken.add(row)
            return []
        missing = list(range(max(through + 1, int(start)), position))[-self.window :]
        if missing:
            self._write(hidden, [int(offset) + q - int(start) for q in missing], [row] * len(missing), missing)
        self._ensure_programs()
        dstate, tstate, _append, step = self._programs
        main_step = hidden[int(offset) + local][self.hidden_size :]
        with self._context():
            ids, self._rings = step(
                dstate,
                tstate,
                self._rings,
                jnp.asarray(row, dtype=jnp.int32),
                jnp.asarray(int(seed_token), dtype=jnp.int32),
                main_step,
                jnp.asarray(position, dtype=jnp.int32),
            )
        self._through[row] = position
        return [int(t) for t in np.asarray(ids)[1 : 1 + int(k)]]

    def draft(self, *args: tp.Any, **kwargs: tp.Any) -> tp.NoReturn:
        """Unsupported: this drafter is driven through :meth:`draft_window` (``whole_block_draft``)."""
        del args, kwargs
        raise NotImplementedError("DeepseekV41DSparkDrafter drafts whole windows via `draft_window`.")


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
