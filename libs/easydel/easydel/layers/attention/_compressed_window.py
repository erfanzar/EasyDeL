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

"""Compressed-window attention: shared-KV MQA with sinks over a sliding window plus compressed KV entries.

:class:`CompressedWindowAttention` is the :class:`UnifiedAttention` subclass behind DeepSeek-V4 / V4.1 attention. It
owns everything the families share:

- the projections, in checkpoint order: ``q_a_proj -> q_a_norm -> q_b_proj`` (optional unweighted per-head query RMS
  norm), a single shared KV head ``kv_proj -> kv_norm`` read as both key and value, the grouped output projection
  ``o_a_proj`` (:class:`~easydel.layers.linears.GroupedLinear`) + ``o_b_proj``, and per-head ``sinks``;
- partial interleaved RoPE on the trailing channels, and the inverse rotation of the output (``K == V`` carried RoPE);
- the attends: the dense fp32 sink softmax (:func:`~easydel.layers.compressed_window.sink_attention`) and the
  registered ``compressed_window_attention`` / ``compressed_window_decode`` operations, created through
  :class:`~easydel.operations.OperationRegistry`.

Families keep their cache transitions (ring writes, compressor / indexer streams, packed serving) and decide which
attend runs. :class:`WindowRingDraftAttentionMixin` adds the DSpark draft path (a block of draft queries over a
sliding-window ring of main K/V) on top of the same projections.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Float, Int

from easydel.layers.compressed_window import (
    ring_decode_write,
    ring_prefill_write,
    ring_slot_positions,
    ring_visibility_bias,
    sink_attention,
)
from easydel.layers.linears import ColumnParallelLinear, GroupedLinear, RowParallelLinear
from easydel.layers.norms import RMSNorm, rms_norm
from easydel.layers.rotary import apply_rope_interleaved_partial, labeled_rope_cos_sin
from easydel.operations import OperationMetadata, OperationRegistry

from ._unified import UnifiedAttention

__all__ = ("CompressedWindowAttention", "WindowRingDraftAttentionMixin")


class CompressedWindowAttention(UnifiedAttention):
    """Shared-KV MQA with per-head sinks over a sliding window plus compressed KV entries.

    Config fields read: ``hidden_size``, ``num_attention_heads``, ``head_dim``, ``q_lora_rank``, ``o_groups``,
    ``o_lora_rank``, ``sliding_window``, ``rms_norm_eps``, ``initializer_range``, ``attention_bias`` (optional) and the
    label-keyed rope parameters (``rope_parameters_for_label``).

    Attributes:
        rope_label: Rope label of this layer (``"main"`` / ``"compress"``).
        q_head_norm_eps: Epsilon of the unweighted per-head query RMS norm after ``q_b_proj``, or ``None`` (no norm).
        scaling: Logit scale ``head_dim ** -0.5``.
    """

    def __init__(
        self,
        config: tp.Any,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
        rope_label: str,
        q_head_norm_eps: float | None = None,
    ):
        """Build the projections, sinks and the registered attend operations.

        Args:
            config: Model configuration.
            dtype: Activation dtype.
            param_dtype: Parameter storage dtype.
            precision: Matmul precision.
            rngs: Random number generators (consumed in projection order, then ``sinks``).
            layer_idx: Layer index.
            rope_label: Rope label of this layer.
            q_head_norm_eps: Per-head query norm epsilon, or ``None``.
        """
        self.rope_label = rope_label
        self.q_head_norm_eps = None if q_head_norm_eps is None else float(q_head_norm_eps)
        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="compressed_window",
            causal=True,
            sliding_window=config.sliding_window,
        )

    def define_network(
        self,
        config: tp.Any,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: spx.Rngs,
    ) -> None:
        """Create the checkpoint-ordered projections, ``sinks`` and the two attend operations."""
        # infra.utils imports layers, so this dependency must stay lazy.
        from easydel.infra.utils import ArrayParam

        linear_kwargs = dict(
            use_bias=bool(getattr(config, "attention_bias", False)),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        norm_kwargs = dict(eps=config.rms_norm_eps, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.scaling = self.head_dim**-0.5
        self.q_a_proj = ColumnParallelLinear(config.hidden_size, config.q_lora_rank, **linear_kwargs)
        self.q_a_norm = RMSNorm(dim=config.q_lora_rank, **norm_kwargs)
        self.q_b_proj = ColumnParallelLinear(config.q_lora_rank, self.num_heads * self.head_dim, **linear_kwargs)
        self.kv_proj = ColumnParallelLinear(config.hidden_size, self.head_dim, **linear_kwargs)
        self.kv_norm = RMSNorm(dim=self.head_dim, **norm_kwargs)
        self.o_a_proj = GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.o_b_proj = RowParallelLinear(config.o_groups * config.o_lora_rank, config.hidden_size, **linear_kwargs)
        self.sinks = ArrayParam.bound(shape=(self.num_heads,), dtype=param_dtype, init_method="zeros", key=rngs.param)
        metadata = OperationMetadata(runtime_dtype=dtype, runtime_softmax_dtype=jnp.float32, base_config=config)
        self.attn_op = OperationRegistry.create("compressed_window_attention", metadata)
        self.decode_op = OperationRegistry.create("compressed_window_decode", metadata)

    # ------------------------------------------------------------------ projections
    def project_query(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        cos: Array,
        sin: Array,
    ) -> tuple[Array, Array]:
        """Return ``(q_residual [B, S, q_lora], q [B, H, S, D])``: LoRA query, optional head norm, RoPE."""
        batch, seq, _ = hidden_states.shape
        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).reshape(batch, seq, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        if self.q_head_norm_eps is not None:
            q = rms_norm(q, self.q_head_norm_eps)
        return q_residual, apply_rope_interleaved_partial(q, cos, sin, unsqueeze_dim=1)

    def project_kv(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        cos: Array,
        sin: Array,
        *,
        layout: tp.Callable[[Array], Array] | None = None,
    ) -> Array:
        """Post-RoPE shared key/value ``[B, S, D]``; ``layout`` (e.g. a replication constraint) runs before the norm."""
        kv = self.kv_proj(hidden_states)
        if layout is not None:
            kv = layout(kv)
        kv = self.kv_norm(kv)[:, None]
        return apply_rope_interleaved_partial(kv, cos, sin, unsqueeze_dim=1)[:, 0]

    def project_output(self, attn_output: Float[Array, "batch seq heads dim"], cos: Array, sin: Array) -> Array:
        """Undo the value RoPE (``K == V``) at the query positions, then the grouped output projection."""
        batch, seq = attn_output.shape[:2]
        attn_output = apply_rope_interleaved_partial(attn_output, cos, -sin, unsqueeze_dim=2)
        grouped = self.o_a_proj(attn_output.reshape(batch, seq, self.config.o_groups, -1)).reshape(batch, seq, -1)
        return checkpoint_name(self.o_b_proj(grouped), "attn_output")

    # ------------------------------------------------------------------ attends
    def sink_attend(self, q: Array, kv: Array, attn_bias: Array) -> Array:
        """Dense fp32 sink softmax ``[B, H, S, D]`` over ``kv [B, T, D]`` with bias ``[B, 1, S, T]``."""
        return sink_attention(q, kv, attn_bias, self.sinks.value, scaling=self.scaling, precision=self.precision)

    def fused_full_attend(
        self,
        q: Array,
        kv: Array,
        bias: Array,
        *,
        platform: str | None,
        token_valid: Array | None = None,
        entry_topk: Array | None = None,
    ) -> Array:
        """Full-sequence attend through the registered ``compressed_window_attention`` operation.

        The KV axis is ``[own token KVs (q length) | compressed entries]``; the window band-skips token tiles. The
        operation builds the head-parallel ``shard_map`` itself when a Pallas kernel runs on a multi-device mesh.

        Args:
            q: Rotated queries ``[B, H, S, D]``.
            kv: ``[B, S + T, D]``.
            bias: Additive fp32 bias ``[B, S, L]`` (``L`` may be ``0`` when ``token_valid`` / ``entry_topk`` carry
                the mask).
            platform: Kernel platform (``None`` = the operation's choice).
            token_valid: Optional per-key validity of the token region ``[B, S]``.
            entry_topk: Optional selected entry ids ``[B, S, k]`` (``-1`` padded).
        """
        return self.attn_op(
            q,
            kv,
            bias,
            self.sinks.value.astype(jnp.float32),
            softmax_scale=self.scaling,
            window=self.sliding_window,
            token_kv_len=q.shape[2],
            platform=platform,
            token_valid=token_valid,
            entry_topk=entry_topk,
        ).attention_outputs

    def fused_decode_attend(self, q: Array, kv: Array, bias: Array, *, platform: str | None) -> Array:
        """Decode attend through the registered ``compressed_window_decode`` operation.

        Args:
            q: ``[B, H, S, D]`` (``S == 1`` on decode).
            kv: Ring plus gathered entries ``[B, L, D]`` (``K == V``).
            bias: Additive fp32 bias ``[B, S, L]`` (head-broadcast).
            platform: Kernel platform.
        """
        return self.decode_op(
            q,
            kv,
            bias,
            self.sinks.value.astype(jnp.float32),
            softmax_scale=self.scaling,
            platform=platform,
        ).attention_outputs


class WindowRingDraftAttentionMixin:
    """DSpark draft attention over a sliding-window ring of main K/V (mix into a :class:`CompressedWindowAttention`).

    Every stage turns the backbone's projected main hidden ``main_x`` into window K/V with its own ``kv_proj`` /
    ``kv_norm`` (token ``p`` at ring slot ``p % window``); a draft block's queries attend to the valid ring plus the
    block's own K/V, bidirectionally, at positions ``p + 1 .. p + block``.
    """

    def main_kv(self, main_x: Array, position_ids: Int[Array, "batch seq"]) -> Array:
        """Project main hidden states to post-RoPE K/V vectors ``[B, S, D]``."""
        cos, sin = labeled_rope_cos_sin(self.config, self.rope_label, position_ids, main_x.dtype)
        return self.project_kv(main_x, cos, sin)

    def seed_ring(self, main_x: Array) -> Array:
        """Seed a ring from prompt states, projecting only positions retained in the window."""
        batch, seq, _ = main_x.shape
        window = int(self.sliding_window)
        start = max(0, seq - window)
        positions_1d = jnp.arange(start, seq, dtype=jnp.int32)
        positions = jnp.broadcast_to(positions_1d[None], (batch, positions_1d.shape[0]))
        kv = self.main_kv(main_x[:, start:, :], positions)
        ring = jnp.zeros((batch, window, kv.shape[-1]), kv.dtype)
        return ring_prefill_write(ring, kv, window, start_position=start)

    def draft_attend(
        self,
        hidden_states: Array,
        main_x: Array,
        ring: Array,
        position: Int[Array, "batch"],  # noqa: F821
    ) -> tuple[Array, Array]:
        """Write the current main K/V into the ring, then attend the draft block over ring plus block K/V."""
        batch, block, _ = hidden_states.shape
        window = int(self.sliding_window)
        main_kv = self.main_kv(main_x, position[:, None])
        ring = ring_decode_write(ring, main_kv[:, 0], position, window)

        draft_positions = position[:, None] + 1 + jnp.arange(block, dtype=jnp.int32)[None, :]
        cos, sin = labeled_rope_cos_sin(self.config, self.rope_label, draft_positions, hidden_states.dtype)
        _, q = self.project_query(hidden_states, cos, sin)
        kv = self.project_kv(hidden_states, cos, sin)

        keys = jnp.concatenate([ring.astype(kv.dtype), kv], axis=1)
        ring_bias = ring_visibility_bias(ring_slot_positions(position, window))
        bias = jnp.concatenate([ring_bias, jnp.zeros((batch, block), jnp.float32)], axis=1)[:, None, None, :]
        attn = self.sink_attend(q, keys, bias)
        return self.project_output(attn.transpose(0, 2, 1, 3), cos, sin), ring
