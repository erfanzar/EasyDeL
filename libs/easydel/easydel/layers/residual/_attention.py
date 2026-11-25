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

"""Attention-based residual connections (Block Attention Residuals & Looped Message Gates).

Rather than summing every layer output uniformly into a single residual stream,
attention-residual architectures partition depth into residual blocks and attend
over completed block summaries plus the current partial block in float32 using
learned query vectors:

- :class:`BlockAttentionResidual`: attends over ``[block_states, partial_block]``
  with ``num_queries`` learned float32 queries (e.g. 1 query for sliding
  attention / MLP / final output, 3 queries for separate Q/K/V streams on full
  attention).
- :class:`LoopMessageAttentionResidual`: attends over the previous loop pass's
  completed block states + final state for each physical layer inside a looped
  core, producing per-site cross-pass messages (e.g. ``[q, k, v, mlp]``).
- :class:`LoopMessageGate`: low-rank forget/message gate blending a residual
  stream with an incoming cross-pass message:
  ``(1 - tanh(forget)) * hidden_states + tanh(message_gate) * message``.

Parameter attribute names (``query``, ``norm``, ``down_proj``,
``forget_gate_proj``, ``message_gate_proj``) match canonical Hugging Face
checkpoints directly so conversion requires no extra renaming.
"""

from __future__ import annotations

import jax
import spectrax as spx
from jax import numpy as jnp
from jaxtyping import Array, Float

from ..linears import ColumnParallelLinear
from ..norms import RMSNorm

__all__ = (
    "BlockAttentionResidual",
    "LoopMessageAttentionResidual",
    "LoopMessageGate",
)


def _unweighted_fp32_rmsnorm(eps: float = 1e-6) -> RMSNorm:
    """Build a parameter-free float32 RMSNorm for residual key/value normalization."""
    return RMSNorm(
        dim=1,
        eps=eps,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        with_scale=False,
        fp32_scale=True,
        output_dtype=None,
        rngs=spx.Rngs(0),
    )


class BlockAttentionResidual(spx.Module):
    """Attention-weighted aggregation over completed residual blocks and the current partial block.

    Computes float32 unweighted RMSNorm over the stacked block summaries and
    current partial residual, scores them with ``num_queries`` learned float32
    query vectors scaled by ``1 / sqrt(hidden_size)``, applies float32 softmax,
    and returns one aggregated ``[batch, seq_len, hidden_size]`` tensor per query.
    """

    def __init__(
        self,
        hidden_size: int,
        num_queries: int = 1,
        *,
        eps: float = 1e-6,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize the block attention residual module.

        Args:
            hidden_size: Width of each residual stream.
            num_queries: Number of independent attention queries (1 produces a
                1-D ``[hidden_size]`` query parameter; ``> 1`` produces
                ``[num_queries, hidden_size]``).
            eps: Epsilon for the internal parameter-free float32 RMSNorm.
            rngs: Random number generator container.
        """
        from easydel.infra.utils import ArrayParam

        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if num_queries <= 0:
            raise ValueError(f"num_queries must be positive, got {num_queries}.")
        self.num_queries = int(num_queries)
        self.hidden_size = int(hidden_size)
        shape = (self.hidden_size,) if self.num_queries == 1 else (self.num_queries, self.hidden_size)
        self.query = ArrayParam.bound(
            shape=shape,
            dtype=jnp.float32,
            init_method="zeros",
            key=rngs.param,
        )
        self.norm = _unweighted_fp32_rmsnorm(eps=eps)

    def forward(
        self,
        block_states: Float[Array, "batch seq_len num_blocks hidden_dim"],
        partial_block_states: Float[Array, "batch seq_len hidden_dim"],
    ) -> tuple[Array, ...]:
        """Aggregate ``block_states`` and ``partial_block_states`` for each query.

        Args:
            block_states: Completed residual block summaries
                ``[batch, seq_len, num_blocks, hidden_size]``.
            partial_block_states: Current partial block residual
                ``[batch, seq_len, hidden_size]``.

        Returns:
            Tuple of length ``num_queries`` containing float32 aggregated
            tensors of shape ``[batch, seq_len, hidden_size]``.
        """
        stacked = jnp.concatenate([block_states, partial_block_states[..., None, :]], axis=-2).astype(jnp.float32)
        stacked = self.norm(stacked)
        q = self.query.value.astype(jnp.float32).reshape(self.num_queries, self.hidden_size)
        scores = jnp.einsum("qh,bsnh->bsqn", q, stacked) * (self.hidden_size**-0.5)
        weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bsqn,bsnh->bsqh", weights, stacked)
        return tuple(out[:, :, i, :] for i in range(self.num_queries))


class LoopMessageAttentionResidual(spx.Module):
    """Cross-pass block attention residual producing per-site messages for each looped layer."""

    def __init__(
        self,
        hidden_size: int,
        num_loop_layers: int,
        num_messages: int = 4,
        *,
        eps: float = 1e-6,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize the cross-pass message attention residual.

        Args:
            hidden_size: Width of each residual stream.
            num_loop_layers: Number of physical layers inside the looped core.
            num_messages: Number of message streams produced per looped layer
                (defaults to 4 for ``[q, k, v, mlp]``).
            eps: Epsilon for the internal parameter-free float32 RMSNorm.
            rngs: Random number generator container.
        """
        from easydel.infra.utils import ArrayParam

        if hidden_size <= 0 or num_loop_layers <= 0 or num_messages <= 0:
            raise ValueError(
                f"hidden_size ({hidden_size}), num_loop_layers ({num_loop_layers}), "
                f"and num_messages ({num_messages}) must all be positive."
            )
        self.hidden_size = int(hidden_size)
        self.num_loop_layers = int(num_loop_layers)
        self.num_messages = int(num_messages)
        self.query = ArrayParam.bound(
            shape=(self.num_loop_layers, self.num_messages, self.hidden_size),
            dtype=jnp.float32,
            init_method="zeros",
            key=rngs.param,
        )
        self.norm = _unweighted_fp32_rmsnorm(eps=eps)

    def forward(
        self,
        block_states: Float[Array, "batch seq_len num_blocks hidden_dim"],
        partial_block_states: Float[Array, "batch seq_len hidden_dim"],
        loop_layer_idx: int,
    ) -> Float[Array, "batch seq_len num_messages hidden_dim"]:
        """Retrieve cross-pass messages for looped layer ``loop_layer_idx``.

        Args:
            block_states: Previous pass's completed block states
                ``[batch, seq_len, num_blocks, hidden_size]``.
            partial_block_states: Previous pass's terminal residual state
                ``[batch, seq_len, hidden_size]``.
            loop_layer_idx: Zero-based index inside the looped core.

        Returns:
            Float32 message tensor ``[batch, seq_len, num_messages, hidden_size]``.
        """
        stacked = jnp.concatenate([block_states, partial_block_states[..., None, :]], axis=-2).astype(jnp.float32)
        stacked = self.norm(stacked)
        q = self.query.value[loop_layer_idx].astype(jnp.float32)
        scores = jnp.einsum("qh,bsnh->bsqn", q, stacked) * (self.hidden_size**-0.5)
        weights = jax.nn.softmax(scores, axis=-1)
        return jnp.einsum("bsqn,bsnh->bsqh", weights, stacked)


class LoopMessageGate(spx.Module):
    """Low-rank gated mixer ``(1 - tanh(forget)) * hidden_states + tanh(message_gate) * message``."""

    def __init__(
        self,
        hidden_size: int,
        gate_rank: int,
        *,
        eps: float = 1e-6,
        initializer_range: float = 0.02,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        rngs: spx.Rngs,
    ) -> None:
        """Initialize the low-rank cross-pass message gate.

        Args:
            hidden_size: Hidden representation dimension.
            gate_rank: Bottleneck rank for the shared down-projection.
            eps: Epsilon for the internal parameter-free float32 RMSNorm.
            initializer_range: Standard deviation for linear weight init.
            dtype: Activation dtype for ``down_proj``.
            param_dtype: Storage dtype for ``down_proj`` (gate up-projections
                remain float32 to preserve small-gate precision).
            precision: Matmul precision.
            rngs: Random number generator container.
        """
        kernel_init = jax.nn.initializers.normal(initializer_range)
        self.norm = _unweighted_fp32_rmsnorm(eps=eps)
        self.down_proj = ColumnParallelLinear(
            2 * hidden_size,
            gate_rank,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            precision=precision,
            rngs=rngs,
        )
        self.forget_gate_proj = ColumnParallelLinear(
            gate_rank,
            hidden_size,
            use_bias=False,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            kernel_init=kernel_init,
            precision=precision,
            rngs=rngs,
        )
        self.message_gate_proj = ColumnParallelLinear(
            gate_rank,
            hidden_size,
            use_bias=False,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            kernel_init=kernel_init,
            precision=precision,
            rngs=rngs,
        )

    def forward(self, hidden_states: Array, message: Array) -> Array:
        """Blend ``hidden_states`` and ``message`` in float32 via learned forget/gain gates."""
        h = hidden_states.astype(jnp.float32)
        m = message.astype(jnp.float32)
        gate_in = jnp.concatenate([self.norm(h), self.norm(m)], axis=-1)
        gate_hidden = self.down_proj(gate_in.astype(self.down_proj.weight.value.dtype)).astype(jnp.float32)
        forget_gate = jnp.tanh(self.forget_gate_proj(gate_hidden))
        message_gate = jnp.tanh(self.message_gate_proj(gate_hidden))
        return (1.0 - forget_gate) * h + message_gate * m
