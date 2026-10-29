# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""XLA registry entry (priority 0 fallback) for the chunked per-channel-decay KDA forward."""

from __future__ import annotations

from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl import kda_chunk_xla, validate_options


@kernel_registry.register("kda_chunk", Platform.XLA, Backend.ANY)
def kda_chunk(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len v_dim"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len head_dim"],
    initial_state: Float[Array, "batch num_heads head_dim v_dim"] | None = None,
    segment_ids: Int[Array, "batch seq_len"] | None = None,
    *,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    continue_initial_state: bool = False,
    return_chunk_states: bool = False,
    precision: str = "highest",
    block_h: int = 1,
    backward: str = "pallas",
    bwd_reduce: str = "vpu",
) -> tuple[Array, ...]:
    """Chunked KDA with per-channel decay (differentiable through plain autodiff).

    Args:
        query: Queries ``[B, H, L, K]``.
        key: Keys ``[B, H, L, K]``.
        value: Values ``[B, H, L, V]``.
        beta: Delta-rule gates ``[B, H, L]``.
        decay: Per-token per-channel log-decay ``[B, H, L, K]`` (``<= 0``).
        initial_state: Optional state ``[B, H, K, V]``.
        segment_ids: Optional packed-row ids ``[B, L]``; the state resets when the id changes and negative ids are
            padding (zero output, no state change in continuation mode).
        chunk_size: Chunk length.
        use_qk_l2norm: L2-normalise queries and keys first.
        continue_initial_state: With ``segment_ids``: padding does not break a run and the first live run inherits
            ``initial_state`` (cached continuation). ``False``: the first token starts from zero (packed training).
        return_chunk_states: Also return the state after every chunk ``[B, C, H, K, V]``.
        precision: ``"highest"`` or ``"default"`` for the non-inverse dots.
        block_h: Pallas tuning knob (heads per grid step); ignored here.
        backward: Pallas backward selector; validated, ignored here.
        bwd_reduce: Pallas backward reduction mode; validated, ignored here.

    Returns:
        ``(out [B, H, L, V] fp32, final_state [B, H, K, V] fp32)`` (+ ``chunk_states``).
    """
    validate_options(chunk_size, precision, backward, bwd_reduce)
    if int(block_h) <= 0:
        raise ValueError(f"block_h must be positive, got {block_h}")
    return kda_chunk_xla(
        query,
        key,
        value,
        beta,
        decay,
        initial_state,
        segment_ids,
        chunk_size=int(chunk_size),
        use_qk_l2norm=use_qk_l2norm,
        continue_initial_state=continue_initial_state,
        return_chunk_states=return_chunk_states,
        precision=precision,
    )


__all__ = ("kda_chunk",)
