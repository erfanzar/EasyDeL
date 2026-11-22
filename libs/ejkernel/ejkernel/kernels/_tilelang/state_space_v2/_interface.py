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

"""Tile-lang SSM-2 (Mamba2) forward.

Stacked registration covers ``state_space_v2``, ``mamba2`` and ``ssm2``
on ``Platform.TILELANG`` — all three route through the same native kernel.
"""

from collections.abc import Callable

import jax
import jaxtyping
from beartype import beartype
from jax import lax
from jaxtyping import Array, Float

from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from .._gate_impl import rmsnorm_silu_gate_tilelang, silu_gate_tilelang
from ._impl import ssm2_tilelang


def _check_silu_act_fn(act_fn):
    if act_fn is not None and act_fn not in (jax.nn.silu, jax.nn.swish):
        name = getattr(act_fn, "__name__", repr(act_fn))
        raise EjkernelRuntimeError(f"tile-lang state_space_v2 only supports silu output gating, got {name}.")


@kernel_registry.register("state_space_v2", Platform.TILELANG, Backend.GPU)
@kernel_registry.register("ssm2", Platform.TILELANG, Backend.GPU)
@kernel_registry.register("mamba2", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def state_space_v2(
    x: Float[Array, "batch seq_len num_heads head_dim"],
    A: Float[Array, "num_heads"],
    B: Float[Array, "batch seq_len n_groups ssm_state_size"],
    C: Float[Array, "batch seq_len n_groups ssm_state_size"],
    D: Float[Array, "num_heads"],
    dt: Float[Array, "batch seq_len num_heads"],
    gate: Float[Array, "batch seq_len intermediate_size"] | None = None,
    initial_state: Float[Array, "batch num_heads head_dim ssm_state_size"] | None = None,
    conv_state: Float[Array, "batch conv_dim d_conv"] | None = None,
    n_groups: int = 1,
    act_fn: Callable[[Array], Array] | None = None,
    use_gated_rmsnorm: bool = False,
    rmsnorm_eps: float = 1e-5,
    precision: lax.Precision | None = None,
    *,
    block_e: int = 128,
) -> tuple[
    Float[Array, "batch seq_len intermediate_size"],
    Float[Array, "batch num_heads head_dim ssm_state_size"],
    Float[Array, "batch conv_dim d_conv"] | None,
]:
    """Tile-lang SSM-2 (Mamba2) forward (and differentiable backward).

    Registered as ``"state_space_v2"``, ``"ssm2"``, and ``"mamba2"`` on
    ``Platform.TILELANG / Backend.GPU``.

    ``conv_state`` is passed through unchanged (the convolutional state is
    not updated here; update it upstream before calling this function).

    Optional output gating modes:

    * ``gate`` + ``act_fn=silu`` + ``use_gated_rmsnorm=False``:
      applies ``y = silu_gate(y, gate)``.
    * ``gate`` + ``act_fn=silu`` + ``use_gated_rmsnorm=True``:
      applies ``y = rmsnorm_silu_gate(y, gate, rmsnorm_eps)``.
    * ``gate=None``: no gating applied.

    Only ``jax.nn.silu`` / ``jax.nn.swish`` are accepted as ``act_fn`` when
    ``gate`` is not ``None``.

    Args:
        x: input ``(batch, seq_len, num_heads, head_dim)``.
        A: per-head state-transition log-eigenvalue, ``(num_heads,)``.
        B: projected B input, ``(batch, seq_len, n_groups, ssm_state_size)``.
        C: projected C input, same shape as ``B``.
        D: per-head skip-connection scale, ``(num_heads,)``.
        dt: per-head time-delta, ``(batch, seq_len, num_heads)``.
        gate: optional silu-gate tensor ``(batch, seq_len, intermediate_size)``.
        initial_state: optional fp32 initial hidden state
            ``(batch, num_heads, head_dim, ssm_state_size)``; defaults to zeros.
        conv_state: optional convolutional state (pass-through, not modified).
        n_groups: number of B/C groups ``G``; must equal ``B.shape[-2]``.
        act_fn: activation for the output gate; only silu/swish accepted.
        use_gated_rmsnorm: if ``True`` apply RMSNorm before gating
            (requires ``gate`` to be provided).
        rmsnorm_eps: epsilon for RMSNorm when ``use_gated_rmsnorm=True``
            (default 1e-5).
        precision: not supported — raises ``EjkernelRuntimeError`` if given.

    Returns:
        ``(y, hf, conv_state)`` — ``y`` is ``(batch, seq_len, num_heads * head_dim)``
        in the input dtype; ``hf`` is fp32
        ``(batch, num_heads, head_dim, ssm_state_size)``; ``conv_state`` is
        returned unchanged.

    Raises:
        EjkernelRuntimeError: if ``n_groups`` mismatches ``B.shape[-2]``,
            ``num_heads`` is not divisible by ``n_groups``, ``precision`` is
            provided, or ``act_fn`` is not silu/swish.
    """
    if n_groups != B.shape[-2]:
        raise EjkernelRuntimeError("tile-lang state_space_v2 requires n_groups to match B/C shape.")
    if precision is not None:
        raise EjkernelRuntimeError("tile-lang state_space_v2 v0 does not yet support custom precision.")

    num_heads = x.shape[2]
    if num_heads % n_groups != 0:
        raise EjkernelRuntimeError(f"num_heads={num_heads} must be divisible by n_groups={n_groups}.")

    y, hf = ssm2_tilelang(x, A, B, C, D, dt, initial_state=initial_state)
    B_, S_, H_, P_ = y.shape
    y_flat = y.reshape(B_, S_, H_ * P_)
    if gate is not None:
        _check_silu_act_fn(act_fn)
        if use_gated_rmsnorm:
            y_flat = rmsnorm_silu_gate_tilelang(y_flat, gate, float(rmsnorm_eps))
        else:
            y_flat = silu_gate_tilelang(y_flat, gate, int(block_e))
    return y_flat, hf, conv_state


__all__ = ["state_space_v2"]
