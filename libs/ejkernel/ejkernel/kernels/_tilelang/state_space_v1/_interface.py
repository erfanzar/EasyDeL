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

"""Tile-lang SSM-1 (Mamba) forward.

Registered for three names that the XLA reference aliases together:
``state_space_v1``, ``mamba1``, ``ssm1``. Each invocation routes through
the same native tile-lang kernel ([_kernel.py](_kernel.py)).

Optional output gating uses a native TileLang silu-gate kernel with a
native VJP. ``conv_state`` is returned unchanged, matching the XLA
reference cache pass-through behavior.
"""

from collections.abc import Callable

import jax
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from .._gate_impl import silu_gate_tilelang
from ._impl import ssm1_tilelang


def _check_silu_act_fn(act_fn):
    if act_fn is not None and act_fn not in (jax.nn.silu, jax.nn.swish):
        name = getattr(act_fn, "__name__", repr(act_fn))
        raise EjkernelRuntimeError(f"tile-lang state_space_v1 only supports silu output gating, got {name}.")


def _impl(
    hidden_states,
    A,
    B,
    C,
    D,
    dt,
    gate,
    initial_state,
    conv_state,
    act_fn,
    *,
    block_d: int,
    block_e: int,
):
    y, hf = ssm1_tilelang(hidden_states, A, B, C, D, dt, initial_state=initial_state, block_d=int(block_d))
    if gate is not None:
        _check_silu_act_fn(act_fn)
        y = silu_gate_tilelang(y, gate, int(block_e))
    return y, hf, conv_state


@kernel_registry.register("state_space_v1", Platform.TILELANG, Backend.GPU)
@kernel_registry.register("ssm1", Platform.TILELANG, Backend.GPU)
@kernel_registry.register("mamba1", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def state_space_v1(
    hidden_states: Float[Array, "batch seq_len intermediate_size"],
    A: Float[Array, "intermediate_size ssm_state_size"],
    B: Float[Array, "batch seq_len ssm_state_size"],
    C: Float[Array, "batch seq_len ssm_state_size"],
    D: Float[Array, "intermediate_size"],
    dt: Float[Array, "batch seq_len intermediate_size"],
    gate: Float[Array, "batch seq_len intermediate_size"] | None = None,
    initial_state: Float[Array, "batch intermediate_size ssm_state_size"] | None = None,
    conv_state: Float[Array, "batch intermediate_size d_conv"] | None = None,
    act_fn: Callable[[Array], Array] | None = None,
    *,
    block_d: int = 64,
    block_e: int = 128,
) -> tuple[
    Float[Array, "batch seq_len intermediate_size"],
    Float[Array, "batch intermediate_size ssm_state_size"],
    Float[Array, "batch intermediate_size d_conv"] | None,
]:
    """Tile-lang SSM-1 (Mamba selective scan) forward.

    Registered as ``"state_space_v1"``, ``"ssm1"``, and ``"mamba1"`` on
    ``Platform.TILELANG / Backend.GPU``.

    ``conv_state`` is passed through unchanged (the convolutional state is not
    updated here; update it upstream before calling this function).

    Optional output gating with ``gate`` applies a silu gate:
    ``y = silu_gate(y, gate)``.  Only ``jax.nn.silu`` / ``jax.nn.swish`` are
    accepted; other activations raise ``EjkernelRuntimeError``.

    Args:
        hidden_states: ``(batch, seq_len, intermediate_size)``.
        A: state-transition log-eigenvalues, ``(intermediate_size, ssm_state_size)``.
        B: projected B input, ``(batch, seq_len, ssm_state_size)``.
        C: projected C input, ``(batch, seq_len, ssm_state_size)``.
        D: skip-connection scale, ``(intermediate_size,)``.
        dt: time-delta, ``(batch, seq_len, intermediate_size)``.
        gate: optional silu-gate tensor ``(batch, seq_len, intermediate_size)``.
        initial_state: optional fp32 initial hidden state
            ``(batch, intermediate_size, ssm_state_size)``; defaults to zeros.
        conv_state: optional convolutional state (pass-through, not modified).
        act_fn: activation function for the output gate; only ``jax.nn.silu``
            / ``jax.nn.swish`` are accepted (or ``None`` when ``gate`` is
            ``None``).

    Returns:
        ``(y, hf, conv_state)`` — ``y`` is ``(batch, seq_len, intermediate_size)``
        in the input dtype; ``hf`` is fp32
        ``(batch, intermediate_size, ssm_state_size)``; ``conv_state`` is
        returned unchanged.

    Raises:
        EjkernelRuntimeError: if ``act_fn`` is not silu/swish.
    """
    return _impl(
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        gate,
        initial_state,
        conv_state,
        act_fn,
        block_d=int(block_d),
        block_e=int(block_e),
    )


__all__ = ["state_space_v1"]
