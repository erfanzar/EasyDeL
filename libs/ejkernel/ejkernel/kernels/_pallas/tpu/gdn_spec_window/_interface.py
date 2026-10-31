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

"""TPU Pallas interface for the GDN speculative-window state scan."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Bool, Float

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import gdn_spec_window_states as _gdn_spec_window_states_pallas


@kernel_registry.register("gdn_spec_window_states", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def gdn_spec_window_states(
    mixed_qkv: Float[Array, "batch num_steps qkv_dim"],
    b: Float[Array, "batch num_steps num_v_heads"],
    a: Float[Array, "batch num_steps num_v_heads"],
    A_log: Float[Array, "num_v_heads"],
    dt_bias: Float[Array, "num_v_heads"],
    step_valid: Bool[Array, "batch num_steps"],
    recurrent_state: Float[Array, "batch num_v_heads key_dim value_dim"],
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    apply_silu: bool = False,
) -> Float[Array, "num_steps batch num_v_heads key_dim value_dim"]:
    """Run the GDN speculative-window state scan via the fused TPU Pallas kernel.

    Registered with ``kernel_registry`` under ``"gdn_spec_window_states"`` for
    the Pallas platform / TPU backend. Input preparation is shared with the
    XLA reference; only the sequential state recurrence is fused into a
    single VMEM-resident kernel.

    Args:
        mixed_qkv: Post-conv packed rows ``[batch, num_steps, qkv_dim]``.
        b: Raw beta gate rows ``[batch, num_steps, n_v]``.
        a: Raw alpha gate rows ``[batch, num_steps, n_v]``.
        A_log: Per-value-head log-decay parameter ``[n_v]``.
        dt_bias: Per-value-head delta-time bias ``[n_v]``.
        step_valid: ``[batch, num_steps]`` per-step validity mask.
        recurrent_state: Window-entry state ``[batch, n_v, d_k, d_v]``.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Key/query head dim.
        d_v: Value head dim.
        apply_silu: Apply SiLU to ``mixed_qkv`` before slicing.

    Returns:
        ``[num_steps, batch, n_v, d_k, d_v]`` per-prefix recurrent states.
    """
    return _gdn_spec_window_states_pallas(
        mixed_qkv,
        b,
        a,
        A_log,
        dt_bias,
        step_valid,
        recurrent_state,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        apply_silu=apply_silu,
    )
