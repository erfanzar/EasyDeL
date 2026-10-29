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

"""TileLang Kernel Delta Attention (KDA) interface.

Registers ``kernel_delta_attention`` and ``kda`` under
``Platform.TILELANG / Backend.GPU``.  Both names share the same
implementation: the GDR recurrent kernel from
:func:`ejkernel.kernels._tilelang.gated_delta_rule._impl.delta_rule_tilelang`.

KDA is a variant of the Gated Delta Rule where the softmax scale can be
explicitly controlled via ``softmax_scale`` instead of using the default
``1 / sqrt(qk_head_dim)``.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ..gated_delta_rule._impl import delta_rule_tilelang


def _impl_kda(
    query,
    key,
    value,
    beta,
    decay,
    softmax_scale,
    chunk_size,
    initial_state,
    use_qk_l2norm,
    use_chunked,
):
    """Validate KDA-specific arguments and dispatch to the GDR kernel.

    Args:
        query: ``[batch, seq_len, num_heads, qk_head_dim]`` float tensor.
        key: ``[batch, seq_len, num_heads, qk_head_dim]`` float tensor.
        value: ``[batch, seq_len, num_heads, v_head_dim]`` float tensor.
        beta: ``[batch, seq_len, num_heads]`` update gate.
        decay: Optional ``[batch, seq_len, num_heads]`` log-decay.
        softmax_scale: Optional scalar scale; defaults to
            ``1 / sqrt(qk_head_dim)`` inside
            :func:`~ejkernel.kernels._tilelang.gated_delta_rule._impl.delta_rule_tilelang`.
        chunk_size: Accepted for API compatibility; has no effect on the
            TileLang kernel.
        initial_state: Optional float32 ``[batch, num_heads, qk_head_dim,
            v_head_dim]`` initial state.
        use_qk_l2norm: Whether to L2-normalise queries and keys.
        use_chunked: Accepted for API compatibility; has no effect.

    Returns:
        ``(output, final_state)`` — same contract as
        :func:`~ejkernel.kernels._tilelang.gated_delta_rule._impl.delta_rule_tilelang`.

    Raises:
        EjkernelRuntimeError: If ``chunk_size <= 0`` or ``use_chunked`` is
            not a bool.
    """
    if chunk_size <= 0:
        raise EjkernelRuntimeError("tile-lang kernel_delta_attention requires chunk_size > 0.")
    if not isinstance(use_chunked, bool):
        raise EjkernelRuntimeError("tile-lang kernel_delta_attention requires use_chunked to be a bool.")
    return delta_rule_tilelang(
        query,
        key,
        value,
        beta,
        decay,
        initial_state=initial_state,
        softmax_scale=softmax_scale,
        use_qk_l2norm=use_qk_l2norm,
    )


@kernel_registry.register("kernel_delta_attention", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def kernel_delta_attention(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_heads v_head_dim"],
    beta: Float[Array, "batch seq_len num_heads"],
    decay: Float[Array, "batch seq_len num_heads"] | None = None,
    *,
    softmax_scale: float | None = None,
    chunk_size: int = 64,
    initial_state: Float[Array, "batch num_heads qk_head_dim v_head_dim"] | None = None,
    use_qk_l2norm: bool = True,
    use_chunked: bool = True,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "batch num_heads qk_head_dim v_head_dim"],
]:
    """Run Kernel Delta Attention (KDA) on GPU via TileLang.

    KDA is functionally equivalent to the Gated Delta Rule with an explicit
    ``softmax_scale`` argument.  Both ``chunk_size`` and ``use_chunked`` are
    accepted for API compatibility with the XLA backend but have no effect
    on the TileLang kernel.

    Args:
        query: ``[batch, seq_len, num_heads, qk_head_dim]`` float tensor.
        key: ``[batch, seq_len, num_heads, qk_head_dim]`` float tensor.
        value: ``[batch, seq_len, num_heads, v_head_dim]`` float tensor.
        beta: ``[batch, seq_len, num_heads]`` per-timestep update gate.
        decay: Optional ``[batch, seq_len, num_heads]`` log-decay.  Pass
            ``None`` to disable forgetting.
        softmax_scale: Optional scale applied to query vectors.  Defaults to
            ``1 / sqrt(qk_head_dim)``.
        chunk_size: Accepted for API compatibility; ignored by this backend.
        initial_state: Optional float32
            ``[batch, num_heads, qk_head_dim, v_head_dim]`` initial state.
        use_qk_l2norm: Whether to L2-normalise queries and keys inside the
            kernel.
        use_chunked: Accepted for API compatibility; ignored by this backend.

    Returns:
        ``(output, final_state)`` — shapes
        ``[batch, seq_len, num_heads, v_head_dim]`` and
        ``[batch, num_heads, qk_head_dim, v_head_dim]`` (float32).

    Raises:
        EjkernelRuntimeError: If ``chunk_size <= 0`` or ``use_chunked`` is
            not a bool.
    """
    return _impl_kda(
        query,
        key,
        value,
        beta,
        decay,
        softmax_scale,
        chunk_size,
        initial_state,
        use_qk_l2norm,
        use_chunked,
    )


@kernel_registry.register("kda", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def kda(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_heads v_head_dim"],
    beta: Float[Array, "batch seq_len num_heads"],
    decay: Float[Array, "batch seq_len num_heads"] | None = None,
    *,
    softmax_scale: float | None = None,
    chunk_size: int = 64,
    initial_state: Float[Array, "batch num_heads qk_head_dim v_head_dim"] | None = None,
    use_qk_l2norm: bool = True,
    use_chunked: bool = True,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "batch num_heads qk_head_dim v_head_dim"],
]:
    """Short alias for :func:`kernel_delta_attention` — identical behaviour.

    Registered separately under ``"kda"`` in the TileLang kernel registry so
    that callers using the abbreviated name get the same GPU kernel.

    Args:
        query: ``[batch, seq_len, num_heads, qk_head_dim]`` float tensor.
        key: ``[batch, seq_len, num_heads, qk_head_dim]`` float tensor.
        value: ``[batch, seq_len, num_heads, v_head_dim]`` float tensor.
        beta: ``[batch, seq_len, num_heads]`` per-timestep update gate.
        decay: Optional ``[batch, seq_len, num_heads]`` log-decay.
        softmax_scale: Optional scale for query vectors.
        chunk_size: Accepted for API compatibility; ignored.
        initial_state: Optional float32 initial hidden state.
        use_qk_l2norm: Whether to L2-normalise queries and keys.
        use_chunked: Accepted for API compatibility; ignored.

    Returns:
        ``(output, final_state)`` — same shapes as
        :func:`kernel_delta_attention`.

    Raises:
        EjkernelRuntimeError: If ``chunk_size <= 0`` or ``use_chunked`` is
            not a bool.
    """
    return _impl_kda(
        query,
        key,
        value,
        beta,
        decay,
        softmax_scale,
        chunk_size,
        initial_state,
        use_qk_l2norm,
        use_chunked,
    )


__all__ = ["kda", "kernel_delta_attention"]
