# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":func:`spectrax.runtime.primitives.boundary`: inline stage-break marker.

Placing ``boundary(x)`` inside a module's ``forward``
declares a pipeline stage split at that point: every op before the
first boundary belongs to stage 0, between the first and second
boundary to stage 1, and so on.

Outside a pipeline context, :func:`boundary` is the identity function
— modules that use it still run correctly on a single device, which
makes it safe to litter your model with boundary hints before you've
decided how to parallelize.

Under the spectrax pipeline runtimes, the primary stage-break
contract is the explicit :class:`~spectrax.nn.PipelineSequential`
container: each element of its ``stages`` is a distinct pipeline stage.
The :func:`boundary` primitive is a convenience marker for models that
aren't structured as a flat sequence of modules; it's preserved in the
traced jaxpr so future versions (v2) can split mid-forward at boundary
calls. In the current release it lowers to an identity and is silently
ignored by the SPMD orchestrator.
"""

from __future__ import annotations

import jax

__all__ = ["boundary"]


@jax.custom_jvp
def _boundary(x: object) -> object:
    """Identity primitive that survives jaxpr lowering as a named op.

    Decorating with :func:`jax.custom_jvp` keeps the call from being
    constant-folded or otherwise erased during tracing — the jaxpr
    retains a distinct ``custom_jvp_call`` site that downstream
    pipeline passes can recognise and use as a stage-split anchor.

    Args:
        x: object JAX-compatible value.

    Returns:
        ``x`` unchanged.
    """
    return x


@_boundary.defjvp
def _boundary_jvp(primals: tuple[object, ...], tangents: tuple[object, ...]) -> tuple[object, object]:
    """JVP rule for :func:`_boundary` — pass primal and tangent through.

    Because :func:`_boundary` is mathematically the identity, the
    correct tangent rule is also identity. Defining it explicitly
    (rather than letting JAX infer one) keeps the marker visible in
    both the forward and the linearised jaxpr.

    Args:
        primals: One-element tuple ``(x,)``.
        tangents: One-element tuple ``(t,)``.

    Returns:
        ``(x, t)`` — primal and tangent forwarded unchanged.
    """
    (x,) = primals
    (t,) = tangents
    return x, t


def boundary(x: object) -> object:
    """Mark a pipeline stage boundary inside a ``forward`` method.

    Outside a pipeline context this is a no-op identity function, so
    modules peppered with ``boundary`` calls still run unchanged on a
    single device. Under the spectrax pipeline runtimes, a future
    jaxpr-splitting pass (v2) will cut the forward pass at each
    boundary and lower each piece onto a separate mesh shard; the
    current SPMD orchestrator uses
    :class:`~spectrax.nn.PipelineSequential` as the canonical
    source of stage structure and treats ``boundary`` as advisory.

    Args:
        x: object JAX-compatible value. Passed through unchanged.

    Returns:
        ``x`` unmodified. The call is preserved in the jaxpr as an
        identity named op so it can be introspected by transforms.
    """
    return _boundary(x)
