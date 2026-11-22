"""Shared fused-divergence dispatch for offline and generalized distillation."""

import typing as tp

import jax
from ejkernel.modules.operations import fused_kl_divergence
from jax import numpy as jnp
from jax.sharding import PartitionSpec


def resolve_kl_direction(beta: float | None) -> str:
    """Resolve the EasyDeL/TRL beta convention to a fused-KL direction.

    Args:
        beta: None or nonpositive selects forward KL; values at least one
            select reverse KL; intermediate values select generalized JSD.

    Returns:
        The fused-kernel direction name.
    """
    if beta is None or beta <= 0.0:
        return "forward"
    if beta >= 1.0:
        return "reverse"
    return "jsd"


def fused_kl_vocab_parallel(
    student_logits,
    teacher_logits,
    weights=None,
    *,
    direction: str,
    temperature: float,
    beta: float = 0.5,
    return_teacher_entropy: bool = False,
    mesh: tp.Any = None,
    logit_spec: PartitionSpec | None = None,
    token_spec: PartitionSpec | None = None,
    platform: str | None = "xla",
):
    """Compute fused KL, padding token axes for vocab-parallel shard divisibility.

    Args:
        student_logits: Student logits with vocabulary on the last axis.
        teacher_logits: Frozen teacher logits of matching shape.
        weights: Optional token weights.
        direction: Forward, reverse, or JSD divergence.
        temperature: Softmax temperature.
        beta: Generalized-JSD mixture coefficient.
        return_teacher_entropy: Whether to return the teacher entropy split.
        mesh: Optional tensor-parallel mesh.
        logit_spec: Partition spec for logits on the mesh.
        token_spec: Partition spec for token weights on the mesh.
        platform: Kernel platform for the non-parallel path.

    Returns:
        Fused divergence output with a scalar masked-mean loss.
    """
    kwargs = {}
    if mesh is not None:
        pads = []
        for index, size in enumerate(student_logits.shape[:-1]):
            axes = token_spec[index] if index < len(token_spec) else None
            axes = axes if isinstance(axes, tuple) else (() if axes is None else (axes,))
            multiple = 1
            for axis in axes:
                multiple *= mesh.shape[axis]
            pads.append((-size) % multiple)
        if any(pads):
            if weights is None:
                weights = jnp.ones(student_logits.shape[:-1], dtype=jnp.float32)
            token_padding = tuple((0, pad) for pad in pads)
            student_logits = jnp.pad(student_logits, (*token_padding, (0, 0)))
            teacher_logits = jnp.pad(teacher_logits, (*token_padding, (0, 0)))
            weights = jnp.pad(weights, token_padding)
        kwargs = {
            "mesh": mesh,
            "in_specs": (logit_spec, logit_spec, token_spec),
            "out_specs": PartitionSpec(),
        }
        platform = "xla"
    return fused_kl_divergence(
        student_logits,
        jax.lax.stop_gradient(teacher_logits),
        weights,
        reduction="mean",
        direction=direction,
        temperature=temperature,
        beta=float(beta),
        return_teacher_entropy=return_teacher_entropy,
        platform=platform,
        **kwargs,
    )
