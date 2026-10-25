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

"""Shared ``shard_map`` construction for the per-row fused distillation losses."""

from __future__ import annotations

from collections.abc import Callable
from functools import reduce
from operator import mul
from typing import Any

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec

from .fused_kl_divergence import _infer_leading_axes, _infer_vocab_axis


def build_row_loss_shard_map(
    run: Callable[..., Any],
    logit_args: tuple,
    weights,
    attention_mask,
    *,
    reduction: str,
    vocab_parallel_axis: str | None,
    run_kwargs: dict[str, Any],
    mesh: Mesh | None,
    in_specs: tuple[PartitionSpec | None, ...] | None,
    out_specs: PartitionSpec | None,
    check_vma: bool,
):
    """Wrap a per-row loss ``run`` in ``shard_map`` with automatic collectives.

    The last entry of ``in_specs[0]`` is the vocab-parallel mesh axis (unless
    ``vocab_parallel_axis`` is given). For ``"sum"`` / ``"mean"`` reductions the
    leading-axis mesh names (``in_specs[0][:-1]``) form the ``psum`` axes of the
    loss and of the weight count; each shard computes a local weighted sum and
    the global mean is formed once from the psum-ed sum and count.

    Args:
        run: Operation ``run`` bound method, called as
            ``run(*logit_args, weights, attention_mask=..., reduction=...,
            vocab_parallel_axis=..., **run_kwargs)``.
        logit_args: Logit tensors (one for entropy, student + teacher for KL-style).
        weights: Optional per-row weights, or ``None``.
        attention_mask: Optional extra mask folded into the weights, or ``None``.
        reduction: Outer reduction: ``"none"``, ``"sum"`` or ``"mean"``.
        vocab_parallel_axis: Explicit vocab mesh axis, or ``None`` to infer it.
        run_kwargs: Remaining keyword arguments forwarded to ``run``.
        mesh: Device mesh.
        in_specs: Specs for ``(*logit_args, [weights], [attention_mask])``.
        out_specs: Output spec.
        check_vma: Forwarded to ``shard_map``.

    Returns:
        ``(shard_map_fn, call_args)``.

    Raises:
        ValueError: If ``mesh`` / ``in_specs`` / ``out_specs`` is missing or
            ``in_specs`` is shorter than the call arguments.
    """
    if mesh is None or in_specs is None or out_specs is None:
        raise ValueError("mesh, in_specs and out_specs must be provided for shard_map execution")
    if vocab_parallel_axis is None:
        vocab_parallel_axis = _infer_vocab_axis(in_specs[0])

    first_spec = in_specs[0]
    leading_axes = _infer_leading_axes(PartitionSpec(*first_spec[:-1]) if first_spec is not None else None)

    call_args = tuple(logit_args)
    if weights is not None:
        call_args = (*call_args, weights)
    if attention_mask is not None:
        call_args = (*call_args, attention_mask)
    if len(in_specs) < len(call_args):
        raise ValueError(f"in_specs length {len(in_specs)} < number of call arguments {len(call_args)}")
    actual_in_specs = tuple(in_specs[: len(call_args)])

    n_logits = len(logit_args)
    has_weights = weights is not None
    has_mask = attention_mask is not None
    inner_reduction = "none" if reduction == "none" else "sum"

    def _per_device(*args):
        """Run one shard-local call and merge scalar reductions across the leading axes."""
        logits = args[:n_logits]
        rest = list(args[n_logits:])
        ws = rest.pop(0) if has_weights else None
        ms = rest.pop(0) if has_mask else None
        local_out = run(
            *logits,
            ws,
            attention_mask=ms,
            reduction=inner_reduction,
            vocab_parallel_axis=vocab_parallel_axis,
            **run_kwargs,
        )
        if reduction == "none":
            return local_out
        if ws is None:
            cnt_weights = jnp.ones((reduce(mul, logits[0].shape[:-1], 1),), dtype=jnp.float32)
        else:
            cnt_weights = ws.astype(jnp.float32).reshape(-1)
        if ms is not None:
            cnt_weights = cnt_weights * ms.astype(jnp.float32).reshape(-1)
        cnt_local = cnt_weights.sum()
        total = jax.lax.psum(local_out, leading_axes) if leading_axes else local_out
        cnt = jax.lax.psum(cnt_local, leading_axes) if leading_axes else cnt_local
        if reduction == "sum":
            return total
        return total / jnp.maximum(cnt, 1e-8)

    wrapped = shard_map(_per_device, mesh=mesh, in_specs=actual_in_specs, out_specs=out_specs, check_vma=check_vma)
    return wrapped, call_args


def resolve_shard_map_method(
    vocab_parallel_axis: str | None,
    mesh: Mesh | None,
    in_specs: tuple[PartitionSpec | None, ...] | None,
    out_specs: PartitionSpec | None,
    check_vma: bool,
) -> tuple[str | None, bool]:
    """Return ``(method, check_vma)`` for the executor call.

    ``method`` is ``"shard_map"`` when ``mesh`` / ``in_specs`` / ``out_specs`` are all given.
    Vocab-parallel gradients need varying-manual-axis checking, otherwise the per-shard
    normalization term is dropped from the VJP, so ``check_vma`` is forced on whenever a
    vocab axis is given or inferable from ``in_specs[0]``.
    """
    if mesh is None or in_specs is None or out_specs is None:
        return None, check_vma
    axis = vocab_parallel_axis if vocab_parallel_axis is not None else _infer_vocab_axis(in_specs[0])
    return "shard_map", (True if axis is not None else check_vma)
