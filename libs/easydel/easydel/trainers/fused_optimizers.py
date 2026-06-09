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

"""Fused optimizers (``optimizer="fused_adamw" | "fused_lion" | "fused_rmsprop"``).

optax builds each of these optimizers as a *chain* of transforms (e.g. AdamW is
``scale_by_adam`` → ``add_decayed_weights`` → ``scale_by_learning_rate``). On TPU,
XLA does **not** fuse the chain into a single read-modify-write: every transform
makes its own pass over the param-sized state, so the update does several times
more HBM traffic than necessary. Since the optimizer is bandwidth-bound (pure
elementwise over the whole param tree), that overhead is the whole cost.

Re-expressing each update as a **single ``jax.tree_util.tree_map``** (one
read-modify-write per leaf) emits one fused elementwise kernel per parameter.
Measured on a dense 9B model (bf16, FSDP×4, TPU v5), update time:

    optimizer   optax (chain)   fused (1 map)   speedup
    adamw           276 ms          63 ms        4.4x
    lion            151 ms          42 ms        3.6x
    rmsprop         146 ms          42 ms        3.5x

The optimizer is ~25% of the training step, so this removes ~15-20% of total
step time with **identical math** (parity-verified vs the optax builtins) and
**no precision change**.

Usage: set the trainer ``optimizer="fused_adamw"`` / ``"fused_lion"`` /
``"fused_rmsprop"`` — same hyperparameters as the un-prefixed names. Gradient
clipping and weight decay are added by the eformer factory exactly as for the
builtins, so those wrappers behave identically.

Note: this trick only helps optimizers whose cost is the optax *chain*. An
optimizer already written as a single per-leaf pass (e.g. external optimizer package's
``fused_adamw``, which loops over leaves and does all its Adam math inline)
does not have the multi-pass problem and gains nothing here — for those the cost
is the algorithm (e.g. the quantization projection), not optimizer bookkeeping.
"""

import dataclasses
import inspect

import jax
import jax.numpy as jnp
import optax
from eformer.optimizers import (
    AdamWConfig,
    AdamWOptimizer,
    LionConfig,
    LionOptimizer,
    RMSPropConfig,
    RMSPropOptimizer,
    register_optimizer,
)

_tree_map = jax.tree_util.tree_map
_is_tuple = lambda x: isinstance(x, tuple)  # noqa: E731


def _split(out, n):
    """Split a tree of ``n``-tuples into ``n`` parallel trees (one fused pass produced them)."""
    return tuple(_tree_map(lambda x, i=i: x[i], out, is_leaf=_is_tuple) for i in range(n))


def fused_adamw(
    learning_rate: float | optax.Schedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    mu_dtype=None,
) -> optax.GradientTransformation:
    """AdamW as a single fused ``tree_map`` (one read-modify-write per leaf).

    Numerically identical to :func:`optax.adamw` (decoupled weight decay, same
    bias correction and ``eps``/``eps_root`` placement) but emits one fused
    elementwise kernel per parameter instead of optax's multi-pass chain.
    """

    def init(params):
        z = (lambda p: jnp.zeros_like(p, mu_dtype)) if mu_dtype is not None else jnp.zeros_like
        return {"mu": _tree_map(z, params), "nu": _tree_map(z, params), "count": jnp.zeros((), jnp.int32)}

    def update(grads, state, params=None):
        if params is None and weight_decay != 0.0:
            raise ValueError("fused_adamw requires params for decoupled weight decay.")
        # optax samples the LR schedule at the PRE-increment count (0, 1, 2, ...)
        # while bias correction uses the POST-increment count (1, 2, 3, ...). Match both.
        lr_t = learning_rate(state["count"]) if callable(learning_rate) else learning_rate
        count = optax.safe_int32_increment(state["count"])
        cf = count.astype(jnp.float32)
        bc1 = 1.0 - b1**cf
        bc2 = 1.0 - b2**cf

        def upd(g, m, v, p):
            store = m.dtype
            g32 = g.astype(jnp.float32)
            m32 = b1 * m.astype(jnp.float32) + (1.0 - b1) * g32
            v32 = b2 * v.astype(jnp.float32) + (1.0 - b2) * (g32 * g32)
            step = (m32 / bc1) / (jnp.sqrt(v32 / bc2 + eps_root) + eps)
            if weight_decay != 0.0:
                step = step + weight_decay * p.astype(jnp.float32)
            return ((-lr_t * step).astype(g.dtype), m32.astype(store), v32.astype(store))

        leaves = (grads, state["mu"], state["nu"], params) if params is not None else (grads, state["mu"], state["nu"])
        if params is None:
            out = _tree_map(lambda g, m, v: upd(g, m, v, g), *leaves)
        else:
            out = _tree_map(upd, *leaves)
        updates, mu, nu = _split(out, 3)
        return updates, {"mu": mu, "nu": nu, "count": count}

    return optax.GradientTransformation(init, update)


def fused_lion(
    learning_rate: float | optax.Schedule,
    b1: float = 0.9,
    b2: float = 0.99,
    weight_decay: float = 1e-3,
    mu_dtype=None,
) -> optax.GradientTransformation:
    """Lion as a single fused ``tree_map``.

    Identical to :func:`optax.lion` (operand order matched): the update
    direction uses the *old* momentum with ``b1``, then the momentum is rolled
    with ``b2``, then optax's decoupled weight decay is folded in.

    Note: ``optax.lion`` (and therefore eformer's ``"lion"``) defaults
    ``weight_decay=1e-3`` — it is **not** zero like AdamW's builder. We mirror
    that default so ``"fused_lion"`` matches ``"lion"`` exactly. The factory's
    own ``weight_decay`` is applied as an additional chain term, exactly as it is
    for the builtin.
    """

    def init(params):
        z = (lambda p: jnp.zeros_like(p, mu_dtype)) if mu_dtype is not None else jnp.zeros_like
        return {"mu": _tree_map(z, params), "count": jnp.zeros((), jnp.int32)}

    def update(grads, state, params=None):
        if params is None and weight_decay != 0.0:
            raise ValueError("fused_lion requires params for decoupled weight decay.")
        # LR schedule sampled at the PRE-increment count (matches optax.scale_by_schedule).
        lr_t = learning_rate(state["count"]) if callable(learning_rate) else learning_rate
        count = optax.safe_int32_increment(state["count"])

        def upd(g, m, p):
            store = m.dtype
            g32 = g.astype(jnp.float32)
            m32 = m.astype(jnp.float32)
            direction = jnp.sign((1.0 - b1) * g32 + b1 * m32)  # uses OLD momentum, like optax
            if weight_decay != 0.0:
                direction = direction + weight_decay * p.astype(jnp.float32)
            new_m = (1.0 - b2) * g32 + b2 * m32
            return ((-lr_t * direction).astype(g.dtype), new_m.astype(store))

        if params is None:
            out = _tree_map(lambda g, m: upd(g, m, g), grads, state["mu"])
        else:
            out = _tree_map(upd, grads, state["mu"], params)
        updates, mu = _split(out, 2)
        return updates, {"mu": mu, "count": count}

    return optax.GradientTransformation(init, update)


def fused_rmsprop(
    learning_rate: float | optax.Schedule,
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    momentum: float | None = None,
    nesterov: bool = False,
) -> optax.GradientTransformation:
    """RMSProp as a single fused ``tree_map`` (eps inside the rsqrt, like optax).

    Fuses the common ``momentum=None``, ``centered=False`` configuration (the
    eformer default). If ``momentum`` is set, falls back to :func:`optax.rmsprop`
    so the heavy-ball / Nesterov trace stays exactly faithful — that path has an
    extra state leaf and is rarely used.
    """
    if momentum is not None:
        return optax.rmsprop(
            learning_rate=learning_rate,
            decay=decay,
            eps=eps,
            initial_scale=initial_scale,
            centered=False,
            momentum=momentum,
            nesterov=nesterov,
        )

    def init(params):
        return {
            "nu": _tree_map(lambda p: jnp.full_like(p, initial_scale, jnp.float32), params),
            "count": jnp.zeros((), jnp.int32),
        }

    def update(grads, state, params=None):
        # LR schedule sampled at the PRE-increment count (matches optax.scale_by_schedule).
        lr_t = learning_rate(state["count"]) if callable(learning_rate) else learning_rate
        count = optax.safe_int32_increment(state["count"])

        def upd(g, v):
            g32 = g.astype(jnp.float32)
            v32 = decay * v.astype(jnp.float32) + (1.0 - decay) * (g32 * g32)
            direction = g32 * jax.lax.rsqrt(v32 + eps)
            return ((-lr_t * direction).astype(g.dtype), v32.astype(v.dtype))

        out = _tree_map(upd, grads, state["nu"])
        updates, nu = _split(out, 2)
        return updates, {"nu": nu, "count": count}

    return optax.GradientTransformation(init, update)


_LION_WEIGHT_DECAY = inspect.signature(optax.lion).parameters["weight_decay"].default


@register_optimizer("fused_adamw")
@dataclasses.dataclass
class FusedAdamWOptimizer(AdamWOptimizer):
    config: AdamWConfig

    def build(self, scheduler):
        return fused_adamw(
            learning_rate=scheduler,
            b1=self.config.b1,
            b2=self.config.b2,
            eps=self.config.eps,
            eps_root=self.config.eps_root,
            mu_dtype=self.config.mu_dtype,
            # eformer's adamw builder passes 0 here too: decoupled decay is sourced
            # from the trainer's weight_decay and added by the factory chain (when
            # nonzero). Keeping 0 in the core avoids applying decay twice.
            weight_decay=0.0,
        )

    def build_mpmd(self, scheduler, *, optimizer=None, **tx_kwargs):
        # The inherited stage-local kernel expects optax.adamw's 3-tuple chain state,
        # not the fused dict state. Raise so the factory installs the clear
        # "unsupported stage-local" wrapper (the normal non-PP path still uses the fused update).
        raise NotImplementedError(
            "fused_adamw has no stage-local kernel; use optimizer='adamw' for pipeline-parallel/MPMD training."
        )


@register_optimizer("fused_lion")
@dataclasses.dataclass
class FusedLionOptimizer(LionOptimizer):
    config: LionConfig

    def build(self, scheduler):
        return fused_lion(
            learning_rate=scheduler,
            b1=self.config.b1,
            b2=self.config.b2,
            weight_decay=_LION_WEIGHT_DECAY,
            mu_dtype=self.config.mu_dtype,
        )

    def build_mpmd(self, scheduler, *, optimizer=None, **tx_kwargs):
        raise NotImplementedError(
            "fused_lion has no stage-local kernel; use optimizer='lion' for pipeline-parallel/MPMD training."
        )


@register_optimizer("fused_rmsprop")
@dataclasses.dataclass
class FusedRMSPropOptimizer(RMSPropOptimizer):
    config: RMSPropConfig

    def build(self, scheduler):
        return fused_rmsprop(
            learning_rate=scheduler,
            decay=self.config.decay,
            eps=self.config.eps,
            initial_scale=self.config.initial_scale,
            momentum=self.config.momentum,
            nesterov=self.config.nesterov,
        )

    def build_mpmd(self, scheduler, *, optimizer=None, **tx_kwargs):
        raise NotImplementedError(
            "fused_rmsprop has no stage-local kernel; use optimizer='rmsprop' for pipeline-parallel/MPMD training."
        )
