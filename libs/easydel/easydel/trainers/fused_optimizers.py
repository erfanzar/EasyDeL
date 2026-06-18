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

"""EasyDeL-local optimizer registrations.

Includes fused variants (``optimizer="fused_adamw" | "fused_lion" | "fused_rmsprop"``)
and research optimizers/wrappers that live at the EasyDeL layer rather than in
the shared eFormer optimizer surface.

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
optimizer already written as a single per-leaf pass does not have the multi-pass
problem and gains nothing here — for those the cost is the algorithm, not
optimizer bookkeeping.
"""

import dataclasses
import inspect
import math
import typing as tp
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from eformer.optimizers import (
    AdamWConfig,
    AdamWOptimizer,
    LionConfig,
    LionOptimizer,
    MuonConfig,
    OptimizerBuilder,
    RMSPropConfig,
    RMSPropOptimizer,
    SerializationMixin,
    register_optimizer,
)
from eformer.optimizers._tx import mars as eformer_mars

_tree_map = jax.tree_util.tree_map


def _is_tuple(value: tp.Any) -> bool:
    return isinstance(value, tuple)


def _update_accepts_extra_args(update_fn: tp.Callable[..., tp.Any], extra_args: tp.Mapping[str, tp.Any]) -> bool:
    if not extra_args:
        return False
    try:
        parameters = inspect.signature(update_fn).parameters
    except (TypeError, ValueError):
        return False
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return True
    return all(key in parameters for key in extra_args)


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


def _zeros_like_with_optional_dtype(array: jax.Array, dtype: tp.Any | None) -> jax.Array:
    return jnp.zeros_like(array, dtype=dtype) if dtype is not None else jnp.zeros_like(array)


def _full_like_with_optional_dtype(array: jax.Array, value: float, dtype: tp.Any | None) -> jax.Array:
    return jnp.full_like(array, value, dtype=dtype) if dtype is not None else jnp.full_like(array, value)


def _compute_dtype(array: jax.Array) -> tp.Any:
    return jnp.complex64 if jnp.issubdtype(array.dtype, jnp.complexfloating) else jnp.float32


def _abs_sq(array: jax.Array) -> jax.Array:
    return jnp.real(array * jnp.conj(array))


def _mars_coeff(mars_gamma: float, mars_beta: float) -> float:
    return mars_gamma * mars_beta / max(1.0 - mars_beta, 1e-8)


class FusedAdaptiveState(NamedTuple):
    mu: optax.Updates
    nu: optax.Updates
    count: jax.Array
    prev_grad: optax.Updates | None


class ScheduleFreePlusState(NamedTuple):
    z: optax.Params
    x: optax.Params
    mu: optax.Updates
    nu: optax.Updates
    l1_ema: jax.Array
    weight_sum: jax.Array
    lr_max: jax.Array
    count: jax.Array


class ScheduleFreePlusHyperballState(NamedTuple):
    z: optax.Params
    x: optax.Params
    mu: optax.Updates
    nu: optax.Updates
    l1_ema: jax.Array
    weight_sum: jax.Array
    lr_max: jax.Array
    radii: optax.Params
    mask: optax.Params
    count: jax.Array


def _is_array_like(value: tp.Any) -> bool:
    return hasattr(value, "dtype") and hasattr(value, "shape")


def _tree_scalar_sum(tree, leaf_fn) -> jax.Array:
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in jtu.tree_leaves(tree):
        if _is_array_like(leaf):
            total = total + leaf_fn(jnp.asarray(leaf)).astype(jnp.float32)
    return total


def _tree_size(tree) -> int:
    return sum(int(leaf.size) for leaf in jtu.tree_leaves(tree) if _is_array_like(leaf))


_SCHEDULE_FREE_PLUS_L1_SCALE: tp.Final[float] = math.sqrt(math.pi / 2.0)


def _tree_inner_product(left, right) -> jax.Array:
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for left_leaf, right_leaf in zip(jtu.tree_leaves(left), jtu.tree_leaves(right), strict=True):
        if _is_array_like(left_leaf) and _is_array_like(right_leaf):
            work_dtype = jnp.promote_types(_compute_dtype(left_leaf), _compute_dtype(right_leaf))
            total = total + jnp.real(
                jnp.vdot(
                    jnp.asarray(left_leaf, dtype=work_dtype),
                    jnp.asarray(right_leaf, dtype=work_dtype),
                )
            ).astype(jnp.float32)
    return total


def _path_key_to_str(key: tp.Any) -> str:
    if hasattr(key, "name"):
        return str(key.name)
    if hasattr(key, "key"):
        return str(key.key)
    if hasattr(key, "idx"):
        return str(key.idx)
    return str(key)


def _path_to_str(path: tuple[tp.Any, ...]) -> str:
    return ".".join(_path_key_to_str(key) for key in path)


_HYPERBALL_EXCLUDED_PATH_TOKENS: tp.Final[tuple[str, ...]] = (
    "embed",
    "embedding",
    "wte",
    "wpe",
    "norm",
    "layernorm",
    "layer_norm",
    "rmsnorm",
    "ln_",
)
_MUON_HYPERBALL_ADAMH_PATH_TOKENS: tp.Final[tuple[str, ...]] = (
    "lm_head",
    "output_projection",
    "final_logits",
)


def _path_contains_any(path_str: str, tokens: tuple[str, ...]) -> bool:
    path_lower = path_str.lower()
    return any(token in path_lower for token in tokens)


def _is_default_hyperball_matrix(path_str: str, param: tp.Any, min_param_ndim: int) -> bool:
    return (
        _is_array_like(param)
        and getattr(param, "ndim", 0) >= min_param_ndim
        and not _path_contains_any(path_str, _HYPERBALL_EXCLUDED_PATH_TOKENS)
    )


def _is_default_muon_hyperball_matrix(path_str: str, param: tp.Any) -> bool:
    return (
        _is_default_hyperball_matrix(path_str, param, 2)
        and getattr(param, "ndim", 0) == 2
        and not _path_contains_any(path_str, _MUON_HYPERBALL_ADAMH_PATH_TOKENS)
    )


def fused_novograd(
    learning_rate: float | optax.Schedule,
    b1: float = 0.9,
    b2: float = 0.25,
    eps: float = 1e-6,
    eps_root: float = 0.0,
    weight_decay: float | optax.Schedule = 0.0,
    mu_dtype: tp.Any | None = None,
    nu_dtype: tp.Any | None = None,
    norm_mode: str = "sum",
    decoupled_weight_decay: bool = False,
    nesterov: bool = False,
    mars_gamma: float = 0.0,
    mars_beta: float | None = None,
    mars_prev_grad_dtype: tp.Any = jnp.bfloat16,
    grad_averaging: bool = True,
) -> optax.GradientTransformation:
    if norm_mode not in ("sum", "mean"):
        raise ValueError(f"Unsupported NovoGrad norm_mode: {norm_mode!r}.")

    use_mars = mars_gamma > 0.0
    mars_beta_value = b1 if mars_beta is None else mars_beta
    mars_c = _mars_coeff(mars_gamma, mars_beta_value)
    weight_decay_is_schedule = callable(weight_decay)
    has_weight_decay = weight_decay_is_schedule or weight_decay != 0.0

    def init(params):
        def init_nu(param):
            dtype = param.dtype if nu_dtype is None else nu_dtype
            return jnp.asarray(0.0, dtype=dtype)

        mu = _tree_map(lambda param: _zeros_like_with_optional_dtype(param, mu_dtype), params)
        nu = _tree_map(init_nu, params)
        prev_grad = None
        if use_mars:
            prev_grad = _tree_map(lambda param: jnp.zeros_like(param, dtype=mars_prev_grad_dtype), params)
        return FusedAdaptiveState(mu=mu, nu=nu, count=jnp.zeros((), jnp.int32), prev_grad=prev_grad)

    def update(grads, state, params=None):
        if params is None:
            if has_weight_decay:
                raise ValueError("fused_novograd requires params when weight_decay is nonzero.")
            params = _tree_map(jnp.zeros_like, grads)

        old_count = state.count
        count = optax.safe_int32_increment(old_count)
        is_first_step = count == jnp.asarray(1, dtype=count.dtype)
        lr_t = learning_rate(old_count) if callable(learning_rate) else learning_rate
        wd_t = weight_decay(old_count) if weight_decay_is_schedule else weight_decay

        def upd(grad, momentum, variance, param, prev_grad):
            momentum_store_dtype = momentum.dtype
            variance_store_dtype = variance.dtype
            grad_work = grad.astype(_compute_dtype(grad))
            if use_mars:
                if prev_grad is None:
                    raise ValueError("MARS state is missing prev_grad.")
                corrected = grad_work + jnp.asarray(mars_c, dtype=grad_work.dtype) * (
                    grad_work - prev_grad.astype(_compute_dtype(grad))
                )
                next_prev = grad.astype(mars_prev_grad_dtype)
            else:
                corrected = grad_work
                next_prev = None

            norm_sq = jnp.sum(_abs_sq(corrected))
            if norm_mode == "mean":
                norm_sq = norm_sq / jnp.asarray(corrected.size, dtype=norm_sq.dtype)

            variance32 = variance.astype(jnp.float32)
            variance_new = jnp.where(is_first_step, norm_sq, b2 * variance32 + (1.0 - b2) * norm_sq)
            denom = jnp.sqrt(variance_new + jnp.asarray(eps_root, dtype=variance_new.dtype)) + jnp.asarray(
                eps,
                dtype=variance_new.dtype,
            )
            normalized = corrected / denom
            if has_weight_decay and not decoupled_weight_decay:
                normalized = normalized + jnp.asarray(wd_t, dtype=normalized.dtype) * param.astype(normalized.dtype)

            momentum_work = momentum.astype(_compute_dtype(grad))
            momentum_new = jnp.where(is_first_step, normalized, b1 * momentum_work + (normalized if not grad_averaging else (1.0 - b1) * normalized))
            direction = normalized + b1 * momentum_new if nesterov else momentum_new
            if has_weight_decay and decoupled_weight_decay:
                direction = direction + jnp.asarray(wd_t, dtype=direction.dtype) * param.astype(direction.dtype)
            update_leaf = (-jnp.asarray(lr_t, dtype=direction.dtype) * direction).astype(param.dtype)
            return update_leaf, momentum_new.astype(momentum_store_dtype), variance_new.astype(variance_store_dtype), next_prev

        if use_mars:
            out = _tree_map(upd, grads, state.mu, state.nu, params, state.prev_grad)
        else:
            out = _tree_map(lambda g, m, v, p: upd(g, m, v, p, None), grads, state.mu, state.nu, params)
        updates = _tree_map(lambda value: value[0], out, is_leaf=_is_tuple)
        mu = _tree_map(lambda value: value[1], out, is_leaf=_is_tuple)
        nu = _tree_map(lambda value: value[2], out, is_leaf=_is_tuple)
        prev_grad = _tree_map(lambda value: value[3], out, is_leaf=_is_tuple) if use_mars else None
        return updates, FusedAdaptiveState(mu=mu, nu=nu, count=count, prev_grad=prev_grad)

    return optax.GradientTransformation(init, update)


def fused_yogi(
    learning_rate: float | optax.Schedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-3,
    eps_root: float = 0.0,
    initial_accumulator_value: float = 1e-6,
    weight_decay: float | optax.Schedule = 0.0,
    decoupled_weight_decay: bool = True,
    mu_dtype: tp.Any | None = None,
    nu_dtype: tp.Any | None = None,
    nesterov: bool = False,
    mars_gamma: float = 0.0,
    mars_beta: float | None = None,
    mars_prev_grad_dtype: tp.Any = jnp.bfloat16,
) -> optax.GradientTransformation:
    use_mars = mars_gamma > 0.0
    mars_beta_value = b1 if mars_beta is None else mars_beta
    mars_c = _mars_coeff(mars_gamma, mars_beta_value)
    weight_decay_is_schedule = callable(weight_decay)
    has_weight_decay = weight_decay_is_schedule or weight_decay != 0.0

    def init(params):
        mu = _tree_map(lambda param: _full_like_with_optional_dtype(param, initial_accumulator_value, mu_dtype), params)
        nu = _tree_map(lambda param: _full_like_with_optional_dtype(param, initial_accumulator_value, nu_dtype), params)
        prev_grad = None
        if use_mars:
            prev_grad = _tree_map(lambda param: jnp.zeros_like(param, dtype=mars_prev_grad_dtype), params)
        return FusedAdaptiveState(mu=mu, nu=nu, count=jnp.zeros((), jnp.int32), prev_grad=prev_grad)

    def update(grads, state, params=None):
        if params is None:
            if has_weight_decay:
                raise ValueError("fused_yogi requires params when weight_decay is nonzero.")
            params = _tree_map(jnp.zeros_like, grads)

        old_count = state.count
        count = optax.safe_int32_increment(old_count)
        nesterov_count = optax.safe_int32_increment(count)
        lr_t = learning_rate(old_count) if callable(learning_rate) else learning_rate
        wd_t = weight_decay(old_count) if weight_decay_is_schedule else weight_decay
        count_f = count.astype(jnp.float32)
        nesterov_count_f = nesterov_count.astype(jnp.float32)
        bc1 = 1.0 - b1**count_f
        bc1_next = 1.0 - b1**nesterov_count_f
        bc2 = 1.0 - b2**count_f

        def upd(grad, momentum, variance, param, prev_grad):
            momentum_store_dtype = momentum.dtype
            variance_store_dtype = variance.dtype
            work_dtype = _compute_dtype(grad)
            grad_work = grad.astype(work_dtype)
            if use_mars:
                if prev_grad is None:
                    raise ValueError("MARS state is missing prev_grad.")
                corrected = grad_work + jnp.asarray(mars_c, dtype=work_dtype) * (
                    grad_work - prev_grad.astype(work_dtype)
                )
                next_prev = grad.astype(mars_prev_grad_dtype)
            else:
                corrected = grad_work
                next_prev = None

            if has_weight_decay and not decoupled_weight_decay:
                corrected = corrected + jnp.asarray(wd_t, dtype=work_dtype) * param.astype(work_dtype)

            momentum_new = b1 * momentum.astype(work_dtype) + (1.0 - b1) * corrected
            grad_sq = _abs_sq(corrected).astype(jnp.float32)
            variance_new = variance.astype(jnp.float32) - (1.0 - b2) * jnp.sign(
                variance.astype(jnp.float32) - grad_sq
            ) * grad_sq
            if nesterov:
                mu_hat_next = momentum_new / jnp.asarray(bc1_next, dtype=work_dtype)
                grad_hat = corrected / jnp.asarray(bc1, dtype=work_dtype)
                mu_hat = b1 * mu_hat_next + (1.0 - b1) * grad_hat
            else:
                mu_hat = momentum_new / jnp.asarray(bc1, dtype=work_dtype)
            nu_hat = variance_new / jnp.asarray(bc2, dtype=variance_new.dtype)
            direction = mu_hat / (
                jnp.sqrt(nu_hat + jnp.asarray(eps_root, dtype=nu_hat.dtype)) + jnp.asarray(eps, dtype=nu_hat.dtype)
            )
            if has_weight_decay and decoupled_weight_decay:
                direction = direction + jnp.asarray(wd_t, dtype=direction.dtype) * param.astype(direction.dtype)
            update_leaf = (-jnp.asarray(lr_t, dtype=direction.dtype) * direction).astype(param.dtype)
            return update_leaf, momentum_new.astype(momentum_store_dtype), variance_new.astype(variance_store_dtype), next_prev

        if use_mars:
            out = _tree_map(upd, grads, state.mu, state.nu, params, state.prev_grad)
        else:
            out = _tree_map(lambda g, m, v, p: upd(g, m, v, p, None), grads, state.mu, state.nu, params)
        updates = _tree_map(lambda value: value[0], out, is_leaf=_is_tuple)
        mu = _tree_map(lambda value: value[1], out, is_leaf=_is_tuple)
        nu = _tree_map(lambda value: value[2], out, is_leaf=_is_tuple)
        prev_grad = _tree_map(lambda value: value[3], out, is_leaf=_is_tuple) if use_mars else None
        return updates, FusedAdaptiveState(mu=mu, nu=nu, count=count, prev_grad=prev_grad)

    return optax.GradientTransformation(init, update)


def _schedule_free_beta(
    count: jax.Array,
    *,
    beta: float,
    beta_final: float | None,
    beta_anneal_steps: int,
) -> jax.Array:
    if beta_final is None or beta_anneal_steps <= 0:
        return jnp.asarray(beta, dtype=jnp.float32)
    tau = jnp.minimum(
        count.astype(jnp.float32) / jnp.asarray(beta_anneal_steps, dtype=jnp.float32),
        1.0,
    )
    log_start = jnp.log1p(-jnp.asarray(beta, dtype=jnp.float32))
    log_end = jnp.log1p(-jnp.asarray(beta_final, dtype=jnp.float32))
    return 1.0 - jnp.exp((1.0 - tau) * log_start + tau * log_end)


def _schedule_free_plus_state_dtype(param: jax.Array, state_dtype: tp.Any | None) -> tp.Any:
    return param.dtype if state_dtype is None else state_dtype


def _schedule_free_plus_find_state(state: tp.Any) -> ScheduleFreePlusState | ScheduleFreePlusHyperballState:
    if isinstance(state, (ScheduleFreePlusState, ScheduleFreePlusHyperballState)):
        return state
    if isinstance(state, (tuple, list)):
        for child in state:
            try:
                return _schedule_free_plus_find_state(child)
            except ValueError:
                pass
    if isinstance(state, dict):
        for child in state.values():
            try:
                return _schedule_free_plus_find_state(child)
            except ValueError:
                pass
    raise ValueError("No ScheduleFreePlusState found in optimizer state.")


def schedule_free_plus_eval_params(state: tp.Any) -> optax.Params:
    return _schedule_free_plus_find_state(state).x


def schedule_free_plus_train_params(
    state: tp.Any,
    *,
    beta: float = 0.9,
    beta_final: float | None = None,
    beta_anneal_steps: int = 0,
) -> optax.Params:
    sf_state = _schedule_free_plus_find_state(state)
    last_update_count = jnp.maximum(
        sf_state.count - jnp.asarray(1, dtype=sf_state.count.dtype),
        jnp.asarray(0, dtype=sf_state.count.dtype),
    )
    beta_t = _schedule_free_beta(
        last_update_count,
        beta=beta,
        beta_final=beta_final,
        beta_anneal_steps=beta_anneal_steps,
    )
    train_params = _tree_map(lambda z, x: (1.0 - beta_t) * z + beta_t * x, sf_state.z, sf_state.x)
    if isinstance(sf_state, ScheduleFreePlusHyperballState):
        train_params = _tree_map(
            lambda value, radius, apply_hyperball: _hyperball_project_if_masked(
                value,
                radius,
                apply_hyperball,
                1e-30,
            ),
            train_params,
            sf_state.radii,
            sf_state.mask,
        )
    return train_params


def schedule_free_plus_adamw(
    learning_rate: float | optax.Schedule,
    *,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    beta: float = 0.9,
    beta_final: float | None = 0.965,
    beta_anneal_steps: int = 0,
    averaging_warmup_steps: int = 0,
    r: float = 1.0,
    weight_lr_power: float = 2.0,
    polyak: bool = False,
    inverse_l1_weighting: bool = False,
    l1_beta: float = 0.9,
    l1_eps: float = 1e-12,
    normalize_l1_by_size: bool = False,
    adamc_weight_decay: bool = True,
    mu_dtype: tp.Any | None = None,
    nu_dtype: tp.Any | None = None,
    state_dtype: tp.Any | None = jnp.float32,
) -> optax.GradientTransformationExtraArgs:
    if not 0.0 <= b1 < 1.0:
        raise ValueError("b1 must be in [0, 1).")
    if not 0.0 <= b2 < 1.0:
        raise ValueError("b2 must be in [0, 1).")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1).")
    if beta_final is not None and not 0.0 <= beta_final < 1.0:
        raise ValueError("beta_final must be in [0, 1) when set.")
    if averaging_warmup_steps < 0:
        raise ValueError("averaging_warmup_steps must be >= 0.")
    if beta_anneal_steps < 0:
        raise ValueError("beta_anneal_steps must be >= 0.")
    if r < 0.0:
        raise ValueError("r must be >= 0.")
    if weight_lr_power < 0.0:
        raise ValueError("weight_lr_power must be >= 0.")
    if not 0.0 <= l1_beta < 1.0:
        raise ValueError("l1_beta must be in [0, 1).")
    if l1_eps <= 0.0:
        raise ValueError("l1_eps must be > 0.")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be >= 0.")

    mu_dtype = jax.dtypes.canonicalize_dtype(mu_dtype)
    nu_dtype = jax.dtypes.canonicalize_dtype(nu_dtype)
    state_dtype = jax.dtypes.canonicalize_dtype(state_dtype)

    def init(params):
        def init_state_param(param):
            dtype = _schedule_free_plus_state_dtype(param, state_dtype)
            return param.astype(dtype)

        z = _tree_map(init_state_param, params)
        x = _tree_map(init_state_param, params)
        mu = _tree_map(lambda param: _zeros_like_with_optional_dtype(param, mu_dtype), z)
        nu = _tree_map(lambda param: _zeros_like_with_optional_dtype(param, nu_dtype), z)
        return ScheduleFreePlusState(
            z=z,
            x=x,
            mu=mu,
            nu=nu,
            l1_ema=jnp.zeros((), dtype=jnp.float32),
            weight_sum=jnp.zeros((), dtype=jnp.float32),
            lr_max=jnp.asarray(l1_eps, dtype=jnp.float32),
            count=jnp.zeros((), dtype=jnp.int32),
        )

    def update(grads, state, params=None, **extra_args):
        if params is None:
            raise ValueError("schedule_free_plus requires params in tx.update.")

        count = optax.safe_int32_increment(state.count)
        count_f = count.astype(jnp.float32)
        beta_t = _schedule_free_beta(
            state.count,
            beta=beta,
            beta_final=beta_final,
            beta_anneal_steps=beta_anneal_steps,
        )
        lr_t = learning_rate(state.count) if callable(learning_rate) else learning_rate
        lr_t = jnp.asarray(lr_t, dtype=jnp.float32)

        raw_l1 = _tree_scalar_sum(grads, lambda grad: jnp.sum(jnp.abs(grad)))
        if normalize_l1_by_size:
            raw_l1 = raw_l1 / jnp.asarray(max(_tree_size(grads), 1), dtype=jnp.float32)
        raw_l1 = raw_l1 * jnp.asarray(_SCHEDULE_FREE_PLUS_L1_SCALE, dtype=jnp.float32)
        l1_ema = l1_beta * state.l1_ema + (1.0 - l1_beta) * raw_l1
        l1_hat = l1_ema / jnp.maximum(1.0 - jnp.asarray(l1_beta, dtype=jnp.float32) ** count_f, l1_eps)

        alpha = lr_t
        value = extra_args.get("value", extra_args.get("loss", None))
        if polyak and value is not None:
            z_minus_x = _tree_map(lambda z, x: z - x, state.z, state.x)
            inner_correction = beta_t * _tree_inner_product(grads, z_minus_x)
            polyak_scalar = jnp.maximum(jnp.asarray(value, dtype=jnp.float32) + inner_correction, 0.0) / jnp.maximum(
                l1_hat,
                jnp.asarray(l1_eps, dtype=jnp.float32),
            )
            alpha = lr_t * polyak_scalar
        elif inverse_l1_weighting:
            alpha = lr_t / jnp.maximum(l1_hat, jnp.asarray(l1_eps, dtype=jnp.float32))

        lr_max = jnp.maximum(state.lr_max, jnp.abs(alpha))
        weight = (count_f**jnp.asarray(r, dtype=jnp.float32)) * (
            lr_max**jnp.asarray(weight_lr_power, dtype=jnp.float32)
        )
        use_warm_average = count <= jnp.asarray(averaging_warmup_steps, dtype=count.dtype)
        weight_sum = jnp.where(use_warm_average, state.weight_sum, state.weight_sum + weight)
        c_t = jnp.where(
            use_warm_average,
            jnp.asarray(1.0, dtype=jnp.float32),
            weight / jnp.maximum(weight_sum, jnp.asarray(l1_eps, dtype=jnp.float32)),
        )

        bc1 = 1.0 - jnp.asarray(b1, dtype=jnp.float32) ** count_f
        bc2 = 1.0 - jnp.asarray(b2, dtype=jnp.float32) ** count_f

        def update_leaf(grad, param, z_prev, x_prev, mu_prev, nu_prev):
            grad_work = grad.astype(_compute_dtype(grad))
            z_work = z_prev.astype(_compute_dtype(z_prev))
            x_work = x_prev.astype(_compute_dtype(x_prev))
            y_prev = (1.0 - beta_t) * z_work + beta_t * x_work
            mu_new = b1 * mu_prev.astype(_compute_dtype(grad)) + (1.0 - b1) * grad_work
            nu_new = b2 * nu_prev.astype(jnp.float32) + (1.0 - b2) * _abs_sq(grad_work).astype(jnp.float32)
            mu_hat = mu_new / jnp.asarray(bc1, dtype=mu_new.dtype)
            nu_hat = nu_new / jnp.asarray(bc2, dtype=nu_new.dtype)
            direction = mu_hat / (
                jnp.sqrt(nu_hat + jnp.asarray(eps_root, dtype=nu_hat.dtype)) + jnp.asarray(eps, dtype=nu_hat.dtype)
            )
            decay_scale = alpha * alpha if adamc_weight_decay else alpha
            z_new = z_work - jnp.asarray(decay_scale * weight_decay, dtype=z_work.dtype) * y_prev
            z_new = z_new - jnp.asarray(alpha, dtype=direction.dtype) * direction
            x_new = (1.0 - c_t) * x_work + c_t * z_new
            y_new = (1.0 - beta_t) * z_new + beta_t * x_new
            return (
                (y_new - param.astype(y_new.dtype)).astype(param.dtype),
                z_new.astype(z_prev.dtype),
                x_new.astype(x_prev.dtype),
                mu_new.astype(mu_prev.dtype),
                nu_new.astype(nu_prev.dtype),
            )

        out = _tree_map(update_leaf, grads, params, state.z, state.x, state.mu, state.nu)
        updates = _tree_map(lambda value: value[0], out, is_leaf=_is_tuple)
        z = _tree_map(lambda value: value[1], out, is_leaf=_is_tuple)
        x = _tree_map(lambda value: value[2], out, is_leaf=_is_tuple)
        mu = _tree_map(lambda value: value[3], out, is_leaf=_is_tuple)
        nu = _tree_map(lambda value: value[4], out, is_leaf=_is_tuple)
        return updates, ScheduleFreePlusState(
            z=z,
            x=x,
            mu=mu,
            nu=nu,
            l1_ema=l1_ema,
            weight_sum=weight_sum,
            lr_max=lr_max,
            count=count,
        )

    return optax.GradientTransformationExtraArgs(init, update)


_LAYER_ID_MLP = 0
_LAYER_ID_ATTENTION = 1
_LAYER_ID_EMBEDDING = 2
_LAYER_ID_LM_HEAD = 3
_LAYER_ID_NORM = 4
_PRISM_INVROOT_R2_COEFFICIENTS: tp.Final[tuple[tuple[float, float, float], ...]] = (
    (7.42487, -18.3958, 12.8967),
    (3.48773, -2.33004, 0.440469),
    (2.77661, -2.07064, 0.463023),
    (1.99131, -1.37394, 0.387593),
    (15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0),
)


def _layer_type_id(path_str: str) -> int:
    path_lower = path_str.lower()
    if any(token in path_lower for token in ("lm_head", "output_projection", "final_logits")):
        return _LAYER_ID_LM_HEAD
    if any(token in path_lower for token in ("embed", "embedding", "wte", "wpe")):
        return _LAYER_ID_EMBEDDING
    if any(token in path_lower for token in ("norm", "layernorm", "layer_norm", "rmsnorm", "ln_")):
        return _LAYER_ID_NORM
    if any(
        token in path_lower
        for token in (
            "self_attn",
            "attention",
            "linear_attn",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "in_proj_qkvz",
            "in_proj_ba",
            "out_proj",
            "conv1d",
            "a_log",
            "dt_bias",
            "attn",
            "qkv",
        )
    ):
        return _LAYER_ID_ATTENTION
    return _LAYER_ID_MLP


def _infer_input_features_axis(path_str: str, param: tp.Any) -> int:
    if not _is_array_like(param) or getattr(param, "ndim", 0) != 2:
        return -1
    path_parts = [part for part in path_str.lower().split(".") if part]
    if path_parts and (path_parts[-1] == "weight" or path_parts[-2:] == ["weight", "value"]):
        return 0
    return -1


def _normalize_axis(axis: int, ndim: int) -> int:
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"`axis` must be in [-{ndim}, {ndim - 1}].")
    return axis


def _is_prism_matrix(path_str: str, param: tp.Any) -> bool:
    if not _is_array_like(param) or getattr(param, "ndim", 0) != 2:
        return False
    return _layer_type_id(path_str) not in (_LAYER_ID_EMBEDDING, _LAYER_ID_NORM)


def _group_stats_shape(shape: tuple[int, ...], group_size: int, axis: int) -> tuple[int, ...]:
    axis = _normalize_axis(axis, len(shape))
    grouped = list(shape)
    axis_size = int(grouped.pop(axis))
    n_groups = max(math.ceil(axis_size / int(group_size)), 1)
    return (*grouped, n_groups)


def _group_mean_square_along_axis(array: jax.Array, group_size: int, axis: int) -> jax.Array:
    array = jnp.asarray(array)
    axis = _normalize_axis(axis, array.ndim)
    grouped = jnp.moveaxis(array, axis, -1) if axis != array.ndim - 1 else array
    length = int(grouped.shape[-1])
    n_groups = max(math.ceil(length / int(group_size)), 1)
    padded_length = n_groups * int(group_size)
    pad = padded_length - length
    if pad:
        grouped = jnp.pad(grouped, [(0, 0)] * (grouped.ndim - 1) + [(0, pad)])
    blocks = grouped.reshape((*grouped.shape[:-1], n_groups, int(group_size)))
    starts = jnp.arange(n_groups) * int(group_size)
    valid_counts = jnp.minimum(int(group_size), jnp.maximum(length - starts, 0)).astype(array.dtype)
    valid_counts = valid_counts.reshape((1,) * (grouped.ndim - 1) + (n_groups,))
    return jnp.sum(jnp.square(blocks), axis=-1) / jnp.maximum(valid_counts, jnp.asarray(1.0, dtype=array.dtype))


def _expand_group_values_along_axis(
    group_values: jax.Array,
    *,
    target_shape: tuple[int, ...],
    group_size: int,
    axis: int,
) -> jax.Array:
    axis = _normalize_axis(axis, len(target_shape))
    length = int(target_shape[axis])
    n_groups = int(group_values.shape[-1])
    expanded = jnp.repeat(group_values[..., None], int(group_size), axis=-1)
    expanded = expanded.reshape((*group_values.shape[:-1], n_groups * int(group_size)))[..., :length]
    return jnp.moveaxis(expanded, -1, axis) if axis != len(target_shape) - 1 else expanded


def _normalize_prism_update(
    update: jax.Array,
    second_moment: jax.Array,
    *,
    group_size: int,
    axis: int,
    beta2: float,
    epsilon: float,
    rms_scale: float,
) -> tuple[jax.Array, jax.Array]:
    compute_dtype = jnp.result_type(update.dtype, jnp.float32)
    update_f32 = update.astype(compute_dtype)
    group_ms = _group_mean_square_along_axis(update_f32, group_size, axis)
    second_next = (
        jnp.asarray(beta2, dtype=compute_dtype) * second_moment.astype(compute_dtype)
        + (1.0 - jnp.asarray(beta2, dtype=compute_dtype)) * group_ms
    )
    expanded_second = _expand_group_values_along_axis(
        second_next,
        target_shape=update.shape,
        group_size=group_size,
        axis=axis,
    )
    normalized = update_f32 / jnp.sqrt(expanded_second + jnp.asarray(epsilon, dtype=compute_dtype))
    if float(rms_scale) > 0.0:
        norm = jnp.linalg.norm(normalized.astype(jnp.float32)).astype(compute_dtype)
        min_norm = jnp.asarray(max(float(epsilon), 1e-30), dtype=compute_dtype)
        target_norm = jnp.asarray(rms_scale, dtype=compute_dtype) * jnp.sqrt(
            jnp.asarray(normalized.size, dtype=compute_dtype)
        )
        normalized = normalized * (target_norm / jnp.maximum(norm, min_norm))
    return normalized.astype(update.dtype), second_next.astype(second_moment.dtype)


def _symmetrize(matrix: jax.Array) -> jax.Array:
    return 0.5 * (matrix + matrix.T)


def _prism_r2_coefficients(inv_steps: int, inv_scale: float) -> tp.Iterator[tuple[float, float, float]]:
    scale = float(inv_scale)
    for idx in range(max(int(inv_steps), 1)):
        a, b, c = _PRISM_INVROOT_R2_COEFFICIENTS[min(idx, len(_PRISM_INVROOT_R2_COEFFICIENTS) - 1)]
        yield a / scale, b / scale**3, c / scale**5


def _matmul_inverse_square_root(
    matrix: jax.Array,
    gram: jax.Array,
    *,
    inv_steps: int,
    inv_epsilon: float,
    inv_scale: float,
) -> jax.Array:
    compute_dtype = jnp.result_type(matrix.dtype, gram.dtype, jnp.float32)
    out = matrix.astype(compute_dtype)
    precond = _symmetrize(gram.astype(compute_dtype))
    eps = jnp.asarray(inv_epsilon, dtype=compute_dtype)
    identity = jnp.eye(precond.shape[0], dtype=compute_dtype)
    gram_scale = jnp.maximum(jnp.linalg.norm(precond), eps)
    precond = precond / gram_scale + eps * identity
    for a, b, c in _prism_r2_coefficients(inv_steps, inv_scale):
        next_weight = (
            jnp.asarray(a, dtype=compute_dtype) * identity
            + jnp.asarray(b, dtype=compute_dtype) * precond
            + jnp.asarray(c, dtype=compute_dtype) * (precond @ precond)
        )
        out = out @ next_weight
        precond = _symmetrize(precond @ (next_weight @ next_weight))
    return (out * jnp.power(gram_scale, jnp.asarray(-0.5, dtype=compute_dtype))).astype(matrix.dtype)


def _apply_grouped_single_sided_prism_last_axis(
    update: jax.Array,
    prediction_error: jax.Array,
    *,
    group_size: int,
    gamma: float,
    gram_epsilon: float,
    inv_steps: int,
    inv_epsilon: float,
    inv_scale: float,
    dtype: jnp.dtype,
) -> jax.Array:
    length = int(update.shape[-1])
    n_groups = max(math.ceil(length / int(group_size)), 1)
    padded_length = n_groups * int(group_size)
    pad = padded_length - length
    update_padded = update
    error_padded = prediction_error
    if pad:
        pad_width = [(0, 0)] * (update.ndim - 1) + [(0, pad)]
        update_padded = jnp.pad(update_padded, pad_width)
        error_padded = jnp.pad(error_padded, pad_width)
    update_blocks = update_padded.reshape((*update_padded.shape[:-1], n_groups, int(group_size)))
    error_blocks = error_padded.reshape((*error_padded.shape[:-1], n_groups, int(group_size)))
    update_blocks = jnp.moveaxis(update_blocks, -2, 0)
    error_blocks = jnp.moveaxis(error_blocks, -2, 0)

    def apply_block(update_block: jax.Array, error_block: jax.Array) -> jax.Array:
        return _apply_single_sided_prism(
            update_block,
            error_block,
            axis=-1,
            group_size=None,
            gamma=gamma,
            gram_epsilon=gram_epsilon,
            inv_steps=inv_steps,
            inv_epsilon=inv_epsilon,
            inv_scale=inv_scale,
            dtype=dtype,
        )

    out_blocks = jax.vmap(apply_block)(update_blocks, error_blocks)
    return jnp.moveaxis(out_blocks, 0, -2).reshape(update_padded.shape)[..., :length].astype(update.dtype)


def _apply_single_sided_prism(
    update: jax.Array,
    prediction_error: jax.Array,
    *,
    axis: int,
    group_size: int | None,
    gamma: float,
    gram_epsilon: float,
    inv_steps: int,
    inv_epsilon: float,
    inv_scale: float,
    dtype: jnp.dtype,
) -> jax.Array:
    axis = _normalize_axis(axis, update.ndim)
    update_canonical = jnp.moveaxis(update, axis, -1) if axis != update.ndim - 1 else update
    error_canonical = jnp.moveaxis(prediction_error, axis, -1) if axis != prediction_error.ndim - 1 else prediction_error
    if group_size is not None and int(group_size) > 0 and int(group_size) < int(update_canonical.shape[-1]):
        out = _apply_grouped_single_sided_prism_last_axis(
            update_canonical,
            error_canonical,
            group_size=int(group_size),
            gamma=gamma,
            gram_epsilon=gram_epsilon,
            inv_steps=inv_steps,
            inv_epsilon=inv_epsilon,
            inv_scale=inv_scale,
            dtype=dtype,
        )
        return (jnp.moveaxis(out, -1, axis) if axis != update.ndim - 1 else out).astype(update.dtype)
    compute_dtype = jnp.dtype(dtype)
    update_f32 = update_canonical.astype(compute_dtype)
    error_f32 = error_canonical.astype(compute_dtype)
    gram = update_f32.T @ update_f32
    if float(gamma) > 0.0:
        gram = gram + jnp.asarray(float(gamma) ** 2, dtype=compute_dtype) * (error_f32.T @ error_f32)
    gram = gram + jnp.asarray(gram_epsilon, dtype=compute_dtype) * jnp.eye(gram.shape[0], dtype=compute_dtype)
    out = _matmul_inverse_square_root(
        update_f32,
        gram,
        inv_steps=inv_steps,
        inv_epsilon=inv_epsilon,
        inv_scale=inv_scale,
    )
    out = jnp.moveaxis(out, -1, axis) if axis != update.ndim - 1 else out
    return out.astype(update.dtype)


def _newton_schulz_orthogonalize(matrix: jax.Array, steps: int, eps: float) -> jax.Array:
    value = matrix / (jnp.linalg.norm(matrix) + eps)
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        value = value.T
    a, b, c = (3.4445, -4.7750, 2.0315)

    def body_fn(_: int, current: jax.Array) -> jax.Array:
        aa = current @ current.T
        bb = b * aa + c * (aa @ aa)
        return a * current + bb @ current

    value = jax.lax.fori_loop(0, max(int(steps), 1), body_fn, value)
    return value.T if transposed else value


def _scale_to_unit_operator_norm(matrix: jax.Array, *, power_steps: int, epsilon: float) -> jax.Array:
    compute_dtype = jnp.result_type(matrix.dtype, jnp.float32)
    value = matrix.astype(compute_dtype)
    eps = jnp.asarray(epsilon, dtype=compute_dtype)
    vector = jnp.ones((value.shape[-1],), dtype=compute_dtype)
    vector = vector / jnp.maximum(jnp.linalg.norm(vector), eps)

    def body_fn(_: int, current: jax.Array) -> jax.Array:
        left = value @ current
        left = left / jnp.maximum(jnp.linalg.norm(left), eps)
        next_vector = value.T @ left
        return next_vector / jnp.maximum(jnp.linalg.norm(next_vector), eps)

    vector = jax.lax.fori_loop(0, max(int(power_steps), 1), body_fn, vector)
    op_norm = jnp.maximum(jnp.linalg.norm(value @ vector), eps)
    return (value / op_norm).astype(matrix.dtype)


def _apply_contra_muon(
    update: jax.Array,
    pre_polar_update: jax.Array,
    *,
    strength: float,
    power_steps: int,
    epsilon: float,
) -> jax.Array:
    if float(strength) <= 0.0:
        return update
    compute_dtype = jnp.result_type(update.dtype, pre_polar_update.dtype, jnp.float32)
    eps = jnp.asarray(epsilon, dtype=compute_dtype)
    base = update.astype(compute_dtype)
    contra = _scale_to_unit_operator_norm(
        pre_polar_update.astype(compute_dtype),
        power_steps=power_steps,
        epsilon=epsilon,
    ).astype(compute_dtype)
    base_norm = jnp.linalg.norm(base)
    adjusted = base - jnp.asarray(float(strength) / 2.0, dtype=compute_dtype) * contra
    adjusted_norm = jnp.linalg.norm(adjusted)
    return (adjusted * (base_norm / jnp.maximum(adjusted_norm, eps))).astype(update.dtype)


def _apply_update_clamp_min(
    update: jax.Array,
    param: jax.Array,
    *,
    min_ratio: float | None,
    epsilon: float,
) -> jax.Array:
    if min_ratio is None or float(min_ratio) <= 0.0:
        return update
    compute_dtype = jnp.result_type(update.dtype, param.dtype, jnp.float32)
    eps = jnp.asarray(epsilon, dtype=compute_dtype)
    update_f32 = update.astype(compute_dtype)
    param_norm = jnp.maximum(jnp.linalg.norm(param.astype(compute_dtype)), eps)
    update_norm = jnp.maximum(jnp.linalg.norm(update_f32), eps)
    target_update_norm = jnp.asarray(float(min_ratio), dtype=compute_dtype) * param_norm
    scale = jnp.where(update_norm < target_update_norm, target_update_norm / update_norm, 1.0)
    return (update_f32 * scale).astype(update.dtype)


def _apply_caution(grad: jax.Array, update: jax.Array) -> jax.Array:
    mask = jnp.bitwise_xor(jnp.signbit(grad), jnp.signbit(update))
    cautious = jnp.where(mask, jnp.zeros_like(update), update)
    total = jnp.asarray(update.size, dtype=update.dtype)
    valid = total - jnp.asarray(jnp.sum(mask), dtype=update.dtype)
    return cautious * jnp.where(valid > 0, total / valid, jnp.asarray(1.0, dtype=update.dtype))


class PrismState(NamedTuple):
    momentum: optax.Updates
    norm_second_moment: optax.Updates
    count: jax.Array


def prism_muon(
    learning_rate: float | optax.Schedule,
    *,
    momentum: float = 0.85,
    nesterov: bool = True,
    beta2: float = 0.95,
    group_size: int = 128,
    norm_group_size: int | None = None,
    normuon_epsilon: float = 1e-8,
    normuon_rms_scale: float = 0.2,
    single_sided_prism_gamma: float | None = 0.5,
    single_sided_prism_group_size: int | None = 128,
    single_sided_prism_gram_epsilon: float = 1e-6,
    single_sided_prism_inv_steps: int = 8,
    single_sided_prism_inv_epsilon: float = 1e-5,
    single_sided_prism_inv_scale: float = 1.001,
    single_sided_prism_dtype: jnp.dtype = jnp.float32,
    update_clamp_min: float | None = 0.35,
    update_clamp_epsilon: float = 1e-8,
    contra_muon: float = 0.1,
    contra_muon_power_steps: int = 5,
    contra_muon_epsilon: float = 1e-10,
    use_cautioning: bool = True,
    momentum_dtype: jnp.dtype = jnp.bfloat16,
    norm_dtype: jnp.dtype = jnp.float32,
    newton_schulz_dtype: jnp.dtype = jnp.bfloat16,
    ns_steps: int = 5,
    muon_epsilon: float = 1e-8,
    max_grad_norm: float | None = None,
    weight_decay: float = 0.0,
    mlp_lr_multiplier: float = 1.0,
    attn_lr_multiplier: float = 1.0,
    embed_lr_multiplier: float = 1.0,
    lm_head_lr_multiplier: float = 1.0,
    norm_lr_multiplier: float = 1.0,
) -> optax.GradientTransformation:
    norm_group_size = int(group_size if norm_group_size is None else norm_group_size)
    layer_lr_multipliers = (
        float(mlp_lr_multiplier),
        float(attn_lr_multiplier),
        float(embed_lr_multiplier),
        float(lm_head_lr_multiplier),
        float(norm_lr_multiplier),
    )

    def init(params):
        flat_with_paths, treedef = jtu.tree_flatten_with_path(params)
        momentum_flat = []
        norm_flat = []
        for path, param in flat_with_paths:
            if not _is_array_like(param):
                momentum_flat.append(param)
                norm_flat.append(param)
                continue
            path_str = _path_to_str(path)
            momentum_flat.append(jnp.zeros_like(param, dtype=momentum_dtype))
            if _is_prism_matrix(path_str, param):
                norm_flat.append(
                    jnp.zeros(
                        _group_stats_shape(tuple(param.shape), norm_group_size, _infer_input_features_axis(path_str, param)),
                        dtype=norm_dtype,
                    )
                )
            else:
                norm_flat.append(jnp.asarray(0.0, dtype=norm_dtype))
        return PrismState(
            momentum=jtu.tree_unflatten(treedef, momentum_flat),
            norm_second_moment=jtu.tree_unflatten(treedef, norm_flat),
            count=jnp.asarray(0, dtype=jnp.int32),
        )

    def update(grads, state, params=None):
        if params is None:
            raise ValueError("prism requires params in tx.update.")
        if max_grad_norm is not None:
            grad_norm = optax.global_norm(grads)
            max_norm = jnp.asarray(max_grad_norm, dtype=grad_norm.dtype)
            grad_scale = jnp.minimum(1.0, max_norm / jnp.maximum(grad_norm, jnp.asarray(1e-6, dtype=grad_norm.dtype)))
            grads = _tree_map(lambda grad: grad * grad_scale if _is_array_like(grad) else grad, grads)

        flat_with_paths, treedef = jtu.tree_flatten_with_path(params)
        grads_flat = jtu.tree_leaves(grads)
        momentum_flat = jtu.tree_leaves(state.momentum)
        norm_flat = jtu.tree_leaves(state.norm_second_moment)
        if not (len(grads_flat) == len(momentum_flat) == len(norm_flat) == len(flat_with_paths)):
            raise ValueError("prism state is incompatible with params/grads structure.")

        lr_t = learning_rate(state.count) if callable(learning_rate) else learning_rate
        updates_flat = []
        new_momentum_flat = []
        new_norm_flat = []
        for (path, param), grad, mu_prev, norm_prev in zip(
            flat_with_paths,
            grads_flat,
            momentum_flat,
            norm_flat,
            strict=True,
        ):
            if not _is_array_like(param):
                updates_flat.append(grad)
                new_momentum_flat.append(mu_prev)
                new_norm_flat.append(norm_prev)
                continue
            path_str = _path_to_str(path)
            grad = jnp.asarray(grad)
            param = jnp.asarray(param)
            grad_dtype = grad.dtype
            mu_prev = jnp.asarray(mu_prev, dtype=momentum_dtype)
            layer_id = _layer_type_id(path_str)
            projection_axis = _infer_input_features_axis(path_str, param)
            mu = (momentum * mu_prev + grad).astype(momentum_dtype)
            precond = momentum * mu + grad if nesterov else mu

            if _is_prism_matrix(path_str, param):
                rows = jnp.asarray(param.shape[0], dtype=grad_dtype)
                cols = jnp.asarray(max(param.shape[1], 1), dtype=grad_dtype)
                muon_scale = jnp.sqrt(jnp.maximum(1.0, rows / cols))
                pre_polar_update = precond.astype(grad_dtype)
                if single_sided_prism_gamma is not None:
                    momentum_prediction = ((1.0 - momentum) * mu).astype(grad_dtype)
                    prism_signal = ((1.0 - momentum) * pre_polar_update).astype(grad_dtype)
                    prediction_error = grad.astype(grad_dtype) - momentum_prediction
                    matrix = _apply_single_sided_prism(
                        prism_signal,
                        prediction_error,
                        axis=projection_axis,
                        group_size=single_sided_prism_group_size,
                        gamma=single_sided_prism_gamma,
                        gram_epsilon=single_sided_prism_gram_epsilon,
                        inv_steps=single_sided_prism_inv_steps,
                        inv_epsilon=single_sided_prism_inv_epsilon,
                        inv_scale=single_sided_prism_inv_scale,
                        dtype=single_sided_prism_dtype,
                    )
                else:
                    matrix = _newton_schulz_orthogonalize(
                        pre_polar_update.astype(newton_schulz_dtype),
                        steps=ns_steps,
                        eps=muon_epsilon,
                    )
                precond = _apply_contra_muon(
                    matrix.astype(grad_dtype),
                    pre_polar_update,
                    strength=contra_muon,
                    power_steps=contra_muon_power_steps,
                    epsilon=contra_muon_epsilon,
                )
                precond = precond * muon_scale
                precond, norm_next = _normalize_prism_update(
                    precond,
                    jnp.asarray(norm_prev, dtype=norm_dtype),
                    group_size=norm_group_size,
                    axis=projection_axis,
                    beta2=beta2,
                    epsilon=normuon_epsilon,
                    rms_scale=normuon_rms_scale,
                )
                precond = _apply_update_clamp_min(
                    precond,
                    param,
                    min_ratio=update_clamp_min,
                    epsilon=update_clamp_epsilon,
                )
            else:
                precond = precond.astype(grad_dtype)
                norm_next = norm_prev

            layer_lr = jnp.asarray(lr_t, dtype=precond.dtype) * jnp.asarray(
                layer_lr_multipliers[layer_id],
                dtype=precond.dtype,
            )
            scaled_update = precond * layer_lr
            if use_cautioning:
                scaled_update = _apply_caution(grad, scaled_update)
            update_leaf = -scaled_update
            if weight_decay > 0.0:
                update_leaf = update_leaf - layer_lr.astype(param.dtype) * jnp.asarray(weight_decay, dtype=param.dtype) * param
            updates_flat.append(update_leaf.astype(param.dtype))
            new_momentum_flat.append(mu)
            new_norm_flat.append(norm_next)

        return (
            jtu.tree_unflatten(treedef, updates_flat),
            PrismState(
                momentum=jtu.tree_unflatten(treedef, new_momentum_flat),
                norm_second_moment=jtu.tree_unflatten(treedef, new_norm_flat),
                count=optax.safe_int32_increment(state.count),
            ),
        )

    return optax.GradientTransformation(init, update)


class HyperballState(NamedTuple):
    inner_state: optax.OptState
    radii: optax.Params
    z: optax.Params
    count: jax.Array


def _squared_norm(array: jax.Array) -> jax.Array:
    array = jnp.asarray(array)
    return jnp.sum(jnp.real(array * jnp.conj(array)).astype(jnp.float32))


def _frobenius_norm(array: jax.Array) -> jax.Array:
    return jnp.sqrt(_squared_norm(array))


def _hyperball_norm(array: jax.Array) -> jax.Array:
    array = jnp.asarray(array)
    if array.ndim <= 2:
        return _frobenius_norm(array)
    axes = tuple(range(1, array.ndim))
    squared = jnp.real(array * jnp.conj(array)).astype(jnp.float32)
    return jnp.sqrt(jnp.sum(squared, axis=axes, keepdims=True))


def _normalize_to_hyperball_radius(array: jax.Array, radius: jax.Array, eps: float) -> jax.Array:
    array_norm = _hyperball_norm(array)
    radius_value = radius.astype(jnp.float32)
    return radius_value.astype(array.dtype) * array / jnp.maximum(
        array_norm,
        jnp.asarray(eps, dtype=jnp.float32),
    ).astype(array.dtype)


def _default_hyperball_mask(params, min_param_ndim: int):
    flat_with_paths, treedef = jtu.tree_flatten_with_path(params)
    mask_flat = [
        _is_default_hyperball_matrix(_path_to_str(path), param, min_param_ndim) for path, param in flat_with_paths
    ]
    return jtu.tree_unflatten(treedef, mask_flat)


def _resolve_hyperball_mask(mask, params, min_param_ndim: int):
    if mask is None:
        return _default_hyperball_mask(params, min_param_ndim)
    if callable(mask):
        return mask(params)
    if isinstance(mask, bool):
        return _tree_map(lambda _: mask, params)
    return mask


def _is_static_false(value: tp.Any) -> bool:
    return isinstance(value, bool) and not value


def _hyperball_project_if_masked(
    value: jax.Array,
    radius: jax.Array,
    apply_hyperball: tp.Any,
    eps: float,
) -> jax.Array:
    if _is_static_false(apply_hyperball):
        return value

    def projected_fn(_: None) -> jax.Array:
        projected = _normalize_to_hyperball_radius(value, radius, eps)
        valid = radius.astype(jnp.float32) > jnp.asarray(eps, dtype=jnp.float32)
        return jnp.where(valid, projected, value)

    if isinstance(apply_hyperball, bool):
        return projected_fn(None)
    apply_hyperball = jnp.asarray(apply_hyperball)
    if apply_hyperball.shape == ():
        return jax.lax.cond(apply_hyperball, projected_fn, lambda _: value, operand=None)
    return jnp.where(apply_hyperball, projected_fn(None), value)


def _hyperball_mask_to_state_leaf(apply_hyperball: tp.Any) -> jax.Array:
    return jnp.asarray(apply_hyperball)


def _default_muon_hyperball_dimension_numbers(params):
    dimension_numbers = optax.contrib.MuonDimensionNumbers()
    flat_with_paths, treedef = jtu.tree_flatten_with_path(params)
    dims_flat = [
        dimension_numbers if _is_default_muon_hyperball_matrix(_path_to_str(path), param) else None
        for path, param in flat_with_paths
    ]
    return jtu.tree_unflatten(treedef, dims_flat)


def hyperball(
    inner: optax.GradientTransformation,
    learning_rate: float | optax.Schedule,
    *,
    mask=None,
    min_param_ndim: int = 2,
    radius_scale: float = 1.0,
    eps: float = 1e-30,
) -> optax.GradientTransformation:
    if min_param_ndim < 0:
        raise ValueError("min_param_ndim must be >= 0.")
    if radius_scale <= 0.0:
        raise ValueError("radius_scale must be > 0.")
    if eps <= 0.0:
        raise ValueError("eps must be > 0.")

    def init(params):
        mask_tree = _resolve_hyperball_mask(mask, params, min_param_ndim)

        def init_radius(param, apply_hyperball):
            if _is_static_false(apply_hyperball):
                return jnp.zeros((), dtype=jnp.float32)
            return jnp.asarray(radius_scale, jnp.float32) * _hyperball_norm(param)

        radii = _tree_map(init_radius, params, mask_tree)
        z = _tree_map(
            lambda param, radius, apply_hyperball: _hyperball_project_if_masked(
                jnp.asarray(param), radius, apply_hyperball, eps
            ),
            params,
            radii,
            mask_tree,
        )
        return HyperballState(inner_state=inner.init(params), radii=radii, z=z, count=jnp.zeros((), jnp.int32))

    def update(grads, state, params=None, **extra_args):
        if params is None:
            raise ValueError("hyperball requires params to project constrained leaves.")
        if _update_accepts_extra_args(inner.update, extra_args):
            inner_updates, inner_state = inner.update(grads, state.inner_state, params, **extra_args)
        else:
            inner_updates, inner_state = inner.update(grads, state.inner_state, params)
        mask_tree = _resolve_hyperball_mask(mask, params, min_param_ndim)
        lr_t = learning_rate(state.count) if callable(learning_rate) else learning_rate
        count = optax.safe_int32_increment(state.count)

        def project_leaf(inner_update, param, radius, z_prev, apply_hyperball):
            if _is_static_false(apply_hyperball):
                return inner_update, param + inner_update

            def constrained_update_fn(_: None) -> jax.Array:
                update_norm = _hyperball_norm(inner_update)
                radius_value = radius.astype(jnp.float32)
                step_size = jnp.asarray(lr_t, dtype=jnp.float32) * radius_value
                direction = inner_update / jnp.maximum(update_norm, jnp.asarray(eps, dtype=jnp.float32)).astype(
                    inner_update.dtype
                )
                z_trial = z_prev + step_size.astype(direction.dtype) * direction
                z_next = _normalize_to_hyperball_radius(z_trial, radius_value, eps)
                constrained_update = (z_next - param).astype(inner_update.dtype)
                valid = (radius_value > eps) & (update_norm > eps) & (jnp.asarray(lr_t, dtype=jnp.float32) != 0.0)
                return (
                    jnp.where(valid, constrained_update, jnp.zeros_like(inner_update)),
                    jnp.where(valid, z_next.astype(z_prev.dtype), z_prev),
                )

            if isinstance(apply_hyperball, bool):
                return constrained_update_fn(None)
            apply_hyperball = jnp.asarray(apply_hyperball)
            if apply_hyperball.shape == ():
                return jax.lax.cond(
                    apply_hyperball,
                    constrained_update_fn,
                    lambda _: (inner_update, (param + inner_update).astype(z_prev.dtype)),
                    operand=None,
                )
            constrained_update, constrained_z = constrained_update_fn(None)
            return (
                jnp.where(apply_hyperball, constrained_update, inner_update),
                jnp.where(apply_hyperball, constrained_z, (param + inner_update).astype(z_prev.dtype)),
            )

        out = _tree_map(project_leaf, inner_updates, params, state.radii, state.z, mask_tree)
        updates = _tree_map(lambda value: value[0], out, is_leaf=_is_tuple)
        z = _tree_map(lambda value: value[1], out, is_leaf=_is_tuple)
        return updates, HyperballState(inner_state=inner_state, radii=state.radii, z=z, count=count)

    return optax.GradientTransformationExtraArgs(init, update)


def schedule_free_plus_hyperball_adamw(
    learning_rate: float | optax.Schedule,
    *,
    mask=None,
    min_param_ndim: int = 2,
    radius_scale: float = 1.0,
    hyperball_eps: float = 1e-30,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    beta: float = 0.9,
    beta_final: float | None = 0.965,
    beta_anneal_steps: int = 0,
    averaging_warmup_steps: int = 0,
    r: float = 1.0,
    weight_lr_power: float = 2.0,
    polyak: bool = False,
    inverse_l1_weighting: bool = False,
    l1_beta: float = 0.9,
    l1_eps: float = 1e-12,
    normalize_l1_by_size: bool = False,
    adamc_weight_decay: bool = True,
    normalize_update: bool = True,
    mu_dtype: tp.Any | None = None,
    nu_dtype: tp.Any | None = None,
    state_dtype: tp.Any | None = jnp.float32,
) -> optax.GradientTransformationExtraArgs:
    if min_param_ndim < 0:
        raise ValueError("min_param_ndim must be >= 0.")
    if radius_scale <= 0.0:
        raise ValueError("radius_scale must be > 0.")
    if hyperball_eps <= 0.0:
        raise ValueError("hyperball_eps must be > 0.")
    if not 0.0 <= b1 < 1.0:
        raise ValueError("b1 must be in [0, 1).")
    if not 0.0 <= b2 < 1.0:
        raise ValueError("b2 must be in [0, 1).")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1).")
    if beta_final is not None and not 0.0 <= beta_final < 1.0:
        raise ValueError("beta_final must be in [0, 1) when set.")
    if averaging_warmup_steps < 0:
        raise ValueError("averaging_warmup_steps must be >= 0.")
    if beta_anneal_steps < 0:
        raise ValueError("beta_anneal_steps must be >= 0.")
    if r < 0.0:
        raise ValueError("r must be >= 0.")
    if weight_lr_power < 0.0:
        raise ValueError("weight_lr_power must be >= 0.")
    if not 0.0 <= l1_beta < 1.0:
        raise ValueError("l1_beta must be in [0, 1).")
    if l1_eps <= 0.0:
        raise ValueError("l1_eps must be > 0.")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be >= 0.")

    mu_dtype = jax.dtypes.canonicalize_dtype(mu_dtype)
    nu_dtype = jax.dtypes.canonicalize_dtype(nu_dtype)
    state_dtype = jax.dtypes.canonicalize_dtype(state_dtype)

    def init(params):
        mask_tree = _resolve_hyperball_mask(mask, params, min_param_ndim)
        mask_state = _tree_map(_hyperball_mask_to_state_leaf, mask_tree)

        def init_radius(param, apply_hyperball):
            if _is_static_false(apply_hyperball):
                return jnp.zeros((), dtype=jnp.float32)
            return jnp.asarray(radius_scale, jnp.float32) * _hyperball_norm(param)

        def init_state_param(param, radius, apply_hyperball):
            dtype = _schedule_free_plus_state_dtype(param, state_dtype)
            value = param.astype(dtype)
            return _hyperball_project_if_masked(value, radius, apply_hyperball, hyperball_eps)

        radii = _tree_map(init_radius, params, mask_tree)
        z = _tree_map(init_state_param, params, radii, mask_state)
        x = _tree_map(init_state_param, params, radii, mask_state)
        mu = _tree_map(lambda param: _zeros_like_with_optional_dtype(param, mu_dtype), z)
        nu = _tree_map(lambda param: _zeros_like_with_optional_dtype(param, nu_dtype), z)
        return ScheduleFreePlusHyperballState(
            z=z,
            x=x,
            mu=mu,
            nu=nu,
            l1_ema=jnp.zeros((), dtype=jnp.float32),
            weight_sum=jnp.zeros((), dtype=jnp.float32),
            lr_max=jnp.asarray(l1_eps, dtype=jnp.float32),
            radii=radii,
            mask=mask_state,
            count=jnp.zeros((), dtype=jnp.int32),
        )

    def update(grads, state, params=None, **extra_args):
        if params is None:
            raise ValueError("schedule_free_plus_hyperball requires params in tx.update.")

        count = optax.safe_int32_increment(state.count)
        count_f = count.astype(jnp.float32)
        beta_t = _schedule_free_beta(
            state.count,
            beta=beta,
            beta_final=beta_final,
            beta_anneal_steps=beta_anneal_steps,
        )
        lr_t = learning_rate(state.count) if callable(learning_rate) else learning_rate
        lr_t = jnp.asarray(lr_t, dtype=jnp.float32)

        raw_l1 = _tree_scalar_sum(grads, lambda grad: jnp.sum(jnp.abs(grad)))
        if normalize_l1_by_size:
            raw_l1 = raw_l1 / jnp.asarray(max(_tree_size(grads), 1), dtype=jnp.float32)
        raw_l1 = raw_l1 * jnp.asarray(_SCHEDULE_FREE_PLUS_L1_SCALE, dtype=jnp.float32)
        l1_ema = l1_beta * state.l1_ema + (1.0 - l1_beta) * raw_l1
        l1_hat = l1_ema / jnp.maximum(1.0 - jnp.asarray(l1_beta, dtype=jnp.float32) ** count_f, l1_eps)

        alpha = lr_t
        value = extra_args.get("value", extra_args.get("loss", None))
        if polyak and value is not None:
            z_minus_x = _tree_map(lambda z, x: z - x, state.z, state.x)
            inner_correction = beta_t * _tree_inner_product(grads, z_minus_x)
            polyak_scalar = jnp.maximum(jnp.asarray(value, dtype=jnp.float32) + inner_correction, 0.0) / jnp.maximum(
                l1_hat,
                jnp.asarray(l1_eps, dtype=jnp.float32),
            )
            alpha = lr_t * polyak_scalar
        elif inverse_l1_weighting:
            alpha = lr_t / jnp.maximum(l1_hat, jnp.asarray(l1_eps, dtype=jnp.float32))

        lr_max = jnp.maximum(state.lr_max, jnp.abs(alpha))
        weight = (count_f**jnp.asarray(r, dtype=jnp.float32)) * (
            lr_max**jnp.asarray(weight_lr_power, dtype=jnp.float32)
        )
        use_warm_average = count <= jnp.asarray(averaging_warmup_steps, dtype=count.dtype)
        weight_sum = jnp.where(use_warm_average, state.weight_sum, state.weight_sum + weight)
        c_t = jnp.where(
            use_warm_average,
            jnp.asarray(1.0, dtype=jnp.float32),
            weight / jnp.maximum(weight_sum, jnp.asarray(l1_eps, dtype=jnp.float32)),
        )

        bc1 = 1.0 - jnp.asarray(b1, dtype=jnp.float32) ** count_f
        bc2 = 1.0 - jnp.asarray(b2, dtype=jnp.float32) ** count_f

        def update_leaf(grad, param, z_prev, x_prev, mu_prev, nu_prev, radius, apply_hyperball):
            grad_work = grad.astype(_compute_dtype(grad))
            z_work = z_prev.astype(_compute_dtype(z_prev))
            x_work = x_prev.astype(_compute_dtype(x_prev))
            y_prev = _hyperball_project_if_masked(
                (1.0 - beta_t) * z_work + beta_t * x_work,
                radius,
                apply_hyperball,
                hyperball_eps,
            )
            mu_new = b1 * mu_prev.astype(_compute_dtype(grad)) + (1.0 - b1) * grad_work
            nu_new = b2 * nu_prev.astype(jnp.float32) + (1.0 - b2) * _abs_sq(grad_work).astype(jnp.float32)
            mu_hat = mu_new / jnp.asarray(bc1, dtype=mu_new.dtype)
            nu_hat = nu_new / jnp.asarray(bc2, dtype=nu_new.dtype)
            direction = mu_hat / (
                jnp.sqrt(nu_hat + jnp.asarray(eps_root, dtype=nu_hat.dtype)) + jnp.asarray(eps, dtype=nu_hat.dtype)
            )

            decay_scale = alpha * alpha if adamc_weight_decay else alpha
            regular_z = z_work - jnp.asarray(decay_scale * weight_decay, dtype=z_work.dtype) * y_prev
            regular_z = regular_z - jnp.asarray(alpha, dtype=direction.dtype) * direction
            regular_x = (1.0 - c_t) * x_work + c_t * regular_z
            regular_y = (1.0 - beta_t) * regular_z + beta_t * regular_x

            direction_norm = _hyperball_norm(direction)
            unit_direction = direction / jnp.maximum(
                direction_norm,
                jnp.asarray(hyperball_eps, dtype=jnp.float32),
            ).astype(direction.dtype)
            hyperball_direction = (
                radius.astype(direction.dtype) * unit_direction
                if normalize_update
                else direction
            )
            hyperball_z_base = z_work - jnp.asarray(decay_scale * weight_decay, dtype=z_work.dtype) * y_prev
            hyperball_z_base = _hyperball_project_if_masked(hyperball_z_base, radius, apply_hyperball, hyperball_eps)
            hyperball_z_trial = hyperball_z_base - jnp.asarray(alpha, dtype=direction.dtype) * hyperball_direction
            hyperball_z = _hyperball_project_if_masked(hyperball_z_trial, radius, apply_hyperball, hyperball_eps)
            hyperball_x_trial = (1.0 - c_t) * x_work + c_t * hyperball_z
            hyperball_x = _hyperball_project_if_masked(hyperball_x_trial, radius, apply_hyperball, hyperball_eps)
            hyperball_y_trial = (1.0 - beta_t) * hyperball_z + beta_t * hyperball_x
            hyperball_y = _hyperball_project_if_masked(hyperball_y_trial, radius, apply_hyperball, hyperball_eps)

            if isinstance(apply_hyperball, bool):
                y_new, z_new, x_new = (
                    (regular_y, regular_z, regular_x)
                    if not apply_hyperball
                    else (hyperball_y, hyperball_z, hyperball_x)
                )
            else:
                apply_hyperball = jnp.asarray(apply_hyperball)
                y_new = jnp.where(apply_hyperball, hyperball_y, regular_y)
                z_new = jnp.where(apply_hyperball, hyperball_z, regular_z)
                x_new = jnp.where(apply_hyperball, hyperball_x, regular_x)

            return (
                (y_new - param.astype(y_new.dtype)).astype(param.dtype),
                z_new.astype(z_prev.dtype),
                x_new.astype(x_prev.dtype),
                mu_new.astype(mu_prev.dtype),
                nu_new.astype(nu_prev.dtype),
            )

        out = _tree_map(update_leaf, grads, params, state.z, state.x, state.mu, state.nu, state.radii, state.mask)
        updates = _tree_map(lambda value: value[0], out, is_leaf=_is_tuple)
        z = _tree_map(lambda value: value[1], out, is_leaf=_is_tuple)
        x = _tree_map(lambda value: value[2], out, is_leaf=_is_tuple)
        mu = _tree_map(lambda value: value[3], out, is_leaf=_is_tuple)
        nu = _tree_map(lambda value: value[4], out, is_leaf=_is_tuple)
        return updates, ScheduleFreePlusHyperballState(
            z=z,
            x=x,
            mu=mu,
            nu=nu,
            l1_ema=l1_ema,
            weight_sum=weight_sum,
            lr_max=lr_max,
            radii=state.radii,
            mask=state.mask,
            count=count,
        )

    return optax.GradientTransformationExtraArgs(init, update)


@dataclasses.dataclass
class NormuonConfig(MuonConfig):
    consistent_rms: float | None = 0.2


@dataclasses.dataclass
class NovoGradConfig(SerializationMixin):
    b1: float = 0.9
    b2: float = 0.25
    eps: float = 1e-6
    eps_root: float = 0.0
    mu_dtype: jnp.dtype | None = None
    nu_dtype: jnp.dtype | None = None
    norm_mode: tp.Literal["sum", "mean"] = "sum"
    nesterov: bool = False
    mars_gamma: float = 0.0
    mars_beta: float | None = None
    mars_prev_grad_dtype: jnp.dtype = jnp.bfloat16
    grad_averaging: bool = True


@dataclasses.dataclass
class YogiConfig(SerializationMixin):
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-3
    eps_root: float = 0.0
    initial_accumulator_value: float = 1e-6
    mu_dtype: jnp.dtype | None = None
    nu_dtype: jnp.dtype | None = None
    nesterov: bool = False
    mars_gamma: float = 0.0
    mars_beta: float | None = None
    mars_prev_grad_dtype: jnp.dtype = jnp.bfloat16


@dataclasses.dataclass
class PrismConfig(SerializationMixin):
    momentum: float = 0.85
    nesterov: bool = True
    beta2: float = 0.95
    group_size: int = 128
    norm_group_size: int | None = None
    normuon_epsilon: float = 1e-8
    normuon_rms_scale: float = 0.2
    single_sided_prism_gamma: float | None = 0.5
    single_sided_prism_group_size: int | None = 128
    single_sided_prism_gram_epsilon: float = 1e-6
    single_sided_prism_inv_steps: int = 8
    single_sided_prism_inv_epsilon: float = 1e-5
    single_sided_prism_inv_scale: float = 1.001
    single_sided_prism_dtype: jnp.dtype = jnp.float32
    update_clamp_min: float | None = 0.35
    update_clamp_epsilon: float = 1e-8
    contra_muon: float = 0.1
    contra_muon_power_steps: int = 5
    contra_muon_epsilon: float = 1e-10
    use_cautioning: bool = True
    momentum_dtype: jnp.dtype = jnp.bfloat16
    norm_dtype: jnp.dtype = jnp.float32
    newton_schulz_dtype: jnp.dtype = jnp.bfloat16
    ns_steps: int = 5
    muon_epsilon: float = 1e-8
    max_grad_norm: float | None = None
    weight_decay: float = 0.0
    mlp_lr_multiplier: float = 1.0
    attn_lr_multiplier: float = 1.0
    embed_lr_multiplier: float = 1.0
    lm_head_lr_multiplier: float = 1.0
    norm_lr_multiplier: float = 1.0


@dataclasses.dataclass
class ScheduleFreePlusConfig(SerializationMixin):
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    eps_root: float = 0.0
    weight_decay: float = 0.0
    beta: float = 0.9
    beta_final: float | None = 0.965
    beta_anneal_steps: int = 0
    averaging_warmup_steps: int = 0
    r: float = 1.0
    weight_lr_power: float = 2.0
    polyak: bool = False
    inverse_l1_weighting: bool = False
    l1_beta: float = 0.9
    l1_eps: float = 1e-12
    normalize_l1_by_size: bool = False
    adamc_weight_decay: bool = True
    mu_dtype: jnp.dtype | None = None
    nu_dtype: jnp.dtype | None = None
    state_dtype: jnp.dtype | None = jnp.float32


@dataclasses.dataclass
class HyperballConfig(SerializationMixin):
    base_optimizer: tp.Literal[
        "adamw",
        "muon",
        "normuon",
        "novograd",
        "yogi",
        "mars",
        "prism",
        "schedule_free_plus",
    ] = "adamw"
    mask: tp.Any | None = None
    min_param_ndim: int = 2
    radius_scale: float = 1.0
    hyperball_eps: float = 1e-30
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.0
    beta: float = 0.95
    beta1: float = 0.95
    beta2: float = 0.99
    gamma: float = 0.025
    max_grad_norm: float | None = None
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.775, 2.0315)
    ns_steps: int = 5
    nesterov: bool = True
    adaptive: bool = False
    preconditioning: tp.Literal["frobenius", "spectral", "aol", "schatten"] = "frobenius"
    consistent_rms: float | None = 0.2
    muon_weight_dimension_numbers: tp.Any | None = None
    mu_dtype: jnp.dtype | None = None
    nu_dtype: jnp.dtype | None = None
    norm_mode: tp.Literal["sum", "mean"] = "sum"
    initial_accumulator_value: float = 1e-6
    mars_gamma: float = 0.0
    mars_beta: float | None = None
    mars_prev_grad_dtype: jnp.dtype = jnp.bfloat16
    prism_momentum: float = 0.85
    prism_beta2: float = 0.95
    prism_group_size: int = 128
    prism_norm_group_size: int | None = None
    prism_normuon_epsilon: float = 1e-8
    prism_normuon_rms_scale: float = 0.2
    prism_gamma: float | None = 0.5
    prism_grouped_size: int | None = 128
    prism_gram_epsilon: float = 1e-6
    prism_inv_steps: int = 8
    prism_inv_epsilon: float = 1e-5
    prism_inv_scale: float = 1.001
    prism_dtype: jnp.dtype = jnp.float32
    prism_update_clamp_min: float | None = 0.35
    prism_update_clamp_epsilon: float = 1e-8
    prism_contra_muon: float = 0.1
    prism_contra_muon_power_steps: int = 5
    prism_contra_muon_epsilon: float = 1e-10
    prism_use_cautioning: bool = True
    prism_momentum_dtype: jnp.dtype = jnp.bfloat16
    prism_norm_dtype: jnp.dtype = jnp.float32
    prism_newton_schulz_dtype: jnp.dtype = jnp.bfloat16
    prism_muon_epsilon: float = 1e-8
    prism_weight_decay: float = 0.0
    schedule_free_b1: float = 0.9
    schedule_free_b2: float = 0.95
    schedule_free_eps: float = 1e-8
    schedule_free_eps_root: float = 0.0
    schedule_free_weight_decay: float = 0.0
    schedule_free_beta: float = 0.9
    schedule_free_beta_final: float | None = 0.965
    schedule_free_beta_anneal_steps: int = 0
    schedule_free_averaging_warmup_steps: int = 0
    schedule_free_r: float = 1.0
    schedule_free_weight_lr_power: float = 2.0
    schedule_free_polyak: bool = False
    schedule_free_inverse_l1_weighting: bool = False
    schedule_free_l1_beta: float = 0.9
    schedule_free_l1_eps: float = 1e-12
    schedule_free_normalize_l1_by_size: bool = False
    schedule_free_adamc_weight_decay: bool = True
    schedule_free_normalize_update: bool = True
    schedule_free_mu_dtype: jnp.dtype | None = None
    schedule_free_nu_dtype: jnp.dtype | None = None
    schedule_free_state_dtype: jnp.dtype | None = jnp.float32


@register_optimizer("normuon")
@dataclasses.dataclass
class NormuonOptimizer(OptimizerBuilder):
    config: NormuonConfig

    def build(self, scheduler):
        return optax.contrib.muon(
            learning_rate=scheduler,
            ns_steps=self.config.ns_steps,
            ns_coeffs=self.config.ns_coeffs,
            beta=self.config.beta,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
            weight_decay_mask=self.config.weight_decay_mask,
            mu_dtype=self.config.mu_dtype,
            nesterov=self.config.nesterov,
            adaptive=self.config.adaptive,
            preconditioning=self.config.preconditioning,
            adam_b1=self.config.adam_b1,
            adam_b2=self.config.adam_b2,
            adam_eps_root=self.config.adam_eps_root,
            adam_weight_decay=self.config.adam_weight_decay,
            adam_learning_rate=self.config.adam_learning_rate,
            muon_weight_dimension_numbers=self.config.muon_weight_dimension_numbers,
            consistent_rms=self.config.consistent_rms,
        )

    def build_mpmd(self, scheduler, *, optimizer=None, **tx_kwargs):
        raise NotImplementedError(
            "normuon has no stage-local kernel; use optimizer='muon' for pipeline-parallel/MPMD training."
        )


@register_optimizer("novograd")
@dataclasses.dataclass
class NovoGradOptimizer(OptimizerBuilder):
    config: NovoGradConfig

    def validate(self) -> None:
        if self.config.norm_mode not in ("sum", "mean"):
            raise ValueError("NovoGradConfig.norm_mode must be 'sum' or 'mean'.")

    def build(self, scheduler):
        return fused_novograd(
            learning_rate=scheduler,
            b1=self.config.b1,
            b2=self.config.b2,
            eps=self.config.eps,
            eps_root=self.config.eps_root,
            weight_decay=0.0,
            mu_dtype=self.config.mu_dtype,
            nu_dtype=self.config.nu_dtype,
            norm_mode=self.config.norm_mode,
            nesterov=self.config.nesterov,
            mars_gamma=self.config.mars_gamma,
            mars_beta=self.config.mars_beta,
            mars_prev_grad_dtype=self.config.mars_prev_grad_dtype,
            grad_averaging=self.config.grad_averaging,
        )

    def build_mpmd(self, scheduler, *, optimizer=None, **tx_kwargs):
        raise NotImplementedError("novograd has no stage-local kernel yet.")


@register_optimizer("yogi")
@dataclasses.dataclass
class YogiOptimizer(OptimizerBuilder):
    config: YogiConfig

    def build(self, scheduler):
        return fused_yogi(
            learning_rate=scheduler,
            b1=self.config.b1,
            b2=self.config.b2,
            eps=self.config.eps,
            eps_root=self.config.eps_root,
            initial_accumulator_value=self.config.initial_accumulator_value,
            weight_decay=0.0,
            mu_dtype=self.config.mu_dtype,
            nu_dtype=self.config.nu_dtype,
            nesterov=self.config.nesterov,
            mars_gamma=self.config.mars_gamma,
            mars_beta=self.config.mars_beta,
            mars_prev_grad_dtype=self.config.mars_prev_grad_dtype,
        )

    def build_mpmd(self, scheduler, *, optimizer=None, **tx_kwargs):
        raise NotImplementedError("yogi has no stage-local kernel yet.")


@register_optimizer("prism")
@dataclasses.dataclass
class PrismOptimizer(OptimizerBuilder):
    config: PrismConfig

    def validate(self) -> None:
        if self.config.group_size <= 0:
            raise ValueError("PrismConfig.group_size must be > 0.")
        if self.config.norm_group_size is not None and self.config.norm_group_size <= 0:
            raise ValueError("PrismConfig.norm_group_size must be > 0 when set.")
        if not 0.0 <= self.config.beta2 < 1.0:
            raise ValueError("PrismConfig.beta2 must be in [0, 1).")
        if self.config.normuon_epsilon < 0.0:
            raise ValueError("PrismConfig.normuon_epsilon must be >= 0.")
        if self.config.normuon_rms_scale < 0.0:
            raise ValueError("PrismConfig.normuon_rms_scale must be >= 0.")
        if self.config.single_sided_prism_gamma is not None and self.config.single_sided_prism_gamma < 0.0:
            raise ValueError("PrismConfig.single_sided_prism_gamma must be >= 0 when set.")
        if self.config.single_sided_prism_group_size is not None and self.config.single_sided_prism_group_size <= 0:
            raise ValueError("PrismConfig.single_sided_prism_group_size must be > 0 when set.")
        if self.config.single_sided_prism_gram_epsilon <= 0.0:
            raise ValueError("PrismConfig.single_sided_prism_gram_epsilon must be > 0.")
        if self.config.single_sided_prism_inv_steps <= 0:
            raise ValueError("PrismConfig.single_sided_prism_inv_steps must be > 0.")
        if self.config.single_sided_prism_inv_epsilon <= 0.0:
            raise ValueError("PrismConfig.single_sided_prism_inv_epsilon must be > 0.")
        if self.config.single_sided_prism_inv_scale <= 0.0:
            raise ValueError("PrismConfig.single_sided_prism_inv_scale must be > 0.")
        if self.config.update_clamp_min is not None and self.config.update_clamp_min < 0.0:
            raise ValueError("PrismConfig.update_clamp_min must be >= 0 when set.")
        if self.config.update_clamp_epsilon <= 0.0:
            raise ValueError("PrismConfig.update_clamp_epsilon must be > 0.")
        if self.config.contra_muon < 0.0:
            raise ValueError("PrismConfig.contra_muon must be >= 0.")
        if self.config.contra_muon_power_steps <= 0:
            raise ValueError("PrismConfig.contra_muon_power_steps must be > 0.")
        if self.config.contra_muon_epsilon <= 0.0:
            raise ValueError("PrismConfig.contra_muon_epsilon must be > 0.")
        if self.config.ns_steps <= 0:
            raise ValueError("PrismConfig.ns_steps must be > 0.")
        if self.config.muon_epsilon <= 0.0:
            raise ValueError("PrismConfig.muon_epsilon must be > 0.")
        if self.config.max_grad_norm is not None and self.config.max_grad_norm <= 0.0:
            raise ValueError("PrismConfig.max_grad_norm must be > 0 when set.")
        if self.config.weight_decay < 0.0:
            raise ValueError("PrismConfig.weight_decay must be >= 0.")

    def build(self, scheduler):
        return prism_muon(
            learning_rate=scheduler,
            momentum=self.config.momentum,
            nesterov=self.config.nesterov,
            beta2=self.config.beta2,
            group_size=self.config.group_size,
            norm_group_size=self.config.norm_group_size,
            normuon_epsilon=self.config.normuon_epsilon,
            normuon_rms_scale=self.config.normuon_rms_scale,
            single_sided_prism_gamma=self.config.single_sided_prism_gamma,
            single_sided_prism_group_size=self.config.single_sided_prism_group_size,
            single_sided_prism_gram_epsilon=self.config.single_sided_prism_gram_epsilon,
            single_sided_prism_inv_steps=self.config.single_sided_prism_inv_steps,
            single_sided_prism_inv_epsilon=self.config.single_sided_prism_inv_epsilon,
            single_sided_prism_inv_scale=self.config.single_sided_prism_inv_scale,
            single_sided_prism_dtype=self.config.single_sided_prism_dtype,
            update_clamp_min=self.config.update_clamp_min,
            update_clamp_epsilon=self.config.update_clamp_epsilon,
            contra_muon=self.config.contra_muon,
            contra_muon_power_steps=self.config.contra_muon_power_steps,
            contra_muon_epsilon=self.config.contra_muon_epsilon,
            use_cautioning=self.config.use_cautioning,
            momentum_dtype=self.config.momentum_dtype,
            norm_dtype=self.config.norm_dtype,
            newton_schulz_dtype=self.config.newton_schulz_dtype,
            ns_steps=self.config.ns_steps,
            muon_epsilon=self.config.muon_epsilon,
            max_grad_norm=self.config.max_grad_norm,
            weight_decay=self.config.weight_decay,
            mlp_lr_multiplier=self.config.mlp_lr_multiplier,
            attn_lr_multiplier=self.config.attn_lr_multiplier,
            embed_lr_multiplier=self.config.embed_lr_multiplier,
            lm_head_lr_multiplier=self.config.lm_head_lr_multiplier,
            norm_lr_multiplier=self.config.norm_lr_multiplier,
        )

    def build_mpmd(self, scheduler, *, optimizer=None, **tx_kwargs):
        raise NotImplementedError("prism has no stage-local kernel yet.")


@register_optimizer("schedule_free_plus")
@register_optimizer("schedulefree_plus")
@dataclasses.dataclass
class ScheduleFreePlusOptimizer(OptimizerBuilder):
    config: ScheduleFreePlusConfig
    skip_external_weight_decay: tp.ClassVar[bool] = True

    def validate(self) -> None:
        if not 0.0 <= self.config.b1 < 1.0:
            raise ValueError("ScheduleFreePlusConfig.b1 must be in [0, 1).")
        if not 0.0 <= self.config.b2 < 1.0:
            raise ValueError("ScheduleFreePlusConfig.b2 must be in [0, 1).")
        if self.config.eps <= 0.0:
            raise ValueError("ScheduleFreePlusConfig.eps must be > 0.")
        if self.config.eps_root < 0.0:
            raise ValueError("ScheduleFreePlusConfig.eps_root must be >= 0.")
        if self.config.weight_decay < 0.0:
            raise ValueError("ScheduleFreePlusConfig.weight_decay must be >= 0.")
        if not 0.0 <= self.config.beta < 1.0:
            raise ValueError("ScheduleFreePlusConfig.beta must be in [0, 1).")
        if self.config.beta_final is not None and not 0.0 <= self.config.beta_final < 1.0:
            raise ValueError("ScheduleFreePlusConfig.beta_final must be in [0, 1) when set.")
        if self.config.beta_anneal_steps < 0:
            raise ValueError("ScheduleFreePlusConfig.beta_anneal_steps must be >= 0.")
        if self.config.averaging_warmup_steps < 0:
            raise ValueError("ScheduleFreePlusConfig.averaging_warmup_steps must be >= 0.")
        if self.config.r < 0.0:
            raise ValueError("ScheduleFreePlusConfig.r must be >= 0.")
        if self.config.weight_lr_power < 0.0:
            raise ValueError("ScheduleFreePlusConfig.weight_lr_power must be >= 0.")
        if not 0.0 <= self.config.l1_beta < 1.0:
            raise ValueError("ScheduleFreePlusConfig.l1_beta must be in [0, 1).")
        if self.config.l1_eps <= 0.0:
            raise ValueError("ScheduleFreePlusConfig.l1_eps must be > 0.")

    def build(self, scheduler):
        return schedule_free_plus_adamw(
            scheduler,
            b1=self.config.b1,
            b2=self.config.b2,
            eps=self.config.eps,
            eps_root=self.config.eps_root,
            weight_decay=self.config.weight_decay,
            beta=self.config.beta,
            beta_final=self.config.beta_final,
            beta_anneal_steps=self.config.beta_anneal_steps,
            averaging_warmup_steps=self.config.averaging_warmup_steps,
            r=self.config.r,
            weight_lr_power=self.config.weight_lr_power,
            polyak=self.config.polyak,
            inverse_l1_weighting=self.config.inverse_l1_weighting,
            l1_beta=self.config.l1_beta,
            l1_eps=self.config.l1_eps,
            normalize_l1_by_size=self.config.normalize_l1_by_size,
            adamc_weight_decay=self.config.adamc_weight_decay,
            mu_dtype=self.config.mu_dtype,
            nu_dtype=self.config.nu_dtype,
            state_dtype=self.config.state_dtype,
        )

    def build_mpmd(self, scheduler, *, optimizer=None, **tx_kwargs):
        raise NotImplementedError("schedule_free_plus has no stage-local kernel yet.")


def _build_hyperball_inner(config: HyperballConfig, scheduler: optax.Schedule, base_optimizer: str):
    if base_optimizer == "adamw":
        return fused_adamw(
            learning_rate=scheduler,
            b1=config.b1,
            b2=config.b2,
            eps=config.eps,
            eps_root=config.eps_root,
            mu_dtype=config.mu_dtype,
            weight_decay=0.0,
        )
    if base_optimizer in ("muon", "normuon"):
        muon_weight_dimension_numbers = (
            config.muon_weight_dimension_numbers
            if config.muon_weight_dimension_numbers is not None
            else _default_muon_hyperball_dimension_numbers
        )
        return optax.contrib.muon(
            learning_rate=scheduler,
            ns_steps=config.ns_steps,
            ns_coeffs=config.ns_coeffs,
            beta=config.beta,
            eps=config.eps,
            weight_decay=0.0,
            mu_dtype=config.mu_dtype,
            nesterov=config.nesterov,
            adaptive=config.adaptive,
            preconditioning=config.preconditioning,
            adam_b1=config.b1,
            adam_b2=config.b2,
            adam_eps_root=config.eps_root,
            adam_weight_decay=0.0,
            adam_learning_rate=scheduler,
            muon_weight_dimension_numbers=muon_weight_dimension_numbers,
            consistent_rms=config.consistent_rms if base_optimizer == "normuon" else None,
        )
    if base_optimizer == "novograd":
        return fused_novograd(
            learning_rate=scheduler,
            b1=config.b1,
            b2=config.b2,
            eps=config.eps,
            eps_root=config.eps_root,
            weight_decay=0.0,
            mu_dtype=config.mu_dtype,
            nu_dtype=config.nu_dtype,
            norm_mode=config.norm_mode,
            nesterov=config.nesterov,
            mars_gamma=config.mars_gamma,
            mars_beta=config.mars_beta,
            mars_prev_grad_dtype=config.mars_prev_grad_dtype,
        )
    if base_optimizer == "yogi":
        return fused_yogi(
            learning_rate=scheduler,
            b1=config.b1,
            b2=config.b2,
            eps=config.eps,
            eps_root=config.eps_root,
            initial_accumulator_value=config.initial_accumulator_value,
            weight_decay=0.0,
            mu_dtype=config.mu_dtype,
            nu_dtype=config.nu_dtype,
            nesterov=config.nesterov,
            mars_gamma=config.mars_gamma,
            mars_beta=config.mars_beta,
            mars_prev_grad_dtype=config.mars_prev_grad_dtype,
        )
    if base_optimizer == "mars":
        return eformer_mars(
            learning_rate=scheduler,
            b1=config.beta1,
            b2=config.beta2,
            gamma=config.gamma,
            eps=config.eps,
            max_grad_norm=0.0 if config.max_grad_norm is None else config.max_grad_norm,
        )
    if base_optimizer == "prism":
        return prism_muon(
            learning_rate=scheduler,
            momentum=config.prism_momentum,
            nesterov=config.nesterov,
            beta2=config.prism_beta2,
            group_size=config.prism_group_size,
            norm_group_size=config.prism_norm_group_size,
            normuon_epsilon=config.prism_normuon_epsilon,
            normuon_rms_scale=config.prism_normuon_rms_scale,
            single_sided_prism_gamma=config.prism_gamma,
            single_sided_prism_group_size=config.prism_grouped_size,
            single_sided_prism_gram_epsilon=config.prism_gram_epsilon,
            single_sided_prism_inv_steps=config.prism_inv_steps,
            single_sided_prism_inv_epsilon=config.prism_inv_epsilon,
            single_sided_prism_inv_scale=config.prism_inv_scale,
            single_sided_prism_dtype=config.prism_dtype,
            update_clamp_min=config.prism_update_clamp_min,
            update_clamp_epsilon=config.prism_update_clamp_epsilon,
            contra_muon=config.prism_contra_muon,
            contra_muon_power_steps=config.prism_contra_muon_power_steps,
            contra_muon_epsilon=config.prism_contra_muon_epsilon,
            use_cautioning=config.prism_use_cautioning,
            momentum_dtype=config.prism_momentum_dtype,
            norm_dtype=config.prism_norm_dtype,
            newton_schulz_dtype=config.prism_newton_schulz_dtype,
            ns_steps=config.ns_steps,
            muon_epsilon=config.prism_muon_epsilon,
            max_grad_norm=config.max_grad_norm,
            weight_decay=config.prism_weight_decay,
        )
    if base_optimizer == "schedule_free_plus":
        return schedule_free_plus_adamw(
            learning_rate=scheduler,
            b1=config.schedule_free_b1,
            b2=config.schedule_free_b2,
            eps=config.schedule_free_eps,
            eps_root=config.schedule_free_eps_root,
            weight_decay=config.schedule_free_weight_decay,
            beta=config.schedule_free_beta,
            beta_final=config.schedule_free_beta_final,
            beta_anneal_steps=config.schedule_free_beta_anneal_steps,
            averaging_warmup_steps=config.schedule_free_averaging_warmup_steps,
            r=config.schedule_free_r,
            weight_lr_power=config.schedule_free_weight_lr_power,
            polyak=config.schedule_free_polyak,
            inverse_l1_weighting=config.schedule_free_inverse_l1_weighting,
            l1_beta=config.schedule_free_l1_beta,
            l1_eps=config.schedule_free_l1_eps,
            normalize_l1_by_size=config.schedule_free_normalize_l1_by_size,
            adamc_weight_decay=config.schedule_free_adamc_weight_decay,
            mu_dtype=config.schedule_free_mu_dtype,
            nu_dtype=config.schedule_free_nu_dtype,
            state_dtype=config.schedule_free_state_dtype,
        )
    raise ValueError(f"Unsupported Hyperball base optimizer: {base_optimizer!r}.")


@register_optimizer("hyperball")
@dataclasses.dataclass
class HyperballOptimizer(OptimizerBuilder):
    config: HyperballConfig
    skip_external_weight_decay: tp.ClassVar[bool] = True
    default_base_optimizer: tp.ClassVar[str | None] = None

    def _base_optimizer(self) -> str:
        return self.default_base_optimizer or self.config.base_optimizer

    def validate(self) -> None:
        if self._base_optimizer() not in (
            "adamw",
            "muon",
            "normuon",
            "novograd",
            "yogi",
            "mars",
            "prism",
            "schedule_free_plus",
        ):
            raise ValueError(
                "Hyperball base_optimizer must be one of: adamw, muon, normuon, novograd, yogi, mars, prism, "
                "schedule_free_plus."
            )
        if self.config.min_param_ndim < 0:
            raise ValueError("HyperballConfig.min_param_ndim must be >= 0.")
        if self.config.radius_scale <= 0.0:
            raise ValueError("HyperballConfig.radius_scale must be > 0.")
        if self.config.hyperball_eps <= 0.0:
            raise ValueError("HyperballConfig.hyperball_eps must be > 0.")

    def build(self, scheduler):
        if self._base_optimizer() == "schedule_free_plus":
            return schedule_free_plus_hyperball_adamw(
                scheduler,
                mask=self.config.mask,
                min_param_ndim=self.config.min_param_ndim,
                radius_scale=self.config.radius_scale,
                hyperball_eps=self.config.hyperball_eps,
                b1=self.config.schedule_free_b1,
                b2=self.config.schedule_free_b2,
                eps=self.config.schedule_free_eps,
                eps_root=self.config.schedule_free_eps_root,
                weight_decay=self.config.schedule_free_weight_decay,
                beta=self.config.schedule_free_beta,
                beta_final=self.config.schedule_free_beta_final,
                beta_anneal_steps=self.config.schedule_free_beta_anneal_steps,
                averaging_warmup_steps=self.config.schedule_free_averaging_warmup_steps,
                r=self.config.schedule_free_r,
                weight_lr_power=self.config.schedule_free_weight_lr_power,
                polyak=self.config.schedule_free_polyak,
                inverse_l1_weighting=self.config.schedule_free_inverse_l1_weighting,
                l1_beta=self.config.schedule_free_l1_beta,
                l1_eps=self.config.schedule_free_l1_eps,
                normalize_l1_by_size=self.config.schedule_free_normalize_l1_by_size,
                adamc_weight_decay=self.config.schedule_free_adamc_weight_decay,
                normalize_update=self.config.schedule_free_normalize_update,
                mu_dtype=self.config.schedule_free_mu_dtype,
                nu_dtype=self.config.schedule_free_nu_dtype,
                state_dtype=self.config.schedule_free_state_dtype,
            )
        return hyperball(
            _build_hyperball_inner(self.config, scheduler, self._base_optimizer()),
            scheduler,
            mask=self.config.mask,
            min_param_ndim=self.config.min_param_ndim,
            radius_scale=self.config.radius_scale,
            eps=self.config.hyperball_eps,
        )

    def build_mpmd(self, scheduler, *, optimizer=None, **tx_kwargs):
        raise NotImplementedError("hyperball has no stage-local kernel yet.")


@register_optimizer("adamw_hyperball")
@dataclasses.dataclass
class AdamWHyperballOptimizer(HyperballOptimizer):
    default_base_optimizer: tp.ClassVar[str | None] = "adamw"


@register_optimizer("muon_hyperball")
@dataclasses.dataclass
class MuonHyperballOptimizer(HyperballOptimizer):
    default_base_optimizer: tp.ClassVar[str | None] = "muon"


@register_optimizer("normuon_hyperball")
@dataclasses.dataclass
class NormuonHyperballOptimizer(HyperballOptimizer):
    default_base_optimizer: tp.ClassVar[str | None] = "normuon"


@register_optimizer("novograd_hyperball")
@dataclasses.dataclass
class NovoGradHyperballOptimizer(HyperballOptimizer):
    default_base_optimizer: tp.ClassVar[str | None] = "novograd"


@register_optimizer("yogi_hyperball")
@dataclasses.dataclass
class YogiHyperballOptimizer(HyperballOptimizer):
    default_base_optimizer: tp.ClassVar[str | None] = "yogi"


@register_optimizer("mars_hyperball")
@dataclasses.dataclass
class MarsHyperballOptimizer(HyperballOptimizer):
    default_base_optimizer: tp.ClassVar[str | None] = "mars"


@register_optimizer("prism_hyperball")
@dataclasses.dataclass
class PrismHyperballOptimizer(HyperballOptimizer):
    default_base_optimizer: tp.ClassVar[str | None] = "prism"


@register_optimizer("schedule_free_plus_hyperball")
@register_optimizer("schedulefree_plus_hyperball")
@dataclasses.dataclass
class ScheduleFreePlusHyperballOptimizer(HyperballOptimizer):
    default_base_optimizer: tp.ClassVar[str | None] = "schedule_free_plus"
