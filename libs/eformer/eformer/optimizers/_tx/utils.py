# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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


import functools
import re
import typing as tp
import warnings

import chex
import jax
import optax
from jax import numpy as jnp


def _pytree_path_to_str(path: tuple[tp.Any, ...]) -> str:
    """Format a JAX pytree KeyPath into a canonical dot-separated string."""
    parts: list[str] = []
    for entry in path:
        if hasattr(entry, "key"):
            parts.append(str(entry.key))
        elif hasattr(entry, "name"):
            parts.append(str(entry.name))
        elif hasattr(entry, "idx"):
            parts.append(str(entry.idx))
        else:
            parts.append(str(entry))
    return ".".join(parts)


def build_path_exclusion_mask(
    patterns: tp.Sequence[str],
    base_mask: chex.ArrayTree | tp.Callable[[chex.ArrayTree], chex.ArrayTree] | None = None,
) -> tp.Callable[[chex.ArrayTree], chex.ArrayTree]:
    """Build an optax-compatible mask function excluding leaves matching regex ``patterns``.

    Args:
        patterns: Sequence of regular expressions matched via ``re.search`` against
            each leaf's dot-joined pytree path (e.g. ``["loop_gates", "loop_iter_embeddings"]``).
        base_mask: Optional existing boolean pytree or callable mask to combine via logical AND.

    Returns:
        Callable ``mask_fn(params) -> bool_pytree`` returning ``False`` on excluded
        leaves and ``True`` otherwise.
    """
    compiled = tuple(re.compile(str(p)) for p in patterns if str(p))

    def _mask_fn(params: chex.ArrayTree) -> chex.ArrayTree:
        excl_tree = jax.tree_util.tree_map_with_path(
            lambda path, _leaf: not any(pat.search(_pytree_path_to_str(path)) for pat in compiled),
            params,
        )
        if base_mask is None:
            return excl_tree
        resolved_base = base_mask(params) if callable(base_mask) else base_mask
        return jax.tree_util.tree_map(lambda a, b: bool(a) and bool(b), resolved_base, excl_tree)

    return _mask_fn


def optax_scale_by_path_multipliers(
    lr_multipliers: tp.Mapping[str, float],
) -> optax.GradientTransformation:
    """Scale optimizer updates per parameter leaf based on regex/substring path rules.

    Stateless (``optax.EmptyState``) transformation intended to sit at the end of an
    optimizer chain (after base adaptive scaling and weight decay). First matching
    pattern in ``lr_multipliers`` insertion order wins; unmatched leaves keep ``1.0``.

    Args:
        lr_multipliers: Mapping ``{pattern: multiplier}`` with finite multipliers >= 0.

    Returns:
        An ``optax.GradientTransformation`` with zero optimizer state overhead.
    """
    compiled: list[tuple[re.Pattern[str], float]] = []
    for pat, raw_mult in lr_multipliers.items():
        mult = float(raw_mult)
        if mult < 0.0 or mult != mult or mult == float("inf"):
            raise ValueError(f"LR multiplier for pattern {pat!r} must be finite and >= 0, got {raw_mult!r}.")
        compiled.append((re.compile(str(pat)), mult))
    rules = tuple(compiled)

    def init_fn(params: chex.ArrayTree) -> optax.EmptyState:
        del params
        return optax.EmptyState()

    def update_fn(
        updates: chex.ArrayTree,
        state: optax.EmptyState,
        params: chex.ArrayTree | None = None,
    ) -> tuple[chex.ArrayTree, optax.EmptyState]:
        del params

        def _scale_leaf(path: tuple[tp.Any, ...], upd: tp.Any) -> tp.Any:
            if upd is None:
                return None
            path_str = _pytree_path_to_str(path)
            for pat, mult in rules:
                if pat.search(path_str):
                    if mult == 1.0:
                        return upd
                    return upd * jnp.asarray(mult, dtype=upd.dtype)
            return upd

        scaled = jax.tree_util.tree_map_with_path(_scale_leaf, updates)
        return scaled, state

    return optax.GradientTransformation(init_fn, update_fn)


def _stochastic_cast_bf16_bits_impl(x_f32: jax.Array, rand16: jax.Array) -> jax.Array:
    """Exact IEEE-754 float32 -> bfloat16 stochastic rounding given uint32 ``rand16`` in ``[0, 65535]``.

    Adds ``rand16`` into the low 16 discarded mantissa bits of the sign-magnitude float32
    representation and truncates bits ``0..15``. Carries into bit 16 with exact probability
    ``(bits & 0xFFFF) / 65536`` across every normal, subnormal, power-of-2, positive, and
    negative finite value while never perturbing exact zeros or already-exact bfloat16 values.
    """
    u = jax.lax.bitcast_convert_type(x_f32, jnp.uint32)
    rounded_u = (u + (rand16.astype(jnp.uint32) & jnp.uint32(0xFFFF))) & jnp.uint32(0xFFFF0000)
    rounded_bf16 = jax.lax.bitcast_convert_type(rounded_u, jnp.float32).astype(jnp.bfloat16)
    return jnp.where(jnp.isfinite(x_f32), rounded_bf16, x_f32.astype(jnp.bfloat16))


@jax.custom_vjp
def stochastic_cast_bf16_with_bits(x: jax.Array, rand16: jax.Array) -> jax.Array:
    """Stochastically cast ``x`` to ``bfloat16`` using caller-supplied 16-bit uniform bits (STE backward)."""
    return _stochastic_cast_bf16_bits_impl(jnp.asarray(x, dtype=jnp.float32), rand16)


def _stochastic_cast_bf16_with_bits_fwd(x: jax.Array, rand16: jax.Array):
    x_arr = jnp.asarray(x)
    out = _stochastic_cast_bf16_bits_impl(x_arr.astype(jnp.float32), rand16)
    return out, jnp.empty((), dtype=x_arr.dtype)


def _stochastic_cast_bf16_with_bits_bwd(x_proto: jax.Array, g: jax.Array):
    return g.astype(x_proto.dtype), None


stochastic_cast_bf16_with_bits.defvjp(_stochastic_cast_bf16_with_bits_fwd, _stochastic_cast_bf16_with_bits_bwd)


def _stochastic_cast_impl(x: jax.Array, dtype: jnp.dtype, key: jax.Array) -> jax.Array:
    """Unbiased stochastic rounding from wider floats into ``dtype``."""
    target_dtype = jnp.dtype(dtype)
    if not jnp.issubdtype(target_dtype, jnp.floating):
        raise ValueError(f"stochastic_cast requires a floating-point target dtype, got {target_dtype.name}.")

    x_arr = jnp.asarray(x)
    if jnp.finfo(target_dtype).bits >= 32:
        return x_arr.astype(target_dtype)

    x_f32 = x_arr.astype(jnp.float32)

    if target_dtype == jnp.bfloat16:
        rand16 = jax.random.bits(key, x_f32.shape, dtype=jnp.uint32) >> jnp.uint32(16)
        return _stochastic_cast_bf16_bits_impl(x_f32, rand16)

    if target_dtype == jnp.float16:
        x_rt_narrow = x_f32.astype(target_dtype)
        x_rt = x_rt_narrow.astype(jnp.float32)
        direction = jnp.where(x_f32 >= x_rt, jnp.inf, -jnp.inf).astype(target_dtype)
        neighbor = jnp.nextafter(x_rt_narrow, direction).astype(jnp.float32)
        lo = jnp.minimum(x_rt, neighbor)
        hi = jnp.maximum(x_rt, neighbor)
        span = hi - lo
        safe_span = jnp.where(span > 0.0, span, 1.0)
        prob_hi = jnp.where(span > 0.0, (x_f32 - lo) / safe_span, 0.0)
        u01 = jax.random.uniform(key, x_f32.shape, dtype=jnp.float32)
        rounded = jnp.where(u01 < prob_hi, hi, lo).astype(target_dtype)
        valid = jnp.isfinite(x_f32) & jnp.isfinite(lo) & jnp.isfinite(hi)
        return jnp.where(valid, rounded, x_rt_narrow)

    # General narrow floats (e.g. float8_e4m3fn / float8_e5m2): exact bin ULP = 2^floor(log2(|x|)) * eps.
    finfo = jnp.finfo(target_dtype)
    tiny = float(finfo.tiny)
    eps = float(finfo.eps)
    abs_x = jnp.abs(x_f32)
    ulp = jnp.where(abs_x >= tiny, jnp.exp2(jnp.floor(jnp.log2(jnp.maximum(abs_x, tiny)))) * eps, tiny * eps)
    noise = jax.random.uniform(key, x_f32.shape, dtype=jnp.float32, minval=-0.5, maxval=0.5)
    rounded = (x_f32 + noise * ulp).astype(target_dtype)
    return jnp.where(jnp.isfinite(x_f32), rounded, x_f32.astype(target_dtype))


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _stochastic_cast_ste(x: jax.Array, key: jax.Array, dtype: jnp.dtype) -> jax.Array:
    return _stochastic_cast_impl(x, dtype, key)


def _stochastic_cast_ste_fwd(x: jax.Array, key: jax.Array, dtype: jnp.dtype):
    x_arr = jnp.asarray(x)
    return _stochastic_cast_impl(x_arr, dtype, key), jnp.empty((), dtype=x_arr.dtype)


def _stochastic_cast_ste_bwd(dtype: jnp.dtype, x_proto: jax.Array, g: jax.Array):
    del dtype
    return g.astype(x_proto.dtype), None


_stochastic_cast_ste.defvjp(_stochastic_cast_ste_fwd, _stochastic_cast_ste_bwd)


def stochastic_cast(x: jax.Array, dtype: tp.Any, key: jax.Array) -> jax.Array:
    """Stochastically round ``x`` to floating-point ``dtype`` with STE backward support.

    For ``bfloat16``, uses exact IEEE-754 bitwise 16-bit stochastic rounding (zero
    perturbation on exact bfloat16 values, exact probability ``frac / 65536`` across
    normals, subnormals, power-of-2 boundaries, and signed values). For ``float16``,
    uses exact ``jnp.nextafter`` bracketing. For narrower floats (``float8_*``), uses
    bin-exact ``2^floor(log2(|x|)) * eps`` ULP dither.

    Args:
        x: Input array (typically float32).
        dtype: Target floating-point dtype.
        key: JAX PRNG key.

    Returns:
        Array cast to ``dtype`` with unbiased expectation ``E[stochastic_cast(x)] = x``.
    """
    target_dtype = jnp.dtype(dtype)
    if not jnp.issubdtype(target_dtype, jnp.floating):
        raise ValueError(f"stochastic_cast requires a floating-point target dtype, got {target_dtype.name}.")
    return _stochastic_cast_ste(jnp.asarray(x), key, target_dtype)


class OptaxStochasticRoundingState(tp.NamedTuple):
    """Minimal 4-byte step counter state for :func:`optax_stochastic_round_updates`."""

    count: chex.Array


def optax_stochastic_round_updates(seed: int = 0) -> optax.GradientTransformation:
    """Quantize narrow-float parameter updates via unbiased stochastic rounding.

    Sits at the end of an Optax chain (inside :class:`optax.MultiSteps` when gradient
    accumulation is active). For every narrow-float parameter leaf (e.g. ``bfloat16``
    or ``float16``, where ``finfo(dtype).bits < 32``), computes the exact float32 target
    ``p_f32 + u_f32``, applies :func:`stochastic_cast` into ``p.dtype``, and returns
    ``delta = p_new.astype(float32) - p_f32`` so ``optax.apply_updates(params, updates)``
    lands on ``p_new`` bit-for-bit without needing FP32 master weights.

    Full-precision leaves (``float32`` / ``float64``) pass through untouched. Uses a
    deterministic counter-based key folded by ``(seed, step, leaf_index)`` so sharded
    FSDP/TP parameters generate chip-local random bits with zero cross-device communication.

    Args:
        seed: Base integer seed folded with step count and leaf index.

    Returns:
        An ``optax.GradientTransformation`` carrying only a scalar ``int32`` step counter.
    """
    base_seed = int(seed)

    def init_fn(params: chex.ArrayTree) -> OptaxStochasticRoundingState:
        del params
        return OptaxStochasticRoundingState(count=jnp.zeros((), dtype=jnp.int32))

    def update_fn(
        updates: chex.ArrayTree,
        state: OptaxStochasticRoundingState,
        params: chex.ArrayTree | None = None,
    ) -> tuple[chex.ArrayTree, OptaxStochasticRoundingState]:
        if params is None:
            raise ValueError("Params cannot be None for optax_stochastic_round_updates!")

        step_key = jax.random.fold_in(jax.random.key(base_seed), state.count)
        u_leaves, treedef = jax.tree_util.tree_flatten(updates, is_leaf=lambda x: x is None)
        p_leaves = jax.tree_util.tree_leaves(params, is_leaf=lambda x: x is None)

        new_leaves: list[tp.Any] = []
        for idx, (u_leaf, p_leaf) in enumerate(zip(u_leaves, p_leaves, strict=True)):
            if u_leaf is None or p_leaf is None:
                new_leaves.append(u_leaf)
                continue
            p_arr = jnp.asarray(p_leaf)
            if not jnp.issubdtype(p_arr.dtype, jnp.floating) or jnp.finfo(p_arr.dtype).bits >= 32:
                new_leaves.append(u_leaf)
                continue
            leaf_key = jax.random.fold_in(step_key, idx)
            p_f32 = p_arr.astype(jnp.float32)
            target_f32 = p_f32 + jnp.asarray(u_leaf, dtype=jnp.float32)
            p_new = _stochastic_cast_impl(target_f32, p_arr.dtype, leaf_key)
            # Returning fp32 delta makes optax.apply_updates(p, delta) == p_new bit-exact.
            new_leaves.append(p_new.astype(jnp.float32) - p_f32)

        rounded_updates = jax.tree_util.tree_unflatten(treedef, new_leaves)
        return rounded_updates, OptaxStochasticRoundingState(count=optax.safe_int32_increment(state.count))

    return optax.GradientTransformation(init_fn, update_fn)


class OptaxScheduledWeightDecayState(tp.NamedTuple):
    """State for the scheduled weight decay optimizer.

    This named tuple holds the state required by the scheduled weight decay
    transformation, tracking the current step count for schedule evaluation.

    Attributes:
        count (chex.Array): Integer array tracking the current optimization step.
            Used to evaluate the weight decay schedule function.
    """

    count: chex.Array


def optax_add_scheduled_weight_decay(
    schedule_fn: tp.Callable[[chex.Array], chex.Array],
    mask: chex.ArrayTree | None = None,
) -> optax.GradientTransformation:
    """
    Create an optax optimizer that applies weight decay on a schedule.

    This function is similar to `optax.add_decayed_weights`, but it allows for
    the weight decay rate to be scheduled over training steps.

    Args:
        schedule_fn: A function that takes the current step count as input
                      and returns the weight decay rate.
        mask: A PyTree with the same structure as the parameters.
              A value of True at a particular location indicates that weight
              decay should be applied to that parameter.

    Returns:
        An `optax.GradientTransformation` object representing the optimizer.
    """

    def init_fn(params: chex.ArrayTree) -> OptaxScheduledWeightDecayState:
        """Initialize the state of the scheduled weight decay optimizer.

        Args:
            params (chex.ArrayTree): Parameter tree (unused, but required by optax interface).

        Returns:
            OptaxScheduledWeightDecayState: Initial state with step count set to zero.
        """
        del params
        return OptaxScheduledWeightDecayState(count=jnp.zeros([], jnp.int32))

    def update_fn(
        updates: chex.ArrayTree,
        state: OptaxScheduledWeightDecayState,
        params: chex.ArrayTree | None = None,
    ) -> tuple[chex.ArrayTree, OptaxScheduledWeightDecayState]:
        """Apply scheduled weight decay to the gradient updates.

        Computes the weight decay rate from the schedule function and adds
        the scaled parameters to the gradient updates.

        Args:
            updates (chex.ArrayTree): Gradient updates to be modified.
            state (OptaxScheduledWeightDecayState): Current optimizer state containing step count.
            params (chex.ArrayTree | None): Model parameters for weight decay computation.

        Returns:
            tuple[chex.ArrayTree, OptaxScheduledWeightDecayState]: Tuple containing:
                - Modified gradient updates with weight decay applied.
                - Updated state with incremented step count.

        Raises:
            ValueError: If params is None, as weight decay requires parameter values.
        """
        if params is None:
            raise ValueError("Params cannot be None for weight decay!")

        weight_decay = schedule_fn(state.count)
        updates = jax.tree_util.tree_map(lambda g, p: g + weight_decay * p, updates, params)
        return updates, OptaxScheduledWeightDecayState(count=optax.safe_int32_increment(state.count))

    if mask is not None:
        return optax.masked(optax.GradientTransformation(init_fn, update_fn), mask)
    return optax.GradientTransformation(init_fn, update_fn)


def create_linear_scheduler(
    steps: int,
    learning_rate_start: float,
    learning_rate_end: float,
    warmup_steps: int | None = None,
) -> optax.Schedule:
    """
    Creates a linear learning rate scheduler with optional warmup.

    Args:
        steps (int): Total number of training steps.
        learning_rate_start (float): Initial learning rate.
        learning_rate_end (float): Final learning rate.
        warmup_steps (tp.Optional[int]): Number of warmup steps.

    Returns:
        optax.Schedule: The configured scheduler.
    """
    if warmup_steps:
        scheduler_warmup = optax.linear_schedule(
            init_value=5e-8,
            end_value=learning_rate_start,
            transition_steps=warmup_steps,
        )
        scheduler_decay = optax.linear_schedule(
            init_value=learning_rate_start,
            end_value=learning_rate_end,
            transition_steps=steps - warmup_steps,
        )
        return optax.join_schedules(schedules=[scheduler_warmup, scheduler_decay], boundaries=[warmup_steps])
    else:
        return optax.linear_schedule(
            init_value=learning_rate_start,
            end_value=learning_rate_end,
            transition_steps=steps,
        )


def create_cosine_scheduler(
    steps: int,
    learning_rate: float,
    learning_rate_end: float | None = None,
    warmup_steps: int | None = None,
    exponent: float = 1.0,
) -> optax.Schedule:
    """
    Creates a cosine learning rate scheduler with optional warmup.

    Args:
        steps (int): Total number of training steps.
        learning_rate (float): Peak learning rate.
        learning_rate_end (tp.Optional[float]): Final learning rate.
        warmup_steps (tp.Optional[int]): Number of warmup steps.
        exponent (float): Exponent for the cosine decay.

    Returns:
        optax.Schedule: The configured scheduler.
    """
    if warmup_steps:
        return optax.warmup_cosine_decay_schedule(
            init_value=0.5e-7,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=steps,
            end_value=learning_rate_end if learning_rate_end is not None else 0.0,
            exponent=exponent,
        )
    if learning_rate_end is not None and learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0 when learning_rate_end is set")

    cosine_alpha = 0.0 if learning_rate_end is None else learning_rate_end / learning_rate
    return optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=steps, alpha=cosine_alpha)


def get_base_optimizer(
    optimizer_type: str,
    scheduler: optax.Schedule,
    optimizer_kwargs: dict,
    weight_decay: float = 0.0,
    weight_decay_mask: tp.Any | None = None,
    gradient_accumulation_steps: int = 1,
    clip_grad: float | None = None,
    **kwargs,
) -> optax.GradientTransformation:
    """
    Base function to create an optimizer with a given scheduler.

    Args:
        optimizer_type (str): Type of optimizer ('adafactor', 'adamw', 'lion', 'rmsprop').
        scheduler (optax.Schedule): Learning rate scheduler.
        optimizer_kwargs (dict): Arguments specific to the optimizer.
        weight_decay (float): Weight decay factor.
        weight_decay_mask (tp.Optional[tp.Any]): Mask for weight decay.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        clip_grad (tp.Optional[float]): If provided, gradients will be clipped to this maximum norm.

    Returns:
        optax.GradientTransformation: The configured optimizer.
    """
    for kwarg in kwargs.keys():
        warnings.warn(f"Key {kwarg} is not used for optimizer.", stacklevel=1)

    if optimizer_type == "adafactor":
        optimizer = optax.adafactor(learning_rate=scheduler, **optimizer_kwargs)
    elif optimizer_type == "adamw":
        optimizer_kwargs = dict(optimizer_kwargs)
        optimizer_kwargs.setdefault("weight_decay", 0.0)
        optimizer = optax.adamw(learning_rate=scheduler, **optimizer_kwargs)
    elif optimizer_type == "lion":
        optimizer = optax.lion(learning_rate=scheduler, **optimizer_kwargs)
    elif optimizer_type == "rmsprop":
        optimizer = optax.rmsprop(learning_rate=scheduler, **optimizer_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    chain = [optimizer]

    if clip_grad is not None:
        chain.insert(0, optax.clip_by_global_norm(clip_grad))

    if weight_decay != 0.0:
        chain.append(
            optax_add_scheduled_weight_decay(
                lambda step: -scheduler(step) * weight_decay,
                weight_decay_mask,
            )
        )

    tx = optax.chain(*chain)

    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(tx, gradient_accumulation_steps)

    return tx
