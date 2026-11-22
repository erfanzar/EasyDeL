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
"""Native pytree merge methods used by :mod:`easydel.trainers.merge_callback`.

The merge callback operates on already-materialized EasyDeL pytrees. This
module owns the per-leaf algorithms and a small process-local registry so the
callback can stay independent from any specific merge method implementation.

Built-in methods:
    ``linear``:
        Weighted arithmetic interpolation of matching array leaves.
    ``slerp``:
        Spherical interpolation for dense floating-point tensors, with a linear
        fallback when the two leaves are nearly collinear.
    ``ties``:
        Single-target TIES-style task-vector merge using magnitude pruning.
    ``dare_ties``:
        Single-target DARE-TIES task-vector merge using deterministic random
        pruning and density rescaling.

The registry accepts any callable with the ``MergeMethod`` signature. Custom
methods are called once per matching array leaf and must return an array with
the same shape as the policy leaf.
"""

from __future__ import annotations

import functools
import typing as tp
from dataclasses import dataclass

import jax
import jax.numpy as jnp

MergeMethodName = tp.Literal["linear", "ties", "dare_ties", "slerp"]


@dataclass(frozen=True)
class MergeLeafContext:
    """Immutable per-array parameters passed to a merge method.

    ``merge_pytrees`` constructs one context for each pair of mergeable array
    leaves after it has validated that the policy and target pytrees share the
    same structure and that the two leaves have identical shapes. The context
    contains only scalar values that affect the leaf-level algorithm, so merge
    methods do not need to import or understand :class:`MergeConfig`.

    The weights are already normalized when ``MergeConfig.normalize`` is true.
    Density is selected from ``target_model_density`` using the leaf index modulo
    the density schedule length. ``seed`` and ``leaf_index`` are both included so
    stochastic methods can derive deterministic per-leaf PRNG keys without
    relying on global state.

    Attributes:
        policy_weight: Weight applied to the policy/base leaf. For linear
            interpolation this is the coefficient on ``policy``; task-vector
            methods usually keep the policy leaf as the base and use this value
            only if they explicitly need symmetric weighting.
        target_weight: Weight applied to the target leaf or target task vector.
            In ``ties`` and ``dare_ties`` this scales ``target - policy`` before
            it is added back to the policy leaf.
        density: Fraction of task-vector entries to keep for pruning-based
            methods. ``1.0`` keeps every entry; ``0.0`` keeps none. The config
            validator guarantees the value is in ``[0, 1]``.
        t_value: Interpolation parameter used by methods such as ``slerp``.
            ``0.0`` selects the policy leaf and ``1.0`` selects the target leaf.
        seed: User-provided integer seed for deterministic stochastic methods.
            It is not mutated; methods should combine it with ``leaf_index`` to
            produce per-leaf keys.
        leaf_index: Position of the leaf in the flattened pytree. This gives
            deterministic methods a stable identifier for density scheduling and
            random-key derivation.
    """

    policy_weight: float
    target_weight: float
    density: float
    t_value: float
    seed: int
    leaf_index: int


MergeMethod = tp.Callable[[jax.Array, jax.Array, MergeLeafContext], jax.Array]
_MERGE_METHODS: dict[str, MergeMethod] = {}


def register_merge_method(name: str, fn: MergeMethod) -> None:
    """Install or replace a leaf merge implementation in the local registry.

    Args:
        name: Public method name accepted by ``MergeConfig.method`` or by any
            code that calls :func:`get_merge_method` directly. Names are stored
            exactly as provided, so callers should normalize casing before
            registration if they want case-insensitive behavior.
        fn: Callable invoked as ``fn(policy_leaf, target_leaf, context)`` for
            every compatible array leaf. It must return a JAX array compatible
            with the original policy leaf shape.

    Behavior:
        Registration is process-local and intentionally overwrite-friendly.
        Re-registering an existing name replaces the previous callable, which is
        useful for downstream plugins and tests that need to override a built-in
        merge rule explicitly.
    """

    _MERGE_METHODS[name] = fn


def get_merge_method(name: str) -> MergeMethod:
    """Resolve a merge method name to its registered callable.

    Args:
        name: Registry key to look up.

    Returns:
        The callable previously installed for ``name``.

    Raises:
        ValueError: If ``name`` is unknown. The error message includes all known
            names so config mistakes are actionable without inspecting the
            registry manually.
    """

    try:
        return _MERGE_METHODS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_MERGE_METHODS))
        raise ValueError(f"Unknown merge method {name!r}. Known methods: {known}.") from exc


def registered_merge_methods() -> tuple[str, ...]:
    """Return all registered method names sorted for stable logs and tests.

    The registry is process-local and mutable, so this helper snapshots the
    current keys and sorts them before returning. Consumers can use the result
    for user-facing error messages, config validation, or deterministic unit
    assertions without exposing the underlying mutable dictionary.
    """

    return tuple(sorted(_MERGE_METHODS))


def _linear_merge(policy: jax.Array, target: jax.Array, ctx: MergeLeafContext) -> jax.Array:
    """Merge one leaf with weighted arithmetic interpolation.

    Computes ``policy_weight * policy + target_weight * target``. The caller is
    responsible for normalizing the two weights when requested by the config.
    This method preserves shape and dtype promotion follows normal JAX
    arithmetic rules.
    """

    return ctx.policy_weight * policy + ctx.target_weight * target


def _concrete_named_sharding(arr: jax.Array) -> jax.sharding.NamedSharding | None:
    """Return ``arr.sharding`` when ``arr`` is a concrete array with a ``NamedSharding``."""
    if isinstance(arr, jax.core.Tracer) or not isinstance(arr, jax.Array):
        return None
    sharding = getattr(arr, "sharding", None)
    return sharding if isinstance(sharding, jax.sharding.NamedSharding) else None


def _slerp_merge(policy: jax.Array, target: jax.Array, ctx: MergeLeafContext) -> jax.Array:
    """Merge one leaf using spherical linear interpolation in float32.

    The method treats both leaves as flattened vectors in at least float32 for
    computing the angle between them, then applies the scalar SLERP
    coefficients to the original-shaped arrays and casts back to the promoted
    input dtype. When either leaf has near-zero norm or the vectors are nearly
    collinear / antiparallel (``|dot| >= 0.9995`` or ``sin(omega) <= 1e-6``),
    it falls back cleanly to linear interpolation ``(1 - t) * policy + t * target``
    to avoid NaNs and antiparallel blowups.
    """
    out_dtype = jnp.result_type(policy, target)
    compute_dtype = jnp.promote_types(out_dtype, jnp.float32)
    policy_f = policy.astype(compute_dtype)
    target_f = target.astype(compute_dtype)

    t = jnp.asarray(ctx.t_value, dtype=compute_dtype)
    lerp = (1.0 - t) * policy_f + t * target_f

    policy_mag = jnp.sqrt(jnp.sum(policy_f * policy_f))
    target_mag = jnp.sqrt(jnp.sum(target_f * target_f))
    eps = jnp.asarray(1e-12, dtype=compute_dtype)
    valid_norms = (policy_mag > eps) & (target_mag > eps)

    one = jnp.asarray(1.0, dtype=compute_dtype)
    safe_policy_mag = jnp.where(policy_mag > eps, policy_mag, one)
    safe_target_mag = jnp.where(target_mag > eps, target_mag, one)
    policy_unit = policy_f / safe_policy_mag
    target_unit = target_f / safe_target_mag

    dot = jnp.clip(jnp.sum(policy_unit * target_unit), -1.0, 1.0)
    omega = jnp.arccos(dot)
    sin_omega = jnp.sin(omega)
    use_slerp = valid_norms & (jnp.abs(dot) < 0.9995) & (sin_omega > 1e-6)
    safe_sin_omega = jnp.where(use_slerp, sin_omega, one)

    coeff_policy = jnp.sin((1.0 - t) * omega) / safe_sin_omega
    coeff_target = jnp.sin(t * omega) / safe_sin_omega
    slerp = coeff_policy * policy_f + coeff_target * target_f

    return jnp.where(use_slerp, slerp, lerp).astype(out_dtype)


def _topk_density_mask_impl(values: jax.Array, keep: int) -> jax.Array:
    """Compute the bit-exact top-``keep`` magnitude mask without full-tensor gathering.

    Exploits the monotonicity of IEEE-754 non-negative floats under unsigned
    integer bitcasts so each step only performs an elementwise comparison and a
    scalar count reduction across shards.
    """
    use_64 = values.dtype == jnp.float64 and bool(jax.config.jax_enable_x64)
    float_dtype = jnp.float64 if use_64 else jnp.float32
    uint_dtype = jnp.uint64 if use_64 else jnp.uint32
    num_bits = 64 if use_64 else 32

    abs_f = jnp.abs(values).astype(float_dtype)
    # Canonicalize -0.0 -> +0.0 before bitcast; treat NaNs as zero magnitude so bounds stay finite.
    abs_f = jnp.where(jnp.isnan(abs_f), jnp.asarray(0.0, dtype=float_dtype), abs_f) + jnp.asarray(
        0.0, dtype=float_dtype
    )
    bits = jax.lax.bitcast_convert_type(abs_f, uint_dtype)
    lo = jnp.min(bits)
    hi = jnp.max(bits)
    one = jnp.asarray(1, dtype=uint_dtype)
    count_dtype = jnp.uint64 if bool(jax.config.jax_enable_x64) else jnp.uint32
    keep_target = jnp.asarray(keep, dtype=count_dtype)

    def body(_i: int, bounds: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        low, high = bounds
        diff = high - low
        mid = low + (diff >> one) + (diff & one)
        cnt = jnp.sum((bits >= mid).astype(count_dtype))
        go_right = cnt >= keep_target
        next_low = jnp.where(go_right, mid, low)
        next_high = jnp.where(go_right, high, jnp.where(mid > low, mid - one, low))
        return next_low, next_high

    threshold_bits, _ = jax.lax.fori_loop(0, num_bits, body, (lo, hi))
    return bits >= threshold_bits


@functools.partial(jax.jit, static_argnames=("sharding",))
def _topk_density_mask_sharded(values: jax.Array, keep: jax.Array, sharding: tp.Any) -> jax.Array:
    """Hoisted sharding-preserving JIT helper for :func:`_topk_density_mask`."""
    values = jax.lax.with_sharding_constraint(values, sharding)
    mask = _topk_density_mask_impl(values, keep)
    return jax.lax.with_sharding_constraint(mask, sharding)


def _topk_density_mask(values: jax.Array, density: float) -> jax.Array:
    """Build a sharding-preserving magnitude-pruning mask for a task-vector leaf.

    Args:
        values: Array whose absolute values determine importance.
        density: Fraction of entries to keep. Values at or above the computed
            threshold are kept, so ties at the threshold may retain slightly more
            than the requested count.

    Returns:
        Boolean mask with the same shape and sharding as ``values``.
        ``density <= 0`` returns all false; ``density >= 1`` returns all true.
    """
    density = float(density)
    if values.size == 0 or density >= 1.0:
        return jnp.ones_like(values, dtype=bool)
    if density <= 0.0:
        return jnp.zeros_like(values, dtype=bool)
    keep = max(1, round(values.size * density))
    if keep >= values.size:
        return jnp.ones_like(values, dtype=bool)

    sharding = _concrete_named_sharding(values)
    if sharding is not None:
        count_dtype = jnp.uint64 if bool(jax.config.jax_enable_x64) else jnp.uint32
        return _topk_density_mask_sharded(values, jnp.asarray(keep, dtype=count_dtype), sharding)
    return _topk_density_mask_impl(values, keep)


def _ties_merge(policy: jax.Array, target: jax.Array, ctx: MergeLeafContext) -> jax.Array:
    """Merge one leaf with the two-model TIES task-vector rule.

    TIES is normally defined over multiple task vectors and includes a sign
    election step across models. EasyDeL's callback merges exactly one target
    into one policy tree, so there is no cross-target sign conflict to resolve:
    the target task vector sign is the elected sign. The method therefore:

    1. Computes ``task = target - policy``.
    2. Keeps the largest-magnitude task entries according to ``ctx.density``.
    3. Adds ``ctx.target_weight * task`` back to the policy only at kept entries.

    Non-array leaves and shape mismatches are filtered before this function is
    called, so the implementation assumes both inputs are compatible arrays.
    """

    task = target - policy
    mask = _topk_density_mask(task, ctx.density)
    return policy + ctx.target_weight * jnp.where(mask, task, jnp.zeros_like(task))


def _dare_ties_apply(
    p: jax.Array,
    t: jax.Array,
    seed: jax.Array,
    density: jax.Array,
    target_weight: jax.Array,
    policy_sharding: tp.Any,
) -> jax.Array:
    """Core DARE-TIES per-leaf computation with optional sharding constraints."""
    if policy_sharding is not None:
        p = jax.lax.with_sharding_constraint(p, policy_sharding)
    task = t - p
    key = jax.random.PRNGKey(seed)
    keep = jax.random.uniform(key, task.shape) < density.astype(jnp.float32)
    if policy_sharding is not None:
        keep = jax.lax.with_sharding_constraint(keep, policy_sharding)
    pruned = jnp.where(keep, task / density.astype(task.dtype), jnp.zeros_like(task))
    out = p + target_weight.astype(pruned.dtype) * pruned
    if policy_sharding is not None:
        out = jax.lax.with_sharding_constraint(out, policy_sharding)
    return out


_dare_ties_sharded = jax.jit(_dare_ties_apply, static_argnames=("policy_sharding",))


def _dare_ties_merge(policy: jax.Array, target: jax.Array, ctx: MergeLeafContext) -> jax.Array:
    """Merge one leaf with deterministic, sharding-preserving DARE-TIES pruning.

    DARE randomly drops task-vector entries and rescales retained entries by
    ``1 / density`` so the expected task-vector magnitude is preserved. This
    implementation is deterministic for a fixed config: the PRNG key is derived
    from ``ctx.seed`` and ``ctx.leaf_index`` instead of global process state.
    When inputs carry a ``NamedSharding``, the random mask and merge are
    compiled with a hoisted JIT helper and matching sharding constraints so the
    mask is generated sharded in-place without per-leaf XLA recompilation churn.

    Edge cases:
        - ``density <= 0`` returns the policy leaf unchanged.
        - ``density >= 1`` keeps the full task vector without random masking.
        - Intermediate densities use an elementwise Bernoulli mask and then add
          ``ctx.target_weight * pruned_task`` to the policy leaf.
    """
    density = float(ctx.density)
    if density <= 0.0:
        return policy
    if density >= 1.0:
        return policy + ctx.target_weight * (target - policy)

    seed = int(ctx.seed) + int(ctx.leaf_index) * 1_000_003
    seed_arr = (
        jnp.asarray(seed & 0xFFFFFFFFFFFFFFFF, dtype=jnp.uint64)
        if bool(jax.config.jax_enable_x64)
        else jnp.asarray(seed & 0xFFFFFFFF, dtype=jnp.uint32)
    )
    density_arr = jnp.asarray(density, dtype=jnp.float32)
    weight_arr = jnp.asarray(float(ctx.target_weight), dtype=jnp.float32)
    policy_sharding = _concrete_named_sharding(policy)

    if policy_sharding is not None:
        return _dare_ties_sharded(policy, target, seed_arr, density_arr, weight_arr, policy_sharding)
    return _dare_ties_apply(policy, target, seed_arr, density_arr, weight_arr, None)


register_merge_method("linear", _linear_merge)
register_merge_method("slerp", _slerp_merge)
register_merge_method("ties", _ties_merge)
register_merge_method("dare_ties", _dare_ties_merge)
