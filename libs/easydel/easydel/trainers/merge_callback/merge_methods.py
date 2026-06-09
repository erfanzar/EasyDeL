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


def _slerp_merge(policy: jax.Array, target: jax.Array, ctx: MergeLeafContext) -> jax.Array:
    """Merge one leaf using spherical linear interpolation.

    The method treats both leaves as flattened vectors for the purpose of
    computing the angle between them, then applies the scalar SLERP coefficients
    to the original-shaped arrays. If the vectors are nearly collinear and the
    sine denominator is too small, it falls back to the equivalent linear
    interpolation ``(1 - t) * policy + t * target`` to avoid numerical
    instability.
    """

    policy_norm = policy / jnp.maximum(jnp.linalg.norm(policy), 1e-12)
    target_norm = target / jnp.maximum(jnp.linalg.norm(target), 1e-12)
    dot = jnp.clip(jnp.vdot(policy_norm, target_norm), -1.0, 1.0)
    omega = jnp.arccos(dot)
    sin_omega = jnp.sin(omega)
    t = jnp.asarray(ctx.t_value, dtype=policy.dtype)
    return jnp.where(
        sin_omega > 1e-6,
        jnp.sin((1.0 - t) * omega) / sin_omega * policy + jnp.sin(t * omega) / sin_omega * target,
        (1.0 - t) * policy + t * target,
    )


def _topk_density_mask(values: jax.Array, density: float) -> jax.Array:
    """Build a magnitude-pruning mask for a task-vector leaf.

    Args:
        values: Array whose absolute values determine importance.
        density: Fraction of entries to keep. Values at or above the computed
            threshold are kept, so ties at the threshold may retain slightly more
            than the requested count.

    Returns:
        Boolean mask with the same shape as ``values``. ``density <= 0`` returns
        all false; ``density >= 1`` returns all true.
    """

    density = float(density)
    if density >= 1.0:
        return jnp.ones_like(values, dtype=bool)
    if density <= 0.0:
        return jnp.zeros_like(values, dtype=bool)
    flat = jnp.ravel(jnp.abs(values))
    keep = max(1, round(flat.size * density))
    if keep >= flat.size:
        return jnp.ones_like(values, dtype=bool)
    threshold = jnp.min(jax.lax.top_k(flat, keep)[0])
    return jnp.abs(values) >= threshold


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


def _dare_ties_merge(policy: jax.Array, target: jax.Array, ctx: MergeLeafContext) -> jax.Array:
    """Merge one leaf with deterministic DARE-TIES pruning.

    DARE randomly drops task-vector entries and rescales retained entries by
    ``1 / density`` so the expected task-vector magnitude is preserved. This
    implementation is deterministic for a fixed config: the PRNG key is derived
    from ``ctx.seed`` and ``ctx.leaf_index`` instead of global process state.

    Edge cases:
        - ``density <= 0`` returns the policy leaf unchanged.
        - ``density >= 1`` keeps the full task vector without random masking.
        - Intermediate densities use an elementwise Bernoulli mask and then add
          ``ctx.target_weight * pruned_task`` to the policy leaf.
    """

    density = float(ctx.density)
    if density <= 0.0:
        return policy
    task = target - policy
    if density >= 1.0:
        pruned = task
    else:
        key = jax.random.PRNGKey(int(ctx.seed) + int(ctx.leaf_index) * 1_000_003)
        keep = jax.random.uniform(key, task.shape) < density
        pruned = jnp.where(keep, task / density, jnp.zeros_like(task))
    return policy + ctx.target_weight * pruned


register_merge_method("linear", _linear_merge)
register_merge_method("slerp", _slerp_merge)
register_merge_method("ties", _ties_merge)
register_merge_method("dare_ties", _dare_ties_merge)
