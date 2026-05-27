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
"""Implementation helpers for native EasyDeL pytree merging.

The public callback accepts trainer states, model wrappers, or raw pytrees.
This module normalizes those inputs to graphstates and performs the actual
leaf-wise merge. It contains no model loading, Hub interaction, or mergekit
subprocess calls; all work happens in memory on the pytrees supplied by the
caller.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .merge_config import MergeConfig
from .merge_methods import MergeLeafContext, get_merge_method


def _as_graphstate(state_or_tree: object) -> object:
    """Normalize supported merge inputs to the pytree that should be merged.

    Args:
        state_or_tree: Either an ``EasyDeLState``-like object exposing
            ``graphstate``, a wrapper exposing ``model.graphstate``, or an
            already-extracted pytree.

    Returns:
        The graphstate/pytree to merge. Array leaves are not copied and raw
        pytrees are returned unchanged.
    """
    if hasattr(state_or_tree, "graphstate"):
        return state_or_tree.graphstate
    if hasattr(state_or_tree, "model"):
        model = state_or_tree.model
        if hasattr(model, "graphstate"):
            return model.graphstate
    return state_or_tree


def merge_pytrees(policy_tree: object, target_tree: object, config: "MergeConfig") -> object:
    """Merge matching array leaves from a policy pytree and target pytree.

    Args:
        policy_tree: Base pytree. Its structure defines the output structure,
            and its non-array or incompatible leaves are preserved unchanged.
        target_tree: Target pytree with the same tree definition as
            ``policy_tree``. Matching array leaves provide interpolation values
            or task vectors depending on ``config.method``.
        config: Merge settings controlling method selection, weight
            normalization, density scheduling, interpolation value, and
            deterministic random seed.

    Returns:
        A pytree with the same structure as ``policy_tree``. For every pair of
        leaves where both sides are array-like and shapes match, the registered
        merge method output is inserted. Other leaves remain the original policy
        leaf so metadata and non-trainable structure are not accidentally
        rewritten.

    Raises:
        ValueError: If the two pytrees have different tree definitions or the
            configured merge method name is unknown.
    """

    policy_weight = float(config.policy_model_weight)
    target_weight = float(config.target_model_weight)
    denom = policy_weight + target_weight
    if config.normalize and denom > 0.0:
        policy_weight /= denom
        target_weight /= denom

    policy_leaves, policy_treedef = jax.tree_util.tree_flatten(policy_tree)
    target_leaves, target_treedef = jax.tree_util.tree_flatten(target_tree)
    if policy_treedef != target_treedef:
        raise ValueError("Policy and target pytrees must have the same structure to merge.")

    merged_leaves = list(policy_leaves)
    merge_method = get_merge_method(config.method)
    densities = tuple(float(density) for density in config.target_model_density) or (1.0,)
    for index, (policy_leaf, target_leaf) in enumerate(zip(policy_leaves, target_leaves, strict=True)):
        if not hasattr(policy_leaf, "shape") or not hasattr(target_leaf, "shape"):
            continue
        policy_arr = jnp.asarray(policy_leaf)
        target_arr = jnp.asarray(target_leaf)
        if policy_arr.shape != target_arr.shape:
            continue
        ctx = MergeLeafContext(
            policy_weight,
            target_weight,
            densities[index % len(densities)],
            float(config.t_values),
            int(config.seed),
            index,
        )
        merged_leaves[index] = merge_method(policy_arr, target_arr, ctx)

    return jax.tree_util.tree_unflatten(policy_treedef, merged_leaves)
