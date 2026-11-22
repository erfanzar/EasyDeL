# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sharding helpers (SPMD + MPMD-aware).

Logical axis-name rules, per-initializer sharding metadata, and
:class:`~jax.sharding.PartitionSpec` / :class:`~jax.sharding.NamedSharding`
derivation from a :class:`~spectrax.Module`. The constraint and
extraction helpers route through ``_resolve_constraint_target`` so they
work on plain JAX meshes, :class:`~spectrax.sharding.SpxMesh`, and
:class:`~spectrax.runtime.types.MpMdMesh` (in which case stage-local
sub-meshes are picked automatically).
"""

from .audit import PlacementIssue, PlacementReport, audit_placement
from .logical import current_axis_rules, logical_axis_rules
from .manager import (
    PartitionAxis,
    PartitionManager,
    get_current_partition_manager,
    get_partition_manager,
)
from .mesh import (
    DEFAULT_MESH_AXIS_DIMS,
    DEFAULT_MESH_AXIS_NAMES,
    SpxMesh,
    calculate_host_mesh_shape,
    cpu_context,
    create_cpu_mesh,
    create_mesh,
    current_mesh,
    force_cpu,
    parse_mesh_from_string,
    use_mesh,
)
from .partition import (
    apply_logical_sharding,
    explain_partition_spec_sanitization,
    extract_sharding_structure,
    extract_shardings,
    get_axes_size_in_mesh,
    get_corrected_named_sharding,
    get_current_stage_mesh,
    get_incontext_mesh,
    get_named_sharding,
    get_partition_spec,
    lax_reshard,
    make_shard_and_gather_fns,
    match_partition_rules,
    names_in_current_mesh,
    sanitize_partition_spec_for_mesh_and_shape,
    to_jax_mesh,
    with_partitioning,
    with_sharding_constraint,
    with_sharding_constraint_by_name,
)
from .placement import (
    mesh_axis_product,
    named_sharding_for_shape,
    named_sharding_for_value,
    place_setup_leaf_with_sharding,
    place_setup_tree_with_shardings,
    reshape_with_named_shardings,
    spec_shape_mismatches,
    transpose_with_named_shardings,
    with_named_sharding,
)
from .strict import is_strict_sharding, set_strict_sharding, strict_sharding

__all__ = [
    "DEFAULT_MESH_AXIS_DIMS",
    "DEFAULT_MESH_AXIS_NAMES",
    "PartitionAxis",
    "PartitionManager",
    "PlacementIssue",
    "PlacementReport",
    "SpxMesh",
    "apply_logical_sharding",
    "audit_placement",
    "calculate_host_mesh_shape",
    "cpu_context",
    "create_cpu_mesh",
    "create_mesh",
    "current_axis_rules",
    "current_mesh",
    "explain_partition_spec_sanitization",
    "extract_sharding_structure",
    "extract_shardings",
    "force_cpu",
    "get_axes_size_in_mesh",
    "get_corrected_named_sharding",
    "get_current_partition_manager",
    "get_current_stage_mesh",
    "get_incontext_mesh",
    "get_named_sharding",
    "get_partition_manager",
    "get_partition_spec",
    "is_strict_sharding",
    "lax_reshard",
    "logical_axis_rules",
    "make_shard_and_gather_fns",
    "match_partition_rules",
    "mesh_axis_product",
    "named_sharding_for_shape",
    "named_sharding_for_value",
    "names_in_current_mesh",
    "parse_mesh_from_string",
    "place_setup_leaf_with_sharding",
    "place_setup_tree_with_shardings",
    "reshape_with_named_shardings",
    "sanitize_partition_spec_for_mesh_and_shape",
    "set_strict_sharding",
    "spec_shape_mismatches",
    "strict_sharding",
    "to_jax_mesh",
    "transpose_with_named_shardings",
    "use_mesh",
    "with_named_sharding",
    "with_partitioning",
    "with_sharding_constraint",
    "with_sharding_constraint_by_name",
]
