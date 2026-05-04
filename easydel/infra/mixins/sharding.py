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

"""Sharding mixin for EasyDeL modules.

Provides sharding and partitioning functionality through the EasyShardingMixin
class, which can be combined with EasyDeL models to enable distributed training
and inference across multiple devices.

Classes:
    EasyShardingMixin: Mixin class providing sharding and partitioning methods

Key Features:
    - Automatic sharding rule resolution from variable metadata
    - Model sharding and gathering for distributed training
    - Output sharding application via JIT compilation

Example:
    >>> from easydel.infra.mixins import EasyShardingMixin
    >>> # Model class inherits from EasyShardingMixin
    >>> model = model.shard_model()
    >>> gathered = model.gather_model()
"""

from __future__ import annotations

import re
import typing as tp
from collections.abc import Mapping
from functools import partial
from typing import Self

import jax
import numpy
import spectrax as spx
from jax.sharding import NamedSharding, PartitionSpec
from spectrax import make_shard_and_gather_fns

from easydel.infra.sharding import replicated_named_sharding
from easydel.utils.traversals import flatten_dict


class EasyShardingMixin:
    """Mixin that gives :class:`EasyDeLBaseModule` its sharding/gathering surface area.

    ``EasyShardingMixin`` is the host-class facade over the lower-level
    :mod:`easydel.infra.sharding` resolver. It expects to be mixed into a
    SpecTrax module that exposes a ``config`` carrying a
    :class:`RuntimeShardingResolver` and one or more JAX device meshes; in
    return, it provides the methods used throughout EasyDeL to:

    - **Inspect** how the module is or should be sharded:
      :attr:`shardings`, :attr:`parameters_sharding`,
      :attr:`graphstate_sharding`, :attr:`graphother_sharding`, and
      :attr:`runtime_sharding_resolver` expose the active sharding metadata
      derived from variable annotations.
    - **Resolve** generic sharding pytrees from regex rules:
      :meth:`resolve_shardings_regex` returns one ``(regex, NamedSharding)``
      rule per ``(variable, alias)`` so that pipeline-parallel layers retain
      per-stage submesh assignments, and :meth:`resolve_sharding_for_tree`
      maps an arbitrary parameter pytree onto those rules with a replicated
      fallback for tiny/scalar leaves.
    - **Apply** the resulting sharding to live arrays:
      :meth:`apply_sharding_for_tree` device-puts each leaf, while
      :meth:`shard_model` and :meth:`gather_model` provide the high-level
      "split, transform, recombine" path used by trainers.
      :meth:`apply_out_shardings` and :meth:`fully_gather` use ``spx.jit``
      with ``out_shardings`` so that MPMD meshes route through ``sxjit`` and
      gathering across pipeline stages becomes a scheduled cross-mesh
      transfer rather than an unsharded copy.
    - **Pick a mesh**: :attr:`mesh`, :attr:`explicit_mesh`,
      :attr:`manual_mesh`, :meth:`mesh_call`, and :meth:`_get_mesh` provide
      the canonical entry points so callers don't have to repeat
      ``self.config.mesh`` in every site.

    The mixin holds no state of its own — every operation is derived from
    ``self.config`` and the live SpecTrax variable metadata, so operations
    are safe to call any number of times.

    Attributes:
        config (Any): The host module's :class:`EasyDeLBaseConfig` instance.
            Read for ``mesh``, ``explicit_mesh``, ``manual_mesh``, and
            ``runtime_sharding_resolver``.
    """

    config: tp.Any

    @property
    def parameters_sharding(self: Self) -> spx.State:
        """Compute shape metadata for the default selected trainable state."""
        return self.resolve_sharding_for_tree(self.parameters)

    @property
    def graphstate_sharding(self: Self) -> spx.State:
        """Compute shape metadata for the default selected trainable state."""
        return self.resolve_sharding_for_tree(self.graphstate)

    @property
    def graphother_sharding(self: Self) -> spx.State:
        """Compute shape metadata for the default selected trainable state."""
        return self.resolve_sharding_for_tree(self.graphother)

    @property
    def mesh(self: Self) -> spx.SpxMesh:
        """Get the SpectraX device mesh from the module's configuration.

        Returns the mesh used for distributed training and sharding operations.
        The mesh defines how arrays are partitioned across devices.

        Returns:
            spx.SpxMesh: The device mesh defined in self.config.mesh,
                which specifies the topology of devices for data and model
                parallelism.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> print(model.mesh.devices.shape)
            (8,)  # Example: 8 devices in the mesh
            >>> print(model.mesh.axis_names)
            ('dp', 'fsdp', 'tp', 'sp')
        """
        return self.config.mesh

    @property
    def explicit_mesh(self: Self) -> spx.SpxMesh:
        """Get the explicit-axis SpectraX device mesh from the module's configuration.

        Returns the explicit mesh variant where axes are explicitly named
        and managed. This is useful for advanced sharding strategies.

        Returns:
            spx.SpxMesh: The explicit-axis device mesh defined in
                self.config.explicit_mesh.
        """
        return self.config.explicit_mesh

    @property
    def manual_mesh(self: Self) -> spx.SpxMesh:
        """Get the manual-axis SpectraX device mesh from the module's configuration.

        Returns the manual mesh variant where axis handling is done manually
        by the user. This provides maximum flexibility for custom sharding.

        Returns:
            spx.SpxMesh: The manual-axis device mesh defined in
                self.config.manual_mesh.
        """
        return self.config.manual_mesh

    def mesh_call(self: Self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Call the module under the configured JAX mesh context.

        This is a convenience method equivalent to `with self.mesh: self(*args, **kwargs)`.
        It ensures that all operations within the forward pass respect the mesh sharding
        configuration.

        Args:
            *args: Positional arguments to pass to the module.
            **kwargs: Keyword arguments to pass to the module.

        Returns:
            Any: The module output, with appropriate sharding applied
                based on the mesh configuration.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> # These are equivalent:
            >>> output1 = model.mesh_call(input_ids, attention_mask=mask)
            >>> with model.mesh:
            ...     output2 = model(input_ids, attention_mask=mask)

        Note:
            This method uses self.mesh only. For explicit_mesh or manual_mesh,
            enter those contexts explicitly when needed.
        """
        with self.mesh:
            return self(*args, **kwargs)

    def _get_mesh(self: Self, mesh: spx.SpxMesh | None = None) -> spx.SpxMesh:
        """Retrieve the SpectraX device mesh, with fallback to configuration.

        Gets the mesh to use for sharding operations, prioritizing the provided
        argument over the mesh in the configuration.

        Args:
            mesh: A SpectraX device mesh to use. If None, uses self.config.mesh.

        Returns:
            spx.SpxMesh: The resolved SpectraX device mesh.

        Raises:
            ValueError: If no mesh is provided and none is found in the
                configuration (self.config.mesh is None or config doesn't exist).

        Example:
            >>> mesh = model._get_mesh()  # Uses config mesh
            >>> custom_mesh = Mesh(devices, axis_names)
            >>> mesh = model._get_mesh(custom_mesh)  # Uses provided mesh
        """
        if mesh is None:
            return self.config.mesh
        return mesh

    def resolve_shardings_regex(
        self,
        *,
        mesh: spx.SpxMesh | None = None,
    ) -> tuple[tuple[str, jax.sharding.NamedSharding], ...]:
        """Return per-variable ``(regex, NamedSharding)`` rules.

        One rule per ``(variable, leaf-name-alias)`` pair: the regex pins
        the *literal* path (including each layer index) and the
        ``NamedSharding`` is the variable's live placement, produced by
        ``named_sharding_for_variable``.

        Layer indices are NOT collapsed to ``\\d+`` -- each layer gets
        its own rule.  Under pipeline parallelism that's the only way to
        carry per-stage information through ``(regex, sharding)`` rules:
        layer 0 may live on stage 0's submesh while layer 9 lives on
        stage 1's submesh, and a single ``\\d+``-collapsed rule cannot
        represent both.

        Args:
            mesh: Optional mesh override.  Defaults to the model's
                configured mesh.

        Returns:
            Tuple of ``(slash-form regex, NamedSharding)`` pairs.  The
            regex matches the variable's slash-separated string path
            (the form ``spectrax.match_partition_rules`` and the
            on-disk ``Checkpointer`` use), so the returned rules can be
            consumed by either pipeline.
        """

        def _metadata_rule_path_aliases(path: str) -> tuple[str, ...]:
            """Return compatibility aliases for canonical metadata-derived paths."""
            parts = path.split(".")
            if not parts:
                return (path,)

            aliases = [tuple(parts)]
            lowered_path = ".".join(parts).lower()
            terminal = parts[-1]

            if terminal == "weight":
                aliases.append((*parts[:-1], "kernel"))
                if "norm" in lowered_path:
                    aliases.append((*parts[:-1], "scale"))
                if "embed" in lowered_path:
                    aliases.append((*parts[:-1], "embedding"))

            deduped: list[str] = []
            seen: set[str] = set()
            for alias_parts in aliases:
                alias_path = ".".join(alias_parts)
                if alias_path in seen:
                    continue
                seen.add(alias_path)
                deduped.append(alias_path)
            return tuple(deduped)

        def _metadata_rule_patterns(path: str) -> tuple[str, ...]:
            """Return path regexes for checkpoint and live SpectraX state paths."""
            parts = path.split(".")
            slash_pattern = r"^(?:.*/)?" + "/".join(re.escape(part) for part in parts) + r"(?:/.*)?$"
            dotted_pattern = r"^(?:.*/)?" + re.escape(path) + r"(?:/.*)?$"

            patterns = [slash_pattern, dotted_pattern]
            if len(parts) > 1:
                # Some pytrees stringify module scopes as slash-separated while
                # keeping the leaf variable name attached with a dot.
                hybrid_pattern = (
                    r"^(?:.*/)?"
                    + "/".join(re.escape(part) for part in parts[:-1])
                    + r"\."
                    + re.escape(parts[-1])
                    + r"(?:/.*)?$"
                )
                patterns.append(hybrid_pattern)

            deduped: list[str] = []
            seen_patterns: set[str] = set()
            for pattern in patterns:
                if pattern in seen_patterns:
                    continue
                seen_patterns.add(pattern)
                deduped.append(pattern)
            return tuple(deduped)

        mesh = self._get_mesh(mesh)
        resolver = self.runtime_sharding_resolver.with_mesh(mesh)
        _, graph = spx.export(self)
        graph_collections = graph.collections()

        rules: list[tuple[str, jax.sharding.NamedSharding]] = []
        seen: set[str] = set()
        for path, var in spx.iter_variables(self):
            if getattr(var, "kind", None) not in graph_collections:
                continue
            value = getattr(var, "value", None)
            shape = tuple(value.shape) if hasattr(value, "shape") else None
            if shape is None:
                continue
            ns = resolver.named_sharding_for_variable(var, shape=shape, mesh=mesh)
            if ns is None:
                continue
            for aliased_path in _metadata_rule_path_aliases(path):
                for pattern in _metadata_rule_patterns(aliased_path):
                    if pattern in seen:
                        continue
                    seen.add(pattern)
                    rules.append((pattern, ns))
        return tuple(rules)

    def resolve_sharding_for_tree(self, tree=None, *, mesh: spx.SpxMesh | None = None):
        """Build a NamedSharding pytree mirroring *tree* using regex rules.

        Resolves each leaf path against the model's regex sharding rules and
        falls back to a fully-replicated sharding for tiny or scalar leaves.

        Args:
            tree: A parameter pytree (defaults to ``self.graphstate_shape``).
            mesh: Optional mesh override; defaults to the config mesh.

        Returns:
            Any: A pytree of :class:`NamedSharding` objects with the same
            structure as *tree*.
        """
        from easydel.infra.utils import jax_path_to_string

        mesh = self._get_mesh(mesh)
        if tree is None:
            tree = self.graphstate_shape
        jax_mesh = mesh.jax_mesh if hasattr(mesh, "jax_mesh") else mesh
        empty_sharding = replicated_named_sharding(mesh)
        rules = self.resolve_shardings_regex(mesh=mesh)

        def _spec_for(path: str, leaf: tp.Any):
            """Resolve the NamedSharding for a single ``(path, leaf)`` pair.

            Args:
                path: Stringified pytree path for the leaf.
                leaf: The pytree leaf (typically a shape-bearing array).

            Returns:
                NamedSharding: A regex-matched sharding, or the replicated
                fallback for small/scalar/non-array leaves.
            """
            if (
                (hasattr(leaf, "shape") and int(numpy.prod(leaf.shape)) < 128)
                or (len(getattr(leaf, "shape", ())) == 0)
                or (not hasattr(leaf, "shape"))
            ):
                return empty_sharding
            for pattern, sharding in rules:
                if re.search(pattern, path):
                    out = (
                        sharding
                        if isinstance(sharding, NamedSharding)
                        else NamedSharding(
                            mesh=jax_mesh,
                            spec=PartitionSpec(*tuple(sharding)),
                        )
                    )
                    if hasattr(leaf, "ndim") and len(out.spec) > leaf.ndim:
                        out = NamedSharding(
                            mesh=out.mesh,
                            spec=PartitionSpec(*tuple(out.spec)[: leaf.ndim]),
                        )
                    return out
            return empty_sharding

        return jax.tree_util.tree_map_with_path(lambda path, leaf: _spec_for(jax_path_to_string(path), leaf), tree)

    def apply_sharding_for_tree(self, tree, *, mesh: spx.SpxMesh | None = None):
        """Apply :meth:`resolve_sharding_for_tree` results to a value tree.

        Args:
            tree: A pytree of arrays to be device-placed.
            mesh: Optional mesh override.

        Returns:
            Any: The tree with each leaf device-put under the resolved
            NamedSharding (donating the original buffer).
        """
        shardings = self.resolve_sharding_for_tree(tree, mesh=mesh)
        return jax.tree_util.tree_map(lambda x, s: jax.device_put(x, s, donate=True), tree, shardings)

    @property
    def runtime_sharding_resolver(self):
        """Return the model's runtime sharding resolver."""
        return self.config.runtime_sharding_resolver.with_mesh(self.config.mesh)

    @property
    def shardings(self: Self):
        """Extract the NamedSharding object for each parameter.

        Returns a nested dictionary containing the NamedSharding object
        (if present) for each parameter, or None for unsharded parameters.

        Returns:
            dict: A nested dictionary mirroring the parameter structure,
                containing NamedSharding objects or None.
        """
        return spx.sharding.extract_sharding_structure(self)

    def _apply_sharding_fns(self: Self, sharding_fns: Mapping[str, tp.Callable]) -> Self:
        """Apply sharding or gathering functions to the module's parameters.

        Internal method that applies a mapping of functions to transform
        parameters. Used by shard_model() and gather_model() to distribute
        or collect parameters across devices.

        Args:
            sharding_fns: A mapping from flattened parameter paths (tuples)
                to transformation functions. Each function takes a parameter
                array and returns a transformed (sharded or gathered) array.

        Returns:
            Self: The module instance with sharding/gathering functions applied
                to its parameters.

        Note:
            Parameters that are not callable (e.g., pre-sharded NF4 arrays)
            are left unchanged.
        """
        gdef, params, others = self.split_module()
        sharding_fns = flatten_dict(sharding_fns)
        _shard_keys = list(sharding_fns.keys())

        def _apply_state(state: spx.State) -> spx.State:
            """Apply the sharding-fn map to a single SpectraX state container.

            Args:
                state: A :class:`spx.State` of leaves.

            Returns:
                spx.State: A new state with eligible leaves transformed.
            """
            new_data: dict[str, dict[str, tp.Any]] = {}
            for c, p, leaf in state.items():
                path = tuple((c + "/" + p).split("/"))
                if leaf is not None and path in _shard_keys:
                    fn = sharding_fns[path]
                    if callable(fn):
                        leaf = fn(leaf)
                new_data.setdefault(c, {})[p] = leaf
            return spx.State(new_data)

        params = _apply_state(params)
        others = _apply_state(others)
        self = self.merge_module(gdef, params, others)
        return self

    def shard_model(
        self: Self,
        mesh: spx.SpxMesh | None = None,
        overlay_fns: Mapping[str, tp.Callable] | None = None,
    ) -> Self:
        """Shard the model's parameters according to the device mesh.

        Distributes the model's parameters across devices according to
        variable metadata-derived partition specs and the device mesh.

        Args:
            mesh: JAX device mesh defining the device topology. If None, uses
                mesh from config. Defaults to None.
            overlay_fns: Additional transformation functions that override
                the default sharding for specific parameters. Keys are parameter
                paths, values are transformation functions. Defaults to None.

        Returns:
            Self: The model instance with sharded parameters.

        Example:
            >>> model = LlamaModel(config, dtype, param_dtype, precision, rngs)
            >>> # Shard using config's mesh and rules
            >>> model = model.shard_model()
            >>>
            >>> # Shard with custom mesh
            >>> devices = jax.devices()
            >>> mesh = Mesh(devices, ('dp',))
            >>> model = model.shard_model(mesh=mesh)

        Note:
            After sharding, each parameter will be distributed across devices
            according to its PartitionSpec. Access to the full parameter
            requires gathering (see gather_model()).
        """
        mesh = self._get_mesh(mesh)
        graphdef, graphstate, graphother = self.split_module()
        graphstate = self.apply_sharding_for_tree(graphstate, mesh=mesh)
        graphother = self.apply_sharding_for_tree(graphother, mesh=mesh)
        self = self.merge_module(graphdef, graphstate, graphother)
        if overlay_fns is not None:
            self = self._apply_sharding_fns(overlay_fns)
        return self

    def gather_model(
        self: Self,
        mesh: spx.SpxMesh | None = None,
        overlay_fns: Mapping[str, tp.Callable] | None = None,
    ) -> Self:
        """Gather the model's parameters from distributed devices to host.

        Collects sharded parameters from across devices and consolidates them,
        typically to a single device or the host. This is the inverse of
        shard_model() and is useful for saving checkpoints or inference.

        Args:
            mesh: JAX device mesh from which to gather parameters. If None,
                uses mesh from config. Defaults to None.
            overlay_fns: Additional transformation functions that override
                the default gathering for specific parameters. Defaults to None.

        Returns:
            Self: The model instance with gathered (non-distributed) parameters.

        Example:
            >>> # After distributed training
            >>> model = model.gather_model()
            >>> # Now parameters are on a single device and can be saved
            >>> model.save_pretrained("checkpoint/")

        Note:
            Gathering is typically slower than keeping parameters distributed,
            so it should only be done when necessary (e.g., checkpointing).
        """
        mesh = self._get_mesh(mesh)
        gdef, graphstate, graphother = self.split_module()
        graphstate = jax.tree_util.tree_map(lambda o: jax.device_get(o), graphstate)
        graphother = jax.tree_util.tree_map(lambda o: jax.device_get(o), graphother)
        self = self.merge_module(gdef, graphstate, graphother)
        if overlay_fns is not None:
            self = self._apply_sharding_fns(overlay_fns)
        return self

    def _make_shard_fns(self: Self):
        """Build sanitized shard functions from partition specs."""
        mesh = self._get_mesh(None)
        partition_specs = self.resolve_sharding_for_tree(mesh=mesh)
        shard_fns, _ = make_shard_and_gather_fns(partition_specs=partition_specs, mesh=mesh)
        return shard_fns

    @property
    def _shard_fns(self: Self):
        """Generate sharding functions based on the module's configuration.

        Returns:
            Mapping: A mapping from flattened parameter paths to sharding
                functions that transform arrays to their sharded form.
        """
        return self._make_shard_fns()

    def apply_out_shardings(self, out_shardings):
        """Apply output sharding specifications to the module state.

        Uses JIT compilation with out_shardings to enforce specific sharding
        constraints on the module's state.

        Args:
            out_shardings: Sharding specifications to apply to the module's
                graphstate and graphother components.

        Returns:
            Self: Module with sharding constraints applied to its state.

        Example:
            >>> shardings = jax.tree_map(
            ...     lambda x: replicated_named_sharding(mesh),
            ...     model.split_module()[1:]
            ... )
            >>> model = model.apply_out_shardings(shardings)
        """
        splits = self.split_module()

        # # @erfanzar NOTE: spx.jit (not jax.jit) so MPMD meshes route to
        # sxjit and per-stage out_shardings land on the right submesh.
        @partial(spx.jit, mesh=self.mesh, out_shardings=out_shardings)
        def _call(graphstate, graphother):
            """Identity ``spx.jit`` wrapper that pins outputs to ``out_shardings``.

            Args:
                graphstate: SpectraX parameter state.
                graphother: SpectraX non-parameter state.

            Returns:
                tuple: ``(graphstate, graphother)`` re-emitted with the
                requested out-shardings.
            """
            return graphstate, graphother

        splits[1:] = _call(*splits[1:])
        return self.merge_module(*splits)

    def fully_gather(self: Self) -> Self:
        """Apply JAX sharding constraints to gather all parameters.

        Marks all parameters to have no sharding (PartitionSpec()), effectively
        gathering them to the host or a single device. Uses jax.jit with
        out_shardings to enforce these gathering constraints.

        Returns:
            Self: The model instance with gathering constraints applied,
                where all parameters are replicated (not sharded).

        Example:
            >>> model = model.fully_gather()
            >>> # All parameters are now replicated across devices
        """
        # # @erfanzar NOTE: spx.jit so MPMD meshes hit sxjit -- gathering
        # across stage submeshes onto the full mesh is a cross-mesh transfer
        # the MPMD runtime knows how to schedule.
        gdef, gstate = spx.export(self)
        sharding_tree = self.resolve_sharding_for_tree(gstate)
        shardings = jax.tree_util.tree_map(lambda x: replicated_named_sharding(self.mesh), sharding_tree)

        @partial(spx.jit, mesh=self.mesh, out_shardings=shardings)
        def _apply(state):
            """Identity ``spx.jit`` wrapper enforcing replicated out-shardings.

            Args:
                state: The combined SpectraX state to be re-emitted.

            Returns:
                Any: The same state with replicated out-shardings applied.
            """
            return state

        gstate = _apply(gstate)
        self = spx.bind(gdef, gstate)
        return self
