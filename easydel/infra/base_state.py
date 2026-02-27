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

"""State management for EasyDeL models.

This module provides the EasyDeLState class, which encapsulates all stateful
components of a model during training or inference, including parameters,
optimizer state, and training metadata.

The EasyDeLState class serves as the primary interface for managing model
state throughout the training lifecycle. It integrates with JAX's functional
paradigm while providing convenient methods for gradient updates, checkpointing,
sharding, and state manipulation.

Classes:
    EasyDeLState: Complete state container for models during training/inference.

Module Constants:
    AM: Alias for AsyncCheckpointManager for convenient access.
    WEIGHTS_NAME: Default filename for model parameters ('easydel-model.parameters').
    OPTIMIZER_NAME: Default filename for optimizer state ('easydel-optstate.parameters').
    OPTIMIZER_STRUCT_NAME: Default filename for optimizer structure ('easydel-optstate.structure').
    TX_STRUCT_JSON: Filename for optimizer transformation structure ('tx_structure.json').

Key Features:
    - Unified state management for training and inference workflows.
    - Automatic sharding and partitioning support across device meshes.
    - Checkpoint saving and loading with async support.
    - Gradient application with optimizer integration.
    - State serialization and deserialization.
    - Support for quantization configurations.

Example:
    Basic usage for creating and managing training state::

        >>> from easydel.infra import EasyDeLState
        >>> import optax
        >>>
        >>> # Create state from a model
        >>> state = EasyDeLState.create(
        ...     model=model,
        ...     tx=optax.adamw(learning_rate=1e-4),
        ...     init_opt_state=True
        ... )
        >>>
        >>> # Apply gradients during training
        >>> state = state.apply_gradients(grads=gradients)
        >>>
        >>> # Save checkpoint
        >>> state.save_state("checkpoint_path")
        >>>
        >>> # Load checkpoint
        >>> state = EasyDeLState.load_state(
        ...     "checkpoint_path",
        ...     config=config
        ... )

See Also:
    - :class:`easydel.infra.base_module.EasyDeLBaseModule`: Base module class.
    - :class:`easydel.infra.base_config.EasyDeLBaseConfig`: Configuration class.
"""

from __future__ import annotations

import collections.abc
import contextlib
import os
import pickle
import traceback
import typing as tp
from typing import Self

import jax
import optax  # pyright: ignore[reportMissingTypeStubs]
from eformer import escale as es
from eformer.escale import PartitionAxis
from eformer.loggings import get_logger
from eformer.paths import ePath, ePathLike
from eformer.serialization import AsyncCheckpointManager, Checkpointer
from flax import nnx as nn
from flax import struct
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.factory import TaskType
from easydel.utils.compiling_utils import ejit
from easydel.utils.traversals import flatten_dict, specs_to_name_sharding, unflatten_dict

from .utils import materialize_meta_leaves, sanitize_partition_spec_for_shape

if tp.TYPE_CHECKING:
    from jax.sharding import Mesh

    from easydel.infra.base_config import EasyDeLBaseConfigDict
    from easydel.infra.etils import EasyDeLBackends, EasyDeLPlatforms
    from easydel.layers import QuantizationConfig

    from .base_module import EasyDeLBaseModule, PartitionLike

AM = AsyncCheckpointManager
"""Alias for AsyncCheckpointManager for convenient access."""

WEIGHTS_NAME = "easydel-model.parameters"
"""Default filename for saving model parameters."""

OPTIMIZER_NAME = "easydel-optstate.parameters"
"""Default filename for saving optimizer state tensors."""

OPTIMIZER_STRUCT_NAME = "easydel-optstate.structure"
"""Default filename for saving optimizer state structure (pickle format)."""

TX_STRUCT_JSON = "tx_structure.json"
"""Filename for optimizer transformation structure in JSON format."""

logger = get_logger(__name__)


def _sanitize_partition_specs_for_shape_tree(
    partition_specs: tp.Any,
    shape_tree: tp.Any,
    mesh: "Mesh",
) -> tuple[tp.Any, int]:
    """Sanitize a partition-spec tree against concrete tensor shapes."""
    adjusted = {"count": 0}

    def _sanitize(spec: tp.Any, shape_obj: tp.Any) -> tp.Any:
        if not isinstance(spec, PartitionSpec) or not hasattr(shape_obj, "shape"):
            return spec
        safe_spec = sanitize_partition_spec_for_shape(
            spec=spec,
            shape=tuple(shape_obj.shape),
            mesh=mesh,
        )
        if safe_spec != spec:
            adjusted["count"] += 1
        return safe_spec

    try:
        sanitized = jax.tree_util.tree_map(_sanitize, partition_specs, shape_tree)
        return sanitized, adjusted["count"]
    except Exception:
        # Fallback for structures that don't support aligned tree mapping.
        flat_specs = flatten_dict(partition_specs)
        flat_shapes = flatten_dict(shape_tree)
        adjusted_count = 0
        for key, spec in flat_specs.items():
            if not isinstance(spec, PartitionSpec):
                continue
            shape_obj = flat_shapes.get(key)
            shape = tuple(getattr(shape_obj, "shape", ()))
            if not shape:
                continue
            safe_spec = sanitize_partition_spec_for_shape(spec=spec, shape=shape, mesh=mesh)
            if safe_spec != spec:
                flat_specs[key] = safe_spec
                adjusted_count += 1
        if adjusted_count == 0:
            return partition_specs, adjusted_count
        return unflatten_dict(flat_specs), adjusted_count


class EasyDeLState(struct.PyTreeNode):
    """Complete state container for EasyDeL models during training or inference.

    EasyDeLState encapsulates all stateful components needed for training or inference,
    including model parameters, optimizer state, and training metadata. It provides
    methods for gradient updates, checkpointing, and state management while integrating
    seamlessly with JAX's functional programming paradigm.

    This class is implemented as a Flax struct PyTreeNode, making it compatible with
    JAX transformations like `jit`, `grad`, `vmap`, and `pmap`. The state is immutable;
    all methods that modify state return new instances.

    Attributes:
        step (int | jax.Array): Current training step count. Incremented automatically
            when `apply_gradients` is called. Can be an integer or a JAX array for
            device placement.
        graphdef (nn.GraphDef): The model's computation graph definition. This is a
            non-pytree node that defines the structure of the neural network without
            containing any parameter values.
        graphstate (nn.GraphState): The model's parameter state as a pytree. Contains
            all trainable parameters (nn.Param) extracted from the model.
        graphother (nn.GraphState): Non-parameter model state as a pytree. Contains
            other stateful components like batch normalization statistics, RNG states,
            and any other nn.Variable types that are not parameters.
        tx (optax.GradientTransformation): The optimizer transformation used for
            parameter updates. This is a non-pytree node. Can be None if the state
            is used only for inference.
        opt_state (optax.OptState | None): The optimizer state (e.g., momentum buffers,
            adaptive learning rate accumulators). This is a pytree node. None if the
            optimizer has not been initialized.
        apply_fn (tp.Callable | None): Optional model application function for custom
            forward pass implementations. Defaults to None, in which case the standard
            model call is used.

    Example:
        Creating state from a model and training::

            >>> import optax
            >>> from easydel.infra import EasyDeLState
            >>>
            >>> # Initialize state with model and optimizer
            >>> state = EasyDeLState.create(
            ...     model=my_model,
            ...     tx=optax.adam(1e-3),
            ...     init_opt_state=True
            ... )
            >>>
            >>> # Training loop
            >>> for batch in dataloader:
            ...     grads = compute_gradients(state, batch)
            ...     state = state.apply_gradients(grads=grads)
            ...     print(f"Step: {state.step}")

        Saving and loading checkpoints::

            >>> # Save complete state
            >>> state.save_state("checkpoints/step_1000")
            >>>
            >>> # Load state later
            >>> loaded_state = EasyDeLState.load_state(
            ...     "checkpoints/step_1000",
            ...     dtype=jnp.bfloat16
            ... )

        Working with sharded states::

            >>> # Shard state across devices
            >>> sharded_state = state.shard_state()
            >>>
            >>> # Gather state back to single device
            >>> gathered_state = sharded_state.gather_state()

    Note:
        The state is designed to be immutable. Methods like `apply_gradients`,
        `shard_state`, etc., return new state instances rather than modifying
        the existing state in place.

    See Also:
        - :meth:`create`: Factory method for creating new state instances.
        - :meth:`apply_gradients`: Apply gradients to update parameters.
        - :meth:`save_state`: Save state to disk.
        - :meth:`load_state`: Load state from disk.
    """

    step: int | jax.Array = struct.field(pytree_node=True)
    graphdef: nn.GraphDef = struct.field(pytree_node=False)

    graphstate: nn.GraphState = struct.field(pytree_node=True)
    graphother: nn.GraphState = struct.field(pytree_node=True)

    tx: optax.GradientTransformation | None = struct.field(pytree_node=False)
    opt_state: optax.OptState | None = struct.field(pytree_node=True)
    apply_fn: tp.Callable | None = struct.field(pytree_node=False, default=None)

    def apply_gradients(self: Self, *, grads) -> Self:
        """Apply gradients to update parameters and optimizer state.

        Performs a single optimization step using the provided gradients. This method
        updates the model parameters using the optimizer transformation, updates the
        optimizer state (e.g., momentum buffers), and increments the step counter.

        The method supports custom optimizer hooks through `apply_updates_hook` if
        the optimizer transformation implements this interface. Otherwise, it falls
        back to the standard `optax.apply_updates` function.

        Args:
            grads: Gradient pytree matching the structure of `graphstate`. Must have
                the same tree structure and array shapes as the model parameters.
                Typically computed using `jax.grad` or similar.

        Returns:
            Self: A new EasyDeLState instance with:
                - Updated `graphstate` containing the new parameter values.
                - Updated `opt_state` reflecting the optimizer's internal state.
                - Incremented `step` count (step + 1).

        Raises:
            AssertionError: If `opt_state` is None (optimizer not initialized).
            AssertionError: If `tx` is None (no optimizer transformation provided).

        Example:
            Basic gradient application::

                >>> def loss_fn(params, batch):
                ...     logits = model.apply(params, batch['input'])
                ...     return cross_entropy(logits, batch['target'])
                >>>
                >>> grads = jax.grad(loss_fn)(state.graphstate, batch)
                >>> state = state.apply_gradients(grads=grads)
                >>> print(f"Updated to step {state.step}")

            With gradient clipping (handled by optimizer)::

                >>> tx = optax.chain(
                ...     optax.clip_by_global_norm(1.0),
                ...     optax.adam(1e-3)
                ... )
                >>> state = EasyDeLState.create(model=model, tx=tx, init_opt_state=True)
                >>> state = state.apply_gradients(grads=grads)

        Note:
            This method requires the optimizer to be initialized. Use `init_opt_state=True`
            when creating the state, or call `init_tx` before applying gradients.

        See Also:
            - :meth:`create`: Create state with optimizer initialization.
            - :meth:`init_tx`: Initialize optimizer for existing state.
        """
        assert self.opt_state is not None
        assert self.tx is not None

        updates, new_opt_state = self.tx.update(updates=grads, state=self.opt_state, params=self.graphstate)

        if hasattr(self.tx, "apply_updates_hook"):
            graphstate = self.tx.apply_updates_hook(self.graphstate, updates)
        else:
            graphstate = optax.apply_updates(self.graphstate, updates)

        return self.replace(step=self.step + 1, graphstate=graphstate, opt_state=new_opt_state)

    @classmethod
    def create(
        cls,
        *,  # Force keyword arguments
        step: int | None = None,
        graphdef: nn.GraphDef | None = None,
        graphstate: nn.GraphState | None = None,
        graphother: nn.GraphState | None = None,
        model: nn.Module | None = None,
        tx: optax.GradientTransformation | None = None,
        opt_state: optax.OptState | None = None,
        init_opt_state: bool = False,
    ) -> Self:
        """Create a new EasyDeLState instance.

        Factory method that provides flexible initialization of the state, either from
        an existing `nn.Module` or by providing the graph components (`graphdef`,
        `graphstate`, `graphother`) directly. It also handles optimizer state
        initialization when requested.

        This method enforces mutual exclusivity between providing a model and providing
        graph components directly, ensuring clear and unambiguous state initialization.

        Args:
            step (int | None): The initial training step count. Defaults to 0 if not
                provided. Can be set to a higher value when resuming training.
            graphdef (nn.GraphDef | None): The model's graph definition. Must be
                provided together with `graphstate` and `graphother` if not using
                `model`. Defaults to None.
            graphstate (nn.GraphState | None): The model's parameter state pytree.
                Must be provided together with `graphdef` and `graphother` if not
                using `model`. Defaults to None.
            graphother (nn.GraphState | None): The model's non-parameter state pytree.
                Must be provided together with `graphdef` and `graphstate` if not
                using `model`. Defaults to None.
            model (nn.Module | None): An EasyDeL module instance. If provided,
                `graphdef`, `graphstate`, and `graphother` are automatically extracted
                using `nn.split`. Cannot be provided simultaneously with graph
                components. Defaults to None.
            tx (optax.GradientTransformation | None): The optimizer transformation
                to use for training. Required if `init_opt_state` is True. Can be
                None for inference-only states. Defaults to None.
            opt_state (optax.OptState | None): Pre-computed optimizer state. Useful
                when loading from a checkpoint. Cannot be provided if `init_opt_state`
                is True. Defaults to None.
            init_opt_state (bool): If True, initializes the optimizer state using
                `tx.init(graphstate)`. Requires `tx` to be provided. Mutually
                exclusive with providing `opt_state`. Defaults to False.

        Returns:
            Self: A new EasyDeLState instance configured with the provided components.

        Raises:
            ValueError: If both `model` and graph components (`graphdef`, `graphstate`,
                or `graphother`) are provided.
            ValueError: If graph components are provided partially (e.g., `graphdef`
                without `graphstate` and `graphother`).
            ValueError: If `init_opt_state` is True and `opt_state` is also provided.
            ValueError: If `init_opt_state` is True but `tx` is not provided.

        Example:
            Creating state from a model::

                >>> import optax
                >>> state = EasyDeLState.create(
                ...     model=my_model,
                ...     tx=optax.adamw(1e-4),
                ...     init_opt_state=True
                ... )

            Creating state from graph components::

                >>> graphdef, graphstate, graphother = nn.split(model, nn.Param, ...)
                >>> state = EasyDeLState.create(
                ...     graphdef=graphdef,
                ...     graphstate=graphstate,
                ...     graphother=graphother,
                ...     tx=optax.adam(1e-3),
                ...     init_opt_state=True
                ... )

            Creating inference-only state (no optimizer)::

                >>> state = EasyDeLState.create(
                ...     model=my_model,
                ...     tx=None,
                ...     init_opt_state=False
                ... )

            Resuming from checkpoint with pre-computed optimizer state::

                >>> state = EasyDeLState.create(
                ...     model=my_model,
                ...     tx=my_optimizer,
                ...     opt_state=loaded_opt_state,
                ...     step=1000
                ... )

        Note:
            When using `model`, the function internally calls `nn.split(model, nn.Param, ...)`
            to extract the graph components. The first split (`nn.Param`) contains trainable
            parameters, and the ellipsis (`...`) captures all other state.

        See Also:
            - :meth:`init_tx`: Initialize optimizer after state creation.
            - :meth:`load_state`: Load complete state from checkpoint.
        """
        graph_params_provided = graphdef is not None or graphstate is not None or graphother is not None
        if model is not None and graph_params_provided:
            raise ValueError(
                "Cannot provide both a model and graph-related parameters. "
                "Choose either model or (graphdef, graphstate)."
            )

        if model is not None:
            graphdef, graphstate, graphother = nn.split(model, nn.Param, ...)
        else:
            has_graphdef = graphdef is not None
            has_graphstate = graphstate is not None
            has_graphother = graphother is not None
            provided_count = int(has_graphdef) + int(has_graphstate) + int(has_graphother)
            if provided_count == 0:
                raise ValueError(
                    "Either `model` or all graph components (`graphdef`, `graphstate`, `graphother`) must be provided.",
                )
            if provided_count != 3:
                raise ValueError(
                    "Graph components must be provided together: (`graphdef`, `graphstate`, `graphother`).",
                )
        if init_opt_state and opt_state is not None:
            raise ValueError("When passing `init_opt_state` as `True` you can't also provide `opt_state`")
        if init_opt_state and tx is None:
            raise ValueError("When passing `init_opt_state` as `True` you have to also provide `tx`.")

        if init_opt_state:
            opt_state = tx.init(graphstate)
        if step is None:
            step = 0
        if graphother is not None:
            graphother = materialize_meta_leaves(graphother, seed=42)

        return cls(
            step=step,
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
            tx=tx,
            opt_state=opt_state,
        )

    def init_tx(self: Self, tx: optax.GradientTransformation, partition_rules: PartitionLike = None) -> Self:
        """Initialize the optimizer state with automatic sharding support.

        Initializes the optimizer state (`opt_state`) for the current `graphstate`
        using the provided optimizer transformation (`tx`). This method automatically
        handles sharding based on the model's partition rules, ensuring the optimizer
        state is distributed across devices in the same manner as the model parameters.

        The initialization is performed using JIT compilation with explicit output
        shardings for efficiency, especially important for large models where the
        optimizer state can be significant (e.g., Adam has 2x the size of parameters).

        Args:
            tx (optax.GradientTransformation): The optimizer transformation to
                initialize. Common choices include `optax.adam`, `optax.adamw`,
                `optax.sgd`, or composed transformations using `optax.chain`.
            partition_rules (PartitionLike, optional): Partitioning rules for the
                optimizer state. These rules determine how optimizer state tensors
                are distributed across the device mesh. If None, uses the partition
                rules from the associated model's config. Defaults to None.

        Returns:
            Self: A new EasyDeLState instance with:
                - Initialized and sharded `opt_state`.
                - The provided `tx` set as the optimizer transformation.

        Example:
            Initialize optimizer after creating inference state::

                >>> # Create inference state
                >>> state = EasyDeLState.create(model=model)
                >>>
                >>> # Later, add optimizer for fine-tuning
                >>> tx = optax.adamw(learning_rate=1e-5, weight_decay=0.01)
                >>> state = state.init_tx(tx)

            With custom partition rules::

                >>> custom_rules = (
                ...     (".*kernel.*", PartitionSpec("fsdp", "tp")),
                ...     (".*", PartitionSpec()),
                ... )
                >>> state = state.init_tx(tx, partition_rules=custom_rules)

        Note:
            This method requires the model to have a valid mesh configuration.
            The sharding is computed using `jax.eval_shape` to avoid materializing
            the full optimizer state before sharding decisions are made.

        See Also:
            - :meth:`create`: Create state with optimizer initialization.
            - :meth:`shard_optimizer_state`: Shard existing optimizer state.
        """
        partition_rules = self.model._get_partition_rules(partition_rules)
        mesh = self.model._get_mesh(None)

        from eformer.escale import match_partition_rules

        def make(graphstate):
            return tx.init(graphstate)

        eval_opt_state = jax.eval_shape(lambda: make(self.graphstate))
        partition_specs = match_partition_rules(partition_rules, eval_opt_state)
        partition_specs, adjusted = _sanitize_partition_specs_for_shape_tree(
            partition_specs=partition_specs,
            shape_tree=eval_opt_state,
            mesh=mesh,
        )
        if adjusted:
            logger.warning("Adjusted %d non-divisible optimizer sharding specs during init_tx.", adjusted)
        named_shardings = specs_to_name_sharding(partition_specs, mesh)

        opt_state = ejit(
            make,
            out_shardings=named_shardings,
            in_shardings=(es.extract_shardings(self.graphstate, mesh=mesh),),
        )(self.graphstate)

        return self.replace(tx=tx, opt_state=opt_state)

    def shard_optimizer_state(
        self,
        opt_state: tp.Any | None = None,
        partition_rules: PartitionLike = None,
    ) -> Self:
        """Apply sharding to the optimizer state based on partition rules.

        Distributes the optimizer state across devices according to the specified
        partition rules. This is useful when loading optimizer state from a checkpoint
        that was saved in a gathered (non-sharded) format.

        Args:
            opt_state (tp.Any | None): The optimizer state pytree to shard. If None,
                uses the current `self.opt_state`. This allows sharding an external
                optimizer state while keeping it associated with this state object.
                Defaults to None.
            partition_rules (PartitionLike, optional): Partitioning rules that define
                how each tensor in the optimizer state should be distributed across
                the device mesh. If None, uses the partition rules from the model's
                config. Defaults to None.

        Returns:
            Self: A new EasyDeLState instance with the sharded `opt_state`.

        Raises:
            ValueError: If optimizer state is not initialized (neither `opt_state`
                argument nor `self.opt_state` is available).

        Example:
            Shard optimizer state after loading::

                >>> # Load gathered optimizer state
                >>> state = state.load_optimizer("checkpoint_dir")
                >>> # Apply sharding
                >>> state = state.shard_optimizer_state()

            With custom partition rules::

                >>> rules = (
                ...     (".*mu.*", PartitionSpec("fsdp")),  # First moment
                ...     (".*nu.*", PartitionSpec("fsdp")),  # Second moment
                ...     (".*", PartitionSpec()),
                ... )
                >>> state = state.shard_optimizer_state(partition_rules=rules)

        See Also:
            - :meth:`gather_optimizer_state`: Reverse operation to gather state.
            - :meth:`init_tx`: Initialize optimizer with automatic sharding.
        """
        if opt_state is None and self.opt_state is None:
            raise ValueError("Optimizer state is not initialized.")
        if opt_state is None:
            opt_state = self.opt_state
        partition_rules = self.model._get_partition_rules(partition_rules)
        mesh = self.model._get_mesh(None)

        from eformer.escale import make_shard_and_gather_fns, match_partition_rules

        partition_specs = match_partition_rules(partition_rules, opt_state)
        partition_specs, adjusted = _sanitize_partition_specs_for_shape_tree(
            partition_specs=partition_specs,
            shape_tree=opt_state,
            mesh=mesh,
        )
        if adjusted:
            logger.warning("Adjusted %d non-divisible optimizer sharding specs before shard_optimizer_state.", adjusted)
        shard_fns, _ = make_shard_and_gather_fns(partition_specs=partition_specs, mesh=mesh)
        opt_state = jax.tree_util.tree_map(lambda f, o: f(o), shard_fns, opt_state)
        return self.replace(opt_state=opt_state)

    def gather_optimizer_state(self: Self, partition_rules: PartitionLike = None) -> Self:
        """Gather the optimizer state from distributed devices to a single device.

        Reverses the sharding operation by collecting all shards of the optimizer
        state and reassembling them into complete tensors. This is typically needed
        before saving checkpoints in a portable format or when transitioning from
        distributed to single-device execution.

        Args:
            partition_rules (PartitionLike, optional): Partitioning rules that were
                used to shard the optimizer state. These rules are needed to generate
                the appropriate gather functions. If None, uses the partition rules
                from the model's config. Defaults to None.

        Returns:
            Self: A new EasyDeLState instance with the gathered (non-sharded)
            `opt_state`.

        Raises:
            AssertionError: If `opt_state` is not initialized.

        Example:
            Gather optimizer state before saving::

                >>> # Gather sharded optimizer state
                >>> gathered_state = state.gather_optimizer_state()
                >>> # Now safe to save
                >>> gathered_state.save_optimizer("checkpoint_dir")

        See Also:
            - :meth:`shard_optimizer_state`: Reverse operation to shard state.
            - :meth:`gather_state`: Gather entire state including model.
        """
        assert self.opt_state is not None, "Optimizer state is not initialized."
        partition_rules = self.model._get_partition_rules(partition_rules)
        mesh = self.model._get_mesh(None)

        from eformer.escale import make_shard_and_gather_fns, match_partition_rules

        partition_specs = match_partition_rules(partition_rules, self.opt_state)
        partition_specs, adjusted = _sanitize_partition_specs_for_shape_tree(
            partition_specs=partition_specs,
            shape_tree=self.opt_state,
            mesh=mesh,
        )
        if adjusted:
            logger.warning("Adjusted %d non-divisible optimizer sharding specs before gather_optimizer_state.", adjusted)
        _, gather = make_shard_and_gather_fns(partition_specs=partition_specs, mesh=mesh)
        self = self.replace(opt_state=jax.tree_util.tree_map(lambda f, o: f(o), gather, self.opt_state))
        return self

    def merge(self: Self, tree) -> EasyDeLBaseModule:
        """Merge a parameter tree with the graph definition to reconstruct the model.

        Combines the stored graph definition (`graphdef`) with a given state tree
        (usually parameters) and the non-parameter state (`graphother`) to reconstruct
        a complete model module. This is useful for accessing model methods or
        performing inference with modified parameters.

        Args:
            tree: The pytree (typically `nn.GraphState`) containing the parameters
                to merge. Must have the same structure as `graphstate`.

        Returns:
            EasyDeLBaseModule: The reconstructed model module with the provided
            parameters and the stored non-parameter state.

        Example:
            Reconstruct model with current parameters::

                >>> model = state.merge(state.graphstate)
                >>> output = model(input_data)

            Use with modified parameters::

                >>> # Scale all parameters by 0.5
                >>> scaled_params = jax.tree_map(lambda x: x * 0.5, state.graphstate)
                >>> model = state.merge(scaled_params)

        Note:
            For convenience, use the `model` property to access the reconstructed
            model with the current `graphstate`.

        See Also:
            - :attr:`model`: Property that returns the reconstructed model.
            - :meth:`merge_to_state`: Update state with new parameters.
        """
        return nn.merge(self.graphdef, tree, self.graphother)

    def merge_to_state(self: Self, tree) -> Self:
        """Create a new state with updated parameters.

        Creates a new `EasyDeLState` by replacing the current `graphstate` with the
        provided tree while keeping all other state components unchanged.

        Args:
            tree: The pytree (typically `nn.GraphState`) containing the new parameters.
                Must have the same structure as the current `graphstate`.

        Returns:
            Self: A new EasyDeLState instance with the updated `graphstate`.

        Example:
            Update state with averaged parameters::

                >>> # Exponential moving average of parameters
                >>> ema_params = jax.tree_map(
                ...     lambda ema, new: 0.999 * ema + 0.001 * new,
                ...     ema_state.graphstate,
                ...     state.graphstate
                ... )
                >>> ema_state = ema_state.merge_to_state(ema_params)

        See Also:
            - :meth:`merge`: Reconstruct model from parameters.
        """
        return self.replace(graphstate=tree)

    @property
    def model(self) -> EasyDeLBaseModule:
        """Reconstruct and return the full EasyDeL model module.

        Convenience property that merges the stored graph definition, parameter state,
        and non-parameter state to create a complete model module instance. This is
        equivalent to calling `merge(self.graphstate)`.

        Returns:
            EasyDeLBaseModule: The fully reconstructed model module that can be used
            for inference or to access model methods.

        Example:
            Access model for inference::

                >>> model = state.model
                >>> output = model(input_ids, attention_mask=mask)

            Access model configuration::

                >>> config = state.model.config
                >>> print(f"Hidden size: {config.hidden_size}")

        Note:
            Each access to this property reconstructs the model, which has some
            overhead. For repeated access, consider caching the result::

                >>> model = state.model  # Cache this
                >>> for batch in batches:
                ...     output = model(batch)

        See Also:
            - :meth:`merge`: Explicit merge with custom parameters.
        """
        return nn.merge(self.graphdef, self.graphstate, self.graphother)

    @property
    def size(self) -> int:
        """Calculate the total memory size of the state in bytes.

        Computes the combined memory footprint of the model parameters (`graphstate`)
        and the optimizer state (`opt_state`). This is useful for memory planning and
        estimating checkpoint sizes.

        Returns:
            int: The total size in bytes of all JAX arrays in `graphstate` and
            `opt_state`. Returns 0 for any component that is None.

        Example:
            Check state memory usage::

                >>> size_bytes = state.size
                >>> size_gb = size_bytes / (1024 ** 3)
                >>> print(f"State size: {size_gb:.2f} GB")

            Compare with and without optimizer::

                >>> inference_state = EasyDeLState.create(model=model)
                >>> training_state = EasyDeLState.create(
                ...     model=model, tx=optax.adam(1e-3), init_opt_state=True
                ... )
                >>> print(f"Inference: {inference_state.size / 1e9:.2f} GB")
                >>> print(f"Training: {training_state.size / 1e9:.2f} GB")

        Note:
            This calculation only includes JAX arrays (`jnp.ndarray`). Other Python
            objects in the state are not counted.
        """

        def calculate_size(pytree):
            """Calculate total size of JAX arrays in a pytree."""
            if pytree is None:
                return 0
            leaves, _ = jax.tree_util.tree_flatten(pytree)
            return sum(leaf.size * leaf.itemsize for leaf in leaves if isinstance(leaf, jnp.ndarray))

        opt_state_size = calculate_size(self.opt_state)
        graphstate_size = calculate_size(self.graphstate)
        return opt_state_size + graphstate_size

    def load_optimizer(
        self: Self,
        load_directory: str | ePathLike,
        checkpointer: Checkpointer | None = None,
        tx_template: None | optax.GradientTransformation = None,
    ) -> Self:
        """Load optimizer state from saved checkpoint files.

        Reads the optimizer state from a checkpoint directory, handling both legacy
        formats (separate structure pickle and SafeTensors files) and modern
        TensorStore-based checkpoints. The loaded state is automatically associated
        with the current model configuration.

        Args:
            load_directory (str | ePathLike): Path to the directory containing the
                saved optimizer state files. The directory should contain either:
                - Modern format: TensorStore checkpoint with 'tx' prefix.
                - Legacy format: `OPTIMIZER_STRUCT_NAME` (pickle) and `OPTIMIZER_NAME`
                  (SafeTensors) files.
            checkpointer (Checkpointer | None): Custom checkpointer instance to use
                for loading. If None, a default Checkpointer is created. Useful for
                custom loading configurations. Defaults to None.
            tx_template (optax.GradientTransformation | None): Template optimizer
                transformation used to infer the structure of the optimizer state.
                If None, uses `self.tx`. Required if the checkpoint format needs
                structure inference. Defaults to None.

        Returns:
            Self: A new EasyDeLState instance with:
                - Loaded `opt_state` containing optimizer buffers (momentum, etc.).
                - Updated `step` from the checkpoint metadata.

        Raises:
            FileNotFoundError: If the required optimizer files are not found in
                the specified directory.
            Exception: If deserialization fails or the checkpoint is corrupted.

        Example:
            Load optimizer for continued training::

                >>> state = EasyDeLState.create(model=model, tx=optax.adam(1e-3))
                >>> state = state.load_optimizer("checkpoints/step_5000")
                >>> print(f"Resuming from step {state.step}")

            With custom checkpointer::

                >>> checkpointer = Checkpointer(
                ...     base_path=ePath("checkpoints"),
                ...     save_interval=1000
                ... )
                >>> state = state.load_optimizer(
                ...     "checkpoints/step_5000",
                ...     checkpointer=checkpointer
                ... )

        Note:
            This method attempts multiple loading strategies for backward compatibility:
            1. First tries modern TensorStore format.
            2. Falls back to legacy pickle + SafeTensors format.
            3. Handles tree structure mismatches gracefully.

        See Also:
            - :meth:`save_optimizer`: Save optimizer state.
            - :meth:`load_state`: Load complete state including model.
        """
        load_directory = ePath(load_directory)

        if checkpointer is None:
            checkpointer = Checkpointer(
                base_path=load_directory,
                save_interval=None,
                step_policies=[],
            )
        org_path = load_directory
        optim_path = load_directory if AM.is_tensorstore(load_directory) else load_directory / OPTIMIZER_NAME
        struct_path = load_directory / OPTIMIZER_STRUCT_NAME
        tx_struct_path = load_directory / TX_STRUCT_JSON
        partition_rules = self.model._get_partition_rules(None)

        def new_method(tx_template):
            """Load using modern TensorStore format."""
            path = str(AsyncCheckpointManager.safe_loadpath(org_path))
            tx_template = tx_template if tx_template is not None else self.tx
            template = None
            if tx_template is not None:
                template = jax.eval_shape(tx_template.init, self.graphstate)
            opt_state, metadata = checkpointer.load_pytree(
                mesh=self.model.mesh,
                path=path,
                partition_rules=partition_rules,
                prefix="tx",
                load_treedef=True,
                discover_latest=True,
                discover_raise=False,
                template=template,
            )
            step = metadata.get("step", 0)
            return opt_state, step

        if not tx_struct_path.exists() or not struct_path.exists():
            try:
                opt_state, step = new_method(tx_template)
                logger.info(f"Optimizer state loaded from {load_directory} (step {step}).")
                return self.replace(opt_state=opt_state, step=jnp.asarray(step))
            except Exception:
                traceback.print_exc()

        try:
            if not AsyncCheckpointManager.is_tensorstore(optim_path):
                treedef, step = pickle.loads(struct_path.read_bytes())
                leaves, _ = AsyncCheckpointManager().load(
                    path=AsyncCheckpointManager.safe_loadpath(optim_path),
                    mesh=self.model.mesh,
                    partition_rules=partition_rules,
                    prefix="tx",
                )
                recreated = [None] * len(leaves)
                for i in range(len(leaves)):
                    try:
                        recreated[i] = leaves[f"param_idx_{i}"]
                    except KeyError:
                        recreated[i] = leaves[f"param_{i}"]

                opt_state = jax.tree_util.tree_unflatten(treedef, recreated)
            else:
                opt_state, step = new_method(tx_template)

            logger.info(f"Optimizer state loaded from {load_directory} (step {step}).")
            return self.replace(opt_state=opt_state, step=jnp.asarray(step))
        except Exception as e:
            if "Too many leaves for PyTreeDef" in str(e):
                try:
                    opt_state, step = new_method(tx_template)
                    return self.replace(opt_state=opt_state, step=jnp.asarray(step))
                except Exception:
                    ...
            logger.error(f"Optimizer load failed: {e!s}")
            raise e

    def save_optimizer(
        self,
        save_directory: str | ePathLike,
        float_dtype: jnp.dtype | None = None,
        checkpointer: Checkpointer | None = None,
        step: int | None = None,
    ) -> None:
        """Save the optimizer state to a directory.

        Saves the optimizer state using the Checkpointer with optional dtype conversion
        for reduced checkpoint size. The state is saved as a pytree with metadata
        including the current training step.

        Args:
            save_directory (str | ePathLike): Directory path where the optimizer
                state will be saved. The directory will be created if it doesn't
                exist.
            float_dtype (jnp.dtype | None): Optional dtype to convert floating-point
                values to before saving. Useful for reducing checkpoint size by
                saving in lower precision (e.g., `jnp.float16` or `jnp.bfloat16`).
                Defaults to None (keeps original dtype).
            checkpointer (Checkpointer | None): Custom checkpointer instance to use
                for saving. If None, a default Checkpointer is created. Defaults
                to None.
            step (int | None): Training step to record in checkpoint metadata. If
                None, the value is not explicitly passed to the checkpointer.
                Defaults to None.

        Returns:
            None

        Raises:
            Exception: If the save operation fails (e.g., disk full, permission denied).

        Example:
            Basic optimizer save::

                >>> state.save_optimizer("checkpoints/optimizer_step_1000")

            Save with reduced precision::

                >>> state.save_optimizer(
                ...     "checkpoints/optimizer_step_1000",
                ...     float_dtype=jnp.float16
                ... )

            With custom checkpointer::

                >>> checkpointer = Checkpointer(
                ...     base_path=ePath("checkpoints"),
                ...     save_interval=1000
                ... )
                >>> state.save_optimizer(
                ...     "checkpoints/step_1000",
                ...     checkpointer=checkpointer
                ... )

        Note:
            If `opt_state` is None (no optimizer initialized), this method logs
            an informational message and returns without saving.

        See Also:
            - :meth:`load_optimizer`: Load optimizer state.
            - :meth:`save_state`: Save complete state including model.
        """
        save_directory = ePath(save_directory)
        if checkpointer is None:
            checkpointer = Checkpointer(
                base_path=save_directory,
                save_interval=None,
                step_policies=[],
            )
        if self.opt_state is not None:
            save_directory.mkdir(parents=True, exist_ok=True)
            optim_path = save_directory
            logger.info(f"Coordinated optimizer save through {optim_path}")
            try:
                with self.model.mesh:
                    checkpointer.save_pytree(
                        tree=self.opt_state,
                        mesh=self.model.mesh,
                        dtype=float_dtype,
                        prefix="tx",
                        # Don't pass step here - save_directory is already the checkpoint directory
                        # Passing step would create duplicate run-{step}/run-{step} structure
                    )
            except Exception as e:
                logger.error(f"Optimizer save failed: {e!s}")
                raise
        else:
            logger.info("Current State don't contain any Optimizer.")

    def save_state(
        self,
        save_directory: str | os.PathLike | ePathLike,
        float_dtype: jnp.dtype | None = None,
        save_optimizer: bool = True,
        step: int | None = None,
    ) -> None:
        """Save the complete EasyDeLState to a directory.

        Saves all components of the state including model parameters (using the
        model's `save_pretrained` method) and optionally the optimizer state.
        This creates a complete checkpoint that can be loaded later with
        `load_state`.

        Args:
            save_directory (str | os.PathLike | ePathLike): The directory path
                where the state will be saved. Will be created if it doesn't exist.
            float_dtype (jnp.dtype | None): Optional dtype to cast floating-point
                parameters and optimizer state to before saving. Useful for reducing
                checkpoint size. Defaults to None (keeps original dtypes).
            save_optimizer (bool): If True, saves the optimizer state alongside
                the model parameters. Set to False for inference-only checkpoints.
                Defaults to True.
            step (int | None): Training step to record in checkpoint metadata.
                If None, uses the current `self.step` value. Defaults to None.

        Returns:
            None

        Example:
            Save complete training state::

                >>> state.save_state("checkpoints/step_10000")

            Save inference checkpoint (no optimizer)::

                >>> state.save_state(
                ...     "checkpoints/inference_model",
                ...     save_optimizer=False
                ... )

            Save with reduced precision::

                >>> state.save_state(
                ...     "checkpoints/step_10000",
                ...     float_dtype=jnp.bfloat16
                ... )

        Note:
            The saved checkpoint directory will contain:
            - Model configuration (config.json)
            - Model parameters (easydel-model.parameters or TensorStore)
            - Optimizer state if `save_optimizer=True` (TensorStore format)

        See Also:
            - :meth:`load_state`: Load complete state from checkpoint.
            - :meth:`save_optimizer`: Save only optimizer state.
        """
        save_directory = ePath(save_directory)
        if step is None:
            step = self.step.item() if isinstance(self.step, jnp.ndarray) else self.step
        if save_optimizer:
            self.save_optimizer(save_directory=save_directory, float_dtype=float_dtype, step=step)
        else:
            logger.info("Skipping optimizer saving as requested.")

        self.model.save_pretrained(
            save_directory=save_directory,
            gather_fns=self.model._gather_fns,
            float_dtype=float_dtype,
            step=step,
        )

    @classmethod
    def load_state(
        cls,
        load_directory: str | os.PathLike,
        device: jax.Device | None = "cpu",  # type:ignore
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision | None = None,
        sharding_axis_dims: collections.abc.Sequence[int] = (1, -1, 1, 1, 1),
        sharding_dcn_axis_dims: collections.abc.Sequence[int] | None = None,
        sharding_axis_names: collections.abc.Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"),
        partition_axis: PartitionAxis | None = None,
        shard_fns: collections.abc.Mapping[tuple, tp.Callable] | dict | None = None,
        backend: EasyDeLBackends | None = None,
        platform: EasyDeLPlatforms | None = None,
        config_kwargs: EasyDeLBaseConfigDict | None = None,
        model_task: TaskType = TaskType.AUTO_BIND,
        auto_shard_model: bool = True,
        partition_rules: tuple[tuple[str, PartitionSpec], ...] | None = None,
        quantization_config: "QuantizationConfig | None" = None,
        apply_quantization: bool = False,
        verbose: bool = True,
        tx_template: optax.GradientTransformation | None = None,
        **kwargs,
    ) -> Self:
        """Load an EasyDeLState from a saved checkpoint directory.

        Class method that reconstructs the complete training state from a checkpoint,
        including model configuration, model parameters, and optionally optimizer state.
        Handles various configurations for device placement, data types, sharding, and
        quantization.

        Args:
            load_directory (str | os.PathLike): Path to the directory containing the
                saved state (configuration, model weights, and potentially optimizer
                state).
            device (jax.Device | None): The JAX device to load the model onto initially.
                Common values: 'cpu', 'gpu', 'tpu'. Defaults to 'cpu' for safe loading
                before sharding.
            dtype (jnp.dtype): The data type for computation activations (e.g., forward
                pass intermediate values). Defaults to `jnp.bfloat16`.
            param_dtype (jnp.dtype): The data type for model parameters. Defaults to
                `jnp.bfloat16`. Can differ from `dtype` for mixed-precision training.
            precision (jax.lax.Precision | None): JAX precision level for matrix
                operations (e.g., `jax.lax.Precision.HIGHEST`). Defaults to None
                (uses JAX default).
            sharding_axis_dims (collections.abc.Sequence[int]): Dimensions of the device mesh for
                sharding. Format: (dp, fsdp, ep, tp, sp) where -1 means infer from
                available devices. Defaults to (1, -1, 1, 1, 1).
            sharding_dcn_axis_dims (collections.abc.Sequence[int] | None): Optional dimensions for
                data-center network (DCN) sharding in multi-host setups. Defaults to
                None.
            sharding_axis_names (collections.abc.Sequence[str]): Names for sharding axes matching
                `sharding_axis_dims`. Defaults to ("dp", "fsdp", "ep", "tp", "sp").
            partition_axis (PartitionAxis | None): Configuration object for partitioning
                specific model dimensions. Defaults to None (uses model defaults).
            shard_fns (collections.abc.Mapping[tuple, tp.Callable] | dict | None): Custom sharding
                functions mapped by parameter path tuples. Defaults to None.
            backend (EasyDeLBackends | None): Backend framework (e.g., JAX, PyTorch).
                Defaults to None (auto-detected).
            platform (EasyDeLPlatforms | None): Hardware platform (e.g., TPU, GPU).
                Defaults to None (auto-detected).
            config_kwargs (EasyDeLBaseConfigDict | None): Dictionary of keyword
                arguments to override in the loaded model configuration. Defaults
                to None.
            model_task (TaskType): The model task type (e.g., CAUSAL_LM, SEQ2SEQ_LM).
                Defaults to TaskType.AUTO_BIND (auto-detect from config).
            auto_shard_model (bool): If True, automatically shards the loaded model
                and optimizer state based on sharding configuration. Defaults to True.
            partition_rules (tuple[tuple[str, PartitionSpec], ...] | None): Explicit
                partition rules as (regex_pattern, PartitionSpec) tuples. Defaults to
                None (uses model config rules).
            quantization_config (QuantizationConfig | None): Configuration for model
                quantization. Defaults to None (no quantization).
            apply_quantization (bool): If True, applies quantization to model linear
                modules. Defaults to False.
            verbose (bool): If True, logs detailed information during loading.
                Defaults to True.
            tx_template (optax.GradientTransformation | None): Template optimizer for
                inferring optimizer state structure when loading. Defaults to None.
            **kwargs: Additional keyword arguments passed to the underlying
                `EasyDeLBaseModule.from_pretrained` method.

        Returns:
            Self: An EasyDeLState instance containing:
                - Loaded and configured model.
                - Optimizer state if found in checkpoint.
                - Training step from checkpoint metadata.
                - Sharding applied if `auto_shard_model=True`.

        Raises:
            FileNotFoundError: If the `load_directory` or essential files (config,
                model weights) are not found.
            ValueError: If there are inconsistencies in arguments or configuration.

        Example:
            Basic checkpoint loading::

                >>> state = EasyDeLState.load_state(
                ...     "checkpoints/step_10000",
                ...     dtype=jnp.bfloat16
                ... )
                >>> print(f"Loaded at step {state.step}")

            Loading with specific sharding::

                >>> state = EasyDeLState.load_state(
                ...     "checkpoints/step_10000",
                ...     sharding_axis_dims=(1, 4, 1, 2, 1),  # 8 devices
                ...     auto_shard_model=True
                ... )

            Loading for inference (CPU, no sharding)::

                >>> state = EasyDeLState.load_state(
                ...     "checkpoints/step_10000",
                ...     device="cpu",
                ...     auto_shard_model=False
                ... )

            Loading with quantization::

                >>> from easydel.layers import QuantizationConfig
                >>> state = EasyDeLState.load_state(
                ...     "checkpoints/step_10000",
                ...     quantization_config=QuantizationConfig(bits=8),
                ...     apply_quantization=True
                ... )

        Note:
            - The model configuration is automatically loaded from `config.json`
              in the checkpoint directory.
            - Optimizer state loading is attempted but failures are logged as
              info (not errors), allowing inference-only usage.
            - When `auto_shard_model=True`, both model and optimizer state are
              sharded according to the specified rules.

        See Also:
            - :meth:`save_state`: Save state to checkpoint.
            - :meth:`create`: Create state from an existing model.
        """
        from easydel.modules.auto.auto_configuration import AutoEasyDeLConfig

        from .base_module import EasyDeLBaseModule

        config = AutoEasyDeLConfig.from_pretrained(
            load_directory,
            sharding_axis_dims=sharding_axis_dims,
            sharding_dcn_axis_dims=sharding_dcn_axis_dims,
            sharding_axis_names=sharding_axis_names,
            partition_axis=partition_axis,
            from_torch=False,
            backend=backend,
            platform=platform,
            model_task=model_task,
        )
        model_task = AutoEasyDeLConfig.bind_model_task(model_task, config.architectures)

        class _BaseModuleLoader(EasyDeLBaseModule):
            """Internal loader class for binding model task."""

            _model_task = model_task

        model = _BaseModuleLoader.from_pretrained(
            pretrained_model_name_or_path=load_directory,
            device=device,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            sharding_axis_dims=sharding_axis_dims,
            sharding_dcn_axis_dims=sharding_dcn_axis_dims,
            sharding_axis_names=sharding_axis_names,
            partition_axis=partition_axis,
            shard_fns=shard_fns,
            backend=backend,
            platform=platform,
            config_kwargs=config_kwargs,
            auto_shard_model=auto_shard_model,
            partition_rules=partition_rules,
            quantization_config=quantization_config,
            apply_quantization=apply_quantization,
            verbose=verbose,
            **kwargs,
        )

        state = cls.create(step=jnp.array(0), model=model)
        try:
            cmg = jax.default_device(device) if device is not None else contextlib.nullcontext()
            with cmg:
                state = state.load_optimizer(load_directory=load_directory, tx_template=tx_template)
        except Exception:
            logger.info("EasyDeLState couldn't restore the optimizer/tx (maybe there wasn't any!).")
        if auto_shard_model:
            state = state.shard_state()
        return state

    def shard_with_shape(self: Self, shape) -> Self:
        """Apply sharding constraints based on a reference shape pytree.

        Takes a pytree `shape` with the same structure as the `EasyDeLState` but
        containing sharding annotations (e.g., `NamedSharding`) instead of actual
        array data. Applies these shardings as device placement constraints to the
        corresponding arrays in the current state.

        This is useful for ensuring state arrays are placed on specific devices
        according to a pre-computed sharding plan.

        Args:
            shape: A pytree with the same structure as `self`, where leaves contain
                sharding annotations (e.g., `jax.sharding.NamedSharding`) instead
                of array data.

        Returns:
            Self: A new EasyDeLState instance with sharding constraints applied
            to all arrays according to the provided shape pytree.

        Example:
            Apply sharding from computed shape::

                >>> # Get sharding plan from jax.eval_shape
                >>> shape = jax.eval_shape(lambda: state)
                >>> # Apply actual shardings
                >>> sharded_state = state.shard_with_shape(shape)

        Note:
            This method uses `jax.device_put` internally to place arrays according
            to the sharding specifications in the shape pytree.

        See Also:
            - :meth:`shard_state`: Shard using partition rules.
            - :attr:`shardings`: Get current sharding annotations.
        """
        self = nn.from_tree(
            jax.tree_util.tree_map(
                lambda arr, sharding: jax.device_put(arr, sharding),
                nn.to_tree(self),
                nn.to_tree(shape),
            )
        )
        return self

    def shard_state(
        self,
        partition_rules: PartitionLike = None,
        mesh: Mesh = None,
    ) -> Self:
        """Shard the entire state based on partition rules.

        Applies sharding to both model parameters and optimizer state according to
        the specified partition rules and device mesh. This is the primary method
        for distributing state across multiple devices for data/model parallelism.

        Args:
            partition_rules (PartitionLike, optional): Partitioning rules as a
                sequence of (regex_pattern, PartitionSpec) tuples. Parameters
                matching each pattern are sharded according to the corresponding
                PartitionSpec. If None, uses rules from the model's config.
                Defaults to None.
            mesh (Mesh, optional): The JAX device mesh to shard across. Defines the
                topology of devices (e.g., 2x4 grid for 8 devices). If None, uses
                the model's configured mesh. Defaults to None.

        Returns:
            Self: A new EasyDeLState instance with all components (graphstate,
            graphother, opt_state) sharded across the device mesh.

        Example:
            Shard with default rules::

                >>> sharded_state = state.shard_state()

            Shard with custom rules::

                >>> rules = (
                ...     (".*embed.*", PartitionSpec("tp")),
                ...     (".*kernel.*", PartitionSpec("fsdp", "tp")),
                ...     (".*", PartitionSpec()),
                ... )
                >>> sharded_state = state.shard_state(partition_rules=rules)

            Shard with custom mesh::

                >>> from jax.sharding import Mesh
                >>> import numpy as np
                >>> devices = np.array(jax.devices()).reshape(2, 4)
                >>> mesh = Mesh(devices, ("dp", "tp"))
                >>> sharded_state = state.shard_state(mesh=mesh)

        Note:
            This method shards the entire state pytree, including nested structures.
            For sharding only model or optimizer separately, use `shard_model` or
            `shard_optimizer_state`.

        See Also:
            - :meth:`shard_model`: Shard only model parameters.
            - :meth:`shard_optimizer_state`: Shard only optimizer state.
            - :meth:`gather_state`: Reverse operation to gather state.
        """
        from eformer.escale import make_shard_and_gather_fns, match_partition_rules

        rules = partition_rules or self.model._get_partition_rules(None)
        mesh = mesh or self.model._get_mesh(None)

        def appy_sharding_on_tree(tree):
            """Apply sharding functions to a pytree."""
            partition_specs = match_partition_rules(rules, tree)
            partition_specs, adjusted = _sanitize_partition_specs_for_shape_tree(
                partition_specs=partition_specs,
                shape_tree=tree,
                mesh=mesh,
            )
            if adjusted:
                logger.warning("Adjusted %d non-divisible sharding specs before shard_state.", adjusted)
            shard_fns, _ = make_shard_and_gather_fns(partition_specs, mesh)
            return jax.tree_util.tree_map(lambda f, o: f(o), shard_fns, tree)

        state_for_shard = self.replace(graphother=materialize_meta_leaves(self.graphother, seed=42))
        return appy_sharding_on_tree(state_for_shard)

    def gather_state(self) -> Self:
        """Gather the entire state from distributed devices.

        Reverses sharding by collecting all distributed state components (model
        parameters and optimizer state) from across the device mesh and reassembling
        them into complete tensors on a single device.

        Returns:
            Self: A new EasyDeLState instance with all components gathered to a
            single device.

        Example:
            Gather for checkpointing::

                >>> gathered_state = sharded_state.gather_state()
                >>> gathered_state.save_state("checkpoint")

            Gather for inspection::

                >>> gathered_state = state.gather_state()
                >>> param_values = jax.tree_map(
                ...     lambda x: x.mean().item(),
                ...     gathered_state.graphstate
                ... )

        Note:
            Gathering large models to a single device can cause out-of-memory errors.
            For large models, consider using streaming save methods that don't require
            full gathering.

        See Also:
            - :meth:`shard_state`: Reverse operation to shard state.
            - :meth:`gather_model`: Gather only model parameters.
            - :meth:`gather_optimizer_state`: Gather only optimizer state.
        """
        if self.opt_state is not None:
            self = self.gather_optimizer_state()
        self = self.gather_model()
        return self

    def gather_model(
        self,
        partition_rules: PartitionLike = None,
        mesh: Mesh | None = None,
    ) -> Self:
        """Gather model parameters from distributed devices.

        Collects the sharded model parameters (`graphstate` and `graphother`) from
        across the device mesh and reassembles them into complete tensors. This is
        typically needed before saving model weights in a portable format.

        Args:
            partition_rules (PartitionLike, optional): Partitioning rules that were
                used for the original sharding. Needed to generate appropriate gather
                functions. If None, uses rules from the model's config. Defaults to
                None.
            mesh (Mesh | None): The JAX device mesh to gather from. If None, uses
                the model's configured mesh. Defaults to None.

        Returns:
            Self: A new EasyDeLState instance with gathered `graphstate` and
            `graphother`.

        Example:
            Gather model for saving::

                >>> gathered_state = state.gather_model()
                >>> # Now graphstate contains complete (non-sharded) parameters

            With specific rules::

                >>> gathered_state = state.gather_model(
                ...     partition_rules=custom_rules,
                ...     mesh=custom_mesh
                ... )

        See Also:
            - :meth:`shard_model`: Reverse operation to shard model.
            - :meth:`gather_state`: Gather entire state including optimizer.
        """
        from eformer.escale import make_shard_and_gather_fns, match_partition_rules

        rules = partition_rules or self.model._get_partition_rules(None)
        mesh = mesh or self.model._get_mesh(None)

        def _apply_gather_on_tree(tree, tree_name: str):
            tree = materialize_meta_leaves(tree, seed=42)
            partition_specs = match_partition_rules(rules=rules, tree=tree)
            partition_specs, adjusted = _sanitize_partition_specs_for_shape_tree(
                partition_specs=partition_specs,
                shape_tree=tree,
                mesh=mesh,
            )
            if adjusted:
                logger.warning(
                    "Adjusted %d non-divisible sharding specs before gather_model (%s).",
                    adjusted,
                    tree_name,
                )
            _, gather_fns = make_shard_and_gather_fns(partition_specs=partition_specs, mesh=mesh)
            return jax.tree_util.tree_map(lambda f, o: f(o), gather_fns, tree)

        graphstate = _apply_gather_on_tree(self.graphstate, "graphstate")
        graphother = _apply_gather_on_tree(self.graphother, "graphother")
        self = self.replace(graphstate=graphstate, graphother=graphother)
        return self

    def shard_model(self: Self, partition_rules: PartitionLike = None, mesh: Mesh | None = None) -> Self:
        """Shard model parameters based on partition rules.

        Distributes the model parameters (`graphstate` and `graphother`) across
        devices according to the specified partition rules. This enables data and
        model parallelism for training and inference with large models.

        Args:
            partition_rules (PartitionLike, optional): Partitioning rules as a
                sequence of (regex_pattern, PartitionSpec) tuples. If None, uses
                rules from the model's config. Defaults to None.
            mesh (Mesh | None): The JAX device mesh to shard across. If None, uses
                the model's configured mesh. Defaults to None.

        Returns:
            Self: A new EasyDeLState instance with sharded `graphstate` and
            `graphother`.

        Example:
            Shard model with default rules::

                >>> sharded_state = state.shard_model()

            Shard with custom configuration::

                >>> rules = (
                ...     (".*attention.*kernel.*", PartitionSpec("tp", None)),
                ...     (".*mlp.*kernel.*", PartitionSpec(None, "tp")),
                ...     (".*", PartitionSpec()),
                ... )
                >>> sharded_state = state.shard_model(partition_rules=rules)

        Note:
            This method only shards model parameters, not optimizer state. Use
            `shard_state` to shard both, or `shard_optimizer_state` for optimizer
            only.

        See Also:
            - :meth:`gather_model`: Reverse operation to gather model.
            - :meth:`shard_state`: Shard entire state.
            - :meth:`shard_optimizer_state`: Shard optimizer state.
        """
        rules = partition_rules or self.model._get_partition_rules(None)
        mesh = mesh or self.model._get_mesh(None)

        def appy_sharding_on_tree(tree):
            """Apply sharding functions to a pytree."""
            from eformer.escale import make_shard_and_gather_fns, match_partition_rules

            tree = materialize_meta_leaves(tree, seed=42)
            partition_specs = match_partition_rules(rules, tree)
            partition_specs, adjusted = _sanitize_partition_specs_for_shape_tree(
                partition_specs=partition_specs,
                shape_tree=tree,
                mesh=mesh,
            )
            if adjusted:
                logger.warning("Adjusted %d non-divisible sharding specs before state.shard_model.", adjusted)
            shard_fns, _ = make_shard_and_gather_fns(partition_specs, mesh)
            return jax.tree_util.tree_map(lambda f, o: f(o), shard_fns, tree)

        graphstate = appy_sharding_on_tree(self.graphstate)
        graphother = appy_sharding_on_tree(self.graphother)

        self = self.replace(graphstate=graphstate, graphother=graphother)
        return self

    @property
    def mesh(self) -> Mesh:
        """Get the JAX device mesh from the model.

        Returns the device mesh used for sharding operations. The mesh defines
        the topology of devices and named axes for distributed computation.

        Returns:
            Mesh: The JAX device mesh configured for the model.

        Example:
            Access mesh properties::

                >>> mesh = state.mesh
                >>> print(f"Mesh shape: {mesh.shape}")
                >>> print(f"Mesh axis names: {mesh.axis_names}")
        """
        return self.model.mesh

    @property
    def shardings(self) -> Self:
        """Get the sharding annotations for all state components.

        Retrieves the sharding specification (e.g., `NamedSharding`) for each array
        in the state pytree. Useful for inspecting how state is distributed across
        devices.

        Returns:
            Self: A pytree with the same structure as `self`, where each leaf
            contains the sharding annotation of the corresponding array, or None
            if the array has no sharding information.

        Example:
            Inspect shardings::

                >>> shardings = state.shardings
                >>> # Check specific parameter sharding
                >>> print(shardings.graphstate)

            Verify sharding configuration::

                >>> for path, sharding in jax.tree_util.tree_leaves_with_path(
                ...     state.shardings.graphstate
                ... ):
                ...     print(f"{path}: {sharding}")
        """
        return nn.from_tree(
            jax.tree_util.tree_map(
                lambda x: x.sharding if hasattr(x, "sharding") else None,
                nn.to_tree(self),
            )
        )

    def __repr__(self) -> str:
        """Return a string representation of the EasyDeLState.

        Provides a concise representation showing whether the state includes an
        optimizer and the model information.

        Returns:
            str: String representation in format:
                "[TxIncluded-]EasyDeLState-{model_repr}"
        """
        return ("TxIncluded-" if self.opt_state is not None else "") + "EasyDeLState-" + str(self.model)

    __str__ = __repr__
