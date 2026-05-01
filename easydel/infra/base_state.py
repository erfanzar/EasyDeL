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
import dataclasses
import datetime as dt
import json
import os
import typing as tp
import uuid
from typing import Self

import jax
import optax  # pyright: ignore[reportMissingTypeStubs]
import spectrax as spx
from eformer.loggings import get_logger
from eformer.paths import ePath, ePathLike
from jax import device_get
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from spectrax import PartitionAxis, make_shard_and_gather_fns
from spectrax.serialization import AsyncCheckpointManager, Checkpointer

from easydel.infra.factory import TaskType
from easydel.utils.helpers import is_remote_path
from easydel.utils.traversals import deepcopy_model

from .sharding import MeshLike, replicated_named_sharding, sanitize_partition_specs_for_shape_tree
from .utils import device_put_or_shard_abstract, materialize_meta_leaves


class _PyTreeNode:
    """Drop-in replacement for SpecTrax.struct.PyTreeNode (now using dataclasses + jax.tree_util).

    A dataclass-like base class where some fields are JAX pytree nodes
    and others are static (aux) data. Mimics the ``struct.field(pytree_node=...)``
    API used by EasyDeLState.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dataclasses.dataclass(cls, frozen=True)
        jax.tree_util.register_pytree_with_keys(
            cls,
            cls._tree_flatten_with_keys,
            cls._tree_unflatten,
        )

    @classmethod
    def _tree_flatten_with_keys(cls, obj):
        fields = dataclasses.fields(obj)
        children = []
        aux = []
        for f in fields:
            val = getattr(obj, f.name)
            pytree_node = f.metadata.get("pytree_node", True)
            if pytree_node:
                children.append((jax.tree_util.GetAttrKey(f.name), val))
                aux.append((f.name, None, pytree_node))
            else:
                aux.append((f.name, val, pytree_node))
        return children, tuple(aux)

    @classmethod
    def _tree_unflatten(cls, aux, children):
        kwargs = {}
        child_idx = 0
        for name, val, pytree_node in aux:
            if pytree_node:
                kwargs[name] = children[child_idx]
                child_idx += 1
            else:
                kwargs[name] = val
        return cls(**kwargs)

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


def _field(pytree_node=True, default=dataclasses.MISSING, default_factory=dataclasses.MISSING, **kwargs):
    metadata = dict(kwargs.pop("metadata", {}))
    metadata["pytree_node"] = pytree_node
    if default_factory is not dataclasses.MISSING:
        return dataclasses.field(default_factory=default_factory, metadata=metadata, **kwargs)
    return dataclasses.field(default=default, metadata=metadata, **kwargs)


if tp.TYPE_CHECKING:
    from easydel.infra.base_config import EasyDeLBaseConfigDict
    from easydel.infra.etils import EasyDeLBackends, EasyDeLPlatforms
    from easydel.layers import QuantizationConfig

    from .base_module import EasyDeLBaseModule

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

RESUME_MODEL_SUBDIR = "_resume_model"
"""Subdirectory used for the LoRA-preserving model copy inside merged state checkpoints."""

logger = get_logger(__name__)


def _read_checkpoint_metadata(load_directory: str | os.PathLike | ePathLike) -> dict[str, tp.Any]:
    """Best-effort read of the checkpoint discovery metadata."""
    metadata_path = ePath(load_directory) / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception:
        logger.debug("Failed to read checkpoint metadata from %s.", metadata_path, exc_info=True)
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _has_saved_optimizer_state(load_directory: str | os.PathLike | ePathLike) -> bool:
    """Return whether a checkpoint directory contains resumable optimizer artifacts.

    When ``metadata.json`` explicitly records ``has_optimizer_state``, that
    value is treated as authoritative so stale optimizer files left behind in a
    reused directory do not accidentally turn a weights-only checkpoint back
    into a resumable one. Older checkpoints without that metadata still fall
    back to artifact discovery for backward compatibility.
    """
    load_directory = ePath(load_directory)
    metadata = _read_checkpoint_metadata(load_directory)
    if "has_optimizer_state" in metadata:
        return bool(metadata["has_optimizer_state"])
    if (load_directory / TX_STRUCT_JSON).exists():
        return True
    if (load_directory / OPTIMIZER_STRUCT_NAME).exists():
        return True
    if (load_directory / "tx").exists():
        return True
    if (load_directory / OPTIMIZER_NAME).exists():
        return True
    return False


def _has_resume_model(load_directory: str | os.PathLike | ePathLike) -> bool:
    """Return whether ``load_state`` should load model weights from ``_resume_model``.

    New merged-LoRA checkpoints record this explicitly in ``metadata.json`` so
    model-only resumes can still restore the original LoRA graph while stale
    directories can opt out cleanly. Older checkpoints fall back to the
    historical rule: only prefer ``_resume_model`` when optimizer state is
    present at the checkpoint root.
    """
    load_directory = ePath(load_directory)
    metadata = _read_checkpoint_metadata(load_directory)
    if "has_resume_model" in metadata:
        return bool(metadata["has_resume_model"])
    return (load_directory / RESUME_MODEL_SUBDIR).exists() and _has_saved_optimizer_state(load_directory)


def _get_checkpoint_step(load_directory: str | os.PathLike | ePathLike) -> int | None:
    """Return the recorded checkpoint step from ``metadata.json`` when available."""
    metadata = _read_checkpoint_metadata(load_directory)
    step = metadata.get("step")
    if step is None:
        return None
    try:
        return int(step)
    except (TypeError, ValueError):
        logger.debug("Ignoring non-integer checkpoint step %r from %s.", step, load_directory)
        return None


def _is_optimizer_template_incompatibility(exc: Exception) -> bool:
    """Return whether an optimizer restore error signals a template mismatch."""
    message = str(exc)
    return (isinstance(exc, KeyError) and "Missing array for key" in message) or (
        isinstance(exc, ValueError) and "Array shape mismatch for key" in message
    )


class EasyDeLState(_PyTreeNode):
    """Complete state container for EasyDeL models during training or inference.

    EasyDeLState encapsulates all stateful components needed for training or inference,
    including model parameters, optimizer state, and training metadata. It provides
    methods for gradient updates, checkpointing, and state management while integrating
    seamlessly with JAX's functional programming paradigm.

    This class is implemented as a SpecTrax struct PyTreeNode, making it compatible with
    JAX transformations like `jit`, `grad`, `vmap`, and `pmap`. The state is immutable;
    all methods that modify state return new instances.

    Attributes:
        step (int | jax.Array): Current training step count. Incremented automatically
            when `apply_gradients` is called. Can be an integer or a JAX array for
            device placement.
        graphdef (spx.GraphDef): The model's computation graph definition. This is a
            non-pytree node that defines the structure of the neural network without
            containing any parameter values.
        graphstate (spx.State): The model's parameter state as a pytree. Contains
            all trainable parameters (spx.Parameter) extracted from the model.
        graphother (spx.State): Non-parameter model state as a pytree. Contains
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
        esurge_cache_scope_key (str): Unique key used to scope eSurge compiled caches
            to this state instance. Auto-generated from a UUID on creation.

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

    step: int | jax.Array = _field(pytree_node=True)
    graphdef: spx.GraphDef = _field(pytree_node=False)

    graphstate: spx.State = _field(pytree_node=True)
    graphother: spx.State = _field(pytree_node=True)

    tx: optax.GradientTransformation | None = _field(pytree_node=False)
    opt_state: optax.OptState | None = _field(pytree_node=True)
    apply_fn: tp.Callable | None = _field(pytree_node=False, default=None)
    esurge_cache_scope_key: str = _field(
        pytree_node=False,
        default_factory=lambda: f"state-{uuid.uuid4().hex}",
    )

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
        if self.opt_state is None:
            raise RuntimeError("Optimizer state is not initialized. Call `init_tx()` first.")
        if self.tx is None:
            raise RuntimeError("Optimizer (tx) is not set. Call `init_tx()` first.")

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
        graphdef: spx.GraphDef | None = None,
        graphstate: spx.State | None = None,
        graphother: spx.State | None = None,
        model: spx.Module | None = None,
        trainable_selector: spx.SelectorSugar | None = None,
        tx: optax.GradientTransformation | None = None,
        opt_state: optax.OptState | None = None,
        init_opt_state: bool = False,
    ) -> Self:
        """Create a new EasyDeLState instance.

        Factory method that provides flexible initialization of the state, either from
        an existing `spx.Module` or by providing the graph components (`graphdef`,
        `graphstate`, `graphother`) directly. It also handles optimizer state
        initialization when requested.

        This method enforces mutual exclusivity between providing a model and providing
        graph components directly, ensuring clear and unambiguous state initialization.

        Args:
            step (int | None): The initial training step count. Defaults to 0 if not
                provided. Can be set to a higher value when resuming training.
            graphdef (spx.GraphDef | None): The model's graph definition. Must be
                provided together with `graphstate` and `graphother` if not using
                `model`. Defaults to None.
            graphstate (spx.State | None): The model's parameter state pytree.
                Must be provided together with `graphdef` and `graphother` if not
                using `model`. Defaults to None.
            graphother (spx.State | None): The model's non-parameter state pytree.
                Must be provided together with `graphdef` and `graphstate` if not
                using `model`. Defaults to None.
            model (spx.Module | None): An EasyDeL module instance. If provided,
                `graphdef`, `graphstate`, and `graphother` are automatically extracted
                using `spx.export`. Cannot be provided simultaneously with graph
                components. Defaults to None.
            trainable_selector (spx.SelectorSugar | None): Selector describing which
                collections belong in `graphstate` when `model` is provided. Defaults
                to the model's `trainable_selector`, which is `"parameters"`
                for EasyDeL modules.
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

                >>> gdef, gstate, gother = model.split_module()
                >>> state = EasyDeLState.create(
                ...     graphdef=gdef,
                ...     graphstate=gstate,
                ...     graphother=gother,
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
            When using `model`, the function defers to the model's own split
            logic (``model.split_module(trainable_selector=...)`` when available)
            so the module controls how `graphstate` is selected. Plain SPX
            modules fall back to selector-based partitioning over ``spx.export``.

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
            if hasattr(model, "split_module"):
                if trainable_selector is None:
                    graphdef, graphstate, graphother = model.split_module()
                else:
                    graphdef, graphstate, graphother = model.split_module(trainable_selector=trainable_selector)
            else:
                gdef, state = spx.export(model)
                graphdef = gdef
                selector = trainable_selector
                if selector is None:
                    selector = getattr(model, "trainable_selector", "parameters")
                graphstate, graphother = spx.as_selector(selector).partition_state(model, state)
        else:
            if trainable_selector is not None:
                raise ValueError("`trainable_selector` can only be used when constructing state from `model`.")
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

    def _optimizer_partition_specs(
        self,
        opt_state: tp.Any,
        *,
        mesh: MeshLike | None = None,
    ) -> tp.Any:
        """Derive optimizer-state sharding from parameter metadata when possible."""
        mesh = mesh or self.model._get_mesh(None)

        if self.tx is not None and hasattr(optax, "tree_map_params"):
            try:
                param_partition_specs = self.model.resolve_sharding_for_tree(self.graphstate)
                partition_specs = optax.tree_map_params(
                    self.tx,
                    lambda _param, spec: spec,
                    opt_state,
                    param_partition_specs,
                    transform_non_params=lambda _: PartitionSpec(),
                )
            except Exception:
                partition_specs = spx.match_partition_rules(((".*", PartitionSpec()),), opt_state)
        else:
            partition_specs = spx.match_partition_rules(((".*", PartitionSpec()),), opt_state)

        partition_specs, adjusted = sanitize_partition_specs_for_shape_tree(
            partition_specs=partition_specs,
            shape_tree=opt_state,
            mesh=mesh,
        )
        if adjusted:
            logger.warning("Adjusted %d non-divisible optimizer sharding specs.", adjusted)
        return partition_specs

    @staticmethod
    def _materialized_tree_partition_specs(tree: tp.Any) -> tp.Any:
        tree = materialize_meta_leaves(tree, seed=42)
        return jax.tree_util.tree_map(
            lambda leaf: PartitionSpec() if hasattr(leaf, "shape") else None,
            tree,
            is_leaf=lambda x: x is None,
        )

    def init_tx(self: Self, tx: optax.GradientTransformation) -> Self:
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

        Note:
            This method requires the model to have a valid mesh configuration.
            The sharding is computed using `jax.eval_shape` to avoid materializing
            the full optimizer state before sharding decisions are made.

        See Also:
            - :meth:`create`: Create state with optimizer initialization.
            - :meth:`shard_optimizer_state`: Shard existing optimizer state.
        """
        # # @erfanzar NOTE: use spectrax's MPMD-aware sharding APIs end-to-end.
        # ``spx.extract_sharding_structure(tree, mesh=mesh)`` is the MPMD-aware
        # reader: it sanitizes each leaf's NamedSharding against the resolved
        # stage-local submesh and strips the pipeline axis from the spec, so
        # PP per-stage placements are preserved correctly.  Pair that with
        # ``optax.tree_map_params`` to mirror per-param NamedShardings onto
        # their opt-state slots, then materialise via the spectrax-canonical
        # shard fns from ``make_shard_and_gather_fns`` -- those are per-leaf
        # ``jax.device_put`` closures that handle shape-sanitization and work
        # for multi-submesh placements (which a single ``jit`` cannot, because
        # ``jax.jit`` rejects multi-mesh ``out_shardings`` and ``sxjit``
        # requires ``sxstage_iter`` markers that an init function does not
        # have).
        mesh = self.model._get_mesh(None)
        replicated = replicated_named_sharding(mesh)

        # 1. Per-leaf NamedShardings -- MPMD-aware (per-stage submesh preserved).
        param_shardings = spx.extract_sharding_structure(self.graphstate, mesh=mesh)

        eval_opt_state = jax.eval_shape(lambda: tx.init(self.graphstate))

        # 2. Mirror each param's NamedSharding onto its matching opt-state slot.
        if hasattr(optax, "tree_map_params"):
            out_shardings = optax.tree_map_params(
                tx,
                lambda _param, ns: ns if isinstance(ns, jax.sharding.NamedSharding) else replicated,
                eval_opt_state,
                param_shardings,
                transform_non_params=lambda _: replicated,
                is_leaf=lambda x: isinstance(x, jax.sharding.NamedSharding) or x is None,
            )
        else:
            out_shardings = eval_opt_state

        # 3. Materialise via per-leaf device_put against the resolved
        #    NamedShardings.  Equivalent to what ``make_shard_and_gather_fns``
        #    does internally, but we already have NamedShardings so we don't
        #    need to round-trip through PartitionSpec.
        opt_state_unplaced = tx.init(self.graphstate)
        opt_state = jax.tree_util.tree_map(
            lambda leaf, ns: (
                jax.device_put(leaf, ns)
                if isinstance(ns, jax.sharding.NamedSharding) and hasattr(leaf, "shape")
                else leaf
            ),
            opt_state_unplaced,
            out_shardings,
            is_leaf=lambda x: isinstance(x, jax.sharding.NamedSharding) or x is None,
        )

        return self.replace(tx=tx, opt_state=opt_state)

    def shard_optimizer_state(
        self,
        opt_state: tp.Any | None = None,
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

        See Also:
            - :meth:`gather_optimizer_state`: Reverse operation to gather state.
            - :meth:`init_tx`: Initialize optimizer with automatic sharding.
        """
        if opt_state is None and self.opt_state is None:
            raise ValueError("Optimizer state is not initialized.")
        if opt_state is None:
            opt_state = self.opt_state
        mesh = self.model._get_mesh(None)
        partition_specs = self._optimizer_partition_specs(
            opt_state,
            mesh=mesh,
        )
        shard_fns, _ = make_shard_and_gather_fns(partition_specs=partition_specs, mesh=mesh)
        opt_state = jax.tree_util.tree_map(lambda f, o: f(o), shard_fns, opt_state)
        return self.replace(opt_state=opt_state)

    def gather_optimizer_state(self: Self) -> Self:
        """Gather the optimizer state from distributed devices to a single device.

        Reverses the sharding operation by collecting all shards of the optimizer
        state and reassembling them into complete tensors. This is typically needed
        before saving checkpoints in a portable format or when transitioning from
        distributed to single-device execution.

        Args:
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
        if self.opt_state is None:
            raise RuntimeError("Optimizer state is not initialized.")
        mesh = self.model._get_mesh(None)
        partition_specs = self._optimizer_partition_specs(
            self.opt_state,
            mesh=mesh,
        )
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
            tree: The pytree (typically `spx.State`) containing the parameters
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
        other = jax.tree_util.tree_map(
            lambda x: jax.lax.stop_gradient(x) if hasattr(x, "shape") else x,
            self.graphother,
        )
        full_state = tree.merge(other, copy=False)
        return spx.bind(self.graphdef, full_state)

    def merge_to_state(self: Self, tree) -> Self:
        """Create a new state with updated parameters.

        Creates a new `EasyDeLState` by replacing the current `graphstate` with the
        provided tree while keeping all other state components unchanged.

        Args:
            tree: The pytree (typically `spx.State`) containing the new parameters.
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
        full_state = self.graphstate.merge(self.graphother, copy=False)
        model = spx.bind(self.graphdef, full_state)
        # `Module.__setattr__` bumps the graph-structure epoch even on
        # underscore-prefixed attrs, which fails inside any spx.jit trace
        # (IllegalMutationError). The scope key is metadata only — bypass
        # the spectrax setattr path so accessing `state.model` is pure.
        object.__setattr__(model, "_esurge_cache_scope_key", self.esurge_cache_scope_key)
        # TODO: Make me Dynamic.
        return model

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
                base_path=str(load_directory),
                save_interval=None,
                step_policies=[],
            )

        tx_template = tx_template if tx_template is not None else self.tx

        def _load_tensorstore(template):
            return checkpointer.load_pytree(
                mesh=self.model.mesh,
                path=str(load_directory),
                prefix="tx",
                load_treedef=True,
                discover_latest=False,
                template=template,
                sharding_rules=self.model.resolve_shardings_regex(),
            )

        template = None
        if tx_template is not None:
            try:
                template = jax.eval_shape(tx_template.init, self.graphstate)
            except Exception:
                logger.warning(
                    "Failed to build an optimizer template for TensorStore restore; "
                    "retrying using the saved optimizer structure.",
                    exc_info=True,
                )

        try:
            opt_state, metadata = _load_tensorstore(template)
        except KeyError as exc:
            if template is not None and "Missing array for key" in str(exc):
                logger.error(
                    "Optimizer checkpoint is incompatible with the current optimizer template.",
                    exc_info=True,
                )
            raise
        step = metadata.get("step", 0)
        logger.info(f"Optimizer state loaded from {load_directory} (step {step}).")
        return self.replace(opt_state=opt_state, step=jnp.asarray(step))

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
            if not is_remote_path(save_directory) or jax.process_index() == 0:
                save_directory.mkdir(parents=True, exist_ok=True)
            optim_path = save_directory
            logger.info(f"Coordinated optimizer save through {optim_path}")
            try:
                with self.model.mesh:
                    checkpointer.save_pytree(
                        tree=self.opt_state,
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
        merge_lora_before_save: bool = False,
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
            merge_lora_before_save (bool): If True and the model currently has
                LoRA adapters attached, saves a merged model export at the
                checkpoint root. The original LoRA-wrapped model is also written
                to ``{save_directory}/_resume_model`` so :meth:`load_state` can
                restore the original training graph for both full and
                weights-only resume flows.
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

            Save a merged LoRA export::

                >>> state.save_state(
                ...     "checkpoints/merged_export",
                ...     merge_lora_before_save=True
                ... )

        Note:
            The saved checkpoint directory will contain:
            - Model configuration (config.json)
            - Model parameters (easydel-model.parameters or TensorStore)
            - Optimizer state if `save_optimizer=True` (TensorStore format)
            - ``_resume_model/`` when saving a merged LoRA checkpoint,
              containing the original unmerged model tree for checkpoint resume
            - The current model tree exactly as stored in the state; call
              ``state.gather_model()`` first if you need gathered weights

        See Also:
            - :meth:`load_state`: Load complete state from checkpoint.
            - :meth:`save_optimizer`: Save only optimizer state.
        """
        save_directory = ePath(save_directory)
        if step is None:
            step = self.step.item() if isinstance(self.step, jnp.ndarray) else self.step
        should_save_merged_lora = merge_lora_before_save and getattr(self.model, "lora_is_enabled", False)
        has_resumable_optimizer = save_optimizer and getattr(self, "opt_state", None) is not None
        resume_model_to_save = self.model if should_save_merged_lora else None
        if save_optimizer:
            self.save_optimizer(save_directory=save_directory, float_dtype=float_dtype, step=step)
        else:
            logger.info("Skipping optimizer saving as requested.")

        if resume_model_to_save is not None:
            resume_directory = save_directory / RESUME_MODEL_SUBDIR
            logger.info(
                "Saving merged LoRA checkpoint to %s with a resume-safe LoRA copy in %s.",
                save_directory,
                resume_directory,
            )
            resume_model_to_save.save_pretrained(
                save_directory=resume_directory,
                float_dtype=float_dtype,
                step=step,
            )

        model_to_save = self.model
        if should_save_merged_lora:
            # Work on a detached copy so export-time LoRA unwrapping does not
            # mutate the live training state.
            model_to_save = deepcopy_model(self.model)
            model_to_save.unwrap_lora_to_layers(verbose=False)

        model_to_save.save_pretrained(
            save_directory=save_directory,
            float_dtype=float_dtype,
            step=step,
        )
        try:
            if jax.process_index() == 0:
                metadata = {
                    "step": int(step),
                    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
                    "is_temporary": False,
                    "has_optimizer_state": bool(has_resumable_optimizer),
                    "has_resume_model": bool(resume_model_to_save is not None),
                }
                (save_directory / "metadata.json").write_text(json.dumps(metadata))
        except Exception as exc:
            logger.warning(f"Failed to write checkpoint discovery metadata for {save_directory}: {exc}")

    @classmethod
    def load_state(
        cls,
        load_directory: str | os.PathLike,
        device: jax.Device | None = "cpu",  # type:ignore
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision | None = None,
        sharding_axis_dims: collections.abc.Sequence[int] = (1, 1, -1, 1, 1, 1),
        sharding_dcn_axis_dims: collections.abc.Sequence[int] | None = None,
        sharding_axis_names: collections.abc.Sequence[str] = ("pp", "dp", "fsdp", "ep", "tp", "sp"),
        partition_axis: PartitionAxis | None = None,
        shard_fns: collections.abc.Mapping[tuple, tp.Callable] | dict | None = None,
        backend: EasyDeLBackends | None = None,
        platform: EasyDeLPlatforms | None = None,
        config_kwargs: EasyDeLBaseConfigDict | None = None,
        model_task: TaskType = TaskType.AUTO_BIND,
        auto_shard_model: bool = True,
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
                available devices. Defaults to (1, 1, -1, 1, 1, 1).
            sharding_dcn_axis_dims (collections.abc.Sequence[int] | None): Optional dimensions for
                data-center network (DCN) sharding in multi-host setups. Defaults to
                None.
            sharding_axis_names (collections.abc.Sequence[str]): Names for sharding axes matching
                `sharding_axis_dims`. Defaults to ("pp", "dp", "fsdp", "ep", "tp", "sp").
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
                ...     sharding_axis_dims=(1, 1, 4, 1, 2, 1),  # 8 devices
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
              in the checkpoint directory. If the checkpoint metadata declares
              ``_resume_model/`` as the state-load source, the model/config are
              loaded from that subdirectory so merged LoRA checkpoints can
              still restore the original training graph while leaving a merged
              model at the checkpoint root for direct inference loading.
            - Optimizer state loading is only attempted when the checkpoint
              metadata or legacy artifacts indicate optimizer state is present.
              Failures are logged as info (not errors), allowing weights-only
              resume or inference-style usage.
            - When `auto_shard_model=True`, both model and optimizer state are
              sharded according to the specified rules.

        See Also:
            - :meth:`save_state`: Save state to checkpoint.
            - :meth:`create`: Create state from an existing model.
        """
        from easydel.modules.auto.auto_configuration import AutoEasyDeLConfig

        from .base_module import EasyDeLBaseModule

        load_directory = ePath(load_directory)
        checkpoint_step = _get_checkpoint_step(load_directory)
        has_optimizer_state = _has_saved_optimizer_state(load_directory)
        has_resume_model = _has_resume_model(load_directory)
        model_load_directory = load_directory
        resume_model_directory = load_directory / RESUME_MODEL_SUBDIR
        if resume_model_directory.exists() and has_resume_model:
            model_load_directory = resume_model_directory
        elif resume_model_directory.exists():
            logger.info(
                "Ignoring %s because checkpoint %s does not declare it as the state-load source; loading merged root model.",
                resume_model_directory,
                load_directory,
            )
        if not model_load_directory.exists():
            model_load_directory = load_directory

        config = AutoEasyDeLConfig.from_pretrained(
            model_load_directory,
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
            pretrained_model_name_or_path=model_load_directory,
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
            quantization_config=quantization_config,
            apply_quantization=apply_quantization,
            verbose=verbose,
            **kwargs,
        )

        state = cls.create(
            step=jnp.asarray(0 if checkpoint_step is None else checkpoint_step, dtype=jnp.int32),
            model=model,
        )
        if has_optimizer_state:
            try:
                cmg = jax.default_device(device) if device is not None else contextlib.nullcontext()
                with cmg:
                    state = state.load_optimizer(load_directory=load_directory, tx_template=tx_template)
            except Exception:
                logger.info("EasyDeLState couldn't restore the optimizer/tx (maybe there wasn't any!).")
        else:
            logger.info("Checkpoint %s does not contain optimizer state; skipping optimizer restore.", load_directory)
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
        self = jax.tree_util.tree_map(
            device_put_or_shard_abstract,
            self,
            shape,
        )
        return self

    def shard_state(
        self,
        mesh: MeshLike | None = None,
    ) -> Self:
        """Shard the entire state across the device mesh.

        Applies sharding to both model parameters and optimizer state according to
        variable metadata-derived partition specs and the device mesh. This is the
        primary method for distributing state across multiple devices for data/model
        parallelism.

        Args:
            mesh (Mesh, optional): The JAX device mesh to shard across. Defines the
                topology of devices (e.g., 2x4 grid for 8 devices). If None, uses
                the model's configured mesh. Defaults to None.

        Returns:
            Self: A new EasyDeLState instance with all components (graphstate,
            graphother, opt_state) sharded across the device mesh.

        Example:
            Shard with default rules::

                >>> sharded_state = state.shard_state()

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
        # # @erfanzar NOTE: rewritten to use spectrax's MPMD-aware sharding APIs
        # end-to-end.  Previously this routed each leaf through
        # ``_device_put_with_partition_spec`` (PartitionSpec -> NamedSharding
        # bound to the *full* mesh -> device_put), which collapsed per-stage
        # submeshes the same way ``init_tx`` used to.  Now:
        #   * ``graphstate`` keeps the variable-aware
        #     ``_apply_partition_specs_to_state`` path (already MPMD-correct
        #     via ``named_sharding_for_variable``).
        #   * ``graphother`` reads its existing per-leaf NamedShardings via
        #     ``spx.extract_sharding_structure`` (per-stage submesh preserved).
        #   * ``opt_state`` mirrors per-param NamedShardings onto matching
        #     mu/nu slots via ``optax.tree_map_params`` -- same auto path as
        #     ``init_tx``.
        mesh = mesh or self.model._get_mesh(None)
        replicated = replicated_named_sharding(mesh)

        step = self.step
        if not isinstance(step, jax.Array):
            step = jnp.asarray(step, dtype=jnp.int32)

        graphstate = self.model.apply_sharding_for_tree(self.graphstate)
        graphother = self.model.apply_sharding_for_tree(self.graphother)

        opt_state = self.opt_state
        if opt_state is not None and self.tx is not None and hasattr(optax, "tree_map_params"):
            param_shardings = spx.extract_sharding_structure(graphstate, mesh=mesh)
            opt_shardings = optax.tree_map_params(
                self.tx,
                lambda _p, ns: ns if isinstance(ns, jax.sharding.NamedSharding) else replicated,
                opt_state,
                param_shardings,
                transform_non_params=lambda _: replicated,
                is_leaf=lambda x: isinstance(x, jax.sharding.NamedSharding) or x is None,
            )
            opt_state = jax.tree_util.tree_map(
                lambda leaf, ns: (
                    jax.device_put(leaf, ns)
                    if isinstance(ns, jax.sharding.NamedSharding) and hasattr(leaf, "shape")
                    else leaf
                ),
                opt_state,
                opt_shardings,
                is_leaf=lambda x: isinstance(x, jax.sharding.NamedSharding) or x is None,
            )

        step = jax.device_put(step, replicated)
        return self.replace(step=step, graphstate=graphstate, graphother=graphother, opt_state=opt_state)

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

    def gather_model(self, mesh: MeshLike | None = None) -> Self:
        """Gather model parameters from distributed devices.

        Collects the sharded model parameters (`graphstate` and `graphother`) from
        across the device mesh and reassembles them into complete tensors. This is
        typically needed before saving model weights in a portable format.

        Args:
            mesh (Mesh | None): The JAX device mesh to gather from. If None, uses
                the model's configured mesh. Defaults to None.

        Returns:
            Self: A new EasyDeLState instance with gathered `graphstate` and
            `graphother`.

        Example:
            Gather model for saving::

                >>> gathered_state = state.gather_model()
                >>> # Now graphstate contains complete (non-sharded) parameters

        See Also:
            - :meth:`shard_model`: Reverse operation to shard model.
            - :meth:`gather_state`: Gather entire state including optimizer.
        """
        mesh = mesh or self.model._get_mesh(None)
        graphstate = jax.tree_util.tree_map(lambda o: device_get(o), self.graphstate)

        graphother = materialize_meta_leaves(self.graphother, seed=42)
        graphother_specs = self._materialized_tree_partition_specs(graphother)
        _, graphother_gather = make_shard_and_gather_fns(partition_specs=graphother_specs, mesh=mesh)
        graphother = jax.tree_util.tree_map(lambda f, o: f(o), graphother_gather, graphother)
        self = self.replace(graphstate=graphstate, graphother=graphother)
        return self

    def shard_model(self: Self, mesh: MeshLike | None = None) -> Self:
        """Shard model parameters across the device mesh.

        Distributes the model parameters (`graphstate` and `graphother`) across
        devices according to variable metadata-derived partition specs. This enables
        data and model parallelism for training and inference with large models.

        Args:
            mesh (Mesh | None): The JAX device mesh to shard across. If None, uses
                the model's configured mesh. Defaults to None.

        Returns:
            Self: A new EasyDeLState instance with sharded `graphstate` and
            `graphother`.

        Example:
            Shard model with default rules::

                >>> sharded_state = state.shard_model()

        Note:
            This method only shards model parameters, not optimizer state. Use
            `shard_state` to shard both, or `shard_optimizer_state` for optimizer
            only.

        See Also:
            - :meth:`gather_model`: Reverse operation to gather model.
            - :meth:`shard_state`: Shard entire state.
            - :meth:`shard_optimizer_state`: Shard optimizer state.
        """
        mesh = mesh or self.model._get_mesh(None)
        graphstate = self.model.apply_sharding_for_tree(self.graphstate)
        graphother = self.model.apply_sharding_for_tree(self.graphother)

        graphother = materialize_meta_leaves(self.graphother, seed=42)
        graphother_specs = self._materialized_tree_partition_specs(graphother)
        shard_fns, _ = make_shard_and_gather_fns(graphother_specs, mesh)
        graphother = jax.tree_util.tree_map(lambda f, o: f(o), shard_fns, graphother)

        self = self.replace(graphstate=graphstate, graphother=graphother)
        return self

    @property
    def mesh(self) -> spx.SpxMesh:
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
        return jax.tree_util.tree_map(
            lambda x: x.sharding if hasattr(x, "sharding") else None,
            self,
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
