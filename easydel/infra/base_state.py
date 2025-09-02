# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

Classes:
    EasyDeLState: Complete state container for models during training/inference

Key Features:
    - Unified state management for training and inference
    - Automatic sharding and partitioning support
    - Checkpoint saving and loading
    - Gradient application with optimizer integration
    - State serialization and deserialization

Example:
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
    >>> # Apply gradients
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
"""

from __future__ import annotations

import contextlib
import os
import pickle
import typing as tp
from typing import Self

import jax
import optax
from eformer import escale as es
from eformer.escale import PartitionAxis
from eformer.loggings import get_logger
from eformer.paths import ePath, ePathLike
from eformer.serialization import AsyncCheckpointManager
from flax import nnx as nn
from flax import struct
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.factory import TaskType
from easydel.utils.compiling_utils import ejit
from easydel.utils.traversals import specs_to_name_sharding

if tp.TYPE_CHECKING:
    from jax.sharding import Mesh

    from easydel.infra.base_config import EasyDeLBaseConfigDict
    from easydel.infra.etils import EasyDeLBackends, EasyDeLPlatforms, EasyDeLQuantizationMethods

    from .base_module import EasyDeLBaseModule, PartitionLike

WEIGHTS_NAME = "easydel-model.parameters"
OPTIMIZER_NAME = "easydel-optstate.parameters"
OPTIMIZER_STRUCT_NAME = "easydel-optstate.structure"
TX_STRUCT_JSON = "tx_structure.json"

logger = get_logger(__name__)


class EasyDeLState(struct.PyTreeNode):
    """Complete state container for EasyDeL models.

    Encapsulates all stateful components needed for training or inference,
    including model parameters, optimizer state, and training metadata.
    Provides methods for gradient updates, checkpointing, and state management.

    This class is designed to work seamlessly with JAX's functional programming
    paradigm while providing convenient methods for state manipulation.

    Attributes:
        step: Current training step count.
        graphdef: Model's computation graph structure (non-pytree).
        graphstate: Model parameter state (pytree).
        graphother: Non-parameter model state (pytree).
        tx: Optimizer transformation (non-pytree).
        opt_state: Optimizer state like moments (pytree).
        apply_fn: Optional model application function (non-pytree).

    Methods:
        apply_gradients: Update parameters with gradients
        create: Factory method to create state from model
        save_state: Save state to checkpoint
        load_state: Load state from checkpoint
        shard_state: Apply sharding to state
        gather_state: Gather sharded state

    Example:
        >>> state = EasyDeLState.create(
        ...     model=my_model,
        ...     tx=optax.adam(1e-3)
        ... )
        >>> # Training step
        >>> grads = compute_gradients(...)
        >>> state = state.apply_gradients(grads=grads)
    """

    step: int | jax.Array = struct.field(pytree_node=True)
    graphdef: nn.GraphDef = struct.field(pytree_node=False)

    graphstate: nn.GraphState = struct.field(pytree_node=True)
    graphother: nn.GraphState = struct.field(pytree_node=True)

    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState | None = struct.field(pytree_node=True)
    apply_fn: tp.Callable | None = struct.field(pytree_node=False, default=None)

    def apply_gradients(self: Self, *, grads):
        """Apply gradients to update parameters and optimizer state.

        Performs a single optimization step using the provided gradients.

        Args:
            grads: Gradient pytree matching the structure of graphstate.

        Returns:
            New EasyDeLState with updated parameters, optimizer state,
            and incremented step count.

        Raises:
            AssertionError: If opt_state or tx is not initialized.

        Example:
            >>> grads = jax.grad(loss_fn)(state.graphstate)
            >>> state = state.apply_gradients(grads=grads)
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
        """Creates a new `EasyDeLState` instance.

        This class method provides a flexible way to initialize the state, either from an
        existing `nn.Module` or by providing the graph components (`graphdef`, `graphstate`,
        `graphother`) directly. It also handles optimizer state initialization.

        Args:
            step (tp.Optional[int]):
                The initial training step. Defaults to 0.
            graphdef (tp.Optional[nn.GraphDef]):
                The model's graph definition.
            graphstate (tp.Optional[nn.GraphState]):
                The model's parameter state.
            graphother (tp.Optional[nn.GraphState]):
                The model's non-parameter state.
            model (tp.Optional[nn.Module]):
                An EasyDeL module instance. If provided, `graphdef`, `graphstate`, and `graphother` are derived from it.
                Cannot be provided simultaneously with graph components.
            tx (tp.Optional[optax.GradientTransformation]):
                The optimizer transformation.
            opt_state (tp.Optional[optax.OptState]):
                The initial optimizer state. Cannot be provided if `init_opt_state` is True.
            init_opt_state (bool):
                If True, initializes the optimizer state using `tx.init(graphstate)`.
                Requires `tx` to be provided. Defaults to False.

        Returns:
            EasyDeLState: A new instance of the state.

        Raises:
            ValueError: If `model` and graph components are provided simultaneously.
            ValueError: If graph components are provided partially.
            ValueError: If `init_opt_state` is True and `opt_state` is also provided.
            ValueError: If `init_opt_state` is True but `tx` is not provided.
        """
        graph_params_provided = graphdef is not None or graphstate is not None or graphother is not None
        if model is not None and graph_params_provided:
            raise ValueError(
                "Cannot provide both a model and graph-related parameters. "
                "Choose either model or (graphdef, graphstate)."
            )

        if model is not None:
            graphdef, graphstate, graphother = nn.split(model, nn.Param, ...)

        if graphdef is not None and graphstate is None and graphother is None:
            raise ValueError(
                "When providing graphdef, (graphstate, graphother) must also be provided.",
            )

        if graphstate is not None and graphdef is None and graphother is None:
            raise ValueError(
                "When providing graphstate, (graphdef, graphother) must also be provided.",
            )
        if graphother is not None and graphdef is None and graphstate is None:
            raise ValueError(
                "When providing graphother, (graphstate, graphdef) must also be provided.",
            )
        if init_opt_state and opt_state is not None:
            raise ValueError("When passing `init_opt_state` as `True` you can't also provide `opt_state`")
        if init_opt_state and tx is None:
            raise ValueError("When passing `init_opt_state` as `True` you have to also provide `tx`.")

        if init_opt_state:
            opt_state = tx.init(graphstate)
        if step is None:
            step = 0

        return cls(
            step=step,
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
            tx=tx,
            opt_state=opt_state,
        )

    def init_tx(self: Self, tx: optax.GradientTransformation, partition_rules: PartitionLike = None) -> Self:
        """Initializes the optimizer state (`opt_state`) for the current `graphstate`
        using the provided optimizer transformation (`tx`).

        It automatically handles sharding based on the model's partition rules.

        Args:
            tx (optax.GradientTransformation):
                The optimizer transformation to initialize with.
            partition_rules (PartitionLike, optional):
                Partitioning rules for the optimizer state. If None, uses the rules from the
                associated model's config. Defaults to None.

        Returns:
            EasyDeLState: A new state object with the initialized and potentially
                sharded `opt_state` and the provided `tx`.
        """
        if partition_rules is None:
            partition_rules = self.model.config.get_partition_rules()

        from eformer.escale import match_partition_rules

        def make(graphstate):
            return tx.init(graphstate)

        eval_opt_state = jax.eval_shape(lambda: make(self.graphstate))
        partition_specs = match_partition_rules(partition_rules, eval_opt_state)
        named_shardings = specs_to_name_sharding(partition_specs, self.model.mesh)

        opt_state = ejit(
            make,
            out_shardings=named_shardings,
            in_shardings=(es.extract_shardings(self.graphstate, mesh=self.model.mesh),),
        )(self.graphstate)

        return self.replace(tx=tx, opt_state=opt_state)

    def shard_optimizer_state(
        self,
        opt_state: tp.Any | None = None,
        partition_rules: PartitionLike = None,
    ) -> tp.Any:
        """Applies sharding to the optimizer state based on partition rules.

        Args:
            opt_state (tp.Optional[tp.Any]):
                The optimizer state pytree to shard. If None, uses `self.opt_state`. Defaults to None.
            partition_rules (PartitionLike, optional):
                Partitioning rules. If None, uses rules from the model's config. Defaults to None.

        Returns:
            EasyDeLState: A new state object with the sharded `opt_state`.

        Raises:
            ValueError: If optimizer state is not initialized
                (neither `opt_state` argument nor `self.opt_state` is available).
        """
        if opt_state is None and self.opt_state is None:
            raise ValueError("Optimizer state is not initialized.")
        if opt_state is None:
            opt_state = self.opt_state
        if partition_rules is None:
            partition_rules = self.model.config.get_partition_rules()

        from eformer.escale import make_shard_and_gather_fns, match_partition_rules

        with self.model.mesh:
            partition_specs = match_partition_rules(partition_rules, opt_state)
            shard_fns, _ = make_shard_and_gather_fns(partition_specs)
            opt_state = jax.tree_util.tree_map(lambda f, o: f(o), shard_fns, opt_state)
            return self.replace(opt_state=opt_state)

    def gather_optimizer_state(self: Self, partition_rules=None):
        """Gathers the optimizer state from potentially distributed devices.

        Args:
            partition_rules (PartitionLike, optional):
                Partitioning rules used to determine how the state was sharded. If None,
                uses rules from the model's config. Defaults to None.

        Returns:
            EasyDeLState: A new state object with the gathered `opt_state`.

        Raises:
            AssertionError: If `opt_state` is not initialized.
        """
        assert self.opt_state is not None, "Optimizer state is not initialized."
        if partition_rules is None:
            partition_rules = self.model.config.get_partition_rules()

        from eformer.escale import make_shard_and_gather_fns, match_partition_rules

        partition_specs = match_partition_rules(partition_rules, self.opt_state)
        _, gather = make_shard_and_gather_fns(partition_specs)
        self = self.replace(opt_state=jax.tree_util.tree_map(lambda f, o: f(o), gather, self.opt_state))
        return self

    def merge(self: Self, tree) -> EasyDeLBaseModule:
        """Merges a given state tree (usually parameters) with the graph definition and other state
        components to reconstruct the full model module.

        Args:
            tree: The pytree (e.g., `nn.GraphState`) containing the parameters to merge.

        Returns:
            EasyDeLBaseModule: The reconstructed model module.
        """
        return nn.merge(self.graphdef, tree, self.graphother)

    def merge_to_state(self: Self, tree) -> Self:
        """Creates a new `EasyDeLState` by replacing the current `graphstate` with the provided tree.

        Args:
            tree: The pytree (e.g., `nn.GraphState`) containing the new parameters.

        Returns:
            EasyDeLState: A new state object with the updated `graphstate`.
        """
        return self.replace(graphstate=tree)

    @property
    def model(self) -> EasyDeLBaseModule:
        """Reconstructs and returns the full EasyDeL model module from the state components.

        Returns:
            EasyDeLBaseModule: The model module instance.
        """
        return nn.merge(self.graphdef, self.graphstate, self.graphother)

    @property
    def size(self) -> int:
        """Calculates the total size in bytes of the model parameters (`graphstate`)
        and the optimizer state (`opt_state`).

        Returns:
            int: The total size in bytes.
        """

        def calculate_size(pytree):
            if pytree is None:
                return 0
            leaves, _ = jax.tree_util.tree_flatten(pytree)
            return sum(leaf.size * leaf.itemsize for leaf in leaves if isinstance(leaf, jnp.ndarray))

        opt_state_size = calculate_size(self.opt_state)
        graphstate_size = calculate_size(self.graphstate)
        return opt_state_size + graphstate_size

    def load_optimizer(self: Self, load_directory: str | ePathLike):
        """Loads the optimizer state from saved files.

        Reads the optimizer state structure from a pickle file (`OPTIMIZER_STRUCT_NAME`)
        and the tensor data from a SafeTensors file (`OPTIMIZER_NAME`) within the
        specified directory.

        Args:
            load_directory (tp.Union[str, os.PathLike]):
                The directory containing the saved optimizer state files.

        Returns:
            EasyDeLState: A new state object with the loaded `opt_state`.

        Raises:
            FileNotFoundError: If the required optimizer files are not found.
            Exception: If any error occurs during loading or deserialization.
        """
        load_directory = ePath(load_directory)
        optim_path = (
            load_directory if AsyncCheckpointManager.is_tensorstore(load_directory) else load_directory / OPTIMIZER_NAME
        )
        struct_path = load_directory / OPTIMIZER_STRUCT_NAME
        tx_struct_path = load_directory / TX_STRUCT_JSON
        if not struct_path.exists() and not tx_struct_path.exists():
            raise FileNotFoundError(f"Optimizer files are missing at {load_directory}")
        try:
            if not AsyncCheckpointManager.is_tensorstore(optim_path):
                treedef, step = pickle.loads(struct_path.read_bytes())
                leaves, _ = AsyncCheckpointManager().load(
                    path=AsyncCheckpointManager.safe_loadpath(optim_path),
                    mesh=self.model.mesh,
                    partition_rules=self.model.config.get_partition_rules(),
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
                opt_state, metadata = AsyncCheckpointManager().load_pytree(
                    path=AsyncCheckpointManager.safe_loadpath(optim_path),
                    mesh=self.model.mesh,
                    partition_rules=self.model.config.get_partition_rules(),
                    prefix="tx",
                )
                step = metadata.get("step", 0)

            logger.info(f"Optimizer state loaded from {load_directory} (step {step}).")
            return self.replace(opt_state=opt_state, step=jnp.asarray(step))
        except Exception as e:
            logger.error(f"Optimizer load failed: {e!s}")
            raise e

    def save_optimizer(self, save_directory: str | ePathLike, float_dtype: jnp.dtype | None = None):
        save_directory = ePath(save_directory)
        if self.opt_state is not None:
            save_directory.mkdir(parents=True, exist_ok=True)
            optim_path = save_directory
            logger.info(f"Coordinated optimizer save through {optim_path}")
            try:
                with self.model.mesh:
                    AsyncCheckpointManager().save_pytree(
                        pytree=self.opt_state,
                        path=optim_path,
                        mesh=self.model.mesh,
                        dtype=float_dtype,
                        prefix="tx",
                        extras={"step": self.step.item() if isinstance(self.step, jnp.ndarray) else self.step},
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
    ):
        """Saves the entire `EasyDeLState` to a directory.

        This includes saving the model parameters (using `model.save_pretrained`)
        and optionally the optimizer state.

        Args:
            save_directory (tp.Union[str, os.PathLike]):
                The directory to save the state to.
            float_dtype (tp.Optional[jnp.dtype]):
                Optional dtype to cast floating-point parameters to before saving. Defaults to None.
            verbose (bool):
                If True, logs information during saving. Defaults to True.
            mismatch_allowed (bool):
                Passed to `model.save_pretrained`, allows saving even if the model structure differs
                slightly from expected. Defaults to True.
            save_optimizer (bool):
                If True, saves the optimizer state. Defaults to True.
        """
        save_directory = ePath(save_directory)
        if save_optimizer:
            self.save_optimizer(save_directory=save_directory, float_dtype=float_dtype)
        else:
            logger.info("Skipping optimizer saving as requested.")

        self.model.save_pretrained(
            save_directory=save_directory,
            gather_fns=self.model._gather_fns,
            float_dtype=float_dtype,
        )

    @classmethod
    def load_state(
        cls,
        load_directory: str | os.PathLike,
        device: jax.Device | None = "cpu",  # type:ignore
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision | None = None,
        sharding_axis_dims: tp.Sequence[int] = (1, -1, 1, 1, 1),
        sharding_dcn_axis_dims: tp.Sequence[int] | None = None,
        sharding_axis_names: tp.Sequence[str] = ("dp", "fsdp", "ep", "tp", "sp"),
        partition_axis: PartitionAxis | None = None,
        shard_attention_computation: bool = True,
        shard_fns: tp.Mapping[tuple, tp.Callable] | dict | None = None,
        backend: EasyDeLBackends | None = None,
        platform: EasyDeLPlatforms | None = None,
        config_kwargs: EasyDeLBaseConfigDict | None = None,
        model_task: TaskType = TaskType.AUTO_BIND,
        auto_shard_model: bool = True,
        partition_rules: tuple[tuple[str, PartitionSpec], ...] | None = None,
        quantization_platform: EasyDeLPlatforms | None = None,
        quantization_method: EasyDeLQuantizationMethods | None = None,
        quantization_block_size: int = 128,
        quantization_pattern: str | None = None,
        quantize_tensors: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        """Loads an EasyDeLState from a saved checkpoint directory.

        This class method reconstructs the model configuration, loads the model
        parameters, and optionally loads the optimizer state from files saved
        previously using `save_state`. It handles various configurations for
        device placement, data types, sharding, and quantization.

        Args:
            load_directory:
                Path to the directory containing the saved state
                (configuration, model weights, and potentially optimizer state).
            device:
                The JAX device (e.g., 'cpu', 'gpu', 'tpu') to load the model onto. Defaults to 'cpu'.
            dtype:
                The data type to use for computation (e.g., jnp.bfloat16). Defaults to jnp.bfloat16.
            param_dtype:
                The data type for the model parameters (e.g., jnp.bfloat16). Defaults to jnp.bfloat16.
            precision:
                The JAX precision level (e.g., jax.lax.Precision.HIGHEST). Defaults to None.
            sharding_axis_dims:
                A sequence defining the dimensions of the device mesh for sharding
                (e.g., (1, -1, 1, 1, 1)). Defaults to (1, -1, 1, 1, 1).
            sharding_dcn_axis_dims:
                Optional sequence for data-centric sharding dimensions. Defaults to None.
            sharding_axis_names:
                Names corresponding to the sharding axes (e.g., ("dp", "fsdp",  "ep", "tp", "sp")).
                Defaults to ("dp", "fsdp",  "ep", "tp", "sp").
            partition_axis:
                Configuration object for partitioning specific axes. Defaults to None.
            shard_attention_computation:
                If True, shards the attention computation across devices. Defaults to True.
            shard_fns:
                Optional mapping of parameter path tuples to custom sharding functions. Defaults to None.
            backend:
                The backend framework to use (e.g., EasyDeLBackends.JAX). Defaults to None (auto-detected).
            platform:
                The hardware platform (e.g., EasyDeLPlatforms.TPU). Defaults to None (auto-detected).
            config_kwargs:
                Optional dictionary of keyword arguments to override in the loaded model configuration. Defaults to None.
            model_task:
                The specific task type for the model (e.g., TaskType.CAUSAL_LM). Defaults to TaskType.AUTO_BIND.
            auto_shard_model:
                If True, automatically shards the loaded model and optimizer state based on the
                provided sharding configuration. Defaults to False.
            partition_rules:
                Optional tuple of partition rules (regex, PartitionSpec) to explicitly define sharding.
                Defaults to None (uses model config).
            quantization_platform:
                Platform for quantization (e.g., EasyDeLPlatforms.TPU). Defaults to None.
            quantization_method:
                Quantization method (e.g., EasyDeLQuantizationMethods.AQT). Defaults to None.
            quantization_block_size:
                Block size for quantization methods like GPTQ. Defaults to 128.
            quantization_pattern:
                Regex pattern to match tensor names for quantization. Defaults to None.
            quantize_tensors:
                If True, applies quantization to the loaded tensors. Defaults to True.
            verbose:
                If True, logs detailed information during loading. Defaults to True.
            **kwargs:
                Additional keyword arguments passed directly to the underlying
                `EasyDeLBaseModule.from_pretrained` method.

        Returns:
            An EasyDeLState instance containing the loaded model, optimizer state
            (if found and loaded), and associated configuration.

        Raises:
            FileNotFoundError: If the `load_directory` or essential files within it
                (like configuration or model weights) are not found.
            ValueError: If there are inconsistencies in the provided arguments or
                loaded configuration.
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
            shard_attention_computation=shard_attention_computation,
            shard_fns=shard_fns,
            backend=backend,
            platform=platform,
            config_kwargs=config_kwargs,
            auto_shard_model=auto_shard_model,
            partition_rules=partition_rules,
            quantization_platform=quantization_platform,
            quantization_method=quantization_method,
            quantization_block_size=quantization_block_size,
            quantization_pattern=quantization_pattern,
            quantize_tensors=quantize_tensors,
            verbose=verbose,
            **kwargs,
        )

        state = cls.create(step=jnp.array(0), model=model)
        optimizer_path = ePath(load_directory) / OPTIMIZER_STRUCT_NAME
        tx_struct_path = ePath(load_directory) / TX_STRUCT_JSON
        if optimizer_path.exists() or tx_struct_path.exists():
            cmg = jax.default_device(device) if device is not None else contextlib.nullcontext()
            with cmg:
                state = state.load_optimizer(load_directory=load_directory)
        if auto_shard_model:
            state = state.shard_state()
        return state

    def shard_with_shape(self: Self, shape) -> Self:
        """Applies sharding constraints to the entire state based on a reference shape pytree.

        This method takes a pytree `shape` which has the same structure as the `EasyDeLState`
        but contains sharding annotations (e.g., `NamedSharding`) instead of actual array data.
        It applies these shardings as constraints to the corresponding arrays in the current state.

        Args:
            shape: A pytree with the same structure as `self`, containing sharding annotations.

        Returns:
            EasyDeLState: A new state object with sharding constraints applied.
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
        """Shards the entire state (model parameters and optimizer state) based on partition rules.

        This is a convenience method that calls `shard_model` and `shard_optimizer_state`.

        Args:
            partition_rules (PartitionLike, optional):
                Partitioning rules. If None, uses rules from the model's config. Defaults to None.
            mesh (Mesh, optional):
                The JAX device mesh to shard across. If None, uses model's mesh. Defaults to None.

        Returns:
            EasyDeLState: A new state object with both model and optimizer states sharded.
        """
        from eformer.escale import make_shard_and_gather_fns, match_partition_rules

        rules = partition_rules or self.model._get_partition_rules(None)
        mesh = mesh or self.model._get_mesh(None)

        def appy_sharding_on_tree(tree):
            partition_specs = match_partition_rules(rules, tree)
            shard_fns, _ = make_shard_and_gather_fns(partition_specs, mesh)
            return jax.tree_util.tree_map(lambda f, o: f(o), shard_fns, tree)

        return appy_sharding_on_tree(self)

    def gather_state(self):
        """Gathers the entire state (model parameters and optimizer state) from distributed devices.

        This is a convenience method that calls `gather_model` and `gather_optimizer_state`.

        Returns:
            EasyDeLState: A new state object with both model and optimizer states gathered.
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
        """Gathers the model parameters (`graphstate` and `graphother`) from distributed devices.

        Args:
            partition_rules (PartitionLike, optional):
                Partitioning rules used for the original sharding. If None, uses model config rules. Defaults to None.
            mesh (tp.Optional[Mesh], optional):
                The JAX device mesh to gather from. If None, uses model's mesh. Defaults to None.

        Returns:
            EasyDeLState: A new state object with gathered `graphstate` and `graphother`.
        """
        from eformer.escale import make_shard_and_gather_fns, match_partition_rules

        rules = partition_rules or self.model._get_partition_rules(None)
        mesh = mesh or self.model._get_mesh(None)
        partition_specs = match_partition_rules(rules=rules, tree=self.graphstate)
        _, gather_fns = make_shard_and_gather_fns(partition_specs=partition_specs, mesh=mesh)
        graphstate = jax.tree_util.tree_map(lambda f, o: f(o), gather_fns, self.graphstate)
        graphother = jax.tree_util.tree_map(lambda f, o: f(o), gather_fns, self.graphother)
        self = self.replace(graphstate=graphstate, graphother=graphother)
        return self

    def shard_model(self: Self, partition_rules: PartitionLike = None, mesh: Mesh | None = None) -> Self:
        """
        Shards the model parameters (`graphstate` and `graphother`) based on partition rules.

        Args:
            partition_rules (PartitionLike, optional): Partitioning rules. If None, uses
                model config rules. Defaults to None.
            mesh (tp.Optional[Mesh], optional): The JAX device mesh to shard across. If None,
                uses model's mesh. Defaults to None.

        Returns:
            EasyDeLState: A new state object with sharded `graphstate` and `graphother`.
        """

        rules = partition_rules or self.model._get_partition_rules(None)
        mesh = mesh or self.model._get_mesh(None)

        def appy_sharding_on_tree(tree):
            from eformer.escale import make_shard_and_gather_fns, match_partition_rules

            partition_specs = match_partition_rules(rules, tree)
            shard_fns, _ = make_shard_and_gather_fns(partition_specs, mesh)
            return jax.tree_util.tree_map(lambda f, o: f(o), shard_fns, tree)

        graphstate = appy_sharding_on_tree(self.graphstate)
        graphother = appy_sharding_on_tree(self.graphother)

        self = self.replace(graphstate=graphstate, graphother=graphother)
        return self

    @property
    def mesh(self):
        return self.model.mesh

    @property
    def shardings(self):
        """
        Retrieves the sharding annotations (e.g., `NamedSharding`) for all components
        of the `EasyDeLState` pytree.

        Returns:
            A pytree with the same structure as `self`, containing sharding annotations
            or None for components without sharding.
        """
        return nn.from_tree(
            jax.tree_util.tree_map(
                lambda x: x.sharding if hasattr(x, "sharding") else None,
                nn.to_tree(self),
            )
        )

    def __repr__(self):
        """
        Returns a string representation of the EasyDeLState, primarily showing the model representation.
        """
        return ("TxIncluded-" if self.opt_state is not None else "") + "EasyDeLState-" + str(self.model)

    __str__ = __repr__
