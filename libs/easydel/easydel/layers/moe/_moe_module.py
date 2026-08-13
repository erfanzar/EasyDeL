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

"""Mixture of Experts (MoE) layer implementations for EasyDeL.

This module provides a comprehensive implementation of Mixture of Experts (MoE) layers
for large-scale neural networks. It includes support for various routing strategies,
load balancing techniques, and distributed training optimizations.

Key Components:
    - **BaseMoeModule**: Abstract base class for MoE implementations with common
      utilities for routing, permutation, and metric computation.
    - **ParallelMoELinear**: Batched linear transformation layer for expert networks
      with support for ragged and grouped matrix multiplication.
    - **Routing Strategies**: Multiple routing algorithms including top-k, switch,
      expert choice, and hash-based routing.
    - **Load Balancing**: Various strategies to ensure balanced expert utilization
      including standard, switch transformer, and expert choice methods.
    - **Distributed Support**: Full support for expert parallelism (EP), tensor
      parallelism (TP), and data parallelism (DP) with optimized all-to-all
      communication patterns.

The module is designed for efficient execution on TPUs and GPUs with optimizations
for:
    - Custom VJP for gradient-efficient sorting operations
    - Pallas-based grouped matrix multiplication kernels for TPUs
    - Ragged tensor operations for variable-length expert assignments
    - Automatic sharding and partitioning for distributed training

Example:
    >>> from easydel.layers import BaseMoeModule, ParallelMoELinear
    >>> # Create a custom MoE layer by extending BaseMoeModule
    >>> class CustomMoE(BaseMoeModule):
    ...     def __init__(self, config):
    ...         super().__init__(config)
    ...         # Initialize gate and expert layers
    ...     def forward(self, hidden_states):
    ...         # Implement forward pass using _moe_call_standard helper
    ...         return self._moe_call_standard(...)
"""

from __future__ import annotations

import typing
import typing as tp
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import spectrax as spx
from eformer.loggings import get_logger
from ejkernel.modules import GroupedMatmulConfig, grouped_matmul  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jax import shard_map
from jax._src.mesh import AxisType, MeshAxisName
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float, Int
from spectrax import common_types

from easydel.infra.sharding import inference_mode_forces_decode, resolve_stage_mesh
from easydel.utils.helpers import check_bool_flag
from easydel.utils.inference_mode import is_inference_mode

from ._communication_utils import (
    MoeFusedHooks,
    MoeLoadBalancingStrategy,
    MoEMethods,
    MoeMetrics,
    MoeRoutingStrategy,
    build_ep_traffic_matrix,
    chunk_group_sizes,
    ep_batch_receive_buffer_rows,
    get_all_to_all_params,
    get_experts_location,
    get_moe_partition_spec,
    local_permute,
    permute,
    resolve_eformer_axis,
    sort_activations,
    unpermute,
)
from ._layout_planner import validate_moe_layout

if typing.TYPE_CHECKING:
    from easydel.infra.base_config import EasyDeLBaseConfig

logger = get_logger(__name__)


BATCH = common_types.BATCH
EMPTY = common_types.EMPTY
EMBED = common_types.EMBED
EXPERT = common_types.EXPERT
MODE_TRAIN = common_types.MODE_TRAIN
EP = common_types.EP
DP = common_types.DP
FSDP = common_types.FSDP
TP = common_types.TP
SP = common_types.SP


class BaseMoeModule(spx.Module, ABC):
    """Abstract base class shared by every Mixture-of-Experts module in EasyDeL.

    This is THE base class consumed by ~15 MoE model implementations (DeepSeek,
    Qwen, Mixtral, Llama4, Arctic, Falcon-H1, GLM, GPT-OSS, etc.). It owns the
    bulk of the MoE machinery so that concrete models only need to define their
    own gate/router and expert kernels and then dispatch to :meth:`moe_call`.
    Responsibilities provided here:

    * Token routing helpers (``_replicate_and_sort_tokens``, capacity masking,
      expert-group masks for hierarchical routing).
    * Three execution paths: fused grouped-matmul (:meth:`_sparse_moe_call`),
      dense per-token einsum (:meth:`_moe_call_dense`), and the standard
      permute / call-expert-layer / unpermute path (:meth:`_moe_call_standard`).
    * Distributed sharding plumbing: expert-mesh construction
      (3D ``(dp, expert, tp)`` or stage-local 5D fall-through), partition-spec
      generation via :meth:`get_moe_spec`, and shard-map orchestration for
      ring-of-experts and ragged-all-to-all expert parallelism.
    * Loss & metrics computation (load-balancing loss, router z-loss,
      utilization, entropy) aggregated into :class:`MoeMetrics`.
    * Per-routing-strategy default hook auto-configuration via
      :meth:`_configure_hooks_for_routing_strategy`.

    Subclasses must implement :meth:`forward` (decorated ``@abstractmethod``)
    and are responsible for owning their own gate layer and expert kernels.

    Attributes:
        config (EasyDeLBaseConfig): Model configuration; carries MoE-related
            knobs (`moe_method`, `use_ring_of_experts`, `use_expert_tensor_mode`,
            `fsdp_is_ep_bound`, `sp_is_ep_bound`, etc.).
        runtime_sharding_resolver: Resolver used to translate logical axis
            names (DP/FSDP/EP/TP/SP) into physical mesh axis names.
        n_routed_experts (int): Total number of routable experts.
        num_experts_per_tok (int): Top-k width used by routing.
        hidden_size (int): Hidden dimension of inputs and outputs.
        lbl_coef (float | None): Coefficient for the load-balancing loss
            (``None`` disables it).
        rzl_coef (float | None): Coefficient for the router z-loss
            (``None`` disables it).
        routing_strategy (MoeRoutingStrategy): Top-k / Switch / Expert-Choice /
            Hash strategy applied at the gate.
        load_balancing_strategy (MoeLoadBalancingStrategy): Which aux-loss
            formulation to apply when ``lbl_coef`` is set.
        moe_hooks (MoeFusedHooks): Hook bundle invoked at fixed pipeline
            points; defaults to an empty bundle.
        module_moe_method (MoEMethods): Cached value of ``config.moe_method``;
            picks the execution path inside :meth:`moe_call`.
        expert_mesh / auto_expert_mesh / expert_abstract_mesh: Pre-computed
            expert meshes pulled from the config and used as fallbacks when
            the stage-local mesh cannot be resolved.
        dtype (jnp.dtype): Compute dtype, defaulting to ``bfloat16``.
        _mesh: The raw JAX mesh stored from ``config.mesh`` (used via the
            :attr:`mesh` property after stage resolution).
    """

    def __init__(
        self,
        config: EasyDeLBaseConfig,
        n_routed_experts: int | None = None,
        num_experts_per_tok: int | None = None,
        hidden_size: int | None = None,
        lbl_coef: float | None = None,
        rzl_coef: float | None = None,
        routing_strategy: MoeRoutingStrategy = MoeRoutingStrategy.TOP_K,
        load_balancing_strategy: MoeLoadBalancingStrategy = MoeLoadBalancingStrategy.STANDARD,
        moe_hooks: MoeFusedHooks | None = None,
    ):
        """Initialize the shared MoE state from the model config.

        Mostly a cache of config-derived values plus a default empty
        :class:`MoeFusedHooks` bundle. No parameters are created here; expert
        kernels and gate weights belong to the concrete subclass.

        Args:
            config: Configuration object exposing mesh, partition resolver,
                and ``moe_*`` knobs (method, ring-of-experts flag, expert mesh).
            n_routed_experts: Total expert count. Falls back to
                ``config.n_routed_experts`` when ``None``.
            num_experts_per_tok: Top-k width per token. Falls back to
                ``config.num_experts_per_tok`` when ``None``.
            hidden_size: Hidden dimension of MoE inputs/outputs. Falls back to
                ``config.hidden_size`` when ``None``.
            lbl_coef: Coefficient on the load-balancing aux loss; ``None``
                disables it.
            rzl_coef: Coefficient on the router z-loss; ``None`` disables it.
            routing_strategy: Strategy used to map tokens to experts.
            load_balancing_strategy: Formulation used by
                :meth:`_compute_load_balancing_loss`.
            moe_hooks: Optional bundle of pipeline hooks. ``None`` instantiates
                an empty :class:`MoeFusedHooks`.
        """
        super().__init__()
        self.config = config
        self._mesh = config.mesh
        self.runtime_sharding_resolver = config.runtime_sharding_resolver
        self.n_routed_experts = n_routed_experts or config.n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok or config.num_experts_per_tok
        self.hidden_size = hidden_size or config.hidden_size
        self.lbl_coef = lbl_coef
        self.rzl_coef = rzl_coef
        self.routing_strategy = routing_strategy
        self.load_balancing_strategy = load_balancing_strategy
        self.moe_hooks: MoeFusedHooks | None = MoeFusedHooks() if moe_hooks is None else moe_hooks
        self.module_moe_method = self.config.moe_method

        self.expert_mesh = self.config.expert_mesh
        self.auto_expert_mesh = self.config.auto_expert_mesh
        self.expert_abstract_mesh = self.config.expert_abstract_mesh

        self.dtype = getattr(self, "dtype", jnp.bfloat16)

    @property
    def mesh(self):
        """Resolve the active stage-local mesh for this MoE module.

        Returns:
            The :class:`StageMesh` (or compatible mesh) currently active for
            the surrounding pipeline stage. When no MPMD pipeline is in scope
            this is equivalent to the module's stored ``self._mesh``.
        """
        return resolve_stage_mesh(self._mesh)

    def _build_active_expert_mesh(
        self,
        *,
        axis_types: tuple[jax.sharding.AxisType, ...],
        arr: jax.Array | None = None,
    ) -> spx.SpxMesh:
        """Return the active stage-local mesh used by fused MoE kernels.

        In an MPMD stage, the safest layout is the stage submesh itself:
        ``(dp, fsdp, ep, tp, sp)`` with the pipeline axis already removed.
        The shard-map then marks only ``dp``/``ep``/``tp`` as manual axes.
        This avoids lowering a separate 3D concrete mesh inside the stage HLO,
        which can confuse TPU enhanced-barrier validation when other stage
        work is compiled in the same executable.

        If a production config binds a non-size-one ``fsdp`` or ``sp`` axis
        into expert parallelism, we still fall back to the historical folded
        3D ``(dp, ep, tp)`` mesh so that the expert axis keeps the requested
        size. When the active stage has no resolvable mesh, the pre-computed
        ``self.auto_expert_mesh`` is returned as a fallback.

        Args:
            axis_types: Three :class:`jax.sharding.AxisType` values, one per
                axis of the resulting expert mesh ``(dp, expert, tp)``. Use
                ``Auto`` for the auto-resharding path and ``Manual`` for the
                shard-map path.
            arr: Optional sample array used by :func:`resolve_stage_mesh` to
                determine the active MPMD stage when none has been entered
                explicitly.

        Returns:
            A stage-local :class:`spx.SpxMesh`, either the 5D stage submesh or
            the folded 3D expert mesh when bound axes require folding.
        """
        mesh = resolve_stage_mesh(self._mesh, arr=arr)
        if mesh is None:
            return self.auto_expert_mesh

        runtime_sharding_resolver = self.runtime_sharding_resolver
        dpname = typing.cast(str, resolve_eformer_axis(DP, runtime_sharding_resolver))
        fsdpname = typing.cast(str, resolve_eformer_axis(FSDP, runtime_sharding_resolver))
        epname = typing.cast(str, resolve_eformer_axis(EP, runtime_sharding_resolver))
        tpname = typing.cast(str, resolve_eformer_axis(TP, runtime_sharding_resolver))
        spname = typing.cast(str, resolve_eformer_axis(SP, runtime_sharding_resolver))

        axis_sizes = dict[MeshAxisName, Any](mesh.shape)
        axis_names = tuple(mesh.axis_names)
        mesh_devices = mesh.devices
        pp_size = int(axis_sizes.get("pp", 1)) if "pp" in axis_names else 1
        fsdp_size = int(axis_sizes.get(fsdpname, 1)) if fsdpname is not None else 1
        sp_size = int(axis_sizes.get(spname, 1)) if spname is not None else 1
        can_use_stage_mesh = (
            pp_size == 1
            and dpname in axis_names
            and epname in axis_names
            and tpname in axis_names
            and (not self.config.fsdp_is_ep_bound or fsdp_size == 1)
            and (not self.config.sp_is_ep_bound or sp_size == 1)
        )
        if can_use_stage_mesh:
            stage_axis_names = tuple(axis_name for axis_name in axis_names if axis_name != "pp")
            stage_devices = mesh_devices.reshape(tuple(axis_sizes[axis_name] for axis_name in stage_axis_names))
            manual_axis_types = {
                dpname: axis_types[0],
                epname: axis_types[1],
                tpname: axis_types[2],
            }
            stage_mesh = jax.sharding.Mesh(
                stage_devices,
                axis_names=stage_axis_names,
                axis_types=tuple[AxisType, ...](
                    manual_axis_types.get(axis_name, jax.sharding.AxisType.Auto) for axis_name in stage_axis_names
                ),
            )
            return spx.SpxMesh(jax_mesh=stage_mesh, mpmd_axis=None)

        assigned_axes: dict[str, str] = {}
        size_by_group = {"dp": 1, "ep": 1, "tp": 1}

        def assign_axis(axis_name: str | None, group: str) -> None:
            """Bucket a physical mesh axis into one of the three MoE groups.

            Updates the enclosing ``assigned_axes`` and ``size_by_group``
            tables in place: a previously unseen axis is recorded under
            ``group`` and its size is multiplied into the running product
            for that group. ``None`` and already-assigned axes are skipped.

            Args:
                axis_name: Physical mesh axis name as resolved from the
                    runtime sharding resolver (or ``None`` when the logical
                    axis does not exist).
                group: Target bucket — one of ``"dp"``, ``"ep"`` or ``"tp"``.
            """
            if axis_name is None or axis_name in assigned_axes:
                return
            assigned_axes[axis_name] = group
            size_by_group[group] *= axis_sizes.get(axis_name, 1)

        assign_axis(tpname, "tp")
        assign_axis(epname, "ep")
        assign_axis(dpname, "dp")
        assign_axis(fsdpname, "ep" if self.config.fsdp_is_ep_bound else "dp")
        assign_axis(spname, "ep" if self.config.sp_is_ep_bound else "dp")

        return spx.SpxMesh(
            jax_mesh=jax.sharding.Mesh(
                mesh_devices.flatten().reshape(size_by_group["dp"], size_by_group["ep"], size_by_group["tp"]),
                axis_names=(dpname, epname, tpname),
                axis_types=axis_types,
            ),
            mpmd_axis=None,
        )

    def _active_auto_expert_mesh(self, arr: jax.Array | None = None) -> spx.SpxMesh:
        """Build the stage-local 3-D MoE mesh in auto-resharding mode.

        Thin wrapper over :meth:`_build_active_expert_mesh` that pins all
        three axes to ``jax.sharding.AxisType.Auto`` so that Spectrax/JAX may
        re-distribute tensors transparently when the user takes the
        auto-resharding code path (as opposed to the manual ``shard_map``
        path used by the fused kernels).

        Args:
            arr: Optional sample array used by :func:`resolve_stage_mesh` to
                infer the active MPMD stage. ``None`` falls back to the
                module's stored mesh.

        Returns:
            A new :class:`spx.SpxMesh` of shape ``(dp_size, ep_size,
            tp_size)`` with all axes typed ``Auto``, suitable for sharding
            constraints in the auto-resharding MoE call paths.
        """
        return self._build_active_expert_mesh(
            axis_types=(
                jax.sharding.AxisType.Auto,
                jax.sharding.AxisType.Auto,
                jax.sharding.AxisType.Auto,
            ),
            arr=arr,
        )

    def _ep_carries_batch_active(self, expert_mesh: spx.SpxMesh) -> bool:
        """Whether the MaxText-style "ep carries batch" dispatch is active.

        The ``moe_ep_carries_batch`` knob shards the token batch over the
        expert axis and dispatches tokens to their expert-owning shard with a
        ragged all-to-all (instead of replicating the batch over ep and
        computing at home). It only composes with the SPMD stage-mesh layout
        where dp/fsdp/ep are distinct mesh axes:

        * non-ring only — ring-of-experts all-gathers the batch over ep, the
          exact opposite placement;
        * ``fsdp_is_ep_bound`` / ``sp_is_ep_bound`` must be ``False`` — the
          folded 3-D expert mesh absorbs those axes into ep and has no
          distinct fsdp axis to co-shard the batch with;
        * ``use_expert_tensor_mode`` swaps ep onto tp and has no expert axis
          to dispatch over;
        * ``n_routed_experts`` must divide by ep so every shard owns the same
          contiguous expert block (the traffic matrix assumes it).

        Shape-dependent conditions (batch divisibility by ``dp*fsdp*ep``) are
        checked separately in :meth:`_sparse_moe_call`, which may still
        deactivate the path per layer invocation.

        Args:
            expert_mesh: Active expert mesh from
                :meth:`_build_active_expert_mesh`.

        Returns:
            ``True`` when every static condition for the ep-carries-batch
            dispatch holds on this mesh.
        """
        if not bool(getattr(self.config, "moe_ep_carries_batch", False)):
            return False
        if self.config.use_ring_of_experts or self.config.use_expert_tensor_mode:
            return False
        if self.config.fsdp_is_ep_bound or self.config.sp_is_ep_bound:
            return False
        axis_names = expert_mesh.jax_mesh.axis_names
        ep_axis_name = resolve_eformer_axis(EP, self.runtime_sharding_resolver)
        fsdp_axis_name = resolve_eformer_axis(FSDP, self.runtime_sharding_resolver)
        # Stage-mesh path only: the folded 3-D expert mesh has no distinct
        # fsdp axis, and MPMD stages (pp > 1) always fold.
        if not (isinstance(ep_axis_name, str) and ep_axis_name in axis_names):
            return False
        if not (isinstance(fsdp_axis_name, str) and fsdp_axis_name in axis_names):
            return False
        ep_size = int(expert_mesh.shape[ep_axis_name])
        if ep_size <= 1:
            return False
        if self.n_routed_experts % ep_size != 0:
            logger.warn_once(
                f"moe_ep_carries_batch=True but n_routed_experts={self.n_routed_experts} is not divisible "
                f"by ep={ep_size}; falling back to the batch-replicated ep dispatch."
            )
            return False
        return True

    def _moe_batch_axis_names(
        self,
        expert_mesh: spx.SpxMesh,
        dp_axis_name: str,
        *,
        ep_carries_batch: bool | None = None,
    ) -> tuple[str, ...]:
        """Mesh axes carrying the token batch inside the fused-MoE ``shard_map``.

        On the folded 3-D expert mesh, :meth:`_build_active_expert_mesh` has
        already multiplied every non-expert-bound batch axis (fsdp, and sp when
        not ``sp_is_ep_bound``) into its dp axis, so naming dp alone shards the
        full batch group. On the 5-D SPMD stage mesh, dp and fsdp stay separate
        axes: a dp-only spec replicates the whole global batch into every fsdp
        shard, and the permuted dispatch buffer inside the shard_map then
        scales as ``B*S*k`` rows per device instead of ``B/(dp*fsdp)*S*k``
        (137 GB per core at B=512, S=8192, k=8, H=2048 — over the TPU int32
        allocation bound). Mirroring the folded path, fsdp joins the batch
        axes whenever it is a distinct axis of the active mesh that is not
        bound to expert parallelism. The sp axis keeps its historical
        replicated behavior (the batch is not generally divisible by sp).

        With ``moe_ep_carries_batch`` active the expert axis joins the batch
        axes as well (MaxText's "batch is sharded by ep" training layout), so
        each ep shard holds ``B/(dp*fsdp*ep)`` tokens and the dispatch
        all-to-all moves tokens to their expert-owning shard.

        Args:
            expert_mesh: Active expert mesh as returned by
                :meth:`_build_active_expert_mesh` (5-D stage mesh or folded
                3-D expert mesh).
            dp_axis_name: Resolved physical name of the data-parallel axis.
            ep_carries_batch: Explicit ep-carries-batch decision from the
                caller (which may include shape-dependent conditions such as
                batch divisibility). ``None`` recomputes the static gate via
                :meth:`_ep_carries_batch_active`.

        Returns:
            Tuple of physical mesh axis names to place on the batch dimension
            of the fused-MoE input/gate-logits/output partition specs.
        """
        batch_axis_names = (dp_axis_name,)
        if not self.config.fsdp_is_ep_bound:
            fsdp_axis_name = resolve_eformer_axis(FSDP, self.runtime_sharding_resolver)
            if (
                isinstance(fsdp_axis_name, str)
                and fsdp_axis_name != dp_axis_name
                and fsdp_axis_name in expert_mesh.jax_mesh.axis_names
            ):
                batch_axis_names += (fsdp_axis_name,)
        if ep_carries_batch is None:
            ep_carries_batch = self._ep_carries_batch_active(expert_mesh)
        if ep_carries_batch:
            ep_axis_name = resolve_eformer_axis(EP, self.runtime_sharding_resolver)
            if (
                isinstance(ep_axis_name, str)
                and ep_axis_name not in batch_axis_names
                and ep_axis_name in expert_mesh.jax_mesh.axis_names
            ):
                batch_axis_names += (ep_axis_name,)
        return batch_axis_names

    def get_moe_spec(
        self,
        direction: tp.Literal["row", "column"],
        tensors_are_expert: bool,
        is_bias: bool = False,
        fsdp_shards_experts: bool = False,
    ) -> PartitionSpec:
        """Generate the :class:`PartitionSpec` for an expert weight or bias tensor.

        Thin wrapper over :func:`get_moe_partition_spec` that fills in the
        module-bound configuration (resolver, FSDP/SP-bound flags).

        Args:
            direction: Weight matrix orientation:
                - "column": For column-wise sharding (wi/wu kernels)
                - "row": For row-wise sharding (wd kernel)
            tensors_are_expert: If True, uses expert tensor mode (experts on TP axis).
                If False, uses standard mode (experts on EP axis).
            is_bias: If True, generates spec for bias tensor (2D instead of 3D).
            fsdp_shards_experts: If True, add the FSDP axis to the expert-dim
                placement (matches ``moe_fsdp_shard_expert_weights`` at-rest
                sharding so the fused ``shard_map`` receives the fsdp-local
                expert slice instead of gathering at the boundary).

        Returns:
            PartitionSpec appropriate for the tensor.

        Examples:
            Standard mode (tensors_are_expert=False):
                - Column weight: [expert, None, tp]  # wi/wu: [E, H, M]
                - Row weight: [expert, tp, None]     # wd: [E, M, H]
                - Bias: [expert, None]               # [E, dim]

            Expert tensor mode (tensors_are_expert=True):
                - Column weight: [tp, None, None]    # Experts on TP
                - Row weight: [tp, None, None]       # Experts on TP
                - Bias: [tp, None]                   # [E, dim]
        """

        return get_moe_partition_spec(
            runtime_sharding_resolver=self.runtime_sharding_resolver,
            direction=direction,
            tensors_are_expert=tensors_are_expert,
            is_bias=is_bias,
            fsdp_is_ep_bound=self.config.fsdp_is_ep_bound,
            sp_is_ep_bound=self.config.sp_is_ep_bound,
            module_view=False,
            fsdp_shards_experts=fsdp_shards_experts,
        )

    def _get_sharding_status(self):
        """Resolve all parallelism axis names and their sizes for this MoE layer.

        Queries the runtime sharding resolver to translate the logical axes
        (DP/FSDP/EP/TP/SP) into physical mesh axis names, then reads the
        per-axis sizes from the active stage mesh. Handles both standard
        and expert-tensor parallelism modes:

        In standard mode:
            - Expert axis → EP (expert parallel)
            - Tensor axis → TP (tensor parallel)

        In expert-tensor mode (`use_expert_tensor_mode=True`):
            - Expert axis → TP (tensor parallel)
            - Tensor axis → EP (expert parallel)

        This axis swapping allows alternative sharding strategies for specific use cases.

        Returns:
            A tuple containing 10 elements:
                1. data_axis_name (str): Resolved data parallel axis name
                2. fsdp_axis_name (str): Resolved FSDP axis name
                3. expert_axis_name (str): Resolved expert parallel axis name
                4. tensor_axis_name (str): Resolved tensor parallel axis name
                5. sp_axis_name (str): Resolved sequence parallel axis name
                6. dp_size (int): Data parallel degree (number of devices)
                7. fsdp_size (int): FSDP degree
                8. ep_size (int): Expert parallel degree
                9. tp_size (int): Tensor parallel degree
                10. sp_size (int): Sequence parallel degree

        Note:
            Sizes default to 1 if the axis doesn't exist in the mesh.
        """
        mesh = self.mesh
        runtime_sharding_resolver = self.runtime_sharding_resolver.with_mesh(mesh)
        data_axis_name = resolve_eformer_axis(DP, runtime_sharding_resolver)

        if self.config.use_expert_tensor_mode:
            expert_axis_name = resolve_eformer_axis(TP, runtime_sharding_resolver)
            tensor_axis_name = resolve_eformer_axis(EP, runtime_sharding_resolver)
        else:
            expert_axis_name = resolve_eformer_axis(EP, runtime_sharding_resolver)
            tensor_axis_name = resolve_eformer_axis(TP, runtime_sharding_resolver)

        fsdp_axis_name = resolve_eformer_axis(FSDP, runtime_sharding_resolver)
        sp_axis_name = resolve_eformer_axis(SP, runtime_sharding_resolver)

        dp_size = mesh.shape.get(data_axis_name, 1) if mesh is not None else 1
        ep_size = mesh.shape.get(expert_axis_name, 1) if mesh is not None else 1
        tp_size = mesh.shape.get(tensor_axis_name, 1) if mesh is not None else 1
        fsdp_size = mesh.shape.get(fsdp_axis_name, 1) if mesh is not None else 1
        sp_size = mesh.shape.get(sp_axis_name, 1) if mesh is not None else 1

        return (
            data_axis_name,
            fsdp_axis_name,
            expert_axis_name,
            tensor_axis_name,
            sp_axis_name,
            dp_size,
            fsdp_size,
            ep_size,
            tp_size,
            sp_size,
        )

    def _validate_fused_moe_layout(
        self,
        *,
        global_tokens: int,
        expert_params_bytes: int,
        chunk_size: int | None,
        ep_carries_batch: bool = False,
    ) -> None:
        """Log the fused-MoE layout estimate and hard-fail impossible layouts.

        Runs at trace time from :meth:`_sparse_moe_call` with static shapes
        only. The underlying :func:`validate_moe_layout` memoizes on the full
        layout tuple, so a model with many identical MoE layers computes and
        logs the estimate once per distinct configuration (not per layer or
        per step). Raises when the permuted dispatch buffer exceeds the TPU
        int32 single-allocation bound — a config that can never compile.

        Args:
            global_tokens: Stage-global token count ``B * S`` seen by this
                MoE layer (pre-shard_map shapes are global).
            expert_params_bytes: Total bytes of this layer's expert
                parameters (kernels + biases) at their storage dtype.
            chunk_size: Active ``moe_chunk_size`` (``None`` when chunking is
                disabled).
            ep_carries_batch: Whether the ep-carries-batch dispatch is active
                for this invocation (batch sharded over ep, worst-case a2a
                receive buffer accounted instead of the replicated dispatch
                buffer).

        Raises:
            ValueError: Propagated from :func:`validate_moe_layout` when the
                dispatch buffer exceeds the int32 single-allocation bound.
        """
        if self.config.use_expert_tensor_mode:
            return
        try:
            (_, _, _, _, _, dp_size, fsdp_size, ep_size, tp_size, sp_size) = self._get_sharding_status()
        except Exception:  # pragma: no cover - mesh/resolver unavailable (e.g. abstract stage)
            return
        total_devices = int(dp_size) * int(fsdp_size) * int(ep_size) * int(tp_size) * int(sp_size)
        if total_devices <= 0:
            return
        validate_moe_layout(
            {"dp": dp_size, "fsdp": fsdp_size, "ep": ep_size, "tp": tp_size, "sp": sp_size},
            tokens_per_device=-(-int(global_tokens) // total_devices),
            num_experts_per_tok=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            num_experts=self.n_routed_experts,
            expert_params_bytes=expert_params_bytes,
            dtype_bytes=jnp.dtype(self.dtype).itemsize,
            fsdp_is_ep_bound=bool(self.config.fsdp_is_ep_bound),
            use_ring_of_experts=bool(self.config.use_ring_of_experts),
            chunk_size=chunk_size,
            fsdp_shard_expert_weights=bool(getattr(self.config, "moe_fsdp_shard_expert_weights", False)),
            ep_carries_batch=bool(ep_carries_batch),
        )

    def _replicate_and_sort_tokens(
        self,
        inputs_flat: jax.Array,
        selected_experts: jax.Array,
        use_custom_sort_vjp: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Replicate each token k times and sort the replicas by assigned expert id.

        This function prepares tokens for expert computation by:
        1. Replicating each token k times (once per selected expert)
        2. Sorting all replicated tokens so tokens for the same expert are contiguous
        3. Computing group sizes (how many tokens per expert)
        4. Creating sorted expert ID array aligned with sorted tokens

        The sorted layout enables efficient grouped/ragged matrix multiplication where
        each expert processes its assigned tokens as a contiguous batch.

        Args:
            inputs_flat: Flattened token representations. Shape: (num_tokens, hidden_dim).
            selected_experts: Expert assignments per token. Shape: (num_tokens, k)
                where k = `num_experts_per_tok`.
            use_custom_sort_vjp: Whether to use custom VJP for memory-efficient sorting.
                Defaults to True.

        Returns:
            A tuple containing:
                - sorted_inputs: Token representations sorted by expert. Shape: (num_tokens*k, hidden_dim).
                - sorted_by_expert: Sorting indices for the permutation. Shape: (num_tokens*k,).
                - group_sizes: Number of tokens assigned to each expert. Shape: (n_routed_experts,).
                - sorted_experts: Expert IDs aligned with sorted_inputs. Shape: (num_tokens*k,).

        Example:
            >>> # 2 tokens, 2 experts per token, 4 total experts
            >>> inputs = jnp.ones((2, 128))
            >>> selected = jnp.array([[0, 2], [1, 3]])  # token 0→experts 0,2; token 1→experts 1,3
            >>> sorted_inputs, indices, sizes, expert_ids = _replicate_and_sort_tokens(inputs, selected)
            >>> # sorted_inputs: tokens grouped as [token0_e0, token1_e1, token0_e2, token1_e3]
            >>> # sizes: [1, 1, 1, 1] - one token per expert
        """
        k = selected_experts.shape[-1]
        flat_idx = selected_experts.reshape(-1)
        sorted_by_expert = jnp.argsort(flat_idx)
        replicated = jnp.repeat(inputs_flat, k, axis=0)
        sorted_inputs = sort_activations(replicated, sorted_by_expert, use_custom_sort_vjp)
        group_sizes = jnp.bincount(flat_idx, length=self.n_routed_experts)
        sorted_experts = jnp.repeat(
            jnp.arange(self.n_routed_experts),
            repeats=group_sizes,
            total_repeat_length=flat_idx.shape[0],
        )
        return sorted_inputs, sorted_by_expert, group_sizes, sorted_experts

    def _apply_capacity_mask(
        self,
        selected_experts: jax.Array,
        weights: jax.Array,
        capacity_factor: float,
    ) -> jax.Array:
        """Zero out per-token weights for experts that have exceeded their capacity.

        This method limits the number of tokens each expert can process by zeroing out
        weights for tokens that exceed the expert's capacity. This helps prevent expert
        overload and improves load balancing during training.

        The capacity is computed as:
            capacity = max(ceil(tokens_per_batch / n_experts) * capacity_factor, capacity_factor)

        Tokens are processed in order, and once an expert reaches capacity, subsequent
        token assignments to that expert receive zero weight.

        Args:
            selected_experts: Expert assignments per token. Shape: (B, S, k) where
                B=batch_size, S=seq_len, k=num_experts_per_tok.
            weights: Expert weights per token. Shape: (B, S, k).
            capacity_factor: Multiplier for base capacity. Values > 1.0 allow more tokens,
                values < 1.0 enforce stricter limits. Typically in range [1.0, 2.0].

        Returns:
            Modified weights with overflow tokens masked to zero. Shape: (B, S, k).
            Tokens within capacity retain their original weights; overflow tokens get 0.

        Example:
            >>> # 2 batches, 4 tokens per batch, 2 experts per token, 4 total experts
            >>> experts = jnp.array([[[0, 1], [0, 2], [1, 3], [2, 3]],  # batch 0
            ...                      [[0, 1], [1, 2], [2, 3], [3, 0]]])  # batch 1
            >>> weights = jnp.ones((2, 4, 2))
            >>> masked_weights = _apply_capacity_mask(experts, weights, capacity_factor=1.5)
            >>> # Some tokens will have zero weight if they exceed expert capacity
        """
        B, S, k = selected_experts.shape
        tokens_per_batch = S * k
        cap = int(max(jnp.ceil(tokens_per_batch / self.n_routed_experts) * capacity_factor, capacity_factor))
        expert_mask = jax.nn.one_hot(selected_experts, num_classes=self.n_routed_experts, dtype=jnp.int32)
        fused = expert_mask.reshape(B, S * k, self.n_routed_experts)
        counts = jnp.cumsum(fused, axis=1)
        counts = counts.reshape(B, S, k, self.n_routed_experts)
        keep = (counts <= cap).astype(weights.dtype)
        keep_for_slot = jnp.sum(keep, axis=-1)
        return weights * keep_for_slot

    def _expert_group_mask(self, gate_logits: jax.Array, n_groups: int, topk_groups: int) -> jax.Array:
        """Build a binary mask selecting only experts within the top-k groups.

        This method implements hierarchical or grouped routing where experts are organized
        into groups, and tokens first select top-k groups, then select experts within
        those groups. This can improve routing efficiency and reduce computation when
        the number of experts is very large.

        The algorithm:
        1. Partition experts into n_groups
        2. For each group, compute a group score (sum of top-2 expert logits in that group)
        3. Select topk_groups with highest scores
        4. Create a mask that zeros out logits for experts in non-selected groups

        Args:
            gate_logits: Router logits for all experts. Shape: (batch*seq, n_experts).
            n_groups: Number of expert groups to partition experts into.
                Must evenly divide n_routed_experts.
            topk_groups: Number of groups to select per token. Typically 1 or 2.

        Returns:
            Binary mask for gate logits. Shape: (batch*seq, n_experts).
            1.0 for experts in selected groups, 0.0 for others.

        Example:
            >>> # 8 experts divided into 4 groups of 2 experts each
            >>> logits = jnp.ones((16, 8))  # 16 tokens, 8 experts
            >>> mask = _expert_group_mask(logits, n_groups=4, topk_groups=2)
            >>> # mask will have 1.0 for experts in 2 selected groups, 0.0 for others
            >>> masked_logits = logits * mask  # Zero out non-selected groups
        """
        BS = gate_logits.shape[0]
        experts_per_group = self.n_routed_experts // n_groups
        scores_grouped = gate_logits.reshape(BS, n_groups, experts_per_group)
        top2_vals, _ = jax.lax.top_k(scores_grouped, k=2)
        group_scores = jnp.sum(top2_vals.astype(jnp.float32), axis=-1)
        _, group_idx = jax.lax.top_k(group_scores, k=topk_groups)
        mask_groups = jax.nn.one_hot(group_idx, num_classes=n_groups, dtype=jnp.float32).sum(axis=-2)
        mask = jnp.broadcast_to(mask_groups[..., None], (BS, n_groups, experts_per_group)).reshape(BS, -1)
        return mask

    def _compute_load_balancing_loss(
        self,
        router_probs: jax.Array,
        expert_loads: jax.Array,
        strategy: MoeLoadBalancingStrategy | None = None,
    ) -> Array | float | None:
        """Compute the load balancing auxiliary loss for even expert utilization.

        This method calculates an auxiliary loss term that encourages the router to
        distribute tokens evenly across all experts, preventing expert collapse where
        a few experts receive most tokens while others are underutilized.

        The loss formulation depends on the chosen strategy:

        **STANDARD:**
            loss = lbl_coef * sum(f_i * p_i)
            where f_i = (expert_loads[i] * n_experts) / k is the normalized fraction
            of tokens to expert i, and p_i = mean(router_probs[:, i]) is the mean
            routing probability to expert i across all tokens.

        **SWITCH_TRANSFORMER:**
            loss = lbl_coef * n_experts * sum(f_i * p_i)
            where f_i = expert_loads[i] / n_tokens and p_i = mean(router_probs[:, i]).
            This matches the formulation from the Switch Transformer paper.

        **EXPERT_CHOICE:**
            loss = lbl_coef * variance(expert_loads)
            Penalizes variance in expert loads, encouraging uniform distribution.

        Args:
            router_probs: Router output probabilities for all tokens and experts.
                Shape: (num_tokens, num_experts). These are the softmax-normalized
                routing probabilities.
            expert_loads: Count of tokens assigned to each expert.
                Shape: (num_experts,). Computed via bincount of selected experts.
            strategy: Load balancing strategy to use. If None, uses
                self.load_balancing_strategy.

        Returns:
            The computed load balancing loss as a scalar float, or None if:
                - strategy is MoeLoadBalancingStrategy.NONE
                - self.lbl_coef is None (load balancing disabled)

        Raises:
            ValueError: If an unknown load balancing strategy is provided.

        Example:
            >>> # Compute loss during forward pass
            >>> expert_loads = jnp.bincount(selected_experts.flatten(), length=n_experts)
            >>> lb_loss = self._compute_load_balancing_loss(router_probs, expert_loads)
            >>> if lb_loss is not None:
            ...     total_loss = main_loss + lb_loss
        """
        strategy = strategy or self.load_balancing_strategy

        if strategy == MoeLoadBalancingStrategy.NONE or self.lbl_coef is None:
            return None

        if strategy == MoeLoadBalancingStrategy.STANDARD:
            f = expert_loads * self.n_routed_experts / self.num_experts_per_tok
            p = jnp.mean(router_probs, axis=0)
            return self.lbl_coef * jnp.sum(f * p)

        elif strategy == MoeLoadBalancingStrategy.SWITCH_TRANSFORMER:
            num_tokens = router_probs.shape[0]
            expert_fraction = expert_loads / num_tokens
            router_fraction = jnp.mean(router_probs, axis=0)
            return self.lbl_coef * self.n_routed_experts * jnp.sum(expert_fraction * router_fraction)

        elif strategy == MoeLoadBalancingStrategy.EMPTY_CHOICE:
            return self.lbl_coef * jnp.var(expert_loads)

        else:
            raise ValueError(f"Unknown load balancing strategy: {strategy}")

    def _compute_router_z_loss(self, router_logits: Float[Array, "batch_seq num_experts"]) -> Array | float | None:
        """Compute router z-loss to encourage numerical stability.

        The router z-loss penalizes large router logit magnitudes, which helps prevent
        router saturation and numerical instability. Large logits cause the softmax to
        produce near-one-hot distributions, reducing exploration and making training
        unstable.

        The loss is computed as:
            z_loss = rzl_coef * mean(logsumexp(logits)^2)

        The logsumexp term measures the "size" of the logits - if all logits are large,
        logsumexp will be large. Squaring and averaging creates a penalty that
        encourages the router to keep logits in a reasonable range.

        Args:
            router_logits: Pre-softmax router output logits for all tokens.
                Shape: (batch_size * seq_len, num_experts).

        Returns:
            The computed z-loss as a scalar float, or None if self.rzl_coef is None
            (z-loss disabled).

        Note:
            This loss was introduced in the ST-MoE paper and is commonly used
            alongside load balancing loss for stable MoE training.

        Example:
            >>> # During forward pass
            >>> router_logits = gate_layer(hidden_states_flat)
            >>> z_loss = self._compute_router_z_loss(router_logits)
            >>> if z_loss is not None:
            ...     aux_loss += z_loss
        """
        if self.rzl_coef is None:
            return None

        log_z = jax.nn.logsumexp(router_logits.astype(jnp.float32), axis=-1)
        return self.rzl_coef * jnp.mean(log_z**2)

    def _compute_metrics(
        self,
        router_logits: jax.Array,
        router_probs: jax.Array,
        selected_experts: jax.Array,
        selected_weights: jax.Array,
        expert_loads: jax.Array,
    ) -> MoeMetrics:
        """Compute and aggregate all MoE-related metrics and auxiliary losses.

        This method calculates various metrics useful for monitoring MoE training
        health, including load balancing loss, router z-loss, expert utilization,
        and routing entropy. These metrics help diagnose issues like expert collapse,
        router saturation, and load imbalance.

        Computed metrics include:
            - **load_balancing_loss**: Auxiliary loss encouraging even expert usage
            - **router_z_loss**: Auxiliary loss preventing large router logits
            - **expert_utilization**: Fraction of experts receiving at least one token
            - **routing_entropy**: Average entropy of routing distributions

        Args:
            router_logits: Pre-softmax router output logits.
                Shape: (batch_size * seq_len, num_experts).
            router_probs: Post-softmax routing probabilities.
                Shape: (batch_size * seq_len, num_experts).
            selected_experts: Expert indices selected for each token.
                Shape: (batch_size * seq_len, num_experts_per_tok).
            selected_weights: Weights for selected experts.
                Shape: (batch_size * seq_len, num_experts_per_tok).
            expert_loads: Number of tokens assigned to each expert.
                Shape: (num_experts,).

        Returns:
            MoeMetrics dataclass containing:
                - expert_loads: Input expert_loads array
                - router_probs: Input router_probs array
                - selected_experts: Input selected_experts array
                - selected_weights: Input selected_weights array
                - load_balancing_loss: Computed LB loss or None
                - router_z_loss: Computed z-loss or None
                - expert_utilization: Fraction of utilized experts (0 to 1)
                - routing_entropy: Mean entropy of routing distributions

        Example:
            >>> metrics = self._compute_metrics(
            ...     router_logits, router_probs, selected_experts, weights, loads
            ... )
            >>> print(f"Expert utilization: {metrics.expert_utilization:.2%}")
            >>> print(f"Routing entropy: {metrics.routing_entropy:.3f}")
            >>> if metrics.load_balancing_loss:
            ...     aux_loss = metrics.load_balancing_loss + (metrics.router_z_loss or 0)
        """
        metrics = MoeMetrics(
            expert_loads=expert_loads,
            router_probs=router_probs,
            selected_experts=selected_experts,
            selected_weights=selected_weights,
        )
        metrics.load_balancing_loss = self._compute_load_balancing_loss(router_probs, expert_loads)
        metrics.router_z_loss = self._compute_router_z_loss(router_logits)
        metrics.expert_utilization = jnp.mean(expert_loads > 0)
        metrics.routing_entropy = jnp.mean(-jnp.sum(router_probs * jnp.log(router_probs + 1e-8), axis=-1))
        return metrics

    def _apply_expert_sharding(self, tensor: Float[Array, ...], tensor_type: str = "weight") -> Float[Array, ...]:
        """Place an expert parameter onto the device mesh using a per-type sharding spec.

        Selects a partition spec keyed on ``tensor_type`` and the tensor's
        rank/shape, then calls :func:`jax.device_put` against the resolved
        sharding. The current implementation uses fully-replicated specs
        (every axis ``EMPTY``); the dispatch shell is in place for future
        EP-sharded layouts.

        Args:
            tensor: The tensor to shard. Can be weights, biases, or activations.
            tensor_type: Type hint for determining sharding strategy. Options:
                - "weight_col": Column-parallel weight (output dim partitioned)
                - "weight_row": Row-parallel weight (input dim partitioned)
                - "bias": Bias parameters
                - "weight" (default): Generic weight tensor

        Returns:
            The input tensor with sharding applied, placed on appropriate devices
            according to the resolved partition specification.

        Note:
            Current implementation uses EMPTY (replicated) sharding for all dimensions.
            Future versions may shard the expert dimension across EP devices.

        Example:
            >>> weight = jnp.ones((n_experts, hidden_dim, intermediate_dim))
            >>> sharded_weight = _apply_expert_sharding(weight, "weight_col")
            >>> # weight is now sharded across devices according to mesh configuration
        """
        mesh = self.mesh
        if mesh is None:
            return tensor
        runtime_sharding_resolver = self.runtime_sharding_resolver.with_mesh(mesh)

        if tensor_type == "weight_col":
            if tensor.ndim == 3 and tensor.shape[0] == self.n_routed_experts:
                sharding_spec = runtime_sharding_resolver.resolve(
                    axes=[EMPTY, EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape
                )
            elif tensor.ndim == 2:
                sharding_spec = runtime_sharding_resolver.resolve(
                    axes=[EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape
                )
            else:
                sharding_spec = runtime_sharding_resolver.resolve(axes=[EMPTY], mode=MODE_TRAIN)

        elif tensor_type == "weight_row":
            if tensor.ndim == 3 and tensor.shape[0] == self.n_routed_experts:
                sharding_spec = runtime_sharding_resolver.resolve(
                    axes=[EMPTY, EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape
                )
            elif tensor.ndim == 2:
                sharding_spec = runtime_sharding_resolver.resolve(
                    axes=[EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape
                )
            else:
                sharding_spec = runtime_sharding_resolver.resolve(axes=[EMPTY], mode=MODE_TRAIN)

        elif tensor_type == "bias":
            if tensor.ndim == 2 and tensor.shape[0] == self.n_routed_experts:
                sharding_spec = runtime_sharding_resolver.resolve(
                    axes=[EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape
                )
            else:
                sharding_spec = runtime_sharding_resolver.resolve(axes=[EMPTY], mode=MODE_TRAIN, shape=tensor.shape)

        else:
            if tensor.ndim == 3 and tensor.shape[0] == self.n_routed_experts:
                sharding_spec = runtime_sharding_resolver.resolve(
                    axes=[EMPTY, EMPTY, EMPTY],
                    mode=MODE_TRAIN,
                    shape=tensor.shape,
                )
            elif tensor.ndim == 2 and tensor.shape[0] == self.n_routed_experts:
                sharding_spec = runtime_sharding_resolver.resolve(
                    axes=[EMPTY, EMPTY], mode=MODE_TRAIN, shape=tensor.shape
                )
            else:
                sharding_spec = runtime_sharding_resolver.resolve(axes=[EMPTY], mode=MODE_TRAIN)

        with mesh:
            sharding = spx.get_corrected_named_sharding(tuple(tensor.shape), sharding_spec, raise_mesh_error=False)
        return jax.device_put(tensor, sharding)

    def _get_gate_layer_sharding(self, weight_shape: tuple) -> PartitionSpec:
        """Build the :class:`PartitionSpec` for the router/gate weight matrix.

        The gate maps hidden states to expert logits; the matrix is small
        relative to the expert FFNs, so the default spec replicates it
        across every mesh axis.

        Args:
            weight_shape: Shape of the gate weight matrix, typically (hidden_dim, n_experts).

        Returns:
            PartitionSpec defining how to shard the gate weights. Currently uses
            [EMPTY, EMPTY] (replicated across all dimensions).

        Note:
            Gate weights are typically small relative to expert FFN weights and
            are usually replicated for efficient routing computation.
        """
        return self.runtime_sharding_resolver.resolve(axes=[EMPTY, EMPTY], mode=MODE_TRAIN, shape=weight_shape)

    def _get_gate_layer_bias_sharding(self, bias_shape: tuple) -> PartitionSpec:
        """Build the :class:`PartitionSpec` for the router/gate bias vector.

        Args:
            bias_shape: Shape of the gate bias vector, typically (n_experts,).

        Returns:
            PartitionSpec defining how to shard the gate bias. Currently uses
            [EMPTY] (replicated).

        Note:
            Like gate weights, bias is usually replicated for efficient routing.
        """
        return self.runtime_sharding_resolver.resolve(axes=[EMPTY], mode=MODE_TRAIN, shape=bias_shape)

    def _validate_routing_inputs(
        self, hidden_states: Float[Array, "batch seq hidden_dim"], router_logits: Float[Array, "batch_seq num_experts"]
    ) -> None:
        """Validate input tensor shapes for routing operations.

        Performs shape validation to ensure inputs are compatible with the MoE
        configuration. This helps catch configuration errors early and provides
        clear error messages.

        Checks performed:
            1. Hidden dimension matches config.hidden_size
            2. Router logits expert dimension matches config.n_routed_experts
            3. Router logits batch dimension matches flattened input batch

        Args:
            hidden_states: Input hidden states from transformer layer.
                Shape: (batch_size, seq_len, hidden_dim).
            router_logits: Router output logits for expert selection.
                Shape: (batch_size * seq_len, num_experts).

        Raises:
            ValueError: If any shape validation fails, with a descriptive message
                indicating the mismatch.

        Example:
            >>> # This will raise ValueError if shapes don't match
            >>> self._validate_routing_inputs(hidden_states, router_logits)
        """
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Input hidden dimension {hidden_states.shape[-1]} doesn't "
                f"match config hidden dimension {self.hidden_size}"
            )

        if router_logits.shape[-1] != self.n_routed_experts:
            raise ValueError(
                f"Router logits expert dimension {router_logits.shape[-1]} doesn't match "
                f"config expert count {self.n_routed_experts}"
            )

        if router_logits.shape[0] != hidden_states.shape[0] * hidden_states.shape[1]:
            raise ValueError(
                f"Router logits batch dimension {router_logits.shape[0]} doesn't match "
                f"flattened input batch dimension {hidden_states.shape[0] * hidden_states.shape[1]}"
            )

    def _apply_capacity_constraint(
        self,
        selected_experts: jax.Array,
        selected_weights: jax.Array,
        capacity_factor: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply soft capacity constraint to limit tokens per expert.

        This method implements a soft capacity constraint by adjusting expert
        weights based on how much each expert is over capacity. Unlike hard
        capacity constraints (which drop tokens), this approach reduces weights
        proportionally to the over-capacity ratio.

        The algorithm:
            1. Compute maximum capacity: cap = capacity_factor * n_tokens / n_experts
            2. Count tokens assigned to each expert
            3. Compute over-capacity ratio: max(count / cap, 1.0)
            4. Scale weights by 1 / over_capacity_ratio
            5. Re-normalize weights to sum to 1 per token

        Args:
            selected_experts: Expert indices for each token.
                Shape: (batch_size * seq_len, num_experts_per_tok).
            selected_weights: Weights for selected experts.
                Shape: (batch_size * seq_len, num_experts_per_tok).
            capacity_factor: Multiplier for base capacity. Base capacity is
                n_tokens / n_experts (perfect balance). Values > 1.0 allow
                experts to receive more tokens than perfect balance.
                Defaults to 1.0 if None.

        Returns:
            Tuple of (selected_experts, constrained_weights) where:
                - selected_experts: Unchanged expert indices
                - constrained_weights: Adjusted weights with over-capacity
                  experts receiving reduced contribution. Normalized to sum
                  to 1 per token.

        Note:
            This is a "soft" constraint - no tokens are dropped, but over-used
            experts have their contributions reduced. This maintains gradient
            flow while encouraging load balance.

        Example:
            >>> # Apply capacity constraint with 1.5x buffer
            >>> experts, weights = self._apply_capacity_constraint(
            ...     selected_experts, selected_weights, capacity_factor=1.5
            ... )
        """
        if capacity_factor is None:
            capacity_factor = 1.0
        num_tokens = selected_experts.shape[0]
        max_capacity = int(capacity_factor * num_tokens / self.n_routed_experts)
        expert_counts = jnp.bincount(selected_experts.flatten(), length=self.n_routed_experts)
        over_capacity_ratio = jnp.maximum(expert_counts / max_capacity, 1.0)
        weight_adjustments = 1.0 / over_capacity_ratio[selected_experts]
        constrained_weights = selected_weights * weight_adjustments
        weight_sum = jnp.sum(constrained_weights, axis=-1, keepdims=True)
        constrained_weights = jnp.where(weight_sum > 0, constrained_weights / weight_sum, constrained_weights)
        return selected_experts, constrained_weights

    def _create_expert_mask(
        self,
        selected_experts: Int[Array, "batch_seq k"],
        expert_id: int,
    ) -> Bool[Array, "batch_seq"]:  # noqa: F821
        """Build a boolean mask of tokens routed to ``expert_id``.

        Convenience helper for per-expert analysis, debugging, or processing
        a single expert outside the batched/grouped path.

        Args:
            selected_experts: Expert assignments per token. Shape: (batch*seq, k)
                where k = num_experts_per_tok.
            expert_id: The expert ID to create a mask for (0 to n_routed_experts-1).

        Returns:
            Boolean mask where True indicates the token was assigned to the specified
            expert. Shape: (batch*seq,).

        Example:
            >>> selected = jnp.array([[0, 2], [1, 3], [0, 1]])  # 3 tokens, 2 experts each
            >>> mask = _create_expert_mask(selected, expert_id=0)
            >>> # mask = [True, False, True] - tokens 0 and 2 use expert 0
        """
        return jnp.any(selected_experts == expert_id, axis=-1)

    def _sparse_moe_call(
        self,
        hidden_state: jax.Array,  # [B, S, H]
        gate_layer: spx.Module,  # [H, E]
        wi_kernel: jax.Array | None,  # [E, H, M]
        wu_kernel: jax.Array | None,  # [E, H, M]
        wd_kernel: jax.Array | None = None,  # [E, M, H]
        wi_bias: jax.Array | None = None,  # [E, H]
        wu_bias: jax.Array | None = None,  # [E, H]
        wd_bias: jax.Array | None = None,  # [E, M]
        gate_up_kernel: jax.Array | None = None,  # [E, H, 2M]
        gate_up_bias: jax.Array | None = None,  # [E, 2M]
        ffn_activation: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        gate_hidden_state: jax.Array | None = None,  # [B, S, H]
        hooks: MoeFusedHooks | None = None,
        *,
        act_fn: Callable[[jax.Array], jax.Array],
    ):
        """Fused MoE path using grouped matmul and shard_map.

        This is the core fused MoE implementation that routes tokens to experts,
        permutes them to an expert-grouped layout, applies expert FFNs via grouped
        matmul kernels, and unpermutes/combines outputs. It supports both
        ring-of-experts and all-to-all expert-parallel communication depending on
        configuration and mesh sizes.

        **Architecture Overview:**
            1. **Routing**: Compute router logits via gate_layer and apply softmax
            2. **Permutation**: Sort tokens by expert assignment for grouped computation
            3. **Expert Computation**: Apply grouped matmul for W_i, W_u, activation, W_d
            4. **Communication** (if EP > 1):
               - Ring-of-Experts: All-gather pattern with local expert subsets
               - All-to-All: Ragged all-to-all for token redistribution
            5. **Unpermutation**: Restore token order and combine expert outputs
            6. **Resharding**: Convert from 3D expert mesh back to 5D model mesh

        **Hook Integration:**
            This method reads hooks from `self.moe_hooks` (automatically configured
            by `moe_call()` based on routing strategy). The following hooks
            are used at specific points:

            - `select_hook`: Refines expert selection weights. For TOP_K routing,
              this defaults to weight normalization (softmax). Called during permutation.
            - `refine_weights_hook`: Refines weights before W_i and W_u projections.
            - `refine_inputs_hook`: Refines token representations before expert-parallel
              all-to-all communication in distributed settings.

            Other hooks in `MoeFusedHooks` are not used in this path but could be
            added in future extensions.

        Args:
            hidden_state: Input tensor. Shape: [B, S, H].
            gate_layer: Router module mapping H -> E (produces logits).
            wi_kernel: Expert W_i kernel. Shape: [E, H, M].
            wu_kernel: Expert W_u kernel. Shape: [E, H, M].
            wd_kernel: Expert W_d kernel. Shape: [E, M, H].
            wi_bias: Optional bias for W_i. Shape: [E, H].
            wu_bias: Optional bias for W_u. Shape: [E, H].
            wd_bias: Optional bias for W_d. Shape: [E, M].
            ffn_activation: Optional custom activation combining (w0, w1) -> output.
            act_fn: Activation used when `ffn_activation` is not provided.

        Returns:
            Tuple `(output, router_logits)` where:
            - output: MoE layer output. Shape: [B, S, H].
            - router_logits: Pre-softmax router logits for auxiliary losses. Shape: [B*S, E].

        Example:
            >>> # Setup MoE layer with 8 experts, top-2 routing
            >>> config.n_routed_experts = 8
            >>> config.num_experts_per_tok = 2
            >>> config.use_ring_of_experts = False  # Use all-to-all
            >>>
            >>> # Initialize expert kernels
            >>> wi_kernel = jax.random.normal(key, (8, 768, 3072))  # gate/up
            >>> wu_kernel = jax.random.normal(key, (8, 768, 3072))  # up
            >>> wd_kernel = jax.random.normal(key, (8, 3072, 768))  # down
            >>>
            >>> # Call fused MoE
            >>> hidden_states = jnp.ones((2, 512, 768))  # (batch, seq, hidden)
            >>> output, logits = moe_layer._sparse_moe_call(
            ...     hidden_state=hidden_states,
            ...     gate_layer=gate,
            ...     wi_kernel=wi_kernel,
            ...     wu_kernel=wu_kernel,
            ...     wd_kernel=wd_kernel,
            ...     act_fn=jax.nn.silu,
            ... )
            >>> # output.shape = (2, 512, 768)
            >>> # logits.shape = (1024, 8)  # batch*seq, n_experts
        """

        if gate_up_kernel is None and (wi_kernel is None or wu_kernel is None):
            raise ValueError("MoE requires either gate_up_kernel or both wi_kernel and wu_kernel.")
        # Resolve routing-strategy defaults (e.g. TOP_K weight renormalization) the same
        # way the dense/standard paths do — the fused default path previously read the raw
        # self.moe_hooks and silently skipped the default refine/select hook, so TOP_K MoE
        # models relying on the framework default (Mixtral/Arctic, norm_topk_prob Qwen) were
        # not renormalizing top-k weights. _configure_hooks_for_routing_strategy returns
        # self.moe_hooks unchanged when the model set its own hooks, so this is a no-op there.
        hooks = self._configure_hooks_for_routing_strategy() if hooks is None else hooks
        select_hook = hooks.select_hook if hooks else None
        refine_weights_hook = hooks.refine_weights_hook if hooks else None
        refine_inputs_hook = hooks.refine_inputs_hook if hooks else None
        scale_replicated_inputs = hooks.scale_replicated_inputs if hooks else None
        output_weights_hook = hooks.output_weights_hook if hooks else None
        _BS, _SQLN, HD = hidden_state.shape

        hidden_state = hidden_state.astype(self.dtype)
        gate_hidden_state = hidden_state if gate_hidden_state is None else gate_hidden_state.astype(self.dtype)
        if hooks is not None and hooks.before_gate is not None:
            gate_hidden_state = hooks.before_gate(gate_hidden_state)

        # Flatten by the gate input's OWN width: `gate_hidden_state` exists so the
        # router can score a different tensor than the experts consume (LatentMoE
        # routes on full-width hidden while experts run in a narrower latent), and
        # reusing the expert width `HD` here would reshape it wrongly.
        prein_gate_logits = gate_layer(gate_hidden_state.reshape(-1, gate_hidden_state.shape[-1]))
        if inference_mode_forces_decode():
            # Decode-replicated scope: materialize the tiny [tokens, E] router
            # logits replicated ONCE so the softmax/top-k below run locally —
            # the sharded-logits path pays max/sum all-reduces plus a gather
            # per layer for data that is a few KB at decode.
            prein_gate_logits = spx.with_sharding_constraint(
                prein_gate_logits,
                jax.sharding.PartitionSpec(None, None),
                mesh=self.mesh,
            )
        if hooks is not None and hooks.after_gate is not None:
            prein_gate_logits = hooks.after_gate(prein_gate_logits)

        if hooks is not None and hooks.normalize_gate_logits is not None:
            gate_logits = hooks.normalize_gate_logits(prein_gate_logits)
        else:
            gate_logits = jax.nn.softmax(prein_gate_logits.astype("f4"), axis=-1).astype(prein_gate_logits.dtype)
        if hooks is not None and hooks.before_topk is not None:
            gate_logits = hooks.before_topk(gate_logits)

        expert_mesh: spx.SpxMesh = self._build_active_expert_mesh(
            axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
            arr=hidden_state,
        )
        runtime_sharding_resolver = self.runtime_sharding_resolver

        # Resolve axis names from the runtime sharding resolver rather than the mesh directly.
        dp_axis_name = resolve_eformer_axis(DP, runtime_sharding_resolver)
        expert_axis_name = resolve_eformer_axis(EP, runtime_sharding_resolver)
        tensor_axis_name = resolve_eformer_axis(TP, runtime_sharding_resolver)

        ep_size = expert_mesh.shape[expert_axis_name]
        tp_size = expert_mesh.shape[tensor_axis_name]

        if self.config.use_expert_tensor_mode:
            if tp_size != 1:
                raise ValueError("if using `ExpertTensorMode` Expert Parallel size should be 1.")

        if self.config.use_ring_of_experts and self.n_routed_experts % ep_size != 0:
            owned = (self.n_routed_experts // ep_size) * ep_size
            raise ValueError(
                f"use_ring_of_experts=True requires n_routed_experts "
                f"({self.n_routed_experts}) to be divisible by ep_size ({ep_size}); "
                f"otherwise experts [{owned}..{self.n_routed_experts - 1}] are silently "
                f"dropped from compute."
            )

        moe_chunk_size = getattr(self.config, "moe_chunk_size", None)
        if moe_chunk_size is not None:
            moe_chunk_size = int(moe_chunk_size)
            if moe_chunk_size <= 0:
                raise ValueError(f"`moe_chunk_size` must be a positive integer, got {moe_chunk_size}.")

        # ZeRO-3-style expert weights (moe_fsdp_shard_expert_weights): fsdp-sharded
        # at rest, all-gathered over the fsdp axis inside the shard_map right
        # before use. Active only when the active expert mesh carries a
        # non-trivial fsdp axis of its own (the 5-D SPMD stage mesh; the folded
        # 3-D mesh has no fsdp axis) and the expert dim divides over (ep, fsdp).
        fsdp_axis_name = resolve_eformer_axis(FSDP, runtime_sharding_resolver)
        fsdp_size = (
            int(expert_mesh.shape[fsdp_axis_name])
            if isinstance(fsdp_axis_name, str) and fsdp_axis_name in expert_mesh.jax_mesh.axis_names
            else 1
        )
        wants_fsdp_expert_weights = (
            bool(getattr(self.config, "moe_fsdp_shard_expert_weights", False))
            and not self.config.fsdp_is_ep_bound
            and not self.config.use_expert_tensor_mode
        )
        gather_expert_weights_over_fsdp = (
            wants_fsdp_expert_weights and fsdp_size > 1 and self.n_routed_experts % (ep_size * fsdp_size) == 0
        )
        if wants_fsdp_expert_weights and fsdp_size > 1 and not gather_expert_weights_over_fsdp:
            logger.warn_once(
                f"moe_fsdp_shard_expert_weights=True but n_routed_experts={self.n_routed_experts} is not "
                f"divisible by ep*fsdp={ep_size * fsdp_size}; expert weights stay fsdp-replicated inside the "
                "fused MoE shard_map (any at-rest fsdp sharding is gathered at the boundary instead)."
            )

        # MaxText-style "ep carries batch" (moe_ep_carries_batch): shard the
        # token batch over ep as well and dispatch tokens to their expert-
        # owning shard with a ragged all-to-all. Static mesh/config conditions
        # live in _ep_carries_batch_active; the batch-divisibility check needs
        # the traced shapes, so it lives here and may deactivate per layer.
        ep_carries_batch = self._ep_carries_batch_active(expert_mesh)
        if ep_carries_batch:
            dp_size = (
                int(expert_mesh.shape[dp_axis_name])
                if isinstance(dp_axis_name, str) and dp_axis_name in expert_mesh.jax_mesh.axis_names
                else 1
            )
            batch_ways = dp_size * fsdp_size * ep_size
            if _BS % batch_ways != 0:
                logger.warn_once(
                    f"moe_ep_carries_batch=True but batch={_BS} is not divisible by "
                    f"dp*fsdp*ep={batch_ways}; falling back to (dp, fsdp) batch sharding "
                    "with the batch-replicated ep dispatch for this layer."
                )
                ep_carries_batch = False

        expert_params_bytes = 0
        for _param in (wi_kernel, wu_kernel, gate_up_kernel, wd_kernel, wi_bias, wu_bias, gate_up_bias, wd_bias):
            if _param is None:
                continue
            for _leaf in _param if isinstance(_param, tuple) else (_param,):
                expert_params_bytes += int(_leaf.size) * jnp.dtype(_leaf.dtype).itemsize
        self._validate_fused_moe_layout(
            global_tokens=_BS * _SQLN,
            expert_params_bytes=expert_params_bytes,
            chunk_size=moe_chunk_size,
            ep_carries_batch=ep_carries_batch,
        )

        # Partition specs over the active expert layout. On MPMD stages this is
        # the folded 3D expert mesh; on SPMD (pp=1) it is the 5D stage mesh,
        # where dp and fsdp are separate axes and the batch dimension must name
        # both — otherwise dp=1/fsdp-dominant training meshes replicate the
        # whole global batch into every shard of the region below.
        batch_axis_names = self._moe_batch_axis_names(expert_mesh, dp_axis_name, ep_carries_batch=ep_carries_batch)
        # Shape-aware batch sanitization, INFERENCE TRACES ONLY: serving-style
        # packed batches can be smaller than the batch mesh group (eSurge
        # packs EVERY window — decode and prefill — as [1, N] tokens; with
        # fsdp>1 the ('dp','fsdp') batch spec cannot divide B=1 and shard_map
        # hard-errors). The batch placement must be dropped CONSISTENTLY
        # across x [B,S,H], gate logits [B*S,E], and the output — a partial
        # drop would pair replicated activations with sharded logits inside
        # the body. Batch-replicated execution is numerically identical, just
        # redundant across those shards. The drop is gated to inference
        # traces (eSurge compiles run under ``set_inference_mode()``; pure-
        # decode buckets additionally force decode-mode specs): a TRAINING
        # batch that cannot divide the batch group keeps the loud shard_map
        # error, because silently replicating it would multiply per-shard
        # compute without any signal to the user.
        _batch_group = 1
        _mesh_axis_sizes = dict(expert_mesh.jax_mesh.shape)
        for _axis_name in batch_axis_names:
            _batch_group *= int(_mesh_axis_sizes.get(_axis_name, 1))
        if _batch_group > 1 and _BS % _batch_group != 0 and (is_inference_mode() or inference_mode_forces_decode()):
            batch_axis_names = ()
        batch_axes = (
            batch_axis_names if len(batch_axis_names) > 1 else (batch_axis_names[0] if batch_axis_names else None)
        )
        input_ps = jax.sharding.PartitionSpec(batch_axes, None, None)
        glps = jax.sharding.PartitionSpec(batch_axes, None)

        # Mode-aware decode boundary: inside a ``decode_mode_specs()`` scope the
        # residual stream is replicated over TP (decode_hidden_state_axis=None),
        # so the MoE block must EMIT replicated hidden — otherwise its
        # tp-sharded output re-anchors the ~7/layer norm/qkv/router collectives
        # the decode layout removes (measured: constraint-only forcing made the
        # step WORSE, 435→482 collectives, because this producer didn't flip).
        # The body keeps the cheap psum_scatter over the top-k-expanded rows
        # and adds one small all-gather on the combined [tokens, H/tp] output.
        decode_replicated_out = inference_mode_forces_decode() and not self.config.use_expert_tensor_mode and tp_size > 1
        if decode_replicated_out:
            logger.debug("MoE decode-replicated boundary active for this trace")

        if self.config.use_expert_tensor_mode or decode_replicated_out:
            output_ps = jax.sharding.PartitionSpec(batch_axes, None, None)
        else:
            output_ps = jax.sharding.PartitionSpec(batch_axes, None, tensor_axis_name)

        if ffn_activation is None:

            def ffn_activation(x0: jax.Array, x1: jax.Array) -> jax.Array:
                """Default SwiGLU-style FFN activation: ``act_fn(x0) * x1``.

                Args:
                    x0: Gate-projection output.
                    x1: Up-projection output.

                Returns:
                    Element-wise ``act_fn(x0) * x1``, same shape as inputs.
                """
                return act_fn(x0) * x1

        # Generate weight sharding specs using helper function
        use_expert_tensor = self.config.use_expert_tensor_mode

        # With gather-at-use active the weight in-specs name (ep, fsdp) on the
        # expert dim so each shard receives its fsdp-local expert slice; the
        # shard_map body then all-gathers over fsdp explicitly.
        wspec = partial(self.get_moe_spec, fsdp_shards_experts=gather_expert_weights_over_fsdp)

        def _wspec_for(kernel_arg, direction):
            """Weight spec; quantized (codes, scales) tuples get a matching
            spec tuple — scales ``[E, 1, N]`` drop the contraction-axis
            placement (its axis is size 1)."""
            if kernel_arg is None:
                return None
            base = wspec(direction, use_expert_tensor, is_bias=False)
            if isinstance(kernel_arg, tuple):
                scales_spec = PartitionSpec(base[0], None, base[2] if direction == "column" else None)
                return (base, scales_spec)
            return base

        wikps = _wspec_for(wi_kernel, "column") if gate_up_kernel is None else None
        wukps = _wspec_for(wu_kernel, "column") if gate_up_kernel is None else None
        wgukps = _wspec_for(gate_up_kernel, "column") if gate_up_kernel is not None else None
        wdkps = _wspec_for(wd_kernel, "row")

        wibps = (
            wspec("column", use_expert_tensor, is_bias=True) if gate_up_kernel is None and wi_bias is not None else None
        )
        wubps = (
            wspec("column", use_expert_tensor, is_bias=True) if gate_up_kernel is None and wu_bias is not None else None
        )
        wgubps = (
            wspec("column", use_expert_tensor, is_bias=True)
            if gate_up_kernel is not None and gate_up_bias is not None
            else None
        )
        wdbps = wspec("row", use_expert_tensor, is_bias=True) if wd_bias is not None else None

        preferred_element_type = jnp.bfloat16
        if jnp.dtype(self.dtype) == jnp.float32:
            preferred_element_type = jnp.float32
        gmm_kws = {"preferred_element_type": preferred_element_type}
        if self.config.moe_force_xla_gmm:
            gmm_kws.update(cfg=GroupedMatmulConfig(platform="xla", bypass_xla_tiling=True))
        else:
            if jax.default_backend() == "tpu":
                moe_block_m = int(getattr(self.config, "moe_tiling_size_batch", 1024))
                moe_block_n = int(getattr(self.config, "moe_tiling_size_dim", 1024))
                # Pallas TPU lowering requires block shapes that satisfy hardware constraints.
                # If the configured tile sizes are incompatible, fall back to XLA.
                if (moe_block_m % 8 != 0) or (moe_block_n % 128 != 0):
                    gmm_kws.update(dict(cfg=GroupedMatmulConfig(platform="xla", bypass_xla_tiling=True)))
                else:
                    gmm_kws.update(platform="pallas")
                    if check_bool_flag("DISABLE_MOE_AUTOTUNE_ON_TPU", False):
                        gmm_kws.update(
                            cfg=GroupedMatmulConfig(
                                platform="pallas",
                                bypass_xla_tiling=True,
                                block_m=moe_block_m,
                                block_n=moe_block_n,
                                block_k=512,
                            )
                        )

        def _chunked_dispatch(
            x: jax.Array,
            gate_logits: jax.Array,
            expert_ffn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
            expert_shard_id: jax.Array,
        ) -> jax.Array:
            """Chunk-scanned fused-MoE dispatch body (``moe_chunk_size``).

            Runs inside the shard_map on the shard-local ``[b, s, H]`` token
            block (in ring mode: after the expert-axis all-gather). The
            flattened token stream is zero-padded to a multiple of
            ``moe_chunk_size`` and processed chunk-by-chunk under
            :func:`jax.lax.scan` with :func:`jax.checkpoint` on the chunk
            body, so both forward and backward peak memory are capped at
            roughly ``moe_chunk_size * top_k * H`` per live buffer instead of
            ``local_tokens * top_k * H``. Chunking over tokens is exact for
            token-choice top-k routing — each token's output depends only on
            its own k experts — so only float accumulation order can differ
            from the single-pass dispatch. Padded rows carry all-zero gate
            scores, receive zero combine weights, and are sliced off before
            returning.

            Args:
                x: Shard-local hidden states ``[b_local, s, H]`` (post
                    all-gather in ring mode).
                gate_logits: Shard-local normalized gate scores
                    ``[b_local * s, E]`` aligned with ``x``'s token order.
                expert_ffn: Grouped-matmul FFN core closure (gate/up ->
                    activation -> down -> TP psum_scatter -> down bias).
                expert_shard_id: This shard's index along the expert axis.

            Returns:
                Combined per-token expert outputs ``[b_local, s, H //
                tp_size]`` in the same token order as ``x`` (ring mode
                returns per-shard partials that still need the expert-axis
                ``psum_scatter`` combine).
            """
            local_batch, local_seq, _hd = x.shape
            tokens = local_batch * local_seq
            x2d = x.reshape(tokens, HD)
            num_chunks = -(-tokens // moe_chunk_size)
            padded_tokens = num_chunks * moe_chunk_size
            if padded_tokens != tokens:
                x2d = jnp.pad(x2d, ((0, padded_tokens - tokens), (0, 0)))
                gate_logits = jnp.pad(gate_logits, ((0, padded_tokens - tokens), (0, 0)))
            x_chunks = x2d.reshape(num_chunks, moe_chunk_size, HD)
            logits_chunks = gate_logits.reshape(num_chunks, moe_chunk_size, -1)

            def _combine_chunk(x_chunk: jax.Array, logits_chunk: jax.Array) -> jax.Array:
                """Run the full permute->FFN->combine pipeline on one chunk.

                Args:
                    x_chunk: Token chunk of shape ``[chunk, H]``.
                    logits_chunk: Gate scores for the chunk, ``[chunk, E]``.

                Returns:
                    Combined outputs ``[chunk, H // tp_size]`` in chunk-local
                    token order.
                """
                xc = x_chunk[None, :, :]
                if self.config.use_ring_of_experts:
                    experts_per_shard = self.n_routed_experts // ep_size
                    xs, sorted_sel, weights, group_sizes, selected = permute(
                        inputs=xc,
                        gate_logits=logits_chunk,
                        pre_bias_logits=None,
                        use_custom_sort_vjp=True,
                        roll_to_expert_id=experts_per_shard * expert_shard_id,
                        num_experts_per_tok=self.num_experts_per_tok,
                        num_experts=self.n_routed_experts,
                        dtype=self.dtype,
                        select_hook=select_hook,
                        refine_weights_hook=refine_weights_hook,
                        refine_inputs_hook=refine_inputs_hook,
                        scale_replicated_inputs=scale_replicated_inputs,
                    )
                    inter = expert_ffn(xs, group_sizes[:experts_per_shard], selected)
                    # Same tail-zeroing as the single-pass ring branch: only
                    # rows owned by this shard's local experts may be nonzero
                    # before the expert-axis psum_scatter combine.
                    local_mask = (selected < experts_per_shard)[:, None]
                    inter = jnp.where(local_mask, inter, 0)
                else:
                    xs, sorted_sel, weights, group_sizes, selected = permute(
                        inputs=xc,
                        gate_logits=logits_chunk,
                        pre_bias_logits=None,
                        use_custom_sort_vjp=True,
                        roll_to_expert_id=None,
                        num_experts_per_tok=self.num_experts_per_tok,
                        num_experts=self.n_routed_experts,
                        dtype=self.dtype,
                        select_hook=select_hook,
                        refine_weights_hook=refine_weights_hook,
                        refine_inputs_hook=refine_inputs_hook,
                        scale_replicated_inputs=scale_replicated_inputs,
                    )
                    chunk_shard_group_sizes = None
                    if ep_size > 1:
                        local_expert_size = self.n_routed_experts // ep_size
                        chunk_shard_group_sizes = jnp.sum(group_sizes.reshape(-1, local_expert_size), axis=1)
                        global_group_sizes = group_sizes
                        xs, _local_sorted, group_sizes, selected = local_permute(
                            xs,
                            global_group_sizes[None, :],
                            local_expert_size,
                            shard_index=expert_shard_id,
                            is_offset=True,
                            global_sorted_experts=selected,
                            use_custom_sort_vjp=True,
                        )
                    inter = expert_ffn(xs, group_sizes, selected)
                    if ep_size > 1:
                        # Unlike the ring combine, no tail-zeroing is needed
                        # here: ragged_all_to_all sends only the leading
                        # ``send_sizes`` rows of each shard's buffer and the
                        # receive buffer is fresh zeros, so the TPU grouped-
                        # matmul garbage tail never propagates (verified by a
                        # masked/unmasked A/B on a v5p-2048 ep=4 mesh).
                        rows = moe_chunk_size * self.num_experts_per_tok
                        out_buf = jnp.zeros((rows, HD // tp_size), dtype=inter.dtype)
                        in_off, send_sz, out_off, recv_sz = get_all_to_all_params(
                            chunk_shard_group_sizes,
                            expert_shard_id,
                            ep_size,
                            is_batch_sharded=False,
                        )
                        inter = jax.lax.ragged_all_to_all(
                            inter,
                            out_buf,
                            in_off,
                            send_sz,
                            out_off,
                            recv_sz,
                            axis_name=expert_axis_name,
                        )
                out = unpermute(
                    inter,
                    sorted_sel,
                    weights,
                    batch_size=1,
                    sequence_length=moe_chunk_size,
                    use_custom_sort_vjp=True,
                    weight_modif_fn=output_weights_hook,
                    num_experts_per_tok=self.num_experts_per_tok,
                    dtype=self.dtype,
                )
                return out[0]

            def _scan_body(carry: None, chunk_inputs: tuple[jax.Array, jax.Array]) -> tuple[None, jax.Array]:
                """Scan body: process one ``(x_chunk, logits_chunk)`` pair.

                Args:
                    carry: Unused (``None``); the scan is a pure map.
                    chunk_inputs: Tuple of the chunk's tokens and gate scores.

                Returns:
                    ``(None, combined_chunk_output)``.
                """
                x_chunk, logits_chunk = chunk_inputs
                return carry, _combine_chunk(x_chunk, logits_chunk)

            _, out_chunks = jax.lax.scan(jax.checkpoint(_scan_body), None, (x_chunks, logits_chunks))
            combined = out_chunks.reshape(padded_tokens, -1)
            if padded_tokens != tokens:
                combined = combined[:tokens]
            return combined.reshape(local_batch, local_seq, -1)

        def _sparse_call_body(
            x: jax.Array,
            gate_logits: jax.Array,
            wi_kernel: jax.Array,
            wu_kernel: jax.Array,
            gate_up_kernel: jax.Array | None,
            wd_kernel: jax.Array,
            wi_bias: jax.Array | None,
            wu_bias: jax.Array | None,
            gate_up_bias: jax.Array | None,
            wd_bias: jax.Array | None,
        ):
            """Per-shard fused-MoE body executed under :func:`jax.shard_map`.

            Permutes tokens by expert assignment, runs the three grouped
            matmuls (gate / up / down) with ``ffn_activation`` in between,
            handles inter-shard token redistribution (ring-of-experts or
            ragged all-to-all), and unpermutes the expert outputs back into
            the original token order on each device.

            Args:
                x: Local hidden-state slice of shape ``[B_local, S, H]``.
                gate_logits: Local router logits of shape ``[B_local*S, E]``.
                wi_kernel: Gate kernel of shape ``[E, H, M]``.
                wu_kernel: Up kernel of shape ``[E, H, M]``.
                wd_kernel: Down kernel of shape ``[E, M, H]``.
                wi_bias: Optional gate bias ``[E, H]`` or ``None``.
                wu_bias: Optional up bias ``[E, H]`` or ``None``.
                wd_bias: Optional down bias ``[E, M]`` or ``None``.

            Returns:
                Per-shard MoE output sharded according to ``output_ps``;
                shape matches the input ``x`` modulo the chosen output
                partition spec.
            """
            batch_size, sequence_length, _ = x.shape
            expert_shard_id = jax.lax.axis_index(expert_axis_name)
            reshaped_group_sizes: jax.Array | None = None

            if gather_expert_weights_over_fsdp:
                # ZeRO-3 gather-at-use: rebuild the ep-local expert block from
                # the fsdp shards once per layer invocation — outside any chunk
                # scan, so chunking never re-gathers per chunk. The all_gather's
                # transpose is a psum_scatter over fsdp, so expert-weight
                # gradients leave the shard_map already sharded like the
                # parameters (optimizer states mirror that sharding).
                def _gather_expert_param(t: jax.Array | None) -> jax.Array | None:
                    """All-gather one expert parameter's fsdp shards on dim 0."""
                    if t is None:
                        return None
                    return jax.lax.all_gather(t, fsdp_axis_name, axis=0, tiled=True)

                wi_kernel = _gather_expert_param(wi_kernel)
                wu_kernel = _gather_expert_param(wu_kernel)
                gate_up_kernel = _gather_expert_param(gate_up_kernel)
                wd_kernel = _gather_expert_param(wd_kernel)
                wi_bias = _gather_expert_param(wi_bias)
                wu_bias = _gather_expert_param(wu_bias)
                gate_up_bias = _gather_expert_param(gate_up_bias)
                wd_bias = _gather_expert_param(wd_bias)

            def _mask_dispatch_tail(t: jax.Array, group_sizes: jax.Array) -> jax.Array:
                """Zero rows past ``sum(group_sizes)`` of a dispatch-buffer tensor.

                With ep>1 the expert-sorted buffers keep their full static
                length but only the leading ``sum(group_sizes)`` rows belong to
                this shard's experts; the grouped-matmul kernels never *write*
                the remaining rows — forward or backward — so on TPU that tail
                holds whatever the buffer held, including NaNs (see the ring
                combine note below). The forward is protected (non-ring:
                ``ragged_all_to_all`` sends only the leading rows; ring: the
                explicit ``local_mask``), but the BACKWARD was not:
                ``back_grouped_matmul`` leaves the tail of ``d(x_rows)`` /
                ``d(intermediate)`` unwritten, and the sort-VJP scatter then
                routes that garbage into real token cotangents — the "M3" NaN
                (finite loss, non-finite grads, ep>1 meshes only; the top MoE
                layer's weight grads stay finite while its d(x) is NaN).
                Masking the gmm operands on the valid prefix is a forward
                no-op, and its transpose zeroes exactly those unwritten
                cotangent rows. At ep=1 every row belongs to a group (no
                tail), so this branch is not emitted and the ep=1 graph is
                unchanged.
                """
                if ep_size <= 1:
                    return t
                keep = jnp.arange(t.shape[0], dtype=jnp.int32) < jnp.sum(group_sizes)
                return jnp.where(keep[:, None], t, jnp.zeros_like(t))

            def _expert_ffn(x_rows: jax.Array, group_sizes: jax.Array, selected_experts: jax.Array) -> jax.Array:
                """Grouped-matmul expert FFN core over expert-sorted rows.

                Applies gate/up (or fused gate-up) grouped matmuls with their
                biases, the configured activation, the down grouped matmul,
                the TP ``psum_scatter`` reduction, and the down bias — exactly
                the single-pass pipeline between token permutation and the
                combine, shared by the chunked and single-pass dispatch paths.

                Args:
                    x_rows: Expert-sorted token rows ``[rows, H]``.
                    group_sizes: Rows per (shard-local) expert group.
                    selected_experts: Per-row (shard-local) expert ids used
                        for bias gathers.

                Returns:
                    Down-projected rows ``[rows, H // tp_size]`` (``H`` when
                    ``tp_size == 1``).
                """

                def _expert_gmm(rows, kernel):
                    """Dense bf16 grouped matmul, or — when the kernel is a
                    quantized ``(codes, scales)`` pair from ``kernel_view()`` —
                    the v3 grouped matmul's native quantised-weight path:
                    int8/int4 codes dequantised in-VMEM against per-channel
                    scales (measured 1.9-2.1x over bf16 at decode shapes and
                    1.5x at prefill on v5p; the auto-tiler loses to explicit
                    whole-K/whole-N tiles at MoE shapes, hence the cfg)."""
                    if isinstance(kernel, tuple):
                        codes, scales = kernel
                        n_groups, k_dim, n_dim = codes.shape
                        # `moe_force_xla_gmm` is a user-facing override that the
                        # dense branch honours through `gmm_kws`. Re-check it
                        # here or the knob silently applies to dense experts
                        # only, leaving quantized experts on Pallas.
                        on_pallas = jax.default_backend() == "tpu" and not self.config.moe_force_xla_gmm
                        # Explicit whole-K/whole-N tiles beat the auto-tiler by
                        # ~10x at MoE shapes, but Mosaic requires 128-aligned
                        # slice shapes — non-aligned contractions (e.g. a
                        # tp-sharded down projection with local k=192) must
                        # fall back to the auto-tiler.
                        tile_m = 64 if rows.shape[0] <= 1024 else 128
                        if k_dim % 128 == 0 and n_dim % 128 == 0:
                            cfg = GroupedMatmulConfig(block_m=tile_m, block_k=k_dim, block_n=n_dim)
                        else:
                            # block_k/n=0 sentinel: defer to the v3 kernel's
                            # own tiler, which handles non-aligned contractions.
                            cfg = GroupedMatmulConfig(block_m=tile_m, block_k=0, block_n=0)
                        return grouped_matmul(
                            rows,
                            codes,
                            group_sizes,
                            preferred_element_type=gmm_kws.get("preferred_element_type", jnp.bfloat16),
                            rhs_scale=scales.reshape(n_groups, 1, 1, n_dim),
                            use_v3=True,
                            platform="pallas" if on_pallas else "xla",
                            cfg=cfg,
                        )
                    return grouped_matmul(rows, kernel, group_sizes, **gmm_kws)

                x_rows = _mask_dispatch_tail(x_rows, group_sizes)
                if gate_up_kernel is not None:
                    layer_gate_up = _expert_gmm(x_rows, gate_up_kernel)
                    layer_gate_up = checkpoint_name(layer_gate_up, "mlp_gate_up")
                    if gate_up_bias is not None:
                        layer_gate_up = layer_gate_up + gate_up_bias[selected_experts]
                    layer_gate_up = _mask_dispatch_tail(layer_gate_up, group_sizes)
                    layer_w0, layer_w1 = jnp.split(layer_gate_up, 2, axis=-1)
                else:
                    layer_w0 = _expert_gmm(x_rows, wi_kernel)

                    layer_w0 = checkpoint_name(layer_w0, "mlp_gate")
                    if wi_bias is not None:
                        layer_w0 = layer_w0 + wi_bias[selected_experts]
                    layer_w0 = _mask_dispatch_tail(layer_w0, group_sizes)

                    layer_w1 = _expert_gmm(x_rows, wu_kernel)

                    layer_w1 = checkpoint_name(layer_w1, "mlp_up")
                    if wu_bias is not None:
                        layer_w1 = layer_w1 + wu_bias[selected_experts]
                    layer_w1 = _mask_dispatch_tail(layer_w1, group_sizes)

                intermediate_layer = _mask_dispatch_tail(ffn_activation(layer_w0, layer_w1), group_sizes)

                intermediate_output = _expert_gmm(intermediate_layer, wd_kernel)
                intermediate_output = checkpoint_name(intermediate_output, "mlp_down")

                # TP reduction: psum_scatter to shard output across TP on hidden dimension
                # This matches output_ps = [DP, EMPTY, TP]
                if tp_size > 1:
                    intermediate_output = jax.lax.psum_scatter(
                        intermediate_output,
                        tensor_axis_name,
                        scatter_dimension=1,
                        tiled=True,
                    )

                if wd_bias is not None:
                    intermediate_output = intermediate_output + wd_bias[selected_experts]
                return intermediate_output

            def _dispatch_ep_batch(x: jax.Array, gate_logits: jax.Array) -> jax.Array:
                """MaxText-style "ep carries batch" dispatch (``moe_ep_carries_batch``).

                The token batch is sharded over ep as well, so each shard
                permutes only its own ``B/(dp*fsdp*ep)`` tokens and the
                tokens travel to their expert-owning shard:

                1. ``permute`` sorts the local rows by *global* expert id, so
                   each destination shard's rows are contiguous.
                2. One ``all_gather`` of the per-expert group sizes yields the
                   global routing table; folding it gives the ``[ep, ep]``
                   traffic matrix for both all-to-all directions.
                3. DISPATCH ``ragged_all_to_all`` moves rows into a worst-case
                   sized receive buffer (see
                   :func:`ep_batch_receive_buffer_rows`); its output is
                   ``checkpoint_name``-tagged ``"moe_dispatch"`` so remat
                   policies can save it instead of replaying the collective.
                4. ``local_permute`` re-sorts the received rows by local
                   expert id; the valid rows form the leading prefix, the
                   buffer slack is a zero/garbage tail that every grouped
                   matmul masks away (``_mask_dispatch_tail``) — on TPU the
                   kernels never write rows past ``sum(group_sizes)``, so the
                   tail must be assumed garbage in forward AND backward.
                5. ``_expert_ffn`` (optionally chunk-scanned over the received
                   rows when ``moe_chunk_size`` is set — the a2a stays outside
                   the scan) computes the expert outputs.
                6. Local unsort + COMBINE ``ragged_all_to_all`` (the transposed
                   traffic matrix) returns each row to its home shard at its
                   original sorted position; tagged ``"moe_combine"``.
                7. ``unpermute`` restores token order and applies the top-k
                   combine weights.

                Args:
                    x: Local hidden-state block ``[B_local, S, H]`` (batch
                        sharded over dp, fsdp AND ep).
                    gate_logits: Local normalized gate scores
                        ``[B_local*S, E]``.

                Returns:
                    Combined per-token outputs ``[B_local, S, H // tp_size]``.
                """
                local_expert_size = self.n_routed_experts // ep_size
                x, sorted_selected_experts, weights, group_sizes, _selected_experts = permute(
                    inputs=x,
                    gate_logits=gate_logits,
                    pre_bias_logits=None,
                    use_custom_sort_vjp=True,
                    roll_to_expert_id=None,
                    num_experts_per_tok=self.num_experts_per_tok,
                    num_experts=self.n_routed_experts,
                    dtype=self.dtype,
                    select_hook=select_hook,
                    refine_weights_hook=refine_weights_hook,
                    refine_inputs_hook=refine_inputs_hook,
                    scale_replicated_inputs=scale_replicated_inputs,
                )
                local_rows = x.shape[0]  # B_local * S * k
                local_tokens = local_rows // self.num_experts_per_tok

                # One all_gather feeds both the [ep, ep] traffic matrix and
                # local_permute's per-source group layout.
                global_group_sizes = jax.lax.all_gather(group_sizes, axis_name=expert_axis_name)
                all_shards_group_sizes = build_ep_traffic_matrix(global_group_sizes, ep_size, local_expert_size)

                in_off, send_sz, out_off, recv_sz = get_all_to_all_params(
                    all_shards_group_sizes,
                    expert_shard_id,
                    ep_size,
                    is_batch_sharded=True,
                    is_dispatch=True,
                )
                recv_rows = ep_batch_receive_buffer_rows(
                    local_tokens, self.num_experts_per_tok, ep_size, local_expert_size
                )
                recv_buf = jnp.zeros((recv_rows, HD), dtype=x.dtype)
                x = jax.lax.ragged_all_to_all(
                    x,
                    recv_buf,
                    in_off,
                    send_sz,
                    out_off,
                    recv_sz,
                    axis_name=expert_axis_name,
                )
                x = checkpoint_name(x, "moe_dispatch")

                x, local_sorted_indices, local_group_sizes, local_sorted_experts = local_permute(
                    x,
                    global_group_sizes,
                    local_expert_size,
                    shard_index=expert_shard_id,
                    is_offset=False,
                    use_custom_sort_vjp=True,
                )
                valid_rows = jnp.sum(local_group_sizes)

                def _mask_row_tail(t: jax.Array) -> jax.Array:
                    """Zero rows past the valid prefix of a receive-sized buffer.

                    Forward no-op (the tail is already zero on backends that
                    honor the a2a/gmm zero-fill), but its transpose zeroes the
                    corresponding cotangent rows — on TPU the unwritten tails
                    of ``ragged_all_to_all`` transposes and grouped-matmul
                    backward outputs hold garbage that the sort/scatter
                    transposes would otherwise route into real token
                    cotangents (the 8ef1ddb9 "M3" NaN class).
                    """
                    keep = jnp.arange(t.shape[0], dtype=jnp.int32) < valid_rows
                    return jnp.where(keep[:, None], t, jnp.zeros_like(t))

                if moe_chunk_size is None:
                    intermediate = _expert_ffn(x, local_group_sizes, local_sorted_experts)
                else:
                    # Chunk-scan over the *received* rows: the dispatch a2a is
                    # already done, so the scan only bounds grouped-matmul
                    # live buffers. Group boundaries per chunk come from the
                    # cumulative group offsets (exact, and rows past the valid
                    # prefix fall outside every group automatically).
                    chunk_rows = int(moe_chunk_size) * self.num_experts_per_tok
                    num_chunks = -(-recv_rows // chunk_rows)
                    padded_rows = num_chunks * chunk_rows
                    x_p = x if padded_rows == recv_rows else jnp.pad(x, ((0, padded_rows - recv_rows), (0, 0)))
                    ids_p = (
                        local_sorted_experts
                        if padded_rows == recv_rows
                        else jnp.pad(
                            local_sorted_experts,
                            (0, padded_rows - recv_rows),
                            constant_values=local_expert_size - 1,
                        )
                    )
                    group_ends = jnp.cumsum(local_group_sizes)
                    group_starts = group_ends - local_group_sizes
                    x_chunks = x_p.reshape(num_chunks, chunk_rows, HD)
                    id_chunks = ids_p.reshape(num_chunks, chunk_rows)
                    chunk_offsets = jnp.arange(num_chunks, dtype=group_ends.dtype) * chunk_rows

                    def _scan_body(
                        carry: None, chunk_inputs: tuple[jax.Array, jax.Array, jax.Array]
                    ) -> tuple[None, jax.Array]:
                        """Run ``_expert_ffn`` on one contiguous row chunk."""
                        x_c, ids_c, c0 = chunk_inputs
                        gs_c = chunk_group_sizes(group_starts, group_ends, c0, chunk_rows)
                        return carry, _expert_ffn(x_c, gs_c, ids_c)

                    _, inter_chunks = jax.lax.scan(
                        jax.checkpoint(_scan_body), None, (x_chunks, id_chunks, chunk_offsets)
                    )
                    intermediate = inter_chunks.reshape(padded_rows, -1)
                    if padded_rows != recv_rows:
                        intermediate = intermediate[:recv_rows]

                # Mask the gmm output tail before the unsort moves it around,
                # and the unsorted buffer before the combine reads from it.
                intermediate = _mask_row_tail(intermediate)
                local_output = sort_activations(intermediate, jnp.argsort(local_sorted_indices), True)
                local_output = _mask_row_tail(local_output)

                c_in_off, c_send_sz, c_out_off, c_recv_sz = get_all_to_all_params(
                    all_shards_group_sizes,
                    expert_shard_id,
                    ep_size,
                    is_batch_sharded=True,
                    is_dispatch=False,
                )
                combine_buf = jnp.zeros((local_rows, HD // tp_size), dtype=local_output.dtype)
                combined = jax.lax.ragged_all_to_all(
                    local_output,
                    combine_buf,
                    c_in_off,
                    c_send_sz,
                    c_out_off,
                    c_recv_sz,
                    axis_name=expert_axis_name,
                )
                combined = checkpoint_name(combined, "moe_combine")

                return unpermute(
                    combined,
                    sorted_selected_experts,
                    weights,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    use_custom_sort_vjp=True,
                    weight_modif_fn=output_weights_hook,
                    num_experts_per_tok=self.num_experts_per_tok,
                    dtype=self.dtype,
                )

            if ep_carries_batch:
                # Batch is sharded over ep; chunking (if any) happens inside,
                # after the dispatch all-to-all.
                return _dispatch_ep_batch(x, gate_logits)

            if moe_chunk_size is not None:
                if self.config.use_ring_of_experts:
                    x, gate_logits = tuple(
                        jax.lax.all_gather(z, axis_name=expert_axis_name, tiled=True) for z in (x, gate_logits)
                    )
                    partial_output = _chunked_dispatch(x, gate_logits, _expert_ffn, expert_shard_id)
                    # Sum per-shard expert partials across the expert axis and
                    # hand each shard back its own token block — the same
                    # combine as the single-pass ring branch.
                    return jax.lax.psum_scatter(partial_output, expert_axis_name, scatter_dimension=0, tiled=True)
                return _chunked_dispatch(x, gate_logits, _expert_ffn, expert_shard_id)

            if self.config.use_ring_of_experts:
                x, gate_logits = tuple(
                    jax.lax.all_gather(
                        z,
                        axis_name=expert_axis_name,
                        tiled=True,
                    )
                    for z in (x, gate_logits)
                )

                # "Route" tokens within each shard.
                experts_per_shard = self.n_routed_experts // ep_size
                x, sorted_selected_experts, weights, group_sizes, selected_experts = permute(
                    inputs=x,
                    gate_logits=gate_logits,
                    pre_bias_logits=None,
                    use_custom_sort_vjp=True,
                    roll_to_expert_id=experts_per_shard * expert_shard_id,
                    num_experts_per_tok=self.num_experts_per_tok,
                    num_experts=self.n_routed_experts,
                    dtype=self.dtype,
                    select_hook=select_hook,
                    refine_weights_hook=refine_weights_hook,
                    refine_inputs_hook=refine_inputs_hook,
                    scale_replicated_inputs=scale_replicated_inputs,
                )

                # Keep only this shard's local experts. ``roll_to_expert_id`` above
                # rotated the local experts to ids ``[0, experts_per_shard)`` so their
                # tokens are the leading rows of the sorted buffer; the remaining rows
                # belong to non-local experts.
                #
                # We deliberately do NOT truncate ``x`` to ``sum(group_sizes)`` here.
                # The old code sliced with a traced length
                # (``jax.lax.dynamic_slice_in_dim(x, 0, jnp.sum(group_sizes), ...)``),
                # which is illegal under jit because a dynamic-slice *size* must be a
                # static Python int (it raised ``TracerIntegerConversionError`` at
                # trace time). Instead we leave ``x`` at its full static length and let
                # the grouped matmul below consume only the leading ``sum(group_sizes)``
                # rows: ``ragged_dot_general`` writes zeros for every row past
                # ``sum(group_sizes)`` (the same guarantee the XLA grouped_matmul relies
                # on for its m-alignment padding), so the non-local tail contributes
                # exactly zero and this shard's contribution is unchanged. The zero tail
                # is later summed across expert shards by the ``psum_scatter`` combine.
                group_sizes = group_sizes[:experts_per_shard]
            else:
                x, sorted_selected_experts, weights, group_sizes, selected_experts = permute(
                    inputs=x,
                    gate_logits=gate_logits,
                    pre_bias_logits=None,
                    use_custom_sort_vjp=True,
                    roll_to_expert_id=None,
                    num_experts_per_tok=self.num_experts_per_tok,
                    num_experts=self.n_routed_experts,
                    dtype=self.dtype,
                    select_hook=select_hook,
                    refine_weights_hook=refine_weights_hook,
                    refine_inputs_hook=refine_inputs_hook,
                    scale_replicated_inputs=scale_replicated_inputs,
                )

                if ep_size > 1:
                    local_expert_size = self.n_routed_experts // ep_size
                    reshaped_group_sizes = jnp.sum(group_sizes.reshape(-1, local_expert_size), axis=1)
                    global_group_sizes = group_sizes

                    x, _local_sorted_indices, group_sizes, selected_experts = local_permute(
                        x,
                        global_group_sizes[None, :],
                        local_expert_size,
                        shard_index=expert_shard_id,
                        is_offset=True,
                        global_sorted_experts=selected_experts,
                        use_custom_sort_vjp=True,
                    )

            intermediate_output = _expert_ffn(x, group_sizes, selected_experts)

            if self.config.use_ring_of_experts:
                # The unpermute + expert-axis ``psum_scatter`` combine below sums every
                # row across expert shards, so only the shard that OWNS a token's expert
                # may carry a nonzero value — the non-local tail (rolled expert ids
                # ``>= experts_per_shard``) MUST be exactly zero. The down grouped_matmul
                # is *supposed* to zero rows past ``sum(local group_sizes)``, but that
                # implicit zeroing is not guaranteed on every backend: XLA:CPU honors it,
                # but on TPU the tail holds garbage (ep=4 zero-bias output diverged 373x
                # from the non-ring reference, with NaNs at small magnitudes). ``wd_bias``,
                # added AFTER the matmul, additionally taints the tail via its clamped
                # out-of-bounds gather. Zero the tail explicitly so the combine is correct;
                # this is a no-op on backends that already zero it (bit-identical on CPU).
                local_mask = (selected_experts < (self.n_routed_experts // ep_size))[:, None]
                intermediate_output = jnp.where(local_mask, intermediate_output, 0)
                # ``intermediate_output`` keeps the full static length here, so it already
                # matches ``sorted_selected_experts`` and needs no padding before unpermute.
                output = unpermute(
                    intermediate_output,
                    sorted_selected_experts,
                    weights,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    # Match the non-ring unpermute and the permute calls above, which all
                    # pass a literal ``True``. ``use_custom_sort_vjp`` is a permute/unpermute
                    # argument, not a config field, so ``self.config.use_custom_sort_vjp``
                    # raised AttributeError once the trace reached this (previously
                    # unreachable) ring branch.
                    use_custom_sort_vjp=True,
                    weight_modif_fn=output_weights_hook,
                    num_experts_per_tok=self.num_experts_per_tok,
                    dtype=self.dtype,
                )

                # ``unpermute`` reshaped the ep-gathered token rows with the
                # pre-gather ``batch_size``, folding the gather factor into its
                # last dim; this reshape recovers token-aligned rows. Their true
                # width is the per-shard hidden size: after the TP
                # ``psum_scatter`` above (scatter_dimension=1) each row carries
                # ``HD // tp_size`` features, so reshaping with the global
                # ``HD`` under tp>1 would misfold hidden chunks into the batch
                # dimension right before the expert-axis combine.
                output = jnp.reshape(output, (-1, sequence_length, HD // tp_size))
                output = jax.lax.psum_scatter(output, expert_axis_name, scatter_dimension=0, tiled=True)

            else:
                if ep_size > 1:
                    # Unlike the ring combine, no tail-zeroing is needed here:
                    # the ragged_all_to_all below sends only the leading
                    # ``send_sizes`` rows of each shard's buffer (offsets and
                    # sizes derive from the local group sizes) and the receive
                    # buffer is fresh zeros, so the TPU grouped-matmul garbage
                    # tail never reaches another shard's combine (verified by
                    # a masked/unmasked A/B on a v5p-2048 ep=4 mesh:
                    # identical outputs and finite grads either way).
                    original_inputs_first_dim = batch_size * sequence_length * self.num_experts_per_tok

                    if sorted_selected_experts.shape[0] != original_inputs_first_dim:
                        raise ValueError("original_inputs_first_dim does not match the original tensor shape!")
                    output_shape = jnp.zeros((original_inputs_first_dim, HD // tp_size), dtype=intermediate_output.dtype)

                    input_offsets, send_sizes, output_offsets, recv_sizes = get_all_to_all_params(
                        reshaped_group_sizes,
                        expert_shard_id,
                        ep_size,
                        is_batch_sharded=False,
                    )

                    intermediate_output = jax.lax.ragged_all_to_all(
                        intermediate_output,
                        output_shape,
                        input_offsets,
                        send_sizes,
                        output_offsets,
                        recv_sizes,
                        axis_name=expert_axis_name,
                    )

                output = unpermute(
                    intermediate_output,
                    sorted_selected_experts,
                    weights,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    use_custom_sort_vjp=True,
                    weight_modif_fn=output_weights_hook,
                    num_experts_per_tok=self.num_experts_per_tok,
                    dtype=self.dtype,
                )
            return output

        @partial(
            shard_map,
            mesh=expert_mesh.jax_mesh,
            in_specs=(input_ps, glps, wikps, wukps, wgukps, wdkps, wibps, wubps, wgubps, wdbps),
            out_specs=output_ps,
            check_vma=False,
        )
        def _sparse_call(*call_args):
            """Run the fused body, then match the mode-aware boundary layout.

            In decode-replicated mode the body's tp-sharded ``[B, S, H/tp]``
            result is all-gathered back to full hidden so the block's output
            matches the replicated decode residual stream declared by
            ``output_ps``.
            """
            out = _sparse_call_body(*call_args)
            if decode_replicated_out:
                out = jax.lax.all_gather(out, tensor_axis_name, axis=2, tiled=True)
            return out

        output = _sparse_call(
            hidden_state,
            gate_logits,
            wi_kernel,
            wu_kernel,
            gate_up_kernel,
            wd_kernel,
            wi_bias,
            wu_bias,
            gate_up_bias,
            wd_bias,
        )

        # Reshard output back to original 5D mesh for compatibility with rest of model
        # This ensures the output can be used in residual connections
        original_output_ps = self.runtime_sharding_resolver.resolve(
            axes=(
                [DP, EMPTY, EMPTY] if (self.config.use_expert_tensor_mode or decode_replicated_out) else [DP, EMPTY, TP]
            ),
            mode=MODE_TRAIN,
            shape=output.shape,
        )
        if len(batch_axis_names) > 1:
            # The shard_map output carries the batch over (dp, fsdp); keep the
            # boundary constraint on the same axes instead of the resolver's
            # dp-only spec, which would all-gather the batch over fsdp here.
            original_output_ps = jax.sharding.PartitionSpec(batch_axis_names, *tuple(original_output_ps)[1:])
        # # @erfanzar NOTE: spx.with_sharding_constraint (NOT jax.lax.*) so the
        # constraint is MPMD-aware and the spec lands on the resolved
        # stage-local submesh.
        if self.config.use_sharding_constraint:
            output = spx.with_sharding_constraint(output, original_output_ps, mesh=self.mesh)

        return output, prein_gate_logits

    def moe_call(
        self,
        hidden_state: jax.Array,  # [B, S, H]
        gate_layer: spx.Module,
        expert_layer: spx.Module,
        wi_kernel: jax.Array | None = None,  # [E, H, M]
        wu_kernel: jax.Array | None = None,  # [E, H, M]
        wd_kernel: jax.Array | None = None,  # [E, M, H]
        wi_bias: jax.Array | None = None,  # [E, H]
        wu_bias: jax.Array | None = None,  # [E, H]
        wd_bias: jax.Array | None = None,  # [E, M]
        gate_up_kernel: jax.Array | None = None,  # [E, H, 2M]
        gate_up_bias: jax.Array | None = None,  # [E, 2M]
        ffn_activation: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        reform_router_probs_fn: typing.Callable[[jax.Array], jax.Array] | None = None,
        hooks: MoeFusedHooks | None = None,
        gate_hidden_state: jax.Array | None = None,
        *,
        act_fn: Callable[[jax.Array], jax.Array],
        output_metrics: bool = False,
        layer_idx: int | None = None,
    ):
        """Dispatch to the configured MoE execution path with auto-configured hooks.

        Reads ``self.module_moe_method`` and routes to one of three engines:
        :meth:`_moe_call_standard`, :meth:`_sparse_moe_call` (the fused
        grouped-matmul path, recommended), or :meth:`_moe_call_dense`. Before
        dispatch, :meth:`_configure_hooks_for_routing_strategy` installs the
        default ``select_hook`` for the active routing strategy if the user
        has not already supplied one. The whole call is wrapped in the
        auto-resharding 3-D expert mesh.

        **Hook Auto-Configuration:**
            - **TOP_K**: Normalizes weights by their sum (softmax-like distribution).
            - **TOP_K_NDIV**: Passes weights through unchanged (raw logit values).
            - **SWITCH**: Enforces hard assignment with weight = 1.0.
            - **EMPTY_CHOICE**: Uniform weights across expert selections.
            - **HASH**: Uniform weights for deterministic assignments.

            User-provided hooks on ``self.moe_hooks`` are preserved and override
            the defaults.

        Args:
            hidden_state: Input hidden states fed to the expert FFNs.
                Shape ``[B, S, H]``.
            gate_layer: Router/gate module mapping ``H -> E`` (produces logits).
            expert_layer: Expert layer module consumed by the standard
                (non-fused) path.
            wi_kernel: Expert gate-projection kernel. Shape ``[E, H, M]``.
                ``None`` when ``gate_up_kernel`` is provided instead.
            wu_kernel: Expert up-projection kernel. Shape ``[E, H, M]``.
                ``None`` when ``gate_up_kernel`` is provided instead.
            wd_kernel: Expert down-projection kernel. Shape ``[E, M, H]``.
                Required.
            wi_bias: Optional bias for the gate projection. Shape ``[E, H]``.
            wu_bias: Optional bias for the up projection. Shape ``[E, H]``.
            wd_bias: Optional bias for the down projection. Shape ``[E, M]``.
            gate_up_kernel: Optional fused gate/up kernel of shape
                ``[E, H, 2M]``; mutually exclusive with the split
                ``wi_kernel`` / ``wu_kernel`` pair.
            gate_up_bias: Optional bias for the fused gate/up projection.
                Shape ``[E, 2M]``.
            ffn_activation: Optional custom activation combining
                ``(w0, w1) -> output``. Defaults to ``act_fn(w0) * w1``.
            reform_router_probs_fn: Optional callable applied to router
                probabilities; used by the standard path only.
            hooks: Optional override for ``self.moe_hooks``.
            gate_hidden_state: Optional alternate hidden state fed to the
                gate (e.g. for "shared expert as router" architectures).
                Defaults to ``hidden_state``.
            act_fn: Activation function used when ``ffn_activation`` is
                ``None``.
            output_metrics: When ``True`` and the standard path is taken,
                returns :class:`MoeMetrics` instead of raw logits.
            layer_idx: Optional layer index used by the standard path for
                ``jax.debug.print`` logging.

        Returns:
            Tuple ``(output, aux)`` where ``output`` has shape ``[B, S, H]``
            and ``aux`` is the pre-softmax router logits of shape ``[B*S, E]``
            (or :class:`MoeMetrics` when ``output_metrics`` is requested via
            the standard path).

        Raises:
            ValueError: If ``wd_kernel`` is ``None``.
            NotImplementedError: If ``self.module_moe_method`` is not one of
                the three known :class:`MoEMethods` values.
        """
        if wd_kernel is None:
            raise ValueError("MoE requires wd_kernel.")
        if (
            any(isinstance(_k, tuple) for _k in (wi_kernel, wu_kernel, gate_up_kernel, wd_kernel) if _k is not None)
            and self.module_moe_method != MoEMethods.FUSED_MOE
        ):
            raise NotImplementedError(
                "Quantized (codes, scales) expert kernels are only supported on the FUSED_MOE path."
            )
        with self._active_auto_expert_mesh(hidden_state):
            match self.module_moe_method:
                case MoEMethods.STANDARD_MOE:
                    logger.warn_once(
                        "You are using MoEMethods.STANDARD_MOE which is not really recommended please switch to FUSED_MOE"
                    )
                    return self._moe_call_standard(
                        gate_layer=gate_layer,
                        expert_layer=expert_layer,
                        hidden_state=hidden_state,
                        output_metrics=output_metrics,
                        validate_inputs=False,
                        apply_capacity_constraint=False,
                        reform_router_probs_fn=reform_router_probs_fn,
                        layer_idx=layer_idx,
                    )
                case MoEMethods.FUSED_MOE:
                    return self._sparse_moe_call(
                        hidden_state=hidden_state,
                        gate_hidden_state=gate_hidden_state,
                        gate_layer=gate_layer,
                        wi_kernel=wi_kernel,
                        wu_kernel=wu_kernel,
                        wd_kernel=wd_kernel,
                        wi_bias=wi_bias,
                        wu_bias=wu_bias,
                        wd_bias=wd_bias,
                        gate_up_kernel=gate_up_kernel,
                        gate_up_bias=gate_up_bias,
                        act_fn=act_fn,
                        ffn_activation=ffn_activation,
                        hooks=hooks,
                    )
                case MoEMethods.DENSE_MOE:
                    logger.warn_once(
                        "You are using MoEMethods.DENSE_MOE which is not really recommended please switch to FUSED_MOE"
                    )
                    return self._moe_call_dense(
                        hidden_state=hidden_state,
                        gate_hidden_state=gate_hidden_state,
                        gate_layer=gate_layer,
                        wi_kernel=wi_kernel,
                        wu_kernel=wu_kernel,
                        wd_kernel=wd_kernel,
                        wi_bias=wi_bias,
                        wu_bias=wu_bias,
                        wd_bias=wd_bias,
                        gate_up_kernel=gate_up_kernel,
                        gate_up_bias=gate_up_bias,
                        act_fn=act_fn,
                        ffn_activation=ffn_activation,
                        hooks=hooks,
                    )
                case _:
                    raise NotImplementedError()

    def _moe_call_dense(
        self,
        hidden_state: jax.Array,  # [B, S, H]
        gate_layer: spx.Module,
        wi_kernel: jax.Array | None,  # [E, H, M]
        wu_kernel: jax.Array | None,  # [E, H, M]
        wd_kernel: jax.Array,  # [E, M, H]
        wi_bias: jax.Array | None = None,  # [E, M]
        wu_bias: jax.Array | None = None,  # [E, M]
        wd_bias: jax.Array | None = None,  # [E, H]
        gate_up_kernel: jax.Array | None = None,  # [E, H, 2M]
        gate_up_bias: jax.Array | None = None,  # [E, 2M]
        ffn_activation: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        gate_hidden_state: jax.Array | None = None,  # [B, S, H]
        hooks: MoeFusedHooks | None = None,
        *,
        act_fn: Callable[[jax.Array], jax.Array],
        capacity_factor: float | None = None,
        output_metrics: bool = False,
    ):
        """Dense MoE forward pass using per-token batched matmuls.

        This method implements MoE using dense einsum operations instead of ragged
        or grouped matrix multiplications. For each token, it selects the top-k
        expert weight matrices and computes the expert outputs via batched
        operations.

        **Key Differences from Fused MoE:**
            - Uses jnp.take to select expert weights per token
            - Computes via einsum with token-local expert batching
            - More memory intensive but works without grouped matmul kernels
            - Good for debugging or when Pallas kernels are unavailable

        **Algorithm:**
            1. Compute router logits and probabilities
            2. Select top-k experts per token
            3. For each token, gather weights for selected experts
            4. Compute expert FFN: act(x @ wi) * (x @ wu) @ wd
            5. Combine expert outputs with weighted sum

        **Hook Integration:**
            Uses the MoeFusedHooks system for customization:
            - before_gate: Preprocess hidden states
            - after_gate: Postprocess gate logits
            - before_topk: Modify router probs before selection
            - select_hook: Custom expert selection
            - refine_weights_hook: Modify selected weights
            - before_wo: Modify intermediate before output projection
            - after_wo: Modify expert outputs
            - before_combine: Modify outputs/weights before combination
            - finalize_output: Final output postprocessing

        Args:
            hidden_state: Input tensor. Shape: [B, S, H].
            gate_layer: Router module mapping H -> E (produces logits).
            wi_kernel: Expert W_i (gate) kernel. Shape: [E, H, M].
            wu_kernel: Expert W_u (up) kernel. Shape: [E, H, M].
            wd_kernel: Expert W_d (down) kernel. Shape: [E, M, H].
            wi_bias: Optional bias for W_i. Shape: [E, M].
            wu_bias: Optional bias for W_u. Shape: [E, M].
            wd_bias: Optional bias for W_d. Shape: [E, H].
            ffn_activation: Optional custom activation combining (w0, w1) -> output.
                If None, uses act_fn(w0) * w1 (gated activation).
            act_fn: Activation function used when ffn_activation is not provided.
            capacity_factor: If provided, applies soft capacity constraints
                to limit tokens per expert.
            output_metrics: If True, returns MoeMetrics instead of router logits.

        Returns:
            Tuple of (output, metrics_or_logits) where:
                - output: MoE layer output. Shape: [B, S, H].
                - metrics_or_logits: MoeMetrics if output_metrics=True,
                  else pre-softmax router logits with shape [B*S, E].

        Note:
            This method is slower than fused MoE but useful for debugging
            and as a fallback when grouped matmul kernels are unavailable.
        """
        if gate_up_kernel is None and (wi_kernel is None or wu_kernel is None):
            raise ValueError("MoE requires either gate_up_kernel or both wi_kernel and wu_kernel.")
        hooks = self._configure_hooks_for_routing_strategy() if hooks is None else hooks

        hidden_state = hidden_state.astype(self.dtype)
        gate_hidden_state = hidden_state if gate_hidden_state is None else gate_hidden_state.astype(self.dtype)
        if hooks.before_gate is not None:
            gate_hidden_state = hooks.before_gate(gate_hidden_state)

        batch_size, seq_len, hidden_dim = hidden_state.shape
        tokens = batch_size * seq_len
        hidden_flat = hidden_state.reshape(tokens, hidden_dim)
        # Gate input keeps its own width; see `_sparse_moe_call` for why this is
        # not `hidden_dim` (the expert width).
        gate_hidden_flat = gate_hidden_state.reshape(tokens, gate_hidden_state.shape[-1])

        prein_gate_logits = gate_layer(gate_hidden_flat)
        gate_logits = prein_gate_logits
        if hooks.after_gate is not None:
            gate_logits = hooks.after_gate(gate_logits)

        router_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        if hooks.before_topk is not None:
            router_probs = hooks.before_topk(router_probs)

        selected_weights, selected_experts = get_experts_location(
            gate_logits=router_probs,
            pre_bias_logits=None,
            select_hook=hooks.select_hook,
            refine_weights_hook=hooks.refine_weights_hook,
            num_experts_per_tok=self.num_experts_per_tok,
        )

        weights = selected_weights.astype(self.dtype)
        experts = selected_experts.astype(jnp.int32)

        if capacity_factor is not None and capacity_factor > 0:
            weights_shaped = weights.reshape(batch_size, seq_len, self.num_experts_per_tok)
            experts_shaped = experts.reshape(batch_size, seq_len, self.num_experts_per_tok)
            weights_shaped = self._apply_capacity_mask(experts_shaped, weights_shaped, capacity_factor)
            weights = weights_shaped.reshape(tokens, self.num_experts_per_tok)

        weight_sum = jnp.sum(weights, axis=-1, keepdims=True)
        weights = jnp.where(weight_sum > 0, weights / weight_sum, weights)

        if ffn_activation is None:

            def ffn_activation(x0: jax.Array, x1: jax.Array) -> jax.Array:
                """Default SwiGLU-style FFN activation: ``act_fn(x0) * x1``.

                Args:
                    x0: Gate-projection output.
                    x1: Up-projection output.

                Returns:
                    Element-wise ``act_fn(x0) * x1``, same shape as inputs.
                """
                return act_fn(x0) * x1

        precision = getattr(self, "precision", None)
        hidden_expanded = hidden_flat[:, None, :]

        if gate_up_kernel is not None:
            gate_up_sel = jnp.take(gate_up_kernel.astype(self.dtype), experts, axis=0)
            gate_up = jnp.einsum("tkh,tkhm->tkm", hidden_expanded, gate_up_sel, precision=precision)
            if gate_up_bias is not None:
                gate_up = gate_up + jnp.take(gate_up_bias.astype(self.dtype), experts, axis=0)
            w0, w1 = jnp.split(gate_up, 2, axis=-1)
        else:
            wi_sel = jnp.take(wi_kernel.astype(self.dtype), experts, axis=0)
            w0 = jnp.einsum("tkh,tkhm->tkm", hidden_expanded, wi_sel, precision=precision)
            if wi_bias is not None:
                w0 = w0 + jnp.take(wi_bias.astype(self.dtype), experts, axis=0)

            wu_sel = jnp.take(wu_kernel.astype(self.dtype), experts, axis=0)
            w1 = jnp.einsum("tkh,tkhm->tkm", hidden_expanded, wu_sel, precision=precision)
            if wu_bias is not None:
                w1 = w1 + jnp.take(wu_bias.astype(self.dtype), experts, axis=0)

        intermediate = ffn_activation(w0, w1)
        if hooks.before_wo is not None:
            intermediate = hooks.before_wo(intermediate)

        wd_sel = jnp.take(wd_kernel.astype(self.dtype), experts, axis=0)
        outputs = jnp.einsum("tkm,tkmh->tkh", intermediate, wd_sel, precision=precision)
        if wd_bias is not None:
            outputs = outputs + jnp.take(wd_bias.astype(self.dtype), experts, axis=0)

        if hooks.after_wo is not None:
            outputs = hooks.after_wo(outputs)

        if hooks.before_combine is not None:
            outputs, weights = hooks.before_combine(outputs, weights)

        combined = jnp.sum(outputs * weights[..., None], axis=1)
        output = combined.reshape(batch_size, seq_len, hidden_dim)

        if hooks.finalize_output is not None:
            output = hooks.finalize_output(output)

        if output_metrics:
            expert_mask = (weights > 0).astype(self.dtype)
            expert_loads = jnp.bincount(
                experts.reshape(-1),
                weights=expert_mask.reshape(-1),
                length=self.n_routed_experts,
            ).astype(self.dtype)
            metrics = self._compute_metrics(
                router_logits=prein_gate_logits,
                router_probs=router_probs,
                selected_experts=experts,
                selected_weights=weights,
                expert_loads=expert_loads,
            )
            return output, metrics

        return output, prein_gate_logits

    def _configure_hooks_for_routing_strategy(self) -> MoeFusedHooks:
        """Resolve the effective hooks for the current routing strategy.

        This method ensures each routing strategy has appropriate hook configuration
        without requiring manual setup. Only fills in a default ``refine_weights_hook``
        if one has not been explicitly configured by the user.

        The resolved :class:`MoeFusedHooks` is *returned* rather than written back to
        ``self.moe_hooks``: assigning a new attribute on an :class:`spx.Module` bumps
        the graph epoch, which is an illegal structural mutation when the forward runs
        under a readonly transform (e.g. the jitted training step). Callers thread the
        returned value through instead of relying on a side effect.

        **Hook Configuration by Strategy:**

            TOP_K: Sets `select_hook` to normalize weights by their sum.
                Ensures expert weights sum to 1.0 for proper weighted combination.

            TOP_K_NDIV: Sets `select_hook` to pass through weights unchanged.
                Uses raw logit values without normalization.

            SWITCH: Sets `select_hook` to enforce hard assignment (weight = 1.0).
                Only one expert gets non-zero weight.

            EMPTY_CHOICE: Sets `select_hook` to normalize per-expert selections.
                Each expert receives equal contribution from selected tokens.

            HASH: Sets `select_hook` to uniform weight distribution.
                All assigned experts get equal weight (1/k).
        """
        # Only set default refine_weights_hook if one wasn't already configured by the user.
        if self.moe_hooks.refine_weights_hook is not None:
            return self.moe_hooks

        refine_weights_hook = None
        if self.routing_strategy == MoeRoutingStrategy.TOP_K:
            # TOP_K: Normalize weights by their sum (softmax-like)
            if self.moe_hooks.select_hook is None:

                def normalize_selected_weights(weights: jax.Array) -> jax.Array:
                    """Normalize top-k expert weights by their sum.

                    Ensures weights for each token sum to 1.0, creating a proper
                    probability distribution over selected experts.
                    """
                    # Match HF Mixtral-style behavior: for top-1 routing keep the selected
                    # probability as-is; for top-k>1 normalize by the sum across selected experts.
                    if weights.shape[-1] <= 1:
                        return weights
                    return weights / jnp.maximum(weights.sum(axis=-1, keepdims=True), 1e-8)

                refine_weights_hook = normalize_selected_weights

        elif self.routing_strategy == MoeRoutingStrategy.TOP_K_NDIV:
            # TOP_K_NDIV: Use weights as-is (raw logits, no normalization)
            if self.moe_hooks.select_hook is None:

                def passthrough_weights(weights: jax.Array) -> jax.Array:
                    """Pass through weights unchanged.

                    For TOP_K_NDIV routing, weights are used as raw logit values
                    without normalization. This allows unnormalized combinations.
                    """
                    return weights

                refine_weights_hook = passthrough_weights

        elif self.routing_strategy == MoeRoutingStrategy.SWITCH:
            # SWITCH: Hard assignment - single expert gets weight 1.0, others 0.0
            if self.moe_hooks.select_hook is None:

                def hard_assignment_weights(weights: jax.Array) -> jax.Array:
                    """Enforce hard assignment for SWITCH routing.

                    Only one expert per token is selected (top-1). This hook ensures
                    the weight is exactly 1.0, creating a hard (non-differentiable)
                    expert assignment.
                    """
                    return jnp.ones_like(weights)

                refine_weights_hook = hard_assignment_weights

        elif self.routing_strategy == MoeRoutingStrategy.EMPTY_CHOICE:
            # EMPTY_CHOICE: Expert-driven selection - normalize per expert
            if self.moe_hooks.select_hook is None:

                def expert_choice_weights(weights: jax.Array) -> jax.Array:
                    """Normalize weights for Expert Choice routing.

                    In Expert Choice routing, each expert selects its own top-k tokens.
                    This hook ensures proper weight distribution across the selected
                    tokens for each expert.
                    """
                    # For expert choice, normalize differently - each expert's selections
                    # should have equal contribution
                    num_experts_selected = weights.shape[-1]
                    return jnp.ones_like(weights) / jnp.maximum(num_experts_selected, 1)

                refine_weights_hook = expert_choice_weights

        elif self.routing_strategy == MoeRoutingStrategy.HASH:
            # HASH: Deterministic routing - uniform weights for all assigned experts
            if self.moe_hooks.select_hook is None:

                def uniform_weights(weights: jax.Array) -> jax.Array:
                    """Uniform weights for hash-based routing.

                    In hash-based routing, tokens are deterministically assigned to experts
                    based on token ID. Each expert in the assignment group gets equal weight.
                    """
                    num_experts_per_token = weights.shape[-1]
                    return jnp.ones_like(weights) / jnp.maximum(num_experts_per_token, 1)

                refine_weights_hook = uniform_weights

        if refine_weights_hook is not None:
            return self.moe_hooks.replace(refine_weights_hook=refine_weights_hook)
        return self.moe_hooks

    def _moe_call_standard(
        self,
        gate_layer: spx.Module,
        expert_layer: spx.Module,
        hidden_state: jax.Array,
        output_metrics: bool = False,
        validate_inputs: bool = False,
        apply_capacity_constraint: bool = False,
        reform_router_probs_fn: typing.Callable[[jax.Array], jax.Array] | None = None,
        layer_idx: int | None = None,
    ) -> tuple[jax.Array, MoeMetrics | jax.Array]:
        """Standard MoE forward pass: routing, permutation, expert computation, and combining.

        This method uses the MoeFusedHooks system to allow custom interventions at key
        points during execution:

        **Hook Integration:**
            - `before_gate`: Applied before gate/router computation.
            - `after_gate`: Applied after gate logits computation.
            - `before_topk`: Applied before expert selection (top-k).
            - `refine_weights_hook`: Refines expert weights after selection.
            - `refine_inputs_hook`: Refines token representations before expert computation.
            - `before_combine`: Applied before combining expert outputs.
            - `finalize_output`: Applied to the final output.

        Args:
            gate_layer: Router module mapping hidden states to expert logits.
            expert_layer: Expert layer module for computing expert outputs.
            hidden_state: Input tensor. Shape: [B, S, H].
            output_metrics: Whether to return detailed MoE metrics.
            validate_inputs: Whether to validate input shapes.
            apply_capacity_constraint: Whether to apply capacity constraints.
            reform_router_probs_fn: Optional function to modify router probabilities.

        Returns:
            Tuple of (output, metrics_or_logits) where:
            - output: MoE layer output. Shape: [B, S, H].
            - metrics_or_logits: MoeMetrics if output_metrics=True, else router_logits.
        """
        hooks = self._configure_hooks_for_routing_strategy()

        hidden_state = hidden_state.astype(self.dtype)
        if hooks.before_gate is not None:
            hidden_state = hooks.before_gate(hidden_state)

        batch_size, seq_len, hidden_size = hidden_state.shape
        hidden_state_flat = hidden_state.reshape(-1, hidden_size)

        router_logits = gate_layer(hidden_state_flat).astype(jnp.promote_types(self.dtype, jnp.float32))

        # Store original logits BEFORE any hooks - used for expert selection (matching HF behavior).
        prein_gate_logits = router_logits

        # after_gate hook produces scattered probs for aux loss/logging, but we use original logits for selection.
        if hooks.after_gate is not None:
            router_probs = hooks.after_gate(router_logits)
        else:
            router_probs = jax.nn.softmax(router_logits, axis=-1)

        if reform_router_probs_fn is not None:
            router_probs = reform_router_probs_fn(router_probs)

        if hooks.before_topk is not None:
            router_probs = hooks.before_topk(router_probs)

        if validate_inputs:
            self._validate_routing_inputs(hidden_state, router_logits)

        # Use original logits for expert selection (top-k on logits, then softmax on k selected via refine_weights_hook).
        # This matches HuggingFace behavior where top-k is done on pre-softmax logits.
        selected_weights, selected_experts = get_experts_location(
            gate_logits=prein_gate_logits,
            pre_bias_logits=None,
            select_hook=hooks.select_hook,
            refine_weights_hook=hooks.refine_weights_hook,
            num_experts_per_tok=self.num_experts_per_tok,
        )

        if layer_idx is not None:
            # Get top-k logits (before softmax) for comparison
            top_k_logits_pre, _ = jax.lax.top_k(prein_gate_logits, self.num_experts_per_tok)
            jax.debug.print("  [ED Router L{}] logits[0]: {}", layer_idx, prein_gate_logits[0])
            jax.debug.print(
                "  [ED Router L{}] top_idx[0]: {}, top_logits[0]: {}",
                layer_idx,
                selected_experts[0],
                top_k_logits_pre[0],
            )
            jax.debug.print(
                "  [ED Router L{}] top_weights[0]: {} (sum={})",
                layer_idx,
                selected_weights[0],
                selected_weights[0].sum(),
            )
            jax.debug.print("  [ED Experts L{}] input[0,:5]: {}", layer_idx, hidden_state_flat[0, :5])

        if apply_capacity_constraint:
            selected_experts, selected_weights = self._apply_capacity_constraint(selected_experts, selected_weights)

        if hooks.refine_inputs_hook is not None:
            hidden_state_flat = hooks.refine_inputs_hook(
                hidden_state_flat,
                selected_weights,
                (batch_size, seq_len, hidden_size),
            )

        (
            sorted_inputs,
            sort_order,
            group_sizes,
            sorted_experts,
        ) = self._replicate_and_sort_tokens(hidden_state_flat, selected_experts)

        out_sorted = expert_layer(sorted_inputs, group_sizes, sorted_experts)
        out_unsorted = sort_activations(out_sorted, jnp.argsort(sort_order))
        out_unflat = out_unsorted.reshape(batch_size * seq_len, self.num_experts_per_tok, hidden_size)

        if hooks.before_combine is not None:
            out_unflat, selected_weights = hooks.before_combine(out_unflat, selected_weights)

        output = jnp.sum(out_unflat * selected_weights[..., None], axis=1).reshape(batch_size, seq_len, hidden_size)

        if layer_idx is not None:
            jax.debug.print("  [ED Experts L{}] output[0,:5]: {}", layer_idx, output.reshape(-1, hidden_size)[0, :5])

        if hooks.finalize_output is not None:
            output = hooks.finalize_output(output)

        if output_metrics:
            metrics = self._compute_metrics(
                router_logits,
                router_probs,
                selected_experts,
                selected_weights,
                group_sizes,
            )
            return output, metrics
        return output, router_logits

    @abstractmethod
    def forward(
        self,
        hidden_states: Float[Array, "batch seq hidden_dim"],
        **kwargs,
    ) -> tuple[Float[Array, "batch seq hidden_dim"], MoeMetrics]:
        """Run the MoE forward pass; concrete subclasses must implement this.

        Subclasses are expected to own their gate layer and expert kernels
        and to dispatch into :meth:`moe_call` (or build a custom routing
        pipeline on top of the helpers provided by :class:`BaseMoeModule`).

        Args:
            hidden_states: Input hidden states of shape ``[B, S, H]``.
            **kwargs: Subclass-specific extra arguments (e.g. attention
                outputs that get summed back in via a shared-expert path).

        Returns:
            Tuple ``(output, metrics)`` where ``output`` has the same shape
            as ``hidden_states`` and ``metrics`` is a :class:`MoeMetrics`
            holding routing diagnostics and auxiliary losses.
        """
        pass
