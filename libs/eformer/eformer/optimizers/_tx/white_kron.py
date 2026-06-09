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

from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec
from optax import tree_utils as otu
from optax._src import base, transform
from optax._src.combine import chain
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype

from eformer.pytree import auto_pytree, field

# Path type constants for parameter classification
DENSE_PATH = 0  # Parameters processed with full dense Kronecker factors
LARGE_PATH = 1  # Large parameters processed with mixed dense/diagonal factors
ONE_D_PATH = 2  # 1D parameters processed with diagonal-only factors


@auto_pytree
class DenseState:
    """State container for dense Kronecker-factored preconditioner blocks.

    This class stores the concatenated preconditioner matrices for all dense
    parameter blocks in the model. Dense blocks are small enough to use full
    matrix Kronecker factors rather than diagonal approximations.

    Attributes:
        Ql (jax.Array): Left Kronecker factors, shape [num_blocks, block_size, block_size].
            These are orthogonal/near-orthogonal matrices for left preconditioning.
        Qr (jax.Array): Right Kronecker factors, shape [num_blocks, block_size, block_size].
            These are orthogonal/near-orthogonal matrices for right preconditioning.
        Ll (jax.Array): Lipschitz estimates for left factors, shape [num_blocks].
            Used for adaptive learning rate scaling.
        Lr (jax.Array): Lipschitz estimates for right factors, shape [num_blocks].
            Used for adaptive learning rate scaling.
        valid_rows (jax.Array): Number of valid rows in each block, shape [num_blocks].
            Accounts for padding when original dimensions are not multiples of block_size.
        valid_cols (jax.Array): Number of valid columns in each block, shape [num_blocks].
            Accounts for padding when original dimensions are not multiples of block_size.
        valid_count (int): Number of actual (non-padding) blocks in the state.
        block_size (int): Size of each square block in the Kronecker factorization.
    """

    Ql: jax.Array
    Qr: jax.Array
    Ll: jax.Array
    Lr: jax.Array
    valid_rows: jax.Array
    valid_cols: jax.Array
    valid_count: int = field(pytree_node=False)
    block_size: int = field(pytree_node=False)


@auto_pytree
class LeafState:
    """State container for a single parameter leaf in the White Kron optimizer.

    This class stores the preconditioner state for individual parameter tensors,
    supporting different processing paths based on parameter size and shape.

    The optimizer handles three types of parameters:
        - DENSE_PATH: Small 2D parameters use full dense Kronecker factors
        - LARGE_PATH: Large 2D parameters use mixed dense/diagonal factors
        - ONE_D_PATH: 1D parameters use diagonal-only preconditioners

    Attributes:
        kind (int): Processing path type (DENSE_PATH, LARGE_PATH, or ONE_D_PATH).
        scanned (int): Whether this parameter is part of a scanned layer (0 or 1).
        B (int): Batch dimension size (number of stacked parameter matrices).
        shape (tuple[int, ...] | None): Original parameter shape (excluding batch dim).
        merged (tuple[int, ...] | None): Shape after merging dimensions to 2D (m, n).
        nr (int | None): Number of row blocks when using blocked processing.
        nc (int | None): Number of column blocks when using blocked processing.
        block_size (int | None): Block size for blocked processing.
        diag_left (bool | None): Whether left factor uses diagonal approximation.
        diag_right (bool | None): Whether right factor uses diagonal approximation.
        stack (int | None): Total number of stacked blocks for parallel processing.
        Ql (jax.Array | None): Left Kronecker factor(s).
        Qr (jax.Array | None): Right Kronecker factor(s).
        Ll (jax.Array | None): Left Lipschitz estimates.
        Lr (jax.Array | None): Right Lipschitz estimates.
        valid_rows (jax.Array | None): Valid row counts per block.
        valid_cols (jax.Array | None): Valid column counts per block.
    """

    kind: int = field(pytree_node=False)
    scanned: int = field(pytree_node=False)
    B: int = field(pytree_node=False)
    shape: tuple[int, ...] | None = field(pytree_node=False, default=None)
    merged: tuple[int, ...] | None = field(pytree_node=False, default=None)
    nr: int | None = field(pytree_node=False, default=None)
    nc: int | None = field(pytree_node=False, default=None)
    block_size: int | None = field(pytree_node=False, default=None)
    diag_left: bool | None = field(pytree_node=False, default=None)
    diag_right: bool | None = field(pytree_node=False, default=None)
    stack: int | None = field(pytree_node=False, default=None)
    Ql: jax.Array | None = None
    Qr: jax.Array | None = None
    Ll: jax.Array | None = None
    Lr: jax.Array | None = None
    valid_rows: jax.Array | None = None
    valid_cols: jax.Array | None = None


def _def_scale(
    lr_style: str | None = "adam",
    b1: float = 0.95,
    normalize_grads: bool = False,
    max_size_dense: int = 16384,
    preconditioner_lr: float = 0.7,
    preconditioner_init_scale: float = 1.0,
    preconditioner_update_style: Literal["QUAD", "skew"] = "skew",
    dtype: str | jnp.dtype = jnp.bfloat16,
    scanned_layers: base.Params | None = None,
    block_size: int = 256,
    pipeline_axis_name: str | None = None,
    pipeline_axis_size: int = 1,
    params_partition_specs: PartitionSpec | list | tuple | dict | None = None,
    noise_scale: float = 1e-9,
) -> base.GradientTransformation:
    """Create a White Kron gradient scaling transformation.

    This is the core implementation for White Kron optimizers (Quad and Skew).
    It applies Kronecker-factored preconditioning to gradients using an online
    learning approach to maintain the preconditioner matrices.

    The optimizer classifies parameters into three categories:
        - Dense: Small 2D parameters (both dims <= max_size_dense) use full matrix factors
        - Large: Large 2D parameters use mixed dense/diagonal factors
        - 1D: Scalar and 1D parameters use diagonal-only preconditioning

    Args:
        lr_style (str | None): Learning rate scaling style. "adam" divides updates by 5.0
            for Adam-like behavior. None uses raw preconditioned gradients. Defaults to "adam".
        b1 (float): Momentum coefficient for exponential moving average of gradients.
            Set to 0 to disable momentum. Defaults to 0.95.
        normalize_grads (bool): Whether to normalize gradients to unit norm before
            preconditioning. Can help with stability. Defaults to False.
        max_size_dense (int): Maximum dimension size for dense Kronecker factors.
            Parameters with both dimensions <= this value use full matrix factors.
            Larger parameters use diagonal approximations. Defaults to 16384.
        preconditioner_lr (float): Learning rate for preconditioner updates.
            Controls how quickly the preconditioner adapts. Defaults to 0.7.
        preconditioner_init_scale (float): Initial scale for preconditioner matrices.
            Identity matrices are scaled by this value. Defaults to 1.0.
        preconditioner_update_style (Literal["QUAD", "skew"]): Update rule for
            preconditioners. "QUAD" uses quadratic updates, "skew" uses Procrustes
            orthogonalization. Defaults to "skew".
        dtype (str | jnp.dtype): Data type for preconditioner storage. Must be
            bfloat16 or float32. Defaults to jnp.bfloat16.
        scanned_layers (base.Params | None): PyTree indicating which parameters are
            from scanned layers (True) vs regular layers (False). Defaults to None.
        block_size (int): Block size for blocking large matrices. Controls memory
            and compute tradeoffs. Defaults to 256.
        pipeline_axis_name (str | None): Name of the pipeline parallel axis for
            sharding preconditioner state. Defaults to None.
        pipeline_axis_size (int): Size of the pipeline parallel axis. Defaults to 1.
        params_partition_specs (PartitionSpec | list | tuple | dict | None): Partition
            specs for model parameters. Used to maintain sharding of momentum. Defaults to None.
        noise_scale (float): Scale of random noise added to gradients for numerical
            stability. Defaults to 1e-9.

    Returns:
        base.GradientTransformation: Gradient transformation implementing White Kron
            preconditioning.

    Raises:
        ValueError: If dtype is not bfloat16 or float32.
        ValueError: If preconditioner_update_style is not "QUAD" or "skew".
    """
    dtype = canonicalize_dtype(dtype)
    if dtype not in (jnp.bfloat16, jnp.float32):
        raise ValueError("dtype must be bfloat16 or float32")
    if preconditioner_update_style not in ("QUAD", "skew"):
        raise ValueError("preconditioner_update_style must be QUAD or skew")

    def init_fn(params):
        params_unboxed = jax.tree.map(lambda x: x, params, is_leaf=lambda x: False)

        mu = None
        if b1 > 0:
            mu = jax.tree.map(lambda p: jnp.zeros_like(p, dtype=dtype), params_unboxed)
            if params_partition_specs is not None:
                mu = with_sharding_constraint(mu, params_partition_specs)

        scanned_flags = scanned_layers if scanned_layers is not None else jax.tree.map(lambda _: False, params_unboxed)

        dense_Ql_list: list[jax.Array] = []
        dense_Qr_list: list[jax.Array] = []
        dense_Ll_list: list[jax.Array] = []
        dense_Lr_list: list[jax.Array] = []

        dense_valid_rc: list[tuple[int, int]] = []

        large_state: list[Any] = []

        leaves, _tdef = jax.tree.flatten(params_unboxed)
        flags, _ = jax.tree.flatten(scanned_flags)

        for leaf, scanned in zip(leaves, flags, strict=False):
            p = leaf if scanned else leaf[None, ...]
            B = p.shape[0]
            shape_wo = p.shape[1:]

            merged = _merge_dims(shape_wo)
            if len(merged) <= 1:
                m_flat = int(np.prod(shape_wo)) if len(shape_wo) > 0 else 1
                Ql = jnp.ones((B, m_flat), dtype=dtype) * preconditioner_init_scale
                Ll = jnp.zeros((B,), jnp.float32)
                large_state.append(
                    LeafState(kind=ONE_D_PATH, scanned=int(scanned), B=B, shape=shape_wo, merged=(m_flat,), Ql=Ql, Ll=Ll)
                )
                continue

            m, n = merged
            is_large_m = m > max_size_dense
            is_large_n = n > max_size_dense
            is_dense = (not is_large_m) and (not is_large_n)

            if is_dense:
                nr, nc = (m + block_size - 1) // block_size, (n + block_size - 1) // block_size
                row_sizes = [block_size] * (nr - 1) + [m - block_size * (nr - 1) if nr > 0 else 0]
                col_sizes = [block_size] * (nc - 1) + [n - block_size * (nc - 1) if nc > 0 else 0]
                row_sizes = [rs if rs > 0 else block_size for rs in row_sizes]
                col_sizes = [cs if cs > 0 else block_size for cs in col_sizes]
                for _b in range(B):
                    for ri in range(nr):
                        for cj in range(nc):
                            vr, vc = row_sizes[ri], col_sizes[cj]
                            Ql = _identity_padded(block_size, vr, dtype) * preconditioner_init_scale
                            Qr = _identity_padded(block_size, vc, dtype) * preconditioner_init_scale
                            dense_Ql_list.append(Ql)
                            dense_Qr_list.append(Qr)
                            dense_Ll_list.append(jnp.zeros([], jnp.float32))
                            dense_Lr_list.append(jnp.zeros([], jnp.float32))
                            dense_valid_rc.append((vr, vc))
                large_state.append(
                    LeafState(
                        kind=DENSE_PATH, scanned=int(scanned), B=B, merged=(m, n), nr=nr, nc=nc, block_size=block_size
                    )
                )

            else:
                diag_left = is_large_m
                diag_right = is_large_n

                if diag_left and diag_right:
                    Ql = jnp.ones((B, m), dtype=dtype) * preconditioner_init_scale
                    Qr = jnp.ones((B, n), dtype=dtype) * preconditioner_init_scale
                    Ll = jnp.zeros((B,), jnp.float32)
                    Lr = jnp.zeros((B,), jnp.float32)
                    large_state.append(
                        LeafState(
                            kind=LARGE_PATH,
                            scanned=int(scanned),
                            B=B,
                            merged=(m, n),
                            diag_left=True,
                            diag_right=True,
                            Ql=Ql,
                            Qr=Qr,
                            Ll=Ll,
                            Lr=Lr,
                            stack=B,
                        )
                    )

                elif diag_left != diag_right:
                    block_rows = not diag_left
                    dim_to_block = m if block_rows else n
                    other_dim = n if block_rows else m

                    num_blocks_per_sample = (dim_to_block + block_size - 1) // block_size
                    stack = B * num_blocks_per_sample

                    Q_diag = jnp.broadcast_to(
                        jnp.ones((1, other_dim), dtype=dtype) * preconditioner_init_scale, (stack, other_dim)
                    )

                    Q_blocked_blocks = []
                    for _ in range(B):
                        for i in range(num_blocks_per_sample):
                            v = (
                                block_size
                                if i < num_blocks_per_sample - 1
                                else (
                                    dim_to_block - block_size * (num_blocks_per_sample - 1)
                                    if num_blocks_per_sample > 0
                                    else block_size
                                )
                            )
                            v = v if v > 0 else block_size
                            Q_blocked_blocks.append(_identity_padded(block_size, v, dtype) * preconditioner_init_scale)
                    Q_blocked = jnp.stack(Q_blocked_blocks, axis=0)

                    Ql = Q_blocked if block_rows else Q_diag
                    Qr = Q_diag if block_rows else Q_blocked
                    Ll = jnp.zeros((stack,), jnp.float32)
                    Lr = jnp.zeros((stack,), jnp.float32)

                    large_state.append(
                        LeafState(
                            kind=LARGE_PATH,
                            scanned=int(scanned),
                            B=B,
                            merged=(m, n),
                            diag_left=diag_left,
                            diag_right=diag_right,
                            Ql=Ql,
                            Qr=Qr,
                            Ll=Ll,
                            Lr=Lr,
                            stack=stack,
                            nr=num_blocks_per_sample if block_rows else None,
                            nc=num_blocks_per_sample if not block_rows else None,
                            block_size=block_size,
                        )
                    )
                else:
                    raise AssertionError("unexpected large case.")

        if dense_Ql_list:
            Ql_cat = jnp.stack(dense_Ql_list, axis=0)
            Qr_cat = jnp.stack(dense_Qr_list, axis=0)
            Ll_cat = jnp.stack(dense_Ll_list, axis=0)
            Lr_cat = jnp.stack(dense_Lr_list, axis=0)

            valid_rows = jnp.array([vr for (vr, _) in dense_valid_rc], dtype=jnp.int32)
            valid_cols = jnp.array([vc for (_, vc) in dense_valid_rc], dtype=jnp.int32)

            valid_count = Ql_cat.shape[0]
            if pipeline_axis_size > 1:
                pad = (-valid_count) % pipeline_axis_size
            else:
                pad = 0
            if pad > 0:
                eye = jnp.eye(block_size, dtype=dtype)
                Ql_pad = jnp.broadcast_to(eye, (pad, block_size, block_size))
                Qr_pad = jnp.broadcast_to(eye, (pad, block_size, block_size))
                Ll_pad = jnp.ones((pad,), jnp.float32)
                Lr_pad = jnp.ones((pad,), jnp.float32)
                Ql_cat = jnp.concatenate([Ql_cat, Ql_pad], axis=0)
                Qr_cat = jnp.concatenate([Qr_cat, Qr_pad], axis=0)
                Ll_cat = jnp.concatenate([Ll_cat, Ll_pad], axis=0)
                Lr_cat = jnp.concatenate([Lr_cat, Lr_pad], axis=0)
                valid_rows = jnp.concatenate([valid_rows, jnp.full((pad,), block_size, jnp.int32)], axis=0)
                valid_cols = jnp.concatenate([valid_cols, jnp.full((pad,), block_size, jnp.int32)], axis=0)

            if pipeline_axis_name is not None:
                Ql_cat = with_sharding_constraint(Ql_cat, PartitionSpec(pipeline_axis_name))
                Qr_cat = with_sharding_constraint(Qr_cat, PartitionSpec(pipeline_axis_name))
                Ll_cat = with_sharding_constraint(Ll_cat, PartitionSpec(pipeline_axis_name))
                Lr_cat = with_sharding_constraint(Lr_cat, PartitionSpec(pipeline_axis_name))
                valid_rows = with_sharding_constraint(valid_rows, PartitionSpec(pipeline_axis_name))
                valid_cols = with_sharding_constraint(valid_cols, PartitionSpec(pipeline_axis_name))
            dense_state = DenseState(
                Ql=Ql_cat,
                Qr=Qr_cat,
                Ll=Ll_cat,
                Lr=Lr_cat,
                valid_rows=valid_rows,
                valid_cols=valid_cols,
                valid_count=int(valid_count),
                block_size=int(block_size),
            )
        else:
            dense_state = None

        for i, st in enumerate(large_state):
            if st.kind != LARGE_PATH:
                continue

            updates = {}
            current_Ql, current_Qr, current_Ll, current_Lr = st.Ql, st.Qr, st.Ll, st.Lr
            current_stack = st.stack
            m, n = st.merged

            if pipeline_axis_size > 1:
                pad = (-current_stack) % pipeline_axis_size
            else:
                pad = 0

            if pad > 0:
                if st.diag_left and st.diag_right:
                    current_Ql = jnp.pad(st.Ql, ((0, pad), (0, 0)), constant_values=1.0)
                    current_Qr = jnp.pad(st.Qr, ((0, pad), (0, 0)), constant_values=1.0)
                elif st.diag_left and (not st.diag_right):
                    eye = jnp.eye(st.block_size, dtype=dtype)
                    current_Ql = jnp.pad(st.Ql, ((0, pad), (0, 0)), constant_values=1.0)
                    current_Qr = jnp.concatenate([st.Qr, jnp.broadcast_to(eye, (pad, eye.shape[0], eye.shape[1]))], 0)
                elif (not st.diag_left) and st.diag_right:
                    eye = jnp.eye(st.block_size, dtype=dtype)
                    current_Ql = jnp.concatenate([st.Ql, jnp.broadcast_to(eye, (pad, eye.shape[0], eye.shape[1]))], 0)
                    current_Qr = jnp.pad(st.Qr, ((0, pad), (0, 0)), constant_values=1.0)
                else:
                    raise AssertionError
                current_Ll = jnp.pad(st.Ll, ((0, pad),), constant_values=1.0)
                current_Lr = jnp.pad(st.Lr, ((0, pad),), constant_values=1.0)
                current_stack += pad

            updates["Ql"] = current_Ql
            updates["Qr"] = current_Qr
            updates["Ll"] = current_Ll
            updates["Lr"] = current_Lr
            updates["stack"] = current_stack

            if st.diag_left and st.diag_right:
                current_valid_rows = jnp.full((current_stack,), m, jnp.int32)
                current_valid_cols = jnp.full((current_stack,), n, jnp.int32)
            elif st.diag_left != st.diag_right:
                block_rows = not st.diag_left
                num_blocks_per_sample = st.nr if block_rows else st.nc
                dim_to_block = m if block_rows else n
                other_dim = n if block_rows else m

                if num_blocks_per_sample and num_blocks_per_sample > 0:
                    last_block_v = dim_to_block - st.block_size * (num_blocks_per_sample - 1)
                    v_one_sample = (
                        jnp.full((num_blocks_per_sample,), st.block_size, dtype=jnp.int32).at[-1].set(last_block_v)
                    )
                else:
                    v_one_sample = jnp.array([], dtype=jnp.int32)
                v_all_samples = jnp.tile(v_one_sample, st.B)

                if v_all_samples.shape[0] < current_stack:
                    p = current_stack - v_all_samples.shape[0]
                    pad_vals = jnp.full((p,), st.block_size, dtype=jnp.int32)
                    v_all_samples = jnp.concatenate([v_all_samples, pad_vals], axis=0)

                other_dim_arr = jnp.full_like(v_all_samples, other_dim)
                if block_rows:
                    current_valid_rows = v_all_samples
                    current_valid_cols = other_dim_arr
                else:
                    current_valid_rows = other_dim_arr
                    current_valid_cols = v_all_samples
            else:
                raise AssertionError

            if pipeline_axis_name is not None:
                updates["Ql"] = with_sharding_constraint(updates["Ql"], PartitionSpec(pipeline_axis_name))
                updates["Qr"] = with_sharding_constraint(updates["Qr"], PartitionSpec(pipeline_axis_name))
                updates["Ll"] = with_sharding_constraint(updates["Ll"], PartitionSpec(pipeline_axis_name))
                updates["Lr"] = with_sharding_constraint(updates["Lr"], PartitionSpec(pipeline_axis_name))
                current_valid_rows = with_sharding_constraint(current_valid_rows, PartitionSpec(pipeline_axis_name))
                current_valid_cols = with_sharding_constraint(current_valid_cols, PartitionSpec(pipeline_axis_name))

            updates["valid_rows"] = current_valid_rows
            updates["valid_cols"] = current_valid_cols

            large_state[i] = st.replace(**updates)

        opt_state = dict(count=jnp.zeros([], jnp.int32), mu=mu, large=large_state)
        if dense_state is not None:
            opt_state["dense"] = dense_state
        return opt_state

    def update_fn(updates: base.Updates, state: dict, params: base.Params | None = None):
        step = safe_int32_increment(state["count"])
        plr = jnp.maximum(preconditioner_lr * jax.lax.rsqrt(1.0 + step / 10000.0), 0.4)
        balance = jnp.equal(step % 100, 0)

        if preconditioner_update_style == "QUAD":
            dense_update_fn = _dense_update
            diag_update_fn = _diag_update
        elif preconditioner_update_style == "skew":
            dense_update_fn = _dense_update_q0p5eq1p5
            diag_update_fn = _diag_update_q0p5eq1p5
        else:
            raise ValueError(f"Unknown preconditioner_update_style: {preconditioner_update_style}")

        mu = state["mu"]
        mupd = updates
        if mu is not None and b1 > 0:
            mu = otu.tree_update_moment(updates, mu, b1, 1)
            if params_partition_specs is not None:
                mu = with_sharding_constraint(mu, params_partition_specs)
            mupd = otu.tree_bias_correction(mu, b1, step)
        mu = otu.tree_cast(mu, dtype) if mu is not None else None
        mupd = otu.tree_cast(mupd, dtype)

        if normalize_grads:
            mupd = jax.tree.map(lambda g: g / (jnp.linalg.norm(g) + 1e-6), mupd)

        leaves_u, tdef_u = jax.tree.flatten(mupd)
        perleaf_state: list[Any] = state["large"]

        dense_state = state.get("dense")
        pg_dense_blocks: jax.Array | None = None
        dense_block_count = 0

        if dense_state is not None:
            blocks_list = []
            for leaf, st in zip(leaves_u, perleaf_state, strict=False):
                if st.kind != DENSE_PATH:
                    continue
                B = st.B
                m, n = st.merged
                nr, nc = st.nr, st.nc
                x2d = jnp.reshape(leaf, (B, m, n))
                current_block_size = dense_state.block_size
                blocks, _ = _block2d(x2d, current_block_size)
                blocks_list.append(blocks)
            if blocks_list:
                grads_cat = jnp.concatenate(blocks_list, axis=0)
                dense_block_count = grads_cat.shape[0]
                state_len = dense_state.Ql.shape[0]
                if dense_block_count < state_len:
                    pad = state_len - dense_block_count
                    grads_cat = jnp.concatenate(
                        [grads_cat, jnp.ones((pad, current_block_size, current_block_size), grads_cat.dtype)], axis=0
                    )
                elif dense_block_count > state_len:
                    raise ValueError(
                        "dense concatenation produced more blocks than q state. check block_size/grouping consistency."
                    )
                if pipeline_axis_name is not None:
                    grads_cat = with_sharding_constraint(grads_cat, PartitionSpec(pipeline_axis_name))

                key_dense = jax.random.fold_in(jax.random.PRNGKey(42), step)
                keys = jax.random.split(key_dense, grads_cat.shape[0])
                if pipeline_axis_name is not None:
                    keys = with_sharding_constraint(keys, PartitionSpec(pipeline_axis_name))

                diag_left = False
                diag_right = False
                valid_shape_dense = jnp.stack([dense_state.valid_rows, dense_state.valid_cols], axis=1)
                if pipeline_axis_name is not None:
                    valid_shape_dense = with_sharding_constraint(valid_shape_dense, PartitionSpec(pipeline_axis_name))
                    Ql_c = with_sharding_constraint(dense_state.Ql, PartitionSpec(pipeline_axis_name))
                    Qr_c = with_sharding_constraint(dense_state.Qr, PartitionSpec(pipeline_axis_name))
                    Ll_in = with_sharding_constraint(dense_state.Ll, PartitionSpec(pipeline_axis_name))
                    Lr_in = with_sharding_constraint(dense_state.Lr, PartitionSpec(pipeline_axis_name))
                else:
                    Ql_c = dense_state.Ql
                    Qr_c = dense_state.Qr
                    Ll_in = dense_state.Ll
                    Lr_in = dense_state.Lr

                Ql_in, Qr_in = jax.lax.cond(balance, lambda p: _balance_qs(p[0], p[1]), lambda p: p, (Ql_c, Qr_c))
                Ql_new, Qr_new, Ll_new, Lr_new, Pg_cat = vmap(
                    _preconditioning, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None)
                )(
                    keys,
                    Ql_in,
                    Qr_in,
                    Ll_in,
                    Lr_in,
                    grads_cat,
                    valid_shape_dense,
                    diag_left,
                    diag_right,
                    plr,
                    noise_scale,
                    diag_update_fn,
                    dense_update_fn,
                )
                if pipeline_axis_name is not None:
                    Pg_cat = with_sharding_constraint(Pg_cat, PartitionSpec(pipeline_axis_name))

                state["dense"] = dense_state.replace(
                    Ql=(
                        with_sharding_constraint(otu.tree_cast(Ql_new, dtype), PartitionSpec(pipeline_axis_name))
                        if pipeline_axis_name is not None
                        else otu.tree_cast(Ql_new, dtype)
                    ),
                    Qr=(
                        with_sharding_constraint(otu.tree_cast(Qr_new, dtype), PartitionSpec(pipeline_axis_name))
                        if pipeline_axis_name is not None
                        else otu.tree_cast(Qr_new, dtype)
                    ),
                    Ll=(
                        with_sharding_constraint(otu.tree_cast(Ll_new, jnp.float32), PartitionSpec(pipeline_axis_name))
                        if pipeline_axis_name is not None
                        else otu.tree_cast(Ll_new, jnp.float32)
                    ),
                    Lr=(
                        with_sharding_constraint(otu.tree_cast(Lr_new, jnp.float32), PartitionSpec(pipeline_axis_name))
                        if pipeline_axis_name is not None
                        else otu.tree_cast(Lr_new, jnp.float32)
                    ),
                )

                valid_count = dense_state.valid_count
                Pg_cat = Pg_cat[:valid_count]
                pg_dense_blocks = Pg_cat

                start_idx = 0
                for leaf_idx, (leaf, st) in enumerate(zip(leaves_u, perleaf_state, strict=False)):
                    if st.kind != DENSE_PATH:
                        continue
                    B = st.B
                    m, n = st.merged
                    nr, nc = st.nr, st.nc
                    nb = B * nr * nc
                    blocks = pg_dense_blocks[start_idx : start_idx + nb]
                    start_idx += nb
                    rec = _unblock2d(blocks, (nr, nc, m, n), dense_state.block_size)
                    leaves_u[leaf_idx] = jnp.reshape(rec, leaf.shape)

        for leaf_idx, (leaf, st) in enumerate(zip(leaves_u, perleaf_state, strict=False)):
            if st.kind != LARGE_PATH:
                continue
            B = st.B
            m, n = st.merged
            diag_left = st.diag_left
            diag_right = st.diag_right
            p2d = jnp.reshape(leaf, (B, m, n))

            if diag_left and diag_right:
                Gs = p2d
                stack = st.stack
                if Gs.shape[0] < stack:
                    pad = stack - Gs.shape[0]
                    Gs = jnp.concatenate([Gs, jnp.ones((pad, m, n), Gs.dtype)], axis=0)
                if pipeline_axis_name is not None:
                    Gs = with_sharding_constraint(Gs, PartitionSpec(pipeline_axis_name))

                key = jax.random.fold_in(jax.random.PRNGKey(43), step)
                keys = jax.random.split(key, stack)
                if pipeline_axis_name is not None:
                    keys = with_sharding_constraint(keys, PartitionSpec(pipeline_axis_name))

                if pipeline_axis_name is not None:
                    Ql_c = with_sharding_constraint(st.Ql, PartitionSpec(pipeline_axis_name))
                    Qr_c = with_sharding_constraint(st.Qr, PartitionSpec(pipeline_axis_name))
                    Ll_in = with_sharding_constraint(st.Ll, PartitionSpec(pipeline_axis_name))
                    Lr_in = with_sharding_constraint(st.Lr, PartitionSpec(pipeline_axis_name))
                else:
                    Ql_c, Qr_c, Ll_in, Lr_in = st.Ql, st.Qr, st.Ll, st.Lr

                Ql_in, Qr_in = jax.lax.cond(balance, lambda p: _balance_qs(p[0], p[1]), lambda p: p, (Ql_c, Qr_c))

                valid_shape_large = jnp.stack([st.valid_rows, st.valid_cols], axis=1)
                if pipeline_axis_name is not None:
                    valid_shape_large = with_sharding_constraint(valid_shape_large, PartitionSpec(pipeline_axis_name))

                Ql_new, Qr_new, Ll_new, Lr_new, Pg = vmap(
                    _preconditioning, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None)
                )(
                    keys,
                    Ql_in,
                    Qr_in,
                    Ll_in,
                    Lr_in,
                    Gs,
                    valid_shape_large,
                    True,
                    True,
                    plr,
                    noise_scale,
                    diag_update_fn,
                    dense_update_fn,
                )
                if pipeline_axis_name is not None:
                    Pg = with_sharding_constraint(Pg, PartitionSpec(pipeline_axis_name))

                state["large"][leaf_idx] = st.replace(
                    Ql=(
                        with_sharding_constraint(otu.tree_cast(Ql_new, dtype), PartitionSpec(pipeline_axis_name))
                        if pipeline_axis_name is not None
                        else otu.tree_cast(Ql_new, dtype)
                    ),
                    Qr=(
                        with_sharding_constraint(otu.tree_cast(Qr_new, dtype), PartitionSpec(pipeline_axis_name))
                        if pipeline_axis_name is not None
                        else otu.tree_cast(Qr_new, dtype)
                    ),
                    Ll=(
                        with_sharding_constraint(otu.tree_cast(Ll_new, jnp.float32), PartitionSpec(pipeline_axis_name))
                        if pipeline_axis_name is not None
                        else otu.tree_cast(Ll_new, jnp.float32)
                    ),
                    Lr=(
                        with_sharding_constraint(otu.tree_cast(Lr_new, jnp.float32), PartitionSpec(pipeline_axis_name))
                        if pipeline_axis_name is not None
                        else otu.tree_cast(Lr_new, jnp.float32)
                    ),
                )

                Pg = Pg[:B]
                leaves_u[leaf_idx] = jnp.reshape(Pg, leaf.shape)

            elif diag_left != diag_right:
                block_rows = not diag_left
                unblock_fn_batched = _unblock_rows if block_rows else _unblock_cols
                num_blocks_per_sample = st.nr if block_rows else st.nc
                other_dim = n if block_rows else m

                if block_rows:
                    Gs, meta = _block_rows(p2d, block_size)
                else:
                    Gs, meta = _block_cols(p2d, block_size)
                stack = st.stack
                if Gs.shape[0] < stack:
                    pad = stack - Gs.shape[0]
                    pad_shape = (pad, block_size, other_dim) if block_rows else (pad, other_dim, block_size)
                    Gs = jnp.concatenate([Gs, jnp.ones(pad_shape, Gs.dtype)], axis=0)

                if pipeline_axis_name is not None:
                    Gs = with_sharding_constraint(Gs, PartitionSpec(pipeline_axis_name))

                key_val = 45 if block_rows else 44
                key = jax.random.fold_in(jax.random.PRNGKey(key_val), step)
                keys = jax.random.split(key, stack)
                if pipeline_axis_name is not None:
                    keys = with_sharding_constraint(keys, PartitionSpec(pipeline_axis_name))

                valid_shape_large = jnp.stack([st.valid_rows, st.valid_cols], axis=1)

                if pipeline_axis_name is not None:
                    valid_shape_large = with_sharding_constraint(valid_shape_large, PartitionSpec(pipeline_axis_name))
                    Ql_c = with_sharding_constraint(st.Ql, PartitionSpec(pipeline_axis_name))
                    Qr_c = with_sharding_constraint(st.Qr, PartitionSpec(pipeline_axis_name))
                    Ll_in = with_sharding_constraint(st.Ll, PartitionSpec(pipeline_axis_name))
                    Lr_in = with_sharding_constraint(st.Lr, PartitionSpec(pipeline_axis_name))
                else:
                    Ql_c, Qr_c, Ll_in, Lr_in = st.Ql, st.Qr, st.Ll, st.Lr

                Ql_in, Qr_in = jax.lax.cond(balance, lambda p: _balance_qs(p[0], p[1]), lambda p: p, (Ql_c, Qr_c))

                Ql_new, Qr_new, Ll_new, Lr_new, Pg = vmap(
                    _preconditioning, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None)
                )(
                    keys,
                    Ql_in,
                    Qr_in,
                    Ll_in,
                    Lr_in,
                    Gs,
                    valid_shape_large,
                    diag_left,
                    diag_right,
                    plr,
                    noise_scale,
                    diag_update_fn,
                    dense_update_fn,
                )
                if pipeline_axis_name is not None:
                    Pg = with_sharding_constraint(Pg, PartitionSpec(pipeline_axis_name))

                state["large"][leaf_idx] = st.replace(
                    Ql=(
                        with_sharding_constraint(otu.tree_cast(Ql_new, dtype), PartitionSpec(pipeline_axis_name))
                        if pipeline_axis_name is not None
                        else otu.tree_cast(Ql_new, dtype)
                    ),
                    Qr=(
                        with_sharding_constraint(otu.tree_cast(Qr_new, dtype), PartitionSpec(pipeline_axis_name))
                        if pipeline_axis_name is not None
                        else otu.tree_cast(Qr_new, dtype)
                    ),
                    Ll=(
                        with_sharding_constraint(otu.tree_cast(Ll_new, jnp.float32), PartitionSpec(pipeline_axis_name))
                        if pipeline_axis_name is not None
                        else otu.tree_cast(Ll_new, jnp.float32)
                    ),
                    Lr=(
                        with_sharding_constraint(otu.tree_cast(Lr_new, jnp.float32), PartitionSpec(pipeline_axis_name))
                        if pipeline_axis_name is not None
                        else otu.tree_cast(Lr_new, jnp.float32)
                    ),
                )

                Pg = Pg[: (B * num_blocks_per_sample)]
                rec = unblock_fn_batched(Pg, meta, block_size, B)
                leaves_u[leaf_idx] = jnp.reshape(rec, leaf.shape)

        leaves_mupd, _ = jax.tree.flatten(mupd)
        for leaf_idx, (leaf, st) in enumerate(zip(leaves_u, perleaf_state, strict=False)):  # noqa: B007
            if st.kind != ONE_D_PATH:
                continue
            B = st.B
            g = leaves_mupd[leaf_idx].astype(dtype)
            g2d = jnp.reshape(g, (B, -1))
            key = jax.random.fold_in(jax.random.PRNGKey(46), step)
            keys = jax.random.split(key, B)

            Ql_new, Ll_new, Pg_flat = vmap(_preconditioning_one_d, in_axes=(0, 0, 0, 0, None, None, None))(
                keys, st.Ql, st.Ll, g2d, plr, noise_scale, diag_update_fn
            )

            state["large"][leaf_idx] = st.replace(Ql=otu.tree_cast(Ql_new, dtype), Ll=otu.tree_cast(Ll_new, jnp.float32))
            leaves_u[leaf_idx] = jnp.reshape(Pg_flat, g.shape)

        precond_all = tdef_u.unflatten(leaves_u)

        if params_partition_specs is not None:
            precond_all = with_sharding_constraint(precond_all, params_partition_specs)

        precond_all = jax.tree.map(
            lambda g: g * (1.1 / jnp.maximum(jnp.sqrt(jnp.mean(jnp.square(g))), 1.1)), precond_all
        )

        if lr_style == "adam":
            precond_all = jax.tree.map(lambda g: g / jnp.array(5.0, g.dtype), precond_all)

        state["count"] = step
        state["mu"] = mu
        return precond_all, state

    return base.GradientTransformation(init_fn, update_fn)


def scale_by_skew(
    lr_style: str | None = "adam",
    b1: float = 0.95,
    normalize_grads: bool = False,
    max_size_dense: int = 16384,
    preconditioner_lr: float = 0.7,
    preconditioner_init_scale: float = 1.0,
    dtype: str | jnp.dtype = jnp.bfloat16,
    scanned_layers: base.Params | None = None,
    block_size: int = 256,
    pipeline_axis_name: str | None = None,
    pipeline_axis_size: int = 1,
    params_partition_specs: PartitionSpec | list | tuple | dict | None = None,
    noise_scale: float = 1e-9,
) -> base.GradientTransformation:
    """Create a gradient scaling transformation using skew-style preconditioner updates.

    The skew variant of White Kron uses Procrustes orthogonalization to maintain
    near-orthogonal preconditioner matrices, which can provide more stable training.

    Args:
        lr_style (str | None): Learning rate scaling style. "adam" for Adam-like scaling.
            Defaults to "adam".
        b1 (float): Momentum coefficient. Defaults to 0.95.
        normalize_grads (bool): Whether to normalize gradients. Defaults to False.
        max_size_dense (int): Max dimension for dense factors. Defaults to 16384.
        preconditioner_lr (float): Preconditioner learning rate. Defaults to 0.7.
        preconditioner_init_scale (float): Initial preconditioner scale. Defaults to 1.0.
        dtype (str | jnp.dtype): Storage dtype. Defaults to jnp.bfloat16.
        scanned_layers (base.Params | None): Scanned layer indicators. Defaults to None.
        block_size (int): Block size for matrix partitioning. Defaults to 256.
        pipeline_axis_name (str | None): Pipeline axis name. Defaults to None.
        pipeline_axis_size (int): Pipeline axis size. Defaults to 1.
        params_partition_specs: Parameter partition specs. Defaults to None.
        noise_scale (float): Noise scale for stability. Defaults to 1e-9.

    Returns:
        base.GradientTransformation: Skew-style preconditioned gradient transformation.

    Example:
        >>> import optax
        >>> from eformer.optimizers._tx import scale_by_skew
        >>> optimizer = optax.chain(
        ...     scale_by_skew(b1=0.95),
        ...     optax.add_decayed_weights(0.1),
        ...     optax.scale_by_learning_rate(1e-4),
        ... )
    """
    return _def_scale(
        lr_style=lr_style,
        b1=b1,
        normalize_grads=normalize_grads,
        max_size_dense=max_size_dense,
        preconditioner_lr=preconditioner_lr,
        preconditioner_init_scale=preconditioner_init_scale,
        preconditioner_update_style="skew",
        dtype=dtype,
        scanned_layers=scanned_layers,
        block_size=block_size,
        pipeline_axis_name=pipeline_axis_name,
        pipeline_axis_size=pipeline_axis_size,
        params_partition_specs=params_partition_specs,
        noise_scale=noise_scale,
    )


def scale_by_quad(
    lr_style: str | None = "adam",
    b1: float = 0.95,
    normalize_grads: bool = False,
    max_size_dense: int = 16384,
    preconditioner_lr: float = 0.7,
    preconditioner_init_scale: float = 1.0,
    dtype: str | jnp.dtype = jnp.bfloat16,
    scanned_layers: base.Params | None = None,
    block_size: int = 256,
    pipeline_axis_name: str | None = None,
    pipeline_axis_size: int = 1,
    params_partition_specs: PartitionSpec | list | tuple | dict | None = None,
    noise_scale: float = 1e-9,
) -> base.GradientTransformation:
    """Create a gradient scaling transformation using QUAD-style preconditioner updates.

    The QUAD variant of White Kron uses quadratic preconditioner updates that
    directly minimize a quadratic loss in the preconditioner space.

    Args:
        lr_style (str | None): Learning rate scaling style. "adam" for Adam-like scaling.
            Defaults to "adam".
        b1 (float): Momentum coefficient. Defaults to 0.95.
        normalize_grads (bool): Whether to normalize gradients. Defaults to False.
        max_size_dense (int): Max dimension for dense factors. Defaults to 16384.
        preconditioner_lr (float): Preconditioner learning rate. Defaults to 0.7.
        preconditioner_init_scale (float): Initial preconditioner scale. Defaults to 1.0.
        dtype (str | jnp.dtype): Storage dtype. Defaults to jnp.bfloat16.
        scanned_layers (base.Params | None): Scanned layer indicators. Defaults to None.
        block_size (int): Block size for matrix partitioning. Defaults to 256.
        pipeline_axis_name (str | None): Pipeline axis name. Defaults to None.
        pipeline_axis_size (int): Pipeline axis size. Defaults to 1.
        params_partition_specs: Parameter partition specs. Defaults to None.
        noise_scale (float): Noise scale for stability. Defaults to 1e-9.

    Returns:
        base.GradientTransformation: QUAD-style preconditioned gradient transformation.

    Example:
        >>> import optax
        >>> from eformer.optimizers._tx import scale_by_quad
        >>> optimizer = optax.chain(
        ...     scale_by_quad(b1=0.95),
        ...     optax.add_decayed_weights(0.1),
        ...     optax.scale_by_learning_rate(1e-4),
        ... )
    """
    return _def_scale(
        lr_style=lr_style,
        b1=b1,
        normalize_grads=normalize_grads,
        max_size_dense=max_size_dense,
        preconditioner_lr=preconditioner_lr,
        preconditioner_init_scale=preconditioner_init_scale,
        preconditioner_update_style="QUAD",
        dtype=dtype,
        scanned_layers=scanned_layers,
        block_size=block_size,
        pipeline_axis_name=pipeline_axis_name,
        pipeline_axis_size=pipeline_axis_size,
        params_partition_specs=params_partition_specs,
        noise_scale=noise_scale,
    )


def skew(
    learning_rate: float | Callable[[int], float] = 0.001,
    lr_style: str | None = "adam",
    b1: float = 0.95,
    weight_decay: float = 0.1,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None = None,
    normalize_grads: bool = False,
    max_size_dense: int = 16384,
    preconditioner_lr: float = 0.7,
    preconditioner_init_scale: float = 1.0,
    dtype: str | jnp.dtype = jnp.bfloat16,
    scanned_layers: base.Params | None = None,
    block_size: int = 256,
    pipeline_axis_name: str | None = None,
    pipeline_axis_size: int = 1,
    params_partition_specs: PartitionSpec | list | tuple | dict | None = None,
    noise_scale: float = 1e-9,
) -> base.GradientTransformation:
    """Complete Skew optimizer with weight decay and learning rate scheduling.

    Skew is a Kronecker-factored preconditioned optimizer using Procrustes
    orthogonalization to maintain near-orthogonal preconditioner matrices.
    This provides efficient second-order optimization with stable training.

    Args:
        learning_rate (float | Callable[[int], float]): Learning rate or schedule.
            Defaults to 0.001.
        lr_style (str | None): Learning rate scaling style. Defaults to "adam".
        b1 (float): Momentum coefficient. Defaults to 0.95.
        weight_decay (float): Weight decay coefficient. Defaults to 0.1.
        weight_decay_mask: Mask for selective weight decay. Defaults to None.
        normalize_grads (bool): Whether to normalize gradients. Defaults to False.
        max_size_dense (int): Max dimension for dense factors. Defaults to 16384.
        preconditioner_lr (float): Preconditioner learning rate. Defaults to 0.7.
        preconditioner_init_scale (float): Initial preconditioner scale. Defaults to 1.0.
        dtype (str | jnp.dtype): Storage dtype. Defaults to jnp.bfloat16.
        scanned_layers (base.Params | None): Scanned layer indicators. Defaults to None.
        block_size (int): Block size for matrix partitioning. Defaults to 256.
        pipeline_axis_name (str | None): Pipeline axis name. Defaults to None.
        pipeline_axis_size (int): Pipeline axis size. Defaults to 1.
        params_partition_specs: Parameter partition specs. Defaults to None.
        noise_scale (float): Noise scale for stability. Defaults to 1e-9.

    Returns:
        base.GradientTransformation: Complete Skew optimizer transformation.

    Example:
        >>> from eformer.optimizers._tx import skew
        >>> # With constant learning rate
        >>> optimizer = skew(learning_rate=1e-4, b1=0.95, weight_decay=0.1)
        >>> # With learning rate schedule
        >>> import optax
        >>> schedule = optax.cosine_decay_schedule(1e-4, 10000)
        >>> optimizer = skew(learning_rate=schedule)
    """
    tx = [
        scale_by_skew(
            lr_style=lr_style,
            b1=b1,
            normalize_grads=normalize_grads,
            max_size_dense=max_size_dense,
            preconditioner_lr=preconditioner_lr,
            preconditioner_init_scale=preconditioner_init_scale,
            dtype=dtype,
            scanned_layers=scanned_layers,
            block_size=block_size,
            pipeline_axis_name=pipeline_axis_name,
            pipeline_axis_size=pipeline_axis_size,
            params_partition_specs=params_partition_specs,
            noise_scale=noise_scale,
        )
    ]
    if weight_decay > 0.0:
        tx.append(transform.add_decayed_weights(weight_decay, weight_decay_mask))
    tx.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*tx)


def quad(
    learning_rate: float | Callable[[int], float] = 0.001,
    lr_style: str | None = "adam",
    b1: float = 0.95,
    weight_decay: float = 0.1,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None = None,
    normalize_grads: bool = False,
    max_size_dense: int = 16384,
    preconditioner_lr: float = 0.7,
    preconditioner_init_scale: float = 1.0,
    dtype: str | jnp.dtype = jnp.bfloat16,
    scanned_layers: base.Params | None = None,
    block_size: int = 256,
    pipeline_axis_name: str | None = None,
    pipeline_axis_size: int = 1,
    params_partition_specs: PartitionSpec | list | tuple | dict | None = None,
    noise_scale: float = 1e-9,
) -> base.GradientTransformation:
    """Complete Quad optimizer with weight decay and learning rate scheduling.

    Quad is a Kronecker-factored preconditioned optimizer using quadratic
    preconditioner updates that minimize a quadratic loss function.
    This provides efficient second-order optimization.

    Args:
        learning_rate (float | Callable[[int], float]): Learning rate or schedule.
            Defaults to 0.001.
        lr_style (str | None): Learning rate scaling style. Defaults to "adam".
        b1 (float): Momentum coefficient. Defaults to 0.95.
        weight_decay (float): Weight decay coefficient. Defaults to 0.1.
        weight_decay_mask: Mask for selective weight decay. Defaults to None.
        normalize_grads (bool): Whether to normalize gradients. Defaults to False.
        max_size_dense (int): Max dimension for dense factors. Defaults to 16384.
        preconditioner_lr (float): Preconditioner learning rate. Defaults to 0.7.
        preconditioner_init_scale (float): Initial preconditioner scale. Defaults to 1.0.
        dtype (str | jnp.dtype): Storage dtype. Defaults to jnp.bfloat16.
        scanned_layers (base.Params | None): Scanned layer indicators. Defaults to None.
        block_size (int): Block size for matrix partitioning. Defaults to 256.
        pipeline_axis_name (str | None): Pipeline axis name. Defaults to None.
        pipeline_axis_size (int): Pipeline axis size. Defaults to 1.
        params_partition_specs: Parameter partition specs. Defaults to None.
        noise_scale (float): Noise scale for stability. Defaults to 1e-9.

    Returns:
        base.GradientTransformation: Complete Quad optimizer transformation.

    Example:
        >>> from eformer.optimizers._tx import quad
        >>> # With constant learning rate
        >>> optimizer = quad(learning_rate=1e-4, b1=0.95, weight_decay=0.1)
        >>> # With learning rate schedule
        >>> import optax
        >>> schedule = optax.cosine_decay_schedule(1e-4, 10000)
        >>> optimizer = quad(learning_rate=schedule)
    """
    tx = [
        scale_by_quad(
            lr_style=lr_style,
            b1=b1,
            normalize_grads=normalize_grads,
            max_size_dense=max_size_dense,
            preconditioner_lr=preconditioner_lr,
            preconditioner_init_scale=preconditioner_init_scale,
            dtype=dtype,
            scanned_layers=scanned_layers,
            block_size=block_size,
            pipeline_axis_name=pipeline_axis_name,
            pipeline_axis_size=pipeline_axis_size,
            params_partition_specs=params_partition_specs,
            noise_scale=noise_scale,
        )
    ]
    if weight_decay > 0.0:
        tx.append(transform.add_decayed_weights(weight_decay, weight_decay_mask))
    tx.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*tx)


def get_opt_state_partition_specs(params, **quad_kwargs):
    """Generate partition specs for White Kron optimizer state.

    This utility function creates JAX partition specifications for the optimizer
    state, enabling proper sharding of the optimizer state across devices in
    distributed training scenarios.

    Args:
        params: Model parameters, used to infer state structure and shapes.
        **quad_kwargs: Keyword arguments for the Quad/Skew optimizer, including:
            - lr_style: Learning rate scaling style
            - b1: Momentum coefficient
            - normalize_grads: Whether to normalize gradients
            - max_size_dense: Max dimension for dense factors
            - preconditioner_lr: Preconditioner learning rate
            - preconditioner_init_scale: Initial preconditioner scale
            - dtype: Storage dtype
            - scanned_layers: Scanned layer indicators
            - block_size: Block size for matrix partitioning
            - pipeline_axis_name: Pipeline axis name for sharding
            - pipeline_axis_size: Pipeline axis size
            - params_partition_specs: Parameter partition specs
            - noise_scale: Noise scale for stability
            - weight_decay: Weight decay coefficient (used to determine state structure)

    Returns:
        tuple: Partition specs for the optimizer state. Structure depends on
            whether weight decay is enabled:
            - With weight decay > 0: (precond_specs, None, None)
            - Without weight decay: (precond_specs, None)

    Example:
        >>> import jax.numpy as jnp
        >>> from eformer.optimizers._tx.white_kron import get_opt_state_partition_specs
        >>> params = {"layer1": jnp.zeros((128, 64)), "layer2": jnp.zeros((64, 32))}
        >>> specs = get_opt_state_partition_specs(
        ...     params,
        ...     pipeline_axis_name="dp",
        ...     pipeline_axis_size=4,
        ...     weight_decay=0.1,
        ... )
    """
    _allowed = {
        "lr_style",
        "b1",
        "normalize_grads",
        "max_size_dense",
        "preconditioner_lr",
        "preconditioner_init_scale",
        "dtype",
        "scanned_layers",
        "block_size",
        "pipeline_axis_name",
        "pipeline_axis_size",
        "params_partition_specs",
        "noise_scale",
    }
    precond_kwargs = {k: v for k, v in quad_kwargs.items() if k in _allowed}
    weight_decay = float(quad_kwargs.get("weight_decay", 0.0) or 0.0)
    _no_constraint_kwargs = dict(precond_kwargs)
    _no_constraint_kwargs["params_partition_specs"] = None
    _no_constraint_kwargs["pipeline_axis_name"] = None
    tx = _def_scale(**_no_constraint_kwargs)
    state_shape = jax.eval_shape(tx.init, params)
    pipeline_axis_name = precond_kwargs.get("pipeline_axis_name", None)
    b1 = precond_kwargs.get("b1", 0.95)
    params_partition_specs = precond_kwargs.get("params_partition_specs", None)
    replicated = PartitionSpec()

    def _leading_axis_spec(ndim: int) -> PartitionSpec:
        if pipeline_axis_name is None or ndim == 0:
            return replicated
        return PartitionSpec(*([pipeline_axis_name] + [None] * (ndim - 1)))

    if b1 and b1 > 0:
        if params_partition_specs is not None:
            mu_specs = params_partition_specs
        else:

            def _param_spec(p):
                try:
                    return p.sharding.spec
                except Exception:
                    return replicated

            mu_specs = jax.tree.map(_param_spec, params)
    else:
        mu_specs = None

    def _to_specs(x, key_path: tuple[Any, ...] = ()):
        if isinstance(x, jax.ShapeDtypeStruct):
            return _leading_axis_spec(x.ndim)
        if isinstance(x, LeafState):
            if x.kind == ONE_D_PATH:
                return x.replace(
                    Ql=replicated if isinstance(x.Ql, jax.ShapeDtypeStruct) else None,
                    Qr=None,
                    Ll=replicated if isinstance(x.Ll, jax.ShapeDtypeStruct) else None,
                    Lr=None,
                    valid_rows=replicated if isinstance(x.valid_rows, jax.ShapeDtypeStruct) else None,
                    valid_cols=replicated if isinstance(x.valid_cols, jax.ShapeDtypeStruct) else None,
                )
            return x.replace(
                Ql=_to_specs(x.Ql),
                Qr=_to_specs(x.Qr),
                Ll=_to_specs(x.Ll),
                Lr=_to_specs(x.Lr),
                valid_rows=_to_specs(x.valid_rows),
                valid_cols=_to_specs(x.valid_cols),
            )
        if isinstance(x, DenseState):
            return x.replace(
                Ql=_to_specs(x.Ql),
                Qr=_to_specs(x.Qr),
                Ll=_to_specs(x.Ll),
                Lr=_to_specs(x.Lr),
                valid_rows=_to_specs(x.valid_rows),
                valid_cols=_to_specs(x.valid_cols),
            )
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if k == "mu" and mu_specs is not None:
                    out[k] = mu_specs
                else:
                    out[k] = _to_specs(v, (*key_path, k))
            return out
        if isinstance(x, list | tuple):
            mapped = [_to_specs(v, (*key_path, i)) for i, v in enumerate(x)]
            return type(x)(mapped)
        return None

    precond_specs = _to_specs(state_shape)
    if weight_decay > 0.0:
        return (precond_specs, None, None)
    else:
        return (precond_specs, None)


# Lipschitz estimate decay rate for preconditioner updates
betaL = 0.95


def _diag_update(term1, term2, L, Q, lr_precond):
    """Update diagonal preconditioner using QUAD-style update.

    Args:
        term1: Diagonal term from gradient outer product.
        term2: Normalization term (total elements / dimension).
        L: Current Lipschitz estimate.
        Q: Current diagonal preconditioner.
        lr_precond: Preconditioner learning rate.

    Returns:
        tuple: (updated_Q, updated_L) - New preconditioner and Lipschitz estimate.
    """
    ell = jnp.max(term1) + term2
    L = jnp.maximum(betaL * L + (1 - betaL) * ell, ell)
    z = (lr_precond / (2.0 * L)).astype(Q.dtype)
    gain = 1.0 - z * (term1 - term2)
    Qn = Q * (gain * gain)
    return Qn, L


def _diag_update_q0p5eq1p5(term1, term2, L, Q, lr_precond):
    """Update diagonal preconditioner using skew-style update.

    This variant uses a linear gain update rather than quadratic, providing
    different stability characteristics.

    Args:
        term1: Diagonal term from gradient outer product.
        term2: Normalization term (total elements / dimension).
        L: Current Lipschitz estimate.
        Q: Current diagonal preconditioner.
        lr_precond: Preconditioner learning rate.

    Returns:
        tuple: (updated_Q, updated_L) - New preconditioner and Lipschitz estimate.
    """
    ell = jnp.max(term1) + term2
    L = jnp.maximum(betaL * L + (1 - betaL) * ell, ell)
    z = (lr_precond / L).astype(Q.dtype)
    gain = 1.0 - z * (term1 - term2)
    Qn = Q * gain
    return Qn, L


def _norm_lower_bound(key, A, k=4, iters=5, skh=False):
    """Estimate a lower bound on the spectral norm of a matrix.

    Uses randomized power iteration to efficiently estimate the spectral norm
    without computing a full SVD.

    Args:
        key: JAX random key for initialization.
        A: Input matrix to estimate norm of.
        k: Number of random vectors to use. Defaults to 4.
        iters: Number of power iterations. Defaults to 5.
        skh: If True, use max absolute value scaling; otherwise use diagonal scaling.
            Defaults to False.

    Returns:
        float: Lower bound estimate of the spectral norm of A.
    """
    if skh:
        scale = jnp.max(jnp.abs(A))
    else:
        scale = jnp.max(jnp.diag(A))
    A /= scale
    mean_energies = jnp.mean(A * A, axis=1, keepdims=False)
    j = jnp.argmax(mean_energies)
    power = jax.lax.dynamic_index_in_dim(mean_energies, j, 0, keepdims=False)
    max_vec = jax.lax.dynamic_index_in_dim(A, j, 0, keepdims=False)
    x = (max_vec * jax.lax.rsqrt(power) + jax.random.normal(key, (k, A.shape[1]), A.dtype)) @ A
    for _ in range(iters):
        x = x / jnp.max(jnp.abs(x))
        x = x @ A
    x = (x / jnp.linalg.vector_norm(x, axis=1, keepdims=True)) @ A
    return jnp.max(jnp.linalg.vector_norm(x, axis=1, keepdims=False)) * scale


def _dense_update(key, term1, term2, L, Q, lr_precond):
    """Update dense preconditioner using QUAD-style update.

    Performs a symmetric update step that preserves positive definiteness
    of the preconditioner matrix.

    Args:
        key: JAX random key for norm estimation.
        term1: Matrix term from gradient outer product.
        term2: Normalization term.
        L: Current Lipschitz estimate.
        Q: Current dense preconditioner matrix.
        lr_precond: Preconditioner learning rate.

    Returns:
        tuple: (updated_Q, updated_L) - New preconditioner and Lipschitz estimate.
    """
    ell = _norm_lower_bound(key, term1) + term2
    L = jnp.maximum(betaL * L + (1 - betaL) * ell, ell)
    z = (lr_precond / (2.0 * L)).astype(Q.dtype)
    P = Q - z * (term1 @ Q - term2 * Q)
    P = P - z * (P @ term1 - P * term2)
    Qn = (P + P.T) / 2.0
    return Qn, L


def _dense_update_q0p5eq1p5(key, term1, term2, L, Q, lr_precond):
    """Update dense preconditioner using skew-style update with Procrustes step.

    This variant uses a Procrustes orthogonalization step to maintain near-orthogonal
    preconditioner matrices, which can provide more stable training.

    Args:
        key: JAX random key for norm estimation and Procrustes step.
        term1: Matrix term from gradient outer product.
        term2: Normalization term.
        L: Current Lipschitz estimate.
        Q: Current dense preconditioner matrix.
        lr_precond: Preconditioner learning rate.

    Returns:
        tuple: (updated_Q, updated_L) - New preconditioner and Lipschitz estimate.
    """
    key1, key2 = jax.random.split(key)
    ell = _norm_lower_bound(key1, term1) + term2
    L = jnp.maximum(betaL * L + (1 - betaL) * ell, ell)
    z = (lr_precond / L).astype(Q.dtype)
    Q_updated = Q - z * (term1 @ Q - term2 * Q)
    Qn = _procrustes_step(key2, Q_updated)
    return Qn, L


def _procrustes_step(key, Q, max_step_size=1 / 8):
    """Perform a Procrustes orthogonalization step on a matrix.

    Projects Q towards the nearest orthogonal matrix using a gradient step
    on the skew-symmetric component of Q.

    Args:
        key: JAX random key for norm estimation.
        Q: Input matrix to orthogonalize.
        max_step_size: Maximum step size for the rotation update. Defaults to 1/8.

    Returns:
        jax.Array: Updated matrix closer to orthogonal.
    """
    R = Q.T - Q
    max_abs = jnp.max(jnp.abs(R))

    def inner(R):
        R = R / max_abs
        RQ = R @ Q
        tr_RQ = jnp.trace(RQ)

        def do_rotation():
            a = max_step_size / _norm_lower_bound(key, R, skh=True)
            RRQ = R @ RQ
            tr_RRQ = jnp.trace(RRQ)
            a = jnp.where(tr_RRQ < 0, jnp.minimum(a, -tr_RQ / tr_RRQ), a)
            return Q + a * (RQ + 0.5 * a * RRQ)

        return jax.lax.cond(tr_RQ > 0, do_rotation, lambda: Q)

    return jax.lax.cond(max_abs > jnp.finfo(Q.dtype).tiny, lambda: inner(R), lambda: Q)


def _preconditioning(
    key: jax.Array,
    Ql: jax.Array,
    Qr: jax.Array,
    Ll: jax.Array,
    Lr: jax.Array,
    G: jax.Array,
    valid_shape: jax.Array,
    diag_left: bool,
    diag_right: bool,
    lr_precond: jax.Array,
    noise_scale: float,
    diag_update_fn: Callable,
    dense_update_fn: Callable,
):
    """Apply Kronecker-factored preconditioning to a gradient matrix.

    This is the core preconditioning function that handles all combinations of
    dense/diagonal left and right factors.

    Args:
        key: JAX random key for noise injection and norm estimation.
        Ql: Left Kronecker factor (matrix or diagonal vector).
        Qr: Right Kronecker factor (matrix or diagonal vector).
        Ll: Left Lipschitz estimate.
        Lr: Right Lipschitz estimate.
        G: Gradient matrix to precondition.
        valid_shape: Array of [valid_rows, valid_cols] for handling padding.
        diag_left: Whether left factor is diagonal.
        diag_right: Whether right factor is diagonal.
        lr_precond: Preconditioner learning rate.
        noise_scale: Scale of noise added for numerical stability.
        diag_update_fn: Function for diagonal factor updates.
        dense_update_fn: Function for dense factor updates.

    Returns:
        tuple: (Ql_new, Qr_new, Ll_new, Lr_new, Pg_out) containing updated factors,
            Lipschitz estimates, and the preconditioned gradient.
    """
    key1, key2 = jax.random.split(key)
    m, n = valid_shape[0], valid_shape[1]
    noise = jax.random.normal(key2, G.shape, G.dtype) * noise_scale
    rows = jnp.arange(G.shape[0], dtype=jnp.int32) < m
    cols = jnp.arange(G.shape[1], dtype=jnp.int32) < n
    mask = rows[:, None] & cols[None, :]
    Gn = G + noise * mask
    m, n = jnp.asarray(m, dtype=G.dtype), jnp.asarray(n, dtype=G.dtype)
    total_numel = m * n

    if not diag_left and not diag_right:
        Pg = jax.numpy.linalg.multi_dot([Ql.T, Ql, Gn, Qr.T, Qr])

        key3, key4 = jax.random.split(key1)
        term1L = Pg @ Pg.T
        term2L = total_numel / m
        Ql_new, Ll_new = dense_update_fn(key3, term1L, term2L, Ll, Ql, lr_precond)

        term1R = Pg.T @ Pg
        term2R = total_numel / n
        Qr_new, Lr_new = dense_update_fn(key4, term1R, term2R, Lr, Qr, lr_precond)

        Pg_out = jax.numpy.linalg.multi_dot([Ql_new.T, Ql_new, G, Qr_new.T, Qr_new])

    elif diag_left and not diag_right:
        Pg = (Ql * Ql)[:, None] * jax.numpy.linalg.multi_dot([Gn, Qr.T, Qr])

        term1L = jnp.sum(Pg * Pg, axis=1)
        term2L = total_numel / m
        Ql_new, Ll_new = diag_update_fn(term1L, term2L, Ll, Ql, lr_precond)

        term1R = Pg.T @ Pg
        term2R = total_numel / n
        Qr_new, Lr_new = dense_update_fn(key1, term1R, term2R, Lr, Qr, lr_precond)

        Pg_out = (Ql_new * Ql_new)[:, None] * jax.numpy.linalg.multi_dot([G, Qr_new.T, Qr_new])

    elif not diag_left and diag_right:
        Pg = jax.numpy.linalg.multi_dot([Ql.T, Ql, Gn]) * (Qr * Qr)[None, :]

        term1L = Pg @ Pg.T
        term2L = total_numel / m
        Ql_new, Ll_new = dense_update_fn(key1, term1L, term2L, Ll, Ql, lr_precond)

        term1R = jnp.sum(Pg * Pg, axis=0)
        term2R = total_numel / n
        Qr_new, Lr_new = diag_update_fn(term1R, term2R, Lr, Qr, lr_precond)

        Pg_out = jax.numpy.linalg.multi_dot([Ql_new.T, Ql_new, G]) * (Qr_new * Qr_new)[None, :]

    else:
        Pg = (Ql * Ql)[:, None] * Gn * (Qr * Qr)[None, :]

        term1L = jnp.sum(Pg * Pg, axis=1)
        term2L = total_numel / m
        Ql_new, Ll_new = diag_update_fn(term1L, term2L, Ll, Ql, lr_precond)

        term1R = jnp.sum(Pg * Pg, axis=0)
        term2R = total_numel / n
        Qr_new, Lr_new = diag_update_fn(term1R, term2R, Lr, Qr, lr_precond)

        Pg_out = (Ql_new * Ql_new)[:, None] * G * (Qr_new * Qr_new)[None, :]

    return Ql_new, Qr_new, Ll_new, Lr_new, Pg_out


def _preconditioning_one_d(key, Q, L, G, lr_precond, noise_scale, diag_update_fn):
    """Apply diagonal preconditioning to a 1D gradient vector.

    Simplified preconditioning for 1D parameters using only diagonal factors.

    Args:
        key: JAX random key for noise injection.
        Q: Diagonal preconditioner vector.
        L: Current Lipschitz estimate.
        G: 1D gradient vector to precondition.
        lr_precond: Preconditioner learning rate.
        noise_scale: Scale of noise added for stability.
        diag_update_fn: Function for diagonal factor updates.

    Returns:
        tuple: (Q_new, L_new, Pg_out) containing updated preconditioner,
            Lipschitz estimate, and preconditioned gradient.
    """
    noise = jax.random.normal(key, G.shape, G.dtype) * noise_scale
    Gn = G + noise
    Pg = Q * Q * Gn
    term1 = Pg * Pg
    term2 = 1.0
    Qn, Ln = diag_update_fn(term1, term2, L, Q, lr_precond)
    Pg_out = Qn * Qn * G
    return Qn, Ln, Pg_out


def _balance_qs(Ql, Qr):
    """Balance the scales of left and right Kronecker factors.

    Normalizes the factors so that their maximum absolute values are equal,
    preventing numerical issues from unbalanced factor scales.

    Args:
        Ql: Left Kronecker factors, shape [batch, ...].
        Qr: Right Kronecker factors, shape [batch, ...].

    Returns:
        tuple: (balanced_Ql, balanced_Qr) with equalized scales.
    """

    @vmap
    def _balance_sample(ql, qr):
        nl = jnp.max(jnp.abs(ql))
        nr = jnp.max(jnp.abs(qr))
        geometric_mean = jnp.sqrt(nl * nr)
        sL = geometric_mean / nl
        sR = geometric_mean / nr
        return ql * sL, qr * sR

    return _balance_sample(Ql, Qr)


def _block2d(x, block_size):
    """Block each [m, n] matrix in a [B, m, n] tensor into fixed-size blocks.

    Partitions each 2D matrix into non-overlapping blocks of size [block_size, block_size],
    padding as necessary. Useful for parallel processing of large matrices.

    Args:
        x: Input tensor of shape [B, m, n] containing B matrices.
        block_size: Size of square blocks to partition into.

    Returns:
        tuple: (blocks, meta) where:
            - blocks: Array of shape [B * nr * nc, block_size, block_size]
            - meta: Tuple (nr, nc, m, n) with grid dimensions and original shape
    """
    B, m, n = x.shape
    nr, nc = (m + block_size - 1) // block_size, (n + block_size - 1) // block_size
    pm, pn = nr * block_size, nc * block_size
    dm, dn = pm - m, pn - n
    xpad = jnp.pad(x, ((0, 0), (0, dm), (0, dn)))
    x5 = xpad.reshape(B, nr, block_size, nc, block_size).transpose(0, 1, 3, 2, 4)
    blocks = x5.reshape(B * nr * nc, block_size, block_size)
    return blocks, (nr, nc, m, n)


def _unblock2d(blocks, meta, block_size):
    """Reconstruct matrices from blocked representation (inverse of _block2d).

    Args:
        blocks: Blocked tensor of shape [B * nr * nc, block_size, block_size].
        meta: Tuple (nr, nc, m, n) from _block2d containing grid dimensions and original shape.
        block_size: Size of square blocks.

    Returns:
        jax.Array: Reconstructed tensor of shape [B, m, n] with padding removed.
    """
    nr, nc, m, n = meta
    bs = block_size
    B = blocks.shape[0] // (nr * nc)
    x5 = blocks.reshape(B, nr, nc, bs, bs).transpose(0, 1, 3, 2, 4)
    x = x5.reshape(B, nr * bs, nc * bs)
    return x[:, :m, :n]


def _block_rows(x, block_size):
    """Block matrices along the row dimension only.

    Partitions each matrix into horizontal strips of height block_size.

    Args:
        x: Input tensor of shape [B, m, n].
        block_size: Height of row blocks.

    Returns:
        tuple: (blocks, meta) where:
            - blocks: Array of shape [B * nr, block_size, n]
            - meta: Tuple (nr, m, n) with number of row blocks and original shape
    """
    B, m, n = x.shape
    nr = (m + block_size - 1) // block_size
    pm = nr * block_size
    dm = pm - m
    xpad = jnp.pad(x, ((0, 0), (0, dm), (0, 0)))
    x3 = xpad.reshape(B, nr, block_size, n)
    blocks = x3.reshape(B * nr, block_size, n)
    return blocks, (nr, m, n)


def _unblock_rows(blocks, meta, block_size, B):
    """Reconstruct matrices from row-blocked representation (inverse of _block_rows).

    Args:
        blocks: Row-blocked tensor of shape [B * nr, block_size, n].
        meta: Tuple (nr, m, n) from _block_rows.
        block_size: Height of row blocks.
        B: Batch size.

    Returns:
        jax.Array: Reconstructed tensor of shape [B, m, n].
    """
    nr, m, n = meta
    x3 = blocks.reshape(B, nr, block_size, n)
    x = x3.reshape(B, nr * block_size, n)
    return x[:, :m, :n]


def _block_cols(x, block_size):
    """Block matrices along the column dimension only.

    Partitions each matrix into vertical strips of width block_size.

    Args:
        x: Input tensor of shape [B, m, n].
        block_size: Width of column blocks.

    Returns:
        tuple: (blocks, meta) where:
            - blocks: Array of shape [B * nc, m, block_size]
            - meta: Tuple (nc, m, n) with number of column blocks and original shape
    """
    B, m, n = x.shape
    nc = (n + block_size - 1) // block_size
    pn = nc * block_size
    dn = pn - n
    xpad = jnp.pad(x, ((0, 0), (0, 0), (0, dn)))
    x4 = xpad.reshape(B, m, nc, block_size).transpose(0, 2, 1, 3)
    blocks = x4.reshape(B * nc, m, block_size)
    return blocks, (nc, m, n)


def _unblock_cols(blocks, meta, block_size, B):
    """Reconstruct matrices from column-blocked representation (inverse of _block_cols).

    Args:
        blocks: Column-blocked tensor of shape [B * nc, m, block_size].
        meta: Tuple (nc, m, n) from _block_cols.
        block_size: Width of column blocks.
        B: Batch size.

    Returns:
        jax.Array: Reconstructed tensor of shape [B, m, n].
    """
    nc, m, n = meta
    x4 = blocks.reshape(B, nc, m, block_size).transpose(0, 2, 1, 3)
    x = x4.reshape(B, m, nc * block_size)
    return x[:, :, :n]


def _merge_dims(shape):
    """Merge a multi-dimensional shape into a 2D shape for Kronecker factorization.

    Finds the optimal split point to reshape an N-dimensional tensor into a 2D matrix
    with balanced dimensions, minimizing the aspect ratio.

    Args:
        shape: Original tensor shape as a tuple.

    Returns:
        tuple: Merged 2D shape. Returns original shape if already 1D or 2D,
            or if the tensor is effectively 1D (all but one dim are 1).
    """
    if len(shape) < 2:
        return shape
    if np.prod(shape) == np.max(shape):
        return (np.max(shape),)
    if len(shape) == 2:
        return shape
    dims = list(shape)
    best_ratio, best_split = float("inf"), 1
    for s in range(1, len(dims)):
        lp, rp = np.prod(dims[:s]), np.prod(dims[s:])
        ratio = max(lp, rp) / min(lp, rp)
        if ratio < best_ratio:
            best_ratio, best_split = ratio, s
    return (np.prod(dims[:best_split]), np.prod(dims[best_split:]))


def _identity_padded(block_size, valid, dtype):
    """Create a padded identity matrix for initializing preconditioners.

    Creates an identity matrix of the specified valid size, padded with zeros
    to reach the full block_size.

    Args:
        block_size: Target matrix size after padding.
        valid: Number of valid rows/columns (size of identity submatrix).
        dtype: Data type for the output matrix.

    Returns:
        jax.Array: Identity matrix of shape [block_size, block_size] with
            identity in top-left [valid, valid] corner and zeros elsewhere.
    """
    if valid >= block_size:
        return jnp.eye(block_size, dtype=dtype)
    eye = jnp.eye(valid, dtype=dtype)
    return jnp.pad(eye, ((0, block_size - valid), (0, block_size - valid)), constant_values=0)
