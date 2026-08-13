# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Low-level Pallas TPU kernel for one-shot (direct-exchange) reduce-scatter.

Each device holds a full-shaped partial ``x`` and must end with the reduced
values of its own row slice.  Instead of the ring algorithm's ``tp - 1``
sequential hops, the one-shot exchange sends each destination's slice
directly:

  1. Barrier with all peers (double barrier).
  2. For each peer at ring offset ``o``: remote-DMA ``x[peer_slice]``
     directly into the peer's receive slot ``o - 1``.  The sum is
     order-independent, so slot indexing by offset keeps all destination
     indices static.
  3. Locally copy this device's own slice ``x[my_slice]`` to VMEM.
  4. Wait sends + incoming copies (symmetric traffic, identical byte counts).
  5. Accumulate the ``tp`` slices in float32, cast back, DMA to the output.

One communication phase instead of ``tp - 1``; wire bytes per device match
the ring algorithm.  Used for small, latency-bound tensors; the bandwidth
regime delegates to ``lax.psum_scatter`` (XLA's ring lowering).

MEASURED (v5p-8, tp=4, bf16, chained steady-state): the one-shot kernel is
slower than the XLA lowering at every tested size (64 KiB: 13.3 vs 12.2 us;
512 KiB: 18.2 vs 14.8 us; 2 MiB: 65.5 vs 34.8 us).  One-shot moves
``(tp-1)/tp * 2x`` the wire bytes of reduce-scatter+all-gather and pays a
double barrier plus HBM/VMEM staging, while XLA's collective is already at
its launch/fence latency floor on a directly-connected 4-chip slice.
``mode="auto"`` therefore resolves to the XLA path; ``"one_shot"`` remains
an explicit opt-in and an autotuner candidate for topologies with more ring
hops, where the single-phase exchange may win.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _infer_axis_size(axis_name: str) -> int | None:
    """Infer collective axis size from the active mapped context when available."""
    try:
        return jax.core.concrete_or_error(
            int,
            lax.psum(jnp.array(1, dtype=jnp.int32), axis_name=axis_name),
            f"collective axis '{axis_name}' size must be static.",
        )
    except Exception:
        return None


def _resolve_tp_size(tp_size: int | None, axis_name: str) -> int:
    """Resolve tensor-parallel world size using explicit value, axis context, then global device count."""
    resolved = int(tp_size) if tp_size is not None else (_infer_axis_size(axis_name) or int(jax.device_count()))
    if resolved < 1:
        raise ValueError(f"tp_size must be >= 1, got {resolved}.")
    return resolved


def _all_peer_barrier(my_id, tp_size: int):
    """Double barrier with every peer on the collective axis using TPU semaphores."""
    barrier_sem = pltpu.get_barrier_semaphore()
    for offset in range(1, tp_size):
        peer = lax.rem(my_id + offset, tp_size)
        pl.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=(peer,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier_sem, tp_size - 1)

    @functools.partial(pl.run_scoped, second_barrier=pltpu.SemaphoreType.REGULAR)
    def _(second_barrier):
        for offset in range(1, tp_size):
            peer = lax.rem(my_id + offset, tp_size)
            pl.semaphore_signal(
                second_barrier,
                inc=1,
                device_id=(peer,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pl.semaphore_wait(second_barrier, tp_size - 1)


def _one_shot_reduce_scatter_kernel(
    x_hbm_ref,
    o_hbm_ref,
    recv_hbm_ref,
    local_in_sem,
    local_out_sem,
    send_sems,
    recv_sems,
    own_vmem_ref,
    recv_vmem_ref,
    o_vmem_ref,
    *,
    axis_name: str | tuple[str, ...],
    tp_size: int,
):
    """Single-step Pallas kernel body for the one-shot reduce-scatter.

    Args:
        x_hbm_ref: Local full partial ``[rows, cols]`` in HBM.
        o_hbm_ref: Output shard ``[rows // tp_size, cols]`` in HBM.
        recv_hbm_ref: Receive scratch ``[tp_size - 1, rows // tp_size, cols]``.
        local_in_sem: DMA semaphore for local HBM→VMEM copies.
        local_out_sem: DMA semaphore for the VMEM→HBM output copy.
        send_sems: Per-offset DMA send semaphores ``[tp_size - 1]``.
        recv_sems: Per-offset DMA receive semaphores ``[tp_size - 1]``.
        own_vmem_ref: VMEM staging for this device's own slice.
        recv_vmem_ref: VMEM staging for received peer slices.
        o_vmem_ref: VMEM staging for the summed output shard.
        axis_name: Collective mesh axis name.
        tp_size: Static world size on ``axis_name``.
    """
    my_id = lax.axis_index(axis_name)
    rows_out = o_hbm_ref.shape[0]

    _all_peer_barrier(my_id, tp_size)

    own_copy = pltpu.make_async_copy(
        x_hbm_ref.at[pl.ds(my_id * rows_out, rows_out)],
        own_vmem_ref,
        local_in_sem,
    )
    own_copy.start()

    def _exchange_op(offset: int):
        peer = lax.rem(my_id + offset, tp_size)
        return pltpu.make_async_remote_copy(
            src_ref=x_hbm_ref.at[pl.ds(peer * rows_out, rows_out)],
            dst_ref=recv_hbm_ref.at[offset - 1],
            send_sem=send_sems.at[offset - 1],
            recv_sem=recv_sems.at[offset - 1],
            device_id=(peer,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    for offset in range(1, tp_size):
        _exchange_op(offset).start()
    for offset in range(1, tp_size):
        _exchange_op(offset).wait()

    recv_copy = pltpu.make_async_copy(recv_hbm_ref, recv_vmem_ref, local_in_sem)
    recv_copy.start()
    own_copy.wait()
    recv_copy.wait()

    acc = own_vmem_ref[...].astype(jnp.float32)
    for slot in range(tp_size - 1):
        acc = acc + recv_vmem_ref[slot].astype(jnp.float32)
    o_vmem_ref[...] = acc.astype(o_vmem_ref.dtype)

    out_copy = pltpu.make_async_copy(o_vmem_ref, o_hbm_ref, local_out_sem)
    out_copy.start()
    out_copy.wait()


def validate_one_shot_inputs(x_2d: jax.Array, tp_size: int) -> None:
    """Validate 2D-flattened inputs against the one-shot kernel's constraints.

    Args:
        x_2d: Input flattened to ``[rows, cols]``.
        tp_size: Collective world size.

    Raises:
        ValueError: If ``rows`` is not divisible by ``tp_size``, the per-shard
            row count is not a multiple of 8, or the trailing dimension is not
            a multiple of 128.
    """
    rows, cols = x_2d.shape
    if rows % tp_size != 0:
        raise ValueError(f"one-shot reduce_scatter requires rows ({rows}) divisible by tp_size ({tp_size}).")
    if cols % 128 != 0:
        raise ValueError(f"one-shot reduce_scatter requires the trailing dim to be a multiple of 128, got {cols}.")
    if (rows // tp_size) % 8 != 0:
        raise ValueError(
            f"one-shot reduce_scatter requires the per-shard row count ({rows // tp_size}) to be a multiple of 8."
        )


def _one_shot_reduce_scatter_2d(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    tp_size: int,
    collective_id: int | None,
) -> jax.Array:
    """Launch the one-shot reduce-scatter Pallas kernel on a 2D input."""
    rows, cols = x.shape
    rows_out = rows // tp_size
    shard_bytes = rows_out * cols * x.dtype.itemsize
    vmem_bytes = (tp_size + 1) * shard_bytes + rows_out * cols * 4

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        ],
        scratch_shapes=(
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((tp_size - 1,)),
            pltpu.SemaphoreType.DMA((tp_size - 1,)),
            pltpu.VMEM((rows_out, cols), x.dtype),
            pltpu.VMEM((tp_size - 1, rows_out, cols), x.dtype),
            pltpu.VMEM((rows_out, cols), x.dtype),
        ),
        grid=(1,),
    )
    out_shape = [
        jax.ShapeDtypeStruct((rows_out, cols), x.dtype),
        jax.ShapeDtypeStruct((tp_size - 1, rows_out, cols), x.dtype),
    ]
    bytes_accessed = shard_bytes * (2 * tp_size)
    cost_estimate = pl.CostEstimate(
        flops=rows_out * cols * (tp_size - 1), bytes_accessed=bytes_accessed, transcendentals=0
    )

    return pl.pallas_call(
        functools.partial(_one_shot_reduce_scatter_kernel, axis_name=axis_name, tp_size=tp_size),
        out_shape=out_shape,
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            vmem_limit_bytes=vmem_bytes + 8 * 1024 * 1024,
        ),
        cost_estimate=cost_estimate,
        name=f"one_shot_reduce_scatter_tp{tp_size}",
    )(x)[0]


def reduce_scatter(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    scatter_axis: int = 0,
    tiled: bool = True,
    mode: str = "auto",
    tp_size: int | None = None,
    collective_id: int | None = 0,
) -> jax.Array:
    """Reduce-scatter ``x`` over ``axis_name`` with a latency-optimised one-shot path.

    Args:
        x: Local full-shaped partial values.
        axis_name: pmap / shard_map axis name for the collective.
        scatter_axis: Axis along which the reduced result is scattered.  The
            Pallas one-shot kernel supports ``scatter_axis=0`` on 2D-flattenable
            inputs; other axes take the ``lax.psum_scatter`` path.
        tiled: Tiled scatter semantics (no new leading axis).  Non-tiled
            requests take the ``lax.psum_scatter`` path.
        mode: ``"one_shot"`` forces the direct-exchange kernel (raises when
            unsupported), ``"ring"`` delegates to ``lax.psum_scatter``,
            ``"auto"`` resolves to the
            ``lax.psum_scatter`` path (measured faster on v5p-8; see module
            docstring).
        tp_size: Collective world size; inferred when ``None``.
        collective_id: Barrier-semaphore allocation ID.

    Returns:
        The reduced shard; shape matches ``x`` with ``scatter_axis`` divided
        by the world size.

    Raises:
        ValueError: If ``mode`` is invalid or ``mode="one_shot"`` constraints
            are violated.
    """
    if mode not in ("auto", "one_shot", "ring"):
        raise ValueError(f"mode must be one of 'auto', 'one_shot', 'ring'; got {mode!r}.")
    if isinstance(axis_name, (tuple, list)):
        if mode == "one_shot":
            raise ValueError("one-shot reduce_scatter supports a single collective axis; got a tuple.")
        return lax.psum_scatter(x, axis_name, scatter_dimension=scatter_axis, tiled=tiled)
    tp = _resolve_tp_size(tp_size, axis_name)
    if tp == 1:
        return x

    if mode != "one_shot":
        return lax.psum_scatter(x, axis_name, scatter_dimension=scatter_axis, tiled=tiled)

    if not (scatter_axis == 0 and tiled and x.ndim >= 2):
        raise ValueError(
            "one-shot reduce_scatter supports scatter_axis=0 with tiled=True on rank>=2 inputs; "
            f"got scatter_axis={scatter_axis}, tiled={tiled}."
        )
    x_2d = x.reshape(x.shape[0], -1)
    validate_one_shot_inputs(x_2d, tp)

    out_2d = _one_shot_reduce_scatter_2d(x_2d, axis_name, tp, collective_id)
    return out_2d.reshape((x.shape[0] // tp, *x.shape[1:]))
