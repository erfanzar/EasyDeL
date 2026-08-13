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

"""Low-level Pallas TPU kernel for one-shot (direct-exchange) all-reduce.

Standard ring all-reduce needs ``2 * (tp - 1)`` sequential ICI hops
(reduce-scatter followed by all-gather), which is bandwidth-optimal but
latency-bound for the small tensors that dominate decode steps (e.g. a
``[tokens, hidden]`` o_proj partial of ~128 KiB).  The one-shot algorithm
instead has every device DMA its full local buffer directly to every peer in
parallel, then each device sums the ``tp`` buffers locally:

  1. Barrier with all peers (double barrier, matching the ring kernels).
  2. Start ``tp - 1`` remote DMA copies: local ``x`` → peer's receive slot.
     The slot index is the sender's *offset* to the receiver, so every slot
     is written exactly once and all indexing is static.
  3. Start the local ``x`` → VMEM copy (overlapped with the exchange).
  4. Wait for the sends and the ``tp - 1`` incoming copies (symmetric
     traffic: every device sends and receives identical byte counts, so
     waiting on the local send/recv semaphore pair covers both directions).
  5. Copy the received buffers to VMEM, accumulate in float32, cast back to
     the input dtype, and DMA the result to the output.

Total wire bytes per device match the ring algorithm; the win is a single
communication phase (one logical hop) instead of ``2 * (tp - 1)``.

The kernel is only intended for buffers small enough to comfortably stage
in VMEM; larger inputs should use ``lax.psum``, whose XLA lowering is
already a bandwidth-optimal ring.

Accumulation is performed in float32 regardless of the input dtype (results
are cast back), which can differ from ``lax.psum`` on bfloat16 inputs by
normal rounding differences.

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
    """Double barrier with every peer on the collective axis using TPU semaphores.

    Mirrors ``_local_barrier`` from the ring matmul kernels, generalised from
    the two ring neighbors to all ``tp_size - 1`` peers (the one-shot exchange
    talks to every device directly).
    """
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


def _one_shot_all_reduce_kernel(
    x_hbm_ref,
    o_hbm_ref,
    recv_hbm_ref,
    local_in_sem,
    local_out_sem,
    send_sems,
    recv_sems,
    x_vmem_ref,
    recv_vmem_ref,
    o_vmem_ref,
    *,
    axis_name: str | tuple[str, ...],
    tp_size: int,
):
    """Single-step Pallas kernel body for the one-shot all-reduce.

    Args:
        x_hbm_ref: Local input buffer ``[rows, cols]`` in HBM.
        o_hbm_ref: Output buffer ``[rows, cols]`` in HBM.
        recv_hbm_ref: Receive scratch ``[tp_size - 1, rows, cols]`` in HBM;
            slot ``o - 1`` is written by the peer ``o`` steps behind on the ring.
        local_in_sem: DMA semaphore for local HBM→VMEM copies.
        local_out_sem: DMA semaphore for the VMEM→HBM output copy.
        send_sems: Per-offset DMA send semaphores ``[tp_size - 1]``.
        recv_sems: Per-offset DMA receive semaphores ``[tp_size - 1]``.
        x_vmem_ref: VMEM staging for the local input.
        recv_vmem_ref: VMEM staging for the received peer buffers.
        o_vmem_ref: VMEM staging for the summed output.
        axis_name: Collective mesh axis name.
        tp_size: Static world size on ``axis_name``.
    """
    my_id = lax.axis_index(axis_name)

    _all_peer_barrier(my_id, tp_size)

    x_local_copy = pltpu.make_async_copy(x_hbm_ref, x_vmem_ref, local_in_sem)
    x_local_copy.start()

    def _exchange_op(offset: int):
        peer = lax.rem(my_id + offset, tp_size)
        return pltpu.make_async_remote_copy(
            src_ref=x_hbm_ref,
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

    recv_local_copy = pltpu.make_async_copy(recv_hbm_ref, recv_vmem_ref, local_in_sem)
    recv_local_copy.start()
    x_local_copy.wait()
    recv_local_copy.wait()

    acc = x_vmem_ref[...].astype(jnp.float32)
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
        ValueError: If the trailing dimension is not lane-aligned (multiple of
            128) or the row count is not sublane-aligned (multiple of 8).
    """
    rows, cols = x_2d.shape
    if cols % 128 != 0:
        raise ValueError(f"one-shot all_reduce requires the trailing dim to be a multiple of 128, got {cols}.")
    if rows % 8 != 0:
        raise ValueError(f"one-shot all_reduce requires the leading (flattened) dim to be a multiple of 8, got {rows}.")
    del tp_size


def _one_shot_all_reduce_2d(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    tp_size: int,
    collective_id: int | None,
) -> jax.Array:
    """Launch the one-shot all-reduce Pallas kernel on a 2D input."""
    rows, cols = x.shape
    buffer_bytes = rows * cols * x.dtype.itemsize
    vmem_bytes = (2 * tp_size) * buffer_bytes + rows * cols * 4

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
            pltpu.VMEM((rows, cols), x.dtype),
            pltpu.VMEM((tp_size - 1, rows, cols), x.dtype),
            pltpu.VMEM((rows, cols), x.dtype),
        ),
        grid=(1,),
    )
    out_shape = [
        jax.ShapeDtypeStruct((rows, cols), x.dtype),
        jax.ShapeDtypeStruct((tp_size - 1, rows, cols), x.dtype),
    ]
    bytes_accessed = buffer_bytes * (2 * tp_size)
    cost_estimate = pl.CostEstimate(flops=rows * cols * (tp_size - 1), bytes_accessed=bytes_accessed, transcendentals=0)

    return pl.pallas_call(
        functools.partial(_one_shot_all_reduce_kernel, axis_name=axis_name, tp_size=tp_size),
        out_shape=out_shape,
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            vmem_limit_bytes=vmem_bytes + 8 * 1024 * 1024,
        ),
        cost_estimate=cost_estimate,
        name=f"one_shot_all_reduce_tp{tp_size}",
    )(x)[0]


def all_reduce(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    mode: str = "auto",
    tp_size: int | None = None,
    collective_id: int | None = 0,
) -> jax.Array:
    """All-reduce ``x`` over ``axis_name`` with a latency-optimized one-shot path.

    Args:
        x: Local partial values of any rank.
        axis_name: pmap / shard_map axis name for the collective.
        mode: ``"one_shot"`` forces the direct-exchange Pallas kernel and
            raises when its constraints are violated; ``"ring"`` delegates to
            ``lax.psum`` (XLA's bandwidth-optimal lowering); ``"auto"``
            resolves to the ring path (measured faster at all sizes on v5p-8;
            see the module docstring).
        tp_size: Collective world size; inferred when ``None``.
        collective_id: Barrier-semaphore allocation ID.  Concurrent collectives
            in one program must use distinct IDs.

    Returns:
        The elementwise sum over all devices, same shape/dtype as ``x``.

    Raises:
        ValueError: If ``mode`` is invalid or ``mode="one_shot"`` constraints
            are violated.
    """
    if mode not in ("auto", "one_shot", "ring"):
        raise ValueError(f"mode must be one of 'auto', 'one_shot', 'ring'; got {mode!r}.")
    if isinstance(axis_name, (tuple, list)):
        if mode == "one_shot":
            raise ValueError("one-shot all_reduce supports a single collective axis; got a tuple.")
        return lax.psum(x, axis_name)
    tp = _resolve_tp_size(tp_size, axis_name)
    if tp == 1:
        return x

    if mode != "one_shot":
        return lax.psum(x, axis_name)

    cols = int(x.shape[-1]) if x.ndim >= 1 else 1
    x_2d = x.reshape(-1, cols) if x.ndim != 2 else x
    validate_one_shot_inputs(x_2d, tp)

    out_2d = _one_shot_all_reduce_2d(x_2d, axis_name, tp, collective_id)
    return out_2d.reshape(x.shape)
