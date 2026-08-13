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

"""Low-level Pallas TPU kernel for one-shot (direct-exchange) all-gather.

The ring all-gather needs ``tp - 1`` sequential ICI hops; for the small
tensors that dominate decode steps this is latency-bound.  The one-shot
algorithm has every device DMA its shard directly into every peer's output
buffer at the shard's final offset — a single communication phase and pure
data movement (no VMEM staging, no compute):

  1. Barrier with all peers (double barrier).
  2. Local DMA: own shard → own output slice ``out[my_id * rows : ...]``.
  3. For each peer: remote DMA own shard → peer's ``out[my_id * rows : ...]``
     (the destination window is computed with the sender's device index).
  4. Wait sends + incoming copies (symmetric traffic, identical byte counts).

Wire bytes per device match the ring algorithm; latency is one hop.  Large
buffers delegate to ``lax.all_gather`` (XLA's bandwidth-optimal ring).

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


def _one_shot_all_gather_kernel(
    x_hbm_ref,
    o_hbm_ref,
    local_sem,
    send_sems,
    recv_sems,
    *,
    axis_name: str | tuple[str, ...],
    tp_size: int,
):
    """Single-step Pallas kernel body for the one-shot all-gather.

    Args:
        x_hbm_ref: Local shard ``[rows, cols]`` in HBM.
        o_hbm_ref: Gathered output ``[tp_size * rows, cols]`` in HBM.
        local_sem: DMA semaphore for the local shard → output copy.
        send_sems: Per-offset DMA send semaphores ``[tp_size - 1]``.
        recv_sems: Per-offset DMA receive semaphores ``[tp_size - 1]``.
        axis_name: Collective mesh axis name.
        tp_size: Static world size on ``axis_name``.
    """
    my_id = lax.axis_index(axis_name)
    rows = x_hbm_ref.shape[0]

    _all_peer_barrier(my_id, tp_size)

    own_copy = pltpu.make_async_copy(
        x_hbm_ref,
        o_hbm_ref.at[pl.ds(my_id * rows, rows)],
        local_sem,
    )
    own_copy.start()

    def _exchange_op(offset: int):
        peer = lax.rem(my_id + offset, tp_size)
        return pltpu.make_async_remote_copy(
            src_ref=x_hbm_ref,
            dst_ref=o_hbm_ref.at[pl.ds(my_id * rows, rows)],
            send_sem=send_sems.at[offset - 1],
            recv_sem=recv_sems.at[offset - 1],
            device_id=(peer,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    for offset in range(1, tp_size):
        _exchange_op(offset).start()
    for offset in range(1, tp_size):
        _exchange_op(offset).wait()
    own_copy.wait()


def validate_one_shot_inputs(x_2d: jax.Array, tp_size: int) -> None:
    """Validate 2D-flattened inputs against the one-shot kernel's constraints.

    Args:
        x_2d: Shard flattened to ``[rows, cols]``.
        tp_size: Collective world size.

    Raises:
        ValueError: If the trailing dimension is not a multiple of 128 or the
            row count is not a multiple of 8.
    """
    rows, cols = x_2d.shape
    if cols % 128 != 0:
        raise ValueError(f"one-shot all_gather requires the trailing dim to be a multiple of 128, got {cols}.")
    if rows % 8 != 0:
        raise ValueError(f"one-shot all_gather requires the leading (flattened) dim to be a multiple of 8, got {rows}.")
    del tp_size


def _one_shot_all_gather_2d(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    tp_size: int,
    collective_id: int | None,
) -> jax.Array:
    """Launch the one-shot all-gather Pallas kernel on a 2D shard."""
    rows, cols = x.shape
    shard_bytes = rows * cols * x.dtype.itemsize

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
        out_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
        scratch_shapes=(
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((tp_size - 1,)),
            pltpu.SemaphoreType.DMA((tp_size - 1,)),
        ),
        grid=(1,),
    )
    out_shape = [jax.ShapeDtypeStruct((tp_size * rows, cols), x.dtype)]
    bytes_accessed = shard_bytes * (tp_size + 1)
    cost_estimate = pl.CostEstimate(flops=0, bytes_accessed=bytes_accessed, transcendentals=0)

    return pl.pallas_call(
        functools.partial(_one_shot_all_gather_kernel, axis_name=axis_name, tp_size=tp_size),
        out_shape=out_shape,
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
        cost_estimate=cost_estimate,
        name=f"one_shot_all_gather_tp{tp_size}",
    )(x)[0]


def all_gather(
    x: jax.Array,
    axis_name: str | tuple[str, ...],
    gather_axis: int = 0,
    tiled: bool = True,
    mode: str = "auto",
    tp_size: int | None = None,
    collective_id: int | None = 0,
) -> jax.Array:
    """All-gather ``x`` over ``axis_name`` with a latency-optimised one-shot path.

    Args:
        x: Local shard.
        axis_name: pmap / shard_map axis name for the collective.
        gather_axis: Axis along which shards are concatenated.  The Pallas
            one-shot kernel supports ``gather_axis=0``; other axes take the
            ``lax.all_gather`` path.
        tiled: Tiled gather semantics (concatenate, no new leading axis).
            Non-tiled requests take the ``lax.all_gather`` path.
        mode: ``"one_shot"`` forces the direct-exchange kernel (raises when
            unsupported), ``"ring"`` delegates to ``lax.all_gather``,
            ``"auto"`` resolves to the
            ``lax.all_gather`` path (measured faster on v5p-8; see module
            docstring).
        tp_size: Collective world size; inferred when ``None``.
        collective_id: Barrier-semaphore allocation ID.

    Returns:
        The gathered array; ``gather_axis`` is multiplied by the world size.

    Raises:
        ValueError: If ``mode`` is invalid or ``mode="one_shot"`` constraints
            are violated.
    """
    if mode not in ("auto", "one_shot", "ring"):
        raise ValueError(f"mode must be one of 'auto', 'one_shot', 'ring'; got {mode!r}.")
    if isinstance(axis_name, (tuple, list)):
        if mode == "one_shot":
            raise ValueError("one-shot all_gather supports a single collective axis; got a tuple.")
        return lax.all_gather(x, axis_name=axis_name, axis=gather_axis, tiled=tiled)
    tp = _resolve_tp_size(tp_size, axis_name)
    if tp == 1:
        if tiled:
            return x
        return jnp.expand_dims(x, 0)

    if mode != "one_shot":
        return lax.all_gather(x, axis_name=axis_name, axis=gather_axis, tiled=tiled)

    if not (gather_axis == 0 and tiled and x.ndim >= 2):
        raise ValueError(
            "one-shot all_gather supports gather_axis=0 with tiled=True on rank>=2 inputs; "
            f"got gather_axis={gather_axis}, tiled={tiled}, ndim={x.ndim}."
        )
    x_2d = x.reshape(x.shape[0], -1)
    validate_one_shot_inputs(x_2d, tp)

    out_2d = _one_shot_all_gather_2d(x_2d, axis_name, tp, collective_id)
    return out_2d.reshape((tp * x.shape[0], *x.shape[1:]))
