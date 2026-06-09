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

"""Bidirectional Reduce-Scatter Matmul with M-Split Algorithm.

This implementation uses BOTH left and right neighbors simultaneously to double
the effective communication bandwidth. The key insight is to split the M dimension
into N blocks (one per device), and each block into TOP and BOT halves:
  - LEFT direction handles all TOP halves
  - RIGHT direction handles all BOT halves

This avoids the collision problem where both directions would compute for the
same shard at the midpoint step.

Setup (8 devices example):
- Each device has: x[M, K_shard], y[N, K_shard] where K is sharded
- M is split into 8 blocks, each block split into TOP and BOT halves
- Output: each device gets its M_block (TOP + BOT) with full K reduction
- D0 owns Block 0 (rows 0 to M/8-1)
- D1 owns Block 1 (rows M/8 to 2M/8-1)
- etc.

Bidirectional ring:
- LEFT direction: D0 → D7 → D6 → D5 → D4 → D3 → D2 → D1 → D0 (send to left)
- RIGHT direction: D0 → D1 → D2 → D3 → D4 → D5 → D6 → D7 → D0 (send to right)

M-SPLIT BIDIRECTIONAL ALGORITHM:
================================

Split M into N blocks, each block into TOP (first half) and BOT (second half):
  - LEFT handles: B0_TOP, B1_TOP, B2_TOP, ..., B(N-1)_TOP
  - RIGHT handles: B0_BOT, B1_BOT, B2_BOT, ..., B(N-1)_BOT

Each direction does COMPLETE reduce-scatter (N-1 steps) for its halves.

From Device 0's perspective (8 devices):

  Step 0 (Prologue):
    Barrier with both neighbors
    LEFT:  Compute P₀(B1_TOP) for block 1's top half
    RIGHT: Compute P₀(B7_BOT) for block 7's bot half
    Send both to neighbors

  Steps 1 to N-2: For each step s:
    Signal neighbors, wait for capacity
    Start bidirectional DMA:
      - LEFT: send to D7, receive from D1
      - RIGHT: send to D1, receive from D7
    DELAYED COMPUTATION (overlapped with DMA):
      - LEFT:  Compute P₀(B(s+1)_TOP) for target block's top half
      - RIGHT: Compute P₀(B(7-s)_BOT) for target block's bot half
    Wait for DMAs
    Accumulate computation results to received data

  Final Step:
    Final DMA exchange
    Compute own block contributions:
      - LEFT:  Compute P₀(B0_TOP) for own block's top half
      - RIGHT: Compute P₀(B0_BOT) for own block's bot half
    Accumulate to received data
    Write output:
      - TOP half from LEFT direction
      - BOT half from RIGHT direction

KEY INSIGHTS:
1. NO COLLISION: LEFT always computes TOP halves, RIGHT always computes BOT halves
2. Even at midpoint (step 3 for 8 devices), they compute DIFFERENT halves of same block
3. PERFECTLY BALANCED: Every step has exactly 2 half-block matmuls
4. NO IDLE STEPS: Both directions always have compute work
5. 2X BANDWIDTH: Both ICI directions fully utilized
6. GOOD OVERLAP: Compute overlaps with bidirectional DMA
"""

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

Ref = Any


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


def mod(x: jax.Array, n: int) -> jax.Array:
    """Modulo operation that works with JAX arrays."""
    return lax.rem(x + n, n)


class KernelConfig(NamedTuple):
    """Configuration for the kernel."""

    num_devices: int
    m_block: int
    m_half_block: int
    bm: int = 128
    bn: int = 128
    bk: int = 128


def tiled_matmul_hbm(
    x_hbm_ref: Ref,
    y_hbm_ref: Ref,
    out_hbm_ref: Ref,
    x_vmem_ref: Ref,
    y_vmem_ref: Ref,
    acc_vmem_ref: Ref,
    out_vmem_ref: Ref,
    copy_sem: Ref,
    *,
    m_block_idx: int | jax.Array,
    m_size: int,
    bm: int,
    bn: int,
    bk: int,
):
    """Tiled matmul: out = x[m_block_idx*bm:m_block_idx*bm+m_size, :] @ y.T.

    This function tiles the matmul computation using async_copy for HBM<->VMEM transfers.
    The result OVERWRITES the output buffer (does not accumulate to existing values).

    IMPORTANT: Uses block indices (multiplied by bm) instead of raw offsets to help
    the Mosaic compiler prove tile alignment at compile time.

    Args:
        x_hbm_ref: Full X buffer in HBM [M, K_shard]
        y_hbm_ref: Y buffer in HBM [N, K_shard]
        out_hbm_ref: Output buffer in HBM [m_size, N]
        x_vmem_ref: VMEM scratch for x tile [bm, bk]
        y_vmem_ref: VMEM scratch for y tile [bn, bk]
        acc_vmem_ref: VMEM scratch for accumulator [bm, bn] in float32
        out_vmem_ref: VMEM scratch for output tile [bm, bn]
        copy_sem: DMA semaphore for async copies
        m_block_idx: Starting block index (offset = m_block_idx * bm)
        m_size: Number of rows to process
        bm: Block size for M dimension
        bn: Block size for N dimension
        bk: Block size for K dimension
    """
    _, k_shard = x_hbm_ref.shape
    n_total, _ = y_hbm_ref.shape

    num_m_tiles = m_size // bm
    num_n_tiles = n_total // bn
    num_k_tiles = k_shard // bk

    for m_tile in range(num_m_tiles):
        global_m_tile = m_block_idx + m_tile
        for n_tile in range(num_n_tiles):
            n_start = n_tile * bn

            acc_vmem_ref[...] = jnp.zeros((bm, bn), dtype=jnp.float32)

            for k_tile in range(num_k_tiles):
                k_start = k_tile * bk

                x_copy = pltpu.make_async_copy(
                    src_ref=x_hbm_ref.at[pl.ds(global_m_tile * bm, bm), pl.ds(k_start, bk)],
                    dst_ref=x_vmem_ref,
                    sem=copy_sem,
                )
                x_copy.start()
                x_copy.wait()

                y_copy = pltpu.make_async_copy(
                    src_ref=y_hbm_ref.at[pl.ds(n_start, bn), pl.ds(k_start, bk)],
                    dst_ref=y_vmem_ref,
                    sem=copy_sem,
                )
                y_copy.start()
                y_copy.wait()

                x_f32 = x_vmem_ref[...].astype(jnp.float32)
                y_f32 = y_vmem_ref[...].astype(jnp.float32)
                acc_vmem_ref[...] = acc_vmem_ref[...] + jnp.dot(x_f32, y_f32.T)

            out_vmem_ref[...] = acc_vmem_ref[...].astype(out_hbm_ref.dtype)

            out_copy = pltpu.make_async_copy(
                src_ref=out_vmem_ref,
                dst_ref=out_hbm_ref.at[pl.ds(m_tile * bm, bm), pl.ds(n_start, bn)],
                sem=copy_sem,
            )
            out_copy.start()
            out_copy.wait()


def tiled_add_hbm(
    src_hbm_ref: Ref,
    dst_hbm_ref: Ref,
    src_vmem_ref: Ref,
    dst_vmem_ref: Ref,
    copy_sem: Ref,
    *,
    bm: int,
    bn: int,
):
    """Tiled addition: dst += src, with HBM inputs.

    Adds the source buffer to the destination buffer in-place using async_copy.
    """
    m_size, n_total = src_hbm_ref.shape

    num_m_tiles = m_size // bm
    num_n_tiles = n_total // bn

    for m_tile in range(num_m_tiles):
        m_start = m_tile * bm
        for n_tile in range(num_n_tiles):
            n_start = n_tile * bn

            src_copy = pltpu.make_async_copy(
                src_ref=src_hbm_ref.at[pl.ds(m_start, bm), pl.ds(n_start, bn)],
                dst_ref=src_vmem_ref,
                sem=copy_sem,
            )
            src_copy.start()
            src_copy.wait()

            dst_copy = pltpu.make_async_copy(
                src_ref=dst_hbm_ref.at[pl.ds(m_start, bm), pl.ds(n_start, bn)],
                dst_ref=dst_vmem_ref,
                sem=copy_sem,
            )
            dst_copy.start()
            dst_copy.wait()

            result = src_vmem_ref[...].astype(jnp.float32) + dst_vmem_ref[...].astype(jnp.float32)
            dst_vmem_ref[...] = result.astype(dst_hbm_ref.dtype)

            out_copy = pltpu.make_async_copy(
                src_ref=dst_vmem_ref,
                dst_ref=dst_hbm_ref.at[pl.ds(m_start, bm), pl.ds(n_start, bn)],
                sem=copy_sem,
            )
            out_copy.start()
            out_copy.wait()


def _kernel(
    x_ref: Ref,
    y_ref: Ref,
    out_ref: Ref,
    scratch_ref: Ref,
    computation_scratch_ref: Ref,
    x_vmem_ref: Ref,
    y_vmem_ref: Ref,
    acc_vmem_ref: Ref,
    out_vmem_ref: Ref,
    add_vmem_ref: Ref,
    send_left_sem: Ref,
    recv_left_sem: Ref,
    send_right_sem: Ref,
    recv_right_sem: Ref,
    copy_sem: Ref,
    left_capacity_sem: Ref,
    right_capacity_sem: Ref,
    *,
    config: KernelConfig,
    axis_name: str,
):
    """Bidirectional Reduce-Scatter Matmul Kernel with M-split algorithm.

    Grid: (num_devices,) where each iteration is one ring step.

    Key insight: Split M into N blocks, each block into TOP and BOT halves.
    - LEFT direction handles all TOP halves (reduced via left ring)
    - RIGHT direction handles all BOT halves (reduced via right ring)

    This ensures:
    - No collision at midpoint (different halves)
    - Perfect load balance (every step has 2 half-block matmuls)
    - Full bandwidth utilization (both directions active)
    """
    num_devices = config.num_devices
    m_block = config.m_block
    m_half_block = config.m_half_block
    bm, bn, bk = config.bm, config.bn, config.bk

    ring_step = pl.program_id(0)

    my_id = lax.axis_index(axis_name)
    left_neighbor = mod(my_id - 1, num_devices)
    right_neighbor = mod(my_id + 1, num_devices)

    left_working_slot = lax.rem(ring_step, 2)
    left_receiving_slot = 1 - left_working_slot
    right_working_slot = 2 + lax.rem(ring_step, 2)
    right_receiving_slot = 5 - right_working_slot

    left_compute_slot = 0
    right_compute_slot = 1

    num_steps = num_devices
    is_first_step = ring_step == 0
    is_last_step = ring_step == num_steps - 1

    def get_left_target_block(step):
        """LEFT direction: compute TOP half of block (my_id + step + 1) % N."""
        return mod(my_id + step + 1, num_devices)

    def get_right_target_block(step):
        """RIGHT direction: compute BOT half of block (my_id - step - 1) % N."""
        return mod(my_id - step - 1, num_devices)

    def compute_matmul_top_half(block_idx, out_slot):
        """Compute matmul for TOP half of specified block.

        Result: x[block_top, :] @ y.T → computation_scratch_ref[out_slot]
        """
        m_block_idx = block_idx * (m_block // bm)

        tiled_matmul_hbm(
            x_hbm_ref=x_ref,
            y_hbm_ref=y_ref,
            out_hbm_ref=computation_scratch_ref.at[out_slot],
            x_vmem_ref=x_vmem_ref,
            y_vmem_ref=y_vmem_ref,
            acc_vmem_ref=acc_vmem_ref,
            out_vmem_ref=out_vmem_ref,
            copy_sem=copy_sem,
            m_block_idx=m_block_idx,
            m_size=m_half_block,
            bm=bm,
            bn=bn,
            bk=bk,
        )

    def compute_matmul_bot_half(block_idx, out_slot):
        """Compute matmul for BOT half of specified block.

        Result: x[block_bot, :] @ y.T → computation_scratch_ref[out_slot]
        """
        m_block_idx = block_idx * (m_block // bm) + (m_half_block // bm)

        tiled_matmul_hbm(
            x_hbm_ref=x_ref,
            y_hbm_ref=y_ref,
            out_hbm_ref=computation_scratch_ref.at[out_slot],
            x_vmem_ref=x_vmem_ref,
            y_vmem_ref=y_vmem_ref,
            acc_vmem_ref=acc_vmem_ref,
            out_vmem_ref=out_vmem_ref,
            copy_sem=copy_sem,
            m_block_idx=m_block_idx,
            m_size=m_half_block,
            bm=bm,
            bn=bn,
            bk=bk,
        )

    def accumulate_computation_to_slot(compute_slot, dst_slot):
        """Add computation_scratch_ref[compute_slot] to scratch_ref[dst_slot]."""
        tiled_add_hbm(
            src_hbm_ref=computation_scratch_ref.at[compute_slot],
            dst_hbm_ref=scratch_ref.at[dst_slot],
            src_vmem_ref=add_vmem_ref,
            dst_vmem_ref=out_vmem_ref,
            copy_sem=copy_sem,
            bm=bm,
            bn=bn,
        )

    def copy_computation_to_slot(compute_slot, dst_slot):
        """Copy computation_scratch_ref[compute_slot] to scratch_ref[dst_slot]."""
        local_copy = pltpu.make_async_copy(
            src_ref=computation_scratch_ref.at[compute_slot],
            dst_ref=scratch_ref.at[dst_slot],
            sem=copy_sem,
        )
        local_copy.start()
        local_copy.wait()

    def local_barrier():
        """Barrier with both neighbors using double-barrier pattern."""
        barrier_sem = pltpu.get_barrier_semaphore()

        pltpu.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=(left_neighbor,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        pltpu.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=(right_neighbor,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        pltpu.semaphore_wait(barrier_sem, 2)

        @functools.partial(pl.run_scoped, second_barrier=pltpu.SemaphoreType.REGULAR)
        def _(second_barrier):
            pltpu.semaphore_signal(
                second_barrier,
                inc=1,
                device_id=(left_neighbor,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            pltpu.semaphore_signal(
                second_barrier,
                inc=1,
                device_id=(right_neighbor,),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
            pltpu.semaphore_wait(second_barrier, 2)

    def signal_left_neighbor():
        """Signal left neighbor that we are ready to receive from them."""
        pltpu.semaphore_signal(
            left_capacity_sem,
            inc=1,
            device_id=(left_neighbor,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    def signal_right_neighbor():
        """Signal right neighbor that we are ready to receive from them."""
        pltpu.semaphore_signal(
            right_capacity_sem,
            inc=1,
            device_id=(right_neighbor,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )

    left_target_block = get_left_target_block(ring_step)
    right_target_block = get_right_target_block(ring_step)

    @pl.when(is_first_step)
    def _prologue():
        local_barrier()

        compute_matmul_top_half(left_target_block, left_compute_slot)
        compute_matmul_bot_half(right_target_block, right_compute_slot)

        copy_computation_to_slot(left_compute_slot, left_working_slot)
        copy_computation_to_slot(right_compute_slot, right_working_slot)

    @pl.when(~is_first_step)
    def _main_loop():
        signal_left_neighbor()
        signal_right_neighbor()

        pltpu.semaphore_wait(left_capacity_sem, 1)
        pltpu.semaphore_wait(right_capacity_sem, 1)

        remote_copy_to_left = pltpu.make_async_remote_copy(
            src_ref=scratch_ref.at[left_receiving_slot],
            dst_ref=scratch_ref.at[left_working_slot],
            send_sem=send_left_sem,
            recv_sem=recv_left_sem,
            device_id=(left_neighbor,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        remote_copy_to_left.start()

        remote_copy_to_right = pltpu.make_async_remote_copy(
            src_ref=scratch_ref.at[right_receiving_slot],
            dst_ref=scratch_ref.at[right_working_slot],
            send_sem=send_right_sem,
            recv_sem=recv_right_sem,
            device_id=(right_neighbor,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        remote_copy_to_right.start()

        compute_matmul_top_half(left_target_block, left_compute_slot)
        compute_matmul_bot_half(right_target_block, right_compute_slot)

        remote_copy_to_left.wait()
        remote_copy_to_right.wait()

        @pl.when(is_last_step)
        def _epilogue():
            accumulate_computation_to_slot(left_compute_slot, left_working_slot)
            accumulate_computation_to_slot(right_compute_slot, right_working_slot)

            top_copy = pltpu.make_async_copy(
                src_ref=scratch_ref.at[left_working_slot],
                dst_ref=out_ref.at[pl.ds(0, m_half_block), :],
                sem=copy_sem,
            )
            top_copy.start()
            top_copy.wait()

            bot_copy = pltpu.make_async_copy(
                src_ref=scratch_ref.at[right_working_slot],
                dst_ref=out_ref.at[pl.ds(m_half_block, m_half_block), :],
                sem=copy_sem,
            )
            bot_copy.start()
            bot_copy.wait()

        @pl.when(~is_last_step)
        def _accumulate():
            accumulate_computation_to_slot(left_compute_slot, left_working_slot)
            accumulate_computation_to_slot(right_compute_slot, right_working_slot)


def reduce_scatter_matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    axis_name: str = "x",
    tp_size: int | None = None,
    collective_id: int | None = 0,
    bm: int = 128,
    bn: int = 128,
    bk: int = 128,
) -> jax.Array:
    """Bidirectional reduce-scatter matmul with M-split algorithm.

    Computes: reduce_scatter(x @ y.T, scatter_dim=0)

    Where:
    - x @ y.T is computed with K sharded across devices
    - Result is scattered on M dimension (each device gets M/num_devices rows)

    The algorithm splits M into N blocks (one per device), and each block into
    TOP and BOT halves. LEFT direction handles TOP halves, RIGHT handles BOT halves.
    This achieves:
    - 2x ICI bandwidth utilization
    - Perfect load balance (every step has compute work)
    - No collision (directions always process different halves)

    Args:
        x: Input tensor [M, K_shard] where K is sharded across devices
        y: Weight tensor [N, K_shard] where K is sharded across devices
        axis_name: Name of the device axis for collective operations
        bm: Block size for M dimension (must divide M_half_block)
        bn: Block size for N dimension (must divide N)
        bk: Block size for K dimension (must divide K_shard)

    Returns:
        Output tensor [M_block, N] where M is scattered across devices
    """
    tp_size = _resolve_tp_size(tp_size, axis_name)
    if tp_size == 1:
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(f"Inputs must be 2D, got shapes {x.shape} and {y.shape}.")
        if x.dtype != y.dtype:
            raise ValueError(f"Input dtypes must match, got {x.dtype} and {y.dtype}.")
        if x.shape[1] != y.shape[1]:
            raise ValueError(f"Incompatible shapes for matmul: contracting dimension mismatch: {x.shape} and {y.shape}.")
        return jnp.dot(x, y.T, preferred_element_type=jnp.float32).astype(x.dtype)
    num_devices = int(tp_size)

    m_total, k_shard = x.shape
    n_total, _ = y.shape

    if m_total % num_devices != 0:
        raise ValueError(f"M ({m_total}) must be divisible by num_devices ({num_devices}).")

    m_block = m_total // num_devices

    if m_block % 2 != 0:
        raise ValueError(f"M_block ({m_block}) must be divisible by 2.")

    m_half_block = m_block // 2

    if m_half_block % bm != 0:
        raise ValueError(f"M_half_block ({m_half_block}) must be divisible by bm ({bm}).")
    if n_total % bn != 0:
        raise ValueError(f"N ({n_total}) must be divisible by bn ({bn}).")
    if k_shard % bk != 0:
        raise ValueError(f"K_shard ({k_shard}) must be divisible by bk ({bk}).")

    if x.dtype in (jnp.bfloat16, jnp.float16):
        partial_out = jnp.dot(x, y.T, precision=jax.lax.Precision.DEFAULT)
        return lax.psum_scatter(partial_out, axis_name=axis_name, scatter_dimension=0, tiled=True)

    config = KernelConfig(
        num_devices=num_devices,
        m_block=m_block,
        m_half_block=m_half_block,
        bm=bm,
        bn=bn,
        bk=bk,
    )

    out_shape = jax.ShapeDtypeStruct((m_block, n_total), x.dtype)

    scratch_shape = jax.ShapeDtypeStruct((4, m_half_block, n_total), x.dtype)

    computation_scratch_shape = jax.ShapeDtypeStruct((2, m_half_block, n_total), x.dtype)

    x_vmem_shape = pltpu.VMEM((bm, bk), x.dtype)
    y_vmem_shape = pltpu.VMEM((bn, bk), y.dtype)
    acc_vmem_shape = pltpu.VMEM((bm, bn), jnp.float32)
    out_vmem_shape = pltpu.VMEM((bm, bn), x.dtype)
    add_vmem_shape = pltpu.VMEM((bm, bn), x.dtype)

    grid = (num_devices,)

    def kernel_fn(
        x_ref,
        y_ref,
        out_ref,
        scratch_ref,
        computation_scratch_ref,
        x_vmem_ref,
        y_vmem_ref,
        acc_vmem_ref,
        out_vmem_ref,
        add_vmem_ref,
        send_left_sem,
        recv_left_sem,
        send_right_sem,
        recv_right_sem,
        copy_sem,
        left_capacity_sem,
        right_capacity_sem,
    ):
        _kernel(
            x_ref=x_ref,
            y_ref=y_ref,
            out_ref=out_ref,
            scratch_ref=scratch_ref,
            computation_scratch_ref=computation_scratch_ref,
            x_vmem_ref=x_vmem_ref,
            y_vmem_ref=y_vmem_ref,
            acc_vmem_ref=acc_vmem_ref,
            out_vmem_ref=out_vmem_ref,
            add_vmem_ref=add_vmem_ref,
            send_left_sem=send_left_sem,
            recv_left_sem=recv_left_sem,
            send_right_sem=send_right_sem,
            recv_right_sem=recv_right_sem,
            copy_sem=copy_sem,
            left_capacity_sem=left_capacity_sem,
            right_capacity_sem=right_capacity_sem,
            config=config,
            axis_name=axis_name,
        )

    out, _, _ = pl.pallas_call(
        kernel_fn,
        out_shape=(out_shape, scratch_shape, computation_scratch_shape),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            scratch_shapes=[
                x_vmem_shape,
                y_vmem_shape,
                acc_vmem_shape,
                out_vmem_shape,
                add_vmem_shape,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.REGULAR,
                pltpu.SemaphoreType.REGULAR,
            ],
            grid=grid,
        ),
        compiler_params=pltpu.CompilerParams(collective_id=collective_id, dimension_semantics=("arbitrary",)),
    )(x, y)

    return out


ALGORITHM_DIAGRAM = """
BIDIRECTIONAL REDUCE-SCATTER MATMUL WITH M-SPLIT ALGORITHM
===========================================================

M dimension split into N blocks, each block split into TOP and BOT halves:

Full M dimension (8 devices):
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │ Block 5 │ Block 6 │ Block 7 │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

Each block split:
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│B0_T│B0_B│B1_T│B1_B│B2_T│B2_B│B3_T│B3_B│B4_T│B4_B│B5_T│B5_B│B6_T│B6_B│B7_T│B7_B│
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
  │         │         │         │         │         │         │         │
  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
                            LEFT (all TOP halves)
        └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
                            RIGHT (all BOT halves)

Direction assignment:
  LEFT:  All TOP halves → reduces via left ring (D0→D7→D6→...→D1→D0)
  RIGHT: All BOT halves → reduces via right ring (D0→D1→D2→...→D7→D0)

Device 0's computation schedule (8 devices):
┌──────────┬─────────────────────────┬─────────────────────────┐
│   Step   │     LEFT (TOP half)     │    RIGHT (BOT half)     │
├──────────┼─────────────────────────┼─────────────────────────┤
│    0     │ P₀(B1_TOP) → send to D7 │ P₀(B7_BOT) → send to D1 │
│    1     │ P₀(B2_TOP) + accum      │ P₀(B6_BOT) + accum      │
│    2     │ P₀(B3_TOP) + accum      │ P₀(B5_BOT) + accum      │
│    3     │ P₀(B4_TOP) + accum      │ P₀(B4_BOT) + accum      │  ← Same block, DIFFERENT halves!
│    4     │ P₀(B5_TOP) + accum      │ P₀(B3_BOT) + accum      │
│    5     │ P₀(B6_TOP) + accum      │ P₀(B2_BOT) + accum      │
│    6     │ P₀(B7_TOP) + accum      │ P₀(B1_BOT) + accum      │
│  Final   │ P₀(B0_TOP) → output TOP │ P₀(B0_BOT) → output BOT │
└──────────┴─────────────────────────┴─────────────────────────┘

Pipeline timeline (Device 0):
Time ────────────────────────────────────────────────────────────────────────────►

Step:     0      1      2      3      4      5      6     Final
         ───    ───    ───    ───    ───    ───    ───    ─────

LEFT:    ████   ████   ████   ████   ████   ████   ████   ████
(TOP)    B1_T   B2_T   B3_T   B4_T   B5_T   B6_T   B7_T   B0_T

RIGHT:   ████   ████   ████   ████   ████   ████   ████   ████
(BOT)    B7_B   B6_B   B5_B   B4_B   B3_B   B2_B   B1_B   B0_B

L-DMA:          ════   ════   ════   ════   ════   ════   ════
                →D7    →D7    →D7    →D7    →D7    →D7    →D7

R-DMA:          ════   ════   ════   ════   ════   ════   ════
                →D1    →D1    →D1    →D1    →D1    →D1    →D1

Matmuls: [2]    [2]    [2]    [2]    [2]    [2]    [2]    [2]

KEY BENEFITS:
✓ NO COLLISION: LEFT and RIGHT always process DIFFERENT halves
✓ PERFECT BALANCE: Every step has exactly 2 half-block matmuls
✓ NO IDLE STEPS: Both directions always have compute work
✓ 2X BANDWIDTH: Both ICI directions fully utilized
✓ GOOD OVERLAP: Compute overlaps with bidirectional DMA

Final output at Device 0:
┌─────────────────────────────────────────────────────────────┐
│  Block 0 TOP (from LEFT):  P₀ + P₁ + P₂ + ... + P₇        │
├─────────────────────────────────────────────────────────────┤
│  Block 0 BOT (from RIGHT): P₀ + P₁ + P₂ + ... + P₇        │
└─────────────────────────────────────────────────────────────┘
Combined: Complete Block 0 with full reduction from all 8 devices ✓
"""

if __name__ == "__main__":
    print(ALGORITHM_DIAGRAM)
