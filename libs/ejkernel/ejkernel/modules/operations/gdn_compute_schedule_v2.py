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

"""Public operation wrapper for GDN v2 schedule-table construction.

The schedule table describes how a packed batch is split into decode work,
regular prefill chunks, and transition rows. TPU Pallas GDN v2 consumes this
table from SMEM while XLA/reference paths use the same public operation for
parity and host-side tests.
"""

from __future__ import annotations

import os

import jax
from jaxtyping import Array, Int

from ejkernel.kernels._registry import kernel_registry
from ejkernel.ops import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    Invocation,
    Kernel,
    Tuner,
)
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import GDNComputeScheduleV2Config


class GDNComputeScheduleV2(Kernel[GDNComputeScheduleV2Config, tuple[Array, Array]]):
    """Executor-driven schedule-table builder for packed GDN v2.

    Implements the :class:`~ejkernel.ops.Kernel` protocol for constructing the
    GDN v2 schedule table that splits a packed batch into decode work, prefill
    chunks, and sublane-transition rows. The builder is pure JAX/XLA helper logic
    with no tunable tile parameters; the config only steers registry dispatch.

    Attributes:
        version: Cache-busting version tag for the persistent config cache.
    """

    version = "1"

    def __init__(self):
        """Initialize the GDN v2 schedule-table kernel.

        Registers the operation under ``op_id="gdn_compute_schedule_v2"`` for
        registry and persistent-cache lookups.
        """
        super().__init__(op_id="gdn_compute_schedule_v2")

    def get_impl(self, cfg: GDNComputeScheduleV2Config):
        """Resolve the backend implementation for a config.

        Args:
            cfg: Operation config carrying the requested ``platform`` and
                ``backend``.

        Returns:
            The registered callable building the GDN v2 schedule table for the
            detected platform and the config's backend.
        """
        platform = detect_platform("gdn_compute_schedule_v2", cfg.platform)
        return kernel_registry.get("gdn_compute_schedule_v2", platform=platform, backend=cfg.backend)

    def run(
        self,
        query_start_loc: Int[Array, "num_requests_plus_one"],
        decode_tokens: int | Int[Array, ""],
        num_valid_seqs: int | Int[Array, ""],
        max_tokens: int,
        chunk_size: int,
        BT: int | None = None,
        alignment: int = 8,
        *,
        cfg: GDNComputeScheduleV2Config,
    ) -> tuple[Int[Array, "max_schedule_blocks table_cols"], Int[Array, ""]]:
        """Build the schedule table with the selected backend.

        Args:
            query_start_loc: CSR-style request offsets.
            decode_tokens: Number of leading decode tokens in the packed batch.
            num_valid_seqs: Number of valid request rows.
            max_tokens: Maximum token count in the packed buffer.
            chunk_size: Prefill chunk size.
            BT: Decode token block size. Defaults to ``chunk_size`` in the
                backend when omitted.
            alignment: Token-row alignment used by the schedule builder.
            cfg: Operation config selected by the executor.

        Returns:
            ``(schedule_table, total_blocks)``.
        """
        impl = self.get_impl(cfg)
        return impl(
            query_start_loc=query_start_loc,
            decode_tokens=decode_tokens,
            num_valid_seqs=num_valid_seqs,
            max_tokens=int(max_tokens),
            chunk_size=int(chunk_size),
            BT=None if BT is None else int(BT),
            alignment=int(alignment),
        )

    def heuristic_cfg(self, inv: Invocation[GDNComputeScheduleV2Config, tuple[Array, Array]]):
        """Return the default XLA config for an invocation.

        Args:
            inv: Invocation carrying the call arguments and kwargs.

        Returns:
            A :class:`GDNComputeScheduleV2Config` on ``platform="xla"`` with
            ``backend="any"``.
        """
        return GDNComputeScheduleV2Config(platform="xla", backend="any")

    def candidate_cfgs(self, inv: Invocation[GDNComputeScheduleV2Config, tuple[Array, Array]]):
        """Return the default candidate config list (heuristic only).

        Args:
            inv: Invocation carrying the call arguments and kwargs.

        Returns:
            A single-element list with the heuristic XLA config.
        """
        return [self.heuristic_cfg(inv)]

    def candidate_cfgs_gpu(self, inv: Invocation[GDNComputeScheduleV2Config, tuple[Array, Array]]):
        """Return GPU autotuning candidates (XLA only).

        The schedule builder has no accelerator-specific kernel, so the GPU set
        is a single XLA config.

        Args:
            inv: Invocation carrying the call arguments and kwargs.

        Returns:
            A single-element list with the XLA config.
        """
        return [GDNComputeScheduleV2Config(platform="xla", backend="any")]

    def candidate_cfgs_tpu(self, inv: Invocation[GDNComputeScheduleV2Config, tuple[Array, Array]]):
        """Return TPU autotuning candidates (XLA only).

        The schedule builder runs as pure XLA even when feeding TPU Pallas GDN
        kernels, so the TPU set is a single XLA config.

        Args:
            inv: Invocation carrying the call arguments and kwargs.

        Returns:
            A single-element list with the XLA config.
        """
        return [GDNComputeScheduleV2Config(platform="xla", backend="any")]

    candidate_cfgs_xla = candidate_cfgs
    candidate_cfgs_any = candidate_cfgs


_executor: Executor[GDNComputeScheduleV2Config, tuple[Array, Array]] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=False,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=1, iters=1),
        persistent=PersistentCache("gdn_compute_schedule_v2"),
    )
)


def compute_schedule_table_v2(
    query_start_loc: jax.Array,
    decode_tokens: int | jax.Array,
    num_valid_seqs: int | jax.Array,
    max_tokens: int,
    chunk_size: int,
    BT: int | None = None,
    alignment: int = 8,
    *,
    cfg: GDNComputeScheduleV2Config | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Build the GDN v2 schedule table through the eJKernel operation stack.

    Each schedule row represents either a decode batch, a regular prefill
    chunk, or a transition row for sublane-misaligned request boundaries. TPU
    Pallas kernels consume this table from SMEM to decide which work path each
    grid step should run.

    Args:
        query_start_loc: CSR-style request offsets, shape
            ``[num_requests + 1]``.
        decode_tokens: Number of leading decode tokens in the packed stream.
        num_valid_seqs: Number of valid non-padding request rows.
        max_tokens: Static upper bound on the packed token count.
        chunk_size: Prefill chunk size.
        BT: Decode token block size. Defaults to ``chunk_size`` when omitted
            by the backend.
        alignment: Token-row alignment used for sublane transition metadata.
        cfg: Optional operation config.

    Returns:
        ``(schedule_table, total_blocks)``. ``schedule_table`` has shape
        ``[safe_max_blocks, 11 + 3 * alignment]``.

    Note:
        Column layout of ``schedule_table``:

        * 0: prefill-valid flag.
        * 1: prefill block token offset.
        * 2: prefill request index.
        * 3: prefill valid-token count.
        * 4: decode-valid flag.
        * 5: decode batch token offset.
        * 6: decode starting request id.
        * 7: decode request count.
        * 8: prefill block-is-last flag.
        * 9: prefill block-is-first flag.
        * 10: transition-block flag.
        * ``11 .. 10 + alignment``: per-sublane request ids.
        * next ``alignment``: per-sublane first-token flags.
        * next ``alignment``: per-sublane last-token flags.
    """
    return _executor(
        GDNComputeScheduleV2(),
        query_start_loc,
        decode_tokens,
        num_valid_seqs,
        max_tokens,
        chunk_size,
        BT,
        alignment,
        _cfg=cfg or GDNComputeScheduleV2Config(platform="xla", backend="any"),
    )


gdn_compute_schedule_v2 = compute_schedule_table_v2

__all__ = (
    "GDNComputeScheduleV2",
    "compute_schedule_table_v2",
    "gdn_compute_schedule_v2",
)
