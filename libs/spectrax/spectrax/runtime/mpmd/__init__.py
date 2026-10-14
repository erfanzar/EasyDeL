# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""True MPMD pipeline runtime.

Each physical pipeline rank compiles and executes its own distinct JAX program.
The public entry points here cover both training-style schedules
(``sxcall``, ``sxgrad``, ``sxvalue_and_grad``) and forward-only inference
dispatch (``sxjit`` plus ``MpmdPipelineExecutor``). The executor reuses the
``sxjit`` prepared stage plan to run same-shaped microbatches as a host
wavefront, while the schedule APIs drive full forward/backward MPMD execution.
"""

from __future__ import annotations

from .compiler import (
    compile_ranked_executables,
    get_num_stages,
    get_num_stages_from_grid,
    run_ranked_pipeline,
)
from .markers import (
    cluster_jaxpr_by_markers,
    split_by_markers,
    sxenter_loop,
    sxexit_loop,
    sxloop,
    sxstage_iter,
    sxstage_region,
)
from .per_rank import (
    compile_per_rank_bwd,
    compile_per_rank_fwd,
    extract_rank_actions,
    run_gpipe_per_rank,
)
from .pipeline_executor import (
    MpmdPipelineDispatchStats,
    MpmdPipelineExecutor,
)
from .pscan_compiler import (
    PscanPlan,
    build_pscan_plan,
    dispatch_pscan,
    has_pscan,
)
from .runtime import (
    collect_task_times_ms,
    sxcall,
    sxgrad,
    sxjit,
    sxvalue_and_grad,
)
from .training_step import sxvalue_and_grad_and_apply
from .treduce import (
    Add,
    Concat,
    Max,
    Op,
    pscan_p,
    treduce,
    treduce_i,
)

__all__ = [
    "Add",
    "Concat",
    "Max",
    "MpmdPipelineDispatchStats",
    "MpmdPipelineExecutor",
    "Op",
    "PscanPlan",
    "build_pscan_plan",
    "cluster_jaxpr_by_markers",
    "collect_task_times_ms",
    "compile_per_rank_bwd",
    "compile_per_rank_fwd",
    "compile_ranked_executables",
    "dispatch_pscan",
    "extract_rank_actions",
    "get_num_stages",
    "get_num_stages_from_grid",
    "has_pscan",
    "pscan_p",
    "run_gpipe_per_rank",
    "run_ranked_pipeline",
    "split_by_markers",
    "sxcall",
    "sxenter_loop",
    "sxexit_loop",
    "sxgrad",
    "sxjit",
    "sxloop",
    "sxstage_iter",
    "sxstage_region",
    "sxvalue_and_grad",
    "sxvalue_and_grad_and_apply",
    "treduce",
    "treduce_i",
]
