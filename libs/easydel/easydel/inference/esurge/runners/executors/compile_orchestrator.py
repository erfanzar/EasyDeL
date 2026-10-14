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

"""Startup compile orchestration for eSurge execution.

Extracted verbatim from ``ExecutionManager`` (phase 11.4 of the eSurge
modularization). :class:`CompileOrchestrator` owns the startup-only compile
drivers: bucket-grid planning (:meth:`CompileOrchestrator.compile`), the
per-variant compile helpers (SPMD fused model step, MPMD fused PP model step,
backbone, LM head, sampler), and the post-warmup KV-page reset. It reaches
compiled-executor caches, cache pages, and debug baselines through a manager
back-reference; per-variant calls are dispatched *through* the manager's thin
delegators so instance-level overrides (tests) keep intercepting them.

The runtime-path bucket check ``_require_precompiled_variants`` intentionally
stays on ``ExecutionManager`` — it runs on every execute() call, not at
startup.

This module also hosts the ``_tree_hash`` / ``_tree_hash_diff`` debug helpers
used to record and diff compile-time input baselines; ``ExecutionManager``
imports them back for its DEBUG_MODE recompilation diagnostics.
"""

from __future__ import annotations

import gc
import hashlib
import typing as tp
from functools import partial

import jax
import numpy
from eformer.loggings import ProgressLogger, get_logger
from eformer.pytree import key_path_to_str
from jax import numpy as jnp

from easydel.caching import RaggedPagesCacheConfig, UnifiedAttentionCacheConfig

from ..execution_types import StepFunctionInputs
from .sampler_executor import _get_padded_num_reqs_with_upper_limit

if tp.TYPE_CHECKING:
    from ..execution_manager import ExecutionManager

logger = get_logger("eSurge-CompileOrchestrator")


def _tree_hash(tree):
    """Compute a hash tree for debugging structure/shape/dtype changes.

    Creates a PyTree with the same structure where each leaf is replaced
    by a hash string encoding its type, shape, dtype, and sharding.

    Args:
        tree: PyTree to hash.

    Returns:
        PyTree with same structure where leaves are hash strings encoding
        their original type, shape, dtype, and sharding information.

    Note:
        This is used for debugging recompilation issues by comparing hash
        trees between compilation and execution to detect structural changes.
    """

    def _map(p, x):
        """Hash a single PyTree leaf into a stable string for diffing."""
        p = key_path_to_str(p)
        maybe_info = (
            f"-{type(x)}"
            + "-"
            + str(getattr(x, "shape", "None"))
            + "-"
            + str(getattr(x, "dtype", "None"))
            + "-"
            + str(getattr(x, "sharding", "None"))
        )
        if isinstance(x, int | float | bool):
            maybe_info = f"-{x}"
        return (
            hashlib.md5(
                str(
                    p
                    + str(type(x))
                    + str(getattr(x, "shape", "None"))
                    + str(getattr(x, "dtype", "None"))
                    + str(getattr(x, "sharding", "None"))
                ).encode()
            ).hexdigest()
            + maybe_info
        )

    return jax.tree_util.tree_map_with_path(
        _map,
        tree,
        is_leaf=lambda x: isinstance(
            x,
            jax.Array | numpy.ndarray | int | float | bool | None,
        ),
    )


def _tree_hash_diff(orgin, new):
    """Compare two hash trees and print differences.

    Compares hash trees created by _tree_hash() and prints any paths
    where the hashes differ, helping debug unexpected recompilations.

    Args:
        orgin: Original hash tree (typically from compilation).
        new: New hash tree (typically from execution).

    Returns:
        PyTree of booleans indicating whether each leaf matches.
    """

    def _map(p, t1, t2):
        """Compare two leaf hashes and report mismatches by path."""
        p = key_path_to_str(p)
        oo = t1 == t2
        if not oo:
            print(f"path: {p} out: {oo} orgin: {t1} new: {t2}")
        return oo

    return jax.tree_util.tree_map_with_path(_map, orgin, new, is_leaf=lambda x: isinstance(x, str))


class CompileOrchestrator:
    """Startup compile drivers for ``ExecutionManager``.

    Stateless beyond the manager back-reference: compiled variants live in
    the manager's model/sampler executors, warm-up cache pages on
    ``manager.kv_pages``, and debug hash baselines in
    ``manager._debug_baselines``.
    """

    def __init__(self, manager: "ExecutionManager") -> None:
        """Bind the owning manager.

        Args:
            manager: The :class:`ExecutionManager` whose executors and cache
                pages the compile drivers operate on.
        """
        self._manager = manager

    @staticmethod
    def _get_feasible_compile_pairs(
        num_tokens_paddings: list[int],
        reqs_padds: list[int],
    ) -> list[tuple[int, int]]:
        """Return only schedulable token/request bucket combinations.

        Every scheduled request consumes at least one token in a runner step,
        so a request bucket larger than the token bucket cannot be reached at
        runtime. Pruning those pairs shortens startup and avoids compiling dead
        executables such as `(32 tokens, 256 requests)`.
        """
        feasible_pairs: list[tuple[int, int]] = []
        for num_tokens in num_tokens_paddings:
            for reqs_padd in reqs_padds:
                if int(reqs_padd) > int(num_tokens):
                    continue
                feasible_pairs.append((int(num_tokens), int(reqs_padd)))
        return feasible_pairs

    def compile(
        self,
        num_tokens_paddings: list[int],
        num_reqs_max_model_len: int,
        max_pages_per_req: int,
        max_num_reqs: int,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig,
        num_reqs_paddings: list[int] | None = None,
        prune_infeasible_pairs: bool = True,
    ) -> None:
        """Compile model execution functions for various input configurations.

        Pre-compiles functions for different combinations of token counts and request
        counts to avoid runtime compilation overhead. This enables seamless switching
        between different batch sizes during inference.

        Args:
            num_tokens_paddings: List of token count configurations to compile.
            num_reqs_max_model_len: Maximum number of requests at max model length.
            max_pages_per_req: Maximum number of KV cache pages per request.
            max_num_reqs: Maximum number of concurrent requests.
            metadata: Pages cache metadata containing configuration details.
            prune_infeasible_pairs: Whether to skip compile pairs where the
                padded request bucket exceeds the token bucket. This is only
                safe when runtime compacts interior zero-token rows before
                bucket selection.

        Note:
            Compilation progress is logged using a progress bar. The total number
            of compilations is len(num_tokens_paddings) * number of unique padded
            request counts.

        Example:
            >>> executor.compile(
            ...     num_tokens_paddings=[128, 256, 512, 1024],
            ...     num_reqs_max_model_len=16,
            ...     max_pages_per_req=64,
            ...     max_num_reqs=32,
            ...     metadata=cache_metadata
            ... )
        """

        if self._manager.use_aot_forward:
            self._manager.clear_cache()
        if num_reqs_paddings:
            reqs_padds = sorted({int(n) for n in num_reqs_paddings if 0 < int(n) <= max_num_reqs})
        else:
            ufn = partial(_get_padded_num_reqs_with_upper_limit, min_input_pad=self._manager.min_input_pad)
            reqs_padds = sorted({ufn(n, max_num_reqs) for n in range(1, max_num_reqs + 1)})
        if not reqs_padds:
            reqs_padds = [max_num_reqs]
        sampler_ufn = partial(
            _get_padded_num_reqs_with_upper_limit,
            min_input_pad=int(getattr(self._manager, "_sampler_min_input_pad", 1)),
        )
        sampler_reqs_padds = sorted(
            {
                *reqs_padds,
                *(sampler_ufn(n, max_num_reqs) for n in range(1, max_num_reqs + 1)),
            }
        )
        if prune_infeasible_pairs:
            sampler_compile_pairs = self._get_feasible_compile_pairs(num_tokens_paddings, sampler_reqs_padds)
        else:
            sampler_compile_pairs = [
                (int(num_tokens), int(reqs_padd))
                for num_tokens in num_tokens_paddings
                for reqs_padd in sampler_reqs_padds
            ]

        model_tokens = sorted({int(t) for t in num_tokens_paddings})
        self._manager._model_num_tokens_paddings = model_tokens
        all_reqs_padds = sorted({int(r) for _, r in sampler_compile_pairs})
        sampler_reqs_padds_to_compile = all_reqs_padds
        if self._manager.model.mesh.is_mpmd:
            model_tokens_to_compile = model_tokens
            reqs_to_compile = reqs_padds

            if self._manager._model_executor.supports_pipeline_model_step:
                pp_model_reqs_to_compile = reqs_to_compile
                pp_model_pairs = [
                    (int(num_tokens), int(reqs_padd))
                    for num_tokens in model_tokens_to_compile
                    for reqs_padd in pp_model_reqs_to_compile
                    if self._manager._should_use_fused_pipeline_model_step(
                        num_tokens=int(num_tokens),
                        padded_num_reqs=int(reqs_padd),
                    )
                ]
                pp_model_progress = ProgressLogger("eSurge-pp-model-step", logger)
                for idx, (num_tokens, reqs_padd) in enumerate(pp_model_pairs):
                    pp_model_progress.update(
                        idx,
                        len(pp_model_pairs),
                        f"Compiling [{idx + 1}/{len(pp_model_pairs)}]: {num_tokens:5d} tokens / {reqs_padd:2d} reqs",
                    )
                    self._manager._compile_pipeline_model_step_variant(
                        num_tokens=num_tokens,
                        padded_num_reqs=reqs_padd,
                        max_num_reqs=max_num_reqs,
                        metadata=metadata,
                    )
                pp_model_progress.complete(f"All {len(pp_model_pairs)} fused PP model-step compilations completed")
            backbone_variants = [False]
            if getattr(self._manager._model_executor, "pipeline_runtime", None) is not None:
                backbone_variants.append(True)
            backbone_total = len(model_tokens_to_compile) * len(backbone_variants)
            backbone_progress = ProgressLogger("eSurge-backbone", logger)
            backbone_idx = 0
            for num_tokens in model_tokens_to_compile:
                for use_pipeline_runtime in backbone_variants:
                    mode_name = "runtime" if use_pipeline_runtime else "direct"
                    backbone_progress.update(
                        backbone_idx,
                        backbone_total,
                        f"Compiling [{backbone_idx + 1}/{backbone_total}]: {num_tokens:5d} tokens ({mode_name})",
                    )
                    self._manager._compile_backbone_variant(
                        num_tokens=num_tokens,
                        max_num_reqs=max_num_reqs,
                        metadata=metadata,
                        use_pipeline_runtime=use_pipeline_runtime,
                    )
                    backbone_idx += 1
            backbone_progress.complete(f"All {backbone_total} backbone compilations completed")

            lm_head_progress = ProgressLogger("eSurge-head-sampler", logger)
            lm_head_reqs_to_compile = reqs_to_compile
            total_phase2 = len(lm_head_reqs_to_compile) + len(sampler_reqs_padds_to_compile)
            phase2_idx = 0
            for reqs_padd in lm_head_reqs_to_compile:
                lm_head_progress.update(
                    phase2_idx,
                    total_phase2,
                    f"Compiling [{phase2_idx + 1}/{total_phase2}]: {reqs_padd:2d} padded requests",
                )
                self._manager._compile_lm_head_variant(
                    padded_num_reqs=reqs_padd,
                    max_num_reqs=max_num_reqs,
                    metadata=metadata,
                )
                phase2_idx += 1
            self._manager._reset_kv_pages_after_compile_warmup()
        else:
            reqs_to_compile = all_reqs_padds
            if prune_infeasible_pairs:
                model_compile_pairs = self._get_feasible_compile_pairs(model_tokens, reqs_padds)
            else:
                model_compile_pairs = [
                    (int(num_tokens), int(reqs_padd)) for num_tokens in model_tokens for reqs_padd in reqs_padds
                ]
            model_step_progress = ProgressLogger("eSurge-model-step", logger)
            for idx, (num_tokens, reqs_padd) in enumerate(model_compile_pairs):
                model_step_progress.update(
                    idx,
                    len(model_compile_pairs),
                    f"Compiling [{idx + 1}/{len(model_compile_pairs)}]: {num_tokens:5d} tokens / {reqs_padd:2d} reqs",
                )
                self._manager._compile_model_step_variant(
                    num_tokens=num_tokens,
                    padded_num_reqs=reqs_padd,
                    max_num_reqs=max_num_reqs,
                    metadata=metadata,
                )
            model_step_progress.complete(f"All {len(model_compile_pairs)} fused model-step compilations completed")

            lm_head_progress = ProgressLogger("eSurge-sampler", logger)
            total_phase2 = len(sampler_reqs_padds_to_compile)
            phase2_idx = 0
        sampler_dummy_tokens = max(model_tokens, default=max_num_reqs)
        for reqs_padd in sampler_reqs_padds_to_compile:
            num_tokens = max(int(sampler_dummy_tokens), int(reqs_padd))
            lm_head_progress.update(
                phase2_idx,
                total_phase2,
                f"Compiling [{phase2_idx + 1}/{total_phase2}]: sampler {reqs_padd:2d} padded requests",
            )
            self._manager._compile_sampler_variant(
                num_tokens=num_tokens,
                max_num_reqs=max_num_reqs,
                padded_num_reqs=reqs_padd,
                metadata=metadata,
            )
            phase2_idx += 1
        if self._manager.model.mesh.is_mpmd:
            lm_head_progress.complete(f"All {total_phase2} lm_head/sampler compilations completed")
        else:
            lm_head_progress.complete(f"All {total_phase2} sampler compilations completed")

    def _compile_model_step_variant(
        self,
        *,
        num_tokens: int,
        padded_num_reqs: int,
        max_num_reqs: int,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig,
    ) -> None:
        """Compile the fused SPMD model step for one token/request bucket pair."""
        if self._manager._model_executor.has_model_step(num_tokens, padded_num_reqs):
            return

        compargs = self._manager.get_compile_configurations(
            self._manager.kv_pages,
            self._manager.rng_key,
            num_tokens,
            max_num_reqs,
            padded_num_reqs,
            metadata,
        )
        graphdef, graphstate, graphother, inputs = compargs
        model_out = self._manager._model_executor.compile_model_step(
            num_tokens=num_tokens,
            padded_num_reqs=padded_num_reqs,
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
            inputs=inputs,
        )
        if model_out is not None and not self._manager.model.mesh.is_mpmd:
            self._manager.kv_pages = model_out.kv_pages
        if self._manager.use_aot_forward:
            warm_args = (graphstate, graphother, inputs)
            self._manager._debug_baselines[f"{num_tokens}_{padded_num_reqs}_hash_in_model_step"] = _tree_hash(warm_args)

    def _compile_pipeline_model_step_variant(
        self,
        *,
        num_tokens: int,
        padded_num_reqs: int,
        max_num_reqs: int,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig,
    ) -> None:
        """Compile the fused MPMD PP model step for one token/request bucket."""
        if not self._manager._model_executor.supports_pipeline_model_step:
            return
        if self._manager._model_executor.has_model_step(num_tokens, padded_num_reqs):
            return

        compargs = self._manager.get_compile_configurations(
            self._manager.kv_pages,
            self._manager.rng_key,
            num_tokens,
            max_num_reqs,
            padded_num_reqs,
            metadata,
        )
        graphdef, graphstate, graphother, inputs = compargs
        model_out = self._manager._model_executor.compile_pipeline_model_step(
            num_tokens=num_tokens,
            padded_num_reqs=padded_num_reqs,
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
            inputs=inputs,
        )
        if model_out is not None and not self._manager.model.mesh.is_mpmd:
            self._manager.kv_pages = model_out.kv_pages
        elif model_out is not None:
            self._manager.kv_pages = model_out.kv_pages
        if self._manager.use_aot_forward:
            warm_args = (graphstate, graphother, inputs)
            self._manager._debug_baselines[f"{num_tokens}_{padded_num_reqs}_hash_in_pp_model_step"] = _tree_hash(
                warm_args
            )

    def _compile_backbone_variant(
        self,
        *,
        num_tokens: int,
        max_num_reqs: int,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig,
        use_pipeline_runtime: bool = True,
    ) -> None:
        """Compile the backbone (transformer forward) for a token bucket.

        Keyed by ``num_tokens`` only.  Uses ``max_num_reqs`` as the dummy
        ``padded_num_reqs`` for the warm-up input shapes (metadata shapes are
        fixed); the decode-layout request bucket passed to
        :meth:`ModelStepExecutor.compile_backbone` depends on the variant —
        see the call-site comment.
        """
        if self._manager._has_backbone_variant(num_tokens, use_pipeline_runtime=use_pipeline_runtime):
            return

        compargs = self._manager.get_compile_configurations(
            self._manager.kv_pages,
            self._manager.rng_key,
            num_tokens,
            max_num_reqs,
            max_num_reqs,  # padded_num_reqs = max_num_reqs (shapes are fixed)
            metadata,
        )
        graphdef, graphstate, graphother, inputs = compargs
        # The resident PP-runtime variant only ever serves decode wavefronts,
        # where every request contributes exactly one token — its real request
        # bucket IS the token bucket, so it traces under the decode layout.
        # The direct-dispatch variant serves mixed prefill windows and keeps
        # the conservative max_num_reqs bucket (decode layout only when the
        # token bucket equals max_num_reqs, the historical behavior).
        backbone_out = self._manager._model_executor.compile_backbone(
            num_tokens=num_tokens,
            padded_num_reqs=num_tokens if use_pipeline_runtime else max_num_reqs,
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
            inputs=inputs,
            use_pipeline_runtime=use_pipeline_runtime,
        )
        if backbone_out is not None and not self._manager.model.mesh.is_mpmd:
            self._manager.kv_pages = backbone_out.kv_pages
        elif backbone_out is not None:
            self._manager.kv_pages = backbone_out.kv_pages
        if self._manager.use_aot_forward:
            warm_args = (graphstate, graphother, inputs)
            self._manager._debug_baselines[f"{num_tokens}_hash_in_backbone"] = _tree_hash(warm_args)

    def _reset_kv_pages_after_compile_warmup(self) -> None:
        """Replace compile warm-up cache pages with fresh runtime pages.

        Upfront MPMD compilation executes the same donated backbone functions
        used by decode. That makes compilation truthful, but the warm-up inputs
        are dummy schedules and dummy page tables. Reinitializing here prevents
        deleted donated buffers and dummy KV writes from leaking into the first
        user request while preserving the compiled executables in the executor
        caches.
        """
        if not self._manager.model.mesh.is_mpmd:
            return
        if not hasattr(self._manager, "_cache_quantizer") or not hasattr(self._manager, "_cache_masking_details"):
            return
        self._manager.kv_pages = None
        gc.collect()
        self._manager.kv_pages = self._manager._init_operations_cache_with_retry(
            quantizer=self._manager._cache_quantizer,
            masking_details=self._manager._cache_masking_details,
        )
        self._manager._model_executor.refresh_kv_pages_template(self._manager.kv_pages)

    def _compile_lm_head_variant(
        self,
        *,
        padded_num_reqs: int,
        max_num_reqs: int,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig,
    ) -> None:
        """Compile the lm_head (gather + project) for a request bucket.

        Keyed by ``padded_num_reqs`` only.
        """
        if self._manager._model_executor.has_lm_head(padded_num_reqs):
            return

        # We need graphdef/graphstate/graphother for the lm_head compilation.
        # num_tokens is irrelevant for lm_head — use max_num_reqs as a safe dummy.
        compargs = self._manager.get_compile_configurations(
            self._manager.kv_pages,
            self._manager.rng_key,
            max_num_reqs,  # num_tokens (irrelevant for lm_head, just needs valid value)
            max_num_reqs,
            max_num_reqs,
            metadata,
        )
        graphdef, graphstate, graphother, inputs = compargs
        self._manager._model_executor.compile_lm_head(
            padded_num_reqs=padded_num_reqs,
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
            inputs=inputs,
        )

    def _compile_sampler_variant(
        self,
        *,
        num_tokens: int,
        max_num_reqs: int,
        padded_num_reqs: int,
        metadata: RaggedPagesCacheConfig | UnifiedAttentionCacheConfig,
        inputs: StepFunctionInputs | None = None,
    ) -> None:
        """Compile a sampler variant without requiring a matching model variant."""
        sampler_key = self._manager._sampler_executor.cache_key(padded_num_reqs=padded_num_reqs)
        greedy_sampler_key = self._manager._sampler_executor.cache_key(padded_num_reqs=padded_num_reqs, greedy=True)
        if self._manager._sampler_executor.has(sampler_key) and self._manager._sampler_executor.has(greedy_sampler_key):
            return
        if inputs is None:
            _, _, _, inputs = self._manager.get_compile_configurations(
                self._manager.kv_pages,
                self._manager.rng_key,
                num_tokens,
                max_num_reqs,
                padded_num_reqs,
                metadata,
            )
        if not self._manager._sampler_executor.has(sampler_key):
            self._manager._sampler_executor.compile(
                num_tokens=num_tokens,
                padded_num_reqs=padded_num_reqs,
                inputs=inputs,
                metadata=inputs.batch_metadata,
            )
        if not self._manager._sampler_executor.has(greedy_sampler_key):
            self._manager._sampler_executor.compile(
                num_tokens=num_tokens,
                padded_num_reqs=padded_num_reqs,
                inputs=inputs,
                metadata=inputs.batch_metadata,
                greedy=True,
            )
        if self._manager.use_aot_forward:
            vocab_size = self._manager.model.config.get_text_config().vocab_size
            dummy_logits = jnp.zeros(
                (padded_num_reqs, vocab_size),
                dtype=self._manager.model.dtype,
                out_sharding=self._manager._sampler_sharding,
            )
            sampler_args = (
                jnp.ones((6, padded_num_reqs), dtype=jnp.float32, out_sharding=self._manager._sampler_sharding),
                jnp.zeros((7, padded_num_reqs), dtype=jnp.int32, out_sharding=self._manager._sampler_sharding),
                jnp.array([padded_num_reqs, num_tokens], dtype=jnp.int32, out_sharding=self._manager._sampler_sharding),
                dummy_logits,
                jax.device_put(inputs.rng_key, self._manager._sampler_sharding),
                self._manager._sampler_zero_token_counts,
            )
            self._manager._debug_baselines[f"{num_tokens}_{padded_num_reqs}_hash_in_sampler"] = _tree_hash(sampler_args)
