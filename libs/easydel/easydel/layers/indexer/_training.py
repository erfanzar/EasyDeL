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

"""Shared policy for selecting sparse indexer training attention.

The numerical operation consumes explicit tensors and masks. Eligibility and
config policy live here, without inspecting a model family or layer internals.
"""

from __future__ import annotations

import typing as tp

import jax
from jax import numpy as jnp

# Preserve the existing auto-mode crossover: measured sparse/dense TPU
# forward+backward timings favor sparse starting at 8k tokens.
AUTO_SPARSE_MIN_SEQ = 8192

__all__ = ("AUTO_SPARSE_MIN_SEQ", "pool_bitmap_tail_error", "resolve_dsa_pool_kernel", "use_sparse_training_attention")


def resolve_dsa_pool_kernel(setting: str | None, kpool: int, *, latent_dim: int, latent_dtype: tp.Any) -> str:
    """The ``dsa_pool_kernel`` that runs for pools of ``kpool`` tokens, with ``None`` (the default) resolved.

    ``None`` picks ``"gather"`` where that kernel runs: on TPU, for an even ``kpool`` (it reads a bf16 latent two keys
    per 32-bit word), a latent width that is a multiple of 128 and a 2- or 4-byte latent. Everywhere else it picks
    ``"bitmap"``. On TPU v5p a 12-layer GLM-5.3-Flash forward and backward on 16 sequences of 131072 tokens with 4-way
    tensor parallelism and pools of 4 tokens took 10.06 s with ``"gather"`` and 21.49 s with ``"bitmap"``; the
    gradient norms differed by 0.002%, less than those of two identical runs.

    Args:
        setting: ``EasyDeLBaseConfig.dsa_pool_kernel``: ``"bitmap"``, ``"gather"`` or ``None``.
        kpool: Tokens per pool.
        latent_dim: Width of the shared key/value latent the attention reads (``kv_lora_rank``).
        latent_dtype: Dtype the attention reads the latent in.

    Returns:
        ``"bitmap"`` or ``"gather"``.
    """
    if setting is not None:
        return setting
    gather_runs = (
        int(kpool) % 2 == 0
        and int(latent_dim) % 128 == 0
        and jnp.dtype(latent_dtype).itemsize in (2, 4)
        and jax.default_backend() == "tpu"
    )
    return "gather" if gather_runs else "bitmap"


def pool_bitmap_tail_error(kpool: int, select_tail: bool) -> str | None:
    """Why the ``dsa_pool_kernel="bitmap"`` kernel cannot take a pooled indexer's tail keys (``None`` when it can).

    A pooled indexer that always selects each query's own incomplete pool adds up to ``kpool - 1`` tail keys per
    query; the bitmap kernel carries at most ``ejkernel.modules.DSA_SPARSE_MAX_TAIL_KEYS`` of them. The ``"gather"``
    kernel has no such limit.

    Args:
        kpool: Tokens per pool.
        select_tail: Whether the incomplete pool is always selected.

    Returns:
        The error message, or ``None``.
    """
    from ejkernel.modules import DSA_SPARSE_MAX_TAIL_KEYS  # pyright: ignore[reportMissingTypeStubs]

    tails = kpool - 1 if select_tail else 0
    if tails <= DSA_SPARSE_MAX_TAIL_KEYS:
        return None
    return (
        f"dsa_pool_kernel='bitmap' takes at most {DSA_SPARSE_MAX_TAIL_KEYS} always-selected tail keys per query, "
        f"but pools of {kpool} tokens with the incomplete pool always selected give {tails}. Use "
        f"dsa_pool_kernel='gather' or a pool size <= {DSA_SPARSE_MAX_TAIL_KEYS + 1}."
    )


def use_sparse_training_attention(
    *,
    topk_indices: tp.Any,
    cache_view: tp.Any,
    mask_info: tp.Any,
    output_attentions: bool,
    sliding_window: tp.Any,
    softmax_aux: tp.Any,
    sparse_training: bool | None = None,
    seq_len: int | None = None,
    backend: str | None = None,
) -> bool:
    """Choose sparse training only when it preserves the dense-path features.

    Explicit ``True`` or ``False`` overrides the automatic device/length
    policy, but cannot enable an unsupported cache, sink, window or attention
    output. Automatic mode enables sparse attention on TPU at 8192 tokens
    and above. The caller supplies the public softmax-aux value rather than
    exposing private layer implementation details to the numerical operation.

    Args:
        topk_indices: Selected token indices; ``None`` means no sparse selection.
        cache_view: Existing inference cache, or ``None`` for training.
        mask_info: Padding/segment mask metadata required by the sparse operation.
        output_attentions: Whether the caller needs dense attention weights.
        sliding_window: Optional window unsupported by the sparse training path.
        softmax_aux: Public attention-sink/softmax auxiliary value, or ``None``.
        sparse_training: Configured force-on/force-off flag; ``None`` selects auto.
        seq_len: Query sequence length, required for automatic mode.
        backend: Optional already-resolved JAX backend; defaults to the active
            backend only when automatic mode needs to inspect it.

    Returns:
        Whether sparse training attention supports all requested features and
        is selected by the configured or automatic policy.
    """
    eligible = (
        topk_indices is not None
        and cache_view is None
        and mask_info is not None
        and not output_attentions
        and sliding_window is None
        and softmax_aux is None
    )
    if not eligible:
        return False
    if sparse_training is not None:
        return bool(sparse_training)
    if seq_len is None or int(seq_len) < AUTO_SPARSE_MIN_SEQ:
        return False
    return (jax.default_backend() if backend is None else backend) == "tpu"
