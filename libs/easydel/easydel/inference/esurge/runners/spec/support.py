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
# See the License for the specific language governing permissions and
# limitations under the License.


"""Shared speculative-decoding helpers for the eSurge runner.

The runner-native drafter path implements draft/verify/commit inline in
:class:`~easydel.inference.esurge.runners.model_runner.eSurgeRunner`. The
helpers here are model-agnostic:

- :class:`SpecDecodeStats` — per-generation acceptance/throughput statistics.
- :func:`default_assistant_layer_mapping` — heuristic assistant-layer to
  target-layer mapping for cross-model drafters (e.g. Gemma4 Assistant).
- :func:`build_target_kv_pairs` — gather per-assistant-layer ``(K, V)``
  tensors out of a target model's (possibly paged/hybrid) KV cache.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array


@dataclass
class SpecDecodeStats:
    """Per-generation statistics for speculative decoding.

    Attributes:
        num_target_forwards: Number of target-model forward passes
            (1 prefill + 1 per spec-decode step). Without spec-decode
            a baseline does 1 forward per emitted token.
        num_draft_steps: Number of spec-decode steps taken.
        num_drafts_generated: Total draft tokens proposed.
        num_drafts_accepted: Total draft tokens accepted by the
            target's verification.
        wallclock_s: Wallclock seconds for the generation.
        tokens_generated: Total accepted + corrected tokens emitted.
    """

    num_target_forwards: int = 0
    num_draft_steps: int = 0
    num_drafts_generated: int = 0
    num_drafts_accepted: int = 0
    wallclock_s: float = 0.0
    tokens_generated: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed drafts the target accepted."""
        if self.num_drafts_generated == 0:
            return 0.0
        return self.num_drafts_accepted / self.num_drafts_generated

    @property
    def tokens_per_second(self) -> float:
        """End-to-end decoding throughput."""
        if self.wallclock_s == 0:
            return 0.0
        return self.tokens_generated / self.wallclock_s

    @property
    def speedup_vs_baseline(self) -> float:
        """Algorithmic speedup vs a 1-token-per-forward baseline.

        Equal to ``tokens_generated / num_target_forwards``. A
        no-spec-decode baseline emits exactly 1 token per target
        forward, so this ratio is the hardware-independent speedup
        from parallel verification. ``1.0`` means no benefit.
        """
        if self.num_target_forwards == 0:
            return 0.0
        return self.tokens_generated / self.num_target_forwards

    @property
    def mean_accepted_per_step(self) -> float:
        """Average accepted drafts per spec-decode step."""
        if self.num_draft_steps == 0:
            return 0.0
        return self.num_drafts_accepted / self.num_draft_steps


def default_assistant_layer_mapping(
    num_assistant_layers: int,
    num_target_layers: int,
) -> list[int]:
    """Heuristic assistant-layer → target-layer mapping.

    The Gemma4 Assistant's Q-only attention reads K/V from the target
    model. Each assistant layer must be told which target layer's K/V
    to attend to. Google's training defines the canonical mapping; it
    is **not** published in the open ``Gemma4Assistant`` config, so
    this heuristic is used until a reference is available:

    Map the ``i``-th assistant layer to the ``i``-th target layer
    counting from the END of the target stack — i.e. the assistant's
    final layer attends to the target's final layer, and earlier
    assistant layers attend to progressively earlier target layers.
    Rationale: the assistant is a shallow drafter whose job is to
    mimic the target's *late* representations, which carry the
    next-token signal.

    Args:
        num_assistant_layers: Drafter layer count (4 for published
            Gemma4 assistants).
        num_target_layers: Target model decoder layer count.

    Returns:
        A list of length ``num_assistant_layers`` mapping each
        assistant layer index to a target layer index.
    """
    if num_target_layers < num_assistant_layers:
        return [num_target_layers - 1] * num_assistant_layers
    base = num_target_layers - num_assistant_layers
    return [base + i for i in range(num_assistant_layers)]


def build_target_kv_pairs(
    target_cache: typing.Any,
    layer_mapping: list[int],
    *,
    page_tables: typing.Sequence[typing.Any] | None = None,
    layer_to_group: dict[int, int] | None = None,
    batch_index: int = 0,
    kv_len: int | None = None,
) -> list[tuple[Array, Array] | None]:
    """Extract per-assistant-layer ``(K, V)`` from a target KV cache.

    The Gemma4 Assistant consumes a ``target_key_value_pairs`` list —
    one ``(K, V)`` tuple per assistant layer — sourced from the target
    model's KV cache. This controller reads the target cache's
    per-layer views and gathers the K/V tensors for the target layers
    named in ``layer_mapping``.

    Args:
        target_cache: The target model's KV cache. Must expose a
            ``views`` sequence whose entries have either ``.key`` /
            ``.value`` attributes (``TransformerCache`` /
            ``HybridCache`` shape) or paged-cache ``.key_pages`` /
            ``.value_pages`` attributes.
        layer_mapping: ``layer_mapping[i]`` is the target layer index
            whose K/V feeds assistant layer ``i``.
        page_tables: Optional eSurge page tables, one per cache group,
            used to gather K/V from ragged paged-cache views.
        layer_to_group: Optional target-layer -> page-table-group map.
            If omitted and paged K/V is encountered, group ``0`` is
            used only when a single page table is supplied.
        batch_index: Request row to gather from non-paged or paged
            caches.
        kv_len: Number of tokens to gather. Defaults to the full
            dense-cache length or one page table row's page coverage.

    Returns:
        A list of ``(K, V)`` tuples (or ``None`` for any target layer
        that has no K/V — e.g. a linear-attention layer in a hybrid
        target). The list length equals ``len(layer_mapping)``.

    Raises:
        ValueError: If ``target_cache`` has no ``views`` attribute.
    """
    views = getattr(target_cache, "views", None)
    if views is None:
        raise ValueError(
            "target_cache must expose a 'views' sequence of per-layer "
            "cache views with .key/.value (got "
            f"{type(target_cache).__name__})"
        )
    pairs: list[tuple[Array, Array] | None] = []
    for tgt_idx in layer_mapping:
        if tgt_idx < 0 or tgt_idx >= len(views):
            pairs.append(None)
            continue
        view = views[tgt_idx]
        view = getattr(view, "transformer", view)
        k = getattr(view, "key", None)
        v = getattr(view, "value", None)
        if k is None or v is None:
            key_pages = getattr(view, "key_pages", None)
            value_pages = getattr(view, "value_pages", None)
            if key_pages is None or value_pages is None or page_tables is None:
                pairs.append(None)
                continue

            group_idx = None
            if layer_to_group is not None:
                group_idx = layer_to_group.get(int(tgt_idx))
            elif len(page_tables) == 1:
                group_idx = 0
            if group_idx is None or group_idx < 0 or group_idx >= len(page_tables):
                pairs.append(None)
                continue

            page_table = page_tables[group_idx]
            page_table_cpu = page_table.get_cpu_tensor() if hasattr(page_table, "get_cpu_tensor") else page_table
            page_size = int(getattr(getattr(view, "metadata", None), "page_size", 1) or 1)
            page_size = max(1, page_size)
            if kv_len is None:
                gather_len = int(page_table_cpu.shape[1]) * page_size
            else:
                gather_len = int(kv_len)
            pages_needed = min(int(page_table_cpu.shape[1]), max(1, (gather_len + page_size - 1) // page_size))
            gather_len = min(gather_len, pages_needed * page_size)
            page_ids = jnp.asarray(page_table_cpu[int(batch_index), :pages_needed], dtype=jnp.int32)
            token_offsets = jnp.arange(gather_len, dtype=jnp.int32)
            page_indices = page_ids[token_offsets // page_size]
            offsets = token_offsets % page_size
            pairs.append((key_pages[page_indices, offsets][None, ...], value_pages[page_indices, offsets][None, ...]))
        else:
            if kv_len is None:
                pairs.append((k[int(batch_index) : int(batch_index) + 1], v[int(batch_index) : int(batch_index) + 1]))
            else:
                pairs.append(
                    (
                        k[int(batch_index) : int(batch_index) + 1, : int(kv_len)],
                        v[int(batch_index) : int(batch_index) + 1, : int(kv_len)],
                    )
                )
    return pairs


__all__ = (
    "SpecDecodeStats",
    "build_target_kv_pairs",
    "default_assistant_layer_mapping",
)
