"""Import-safe page-pool sizing shared by homogeneous and mixed caches."""

from __future__ import annotations

import typing as tp
from collections.abc import Mapping
from dataclasses import dataclass


def cap_paged_cache_pages(
    num_pages: int,
    *,
    page_size: int,
    data_parallel_size: int = 1,
    max_cache_tokens: int | None = None,
) -> int:
    """Align a page pool to DP without exceeding an optional token ceiling.

    Args:
        num_pages: Available page count before alignment.
        page_size: Tokens in each page.
        data_parallel_size: Partitions of the page axis.
        max_cache_tokens: Optional positive ceiling on total tokens.

    Returns:
        Positive, DP-aligned page count within the ceiling.

    Raises:
        ValueError: If dimensions are invalid or no aligned page fits.
    """
    if page_size <= 0 or data_parallel_size <= 0:
        raise ValueError("page_size and data_parallel_size must be positive")
    pages = int(num_pages)
    if max_cache_tokens is not None and int(max_cache_tokens) > 0:
        pages = min(pages, int(max_cache_tokens) // int(page_size))
    pages = (pages // int(data_parallel_size)) * int(data_parallel_size)
    if pages <= 0:
        raise ValueError("No DP-aligned cache page fits the HBM budget and max_cache_tokens ceiling")
    return pages


@dataclass(frozen=True)
class PagedCacheBudget:
    """Shared pool sizing result, including the busiest physical stage charge."""

    num_pages: int
    page_bytes: int
    num_hidden_layers: int


def compute_paged_cache_budget(
    layer_page_bytes: Mapping[int, int],
    *,
    available_bytes: int | None,
    page_size: int,
    data_parallel_size: int = 1,
    stage_layer_groups: tp.Iterable[tp.Iterable[int]] | None = None,
    num_pages_override: int | None = None,
    max_cache_tokens: int | None = None,
) -> PagedCacheBudget:
    """Size a shared pool using the maximum physical-stage page footprint.

    Args:
        layer_page_bytes: Bytes occupied by one page in each layer, after
            resolving its actual storage layout and dtype.
        available_bytes: Effective HBM budget, required without an override.
        page_size: Tokens per page.
        data_parallel_size: Number of page-axis partitions.
        stage_layer_groups: Optional physical-stage layer ownership groups.
            Each iterable is consumed once, including generator groups.
        num_pages_override: Optional explicitly requested page count.
        max_cache_tokens: Optional positive total-token ceiling.

    Returns:
        Page count, maximum stage page footprint and busiest stage layer count.

    Raises:
        ValueError: If geometry is empty/invalid or no aligned pool fits.
    """
    if not layer_page_bytes or any(int(size) <= 0 for size in layer_page_bytes.values()):
        raise ValueError("Per-layer page footprints must be nonempty and positive")
    groups = (
        [list(layer_page_bytes)]
        if stage_layer_groups is None
        else [[int(idx) for idx in group if int(idx) in layer_page_bytes] for group in stage_layer_groups]
    )
    groups = [group for group in groups if group] or [list(layer_page_bytes)]
    page_bytes = max(sum(int(layer_page_bytes[idx]) for idx in group) for group in groups)
    if num_pages_override is None:
        if available_bytes is None:
            raise ValueError("available_bytes is required when num_pages_override is absent")
        num_pages_override = int(available_bytes) // page_bytes
    return PagedCacheBudget(
        num_pages=cap_paged_cache_pages(
            num_pages_override,
            page_size=page_size,
            data_parallel_size=data_parallel_size,
            max_cache_tokens=max_cache_tokens,
        ),
        page_bytes=page_bytes,
        num_hidden_layers=max(len(group) for group in groups),
    )
