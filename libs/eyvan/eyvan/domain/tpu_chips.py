# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Which chips of a TPU slice a chip-sized workload may hold, and which ones it gets.

Pure and provider-neutral: callers pass the slice's chip count and its chips per
host (the hardware catalog knows both). Chip indexes are global within the
slice; host ``index // chips_per_host`` owns chip ``index``.

One allocation lives on one host. What one process may hold was measured on a
real v5p host (4 chips) on 2026-11-02, not assumed: any single chip, the whole
host, or a *neighbouring* pair. The four chips form a ring 0-1-2-3-0; the
diagonal pairs {0,2} and {1,3} make libtpu abort with SLICE_FAILURE_INIT_ERROR
whatever bounds it is given. Hosts of any other width offer single chips only,
because nothing here has measured which blocks they accept.
"""

from __future__ import annotations

import typing as tp

#: Chip pairs one process can hold on a 4-chip host, best-first for picking:
#: the two pairs that leave the other pair whole come first.
RING_PAIRS_4: tuple[tuple[int, int], ...] = ((0, 1), (2, 3), (1, 2), (0, 3))


def allowed_chip_counts(chips_per_host: int) -> tuple[int, ...]:
    """Return the chip counts one allocation on a host of ``chips_per_host`` may have."""

    if chips_per_host == 4:
        return (1, 2, 4)
    return (1,) if chips_per_host >= 1 else ()


def _blocks(count: int, chips_per_host: int) -> tuple[tuple[int, ...], ...]:
    """Return every host-local chip set of size ``count`` one process can hold, best-first."""

    if count == 1:
        return tuple((chip,) for chip in range(chips_per_host))
    if count == chips_per_host and count in allowed_chip_counts(chips_per_host):
        return (tuple(range(chips_per_host)),)
    if count == 2 and chips_per_host == 4:
        return RING_PAIRS_4
    return ()


def chip_request_error(
    count: int,
    indexes: tp.Sequence[int] | None,
    *,
    total_chips: int,
    chips_per_host: int,
) -> str | None:
    """Return why a chip request cannot be satisfied on this slice shape, or ``None``."""

    if chips_per_host < 1 or total_chips < chips_per_host or total_chips % chips_per_host:
        return f"the slice shape ({total_chips} chips, {chips_per_host} per host) is not divisible into hosts"
    allowed = allowed_chip_counts(chips_per_host)
    if count not in allowed:
        return f"chips must be one of {', '.join(map(str, allowed))} on a host with {chips_per_host} chips"
    if indexes is None:
        return None
    if len(indexes) != count:
        return "chips must equal the number of chips_indexes"
    out_of_range = [index for index in indexes if index >= total_chips]
    if out_of_range:
        return f"chip index {out_of_range[0]} is outside this slice's chips 0..{total_chips - 1}"
    hosts = {index // chips_per_host for index in indexes}
    if len(hosts) != 1:
        return "chips_indexes must all be on one host (host = index // chips_per_host)"
    host = hosts.pop()
    local = tuple(sorted(index - host * chips_per_host for index in indexes))
    if local not in _blocks(count, chips_per_host):
        pairs = ", ".join(f"{{{a},{b}}}" for a, b in RING_PAIRS_4)
        return (
            f"chips {sorted(indexes)} are not neighbours on their host; a pair must be one of {pairs} "
            "(host-local), because libtpu refuses diagonal pairs"
        )
    return None


def pick_chips(
    free: tp.AbstractSet[int],
    count: int,
    indexes: tp.Sequence[int] | None,
    *,
    total_chips: int,
    chips_per_host: int,
) -> tuple[int, ...] | None:
    """Return the chips to grant, or ``None`` when the request must wait.

    Pinned requests get exactly their chips or nothing. Otherwise the lowest
    host with a usable block is used, trying blocks best-first so a pair lands
    on {0,1} or {2,3} and keeps the other pair whole.
    """

    if chip_request_error(count, indexes, total_chips=total_chips, chips_per_host=chips_per_host) is not None:
        return None
    if indexes is not None:
        wanted = tuple(sorted(indexes))
        return wanted if all(index in free for index in wanted) else None
    blocks = _blocks(count, chips_per_host)
    for host in range(total_chips // chips_per_host):
        base = host * chips_per_host
        for block in blocks:
            chips = tuple(base + chip for chip in block)
            if all(chip in free for chip in chips):
                return chips
    return None


def host_local_chips(chips: tp.Sequence[int], chips_per_host: int) -> tuple[int, tuple[int, ...]]:
    """Return ``(host_index, local_chip_indexes)`` for chips that sit on one host."""

    hosts = {chip // chips_per_host for chip in chips}
    if len(hosts) != 1:
        raise ValueError("a chip allocation must sit on exactly one host")
    host = hosts.pop()
    return host, tuple(sorted(chip - host * chips_per_host for chip in chips))


__all__ = ("RING_PAIRS_4", "allowed_chip_counts", "chip_request_error", "host_local_chips", "pick_chips")
