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

"""Startup check of the recurrent slot contract that eSurge serving layers declare.

A shared recurrent serving layer class (for example the KDA or GDN linear-attention layer) declares in the class
attribute ``esurge_recurrent_slot_contract`` how far it follows the eSurge recurrent step contract:

* ``"slots"``: on every step row ``r`` reads and writes pool slot ``cache_metadata.recurrent_state_indices[r]``, so a
  request keeps its state in one stable slot across SequenceBuffer row moves, several row windows per step and sparse
  row packing.
* ``"verify"``: ``"slots"`` plus the speculative verify contract: every request owns a contiguous group of
  ``W = 1 + K_max`` slots, a row reads slot ``g + recurrent_read_offset[r]`` and writes the state after each of its
  uncertain tokens into ``g + j`` (``recurrent_commit_lens``).

The runner reads the declarations once at startup (:func:`resolve_recurrent_slot_contract`); shared code reads only the
declaration, never a model name or class. A model whose recurrent layers do not all declare a level keeps the identity
layout: slot = row, one row window per step and no sparse row packing (:func:`check_identity_layout_row_windows`).
A declaring model gets pooled slots exactly where the row layout needs them (:func:`uses_pooled_recurrent_slots`).
"""

from __future__ import annotations

import typing

import spectrax as spx

RECURRENT_SLOT_CONTRACT_ATTR = "esurge_recurrent_slot_contract"
"""Class attribute through which a recurrent serving layer declares its contract level."""

SLOT_CONTRACT_LEVELS: tuple[str, ...] = ("none", "slots", "verify")
"""Contract levels, weakest first."""


def declared_slot_contract(module: typing.Any) -> str | None:
    """Return the contract level declared by ``module``'s class.

    Args:
        module: Any object; only its class attribute :data:`RECURRENT_SLOT_CONTRACT_ATTR` is read.

    Returns:
        The declared level, or ``None`` when the class declares none.

    Raises:
        ValueError: If the declared level is not one of :data:`SLOT_CONTRACT_LEVELS`.
    """
    level = getattr(type(module), RECURRENT_SLOT_CONTRACT_ATTR, None)
    if level is None:
        return None
    if level not in SLOT_CONTRACT_LEVELS:
        raise ValueError(
            f"{type(module).__name__}.{RECURRENT_SLOT_CONTRACT_ATTR}={level!r} is not one of {SLOT_CONTRACT_LEVELS}."
        )
    return str(level)


def lowest_slot_contract(declared: typing.Iterable[str], *, num_recurrent_layers: int) -> str:
    """Combine per-module declarations into the model's contract level.

    Args:
        declared: The levels declared by the model's modules, one entry per declaring module.
        num_recurrent_layers: Recurrent layers of the model (from its layer types).

    Returns:
        ``"none"`` when the model has no recurrent layer or fewer declaring modules than recurrent layers, otherwise
        the weakest declared level.
    """
    levels = list(declared)
    if int(num_recurrent_layers) <= 0 or len(levels) < int(num_recurrent_layers):
        return "none"
    return min(levels, key=SLOT_CONTRACT_LEVELS.index)


def resolve_recurrent_slot_contract(model: typing.Any, *, num_recurrent_layers: int) -> str:
    """Walk ``model``'s module tree once and return its recurrent slot contract level.

    Args:
        model: The served model (a :class:`spectrax.Module` tree).
        num_recurrent_layers: Recurrent layers of the model (from its layer types).

    Returns:
        One of :data:`SLOT_CONTRACT_LEVELS`, combined by :func:`lowest_slot_contract`.
    """
    if int(num_recurrent_layers) <= 0 or not isinstance(model, spx.Module):
        return "none"
    declared: list[str] = []
    for module in spx.iter_modules(model, with_path=False):
        level = declared_slot_contract(module)
        if level is not None:
            declared.append(level)
    return lowest_slot_contract(declared, num_recurrent_layers=num_recurrent_layers)


def check_identity_layout_row_windows(
    *,
    family: str,
    rows_per_forward: int,
    max_num_reqs: int,
    speculative_candidate_rows: bool,
) -> None:
    """Refuse several row windows per step for recurrent pools kept in the identity layout (slot = row).

    In the identity layout window row ``r`` owns pool slot ``r``. A step that runs as several row windows (more
    resident requests than one forward takes) would let every later window read and overwrite the first window's
    states.

    Args:
        family: Model family name, used in the error message only.
        rows_per_forward: Request rows one forward takes, ``min(metadata.get_max_num_seqs(), max_num_reqs)``.
        max_num_reqs: Request rows the runner keeps resident.
        speculative_candidate_rows: Whether the pools hold speculative candidate rows (laid out by the row count).

    Raises:
        ValueError: When ``rows_per_forward < max_num_reqs``.
    """
    if int(rows_per_forward) >= int(max_num_reqs):
        return
    reason = (
        "speculative decoding lays the recurrent candidate rows out by the window's row count"
        if speculative_candidate_rows
        else f"its recurrent layers do not declare {RECURRENT_SLOT_CONTRACT_ATTR}, so they address pool slots by "
        "window-local row instead of cache_metadata.recurrent_state_indices"
    )
    raise ValueError(
        f"max_num_seqs={max_num_reqs} exceeds rows per forward={rows_per_forward} for {family}: eSurge would run "
        f"each step as several row windows, and {reason}, so later windows would read and overwrite the first "
        f"window's recurrent states. Lower max_num_seqs to <= {rows_per_forward}, or raise the rows per forward "
        "(metadata.get_max_num_seqs() = 131072 // ceil(max_model_len / page_size) unless the window-aware "
        "runtime cap is on) with a larger page_size or a smaller max_model_len."
    )


def uses_pooled_recurrent_slots(
    *,
    family: str,
    contract: str,
    has_recurrent_layers: bool,
    slot_indexed_cache: bool,
    rank_major_dp: bool,
    speculative_candidate_rows: bool,
    rows_per_forward: int,
    max_num_reqs: int,
    sparse_row_packing: bool,
    group_size: int,
) -> bool:
    """Decide once, at runner init, whether requests own pooled recurrent-state slots shipped with every step.

    Pooled slots (:class:`~easydel.inference.esurge.runners.slot_pool.RecurrentSlotPool`) give every request a stable
    group of ``group_size`` slots, shipped as ``cache_metadata.recurrent_state_indices``. They are required where a row
    can sit past its window-local index (several row windows per step, sparse packing of zero-token rows) and where a
    request owns more than one slot. Elsewhere the identity layout (window row ``r`` owns slot ``r``) is kept: explicit
    slots make the recurrent drivers gather and scatter the pool rows instead of updating them in place.

    Args:
        family: Model family name, used in error messages only.
        contract: The model's slot contract level (:func:`resolve_recurrent_slot_contract`).
        has_recurrent_layers: Whether the model keeps per-request recurrent state.
        slot_indexed_cache: Whether the cache keeps all decode state in per-request slots (compressed-window caches);
            such caches are always pooled.
        rank_major_dp: Whether rank-major SPMD DP runs the steps; its per-rank slot blocks are handled separately.
        speculative_candidate_rows: Whether the pools hold speculative candidate rows (laid out by the row count).
        rows_per_forward: Request rows one forward takes, ``min(metadata.get_max_num_seqs(), max_num_reqs)``.
        max_num_reqs: Request rows the runner keeps resident.
        sparse_row_packing: Whether zero-token rows may be packed out of a window (DP 1 without async scheduling).
        group_size: Slots per request, ``W``.

    Returns:
        ``True`` for pooled slots, ``False`` for the identity layout (and for rank-major DP).

    Raises:
        ValueError: When ``group_size > 1`` without the ``"verify"`` contract, or when the identity layout would need
            several row windows per step (:func:`check_identity_layout_row_windows`).
    """
    if slot_indexed_cache:
        return True
    if not has_recurrent_layers or rank_major_dp:
        return False
    if int(group_size) > 1 and contract != "verify":
        raise ValueError(
            f"{family}: {group_size} recurrent slots per request need recurrent layers that declare "
            f'{RECURRENT_SLOT_CONTRACT_ATTR} = "verify", got {contract!r}.'
        )
    rows_leave_window_index = int(rows_per_forward) < int(max_num_reqs) or bool(sparse_row_packing)
    if contract != "none" and not speculative_candidate_rows and (rows_leave_window_index or int(group_size) > 1):
        return True
    check_identity_layout_row_windows(
        family=family,
        rows_per_forward=rows_per_forward,
        max_num_reqs=max_num_reqs,
        speculative_candidate_rows=speculative_candidate_rows,
    )
    return False
