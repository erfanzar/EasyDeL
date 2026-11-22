"""MPMD scheduler fixtures shared by per-model module tests."""

from __future__ import annotations

from dataclasses import dataclass

import spectrax as spx


@dataclass(frozen=True)
class MPMDScheduleCase:
    name: str
    schedule: spx.Schedule
    virtual_stages: int = 1
    num_hidden_layers: int = 4


LOSS_SCHEDULE_CASES: tuple[MPMDScheduleCase, ...] = (
    MPMDScheduleCase(
        name="flat",
        schedule=spx.GPipe(microbatches=2),
        virtual_stages=1,
        num_hidden_layers=4,
    ),
    MPMDScheduleCase(
        name="virtual",
        schedule=spx.KimiK2(microbatches=2, virtual_stages=2, stage_layout="loop"),
        virtual_stages=2,
        num_hidden_layers=4,
    ),
    MPMDScheduleCase(
        name="bidirectional",
        schedule=spx.DualPipeV(microbatches=4),
        virtual_stages=2,
        num_hidden_layers=4,
    ),
)

GENERATION_SCHEDULE_CASE = MPMDScheduleCase(
    name="dualpipe_generation",
    schedule=spx.DualPipeV(microbatches=4),
    virtual_stages=2,
    num_hidden_layers=4,
)

LOSS_SCHEDULE_KINDS = tuple(case.name for case in LOSS_SCHEDULE_CASES)
GENERATION_SCHEDULE_KIND = GENERATION_SCHEDULE_CASE.name

_SCHEDULE_BY_NAME = {case.name: case for case in (*LOSS_SCHEDULE_CASES, GENERATION_SCHEDULE_CASE)}


def get_mpmd_schedule_case(name: str | None) -> MPMDScheduleCase | None:
    if name is None:
        return None
    return _SCHEDULE_BY_NAME[name]


def scaled_mrope_section(
    head_dim: int,
    partial_rotary_factor: float = 0.25,
    base_section: tuple[int, int, int] = (24, 20, 20),
) -> list[int]:
    """Scale a production mRoPE section to the tiny test rotary dimension."""
    expected = int(head_dim * partial_rotary_factor) // 2
    if expected < len(base_section):
        raise ValueError(
            f"mRoPE section target {expected} is too small for {len(base_section)} axes; "
            f"increase head_dim={head_dim} or partial_rotary_factor={partial_rotary_factor}."
        )

    total = sum(base_section)
    numerators = [axis * expected for axis in base_section]
    section = [max(1, numerator // total) for numerator in numerators]

    remainder_order = sorted(
        range(len(base_section)),
        key=lambda idx: (numerators[idx] % total, base_section[idx]),
        reverse=True,
    )
    while sum(section) < expected:
        for idx in remainder_order:
            if sum(section) >= expected:
                break
            section[idx] += 1

    while sum(section) > expected:
        idx = max(
            (idx for idx, value in enumerate(section) if value > 1),
            key=lambda idx: (section[idx], -base_section[idx]),
        )
        section[idx] -= 1

    return section
