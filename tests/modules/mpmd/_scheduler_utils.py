"""MPMD scheduler fixtures shared by per-model module tests."""

from __future__ import annotations

from dataclasses import dataclass

import spectrax as spx


@dataclass(frozen=True)
class MPMDScheduleCase:
    name: str
    schedule: spx.Schedule
    virtual_stages: int = 1
    num_hidden_layers: int = 2


LOSS_SCHEDULE_CASES: tuple[MPMDScheduleCase, ...] = (
    MPMDScheduleCase(
        name="flat",
        schedule=spx.GPipe(microbatches=2),
        virtual_stages=1,
        num_hidden_layers=2,
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

_SCHEDULE_BY_NAME = {
    case.name: case for case in (*LOSS_SCHEDULE_CASES, GENERATION_SCHEDULE_CASE)
}


def get_mpmd_schedule_case(name: str | None) -> MPMDScheduleCase | None:
    if name is None:
        return None
    return _SCHEDULE_BY_NAME[name]
