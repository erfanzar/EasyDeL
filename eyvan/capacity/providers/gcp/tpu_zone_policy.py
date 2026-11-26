# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Documented deny-only TPU location policy for queued-resource autoscale."""

from __future__ import annotations

from dataclasses import dataclass

from eyvan.capacity.providers.gcp.tpu_types import accelerator_generation


@dataclass(frozen=True, slots=True)
class DocumentedTpuZoneExclusion:
    """One documented TPU generation and zone excluded from queue admission."""

    generation: str
    zone: str
    source_url: str


# Source: https://cloud.google.com/tpu/docs/regions-zones
# The TPU locations table lists TPU7x for
# us-central1-c and v6e for us-central1-b, but does not list v6e in
# us-central1-c. Do not add positive availability assertions here; dynamic
# provider discovery remains the primary source for eligible types.
DOCUMENTED_TPU_QUEUE_ADMISSION_EXCLUSIONS = (
    DocumentedTpuZoneExclusion(
        generation="v6e",
        zone="us-central1-c",
        source_url="https://cloud.google.com/tpu/docs/regions-zones",
    ),
)


def documented_tpu_zone_exclusion(*, zone: str, accelerator_type: str) -> DocumentedTpuZoneExclusion | None:
    """Return a documented deny-only exclusion for one TPU type, if any."""

    generation = accelerator_generation(accelerator_type)
    if generation is None:
        return None
    for exclusion in DOCUMENTED_TPU_QUEUE_ADMISSION_EXCLUSIONS:
        if exclusion.zone == zone and exclusion.generation == generation:
            return exclusion
    return None


__all__ = (
    "DOCUMENTED_TPU_QUEUE_ADMISSION_EXCLUSIONS",
    "DocumentedTpuZoneExclusion",
    "documented_tpu_zone_exclusion",
)
