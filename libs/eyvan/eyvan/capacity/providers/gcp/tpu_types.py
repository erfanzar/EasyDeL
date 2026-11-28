# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Neutral Cloud TPU accelerator-type parsing shared by GCP adapters.

Re-exported from :mod:`eyvan.domain.accelerators` so pure domain modules can
parse TPU shape names without importing provider packages.
"""

from __future__ import annotations

from eyvan.domain.accelerators import (
    TpuAcceleratorType,
    accelerator_generation,
    parse_tpu_accelerator_type,
)

__all__ = ("TpuAcceleratorType", "accelerator_generation", "parse_tpu_accelerator_type")
