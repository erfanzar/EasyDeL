# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Internal compatibility utilities used across Spectrax."""

from .dtype import DTYPE_MAPPING, DTYPE_TO_STRING_MAP, STRING_TO_DTYPE_MAP, put_dtype
from .logging import LazyLogger, ProgressLogger, get_logger

__all__ = (
    "DTYPE_MAPPING",
    "DTYPE_TO_STRING_MAP",
    "STRING_TO_DTYPE_MAP",
    "LazyLogger",
    "ProgressLogger",
    "get_logger",
    "put_dtype",
)
