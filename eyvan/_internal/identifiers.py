# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deterministic internal identifiers shared across controller components."""

from __future__ import annotations

import hashlib


def stable_id(prefix: str, *parts: str | int) -> str:
    """Return a deterministic identifier from a typed sequence of parts."""

    payload = "\0".join(str(part) for part in parts).encode()
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:32]}"
