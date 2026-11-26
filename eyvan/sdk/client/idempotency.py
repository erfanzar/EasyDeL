# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Idempotency-key generation shared by Eyvan clients."""

from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IdempotencyKeyFactory:
    """Generate stable retry keys while preserving caller-supplied values."""

    prefix: str = "sdk"

    def create(self, operation: str | None = None, supplied: str | None = None) -> str:
        """Return ``supplied`` or create a fresh key scoped by ``operation``."""

        if supplied is not None:
            if not supplied:
                raise ValueError("idempotency_key must be a non-empty string")
            return supplied
        operation_part = f"-{operation}" if operation else ""
        return f"{self.prefix}{operation_part}-{uuid.uuid4().hex}"
