# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Serializable in-memory repository for development and focused tests."""

from __future__ import annotations

from eyvan.server.persistence.state import StateRepository, StateSession


class InMemorySession(StateSession):
    """Transaction-local in-memory state snapshot."""


class InMemoryRepository(StateRepository):
    """Copy-on-write in-memory repository with serializable transactions and event waits."""

    session_type = InMemorySession


__all__ = ("InMemoryRepository", "InMemorySession")
