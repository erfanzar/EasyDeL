# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persistence contracts and repository implementations."""

from eyvan.capacity.fleet_state import FleetCellRecord, FleetStateStore, GcsBackend, LocalBackend
from eyvan.server.persistence.base import Repository, RepositorySession
from eyvan.server.persistence.leader import PostgresAdvisoryLeader
from eyvan.server.persistence.local import (
    LocalControlPlaneError,
    LocalControlPlaneInUseError,
    LocalControlPlaneSchemaError,
    LocalJsonRepository,
    LocalJsonSession,
)
from eyvan.server.persistence.memory import InMemoryRepository, InMemorySession
from eyvan.server.persistence.sqlalchemy import (
    SCHEMA_REVISION,
    Base,
    SqlAlchemyRepository,
    SqlAlchemySession,
    create_schema,
)

__all__ = (
    "SCHEMA_REVISION",
    "Base",
    "FleetCellRecord",
    "FleetStateStore",
    "GcsBackend",
    "InMemoryRepository",
    "InMemorySession",
    "LocalBackend",
    "LocalControlPlaneError",
    "LocalControlPlaneInUseError",
    "LocalControlPlaneSchemaError",
    "LocalJsonRepository",
    "LocalJsonSession",
    "PostgresAdvisoryLeader",
    "Repository",
    "RepositorySession",
    "SqlAlchemyRepository",
    "SqlAlchemySession",
    "create_schema",
)
