# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""An idempotency replay of a capacity-pool upsert must not skip authorization.

The replay branch used to return the stored pool before the in-transaction
ownership gate ran, so a caller whose grant had been revoked -- or whose pool
had changed hands -- could still read the current configuration by replaying
the original request.
"""

from __future__ import annotations

import pytest
from test_service import _capacity_pool

from eyvan.records import Principal, Role, RoleGrant
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import AuthorizationError


@pytest.mark.asyncio
async def test_a_replayed_pool_upsert_reauthorizes_the_caller() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    fleet_admin = Principal(subject="system", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    pool = _capacity_pool()
    await service.upsert_capacity_pool(fleet_admin, pool, idempotency_key="pool-1")

    outsider = Principal(subject="mallory", grants=())
    with pytest.raises(AuthorizationError):
        await service.upsert_capacity_pool(outsider, pool, idempotency_key="pool-1")
