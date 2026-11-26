# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Leader-only scheduling, outbox delivery, and cell observation loops.

``EyvanController`` is composed from one mixin per concern, each in its own
module. The split is by responsibility, not by line count: every method
keeps its original body and no name appears in two mixins, so the MRO is
unambiguous and resolution is identical to the single-class form.
"""

from eyvan.server.controller._support import (
    CapacityApplySuperseded,
    LeaderElector,
    LeaderPhase,
    PreemptionCheckpointError,
    SingleProcessLeader,
    StaleOutboxCommand,
    logger,
)
from eyvan.server.controller.capacity_commands import _ControllerCapacityCommands
from eyvan.server.controller.expansion import _ControllerExpansion
from eyvan.server.controller.fleets import _ControllerFleets
from eyvan.server.controller.lifecycle import _ControllerLifecycle
from eyvan.server.controller.log_archival import _ControllerLogArchival
from eyvan.server.controller.loops import _ControllerLoops
from eyvan.server.controller.observation import _ControllerObservation
from eyvan.server.controller.outbox import _ControllerOutbox
from eyvan.server.controller.preemption import _ControllerPreemption
from eyvan.server.controller.reclaim import _ControllerReclaim
from eyvan.server.controller.scheduling import _ControllerScheduling
from eyvan.server.controller.services import HttpServiceHealthProbe, _ControllerServices
from eyvan.server.controller.settlement import _ControllerSettlement
from eyvan.server.controller.staging import _ControllerStaging
from eyvan.server.controller.webhook_delivery import _ControllerWebhookDelivery
from eyvan.server.controller.workload_commands import _ControllerWorkloadCommands


class EyvanController(
    _ControllerLifecycle,
    _ControllerFleets,
    _ControllerScheduling,
    _ControllerOutbox,
    _ControllerCapacityCommands,
    _ControllerWorkloadCommands,
    _ControllerPreemption,
    _ControllerReclaim,
    _ControllerSettlement,
    _ControllerServices,
    _ControllerObservation,
    _ControllerLogArchival,
    _ControllerStaging,
    _ControllerExpansion,
    _ControllerWebhookDelivery,
    _ControllerLoops,
):
    """Turn pure scheduler decisions into fenced, recoverable backend actions."""


__all__ = (
    "CapacityApplySuperseded",
    "EyvanController",
    "HttpServiceHealthProbe",
    "LeaderElector",
    "LeaderPhase",
    "PreemptionCheckpointError",
    "SingleProcessLeader",
    "StaleOutboxCommand",
)
