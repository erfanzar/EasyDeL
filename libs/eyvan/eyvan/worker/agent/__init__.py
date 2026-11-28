# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lazy public entry point for the out-of-process Eyvan cell agent."""

from __future__ import annotations

import typing as tp

if tp.TYPE_CHECKING:
    from fastapi import FastAPI

    from eyvan.config import EyvanSettings
    from eyvan.worker.agent.service import CellAgentService


def create_agent_app(
    service: CellAgentService | None = None,
    settings: EyvanSettings | None = None,
) -> FastAPI:
    """Import and build the cell-agent application on demand."""

    from eyvan.worker.agent.app import create_agent_app as build_agent_app

    return build_agent_app(service=service, settings=settings)


__all__ = ("create_agent_app",)
