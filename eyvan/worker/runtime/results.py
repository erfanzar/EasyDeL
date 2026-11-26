# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Public callable-result API backed by the shared worker protocol."""

from __future__ import annotations

from eyvan.worker.runtime.callable_protocol import FORMAT_VERSION, dump_result, load_result

__all__ = ("FORMAT_VERSION", "dump_result", "load_result")
