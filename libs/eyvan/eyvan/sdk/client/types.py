# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""HTTP query and header aliases used by Eyvan transports."""

from __future__ import annotations

import typing as tp

QueryValue: tp.TypeAlias = str | int | float | bool | None
QueryParams: tp.TypeAlias = tp.Mapping[str, QueryValue]
HeaderMap: tp.TypeAlias = tp.Mapping[str, str]
