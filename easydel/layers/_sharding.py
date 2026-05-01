# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Layer sharding compatibility exports.

Generic sharding resolution lives in ``easydel.infra.sharding``. This module
keeps the historical layer import path without maintaining a second
implementation.
"""

from __future__ import annotations

from easydel.infra.sharding import pick_mesh, resolve_safe_sharding

__all__ = ["pick_mesh", "resolve_safe_sharding"]
