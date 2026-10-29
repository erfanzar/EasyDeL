# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SpecTrax core: variables, modules, graph, state, selectors, and errors.

This subpackage contains the runtime foundation every other spectrax
subpackage builds on. Public names are re-exported here for convenience;
:mod:`spectrax` itself re-exports them again for end users.
"""

from .errors import (
    CyclicGraphError,
    IllegalMutationError,
    LazyInitUnderTransformError,
    SelectorError,
    SpecTraxError,
)
from .lazy_init import lazy_init
from .paths import Path, path_to_str, str_to_path
from .policy import Policy
from .registry import qualified_name, resolve_class
from .sharding import AxisNames, Sharding
from .stage_assignment import assign_stage
from .state import State, StateCallABI, state_call_abi
from .static import Static
from .variable import Buffer, InitPlacementHook, Parameter, Variable, variable_init_placement

__all__ = [
    "AxisNames",
    "Buffer",
    "CyclicGraphError",
    "IllegalMutationError",
    "InitPlacementHook",
    "LazyInitUnderTransformError",
    "Parameter",
    "Path",
    "Policy",
    "SelectorError",
    "Sharding",
    "SpecTraxError",
    "State",
    "StateCallABI",
    "Static",
    "Variable",
    "assign_stage",
    "lazy_init",
    "path_to_str",
    "qualified_name",
    "resolve_class",
    "state_call_abi",
    "str_to_path",
    "variable_init_placement",
]
