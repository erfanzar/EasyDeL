# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainer-local model/state helpers."""

from __future__ import annotations

import spectrax as spx

from easydel.infra.base_state import EasyDeLState


def disable_state_dropout(state: EasyDeLState | None) -> EasyDeLState | None:
    """Return a state whose module graph is in eval mode.

    Optional state slots pass through as ``None``. Concrete states expose their
    module, call ``eval()`` to disable dropout-like stochastic layers, and then
    export the updated graph definition back into the state when supported.
    """
    if state is None:
        return None
    module = state.model
    module.eval()
    try:
        graphdef, _ = spx.export(module)
    except TypeError:
        return state
    return state.replace(graphdef=graphdef)


def reject_string_model_id(model_id: str, *, role: str = "model") -> None:
    """Raise for TRL-style string model identifiers passed to trainers.

    These EasyDeL trainer paths require already-initialized modules or states.
    Rejecting strings prevents hidden model loading, network access, and
    runtime behavior that differs from the rest of the training suite.
    """
    raise ValueError(
        f"EasyDeL trainers do not load {role} from string ids. "
        f"Load {model_id!r} with the appropriate AutoEasyDeLModel class first, "
        "then pass the initialized module or EasyDeLState to the trainer."
    )
