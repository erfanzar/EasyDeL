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
"""Repo-root entry point for HF -> EasyDeL checkpoint conversion.

Delegates to :mod:`easydel.scripts.convert_hf_to_easydel`, which holds the
implementation. This file used to be a full copy of it, and the two drifted:
the copy still validated a 5-axis mesh (``dp,fsdp,ep,tp,sp``) from before the
``pp`` axis existed, so the documented runner rejected the 6-tuple its own
default declared and could not express pipeline parallelism at all. A wrapper
cannot drift.

Usage is unchanged::

    python scripts/convert_hf_to_easydel.py --source <repo|path|gs://...> --out <dir|gs://...>
"""

from __future__ import annotations

import sys as _sys

from easydel.scripts import convert_hf_to_easydel as _impl
from easydel.scripts.convert_hf_to_easydel import *  # noqa: F403
from easydel.scripts.convert_hf_to_easydel import main

# Re-export the module's private helpers too: this path is imported directly by
# tooling and tests (e.g. `from scripts.convert_hf_to_easydel import
# _infer_task_from_hf_config`), and a wrapper that dropped them would break
# those imports just as surely as the drift it replaces.
for _name in dir(_impl):
    if _name.startswith("__"):
        continue
    globals().setdefault(_name, getattr(_impl, _name))
del _name

if __name__ == "__main__":
    _sys.exit(main())
