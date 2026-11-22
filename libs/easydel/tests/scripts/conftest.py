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

"""Put the workspace root on ``sys.path`` for the runner-script tests.

The scripts under test live in the repo-root ``scripts/`` directory, which is
a namespace package outside the ``easydel`` distribution. pytest's rootdir is
``libs/easydel`` (that is where the ini section lives), so ``import
scripts.convert_hf_to_easydel`` is unimportable by default and the whole
module errors at collection — which aborts the entire suite run, not just
this file.
"""

import sys
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]

if (_WORKSPACE_ROOT / "scripts").is_dir() and str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))
