# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""EasyDeL Evaluation Module.

This module provides adapters and utilities for evaluating EasyDeL models
using standard benchmarking frameworks such as lm-evaluation-harness.

Key Components:
    eSurgeLMEvalAdapter: Adapter class that makes eSurge inference engines
        compatible with the lm-evaluation-harness framework.

Example:
    Running evaluations with lm-evaluation-harness::

        >>> from easydel.inference import eSurge
        >>> from easydel.inference.evaluations import eSurgeLMEvalAdapter
        >>> from lm_eval.evaluator import simple_evaluate
        >>>
        >>> # Create eSurge engine
        >>> surge = eSurge(model, processor)
        >>>
        >>> # Wrap with evaluation adapter
        >>> adapter = eSurgeLMEvalAdapter(surge, processor)
        >>>
        >>> # Run evaluations
        >>> results = simple_evaluate(
        ...     model=adapter,
        ...     tasks=["hellaswag", "arc_easy"]
        ... )

Note:
    To use this module, install EasyDeL with the lm_eval extra:
    ``pip install easydel[lm_eval]``
"""

from .esurge_eval import eSurgeLMEvalAdapter

__all__ = ("eSurgeLMEvalAdapter",)
