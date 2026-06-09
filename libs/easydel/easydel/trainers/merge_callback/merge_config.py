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
"""Configuration object for native EasyDeL model-state merging.

The merge callback deliberately works on initialized EasyDeL pytrees instead of
loading model IDs or invoking mergekit. ``MergeConfig`` contains the method and
scalar schedules needed by :func:`merge_pytrees`, plus path/dtype fields that
are useful as metadata when mirroring TRL/mergekit-style config surfaces.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass


@dataclass
class MergeConfig:
    """User-facing settings for merging two initialized EasyDeL pytrees.

    ``MergeModelCallback`` passes this config to :func:`merge_pytrees`, which
    flattens the policy and target pytrees and applies the selected merge method
    to each compatible array leaf. Leaves that are not arrays, or array leaves
    whose shapes do not match, are preserved from the policy tree.

    Attributes:
        method: Name of the registered merge method. Built-ins are ``linear``,
            ``slerp``, ``ties``, and ``dare_ties``.
        policy_model_path: Optional policy model identifier retained only for
            logging or checkpoint metadata. EasyDeL does not load from this path
            inside the merge callback.
        target_model_path: Optional target model identifier retained only for
            logging or checkpoint metadata.
        policy_model_weight: Non-negative coefficient for the policy leaf.
            If ``normalize`` is truthy, it is divided by the sum of policy and
            target weights before the merge method sees it.
        target_model_weight: Non-negative coefficient for the target leaf or
            target task vector. It is normalized with ``policy_model_weight``
            when ``normalize`` is truthy.
        policy_model_density: Compatibility density schedule for the policy
            side. The current two-tree methods do not consume it directly, but
            it is preserved in ``create()`` for metadata/API parity.
        target_model_density: Density schedule used by task-vector methods.
            ``merge_pytrees`` selects ``target_model_density[leaf_index % n]``
            for each array leaf.
        normalize: Truthy value enabling weight normalization. Kept as a float
            for compatibility with mergekit-style config payloads.
        t_values: Interpolation parameter used by ``slerp`` and available to
            custom merge methods through ``MergeLeafContext.t_value``.
        dtype: Output dtype hint retained for metadata/API parity. Current
            in-memory merges leave dtype conversion to normal JAX arithmetic.
        seed: Integer seed used by deterministic stochastic methods such as
            ``dare_ties``.
    """

    method: tp.Literal["linear", "ties", "dare_ties", "slerp"] = "linear"
    policy_model_path: str | None = None
    target_model_path: str | None = None
    policy_model_weight: float = 0.5
    target_model_weight: float = 0.5
    policy_model_density: tuple[float, ...] = (1.0, 0.7, 0.1)
    target_model_density: tuple[float, ...] = (1.0,)
    normalize: float = 1.0
    t_values: float = 0.5
    dtype: str = "float16"
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate method names, weights, and density schedules.

        Raises:
            ValueError: If the method is not registered as a built-in method
                name, any merge weight is negative, or any configured density is
                outside ``[0, 1]``.
        """
        if self.method not in {"linear", "ties", "dare_ties", "slerp"}:
            raise ValueError("`method` must be one of 'linear', 'ties', 'dare_ties', or 'slerp'.")
        if self.policy_model_weight < 0.0 or self.target_model_weight < 0.0:
            raise ValueError("Merge weights must be non-negative.")
        if any(density < 0.0 for density in (*self.policy_model_density, *self.target_model_density)):
            raise ValueError("Merge densities must be non-negative.")
        if any(density > 1.0 for density in (*self.policy_model_density, *self.target_model_density)):
            raise ValueError("Merge densities must be <= 1.0.")

    def create(self) -> dict[str, object]:
        """Serialize merge settings into a plain dictionary.

        Returns:
            A shallow dictionary containing only config values. Array pytrees and
            callable registry entries are intentionally excluded, so the result
            can be stored in checkpoint metadata, logged, or passed to external
            compatibility code that expects a mergekit-like payload.
        """
        return {
            "method": self.method,
            "policy_model_path": self.policy_model_path,
            "target_model_path": self.target_model_path,
            "policy_model_weight": self.policy_model_weight,
            "target_model_weight": self.target_model_weight,
            "policy_model_density": self.policy_model_density,
            "target_model_density": self.target_model_density,
            "normalize": self.normalize,
            "t_values": self.t_values,
            "dtype": self.dtype,
            "seed": self.seed,
        }
