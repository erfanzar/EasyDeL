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
"""Trainer callback wrapper around EasyDeL's native pytree merge.

The callback provides a TRL-compatible place to trigger a merge while keeping
EasyDeL semantics: callers pass initialized states or pytrees, and the callback
stores the merged graphstate in memory. It does not load model IDs, write merged
checkpoints by itself, or push artifacts to the Hub.
"""

from __future__ import annotations

import typing as tp

from ._fn import _as_graphstate, merge_pytrees
from .merge_config import MergeConfig


class MergeModelCallback:
    """Callback that merges policy and target states at save/train-end hooks.

    The callback is intentionally small: it delegates all leaf math to
    :func:`merge_pytrees` and only decides *when* to run the merge. The merged
    object is stored as ``last_merged_state`` so tests or downstream training
    code can decide how to save, inspect, or publish it.

    Args:
        merge_config: Merge settings. If omitted, defaults to a normalized
            linear merge with equal policy/target weights.
        merge_at_every_checkpoint: When true, ``on_save`` attempts a merge
            whenever both policy and target states are supplied in callback
            kwargs. When false, ``on_train_end`` performs the merge instead.
        push_to_hub: Compatibility flag retained for TRL-style constructor
            parity. This callback does not upload by itself; callers can use
            ``last_merged_state`` in their own publishing path.
    """

    def __init__(
        self,
        merge_config: MergeConfig | None = None,
        merge_at_every_checkpoint: bool = False,
        push_to_hub: bool = False,
    ) -> None:
        """Store callback policy and merge configuration.

        Args:
            merge_config: Optional merge settings. ``None`` creates a default
                equal-weight linear merge config.
            merge_at_every_checkpoint: Controls whether ``on_save`` performs
                merges. When false, merging is deferred to ``on_train_end``.
            push_to_hub: Compatibility flag retained for callers that construct
                TRL-style merge callbacks. It is stored but not acted on by this
                in-memory callback.
        """
        self.merge_config = merge_config or MergeConfig()
        self.merge_at_every_checkpoint = bool(merge_at_every_checkpoint)
        self.push_to_hub = bool(push_to_hub)
        # Holds the most recent merge result; None until on_save/on_train_end runs a merge, so callers
        # (and the docstring's promised attribute) never hit AttributeError when reading it early.
        self.last_merged_state = None

    def merge_states(self, policy_state: object, target_state: object) -> object:
        """Merge two states or raw pytrees with this callback's config.

        Args:
            policy_state: Base state, module wrapper, or raw pytree. If it
                exposes ``graphstate`` (directly or under ``.model``), that
                graphstate is merged.
            target_state: Target state, module wrapper, or raw pytree with the
                same pytree definition as the policy graphstate.

        Returns:
            The merged graphstate/pytree. The return value is not wrapped back
            into an ``EasyDeLState`` because callback callers may need to decide
            how optimizer state, checkpoint metadata, or trainable selectors
            should be handled.
        """
        return merge_pytrees(_as_graphstate(policy_state), _as_graphstate(target_state), self.merge_config)

    def on_save(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        """Handle a checkpoint-save callback event.

        Expected keyword arguments:
            ``policy_state`` or ``state``:
                Base state/pytree to merge from.
            ``target_state``:
                Target state/pytree to merge into the policy.

        Behavior:
            If ``merge_at_every_checkpoint`` is false, this hook is a no-op. If
            it is true and both required states are present, the merged pytree is
            stored on ``self.last_merged_state``. Missing state kwargs are
            treated as a no-op so the callback can be attached to trainer
            events that do not provide a target state.
        """
        if self.merge_at_every_checkpoint:
            policy_state = kwargs.get("policy_state") or kwargs.get("state")
            target_state = kwargs.get("target_state")
            if policy_state is not None and target_state is not None:
                self.last_merged_state = self.merge_states(policy_state, target_state)
        del args

    def on_train_end(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        """Handle the end-of-training callback event.

        This hook mirrors :meth:`on_save` but only runs when
        ``merge_at_every_checkpoint`` is false. It reads ``policy_state`` (or
        fallback ``state``) and ``target_state`` from keyword arguments and
        stores the merged pytree in ``last_merged_state`` when both are present.
        Per-checkpoint configurations leave train-end merging disabled to avoid
        performing the same merge twice.
        """
        if not self.merge_at_every_checkpoint:
            policy_state = kwargs.get("policy_state") or kwargs.get("state")
            target_state = kwargs.get("target_state")
            if policy_state is not None and target_state is not None:
                self.last_merged_state = self.merge_states(policy_state, target_state)
        del args
