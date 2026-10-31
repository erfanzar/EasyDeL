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
stores the merged graphstate in memory. Optional publishing uses the merged
policy state or a caller-supplied publisher; model IDs are never loaded.
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
        push_to_hub: Publish after merging, using ``publish_fn`` or the merged
            policy state's model Hub API.
        hub_model_id: Destination repository for the model Hub API.
        publish_fn: Optional publisher receiving the merged graphstate. Useful
            for raw pytrees or custom checkpoint publishing.
        target_state: Default initialized merge target for trainer events.
    """

    def __init__(
        self,
        merge_config: MergeConfig | None = None,
        merge_at_every_checkpoint: bool = False,
        push_to_hub: bool = False,
        hub_model_id: str | None = None,
        publish_fn: tp.Callable[[object], object] | None = None,
        target_state: object | None = None,
    ) -> None:
        """Store callback policy and merge configuration.

        Args:
            merge_config: Optional merge settings. ``None`` creates a default
                equal-weight linear merge config.
            merge_at_every_checkpoint: Controls whether ``on_save`` performs
                merges. When false, merging is deferred to ``on_train_end``.
            push_to_hub: Whether to publish each completed merge.
            hub_model_id: Explicit Hub destination for model publishing.
            publish_fn: Optional custom merged-graphstate publisher.
            target_state: Default merge target, overridable by each event.
        """
        self.merge_config = merge_config or MergeConfig()
        self.merge_at_every_checkpoint = bool(merge_at_every_checkpoint)
        self.push_to_hub = bool(push_to_hub)
        self.hub_model_id = hub_model_id
        self.publish_fn = publish_fn
        self.target_state = target_state
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

    def _merge_and_publish(self, policy_state, target_state, **kwargs) -> None:
        """Store the merge and optionally publish it to the explicit destination."""
        self.last_merged_state = self.merge_states(policy_state, target_state)
        if not self.push_to_hub:
            return
        if self.publish_fn is not None:
            self.publish_fn(self.last_merged_state)
            return
        repo_id = kwargs.get("hub_model_id") or self.hub_model_id
        if repo_id is None:
            raise ValueError("Merge publishing requires `hub_model_id` or `publish_fn`.")
        merge = getattr(policy_state, "merge", None)
        if not callable(merge):
            raise TypeError("Publishing raw pytrees requires `publish_fn`; otherwise pass an EasyDeLState.")
        model = merge(tree=self.last_merged_state)
        model.push_to_hub(repo_id=repo_id, **kwargs.get("hub_kwargs", {}))

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
            policy_state = kwargs.get("policy_state")
            if policy_state is None:
                policy_state = kwargs.get("state")
            target_state = kwargs.get("target_state", self.target_state)
            if policy_state is not None and target_state is not None:
                self._merge_and_publish(
                    policy_state,
                    target_state,
                    hub_model_id=kwargs.get("hub_model_id"),
                    hub_kwargs=kwargs.get("hub_kwargs", {}),
                )
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
            policy_state = kwargs.get("policy_state")
            if policy_state is None:
                policy_state = kwargs.get("state")
            target_state = kwargs.get("target_state", self.target_state)
            if policy_state is not None and target_state is not None:
                self._merge_and_publish(
                    policy_state,
                    target_state,
                    hub_model_id=kwargs.get("hub_model_id"),
                    hub_kwargs=kwargs.get("hub_kwargs", {}),
                )
        del args
