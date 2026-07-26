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
"""Frozen policy copies shared by the online-RL and preference trainers.

Two mechanisms live here, both about holding a copy of the policy that the
compiled training step must not invalidate:

* :class:`OwnedPolicySnapshot` -- the behavior policy an asynchronous rollout
  generates from, with an explicit release.
* :func:`polyak_update_reference_graphstate` -- the periodic reference-model
  refresh used by ``sync_ref_model``.

An asynchronous RL trainer cannot generate from its live training state: the
compiled train step is compiled with ``donate_argnums=(0,)``, so every step
invalidates the buffers a concurrent rollout thread would be reading. The
rollout therefore needs its own frozen copy of the policy -- which is also what
makes the behavior policy well defined, since ``old_per_token_logps`` must
correspond to the weights that actually sampled the tokens.

The snapshot is deliberately *params-only*: the optimizer transform and slots
are dropped, so the extra cost is one copy of ``graphstate``/``graphother``
rather than a second full training state.

Snapshot lifetime is managed explicitly rather than by reference counting.
Python refcounts cannot free a snapshot on time because the cached eSurge
engine keeps its own reference to the graph trees it last generated from, so a
naive ``del`` leaves the previous snapshot resident while the next one is being
allocated -- briefly doubling the extra footprint. :meth:`OwnedPolicySnapshot.release`
deletes the device buffers it owns, making the free deterministic.

Releasing while the engine still references the arrays is safe as long as the
engine is idle: :meth:`EasyGenerationMixin.get_esurge` refreshes engine weights
(``_refresh_esurge_engine_weights``) before the next generation, so the stale
references are replaced before anything executes. Callers must only release
between rollouts, which :meth:`OwnedPolicySnapshot.release` cannot verify for
them.
"""

from __future__ import annotations

import typing as tp

import jax

from easydel.infra.base_state import EasyDeLState
from easydel.utils.helpers import get_logger

logger = get_logger(__name__)


def _copy_leaf(value: tp.Any) -> tp.Any:
    """Return an independently owned copy of a JAX array leaf.

    Args:
        value: Arbitrary pytree leaf.

    Returns:
        A fresh device array for ``jax.Array`` leaves (same shape, dtype, and
        sharding), otherwise ``value`` unchanged.
    """
    if isinstance(value, jax.Array):
        return value.copy()
    return value


def polyak_update_reference_graphstate(
    reference_graphstate: tp.Any,
    policy_graphstate: tp.Any,
    *,
    alpha: float,
) -> tp.Any:
    """Mix the live policy parameters into a frozen reference graphstate.

    Computes ``alpha * policy + (1 - alpha) * reference`` leafwise. The result is
    always freshly allocated, so the reference never ends up aliasing the live
    training state's buffers -- the compiled step donates those and invalidates
    them on the next update. Callers must therefore *not* copy
    ``policy_graphstate`` before calling this: the arithmetic already produces
    new arrays, and a defensive copy just adds a transient full-parameter
    allocation on top of the two operands.

    Args:
        reference_graphstate: Current reference parameters.
        policy_graphstate: Live policy parameters to mix in.
        alpha: Weight on the policy in ``[0, 1]``. ``alpha >= 1.0`` degenerates
            to a plain copy of the policy, which is both cheaper than the
            arithmetic and avoids keeping the outgoing reference resident.

    Returns:
        A new graphstate for the reference model.
    """
    if alpha >= 1.0:
        return jax.tree_util.tree_map(_copy_leaf, policy_graphstate)
    return jax.tree_util.tree_map(
        lambda policy, reference: alpha * policy + (1.0 - alpha) * reference,
        policy_graphstate,
        reference_graphstate,
    )


def drop_optimizer_state(state: tp.Any) -> tp.Any:
    """Return a frozen state without its optimizer transform or slots.

    Teacher, reference, and target states are never stepped -- no trainer reads
    ``tx``, ``opt_state``, ``init_tx``, ``apply_gradients``, or ``save_state`` on
    one. A caller who supplies a state loaded from a *training* checkpoint would
    otherwise keep its optimizer slots (two extra parameter sets for Adam)
    resident for the whole run, so every frozen-state assignment normalizes
    through here.

    Dropping the reference lets the buffers be freed as soon as the caller
    releases their own handle on the state; it never mutates the input.

    Args:
        state: Frozen :class:`EasyDeLState`. Anything else is returned unchanged
            so call sites do not need their own type check.

    Returns:
        ``state`` when it already carries no optimizer, otherwise a copy with
        ``tx`` and ``opt_state`` cleared. Parameter buffers are shared, not
        copied.
    """
    if not isinstance(state, EasyDeLState):
        return state
    if state.tx is None and state.opt_state is None:
        return state
    return state.replace(tx=None, opt_state=None)


def copy_frozen_policy(
    state: EasyDeLState,
    *,
    cache_scope_key: str | None = None,
) -> EasyDeLState:
    """Copy a state's parameters and buffers into a new inference-only state.

    Used wherever a trainer needs its *own* frozen copy of the policy (a
    reference model, a rollout snapshot). Only ``graphstate``/``graphother``/
    ``step`` are copied and the optimizer is dropped, so the copy costs one
    parameter set even when the source carries optimizer slots -- unlike a
    whole-pytree deep copy, which would duplicate those too.

    Args:
        state: Source state. Its buffers are read but never adopted, so the
            caller may keep donating it to a compiled step.
        cache_scope_key: Optional eSurge cache scope for the copy. Pass a key
            distinct from the source's to keep the copy's engine cached
            separately.

    Returns:
        A state owning independent copies of every array leaf.
    """
    return state.replace(
        step=_copy_leaf(state.step),
        graphstate=jax.tree_util.tree_map(_copy_leaf, state.graphstate),
        graphother=jax.tree_util.tree_map(_copy_leaf, state.graphother),
        tx=None,
        opt_state=None,
        **({} if cache_scope_key is None else {"esurge_cache_scope_key": cache_scope_key}),
    )


class OwnedPolicySnapshot:
    """A params-only policy copy whose device buffers this object owns.

    Args:
        state: Snapshot state whose array leaves were freshly copied for this
            object and are therefore safe to delete on release.
        policy_step: Optimizer step of the training state the snapshot was taken
            from, used for staleness accounting.

    Attributes:
        policy_step: Policy version this snapshot represents.
        nbytes: Total logical size of the owned arrays across all devices.
    """

    __slots__ = ("_released", "_state", "nbytes", "policy_step")

    def __init__(self, state: EasyDeLState, policy_step: int) -> None:
        self._state = state
        self._released = False
        self.policy_step = int(policy_step)
        self.nbytes = sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(state) if isinstance(leaf, jax.Array))

    @classmethod
    def from_training_state(
        cls,
        state: EasyDeLState,
        *,
        policy_step: int,
        cache_scope_key: str | None = None,
    ) -> OwnedPolicySnapshot:
        """Copy a training state into an inference-only snapshot.

        The optimizer transform and slots are dropped: generation and policy
        log-prob scoring only need the model graph, so a snapshot costs one copy
        of the parameters and buffers rather than a full training state.

        Args:
            state: Live training state. Its buffers are read but never adopted,
                so the caller may keep donating it to the compiled train step.
            policy_step: Optimizer step this snapshot corresponds to.
            cache_scope_key: Optional eSurge cache scope for the snapshot. Pass a
                key distinct from the training state's so the snapshot's engine
                is cached separately from any engine created off the live state.

        Returns:
            A snapshot owning independent copies of every array leaf.
        """
        snapshot_state = copy_frozen_policy(state, cache_scope_key=cache_scope_key)
        return cls(jax.block_until_ready(snapshot_state), policy_step)

    @property
    def released(self) -> bool:
        """Whether the owned buffers have been deleted."""
        return self._released

    @property
    def state(self) -> EasyDeLState:
        """The snapshot state.

        Raises:
            RuntimeError: If the snapshot has already been released, which would
                otherwise surface later as a deleted-buffer error inside a
                rollout worker.
        """
        if self._released:
            raise RuntimeError(
                f"policy snapshot from step {self.policy_step} was released; "
                "take a new snapshot before submitting another rollout"
            )
        return self._state

    def release(self) -> int:
        """Delete the device buffers this snapshot owns.

        Idempotent. Only call this when no rollout is reading the snapshot; any
        engine that generated from it must be idle, and will pick the next
        snapshot's weights up through the normal engine weight refresh.

        Returns:
            Number of bytes released, summed across devices. ``0`` when the
            snapshot was already released.
        """
        if self._released:
            return 0
        self._released = True
        released = 0
        for leaf in jax.tree_util.tree_leaves(self._state):
            if not isinstance(leaf, jax.Array) or leaf.is_deleted():
                continue
            try:
                released += leaf.nbytes
                leaf.delete()
            except Exception:  # pragma: no cover - best-effort buffer release
                logger.debug("Failed to release a policy-snapshot buffer", exc_info=True)
        self._state = None  # pyright: ignore[reportAttributeAccessIssue]
        return released

    def __repr__(self) -> str:
        """Return a diagnostic summary of the snapshot."""
        return (
            f"{type(self).__name__}(policy_step={self.policy_step}, "
            f"gib={self.nbytes / 1024**3:.2f}, released={self._released})"
        )


__all__ = (
    "OwnedPolicySnapshot",
    "copy_frozen_policy",
    "drop_optimizer_state",
    "polyak_update_reference_graphstate",
)
