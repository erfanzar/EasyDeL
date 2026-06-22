# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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


"""Base actor pool management with health checking."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import ray
from ray.actor import ActorHandle

if TYPE_CHECKING:
    from ..core.cluster import HostInfo

logger = logging.getLogger("ray")

HEALTH_CHECK_TIMEOUT_S = 60
SLICE_ACTOR_START_TIMEOUT_S = 4 * 60 * 60
SCALE_POLL_S = int(os.getenv("EFORMER_SCALE_POLL_S", "30"))
SCALE_ADD_TIMEOUT_S = int(os.getenv("EFORMER_SCALE_ADD_TIMEOUT_S", "604800"))
ActorInfoT = TypeVar("ActorInfoT")

class InsufficientSlicesError(RuntimeError):
    """Raised when the requested number of TPU slices cannot be allocated.

    This exception is raised by SlicePoolManager.scale_multislice when
    none of the requested slice counts can be satisfied, typically due to:
    - Insufficient TPU resources in the cluster
    - Preemption of TPU nodes during scaling
    - Ray autoscaler unable to provision required nodes

    The exception message includes details about requested vs available slices.

    Example:
        >>> manager = SlicePoolManager(tpu_type="v4-32")
        >>> try:
        ...     manager.scale_multislice([4, 8])
        ... except InsufficientSlicesError as e:
        ...     print(f"Could not allocate TPU slices: {e}")
        ...
    """

    pass

@dataclass(frozen=True)
class ActorPoolMember(Generic[ActorInfoT]):
    """Container for an actor handle and its associated metadata.

    Attributes:
        actor: Ray actor handle for remote execution.
        actor_info: Metadata about the actor (type depends on ActorInfoT).
    """

    actor: ActorHandle
    actor_info: ActorInfoT

class ResourcePoolManager(Generic[ActorInfoT]):
    """Abstract base class for managing pools of Ray actors.

    Provides common functionality for scaling, health monitoring, and
    lifecycle management of actor pools. Subclasses should implement
    create_actor() to define how actors are created.

    Attributes:
        _actor_pool: List of active actor pool members.
    """

    def __init__(self) -> None:
        """Initialize an empty actor pool."""
        self._actor_pool: list[ActorPoolMember[ActorInfoT]] = []

    def get_all_actors_in_pool(self) -> list[ActorHandle]:
        """Get all actor handles in the pool.

        Returns:
            List of Ray actor handles.
        """
        return [m.actor for m in self._actor_pool]

    def get_all_pool_members(self) -> list[ActorPoolMember[ActorInfoT]]:
        """Get a copy of all pool members with their metadata.

        Returns:
            List of ActorPoolMember objects containing actors and their info.
        """
        return self._actor_pool.copy()

    def get_actor_pool_name(self) -> str:
        """Get a human-readable name for this actor pool.

        Returns:
            String identifier for the pool, defaults to class name.
        """
        return self.__class__.__name__

    def get_actor_name_from_actor_info(self, actor_info: ActorInfoT) -> str:
        """Generate a human-readable name from actor info.

        Args:
            actor_info: Metadata about the actor.

        Returns:
            String representation of the actor for logging.
        """
        return str(actor_info)

    def create_actor(self) -> ActorHandle:
        """Create a new actor instance.

        Must be implemented by subclasses to define actor creation logic.

        Returns:
            Ray actor handle for the newly created actor.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError

    def _remove_unhealthy_members_from_actor_pool(self) -> None:
        """Remove unhealthy actors from the pool.

        Performs health checks on all actors and removes those that are
        unresponsive, dead, or unhealthy. Attempts to kill removed actors.
        """
        if not self._actor_pool:
            return

        ref_map = {m: m.actor.healthy.remote() for m in self._actor_pool}
        refs = list(ref_map.values())

        done, _ = ray.wait(refs, num_returns=len(refs), timeout=HEALTH_CHECK_TIMEOUT_S)

        done_set = set(done)
        healthy: list[ActorPoolMember[HostInfo]] = []

        for member, ref in ref_map.items():
            name = self.get_actor_name_from_actor_info(member.actor_info)
            if ref in done_set:
                try:
                    if ray.get(ref, timeout=0):
                        healthy.append(member)
                    else:
                        logger.warning(f"Actor {name} reported unhealthy; killing")
                        try:
                            ray.kill(member.actor, no_restart=True)
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"Actor {name} health check exception ({e}); killing")
                    try:
                        ray.kill(member.actor, no_restart=True)
                    except Exception:
                        pass
            else:
                logger.warning(f"Actor {name} health timeout; killing")
                try:
                    ray.kill(member.actor, no_restart=True)
                except Exception:
                    pass

        self._actor_pool = healthy

    def _add_members_to_actor_pool(self, desired_num_actors: int) -> None:
        """Add new actors to the pool to reach desired size.

        Creates new actors asynchronously and waits for them to start.
        Actors that fail to start within the timeout are killed.

        Args:
            desired_num_actors: Target number of actors in the pool.
        """
        current = len(self._actor_pool)
        if current >= desired_num_actors:
            return
        num_to_add = desired_num_actors - current
        logger.info(f"Scaling up pool {self.get_actor_pool_name()} from {current} to {desired_num_actors}")

        actors = [self.create_actor() for _ in range(num_to_add)]
        awaitables = [(actor, actor.get_info.remote()) for actor in actors]

        logger.info(f"Waiting up to {SLICE_ACTOR_START_TIMEOUT_S}s for {num_to_add} slice actors to start...")
        ray.wait([a for _, a in awaitables], num_returns=len(awaitables), timeout=SLICE_ACTOR_START_TIMEOUT_S)

        started = 0
        for actor, info_ref in awaitables:
            try:
                info = ray.get(info_ref, timeout=0)
                self._actor_pool.append(ActorPoolMember(actor, info))
                started += 1
                logger.info(f"Added actor {self.get_actor_name_from_actor_info(info)}")
            except Exception as e:
                logger.warning(f"SliceActor failed to start in time: {e}; killing actor")
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    pass

        logger.info(f"Started {started}/{num_to_add} slice actors")

    def _remove_members_from_actor_pool(self, desired_num_actors: int) -> None:
        """Remove actors to reach the desired pool size.

        Args:
            desired_num_actors: Target number of actors in the pool.
        """
        while len(self._actor_pool) > desired_num_actors:
            member = self._actor_pool.pop()
            name = self.get_actor_name_from_actor_info(member.actor_info)
            try:
                try:
                    ray.get(member.actor.shutdown.remote(), timeout=5)
                except Exception:
                    pass
                ray.kill(member.actor, no_restart=True)
                logger.info(f"Removed actor {name}")
            except Exception as e:
                logger.error(f"Failed to kill actor {name}: {e}")

    def _scale_actor_pool(self, desired_num_actors: int) -> None:
        """Scale the actor pool to the desired size.

        First removes unhealthy actors, then adds or removes actors
        as needed to reach the target size.

        Args:
            desired_num_actors: Target number of actors in the pool.
        """
        self._remove_unhealthy_members_from_actor_pool()
        current = len(self._actor_pool)
        if current < desired_num_actors:
            self._add_members_to_actor_pool(desired_num_actors)
        elif current > desired_num_actors:
            self._remove_members_from_actor_pool(desired_num_actors)

    def drain_actor_pool(self) -> None:
        """Shut down and remove all actors from the pool.

        Attempts graceful shutdown first, then forcefully kills actors.
        Clears the actor pool after draining.
        """
        if not self._actor_pool:
            return

        shutdown_refs = []
        for member in self._actor_pool:
            try:
                shutdown_refs.append(member.actor.shutdown.remote())
            except Exception:
                pass

        try:
            ray.wait(shutdown_refs, num_returns=len(shutdown_refs), timeout=5.0)
        except Exception:
            pass

        for member in self._actor_pool:
            name = self.get_actor_name_from_actor_info(member.actor_info)
            try:
                ray.kill(member.actor, no_restart=True)
                logger.info(f"Killed actor {name}")
            except Exception as e:
                logger.error(f"Failed to kill actor {name}: {e}")

        self._actor_pool = []
