# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Public rich-record queries for the thread-safe tracking index facade."""

from __future__ import annotations

import sqlite3
import typing as tp
from contextlib import AbstractContextManager

from eyvan.track._format import validate_sha256
from eyvan.track._store import TrackingStore
from eyvan.track.tracker import phase2, phase3
from eyvan.track.tracker.records import integer, metric

Record = dict[str, tp.Any]


def _page(limit: int, offset: int, maximum: int, include_replayed: bool = False) -> None:
    integer(limit, minimum=1, maximum=maximum)
    integer(offset, minimum=0)
    if not isinstance(include_replayed, bool):
        raise ValueError("include_replayed must be boolean")


class PhaseThreeIndexMixin:
    """Read APIs using the facade's validated scope and connection ownership."""

    if tp.TYPE_CHECKING:
        _connection: tp.Callable[[], AbstractContextManager[sqlite3.Connection]]
        _scope: tp.Callable[..., tuple[str, ...]]

    def _rich_query(
        self,
        org: str,
        project: str,
        run_id: str,
        query: tp.Callable[..., tp.Any],
        **parameters: tp.Any,
    ) -> tp.Any:
        scope = self._scope(org, project, run_id)
        store = parameters.get("store")
        with self._connection() as db:
            if not phase2.exists(db, scope):
                return None
            if query is phase3.blob:
                result = phase3.blob_reference(db, scope, parameters["digest"])
            else:
                detached_parameters = {**parameters, "store": None} if "store" in parameters else parameters
                result = query(db, scope, **detached_parameters)
        if query is phase3.blob:
            return phase3.blob(result, scope, parameters["digest"], store)
        return phase3.annotate_availability(result, store, scope) if store is not None else result

    def rollouts(
        self,
        org: str,
        project: str,
        run_id: str,
        kind: str | None = None,
        actor: str | None = None,
        limit: int = 50,
        offset: int = 0,
        include_replayed: bool = False,
        *,
        store: TrackingStore | None = None,
    ) -> Record | None:
        """Return filtered sampled-trajectory summaries, newest first."""

        phase2.validate_episode_query(kind, actor)
        _page(limit, offset, 500, include_replayed)
        return self._rich_query(
            org,
            project,
            run_id,
            phase3.rollouts,
            kind=kind,
            actor=actor,
            limit=limit,
            offset=offset,
            include_replayed=include_replayed,
            store=store,
        )

    def rollout_detail(
        self,
        org: str,
        project: str,
        run_id: str,
        rollout_id: str,
        *,
        store: TrackingStore | None = None,
    ) -> Record | None:
        """Return a full trajectory by stable id, even when its attempt is replayed."""

        phase3.validate_id(rollout_id)
        return self._rich_query(org, project, run_id, phase3.rollout_detail, rollout_id=rollout_id, store=store)

    def completions(
        self,
        org: str,
        project: str,
        run_id: str,
        step: int | None = None,
        limit: int = 50,
        offset: int = 0,
        sort: str = "recent",
        include_replayed: bool = False,
    ) -> Record | None:
        """Return completion group summaries and all available distinct steps."""

        if step is not None:
            integer(step, minimum=0, maximum=2**63 - 1)
        if sort not in ("recent", "zero_variance", "reward"):
            raise ValueError("invalid completion sort")
        _page(limit, offset, 500, include_replayed)
        return self._rich_query(
            org,
            project,
            run_id,
            phase3.completions,
            step=step,
            limit=limit,
            offset=offset,
            sort=sort,
            include_replayed=include_replayed,
        )

    def completion_detail(self, org: str, project: str, run_id: str, completion_id: str) -> Record | None:
        """Return a group's complete prompt/responses alongside its summary."""

        phase3.validate_id(completion_id)
        return self._rich_query(org, project, run_id, phase3.completion_detail, completion_id=completion_id)

    def completion_stats(self, org: str, project: str, run_id: str, points: int = 200) -> Record | None:
        """Return per-step response statistics in at most ``points`` buckets."""

        integer(points, minimum=1, maximum=2000)
        return self._rich_query(org, project, run_id, phase3.completion_stats, points=points)

    def tables(
        self,
        org: str,
        project: str,
        run_id: str,
        key: str | None = None,
        step: int | None = None,
        *,
        store: TrackingStore | None = None,
    ) -> Record | None:
        """List table keys or retrieve a winning snapshot, optionally at one step."""

        if key is not None:
            metric(key)
        if step is not None:
            integer(step, minimum=0, maximum=2**63 - 1)
        return self._rich_query(org, project, run_id, phase3.tables, key=key, step=step, store=store)

    def media(
        self,
        org: str,
        project: str,
        run_id: str,
        key: str | None = None,
        limit: int = 100,
        offset: int = 0,
        *,
        store: TrackingStore | None = None,
    ) -> Record | None:
        """Return non-replayed media references with optional live availability."""

        if key is not None:
            metric(key)
        _page(limit, offset, 1000)
        return self._rich_query(org, project, run_id, phase3.media, key=key, limit=limit, offset=offset, store=store)

    def blob(
        self,
        org: str,
        project: str,
        run_id: str,
        digest: str,
        *,
        store: TrackingStore,
    ) -> tuple[bytes, str] | None:
        """Return verified bytes/MIME only for an indexed reference of this run.

        Raises:
            BlobIntegrityError: If the content hash or maximum byte size fails.
            TrackingStoreError: If the store is inaccessible (not merely missing).
        """

        validate_sha256(digest)
        return self._rich_query(org, project, run_id, phase3.blob, digest=digest, store=store)


__all__ = ("PhaseThreeIndexMixin",)
