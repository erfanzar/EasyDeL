# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Which accelerator types a project can see, per zone, quickly.

Quota discovery asks every zone. Running ``gcloud compute tpus accelerator-types
list --zone Z`` per zone starts a fresh Python interpreter for each request;
that overhead can dominate discovery latency.

Use one ``gcloud auth print-access-token`` for a credential -- inheriting the
operator's login without a service-account file or a new credential path -- then
parallel HTTPS requests without per-zone process startup.

**The subprocess path stays.** It works where a raw HTTPS call may not -- a
proxy that only gcloud is configured for, an environment where the token cannot
be minted -- so this is the fast path and that is the fallback, rather than a
replacement.
"""

from __future__ import annotations

import subprocess
import threading
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from eyvan._internal.subprocess_env import provider_cli_env

#: How many zones are read at once. Bound socket use while overlapping network
#: latency across zones.
DEFAULT_CONCURRENCY = 48

#: Bound on one zone's read. Generous because a slow zone must not fail a whole
#: discovery, tight enough that one hung socket cannot hold the deadline.
REQUEST_TIMEOUT_S = 20.0

#: A page big enough that no zone Google currently offers needs a second one.
#: Paging is still handled -- this only avoids the round trips.
PAGE_SIZE = 500

#: Answer codes that mean the credential, not the zone, is the problem. A
#: Google access token lasts about an hour, so a catalog held for the life of
#: the server must re-mint once on these instead of degrading every read.
AUTH_FAILURES = frozenset({401, 403})

_ENDPOINT = "https://tpu.googleapis.com/v2/projects/{project}/locations/{zone}/acceleratorTypes"


class AcceleratorCatalogError(RuntimeError):
    """The accelerator catalogue could not be read over the API."""


def gcloud_access_token(timeout_s: float = 30.0) -> str:
    """Mint a bearer token from whatever credentials gcloud already holds.

    Deliberately not a credentials library. The operator has already logged in
    for every other Eyvan GCP operation, and introducing a second way to
    authenticate would mean two things to configure and two ways to be broken.
    """

    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=provider_cli_env(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AcceleratorCatalogError(f"could not run gcloud to mint a token: {exc}") from exc
    token = (result.stdout or "").strip()
    if result.returncode != 0 or not token:
        detail = (result.stderr or "").strip()[:200]
        raise AcceleratorCatalogError(f"gcloud could not mint an access token: {detail or result.returncode}")
    return token


@dataclass
class ApiAcceleratorCatalog:
    """Read each zone's accelerator types over the Cloud TPU API.

    Attributes:
        token_source: Where the bearer credential comes from. Injected so a test
            never shells out and a caller can supply its own.
        concurrency: Zones read at once.
    """

    token_source: tp.Callable[[], str] = gcloud_access_token
    concurrency: int = DEFAULT_CONCURRENCY
    _token: str | None = field(default=None, init=False, repr=False)
    _token_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.concurrency, bool) or not isinstance(self.concurrency, int) or self.concurrency < 1:
            raise ValueError("concurrency must be a positive integer")

    def _refresh_token(self, stale: str | None) -> str:
        """Re-mint the cached bearer token once, collapsing concurrent refreshes.

        ``stale`` is the token the failed request carried: the first thread to
        report it re-mints, and every other thread -- whose 401 was caused by
        the same aged-out credential -- reuses whatever replaced it.
        """

        with self._token_lock:
            if self._token is None or self._token == stale:
                self._token = self.token_source()
            return self._token

    def types_for_zones(self, project: str, zones: tp.Sequence[str]) -> dict[str, tuple[str, ...]]:
        """Return each zone's accelerator types, omitting zones that failed.

        A zone that cannot be read is left out rather than reported as empty.
        Those are different facts: "this zone offers nothing" would remove it
        from consideration, while "nobody could ask" must not.
        """

        if not zones:
            return {}
        import httpx

        token = self._token or self.token_source()
        self._token = token
        limits = httpx.Limits(max_connections=self.concurrency, max_keepalive_connections=self.concurrency)

        def read(client: httpx.Client, zone: str, token: str) -> tuple[str, tuple[str, ...] | None]:
            found: list[str] = []
            page: str | None = None
            try:
                while True:
                    params: dict[str, object] = {"pageSize": PAGE_SIZE}
                    if page:
                        params["pageToken"] = page
                    headers = {"Authorization": f"Bearer {token}"}
                    response = client.get(_ENDPOINT.format(project=project, zone=zone), params=params, headers=headers)
                    if response.status_code in AUTH_FAILURES:
                        # The cached credential aged out. Mint a fresh one and
                        # retry this page once, as GcsRestObject does; a zone
                        # that still refuses is dropped like any other failure.
                        token = self._refresh_token(token)
                        headers = {"Authorization": f"Bearer {token}"}
                        response = client.get(
                            _ENDPOINT.format(project=project, zone=zone), params=params, headers=headers
                        )
                    if response.status_code != 200:
                        return zone, None
                    payload = response.json()
                    for item in payload.get("acceleratorTypes", []) or []:
                        name = item.get("type") if isinstance(item, dict) else None
                        if isinstance(name, str) and name:
                            found.append(name.lower())
                    page = payload.get("nextPageToken")
                    if not page:
                        break
            except (httpx.HTTPError, ValueError):
                return zone, None
            return zone, tuple(sorted(set(found)))

        results: dict[str, tuple[str, ...]] = {}
        with (
            httpx.Client(timeout=REQUEST_TIMEOUT_S, limits=limits, trust_env=False) as client,
            ThreadPoolExecutor(max_workers=min(self.concurrency, len(zones))) as pool,
        ):
            for zone, types in pool.map(lambda zone: read(client, zone, token), zones):
                if types is not None:
                    results[zone] = types
        if not results:
            raise AcceleratorCatalogError(
                f"no zone of {project!r} could be read over the Cloud TPU API; falling back to gcloud"
            )
        return results


__all__ = (
    "AUTH_FAILURES",
    "DEFAULT_CONCURRENCY",
    "PAGE_SIZE",
    "REQUEST_TIMEOUT_S",
    "AcceleratorCatalogError",
    "ApiAcceleratorCatalog",
    "gcloud_access_token",
)
