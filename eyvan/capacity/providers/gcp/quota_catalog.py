# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reading a project's TPU quota limits without spawning gcloud.

The limits come from one ``gcloud beta quotas info list``, which costs about
four seconds -- almost all of it Python interpreter startup inside gcloud rather
than time spent talking to Google. That was the largest single cost left in a
quota read after the per-zone accelerator listing moved to the API, and the
Cloud console answers the same question instantly because it calls the API.

So does this. It reuses the credential path already established for the
accelerator catalogue -- one ``gcloud auth print-access-token``, inheriting
whatever login the operator has -- and then plain HTTPS.

**The subprocess path stays.** It works where a raw HTTPS call may not: a proxy
only gcloud is configured for, or an environment where the token cannot be
minted. This is the fast path and that is the fallback, not a replacement.
"""

from __future__ import annotations

import threading
import typing as tp
from dataclasses import dataclass, field

from eyvan.capacity.providers.gcp.accelerator_catalog import (
    AUTH_FAILURES,
    AcceleratorCatalogError,
    gcloud_access_token,
)

#: Bound on the whole read. One request, so this is the deadline rather than a
#: per-page budget; generous enough for a slow link, tight enough that a hung
#: socket cannot hold a discovery open.
REQUEST_TIMEOUT_S = 30.0

#: Large enough that no project Google currently serves needs a second page.
#: Paging is still handled -- this only avoids the round trips.
PAGE_SIZE = 500

_ENDPOINT = "https://cloudquotas.googleapis.com/v1/projects/{project}/locations/global/services/{service}/quotaInfos"


class QuotaCatalogError(RuntimeError):
    """The quota limits could not be read over the Cloud Quotas API."""


@dataclass
class ApiQuotaCatalog:
    """Read one project's quota limits over the Cloud Quotas API.

    Attributes:
        service: The service whose quotas are wanted, e.g. ``tpu.googleapis.com``.
        token_source: Where the bearer credential comes from. Injected so a test
            never shells out and a caller can supply its own.
    """

    service: str
    token_source: tp.Callable[[], str] = gcloud_access_token
    _token: str | None = field(default=None, init=False, repr=False)
    _token_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _refresh_token(self, stale: str | None) -> str:
        """Re-mint the cached bearer token once, collapsing concurrent refreshes.

        ``stale`` is the token the failed request carried: the first caller to
        report it re-mints, and every other caller -- whose 401 was caused by
        the same aged-out credential -- reuses whatever replaced it.
        """

        with self._token_lock:
            if self._token is None or self._token == stale:
                self._token = self.token_source()
            return self._token

    def limits(self, project: str) -> list[dict[str, object]]:
        """Return the raw quota entries, shaped as the gcloud reader returns them.

        The REST resource and the gcloud rows carry the same field names for
        everything the caller reads -- ``metric``, ``quotaId`` and
        ``dimensionsInfos`` -- so this deliberately returns the list unchanged
        rather than projecting it. A projection here would be a second schema to
        keep in step with the fallback path, which is how the two would drift.
        """

        import httpx

        entries: list[dict[str, object]] = []
        page: str | None = None
        try:
            token = self._token or self.token_source()
            self._token = token
            with httpx.Client(timeout=REQUEST_TIMEOUT_S, trust_env=False) as client:
                while True:
                    params: dict[str, object] = {"pageSize": PAGE_SIZE}
                    if page:
                        params["pageToken"] = page
                    headers = {"Authorization": f"Bearer {token}"}
                    response = client.get(
                        _ENDPOINT.format(project=project, service=self.service),
                        params=params,
                        headers=headers,
                    )
                    if response.status_code in AUTH_FAILURES:
                        # The cached credential aged out (a Google token lasts
                        # about an hour). Mint a fresh one and retry this page
                        # once, as GcsRestObject does; a request that still
                        # refuses falls back to gcloud below.
                        token = self._refresh_token(token)
                        headers = {"Authorization": f"Bearer {token}"}
                        response = client.get(
                            _ENDPOINT.format(project=project, service=self.service),
                            params=params,
                            headers=headers,
                        )
                    if response.status_code != 200:
                        raise QuotaCatalogError(
                            f"Cloud Quotas API answered {response.status_code} for {project!r}; falling back to gcloud"
                        )
                    payload = response.json()
                    for item in payload.get("quotaInfos", []) or []:
                        if isinstance(item, dict):
                            entries.append(item)
                    page = payload.get("nextPageToken")
                    if not page:
                        break
        except (httpx.HTTPError, ValueError) as exc:
            raise QuotaCatalogError(f"could not read quotas for {project!r}: {exc}") from exc
        except AcceleratorCatalogError as exc:
            # The shared credential helper failed, which is a gcloud problem and
            # not a quota one; say so in this path's own vocabulary.
            raise QuotaCatalogError(str(exc)) from exc
        return entries


__all__ = ("PAGE_SIZE", "REQUEST_TIMEOUT_S", "ApiQuotaCatalog", "QuotaCatalogError")
