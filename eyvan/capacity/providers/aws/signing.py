# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""AWS Signature Version 4, written out rather than imported.

**No new dependency.** ``httpx`` and ``hashlib`` are already here, and SigV4 is
four hashes and a string format. Adding botocore to sign a request would pull in
a client library whose object model then has to be kept in agreement with the
API calls Eyvan actually makes -- the same reasoning that keeps the Kubernetes
transport on plain ``httpx``.

It is also the part of an AWS driver that can be *proved* without an AWS
account. AWS publishes a worked example -- a specific request, credential, and
timestamp, with the canonical request, string-to-sign and signature it must
produce -- and the tests check this implementation against it. Everything else
in this provider is verified against a fake EC2; this is verified against AWS's
own arithmetic.

Session tokens are included in the signature, not merely sent alongside it.
``X-Amz-Security-Token`` is a signed header for temporary credentials, and a
request that sends the token unsigned is rejected in a way that reads like a
clock-skew problem.
"""

from __future__ import annotations

import hashlib
import hmac
import typing as tp
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote

#: The only algorithm EC2 accepts for a signed query API request.
ALGORITHM = "AWS4-HMAC-SHA256"

#: The digest of an empty body, precomputed because every GET has one and the
#: value appears in both the canonical request and a header.
EMPTY_PAYLOAD_SHA256 = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True, slots=True)
class AwsCredentials:
    """What a request is signed with.

    ``session_token`` is present for temporary credentials -- an instance role, a
    web-identity or assumed role -- which is how a control plane on EC2 or EKS
    ordinarily authenticates. Absent for long-lived keys.
    """

    access_key_id: str
    secret_access_key: str
    session_token: str | None = None


def _sign(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode(), hashlib.sha256).digest()


def signing_key(secret_access_key: str, *, date: str, region: str, service: str) -> bytes:
    """Derive the per-day, per-region, per-service key SigV4 signs with.

    Scoped rather than reused: a leaked signing key is useful for one service in
    one region on one day, which is the entire reason the derivation has four
    steps instead of one.
    """

    date_key = _sign(f"AWS4{secret_access_key}".encode(), date)
    region_key = _sign(date_key, region)
    service_key = _sign(region_key, service)
    return _sign(service_key, "aws4_request")


def canonical_query(params: tp.Mapping[str, str]) -> str:
    """Return the query string in the exact form SigV4 hashes.

    Sorted by key, and every component percent-encoded with ``/`` *not* left
    unescaped -- unlike the path. Getting either wrong produces a signature
    mismatch, which AWS reports without saying which half disagreed.
    """

    return "&".join(f"{quote(key, safe='-_.~')}={quote(params[key], safe='-_.~')}" for key in sorted(params))


def canonical_request(
    *,
    method: str,
    path: str,
    query: str,
    headers: tp.Mapping[str, str],
    payload_sha256: str,
) -> tuple[str, str]:
    """Return the canonical request and the signed-header list it declares.

    Header names are lowercased and sorted, values are stripped, and the signed
    list must name exactly the headers included -- a header sent but unsigned is
    a header AWS ignores, and one signed but unsent is a mismatch.
    """

    ordered = sorted((name.lower(), " ".join(value.split())) for name, value in headers.items())
    signed = ";".join(name for name, _value in ordered)
    canonical_headers = "".join(f"{name}:{value}\n" for name, value in ordered)
    request = "\n".join(
        [
            method.upper(),
            path or "/",
            query,
            canonical_headers,
            signed,
            payload_sha256,
        ]
    )
    return request, signed


def authorization_header(
    credentials: AwsCredentials,
    *,
    method: str,
    path: str,
    params: tp.Mapping[str, str],
    headers: tp.Mapping[str, str],
    region: str,
    service: str,
    moment: datetime,
    payload_sha256: str = EMPTY_PAYLOAD_SHA256,
) -> dict[str, str]:
    """Return every header a signed request must carry, including the signature.

    Args:
        credentials: Access key, secret, and session token when temporary.
        method: HTTP method, uppercased by the caller or here.
        path: Absolute request path, already encoded.
        params: Query parameters, unencoded.
        headers: Headers to sign. ``host`` must be among them; the timestamp and
            payload digest are added here.
        region: AWS region, part of the credential scope.
        service: AWS service name, part of the credential scope.
        moment: Request time, timezone-aware and in UTC.
        payload_sha256: Digest of the body.

    Returns:
        The headers to send: everything passed in, plus ``X-Amz-Date``,
        ``Authorization``, and the session token when one is in use.
    """

    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    date = moment.strftime("%Y%m%d")
    # The payload digest is the last line of the canonical request, so it is
    # already signed; `x-amz-content-sha256` as a *header* is an S3 requirement
    # and buys nothing on EC2. Omitting it keeps the signed set to what AWS's
    # own worked example signs, which is what makes that example usable as a
    # check on this implementation rather than on a variation of it.
    signed_headers = {**headers, "x-amz-date": stamp}
    if credentials.session_token:
        # Signed, not merely sent. An unsigned token is rejected with an error
        # that reads like clock skew, which is a long way from the cause.
        signed_headers["x-amz-security-token"] = credentials.session_token

    request, signed = canonical_request(
        method=method,
        path=path,
        query=canonical_query(params),
        headers=signed_headers,
        payload_sha256=payload_sha256,
    )
    scope = f"{date}/{region}/{service}/aws4_request"
    to_sign = "\n".join([ALGORITHM, stamp, scope, hashlib.sha256(request.encode()).hexdigest()])
    signature = hmac.new(
        signing_key(credentials.secret_access_key, date=date, region=region, service=service),
        to_sign.encode(),
        hashlib.sha256,
    ).hexdigest()
    return {
        **signed_headers,
        "Authorization": (
            f"{ALGORITHM} Credential={credentials.access_key_id}/{scope}, SignedHeaders={signed}, Signature={signature}"
        ),
    }


__all__ = (
    "ALGORITHM",
    "EMPTY_PAYLOAD_SHA256",
    "AwsCredentials",
    "authorization_header",
    "canonical_query",
    "canonical_request",
    "signing_key",
)
