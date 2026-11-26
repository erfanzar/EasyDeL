# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Signature Version 4, checked against AWS's own worked example.

The one part of an AWS provider that can be *proved* without an AWS account.
AWS publishes a specific request, credential and timestamp together with the
canonical request, string-to-sign and signature they must produce; this checks
the implementation against that arithmetic rather than against another copy of
itself.

The rest of the provider is verified against a fake EC2, which proves fencing
and idempotency and nothing about whether AWS would accept the bytes. This is
the half that can be certain, so it is worth being certain about: a signature
mismatch is reported by AWS without saying which half of the canonical request
disagreed, and diagnosing that against a live account is an afternoon.
"""

from __future__ import annotations

import hashlib
import typing as tp
from datetime import UTC, datetime
from urllib.parse import urlencode

import httpx

from eyvan.capacity.providers.aws.api import API_VERSION, HttpEc2Api
from eyvan.capacity.providers.aws.signing import (
    AwsCredentials,
    authorization_header,
    canonical_query,
    canonical_request,
    signing_key,
)

# AWS's documented example credential. Not a secret: it is published in the
# SigV4 test suite precisely so implementations can check themselves.
EXAMPLE = AwsCredentials(
    access_key_id="AKIDEXAMPLE",
    secret_access_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
)
MOMENT = datetime(2015, 8, 30, 12, 36, 0, tzinfo=UTC)


def test_the_derived_key_matches_the_documented_one() -> None:
    """The four-step derivation, which is why a leaked key is worth one day.

    AWS documents the intermediate for us-east-1/iam on 2015-08-30.
    """

    key = signing_key(EXAMPLE.secret_access_key, date="20150830", region="us-east-1", service="iam")

    assert key.hex() == "c4afb1cc5771d871763a393e44b703571b55cc28424d1a5e86da6ed3c154a4b9"


def test_the_canonical_request_matches_the_documented_one() -> None:
    """AWS's `get-vanilla` case, byte for byte.

    Every part of this is a place an implementation goes wrong silently: the
    empty query line, the trailing newline after the headers, the signed-header
    list, the digest of an empty body.
    """

    request, signed = canonical_request(
        method="GET",
        path="/",
        query="",
        headers={"Host": "example.amazonaws.com", "X-Amz-Date": "20150830T123600Z"},
        payload_sha256=hashlib.sha256(b"").hexdigest(),
    )

    assert signed == "host;x-amz-date"
    assert request == (
        "GET\n"
        "/\n"
        "\n"
        "host:example.amazonaws.com\n"
        "x-amz-date:20150830T123600Z\n"
        "\n"
        "host;x-amz-date\n"
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_the_signature_matches_the_documented_one() -> None:
    """The end-to-end check: AWS's `get-vanilla` signature for `service`."""

    headers = authorization_header(
        EXAMPLE,
        method="GET",
        path="/",
        params={},
        headers={"host": "example.amazonaws.com"},
        region="us-east-1",
        service="service",
        moment=MOMENT,
    )

    assert "Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31" in headers["Authorization"]
    assert "Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request" in headers["Authorization"]


def test_query_parameters_are_sorted_and_encoded_the_way_the_signature_expects() -> None:
    """Unsorted or under-encoded, and AWS rejects it without saying which."""

    assert canonical_query({"b": "2", "a": "1"}) == "a=1&b=2"
    # `/` is escaped in a query component, unlike in a path. Getting that
    # backwards is the most common way a hand-written signer fails.
    assert canonical_query({"Filter.1.Value": "a/b"}) == "Filter.1.Value=a%2Fb"
    assert canonical_query({"k": "a b"}) == "k=a%20b"


def test_a_session_token_is_signed_rather_than_only_sent() -> None:
    """An unsigned token is refused in a way that reads like clock skew.

    Temporary credentials are how a control plane on EC2 or EKS ordinarily
    authenticates, so this is the common path and not the exotic one.
    """

    headers = authorization_header(
        AwsCredentials("AKID", "secret", session_token="token-value"),
        method="GET",
        path="/",
        params={"Action": "DescribeInstances"},
        headers={"host": "ec2.us-east-1.amazonaws.com"},
        region="us-east-1",
        service="ec2",
        moment=MOMENT,
    )

    assert headers["x-amz-security-token"] == "token-value"
    assert "x-amz-security-token" in headers["Authorization"]


def test_header_values_are_normalized_before_signing() -> None:
    """Interior whitespace is collapsed, which AWS also does before verifying."""

    request, _signed = canonical_request(
        method="POST",
        path="/",
        query="",
        headers={"Host": "example.amazonaws.com", "X-Amz-Meta": "  a   b  "},
        payload_sha256="0" * 64,
    )

    assert "x-amz-meta:a b\n" in request


# --------------------------------------------------------------------------- signing the bytes that get sent


class _XmlResponse:
    """The two things the EC2 client reads off a response."""

    def __init__(self, text: str) -> None:
        self.status_code = 200
        self.text = text


class _CapturingClient:
    """The httpx seam HttpEc2Api posts through, holding the encoded body."""

    def __init__(self) -> None:
        self.url: str | None = None
        self.body: str | None = None
        self.headers: dict[str, str] | None = None

    def post(self, url: str, content: str | None = None, headers: tp.Mapping[str, str] | None = None) -> _XmlResponse:
        self.url = url
        self.body = content
        self.headers = dict(headers or {})
        return _XmlResponse('<DescribeInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2016-11-15/"/>')


def test_the_payload_digest_signs_the_bytes_actually_posted() -> None:
    """AWS hashes the body it receives; signing a different encoding fails every call.

    The filter names arrive in an insertion order that differs from sorted
    order on purpose -- that drift is the whole defect -- and the value carries
    a space, where two urlencoders disagree even at equal order.
    """

    client = _CapturingClient()
    api = HttpEc2Api(region="us-east-1", credentials=EXAMPLE, client=client)

    api.describe_instances({"tag:eyvan:cell": "cell a"})

    posted = client.body
    assert posted == urlencode(
        sorted(
            {
                "Filter.1.Name": "tag:eyvan:cell",
                "Filter.1.Value.1": "cell a",
                "Action": "DescribeInstances",
                "Version": API_VERSION,
            }.items()
        )
    )

    # Re-derive the signature the way AWS does, from the exact bytes received:
    # a digest computed over any other encoding cannot reproduce it.
    moment = datetime.strptime(client.headers["x-amz-date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    recomputed = authorization_header(
        EXAMPLE,
        method="POST",
        path="/",
        params={},
        headers={name: client.headers[name] for name in ("host", "content-type")},
        region="us-east-1",
        service="ec2",
        moment=moment,
        payload_sha256=hashlib.sha256(posted.encode("utf-8")).hexdigest(),
    )
    assert client.headers == recomputed, "every header sent is signed, and over the posted bytes"


def test_httpx_puts_a_string_body_on_the_wire_byte_for_byte() -> None:
    """The wire half of the guarantee: httpx forwards a string body unmodified.

    It also adds no content-type of its own -- the signed one is sent
    explicitly, and a silently different one would be an unsigned header.
    """

    body = urlencode({"Action": "DescribeInstances", "Version": API_VERSION})

    request = httpx.Request("POST", "https://ec2.us-east-1.amazonaws.com/", content=body)

    assert request.content == body.encode("utf-8")
    assert request.headers.get("content-type") is None
