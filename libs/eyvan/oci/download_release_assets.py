#!/usr/bin/env python3
# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Download only the bounded, authenticated asset set of an Eyvan release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NoReturn

RECEIPT_NAME = "eyvan-oci-release-receipt-v1.json"
RECEIPT_BUNDLE_NAME = "eyvan-oci-release-receipt-v1.bundle.json"
ASSET_MANIFEST_SCHEMA = "eyvan.oci.release-assets.v1"
ARTIFACT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")
REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
TAG_RE = re.compile(r"eyvan-v[0-9]+(?:\.[0-9A-Za-z]+)+(?:[-+][0-9A-Za-z.-]+)?\Z")
HEX_RE = re.compile(r"[0-9a-f]{64}\Z")
FULL_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
MAX_RELEASE_ASSETS = 36
MAX_RELEASE_BYTES = 1024 * 1024 * 1024
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_API_DOCUMENT_BYTES = 2 * 1024 * 1024
BOOTSTRAP_LIMITS = {RECEIPT_NAME: 1024 * 1024, RECEIPT_BUNDLE_NAME: 4 * 1024 * 1024}


class DownloadError(RuntimeError):
    """Raised when release metadata or an asset violates the closed policy."""


def _fail(message: str) -> NoReturn:
    raise DownloadError(message)


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _content_length(response: Any, *, maximum: int, expected: int | None = None) -> None:
    values = response.headers.get_all("Content-Length")
    if values is None:
        value = response.headers.get("Content-Length")
        values = [] if value is None else [value]
    if not values:
        return
    if len(values) != 1 or not values[0].isdecimal():
        _fail(f"response has an invalid Content-Length: {values!r}")
    length = int(values[0])
    if length > maximum:
        _fail(f"response Content-Length {length} exceeds the {maximum}-byte hard limit")
    if expected is not None and length != expected:
        _fail(f"response Content-Length {length} does not match GitHub asset size {expected}")


def _read_bounded(response: Any, *, maximum: int) -> bytes:
    _content_length(response, maximum=maximum)
    chunks: list[bytes] = []
    received = 0
    while chunk := response.read(min(1024 * 1024, maximum - received + 1)):
        received += len(chunk)
        if received > maximum:
            _fail(f"response exceeded the {maximum}-byte hard limit")
        chunks.append(chunk)
    return b"".join(chunks)


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow HTTPS asset redirects without forwarding the GitHub token cross-origin."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> urllib.request.Request | None:
        redirected = super().redirect_request(request, file_pointer, code, message, headers, new_url)
        if redirected is None:
            return None
        old = urllib.parse.urlparse(request.full_url)
        new = urllib.parse.urlparse(new_url)
        if new.scheme != "https" or not new.netloc or new.username or new.password:
            _fail(f"asset redirect is not credential-free HTTPS: {new_url!r}")
        if (old.scheme, old.netloc) != (new.scheme, new.netloc):
            redirected.remove_header("Authorization")
        return redirected


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_SafeRedirectHandler())


def _request(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    token: str,
    accept: str,
) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "Accept-Encoding": "identity",
            "Authorization": f"Bearer {token}",
            "User-Agent": "eyvan-qualified-assets/1",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    return opener.open(request, timeout=120)


def _release_metadata(
    opener: urllib.request.OpenerDirector,
    *,
    repository: str,
    tag: str,
    token: str,
    expected_target: str,
) -> dict[str, dict[str, Any]]:
    encoded_tag = urllib.parse.quote(tag, safe="")
    url = f"https://api.github.com/repos/{repository}/releases/tags/{encoded_tag}"
    with _request(opener, url, token=token, accept="application/vnd.github+json") as response:
        if response.status != 200:
            _fail(f"GitHub release lookup returned HTTP {response.status}")
        raw = _read_bounded(response, maximum=MAX_API_DOCUMENT_BYTES)
    try:
        release = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=_pairs_no_duplicates)
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail(f"GitHub release metadata is not strict JSON: {error}")
    if not isinstance(release, dict):
        _fail("GitHub release metadata must be an object")
    if release.get("tag_name") != tag or release.get("target_commitish") != expected_target:
        _fail("GitHub release does not bind the exact tag and source SHA")
    if release.get("draft") is not False or release.get("prerelease") is not False:
        _fail("GitHub release must be a published, non-prerelease release")
    if release.get("immutable") is not True:
        _fail("GitHub release must be provider-reported immutable")
    raw_assets = release.get("assets")
    if not isinstance(raw_assets, list) or len(raw_assets) != MAX_RELEASE_ASSETS:
        _fail(f"GitHub release must contain exactly {MAX_RELEASE_ASSETS} assets")
    assets: dict[str, dict[str, Any]] = {}
    total = 0
    for index, value in enumerate(raw_assets):
        if not isinstance(value, dict):
            _fail(f"GitHub release asset {index} must be an object")
        name = value.get("name")
        asset_id = value.get("id")
        size = value.get("size")
        if not isinstance(name, str) or not ARTIFACT_NAME_RE.fullmatch(name) or name in {".", ".."} or name in assets:
            _fail(f"GitHub release asset {index} has an invalid or duplicate plain filename")
        if isinstance(asset_id, bool) or not isinstance(asset_id, int) or asset_id <= 0:
            _fail(f"GitHub release asset {name!r} has an invalid ID")
        if value.get("state") != "uploaded":
            _fail(f"GitHub release asset {name!r} is not fully uploaded")
        if isinstance(size, bool) or not isinstance(size, int) or not (1 <= size <= MAX_ARTIFACT_BYTES):
            _fail(f"GitHub release asset {name!r} has an invalid or excessive byte size")
        expected_url = f"https://api.github.com/repos/{repository}/releases/assets/{asset_id}"
        if value.get("url") != expected_url:
            _fail(f"GitHub release asset {name!r} has an unexpected API URL")
        total += size
        if total > MAX_RELEASE_BYTES:
            _fail(f"GitHub release assets exceed the {MAX_RELEASE_BYTES}-byte aggregate hard limit")
        assets[name] = {"id": asset_id, "name": name, "size": size, "url": expected_url}
    return assets


def _manifest_records(path: Path) -> dict[str, str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=_pairs_no_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail(f"cannot read strict asset manifest: {error}")
    if raw != _canonical_bytes(value) or not isinstance(value, dict) or set(value) != {"artifacts", "schemaVersion"}:
        _fail("asset manifest must be canonical closed-schema JSON")
    if value["schemaVersion"] != ASSET_MANIFEST_SCHEMA or not isinstance(value["artifacts"], list):
        _fail("asset manifest has the wrong schema")
    if len(value["artifacts"]) != MAX_RELEASE_ASSETS - 2:
        _fail("asset manifest must contain exactly 34 artifacts")
    records: dict[str, str] = {}
    for index, record in enumerate(value["artifacts"]):
        if not isinstance(record, dict) or set(record) != {"name", "sha256"}:
            _fail(f"asset manifest record {index} has the wrong fields")
        name = record["name"]
        digest = record["sha256"]
        if (
            not isinstance(name, str)
            or not ARTIFACT_NAME_RE.fullmatch(name)
            or name in {".", "..", RECEIPT_NAME, RECEIPT_BUNDLE_NAME}
            or name in records
        ):
            _fail(f"asset manifest record {index} has an invalid or duplicate filename")
        if not isinstance(digest, str) or not HEX_RE.fullmatch(digest):
            _fail(f"asset manifest record {index} has an invalid SHA-256")
        records[name] = digest
    return records


def _download_asset(
    opener: urllib.request.OpenerDirector,
    asset: dict[str, Any],
    *,
    destination: Path,
    token: str,
    hard_limit: int,
    expected_sha256: str | None,
) -> None:
    size = asset["size"]
    if size > hard_limit:
        _fail(f"asset {asset['name']!r} exceeds its {hard_limit}-byte hard limit")
    temporary = destination / f".{asset['name']}.part"
    target = destination / asset["name"]
    if target.exists() or target.is_symlink() or temporary.exists() or temporary.is_symlink():
        _fail(f"refusing to replace existing asset path {target}")
    digest = hashlib.sha256()
    received = 0
    try:
        with _request(
            opener,
            asset["url"],
            token=token,
            accept="application/octet-stream",
        ) as response:
            if response.status != 200:
                _fail(f"asset {asset['name']!r} returned HTTP {response.status}")
            content_encoding = response.headers.get("Content-Encoding")
            if content_encoding not in {None, "identity"}:
                _fail(f"asset {asset['name']!r} used unexpected Content-Encoding {content_encoding!r}")
            _content_length(response, maximum=hard_limit, expected=size)
            with temporary.open("xb") as output:
                while chunk := response.read(min(1024 * 1024, size - received + 1)):
                    received += len(chunk)
                    if received > size or received > hard_limit:
                        _fail(f"asset {asset['name']!r} exceeded its exact or hard byte limit")
                    output.write(chunk)
                    digest.update(chunk)
        if received != size:
            _fail(f"asset {asset['name']!r} size mismatch: {received} != {size}")
        actual_sha256 = digest.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            _fail(f"asset {asset['name']!r} SHA-256 mismatch")
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validated_destination(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        _fail("destination must be a real directory, not a symlink")
    resolved = path.resolve(strict=True)
    if any(resolved.iterdir()):
        _fail("destination must be empty before a bounded release download")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expected-target", required=True)
    parser.add_argument("--token-env", default="GH_TOKEN")
    parser.add_argument("--destination", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--receipt-sha256")
    mode.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    if not REPOSITORY_RE.fullmatch(args.repository):
        raise SystemExit("--repository must be an exact owner/name")
    if not TAG_RE.fullmatch(args.tag):
        raise SystemExit("--tag must be an exact eyvan-v<version> tag")
    if not FULL_SHA_RE.fullmatch(args.expected_target):
        raise SystemExit("--expected-target must be a full lowercase Git SHA")
    token = os.environ.get(args.token_env, "")
    if not token or any(character.isspace() for character in token):
        raise SystemExit(f"{args.token_env} must contain one non-empty GitHub token")

    try:
        destination = _validated_destination(args.destination)
        opener = _opener()
        assets = _release_metadata(
            opener,
            repository=args.repository,
            tag=args.tag,
            token=token,
            expected_target=args.expected_target,
        )
        if args.receipt_sha256 is not None:
            if not HEX_RE.fullmatch(args.receipt_sha256):
                _fail("--receipt-sha256 must be 64 lowercase hexadecimal characters")
            expected = {RECEIPT_NAME: args.receipt_sha256, RECEIPT_BUNDLE_NAME: None}
            for name, expected_sha256 in expected.items():
                if name not in assets:
                    _fail(f"GitHub release lacks fixed bootstrap asset {name!r}")
                _download_asset(
                    opener,
                    assets[name],
                    destination=destination,
                    token=token,
                    hard_limit=BOOTSTRAP_LIMITS[name],
                    expected_sha256=expected_sha256,
                )
            print("downloaded the fixed receipt and detached bundle within hard byte limits")
        else:
            records = _manifest_records(args.manifest)
            expected_names = {*records, RECEIPT_NAME, RECEIPT_BUNDLE_NAME}
            if set(assets) != expected_names:
                missing = sorted(expected_names - set(assets))
                unexpected = sorted(set(assets) - expected_names)
                _fail(f"GitHub release asset set is not closed: missing={missing}, unexpected={unexpected}")
            for name, expected_sha256 in records.items():
                _download_asset(
                    opener,
                    assets[name],
                    destination=destination,
                    token=token,
                    hard_limit=MAX_ARTIFACT_BYTES,
                    expected_sha256=expected_sha256,
                )
            print("downloaded exactly 34 authenticated receipt artifacts within hard byte limits")
    except (DownloadError, OSError, urllib.error.URLError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
