# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Keep duplicated security documentation aligned with observed runtime contracts."""

import re
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("document", "obsolete_claims"),
    (
        (
            "docs/security.md",
            (
                "signed, digest-verified inputs",
                "signed with the submitter's key context",
            ),
        ),
        (
            "docs/tutorial/01-what-is-eyvan.md",
            (
                "│ signed callable artifact + workload record",
                "dumps the bound callable, signs it",
            ),
        ),
        (
            "docs/tutorial/03-sdk.md",
            (
                "A `signing_key` is refused for remote planes",
                "pickled (cloudpickle), signed, and uploaded",
            ),
        ),
        (
            "docs/tutorial/12-security.md",
            (
                "signed, digest-verified inputs",
                "Signing keys are refused on remote planes",
            ),
        ),
    ),
)
def test_standalone_callable_docs_do_not_promise_signature_authentication(
    document: str, obsolete_claims: tuple[str, ...]
) -> None:
    root = Path(__file__).resolve().parents[1]
    text = " ".join((root / document).read_text(encoding="utf-8").split())
    for claim in obsolete_claims:
        assert re.search(rf"(?<!\w){re.escape(claim)}", text) is None, (
            f"{document} advertises unsupported standalone callable signing: {claim}"
        )
    assert "unsigned" in text
    assert "trusted project" in text
    assert "digest" in text
    if document != "docs/tutorial/01-what-is-eyvan.md":
        assert "loopback" in text
        assert "does not distribute" in text
        assert "custom" in text
