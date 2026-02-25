from __future__ import annotations

from pydantic import BaseModel

from easydel.inference.oai_proxies import InferenceApiRouter


class _Meta(BaseModel):
    key: str
    count: int


def test_format_metadata_sse_uses_standard_event_framing():
    payload = InferenceApiRouter._format_metadata_sse(_Meta(key="value", count=2)).decode()

    assert payload.startswith("event: metadata\ndata: ")
    assert payload.endswith("\n\n")
    assert '"key":"value"' in payload
    assert '"count":2' in payload
