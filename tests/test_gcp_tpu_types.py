# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import pytest

from eyvan.capacity.providers.gcp.tpu_types import accelerator_generation, parse_tpu_accelerator_type


def test_parser_returns_generation_and_public_slice_size_for_valid_types() -> None:
    parsed = parse_tpu_accelerator_type("v5litepod-8")

    assert parsed is not None
    assert parsed.accelerator_type == "v5litepod-8"
    assert parsed.generation == "v5litepod"
    assert parsed.size == 8
    assert accelerator_generation("v5p-512") == "v5p"


def test_parser_rejects_noncanonical_and_incomplete_accelerator_types() -> None:
    assert parse_tpu_accelerator_type("V5P-16") is None
    assert parse_tpu_accelerator_type("v5p-0") is None
    assert parse_tpu_accelerator_type("v5p") is None
    assert accelerator_generation("not-a-tpu") is None


def _qr(raw: dict):
    from eyvan.capacity.providers.gcp.queued_resources import GcpQueuedResourceState, QueuedResource

    return QueuedResource(
        qr_id=raw.get("name", "qr-1"),
        state=GcpQueuedResourceState.ACTIVE,
        node_ids=(),
        accelerator_type=None,
        raw=raw,
    )


def _spec_labels(uid: str = "uid-1", cell: str = "cell-a"):
    from eyvan.capacity.providers.gcp.queued_resources import QueuedResourceSpec

    return QueuedResourceSpec(
        name="cell-a",
        accelerator_type="v5p-16",
        zone="us-east5-a",
        project="p",
        capacity_class="spot",
        labels={"eyvan-uid": uid, "eyvan-cell": cell},
        runtime_version="v2-alpha-tpuv5",
    )


def test_an_unlabelled_active_queued_resource_is_not_declared_foreign() -> None:
    """Use the fenced recorded QR ID as ownership evidence when Cloud TPU omits labels and metadata."""

    from eyvan.capacity.providers.gcp.queued_resources import GcloudTpuProvider

    bare = _qr({"name": "qr-1", "tpu": {"nodeSpec": [{"node": {"acceleratorType": "v5p-16"}}]}})

    GcloudTpuProvider._validate_qr(bare, _spec_labels(), recorded_qr_id="qr-1")


def test_an_unlabelled_queued_resource_we_never_recorded_is_still_refused() -> None:
    """The fallback is our own record, not "assume everything is ours"."""

    from eyvan.capacity.providers.gcp.queued_resources import GcloudError, GcloudTpuProvider

    bare = _qr({"name": "qr-1", "tpu": {"nodeSpec": [{"node": {}}]}})

    with pytest.raises(GcloudError, match="not owned"):
        GcloudTpuProvider._validate_qr(bare, _spec_labels(), recorded_qr_id="a-different-qr")


def test_a_labelled_foreign_queued_resource_is_still_refused() -> None:
    """When the provider does answer, its answer stays authoritative."""

    from eyvan.capacity.providers.gcp.queued_resources import GcloudError, GcloudTpuProvider

    foreign = _qr({"name": "qr-1", "labels": {"eyvan-uid": "someone-else", "eyvan-cell": "other"}})

    with pytest.raises(GcloudError, match="not owned"):
        GcloudTpuProvider._validate_qr(foreign, _spec_labels(), recorded_qr_id="qr-1")


def test_a_labelled_matching_queued_resource_is_accepted() -> None:
    """The path that already worked must keep working."""

    from eyvan.capacity.providers.gcp.queued_resources import GcloudTpuProvider

    ours = _qr({"name": "qr-1", "labels": {"eyvan-uid": "uid-1", "eyvan-cell": "cell-a"}})

    GcloudTpuProvider._validate_qr(ours, _spec_labels(), recorded_qr_id="qr-1")


def test_an_active_queued_resource_can_still_be_released(monkeypatch) -> None:
    """Why ACTIVE v5p-128 queued resources survived for days doing nothing.

    Cloud TPU stops echoing labels and metadata once a queued resource is
    provisioned, and ``delete`` required them. Cells released while still
    WAITING went away cleanly; every one that reached ACTIVE became undeletable
    *by Eyvan* at the exact moment it became usable -- and therefore billable.

    Refusing there protects nothing. The caller is deleting an id recorded on
    its own fenced cell record.
    """

    import subprocess

    from eyvan.capacity.providers.gcp import queued_resources as module

    provider = module.GcloudTpuProvider()
    bare = _qr({"name": "qr-1", "tpu": {"nodeSpec": [{"node": {"acceleratorType": "v5p-16"}}]}})
    # Present for the ownership check, gone once the delete has been issued --
    # otherwise the removal poll waits out its full timeout.
    seen: list[int] = []

    def _describe(self, *args, **kwargs):
        seen.append(1)
        return bare if len(seen) == 1 else None

    issued: list[tuple[str, ...]] = []

    def _record(self, args):
        issued.append(tuple(args))
        return subprocess.CompletedProcess(list(args), 0, "", "")

    monkeypatch.setattr(module.GcloudTpuProvider, "describe_qr", _describe)
    monkeypatch.setattr(module.GcloudTpuProvider, "_run", _record)

    provider.delete("qr-1", project="p", zone="z", management_uid="uid-1")

    assert any("delete" in args for args in issued), "an ACTIVE resource must still be releasable"


def test_a_labelled_foreign_queued_resource_is_never_deleted(monkeypatch) -> None:
    """When the provider does answer, a mismatch still blocks deletion."""

    from eyvan.capacity.providers.gcp import queued_resources as module

    provider = module.GcloudTpuProvider()
    foreign = _qr({"name": "qr-1", "labels": {"eyvan-uid": "someone-else"}})
    monkeypatch.setattr(module.GcloudTpuProvider, "describe_qr", lambda *a, **k: foreign)
    monkeypatch.setattr(
        module.GcloudTpuProvider, "_run", lambda self, args: (_ for _ in ()).throw(AssertionError("must not delete"))
    )

    with pytest.raises(module.GcloudError, match="refusing to delete"):
        provider.delete("qr-1", project="p", zone="z", management_uid="uid-1")


def test_deleting_without_an_ownership_uid_is_still_refused(monkeypatch) -> None:
    """The caller must still assert which cell owns the resource."""

    from eyvan.capacity.providers.gcp import queued_resources as module

    provider = module.GcloudTpuProvider()
    bare = _qr({"name": "qr-1", "tpu": {"nodeSpec": [{"node": {}}]}})
    monkeypatch.setattr(module.GcloudTpuProvider, "describe_qr", lambda *a, **k: bare)

    with pytest.raises(module.GcloudError, match="refusing to delete"):
        provider.delete("qr-1", project="p", zone="z", management_uid=None)
