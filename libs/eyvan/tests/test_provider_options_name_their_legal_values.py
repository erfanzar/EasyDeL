# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A rejected ``provider_options`` key or value has to name what it would accept.

Standing up a head node on GCE hit this three times in one afternoon. The
validators were right every time and the messages were unfinishable: "unknown
TPU provider option: bootstrap-egress" does not say which keys exist, and
"requires provider_options.bootstrap_egress" does not say that exactly three
values are legal -- a set that, at the time, appeared in no documentation
either. Each round trip cost a read through the source of a package the
operator had just installed.

So these tests do not check that a sentence contains a substring. They read the
names *out of* the message and feed them back through the same validator, which
makes the message's promise a property the code has to keep: everything it
offers must actually be accepted, and every legal value must actually be
offered.
"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from eyvan.capacity.fleet_config import EyvanFleetConfig
from eyvan.capacity.providers.gcp.compute_instances import normalize_compute_config
from eyvan.capacity.providers.gcp.queued_resources import (
    BOOTSTRAP_EGRESS_MODES,
    normalize_tpu_config,
)
from eyvan.sdk.commands.autoscale import _EGRESS_CHOICES

_KEY = re.compile(r"[a-z][a-z0-9_]*")


def _offered_keys(message: str) -> list[str]:
    """Return the accepted keys a rejection message lists after 'accepted keys are'."""

    _, separator, listed = message.partition("accepted keys are ")
    assert separator, message
    return _KEY.findall(listed)


def _offered_egress(message: str) -> list[str]:
    """Return the bootstrap_egress values a rejection message offers."""

    _, separator, listed = message.partition(": ")
    assert separator, message
    return re.findall(r"[a-z]+(?:-[a-z]+)+", listed)


def _native_compute_fleet() -> dict:
    """One minimal eyvan-native Compute fleet, missing only its egress choice."""

    return {
        "version": 1,
        "name": "research",
        "provider": {"type": "gcp", "project": "gcp-project", "region": "us-central1", "zone": "us-central1-a"},
        "agent_endpoint": "https://agent.example.com",
        "executor": "eyvan-native",
        "node_types": {
            "cpu": {
                "kind": "cpu",
                "machine_type": "n2-standard-8",
                "resources": {"cpu": 8, "memory_gib": 32},
            }
        },
    }


def test_an_unknown_tpu_option_lists_keys_that_are_all_really_accepted() -> None:
    """Every key the message offers must survive the validator that printed it."""

    with pytest.raises(ValueError) as rejected:
        normalize_tpu_config({"provider_options": {"bootstrap-egress": "cloud-nat"}})

    message = str(rejected.value)
    offered = _offered_keys(message)

    assert "unknown TPU provider option: bootstrap-egress" in message
    assert "bootstrap_egress" in offered
    assert "runtime_version" in offered
    assert not [key for key in offered if key not in _TPU_SAMPLES], "add a sample for every offered TPU option"
    for key in offered:
        normalize_tpu_config({"provider_options": {key: _TPU_SAMPLES[key]}})


def test_an_unknown_compute_option_lists_keys_that_are_all_really_accepted() -> None:
    """The Compute path had the same gap and the same one-line fix."""

    with pytest.raises(ValueError) as rejected:
        normalize_compute_config({"provider_options": {"machine-type": "n2-standard-8"}})

    message = str(rejected.value)
    offered = _offered_keys(message)

    assert "unknown Compute provider option: machine-type" in message
    assert "bootstrap_egress" in offered
    assert "machine_type" in offered
    assert not [key for key in offered if key not in _COMPUTE_SAMPLES], "add a sample for every offered Compute option"
    for key in offered:
        normalize_compute_config({"machine_type": "n2-standard-8", "provider_options": {key: _COMPUTE_SAMPLES[key]}})


_TPU_SAMPLES: dict[str, object] = {
    "bootstrap_egress": "cloud-nat",
    "internal_ips": True,
    "metadata": {"team": "research"},
    "network": "default",
    "node_count": 1,
    "node_prefix": None,
    "runtime_version": "v2-alpha-tpuv6e",
    "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
    "service_account": "eyvan@example.iam.gserviceaccount.com",
    "startup_script": "#!/bin/sh\ntrue\n",
    "subnetwork": "default",
    "tags": ["eyvan"],
}

_COMPUTE_SAMPLES: dict[str, object] = {
    "bootstrap_egress": "cloud-nat",
    "boot_disk_size_gb": 100,
    "boot_disk_type": "pd-balanced",
    "gpu_count": 0,
    "gpu_type": None,
    "image": None,
    "image_family": None,
    "image_project": None,
    "internal_ip_only": True,
    "machine_type": "n2-standard-8",
    "maintenance_policy": "TERMINATE",
    "metadata": {"team": "research"},
    "network": "default",
    "reservation": None,
    "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
    "service_account": "eyvan@example.iam.gserviceaccount.com",
    "startup_script": "#!/bin/sh\ntrue\n",
    "subnetwork": "default",
    "tags": ["eyvan"],
    "tpu_count": 0,
    "tpu_type": None,
}


@pytest.mark.parametrize(
    ("normalize", "desired", "label"),
    (
        (normalize_tpu_config, {}, "TPU"),
        (normalize_compute_config, {"machine_type": "n2-standard-8"}, "Compute"),
    ),
)
def test_a_rejected_egress_value_offers_every_value_that_works(normalize, desired: dict, label: str) -> None:
    """The three legal modes are not guessable, so refusing without them is a dead end."""

    with pytest.raises(ValueError) as rejected:
        normalize({**desired, "provider_options": {**desired, "bootstrap_egress": "internet"}})

    offered = _offered_egress(str(rejected.value))

    assert f"{label} option 'bootstrap_egress' must be one of" in str(rejected.value)
    assert offered == sorted(BOOTSTRAP_EGRESS_MODES)
    for value in offered:
        normalized = normalize({**desired, "provider_options": {**desired, "bootstrap_egress": value}})
        assert normalized["bootstrap_egress"] == value


def test_a_fleet_missing_its_egress_choice_is_told_which_values_exist() -> None:
    """This is the message an operator meets first, before any provider call."""

    with pytest.raises(ValidationError) as rejected:
        EyvanFleetConfig.model_validate(_native_compute_fleet())

    message = str(rejected.value)
    offered = re.findall(r"[a-z]+(?:-[a-z]+)+", message.partition("(one of: ")[2].partition(")")[0])

    assert "provider_options.bootstrap_egress" in message
    assert offered == sorted(BOOTSTRAP_EGRESS_MODES)
    for value in offered:
        raw = _native_compute_fleet()
        raw["node_types"]["cpu"]["provider_options"] = {"bootstrap_egress": value}
        pool = EyvanFleetConfig.model_validate(raw).pool_payloads()[0]
        assert pool["provider_options"]["bootstrap_egress"] == value


def test_the_interactive_setup_offers_exactly_the_values_the_validators_accept() -> None:
    """A menu that drifts from the validator hands the operator a rejected fleet."""

    assert {choice[1] for choice in _EGRESS_CHOICES} == set(BOOTSTRAP_EGRESS_MODES)
