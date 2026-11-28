# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
from pydantic import ValidationError

from eyvan.capacity.fleet_config import EyvanFleetConfig


def _fleet(options, backend="gcp-compute"):
    return {
        "name": "network-test",
        "provider": {"type": "gcp", "project": "test-project", "region": "us-east5", "zone": "us-east5-a"},
        "capacity_class": "spot",
        "pools": {
            "v5p-8": {
                "capacity_backend": backend,
                "min_workers": 0,
                "max_workers": 1,
                "provider_options": options,
            }
        },
    }


@pytest.mark.parametrize("egress", ["public-internet", "PUBLIC_INTERNET"])
def test_a_private_compute_fleet_rejects_direct_public_bootstrap_before_apply(egress):
    options = {"bootstrap_egress": egress, "internal_ip_only": True}
    with pytest.raises(ValidationError, match="internal_ip_only"):
        EyvanFleetConfig.model_validate(_fleet(options))


def test_a_compute_fleet_accepts_explicit_public_address_configuration():
    config = EyvanFleetConfig.model_validate(_fleet({"bootstrap_egress": "public-internet", "internal_ip_only": False}))
    assert config.pool_payloads()[0]["provider_options"]["internal_ip_only"] is False


@pytest.mark.parametrize("egress", ["cloud-nat", "private-google-access"])
def test_a_private_compute_fleet_preserves_its_explicit_egress_route(egress):
    config = EyvanFleetConfig.model_validate(_fleet({"bootstrap_egress": egress, "internal_ip_only": True}))
    assert config.pool_payloads()[0]["provider_options"]["internal_ip_only"] is True


def test_a_private_legacy_tpu_fleet_rejects_direct_public_bootstrap():
    with pytest.raises(ValidationError, match="internal_ips"):
        EyvanFleetConfig.model_validate(_fleet({"bootstrap_egress": "public-internet", "internal_ips": True}, "gcp-tpu"))


def test_a_legacy_tpu_fleet_keeps_its_existing_public_address_default():
    config = EyvanFleetConfig.model_validate(_fleet({"bootstrap_egress": "public-internet"}, "gcp-tpu"))
    assert config.pool_payloads()[0]["capacity_class"] == "spot"
