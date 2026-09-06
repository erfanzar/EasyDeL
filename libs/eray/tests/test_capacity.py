# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for eray.capacity: the pool planner, the fake-backed reconcile
loop, capacity-tier proto mapping, spec persistence, and the CLI."""

from __future__ import annotations

import eray.capacity.state as state_module
import eray.cli.capacity as cli_capacity
import pytest
from click.testing import CliRunner
from eray.capacity.fake import FakeClock, FakeQrService
from eray.capacity.gcp_qr import capacity_mode
from eray.capacity.pool import CapacityPool, PoolSpec, plan_pool
from eray.capacity.state import load_pool_specs, remove_pool_spec, save_pool_spec
from eray.capacity.types import (
    CAPACITY_POOL_LABEL,
    CapacityType,
    InfraError,
    QueuedResourceInfo,
    TpuCreateRequest,
    generate_qr_suffix,
    parse_duration_s,
    validate_labels,
    validate_resource_name,
)

try:
    import google.cloud.tpu_v2alpha1  # noqa: F401

    _HAS_GCP_EXTRA = True
except ImportError:
    _HAS_GCP_EXTRA = False

ZONE_A = "us-east5-a"
ZONE_B = "us-central2-b"
NOW = 1_700_000_000.0


def make_spec(**overrides) -> PoolSpec:
    defaults = dict(
        name="testpool",
        accelerator_type="v5p-128",
        zones=(ZONE_A, ZONE_B),
        project="test-project",
        count=2,
        buffer=0,
        zone_wait_timeout_s=3600.0,
    )
    defaults.update(overrides)
    return PoolSpec(**defaults)


def qr(name: str, state: str, zone: str = ZONE_A, *, age: float = 60.0, acc: str = "v5p-128") -> QueuedResourceInfo:
    return QueuedResourceInfo(
        name=name,
        state=state,
        zone=zone,
        labels={CAPACITY_POOL_LABEL: "testpool"},
        create_time=NOW - age,
        accelerator_type=acc,
    )


def kinds(actions) -> list[str]:
    return [a.kind for a in actions]


# ============================================================================
# Pure helpers
# ============================================================================


class TestHelpers:
    def test_capacity_mode_mapping(self):
        assert capacity_mode(CapacityType.SPOT) == "spot"
        assert capacity_mode(CapacityType.RESERVED) == "guaranteed_reserved"
        assert capacity_mode(CapacityType.GUARANTEED) == "guaranteed"
        assert capacity_mode(CapacityType.ON_DEMAND) == "none"

    def test_capacity_mode_accepts_plain_strings(self):
        assert capacity_mode("spot") == "spot"
        with pytest.raises(ValueError):
            capacity_mode("preemptible")

    def test_parse_duration(self):
        assert parse_duration_s("72h") == 72 * 3600
        assert parse_duration_s("30m") == 1800
        assert parse_duration_s("45s") == 45
        assert parse_duration_s("2d") == 2 * 86400
        assert parse_duration_s("90") == 90
        with pytest.raises(ValueError):
            parse_duration_s("soon")

    def test_qr_suffix_is_name_safe_and_unique(self):
        a, b = generate_qr_suffix(), generate_qr_suffix()
        assert a != b
        validate_resource_name(f"pool-{a}", "queued resource")

    def test_validation_rejects_bad_names_and_labels(self):
        with pytest.raises(ValueError):
            validate_resource_name("Has-Caps", "queued resource")
        with pytest.raises(ValueError):
            validate_resource_name("x" * 64, "queued resource")
        with pytest.raises(ValueError):
            validate_labels({"UPPER": "v"})
        validate_labels({"eray-capacity-pool": "ok-value_1"})

    def test_pool_spec_validation(self):
        with pytest.raises(ValueError):
            make_spec(name="x" * 60)  # no room for suffix
        with pytest.raises(ValueError):
            make_spec(zones=())
        with pytest.raises(ValueError):
            make_spec(count=-1)
        assert make_spec(count=0, buffer=3).desired_total == 3
        assert make_spec(count=2, buffer=1).desired_total == 3


# ============================================================================
# Plan truth table (pure)
# ============================================================================


class TestPlanTruthTable:
    def test_empty_cloud_creates_deficit_in_first_zone(self):
        actions = plan_pool(make_spec(count=2), [], now=NOW)
        assert kinds(actions) == ["create", "create"]
        assert all(a.zone == ZONE_A for a in actions)

    def test_desired_met_plans_nothing(self):
        observed = [qr("a", "ACTIVE"), qr("b", "WAITING_FOR_RESOURCES")]
        assert plan_pool(make_spec(count=2), observed, now=NOW) == []

    def test_suspended_is_deleted_and_replaced(self):
        observed = [qr("a", "ACTIVE"), qr("b", "SUSPENDED")]
        actions = plan_pool(make_spec(count=2), observed, now=NOW)
        assert kinds(actions) == ["delete", "create"]
        assert actions[0].name == "b"
        assert actions[0].force is True
        assert actions[0].reason == "dead:SUSPENDED"

    def test_failed_is_deleted_and_replaced(self):
        actions = plan_pool(make_spec(count=1), [qr("a", "FAILED")], now=NOW)
        assert kinds(actions) == ["delete", "create"]
        assert actions[0].reason == "dead:FAILED"

    def test_suspending_gets_replacement_but_no_delete_yet(self):
        observed = [qr("a", "ACTIVE"), qr("b", "SUSPENDING")]
        actions = plan_pool(make_spec(count=2), observed, now=NOW)
        assert kinds(actions) == ["create"]

    def test_deleting_is_ignored_and_replaced(self):
        actions = plan_pool(make_spec(count=1), [qr("a", "DELETING")], now=NOW)
        assert kinds(actions) == ["create"]

    def test_stuck_waiting_rotates_to_next_zone(self):
        observed = [qr("a", "WAITING_FOR_RESOURCES", ZONE_A, age=7200.0)]
        actions = plan_pool(make_spec(count=1), observed, now=NOW)
        assert kinds(actions) == ["delete", "create"]
        assert actions[0].reason == "zone-rotate"
        assert actions[0].force is False
        assert actions[1].zone == ZONE_B

    def test_fresh_waiting_does_not_rotate(self):
        observed = [qr("a", "WAITING_FOR_RESOURCES", ZONE_A, age=60.0)]
        assert plan_pool(make_spec(count=1), observed, now=NOW) == []

    def test_single_zone_pool_never_rotates(self):
        spec = make_spec(zones=(ZONE_A,), count=1)
        observed = [qr("a", "WAITING_FOR_RESOURCES", ZONE_A, age=10_000_000.0)]
        assert plan_pool(spec, observed, now=NOW) == []

    def test_no_rotation_when_other_zones_blocked(self):
        observed = [qr("a", "WAITING_FOR_RESOURCES", ZONE_A, age=7200.0)]
        actions = plan_pool(make_spec(count=1), observed, now=NOW, zone_blocks={ZONE_B: NOW + 100})
        assert actions == []

    def test_missing_create_time_counts_as_fresh(self):
        observed = [
            QueuedResourceInfo(name="a", state="WAITING_FOR_RESOURCES", zone=ZONE_A, create_time=None)
        ]
        assert plan_pool(make_spec(count=1), observed, now=NOW) == []

    def test_buffer_holds_capacity_at_zero_count(self):
        actions = plan_pool(make_spec(count=0, buffer=1), [], now=NOW)
        assert kinds(actions) == ["create"]

    def test_excess_sheds_pending_before_active(self):
        observed = [qr("act", "ACTIVE", age=500.0), qr("wait", "WAITING_FOR_RESOURCES", age=100.0)]
        actions = plan_pool(make_spec(count=1), observed, now=NOW)
        assert kinds(actions) == ["delete"]
        assert actions[0].name == "wait"
        assert actions[0].force is False

    def test_excess_sheds_newest_active_and_forces(self):
        observed = [qr("old", "ACTIVE", age=5000.0), qr("new", "ACTIVE", age=100.0)]
        actions = plan_pool(make_spec(count=1), observed, now=NOW)
        assert kinds(actions) == ["delete"]
        assert actions[0].name == "new"
        assert actions[0].force is True

    def test_all_zones_blocked_yields_note_not_create(self):
        blocks = {ZONE_A: NOW + 100, ZONE_B: NOW + 100}
        actions = plan_pool(make_spec(count=1), [], now=NOW, zone_blocks=blocks)
        assert kinds(actions) == ["note"]
        assert "all-zones-blocked" in actions[0].reason

    def test_type_mismatch_pending_is_shed_active_is_kept(self):
        observed = [
            qr("wrong-wait", "WAITING_FOR_RESOURCES", acc="v5p-64"),
            qr("wrong-act", "ACTIVE", acc="v5p-64"),
        ]
        actions = plan_pool(make_spec(count=1), observed, now=NOW)
        by_kind = {a.kind: a for a in actions}
        assert by_kind["delete"].name == "wrong-wait"
        assert by_kind["note"].name == "wrong-act"
        assert any(a.kind == "create" for a in actions)  # neither counts toward desired

    def test_rotation_replacement_prefers_zone_after_stuck_one(self):
        spec = make_spec(zones=(ZONE_A, ZONE_B, "us-west4-a"), count=1)
        observed = [qr("a", "WAITING_FOR_RESOURCES", ZONE_B, age=7200.0)]
        actions = plan_pool(spec, observed, now=NOW)
        create = next(a for a in actions if a.kind == "create")
        assert create.zone == "us-west4-a"


# ============================================================================
# Fake-backed reconcile (integration)
# ============================================================================


def make_pool(spec: PoolSpec | None = None, **spec_overrides):
    spec = spec or make_spec(**spec_overrides)
    clock = FakeClock(NOW)
    service = FakeQrService(project_id=spec.project, clock=clock)
    pool = CapacityPool(service, spec, clock=clock)
    return pool, service, clock


class TestReconcile:
    def test_provision_from_empty_then_activate(self):
        pool, service, _ = make_pool(count=2)
        report = pool.reconcile()
        assert len(report.executed) == 2
        assert len(service.qr_names()) == 2
        assert all(state == "WAITING_FOR_RESOURCES" for state in service.states_by_name().values())
        service.activate_pending()
        report = pool.reconcile()
        assert report.active == 2
        assert report.actions == []

    def test_suspended_is_reclaimed_end_to_end(self):
        pool, service, _ = make_pool(count=2)
        pool.reconcile()
        service.activate_pending()
        victim = sorted(service.qr_names())[0]
        service.set_state(victim, ZONE_A, "SUSPENDED")

        report = pool.reconcile()
        deleted = [d for d in service.deletes if d[0] == victim]
        assert deleted and deleted[0][2] is True  # force delete
        assert victim not in service.qr_names()
        assert len(service.qr_names()) == 2  # replacement requested immediately
        assert not report.errors

    def test_kill_test_missing_qr_is_recreated(self):
        # A QR deleted out from under the pool must be replaced within
        # one evaluation interval.
        pool, service, _ = make_pool(count=1)
        pool.reconcile()
        (name,) = service.qr_names()
        service.queued_resource_delete(name, ZONE_A)  # external force-delete
        report = pool.reconcile()
        assert len(service.qr_names()) == 1
        assert next(iter(service.qr_names())) != name
        assert not report.errors

    def test_quota_stockout_falls_through_to_next_zone(self):
        pool, service, _ = make_pool(count=1)
        service.stockout_zones.add(ZONE_A)
        report = pool.reconcile()
        assert len(service.qr_names()) == 1
        created = service.creates[-1]
        assert created.zone == ZONE_B
        assert pool.zone_blocks[ZONE_A] > NOW
        # Fallthrough that landed is the feature working: a warning, not an
        # error — cron passes exercising it must exit 0.
        assert any("exhausted" in w for w in report.warnings)
        assert report.errors == []

    def test_second_create_skips_freshly_blocked_zone(self):
        pool, service, _ = make_pool(count=2)
        service.stockout_zones.add(ZONE_A)
        pool.reconcile()
        assert len(service.qr_names()) == 2
        assert all(req.zone == ZONE_B for req in service.creates if req.name in service.qr_names())
        # Only one failed attempt against the stocked-out zone.
        assert sum(1 for req in service.creates if req.zone == ZONE_A) == 1

    def test_quota_block_expires(self):
        pool, service, clock = make_pool(count=1)
        service.stockout_zones.add(ZONE_A)
        pool.reconcile()
        service.stockout_zones.clear()
        (name,) = service.qr_names()
        service.queued_resource_delete(name, ZONE_B)
        clock.advance(make_spec().quota_block_s + 1)
        pool.reconcile()
        assert service.creates[-1].zone == ZONE_A

    def test_all_zones_stocked_out_reports_error_and_recovers(self):
        pool, service, clock = make_pool(count=1)
        service.stockout_zones.update({ZONE_A, ZONE_B})
        report = pool.reconcile()
        assert report.errors
        assert len(service.qr_names()) == 0
        service.stockout_zones.clear()
        clock.advance(make_spec().quota_block_s + 1)
        report = pool.reconcile()
        assert len(service.qr_names()) == 1
        assert not report.errors

    def test_silent_stockout_rotates_after_timeout(self):
        pool, service, clock = make_pool(count=1)
        service.silent_stockout_zones.add(ZONE_A)
        pool.reconcile()
        (stuck,) = service.qr_names()
        clock.advance(3601.0)
        report = pool.reconcile()
        assert stuck not in service.qr_names()
        assert len(service.qr_names()) == 1
        assert service.creates[-1].zone == ZONE_B
        assert not report.errors

    def test_buffer_holds_and_reclaims_warm_spare(self):
        pool, service, _ = make_pool(count=0, buffer=1)
        pool.reconcile()
        assert len(service.qr_names()) == 1
        service.activate_pending()
        (name,) = service.qr_names()
        service.set_state(name, ZONE_A, "SUSPENDED")
        pool.reconcile()
        assert name not in service.qr_names()
        assert len(service.qr_names()) == 1

    def test_create_failure_does_not_leak_half_registered_qr(self):
        pool, service, _ = make_pool(count=1)
        service.inject_failure(
            "queued_resource_create", InfraError("deadline exceeded"), register_before_failing=True
        )
        report = pool.reconcile()
        assert report.errors
        # The half-registered QR was best-effort deleted.
        assert service.qr_names() == set()
        assert service.deletes and service.deletes[-1][0] == service.creates[-1].name
        # Next pass provisions cleanly.
        report = pool.reconcile()
        assert len(service.qr_names()) == 1
        assert not report.errors

    def test_scale_down_sheds_to_new_count(self):
        pool, service, _ = make_pool(count=3)
        pool.reconcile()
        service.activate_pending()
        import dataclasses

        pool.spec = dataclasses.replace(pool.spec, count=1)
        pool.reconcile()
        assert len(service.qr_names()) == 1

    def test_adopt_prefix_counts_unlabeled_fleet(self):
        # Migration path: legacy hand-created QRs carry no labels; adopt
        # them by name prefix so the pool holds off on new creates while
        # they live.
        pool, service, _ = make_pool(count=2, adopt_prefix="demo-qr")
        for suffix in ("b", "c"):
            service.queued_resource_create(
                TpuCreateRequest(
                    name=f"demo-qr-{suffix}",
                    zone=ZONE_A,
                    accelerator_type="v5p-128",
                    runtime_version="v2-alpha-tpuv5",
                )
            )
        service.activate_pending()
        report = pool.reconcile()
        assert report.actions == []
        assert report.active == 2

    def test_adopted_suspended_is_replaced_by_labeled_pool_qr(self):
        pool, service, _ = make_pool(count=1, adopt_prefix="demo-qr")
        service.queued_resource_create(
            TpuCreateRequest(
                name="demo-qr-a",
                zone=ZONE_A,
                accelerator_type="v5p-128",
                runtime_version="v2-alpha-tpuv5",
            )
        )
        service.set_state("demo-qr-a", ZONE_A, "SUSPENDED")
        pool.reconcile()
        assert "demo-qr-a" not in service.qr_names()
        (replacement_req,) = [r for r in service.creates if r.name in service.qr_names()]
        assert replacement_req.labels[CAPACITY_POOL_LABEL] == "testpool"

    def test_dry_run_never_touches_cloud(self):
        spec = make_spec(count=2)
        clock = FakeClock(NOW)
        service = FakeQrService(clock=clock)
        service.queued_resource_create(
            TpuCreateRequest(
                name="testpool-existing",
                zone=ZONE_A,
                accelerator_type="v5p-128",
                runtime_version="v2-alpha-tpuv5",
                labels={CAPACITY_POOL_LABEL: "testpool"},
            )
        )
        service.set_state("testpool-existing", ZONE_A, "SUSPENDED")
        creates_before, deletes_before = len(service.creates), len(service.deletes)

        pool = CapacityPool(service, spec, dry_run=True, clock=clock)
        report = pool.reconcile()
        assert kinds(report.actions) == ["delete", "create", "create"]
        assert report.executed == []
        assert len(service.creates) == creates_before
        assert len(service.deletes) == deletes_before

    def test_observe_failure_takes_no_actions(self):
        # An API outage must look like "cannot see", never "own nothing":
        # acting on the emptiness would mass-create every pass.
        from eray.capacity.types import InfraUnavailableError

        pool, service, _ = make_pool(count=4)
        service.inject_failure("queued_resource_list", InfraUnavailableError("503 backend"))
        report = pool.reconcile()
        assert report.actions == []
        assert report.executed == []
        assert any("observe failed" in e for e in report.errors)
        assert service.qr_names() == set()  # zero creates

    def test_unknown_accelerator_rejected_at_spec_construction(self):
        with pytest.raises(ValueError, match="unknown accelerator"):
            make_spec(accelerator_type="v7p-256")

    def test_excess_provisioning_shed_uses_force(self):
        observed = [qr("act", "ACTIVE", age=500.0), qr("prov", "PROVISIONING", age=100.0)]
        actions = plan_pool(make_spec(count=1), observed, now=NOW)
        assert kinds(actions) == ["delete"]
        assert actions[0].name == "prov"
        assert actions[0].force is True  # node creation already started

    def test_other_pools_qrs_are_untouched(self):
        pool, service, _ = make_pool(count=1)
        service.queued_resource_create(
            TpuCreateRequest(
                name="otherpool-abc",
                zone=ZONE_A,
                accelerator_type="v5p-128",
                runtime_version="v2-alpha-tpuv5",
                labels={CAPACITY_POOL_LABEL: "otherpool"},
            )
        )
        service.set_state("otherpool-abc", ZONE_A, "SUSPENDED")
        pool.reconcile()
        assert "otherpool-abc" in service.qr_names()
        assert not any(d[0] == "otherpool-abc" for d in service.deletes)


# ============================================================================
# Spec persistence
# ============================================================================


class TestState:
    def test_spec_roundtrip(self, tmp_path):
        path = tmp_path / "capacity.json"
        spec = make_spec(count=4, buffer=1, labels={"team": "research"}, adopt_prefix="demo-qr")
        save_pool_spec(spec, path)
        loaded = load_pool_specs(path)
        assert loaded == {"testpool": spec}
        assert loaded["testpool"].capacity is CapacityType.SPOT
        assert isinstance(loaded["testpool"].zones, tuple)

    def test_remove(self, tmp_path):
        path = tmp_path / "capacity.json"
        save_pool_spec(make_spec(), path)
        assert remove_pool_spec("testpool", path) is True
        assert remove_pool_spec("testpool", path) is False
        assert load_pool_specs(path) == {}

    def test_corrupt_state_file_fails_actionably(self, tmp_path):
        path = tmp_path / "capacity.json"
        path.write_text("{not json")
        with pytest.raises(RuntimeError, match="not valid JSON"):
            load_pool_specs(path)

    def test_unknown_keys_ignored(self, tmp_path):
        import json

        path = tmp_path / "capacity.json"
        save_pool_spec(make_spec(), path)
        doc = json.loads(path.read_text())
        doc["pools"]["testpool"]["future_field"] = 42
        path.write_text(json.dumps(doc))
        assert load_pool_specs(path)["testpool"] == make_spec()


# ============================================================================
# Proto construction (needs google-cloud-tpu — the eray[gcp] extra)
# ============================================================================


@pytest.mark.skipif(not _HAS_GCP_EXTRA, reason="eray[gcp] extra not installed")
class TestProtoBuild:
    def _request(self, **overrides):
        defaults = dict(
            name="testpool-x",
            zone=ZONE_A,
            accelerator_type="v5p-128",
            runtime_version="v2-alpha-tpuv5",
            labels={"eray-capacity-pool": "testpool"},
            metadata={"startup-script": "echo hi"},
        )
        defaults.update(overrides)
        return TpuCreateRequest(**defaults)

    def test_spot_sets_spot_block(self):
        from eray.capacity.gcp_qr import build_queued_resource

        proto = build_queued_resource(self._request(capacity_type=CapacityType.SPOT), "proj")
        pb = type(proto).pb(proto)
        assert pb.HasField("spot")
        assert not pb.HasField("guaranteed")

    def test_reserved_sets_guaranteed_reserved(self):
        from eray.capacity.gcp_qr import build_queued_resource

        proto = build_queued_resource(self._request(capacity_type=CapacityType.RESERVED), "proj")
        pb = type(proto).pb(proto)
        assert pb.HasField("guaranteed")
        assert proto.guaranteed.reserved is True
        assert not pb.HasField("spot")

    def test_on_demand_sets_no_scheduling_block(self):
        from eray.capacity.gcp_qr import build_queued_resource

        proto = build_queued_resource(self._request(capacity_type=CapacityType.ON_DEMAND), "proj")
        pb = type(proto).pb(proto)
        assert not pb.HasField("spot")
        assert not pb.HasField("guaranteed")

    def test_guaranteed_without_reserved(self):
        from eray.capacity.gcp_qr import build_queued_resource

        proto = build_queued_resource(self._request(capacity_type=CapacityType.GUARANTEED), "proj")
        assert proto.guaranteed.reserved is False

    def test_node_spec_carries_identity_and_labels(self):
        from eray.capacity.gcp_qr import build_queued_resource

        proto = build_queued_resource(self._request(), "proj")
        spec = proto.tpu.node_spec[0]
        assert spec.parent == f"projects/proj/locations/{ZONE_A}"
        assert spec.node_id == "testpool-x"
        assert spec.node.accelerator_type == "v5p-128"
        assert dict(spec.node.labels) == {"eray-capacity-pool": "testpool"}
        assert dict(spec.node.metadata) == {"startup-script": "echo hi"}

    def test_valid_until_maps_to_queueing_policy(self):
        from eray.capacity.gcp_qr import build_queued_resource

        proto = build_queued_resource(self._request(valid_until_duration="72h"), "proj")
        assert proto.queueing_policy.valid_until_duration.total_seconds() == 72 * 3600


# ============================================================================
# CLI
# ============================================================================


@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    """Isolated CLI environment: fake service + tmp state file."""
    clock = FakeClock(NOW)
    service = FakeQrService(clock=clock)
    monkeypatch.setattr(cli_capacity, "make_service", lambda project: service)
    monkeypatch.setattr(state_module, "STATE_PATH", tmp_path / "capacity.json")
    from eray.cli.main import cli

    return CliRunner(), cli, service


class TestCli:
    def test_provision_creates_and_saves(self, cli_env):
        runner, cli, service = cli_env
        result = runner.invoke(
            cli,
            ["tpu", "provision", "-a", "v5p-128", "--zones", f"{ZONE_A},{ZONE_B}", "-p", "test-project", "--count", "2"],
        )
        assert result.exit_code == 0, result.output
        assert len(service.qr_names()) == 2
        assert "eray-v5p-128" in load_pool_specs()

    def test_provision_dry_run_saves_nothing_touches_nothing(self, cli_env):
        runner, cli, service = cli_env
        result = runner.invoke(
            cli,
            ["tpu", "provision", "-a", "v5p-128", "--zones", ZONE_A, "-p", "test-project", "--dry-run"],
        )
        assert result.exit_code == 0, result.output
        assert service.qr_names() == set()
        assert load_pool_specs() == {}
        assert "would create" in result.output

    def test_reclaim_recovers_suspended(self, cli_env):
        runner, cli, service = cli_env
        runner.invoke(
            cli,
            ["tpu", "provision", "-a", "v5p-128", "--zones", ZONE_A, "-p", "test-project", "--count", "1"],
        )
        (name,) = service.qr_names()
        service.set_state(name, ZONE_A, "SUSPENDED")
        result = runner.invoke(cli, ["tpu", "reclaim"])
        assert result.exit_code == 0, result.output
        assert name not in service.qr_names()
        assert len(service.qr_names()) == 1

    def test_wait_rejects_dry_run(self, cli_env):
        runner, cli, _ = cli_env
        result = runner.invoke(
            cli,
            ["tpu", "provision", "-a", "v4-8", "--zones", ZONE_A, "-p", "test-project", "--dry-run", "--wait", "5"],
        )
        assert result.exit_code != 0
        assert "--wait cannot be combined" in result.output

    def test_reclaim_isolates_broken_pool(self, cli_env, monkeypatch):
        # One pool whose service cannot observe must not stop the loop from
        # reconciling the healthy pool.
        from eray.capacity.types import InfraUnavailableError

        runner, cli, healthy_service = cli_env
        for pname, proj in (("pool-a", "proj-a"), ("pool-b", "proj-b")):
            runner.invoke(
                cli,
                ["tpu", "provision", "-a", "v4-8", "--zones", ZONE_A, "-p", proj, "--count", "1", "--name", pname],
            )

        broken = FakeQrService(project_id="proj-a")
        broken.inject_failure("queued_resource_list", InfraUnavailableError("outage"))

        def per_project(project):
            return broken if project == "proj-a" else healthy_service

        monkeypatch.setattr(cli_capacity, "make_service", per_project)
        result = runner.invoke(cli, ["tpu", "reclaim"])
        assert result.exit_code != 0  # the broken pool is an error...
        assert "observe failed" in result.output
        # ...but the healthy pool was still reconciled this pass (its
        # desired state remains satisfied — no exception aborted the loop).
        assert "pool-b" in result.output

    def test_reclaim_without_pools_errors(self, cli_env):
        runner, cli, _ = cli_env
        result = runner.invoke(cli, ["tpu", "reclaim"])
        assert result.exit_code != 0
        assert "no saved capacity pools" in result.output

    def test_release_deletes_and_forgets(self, cli_env):
        runner, cli, service = cli_env
        runner.invoke(
            cli,
            ["tpu", "provision", "-a", "v5p-128", "--zones", ZONE_A, "-p", "test-project", "--count", "2"],
        )
        result = runner.invoke(cli, ["tpu", "release", "--name", "eray-v5p-128", "--yes"])
        assert result.exit_code == 0, result.output
        assert service.qr_names() == set()
        assert load_pool_specs() == {}
