# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``eyvan race tpu``: request one TPU shape in many zones at once, then clean up.

Every test runs against an in-memory queue that behaves like the Cloud TPU
queued-resource API where it matters: refusals arrive on the create operation,
deletes are refused while a request is PROVISIONING, and nothing is gone until
a listing says so.
"""

from __future__ import annotations

import _thread
import functools
import json
import threading

import pytest
from click.testing import CliRunner

from eyvan.capacity.providers.gcp import tpu_race
from eyvan.capacity.providers.gcp.tpu_race import RaceOutcome, RaceTiming, TpuRace, gcloud_token

PROJECT = "example-spot-project"
FAST = RaceTiming(timeout_s=5.0, cleanup_timeout_s=5.0, poll_s=0.01, operation_poll_s=0.01, stagger_s=0.0)


class FakeQueue:
    """Per-zone plans: ``states`` the request walks through, ``refuse`` (code 7), ``reject``,
    ``refuse_deletes`` (deletes answered FAILED_PRECONDITION first), ``offers``."""

    def __init__(self, plans: dict[str, dict]) -> None:
        self.plans = plans
        self.requests: dict[tuple[str, str], dict] = {}
        self.lock = threading.Lock()
        self.created: list[tuple[str, str, dict]] = []
        self.deleted: list[tuple[str, str]] = []
        self.refused_deletes: list[tuple[str, str]] = []
        self.on_get: list = []

    def list_zones(self, project: str) -> list[str]:
        return sorted(self.plans)

    def offers(self, project: str, zone: str, accelerator_type: str) -> bool | None:
        return self.plans[zone].get("offers", True)

    def create(self, project, zone, name, body):
        plan = self.plans[zone]
        if plan.get("reject"):
            return 400, {"error": {"code": 3, "message": "Queueing is not supported"}}
        with self.lock:
            self.created.append((zone, name, body))
            if not plan.get("refuse"):
                self.requests[(zone, name)] = {
                    "states": list(plan.get("states", ["WAITING_FOR_RESOURCES"])),
                    "refuse_deletes": plan.get("refuse_deletes", 0),
                }
        return 200, {"name": f"projects/{project}/locations/{zone}/operations/op-{name}", "done": False}

    def operation(self, name):
        zone = name.split("/")[3]
        if self.plans[zone].get("refuse"):
            message = "User does not have permission to submit requests into this queue"
            return 200, {"name": name, "done": True, "error": {"code": 7, "message": message}}
        return 200, {"name": name, "done": True}

    def get(self, project, zone, name):
        for hook in list(self.on_get):
            hook(zone)
        with self.lock:
            request = self.requests.get((zone, name))
            if request is None:
                return 404, {}
            states = request["states"]
            state = states.pop(0) if len(states) > 1 else states[0]
        return 200, {"state": {"state": state}}

    def delete(self, project, zone, name):
        with self.lock:
            request = self.requests.get((zone, name))
            if request is None:
                return 404, {}
            if request["refuse_deletes"] > 0:
                request["refuse_deletes"] -= 1
                self.refused_deletes.append((zone, name))
                return 400, {"error": {"code": 9, "message": "cannot delete while PROVISIONING"}}
            del self.requests[(zone, name)]
            self.deleted.append((zone, name))
        return 200, {"name": "operations/delete"}

    def list_requests(self, project, zone):
        with self.lock:
            return 200, [name for (z, name) in self.requests if z == zone]

    def list_nodes(self, project, zone):
        return 200, []

    def delete_node(self, project, zone, name):
        return 200, {}


def _race(queue: FakeQueue, **options) -> TpuRace:
    options.setdefault("timing", FAST)
    return TpuRace(queue, project=PROJECT, accelerator_type="v6e-8", **options)


def _by_zone(report) -> dict:
    return {zone.zone: zone for zone in report.zones}


def test_measure_only_reports_every_zone_and_deletes_every_request(tmp_path) -> None:
    queue = FakeQueue(
        {
            "us-east5-b": {"states": ["WAITING_FOR_RESOURCES", "PROVISIONING", "ACTIVE"]},
            "us-central1-a": {"refuse": True},
            "asia-east1-c": {"reject": True},
            "us-south1-a": {"states": ["WAITING_FOR_RESOURCES", "SUSPENDING"]},
            "us-west1-c": {"states": ["FAILED"]},
        }
    )
    race = _race(queue, receipt=tmp_path / "r.json")

    report = race.run()
    zones = _by_zone(report)

    assert zones["us-east5-b"].outcome is RaceOutcome.GRANTED and not zones["us-east5-b"].kept
    assert [state for _, state in zones["us-east5-b"].states] == ["WAITING_FOR_RESOURCES", "PROVISIONING", "ACTIVE"]
    # Code 7 arrives on the operation, after the create call already said 200.
    assert zones["us-central1-a"].outcome is RaceOutcome.REFUSED and zones["us-central1-a"].error_code == 7
    assert zones["asia-east1-c"].outcome is RaceOutcome.REJECTED and zones["asia-east1-c"].cleanup is None
    assert zones["us-south1-a"].outcome is RaceOutcome.SUSPENDED
    assert zones["us-west1-c"].outcome is RaceOutcome.FAILED
    assert queue.requests == {}, "a request survived the race"
    assert report.sweep is not None and report.sweep.clean and report.clean
    receipt = json.loads((tmp_path / "r.json").read_text())
    assert receipt["clean"] is True and receipt["kept"] is None
    assert receipt["zones"][0]["name"].startswith(f"eyvan-race-{report.run_id}-")


def test_concurrent_zone_threads_never_break_the_receipt(tmp_path) -> None:
    """Every zone thread rewrites the receipt; unserialized, two renames of one temp file collide."""

    zones = [f"us-east{index}-b" for index in range(1, 10)]
    queue = FakeQueue({zone: {"states": ["ACTIVE"]} for zone in zones})
    errors: list[BaseException] = []
    previous = threading.excepthook
    threading.excepthook = lambda arguments: errors.append(arguments.exc_value)
    try:
        report = _race(queue, receipt=tmp_path / "r.json").run()
    finally:
        threading.excepthook = previous

    assert errors == []
    assert json.loads((tmp_path / "r.json").read_text())["clean"] is True and report.clean


def test_spot_label_and_name_are_on_every_request() -> None:
    queue = FakeQueue({"us-east5-b": {"states": ["ACTIVE"]}})
    report = _race(queue).run()

    ((_zone, name, body),) = queue.created
    assert name == f"eyvan-race-{report.run_id}-us-east5-b"
    assert body["spot"] == {}
    node = body["tpu"]["nodeSpec"][0]
    assert node["node"]["labels"] == {"eyvan-race": report.run_id}
    assert node["node"]["acceleratorType"] == "v6e-8" and node["node"]["runtimeVersion"] == "v2-alpha-tpuv6e"


def test_on_demand_sends_no_spot_block() -> None:
    queue = FakeQueue({"us-east5-b": {"states": ["ACTIVE"]}})
    _race(queue, spot=False).run()

    assert "spot" not in queue.created[0][2]


def test_a_provisioning_request_is_waited_out_then_deleted() -> None:
    """The API refuses to delete a PROVISIONING request; giving up there leaks a billed TPU."""

    queue = FakeQueue({"us-east5-b": {"states": ["PROVISIONING"], "refuse_deletes": 2}})
    report = _race(queue, timing=RaceTiming(timeout_s=0.05, cleanup_timeout_s=5.0, poll_s=0.01, stagger_s=0.0)).run()

    zone = _by_zone(report)["us-east5-b"]
    assert zone.outcome is RaceOutcome.WAITING
    assert len(queue.refused_deletes) == 2
    assert zone.cleanup == "deleted" and queue.requests == {}
    assert report.clean


def test_keep_first_keeps_exactly_one_grant_and_deletes_the_rest() -> None:
    queue = FakeQueue({zone: {"states": ["ACTIVE"]} for zone in ("us-east5-b", "us-central1-a", "us-west1-c")})
    report = _race(queue, keep_first=True).run()

    kept = [zone for zone in report.zones if zone.kept]
    assert len(kept) == 1 and kept[0].cleanup == "kept"
    assert list(queue.requests) == [(kept[0].zone, kept[0].name)]
    assert report.kept is kept[0] and report.clean


def test_keep_first_stops_the_zones_still_waiting() -> None:
    queue = FakeQueue(
        {
            "us-east5-b": {"states": ["WAITING_FOR_RESOURCES", "ACTIVE"]},
            "us-central1-b": {"states": ["WAITING_FOR_RESOURCES"]},
        }
    )
    report = _race(queue, keep_first=True, timing=RaceTiming(timeout_s=60.0, poll_s=0.01, stagger_s=0.0)).run()

    zones = _by_zone(report)
    assert zones["us-east5-b"].kept
    assert zones["us-central1-b"].outcome is RaceOutcome.CANCELLED and zones["us-central1-b"].cleanup == "deleted"
    assert list(queue.requests) == [("us-east5-b", zones["us-east5-b"].name)]


def test_ctrl_c_stops_the_race_but_not_the_cleanup() -> None:
    queue = FakeQueue({zone: {"states": ["WAITING_FOR_RESOURCES"]} for zone in ("us-east5-b", "us-central1-b")})
    fired = threading.Event()

    def interrupt_once_everything_exists(zone: str) -> None:
        if len(queue.created) == 2 and not fired.is_set():
            fired.set()
            _thread.interrupt_main()

    queue.on_get.append(interrupt_once_everything_exists)
    report = _race(queue, timing=RaceTiming(timeout_s=60.0, poll_s=0.01, stagger_s=0.0)).run()

    assert fired.is_set() and report.interrupted
    assert {zone.outcome for zone in report.zones} == {RaceOutcome.CANCELLED}
    assert queue.requests == {} and report.clean


def test_an_interim_receipt_says_running_not_error(tmp_path) -> None:
    """Read mid-race (or after a crash), an unsettled zone must not look like a failure."""

    queue = FakeQueue({"us-east5-b": {"states": ["WAITING_FOR_RESOURCES", "ACTIVE"]}})
    receipt = tmp_path / "r.json"
    seen: list[str] = []

    def read_receipt(zone: str) -> None:
        if not seen and receipt.exists():
            seen.append(json.loads(receipt.read_text())["zones"][0]["outcome"])

    queue.on_get.append(read_receipt)
    report = _race(queue, receipt=receipt).run()

    assert seen == ["running"]
    assert _by_zone(report)["us-east5-b"].outcome is RaceOutcome.GRANTED


def test_without_named_zones_only_zones_offering_the_shape_are_raced() -> None:
    queue = FakeQueue(
        {
            "us-east5-b": {"states": ["ACTIVE"]},
            "europe-west4-a": {"offers": False},
            "asia-south1-b": {"offers": None},
        }
    )
    report = _race(queue).run()

    assert [zone.zone for zone in report.zones] == ["us-east5-b"]
    assert report.not_offered == ["europe-west4-a"] and report.unknown_offer == ["asia-south1-b"]


def test_an_unlistable_zone_is_never_reported_clean() -> None:
    queue = FakeQueue({"us-east5-b": {"states": ["ACTIVE"]}})
    queue.list_requests = lambda project, zone: (403, [])  # type: ignore[method-assign]
    report = _race(queue).run()

    assert report.sweep is not None and report.sweep.unreadable and not report.clean


@pytest.mark.parametrize("fields", [{"project": "Not A Project"}, {"accelerator_type": "v6e"}, {"zones": ["east"]}])
def test_bad_inputs_are_refused_before_anything_is_created(fields) -> None:
    queue = FakeQueue({})
    options = {"project": PROJECT, "accelerator_type": "v6e-8", **fields}
    with pytest.raises(ValueError):
        TpuRace(queue, **options)
    assert queue.created == []


# --------------------------------------------------------------------------- the account


def test_the_token_is_minted_as_the_named_account_without_changing_gcloud_config() -> None:
    calls: list[tuple[list[str], dict]] = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs["env"]))
        import subprocess

        return subprocess.CompletedProcess(argv, 0, stdout="tok\n", stderr="")

    assert gcloud_token("me@example.com", runner=runner) == "tok"
    assert gcloud_token(None, runner=runner) == "tok"

    (first, first_env), (second, second_env) = calls
    assert first == second == ["gcloud", "auth", "print-access-token"]
    assert first_env["CLOUDSDK_CORE_ACCOUNT"] == "me@example.com"
    assert second_env.get("CLOUDSDK_CORE_ACCOUNT") == __import__("os").environ.get("CLOUDSDK_CORE_ACCOUNT")


# --------------------------------------------------------------------------- the command


@pytest.fixture
def cli(monkeypatch):
    from eyvan.sdk.commands import race as race_command

    queue = FakeQueue({"us-east5-b": {"states": ["ACTIVE"]}, "us-central1-a": {"refuse": True}})
    accounts: list[str | None] = []

    def factory(account):
        accounts.append(account)
        return queue

    monkeypatch.setattr(race_command, "_api_factory", factory)
    monkeypatch.setattr(race_command, "gcloud_active_account", lambda: "active@example.com")
    monkeypatch.setattr(
        race_command,
        "RaceTiming",
        functools.partial(RaceTiming, poll_s=0.01, operation_poll_s=0.01, stagger_s=0.0, cleanup_timeout_s=5.0),
    )
    return race_command, queue, accounts


def test_the_command_races_as_the_named_account_and_prints_a_receipt(cli, tmp_path) -> None:
    race_command, queue, accounts = cli
    receipt = tmp_path / "race.json"

    result = CliRunner().invoke(
        race_command.race,
        ["tpu", "v6e-8", "--project", PROJECT, "--zone", "us-east5-b", "--zone", "us-central1-a",
         "--account", "me@example.com", "--yes", "--json", "--receipt", str(receipt)],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert accounts == ["me@example.com"]
    document = json.loads(result.stdout)
    outcomes = {zone["zone"]: zone["outcome"] for zone in document["zones"]}
    assert outcomes == {"us-east5-b": "granted", "us-central1-a": "refused"}
    assert document["clean"] is True and document["account"] == "me@example.com"
    assert json.loads(receipt.read_text())["run_id"] == document["run_id"]
    assert queue.requests == {}


def test_the_command_refuses_to_create_anything_unattended_without_yes(cli, tmp_path) -> None:
    race_command, queue, _ = cli

    result = CliRunner().invoke(
        race_command.race, ["tpu", "v6e-8", "--project", PROJECT, "--receipt", str(tmp_path / "r.json")]
    )

    assert result.exit_code != 0 and "--yes" in result.output
    assert queue.created == []


def test_keep_first_prints_how_to_attach_and_delete_the_grant(cli, tmp_path) -> None:
    race_command, queue, _ = cli

    result = CliRunner().invoke(
        race_command.race,
        ["tpu", "v6e-8", "--project", PROJECT, "--keep-first", "--yes", "--receipt", str(tmp_path / "r.json")],
    )

    assert result.exit_code == 0, result.output
    ((zone, name),) = queue.requests
    assert f"eyvan gcp connect --tpu {name} --project {PROJECT} --zone {zone}" in result.output
    assert f"queued-resources delete {name} --zone {zone}" in result.output


def test_sweep_deletes_leftovers_and_spares_nothing_unless_kept(cli) -> None:
    race_command, queue, _ = cli
    run_id = "0a1b2c3d"
    queue.requests[("us-east5-b", f"eyvan-race-{run_id}-us-east5-b")] = {"states": ["ACTIVE"], "refuse_deletes": 0}
    queue.requests[("us-east5-b", "someone-elses-tpu")] = {"states": ["ACTIVE"], "refuse_deletes": 0}

    result = CliRunner().invoke(race_command.race, ["sweep", run_id, "--project", PROJECT, "--zone", "us-east5-b"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["clean"] is True
    assert list(queue.requests) == [("us-east5-b", "someone-elses-tpu")]


def test_race_is_registered_on_the_main_cli() -> None:
    from eyvan.sdk.cli import main

    assert "race" in main.commands and {"tpu", "sweep"} <= set(main.commands["race"].commands)


def test_the_module_never_runs_gcloud_config() -> None:
    from pathlib import Path

    from eyvan.sdk.commands import race as race_command

    for path in (Path(tpu_race.__file__), Path(race_command.__file__)):
        text = path.read_text(encoding="utf-8")
        assert '"config", "set"' not in text and "config set account" not in text.replace("`", "")


def test_sweep_accepts_receipt_inherits_account_and_lists_zones_when_omitted(cli, tmp_path) -> None:
    race_command, queue, accounts = cli
    run_id = "deadbeef"
    queue.requests[("us-east5-b", f"eyvan-race-{run_id}-us-east5-b")] = {"states": ["ACTIVE"], "refuse_deletes": 0}
    receipt = tmp_path / "race.json"
    receipt.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "project": PROJECT,
                "account": "receipt@example.com",
                "zones": [{"zone": "us-east5-b"}],
            }
        ),
        encoding="utf-8",
    )

    res1 = CliRunner().invoke(race_command.race, ["sweep", "--receipt", str(receipt)])
    assert res1.exit_code == 0, res1.output
    assert json.loads(res1.stdout)["clean"] is True
    assert accounts == ["receipt@example.com"]
    assert queue.requests == {}

    # Printed recovery hint (`eyvan race sweep <run_id> --project <project>` without --zone) uses list_zones.
    run_id2 = "cafebabe"
    queue.requests[("us-east5-b", f"eyvan-race-{run_id2}-us-east5-b")] = {"states": ["ACTIVE"], "refuse_deletes": 0}
    res2 = CliRunner().invoke(race_command.race, ["sweep", run_id2, "--project", PROJECT])
    assert res2.exit_code == 0, res2.output
    assert json.loads(res2.stdout)["clean"] is True
    assert queue.requests == {}


def test_cleaner_sweep_waits_for_async_node_deletion_until_404() -> None:
    queue = FakeQueue({"us-east5-b": {"states": ["ACTIVE"]}})
    run_id = "11223344"
    node_name = f"eyvan-race-{run_id}-us-east5-b"
    remaining_polls = [2]
    node_deleted: list[str] = []

    def list_nodes(project, zone):
        del project
        return 200, [node_name] if zone == "us-east5-b" and remaining_polls[0] > 0 else []

    def delete_node(project, zone, name):
        del project, zone
        node_deleted.append(name)
        return 200, {"name": "operations/delete-node"}

    def get_node(project, zone, name):
        del project, zone, name
        if remaining_polls[0] > 0:
            remaining_polls[0] -= 1
            return 200, {"state": "DELETING"}
        return 404, {}

    queue.list_nodes = list_nodes  # type: ignore[method-assign]
    queue.delete_node = delete_node  # type: ignore[method-assign]
    queue.get_node = get_node  # type: ignore[attr-defined]

    sweep = tpu_race.sweep_race(queue, project=PROJECT, run_id=run_id, zones=["us-east5-b"], timing=FAST)
    assert node_deleted == [node_name]
    assert remaining_polls == [0]
    assert sweep.clean is True


def test_cleaner_delete_reissues_promptly_after_leaving_provisioning() -> None:
    import time

    queue = FakeQueue({"us-east5-b": {"states": ["PROVISIONING", "WAITING_FOR_RESOURCES"], "refuse_deletes": 1}})
    get_calls = [0]
    original_get = queue.get

    def counting_get(project, zone, name):
        get_calls[0] += 1
        return original_get(project, zone, name)

    queue.get = counting_get  # type: ignore[method-assign]
    queue.requests[("us-east5-b", "eyvan-race-aaaa1111-us-east5-b")] = {
        "states": ["PROVISIONING", "WAITING_FOR_RESOURCES"],
        "refuse_deletes": 1,
    }
    cleaner = tpu_race._Cleaner(queue, PROJECT, FAST, time.monotonic)
    status = cleaner.delete("us-east5-b", "eyvan-race-aaaa1111-us-east5-b")
    assert status == "deleted"
    assert len(queue.refused_deletes) == 1
    assert get_calls[0] <= 4, f"expected prompt re-DELETE after leaving PROVISIONING, saw {get_calls[0]} get calls"


def test_race_zone_fails_fast_on_non_200_operation_poll() -> None:
    queue = FakeQueue({"us-east5-b": {"states": ["ACTIVE"]}})
    queue.operation = lambda name: (403, {"error": {"code": 7, "message": "operation poll forbidden"}})  # type: ignore[method-assign]
    report = _race(queue, zones=["us-east5-b"]).run()
    [zone] = report.zones
    assert zone.outcome is RaceOutcome.ERROR
    assert "403" in (zone.error or "") or "forbidden" in (zone.error or "")


def test_tpu7x_default_runtime_version() -> None:
    from eyvan.capacity.providers.gcp.queued_resources import default_runtime_version

    assert default_runtime_version("tpu7x-4") == "v2-alpha-tpuv7x"


def test_rest_call_skips_retries_on_permanent_quota_429_and_retries_rate_limit_429() -> None:
    import httpx

    # Permanent quota exhaustion (Limit: 0) must return immediately with zero retry sleeps.
    perm_calls = [0]
    perm_sleeps: list[float] = []

    def perm_handler(request: httpx.Request) -> httpx.Response:
        del request
        perm_calls[0] += 1
        return httpx.Response(
            429,
            json={
                "error": {
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Quota limit 'tpu-v6e-preemptible' has been exceeded. Limit: 0",
                }
            },
        )

    rest_perm = tpu_race.TpuQueueRest(
        token_source=lambda: "tok",
        transport=httpx.MockTransport(perm_handler),
        sleep=perm_sleeps.append,
    )
    status, doc = rest_perm.create(PROJECT, "us-east5-b", "eyvan-race-12345678-us-east5-b", {})
    assert status == 429
    assert perm_calls == [1] and perm_sleeps == []
    assert "Limit: 0" in tpu_race.error_text(doc)

    # Transient rate-limit 429 still retries and succeeds on the next attempt.
    rate_calls = [0]
    rate_sleeps: list[float] = []

    def rate_handler(request: httpx.Request) -> httpx.Response:
        del request
        rate_calls[0] += 1
        if rate_calls[0] == 1:
            return httpx.Response(
                429,
                json={
                    "error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Rate limit exceeded per minute"}
                },
            )
        return httpx.Response(200, json={"name": "operations/ok", "done": True})

    rest_rate = tpu_race.TpuQueueRest(
        token_source=lambda: "tok",
        transport=httpx.MockTransport(rate_handler),
        sleep=rate_sleeps.append,
    )
    status2, doc2 = rest_rate.create(PROJECT, "us-east5-b", "eyvan-race-12345678-us-east5-b", {})
    assert status2 == 200 and doc2["done"] is True
    assert rate_calls == [2] and rate_sleeps == [5.0]

    # Google APIs can pair ErrorInfo reason="QUOTA_EXCEEDED" with a per-minute rate limit; must still retry.
    paired_calls = [0]
    paired_sleeps: list[float] = []

    def paired_handler(request: httpx.Request) -> httpx.Response:
        del request
        paired_calls[0] += 1
        if paired_calls[0] == 1:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "code": 429,
                        "status": "RESOURCE_EXHAUSTED",
                        "message": "Read requests per minute quota exceeded",
                        "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "QUOTA_EXCEEDED"}],
                    }
                },
            )
        return httpx.Response(200, json={"name": "operations/ok2", "done": True})

    rest_paired = tpu_race.TpuQueueRest(
        token_source=lambda: "tok",
        transport=httpx.MockTransport(paired_handler),
        sleep=paired_sleeps.append,
    )
    status3, doc3 = rest_paired.create(PROJECT, "us-east5-b", "eyvan-race-12345678-us-east5-b", {})
    assert status3 == 200 and doc3["done"] is True
    assert paired_calls == [2] and paired_sleeps == [5.0]


def test_cleaner_emits_teardown_transitions_and_reissues_delete_immediately_on_suspended() -> None:
    import time

    events: list[tuple[str, str]] = []
    delete_calls = [0]
    get_states = ["SUSPENDING", "SUSPENDED"]
    get_calls = [0]

    class _SuspendingQueue(FakeQueue):
        def delete(self, project, zone, name):
            del project, zone, name
            delete_calls[0] += 1
            return 200, {"name": f"operations/del-{delete_calls[0]}"}

        def get(self, project, zone, name):
            del project, zone, name
            get_calls[0] += 1
            if delete_calls[0] >= 2:
                return 404, {}
            state = get_states.pop(0) if get_states else "SUSPENDED"
            return 200, {"state": {"state": state}}

    queue = _SuspendingQueue({"us-east5-b": {}})
    cleaner = tpu_race._Cleaner(queue, PROJECT, FAST, time.monotonic, on_event=lambda z, m: events.append((z, m)))
    result = cleaner.delete("us-east5-b", "eyvan-race-aaaa1111-us-east5-b")
    assert result == "deleted"
    assert delete_calls == [2]
    assert get_calls[0] <= 3, f"expected immediate re-DELETE on SUSPENDED, saw {get_calls[0]} get calls"
    assert events == [("us-east5-b", "deleting: suspending"), ("us-east5-b", "deleting: suspended")]


def test_cleaner_stuck_suspended_qr_honors_poll_sleep_and_cleanup_timeout(monkeypatch) -> None:
    now = [0.0]
    sleeps: list[float] = []
    delete_calls = [0]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    class _StuckSuspendedQueue(FakeQueue):
        def delete(self, project, zone, name):
            del project, zone, name
            delete_calls[0] += 1
            return 200, {"name": f"operations/del-{delete_calls[0]}"}

        def get(self, project, zone, name):
            del project, zone, name
            return 200, {"state": {"state": "SUSPENDED"}}

    monkeypatch.setattr(tpu_race.time, "sleep", fake_sleep)
    timing = RaceTiming(timeout_s=5.0, cleanup_timeout_s=0.05, poll_s=0.02, operation_poll_s=0.01, stagger_s=0.0)
    cleaner = tpu_race._Cleaner(_StuckSuspendedQueue({"us-east5-b": {}}), PROJECT, timing, lambda: now[0])
    outcome = cleaner.delete("us-east5-b", "eyvan-race-bbbb2222-us-east5-b")
    assert outcome.startswith("delete_timeout"), outcome
    assert sleeps == [0.02, 0.02, 0.02]
    assert delete_calls[0] <= 5, f"expected bounded deletes honoring poll_s and timeout, got {delete_calls[0]}"


def test_ai_zone_suffix_accepted_and_summary_condenses_long_not_offered_list(capsys) -> None:
    from eyvan.sdk.commands import race as race_command

    queue = FakeQueue({"us-south1-ai1b": {"states": ["ACTIVE"]}})
    race = TpuRace(queue, project=PROJECT, accelerator_type="v6e-4", zones=["us-south1-ai1b"], timing=FAST)
    report = race.run()
    assert [z.zone for z in report.zones] == ["us-south1-ai1b"] and report.clean

    report.not_offered = [f"us-central{i}-a" for i in range(15)]
    race_command._summary(report)
    out = capsys.readouterr().out
    assert "not offered in 15 zones (full list in receipt / --json)" in out
    assert "us-central14-a" not in out


def test_race_enables_external_ips_by_default_and_honors_internal_ips_flag(cli, tmp_path) -> None:
    race_command, queue, _ = cli

    res_default = CliRunner().invoke(
        race_command.race,
        ["tpu", "v6e-8", "--project", PROJECT, "--zone", "us-east5-b", "--yes", "--receipt", str(tmp_path / "r1.json")],
    )
    assert res_default.exit_code == 0, res_default.output
    default_node = queue.created[-1][2]["tpu"]["nodeSpec"][0]["node"]
    assert default_node["networkConfig"] == {"enableExternalIps": True}

    res_internal = CliRunner().invoke(
        race_command.race,
        [
            "tpu",
            "v6e-8",
            "--project",
            PROJECT,
            "--zone",
            "us-east5-b",
            "--internal-ips",
            "--yes",
            "--receipt",
            str(tmp_path / "r2.json"),
        ],
    )
    assert res_internal.exit_code == 0, res_internal.output
    internal_node = queue.created[-1][2]["tpu"]["nodeSpec"][0]["node"]
    assert "networkConfig" not in internal_node
