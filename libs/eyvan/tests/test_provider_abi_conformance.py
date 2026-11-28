# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The conformance suite, tested the only way that means anything: by breaking it.

A suite that only ever runs against a correct provider proves nothing about what
it would catch. Every property below is defeated by a deliberately broken
subclass of the shipped on-premises driver -- each break is a bug that has real
precedent -- and the test asserts *exactly* which checks fail, so a check that
silently stopped asserting anything would show up as a passing broken provider.

The last test closes the loop: every published check is defeated by at least one
of these, so none of them is dead weight.
"""

from __future__ import annotations

import typing as tp

import pytest
from test_provider_abi_onprem import OnPremSshConformanceHarness

from eyvan.capacity import CapacityStatus
from eyvan.capacity.conformance import (
    CONFORMANCE_CHECK_NAMES,
    CONFORMANCE_CHECKS,
    KNOWN_CAPACITY_CLASSES,
    CapacityProviderConformanceTests,
    ConformanceReport,
    ConformanceResult,
    ConformanceViolation,
    ProviderConformanceHarness,
    run_conformance,
)
from eyvan.capacity.drivers.base import (
    CapacityProviderError,
    CapacityProviderFailureCode,
    CellCapabilities,
    CellObservation,
    MutationContext,
)
from eyvan.capacity.drivers.onprem_ssh import HostMarker, OnPremSshCapacityDriver
from eyvan.capacity.fleet_state import FleetCellRecord

# --------------------------------------------------------------------------- the honest provider


class TestOnPremSshConformance(CapacityProviderConformanceTests):
    """The shipped on-premises provider passes the suite it is published with."""

    @pytest.fixture(autouse=True)
    def _isolated(self, tmp_path) -> None:
        self._tmp_path = tmp_path

    def harness(self) -> ProviderConformanceHarness:
        """Return a harness over a private fake rack."""

        return OnPremSshConformanceHarness(self._tmp_path)


def test_the_shipped_provider_passes_every_published_check(tmp_path) -> None:
    """One report, so a new check cannot be added without this provider meeting it."""

    report = run_conformance(OnPremSshConformanceHarness(tmp_path))

    assert report.failed_checks == (), report.describe()
    assert tuple(result.check for result in report.results) == CONFORMANCE_CHECK_NAMES
    assert report.passed


def test_the_routable_vocabulary_includes_flex_start() -> None:
    """A provider whose cells are wait-then-window capacity can say so honestly.

    ``flex_start`` is routed end to end -- the domain enum, pool records, and the
    fleet loader all ship it -- so declaring it must not read as an invented name.
    """

    assert "flex_start" in KNOWN_CAPACITY_CLASSES


# --------------------------------------------------------------------------- broken providers


class InventsACapacityClass(OnPremSshCapacityDriver):
    """Declares a capacity class the scheduler has never heard of."""

    def capabilities(self) -> CellCapabilities:
        """Return capabilities no pool can ever match."""

        return CellCapabilities(
            provider=self.provider_name,
            executor=self.execution_backend,
            accelerator_types=("gpu",),
            capacity_classes=("preemptible",),
        )


class ReportsAnUnknownCellAsUnobserved(OnPremSshCapacityDriver):
    """Confuses "I have no record of this" with "I could not look"."""

    def observe(self, cell_id: str) -> CellObservation:
        """Report a cell with no durable record as unknown rather than absent."""

        observation = super().observe(cell_id)
        if observation.state is CapacityStatus.ABSENT:
            return CellObservation(cell_id=cell_id, state=CapacityStatus.UNOBSERVED, generation=0)
        return observation


class ForgetsToClaim(OnPremSshCapacityDriver):
    """Writes the durable record and never touches the hardware."""

    def _claim(self, host: str, **fence: object) -> HostMarker:
        """Skip the provider-side claim entirely."""

        del host
        return HostMarker(
            cell_id=str(fence["cell_id"]),
            management_uid=str(fence["management_uid"]),
            provider_generation=int(tp.cast(int, fence["provider_generation"])),
            fence=str(fence["fence"]),
        )


class ProvisionsAgainOnReplay(OnPremSshCapacityDriver):
    """Forgets the idempotency key, so every retry is a new incarnation."""

    @staticmethod
    def _accept_mutation(record: FleetCellRecord, context: MutationContext) -> bool:
        """Keep the fence, drop the replay suppression."""

        OnPremSshCapacityDriver._accept_mutation(record, context)
        return True


class TrustsAnyFence(OnPremSshCapacityDriver):
    """Accepts a superseded generation or a token that owns nothing."""

    @staticmethod
    def _accept_mutation(record: FleetCellRecord, context: MutationContext) -> bool:
        """Drop the fence, keep the replay suppression."""

        return context.idempotency_key not in record.idempotency_keys


class AdoptsWhateverIsThere(OnPremSshCapacityDriver):
    """Reads the marker and then reports what it hoped to find."""

    def _observe_host(self, host: str, **fence: object) -> HostMarker | None:
        """Return the expected identity whenever any claim exists."""

        observed = super()._observe_host(host, **tp.cast(tp.Any, fence))
        if observed is None:
            return None
        return HostMarker(
            cell_id=str(fence["cell_id"]),
            management_uid=str(fence["management_uid"]),
            provider_generation=int(tp.cast(int, fence["provider_generation"])),
            fence=str(fence["fence"]),
        )


class ReportsUnreachableAsAbsent(OnPremSshCapacityDriver):
    """Turns "the rack did not answer" into "the rack is gone"."""

    def observe(self, cell_id: str) -> CellObservation:
        """Collapse an unobserved cell into a definitive loss."""

        observation = super().observe(cell_id)
        if observation.state is CapacityStatus.UNOBSERVED:
            return CellObservation(cell_id=cell_id, state=CapacityStatus.ABSENT, generation=observation.generation)
        return observation


class PretendsAnUnreachableClaimSucceeded(OnPremSshCapacityDriver):
    """Swallows a transport failure and carries on as if the host answered."""

    def _claim(self, host: str, **fence: object) -> HostMarker:
        """Fabricate a claim whenever the host could not be reached."""

        try:
            return super()._claim(host, **tp.cast(tp.Any, fence))
        except CapacityProviderError as exc:
            if exc.code is not CapacityProviderFailureCode.PROVIDER_UNAVAILABLE:
                raise
            return HostMarker(
                cell_id=str(fence["cell_id"]),
                management_uid=str(fence["management_uid"]),
                provider_generation=int(tp.cast(int, fence["provider_generation"])),
                fence=str(fence["fence"]),
            )

    def _run_lifecycle_command(self, host: str, argv, *, phase: str) -> None:
        """Treat a launcher that was never reached as a launcher that ran."""

        try:
            super()._run_lifecycle_command(host, argv, phase=phase)
        except CapacityProviderError as exc:
            if exc.code is not CapacityProviderFailureCode.PROVIDER_UNAVAILABLE:
                raise


class TreatsEveryFailureAsTransient(OnPremSshCapacityDriver):
    """Reports a permanent refusal as something worth retrying forever."""

    @staticmethod
    def _configuration(detail: str) -> CapacityProviderError:
        """Reclassify a conclusive refusal as a retryable outage."""

        return CapacityProviderError(
            detail,
            code=CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
            retryable=True,
        )


class LeavesMarkersBehind(OnPremSshCapacityDriver):
    """Tombstones the record while the hardware stays claimed."""

    def _release_hosts(self, cell_id: str, cleanup, *, revalidate) -> None:
        """Skip provider-side cleanup."""

        del cell_id, cleanup, revalidate


class RefusesToReplayARelease(OnPremSshCapacityDriver):
    """Cannot complete a release whose record is already gone."""

    def release(self, cell_id: str, context: MutationContext) -> CellObservation:
        """Fail every replay, so the command retries under its lease forever."""

        if self.state.get(cell_id) is None:
            raise CapacityProviderError(
                f"cell {cell_id!r} has no record to release",
                code=CapacityProviderFailureCode.CONFIGURATION,
                retryable=False,
            )
        return super().release(cell_id, context)


BROKEN_PROVIDERS: tuple[tuple[type[OnPremSshCapacityDriver], tuple[str, ...]], ...] = (
    (InventsACapacityClass, ("capabilities-are-routable",)),
    (
        ReportsAnUnknownCellAsUnobserved,
        # Two checks, honestly: "absent" is the same fact before a cell exists
        # and after it is released, and the controller reconciles forever
        # without it.
        ("unknown-cell-is-absent", "release-frees-the-substrate"),
    ),
    (
        ForgetsToClaim,
        # A fabricated claim also hides an unreachable provider during apply.
        (
            "apply-publishes-a-fenced-incarnation",
            "apply-is-idempotent-under-replay",
            "unreachable-provider-does-not-fake-capacity",
            "permanent-failure-is-conclusive",
        ),
    ),
    (ProvisionsAgainOnReplay, ("apply-is-idempotent-under-replay",)),
    (TrustsAnyFence, ("stale-generation-cannot-mutate", "stale-fencing-token-cannot-mutate")),
    (AdoptsWhateverIsThere, ("leaked-incarnation-is-not-adopted",)),
    (ReportsUnreachableAsAbsent, ("unreachable-provider-is-not-a-conclusion",)),
    (PretendsAnUnreachableClaimSucceeded, ("unreachable-provider-does-not-fake-capacity",)),
    (TreatsEveryFailureAsTransient, ("permanent-failure-is-conclusive",)),
    (LeavesMarkersBehind, ("release-frees-the-substrate", "release-is-idempotent")),
    (RefusesToReplayARelease, ("release-is-idempotent",)),
)


@pytest.mark.parametrize(
    ("factory", "expected"),
    BROKEN_PROVIDERS,
    ids=[factory.__name__ for factory, _ in BROKEN_PROVIDERS],
)
def test_the_suite_catches_a_deliberately_broken_provider(tmp_path, factory, expected) -> None:
    """Each break fails exactly the checks that defend the property it violates."""

    report = run_conformance(OnPremSshConformanceHarness(tmp_path, factory=factory))

    assert report.failed_checks == expected, report.describe()
    assert not report.passed
    assert all(result.detail for result in report.failures), "a failure must say what it observed"


def test_every_published_check_is_defeated_by_some_broken_provider() -> None:
    """A check nothing can fail is a check that is no longer asserting anything."""

    defeated = {name for _, expected in BROKEN_PROVIDERS for name in expected}

    assert defeated == set(CONFORMANCE_CHECK_NAMES)


def test_a_failing_report_names_the_property_and_raises_on_require(tmp_path) -> None:
    """The report is the artefact an external provider reads; it must be legible."""

    report = run_conformance(OnPremSshConformanceHarness(tmp_path, factory=AdoptsWhateverIsThere))

    assert "NEEDS_RECREATE" in report.describe()
    with pytest.raises(ConformanceViolation, match="leaked-incarnation-is-not-adopted"):
        report.require()


def test_a_provider_that_raises_an_unexpected_error_fails_that_check_only(tmp_path) -> None:
    """One exploding property must not hide the state of the other eleven."""

    class ExplodesOnDrain(OnPremSshCapacityDriver):
        def drain(self, cell_id: str, context: MutationContext) -> CellObservation:
            """Raise something nobody classified."""

            raise ZeroDivisionError("provider bug")

    report = run_conformance(OnPremSshConformanceHarness(tmp_path, factory=ExplodesOnDrain))

    assert report.failed_checks == ("stale-generation-cannot-mutate", "release-frees-the-substrate")
    failure = next(result for result in report.failures if result.check == "release-frees-the-substrate")
    assert "ZeroDivisionError" in failure.detail


def test_the_checks_and_the_pytest_mixin_cover_the_same_properties() -> None:
    """A property added to one and not the other would silently stop being run."""

    mixin_tests = {
        name.removeprefix("test_") for name in dir(CapacityProviderConformanceTests) if name.startswith("test_")
    }
    published = {check.name.replace("-", "_") for check in CONFORMANCE_CHECKS}

    assert mixin_tests == published


def test_a_report_with_no_results_is_not_silently_passing() -> None:
    """An empty run is a mistake, and require() must not bless it."""

    empty = ConformanceReport()
    failing = ConformanceReport(results=(ConformanceResult(check="x", passed=False, detail="why"),))

    assert empty.passed and empty.failed_checks == ()
    assert not failing.passed
    with pytest.raises(ConformanceViolation, match="why"):
        failing.require()
