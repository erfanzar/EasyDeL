# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The conformance suite every Eyvan capacity provider must pass.

``capacity/drivers/kubernetes.py`` says the thing this module exists to make
enforceable: it is deliberately a near-clone of ``manual.py`` because "the parts
that are hard to get right are backend-neutral". That is a claim about *other
people's* providers as much as Eyvan's own, and until now nothing could check
it -- an out-of-tree provider could implement the five methods of
:class:`~eyvan.capacity.drivers.base.CapacityDriver` and get every one of the
hard parts wrong while satisfying the type checker.

So this suite does not test that methods exist. It tests the four properties a
control plane actually depends on, each of which has a matching failure that has
happened in a real fleet:

1. **A stale fence cannot mutate.** A driver that accepts an older generation or
   a foreign fencing token lets a superseded controller resurrect capacity that
   another incarnation already owns. Both the durable record and the provider
   substrate must be untouched -- raising *after* mutating is the same bug with
   a better error message.
2. **Create is idempotent under replay.** The outbox retries; a driver that
   provisions again on replay leaks the first incarnation, and on backends that
   derive a worker token from the incarnation it also locks out every worker
   that was already running.
3. **A leaked incarnation is detected and never adopted.** Provider-side state
   that does not match the durable fence belongs to a previous incarnation.
   Adopting it publishes endpoints for machines this cell does not own; the
   only safe report is ``NEEDS_RECREATE``, with nothing published.
4. **An inconclusive call is not a conclusion.** A control API that could not be
   reached says nothing about the fleet. Reporting it as ``ABSENT`` deletes
   live capacity; reporting it as a permanent failure discards a command that
   should be retried; reporting a *permanent* failure as retryable holds a
   lease forever and makes the cell undeletable.

**How to use it.** Implement :class:`ProviderConformanceHarness` for your
provider -- six hooks, each of which is a fact about your substrate that only
you can produce -- and either subclass :class:`CapacityProviderConformanceTests`
in your pytest suite (one test per property, so a failure names the property) or
call :func:`run_conformance` and assert on the report.

**What it deliberately does not do.** It never touches your internals. Every
assertion goes through :class:`~eyvan.capacity.drivers.base.CapacityDriver` plus
your harness, because a suite that reached into a driver would be testing this
implementation rather than the contract, and would stop external providers from
being written differently -- which is the whole point of having an ABI.
"""

from __future__ import annotations

import typing as tp
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass

from eyvan._internal.json_types import JsonObject
from eyvan.capacity import (
    DEFINITIVE_CAPACITY_LOSS_STATUSES,
    READY_CAPACITY_STATUSES,
    CapacityStatus,
)
from eyvan.capacity.drivers.base import (
    CapacityDriver,
    CapacityProviderError,
    CapacityProviderFailureCode,
    CellObservation,
    MutationContext,
    StaleMutationError,
    canonical_execution_backend,
)

#: Capacity classes the scheduler routes on. A provider that invents a name here
#: does not fail loudly -- it simply never matches a pool, which is the kind of
#: bug that looks like "the provider is slow to provision".
KNOWN_CAPACITY_CLASSES = frozenset({"spot", "on-demand", "reserved", "guaranteed", "flex_start"})

#: Statuses that assert something conclusive about a cell. An observation that
#: could not reach the provider must not be one of these.
CONCLUSIVE_STATUSES = frozenset(DEFINITIVE_CAPACITY_LOSS_STATUSES | READY_CAPACITY_STATUSES)

#: Failure codes that mean "this will fail identically forever". A provider that
#: returns one of these must set ``retryable=False``, or the outbox backs off on
#: a command that can never succeed while its lease keeps the cell undeletable.
CONCLUSIVE_FAILURE_CODES = frozenset(
    {
        CapacityProviderFailureCode.CONFIGURATION,
        CapacityProviderFailureCode.QUOTA_EXCEEDED,
        CapacityProviderFailureCode.PROVIDER_ERROR,
    }
)


class ConformanceViolation(AssertionError):
    """One conformance property was not satisfied."""


@dataclass(frozen=True, slots=True)
class ConformanceResult:
    """The outcome of one conformance check."""

    check: str
    passed: bool
    detail: str = ""

    def describe(self) -> str:
        """Render a one-line result."""

        verdict = "PASS" if self.passed else "FAIL"
        return f"{verdict} {self.check}" + (f": {self.detail}" if self.detail else "")


@dataclass(frozen=True, slots=True)
class ConformanceReport:
    """Every conformance result from one run, passing and failing."""

    results: tuple[ConformanceResult, ...] = ()

    @property
    def passed(self) -> bool:
        """Return whether every check passed."""

        return all(result.passed for result in self.results)

    @property
    def failures(self) -> tuple[ConformanceResult, ...]:
        """Return only the failing results."""

        return tuple(result for result in self.results if not result.passed)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        """Return the names of the failing checks, in run order."""

        return tuple(result.check for result in self.failures)

    def require(self) -> None:
        """Raise if any check failed.

        Raises:
            ConformanceViolation: Listing every failure.
        """

        if self.passed:
            return
        detail = "\n".join(result.describe() for result in self.failures)
        raise ConformanceViolation(f"{len(self.failures)} capacity provider conformance failure(s):\n{detail}")

    def describe(self) -> str:
        """Render the whole report."""

        return "\n".join(result.describe() for result in self.results)


class ProviderConformanceHarness(ABC):
    """What a provider must be able to say about its own substrate.

    Six hooks. Each one exists because a property above cannot be checked from
    outside without it, and each is a *fact*, never a promise: a harness that
    fakes ``provider_incarnations`` is lying to its own test suite.
    """

    @abstractmethod
    def new_driver(self) -> CapacityDriver:
        """Return a fresh driver over an isolated substrate and durable store.

        Called once per check. Two drivers returned by two calls must not share
        state, or a check would observe another check's cells.
        """

    @abstractmethod
    def desired_capacity(self, cell_id: str) -> JsonObject:
        """Return a valid ``apply_capacity`` document for ``cell_id``."""

    @abstractmethod
    def provider_incarnations(self, driver: CapacityDriver, cell_id: str) -> int:
        """Return how many provider-side incarnations exist for ``cell_id``.

        This is the leak counter: VMs, Jobs, host claims -- whatever your
        provider creates that would keep costing money or holding hardware if
        Eyvan forgot about it. It must count the substrate, not the record.
        """

    @abstractmethod
    def leak_previous_incarnation(self, driver: CapacityDriver, cell_id: str) -> None:
        """Make the substrate hold an incarnation the durable fence disowns.

        Simulate what a crash between two mutations leaves behind: the resource
        of a previous generation, a foreign fencing token, or another cell's
        claim on the same hardware. The cell must not adopt it.
        """

    @abstractmethod
    def provider_unreachable(self, driver: CapacityDriver) -> AbstractContextManager[None]:
        """Make provider calls fail *inconclusively* inside the context.

        A refused connection, a timeout, a 503 -- anything after which the
        driver cannot know whether the call took effect.
        """

    @abstractmethod
    def provider_rejects_permanently(self, driver: CapacityDriver) -> AbstractContextManager[None]:
        """Make provider calls fail *conclusively* inside the context.

        Missing credentials, a namespace that does not exist, a manifest the
        API refuses. Retrying is guaranteed to fail identically.
        """

    # ---- optional description ------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Return the provider name used in conformance diagnostics."""

        return type(self).__name__


def context(key: str, *, generation: int = 1, token: str = "fence-1") -> MutationContext:
    """Return a mutation context for a conformance check."""

    return MutationContext(idempotency_key=key, generation=generation, fencing_token=token)


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ConformanceViolation(detail)


def _endpoints(observation: CellObservation) -> tuple[str, ...]:
    published = observation.metadata.get("native_worker_endpoints")
    endpoints = tuple(entry for entry in published if isinstance(entry, str)) if isinstance(published, list) else ()
    if observation.execution_endpoint:
        endpoints = (*endpoints, observation.execution_endpoint)
    return endpoints


def _provider_generation(observation: CellObservation) -> int | None:
    value = observation.metadata.get("provider_generation")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _unchanged(before: CellObservation, after: CellObservation, action: str) -> None:
    _require(
        before.generation == after.generation,
        f"{action} changed the durable generation from {before.generation} to {after.generation}",
    )
    _require(
        before.state == after.state,
        f"{action} changed the cell state from {before.state} to {after.state}",
    )
    _require(
        _provider_generation(before) == _provider_generation(after),
        f"{action} changed the provider generation",
    )


# --------------------------------------------------------------------------- checks


def check_capabilities_are_routable(harness: ProviderConformanceHarness) -> None:
    """Capabilities must use the vocabularies the scheduler routes on."""

    capabilities = harness.new_driver().capabilities()
    _require(bool(capabilities.provider.strip()), "capabilities() must name a provider")
    _require(
        capabilities.executor == canonical_execution_backend(capabilities.executor),
        f"executor {capabilities.executor!r} is not a canonical execution backend name",
    )
    _require(bool(capabilities.accelerator_types), "capabilities() must declare at least one accelerator type")
    _require(bool(capabilities.capacity_classes), "capabilities() must declare at least one capacity class")
    unknown = sorted(set(capabilities.capacity_classes) - KNOWN_CAPACITY_CLASSES)
    _require(
        not unknown,
        f"capacity class(es) {unknown} are outside the scheduler's vocabulary "
        f"{sorted(KNOWN_CAPACITY_CLASSES)}; a pool would never match them",
    )


def check_unknown_cell_is_absent(harness: ProviderConformanceHarness) -> None:
    """Observing a cell that was never applied is ABSENT, not an error."""

    observation = harness.new_driver().observe("never-applied")
    _require(
        observation.state is CapacityStatus.ABSENT,
        f"observing an unknown cell reported {observation.state}, expected ABSENT",
    )
    _require(observation.cell_id == "never-applied", "an observation must carry the cell it describes")


def check_apply_publishes_a_fenced_incarnation(harness: ProviderConformanceHarness) -> None:
    """Apply creates exactly one incarnation and reports it under the fence."""

    driver = harness.new_driver()
    observation = driver.apply_capacity("cell-a", harness.desired_capacity("cell-a"), context("apply", generation=7))
    _require(observation.cell_id == "cell-a", "apply_capacity must observe the cell it was given")
    _require(
        observation.generation == 7,
        f"apply_capacity reported generation {observation.generation}, expected the mutation's 7",
    )
    _require(
        observation.state not in DEFINITIVE_CAPACITY_LOSS_STATUSES,
        f"apply_capacity reported {observation.state}, which tells the controller the cell is gone",
    )
    for key in ("provider", "capacity_backend", "provider_generation"):
        _require(key in observation.metadata, f"an observation must carry {key!r} in its metadata")
    incarnations = harness.provider_incarnations(driver, "cell-a")
    _require(incarnations == 1, f"apply_capacity created {incarnations} provider incarnations, expected 1")


def check_apply_is_idempotent_under_replay(harness: ProviderConformanceHarness) -> None:
    """Replaying one mutation must not create a second incarnation."""

    driver = harness.new_driver()
    desired = harness.desired_capacity("cell-a")
    replayed = context("apply", generation=3)
    first = driver.apply_capacity("cell-a", desired, replayed)
    second = driver.apply_capacity("cell-a", desired, replayed)
    incarnations = harness.provider_incarnations(driver, "cell-a")
    _require(
        incarnations == 1,
        f"replaying one mutation left {incarnations} provider incarnations; the first one leaked",
    )
    _require(
        _provider_generation(first) == _provider_generation(second),
        "replaying one mutation bumped the provider generation, so every worker of the first "
        "incarnation is now fenced out by a token it cannot derive",
    )


def check_stale_generation_cannot_mutate(harness: ProviderConformanceHarness) -> None:
    """An older generation changes nothing, durably or on the provider."""

    driver = harness.new_driver()
    desired = harness.desired_capacity("cell-a")
    driver.apply_capacity("cell-a", desired, context("apply", generation=5))
    before = driver.observe("cell-a")
    incarnations = harness.provider_incarnations(driver, "cell-a")
    stale = context("stale", generation=4)
    for action, call in (
        ("apply_capacity", lambda: driver.apply_capacity("cell-a", desired, stale)),
        ("reconcile", lambda: driver.reconcile("cell-a", stale)),
        ("drain", lambda: driver.drain("cell-a", stale)),
        ("release", lambda: driver.release("cell-a", stale)),
    ):
        try:
            call()
        except StaleMutationError:
            pass
        else:
            raise ConformanceViolation(f"{action} accepted a generation older than the cell's")
        after = driver.observe("cell-a")
        _unchanged(before, after, f"a rejected stale {action}")
        now = harness.provider_incarnations(driver, "cell-a")
        _require(
            now == incarnations,
            f"a rejected stale {action} changed the provider from {incarnations} to {now} incarnations",
        )


def check_stale_fencing_token_cannot_mutate(harness: ProviderConformanceHarness) -> None:
    """A foreign fencing token at the same generation changes nothing."""

    driver = harness.new_driver()
    desired = harness.desired_capacity("cell-a")
    driver.apply_capacity("cell-a", desired, context("apply", generation=5))
    before = driver.observe("cell-a")
    incarnations = harness.provider_incarnations(driver, "cell-a")
    usurper = context("usurp", generation=5, token="someone-else")
    for action, call in (
        ("apply_capacity", lambda: driver.apply_capacity("cell-a", desired, usurper)),
        ("reconcile", lambda: driver.reconcile("cell-a", usurper)),
        ("release", lambda: driver.release("cell-a", usurper)),
    ):
        try:
            call()
        except StaleMutationError:
            pass
        else:
            raise ConformanceViolation(f"{action} accepted a fencing token that does not own the cell")
        after = driver.observe("cell-a")
        _unchanged(before, after, f"a rejected {action} with a foreign fencing token")
        now = harness.provider_incarnations(driver, "cell-a")
        _require(
            now == incarnations,
            f"a rejected {action} with a foreign token changed the provider substrate",
        )


def check_leaked_incarnation_is_not_adopted(harness: ProviderConformanceHarness) -> None:
    """Provider state that the fence disowns is reported, never adopted."""

    driver = harness.new_driver()
    driver.apply_capacity("cell-a", harness.desired_capacity("cell-a"), context("apply"))
    harness.leak_previous_incarnation(driver, "cell-a")
    observation = driver.observe("cell-a")
    _require(
        observation.state is CapacityStatus.NEEDS_RECREATE,
        f"a leaked incarnation was reported as {observation.state}; NEEDS_RECREATE is the only "
        "state that tells the controller to replace it rather than schedule onto it",
    )
    published = _endpoints(observation)
    _require(
        not published,
        f"a leaked incarnation published execution endpoints {published}; work would be sent to "
        "machines this cell does not own",
    )


def check_unreachable_provider_is_not_a_conclusion(harness: ProviderConformanceHarness) -> None:
    """An observation that could not reach the provider concludes nothing."""

    driver = harness.new_driver()
    driver.apply_capacity("cell-a", harness.desired_capacity("cell-a"), context("apply"))
    with harness.provider_unreachable(driver):
        try:
            observation = driver.observe("cell-a")
        except CapacityProviderError as exc:
            _require(
                exc.retryable,
                f"an unreachable provider raised a non-retryable {exc.code}; the command would be "
                "discarded although nothing was learned",
            )
            _require(
                exc.code is CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
                f"an unreachable provider raised {exc.code}, which the availability belief reads as "
                "a fact about the pool rather than about the link",
            )
            return
        except Exception as exc:  # any other exception type is itself the violation
            raise ConformanceViolation(
                f"an unreachable provider raised {type(exc).__name__}, which carries no retryability "
                "or failure code, so the controller cannot tell it from a real failure"
            ) from exc
    _require(
        observation.state not in CONCLUSIVE_STATUSES,
        f"an unreachable provider reported {observation.state}; that is a conclusion about capacity nothing observed",
    )
    _require(
        observation.failure_code is CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
        "an inconclusive observation must say why: failure_code PROVIDER_UNAVAILABLE",
    )
    _require(not _endpoints(observation), "an unobserved cell must not publish execution endpoints")


def check_unreachable_provider_does_not_fake_capacity(harness: ProviderConformanceHarness) -> None:
    """A mutation that could not be established is retryable and claims nothing."""

    driver = harness.new_driver()
    with harness.provider_unreachable(driver):
        try:
            driver.apply_capacity("cell-a", harness.desired_capacity("cell-a"), context("apply"))
        except CapacityProviderError as exc:
            _require(
                exc.retryable and exc.code is CapacityProviderFailureCode.PROVIDER_UNAVAILABLE,
                f"an unreachable provider failed apply with {exc.code} retryable={exc.retryable}; "
                "an unestablished call must stay retryable and must not blame the pool",
            )
        except Exception as exc:  # any other exception type is itself the violation
            raise ConformanceViolation(
                f"an unreachable provider failed apply with {type(exc).__name__} instead of a "
                "classified CapacityProviderError"
            ) from exc
        else:
            raise ConformanceViolation("apply_capacity reported success while the provider was unreachable")
    if harness.provider_incarnations(driver, "cell-a") == 0:
        observation = driver.observe("cell-a")
        _require(
            observation.state not in READY_CAPACITY_STATUSES,
            f"apply_capacity failed and created nothing, yet the cell reports {observation.state}; "
            "the scheduler would place work on capacity that does not exist",
        )


def check_permanent_failure_is_conclusive(harness: ProviderConformanceHarness) -> None:
    """A failure that retrying cannot fix must say so."""

    driver = harness.new_driver()
    with harness.provider_rejects_permanently(driver):
        try:
            driver.apply_capacity("cell-a", harness.desired_capacity("cell-a"), context("apply"))
        except CapacityProviderError as exc:
            _require(
                not exc.retryable,
                f"a permanent failure was reported as retryable ({exc.code}); the outbox would back "
                "off forever while its lease keeps the cell undeletable",
            )
            _require(
                exc.code in CONCLUSIVE_FAILURE_CODES,
                f"a permanent failure was reported as {exc.code}, which the controller retries",
            )
            return
        except Exception as exc:  # any other exception type is itself the violation
            raise ConformanceViolation(
                f"a permanently failing provider raised {type(exc).__name__} instead of a classified "
                "CapacityProviderError, so nothing can decide whether to retry it"
            ) from exc
    raise ConformanceViolation("apply_capacity reported success while the provider was permanently refusing")


def check_release_frees_the_substrate(harness: ProviderConformanceHarness) -> None:
    """Drain keeps the cell releasable; release leaves nothing behind."""

    driver = harness.new_driver()
    driver.apply_capacity("cell-a", harness.desired_capacity("cell-a"), context("apply"))
    drained = driver.drain("cell-a", context("drain"))
    _require(
        drained.state is CapacityStatus.DRAINING,
        f"drain reported {drained.state}, expected DRAINING",
    )
    released = driver.release("cell-a", context("release", generation=2))
    _require(
        released.state is CapacityStatus.RELEASED,
        f"release reported {released.state}, expected RELEASED",
    )
    remaining = harness.provider_incarnations(driver, "cell-a")
    _require(remaining == 0, f"release left {remaining} provider incarnations behind")
    observation = driver.observe("cell-a")
    _require(
        observation.state in DEFINITIVE_CAPACITY_LOSS_STATUSES,
        f"a released cell observes as {observation.state}; the controller would keep reconciling it",
    )


def check_release_is_idempotent(harness: ProviderConformanceHarness) -> None:
    """Replaying a release succeeds instead of holding the command forever."""

    driver = harness.new_driver()
    driver.apply_capacity("cell-a", harness.desired_capacity("cell-a"), context("apply"))
    release = context("release", generation=2)
    driver.release("cell-a", release)
    replayed = driver.release("cell-a", release)
    _require(
        replayed.state is CapacityStatus.RELEASED,
        f"replaying a release reported {replayed.state}; a release command that cannot complete "
        "twice holds its lease and keeps the cell undeletable",
    )
    remaining = harness.provider_incarnations(driver, "cell-a")
    _require(remaining == 0, f"a replayed release left {remaining} provider incarnations behind")


@dataclass(frozen=True, slots=True)
class ConformanceCheck:
    """One named conformance property."""

    name: str
    run: tp.Callable[[ProviderConformanceHarness], None]

    @property
    def description(self) -> str:
        """Return the property this check defends, from the check's docstring."""

        summary = (self.run.__doc__ or "").strip().splitlines()
        return summary[0] if summary else self.name


CONFORMANCE_CHECKS: tuple[ConformanceCheck, ...] = (
    ConformanceCheck("capabilities-are-routable", check_capabilities_are_routable),
    ConformanceCheck("unknown-cell-is-absent", check_unknown_cell_is_absent),
    ConformanceCheck("apply-publishes-a-fenced-incarnation", check_apply_publishes_a_fenced_incarnation),
    ConformanceCheck("apply-is-idempotent-under-replay", check_apply_is_idempotent_under_replay),
    ConformanceCheck("stale-generation-cannot-mutate", check_stale_generation_cannot_mutate),
    ConformanceCheck("stale-fencing-token-cannot-mutate", check_stale_fencing_token_cannot_mutate),
    ConformanceCheck("leaked-incarnation-is-not-adopted", check_leaked_incarnation_is_not_adopted),
    ConformanceCheck("unreachable-provider-is-not-a-conclusion", check_unreachable_provider_is_not_a_conclusion),
    ConformanceCheck("unreachable-provider-does-not-fake-capacity", check_unreachable_provider_does_not_fake_capacity),
    ConformanceCheck("permanent-failure-is-conclusive", check_permanent_failure_is_conclusive),
    ConformanceCheck("release-frees-the-substrate", check_release_frees_the_substrate),
    ConformanceCheck("release-is-idempotent", check_release_is_idempotent),
)

CONFORMANCE_CHECK_NAMES: tuple[str, ...] = tuple(check.name for check in CONFORMANCE_CHECKS)


def run_conformance(
    harness: ProviderConformanceHarness,
    *,
    checks: tp.Sequence[ConformanceCheck] = CONFORMANCE_CHECKS,
) -> ConformanceReport:
    """Run the conformance suite and report every outcome.

    Args:
        harness: The provider's harness.
        checks: The checks to run, defaulting to all of them.

    Returns:
        A report with one result per check. Unexpected exceptions are failures
        of the check that raised them, not of the run: a provider that crashes
        one property must still be measured on the rest.
    """

    results: list[ConformanceResult] = []
    for check in checks:
        try:
            check.run(harness)
        except ConformanceViolation as exc:
            results.append(ConformanceResult(check=check.name, passed=False, detail=str(exc)))
        except Exception as exc:  # a provider may raise anything
            results.append(
                ConformanceResult(
                    check=check.name,
                    passed=False,
                    detail=f"raised {type(exc).__name__}: {exc}",
                )
            )
        else:
            results.append(ConformanceResult(check=check.name, passed=True))
    return ConformanceReport(results=tuple(results))


class CapacityProviderConformanceTests:
    """Pytest mix-in: one test per property, so a failure names the property.

    Subclass it in your provider's test suite and implement :meth:`harness`::

        class TestAcmeConformance(CapacityProviderConformanceTests):
            def harness(self) -> ProviderConformanceHarness:
                return AcmeHarness()

    Collected by pytest because the class name starts with ``Test`` in *your*
    module; this base class deliberately does not, so importing it does not run
    the suite against nothing.
    """

    def harness(self) -> ProviderConformanceHarness:
        """Return the harness under test."""

        raise NotImplementedError("a conformance test class must implement harness()")

    def test_capabilities_are_routable(self) -> None:
        """Capabilities use the vocabularies the scheduler routes on."""

        check_capabilities_are_routable(self.harness())

    def test_unknown_cell_is_absent(self) -> None:
        """An unknown cell observes as ABSENT rather than raising."""

        check_unknown_cell_is_absent(self.harness())

    def test_apply_publishes_a_fenced_incarnation(self) -> None:
        """Apply creates one incarnation and reports it under the fence."""

        check_apply_publishes_a_fenced_incarnation(self.harness())

    def test_apply_is_idempotent_under_replay(self) -> None:
        """A replayed mutation does not create a second incarnation."""

        check_apply_is_idempotent_under_replay(self.harness())

    def test_stale_generation_cannot_mutate(self) -> None:
        """An older generation changes nothing, durably or on the provider."""

        check_stale_generation_cannot_mutate(self.harness())

    def test_stale_fencing_token_cannot_mutate(self) -> None:
        """A foreign fencing token changes nothing."""

        check_stale_fencing_token_cannot_mutate(self.harness())

    def test_leaked_incarnation_is_not_adopted(self) -> None:
        """A leaked incarnation is reported NEEDS_RECREATE and not adopted."""

        check_leaked_incarnation_is_not_adopted(self.harness())

    def test_unreachable_provider_is_not_a_conclusion(self) -> None:
        """An unreachable provider yields UNKNOWN, never a conclusion."""

        check_unreachable_provider_is_not_a_conclusion(self.harness())

    def test_unreachable_provider_does_not_fake_capacity(self) -> None:
        """An unestablished mutation is retryable and claims no capacity."""

        check_unreachable_provider_does_not_fake_capacity(self.harness())

    def test_permanent_failure_is_conclusive(self) -> None:
        """A permanent failure is reported as non-retryable."""

        check_permanent_failure_is_conclusive(self.harness())

    def test_release_frees_the_substrate(self) -> None:
        """Release leaves no provider incarnation behind."""

        check_release_frees_the_substrate(self.harness())

    def test_release_is_idempotent(self) -> None:
        """A replayed release completes instead of holding its lease."""

        check_release_is_idempotent(self.harness())


__all__ = (
    "CONCLUSIVE_FAILURE_CODES",
    "CONCLUSIVE_STATUSES",
    "CONFORMANCE_CHECKS",
    "CONFORMANCE_CHECK_NAMES",
    "KNOWN_CAPACITY_CLASSES",
    "CapacityProviderConformanceTests",
    "ConformanceCheck",
    "ConformanceReport",
    "ConformanceResult",
    "ConformanceViolation",
    "ProviderConformanceHarness",
    "context",
    "run_conformance",
)
