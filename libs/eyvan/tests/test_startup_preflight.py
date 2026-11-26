# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Standing up a network-bound head must not be a crash-fix-restart loop.

Four settings a network bind requires -- ``EYVAN_TLS_CERT_FILE`` (with
``EYVAN_TLS_KEY_FILE``, or an explicit insecure opt-in), ``EYVAN_REPOSITORY_BACKEND``,
``EYVAN_METRICS_TOKEN``, ``EYVAN_ARTIFACT_TOKEN_KEY`` -- were each discovered by
reading a traceback, fixing the one variable it named, restarting, and meeting
the next. Every one of those refusals is correct and each explains itself well.
The defect was that they were serialized, on a machine that was already
provisioned and billing while the operator went around the loop three times.

These tests hold two things:

* the server names all of them at once, with every original explanation intact;
* ``eyvan doctor`` reports the same set, from the same function, before anything
  has been started -- and fails if the two ever diverge.
"""

from __future__ import annotations

import typing as tp

import pytest
from click.testing import CliRunner
from test_doctor import _healthy, _Store

from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.commands.doctor import (
    CALLABLE_TRUST_CHECK,
    SERVER_STARTUP_CHECK,
    CheckStatus,
    DoctorCommandService,
    doctor,
)
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.contexts import EyvanContextConfig
from eyvan.server.api.server_bind import ServerBindPolicy
from eyvan.server.api.server_startup import (
    ServerStartupSettings,
    StartupConfigurationSource,
    unsatisfied_startup_requirements,
)

#: Every variable that decides whether a network bind is startable. Cleared
#: before each case so a developer's own exported environment cannot decide the
#: outcome of an assertion about which settings are missing.
_STARTUP_VARIABLES = (
    "EYVAN_TLS_CERT_FILE",
    "EYVAN_TLS_KEY_FILE",
    "EYVAN_INSECURE_PLAINTEXT_BIND",
    "EYVAN_REPOSITORY_BACKEND",
    "EYVAN_METRICS_TOKEN",
    "EYVAN_ARTIFACT_TOKEN_KEY",
    "EYVAN_DEV_AUTH_ENABLED",
)

_NETWORK = ServerBindPolicy.from_host("0.0.0.0")
_LOOPBACK = ServerBindPolicy.from_host("127.0.0.1")

#: The refusals as they were written, copied here rather than imported. An
#: imported constant would follow a rewrite and assert nothing; these fail if a
#: single explanation is shortened, softened, or dropped in the collection.
_POSTGRES_SENTENCE = (
    "a network-bound Eyvan server requires EYVAN_REPOSITORY_BACKEND=postgres so every replica "
    "shares durable control-plane state"
)
_TLS_SENTENCE = (
    "a network-bound Eyvan server refuses to serve plaintext HTTP: OIDC bearer tokens, bootstrap "
    "credentials, and capability tokens would cross the network unencrypted. Set EYVAN_TLS_CERT_FILE "
    "and EYVAN_TLS_KEY_FILE to serve TLS, or set EYVAN_INSECURE_PLAINTEXT_BIND=1 (or pass "
    "--insecure-plaintext to eyvan-server) to keep the pre-0.1 plaintext behavior as an explicit, "
    "reviewed decision"
)
_METRICS_SENTENCE = (
    "a network-bound Eyvan server requires EYVAN_METRICS_TOKEN so /metrics does not expose "
    "per-tenant spend to anyone who can reach the port"
)
_ARTIFACT_SENTENCE = (
    "EYVAN_ARTIFACT_TOKEN_KEY must be set when the Eyvan repository backend is not local; an ephemeral "
    "per-process signing key would break managed-artifact token validation across replicas and restarts"
)
_DEV_AUTH_SENTENCE = (
    "Eyvan development authentication may only be used with a literal loopback server bind; "
    "disable it or configure OIDC for a network-bound deployment"
)


@pytest.fixture(autouse=True)
def _clean_startup_environment(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Start every case from an environment that configures nothing."""

    for variable in _STARTUP_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    # The unrelated checks in a full doctor run must not depend on what this
    # machine has staged or is running.
    monkeypatch.setattr("eyvan.sdk.commands.doctor.stale_components", lambda **_kwargs: ())
    monkeypatch.setattr("eyvan.sdk.commands.doctor_preflights.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "eyvan.sdk.commands.doctor_preflights.DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH",
        tmp_path / "no-runtime-profile.json",
    )
    monkeypatch.setenv("EYVAN_ARTIFACT_STORE_URI", str(tmp_path))


def _configure(monkeypatch: pytest.MonkeyPatch, environment: dict[str, str]) -> None:
    for name, value in environment.items():
        monkeypatch.setenv(name, value)


def _session(handler: tp.Callable[[str], tp.Any] | None = None) -> CliSession:
    """A session whose control plane answers everything, so only the new check can fail."""

    return CliSession(
        client=_FakeClient(handler or _healthy()),
        identity=ClientIdentity(),
        context_store=_Store(),
        context_config=EyvanContextConfig(),
        base_url="http://test",
        org_id="acme",
        project_id="research",
        overrides=ContextOverrides(),
    )


class _FakeClient:
    """Answers the reads doctor's other checks make; this module asserts the new one."""

    def __init__(self, handler: tp.Callable[[str], tp.Any]) -> None:
        self._handler = handler

    def request(self, method: str, path: str, **_kwargs: tp.Any) -> tp.Any:
        del method
        return self._handler(path)


def _requirement_names(text: str) -> tuple[str, ...]:
    """Parse the ``  - NAME: why`` bullets out of whatever a surface actually emitted.

    Deliberately reads the rendered text rather than calling the shared function
    a second time: this is what an operator sees, and a surface that stopped
    using the shared source of truth would emit something else.
    """

    return tuple(line.removeprefix("  - ").split(":", 1)[0] for line in text.splitlines() if line.startswith("  - "))


def _server_refusal(host: str = "0.0.0.0") -> str:
    """Return what the server refuses to start with, or ``""`` when it starts."""

    try:
        ServerStartupSettings.from_environment(ServerBindPolicy.from_host(host))
    except ValueError as error:
        return str(error)
    return ""


def _doctor_check(host: str = "0.0.0.0", handler: tp.Callable[[str], tp.Any] | None = None):
    report = DoctorCommandService(_session(handler), server_bind_host=host).report()
    return next(check for check in report.checks if check.name == SERVER_STARTUP_CHECK)


def test_a_network_bind_names_every_missing_setting_in_one_refusal() -> None:
    """The whole point: one traceback, every variable, no restart between them."""

    message = _server_refusal()

    assert _requirement_names(message) == (
        "EYVAN_TLS_CERT_FILE",
        "EYVAN_REPOSITORY_BACKEND",
        "EYVAN_METRICS_TOKEN",
        "EYVAN_ARTIFACT_TOKEN_KEY",
    )
    assert "4 unsatisfied requirements" in message


def test_every_explanation_is_preserved_word_for_word() -> None:
    """Collecting the refusals must not summarize any of them.

    Each sentence is the only place the reason for that setting is written down;
    an operator deciding whether to set a metrics token reads this and nothing
    else.
    """

    message = _server_refusal()

    assert _TLS_SENTENCE in message
    assert _POSTGRES_SENTENCE in message
    assert _METRICS_SENTENCE in message
    assert _ARTIFACT_SENTENCE in message


def test_the_artifact_key_is_reported_before_the_backend_is_corrected() -> None:
    """The exact serialization: the key requirement used to appear only later.

    ``EYVAN_ARTIFACT_TOKEN_KEY`` is refused when the backend is not local, and a
    network bind must be Postgres -- so it is required from the first attempt,
    not from the attempt after the backend was fixed. Reporting it only once the
    backend already said ``postgres`` is what cost a restart.
    """

    settings = EyvanSettings.from_env()

    assert settings.repository_backend is RepositoryBackend.MEMORY
    names = tuple(
        requirement.environment_variable
        for requirement in unsatisfied_startup_requirements(
            _NETWORK, settings, source=StartupConfigurationSource.ENVIRONMENT
        )
    )
    assert "EYVAN_ARTIFACT_TOKEN_KEY" in names


def test_fixing_one_setting_never_reveals_a_setting_that_was_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """Walk the loop that was actually walked; it must strictly shrink.

    Each step fixes exactly what the previous refusal named. If any requirement
    were still discovered late, a later message would name a variable an earlier
    one did not.
    """

    seen: list[tuple[str, ...]] = [_requirement_names(_server_refusal())]

    # The TLS refusal names EYVAN_TLS_CERT_FILE; its own explanation says the
    # key file is needed too, so the walk sets both halves at this step.
    monkeypatch.setenv("EYVAN_TLS_CERT_FILE", "/tmp/eyvan-walk-cert.pem")
    monkeypatch.setenv("EYVAN_TLS_KEY_FILE", "/tmp/eyvan-walk-key.pem")
    seen.append(_requirement_names(_server_refusal()))
    monkeypatch.setenv("EYVAN_REPOSITORY_BACKEND", "postgres")
    seen.append(_requirement_names(_server_refusal()))
    monkeypatch.setenv("EYVAN_METRICS_TOKEN", "scrape-token")
    seen.append(_requirement_names(_server_refusal()))
    monkeypatch.setenv("EYVAN_ARTIFACT_TOKEN_KEY", "k" * 32)

    assert seen == [
        ("EYVAN_TLS_CERT_FILE", "EYVAN_REPOSITORY_BACKEND", "EYVAN_METRICS_TOKEN", "EYVAN_ARTIFACT_TOKEN_KEY"),
        ("EYVAN_REPOSITORY_BACKEND", "EYVAN_METRICS_TOKEN", "EYVAN_ARTIFACT_TOKEN_KEY"),
        ("EYVAN_METRICS_TOKEN", "EYVAN_ARTIFACT_TOKEN_KEY"),
        ("EYVAN_ARTIFACT_TOKEN_KEY",),
    ]
    assert _server_refusal() == ""


def test_a_single_unsatisfied_requirement_still_explains_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """One missing setting reads as a sentence, not as a list of one with no reason."""

    _configure(
        monkeypatch,
        {
            "EYVAN_TLS_CERT_FILE": "/tmp/eyvan-single-cert.pem",
            "EYVAN_TLS_KEY_FILE": "/tmp/eyvan-single-key.pem",
            "EYVAN_REPOSITORY_BACKEND": "postgres",
            "EYVAN_METRICS_TOKEN": "scrape-token",
            "EYVAN_ARTIFACT_TOKEN_KEY": "k" * 32,
            "EYVAN_DEV_AUTH_ENABLED": "1",
        },
    )

    message = _server_refusal()

    assert "1 unsatisfied requirement:" in message
    assert _DEV_AUTH_SENTENCE in message
    assert _requirement_names(message) == ("EYVAN_DEV_AUTH_ENABLED",)


_ENVIRONMENTS: tuple[tuple[str, dict[str, str]], ...] = (
    ("nothing configured", {}),
    ("backend only", {"EYVAN_REPOSITORY_BACKEND": "postgres"}),
    ("backend and metrics", {"EYVAN_REPOSITORY_BACKEND": "postgres", "EYVAN_METRICS_TOKEN": "scrape-token"}),
    ("wrong backend", {"EYVAN_REPOSITORY_BACKEND": "local", "EYVAN_METRICS_TOKEN": "scrape-token"}),
    (
        "development auth left on",
        {
            "EYVAN_REPOSITORY_BACKEND": "postgres",
            "EYVAN_METRICS_TOKEN": "scrape-token",
            "EYVAN_ARTIFACT_TOKEN_KEY": "k" * 32,
            "EYVAN_DEV_AUTH_ENABLED": "true",
        },
    ),
    (
        "tls only",
        {
            "EYVAN_TLS_CERT_FILE": "/tmp/eyvan-tls-cert.pem",
            "EYVAN_TLS_KEY_FILE": "/tmp/eyvan-tls-key.pem",
        },
    ),
    (
        "fully configured",
        {
            "EYVAN_TLS_CERT_FILE": "/tmp/eyvan-tls-cert.pem",
            "EYVAN_TLS_KEY_FILE": "/tmp/eyvan-tls-key.pem",
            "EYVAN_REPOSITORY_BACKEND": "postgres",
            "EYVAN_METRICS_TOKEN": "scrape-token",
            "EYVAN_ARTIFACT_TOKEN_KEY": "k" * 32,
        },
    ),
    (
        "explicit insecure opt-in",
        {
            "EYVAN_INSECURE_PLAINTEXT_BIND": "1",
            "EYVAN_REPOSITORY_BACKEND": "postgres",
            "EYVAN_METRICS_TOKEN": "scrape-token",
            "EYVAN_ARTIFACT_TOKEN_KEY": "k" * 32,
        },
    ),
)


@pytest.mark.parametrize(
    "environment",
    [environment for _name, environment in _ENVIRONMENTS],
    ids=[name for name, _environment in _ENVIRONMENTS],
)
def test_doctor_reports_exactly_what_the_server_refuses_on(
    monkeypatch: pytest.MonkeyPatch, environment: dict[str, str]
) -> None:
    """The divergence guard, read off both surfaces' emitted text.

    A second hand-maintained list in ``doctor`` would answer some of these
    environments differently, and the failure mode of that is the worst one
    available here: a preflight that says ready for a server that will refuse.
    """

    _configure(monkeypatch, environment)

    check = _doctor_check()
    refusal = _server_refusal()

    assert _requirement_names(check.remedy) == _requirement_names(refusal)
    assert (check.status is CheckStatus.FAIL) is bool(refusal)


def test_doctor_names_the_settings_in_its_one_line_detail() -> None:
    """The detail is what a terminal shows first; it has to be the whole set."""

    check = _doctor_check()

    assert check.status is CheckStatus.FAIL
    for name in (
        "EYVAN_TLS_CERT_FILE",
        "EYVAN_REPOSITORY_BACKEND",
        "EYVAN_METRICS_TOKEN",
        "EYVAN_ARTIFACT_TOKEN_KEY",
    ):
        assert name in check.detail
    assert "0.0.0.0" in check.detail


def test_doctor_carries_the_same_explanations_the_server_would_print() -> None:
    """Asking beforehand must not be a weaker answer than being refused."""

    check = _doctor_check()

    assert _TLS_SENTENCE in check.remedy
    assert _POSTGRES_SENTENCE in check.remedy
    assert _METRICS_SENTENCE in check.remedy
    assert _ARTIFACT_SENTENCE in check.remedy


def test_a_loopback_preflight_asks_for_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local development requires none of this, and doctor must not pretend otherwise.

    The zero-configuration loopback path is the reason these settings can be
    required at all: nothing about `eyvan up` on a laptop changes here.
    """

    check = _doctor_check(host="127.0.0.1")

    assert check.status is CheckStatus.PASS
    assert check.remedy == ""
    assert (
        unsatisfied_startup_requirements(
            _LOOPBACK, EyvanSettings.from_env(), source=StartupConfigurationSource.ENVIRONMENT
        )
        == ()
    )
    assert ServerStartupSettings.from_environment(_LOOPBACK).settings.repository_backend is RepositoryBackend.LOCAL


def test_the_default_bind_is_the_one_the_server_uses() -> None:
    """``eyvan doctor`` with no flag must answer for ``eyvan-server`` with no flag."""

    default_check = DoctorCommandService(_session()).report().checks[0]

    assert default_check.name == SERVER_STARTUP_CHECK
    assert default_check.status is CheckStatus.PASS
    assert "127.0.0.1" in default_check.detail


def test_the_check_runs_before_a_control_plane_exists() -> None:
    """The moment it is needed is the moment nothing is listening yet."""

    def unreachable(_path: str) -> tp.Any:
        raise ConnectionRefusedError("connection refused")

    report = DoctorCommandService(_session(unreachable), server_bind_host="0.0.0.0").report()

    startup = next(check for check in report.checks if check.name == SERVER_STARTUP_CHECK)
    assert report.checks[0].name == SERVER_STARTUP_CHECK
    assert startup.status is CheckStatus.FAIL
    assert _requirement_names(startup.remedy) == _requirement_names(_server_refusal())


def test_a_missing_setting_exits_non_zero_even_when_everything_else_passes() -> None:
    """Exit code is the only part of this a script can read."""

    with pytest.raises(SystemExit) as raised:
        DoctorCommandService(_session(), server_bind_host="0.0.0.0").run()

    assert raised.value.code == 1
    report = DoctorCommandService(_session(), server_bind_host="0.0.0.0").report()
    assert [check.name for check in report.checks if check.status is CheckStatus.FAIL] == [SERVER_STARTUP_CHECK]


def test_the_command_takes_the_bind_address_as_an_option() -> None:
    """The check is reachable from the surface an operator actually has."""

    result = CliRunner().invoke(doctor, ["--server-host", "0.0.0.0"], obj=_session())

    assert result.exit_code == 1
    assert "EYVAN_ARTIFACT_TOKEN_KEY" in result.stdout

    passing = CliRunner().invoke(doctor, [], obj=_session())

    assert passing.exit_code == 0


def test_a_network_bind_with_missing_settings_still_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing here relaxes a requirement; collecting them is not forgiving them."""

    _configure(monkeypatch, {"EYVAN_REPOSITORY_BACKEND": "postgres", "EYVAN_ARTIFACT_TOKEN_KEY": "k" * 32})

    with pytest.raises(ValueError, match="EYVAN_METRICS_TOKEN"):
        ServerStartupSettings.from_environment(_NETWORK)


def test_a_network_bind_that_trusts_no_project_warns_without_turning_the_run_red(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The silent half of standing a head up, given a surface before it bites.

    A plane that trusts no project serves every command workload correctly, so
    this is deliberately not a startup requirement -- refusing there would break
    deployments the setting does not concern, and `unsatisfied_startup_requirements`
    is exactly the list the server refuses to start without.

    What it must not do is stay invisible. Without this, the whole `@execute` API
    authenticates and is then refused with a 403 naming the workload rather than
    the unset variable, and nothing anywhere says so first.
    """

    _configure(
        monkeypatch,
        {
            "EYVAN_TLS_CERT_FILE": "/tmp/eyvan-trust-cert.pem",
            "EYVAN_TLS_KEY_FILE": "/tmp/eyvan-trust-key.pem",
            "EYVAN_REPOSITORY_BACKEND": "postgres",
            "EYVAN_ARTIFACT_TOKEN_KEY": "k" * 32,
            "EYVAN_METRICS_TOKEN": "m" * 32,
        },
    )

    report = DoctorCommandService(_session(), server_bind_host="0.0.0.0").report()
    trust = next(check for check in report.checks if check.name == CALLABLE_TRUST_CHECK)

    assert trust.status is CheckStatus.WARN
    assert "403" in trust.detail
    assert "EYVAN_TRUSTED_CALLABLE_PROJECTS" in (trust.remedy or "")
    # A warning never turns a healthy environment red: an operator running only
    # command workloads is not misconfigured.
    assert report.ok is True

    monkeypatch.setenv("EYVAN_TRUSTED_CALLABLE_PROJECTS", "acme/research")
    granted = DoctorCommandService(_session(), server_bind_host="0.0.0.0").report()
    named = next(check for check in granted.checks if check.name == CALLABLE_TRUST_CHECK)

    assert named.status is CheckStatus.PASS
    assert "acme/research" in named.detail


def test_a_malformed_environment_is_reported_rather_than_crashing_the_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`doctor` must survive the exact condition it exists to diagnose.

    `EyvanSettings` is a plain frozen dataclass and raises `ValueError`; the
    callable-trust check caught pydantic's `ValidationError`, which it can never
    raise. So a bad EYVAN_* value crashed the whole preflight with a traceback
    instead of reporting which setting was wrong -- in the one command an
    operator runs precisely because something is misconfigured.
    """

    _configure(monkeypatch, {"EYVAN_OUTBOX_CLAIM_TTL_S": "-5"})

    report = DoctorCommandService(_session(), server_bind_host="0.0.0.0").report()

    startup = next(check for check in report.checks if check.name == SERVER_STARTUP_CHECK)
    trust = next(check for check in report.checks if check.name == CALLABLE_TRUST_CHECK)

    assert startup.status is CheckStatus.FAIL
    assert "does not parse" in startup.detail
    # Skipped, not crashed, and distinct from "no project is trusted".
    assert trust.status is CheckStatus.SKIP
    assert report.ok is False
