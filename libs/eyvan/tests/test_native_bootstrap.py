# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import base64
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from eyvan.capacity.drivers.base import CapacityStatus, CellObservation
from eyvan.records import NativeContainerConfig, NativeContainerMount
from eyvan.worker.runtime import ConfiguredExecutionResolver, ExecutionRuntimeRegistry
from eyvan.worker.runtime.native_bootstrap import build_native_worker_startup_script, derive_native_worker_token
from eyvan.worker.runtime.native_slice import JaxClusterDetectionMode, NativeSliceExecutor

_SHA256 = "a" * 64


def test_compute_bootstrap_fetches_a_cell_scoped_token_from_secret_manager_and_runs_fleet_setup() -> None:
    token = derive_native_worker_token("agent-global-secret", "cell-uid")
    secret_resource = "projects/project-a/secrets/eyvan-native-cell-uid/versions/latest"
    script = build_native_worker_startup_script(
        package="gs://artifacts/eyvan/eyvan.whl",
        package_sha256=_SHA256,
        package_generation="1700000000000000",
        port=8123,
        state_dir="/var/lib/eyvan/worker",
        worker_secret_resource=secret_resource,
        wheelhouse="gs://artifacts/eyvan/wheelhouse.zip",
        wheelhouse_sha256=_SHA256,
        wheelhouse_generation="1700000000000001",
        before_start="echo setup-runtime",
    )

    assert token is not None
    assert "agent-global-secret" not in script
    assert token not in script
    assert secret_resource in script
    assert "gcloud secrets versions access" in script
    assert script.index("echo setup-runtime") < script.index("gcloud storage cp")
    assert "gcloud storage cp 'gs://artifacts/eyvan/eyvan.whl#1700000000000000'" in script
    assert "gcloud storage cp 'gs://artifacts/eyvan/wheelhouse.zip#1700000000000001'" in script
    assert script.count("sha256sum --check --strict -") == 2
    assert script.index("eyvan.whl#1700000000000000") < script.index("sha256sum --check --strict -")
    assert script.index("sha256sum --check --strict -") < script.index("pip install")
    assert "/opt/eyvan-native/venv/bin/python -m pip install" in script
    assert "/var/lib/eyvan/bootstrap/eyvan.whl" in script
    assert "eyvan-worker.whl" not in script
    assert "gcloud storage cp 'gs://artifacts/eyvan/wheelhouse.zip#1700000000000001'" in script
    assert "--no-index --find-links /var/lib/eyvan/bootstrap/wheels" in script
    assert "eyvan.worker.runtime.native_worker --host 0.0.0.0 --port 8123" in script
    assert "--secret-file /run/eyvan-native/worker-secret" in script
    assert "--systemd-transient" in script
    assert "EYVAN_SYSTEMD_EXIT_TYPE_CGROUP" in script
    assert "EnvironmentFile=/etc/eyvan-native/systemd-runtime.env" in script
    assert "useradd --system --gid eyvan-workload" in script
    assert "Environment=VIRTUAL_ENV=/opt/eyvan-native/venv" in script
    assert "Environment=PATH=/opt/eyvan-native/venv/bin:/snap/bin:" in script
    assert "Environment=EYVAN_NATIVE_WORKLOAD_USER=eyvan-workload" in script
    assert "runuser -u eyvan-workload -- /opt/eyvan-native/venv/bin/python -c 'import sys'" in script
    assert "virtual environment is not executable by the workload user" in script
    assert "runuser -u eyvan-workload -- test -r" in script
    assert "/dev/vfio/devices/*" in script
    assert '[ -c "${device}" ] || continue' in script
    assert "setfacl -m u:eyvan-workload:rw" in script
    assert 'command apt-get -o DPkg::Lock::Timeout=300 "$@"' in script
    assert "eyvan_apt_get install -y --no-install-recommends acl" in script
    assert "python3 -m ensurepip --version" in script
    assert "eyvan_apt_get install -y --no-install-recommends python3-venv" in script
    assert "rm -rf /opt/eyvan-native/venv" in script
    assert "systemctl enable eyvan-native-worker.service" in script
    assert "systemctl restart eyvan-native-worker.service" in script
    assert "/opt/eyvan-native/multihost/sitecustomize.py" in script
    assert "import eyvan.worker.runtime._multihost_site as module" in script
    assert "/opt/eyvan-native/multihost/eyvan_multihost.py" in script
    assert "import eyvan.worker.distributed_environment as module" in script
    assert "/opt/eyvan-native/multihost/eyvan_distributed_environment.py" in script
    # The generated sitecustomize delegates to one staged function rather than
    # re-implementing the scrub-then-initialize ordering inline. Ordering written
    # in a heredoc can only be asserted as a *string*; ordering written in Python
    # is covered behaviourally in ``test_multihost``, which is where it belongs.
    assert "from eyvan_multihost import initialize_worker_runtime" in script
    assert "initialize_worker_runtime()" in script
    assert "remove_gce_tpu_topology_overrides(os.environ)" not in script
    assert "def _initialize_eyvan_multihost" not in script


def test_compute_bootstrap_apt_helper_executes_with_lock_timeout_and_preserves_failures(tmp_path: Path) -> None:
    apt_args = tmp_path / "apt-args"
    fake_apt = tmp_path / "apt-get"
    fake_apt.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$EYVAN_APT_ARGS"\nexit "${EYVAN_APT_EXIT_CODE:-0}"\n')
    fake_apt.chmod(0o755)
    script = build_native_worker_startup_script(
        package="eyvan==0.1.0",
        port=8123,
        state_dir="/var/lib/eyvan/worker",
        worker_secret_resource="projects/project-a/secrets/eyvan-native-cell/versions/latest",
        before_start="eyvan_apt_get install -y docker.io\nexit 0",
    )
    environment = {
        "EYVAN_APT_ARGS": str(apt_args),
        "EYVAN_APT_EXIT_CODE": "0",
        "PATH": str(tmp_path),
    }
    bash = shutil.which("bash")
    assert bash is not None

    syntax = subprocess.run([bash, "-n"], input=script, text=True, capture_output=True, check=False)
    succeeded = subprocess.run([bash], input=script, text=True, capture_output=True, env=environment, check=False)

    assert syntax.returncode == 0, syntax.stderr
    assert succeeded.returncode == 0, succeeded.stderr
    assert apt_args.read_text().splitlines() == [
        "-o",
        "DPkg::Lock::Timeout=300",
        "install",
        "-y",
        "docker.io",
    ]

    environment["EYVAN_APT_EXIT_CODE"] = "23"
    failed = subprocess.run([bash], input=script, text=True, capture_output=True, env=environment, check=False)
    missing = subprocess.run(
        [bash],
        input=script,
        text=True,
        capture_output=True,
        env={**environment, "PATH": str(tmp_path / "missing")},
        check=False,
    )

    assert failed.returncode == 23
    assert missing.returncode == 127
    assert "bootstrap requires apt-get" in missing.stderr


def test_compute_bootstrap_writes_the_declarative_runtime_environment_for_workloads() -> None:
    script = build_native_worker_startup_script(
        package="gs://artifacts/eyvan/eyvan.whl",
        package_sha256=_SHA256,
        port=8123,
        state_dir="/var/lib/eyvan/worker",
        worker_secret_resource="projects/project-a/secrets/eyvan-native-cell/versions/latest",
        runtime_env={
            "HF_HOME": "/var/lib/eyvan/workload/hf",
            "JAX_COMPILATION_CACHE_DIR": "/var/lib/eyvan/workload/jax-cache",
        },
    )

    assert "/etc/eyvan-native/workload-runtime.json" in script
    assert "--runtime-env-file /etc/eyvan-native/workload-runtime.json" in script
    # The file must be created private: the redirect itself runs under a
    # private umask, not only the chmod line that follows it.
    assert "(umask 077; printf" in script
    assert script.index("workload-runtime.json") < script.index("eyvan.worker.runtime.native_worker --host 0.0.0.0")


def test_compute_bootstrap_writes_a_typed_oci_runtime_for_the_native_worker() -> None:
    container = NativeContainerConfig(
        image="us-docker.pkg.dev/project-a/easydel/runtime@sha256:" + "a" * 64,
        pull_policy="always",
        working_dir="/workspace",
        privileged=True,
        mounts=(NativeContainerMount(source="/mnt/datasets", target="/datasets"),),
    )
    script = build_native_worker_startup_script(
        package="gs://artifacts/eyvan/eyvan.whl",
        package_sha256=_SHA256,
        port=8123,
        state_dir="/var/lib/eyvan/worker",
        worker_secret_resource="projects/project-a/secrets/eyvan-native-cell/versions/latest",
        runtime_container=container,
        container_artifact_mounts=("/opt/eyvan-runtime/easydel.whl",),
    )

    payload_line = next(line for line in script.splitlines() if "container-runtime.json" in line and "base64" in line)
    encoded_payload = max(shlex.split(payload_line), key=len)
    payload = json.loads(base64.b64decode(encoded_payload))

    assert payload == {
        "artifact_mounts": ["/opt/eyvan-runtime/easydel.whl"],
        "container": container.model_dump(mode="json"),
    }
    assert "--container-runtime-file /etc/eyvan-native/container-runtime.json" in script
    assert "(umask 077; printf" in script
    assert "docker version >/dev/null 2>&1" in script
    assert script.index("container-runtime.json") < script.index("eyvan.worker.runtime.native_worker --host 0.0.0.0")


def test_compute_bootstrap_rejects_container_artifacts_without_an_oci_runtime() -> None:
    with pytest.raises(ValueError, match="require a container runtime"):
        build_native_worker_startup_script(
            package="gs://artifacts/eyvan/eyvan.whl",
            package_sha256=_SHA256,
            port=8123,
            state_dir="/var/lib/eyvan/worker",
            worker_secret_resource="projects/project-a/secrets/eyvan-native-cell/versions/latest",
            container_artifact_mounts=("/opt/eyvan-runtime/easydel.whl",),
        )


def test_compute_bootstrap_rejects_an_environment_that_cannot_be_loaded_by_a_native_worker() -> None:
    with pytest.raises(ValueError, match="environment"):
        build_native_worker_startup_script(
            package="gs://artifacts/eyvan/eyvan.whl",
            package_sha256=_SHA256,
            port=8123,
            state_dir="/var/lib/eyvan/worker",
            worker_secret_resource="projects/project-a/secrets/eyvan-native-cell/versions/latest",
            runtime_env={"NOT-AN-ENVIRONMENT-NAME": "value"},
        )


def test_compute_bootstrap_rejects_a_gcs_package_without_a_wheel_filename() -> None:
    with pytest.raises(ValueError, match=r"must name a .whl file"):
        build_native_worker_startup_script(
            package="gs://artifacts/eyvan/eyvan",
            port=8123,
            state_dir="/var/lib/eyvan/worker",
            worker_secret_resource="projects/project-a/secrets/eyvan-native-cell/versions/latest",
        )


def test_compute_bootstrap_refuses_an_unverified_gcs_wheel() -> None:
    with pytest.raises(ValueError, match="package SHA-256 is required"):
        build_native_worker_startup_script(
            package="gs://artifacts/eyvan/eyvan.whl",
            port=8123,
            state_dir="/var/lib/eyvan/worker",
            worker_secret_resource="projects/project-a/secrets/eyvan-native-cell/versions/latest",
        )


def test_compute_bootstrap_rejects_a_non_numeric_gcs_generation() -> None:
    with pytest.raises(ValueError, match="positive decimal integer"):
        build_native_worker_startup_script(
            package="gs://artifacts/eyvan/eyvan.whl",
            package_sha256=_SHA256,
            package_generation="latest",
            port=8123,
            state_dir="/var/lib/eyvan/worker",
            worker_secret_resource="projects/project-a/secrets/eyvan-native-cell/versions/latest",
        )


def test_native_resolver_derives_vm_endpoint_and_cell_token_without_importing_ray(tmp_path: Path, monkeypatch) -> None:
    captured: list[dict] = []
    registry = ExecutionRuntimeRegistry()

    class FakeNativeDriver:
        def health(self) -> bool:
            return True

    def factory(**kwargs):
        captured.append(kwargs)
        return FakeNativeDriver()

    registry.register("eyvan-native", factory)
    resolver = ConfiguredExecutionResolver(
        "eyvan-native",
        registry=registry,
        state_dir=tmp_path,
        native_worker_port=8123,
        native_worker_token="agent-global-secret",
    )
    observation = CellObservation(
        "cell-a",
        CapacityStatus.HEALTHY,
        1,
        metadata={"head_ip": "10.20.0.7", "management_uid": "cell-uid", "provider_generation": 1},
    )
    monkeypatch.delitem(sys.modules, "eyvan.worker.runtime.ray_jobs", raising=False)
    monkeypatch.delitem(sys.modules, "ray", raising=False)

    driver = resolver.resolve("cell-a", observation)

    assert isinstance(driver, FakeNativeDriver)
    assert captured == [
        {
            "endpoint": "http://10.20.0.7:8123",
            "state_dir": tmp_path,
            "token": derive_native_worker_token("agent-global-secret", "cell-uid", provider_generation=1),
        }
    ]
    assert "eyvan.worker.runtime.ray_jobs" not in sys.modules
    assert "ray" not in sys.modules


def test_native_resolver_rotates_its_client_when_a_recreated_cell_reuses_an_ip(tmp_path: Path) -> None:
    captured: list[dict] = []
    registry = ExecutionRuntimeRegistry()

    class FakeNativeDriver:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    registry.register("eyvan-native", lambda **kwargs: captured.append(kwargs) or FakeNativeDriver())
    resolver = ConfiguredExecutionResolver(
        "eyvan-native",
        registry=registry,
        state_dir=tmp_path,
        native_worker_token="agent-global-secret",
    )
    first = CellObservation(
        "cell-a",
        CapacityStatus.HEALTHY,
        1,
        execution_endpoint="http://10.20.0.7:8100",
        metadata={"management_uid": "cell-uid", "provider_generation": 1},
    )
    replacement = CellObservation(
        "cell-a",
        CapacityStatus.HEALTHY,
        2,
        execution_endpoint="http://10.20.0.7:8100",
        metadata={"management_uid": "cell-uid", "provider_generation": 2},
    )

    first_driver = resolver.resolve("cell-a", first)
    replacement_driver = resolver.resolve("cell-a", replacement)

    assert first_driver is not replacement_driver
    assert len(captured) == 2
    assert captured[0]["token"] != captured[1]["token"]
    assert first_driver.closed is True

    resolver.close_cell("cell-a")

    assert replacement_driver.closed is True


def test_native_resolver_uses_gce_topology_for_a_multi_node_tpu_qr(tmp_path: Path) -> None:
    resolver = ConfiguredExecutionResolver(
        "eyvan-native",
        state_dir=tmp_path,
        native_worker_token="agent-global-secret",
    )
    observation = CellObservation(
        "cell-a",
        CapacityStatus.HEALTHY,
        1,
        metadata={
            "management_uid": "cell-uid",
            "provider_generation": 1,
            "native_worker_endpoints": ["http://worker-a:8100", "http://worker-b:8100"],
            "tpu_config": {"node_count": 2},
        },
    )
    try:
        driver = resolver.resolve("cell-a", observation)

        assert isinstance(driver, NativeSliceExecutor)
        assert driver.jax_cluster_detection is JaxClusterDetectionMode.GCE_TPU
    finally:
        resolver.close()


def test_native_worker_module_can_be_imported_without_ray() -> None:
    code = "\n".join(
        (
            "import sys",
            "import eyvan.worker.runtime.native_remote",
            "import eyvan.worker.runtime.native_worker",
            "assert 'ray' not in sys.modules",
            "assert 'eyvan.worker.runtime.ray_jobs' not in sys.modules",
        )
    )

    # A clean interpreter catches accidental transitive imports rather than
    # merely inspecting the current test process.
    import subprocess

    subprocess.run((sys.executable, "-c", code), check=True)


def test_a_rebuilt_wheel_replaces_the_installed_package() -> None:
    """Force-reinstall a rebuilt wheel so unchanged package versions cannot retain stale bootstrap modules."""

    script = build_native_worker_startup_script(
        package="gs://artifacts/eyvan/runtime/wheels/abc/eyvan-0.1.0-py3-none-any.whl",
        package_sha256=_SHA256,
        port=8123,
        state_dir="/var/lib/eyvan/worker",
        worker_secret_resource="projects/p/secrets/s/versions/latest",
    )

    assert "--force-reinstall" in script


def test_a_rebuilt_local_wheel_replaces_the_installed_package() -> None:
    """Force-reinstall local wheel paths copied by gcp connect without reinstalling their dependencies."""

    script = build_native_worker_startup_script(
        package="/var/lib/eyvan/attach/eyvan-0.0.1-py3-none-any.whl",
        port=8100,
        state_dir="/var/lib/eyvan/native-worker",
        worker_secret_file="/etc/eyvan-native/attached-worker-secret",
    )

    forced = [line for line in script.splitlines() if "--force-reinstall" in line]
    assert forced and all("--no-deps" in line for line in forced)


def test_a_package_name_is_not_force_reinstalled() -> None:
    script = build_native_worker_startup_script(
        package="eyvan",
        port=8100,
        state_dir="/var/lib/eyvan/native-worker",
        worker_secret_file="/etc/eyvan-native/attached-worker-secret",
    )

    assert "--force-reinstall" not in script


def test_the_forced_reinstall_does_not_rebuild_the_dependency_tree() -> None:
    """Every bootstrap pays this cost, so it must stay bounded to one package."""

    script = build_native_worker_startup_script(
        package="gs://artifacts/eyvan/runtime/wheels/abc/eyvan-0.1.0-py3-none-any.whl",
        package_sha256=_SHA256,
        port=8123,
        state_dir="/var/lib/eyvan/worker",
        worker_secret_resource="projects/p/secrets/s/versions/latest",
    )

    forced = [line for line in script.splitlines() if "--force-reinstall" in line]
    assert forced, "the forced reinstall must be present for this to mean anything"
    assert all("--no-deps" in line for line in forced)


def test_a_failed_venv_install_does_not_abort_before_eyvans_own_message(tmp_path: Path) -> None:
    """Reach Eyvan's actionable venv diagnostic when apt fails under shell fail-fast settings.

    Execute the generated conditional, locating it by a pinned literal so a
    restructure cannot silently bypass the behavior under test.
    """

    fake_apt = tmp_path / "apt-get"
    fake_apt.write_text('#!/bin/sh\nexit "${EYVAN_APT_EXIT_CODE:-0}"\n')
    fake_apt.chmod(0o755)
    script = build_native_worker_startup_script(
        package="eyvan==0.1.0",
        port=8123,
        state_dir="/var/lib/eyvan/worker",
        worker_secret_resource="projects/project-a/secrets/eyvan-native-cell/versions/latest",
    )
    lines = script.splitlines()
    opener = next(index for index, line in enumerate(lines) if "no-install-recommends python3-venv" in line)
    assert "eyvan_apt_get update" in lines[opener], "the update must share the conditional, not precede it"
    guarded = "\n".join(line.strip() for line in lines[opener : opener + 3])
    assert guarded.endswith("fi"), guarded

    preamble = "\n".join(
        (
            "set -euo pipefail",
            'eyvan_apt_get() { command apt-get -o DPkg::Lock::Timeout=300 "$@"; }',
        )
    )
    probe = f"{preamble}\n{guarded}\necho REACHED-THE-NEXT-CHECK\n"
    bash = shutil.which("bash")
    assert bash is not None

    result = subprocess.run(
        [bash],
        input=probe,
        text=True,
        capture_output=True,
        env={"PATH": str(tmp_path), "EYVAN_APT_EXIT_CODE": "100"},
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "REACHED-THE-NEXT-CHECK" in result.stdout
    assert "could not install python3-venv from apt" in result.stderr


def test_a_python_without_ensurepip_falls_back_to_one_that_has_it(tmp_path: Path) -> None:
    """Try installed versioned interpreters with ensurepip before falling back to apt.

    Execute the generated selection with a failing ``python3`` and a working
    versioned sibling so the check covers behavior rather than script text alone.
    """

    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    (stub_dir / "python3").write_text("#!/bin/sh\nexit 1\n")
    (stub_dir / "python3.11").write_text("#!/bin/sh\nexit 0\n")
    for name in ("python3", "python3.11"):
        (stub_dir / name).chmod(0o755)

    script = build_native_worker_startup_script(
        package="eyvan==0.1.0",
        port=8123,
        state_dir="/var/lib/eyvan/worker",
        worker_secret_resource="projects/project-a/secrets/eyvan-native-cell/versions/latest",
    )
    selection = _interpreter_selection(script)
    assert "ensurepip" in selection, selection

    probe = f'set -eu\nPATH="{stub_dir}:$PATH"\n{selection}\necho "SELECTED=${{eyvan_python}}"\n'
    bash = shutil.which("bash")
    assert bash is not None

    result = subprocess.run([bash], input=probe, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert f"SELECTED={stub_dir}/python3.11" in result.stdout, result.stdout
    assert "has no ensurepip" in result.stderr


def test_the_default_interpreter_is_kept_when_it_already_works(tmp_path: Path) -> None:
    """The search must not move a host that was never broken onto a sibling."""

    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    (stub_dir / "python3").write_text("#!/bin/sh\nexit 0\n")
    (stub_dir / "python3").chmod(0o755)

    script = build_native_worker_startup_script(
        package="eyvan==0.1.0",
        port=8123,
        state_dir="/var/lib/eyvan/worker",
        worker_secret_resource="projects/project-a/secrets/eyvan-native-cell/versions/latest",
    )
    selection = _interpreter_selection(script)

    probe = f'set -eu\nPATH="{stub_dir}:$PATH"\n{selection}\necho "SELECTED=${{eyvan_python}}"\n'
    result = subprocess.run([shutil.which("bash")], input=probe, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "SELECTED=python3" in result.stdout


def test_an_unsupported_default_python_is_skipped_even_when_it_has_ensurepip(tmp_path: Path) -> None:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    (stub_dir / "python3").write_text('#!/bin/sh\nif [ "$1" = "-c" ]; then exit 1; fi\nexit 0\n')
    (stub_dir / "python3.11").write_text("#!/bin/sh\nexit 0\n")
    for name in ("python3", "python3.11"):
        (stub_dir / name).chmod(0o755)
    script = build_native_worker_startup_script(
        package="eyvan==0.1.0",
        port=8123,
        state_dir="/var/lib/eyvan/worker",
        worker_secret_resource="projects/p/secrets/s/versions/1",
    )
    selection = _interpreter_selection(script)
    result = subprocess.run(
        [shutil.which("bash")],
        input=f'PATH="{stub_dir}:$PATH"\n{selection}\necho "SELECTED=${{eyvan_python}}"\n',
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"SELECTED={stub_dir}/python3.11" in result.stdout


@pytest.mark.parametrize("version", ["3.10", "3.10.19", "3.9", "3.14"])
def test_an_unsupported_declared_worker_python_is_refused_before_bootstrap(version) -> None:
    with pytest.raises(ValueError, match=r"Python 3\.11"):
        build_native_worker_startup_script(
            package="eyvan==0.1.0",
            port=8123,
            state_dir="/var/lib/eyvan/worker",
            worker_secret_resource="projects/p/secrets/s/versions/1",
            runtime_python=version,
        )


def _interpreter_selection(script: str) -> str:
    """Return the interpreter-selection block, ending where the apt guard begins.

    Sliced on the guard rather than on a closing ``fi``: the block contains a
    nested ``if`` inside a ``for``, so the first ``fi`` closes the wrong one and
    the extracted fragment will not parse.
    """

    lines = script.splitlines()
    start = next(index for index, line in enumerate(lines) if line.strip() == "eyvan_python=python3")
    end = next(
        index
        for index in range(start + 1, len(lines))
        if lines[index].strip().startswith('if ! "${eyvan_python}" -m ensurepip')
    )
    return "\n".join(line.strip() for line in lines[start:end])


def test_a_declared_python_builds_the_worker_environment_with_uv() -> None:
    """A TPU image ships whatever Python it ships; a pickled callable does not care.

    A serialized callable is pinned to the interpreter that pickled it, and the
    images in use are 3.11 against a 3.13 laptop -- a mismatch discovered only
    once a slice is ACTIVE and the workload dispatched. Declaring the
    interpreter builds the environment on it instead.

    uv rather than apt or ensurepip, because the whole premise is that the image
    does not already have this Python, and uv fetches it. Deliberately without
    ``--system-site-packages``: those packages are built for the image's own
    interpreter, and importing them under another is the failure being avoided,
    so the fleet installs its own accelerator stack in ``runtime.init_commands``.
    """

    script = build_native_worker_startup_script(
        package="eyvan==0.1.0",
        port=8123,
        state_dir="/var/lib/eyvan/worker",
        worker_secret_resource="projects/p/secrets/s/versions/1",
        runtime_python="3.13",
    )

    assert "uv venv --seed --python 3.13 /opt/eyvan-native/venv" in script
    # --seed is load-bearing, not tidiness. The block immediately after this one
    # recreates any environment whose `python -m pip` does not answer, and a uv
    # venv has no pip unless asked -- so an unseeded 3.13 environment is built
    # and then instantly replaced by one on the image's own interpreter, which
    # is precisely the failure being prevented.
    seeded = script.index("uv venv --seed --python 3.13")
    recreate = script.index("-m pip --version >/dev/null 2>&1; then", seeded)
    assert seeded < recreate, "the recreate guard must come after the declared build"
    # The declared environment must not inherit the image's interpreter's
    # packages; that is the bug, not a convenience.
    assert "uv venv --python 3.13 --system-site-packages" not in script
    # uv is fetched only when the host lacks it, and its absence is a typed
    # failure rather than a venv silently built on the wrong Python.
    assert "command -v uv" in script
    assert "Eyvan needs uv to build a Python 3.13 worker environment" in script


def test_a_full_declared_python_version_matches_the_interpreter_guard() -> None:
    """A full version ("3.11.9") is a valid uv declaration and still a 3.11 pin.

    The guard compares the running interpreter's major.minor, so the declared
    text is reduced to the same two components before the two are compared.
    Comparing the declared *text* against that join never matched -- "3.11.9"
    is never "3.11" -- and the venv was torn down and rebuilt on every boot,
    which is the exact every-boot rebuild the guard exists to prevent.
    """

    script = build_native_worker_startup_script(
        package="eyvan==0.1.0",
        port=8123,
        state_dir="/var/lib/eyvan/worker",
        worker_secret_resource="projects/p/secrets/s/versions/1",
        runtime_python="3.11.9",
    )

    # uv receives the declaration as spelled: a full version names the exact
    # interpreter to fetch.
    assert "uv venv --seed --python 3.11.9 /opt/eyvan-native/venv" in script
    # The guard compares major.minor to major.minor, as a quoted string, so an
    # interpreter that already matches keeps its venv across boots.
    assert "sys.version_info[:2])) == '3.11' else 1)" in script
    assert "'3.11.9' else 1)" not in script


def test_an_undeclared_python_renders_exactly_the_script_it_always_did() -> None:
    """A fleet that declares nothing must boot the way it always booted."""

    plain = build_native_worker_startup_script(
        package="eyvan==0.1.0",
        port=8123,
        state_dir="/var/lib/eyvan/worker",
        worker_secret_resource="projects/p/secrets/s/versions/1",
    )

    assert "uv venv" not in plain
    # Still the image's own interpreter, still inheriting its JAX and libtpu.
    assert "-m venv --system-site-packages /opt/eyvan-native/venv" in plain


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow


def test_the_per_job_dependency_cache_is_created_for_the_workload_user() -> None:
    """Create a writable eyvan-workload dependency cache so jobs can reuse installed packages."""

    from eyvan.worker.runtime.dependency_entry import DEFAULT_CACHE_ROOT

    script = build_native_worker_startup_script(
        package="gs://artifacts/eyvan/eyvan.whl",
        package_sha256=_SHA256,
        package_generation="1700000000000000",
        port=8123,
        state_dir="/var/lib/eyvan/worker",
        worker_secret_resource="projects/project-a/secrets/eyvan-native-cell-uid/versions/latest",
    )

    creation = next(line for line in script.splitlines() if "install -d -o eyvan-workload" in line)
    assert str(DEFAULT_CACHE_ROOT) in creation.split()
    assert "-m 0750" in creation
