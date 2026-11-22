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

"""Shared eray test fixtures."""

from __future__ import annotations

import tempfile
from functools import wraps

import eray.provision.tunnel as tunnel_module
import pytest


@pytest.fixture(autouse=True)
def _isolate_tunnel_store(tmp_path, monkeypatch):
    """Point the tunnel store at a per-test tmp dir for every test.

    The tunnel store (`~/.eray/tunnels.json`) is process-global mutable
    state, and address resolution (`eray resources`/`eray status`) now
    consults it to auto-detect an open tunnel's port. Without this, tests
    would read (and mutate) the developer's real tunnels — non-hermetic,
    and flaky on a machine that actually has tunnels open. Autouse so no
    test can forget it.
    """
    monkeypatch.setattr(tunnel_module, "STORE_PATH", tmp_path / "tunnels.json")
    monkeypatch.setattr(tunnel_module, "LOG_DIR", tmp_path / "tunnel-logs")
    # Shrink the post-spawn liveness probe so the suite doesn't pay the full
    # production wait on every open_tunnel; still long enough for a `python -c
    # pass` forwarder to exit and be caught as an immediate failure.
    monkeypatch.setattr(tunnel_module, "_STARTUP_PROBE_S", 0.4)


@pytest.fixture(scope="module")
def local_ray():
    """Own a fake-accelerator cluster for opt-in host orchestration tests only.

    No workload initializes an accelerator backend. The returned actor keeps
    production scheduling behavior but cannot kill VFIO holders or remove the
    real TPU lock. Driver waits (including swarm_execute's internal ray.get)
    are bounded, and the owned cluster is shut down even on test/setup errors.
    """
    ray = pytest.importorskip("ray")
    from eray.pool.device_host import DeviceHostActor
    from ray._private import ray_constants

    if ray.is_initialized():
        raise RuntimeError("host orchestration tests require their own local Ray cluster")

    class HostOnlyActor(DeviceHostActor.__ray_metadata__.modified_class):
        def _kill_vfio_holders(self):
            pass

        def _hacky_remove_tpu_lockfile(self):
            pass

    get = ray.get

    @wraps(get)
    def bounded_get(object_refs, *, timeout=None, **kwargs):
        return get(object_refs, timeout=30 if timeout is None else min(timeout, 30), **kwargs)

    with pytest.MonkeyPatch.context() as patch, tempfile.TemporaryDirectory(prefix="eray-test-") as temp_dir:
        # Ray's automatic uv hook copies the workspace but not its .venv.
        # With uv run --no-sync, workers then repeatedly fail to import ray.
        # Use the driver's installed editable environment for these local tests.
        patch.setattr(ray_constants, "RAY_ENABLE_UV_RUN_RUNTIME_ENV", False)
        patch.setattr(ray, "get", bounded_get)
        try:
            ray.init(
                address="local",
                num_cpus=8,
                num_gpus=4,
                resources={"TPU": 4},
                include_dashboard=False,
                log_to_driver=False,
                runtime_env={
                    "env_vars": {
                        "JAX_PLATFORMS": "cpu",
                        "JAX_PLATFORM_NAME": "cpu",
                        "ENABLE_DISTRIBUTED_INIT": "0",
                    }
                },
                _temp_dir=temp_dir,
            )
            yield ray.remote(HostOnlyActor)
        finally:
            ray.shutdown()
