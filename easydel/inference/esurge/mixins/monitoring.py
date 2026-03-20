# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from typing import Any

from ..metrics import get_metrics_collector, initialize_metrics


def _panel(
    title: str,
    expr: str,
    ds_uid: str,
    grid_x: int,
    grid_y: int,
    grid_w: int = 12,
    grid_h: int = 8,
    panel_id: int = 1,
    panel_type: str = "timeseries",
    unit: str = "",
    legend_mode: str = "list",
) -> dict:
    """Build a single Grafana dashboard panel dict."""
    target = {
        "datasource": {"type": "prometheus", "uid": ds_uid},
        "expr": expr,
        "legendFormat": "{{__name__}}",
        "refId": "A",
    }
    field_config: dict[str, Any] = {"defaults": {"unit": unit}, "overrides": []}
    panel: dict[str, Any] = {
        "id": panel_id,
        "title": title,
        "type": panel_type,
        "datasource": {"type": "prometheus", "uid": ds_uid},
        "targets": [target],
        "fieldConfig": field_config,
        "gridPos": {"h": grid_h, "w": grid_w, "x": grid_x, "y": grid_y},
        "options": {"legend": {"displayMode": legend_mode}, "tooltip": {"mode": "multi"}},
    }
    if panel_type == "stat":
        panel["options"] = {
            "reduceOptions": {"calcs": ["lastNotNull"]},
            "colorMode": "value",
            "graphMode": "area",
        }
    return panel


def _build_esurge_dashboard_model(datasource_uid: str) -> dict:
    """Return a complete Grafana dashboard dict for eSurge metrics."""
    panels = [
        # ── Row 0: key stat panels ──
        _panel("Tokens / sec", "esurge_tokens_per_second", datasource_uid, 0, 0, 6, 4, 1, "stat", "ops"),
        _panel("Running Requests", "esurge_running_requests", datasource_uid, 6, 0, 6, 4, 2, "stat", "short"),
        _panel("Waiting Requests", "esurge_waiting_requests", datasource_uid, 12, 0, 6, 4, 3, "stat", "short"),
        _panel("Batch Size", "esurge_batch_size", datasource_uid, 18, 0, 6, 4, 4, "stat", "short"),
        # ── Row 1: throughput & latency ──
        _panel("Throughput (tok/s)", "esurge_tokens_per_second", datasource_uid, 0, 4, 12, 8, 5, unit="ops"),
        _panel(
            "Request Latency (p50 / p99)",
            "histogram_quantile(0.5, rate(esurge_request_duration_seconds_bucket[1m]))",
            datasource_uid,
            12,
            4,
            12,
            8,
            6,
            unit="s",
        ),
        # ── Row 2: TTFT & schedule time ──
        _panel(
            "Time to First Token (p50 / p99)",
            "histogram_quantile(0.5, rate(esurge_time_to_first_token_seconds_bucket[1m]))",
            datasource_uid,
            0,
            12,
            12,
            8,
            7,
            unit="s",
        ),
        _panel(
            "Schedule Duration (p50)",
            "histogram_quantile(0.5, rate(esurge_schedule_duration_seconds_bucket[1m]))",
            datasource_uid,
            12,
            12,
            12,
            8,
            8,
            unit="s",
        ),
        # ── Row 3: cache ──
        _panel("Cache Pages (total vs used)", "esurge_cache_pages_total", datasource_uid, 0, 20, 8, 8, 9),
        _panel("Cache Pages Used", "esurge_cache_pages_used", datasource_uid, 8, 20, 8, 8, 10),
        _panel("Cache Hit Rate", "esurge_cache_hit_rate", datasource_uid, 16, 20, 8, 8, 11, unit="percentunit"),
        # ── Row 4: model execution ──
        _panel(
            "Model Execution Time (p50)",
            "histogram_quantile(0.5, rate(esurge_model_execution_duration_seconds_bucket[1m]))",
            datasource_uid,
            0,
            28,
            12,
            8,
            12,
            unit="s",
        ),
        _panel(
            "Tokens Generated (rate)",
            "rate(esurge_tokens_generated_total[1m])",
            datasource_uid,
            12,
            28,
            12,
            8,
            13,
            unit="ops",
        ),
    ]

    dashboard = {
        "id": None,
        "uid": "esurge-overview",
        "title": "eSurge Engine Overview",
        "tags": ["esurge", "inference"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "5s",
        "time": {"from": "now-15m", "to": "now"},
        "panels": panels,
    }
    return dashboard


class EngineMonitoringMixin:
    """Mixin providing monitoring and observability for the eSurge engine.

    Manages Prometheus metrics exporting, Grafana dashboard provisioning
    (via local binary or Docker), and runtime performance metric collection.
    Supports auto-starting a Grafana instance pre-configured with a
    Prometheus data source for visualizing inference throughput, latency,
    and resource utilization.

    Methods:
        start_monitoring: Initialize Prometheus exporter and optional Grafana.
        stop_monitoring: Shut down all monitoring services.
        get_metrics_summary: Retrieve current performance metrics snapshot.
        monitoring_active: Property indicating whether monitoring is running.
    """

    def _prepare_grafana_provisioning(
        self,
        datasource_name: str,
        datasource_uid: str,
        datasource_url: str,
        for_docker: bool = False,
    ) -> str:
        """Create temporary provisioning config for Grafana.

        Creates a temporary directory with Grafana provisioning YAML files
        for auto-configuring a Prometheus data source.

        Args:
            datasource_name: Display name for the data source.
            datasource_uid: Unique identifier for the data source.
            datasource_url: URL of the Prometheus metrics endpoint.
            for_docker: If True, use container-relative paths in provider
                config (the provisioning root is mounted at
                ``/etc/grafana/provisioning`` inside the container).

        Returns:
            Path to the provisioning root directory.
        """
        provisioning_root = tempfile.mkdtemp(prefix="esurge_grafana_")
        datasources_dir = os.path.join(provisioning_root, "datasources")
        dashboards_dir = os.path.join(provisioning_root, "dashboards")
        dashboard_json_dir = os.path.join(provisioning_root, "dashboard_json")
        os.makedirs(datasources_dir, exist_ok=True)
        os.makedirs(dashboards_dir, exist_ok=True)
        os.makedirs(dashboard_json_dir, exist_ok=True)

        datasource_config = f"""apiVersion: 1
datasources:
  - name: "{datasource_name}"
    uid: "{datasource_uid}"
    type: prometheus
    access: proxy
    url: "{datasource_url}"
    isDefault: true
    editable: true
    jsonData:
      timeInterval: "1s"
"""
        with open(os.path.join(datasources_dir, "esurge-prometheus.yaml"), "w", encoding="utf-8") as f:
            f.write(datasource_config)

        # Docker mounts provisioning_root at /etc/grafana/provisioning,
        # so the dashboard JSON dir is at a fixed container path.
        # Local Grafana uses the real host path.
        if for_docker:
            json_path_in_config = "/etc/grafana/provisioning/dashboard_json"
        else:
            json_path_in_config = dashboard_json_dir

        provider_config = f"""apiVersion: 1
providers:
  - name: "esurge-autoprovisioned"
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: {json_path_in_config}
      foldersFromFilesStructure: false
"""
        with open(os.path.join(dashboards_dir, "provider.yaml"), "w", encoding="utf-8") as f:
            f.write(provider_config)

        dashboard_model = _build_esurge_dashboard_model(datasource_uid)
        with open(os.path.join(dashboard_json_dir, "esurge-overview.json"), "w", encoding="utf-8") as f:
            json.dump(dashboard_model, f, indent=2)

        return provisioning_root

    def _start_local_grafana_service(
        self,
        provisioning_root: str,
        grafana_host: str | None,
        grafana_port: int,
        grafana_admin_user: str,
        grafana_admin_password: str,
        allow_anonymous: bool,
    ) -> str | None:
        """Start Grafana using a locally installed grafana-server binary.

        Args:
            provisioning_root: Path to provisioning config directory.
            grafana_host: Host for reporting Grafana URL.
            grafana_port: Port to run Grafana on.
            grafana_admin_user: Admin username.
            grafana_admin_password: Admin password.
            allow_anonymous: Enable anonymous admin access.

        Returns:
            Grafana URL if started successfully, None otherwise.
        """
        if self._grafana_process:
            return self._grafana_url

        grafana_exe = shutil.which("grafana-server")
        if not grafana_exe:
            return None

        data_dir = os.path.join(provisioning_root, "data")
        plugins_dir = os.path.join(provisioning_root, "plugins")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(plugins_dir, exist_ok=True)

        env = os.environ.copy()
        env.update(
            {
                "GF_PATHS_PROVISIONING": provisioning_root,
                "GF_PATHS_DATA": data_dir,
                "GF_PATHS_PLUGINS": plugins_dir,
                "GF_SECURITY_ADMIN_USER": grafana_admin_user,
                "GF_SECURITY_ADMIN_PASSWORD": grafana_admin_password,
                "GF_SERVER_HTTP_PORT": str(grafana_port),
            }
        )
        if allow_anonymous:
            env["GF_AUTH_ANONYMOUS_ENABLED"] = "true"
            env["GF_AUTH_ANONYMOUS_ORG_ROLE"] = "Admin"

        possible_homepaths = ["/usr/share/grafana", "/usr/local/share/grafana"]
        cmd = [grafana_exe, "server"]
        homepath = next((p for p in possible_homepaths if os.path.isdir(p)), None)
        if homepath:
            cmd.extend(["--homepath", homepath])

        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            self._info(" Failed to start local Grafana server: %s", exc)
            return None

        self._grafana_process = proc
        self._grafana_temp_dir = provisioning_root
        self._grafana_url = f"http://{grafana_host or 'localhost'}:{grafana_port}"
        self._info(" Grafana started (local binary) at %s", self._grafana_url)
        return self._grafana_url

    def _start_docker_grafana_service(
        self,
        provisioning_root: str,
        grafana_host: str | None,
        grafana_port: int,
        grafana_image: str,
        grafana_admin_user: str,
        grafana_admin_password: str,
        allow_anonymous: bool,
        datasource_url: str,
    ) -> str | None:
        """Start Grafana using Docker.

        Args:
            provisioning_root: Path to provisioning config directory (mounted).
            grafana_host: Host for reporting Grafana URL.
            grafana_port: Host port to expose Grafana.
            grafana_image: Docker image to use.
            grafana_admin_user: Admin username.
            grafana_admin_password: Admin password.
            allow_anonymous: Enable anonymous admin access.
            datasource_url: Prometheus URL for the container to connect to.

        Returns:
            Grafana URL if started successfully, None otherwise.
        """
        if self._grafana_container_name:
            return self._grafana_url

        docker_exe = shutil.which("docker")
        if not docker_exe:
            return None

        container_name = f"esurge-grafana-{uuid.uuid4().hex[:8]}"
        cmd = [
            docker_exe,
            "run",
            "--rm",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{grafana_port}:3000",
            "-v",
            f"{provisioning_root}:/etc/grafana/provisioning",
            "--add-host",
            "host.docker.internal:host-gateway",
            "-e",
            f"GF_SECURITY_ADMIN_USER={grafana_admin_user}",
            "-e",
            f"GF_SECURITY_ADMIN_PASSWORD={grafana_admin_password}",
        ]
        if allow_anonymous:
            cmd.extend(["-e", "GF_AUTH_ANONYMOUS_ENABLED=true", "-e", "GF_AUTH_ANONYMOUS_ORG_ROLE=Admin"])
        cmd.append(grafana_image)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            err_output = exc.stderr.strip() if exc.stderr else str(exc)
            self._info(" Failed to start Grafana automatically: %s", err_output)
            return None
        except Exception as exc:
            self._info(" Failed to start Grafana automatically: %s", exc)
            return None

        self._grafana_container_name = container_name
        self._grafana_container_id = result.stdout.strip() or container_name
        self._grafana_temp_dir = provisioning_root
        self._grafana_url = f"http://{grafana_host or 'localhost'}:{grafana_port}"
        self._info(" Grafana started (Docker) at %s (datasource -> %s)", self._grafana_url, datasource_url)
        return self._grafana_url

    def _start_prometheus_server(
        self,
        scrape_target: str,
        prometheus_server_port: int = 9090,
    ) -> str | None:
        """Start a Prometheus server that scrapes the prometheus_client endpoint.

        Args:
            scrape_target: The host:port of the prometheus_client exporter
                           (e.g. ``"localhost:11184"``).
            prometheus_server_port: Port for the Prometheus server itself.

        Returns:
            Prometheus server URL if started, None otherwise.
        """
        if self._prometheus_process:
            return f"http://localhost:{prometheus_server_port}"

        prometheus_exe = shutil.which("prometheus")
        if not prometheus_exe:
            return None

        prom_dir = tempfile.mkdtemp(prefix="esurge_prometheus_")
        data_dir = os.path.join(prom_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        config = (
            "global:\n"
            "  scrape_interval: 2s\n"
            "  evaluation_interval: 2s\n"
            "scrape_configs:\n"
            "  - job_name: esurge\n"
            "    metrics_path: /metrics\n"
            f"    static_configs:\n"
            f"      - targets: ['{scrape_target}']\n"
        )
        config_path = os.path.join(prom_dir, "prometheus.yml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config)

        cmd = [
            prometheus_exe,
            f"--config.file={config_path}",
            f"--storage.tsdb.path={data_dir}",
            f"--web.listen-address=0.0.0.0:{prometheus_server_port}",
            "--storage.tsdb.retention.time=1h",
            "--web.enable-lifecycle",
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            self._info(" Failed to start Prometheus server: %s", exc)
            shutil.rmtree(prom_dir, ignore_errors=True)
            return None

        self._prometheus_process = proc
        self._prometheus_temp_dir = prom_dir
        url = f"http://localhost:{prometheus_server_port}"
        self._info(" Prometheus server started at %s (scraping %s)", url, scrape_target)
        return url

    def _stop_prometheus_server(self) -> None:
        """Stop the Prometheus server if it was started by the engine."""
        if self._prometheus_process:
            try:
                self._prometheus_process.terminate()
                self._prometheus_process.wait(timeout=5)
                self._info(" Prometheus server stopped")
            except Exception:
                try:
                    self._prometheus_process.kill()
                except Exception:
                    pass
            self._prometheus_process = None
        if self._prometheus_temp_dir:
            shutil.rmtree(self._prometheus_temp_dir, ignore_errors=True)
            self._prometheus_temp_dir = None

    def _provision_running_grafana(
        self,
        grafana_host: str | None,
        grafana_port: int,
        grafana_admin_user: str,
        grafana_admin_password: str,
        datasource_name: str,
        datasource_uid: str,
        datasource_url: str,
    ) -> str | None:
        """Provision datasource and dashboard on an already-running Grafana via HTTP API.

        Returns:
            Grafana URL if provisioning succeeded, None otherwise.
        """
        import base64
        import urllib.error
        import urllib.request

        grafana_base = f"http://{grafana_host or 'localhost'}:{grafana_port}"
        auth = base64.b64encode(f"{grafana_admin_user}:{grafana_admin_password}".encode()).decode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {auth}",
        }

        # Check if Grafana is reachable
        try:
            req = urllib.request.Request(f"{grafana_base}/api/health", headers=headers)
            urllib.request.urlopen(req, timeout=3)
        except Exception:
            return None

        self._info(" Detected running Grafana at %s, provisioning via API...", grafana_base)

        # Upsert datasource
        datasource_provisioned = False
        ds_payload = json.dumps(
            {
                "name": datasource_name,
                "uid": datasource_uid,
                "type": "prometheus",
                "access": "proxy",
                "url": datasource_url,
                "isDefault": True,
                "jsonData": {"timeInterval": "1s"},
            }
        ).encode()
        try:
            req = urllib.request.Request(
                f"{grafana_base}/api/datasources",
                data=ds_payload,
                headers=headers,
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
            self._info("  Datasource '%s' created", datasource_name)
            datasource_provisioned = True
        except urllib.error.HTTPError as e:
            if e.code == 409:
                # Already exists — update it
                try:
                    req = urllib.request.Request(
                        f"{grafana_base}/api/datasources/uid/{datasource_uid}",
                        data=ds_payload,
                        headers=headers,
                        method="PUT",
                    )
                    urllib.request.urlopen(req, timeout=5)
                    self._info("  Datasource '%s' updated", datasource_name)
                    datasource_provisioned = True
                except Exception as update_exc:
                    self._info("  Failed to update datasource '%s': %s", datasource_name, update_exc)
            else:
                self._info("  Failed to create datasource: %s", e)
        if not datasource_provisioned:
            return None

        # Upsert dashboard
        dashboard_model = _build_esurge_dashboard_model(datasource_uid)
        dashboard_payload = json.dumps(
            {
                "dashboard": dashboard_model,
                "overwrite": True,
                "message": "Auto-provisioned by eSurge",
            }
        ).encode()
        try:
            req = urllib.request.Request(
                f"{grafana_base}/api/dashboards/db",
                data=dashboard_payload,
                headers=headers,
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=5)
            resp_data = json.loads(resp.read().decode())
            dashboard_url = f"{grafana_base}{resp_data.get('url', '/d/esurge-overview')}"
            self._info("  Dashboard provisioned: %s", dashboard_url)
        except Exception as e:
            self._info("  Failed to provision dashboard: %s", e)
            return None

        self._grafana_url = grafana_base
        return grafana_base

    def _start_grafana_service(
        self,
        prometheus_url: str | None,
        grafana_host: str | None,
        grafana_port: int,
        grafana_image: str,
        grafana_admin_user: str,
        grafana_admin_password: str,
        allow_anonymous: bool,
        datasource_name: str,
        datasource_uid: str | None,
        datasource_url: str | None,
        use_docker: bool,
    ) -> str | None:
        """Attempt to launch Grafana wired to the Prometheus endpoint.

        First checks if Grafana is already running and provisions via the
        HTTP API.  Otherwise tries a local grafana-server binary, then
        optionally falls back to Docker.

        Args:
            prometheus_url: URL of the Prometheus metrics endpoint.
            grafana_host: Host for reporting Grafana URL.
            grafana_port: Port to run Grafana on.
            grafana_image: Docker image (used if use_docker=True).
            grafana_admin_user: Admin username.
            grafana_admin_password: Admin password.
            allow_anonymous: Enable anonymous admin access.
            datasource_name: Display name for the data source.
            datasource_uid: Unique identifier for the data source.
            datasource_url: Override URL for the Prometheus data source.
            use_docker: Allow falling back to Docker.

        Returns:
            Grafana URL if started successfully, None otherwise.
        """
        if self._grafana_container_name or self._grafana_process or self._grafana_url:
            return self._grafana_url

        if not prometheus_url:
            self._info(" Grafana autostart skipped: Prometheus URL unavailable")
            return None

        datasource_uid = datasource_uid or "esurge-prometheus"

        # prometheus_url is the raw exporter endpoint (e.g. http://0.0.0.0:11184/metrics).
        # Grafana needs a real Prometheus *server* with query APIs, not the
        # raw /metrics endpoint.  Start one if possible.
        from urllib.parse import urlparse

        parsed = urlparse(prometheus_url)
        exporter_host = parsed.hostname or "localhost"
        exporter_port = parsed.port or 11184
        scrape_target = f"{exporter_host}:{exporter_port}"
        # Normalise 0.0.0.0 → localhost for Prometheus scraping
        if exporter_host == "0.0.0.0":
            scrape_target = f"localhost:{exporter_port}"

        prom_server_url = datasource_url  # user override
        if not prom_server_url:
            prom_server_url = self._start_prometheus_server(
                scrape_target=scrape_target,
                prometheus_server_port=9090,
            )
        if not prom_server_url:
            self._info(
                " Grafana autostart skipped: could not start Prometheus server "
                "(install prometheus: https://prometheus.io/download/)"
            )
            return None

        # For Docker-based Grafana, rewrite localhost → host.docker.internal
        docker_prom_url = (
            prom_server_url.replace("0.0.0.0", "host.docker.internal")
            .replace("localhost", "host.docker.internal")
            .replace("127.0.0.1", "host.docker.internal")
        )

        # If Grafana is already running, provision via HTTP API
        api_url = self._provision_running_grafana(
            grafana_host=grafana_host,
            grafana_port=grafana_port,
            grafana_admin_user=grafana_admin_user,
            grafana_admin_password=grafana_admin_password,
            datasource_name=datasource_name,
            datasource_uid=datasource_uid,
            datasource_url=prom_server_url,
        )
        if api_url:
            return api_url

        # Try local grafana-server first — always use host-local URL
        local_provisioning = self._prepare_grafana_provisioning(
            datasource_name=datasource_name,
            datasource_uid=datasource_uid,
            datasource_url=prom_server_url,
            for_docker=False,
        )
        local_url = self._start_local_grafana_service(
            provisioning_root=local_provisioning,
            grafana_host=grafana_host,
            grafana_port=grafana_port,
            grafana_admin_user=grafana_admin_user,
            grafana_admin_password=grafana_admin_password,
            allow_anonymous=allow_anonymous,
        )
        if local_url:
            return local_url
        shutil.rmtree(local_provisioning, ignore_errors=True)

        if not use_docker:
            self._info(" Grafana autostart skipped: local server unavailable and Docker disabled")
            return None

        # Docker fallback — use container-relative paths and Docker URLs
        docker_provisioning = self._prepare_grafana_provisioning(
            datasource_name=datasource_name,
            datasource_uid=datasource_uid,
            datasource_url=docker_prom_url,
            for_docker=True,
        )
        docker_url = self._start_docker_grafana_service(
            provisioning_root=docker_provisioning,
            grafana_host=grafana_host,
            grafana_port=grafana_port,
            grafana_image=grafana_image,
            grafana_admin_user=grafana_admin_user,
            grafana_admin_password=grafana_admin_password,
            allow_anonymous=allow_anonymous,
            datasource_url=docker_prom_url,
        )
        if docker_url:
            return docker_url

        shutil.rmtree(docker_provisioning, ignore_errors=True)
        return None

    def _stop_grafana_service(self) -> None:
        """Stop the Grafana container/process if it was started by the engine.

        Cleans up Docker containers, local processes, and temporary provisioning
        directories. Safe to call even if Grafana was not started.
        """
        # Stop processes/containers BEFORE removing their temp dirs
        container = self._grafana_container_id or self._grafana_container_name
        docker_exe = shutil.which("docker") if container else None
        if container and docker_exe:
            try:
                subprocess.run(
                    [docker_exe, "stop", container],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                self._info(" Grafana container stopped")
            except Exception:
                self._info(" Failed to stop Grafana container %s", container)

        if self._grafana_process:
            try:
                self._grafana_process.terminate()
                self._grafana_process.wait(timeout=2)
                self._info(" Grafana process stopped")
            except Exception:
                self._info(" Failed to stop Grafana process gracefully")
            self._grafana_process = None

        self._stop_prometheus_server()

        # Now safe to remove temp dirs
        if self._grafana_temp_dir:
            shutil.rmtree(self._grafana_temp_dir, ignore_errors=True)

        self._grafana_container_name = None
        self._grafana_container_id = None
        self._grafana_temp_dir = None
        self._grafana_url = None

    def start_monitoring(
        self,
        dashboard_port: int | None = None,
        prometheus_port: int = 11184,
        dashboard_host: str | None = None,
        enable_prometheus: bool = True,
        enable_dashboard: bool | None = None,
        enable_console: bool = False,
        log_file: str | None = None,
        log_interval: float = 10.0,
        history_size: int = 1000,
        enable_detailed_logging: bool = True,
        start_grafana: bool = True,
        grafana_port: int = 3000,
        grafana_host: str | None = None,
        grafana_image: str = "grafana/grafana-oss:latest",
        grafana_use_docker: bool = False,
        grafana_admin_user: str = "admin",
        grafana_admin_password: str = "admin",
        grafana_allow_anonymous: bool = True,
        grafana_datasource_name: str = "eSurge Prometheus",
        grafana_datasource_uid: str | None = None,
        grafana_datasource_url: str | None = None,
    ) -> dict[str, str]:
        """Start Prometheus-based monitoring for the engine.

        Initializes the Prometheus metrics exporter, optional console monitor,
        and (by default) tries to auto-start a Grafana instance with a
        pre-provisioned Prometheus data source (local grafana-server first,
        optionally Docker if enabled).

        Args:
            dashboard_port: Deprecated; no longer used (kept for compatibility).
            prometheus_port: Port for Prometheus metrics endpoint.
            dashboard_host: Deprecated; no longer used (kept for compatibility).
            enable_prometheus: Start Prometheus metrics server.
            enable_dashboard: Deprecated; no longer used (kept for compatibility).
            enable_console: Start console monitor with rich display.
            log_file: Optional file path for metrics logging.
            log_interval: Interval in seconds between metric logs.
            history_size: Number of historical metrics to retain.
            enable_detailed_logging: Enable detailed metric logging.
            start_grafana: Auto-start Grafana (via Docker) pointed at the Prometheus endpoint.
            grafana_port: Host port to expose Grafana.
            grafana_host: Hostname to use when reporting Grafana URL (defaults to localhost).
            grafana_image: Docker image for Grafana (used when grafana_use_docker=True).
            grafana_use_docker: Allow falling back to Docker for Grafana when local server is unavailable.
            grafana_admin_user: Admin username for Grafana.
            grafana_admin_password: Admin password for Grafana.
            grafana_allow_anonymous: Allow anonymous admin access to Grafana (for quick local use).
            grafana_datasource_name: Display name for the auto-provisioned Prometheus data source.
            grafana_datasource_uid: UID for the Prometheus data source (auto-generated if None).
            grafana_datasource_url: Override URL for the Prometheus data source inside Grafana.

        Returns:
            Dictionary of service URLs:
            - 'prometheus': Prometheus metrics endpoint
            - 'grafana': Grafana UI (when auto-start succeeds)
        """
        if self._monitoring_initialized:
            if start_grafana and not self._grafana_container_name and not self._grafana_url:
                existing_urls = self._monitoring_urls or {}
                prometheus_url = existing_urls.get("prometheus")
                grafana_url = self._start_grafana_service(
                    prometheus_url=prometheus_url,
                    grafana_host=grafana_host or dashboard_host,
                    grafana_port=grafana_port,
                    grafana_image=grafana_image,
                    grafana_admin_user=grafana_admin_user,
                    grafana_admin_password=grafana_admin_password,
                    allow_anonymous=grafana_allow_anonymous,
                    datasource_name=grafana_datasource_name,
                    datasource_uid=grafana_datasource_uid,
                    datasource_url=grafana_datasource_url,
                    use_docker=grafana_use_docker,
                )
                if grafana_url:
                    existing_urls["grafana"] = grafana_url
                    self._monitoring_urls = existing_urls
            self._info("Monitoring already initialized for this eSurge instance")
            return self._monitoring_urls or {}

        self._info("Starting eSurge monitoring services (Prometheus exporter)...")

        if not get_metrics_collector():
            initialize_metrics(
                log_file=log_file,
                log_interval=log_interval,
                history_size=history_size,
                enable_detailed_logging=enable_detailed_logging,
            )
            self._info(" Metrics collection initialized")

        urls: dict[str, str] = {}

        if enable_prometheus:
            try:
                from ..monitoring import start_monitoring_server

                self._monitoring_server = start_monitoring_server(prometheus_port=prometheus_port, update_interval=1.0)
                host_for_logs = dashboard_host or "0.0.0.0"
                urls["prometheus"] = f"http://{host_for_logs}:{prometheus_port}/metrics"
                self._info(f" Prometheus metrics: {urls['prometheus']}")
                self._info(" Point Grafana at the Prometheus endpoint to visualize eSurge metrics.")
            except ImportError:
                self._info(" Prometheus monitoring unavailable (install prometheus-client)")
            except Exception as e:
                self._info(f" Failed to start Prometheus server: {e}")
        elif start_grafana:
            self._info(" Grafana autostart skipped because Prometheus exporter is disabled")

        if enable_dashboard or dashboard_port or dashboard_host:
            self._info(
                " The built-in web dashboard has been removed. "
                "Use Prometheus + Grafana (or another Prometheus UI) for charts."
            )

        if start_grafana and enable_prometheus:
            grafana_url = self._start_grafana_service(
                prometheus_url=urls.get("prometheus"),
                grafana_host=grafana_host or dashboard_host,
                grafana_port=grafana_port,
                grafana_image=grafana_image,
                grafana_admin_user=grafana_admin_user,
                grafana_admin_password=grafana_admin_password,
                allow_anonymous=grafana_allow_anonymous,
                datasource_name=grafana_datasource_name,
                datasource_uid=grafana_datasource_uid,
                datasource_url=grafana_datasource_url,
                use_docker=grafana_use_docker,
            )
            if grafana_url:
                urls["grafana"] = grafana_url
                self._info(f" Grafana UI: {grafana_url}")

        if enable_console:
            try:
                from ..monitoring import start_console_monitor

                self._info(" Starting console monitor...")
                start_console_monitor(refresh_rate=1.0)
            except ImportError:
                self._info(" Console monitor unavailable (install rich)")
            except Exception as e:
                self._info(f" Failed to start console monitor: {e}")

        self._monitoring_initialized = True
        if urls:
            self._info(" Monitoring services started successfully!")
            self._info(" Metrics will be automatically collected during inference")
        else:
            self._info(" No monitoring services were started successfully")
        self._monitoring_urls = urls
        return urls

    def stop_monitoring(self) -> None:
        """Stop all monitoring services.

        Gracefully shuts down Prometheus server and console monitor
        if they are running.
        """
        if not self._monitoring_initialized:
            self._info("No monitoring services to stop")
            return
        self._info("Stopping eSurge monitoring services...")

        if self._monitoring_server:
            try:
                self._monitoring_server.stop()
                self._info(" Prometheus server stopped")
            except Exception as e:
                self._info(f" Error stopping Prometheus server: {e}")
            self._monitoring_server = None

        # Stop the console monitor if it was started
        try:
            from .. import monitoring as _mon_module

            if _mon_module._console_monitor is not None:
                _mon_module._console_monitor.stop()
                _mon_module._console_monitor = None
        except Exception:
            pass

        self._stop_grafana_service()

        self._monitoring_initialized = False
        self._monitoring_urls = None
        self._info(" Monitoring services stopped")

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get current performance metrics summary.

        Returns:
            Dictionary containing:
            - requests_per_second: Current request throughput
            - average_latency: Average request latency
            - average_ttft: Average time to first token
            - average_throughput: Average tokens/second
            - total_completed: Total completed requests
            - total_failed: Total failed requests
            - total_tokens: Total tokens generated
            - active_requests: Currently active requests
            - queue_size: Pending requests in queue
            - running_requests: Currently running requests
        """
        metrics_collector = get_metrics_collector()
        if not metrics_collector:
            return {"error": "Metrics collection not initialized"}
        system_metrics = metrics_collector.get_system_metrics()
        return {
            "requests_per_second": system_metrics.requests_per_second,
            "average_latency": system_metrics.average_latency,
            "average_ttft": system_metrics.average_ttft,
            "average_throughput": system_metrics.average_throughput,
            "total_completed": system_metrics.total_requests_completed,
            "total_failed": system_metrics.total_requests_failed,
            "total_tokens": system_metrics.total_tokens_generated,
            "active_requests": len(self._active_requests),
            "queue_size": self.num_pending_requests,
            "running_requests": self.num_running_requests,
        }

    @property
    def monitoring_active(self) -> bool:
        """Check if monitoring services are currently active.

        Returns:
            True if monitoring has been initialized and is running.
        """
        return self._monitoring_initialized
