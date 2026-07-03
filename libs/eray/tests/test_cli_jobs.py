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

"""Tests for eray.cli.jobs — run/status/logs/stop."""

import time
from types import SimpleNamespace

from click.testing import CliRunner
from eray.cli import jobs


class FakeClient:
    """Minimal JobSubmissionClient stand-in capturing calls."""

    def __init__(self, jobs_list=None, logs=""):
        self.jobs_list = jobs_list or []
        self.logs = logs
        self.submitted = None
        self.stopped = None

    def list_jobs(self):
        return self.jobs_list

    def get_job_logs(self, submission_id):
        return self.logs

    def submit_job(self, *, entrypoint, submission_id, runtime_env, metadata):
        self.submitted = {
            "entrypoint": entrypoint,
            "submission_id": submission_id,
            "runtime_env": runtime_env,
            "metadata": metadata,
        }
        return submission_id

    def stop_job(self, submission_id):
        self.stopped = submission_id


def _job(sid, status, entrypoint="python launch.py", age_s=60.0, ended=False):
    now_ms = time.time() * 1000
    return SimpleNamespace(
        submission_id=sid,
        status=status,
        entrypoint=entrypoint,
        start_time=now_ms - age_s * 1000,
        end_time=now_ms - 1000 if ended else None,
    )


class TestEnvInheritance:
    def test_deny_list_filters_host_state(self):
        env = {
            "PATH": "/usr/bin",
            "HOME": "/home/x",
            "SSH_AUTH_SOCK": "/tmp/s",
            "RAY_ADDRESS": "auto",
            "LD_LIBRARY_PATH": "/lib",
            "VIRTUAL_ENV": "/venv",
            "HF_TOKEN": "hf_secret",
            "WANDB_API_KEY": "wandb_x",
            "PYTHONPATH": "libs/easydel",
            "TOKENIZERS_PARALLELISM": "true",
        }
        out = jobs.inherited_env(env)
        assert set(out) == {"HF_TOKEN", "WANDB_API_KEY", "PYTHONPATH", "TOKENIZERS_PARALLELISM"}

    def test_mask_value_hides_secrets(self):
        assert jobs.mask_value("HF_TOKEN", "hf_abcdefghijklmno") == "hf_a…(18 chars)"
        assert jobs.mask_value("TOKENIZERS_PARALLELISM", "true") == "true"


class TestAddressResolution:
    def test_variants(self, monkeypatch):
        monkeypatch.delenv("RAY_ADDRESS", raising=False)
        assert jobs.resolve_address("http://1.2.3.4:8265") == "http://1.2.3.4:8265"
        assert jobs.resolve_address("1.2.3.4:8265") == "http://1.2.3.4:8265"
        assert jobs.resolve_address("1.2.3.4:6379") == "http://1.2.3.4:8265"
        assert jobs.resolve_address("1.2.3.4") == "http://1.2.3.4:8265"
        assert jobs.resolve_address(None) == jobs.DEFAULT_DASHBOARD
        monkeypatch.setenv("RAY_ADDRESS", "http://10.0.0.1:8265")
        assert jobs.resolve_address(None) == "http://10.0.0.1:8265"


class TestVerdicts:
    def test_succeeded_with_traceback_is_failed(self):
        err, _ = jobs.scan_log_tail("blah\nTraceback (most recent call last):\n  ...")
        assert err == "remote-raise"
        assert jobs.verdict_for("SUCCEEDED", err) == "failed(remote-raise)"

    def test_clean_succeeded_is_ok(self):
        err, _ = jobs.scan_log_tail("all done")
        assert err is None
        assert jobs.verdict_for("SUCCEEDED", None) == "ok"

    def test_signature_precedence(self):
        text = "Failed to merge the Job's runtime env\nTraceback (most recent call last):"
        err, _ = jobs.scan_log_tail(text)
        assert err == "env-conflict"

    def test_phase_step_with_metric(self):
        text = "loaded state step: 0\n{'kl_loss': 2.701, 'train_step': 7}"
        _, phase = jobs.scan_log_tail(text)
        assert phase == "step 7 (kl 2.701)"

    def test_phase_marker_last_wins(self):
        text = "Uploading package\nloaded state step: 0"
        _, phase = jobs.scan_log_tail(text)
        assert phase == "loaded"


class TestPackageGuard:
    def test_skips_excluded_dirs(self, tmp_path):
        (tmp_path / "keep.bin").write_bytes(b"x" * 1000)
        git = tmp_path / ".git"
        git.mkdir()
        (git / "big.pack").write_bytes(b"y" * 100_000)
        assert jobs.package_size_bytes(tmp_path) == 1000


class TestRunCommand:
    def test_defaults_inject_env_and_working_dir(self, monkeypatch, tmp_path):
        fake = FakeClient()
        monkeypatch.setattr(jobs, "make_client", lambda address: fake)
        monkeypatch.setattr(jobs, "_history_append", lambda record: None)
        monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
            result = runner.invoke(jobs.run, ["--", "python", "launch.py"])
            assert result.exit_code == 0, result.output
            sub = fake.submitted
            assert sub["entrypoint"] == "python launch.py"
            assert sub["runtime_env"]["working_dir"] == fs
            assert sub["runtime_env"]["env_vars"]["HF_TOKEN"] == "hf_secret_value"
            assert "PATH" not in sub["runtime_env"]["env_vars"]
            assert sub["submission_id"].startswith("launch-")

    def test_env_override_and_no_working_dir(self, monkeypatch, tmp_path):
        fake = FakeClient()
        monkeypatch.setattr(jobs, "make_client", lambda address: fake)
        monkeypatch.setattr(jobs, "_history_append", lambda record: None)
        monkeypatch.setenv("HF_TOKEN", "from_shell")
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                jobs.run,
                ["--no-working-dir", "--env", "HF_TOKEN=override", "--id", "myjob", "--", "python", "x.py"],
            )
            assert result.exit_code == 0, result.output
            sub = fake.submitted
            assert "working_dir" not in sub["runtime_env"]
            assert sub["runtime_env"]["env_vars"]["HF_TOKEN"] == "override"
            assert sub["submission_id"] == "myjob"


class TestStatusCommand:
    def test_verdict_column_and_exit_code(self, monkeypatch):
        lying = _job("lying-job", "SUCCEEDED", ended=True)
        healthy = _job("healthy-job", "RUNNING")
        fake = FakeClient(jobs_list=[lying, healthy])
        monkeypatch.setattr(jobs, "make_client", lambda address: fake)
        monkeypatch.setattr(
            jobs,
            "get_log_tail",
            lambda client, sid: "Traceback (most recent call last):" if sid == "lying-job" else "{'train_step': 3}",
        )
        runner = CliRunner()
        result = runner.invoke(jobs.status, [])
        assert "failed(remote-raise)" in result.output
        assert "step 3" in result.output
        assert result.exit_code == 1  # a failing verdict is CI-visible

    def test_all_healthy_exits_zero(self, monkeypatch):
        fake = FakeClient(jobs_list=[_job("ok-job", "RUNNING")])
        monkeypatch.setattr(jobs, "make_client", lambda address: fake)
        monkeypatch.setattr(jobs, "get_log_tail", lambda client, sid: "loaded state step: 0")
        runner = CliRunner()
        result = runner.invoke(jobs.status, ["--json"])
        assert result.exit_code == 0, result.output
        assert '"verdict": "-"' in result.output


class TestStopCommand:
    def test_stop_last(self, monkeypatch):
        older = _job("older", "STOPPED", age_s=600, ended=True)
        newest = _job("newest", "RUNNING", age_s=30)
        fake = FakeClient(jobs_list=[older, newest])
        monkeypatch.setattr(jobs, "make_client", lambda address: fake)
        runner = CliRunner()
        result = runner.invoke(jobs.stop, ["--last"])
        assert result.exit_code == 0, result.output
        assert fake.stopped == "newest"

    def test_stop_requires_target(self, monkeypatch):
        monkeypatch.setattr(jobs, "make_client", lambda address: FakeClient())
        runner = CliRunner()
        result = runner.invoke(jobs.stop, [])
        assert result.exit_code != 0


class TestPatternsConfig:
    def test_defaults_used_without_override(self):
        err, phase = jobs.scan_log_tail("Traceback (most recent call last):\n{'train_step': 4, 'kl_loss': 1.5}")
        assert err == "remote-raise"
        assert phase == "step 4 (kl 1.5)"

    def test_project_override_replaces_patterns(self, monkeypatch, tmp_path):
        override = tmp_path / "patterns.json"
        override.write_text(
            '{"errors": [["MY_FATAL", "custom-fatal"]], "step_metric": "iter", "progress_metrics": ["ppl"]}'
        )
        monkeypatch.setenv(jobs.PATTERNS_ENV, str(override))
        patterns = jobs.load_patterns()
        err, phase = jobs.scan_log_tail("MY_FATAL happened\n{'iter': 12, 'ppl': 3.5}", patterns)
        assert err == "custom-fatal"
        assert phase == "step 12 (ppl 3.5)"
        # default errors were replaced wholesale, not appended
        err2, _ = jobs.scan_log_tail("Traceback (most recent call last):", patterns)
        assert err2 is None

    def test_unreadable_override_is_ignored(self, monkeypatch, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        monkeypatch.setenv(jobs.PATTERNS_ENV, str(bad))
        patterns = jobs.load_patterns()
        assert patterns["step_metric"] == "train_step"


class TestAlerts:
    def test_alert_fires_on_latest_value(self):
        text = "{'kl_loss': 1.0}\n{'kl_loss': 9.2}"
        assert jobs.evaluate_alerts(text, ("kl_loss>5",)) == ["kl_loss=9.2 > 5.0"]

    def test_alert_quiet_below_threshold_or_missing(self):
        assert jobs.evaluate_alerts("{'kl_loss': 1.0}", ("kl_loss>5", "step_time>120")) == []

    def test_bad_expression_raises(self):
        import click as _click
        import pytest as _pytest

        with _pytest.raises(_click.UsageError):
            jobs.evaluate_alerts("", ("kl_loss !! 5",))


class TestPackagesCleanup:
    def test_stale_detection_respects_referenced(self, tmp_path):
        wdf = tmp_path / "runtime_resources" / "working_dir_files"
        wdf.mkdir(parents=True)
        (wdf / "_ray_pkg_live").mkdir()
        (wdf / "_ray_pkg_live" / "f").write_bytes(b"x" * 10)
        (wdf / "_ray_pkg_stale").mkdir()
        (wdf / "_ray_pkg_stale" / "f").write_bytes(b"y" * 20)
        (wdf / "unrelated").mkdir()

        stale = jobs.find_stale_packages(str(tmp_path), {"_ray_pkg_live"})
        assert [(p.rsplit("/", 1)[-1], s) for p, s in stale] == [("_ray_pkg_stale", 20)]

    def test_referenced_packages_from_live_jobs(self):
        running = _job("a", "RUNNING")
        running.runtime_env = {"working_dir": "gcs://_ray_pkg_live.zip"}
        done = _job("b", "SUCCEEDED", ended=True)
        done.runtime_env = {"working_dir": "gcs://_ray_pkg_old.zip"}
        assert jobs.referenced_packages([running, done]) == {"_ray_pkg_live"}


class TestRerun:
    def test_rerun_resubmits_recorded_env(self, monkeypatch):
        class FakeInfoClient(FakeClient):
            def get_job_info(self, sid):
                return SimpleNamespace(
                    submission_id=sid,
                    entrypoint="python launch.py",
                    runtime_env={"env_vars": {"HF_TOKEN": "x"}},
                    metadata={"git_sha": "abc"},
                    status="STOPPED",
                    start_time=0,
                    end_time=1,
                )

        fake = FakeInfoClient()
        monkeypatch.setattr(jobs, "make_client", lambda address: fake)
        runner = CliRunner()
        result = runner.invoke(jobs.rerun, ["old-job"])
        assert result.exit_code == 0, result.output
        assert fake.submitted["entrypoint"] == "python launch.py"
        assert fake.submitted["runtime_env"] == {"env_vars": {"HF_TOKEN": "x"}}
        assert fake.submitted["metadata"]["rerun_of"] == "old-job"
        assert fake.submitted["submission_id"].startswith("old-job-r")


class TestBounceGate:
    def test_bounce_requires_confirmation(self):
        from eray.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["tpu", "bounce", "--ips", "10.0.0.1", "--tpu-type", "v4-8"])
        assert result.exit_code != 0
        assert "yes-kill-jobs" in result.output


class TestProgressMetricsDisplay:
    def test_all_present_metrics_shown_with_step_time(self):
        text = "{'loss': 2.964, 'train_step': 2, 'kl_loss': 2.964, 'train_step_time': 79.844}"
        _, phase = jobs.scan_log_tail(text)
        # loss == kl_loss in distill logs: deduped; step time always shown
        assert phase == "step 2 (kl 2.964, s/step 79.844)"

    def test_distinct_metrics_all_listed(self):
        text = "{'kl_loss': 1.5, 'loss': 2.0, 'train_step': 9, 'train_step_time': 81.2}"
        _, phase = jobs.scan_log_tail(text)
        assert phase == "step 9 (kl 1.5, loss 2, s/step 81.2)"
