"""Host phase timings are configured explicitly and written as JSONL."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from easydel.trainers.trainer.trainer import Trainer
from easydel.trainers.training_configurations import TrainingArguments
from eformer.paths import GCSPath, LocalPath
from google.api_core.exceptions import Forbidden, RetryError, ServiceUnavailable
from google.auth.exceptions import DefaultCredentialsError, RefreshError

pytestmark = pytest.mark.cpu_ok


def test_host_phase_log_appends_complete_records(tmp_path, monkeypatch):
    monkeypatch.setattr("easydel.trainers.trainer.trainer.jax.process_index", lambda: 2)
    path = tmp_path / "phases.jsonl"
    host = SimpleNamespace(
        arguments=SimpleNamespace(host_phase_log_path=str(path)),
        _last_step_put_dispatch_seconds=0.12345,
        _last_step_device_sync_seconds=0.23456,
        _last_step_preprocess_seconds=0.34567,
    )
    Trainer._append_host_phase_log(host)
    Trainer._append_host_phase_log(host)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    for record in records:
        assert record["proc"] == 2
        assert record["put"] == 0.123
        assert record["dev"] == 0.235
        assert record["pre"] == 0.346
        assert isinstance(record["t"], float)


def test_host_phase_path_round_trips_in_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(TrainingArguments, "_setup_distributed", lambda self: None)
    args = TrainingArguments(host_phase_log_path=str(tmp_path / "phases.jsonl"), use_wandb=False, report_metrics=False)
    serialized = json.loads(args.to_json_string())
    assert serialized["host_phase_log_path"] == str(tmp_path / "phases.jsonl")


def test_host_phase_path_does_not_shift_existing_positional_arguments():
    signature = inspect.signature(TrainingArguments)
    assert signature.parameters["host_phase_log_path"].kind is inspect.Parameter.KEYWORD_ONLY
    positional_names = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    index = positional_names.index("training_time_limit")
    assert positional_names[index + 1] == "train_on_inputs"
    values = [object() for _ in positional_names[: index + 3]]
    bound = signature.bind_partial(*values, host_phase_log_path="timings.jsonl")
    assert bound.arguments["train_on_inputs"] is values[index + 1]
    assert bound.arguments["trainer_prefix"] is values[index + 2]
    assert bound.arguments["host_phase_log_path"] == "timings.jsonl"


def test_local_log_never_reads_or_rewrites_previous_records(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "phases.jsonl"
    host = SimpleNamespace(arguments=SimpleNamespace(host_phase_log_path=str(path)))
    monkeypatch.setattr("easydel.trainers.trainer.trainer.jax.process_index", lambda: 0)

    def forbid_rewrite(*args, **kwargs):
        raise AssertionError("Appending must not read or rewrite the accumulated log")

    monkeypatch.setattr(LocalPath, "read_text", forbid_rewrite)
    monkeypatch.setattr(LocalPath, "write_text", forbid_rewrite)
    for _ in range(100):
        Trainer._append_host_phase_log(host)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 100
    assert all(record["proc"] == 0 for record in records)


def test_failed_local_append_preserves_existing_records(tmp_path, monkeypatch):
    path = tmp_path / "phases.jsonl"
    previous = '{"proc": 7, "t": 1.0}\n'
    path.write_text(previous)
    host = SimpleNamespace(arguments=SimpleNamespace(host_phase_log_path=str(path)))
    monkeypatch.setattr("easydel.trainers.trainer.trainer.jax.process_index", lambda: 0)
    original_open = Path.open

    class FailedAppend:
        def __enter__(self):
            self.stream = original_open(path, "a", encoding="utf-8")
            return self

        def write(self, record):
            self.stream.write(record[:5])
            raise OSError("disk full during append")

        def __exit__(self, *args):
            self.stream.close()

    def open_with_failure(self, mode="r", *args, **kwargs):
        if self == path:
            assert mode == "a", "An append failure must never truncate the original log"
            return FailedAppend()
        return original_open(self, mode, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, "open", open_with_failure)
        Trainer._append_host_phase_log(host)
    assert path.read_text().startswith(previous)
    assert json.loads(path.read_text().splitlines()[0]) == {"proc": 7, "t": 1.0}


def test_cloud_log_uses_bounded_immutable_chunks_and_preserves_records_on_failure(monkeypatch):
    prefix = "gs://bucket/host-timings.jsonl"
    previous = '{"proc": 7, "t": 1.0}\n'
    objects = {prefix: previous, f"{prefix}/old.jsonl": previous}
    write_sizes = []
    fail_write = False

    class MemoryCloudPath:
        """Host-only object-store stand-in; forbids reads and overwriting any object."""

        def __init__(self, path):
            self.path = path

        def __truediv__(self, child):
            return MemoryCloudPath(f"{self.path}/{child}")

        def read_text(self):
            raise AssertionError("Cloud appends must not download earlier records")

        def write_text(self, record):
            assert self.path not in objects
            write_sizes.append(len(record))
            if fail_write:
                raise OSError("upload failed")
            objects[self.path] = record

    monkeypatch.setattr("easydel.trainers.trainer.trainer.ePath", MemoryCloudPath)
    monkeypatch.setattr("easydel.trainers.trainer.trainer.jax.process_index", lambda: 2)
    # Even identical clock ticks cannot collide across steps or restarts.
    monkeypatch.setattr("easydel.trainers.trainer.trainer._time.time_ns", lambda: 1)
    host = SimpleNamespace(arguments=SimpleNamespace(host_phase_log_path=prefix))
    for _ in range(100):
        Trainer._append_host_phase_log(host)
    assert len(objects) == 102
    assert objects[prefix] == previous
    assert objects[f"{prefix}/old.jsonl"] == previous
    records = [json.loads(value) for key, value in objects.items() if key not in (prefix, f"{prefix}/old.jsonl")]
    assert all(record["proc"] == 2 for record in records)
    assert max(write_sizes) < 256
    snapshot = objects.copy()
    fail_write = True
    Trainer._append_host_phase_log(host)
    assert objects == snapshot


@pytest.mark.parametrize(
    "error",
    [
        Forbidden("permission denied"),
        ServiceUnavailable("storage unavailable"),
        RetryError("upload retries exhausted", cause=ServiceUnavailable("storage unavailable")),
        RefreshError("credentials expired"),
        RuntimeError("backend unavailable"),
    ],
    ids=["forbidden", "service-unavailable", "retry-exhausted", "credentials-refresh", "backend-runtime"],
)
def test_gcs_sdk_upload_failures_preserve_records_and_allow_subsequent_logging(monkeypatch, error):
    """Exercise GCSPath itself with an in-memory client, never a live cloud write."""
    prefix = "gs://bucket/host-timings.jsonl"
    previous = '{"proc": 7, "t": 1.0}\n'
    objects = {prefix: previous, f"{prefix}/old.jsonl": previous}
    fail_write = True

    def blob(name):
        def upload_from_string(record, **kwargs):
            if fail_write:
                raise error
            key = f"gs://bucket/{name}"
            assert key not in objects
            objects[key] = record

        return SimpleNamespace(upload_from_string=upload_from_string)

    client = SimpleNamespace(bucket=lambda name: SimpleNamespace(blob=blob))
    monkeypatch.setattr("easydel.trainers.trainer.trainer.ePath", lambda path: GCSPath(path, client=client))
    monkeypatch.setattr("easydel.trainers.trainer.trainer.jax.process_index", lambda: 2)
    host = SimpleNamespace(arguments=SimpleNamespace(host_phase_log_path=prefix))
    Trainer._append_host_phase_log(host)
    assert objects == {prefix: previous, f"{prefix}/old.jsonl": previous}
    fail_write = False
    Trainer._append_host_phase_log(host)
    assert len(objects) == 3
    assert objects[prefix] == previous
    assert objects[f"{prefix}/old.jsonl"] == previous
    new_record = next(value for key, value in objects.items() if key not in (prefix, f"{prefix}/old.jsonl"))
    assert json.loads(new_record)["proc"] == 2


@pytest.mark.parametrize("error_type", [DefaultCredentialsError, RefreshError])
def test_gcs_credential_initialization_failure_is_nonfatal(monkeypatch, error_type):
    def unavailable_client():
        raise error_type("credentials unavailable")

    # Use the real ePath/GCSPath dispatch, but prevent any credential/network lookup.
    monkeypatch.setattr("eformer.paths.storage.Client", unavailable_client)
    monkeypatch.setattr("easydel.trainers.trainer.trainer.jax.process_index", lambda: 0)
    host = SimpleNamespace(arguments=SimpleNamespace(host_phase_log_path="gs://bucket/phases"))
    Trainer._append_host_phase_log(host)


@pytest.mark.parametrize("signal_type", [KeyboardInterrupt, SystemExit])
def test_host_log_does_not_swallow_cancellation(monkeypatch, signal_type):
    def interrupted_path(path):
        raise signal_type()

    monkeypatch.setattr("easydel.trainers.trainer.trainer.ePath", interrupted_path)
    monkeypatch.setattr("easydel.trainers.trainer.trainer.jax.process_index", lambda: 0)
    host = SimpleNamespace(arguments=SimpleNamespace(host_phase_log_path="gs://bucket/phases"))
    with pytest.raises(signal_type):
        Trainer._append_host_phase_log(host)


def test_disabled_host_log_performs_no_path_io(monkeypatch):
    def forbid_io(path):
        raise AssertionError("Disabled logging must not resolve or write a path")

    monkeypatch.setattr("easydel.trainers.trainer.trainer.ePath", forbid_io)
    Trainer._append_host_phase_log(SimpleNamespace(arguments=SimpleNamespace(host_phase_log_path=None)))
