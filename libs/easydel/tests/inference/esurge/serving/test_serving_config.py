"""Serving package: YAML validation, API-key file, coordination token, bind addresses and engine kwargs."""

from __future__ import annotations

import os
import stat
import threading
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.serving import ApiKeyFileError, ServingConfig, ServingConfigError, load_or_create_api_key
from easydel.inference.esurge.serving.__main__ import main
from easydel.inference.esurge.serving.app import build_engine_sections, build_loading_kwargs
from easydel.inference.esurge.serving.cluster import public_urls, resolve_bind_host, shared_coordination_token

pytestmark = pytest.mark.cpu_ok

GLM = {
    "model": {
        "checkpoint": "gs://bucket/GLM-5.3-Flash-easydel-bf16",
        "tokenizer": "/opt/glm/tokenizer",
        "served_name": "GLM-5.3-Flash-W8A8",
        "sharding_axis_dims": [1, 4, 1, 4, 4, 1],
        "attn_mechanism": "multi_latent_ragged_page_attention_v2",
        "quantization": {"dtype": "w8a8"},
    },
    "esurge": {
        "runtime": {"max_model_len": 131072, "max_num_seqs": 128},
        "cache": {"hbm_utilization": 0.97, "page_size": 128, "data_parallelism_axis": "dp"},
    },
    "server": {"port": 8000, "api_key_file": "/tmp/k"},
}


def test_full_deployment_parses_with_defaults():
    config = ServingConfig.from_mapping(GLM)
    assert config.model.sharding_axis_dims == (1, 4, 1, 4, 4, 1)
    assert config.model.quantization.dtype == "w8a8" and config.model.quantization.pattern is None
    assert config.model.tokenizer_source == "/opt/glm/tokenizer"
    assert config.server.host == "internal"
    assert config.coordination.bind_host == "internal"
    assert config.coordination.control_port == 19666


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        ({"extra": 1}, "unknown top-level"),
        ({"model": {"checkpoint": "x", "nope": 1}}, "unknown keys"),
        ({"model": {"checkpoint": ""}}, "checkpoint"),
        ({"model": {"checkpoint": "x", "sharding_axis_dims": [1, 2]}}, "6 entries"),
        ({"model": {"checkpoint": "x", "sharding_axis_dims": [-1, -1, 1, 1, 1, 1]}}, "at most one -1"),
        ({"server": {"port": 70000}}, "port"),
        ({"server": {"api_key_file": ""}}, "api_key_file"),
        ({"esurge": {"distributed": {}}}, "owned by the `coordination`"),
        ({"coordination": {"heartbeat_timeout_s": 0}}, "positive"),
    ],
)
def test_invalid_deployments_are_rejected(patch, match):
    with pytest.raises(ServingConfigError, match=match):
        ServingConfig.from_mapping({**GLM, **patch})


def test_cli_check_validates_without_loading_a_model(tmp_path, capsys):
    good = tmp_path / "good.yaml"
    good.write_text("model: {checkpoint: some/model}\n")
    assert main(["check", "--config", str(good)]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: {checkpoint: some/model}\nserver: {port: 0}\n")
    assert main(["check", "--config", str(bad)]) == 2
    assert "server.port" in capsys.readouterr().err


def test_api_key_file_is_created_private_and_stable(tmp_path):
    path = tmp_path / "nested" / "key"
    key, created = load_or_create_api_key(path)
    assert created and key.startswith("sk-") and len(key) >= 32
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load_or_create_api_key(path) == (key, False)


def test_api_key_file_creation_race_yields_one_key(tmp_path):
    path = tmp_path / "key"
    keys: list[str] = []
    threads = [threading.Thread(target=lambda: keys.append(load_or_create_api_key(path)[0])) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(keys)) == 1


def test_unusable_api_key_files_fail_closed(tmp_path):
    shared = tmp_path / "shared"
    shared.write_text("sk-" + "a" * 40)
    os.chmod(shared, 0o644)
    with pytest.raises(ApiKeyFileError, match="other users"):
        load_or_create_api_key(shared)
    short = tmp_path / "short"
    short.write_text("sk-short")
    os.chmod(short, 0o600)
    with pytest.raises(ApiKeyFileError, match="at least"):
        load_or_create_api_key(short)
    empty = tmp_path / "empty"
    empty.write_text("\n")
    os.chmod(empty, 0o600)
    with pytest.raises(ApiKeyFileError):
        load_or_create_api_key(empty)


class _Store:
    def __init__(self):
        self.values = {}
        self.cond = threading.Condition()

    def key_value_set(self, key, value, allow_overwrite=False):
        with self.cond:
            if key in self.values and not allow_overwrite:
                raise RuntimeError("ALREADY_EXISTS")
            self.values[key] = value
            self.cond.notify_all()

    def blocking_key_value_get(self, key, timeout_in_ms):
        with self.cond:
            if not self.cond.wait_for(lambda: key in self.values, timeout_in_ms / 1000):
                raise TimeoutError(key)
            return self.values[key]


def test_coordination_token_is_shared_per_launch_and_never_reused():
    store = _Store()
    results: list[str] = []
    worker = threading.Thread(
        target=lambda: results.append(shared_coordination_token(19666, is_leader=False, timeout_s=5, client=store))
    )
    worker.start()
    leader_token = shared_coordination_token(19666, is_leader=True, timeout_s=5, client=store)
    worker.join()
    assert results == [leader_token] and len(leader_token) >= 40
    with pytest.raises(RuntimeError):
        shared_coordination_token(19666, is_leader=True, timeout_s=5, client=store)
    with pytest.raises(RuntimeError, match="did not publish"):
        shared_coordination_token(1, is_leader=False, timeout_s=0.01, client=store)


def test_bind_hosts_and_urls():
    assert resolve_bind_host("10.0.0.5") == "10.0.0.5"
    assert resolve_bind_host("internal", internal_ip=lambda: "10.0.0.7") == "10.0.0.7"
    with pytest.raises(RuntimeError):
        resolve_bind_host("internal", internal_ip=lambda: None)
    ips = {"internal_ip": lambda: "10.0.0.7", "external_ip": lambda: "34.1.2.3"}
    assert public_urls("10.0.0.7", 8000, **ips) == {"internal": "http://10.0.0.7:8000/v1"}
    assert public_urls("0.0.0.0", 8000, **ips) == {
        "internal": "http://10.0.0.7:8000/v1",
        "external": "http://34.1.2.3:8000/v1",
    }


def test_engine_sections_add_zmq_only_for_multi_host():
    config = ServingConfig.from_mapping(GLM)
    single = build_engine_sections(config, world_size=1, token=None, control_bind_host=None)
    assert "distributed" not in single and single["cache"]["page_size"] == 128
    multi = build_engine_sections(config, world_size=16, token="t" * 43, control_bind_host="10.0.0.7")
    assert multi["distributed"] == {
        "coordination": "zmq",
        "distributed_auth_token": "t" * 43,
        "distributed_control_port": 19666,
        "distributed_control_bind_host": "10.0.0.7",
        "distributed_heartbeat_timeout_s": 120.0,
    }
    with pytest.raises(ValueError):
        build_engine_sections(config, world_size=16, token=None, control_bind_host="10.0.0.7")
    with pytest.raises(ValueError, match="unknown sections"):
        build_engine_sections(
            ServingConfig.from_mapping({**GLM, "esurge": {"runtme": {}}}),
            world_size=1,
            token=None,
            control_bind_host=None,
        )


def test_loading_kwargs_use_the_model_quantization_pattern():
    ed = SimpleNamespace(QuantizationConfig=lambda **kw: kw)
    text = SimpleNamespace(default_quantization_pattern=lambda dtype: f"pattern-for-{dtype}")
    kwargs = build_loading_kwargs(ServingConfig.from_mapping(GLM), text, ed=ed)
    assert kwargs["sharding_axis_dims"] == (1, 4, 1, 4, 4, 1)
    assert kwargs["sharding_axis_names"] == ("pp", "dp", "fsdp", "ep", "tp", "sp")
    assert kwargs["config_kwargs"] == {"attn_mechanism": "multi_latent_ragged_page_attention_v2"}
    assert kwargs["quantization_config"] == {"dtype": "w8a8", "pattern": "pattern-for-w8a8"}
    with pytest.raises(ValueError, match="default_quantization_pattern"):
        build_loading_kwargs(ServingConfig.from_mapping(GLM), object(), ed=ed)
