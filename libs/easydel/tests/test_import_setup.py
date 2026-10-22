"""Regression coverage for import setup without initializing any JAX backend."""

import importlib.metadata
import io
import subprocess
import sys
import textwrap
import urllib.request

import easydel
import pytest
from easydel import _linkup

pytestmark = pytest.mark.cpu_ok


def _run_python(source):
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(source)], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_import_accepts_versions_from_updated_distribution_requirements():
    _run_python("""
        import importlib.metadata as metadata

        original_requires = metadata.requires
        original_version = metadata.version
        versions = {"eformer": "9.1.0", "ejkernel": "9.2.0"}
        metadata.requires = lambda name: (
            [f"{key}=={value}" for key, value in versions.items()]
            if name == "easydel" else original_requires(name)
        )
        metadata.version = lambda name: versions.get(name) or original_version(name)
        import easydel
        assert easydel.__version__
    """)


def test_dependency_mismatch_raises_informative_import_error(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "requires", lambda name: ["eformer==9.1.0"])
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.0.0")
    with pytest.raises(ImportError, match=r"eformer==9\.1\.0.*eformer 9\.0\.0"):
        _linkup._validate_dependency_versions()


def test_dependency_ranges_and_inactive_extras_are_respected(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "requires",
        lambda name: ["ejkernel>=9,<10", "ejkernel[cuda]==8; extra == 'cuda'"],
    )
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.2.0")
    _linkup._validate_dependency_versions()


@pytest.mark.parametrize("selector", ["v5litepod-8", "v5lite-8", "ct5lp-hightpu-4t", "V5LITEPOD_16"])
def test_v5e_gce_names_apply_generation_defaults(monkeypatch, selector):
    monkeypatch.setenv("EASYDEL_TARGETED_TPU_GENERATION", selector)
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--user_flag=true")
    monkeypatch.setenv("XLA_FLAGS", "")
    _linkup._maybe_apply_targeted_tpu_flags()
    flags = _linkup._os.environ["LIBTPU_INIT_ARGS"].split()
    assert "--user_flag=true" in flags
    assert "--xla_tpu_enable_async_collective_fusion=true" in flags


def test_v5e_machine_type_metadata_detection(monkeypatch):
    for name in (
        "EASYDEL_TARGETED_TPU_GENERATION",
        "TPU_ACCELERATOR_TYPE",
        "TPU_TYPE",
        "TPU_VERSION",
        "ACCELERATOR_TYPE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        _linkup,
        "_get_metadata_value",
        lambda path: (
            "projects/project/zones/zone/machineTypes/ct5lp-hightpu-4t" if path == "instance/machine-type" else None
        ),
    )
    assert _linkup._detect_tpu_generation_without_jax() == "v5e"


def test_metadata_read_uses_bounded_timeout(monkeypatch):
    def urlopen(request, *, timeout):
        assert request.full_url.endswith("/instance/attributes/accelerator-type")
        assert timeout == 0.05
        return io.BytesIO(b"v5litepod-8\n")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert _linkup._get_metadata_value("instance/attributes/accelerator-type") == "v5litepod-8"


@pytest.mark.parametrize("explicit", [False, True])
def test_traceback_filtering_updates_already_imported_jax(monkeypatch, explicit):
    if explicit:
        monkeypatch.setenv("JAX_TRACEBACK_FILTERING", "auto")
    else:
        monkeypatch.delenv("JAX_TRACEBACK_FILTERING", raising=False)
    monkeypatch.setenv("EASYDEL_AUTO", "1")
    _run_python(f"""
        import jax
        jax.config.update("jax_traceback_filtering", "auto")
        import easydel
        assert jax.config.jax_traceback_filtering == {"auto" if explicit else "off"!r}
    """)


def test_disabled_auto_setup_preserves_traceback_filtering(monkeypatch):
    monkeypatch.setenv("EASYDEL_AUTO", "0")
    monkeypatch.delenv("JAX_TRACEBACK_FILTERING", raising=False)
    _run_python("""
        import jax
        jax.config.update("jax_traceback_filtering", "auto")
        import easydel
        assert jax.config.jax_traceback_filtering == "auto"
    """)


def test_hf_gated_repo_errors_remain_errors_under_pytest(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "gated_repo_regression")
    _run_python("""
        from transformers import AutoConfig, PretrainedConfig

        def gated_config(cls, *args, **kwargs):
            raise OSError("You are trying to access a gated repo")

        PretrainedConfig.get_config_dict = classmethod(gated_config)
        import easydel
        try:
            AutoConfig.from_pretrained("example/gated-model")
        except OSError as error:
            assert "gated repo" in str(error)
        else:
            raise AssertionError("AutoConfig must preserve its OSError contract")
    """)


def test_lazy_embedding_export_remains_available_once():
    assert easydel.__all__.count("AutoEasyDeLModelForEmbedding") == 1
    assert "AutoEasyDeLModelForEmbedding" in dir(easydel)


def test_auto_environment_emits_reduce_scatter_flag_once(monkeypatch):
    monkeypatch.setenv("EASYDEL_AUTO", "1")
    monkeypatch.setenv("XLA_FLAGS", "")
    monkeypatch.setenv("EASYDEL_TARGETED_TPU_GENERATION", "none")
    # Keep all environment changes local to this test.
    with monkeypatch.context() as context:
        context.setattr(_linkup._os, "environ", dict(_linkup._os.environ))
        _linkup._apply_auto_environment()
        flags = _linkup._os.environ["XLA_FLAGS"].split()
    assert flags.count("--xla_gpu_enable_reduce_scatter_combine_by_dim=false") == 1
