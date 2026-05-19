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

"""Import-time compatibility and environment setup for EasyDeL."""

from __future__ import annotations

import os as _os
import pickle as _pickle
import sys as _sys
import types as _types
from importlib import machinery as _machinery
from importlib import util as _importlib_util
from logging import getLogger as _getlogger

from .utils import check_bool_flag as _check_bool_flag

_TPU_FLAGS_BY_GENERATION = {
    "v4": ("--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE "),
    "v5e": (
        "--xla_tpu_enable_data_parallel_all_reduce_opt=true "
        "--xla_tpu_data_parallel_opt_different_sized_ops=true "
        "--xla_tpu_enable_async_collective_fusion=true "
        "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
        "--xla_tpu_overlap_compute_collective_tc=true "
        "--xla_enable_async_all_gather=true "
    ),
    "v5p": (
        "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
        "--xla_tpu_megacore_fusion_allow_ags=false "
        "--xla_enable_async_collective_permute=true "
        "--xla_tpu_enable_ag_backward_pipelining=true "
        "--xla_tpu_enable_data_parallel_all_reduce_opt=true "
        "--xla_tpu_data_parallel_opt_different_sized_ops=true "
        "--xla_tpu_enable_async_collective_fusion=true "
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
        "--xla_tpu_overlap_compute_collective_tc=true "
        "--xla_enable_async_all_gather=true "
    ),
    "v6e": (
        "--xla_tpu_scoped_vmem_limit_kib=98304 "
        "--xla_enable_async_all_gather=true "
        "--xla_tpu_overlap_compute_collective_tc=true "
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
        "--xla_tpu_enable_async_collective_fusion=true "
        "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
    ),
}

_TPU_GENERATION_ALIASES = {
    "4": "v4",
    "tpu-v4": "v4",
    "v4": "v4",
    "5e": "v5e",
    "tpu-v5e": "v5e",
    "v5e": "v5e",
    "5p": "v5p",
    "tpu-v5p": "v5p",
    "v5p": "v5p",
    "6": "v6e",
    "6e": "v6e",
    "max-6": "v6e",
    "max6": "v6e",
    "tpu-v6e": "v6e",
    "trillium": "v6e",
    "v6": "v6e",
    "v6e": "v6e",
}


def _normalize_tpu_generation(value: str | None) -> str | None:
    """Normalize a user-provided TPU generation selector."""
    if value is None:
        return None
    normalized = value.strip().lower().replace("_", "-")
    if not normalized:
        return None
    if normalized in _TPU_GENERATION_ALIASES:
        return _TPU_GENERATION_ALIASES[normalized]
    for token, generation in (
        ("v6e", "v6e"),
        ("ct6e", "v6e"),
        ("trillium", "v6e"),
        ("v5p", "v5p"),
        ("ct5p", "v5p"),
        ("v5e", "v5e"),
        ("ct5e", "v5e"),
        ("v4", "v4"),
        ("ct4", "v4"),
    ):
        if normalized.startswith(token) or f"-{token}" in normalized:
            return generation
    return None


def _get_metadata_value(path: str) -> str | None:
    """Read a GCE metadata value without importing or initializing JAX."""
    try:
        import urllib.request as _urllib_request

        timeout = float(_os.getenv("EASYDEL_TPU_METADATA_TIMEOUT", "0.05"))
        request = _urllib_request.Request(
            f"http://169.254.169.254/computeMetadata/v1/{path}",
            headers={"Metadata-Flavor": "Google"},
        )
        with _urllib_request.urlopen(request, timeout=max(timeout, 0.001)) as response:
            value = response.read(128).decode("utf-8", errors="ignore").strip()
    except Exception:
        return None
    return value or None


def _detect_tpu_generation_without_jax() -> str | None:
    """Infer TPU generation from environment or metadata without touching JAX."""
    for env_name in (
        "EASYDEL_TARGETED_TPU_GENERATION",
        "TPU_ACCELERATOR_TYPE",
        "TPU_TYPE",
        "TPU_VERSION",
        "ACCELERATOR_TYPE",
    ):
        generation = _normalize_tpu_generation(_os.getenv(env_name))
        if generation is not None:
            return generation

    for metadata_path in (
        "instance/attributes/accelerator-type",
        "instance/attributes/acceleratorType",
        "instance/machine-type",
    ):
        generation = _normalize_tpu_generation(_get_metadata_value(metadata_path))
        if generation is not None:
            return generation
    return None


def _maybe_apply_targeted_tpu_flags() -> None:
    """Apply TPU flags for an explicit or lazily detected generation."""
    explicit_generation = _os.getenv("EASYDEL_TARGETED_TPU_GENERATION")
    generation = _normalize_tpu_generation(explicit_generation)
    if explicit_generation is None or not explicit_generation.strip():
        generation = _detect_tpu_generation_without_jax()
    if generation is None:
        return
    flags = _TPU_FLAGS_BY_GENERATION.get(generation)
    if not flags:
        return
    if explicit_generation is None or not explicit_generation.strip():
        _os.environ["EASYDEL_TARGETED_TPU_GENERATION"] = generation
    _os.environ["LIBTPU_INIT_ARGS"] = (_os.getenv("LIBTPU_INIT_ARGS", "") + " " + flags).strip()


def _ensure_optional_deepspeed_stub() -> None:
    """Provide a minimal deepspeed module for remote-code import checks."""
    try:
        if _importlib_util.find_spec("deepspeed") is not None:
            return
    except (ModuleNotFoundError, ValueError):
        return

    if "deepspeed" in _sys.modules:
        return

    _stub = _types.ModuleType("deepspeed")
    _stub.__version__ = "0.0.0"
    _stub.__spec__ = _machinery.ModuleSpec(name="deepspeed", loader=None)
    _sys.modules["deepspeed"] = _stub


def _patch_removed_jax_config_flags() -> None:
    """Ignore config flags that older dependencies may set on newer JAX."""
    try:
        import jax as _jax
    except Exception:
        return

    config = getattr(_jax, "config", None)
    update = getattr(config, "update", None)
    if update is None or getattr(update, "_easydel_removed_flag_patch", False):
        return

    removed_flags = {"jax_pmap_shmap_merge"}

    def _patched_update(name, value):
        if name in removed_flags:
            return None
        return update(name, value)

    _patched_update._easydel_removed_flag_patch = True  # type: ignore[attr-defined]
    config.update = _patched_update


def _patch_transformers_import_utils() -> None:
    """Backfill removed transformers import-utils symbols for remote model code."""
    try:
        from transformers.utils import import_utils as _hf_import_utils
    except Exception:
        return

    if not hasattr(_hf_import_utils, "is_torch_fx_available"):

        def _is_torch_fx_available() -> bool:
            try:
                is_torch_available = getattr(_hf_import_utils, "is_torch_available", None)
                return bool(is_torch_available()) if callable(is_torch_available) else False
            except Exception:
                return False

        _hf_import_utils.is_torch_fx_available = _is_torch_fx_available


def _patch_transformers_rope_scaling_property() -> None:
    """Normalize HF ``rope_scaling`` property for legacy DeepSeek remote modules."""
    try:
        from transformers.configuration_utils import PretrainedConfig as _HFPretrainedConfig
    except Exception:
        return

    rope_scaling_prop = getattr(_HFPretrainedConfig, "rope_scaling", None)
    if not isinstance(rope_scaling_prop, property):
        return

    original_get = rope_scaling_prop.fget
    if original_get is None or getattr(original_get, "_easydel_rope_scaling_patch", False):
        return

    def _patched_get(self):
        value = original_get(self)
        if getattr(self, "model_type", None) in {"deepseek_v2", "deepseek_v3"} and isinstance(value, dict):
            rope_type = value.get("rope_type", value.get("type"))
            if rope_type in (None, "default"):
                return None
        return value

    _patched_get._easydel_rope_scaling_patch = True  # type: ignore[attr-defined]
    _HFPretrainedConfig.rope_scaling = property(
        _patched_get,
        rope_scaling_prop.fset,
        rope_scaling_prop.fdel,
        rope_scaling_prop.__doc__,
    )


# OCD Maybe? idk why it's needed but it's here and i had to spend 1 hour to figure it out
def _patch_transformers_mrope_default_rope_validation() -> None:
    """Silence HF warnings for mRoPE metadata carried on ``rope_type='default'``."""
    try:
        from transformers.modeling_rope_utils import RotaryEmbeddingConfigMixin as _HFRotaryEmbeddingConfigMixin
    except Exception:
        return

    original_check = getattr(_HFRotaryEmbeddingConfigMixin, "_check_received_keys", None)
    if original_check is None or getattr(original_check, "_easydel_mrope_default_validation_patch", False):
        return

    mrope_metadata_keys = {"mrope_section", "mrope_interleaved"}

    def _patched_check_received_keys(
        rope_type,
        received_keys,
        required_keys,
        optional_keys=None,
        ignore_keys=None,
    ):
        if rope_type == "default" and mrope_metadata_keys.intersection(received_keys):
            ignore_keys = set(ignore_keys or ()) | mrope_metadata_keys
        return original_check(rope_type, received_keys, required_keys, optional_keys, ignore_keys)

    _patched_check_received_keys._easydel_mrope_default_validation_patch = True  # type: ignore[attr-defined]
    _HFRotaryEmbeddingConfigMixin._check_received_keys = staticmethod(_patched_check_received_keys)


def _patch_transformers_init_weights_tie_signature() -> None:
    """Handle legacy remote-model ``tie_weights()`` signature changes."""
    try:
        from transformers.modeling_utils import PreTrainedModel as _HFPreTrainedModel
    except Exception:
        return

    original_init_weights = getattr(_HFPreTrainedModel, "init_weights", None)
    if original_init_weights is None or getattr(original_init_weights, "_easydel_tie_patch", False):
        return

    def _patched_init_weights(self):
        try:
            return original_init_weights(self)
        except TypeError as exc:
            if "recompute_mapping" not in str(exc):
                raise
            return self.tie_weights()

    _patched_init_weights._easydel_tie_patch = True  # type: ignore[attr-defined]
    _HFPreTrainedModel.init_weights = _patched_init_weights


def _patch_eformer_exception_serialization() -> None:
    """Replace non-picklable remote exceptions with a safe fallback."""
    try:
        from eformer.executor.ray.types import ExceptionInfo as _ExceptionInfo
    except Exception:
        return

    original_ser_exc_info = _ExceptionInfo.ser_exc_info.__func__
    if getattr(original_ser_exc_info, "_easydel_picklable_exc_patch", False):
        return

    def _coerce_picklable_exception(exception: BaseException | None) -> BaseException | None:
        if exception is None:
            return None
        try:
            _pickle.loads(_pickle.dumps(exception))
            return exception
        except Exception:
            exc_type = f"{exception.__class__.__module__}.{exception.__class__.__qualname__}"
            try:
                message = str(exception)
            except Exception:
                message = repr(exception)
            fallback = RuntimeError(f"{exc_type}: {message}")
            notes = getattr(exception, "__notes__", None)
            if notes:
                for note in notes:
                    try:
                        fallback.add_note(note)
                    except Exception:
                        break
            return fallback

    def _patched_ser_exc_info(cls, exception: BaseException | None = None):
        exc_info = original_ser_exc_info(cls, exception)
        exc_info.ex = _coerce_picklable_exception(exc_info.ex)
        return exc_info

    _patched_ser_exc_info._easydel_picklable_exc_patch = True  # type: ignore[attr-defined]
    _ExceptionInfo.ser_exc_info = classmethod(_patched_ser_exc_info)


def _patch_transformers_autoconfig_gated_repo_skip() -> None:
    """Convert gated-repo config load failures to ``pytest.skip`` under pytest."""
    try:
        from transformers import AutoConfig as _HFAutoConfig
    except Exception:
        return

    auto_config_from_pretrained = _HFAutoConfig.__dict__.get("from_pretrained", None)
    if not isinstance(auto_config_from_pretrained, classmethod):
        return
    original_fn = auto_config_from_pretrained.__func__
    if getattr(original_fn, "_easydel_gated_repo_patch", False):
        return

    def _patched_from_pretrained(cls, *args, **kwargs):
        import os as _runtime_os

        try:
            return original_fn(cls, *args, **kwargs)
        except OSError as exc:
            if "PYTEST_CURRENT_TEST" not in _runtime_os.environ:
                raise
            message = str(exc).lower()
            is_gated_error = (
                "gated repo" in message
                or "gated model" in message
                or "you are trying to access a gated repo" in message
                or "access to this model is restricted" in message
                or "access to this repository is restricted" in message
                or ("401" in message and "huggingface" in message)
            )
            if not is_gated_error:
                raise
            import unittest as _unittest

            model_id = args[0] if args else kwargs.get("pretrained_model_name_or_path", "<unknown>")
            raise _unittest.SkipTest(f"Skipping gated Hugging Face repo during tests: {model_id}") from exc

    _patched_from_pretrained._easydel_gated_repo_patch = True  # type: ignore[attr-defined]
    _HFAutoConfig.from_pretrained = classmethod(_patched_from_pretrained)


def _apply_auto_environment() -> None:
    """Apply EasyDeL's default environment and logging setup."""
    if not _check_bool_flag("EASYDEL_AUTO", True):
        return

    _sys.setrecursionlimit(10000)

    _getlogger("jax._src.xla_bridge").setLevel(30)
    _getlogger("jax._src.mesh_utils").setLevel(30)
    _getlogger("jax._src.distributed").setLevel(30)

    _getlogger("httpx").setLevel(30)
    _getlogger("httpcore").setLevel(30)
    _getlogger("datasets").setLevel(30)

    _getlogger("numexpr.utils").setLevel(30)
    _getlogger("numexpr").setLevel(30)

    _getlogger("eray-executor").setLevel(30)
    _getlogger("absl").setLevel(30)

    _os.environ["NUMEXPR_NUM_THREADS"] = "8"

    _os.environ["KMP_AFFINITY"] = "noverbose"
    _os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    _os.environ.setdefault("GLOG_minloglevel", "3")
    _os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    _os.environ["CACHE_TRITON_KERNELS"] = "1"
    _os.environ.setdefault("TPU_MIN_LOG_LEVEL", "4")
    _os.environ.setdefault("TPU_STDERR_LOG_LEVEL", "4")

    _os.environ.setdefault("TPU_LOG_DIR", "disabled")
    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    _os.environ["XLA_FLAGS"] = (
        _os.getenv("XLA_FLAGS", "") + " "
        "--xla_gpu_triton_gemm_any=true  "
        "--xla_gpu_enable_while_loop_double_buffering=true  "
        "--xla_gpu_enable_pipelined_all_gather=true  "
        "--xla_gpu_enable_pipelined_reduce_scatter=true  "
        "--xla_gpu_enable_pipelined_all_reduce=true  "
        "--xla_gpu_enable_reduce_scatter_combine_by_dim=false  "
        "--xla_gpu_enable_all_gather_combine_by_dim=false  "
        "--xla_gpu_enable_reduce_scatter_combine_by_dim=false  "
        "--xla_gpu_all_gather_combine_threshold_bytes=33554432 "
        "--xla_gpu_reduce_scatter_combine_threshold_bytes=33554432 "
        "--xla_gpu_all_reduce_combine_threshold_bytes=33554432 "
        "--xla_gpu_multi_streamed_windowed_einsum=true  "
        "--xla_gpu_enable_latency_hiding_scheduler=true  "
        "--xla_gpu_enable_cublaslt=true "
        "--xla_gpu_enable_cudnn_fmha=true "
        "--xla_gpu_force_compilation_parallelism=4 "
        "--xla_gpu_enable_shared_constants=true "
        "--xla_gpu_enable_triton_gemm=true "
        "--xla_gpu_enable_command_buffer='' "
        "--xla_disable_hlo_passes=collective-permute-motion "
    )
    _maybe_apply_targeted_tpu_flags()
    _os.environ.update(
        {
            "NCCL_LL128_BUFFSIZE": "-2",
            "NCCL_LL_BUFFSIZE": "-2",
            "NCCL_PROTO": "SIMPLE,LL,LL128",
        }
    )
    if _os.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION", None) is None:
        _os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
    if _os.getenv("JAX_TRACEBACK_FILTERING", None) is None:
        _os.environ["JAX_TRACEBACK_FILTERING"] = "off"


def apply_linkup() -> None:
    """Run EasyDeL import-time compatibility patches and environment setup."""
    _ensure_optional_deepspeed_stub()
    _patch_transformers_rope_scaling_property()
    _patch_transformers_mrope_default_rope_validation()
    _patch_transformers_import_utils()
    _patch_eformer_exception_serialization()
    _patch_transformers_autoconfig_gated_repo_skip()
    _patch_transformers_init_weights_tie_signature()
    _patch_removed_jax_config_flags()
    _apply_auto_environment()


def initialize_distributed(logger) -> bool:
    """Initialize distributed JAX when enabled.

    Returns:
        Whether distributed initialization was enabled.
    """
    _os.environ.setdefault("JAX_ENABLE_PREEMPTION_SERVICE", "true")
    distributed_init_enabled = _check_bool_flag("ENABLE_DISTRIBUTED_INIT", True)
    if distributed_init_enabled:
        import jax as _jax

        _jax.config.update("jax_enable_preemption_service", True)
        if _jax.distributed.is_initialized():
            logger.debug("JAX distributed already initialized; using existing setup.")
        else:
            from eformer.executor import DistributedConfig as _DistributedConfig

            try:
                _DistributedConfig().initialize()
            except RuntimeError:
                if _jax.distributed.is_initialized():
                    logger.debug("JAX distributed already initialized; using existing setup.")
                else:
                    raise
    else:
        distributed_msg = (
            "Skipping initialization of `DistributedConfig` (ENABLE_DISTRIBUTED_INIT=0), "
            "you can initialize that via `ed.init_cluster()`."
        )
        if "ENABLE_DISTRIBUTED_INIT" in _os.environ:
            logger.info(distributed_msg)
        else:
            logger.debug(distributed_msg)
    return distributed_init_enabled
